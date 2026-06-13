"""
Volve Stage-2 data module (TZ-01 §2). Single source of truth for loading,
cleaning, assembly, split, scaling, and windowing.
No model script may reimplement any of these steps.
"""
import hashlib
import os
from datetime import datetime as _dt

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import (
    DATASET, RESULTS_DIR, DATA_CARD, EXPECTED,
    PRODUCERS, INJECTORS, WELL_ABBREV,
    CAL_START, CAL_END, TRAIN_END, TEST_START,
    PRODUCER_CHANNELS, INJECTOR_CHANNELS,
    TARGETS, TARGET_SOURCES, ZERO_AS_MISSING, VAL_FRACTION,
)

# ---------------------------------------------------------------------------
# Constants for physical cleaning rules (TZ-01 §2.3, §12.3)
# ---------------------------------------------------------------------------
# Allocation artifacts: well → date of the erroneous record (HRS=0.625)
_ALLOC = {
    "15 9-F-12 H": pd.Timestamp("2012-08-20"),
    "15 9-F-14 H": pd.Timestamp("2012-08-13"),
}
# Small negative WPR rows: well → date
_NEG_WPR = {
    "15 9-F-14 H": pd.Timestamp("2009-03-03"),   # WPR = -0.04
    "15 9-F-12 H": pd.Timestamp("2008-04-29"),   # WPR = -0.59
}

CALENDAR = pd.date_range(CAL_START, CAL_END, freq="D")


# ---------------------------------------------------------------------------
# Step 2.1: Load & canonicalize
# ---------------------------------------------------------------------------
def _load_sheet(wb_path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(wb_path, sheet_name=sheet)
    df = df.dropna(how="all")
    # Identify DATE column (may be named DATEPRD, DATE, etc.)
    date_col = next(c for c in df.columns if "DATE" in str(c).upper())
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "DATE"}).set_index("DATE").sort_index()
    dups = df.index[df.index.duplicated()]
    if len(dups):
        raise ValueError(f"Duplicate dates in sheet {sheet!r}: {dups[:3].tolist()}")
    return df


# ---------------------------------------------------------------------------
# Step 2.2: Calendar alignment
# ---------------------------------------------------------------------------
def _reindex_to_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Clip to CAL_END and reindex to the full daily calendar."""
    df = df.loc[df.index <= pd.Timestamp(CAL_END)]
    return df.reindex(CALENDAR)  # missing dates → NaN


# ---------------------------------------------------------------------------
# Step 2.3: Physical cleaning
# ---------------------------------------------------------------------------
def _physical_clean(df: pd.DataFrame, well: str, counts: dict,
                    is_producer: bool = True) -> pd.DataFrame:
    # HRS: fill pre-commissioning / post-shutdown NaN with 0
    df["HRS"] = df["HRS"].fillna(0.0)

    # Rule 1: clip HRS > 24.  Count only for producers (TZ expected=13 producers only).
    n_over = int((df["HRS"] > 24).sum())
    if is_producer:
        counts["hrs_gt_24"] = counts.get("hrs_gt_24", 0) + n_over
    df["HRS"] = df["HRS"].clip(0.0, 24.0)

    # Rule 2: small negative WPR → 0 (producers only; injectors have no WPR)
    if well in _NEG_WPR and "WPR" in df.columns:
        dt = _NEG_WPR[well]
        if dt in df.index and pd.notna(df.at[dt, "WPR"]) and df.at[dt, "WPR"] < 0:
            df.at[dt, "WPR"] = 0.0
            counts["small_negative_wpr"] = counts.get("small_negative_wpr", 0) + 1

    # Rule 3: allocation artifacts — convert rate to daily equivalent rate×HRS/24
    if well in _ALLOC and "OPR" in df.columns:
        dt = _ALLOC[well]
        if dt in df.index:
            hrs = df.at[dt, "HRS"]
            if 0.0 < hrs < 24.0:   # sanity: HRS already clipped, expect 0.625
                for col in TARGET_SOURCES:
                    if col in df.columns and pd.notna(df.at[dt, col]):
                        df.at[dt, col] = df.at[dt, col] * hrs / 24.0
                if "WPR" in df.columns and pd.notna(df.at[dt, "WPR"]) and df.at[dt, "WPR"] < 0:
                    df.at[dt, "WPR"] = 0.0
                counts["allocation_artifacts"] = counts.get("allocation_artifacts", 0) + 1

    # Rule 4: HRS > 12 and OPR = 0 → flag, keep.  Producers only (injectors have no OPR).
    if is_producer:
        opr = df["OPR"].fillna(0.0) if "OPR" in df.columns else pd.Series(0.0, index=df.index)
        inc = ((df["HRS"] > 12) & (opr == 0.0)).astype(int)
        counts["inconsistent_day"] = counts.get("inconsistent_day", 0) + int(inc.sum())
        df["_inc"] = inc
    else:
        df["_inc"] = 0

    return df


# ---------------------------------------------------------------------------
# Step 2.4: Zero-as-missing recoding
# ---------------------------------------------------------------------------
def _zero_as_missing(df: pd.DataFrame, well: str, counts: dict) -> pd.DataFrame:
    opr = df["OPR"].fillna(0.0) if "OPR" in df.columns else pd.Series(0.0, index=df.index)
    ab = WELL_ABBREV[well]
    for col in ZERO_AS_MISSING:
        if col not in df.columns:
            continue
        mask = (df[col] == 0.0) & (opr > 0.0)
        n = int(mask.sum())
        if n:
            counts[f"zero_prod_{ab}_{col}"] = n
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Step 2.6: Imputation — forward-fill gaps ≤ 3 days
# ---------------------------------------------------------------------------
def _ffill_limited(s: pd.Series, max_gap: int = 3):
    """Forward-fill NaN runs of length ≤ max_gap. Returns (filled, imputed_bool)."""
    was_nan = s.isna()
    if not was_nan.any():
        return s.copy(), pd.Series(False, index=s.index)
    run_id = (~was_nan).cumsum()
    run_len = was_nan.groupby(run_id).transform("sum")
    filled = s.ffill()
    long_gap = was_nan & (run_len > max_gap)
    filled[long_gap] = np.nan
    imputed = was_nan & filled.notna()
    return filled, imputed


def _impute_well(df: pd.DataFrame, feat_cols: list, well: str,
                 impute_out: dict) -> pd.DataFrame:
    ab = WELL_ABBREV[well]
    active = df["HRS"] > 0.0
    for col in feat_cols:
        if col not in df.columns or col == "HRS":
            continue
        filled, imputed = _ffill_limited(df[col], max_gap=3)
        if imputed.any():
            impute_out[f"imputed_{ab}_{col}"] = imputed.astype(int)
        still_nan = filled.isna()
        n_active_nan = int((still_nan & active).sum())
        if n_active_nan:
            impute_out[f"active_unfilled_{ab}_{col}"] = n_active_nan
        # Zero-fill remaining NaN, then enforce inactive days → 0.
        # This prevents forward-fill from leaking active-period values
        # across shutdown boundaries (TZ §2.2: inactive days inputs = 0).
        filled = filled.fillna(0.0)
        filled[~active] = 0.0
        df.loc[:, col] = filled
    return df


# ---------------------------------------------------------------------------
# Step 2.10: Data card
# ---------------------------------------------------------------------------
def _write_data_card(file_hash: str, counts: dict, impute_out: dict,
                     feat_cols: list, wide: pd.DataFrame,
                     train_mask: pd.Series) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def _row(key, expected, actual):
        ok = "✓" if actual == expected else f"⚠ got {actual}"
        return f"| {key} | {expected} | {actual} | {ok} |"

    checks = [
        ("calendar_days",              EXPECTED["calendar_days"]),
        ("train_rows",                 EXPECTED["train_rows"]),
        ("test_rows",                  EXPECTED["test_rows"]),
        ("hrs_gt_24",                  EXPECTED["hrs_gt_24"]),
        ("allocation_artifacts",       EXPECTED["allocation_artifacts"]),
        ("small_negative_wpr",         EXPECTED["small_negative_wpr"]),
        ("inconsistent_day",           EXPECTED["inconsistent_day"]),
        ("zero_production_days_train", EXPECTED["zero_production_days_train"]),
    ]
    mismatches = [k for k, e in checks if counts.get(k, "?") != e]
    header = []
    if mismatches:
        header = [f"> **WARNING — {len(mismatches)} count mismatch(es): {mismatches}**", ""]

    zero_opr = wide.loc[train_mask, "OPR_field"]
    train_opr_mean = float(zero_opr.mean())
    test_opr_mean  = float(wide.loc[~train_mask, "OPR_field"].mean())

    lines = header + [
        "# Data Card — Volve Stage-2",
        "",
        f"Generated: {_dt.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source SHA-256 (first 16 chars): `{file_hash}`",
        "",
        "## Row counts vs TZ-01 §2 expected",
        "",
        "| Check | Expected | Actual | Status |",
        "| --- | --- | --- | --- |",
    ]
    for key, exp in checks:
        lines.append(_row(key, exp, counts.get(key, "?")))

    # Zero-as-missing breakdown
    lines += [
        "",
        "## Zero-as-missing counts (TZ-01 §2.4)",
        "",
        "| Well × channel | Days zero while OPR>0 |",
        "| --- | --- |",
    ]
    for k, v in counts.items():
        if k.startswith("zero_prod_"):
            lines.append(f"| {k[10:]} | {v} |")

    lines += [
        "",
        "## Imputation summary (TZ-01 §2.6)",
        "",
        "| Flag | Value |",
        "| --- | --- |",
    ]
    for k, v in impute_out.items():
        if not isinstance(v, pd.Series):
            lines.append(f"| {k} | {v} |")
        else:
            lines.append(f"| {k} (imputed days) | {int(v.sum())} |")

    lines += [
        "",
        "## Feature roster",
        "",
        f"Total feature columns: {len(feat_cols)}",
        "",
        "```",
        ", ".join(feat_cols),
        "```",
        "",
        "## Split summary",
        "",
        f"Train: {CAL_START} – {TRAIN_END} ({counts.get('train_rows','?')} rows)",
        f"Test:  {TEST_START} – {CAL_END} ({counts.get('test_rows','?')} rows)",
        f"Val:   last {VAL_FRACTION*100:.0f}% of train",
        "",
        f"Train OPR mean: {train_opr_mean:.1f} m³/day",
        f"Test  OPR mean: {test_opr_mean:.1f} m³/day  ← M2 regime shift",
        "",
        "## Frozen decisions (from §12 of preprocessing plan)",
        "",
        "- F-12 ABHP/ABHT excluded → `f12_bhp_dead` indicator (1 from 2011-01-01)",
        "- F-1C AAP and F-14 AAP excluded",
        "- Injectors: HRS + WIR only (all pressure/T channels ≥94.6% NaN)",
        "- AW (sum of 5 producer activity masks) included as feature",
        "- GPR kept as third target; GOR ≈ 150",
        "- Zero-production days kept (rule predictor handles them)",
        "- RobustScaler fit on train only; targets inverse-transformed before metrics",
        "- NOTE: spec says '3135 days' counting day-intervals; actual row count = 3136",
    ]

    with open(DATA_CARD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[data_card] written: {DATA_CARD}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_dataset() -> dict:
    """
    Build the full Volve Stage-2 dataset per TZ-01 §2.
    Returns a dict with all splits, scalers, the assembled wide frame, and metadata.
    Call this once; pass the result to make_windows() and model scripts.
    """
    wb_path = DATASET
    if not os.path.exists(wb_path):
        raise FileNotFoundError(f"Dataset not found: {wb_path}")

    with open(wb_path, "rb") as fh:
        file_hash = hashlib.sha256(fh.read()).hexdigest()[:16]

    counts: dict = {}
    impute_out: dict = {}

    # ---- §2.1 / §2.2: Load and align each well -------------------------
    well_dfs: dict = {}
    for well in PRODUCERS + INJECTORS:
        df = _load_sheet(wb_path, well)
        df = _reindex_to_calendar(df)
        df = _physical_clean(df, well, counts, is_producer=(well in PRODUCERS))
        df = _zero_as_missing(df, well, counts)
        well_dfs[well] = df

    # ---- §2.7: Field targets — sum BEFORE exclusions/imputation ---------
    # Computed from physically-cleaned data; zero-fill inactive days.
    tgt = pd.DataFrame(0.0, index=CALENDAR, columns=TARGETS)
    for well in PRODUCERS:
        wdf = well_dfs[well]
        for src, field_col in zip(TARGET_SOURCES, TARGETS):
            if src in wdf.columns:
                tgt[field_col] += wdf[src].fillna(0.0)

    # ---- §2.5: Channel exclusions ---------------------------------------
    # F-12: drop ABHP, ABHT; add f12_bhp_dead binary indicator
    f12_dead = (CALENDAR >= pd.Timestamp("2011-01-01")).astype(int)
    f12_bhp_dead = pd.Series(f12_dead, index=CALENDAR, name="f12_bhp_dead")
    for col in ["ABHP", "ABHT"]:
        well_dfs["15 9-F-12 H"].drop(columns=[col], inplace=True, errors="ignore")
    # F-1C and F-14: drop AAP
    for well in ["15 9-F-1 C", "15 9-F-14 H"]:
        well_dfs[well].drop(columns=["AAP"], inplace=True, errors="ignore")
    # Injectors: keep only HRS, WIR (+ _inc flag)
    for well in INJECTORS:
        keep = [c for c in well_dfs[well].columns if c in ("HRS", "WIR", "_inc")]
        well_dfs[well] = well_dfs[well][[c for c in keep if c in well_dfs[well].columns]].copy()

    # ---- §2.6: Imputation per well --------------------------------------
    # Producers: only genuine producer channels (TZ-02 R4 — exclude dead *_WIR
    # columns the producer sheets carry as all-NaN→0; they are not in
    # PRODUCER_CHANNELS and must not be imputed/flagged or assembled).
    for well in PRODUCERS:
        feat = [c for c in well_dfs[well].columns if c in PRODUCER_CHANNELS]
        well_dfs[well] = _impute_well(well_dfs[well], feat, well, impute_out)
    for well in INJECTORS:
        feat = [c for c in well_dfs[well].columns if c != "_inc"]
        well_dfs[well] = _impute_well(well_dfs[well], feat, well, impute_out)

    # ---- §2.2 / §2.7: Activity masks and AW ----------------------------
    masks: dict = {}
    for well in PRODUCERS + INJECTORS:
        ab = WELL_ABBREV[well]
        hrs = well_dfs[well]["HRS"].fillna(0.0)
        masks[f"active_{ab}"] = (hrs > 0.0).astype(int)
    AW = sum(masks[f"active_{WELL_ABBREV[w]}"] for w in PRODUCERS)

    # ---- §2.7: Assemble wide frame --------------------------------------
    parts: dict = {}
    # Producer feature columns (well-prefixed, post-exclusion)
    for well in PRODUCERS:
        ab = WELL_ABBREV[well]
        wdf = well_dfs[well]
        for col in wdf.columns:
            # TZ-02 R4: restrict to PRODUCER_CHANNELS (post-exclusion) so dead
            # *_WIR columns the producer sheets carry never enter the feature set.
            if col not in PRODUCER_CHANNELS:
                continue
            parts[f"{ab}_{col}"] = wdf[col]
    # Injector HRS + WIR
    for well in INJECTORS:
        ab = WELL_ABBREV[well]
        wdf = well_dfs[well]
        for col in ["HRS", "WIR"]:
            if col in wdf.columns:
                parts[f"{ab}_{col}"] = wdf[col]
    # Activity masks
    for key, val in masks.items():
        parts[key] = val
    parts["AW"]           = pd.Series(AW, index=CALENDAR)
    parts["f12_bhp_dead"] = f12_bhp_dead
    # Imputed-flag columns (Series only)
    for key, val in impute_out.items():
        if isinstance(val, pd.Series):
            parts[key] = val
    # Inconsistency flags per producer
    for well in PRODUCERS:
        ab = WELL_ABBREV[well]
        if "_inc" in well_dfs[well].columns:
            s = well_dfs[well]["_inc"]
            if s.any():
                parts[f"inconsistent_{ab}"] = s

    wide = pd.DataFrame(parts, index=CALENDAR)
    for col in TARGETS:
        wide[col] = tgt[col].values

    # Fill any residual NaN in non-target columns with 0
    feat_cols = [c for c in wide.columns if c not in TARGETS]
    wide[feat_cols] = wide[feat_cols].fillna(0.0)

    # ---- Counts ---------------------------------------------------------
    train_mask = wide.index <= pd.Timestamp(TRAIN_END)
    test_mask  = wide.index >= pd.Timestamp(TEST_START)
    counts["calendar_days"]              = len(CALENDAR)
    counts["train_rows"]                 = int(train_mask.sum())
    counts["test_rows"]                  = int(test_mask.sum())
    counts["zero_production_days_train"] = int(
        (wide.loc[train_mask, "OPR_field"] == 0.0).sum()
    )
    # Convenience alias for test assertions
    counts["f12_abhp_zero_producing"] = counts.get("zero_prod_F12H_ABHP", 0)

    # ---- §2.8: Split & scaling ------------------------------------------
    n_train = counts["train_rows"]
    n_val   = int(round(n_train * VAL_FRACTION))
    val_start_idx = n_train - n_val          # index in the full calendar
    val_start_dt  = wide.index[val_start_idx]

    val_mask        = train_mask & (wide.index >= val_start_dt)
    pure_train_mask = train_mask & (wide.index < val_start_dt)

    # Columns that should NOT be scaled (binary / integer flags)
    binary_cols = [
        c for c in feat_cols
        if c.startswith("active_")
        or c.startswith("imputed_")
        or c.startswith("inconsistent_")
        or c in ("AW", "f12_bhp_dead")
    ]
    scaled_cols = [c for c in feat_cols if c not in binary_cols]
    sc_idx = [feat_cols.index(c) for c in scaled_cols]

    X_all = wide[feat_cols].values.astype(float)
    y_all = wide[TARGETS].values.astype(float)

    X_scaler = RobustScaler()
    X_scaler.fit(X_all[train_mask][:, sc_idx])
    X_sc = X_all.copy()
    X_sc[:, sc_idx] = X_scaler.transform(X_all[:, sc_idx])

    y_scaler = RobustScaler()
    y_scaler.fit(y_all[train_mask])

    tr_i = np.where(train_mask)[0]
    va_i = np.where(val_mask)[0]
    te_i = np.where(test_mask)[0]

    # ---- §2.10: Data card -----------------------------------------------
    _write_data_card(file_hash, counts, impute_out, feat_cols, wide, train_mask)

    return {
        "wide":          wide,
        "feat_cols":     feat_cols,
        "scaled_cols":   scaled_cols,
        "binary_cols":   binary_cols,
        "sc_idx":        sc_idx,
        "X_scaled":      X_sc,     # shape (n_calendar, n_feat)
        "y_raw":         y_all,    # shape (n_calendar, 3)
        "train_mask":    train_mask,
        "val_mask":      val_mask,
        "test_mask":     test_mask,
        "train_dates":   wide.index[train_mask],
        "val_dates":     wide.index[val_mask],
        "test_dates":    wide.index[test_mask],
        "X_train":       X_sc[tr_i], "y_train": y_all[tr_i],
        "X_val":         X_sc[va_i], "y_val":   y_all[va_i],
        "X_test":        X_sc[te_i], "y_test":  y_all[te_i],
        "X_scaler":      X_scaler,
        "y_scaler":      y_scaler,
        "counts":        counts,
        "file_hash":     file_hash,
        "CALENDAR":      CALENDAR,
    }


# ---------------------------------------------------------------------------
# §2.9: Windowing
# ---------------------------------------------------------------------------
def make_windows(ds: dict, L: int, regime: str = "B") -> dict:
    """
    Build sliding-window tensors.
    L:      lookback in days.
    regime: 'B' — covariates only (primary, no target feedback).
            'A' — adds lagged true targets y(t-L)..y(t-1) as channels.
    Returns X_train/val/test (n_samples, L, n_channels), y_*/val/test (n_samples, 3).
    Test windows may reach back into train-period inputs (legal).
    Regime B test tensors must contain no target-derived channels.
    """
    X_sc  = ds["X_scaled"]   # (n, F)
    y_raw = ds["y_raw"]      # (n, 3)
    n     = len(X_sc)

    y_sc_full = None
    if regime == "A":
        y_sc_full = np.zeros_like(y_raw)
        y_sc_full[:] = ds["y_scaler"].transform(y_raw)

    def _window(t):
        start = t - L + 1
        if start < 0:
            return None
        xw = X_sc[start : t + 1]  # (L, F)
        if regime == "A":
            # Lagged targets: indices t-L .. t-1
            yw = y_sc_full[max(0, start - 1) : t]
            if len(yw) < L:
                pad = np.zeros((L - len(yw), 3))
                yw  = np.vstack([pad, yw])
            xw = np.concatenate([xw, yw], axis=1)
        return xw

    def _build(idx_arr):
        Xs, ys = [], []
        for t in idx_arr:
            xw = _window(t)
            if xw is None:
                continue
            Xs.append(xw)
            ys.append(y_raw[t])
        if not Xs:
            nchan = X_sc.shape[1] + (3 if regime == "A" else 0)
            return np.empty((0, L, nchan)), np.empty((0, 3))
        return np.array(Xs, dtype=float), np.array(ys, dtype=float)

    tr_i = np.where(ds["train_mask"])[0]
    va_i = np.where(ds["val_mask"])[0]
    te_i = np.where(ds["test_mask"])[0]

    X_tr, y_tr = _build(tr_i)
    X_va, y_va = _build(va_i)
    X_te, y_te = _build(te_i)

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_va, "y_val":   y_va,
        "X_test":  X_te, "y_test":  y_te,
        "L": L, "regime": regime,
        "n_feat": X_tr.shape[2] if len(X_tr) > 0 else X_sc.shape[1],
    }

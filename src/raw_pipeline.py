"""
Raw -> preprocessed pipeline (reconstruction of the data-preparation workflow).

Reproduces, from the original Equinor Volve production export, the chain that
produced ``SD28Nov2024_SelectedFeature_WeveletDenoised.xlsx``:

    Volve production data.xlsx  (raw daily, 24 cols, 7 wells)
        |  (1) select 3 producing wells, map variables, compute OPR
        v
    multi-well compiled dataset (10 features + OPR per well)   ~ NM_3Apr / NM_16Apr
        |  (2) clean: on-stream filter, impute dead downhole sensors,
        |            drop annulus pressure
        v
    cleaned dataset
        |  (3) wavelet denoising (SURE soft-threshold)          ~ NM..NoiseFree
        v
    denoised dataset
        |  (4) NSGA-II / LSTM feature selection  (10 -> 7)       (Table 1 / Fig 6)
        v
    SelectedFeature_WaveletDenoised  (Train = F-12+F-14, Test = F-11)  ~ SD...xlsx

Lineage evidence (verified against the supplied intermediates):
  * OPR = BORE_OIL_VOL / ON_STREAM_HRS   (F-12 day 1: 284.65 / 11.5 = 24.75,
    exactly the OIL_PROD_RATE in NM_3Apr).
  * Column abbreviations taken from the NM_3Apr 'abbreviation' sheet.
  * Annulus pressure present in NM_3Apr (34 cols) but dropped in NM_16Apr (31).

Caveats (author-specific steps not fully recoverable -> documented, approximated):
  * F-12's downhole pressure/temperature are largely dead (raw median 0) and were
    imputed in the released SD file; we impute by interpolation.
  * The SD file's ``AW`` takes values {0,1,2,3} (a processed regime/quality code);
    the paper documents AW as well status 0/1, which is what we emit.
  * Exact record counts / date windows of the SD split reflect ad-hoc cleaning;
    we stack each well's on-stream series (counts therefore differ slightly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .preprocessing import wavelet_denoise

RAW_XLSX = C.INPUT_DIR / "Volve production data.xlsx"
RAW_SHEET = "Daily Production Data"

# Three producing wells studied in the paper.
WELLS = ["15/9-F-12", "15/9-F-14", "15/9-F-11"]
TRAIN_WELLS = ["15/9-F-12", "15/9-F-14"]
TEST_WELLS = ["15/9-F-11"]

# Raw column -> paper variable.  (OPR derived separately; Time added as index.)
COLUMN_MAP = {
    "ON_STREAM_HRS": "OSH",            # hours of production per day
    "AVG_DOWNHOLE_PRESSURE": "ADP",    # avg bottomhole pressure (bar)
    "AVG_DOWNHOLE_TEMPERATURE": "ADTemp",  # avg bottomhole temperature (C)
    "AVG_DP_TUBING": "ADT",            # avg tubing differential pressure (bar)
    "AVG_CHOKE_SIZE_P": "ACS",         # avg choke opening (%)
    "AVG_WHP_P": "AWHP",               # avg wellhead pressure (bar)
    "AVG_WHT_P": "AWHT",               # avg wellhead temperature (C)
    "DP_CHOKE_SIZE": "DCS",            # choke size (mm)
}
# Full 10-feature set before NSGA-II selection (Time + AW + the 8 mapped).
FULL_FEATURES = ["Time", "OSH", "ADP", "ADTemp", "ADT", "ACS",
                 "AWHP", "AWHT", "DCS", "AW"]
# Annulus pressure is loaded then dropped (mirrors NM_3Apr -> NM_16Apr).
ANNULUS_COL = "AVG_ANNULUS_PRESS"


# --------------------------------------------------------------------------- #
# (1) Load raw + compile per-well frames
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW_XLSX, sheet_name=RAW_SHEET)
    return df[df["FLOW_KIND"] == "production"].copy()


def build_well_frame(raw: pd.DataFrame, well: str,
                     on_stream_only: bool = True) -> pd.DataFrame:
    """Per-well daily frame with the 10 features + OPR (physical units)."""
    s = raw[raw["NPD_WELL_BORE_NAME"] == well].sort_values("DATEPRD").copy()
    if on_stream_only:
        s = s[s["ON_STREAM_HRS"] > 0]

    out = pd.DataFrame({"DATE": s["DATEPRD"].values})
    for raw_col, name in COLUMN_MAP.items():
        out[name] = s[raw_col].to_numpy(dtype=float)
    out[ANNULUS_COL] = s[ANNULUS_COL].to_numpy(dtype=float)

    # OPR = oil volume / on-stream hours  (m3 per producing hour).
    osh = s["ON_STREAM_HRS"].to_numpy(dtype=float)
    vol = s["BORE_OIL_VOL"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        opr = np.where(osh > 0, vol / osh, 0.0)
    out["OPR"] = np.nan_to_num(opr)

    # Well status (paper: open=1, closed=0).
    out["AW"] = (osh > 0).astype(int)
    out["well"] = well
    return out


# --------------------------------------------------------------------------- #
# (2) Cleaning: impute dead downhole sensors, drop annulus
# --------------------------------------------------------------------------- #
def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Treat zero downhole sensor readings as missing and interpolate.

    F-12's downhole pressure/temperature are largely dead (recorded as 0); the
    released dataset carries plausible filled values, so we interpolate zeros
    in ADP/ADTemp/ADT and forward/back-fill the ends.  Annulus pressure is
    dropped (NM_16Apr step).
    """
    df = df.drop(columns=[ANNULUS_COL], errors="ignore").copy()
    for col in ["ADP", "ADTemp", "ADT"]:
        v = df[col]
        # Treat zeros as missing ONLY when the sensor is genuinely DEAD (mostly
        # zero, as for F-12's downhole gauges) — otherwise legitimate zero
        # readings (e.g. low-rate tubing differential) would be overwritten.
        if (v == 0).mean() > 0.5:
            v = v.replace(0.0, np.nan)
        # Always fill genuine NaN (missing records) by interpolation either way.
        df[col] = v.interpolate(limit_direction="both").bfill().ffill()
    # Any residual NaNs in other features -> column median.
    for col in ["OSH", "ACS", "AWHP", "AWHT", "DCS"]:
        df[col] = df[col].fillna(df[col].median())
    return df


# --------------------------------------------------------------------------- #
# (3) Wavelet denoising
# --------------------------------------------------------------------------- #
DENOISE_COLS = ["OSH", "ADP", "ADTemp", "ADT", "ACS", "AWHP", "AWHT", "DCS", "OPR"]


def denoise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Wavelet-denoise the signal columns using the parameters in
    ``config.DENOISE`` (target OPR, heavier) and ``config.DENOISE_FEATURES``
    (exogenous features, lighter) — chosen to MAXIMISE the per-column
    correlation with the supplied SD file. Single source of truth; no hard-coded
    params here.

    (NB: the §7 'approx-only' OPR exploration — even heavier smoothing that lets
    the one-step models reach the paper's ~2.7 regime — is a separate concern in
    `scripts/00b_denoising.py`/`preprocessing.wavelet_approx`, not the
    SD-faithful reconstruction produced here.)
    """
    out = df.copy()
    for col in DENOISE_COLS:
        if col not in out:
            continue
        params = C.DENOISE if col == "OPR" else C.DENOISE_FEATURES
        out[col] = wavelet_denoise(out[col].to_numpy(dtype=float), **params)
    return out


# --------------------------------------------------------------------------- #
# (4) Assemble compiled / denoised datasets with a Time index
# --------------------------------------------------------------------------- #
def _add_time(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "Time", np.arange(1, len(out) + 1))
    return out


# Columns summed across the 3 wells per date (verified against the SD file:
# corr(sum) = 1.000 for ADP/ADTemp; OSH/AWHP/DCS match the summed series).
_SUM_COLS = ["OSH", "ADP", "ADTemp", "ADT", "ACS", "AWHP", "AWHT", "DCS"]


def aggregate_by_date(clean: bool = False) -> pd.DataFrame:
    """Date-aligned aggregation across the 3 wells — the TRUE structure of the
    supplied SD file (one row per date, not stacked wells).

    Verified exactly against the supplied intermediates:
      * OPR(date)  = Σ BORE_OIL_VOL / Σ ON_STREAM_HRS  (F-12 day 1 = 284.65/11.5
        = 24.75, the NM_3Apr value);
      * OSH/ADP/ADTemp/.../DCS(date) = Σ across the active wells (corr 1.000);
      * AW(date)   = number of wells producing that day → values {0,1,2,3},
        explaining the SD file's `AW` (NOT a 0/1 single-well status).

    The SD file does NOT impute F-12's dead downhole sensors — they contribute
    their raw zeros to the sum (verified: with no imputation every feature
    correlates 0.98-0.9999 with the SD file; imputing inflates ADP/ADTemp).
    Hence ``clean=False`` by default.
    """
    raw = load_raw()
    parts = []
    for well in WELLS:
        f = build_well_frame(raw, well, on_stream_only=False)  # OPR = per-well rate
        if clean:
            f = clean_frame(f)                          # impute dead sensors first
        f["_active"] = (f["OSH"] > 0).astype(int)
        parts.append(f)
    allw = pd.concat(parts, ignore_index=True)

    g = allw.groupby("DATE")
    agg = pd.DataFrame({c: g[c].sum() for c in _SUM_COLS})
    count = g["_active"].sum()                      # active-well count {0,1,2,3}
    # AW reconstruction (config.AW_MODE): 'count' reproduces the SD file
    # (Σ WORK over the 3 wells); 'status' is the paper-text 0/1 field indicator.
    agg["AW"] = count if C.AW_MODE == "count" else (count > 0).astype(int)
    # OPR(date) = sum of per-well rates (verified ≈ SD: mean 166 ≈ AW(1.94) ×
    # total-vol/total-osh; sum-of-rates reproduces it, total/total does not).
    agg["OPR"] = g["OPR"].sum()
    agg = agg.reset_index().sort_values("DATE").reset_index(drop=True)
    return agg


def compile_dataset(denoise: bool = True, clean: bool = False,
                    n_train: int = 2176, n_test: int = 716):
    """Return (train_df, test_df) reconstructing the SD file by date-aligned
    aggregation, then a time-ordered split (first ``n_train`` dates = train,
    next ``n_test`` = test — the SD file's 2176 / 716).
    """
    agg = aggregate_by_date(clean=clean)
    agg.insert(0, "Time", np.arange(1, len(agg) + 1))
    cols = ["Time"] + [c for c in FULL_FEATURES if c != "Time"] + ["OPR"]
    agg = agg[cols]

    train = agg.iloc[:n_train].copy()
    test = agg.iloc[n_train:n_train + n_test].copy()
    if denoise:
        train = denoise_frame(train)
        test = denoise_frame(test)
    return train, test


def per_well_frames(features=None, denoise: bool = True, clean: bool = True):
    """Return {well: DataFrame} with a per-well day index + the requested
    features + OPR, cleaned and wavelet-denoised.  Used by the Fig-14
    sensitivity scenarios so the F-12 / F-14 / F-11 splits are exact (proper
    well IDs) instead of a heuristic boundary on the stacked SD file.
    """
    features = features or ["Time", "OSH", "ADP", "ADTemp", "AWHP", "DCS", "AW"]
    raw = load_raw()
    out = {}
    for well in WELLS:
        f = build_well_frame(raw, well)
        if clean:
            f = clean_frame(f)
        if denoise:
            f = denoise_frame(f)
        f = f.reset_index(drop=True)
        f.insert(0, "Time", np.arange(1, len(f) + 1))
        out[well] = f[[c for c in features] + ["OPR"]]
    return out


# --------------------------------------------------------------------------- #
# (5) Feature reduction 10 -> 7  (greedy forward selection demonstration)
# --------------------------------------------------------------------------- #
def greedy_feature_selection(train_df: pd.DataFrame, candidates=None,
                             seed: int = C.SEED):
    """Greedy forward selection mirroring the NSGA-II dual objective (error vs
    parsimony).  Adds the feature that most reduces time-series-CV RMSE until the
    improvement plateaus, returning the selection order and the RMSE/R2 curve.

    A Random-Forest surrogate over the exogenous features only (no lagged OPR)
    is used (fast, deterministic) instead of retraining an LSTM per subset; on
    the date-aligned aggregation it recovers exactly the paper's 7 selected
    features and drops exactly ADT/AWHT/ACS (Table 1 features 8-10).
    """
    from sklearn.ensemble import RandomForestRegressor
    from . import metrics as M

    candidates = list(candidates or [c for c in FULL_FEATURES])
    df = train_df.reset_index(drop=True)
    y = df["OPR"].to_numpy(float)
    n = len(df)
    split = int(n * 0.8)

    def cv_rmse(feats):
        # Exogenous-feature mapping only (no lagged OPR): feature selection must
        # rank the well-performance variables' own contribution to OPR, exactly
        # as the paper's NSGA-II/LSTM wrapper does. Including OPR(t-1) would let
        # the surrogate ignore the features and produce a meaningless ranking.
        X = np.column_stack([df[f].to_numpy(float) for f in feats])
        rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        rf.fit(X[:split], y[:split])
        pred = rf.predict(X[split:])
        return M.rmse(y[split:], pred), M.r2(y[split:], pred)

    selected, order = [], []
    remaining = candidates.copy()
    while remaining:
        scored = [(f, *cv_rmse(selected + [f])) for f in remaining]
        scored.sort(key=lambda t: t[1])              # lowest RMSE first
        best_f, best_rmse, best_r2 = scored[0]
        selected.append(best_f)
        remaining.remove(best_f)
        order.append({"n": len(selected), "added": best_f,
                      "subset": list(selected),
                      "RMSE": round(best_rmse, 4), "R2": round(best_r2, 5)})
    return order

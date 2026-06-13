"""
Evaluation module (TZ-01 §3). All metrics are on raw physical values (m³/day).
Models must inverse-transform predictions before calling evaluate().
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import TARGETS, TABLES_DIR, EXPECTED


def rmse(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2(y_true, y_pred) -> float:
    return float(r2_score(np.asarray(y_true, float), np.asarray(y_pred, float)))


def block_bootstrap_rmse(y_true, y_pred, n_boot: int = 2000, block: int = 30,
                         seed: int = 42, alpha: float = 0.05) -> tuple:
    """
    Block-bootstrap 95% CI on RMSE (block=30 days per TZ-01 §3).
    Returns (lo, hi) for the (1-alpha) interval.
    """
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(yt)
    n_blocks = int(np.ceil(n / block))
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        boot.append(rmse(yt[idx], yp[idx]))
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def persistence_rmse(y_full: np.ndarray, test_days: int) -> dict:
    """
    Compute persistence baseline (y(t)=y(t-1)) on the test set.
    y_full: shape (n_calendar, 3) — raw field targets over the full calendar.
    Returns {target: rmse_value}.
    Expected: OPR≈7.69, GPR≈1149, WPR≈19.6 (within 1%).
    """
    y_full = np.asarray(y_full, float)
    ts = len(y_full) - test_days
    y_true = y_full[ts:]
    y_pred = y_full[ts - 1 : ts - 1 + test_days]
    return {tgt: rmse(y_true[:, i], y_pred[:, i]) for i, tgt in enumerate(TARGETS)}


def rmse_r2_consistent(rmse_val: float, r2_val: float, y_true,
                       tol: float = 0.02) -> bool:
    """
    R²/RMSE consistency check (guards against Stage-1 Table-1 error, TZ-01 §3).
    Implied variance = RMSE² / (1 − R²) must match actual variance within tol.
    """
    yt = np.asarray(y_true, float)
    actual_var = float(np.var(yt, ddof=0))
    denom = 1.0 - r2_val
    if abs(denom) < 1e-12 or actual_var < 1e-12:
        return True
    implied_var = rmse_val ** 2 / denom
    return abs(implied_var - actual_var) / actual_var < tol


def skill_score(y_true, y_pred, y_ref) -> float:
    """
    Skill score = 1 − MSE_model / MSE_reference (TZ-02 R0).
    >0 means the model beats the naive reference; 0 ties it; <0 is worse.
    Robust to the low-test-variance distortion that breaks R² ranking.
    """
    yt, yp, yr = (np.asarray(a, float) for a in (y_true, y_pred, y_ref))
    mse_model = float(np.mean((yt - yp) ** 2))
    mse_ref   = float(np.mean((yt - yr) ** 2))
    if mse_ref < 1e-12:
        return float("nan")
    return float(1.0 - mse_model / mse_ref)


def evaluate(y_true, y_pred, *, split: str, model: str, regime: str,
             seed: int, y_train=None, persist_pred=None) -> list:
    """
    Compute per-target metrics and return rows for the master table.
    y_true, y_pred: array-like (n_samples, 3) — RAW physical units (m³/day).
    Includes R²/RMSE consistency self-check per target.

    TZ-02 R0 additions (computed only when the optional inputs are supplied):
    - skill_vs_trainmean: skill vs predicting the train mean as a constant
      (climatology reference). Requires y_train (raw train targets).
    - skill_vs_persistence: skill vs y(t)=y(t-1) (Regime A only). Requires
      persist_pred aligned row-for-row with y_true.
    - R2_low_variance_warning: True when Var(test_target) < 0.5·Var(train_target),
      i.e. R² is distorted by low test variance and must not drive ranking.
    """
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    ytr = np.asarray(y_train, float) if y_train is not None else None
    yps = np.asarray(persist_pred, float) if persist_pred is not None else None
    rows = []
    for i, tgt in enumerate(TARGETS):
        rm = rmse(yt[:, i], yp[:, i])
        ma = mae(yt[:, i], yp[:, i])
        r  = r2(yt[:, i], yp[:, i])
        consistent = rmse_r2_consistent(rm, r, yt[:, i])
        row = {
            "model": model, "regime": regime, "target": tgt,
            "split": split, "seed": seed,
            "RMSE": round(rm, 4), "MAE": round(ma, 4), "R2": round(r, 4),
            "RMSE_R2_ok": consistent,
        }
        if ytr is not None:
            train_mean_i = float(np.mean(ytr[:, i]))
            ref = np.full_like(yt[:, i], train_mean_i)
            s = skill_score(yt[:, i], yp[:, i], ref)
            row["skill_vs_trainmean"] = None if np.isnan(s) else round(s, 4)
            var_tr = float(np.var(ytr[:, i], ddof=0))
            var_te = float(np.var(yt[:, i], ddof=0))
            row["R2_low_variance_warning"] = bool(var_te < 0.5 * var_tr)
        if regime == "A" and yps is not None:
            s = skill_score(yt[:, i], yp[:, i], yps[:, i])
            row["skill_vs_persistence"] = None if np.isnan(s) else round(s, 4)
        rows.append(row)
    return rows


def append_master_table(rows: list) -> str:
    """Append rows to results/tables/metrics_master.csv (long format)."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, "metrics_master.csv")
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["model", "regime", "target", "split", "seed"], keep="last"
        )
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    return path

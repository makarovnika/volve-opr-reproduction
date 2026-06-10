"""
Statistical error metrics used throughout the paper (Sec. 2.4 / Supp. Sec. 4).

All metrics operate on OPR values expressed in physical units (m3/day) so the
numbers are directly comparable with Tables 3-4.

    ARE  - Average Relative Error          (signed, can be negative)
    MAE  - Mean Absolute Error             (m3/day)
    RMSE - Root Mean Square Error          (m3/day)
    R2   - Coefficient of determination
    SD   - Standard deviation of residuals (m3/day)
"""
from __future__ import annotations

import numpy as np


def _clean(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return y_true, y_pred


def are(y_true, y_pred, eps: float = 1.0) -> float:
    """Average relative error: mean of (pred - meas) / meas.

    Rows where measured OPR ≈ 0 (shut-in days) are EXCLUDED rather than having
    their denominator floored: flooring keeps the full numerator, so a zero-truth
    row injects a spurious relative error of (pred / eps) that can dominate the
    mean. Excluding them yields a meaningful relative error over producing rows.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    mask = np.abs(y_true) >= eps
    if not mask.any():
        return 0.0
    return float(np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def sd_residual(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    res = y_pred - y_true
    return float(np.std(res, ddof=0))


def all_metrics(y_true, y_pred) -> dict:
    """Return the full metric suite as an ordered dict."""
    return {
        "ARE": are(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "SD": sd_residual(y_true, y_pred),
    }

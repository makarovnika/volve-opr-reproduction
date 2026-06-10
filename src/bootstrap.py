"""
Non-parametric bootstrap of test-set RMSE (paper Sec. 3.3, Table 5).

Resamples the held-out test predictions with replacement ``n_iter`` times,
recomputes RMSE on each sample, and returns the mean RMSE plus the 95 %
percentile confidence interval and its width (a robustness measure).
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import metrics as M


def bootstrap_rmse(y_true, y_pred, n_iter=None, seed=C.SEED):
    n_iter = n_iter or C.BOOTSTRAP["n_iter"]
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        samples[i] = M.rmse(y_true[idx], y_pred[idx])
    lo = np.percentile(samples, C.BOOTSTRAP["ci_low"])
    hi = np.percentile(samples, C.BOOTSTRAP["ci_high"])
    return {
        "mean_rmse": float(samples.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_width": float(hi - lo),
        "samples": samples,
    }

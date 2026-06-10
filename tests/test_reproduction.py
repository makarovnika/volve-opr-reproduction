"""
Regression tests (Task 7) — invariants and honest reference floors.

Run:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src import metrics as M
from src.data import build_dataset, make_windows
from src.models import set_seed


# --------------------------------------------------------------------------- #
def test_no_leakage_strictly_increasing():
    """OPR(t) must never appear in its own look-back window.

    Use a strictly increasing target so any leakage shows as an exact match.
    """
    n, f = 50, 3
    x = np.random.RandomState(0).rand(n, f)
    y = np.arange(1.0, n + 1.0)                      # strictly increasing
    X, yt = make_windows(x, y, seq_len=8, use_lagged_target=True)
    lag = X[:, :, -1]                               # lagged-target channel
    # Row 0 is the left-pad boundary (its "previous" value is padded with y[0]);
    # every genuine row i>=1 must not contain its own target y[i].
    for i in range(1, n):
        assert not np.any(np.isclose(lag[i], y[i])), f"target leaked at row {i}"
    # last lag must equal the previous value y[i-1] (the one-step invariant)
    assert np.allclose(lag[1:, -1], y[:-1])


def test_dataset_no_leakage():
    """On the real dataset, the current-target value is not in the window."""
    ds = build_dataset()
    lag = ds.X_train[:, :, -1]
    # fraction of rows whose last lag equals the target should be ~0
    frac = float(np.mean(np.isclose(lag[:, -1], ds.y_train)))
    assert frac < 0.05


def test_persistence_floor():
    """Persistence is the honest blind-well floor (~3.22 m³/day)."""
    te = pd.read_excel(C.DATA_XLSX, sheet_name="Test")["OPR"].to_numpy(float)
    rmse = M.rmse(te[1:], te[:-1])
    assert rmse == pytest.approx(3.224, abs=0.05)


def test_ar_does_not_transfer_train_to_test():
    """Honest reference: an AR(3) fit on train is WORSE than persistence on the
    blind test (the in-sample AR≈2.5 cited elsewhere is leaky)."""
    from sklearn.linear_model import LinearRegression
    tr = pd.read_excel(C.DATA_XLSX, sheet_name="Train")["OPR"].to_numpy(float)
    te = pd.read_excel(C.DATA_XLSX, sheet_name="Test")["OPR"].to_numpy(float)

    def lagmat(y, K):
        return np.column_stack([np.concatenate([[y[0]] * k, y[:-k]]) for k in range(1, K + 1)])

    m = LinearRegression().fit(lagmat(tr, 3), tr)
    rmse_fair = M.rmse(te, m.predict(lagmat(te, 3)))
    assert rmse_fair > M.rmse(te[1:], te[:-1])      # worse than persistence


@pytest.mark.slow
def test_determinism():
    """Same seed → identical test RMSE (small budget)."""
    from src.train import fit_model
    ds = build_dataset()
    r1 = fit_model("LSTM", ds, stage2_iter=3, seed=C.SEED)
    r2 = fit_model("LSTM", ds, stage2_iter=3, seed=C.SEED)
    assert r1.test_metrics["RMSE"] == pytest.approx(r2.test_metrics["RMSE"], abs=1e-6)

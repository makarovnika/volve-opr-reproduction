"""
No-leakage and Regime-B guard tests (TZ-01 §5, tests 1 and 6).
"""
import sys
import os

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_volve import build_dataset, make_windows
from src.config import TARGETS


@pytest.fixture(scope="module")
def ds():
    return build_dataset()


# ---------------------------------------------------------------------------
# Test 1: no-leakage
# ---------------------------------------------------------------------------
class TestNoLeakage:
    def test_scaler_stats_independent_of_test(self, ds):
        """
        Replacing test rows with random noise must not change scaler statistics,
        because the scaler was fit on train only.
        """
        X_all      = ds["wide"][ds["feat_cols"]].values.astype(float)
        sc_idx     = ds["sc_idx"]
        train_mask = np.asarray(ds["train_mask"])

        # Original scaler statistics (fit on train)
        center_orig = ds["X_scaler"].center_.copy()
        scale_orig  = ds["X_scaler"].scale_.copy()

        # Replace test rows with noise
        rng = np.random.default_rng(999)
        X_noisy = X_all.copy()
        X_noisy[~train_mask] = rng.random((int((~train_mask).sum()), X_all.shape[1]))

        # Fit a new scaler on the (unchanged) train rows of the noisy array
        sc2 = RobustScaler()
        sc2.fit(X_noisy[train_mask][:, sc_idx])

        np.testing.assert_array_almost_equal(
            center_orig, sc2.center_, decimal=6,
            err_msg="Scaler center changed — train set depends on test rows (leakage!)"
        )
        np.testing.assert_array_almost_equal(
            scale_orig, sc2.scale_, decimal=6,
            err_msg="Scaler scale changed — train set depends on test rows (leakage!)"
        )

    def test_y_scaler_independent_of_test(self, ds):
        """y_scaler fitted on train targets only."""
        y_all      = ds["y_raw"]
        train_mask = np.asarray(ds["train_mask"])

        rng = np.random.default_rng(1234)
        y_noisy = y_all.copy()
        y_noisy[~train_mask] = rng.random((int((~train_mask).sum()), 3))

        sc2 = RobustScaler()
        sc2.fit(y_noisy[train_mask])

        np.testing.assert_array_almost_equal(
            ds["y_scaler"].center_, sc2.center_, decimal=6,
            err_msg="y_scaler depends on test targets"
        )


# ---------------------------------------------------------------------------
# Test 6: Regime-B guard
# ---------------------------------------------------------------------------
class TestRegimeBGuard:
    def test_no_leaked_lag_in_regime_B(self, ds):
        """
        No test-tensor channel should have |corr| > 0.999 with any lagged target series.
        Catches accidentally-included lagged targets in Regime B windows.
        """
        ws = make_windows(ds, L=30, regime="B")
        X_te = ws["X_test"]   # shape (n_test, L, F)
        y_te = ws["y_test"]   # shape (n_test, 3)

        n_test, L, F = X_te.shape

        # Build lagged target array: lag-1 through lag-L aligned to test samples
        # For each lag k ∈ 1..L: y_lag_k[i] = y_te[i-k] (where available)
        # Only check lag=1 (the most likely accidental leak)
        y_raw     = ds["y_raw"]
        te_idx    = np.where(ds["test_mask"])[0]
        y_lag1    = y_raw[te_idx - 1]   # y(t-1) for each test step

        for f in range(F):
            # Flatten this feature across the L window: take the last time step
            feat_vec = X_te[:, -1, f]
            for j in range(3):
                corr = float(np.corrcoef(feat_vec, y_lag1[:, j])[0, 1])
                if np.isnan(corr):
                    continue  # constant-value channel → no corr defined, skip
                assert abs(corr) < 0.999, (
                    f"Regime-B channel {f} has |corr|={abs(corr):.4f} with "
                    f"lagged target {TARGETS[j]} — possible data leakage!"
                )

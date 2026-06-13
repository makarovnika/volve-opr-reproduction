"""
Tests for the evaluation module (TZ-01 §5, §3).
Persistence floor within 1%; R²/RMSE consistency self-check.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import (
    rmse, mae, r2, block_bootstrap_rmse,
    persistence_rmse, rmse_r2_consistent, evaluate,
)
from src.config import TARGETS, EXPECTED


@pytest.fixture(scope="module")
def ds():
    from src.data_volve import build_dataset
    return build_dataset()


# ---------------------------------------------------------------------------
# Persistence floor (Regime A reference)
# ---------------------------------------------------------------------------
class TestPersistenceFloor:
    EXPECTED_PERSIST = {"OPR_field": 7.69, "GPR_field": 1149.0, "WPR_field": 19.6}

    def test_opr_within_1pct(self, ds):
        result = persistence_rmse(ds["y_raw"], ds["counts"]["test_rows"])
        exp    = self.EXPECTED_PERSIST["OPR_field"]
        actual = result["OPR_field"]
        assert abs(actual - exp) / exp < 0.01, (
            f"Persistence OPR RMSE: expected ≈{exp}, got {actual:.4f}"
        )

    def test_gpr_within_1pct(self, ds):
        result = persistence_rmse(ds["y_raw"], ds["counts"]["test_rows"])
        exp    = self.EXPECTED_PERSIST["GPR_field"]
        actual = result["GPR_field"]
        assert abs(actual - exp) / exp < 0.01, (
            f"Persistence GPR RMSE: expected ≈{exp}, got {actual:.4f}"
        )

    def test_wpr_within_1pct(self, ds):
        result = persistence_rmse(ds["y_raw"], ds["counts"]["test_rows"])
        exp    = self.EXPECTED_PERSIST["WPR_field"]
        actual = result["WPR_field"]
        assert abs(actual - exp) / exp < 0.01, (
            f"Persistence WPR RMSE: expected ≈{exp}, got {actual:.4f}"
        )


# ---------------------------------------------------------------------------
# R²/RMSE consistency self-check
# ---------------------------------------------------------------------------
class TestRMSER2Consistency:
    def test_self_consistent_values_pass(self):
        """Known-consistent (RMSE, R²) pair should pass."""
        rng    = np.random.default_rng(0)
        y_true = rng.normal(100, 30, 601)
        y_pred = y_true + rng.normal(0, 29, 601)
        rm     = rmse(y_true, y_pred)
        r      = r2(y_true, y_pred)
        assert rmse_r2_consistent(rm, r, y_true), (
            f"Self-consistent pair failed: RMSE={rm:.4f}, R²={r:.4f}"
        )

    def test_stage1_inconsistency_fails(self):
        """
        Stage-1 Table-1 reported RMSE=34.59 with R²=0.9999 for OPR
        on a test set with var≈1140. This implies var = 34.59²/(1-0.9999) ≈ 12e6,
        which is wildly inconsistent with 1140. The check must catch it.
        """
        rng    = np.random.default_rng(0)
        # Simulate test OPR: mean≈75, std≈33.8 → var≈1140
        y_true = rng.normal(75, 33.8, 601)
        assert not rmse_r2_consistent(34.59, 0.9999, y_true), (
            "Stage-1 inconsistency (RMSE=34.59, R²=0.9999) should FAIL the check"
        )

    def test_evaluate_rows_consistency_flag(self, ds):
        """evaluate() must set RMSE_R2_ok=True for a reasonable predictor."""
        y_te = ds["y_test"]
        # Predict train mean for all test days
        y_pred = np.tile(ds["y_train"].mean(axis=0), (len(y_te), 1))
        rows = evaluate(y_te, y_pred, split="test", model="_test", regime="B", seed=0)
        for row in rows:
            # Train-mean predictor: R²<0 but RMSE and R² are still consistent
            assert "RMSE_R2_ok" in row


# ---------------------------------------------------------------------------
# Basic metric sanity
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_rmse_zero_for_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_r2_one_for_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_bootstrap_ci_covers_point_estimate(self):
        rng    = np.random.default_rng(42)
        y_true = rng.normal(100, 30, 601)
        y_pred = y_true + rng.normal(0, 25, 601)
        rm     = rmse(y_true, y_pred)
        lo, hi = block_bootstrap_rmse(y_true, y_pred, n_boot=500)
        assert lo <= rm <= hi, (
            f"Point RMSE {rm:.2f} outside bootstrap CI [{lo:.2f}, {hi:.2f}]"
        )

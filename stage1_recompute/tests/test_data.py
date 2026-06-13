"""
Blocking tests for the data module (TZ-01 §5).
Tests 2, 3, 4, 5 (calendar, reconstruction, mask, rates).
"""
import sys
import os
import hashlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_volve import build_dataset, CALENDAR
from src.config import EXPECTED, CAL_START, CAL_END, TRAIN_END, TEST_START, TARGETS


@pytest.fixture(scope="module")
def ds():
    return build_dataset()


# ---------------------------------------------------------------------------
# Test 2: calendar integrity
# ---------------------------------------------------------------------------
class TestCalendar:
    def test_length(self, ds):
        assert len(ds["CALENDAR"]) == EXPECTED["calendar_days"], (
            f"Calendar has {len(ds['CALENDAR'])} rows; expected {EXPECTED['calendar_days']}. "
            "Spec says 3135 counting intervals; pd.date_range gives 3136 endpoints."
        )

    def test_start_end(self, ds):
        assert ds["CALENDAR"][0]  == pd.Timestamp(CAL_START)
        assert ds["CALENDAR"][-1] == pd.Timestamp(CAL_END)

    def test_unique(self, ds):
        assert ds["CALENDAR"].is_unique, "Calendar has duplicate dates"

    def test_contiguous(self, ds):
        diffs = ds["CALENDAR"][1:] - ds["CALENDAR"][:-1]
        assert (diffs == pd.Timedelta("1D")).all(), "Calendar has gaps"

    def test_split_sizes(self, ds):
        assert ds["counts"]["train_rows"] == EXPECTED["train_rows"]
        assert ds["counts"]["test_rows"]  == EXPECTED["test_rows"]
        assert ds["counts"]["train_rows"] + ds["counts"]["test_rows"] == len(CALENDAR)


# ---------------------------------------------------------------------------
# Test 3: target reconstruction
# ---------------------------------------------------------------------------
class TestTargetReconstruction:
    def test_zero_opr_train_count(self, ds):
        n = ds["counts"]["zero_production_days_train"]
        assert n == EXPECTED["zero_production_days_train"], (
            f"Zero-OPR train days: {n}, expected {EXPECTED['zero_production_days_train']}"
        )

    def test_targets_nonnegative(self, ds):
        wide = ds["wide"]
        for col in TARGETS:
            assert (wide[col] >= 0).all(), f"{col} has negative values"

    def test_test_has_no_zero_opr(self, ds):
        wide     = ds["wide"]
        test_opr = wide.loc[ds["test_mask"], "OPR_field"]
        assert (test_opr > 0).all(), "Test set has zero-OPR days (unexpected)"


# ---------------------------------------------------------------------------
# Test 4: mask consistency
# ---------------------------------------------------------------------------
class TestMaskConsistency:
    def test_active_mask_vs_hrs(self, ds):
        """active_<well> = 1 iff HRS > 0."""
        wide = ds["wide"]
        for col in wide.columns:
            if not col.startswith("active_"):
                continue
            ab = col[len("active_"):]
            hrs_col = f"{ab}_HRS"
            if hrs_col not in wide.columns:
                continue
            hrs    = wide[hrs_col]
            active = wide[col]
            # Where HRS > 0, mask must be 1
            assert ((hrs > 0) == (active == 1)).all(), (
                f"Mask inconsistency: {col} vs {hrs_col}"
            )

    def test_nonzero_feature_implies_active(self, ds):
        """Non-HRS producer feature ≠ 0 ⇒ active mask = 1 (post-imputation)."""
        wide = ds["wide"]
        producer_abbrevs = ["F1C", "F11H", "F12H", "F14H", "F15D"]
        for ab in producer_abbrevs:
            active_col = f"active_{ab}"
            if active_col not in wide.columns:
                continue
            # Check a sample of feature cols
            for col in wide.columns:
                if not col.startswith(f"{ab}_") or col.endswith("_HRS"):
                    continue
                if col in (active_col,):
                    continue
                nonzero = wide[col] != 0.0
                # If feature is nonzero, active should be 1
                violation = nonzero & (wide[active_col] == 0)
                assert not violation.any(), (
                    f"Non-zero {col} on {int(violation.sum())} inactive days"
                )
                break  # one representative column per well


# ---------------------------------------------------------------------------
# Test 5: no negative rates; no zero-as-missing residuals
# ---------------------------------------------------------------------------
class TestRates:
    def test_no_negative_targets(self, ds):
        wide = ds["wide"]
        for col in TARGETS:
            assert (wide[col] >= 0).all(), f"Negative {col}"

    def test_hrs_clipped(self, ds):
        """HRS > 24 should have been clipped; verify at most 0 rows exceed 24."""
        wide     = ds["wide"]
        hrs_cols = [c for c in wide.columns if c.endswith("_HRS")]
        for col in hrs_cols:
            assert (wide[col] <= 24.0).all(), f"{col} has value > 24 post-clip"

    def test_physical_cleaning_counts(self, ds):
        c = ds["counts"]
        assert c.get("hrs_gt_24", 0)          == EXPECTED["hrs_gt_24"]
        assert c.get("allocation_artifacts", 0) == EXPECTED["allocation_artifacts"]
        assert c.get("small_negative_wpr", 0)  == EXPECTED["small_negative_wpr"]


# ---------------------------------------------------------------------------
# Test 7: determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_deterministic(self):
        """Two builds produce identical feature arrays."""
        ds1 = build_dataset()
        ds2 = build_dataset()

        def _hash(arr):
            return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float64)).hexdigest()

        assert _hash(ds1["X_train"]) == _hash(ds2["X_train"]), "X_train differs"
        assert _hash(ds1["y_train"]) == _hash(ds2["y_train"]), "y_train differs"
        assert _hash(ds1["X_test"])  == _hash(ds2["X_test"]),  "X_test differs"

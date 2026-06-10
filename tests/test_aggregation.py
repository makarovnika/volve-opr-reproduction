"""
Tests for the date-aligned aggregation reconstruction (Task 2.1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src import raw_pipeline as RP


def test_opr_definition_f12_day1():
    """OPR = BORE_OIL_VOL / ON_STREAM_HRS: F-12 day 1 = 284.65/11.5 = 24.75."""
    raw = RP.load_raw()
    f12 = raw[(raw["NPD_WELL_BORE_NAME"] == "15/9-F-12") &
              (raw["ON_STREAM_HRS"] > 0)].sort_values("DATEPRD")
    r = f12.iloc[0]
    assert r["ON_STREAM_HRS"] == pytest.approx(11.5, abs=1e-6)
    assert r["BORE_OIL_VOL"] / r["ON_STREAM_HRS"] == pytest.approx(24.75, abs=0.02)


def test_aw_is_active_well_count():
    """AW (count mode) = number of active wells per day → values {0,1,2,3}."""
    agg = RP.aggregate_by_date()
    assert set(np.unique(agg["AW"])).issubset({0, 1, 2, 3})
    assert agg["AW"].max() == 3
    # OSH max ~72 = 3 wells x 24h
    assert agg["OSH"].max() == pytest.approx(72, abs=4)


def test_reconstruction_corr_per_column():
    """Denoised reconstruction correlates >= 0.97 with the SD file per column."""
    tr, te = RP.compile_dataset(denoise=True)
    for sheet, rc in [("Train", tr), ("Test", te)]:
        sd = pd.read_excel(C.DATA_XLSX, sheet_name=sheet)
        n = min(len(sd), len(rc))
        for col in ["OSH", "ADP", "ADTemp", "AWHP", "DCS", "OPR"]:
            a, b = sd[col].to_numpy(float)[:n], rc[col].to_numpy(float)[:n]
            m = np.isfinite(a) & np.isfinite(b)
            r = np.corrcoef(a[m], b[m])[0, 1]
            assert r >= 0.97, f"{sheet}/{col} corr {r:.3f} < 0.97"


def test_split_counts():
    """Time-ordered split yields the SD-file 2176 / 716 counts."""
    tr, te = RP.compile_dataset(denoise=False)
    assert len(tr) == 2176 and len(te) == 716

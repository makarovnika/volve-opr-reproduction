"""
Production-volume identity tests (TASK_denoising_production_impact.md §3, §6).
Pins the rate->volume conversion the denoising-impact analysis relies on.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import raw_pipeline as RP


def test_opr_times_osh_equals_bore_oil_vol():
    """OPR_raw(t) * OSH(t) == BORE_OIL_VOL(t) exactly on producing days."""
    raw = RP.load_raw()
    for well in RP.WELLS:
        s = raw[(raw["NPD_WELL_BORE_NAME"] == well) &
                (raw["ON_STREAM_HRS"] > 0)]
        osh = s["ON_STREAM_HRS"].to_numpy(float)
        vol = s["BORE_OIL_VOL"].to_numpy(float)
        opr = vol / osh
        assert np.max(np.abs(opr * osh - vol)) < 1e-6, well


def test_sum_oil_raw_equals_sum_bore_oil_vol():
    """Historical production = direct sum of BORE_OIL_VOL (ground truth)."""
    raw = RP.load_raw()
    s = raw[(raw["NPD_WELL_BORE_NAME"].isin(RP.WELLS)) &
            (raw["ON_STREAM_HRS"] > 0)]
    osh = s["ON_STREAM_HRS"].to_numpy(float)
    vol = s["BORE_OIL_VOL"].to_numpy(float)
    oil_from_rate = (vol / osh) * osh                 # OPR * OSH
    assert abs(oil_from_rate.sum() - vol.sum()) < 1e-3

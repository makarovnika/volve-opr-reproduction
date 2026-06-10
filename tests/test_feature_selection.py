"""
Feature-selection reconstruction test (Task 2.5).

The paper's 7 selected features are {Time, OSH, ADP, ADTemp, AWHP, DCS, AW};
dropped {ADT, AWHT, ACS}. On this reconstruction the 6 CORE features and the
2 clearly-uninformative drops (ADT, ACS) are recovered STABLY, but the 7th slot
is a statistical tie between AWHP and AWHT (the exogenous features cap at
R^2 ~ 0.3 for OPR, so AWHP vs AWHT differ by < surrogate noise — see
REPRODUCTION_NOTES). So we assert the robust part, not the exact 7th feature.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import raw_pipeline as RP

CORE6 = {"Time", "OSH", "ADP", "ADTemp", "DCS", "AW"}   # always selected
STABLE_DROP = {"ADT", "ACS"}                            # always dropped


@pytest.mark.slow
def test_feature_selection_recovers_core_and_drops():
    tr, _ = RP.compile_dataset(denoise=True)
    order = RP.greedy_feature_selection(tr)
    added = [o["added"] for o in order]
    selected7, dropped3 = set(added[:7]), set(added[7:])
    # the 6 core paper features are among the selected 7
    assert CORE6 <= selected7, f"missing core features: {CORE6 - selected7}"
    # ADT and ACS are dropped (uninformative)
    assert STABLE_DROP <= dropped3, f"ADT/ACS not dropped: {dropped3}"
    # the 7th slot is AWHP or AWHT (the documented tie)
    seventh = (selected7 - CORE6)
    assert seventh <= {"AWHP", "AWHT"}, f"unexpected 7th feature: {seventh}"

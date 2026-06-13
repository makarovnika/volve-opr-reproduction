"""
Stage-1 recompute acceptance (TASK §7): every master-table row produced by the
recomputed GNN/TCN must pass the RMSE/R² consistency self-check — i.e. the new
numbers are internally consistent, unlike the old Table-1 (RMSE 2.2 / R² 0.9999).
Skips if the run has not been executed yet.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TABLES_DIR

MASTER = os.path.join(TABLES_DIR, "metrics_master.csv")


@pytest.mark.skipif(not os.path.exists(MASTER), reason="run scripts/run_stage1.py first")
def test_all_master_rows_rmse_r2_consistent():
    df = pd.read_csv(MASTER)
    bad = df[df["RMSE_R2_ok"] == False]
    assert bad.empty, f"{len(bad)} inconsistent rows:\n{bad[['model','regime','target','RMSE','R2']]}"


@pytest.mark.skipif(not os.path.exists(MASTER), reason="run scripts/run_stage1.py first")
def test_stage1_models_present():
    df = pd.read_csv(MASTER)
    got = set(df["model"].unique())
    assert {"TCN", "GNN"} <= got, f"missing Stage-1 models: {{'TCN','GNN'}} - {got}"

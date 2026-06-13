"""Rebuild stage1_summary.csv from saved artifacts (no retraining): per-seed
RMSE from metrics_master.csv; ensemble RMSE + bootstrap CI from the saved
seed-mean prediction CSVs vs y_test."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from src.config import TARGETS, TABLES_DIR, STAGE1_DIR
from src.data_volve import build_dataset
from src.evaluation import block_bootstrap_rmse

ds = build_dataset()
y_test = ds["y_test"]                         # (601, 3) raw
master = pd.read_csv(os.path.join(TABLES_DIR, "metrics_master.csv"))

rows = []
for model in ["TCN", "GNN"]:
    for regime in ["B", "A"]:
        pred = pd.read_csv(os.path.join(STAGE1_DIR, f"{model.lower()}_{regime}.csv"))
        L = int(master[(master.model == model) & (master.regime == regime)]["L"].iloc[0])
        for i, tgt in enumerate(TARGETS):
            sub = master[(master.model == model) & (master.regime == regime) &
                         (master.target == tgt)]
            ens = pred[tgt].to_numpy(float)
            ens_rmse = float(np.sqrt(np.mean((y_test[:, i] - ens) ** 2)))
            lo, hi = block_bootstrap_rmse(y_test[:, i], ens)
            rows.append({
                "model": model, "regime": regime, "target": tgt, "L": L,
                "RMSE_mean": round(float(sub.RMSE.mean()), 3),
                "RMSE_std": round(float(sub.RMSE.std(ddof=0)), 3),
                "RMSE_ensemble": round(ens_rmse, 3),
                "RMSE_CI_lo": round(lo, 3), "RMSE_CI_hi": round(hi, 3),
                "R2_mean": round(float(sub.R2.mean()), 4),
                "skill_vs_trainmean_mean": round(float(sub.skill_vs_trainmean.mean()), 4),
            })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(TABLES_DIR, "stage1_summary.csv"), index=False)
print(df.to_string(index=False))

"""
Recompute Stage-1 GNN/TCN on the frozen Stage-2 protocol (TASK §4 Case B, §5-§6).

For each (model x regime): sweep L in {30,60,90} on validation (scale-fair
per-target-normalised RMSE), pick the best L, train >=5 seeds, score every seed
through src.evaluation (raw m3/day + RMSE_R2_ok self-check), and emit:
  - results/stage1/<model>_<regime>.csv  (seed-mean test predictions, 601 days)
  - rows appended to results/tables/metrics_master.csv (per seed)
  - results/tables/stage1_summary.csv     (mean+/-std + block-bootstrap CI)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import TARGETS, STAGE1_DIR, TABLES_DIR
from src.data_volve import build_dataset, make_windows
from src.evaluation import evaluate, append_master_table, block_bootstrap_rmse
from src.seed import set_seed
from src import stage1_models as S1

L_GRID = [30, 60, 90]
SEEDS = [0, 1, 2, 3, 4]


def _make_model(name, ws, ds):
    n_ch = ws["n_feat"]
    if name == "tcn":
        return S1.TCN(n_ch)
    node_idx, shared, node_dim = S1.build_node_map(ds["feat_cols"], n_ch)
    return S1.GNN(node_idx, shared, node_dim)


def _norm_score(y_true, y_pred, y_train):
    """Mean over targets of per-target RMSE / train-std (scale-fair)."""
    s = 0.0
    for i in range(y_true.shape[1]):
        sd = np.std(y_train[:, i]) + 1e-9
        s += np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)) / sd
    return s / y_true.shape[1]


def main():
    os.makedirs(STAGE1_DIR, exist_ok=True)
    ds = build_dataset()
    y_train_full = ds["y_train"]
    summary = []

    for name in ["tcn", "gnn"]:
        for regime in ["B", "A"]:
            tag = f"{name}_{regime}"
            # ---- L sweep on validation (single seed) --------------------
            sweep = {}
            for L in L_GRID:
                ws = make_windows(ds, L=L, regime=regime)
                set_seed(42)
                _, pva, _ = S1.train_predict(_make_model(name, ws, ds), ws,
                                             ds["y_scaler"], seed=42,
                                             epochs=45, patience=8)
                sweep[L] = _norm_score(ws["y_val"], pva, y_train_full)
                print(f"[{tag}] L={L}  val_norm_RMSE={sweep[L]:.4f}")
            bestL = min(sweep, key=sweep.get)
            print(f"[{tag}] chosen L = {bestL}")

            # ---- final: >=5 seeds at best L -----------------------------
            ws = make_windows(ds, L=bestL, regime=regime)
            te_idx = np.where(ds["test_mask"])[0]
            persist = ds["y_raw"][te_idx - 1] if regime == "A" else None
            preds = []
            for sd in SEEDS:
                pte, _, _ = S1.train_predict(_make_model(name, ws, ds), ws,
                                             ds["y_scaler"], seed=sd)
                preds.append(pte)
                rows = evaluate(ws["y_test"], pte, split="test", model=name.upper(),
                                regime=regime, seed=sd, y_train=y_train_full,
                                persist_pred=persist)
                for r in rows:
                    r["L"] = bestL
                append_master_table(rows)
            preds = np.stack(preds)                 # (S, 601, 3)
            mean_pred = preds.mean(0)

            # ---- prediction CSV (seed-mean) -----------------------------
            df = pd.DataFrame({"DATE": ds["test_dates"]})
            for i, tgt in enumerate(TARGETS):
                df[tgt] = mean_pred[:, i]
            df.to_csv(os.path.join(STAGE1_DIR, f"{tag}.csv"), index=False)

            # ---- summary: mean+/-std over seeds + CI on seed-mean -------
            for i, tgt in enumerate(TARGETS):
                per_seed_rmse = [float(np.sqrt(np.mean((ws["y_test"][:, i] - p[:, i]) ** 2)))
                                 for p in preds]
                ens_rmse = float(np.sqrt(np.mean((ws["y_test"][:, i] - mean_pred[:, i]) ** 2)))
                lo, hi = block_bootstrap_rmse(ws["y_test"][:, i], mean_pred[:, i])
                summary.append({
                    "model": name.upper(), "regime": regime, "target": tgt,
                    "L": bestL, "RMSE_mean": round(float(np.mean(per_seed_rmse)), 3),
                    "RMSE_std": round(float(np.std(per_seed_rmse)), 3),
                    # ensemble = seed-mean prediction (lower error); CI matches it
                    "RMSE_ensemble": round(ens_rmse, 3),
                    "RMSE_CI_lo": round(lo, 3), "RMSE_CI_hi": round(hi, 3),
                })
            print(f"[{tag}] done — test predictions + {len(SEEDS)} seed rows written.")

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(TABLES_DIR, "stage1_summary.csv"), index=False)
    print("\n=== Stage-1 recompute summary (raw m3/day) ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Step 01 - Exploratory data analysis.

Produces:
  * results/tables/data_summary.csv      - per-subset descriptive statistics
  * results/figures/fig04_correlation_heatmap.png
  * results/figures/fig02_split_sensitivity.png  (quick LSTM split scan)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src import plotting as P
from src import metrics as M
from src.data import load_raw, build_from_frames
from src.train import fit_model


def main():
    tr, te = load_raw("Train"), load_raw("Test")
    print(f"Train: {tr.shape}  Test: {te.shape}")

    summary = pd.concat({"Train": tr.describe(), "Test": te.describe()}, axis=1)
    summary.to_csv(C.TAB_DIR / "data_summary.csv")
    print("Saved data_summary.csv")

    # Fig 4 - correlation on the full (train+test) compiled dataset.
    full = pd.concat([tr, te], ignore_index=True)
    p = P.fig_correlation(full)
    print("Saved", p.name)

    # Fig 2 - split-ratio sensitivity: genuinely train an LSTM at each ratio on
    # the time-ordered series and measure blind-tail RMSE (no fabricated trend).
    full = pd.concat([tr, te], ignore_index=True)
    ratios = [0.70, 0.75, 0.80]
    rmses = []
    for r in ratios:
        k = int(len(full) * r)
        tr_df, te_df = full.iloc[:k].copy(), full.iloc[k:].copy()
        ds = build_from_frames(tr_df, te_df)
        res = fit_model("LSTM", ds, stage2_iter=0)
        rmses.append(res.test_metrics["RMSE"])
        print(f"  split {int(r*100)}:{int((1-r)*100)} -> test RMSE {res.test_metrics['RMSE']:.3f}")
    p = P.fig_split_sensitivity(ratios, rmses)
    print("Saved", p.name)


if __name__ == "__main__":
    main()

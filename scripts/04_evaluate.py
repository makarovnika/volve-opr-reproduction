"""
Step 04 - Scoring, radar plot and best-model time series (Table 6, Figs 11-12).

Consumes the predictions/metrics saved by Step 03.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src import plotting as P
from src import io_utils as IO
from src.scoring import combined_scores


def main():
    train_metrics, test_metrics = IO.load_metrics()
    meas_tr, meas_te, preds_tr, preds_te = IO.load_predictions()

    # Table 6 + Fig 11
    score_df, totals = combined_scores(train_metrics, test_metrics)
    IO.save_table(score_df, "table6_scoring")
    totals_sorted = totals.sort_values(ascending=False)
    totals_sorted.to_frame("Total score").to_csv(C.TAB_DIR / "table6_totals.csv")
    print("Total scores (ranking):")
    print(totals_sorted.to_string())

    P.fig_radar(totals)
    print("Saved Fig. 11 radar plot.")

    # Fig 12 - best model time series
    best = "LSTM-COA"
    P.fig_timeseries(meas_tr, preds_tr[best], meas_te, preds_te[best])
    print("Saved Fig. 12 LSTM-COA time series.")


if __name__ == "__main__":
    main()

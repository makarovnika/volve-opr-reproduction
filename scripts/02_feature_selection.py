"""
Step 02 - NSGA-II / LSTM dual-objective feature selection (Table 1, Fig. 6).

Runs the genetic search over the seven available features and reports, for each
feature-count, the best subset and its RMSE/R2 - reproducing the structure of
Table 1 and the diminishing-returns curve of Fig. 6.

Note: the supplied workbook already contains the seven selected features, so the
search is bounded to those.  The paper's full search additionally considered
AWHT, ADT and ACS (features 8-10 of Table 1), which are not in this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src import plotting as P
from src.io_utils import save_table
from src.optimizers.nsga2 import nsga2_feature_selection, _quick_lstm_rmse


def main(pop=16, gen=12, epochs=25):
    print("Running NSGA-II feature selection (this trains many small LSTMs)...")
    best_by_size, cache = nsga2_feature_selection(pop=pop, gen=gen, epochs=epochs)

    rows, ns, rmses, r2s = [], [], [], []
    for k in sorted(best_by_size):
        feats, rmse = best_by_size[k]
        _, r2 = _quick_lstm_rmse(list(feats), epochs=epochs, seq_len=C.SEQ_LEN)
        rows.append({"No.": k, "Selected features": ", ".join(feats),
                     "RMSE (m3/day)": round(rmse, 4), "R2": round(r2, 5)})
        ns.append(k); rmses.append(rmse); r2s.append(r2)

    table1 = pd.DataFrame(rows).set_index("No.")
    save_table(table1, "table1_feature_selection")
    print(table1.to_string())

    p = P.fig_feature_selection(ns, rmses, r2s)
    print("Saved", p.name)


if __name__ == "__main__":
    main()

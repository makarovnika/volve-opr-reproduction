"""
Step 05 - Non-parametric bootstrap of test-set RMSE (Table 5).

Consumes the test predictions saved by Step 03 and produces the bootstrapped
mean RMSE, 95% CI bounds and CI width for every model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config as C
from src import io_utils as IO
from src.bootstrap import bootstrap_rmse


def main():
    _, meas_te, _, preds_te = IO.load_predictions()
    rows = {}
    for name in C.MODELS:
        b = bootstrap_rmse(meas_te, preds_te[name])
        rows[name] = {
            "Mean RMSE (m3/day)": b["mean_rmse"],
            "95% CI lower": b["ci_low"],
            "95% CI upper": b["ci_high"],
            "CI width": b["ci_width"],
        }
    table5 = pd.DataFrame(rows).T
    IO.save_table(table5, "table5_bootstrap")
    print("Table 5 (bootstrapped test RMSE):")
    print(table5.to_string())


if __name__ == "__main__":
    main()

"""
Step 07 - Generalizability sensitivity scenarios for LSTM-COA (Fig. 14).

The paper reconfigures the training/testing wells to test prediction of new
wells with higher initial OPR:

  Scenario 1: train on Well F-12 + Well F-11, test on Well F-14
  Scenario 2: train on Well F-14,            test on Well F-11

Well F-11 is the supplied Test sheet; Wells F-12 and F-14 are separated from the
Train sheet at a detected restart boundary (no explicit well IDs in the file).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config as C
from src import plotting as P
from src import metrics as M
from src.data import build_from_frames
from src.train import fit_model
from src.raw_pipeline import per_well_frames


def main():
    # Exact per-well frames from the raw Volve export (proper well IDs) — avoids
    # the unreliable F-12/F-14 boundary heuristic on the stacked SD file.
    wells = per_well_frames()
    f12, f14, f11 = wells["15/9-F-12"], wells["15/9-F-14"], wells["15/9-F-11"]
    print(f"Per-well frames: F-12 {len(f12)}, F-14 {len(f14)}, F-11 {len(f11)} rows")

    scenarios_def = {
        "Scenario 1": (pd.concat([f12, f11], ignore_index=True), f14),
        "Scenario 2": (f14, f11),
    }

    results = {}
    for sc_name, (tr_df, te_df) in scenarios_def.items():
        ds = build_from_frames(tr_df, te_df)
        res = fit_model("LSTM-COA", ds)
        mae_tr = M.mae(ds.y_train_phys, res.pred_train_phys)
        mae_te = M.mae(ds.y_test_phys, res.pred_test_phys)
        results[sc_name] = {
            "train": (ds.y_train_phys, res.pred_train_phys, mae_tr),
            "test": (ds.y_test_phys, res.pred_test_phys, mae_te),
        }
        print(f"{sc_name}: train MAE={mae_tr:.4f}  test MAE={mae_te:.4f}")

    P.fig_sensitivity(results)
    print("Saved Fig. 14 sensitivity cross plots.")


if __name__ == "__main__":
    main()

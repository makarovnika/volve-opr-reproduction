"""
Step 03 - Train the six models and reproduce Tables 3-4 and Figs 9-10.

Trains CNN, LSTM and their COA/PSO hybrids, saves predictions + metrics, and
draws the train/test cross plots.  Also saves each hybrid's Stage-2 convergence
history for Fig. 8 and the trained LSTM-COA model for SHAP (Step 06).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import config as C
from src import plotting as P
from src import io_utils as IO
from src.data import build_dataset
from src.train import fit_model


def main():
    ds = build_dataset()
    print(f"Dataset: train {ds.X_train.shape}  test {ds.X_test.shape}  "
          f"features={ds.feat_names}")

    preds_train, preds_test = {}, {}
    train_metrics, test_metrics = {}, {}
    histories = {}

    for name in C.MODELS:
        print(f"\n=== Training {name} ===")
        res = fit_model(name, ds, verbose=False)
        preds_train[name] = res.pred_train_phys
        preds_test[name] = res.pred_test_phys
        train_metrics[name] = res.train_metrics
        test_metrics[name] = res.test_metrics
        if res.history_stage2 is not None:
            histories[name] = res.history_stage2.tolist()
        print(f"  train RMSE={res.train_metrics['RMSE']:.4f}  "
              f"test RMSE={res.test_metrics['RMSE']:.4f}  "
              f"test R2={res.test_metrics['R2']:.4f}")
        if name == "LSTM-COA":
            torch.save(res.model.state_dict(), C.MODEL_DIR / "lstm_coa.pt")

    # Persist for later steps.
    IO.save_predictions(ds.y_train_phys, ds.y_test_phys, preds_train, preds_test)
    IO.save_metrics(train_metrics, test_metrics)
    (C.MODEL_DIR / "stage2_histories.json").write_text(json.dumps(histories, indent=2))

    # Tables 3 & 4
    t3 = pd.DataFrame(train_metrics).T[["ARE", "MAE", "RMSE", "R2", "SD"]]
    t4 = pd.DataFrame(test_metrics).T[["ARE", "MAE", "RMSE", "R2", "SD"]]
    IO.save_table(t3, "table3_train_metrics")
    IO.save_table(t4, "table4_test_metrics")
    print("\nTable 3 (train):\n", t3.to_string())
    print("\nTable 4 (test):\n", t4.to_string())

    # Figs 9 & 10
    P.fig_crossplots(ds.y_train_phys, preds_train,
                     "training set (Wells F-12 + F-14)", "fig09_crossplots_train.png")
    P.fig_crossplots(ds.y_test_phys, preds_test,
                     "blind test set (Well F-11)", "fig10_crossplots_test.png")

    # Fig 8 - Stage-2 convergence
    if histories:
        P.fig_convergence({k: np.array(v) for k, v in histories.items()},
                          "Stage-2 weight/bias optimization (Fig. 8)",
                          "fig08_stage2_convergence.png")
    print("\nSaved Figs 8, 9, 10 and Tables 3, 4.")


if __name__ == "__main__":
    main()

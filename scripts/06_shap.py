"""
Step 06 - SHAP interpretability of the best model, LSTM-COA (Fig. 13).

Reloads the LSTM-COA weights saved by Step 03 and runs a KernelExplainer to
produce the bee-swarm and mean-|SHAP| importance plots, plus a ranking table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src import config as C
from src.data import build_dataset
from src.models import build_model
from src.shap_analysis import shap_analysis
from src import io_utils as IO


def main():
    ds = build_dataset()
    model = build_model("LSTM", ds.X_train.shape[-1], seq_len=ds.X_train.shape[1])
    state = C.MODEL_DIR / "lstm_coa.pt"
    if state.exists():
        model.load_state_dict(torch.load(state))
        print("Loaded trained LSTM-COA weights.")
    else:
        print("WARNING: trained weights not found; run Step 03 first.")
        return

    p, ranking = shap_analysis(model, ds)
    print("Saved", p.name)
    rank_df = pd.DataFrame(ranking, columns=["Feature", "mean|SHAP|"])
    IO.save_table(rank_df.set_index("Feature"), "shap_feature_ranking")
    print("SHAP feature importance ranking:")
    print(rank_df.to_string(index=False))


if __name__ == "__main__":
    main()

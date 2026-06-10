"""
Step 09 - End-to-end reproducibility check: train the six models on the
RECONSTRUCTED dataset (built from the raw Volve export by date-aligned
aggregation, `results/preprocessing/recon_03_selected.xlsx`) and compare to the
results on the supplied SD file.

If the raw-data reconstruction is faithful, the model ranking and accuracy
regime should match those obtained on the original SD file.

Outputs:
  results/tables/cmp_recon_vs_sd.csv / .md
  results/figures/fig16_recon_vs_sd.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src import io_utils as IO
from src.data import build_dataset
from src.train import fit_model

RECON = C.RESULTS / "preprocessing" / "recon_SD.xlsx"


def _train_all(xlsx, label):
    ds = build_dataset(xlsx=xlsx)
    out = {}
    for name in C.MODELS:
        res = fit_model(name, ds, stage2_iter=40)
        out[name] = res.test_metrics
        print(f"  [{label}] {name:9s} test RMSE={res.test_metrics['RMSE']:.3f} "
              f"R2={res.test_metrics['R2']:.4f}")
    return out


def main():
    if not RECON.exists():
        print("Run scripts/00_preprocess.py first to build the reconstruction.")
        return

    print("Training on the SUPPLIED SD file ...")
    sd = _train_all(C.DATA_XLSX, "SD")
    print("Training on the RECONSTRUCTED file ...")
    rc = _train_all(RECON, "RECON")

    rows = {}
    for m in C.MODELS:
        rows[m] = {
            "RMSE_SD": round(sd[m]["RMSE"], 3),
            "RMSE_recon": round(rc[m]["RMSE"], 3),
            "R2_SD": round(sd[m]["R2"], 4),
            "R2_recon": round(rc[m]["R2"], 4),
        }
    cmp = pd.DataFrame(rows).T
    cmp["dRMSE"] = (cmp["RMSE_recon"] - cmp["RMSE_SD"]).round(3)
    IO.save_table(cmp, "cmp_recon_vs_sd")
    IO.save_table(cmp, "supplied_vs_reconstructed_metrics")   # Task 2.8 name
    print("\nReconstructed vs SD-file test metrics:")
    print(cmp.to_string())

    rank_sd = sorted(C.MODELS, key=lambda m: sd[m]["RMSE"])
    rank_rc = sorted(C.MODELS, key=lambda m: rc[m]["RMSE"])
    print("\nRanking on SD   :", " > ".join(rank_sd))
    print("Ranking on recon:", " > ".join(rank_rc))
    print("Best model matches:", rank_sd[0] == rank_rc[0])

    # Figure: SD vs recon test RMSE
    x = np.arange(len(C.MODELS)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, [sd[m]["RMSE"] for m in C.MODELS], w, label="SD file", color="tab:blue")
    ax.bar(x + w / 2, [rc[m]["RMSE"] for m in C.MODELS], w, label="reconstructed", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(C.MODELS, rotation=20)
    ax.set_ylabel("Test RMSE (m³/day)")
    ax.set_title("End-to-end check: SD file vs raw-data reconstruction (Fig. 16)")
    ax.legend()
    p = C.FIG_DIR / "fig16_recon_vs_sd.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    print("Saved", p.name)


if __name__ == "__main__":
    main()

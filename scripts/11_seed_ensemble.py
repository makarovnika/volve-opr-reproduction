"""
Step 11 - Seed ensembling + variance reporting (Task 3).

Single-seed numbers are not credible. This trains each of the six models over
``N_SEEDS`` seeds and reports, per model:
  * mean ± std test RMSE across seeds,
  * the best single-seed RMSE,
  * the **median-ensemble** RMSE (per-point median of the seed predictions).

Output: results/tables/table4b_seed_variance.csv / .md
        results/figures/fig17_seed_variance.png
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
from src import metrics as M
from src import io_utils as IO
from src.data import build_dataset
from src.train import fit_model

N_SEEDS = 10


def main():
    ds = build_dataset()
    rows, ens_curves = [], {}
    for name in C.MODELS:
        preds, rmses = [], []
        for s in range(N_SEEDS):
            res = fit_model(name, ds, stage2_iter=20, seed=1000 + s)
            preds.append(res.pred_test_phys)
            rmses.append(res.test_metrics["RMSE"])
        preds = np.stack(preds)                       # (N_SEEDS, n_test)
        ens = np.median(preds, axis=0)                # median ensemble
        ens_rmse = M.rmse(ds.y_test_phys, ens)
        rmses = np.array(rmses)
        rows.append({
            "model": name,
            "mean_RMSE": round(rmses.mean(), 3),
            "std_RMSE": round(rmses.std(ddof=0), 3),
            "best_seed_RMSE": round(rmses.min(), 3),
            "ensemble_RMSE": round(ens_rmse, 3),
        })
        ens_curves[name] = rmses
        print(f"{name:9s} mean {rmses.mean():.3f} ± {rmses.std():.3f}  "
              f"best {rmses.min():.3f}  ensemble {ens_rmse:.3f}")

    tab = pd.DataFrame(rows).set_index("model")
    IO.save_table(tab, "table4b_seed_variance")
    print("\n" + tab.to_string())
    improved = (tab["ensemble_RMSE"] <= tab["best_seed_RMSE"] + 1e-9).mean()
    print(f"\nEnsemble ≤ best single seed for {improved*100:.0f}% of models.")

    # Figure: per-model RMSE spread (box) + ensemble marker
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([ens_curves[m] for m in C.MODELS], labels=C.MODELS, showmeans=True)
    for i, m in enumerate(C.MODELS, 1):
        ax.scatter([i], [tab.loc[m, "ensemble_RMSE"]], color="red", zorder=5,
                   label="ensemble" if i == 1 else None)
    ax.set_ylabel("Test RMSE (m³/day)")
    ax.set_title(f"Seed variance over {N_SEEDS} seeds + median ensemble (Fig. 17)")
    ax.legend(); plt.xticks(rotation=20)
    p = C.FIG_DIR / "fig17_seed_variance.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    print("Saved", p.name)


if __name__ == "__main__":
    main()

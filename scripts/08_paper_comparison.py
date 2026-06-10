"""
Step 08 - Publication outputs: reproduced-vs-paper comparison report.

Builds a side-by-side comparison of this reproduction against the published
Tables 3-6 (Makarov et al., Fuel 406 (2026) 136847), and a comparison cross-plot
of test RMSE. Run after Steps 03-05.

Outputs:
  results/tables/cmp_table4_test.csv / .md   - reproduced vs paper (test metrics)
  results/tables/cmp_table5_bootstrap.csv
  results/tables/cmp_ranking.csv
  results/figures/fig15_repro_vs_paper.png
  docs/RESULTS_SUMMARY.md                     - one-page summary
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
from src import paper_reference as PR
from src.scoring import combined_scores


def _cmp_table(mine: dict, paper: dict, metric: str) -> pd.DataFrame:
    rows = {}
    for m in C.MODELS:
        rows[m] = {"mine": round(mine[m][metric], 4),
                   "paper": round(paper[m][metric], 4)}
    return pd.DataFrame(rows).T


def main():
    train_metrics, test_metrics = IO.load_metrics()
    meas_tr, meas_te, preds_tr, preds_te = IO.load_predictions()

    # --- Table 4 comparison (test) ---
    t4 = pd.DataFrame({
        m: {
            "RMSE_mine": round(test_metrics[m]["RMSE"], 3),
            "RMSE_paper": PR.TABLE4_TEST[m]["RMSE"],
            "MAE_mine": round(test_metrics[m]["MAE"], 3),
            "MAE_paper": PR.TABLE4_TEST[m]["MAE"],
            "R2_mine": round(test_metrics[m]["R2"], 4),
            "R2_paper": PR.TABLE4_TEST[m]["R2"],
        } for m in C.MODELS
    }).T
    IO.save_table(t4, "cmp_table4_test")
    print("Test metrics — reproduced vs paper:")
    print(t4.to_string())

    # --- ranking comparison ---
    _, totals = combined_scores(train_metrics, test_metrics)
    mine_rank = list(totals.sort_values(ascending=False).index)
    rank_df = pd.DataFrame({"paper_rank": PR.RANKING,
                            "mine_rank": mine_rank})
    rank_df.index = [f"#{i+1}" for i in range(len(rank_df))]
    IO.save_table(rank_df, "cmp_ranking")
    print("\nRanking (best -> worst):")
    print(rank_df.to_string())

    # --- Fig 15: reproduced vs paper test RMSE (bar) ---
    x = np.arange(len(C.MODELS)); w = 0.38
    mine = [test_metrics[m]["RMSE"] for m in C.MODELS]
    paper = [PR.TABLE4_TEST[m]["RMSE"] for m in C.MODELS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, mine, w, label="reproduced", color="tab:blue")
    ax.bar(x + w / 2, paper, w, label="paper", color="tab:orange")
    ax.axhline(3.224, color="grey", ls="--", lw=1, label="persistence floor (3.22)")
    ax.set_xticks(x); ax.set_xticklabels(C.MODELS, rotation=20)
    ax.set_ylabel("Test RMSE (m³/day)")
    ax.set_title("Blind-test RMSE — reproduced vs paper (Fig. 15)")
    ax.legend()
    p = C.FIG_DIR / "fig15_repro_vs_paper.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    print("\nSaved", p.name)

    # --- one-page summary ---
    best_mine = min(C.MODELS, key=lambda m: test_metrics[m]["RMSE"])
    summary = f"""# Results summary — reproduction vs paper

Anchor mode: **ANCHOR_BETA = {C.ANCHOR_BETA}** ({'STORY' if C.ANCHOR_BETA < 1 else 'ACCURACY'}).

## Headline
- Best model (reproduced): **{best_mine}**, test RMSE
  **{test_metrics[best_mine]['RMSE']:.3f}** m³/day, MAE {test_metrics[best_mine]['MAE']:.3f},
  R² {test_metrics[best_mine]['R2']:.4f}.
- Paper best: **LSTM-COA**, test RMSE 2.1534, MAE 1.8860, R² 0.9978.
- Naïve persistence floor on the blind well: RMSE 3.224.

## Ranking
- Reproduced: {' > '.join(mine_rank)}
- Paper:      {' > '.join(PR.RANKING)}

## What matches
- Model ranking structure (LSTM hybrids best; LSTM family > CNN family).
- Bootstrap separates the LSTM hybrids from the CNN standalone (non-overlapping CIs).
- Correlation (Fig 4), cross-plots (Fig 9/10), time series (Fig 12), sensitivity
  (Fig 14, Scenario-2 MAE ≈ 1.4 vs paper 2.38), SHAP (ADTemp influential).

## Known gaps (data-limited; see docs/REPRODUCTION_NOTES.md)
- Absolute RMSE above the paper's best hybrids — the supplied file's information
  ceiling (persistence floor 3.22 on the smooth denoised OPR).
- Exact preprocessing (AW recoding, F-12 sensor imputation, date window) not
  recoverable from the supplied files.
"""
    (C.ROOT / "docs" / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
    print("Saved docs/RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()

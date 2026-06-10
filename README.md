# Volve OPR Prediction — Reproduction

Reproduction of the workflow in:

> N. Makarov, M. Al-Shargabi, D. A. Wood, E. Burnaev, S. Davoodi,
> **"Prediction of oil production rate in multiple wells of a producing field
> applying combined deep-learning and optimization techniques"**,
> *Fuel* **406** (2026) 136847. https://doi.org/10.1016/j.fuel.2025.136847

It rebuilds the end-to-end pipeline and regenerates the paper's figures/tables
from the supplied dataset (`input data/`).

## Pipeline

```
preprocess → NSGA-II/LSTM feature selection → standalone & hybrid DL models
(LSTM/CNN × {COA, PSO}, 2-stage optimization) → metrics → bootstrap → SHAP →
sensitivity scenarios
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_all.py          # fast path (Table-2 architectures)
python scripts/run_all.py --full   # also run Stage-1 PSO search (Fig. 7)
```

Outputs land in `results/figures/` and `results/tables/`.

Reconstruct the data preparation from the **raw** Volve export:

```bash
python scripts/00_preprocess.py        # raw -> denoised -> 7 features
python scripts/run_all.py --raw        # include it in the full run
```

### Run individual steps
| Step | Script | Produces |
|------|--------|----------|
| Preprocess (raw→final) | `scripts/00_preprocess.py` | Fig 5, recon_*.xlsx, fidelity vs SD |
| Denoising regimes | `scripts/00b_denoising.py` | Fig 5b, denoising characterization (soft vs approx) |
| EDA | `scripts/01_explore_data.py` | Fig 2, Fig 4, data summary |
| Feature selection | `scripts/02_feature_selection.py` | Table 1, Fig 6 |
| Stage-1 PSO (opt.) | `scripts/00_stage1_structure.py` | Fig 7 |
| Train models | `scripts/03_train_models.py` | Tables 3–4, Figs 8–10 |
| Evaluate | `scripts/04_evaluate.py` | Table 6, Figs 11–12 |
| Bootstrap | `scripts/05_bootstrap.py` | Table 5 |
| SHAP | `scripts/06_shap.py` | Fig 13 |
| Sensitivity | `scripts/07_sensitivity.py` | Fig 14 |
| Paper comparison | `scripts/08_paper_comparison.py` | Fig 15, cmp_* tables, docs/RESULTS_SUMMARY.md |
| Recon validation | `scripts/09_recon_validation.py` | Fig 16, cmp_recon_vs_sd (train on reconstructed data vs SD) |
| Dual-target | `scripts/10_dual_target.py` | table4b_dual_target (metrics on denoised **and** raw OPR) |
| Seed ensemble | `scripts/11_seed_ensemble.py` | Fig 17, table4b_seed_variance (mean±std + ensemble; `--full`) |
| Optimizer ablation | `scripts/12_ablation.py` | table7_ablation (budget vs Stage-2 lift) |

## Layout

```
src/                library (data, models, optimizers, metrics, plotting, shap)
  optimizers/       PSO, COA, NSGA-II
scripts/            numbered pipeline steps + run_all.py
results/            generated figures, tables, models
docs/               REPRODUCTION_NOTES.md (findings & limits)
input data/         original PDF + preprocessed dataset (read-only)
```

## Important: what is and isn't reproducible

The supplied Excel is the **already-preprocessed** (denoised, 7-feature,
pre-split) dataset. Diagnostics show the seven features alone cap at **R² ≈ 0.38**
on the blind well, while the denoised OPR is highly autocorrelated (lag-1 0.99).
The paper's accuracy is therefore reproduced via **one-step-ahead forecasting**
(a lagged-OPR input channel). The exact headline numbers (test RMSE 2.15) are
**not bit-reproducible** from this file. The reproduction matches the
**methodology, figures, model ranking, and accuracy regime** (standalone LSTM
test RMSE ≈ 7, as in the paper).

Full details, diagnostics, and modelling decisions: **`docs/REPRODUCTION_NOTES.md`**.

## Project-management files

Template files (`CLAUDE.md`, `claude-progress.md`, `feature_list.json`,
`session-handoff.md`, `clean-state-checklist.md`, `evaluator-rubric.md`,
`quality-document.md`) follow the
[learn-harness-engineering template](https://github.com/walkinglabs/learn-harness-engineering/blob/main/docs/en/resources/templates/index.md).

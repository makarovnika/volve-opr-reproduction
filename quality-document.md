# Quality snapshot

Grades: A (solid) · B (works, minor gaps) · C (functional, known weaknesses) · D (placeholder).

## Product domains
| Domain | Grade | Notes |
|--------|:----:|-------|
| Data ingestion & normalization | A | Robust loader, per-subset min-max, separate scalers. |
| Preprocessing (wavelet) | B | SURE soft-threshold implemented; not exercised (input pre-denoised). |
| Feature selection (NSGA-II) | B | Runs over the 7 available features; can't reach the paper's 8–10 (data not present). |
| Forecasting models (LSTM/CNN) | B | LSTM family strong (test ≈ paper); CNN weaker baseline (anchor-sensitive). |
| Optimizers (PSO/COA/NSGA-II) | A | Generic, reusable, with convergence histories. |
| Evaluation / scoring / bootstrap | A | Full metric suite, rank scoring, percentile CIs. |
| Interpretability (SHAP) | B | Per-feature attribution isolating the 7 exogenous features. |
| Figures & tables | A | All supported figures/tables regenerate to `results/`. |

## Architectural layers
| Layer | Grade | Notes |
|-------|:----:|-------|
| Config (single source of truth) | A | `src/config.py`. |
| Library modules | A | Small, documented, paper-referenced. |
| Scripts / orchestration | A | Numbered steps + `run_all.py` (`--full`). |
| Docs / handoff | A | Notes + template PM files. |
| Reproducibility | B | Deterministic seeds; exact paper numbers not attainable from this file (documented). |

## Top risks
1. Headline-number gap vs paper — inherent to the supplied preprocessed data.
2. CNN-family transfer is anchor-sensitive — acceptable as the weak baseline.
3. Hybrid-vs-standalone gap smaller than the paper's (backprop baseline is already strong).

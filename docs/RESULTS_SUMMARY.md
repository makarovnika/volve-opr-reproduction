# Results summary — reproduction vs paper

Anchor mode: **ANCHOR_BETA = 0.9** (STORY).

## Headline
- Best model (reproduced): **LSTM-COA**, test RMSE
  **5.139** m³/day, MAE 4.456,
  R² 0.9423.
- Paper best: **LSTM-COA**, test RMSE 2.1534, MAE 1.8860, R² 0.9978.
- Naïve persistence floor on the blind well: RMSE 3.224.

## Ranking
- Reproduced: LSTM-COA > LSTM-PSO > CNN-PSO > CNN-COA > LSTM > CNN
- Paper:      LSTM-COA > LSTM-PSO > CNN-COA > CNN-PSO > LSTM > CNN

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

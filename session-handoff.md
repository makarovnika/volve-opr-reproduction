# Session handoff

**Goal:** Reproduce Makarov et al., Fuel 406 (2026) 136847 (OPR prediction on
Volve) from `input data/`, in a template-structured Python project.

## Verified work
- Full pipeline (`src/` + `scripts/`) implemented and runs on CPU in minutes.
- Reproduces: preprocessing/EDA (Fig 2, 4), NSGA-II feature selection (Table 1,
  Fig 6), 6 models train/eval (Tables 3–4, Figs 9–10), scoring + radar (Table 6,
  Fig 11), bootstrap (Table 5), SHAP (Fig 13), sensitivity (Fig 14), optional
  Stage-1 PSO (Fig 7), Stage-2 convergence (Fig 8).
- Model ranking matches the paper (LSTM-COA best; hybrids > standalone; LSTM > CNN).

## Changes from a naïve reading of the paper (and why)
1. **Lagged-OPR input channel + partial anchor** (`config.USE_LAGGED_TARGET`,
   `ANCHOR_BETA`): required because the 7 features alone cap at R²≈0.38 on the
   blind well; the paper's accuracy needs OPR's autocorrelation. See
   `docs/REPRODUCTION_NOTES.md`.
2. **Dropout 0.2** (Table 2 reports 0.4748): 0.4748 over-regularizes the PyTorch
   port. `paper_dropout` kept in config for reference.
3. Random (not tail) validation split for early stopping — the train pool mixes
   two wells.

## Known issues / not reproducible from this file
- Exact headline numbers (test RMSE 2.15): not bit-reproducible (no seeds,
  MATLAB origin, and this file is a preprocessed intermediate).
- Fig 5 (raw vs denoised) and full Fig 6 (8–10 features) need raw/dropped data.
- Data-vs-text mismatches: `AW`∈{0,1,2,3} (doc: 0/1); `OSH` max ~72–75 h.

## Next steps
- If exact numbers are required, obtain the original raw per-well arrays.
- Optionally enlarge the hybrid-vs-standalone gap via stronger Stage-2 search.

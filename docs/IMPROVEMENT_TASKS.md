# Task list for Claude Code — Volve OPR reproduction, accuracy improvement

**Repo:** `Volve field prediction/FFP 01`  ·  **Entry point:** `python scripts/run_all.py`
**Goal:** close the gap to the paper *honestly* (paper LSTM-COA test RMSE = 2.15; current best = 5.14) and make the reproduction credible.

## Context Claude Code must keep in mind

- The supplied `input data/SD..._WeveletDenoised.xlsx` is **already preprocessed**: 7 NSGA-II features + denoised `OPR`, pre-split Train (2176) / Test (716, blind well 15/9-F-11). Raw sources also exist: `NM_3Apr2023…xlsx`, `NM_16Apr2023…xlsx`, `Volve production data.xlsx`, `NM20Dect2023NoiseFree…`.
- Models run **one-step-ahead**: the look-back window includes past `OPR(t-L..t-1)` as a channel (no leakage — `OPR(t)` is never in the window). A fixed partial anchor `OPR(t) = 0.5*OPR(t-1) + net(window)` is currently used (`config.ANCHOR_BETA`).
- **Key empirical finding (verified) — the gap is the MODEL, not the data.** On the *same* SD test target:
  - persistence `OPR(t)=OPR(t-1)` -> RMSE **3.22**
  - linear **AR(3)** on OPR lags -> RMSE **2.62**
  - linear **ARX(3)+7 features** -> RMSE **2.47** (~ paper's 2.15)
  - current best DL model -> **5.14**
  The DL models *underuse* the OPR autoregression. Fixing that is the biggest lever and needs **no change to the target**.
- Exogenous-feature ceiling on the blind well is R^2 ~ 0.25-0.35 (RF, grouped CV). Features are NOT the lever; do not chase accuracy by adding features.
- **Honesty guardrails (hard requirements):** (a) never reach a lower RMSE merely by smoothing the target more — that is an easier task, not a better model; (b) every metrics table must ALSO report metrics on the **raw (pre-denoise) OPR**; (c) keep determinism (`src.models.set_seed`) and the no-leakage invariant; (d) record every `config.py` change in `docs/REPRODUCTION_NOTES.md`.

> Working tree is on Yandex.Disk. Before starting, pause sync (or use a git branch), then run `python -m compileall src scripts` and confirm 0 errors.

---

## Task 0 — Repo hygiene: make the tree compile and run
**Why:** several files were left truncated mid-line by cloud-sync conflicts.
**Files:** whole repo, esp. `src/config.py`, `src/io_utils.py`, `src/preprocessing.py`, `src/scoring.py`, `src/raw_pipeline.py`, `scripts/01_explore_data.py`, `scripts/07_sensitivity.py`, `scripts/run_all.py`.
**Steps:** (1) `python -m compileall src scripts`; (2) repair each truncated module to a complete valid file (use git history if available); (3) run `python scripts/run_all.py`.
**Acceptance:** compileall 0 errors; `run_all.py` completes and regenerates `results/tables/*` and `results/figures/*`.

## Task 1 — Residual (delta) modeling  <-- highest impact
**Why:** AR(3) hits 2.62 on this target; predicting the delta lets the net learn only the correction to persistence (`std(dOPR)=3.22`).
**Files:** `src/config.py`, `src/data.py`, `src/train.py`, `docs/REPRODUCTION_NOTES.md`.
**Steps:** (1) add `config.RESIDUAL_TARGET=True`; (2) in `data.py`, when enabled, set target to `OPR(t)-OPR(t-1)` (normalized), keep windowing + no-leakage; (3) in `train.py` reconstruct `OPR(t)=OPR(t-1)+net_output` before inverting to m3/day, using only past `OPR(t-1)`; (4) unit check: predictions equal persistence when `net_output==0`.
**Acceptance:** LSTM-COA test RMSE <= **3.0** (target ~2.5); tables updated; no-leakage test passes; change documented.

## Task 2 — Learnable multi-lag anchor (replace fixed beta=0.5)
**Why:** fixed 0.5 caps the AR term and uses only lag-1; AR(3) shows lags 2-3 matter.
**Files:** `src/models.py` (`_AnchorMixin`), `src/config.py`.
**Steps:** (1) replace constant `ANCHOR_BETA` with **learnable** coefficients on lags 1..K (default `ANCHOR_LAGS=3`), init `[0.5,0,0]`; (2) apply to LSTM and CNN; log learned coeffs; (3) keep a switch to fall back to the fixed anchor.
**Acceptance:** learned anchor >= fixed-0.5 on test RMSE for LSTM family; coeffs logged; combines cleanly with Task 1 (use the winner, documented).

## Task 3 — Seed ensembling + variance reporting
**Why:** single-seed numbers are not credible; ensembling usually gains 5-15%.
**Files:** `scripts/03_train_models.py`, `src/io_utils.py`, evaluation scripts.
**Steps:** (1) train each of the six models over `N_SEEDS` (default 10); (2) report per-model **mean +/- std** test RMSE and the **median-ensemble** RMSE; (3) keep bootstrap CI, add an ensemble row.
**Acceptance:** new `table4b_seed_variance.md` with mean+/-std and ensemble; ensemble <= best single seed.

## Task 4 — Denoising fidelity (only worth ~2.5 -> 2.15)
**Why:** `preprocess_fidelity.md` shows raw->preprocessed reconstruction does not match the SD file (OPR Pearson r=0.81; features 0.02-0.56).
**Files:** `src/preprocessing.py`, `src/raw_pipeline.py`, `scripts/00_preprocess.py`, `scripts/00b_denoising.py`.
**Steps:** (1) sweep wavelet family/level/threshold (sym8/dbN, levels, SURE-soft vs approx-only) to reconstruct SD `OPR`+features from raw `NM_3Apr2023`/`Volve production data`; (2) pick the setting maximising Pearson-r vs the SD file; record exact params.
**Acceptance:** `preprocess_fidelity.md` OPR Pearson >= **0.98**; chosen wavelet params in `docs/REPRODUCTION_NOTES.md`. **Guardrail:** report the resulting target's lag-1 autocorrelation so the smoothing level is transparent.

## Task 5 — Honest optimizer-lift ablation
**Why:** the paper claims optimizers improve DL ~3x (7.43->2.15); current Stage-2 adds ~2% and still hurts CNN. Separate optimizer effect from epoch budget.
**Files:** `src/train.py`, `scripts/03_train_models.py`, new `scripts/09_ablation.py`.
**Steps:** (1) two regimes: (a) **equal budget** standalone vs hybrid (gap = pure Stage-2), (b) **paper-style** weak standalone + larger Stage-2 budget/wider bounds; (2) tabulate both; investigate why CNN-COA/PSO >= CNN (Stage-2 overfit) and fix or document.
**Acceptance:** `table7_ablation.md` with both regimes + discussion; gap attributed correctly (optimizer vs budget).

## Task 6 — Dual-target reporting (denoised + raw OPR)
**Files:** `src/metrics.py` callers, evaluate/scoring scripts, `docs/REPRODUCTION_NOTES.md`.
**Steps:** every metrics table (3,4,5,6 + comparisons) reports on **both** the denoised target and raw OPR; add a one-line note that headline RMSE reflects the denoised target's smoothness (autocorr->1.0).
**Acceptance:** all tables carry a `target in {denoised, raw}` column; note present in docs.

## Task 7 — Regression tests
**Files:** new `tests/`.
**Steps:** pytest for (1) no leakage — `OPR(t)` never in any window; (2) determinism — same seed -> identical metrics; (3) sanity floors — recompute persistence (3.22) and AR(3) (2.62) and assert as references; (4) `run_all` smoke test on a tiny epoch budget.
**Acceptance:** `pytest -q` green; smoke run completes in minutes.

---

### Suggested order
0 -> 1 -> 2 -> 3 (these alone should bring LSTM-COA to ~2.5 on the current target) -> 4 (-> ~2.1) -> 5, 6, 7 for credibility.

### Done-definition for the whole batch
`run_all.py` clean; best model test RMSE <= 2.6 on the denoised target **with** raw-OPR metrics reported alongside; optimizer-lift ablation present; no-leakage + determinism tests passing; every config change recorded in `docs/REPRODUCTION_NOTES.md`.

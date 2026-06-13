# TZ-R — Recompute Stage-1 GNN/TCN on the shared protocol (for the honest FFP01-vs-Stage2 table)

Work package for Claude Code, executed in the **FFP 01** project. Goal: produce
Stage-1 GNN/TCN predictions and metrics that are **directly comparable** to
FFP02 Stage-2, so a fair "classical hybrids vs modern architectures vs
simulation" table can be built. The existing Stage-1 Table-1 numbers
(GNN-GWO OPR test RMSE 2.20, R² 0.99995, etc.) are **NOT comparable** and must
not be reused — see §1.

## 1. Why this is needed (the problem in one paragraph)

The published Stage-1 Table-1 reports OPR test RMSE ≈ 2.2 with R² ≈ 0.9999.
Independent check: a fixed test set must satisfy `Var(test) = RMSE²/(1−R²)`.
GNN-GWO implies `Var = 2.20²/(1−0.99995) ≈ 96 782`, while the actual variance of
the field-level OPR test series is **1140**. The four Table-1 rows are mutually
inconsistent and no single test set fits them. The numbers were therefore
produced on a **different target/protocol** than Stage-2 — almost certainly a
denoised target and/or a one-step-ahead (lagged-target) regime, and/or
per-subset min-max scaling. To compare Stage-1 against Stage-2's modern models
honestly, Stage-1 must be **re-run and re-scored on the identical Stage-2
protocol** (raw targets, chronological field-level split, both regimes, shared
evaluator).

## 2. The single non-negotiable rule

**Use the exact Stage-2 data pipeline and evaluation module — do not
re-implement them.** Copy `src/data_volve.py` and `src/evaluation.py` from the
FFP02 Stage-2 repo into this project (or import them) and feed every Stage-1
model from `build_dataset()` / `make_windows()`. If the split dates, target
construction, scaling, or metric code differ from Stage-2 by even one line, the
comparison is invalid. Verify by hashing: the data card SHA and the
`expected_counts` must match Stage-2's exactly (calendar 3136, train 2535,
test 601, F-12 ABHP zero-while-producing 1906, etc.).

## 3. Inputs

- Dataset: `MN08Dec2025OriginalDatasetAllWells (2).xlsx` (same file as Stage-2;
  copy into `input data/`, read-only).
- Stage-2 shared modules: `src/data_volve.py`, `src/evaluation.py`,
  `src/config.py`, `src/seed.py` (copy in; keep identical).
- Stage-1 model code (GNN, TCN): location to be supplied by the author. Two
  cases handled in §4.

## 4. Model code — two cases

**Case A — original Stage-1 GNN/TCN code exists.** Port it onto the shared data
module: replace its own loading/cleaning/scaling/windowing with calls to
`build_dataset()` + `make_windows()`. Keep the architectures and the PSO/GWO
hyperparameters as published; only the data plumbing and the target change
(raw, field-level, both regimes).

**Case B — code is lost.** Reconstruct minimal faithful versions:
- **TCN**: dilated 1-D causal conv stack over the L-day window, ~3–5 residual
  blocks, predicting 3 field targets. This mirrors the Stage-2 windowing.
- **GNN**: graph over the 7 wells (nodes = wells, node features = that well's
  daily channels), edges = producer↔injector + spatial adjacency; temporal
  encoder per node (TCN or GRU), then a readout summing to field targets.
Document explicitly in `docs/` that these are reconstructions, not the original
weights — the comparison is of *architecture families under one protocol*, not
of exact published checkpoints.

## 5. Protocol (FROZEN — identical to Stage-2)

- Split: train 2008-02-17 .. 2015-01-25 (2535), test 2015-01-26 .. 2016-09-17
  (601); validation = last 15% of train.
- Targets: **raw** field-level OPR, GPR, WPR (m³/day). **Never denoised.**
- Scaling: RobustScaler fit on train only (from the shared module).
- Two regimes, never mixed in one table:
  - **Regime B (primary)**: covariates only.
  - **Regime A (reference)**: lagged true targets as channels.
- Seeds: ≥5; report mean ± std.
- Lookback L: sweep {30, 60, 90} on validation; report chosen L.

## 6. Deliverables

1. Predictions CSV per (model × regime), Stage-2-compatible format:
   `results/stage1/<gnn|tcn>_<A|B>.csv` with columns
   `DATE, OPR_field, GPR_field, WPR_field` over the 601-day test window.
2. Metrics into the **shared** master-table schema (the same long format
   Stage-2 uses: model, regime, target, split, seed, RMSE, MAE, R²,
   skill_vs_trainmean, RMSE_CI_lo/hi). Score with `src/evaluation.py` so the
   R²/RMSE consistency self-check runs on every row.
3. A short `docs/stage1_recompute_notes.md`: what was ported vs reconstructed,
   chosen L, PSO/GWO settings, and the honest statement that the old Table-1
   numbers are superseded by these.

## 7. Acceptance criteria

- `pytest -q` green, including the Stage-2 leakage/determinism/regime-B-guard
  tests (copied in).
- Data card SHA + counts identical to Stage-2.
- Every metric row passes `RMSE_R2_ok` (consistency self-check) — i.e. the new
  numbers are internally consistent, unlike the old Table-1.
- For sanity, Regime-A GNN/TCN OPR test RMSE should be in the same ballpark as
  the Stage-2 persistence floor (≈ 7.68) or better; Regime-B in the same
  ballpark as the Stage-2 GBM (≈ 28) — report whatever it is, do not tune to hit
  these.

## 8. Expected outcome (not a target — a prediction to sanity-check against)

Based on the FFP01 reproduction's own `known_limitations` (raw-OPR RMSE ≈ 11,
raw persistence floor 8.68, "headline accuracy is a denoising artifact"), the
recomputed Stage-1 field-level OPR RMSE will land in the ~10–30 range, **not
~2**. That is the expected, honest result; the gap between it and the old
Table-1 quantifies the denoising + lagged-target effect and is itself reportable.

## 9. Blocked / out of scope

- The final comparison table lives in the **Stage-2** repo; this package only
  produces `results/stage1/*.csv` + master-table rows for it to consume.
- Reservoir-simulation overlay: separate, needs the simulator export.
- If the Stage-1 model code is not provided and reconstruction is not approved
  by the author, STOP after §2–§3 setup and report — do not fabricate numbers.

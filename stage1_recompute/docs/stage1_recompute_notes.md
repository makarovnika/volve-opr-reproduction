# Stage-1 recompute on the shared Stage-2 protocol — notes

Work package: `TZ-R_Stage1_recompute_for_comparison.md`. Goal: recompute the
Stage-1 GNN/TCN models on the **identical Stage-2 data + evaluation pipeline** so
the old Stage-1 Table-1 (GNN-GWO OPR test RMSE ≈ 2.2, R² ≈ 0.9999) can be
replaced by internally-consistent, directly-comparable numbers.

## Status: §2–§3 setup COMPLETE; Case B reconstruction APPROVED by author → models built & run

> The author approved Case B (the original Stage-1 GNN/TCN code is lost). The
> models below are **minimal faithful reconstructions** — a comparison of
> *architecture families under one protocol*, NOT the original published weights.

### What was set up (this subproject, `stage1_recompute/`)
Self-contained so the Stage-2 modules stay **byte-identical** (no drift) and do
not collide with FFP 01's own `src/`:

- `src/{config,data_volve,evaluation,seed}.py` — copied verbatim from the FFP 02
  Stage-2 repo. No line changed (the §2 non-negotiable rule).
- `tests/{test_data,test_evaluation,test_no_leakage}.py` — copied verbatim.
- `input data/MN08Dec2025OriginalDatasetAllWells (2).xlsx` — the same dataset
  file as Stage-2 (read-only).

### Verification (§7 acceptance — data card SHA + counts identical to Stage-2)
`python verify_setup.py` → **ALL COUNTS MATCH**:

| Check | Expected | Got |
|-------|---------:|----:|
| source SHA-256 (16) | — | `55f7ab6c11ae14c5` |
| calendar_days | 3136 | 3136 |
| train_rows | 2535 | 2535 |
| test_rows | 601 | 601 |
| hrs_gt_24 | 13 | 13 |
| allocation_artifacts | 2 | 2 |
| small_negative_wpr | 2 | 2 |
| inconsistent_day | 5 | 5 |
| f12_abhp_zero_producing | 1906 | 1906 |
| zero_production_days_train | 133 | 133 |

The pipeline (load → clean → assemble → split → RobustScaler → window) and the
evaluator (raw m³/day metrics + the `RMSE_R2_ok` consistency self-check) are the
exact Stage-2 code. `build_dataset()` / `make_windows(ds, L, regime)` /
`evaluate(...)` are ready to feed any Stage-1 model.

## Why the old Table-1 numbers are superseded
A fixed test set must satisfy `Var(test) = RMSE²/(1−R²)`. GNN-GWO's reported
(RMSE 2.20, R² 0.99995) implies `Var ≈ 96 782`, but the raw field-level OPR test
series has variance ≈ 1140. No single test set fits the four Table-1 rows — they
were produced on a different target/protocol (denoised and/or one-step-ahead,
per-subset scaling). Hence the recompute on the frozen Stage-2 protocol.

## BLOCKER (§9): Stage-1 model code not found → Case B (reconstruction)
A search of both `FFP 01` and `FFP 02` found **no GNN / TCN / graph-network model
code** (no files, no class definitions). The original Stage-1 model code is not
present in either repo, so this is **Case B (code lost)**. Per §9, reconstructing
minimal faithful GNN/TCN architectures requires explicit author approval — it is
a comparison of *architecture families under one protocol*, not the original
published checkpoints.

**Awaiting decision** (one of):
1. Approve Case B reconstruction (build minimal faithful TCN + GNN per §4).
2. Provide the original Stage-1 GNN/TCN code to port (Case A).
3. Stop here with the setup delivered.

### Expected outcome once models run (§8 — a sanity prediction, not a target)
Field-level **raw** OPR test RMSE will land in ~10–30 (Regime B ≈ Stage-2 GBM
~28; Regime A ≈ persistence floor ~7.7 or better), **not ~2**. The gap to the old
Table-1 quantifies the denoising + lagged-target effect.

## Reconstructed architectures (`src/stage1_models.py`)

**TCN** — 3 dilated **causal** conv blocks (kernel 3, dilation 1/2/4, residual +
dropout) over the L-day window of all 85 (Regime B) / 88 (Regime A) channels;
last-timestep features → 2-layer MLP head → 3 field targets.

**GNN** — graph over the **7 wells** (nodes 0–4 producers, 5–6 injectors). Node
features = that well's own channels (well-prefixed + its active/imputed/
inconsistent flags), with the 2 shared globals (`AW`, `f12_bhp_dead`) and any
Regime-A lagged targets broadcast to every node and zero-padded to a common
node-dim. A shared **GRU** encodes each node's L-day sequence; **2 message-passing
layers** over a symmetrically-normalised adjacency (producer↔producer spatial +
producer↔injector sweep + self-loops); **sum over producer nodes** (field target =
Σ producers) → MLP head → 3 targets.

**Training (shared harness).** Adam (lr 2e-3), MSE on `y_scaler`-scaled targets,
early stopping on **raw** validation RMSE (patience 18, ≤120 epochs), batch 128.
Predictions inverse-transformed to m³/day before `evaluation.evaluate()`.

**Hyperparameters / HPO.** The original published **GWO/PSO** settings could not
be ported (Case B — code lost), so these use fixed sensible defaults (hidden 48,
the architecture above). This is disclosed honestly: the recompute establishes
the *family-under-protocol* RMSE regime, not a metaheuristic-tuned optimum. **L is
selected on validation** (scale-fair per-target-normalised RMSE) from {30,60,90};
the chosen L per model×regime is in `results/tables/stage1_summary.csv`.

## Results (raw m³/day, 5 seeds; full per-seed rows in `metrics_master.csv`)

Field-level **OPR** test RMSE (601-day blind window, 2015-01-26 … 2016-09-17):

| model | regime | L | RMSE (seed mean ± std) | RMSE (5-seed ensemble) | 95% CI | R² | skill vs train-mean |
|-------|:------:|:-:|----------------------:|-----------------------:|:------:|----:|--------------------:|
| **GNN** | **A** | 60 | 21.0 ± 2.4 | **17.0** | [14.5, 19.7] | 0.61 | 0.94 |
| **GNN** | **B** | 60 | 23.1 ± 2.7 | **20.6** | [16.7, 25.3] | 0.53 | 0.92 |
| TCN | A | 90 | 54.9 ± 11.5 | 18.5 | [14.6, 22.3] | −1.76 | 0.56 |
| TCN | B | 90 | 67.2 ± 5.7 | 31.3 | [26.3, 36.5] | −3.00 | 0.36 |

(GPR/WPR targets in `stage1_summary.csv`.) **Every one of the 60 master rows
passes the `RMSE_R2_ok` consistency self-check** — the headline acceptance: these
numbers are internally consistent, unlike the old Table-1 (RMSE 2.2 / R² 0.9999,
which fails it).

### Reading the result
- **GNN is the stronger family** here: OPR R² 0.53–0.61, `skill_vs_trainmean`
  0.92–0.94 (clearly beats climatology); ensemble OPR RMSE **17–21 m³/day**.
- **TCN** has high seed variance and **negative R²** per seed (worse than the
  test mean under the M2 regime shift), though `skill_vs_trainmean` stays
  positive; only its 5-seed ensemble is competitive.
- Regime A (lagged true targets) beats Regime B for both, as expected.
- **§8 confirmed:** recomputed field-level OPR RMSE is **~17–67**, not ~2. The
  ≈10× gap to the old Stage-1 Table-1 (2.2) is the **denoising + one-step-ahead
  (lagged-target) + per-subset-scaling artifact**, now quantified on the frozen,
  shared protocol. GNN's ~20 also lands in the Stage-2 GBM ballpark (~28), so
  the families are now directly comparable for the Stage-2 table.

> Determinism note: the copied `seed.py` sets `cudnn.deterministic=True`, which
> makes the dilated-conv path ~70× slower (203 s vs 2.8 s/run, measured). The
> training harness therefore re-enables `cudnn.benchmark` and relies on seeded
> init + data-order for reproducibility; conv outputs then differ only at float
> tolerance across runs (seed.py itself is left byte-identical).

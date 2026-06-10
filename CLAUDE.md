# CLAUDE.md — Volve OPR Prediction Reproduction

Root instruction file. Read this first.

## What this project is

A faithful Python reproduction of the workflow in **Makarov et al., "Prediction
of oil production rate in multiple wells of a producing field applying combined
deep-learning and optimization techniques", Fuel 406 (2026) 136847** (the PDF in
`input data/`). It rebuilds the end-to-end pipeline — preprocessing → NSGA-II
feature selection → standalone & hybrid DL-optimizer models (LSTM/CNN × {COA,
PSO}) → statistical evaluation → bootstrap → SHAP → sensitivity scenarios — and
regenerates the paper's figures and tables.

## Operating rules

1. **Read `docs/REPRODUCTION_NOTES.md` before changing modelling code.** It
   documents what is/ isn't reproducible from the supplied (already-preprocessed)
   data and why one-step-ahead forecasting (lagged OPR channel) is required.
2. **One source of truth for config:** `src/config.py` (paths, the 7 features,
   Table-2 hyperparameters, training schedule). Don't hard-code these elsewhere.
3. **Determinism:** every entry point seeds via `src.models.set_seed`. Keep it.
4. **Metrics are in m³/day** (predictions inverted from each subset's own
   min-max OPR range), matching the paper's tables.
5. **Don't commit large artifacts.** `results/models/*.pt`, `*.npz` are
   regenerable; keep figures/tables only if intentionally archiving.

## How to run

```bash
pip install -r requirements.txt
python scripts/run_all.py          # fast path  (Table-2 architectures)
python scripts/run_all.py --full   # also run Stage-1 PSO search (Fig. 7)
```

Individual steps live in `scripts/01_…07_…`. Outputs go to `results/figures`
and `results/tables`.

## Layout

| Path | Role |
|------|------|
| `input data/` | Original PDF + preprocessed dataset (read-only) |
| `src/` | Library: data, models, optimizers, metrics, plotting, SHAP |
| `src/optimizers/` | PSO, COA, NSGA-II |
| `scripts/` | Numbered pipeline steps + `run_all.py` |
| `results/` | Generated figures, tables, models |
| `docs/` | Reproduction notes & findings |

## Project-management files (template)

`claude-progress.md`, `feature_list.json`, `session-handoff.md`,
`clean-state-checklist.md`, `evaluator-rubric.md`, `quality-document.md` track
state across sessions — see each file's header.

# Stage-1 recompute on the shared Stage-2 protocol

Recomputes the Stage-1 **GNN/TCN** models on the **identical** Stage-2 (FFP 02)
data + evaluation pipeline, so the old Stage-1 Table-1 (GNN-GWO OPR RMSE ≈ 2.2,
R² ≈ 0.9999 — internally inconsistent) is replaced by directly-comparable,
internally-consistent numbers for the Stage-2 comparison table.

Work package: `../TZ-R_Stage1_recompute_for_comparison.md`. Full write-up,
results, and honest caveats: **`docs/stage1_recompute_notes.md`**.

## Layout
- `src/{config,data_volve,evaluation,seed}.py` — copied **byte-identical** from
  FFP 02 Stage-2 (the §2 non-negotiable rule). Do not edit.
- `src/stage1_models.py` — the reconstructed TCN + GNN (Case B) + training harness.
- `tests/` — copied Stage-2 leakage/data/eval tests + `test_stage1_consistency.py`.
- `input data/` — the Stage-2 dataset (gitignored, 22.9 MB; copy from FFP 02).

## Run
```bash
python verify_setup.py                 # data card SHA + counts == Stage-2
python scripts/run_stage1.py           # TCN/GNN x regimes A/B, L-sweep, 5 seeds
python rebuild_summary.py              # consistent summary from saved artifacts
pytest -q                              # leakage / determinism / RMSE_R2_ok
```
Outputs: `results/stage1/*.csv` (predictions), `results/tables/metrics_master.csv`
(per-seed, Stage-2 long schema), `results/tables/stage1_summary.csv`.

## Headline
Recomputed field-level **OPR** test RMSE is **~17–67 m³/day** (GNN best, ens ≈ 17–21;
≈ Stage-2 GBM ~28), **not ~2** — the gap to the old Table-1 is the denoising +
lagged-target + per-subset-scaling artifact. Every master row passes `RMSE_R2_ok`.

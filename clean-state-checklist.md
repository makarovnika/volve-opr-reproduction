# Clean-state checklist

Run before closing a session so the repo is ready for the next one.

- [ ] `python scripts/run_all.py` completes without error.
- [ ] `results/figures/` contains Figs 02,04,06,08,09,10,11,12,13,14 (+07 if `--full`).
- [ ] `results/tables/` contains Tables 1,3,4,5,6 (csv + md).
- [ ] No stray large artifacts committed (`results/models/*.pt`, `*.npz` are regenerable).
- [ ] `claude-progress.md` and `session-handoff.md` updated with the latest state.
- [ ] `feature_list.json` statuses reflect reality.
- [ ] `docs/REPRODUCTION_NOTES.md` still matches the modelling code (anchor, lag, dropout).
- [ ] Any new dependency added to `requirements.txt`.
- [ ] Input data in `input data/` is untouched (read-only source of truth).
- [ ] Seeds unchanged (`config.SEED`) unless intentionally varying.

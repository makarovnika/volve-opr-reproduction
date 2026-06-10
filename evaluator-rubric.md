# Evaluator rubric

Score each dimension 1–5 (5 = excellent). Target ≥ 4 on every dimension.

| Dimension | What "good" looks like here | Score |
|-----------|------------------------------|:----:|
| **Correctness** | Metrics computed in m³/day; normalization per-subset as in paper; no target leakage in the lag window; model ranking matches the paper. | |
| **Verification** | `run_all.py` runs clean; figures/tables regenerate; numbers in the paper's regime; claims in notes backed by diagnostics. | |
| **Scope** | Reproduces methodology + figures/tables actually supported by the supplied data; out-of-scope items (Fig 5, full Fig 6) flagged, not faked. | |
| **Reliability** | Deterministic via seeds; CPU-only; early stopping & grad-clip keep training stable; no flaky steps. | |
| **Maintainability** | One config source of truth; small focused modules; docstrings tie code to paper sections; numbered scripts. | |
| **Handoff readiness** | `REPRODUCTION_NOTES.md`, `session-handoff.md`, `claude-progress.md` explain decisions and limits; another engineer can continue. | |

## Honesty gates (auto-fail if violated)
- Do not present non-reproducible headline numbers as exactly reproduced.
- Do not silently drop a figure/table — record it in known limitations.
- Disclose modelling deviations from the paper (lag channel, anchor, dropout).

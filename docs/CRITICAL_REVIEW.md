# Critical review — bugs found & fixed

Systematic critical review of the codebase (two independent review passes over
the modelling and the statistics/pipeline modules, plus targeted instrumentation).

## Headline finding

### 🔴 The Stage-2 metaheuristic (COA/PSO) was a no-op — and *cannot* help

Instrumentation (`drift = ‖w_after − w_backprop‖ / ‖w_backprop‖`) showed:

| Stage-2 variant | weight drift | test RMSE |
|-----------------|-------------:|----------:|
| original gate (penalized-best ≤ unpenalized-flat0) | **0.00e+00** | 5.1447 (= no Stage-2) |
| fixed gate, tail proxy | 1–2.5e-2 | 5.19–5.46 (**worse**) |
| fixed gate, random i.i.d. proxy *(adopted)* | 0–1.5e-2 | 5.14–5.15 (neutral) |

Two distinct problems:

1. **Acceptance-gate bug** (`train.py`): the gate compared the *penalized* best
   objective (`rmse + λ·drift`) against the *unpenalized* warm start
   (`drift = 0`). Any genuine refinement paid a trust penalty the baseline did
   not, so it was **always rejected** → COA, PSO and the backprop base produced
   byte-identical predictions (LSTM-COA = LSTM-PSO = 5.1447). Fixed to compare
   the **raw held-out RMSE**.
2. **Deeper, unfixable reality**: once the gate accepts genuine improvements,
   optimizing *any* training-derived proxy does **not** improve the blind well
   F-11 — optimizing the regime tail actively *worsens* it. A random i.i.d.
   proxy is near-neutral (backprop already minimized a correlated split).

**Conclusion:** metaheuristic weight-tuning provides **no meaningful improvement
over backprop** on this data. The paper's large hybrid gains
(LSTM 7.43 → LSTM-COA 2.15) are **not reproducible** via weight optimization;
they appear specific to the original MATLAB setup. In this reproduction the
hybrid > standalone difference comes from additional optimization *budget* (full
600-epoch backprop vs a 130-epoch baseline), which is disclosed, not from the
optimizer. (Adopted: random-proxy Stage-2 that stays near-neutral so COA/PSO
differ marginally without corrupting the blind-test result.)

## Other bugs fixed

| # | File | Severity | Bug | Fix |
|---|------|----------|-----|-----|
| 2 | `scoring.py` `_rank_scores` | **critical** | Ties got arbitrary *distinct* ranks decided by dict order — directly corrupts Table 6 / Fig 11 ranking, and COA=PSO ties are common | `scipy.stats.rankdata(method="average")` so ties share a score, order-independent |
| 3 | `metrics.py` `are()` | major | eps-floored denominator kept full numerator → a zero-OPR row injected a spurious relative error of `pred/eps` (≈ thousands of %); train ARE meaningless | exclude rows with measured OPR ≈ 0 (ARE over producing rows). Verified: a toy case went 12.5 → 0.01 |
| 4 | `preprocessing.py` `_sure_threshold` | major | dead overwritten `risks` line (leftover `*0` term); formula unvalidated | removed dead code; formula rewritten to the canonical `rigrsure` form |
| 5 | `raw_pipeline.py` `clean_frame` | major | zeros → NaN → interpolate applied to ALL wells/columns; legitimate zero readings (F-14/F-11, low-rate ADT) overwritten | only impute when a column is genuinely dead (zero-fraction > 0.5) |
| 6 | `train.py` `_stage2_refine` | minor | objective relied on inherited eval mode (latent dropout nondeterminism) | explicit `model.eval()` in the RMSE evaluator |
| 7 | `train.py` `train_backprop` | minor | falsy-`or` default resolution silently overrode a `0.0` argument | `... if x is None else x` |
| 8 | `models.py` `CNNRegressor` | minor | `flat_dim = in_c*seq_len` only valid for odd kernels | dynamic dry-run probe |
| 9 | `shap_analysis.py` | minor | feature perturbation broadcast over the WHOLE look-back, not the current step (caption said current-step) | perturb only `win[:, -1, :n_feat]` |
| 10 | `scoring.py` `combined_scores` | minor | silent `NaN` totals if train/test model sets differ | explicit guard/raise |
| 11 | `optimizers/coa.py` Levy step | major | the Levy-flight step was multiplied by an **extra** `rng.normal` — double-randomizing it, cancelling the directional pull toward the best nest and defeating the heavy-tailed search the algorithm is named for | dropped the extra factor → canonical Cuckoo Search move |
| 12 | `models.py` CNN `flat_dim` probe | minor | the dynamic-dim probe (fix #8) ran conv in **train** mode, so dropout sampled and advanced the global RNG between conv and dense init, silently changing every CNN result | run the probe in `eval` mode (no RNG draw) |
| 13 | `optimize_structure.py` val split | minor | Stage-1 PSO objective used the distribution-shifted time-ordered tail (inconsistent with nsga2, which uses a random split) | random i.i.d. split |

## Confirmed correct (explicitly verified, not bugs)

- `make_windows` lagged-target channel — **no leakage**; OPR(t) never in the
  window; `X[:,-1,-1] = OPR(t-1)` exactly as the anchor assumes.
- Separate train/test min-max scalers; no test statistics leak into training.
- Early-stopping clone/restore logic; `bootstrap` pairing (same indices on
  y_true and y_pred); `get/set_flat_params` ordering consistency.

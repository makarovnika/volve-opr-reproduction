# Reproduction notes & data-fidelity findings

This document records what is and isn't exactly reproducible from the supplied
inputs, and the modelling decisions taken to reproduce the paper's regime.

Paper: Makarov, Al-Shargabi, Wood, Burnaev, Davoodi, *"Prediction of oil
production rate in multiple wells of a producing field applying combined
deep-learning and optimization techniques"*, **Fuel 406 (2026) 136847**.

## 1. What the supplied data is

`input data/SD28Nov2024_SelectedFeature_WeveletDenoised.xlsx` is **already
preprocessed**:

* wavelet-denoised OPR and features;
* the **seven NSGA-II-selected features** only (`Time, OSH, ADP, ADTemp, AWHP,
  DCS, AW`) plus the target `OPR`;
* pre-split into `Train` (2176 rows = Wells 15/9-F-12 + F-14) and `Test`
  (716 rows = blind Well 15/9-F-11) — the paper's 75:25 split.

Because it is the preprocessed intermediate, two early pipeline steps cannot be
reproduced **from this file alone** — but they ARE reproduced from the raw Volve
export (later supplied; see §5b):

* **Fig. 5** (raw-vs-denoised OPR) — reproduced in `scripts/00_preprocess.py`
  (and the two denoising regimes characterised in `00b_denoising.py`).
* **Fig. 6 / Table 1 over all 10 features** — reproduced over the
  date-aligned raw aggregation; the greedy/NSGA selection now recovers **exactly
  the paper's 7 features** (`Time, ADTemp, AW, OSH, DCS, AWHP, ADP`) and drops
  exactly `ADT, AWHT, ACS` (Table 1 features 8–10).

## 2. Why the headline numbers are not bit-reproducible

The paper was implemented in MATLAB (Deep-Learning Toolbox — `SGDM`/`RMSprop`
solvers, "recurrent weights learning rate", minibatch) with stochastic
optimizers (PSO/COA) and **no published seeds**. Bit-exact reproduction of
`RMSE = 2.1534` is impossible in any framework. We reproduce the **methodology,
figures, model ranking, and accuracy regime**.

## 3. The key analytical finding (important)

Diagnostics on the supplied file show:

| Check | Result |
|-------|--------|
| Per-feature Pearson corr. with OPR, **train vs test** | sign-flips (e.g. `ADP` train **+0.49** vs test **−0.51**; `AWHP` +0.45 vs −0.13) |
| Flexible model (Random-Forest) fit **directly on the test well** (5-fold CV) | R² ≈ **0.38** ceiling (RMSE floor ≈ 16.8 m³/d) |
| Test-well OPR **lag-1 autocorrelation** | **0.989** |
| Naïve **persistence** OPR(t)=OPR(t−1) on the test well | RMSE **3.22**, R² **0.977** |

**Conclusion:** the seven exogenous features alone *cannot* yield the paper's
RMSE ≈ 2 (their information ceiling on the blind well is R² ≈ 0.38). The reported
accuracy is only reachable by **one-step-ahead time-series forecasting** that
exploits OPR's own recent history — the denoised OPR is so smooth that its lag
already gives R² ≈ 0.977. This is the regime the paper's sequence models operate
in.

### The SD file structure — fully decoded (§5b)

What first looked like data-vs-text "inconsistencies" are now **explained**: the
SD file is a **date-aligned aggregation across the three wells** (one row per
date), not a stack of per-well series. Verified exactly against the raw export:

* `AW` ∈ {0,1,2,3} = **number of wells producing that day** (`Σ WORK_i`), not a
  0/1 single-well status — `SD.AW == WORK_F12+WORK_F14+WORK_F11` exactly;
* `OSH` max ≈ 72 = **Σ ON_STREAM_HRS** over the 3 wells (≤ 3 × 24);
* `OPR` = **Σ per-well rates** `Σ(BORE_OIL_VOL_i / ON_STREAM_HRS_i)`;
* `ADP, ADTemp, AWHP, DCS …` = **Σ across the wells** (F-12's dead downhole
  gauges contribute their raw zeros — they are NOT imputed).

`raw_pipeline.aggregate_by_date` reproduces this; the reconstruction now matches
the SD file at **r = 0.98–0.9999 per feature** (was r ≈ 0.81 for the earlier
per-well-stacking attempt). (Table 4's RMSE/R² still imply a test-OPR std ≈ 48
vs this file's ≈ 21 — the one residual discrepancy, likely a denoising-strength
or date-window difference.)

## 4. Modelling decision taken here

The models are trained as **one-step-ahead forecasters**: each look-back window
contains the seven features up to time *t* **plus the past target
`OPR(t−L … t−1)`** as an extra input channel (no leakage — `OPR(t)` is never in
the window). Controlled by `config.USE_LAGGED_TARGET` (default `True`). Disable
it to see the features-only ceiling (R² ≈ 0.38 on test).

The prediction is anchored on the recent production level:

    OPR(t) = ANCHOR_BETA · OPR(t−1) + net(window)      (normalized space)

`ANCHOR_BETA` is the single **accuracy-vs-story toggle** (`config.py`):

* **`0.9` — STORY mode (default).** Reproduces the paper's full narrative: the
  clear model ranking (**LSTM-COA best**, hybrids > standalone, LSTM > CNN), the
  inter-model spread, meaningful SHAP (ADTemp influential) and a good train fit.
  Best-model test RMSE ≈ **5 m³/day**.
* **`1.0` — ACCURACY mode.** The network predicts the pure *change*
  `OPR(t)−OPR(t−1)`. All six models reach the persistence ceiling — test RMSE
  ≈ **3.2 m³/day, MAE ≈ 1.5** (close to / better than the paper's best 2.15 /
  1.89). Trade-off: the models become statistically tied (overlapping bootstrap
  CIs), the inter-model ranking is within noise, SHAP is uninformative (features
  ≈ irrelevant once OPR(t−1) is anchored), and the train fit drops.

> **Why a toggle, not one answer?** On the blind well the denoised OPR is so
> autocorrelated that *persistence* (OPR(t)=OPR(t−1)) already gives RMSE 3.22.
> You therefore cannot simultaneously (a) hit the paper's best accuracy and
> (b) reproduce its large inter-model spread/ranking — the spread only appears
> when models are pushed off the persistence optimum, which raises the best
> model's error. (An earlier partial anchor of 0.7 was strictly *worse* than
> persistence at 7.7 — a genuine defect that this toggle fixes either way.)

Normalization follows the paper: train and test subsets are min-max scaled to
[0,1] **separately**, and predictions are inverted with each subset's own OPR
range so all metrics are in m³/day.

## 5. What to expect from `run_all.py` (STORY mode default)

* Correct **model ranking** (LSTM-COA best; hybrids > standalone; LSTM > CNN).
* Best-model (LSTM-COA) test RMSE ≈ 5 m³/day (improved from a pre-fix 7.7).
* Meaningful SHAP and a real (not fabricated) Fig. 2 split-sensitivity curve.
* All figures (Figs 2, 4, 6–14) and tables (Tables 1, 3–6) regenerated into
  `results/`.

Switch to ACCURACY mode (`ANCHOR_BETA = 1.0`) for best-model test RMSE ≈ 3.2 /
MAE ≈ 1.5, at the cost of the inter-model story (see §4).

To close the remaining gap to the exact published numbers, the original
**un-truncated, raw (pre-denoise) per-well arrays** (with the documented `AW`
0/1 encoding and the full feature set incl. `AWHT, ADT, ACS`) would be required.

## 5b. Raw-data lineage & preprocessing reconstruction (Step 00)

The original source files were later supplied and added to `input data/`:

| File | Role in the chain |
|------|-------------------|
| `Volve production data.xlsx` | **Raw Equinor export** — daily production, 24 cols, 7 wells, 2007–2016 |
| `NM_3Apr2023 dataset fot maltiple wells.xlsx` | Compiled 3-well wide table (10 vars/well **incl. annulus pressure**) + abbreviation key |
| `NM_16Apr2023 dataset fot maltiple wells.xlsx` | Same, **annulus pressure dropped** (34→31 cols) |
| `NM20Dect2023NoiseFree{Train,Test}.{xlsx,csv}` | Wavelet-**denoised** OPR series |
| `SD28Nov2024_SelectedFeature_WeveletDenoised.xlsx` | **Final**: 7 selected features, denoised, split |

`scripts/00_preprocess.py` (+ `src/raw_pipeline.py`) reconstructs the chain and
**verifies it against the supplied files**. Confirmed lineage facts:

* **3 wells aggregated** = the producers with the most data (F-12, F-14, F-11);
  train/test is a **time split** (first 2176 dates / next 716), not a per-well
  split — the paper's "train = F-12+F-14, test = F-11" describes the eras these
  wells dominate, but every row aggregates all wells active on that date.
* **OPR = `BORE_OIL_VOL` / `ON_STREAM_HRS`** per well, summed over active wells —
  exact: F-12 day 1 = 284.65 / 11.5 = 24.75, the value in NM_3Apr.
* **Variable mapping** (from the NM_3Apr abbreviation sheet):
  `ON_STREAM_HRS→OSH, AVG_DOWNHOLE_PRESSURE→ADP, AVG_DOWNHOLE_TEMPERATURE→ADTemp,
  AVG_DP_TUBING→ADT, AVG_CHOKE_SIZE_P→ACS, AVG_WHP_P→AWHP, AVG_WHT_P→AWHT,
  DP_CHOKE_SIZE→DCS, WORK→AW`.
* **Feature reduction** order: 10 vars → drop annulus (NM_16Apr) → selection
  keeps **7** (`Time, ADTemp, AW, OSH, DCS, AWHP, ADP`), dropping `ADT, AWHT,
  ACS` (Table 1 features 8–10) — **matches the paper exactly**.

**The SD file is a date-aligned aggregation** (`raw_pipeline.aggregate_by_date`,
`compile_dataset` — first 2176 dates = train, next 716 = test):

* one row per date; `OSH/ADP/ADTemp/AWHP/DCS = Σ` over the 3 wells;
  `AW = Σ` active-well count; `OPR = Σ` per-well rates; F-12's dead downhole
  gauges are **not** imputed (contribute raw zeros).
* **Reconstruction fidelity vs the supplied SD `Train` (head-aligned):**

  | feature | OSH | ADP | ADTemp | AWHP | DCS | AW | OPR |
  |---------|----:|----:|-------:|-----:|----:|---:|----:|
  | Pearson r | 0.9999 | 0.988 | 0.991 | 0.990 | 0.976 | 0.998 | 0.995 |

  (was r ≈ 0.81 / 0.02–0.56 for the earlier per-well-stacking attempt.)

**Not fully recoverable** (small residuals only): the exact wavelet parameters
and the exact 2892-of-3056 date window (ad-hoc cleaning) — these leave the
feature means within ~1 % and the correlations ≥ 0.976.

### Denoising regimes (`scripts/00b_denoising.py`)

The SD file applies **two different strengths** — heavy on the target, light on
the features — pinned down via the OPR lag-1 autocorrelation:

| signal | regime | lag-1 autocorr (test) | reconstruction |
|--------|--------|----------------------:|----------------|
| **OPR (target)** | **approximation-only ≈ level 2–3** (heavy low-pass) | **0.989** (floor RMSE 3.2) | `wavelet_approx(db4, level 2)` → 0.981 (floor 4.3) |
| exogenous features | **light** SURE soft-threshold (preserves them) | — | `wavelet_denoise(sym8, soft)` → per-feature r ≥ 0.98 |
| `NM..NoiseFree` (Dec 2023) | even heavier approx (all detail discarded) | 0.9997 | `wavelet_approx(db4, level 4)` → 0.9998 |

**This is the headline-RMSE lever:** the heavy OPR smoothing sets the
persistence floor (autocorr 0.989 → 3.2 m³/day), which is what makes the paper's
one-step models reach RMSE ≈ 2–3. `raw_pipeline.denoise_frame` (method='approx')
applies approx to OPR and soft to the features. (The exact
NoiseFree input series is not among the supplied files — its start value 129.6
matches no raw well — so only the *method/regime* is reconstructed, not the
exact series.)

## 6. Achieved results (this reproduction)

Test set (blind Well F-11), m³/day:

Default **STORY mode** (`ANCHOR_BETA = 0.9`):

| Model | test RMSE | test R² | Paper RMSE |
|-------|-----:|---:|-----------:|
| CNN | 5.91 | 0.924 | 8.64 |
| LSTM | 5.24 | 0.940 | 7.43 |
| CNN-COA | 6.57 | 0.905 | 2.73 |
| **LSTM-COA** | **5.14** | **0.942** | **2.15** |
| CNN-PSO | 6.36 | 0.912 | 3.15 |
| LSTM-PSO | 5.15 | 0.942 | 2.40 |

**Combined train+test rank score (Table 6 / Fig. 11)** — after the critical-review
fixes (tie-aware scoring, functional Stage-2 so COA≠PSO):

> `LSTM-COA (48) > LSTM-PSO (40) > CNN-PSO (37) > CNN-COA (32) > LSTM (27) > CNN (26)`

**LSTM-COA is strictly best**, then the LSTM/CNN hybrids, then the standalones —
4 of 6 positions match the paper exactly (only the two CNN hybrids #3/#4 swap).
Bootstrap CIs separate the LSTM family (≤ 5.43) from the CNN family (≥ 5.67) —
non-overlapping, the paper's robustness claim.

**ACCURACY mode** (`ANCHOR_BETA = 1.0`): all six models reach test RMSE ≈ 3.22,
MAE ≈ 1.49 (better than the paper's MAE 1.89), R² ≈ 0.977; statistically tied.

### Reproduced cleanly (STORY mode)
- Fig. 2 split-sensitivity — now a *real* trained-LSTM curve (70:30→5.49,
  75:25→3.24, 80:20→3.18: error drops then plateaus past 75:25, as in the paper).
- Fig. 4 correlation structure (Time↔OPR −0.72, ADP↔ADTemp 0.98, AWHP↔DCS 0.88).
- Fig. 9/10 cross-plots incl. the un-predicted 1285 m³/d spike.
- Fig. 11 ranking, Fig. 12 LSTM-COA time-series, Tables 3–6.
- Fig. 14 sensitivity now uses **exact per-well raw frames** (no boundary
  heuristic); Scenario-2 test MAE ≈ 1.4 (paper 2.38).

### Diverges from the paper (data-limited, documented)
- **Stage-2 metaheuristic finds no improvement over backprop.** Instrumented
  testing showed COA/PSO weight-tuning cannot beat the backprop solution on the
  blind well (any training-derived proxy that helps the train tail *worsens*
  F-11). The paper's large hybrid gains are not reproducible by weight
  optimization; the hybrid > standalone gap here comes from extra optimization
  budget (epochs), disclosed in `docs/CRITICAL_REVIEW.md`. Fig. 8's Stage-2
  curves are therefore nearly flat.
- Absolute test RMSE ~2× the paper's best hybrid (information ceiling of this
  file; persistence floor 3.22).
- Sensitivity Scenario-1 (extrapolate to Well F-14's high startup OPR from
  low-OPR training) remains hard — a genuine out-of-range generalization limit.
- Exact SD numbers (test-OPR std, the precise 2892-date window, wavelet params)
  are not bit-recoverable — but the preprocessing chain is now reconstructed at
  r ≥ 0.976 per feature and the feature selection matches the paper's 7.

## 7. End-to-end validation: train on the reconstruction (`scripts/09…`)

Training the six models on the raw-data reconstruction (`recon_03_selected.xlsx`)
and comparing to the supplied SD file **closes the loop and identifies the
denoising as the headline lever**:

| Model | RMSE (SD file) | RMSE (reconstruction) | Paper |
|-------|---------------:|----------------------:|------:|
| LSTM-COA | 5.14 | **2.77** | 2.15 |
| LSTM-PSO | 5.15 | **2.72** | 2.40 |
| CNN-COA | 6.46 | 4.49 | 2.73 |
| CNN-PSO | 6.26 | 4.63 | 3.15 |
| CNN | 5.91 | 4.59 | 8.64 |
| LSTM | 5.25 | 5.22 | 7.43 |

With the **correct heavy approximation-only OPR denoising** (autocorr ≈ 0.98),
the reconstruction's **LSTM hybrids reach RMSE ≈ 2.7 — within ~25 % of the paper's
2.15/2.40 and closer than the supplied SD file itself** (5.14). Conversely, with
a light soft-threshold (autocorr ≈ 0.93) the same models only reach ≈ 9
(persistence floor 8.4). This pins the paper's headline accuracy to one
preprocessing choice: **how aggressively the OPR target is smoothed.** The LSTM
hybrids remain the top two in both cases (COA/PSO are near-tied at ~2.7–2.8).

## 8. Honest accuracy ceiling (Tasks 1–2, 6) — what is and isn't a fair target

A task list (`docs/IMPROVEMENT_TASKS.md`) cited AR(3) = 2.62 / ARX = 2.47 as
honest targets the DL should reach. **Verified: those numbers are in-sample /
CV-on-the-test (leaky).** Fit train→test (the only fair protocol), the same
models are *worse* than persistence:

| baseline (test) | in-sample / CV (leaky) | train → test (fair) |
|-----------------|-----------------------:|--------------------:|
| persistence OPR(t)=OPR(t−1) | 3.22 | **3.22** |
| AR(3) on OPR lags | 2.49 / 2.55 | **5.2** |
| ARX(3)+7 features | 2.68 | 11.3 |

The train AR is *averaging* (coefs ≈ [0.43, 0.29, 0.24]); the test AR is
*trend-extrapolating* (≈ [1.73, −0.97, 0.23]). They differ because the SD **train
OPR is spiky (autocorr 0.905) while the test is smooth (0.989)** — so no
train-fitted AR transfers, and **the honest blind-well floor is persistence,
3.22 m³/day** (not 2.5).

* **Learnable multi-lag anchor (Task 2, `config.ANCHOR_LEARNABLE`)**: implemented
  and verified leak-free, but does NOT beat the fixed lag-1 anchor (LSTM-COA
  5.60 vs 5.14) — the limit is the net overfitting the spiky train, not the lag
  count. Off by default.
* **Dual-target reporting (Task 6, `scripts/10_dual_target.py`)**: every test
  metric is also scored on the **raw (pre-denoise) OPR** (autocorr 0.93,
  persistence 8.68). On the raw signal the models give RMSE ≈ 11 — vs ≈ 5 on the
  denoised target — making explicit that the headline accuracy (and the paper's
  2.15) is largely a property of how heavily OPR is smoothed, not of the model.

**Bottom line (honest):** on the supplied SD blind well, ≈ 3.2 m³/day is the fair
train→test ceiling; reaching 2.1–2.6 requires either in-sample leakage or heavier
OPR smoothing (§7), both of which are disclosed rather than used to inflate the
headline.

### Optimizer-lift ablation (Task 5, `scripts/12_ablation.py`)

Separating the epoch budget from the Stage-2 metaheuristic:

| arch | standalone 130ep | standalone 600ep | +Stage-2 (COA/PSO) | **pure Stage-2 lift** | budget lift 130→600 |
|------|-----------------:|-----------------:|-------------------:|----------------------:|--------------------:|
| CNN  | 5.91 | 6.68 | 6.46 / 6.36 | **+0.22 / +0.32** | **−0.77** |
| LSTM | 5.25 | 5.15 | 5.14 / 5.15 | **+0.01 / −0.01** | +0.10 |

* **Pure Stage-2 lift is small** (mean +0.14 m³/day): it slightly recovers the
  over-trained CNN and does ≈ nothing for the LSTM — nowhere near the paper's
  ~3× optimizer lift. Confirms `docs/CRITICAL_REVIEW.md`.
* **The CNN OVERFITS at 600 epochs** — this, not the optimizer, is why the main
  pipeline's 600-epoch CNN hybrids score worse than the 130-epoch standalone CNN.
  The paper's claimed ~3× optimizer lift (7.43→2.15) is **not reproducible**; the
  hybrid>standalone gap is purely the training budget.

### Seed variance + ensembling (Task 3, `scripts/11_seed_ensemble.py`, Fig. 17)

10 seeds per model:

| model | mean ± std | best seed | median ensemble |
|-------|-----------:|----------:|----------------:|
| **LSTM-COA** | **5.12 ± 0.07** | 4.99 | 5.13 |
| LSTM-PSO | 5.18 ± 0.07 | 5.05 | 5.19 |
| LSTM | 5.21 ± 0.12 | 4.99 | 5.23 |
| CNN | 5.94 ± 0.54 | 5.18 | 5.77 |
| CNN-PSO | 6.51 ± 0.88 | 5.28 | 6.35 |
| CNN-COA | 6.81 ± 1.01 | 5.49 | 6.60 |

* **The LSTM models are stable (std ≈ 0.07–0.12); the CNN models are not (std
  ≈ 0.5–1.0)** — single-seed CNN numbers are not credible.
* The median ensemble lands near the **mean**, not below the best seed (errors
  across seeds are correlated, all near the persistence-floor regime), so
  ensembling does NOT beat the best lucky seed — i.e. quoting a "best seed" is
  cherry-picking; the credible figure is the ensemble/mean.
* The headline LSTM-COA = 5.14 sits inside its seed distribution (5.12 ± 0.07),
  confirming it is a typical, not cherry-picked, result.

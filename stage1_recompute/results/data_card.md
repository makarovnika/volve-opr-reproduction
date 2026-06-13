# Data Card — Volve Stage-2

Generated: 2026-06-13 14:42
Source SHA-256 (first 16 chars): `55f7ab6c11ae14c5`

## Row counts vs TZ-01 §2 expected

| Check | Expected | Actual | Status |
| --- | --- | --- | --- |
| calendar_days | 3136 | 3136 | ✓ |
| train_rows | 2535 | 2535 | ✓ |
| test_rows | 601 | 601 | ✓ |
| hrs_gt_24 | 13 | 13 | ✓ |
| allocation_artifacts | 2 | 2 | ✓ |
| small_negative_wpr | 2 | 2 | ✓ |
| inconsistent_day | 5 | 5 | ✓ |
| zero_production_days_train | 133 | 133 | ✓ |

## Zero-as-missing counts (TZ-01 §2.4)

| Well × channel | Days zero while OPR>0 |
| --- | --- |
| F1C_AAP | 1 |
| F11H_ABHP | 2 |
| F11H_ABHT | 2 |
| F11H_ADPT | 2 |
| F11H_AAP | 3 |
| F11H_AWHP | 2 |
| F11H_AWHT | 2 |
| F12H_ABHP | 1906 |
| F12H_ABHT | 1906 |
| F12H_AWHP | 1 |
| F14H_ABHP | 15 |
| F14H_ABHT | 15 |
| F14H_AAP | 850 |
| F14H_AWHP | 1 |

## Imputation summary (TZ-01 §2.6)

| Flag | Value |
| --- | --- |
| imputed_F11H_ABHP (imputed days) | 3 |
| active_unfilled_F11H_ABHP | 5 |
| imputed_F11H_ABHT (imputed days) | 3 |
| active_unfilled_F11H_ABHT | 5 |
| imputed_F11H_ADPT (imputed days) | 3 |
| active_unfilled_F11H_ADPT | 5 |
| imputed_F11H_AAP (imputed days) | 4 |
| active_unfilled_F11H_AAP | 5 |
| imputed_F11H_ACS (imputed days) | 3 |
| imputed_F11H_AWHP (imputed days) | 3 |
| active_unfilled_F11H_AWHP | 5 |
| imputed_F11H_AWHT (imputed days) | 3 |
| active_unfilled_F11H_AWHT | 5 |
| imputed_F11H_CZ (imputed days) | 6 |
| imputed_F12H_ADPT (imputed days) | 70 |
| active_unfilled_F12H_ADPT | 6 |
| imputed_F12H_AAP (imputed days) | 70 |
| active_unfilled_F12H_AAP | 13 |
| imputed_F12H_ACS (imputed days) | 92 |
| imputed_F12H_AWHP (imputed days) | 71 |
| imputed_F12H_AWHT (imputed days) | 70 |
| imputed_F12H_CZ (imputed days) | 70 |
| imputed_F14H_ABHP (imputed days) | 72 |
| active_unfilled_F14H_ABHP | 18 |
| imputed_F14H_ABHT (imputed days) | 72 |
| active_unfilled_F14H_ABHT | 18 |
| imputed_F14H_ADPT (imputed days) | 69 |
| active_unfilled_F14H_ADPT | 6 |
| imputed_F14H_ACS (imputed days) | 75 |
| imputed_F14H_AWHP (imputed days) | 70 |
| imputed_F14H_AWHT (imputed days) | 69 |
| imputed_F14H_CZ (imputed days) | 69 |
| imputed_F15D_ABHP (imputed days) | 2 |
| imputed_F15D_ABHT (imputed days) | 2 |
| imputed_F15D_ADPT (imputed days) | 2 |
| imputed_F15D_AAP (imputed days) | 2 |
| imputed_F15D_ACS (imputed days) | 2 |
| imputed_F15D_AWHP (imputed days) | 2 |
| imputed_F15D_AWHT (imputed days) | 2 |
| imputed_F15D_CZ (imputed days) | 2 |

## Feature roster

Total feature columns: 85

```
F1C_HRS, F1C_ABHP, F1C_ABHT, F1C_ADPT, F1C_ACS, F1C_AWHP, F1C_AWHT, F1C_CZ, F11H_HRS, F11H_ABHP, F11H_ABHT, F11H_ADPT, F11H_AAP, F11H_ACS, F11H_AWHP, F11H_AWHT, F11H_CZ, F12H_HRS, F12H_ADPT, F12H_AAP, F12H_ACS, F12H_AWHP, F12H_AWHT, F12H_CZ, F14H_HRS, F14H_ABHP, F14H_ABHT, F14H_ADPT, F14H_ACS, F14H_AWHP, F14H_AWHT, F14H_CZ, F15D_HRS, F15D_ABHP, F15D_ABHT, F15D_ADPT, F15D_AAP, F15D_ACS, F15D_AWHP, F15D_AWHT, F15D_CZ, F4AH_HRS, F4AH_WIR, F5AH_HRS, F5AH_WIR, active_F1C, active_F11H, active_F12H, active_F14H, active_F15D, active_F4AH, active_F5AH, AW, f12_bhp_dead, imputed_F11H_ABHP, imputed_F11H_ABHT, imputed_F11H_ADPT, imputed_F11H_AAP, imputed_F11H_ACS, imputed_F11H_AWHP, imputed_F11H_AWHT, imputed_F11H_CZ, imputed_F12H_ADPT, imputed_F12H_AAP, imputed_F12H_ACS, imputed_F12H_AWHP, imputed_F12H_AWHT, imputed_F12H_CZ, imputed_F14H_ABHP, imputed_F14H_ABHT, imputed_F14H_ADPT, imputed_F14H_ACS, imputed_F14H_AWHP, imputed_F14H_AWHT, imputed_F14H_CZ, imputed_F15D_ABHP, imputed_F15D_ABHT, imputed_F15D_ADPT, imputed_F15D_AAP, imputed_F15D_ACS, imputed_F15D_AWHP, imputed_F15D_AWHT, imputed_F15D_CZ, inconsistent_F1C, inconsistent_F15D
```

## Split summary

Train: 2008-02-17 – 2015-01-25 (2535 rows)
Test:  2015-01-26 – 2016-09-17 (601 rows)
Val:   last 15% of train

Train OPR mean: 152.4 m³/day
Test  OPR mean: 75.3 m³/day  ← M2 regime shift

## Frozen decisions (from §12 of preprocessing plan)

- F-12 ABHP/ABHT excluded → `f12_bhp_dead` indicator (1 from 2011-01-01)
- F-1C AAP and F-14 AAP excluded
- Injectors: HRS + WIR only (all pressure/T channels ≥94.6% NaN)
- AW (sum of 5 producer activity masks) included as feature
- GPR kept as third target; GOR ≈ 150
- Zero-production days kept (rule predictor handles them)
- RobustScaler fit on train only; targets inverse-transformed before metrics
- NOTE: spec says '3135 days' counting day-intervals; actual row count = 3136

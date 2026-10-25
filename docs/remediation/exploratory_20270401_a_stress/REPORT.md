# A-default stress exploratory B=600

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**
- **Exploratory B=600 A-default stress. Not a locked evaluation.**

- Seed `20270401`; 600 replicates; complete data; 11 scenarios × 2 sizes × 5 analyses.
- Frozen matrix from `timing_pilot_20270301_a_stress`. Independent of that timing seed.
- Mechanisms: covariance shift (`sigma_p` ×1.6, `ell` ×2), heavy tails (t(3) measurement ×1.5), delayed alternatives.
- A is the subject. B and C1 are comparators. Production defaults stay `[ancova]`. C1 is not retuned.
- Fits recorded: **66000**. Study datasets: **13200**.
- Wall clock: **31.234 s**.
- Common random numbers: one weekly latent path per (scenario, size, replicate); biweekly is a visit subset.
- Working covariance: unstressed `HierarchicalCovariance.animal_cov` (historical plug-in stand-in).
- Failed analyses count as non-rejections and remain in the FWER/power denominator.
- Delayed FWER uses partial-null cells only (no delayed global-null; shape is irrelevant at (0,0)).

## Sourced tolerances (not exact 0.05 control)

From `configs/new_study.example.yaml` and `optimize.py::feasibility`:

- FWER point ceiling 0.065 (0.05+0.015); CI ceiling 0.095 (0.05+0.045).
- Coverage floor 0.92; bias fraction 0.15; failure rate 0.02.
- Power target 0.8 with CI floor 0.74 applies to a chosen design, not every small n.
- Passing a tolerated FWER gate is not “FWER = 0.05.”

## Review by mechanism (exploratory, not locked)

| Analysis | Mechanism | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference | Power if chosen |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | 6/6/6 | 0.055 [0.076] | 0.945 | 0.007 | 0.000 | 0.992 [0.981] | yes | yes |
| `B_rm_gls_quadratic` | covariance_shift | weekly | 6/6/6 | 0.023 [0.039] | 0.967 | 0.007 | 0.000 | 0.992 [0.981] | yes | yes |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | 6/6/6 | 0.030 [0.047] | 0.963 | 0.007 | 0.000 | 0.987 [0.974] | yes | yes |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | 6/6/6 | 0.053 [0.074] | 0.938 | 0.007 | 0.000 | 0.995 [0.985] | yes | yes |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | 6/6/6 | 0.053 [0.074] | 0.938 | 0.007 | 0.000 | 0.995 [0.985] | yes | yes |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | 8/8/8 | 0.060 [0.082] | 0.943 | 0.003 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | covariance_shift | weekly | 8/8/8 | 0.035 [0.053] | 0.958 | 0.002 | 0.000 | 0.997 [0.988] | yes | yes |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | 8/8/8 | 0.033 [0.051] | 0.963 | 0.003 | 0.000 | 0.997 [0.988] | yes | yes |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | 8/8/8 | 0.060 [0.082] | 0.937 | 0.003 | 0.000 | 0.997 [0.988] | yes | yes |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | 8/8/8 | 0.060 [0.082] | 0.937 | 0.003 | 0.000 | 0.997 [0.988] | yes | yes |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | 6/6/6 | 0.050 [0.071] | 0.948 | 0.004 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | heavy_tails | weekly | 6/6/6 | 0.020 [0.035] | 0.968 | 0.004 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | 6/6/6 | 0.027 [0.043] | 0.960 | 0.004 | 0.000 | 1.000 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | 6/6/6 | 0.052 [0.073] | 0.945 | 0.004 | 0.000 | 1.000 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | 6/6/6 | 0.052 [0.073] | 0.945 | 0.004 | 0.000 | 1.000 [0.994] | yes | yes |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | 8/8/8 | 0.052 [0.073] | 0.943 | 0.002 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | heavy_tails | weekly | 8/8/8 | 0.030 [0.047] | 0.962 | 0.001 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | 8/8/8 | 0.040 [0.059] | 0.958 | 0.002 | 0.000 | 1.000 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | 8/8/8 | 0.060 [0.082] | 0.940 | 0.002 | 0.000 | 1.000 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | 8/8/8 | 0.060 [0.082] | 0.940 | 0.002 | 0.000 | 1.000 [0.994] | yes | yes |
| `A_endpoint_ancova` | delayed | endpoint_only | 6/6/6 | 0.062 [0.084] | 0.938 | 0.001 | 0.000 | 0.998 [0.991] | yes | yes |
| `B_rm_gls_quadratic` | delayed | weekly | 6/6/6 | 0.050 [0.071] | 0.950 | 0.005 | 0.000 | 0.998 [0.991] | yes | yes |
| `B_rm_gls_quadratic` | delayed | biweekly | 6/6/6 | 0.055 [0.076] | 0.945 | 0.001 | 0.000 | 0.998 [0.991] | yes | yes |
| `C1_mmrm_visit_contrast` | delayed | weekly | 6/6/6 | 0.067 [0.090] | 0.933 | 0.001 | 0.000 | 0.998 [0.991] | no | yes |
| `C1_mmrm_visit_contrast` | delayed | biweekly | 6/6/6 | 0.067 [0.090] | 0.933 | 0.001 | 0.000 | 0.998 [0.991] | no | yes |
| `A_endpoint_ancova` | delayed | endpoint_only | 8/8/8 | 0.038 [0.057] | 0.945 | 0.008 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | delayed | weekly | 8/8/8 | 0.042 [0.061] | 0.948 | 0.005 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | delayed | biweekly | 8/8/8 | 0.042 [0.061] | 0.942 | 0.008 | 0.000 | 1.000 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | delayed | weekly | 8/8/8 | 0.047 [0.067] | 0.935 | 0.008 | 0.000 | 1.000 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | delayed | biweekly | 8/8/8 | 0.047 [0.067] | 0.935 | 0.008 | 0.000 | 1.000 [0.994] | yes | yes |

“Inference” is FWER + coverage + alternative bias + failures within that mechanism. It is not a primary-analysis decision and not design qualification.

## Cells that miss a sourced inference tolerance

- `C1_mmrm_visit_contrast` delayed weekly 6/6/6: FWER 0.067 (worst delayed_partial_null_low)
- `C1_mmrm_visit_contrast` delayed biweekly 6/6/6: FWER 0.067 (worst delayed_partial_null_low)

## Per-cell operating characteristics

Full rows are in `cell_oc.csv`. Replicate-level draws are not archived; this runner plus the seed reproduces them.

| Analysis | Mechanism | Schedule | Scenario | n/sex | FWER | Coverage min | Bias max | Power-all | Fail |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_alternative | 6/6/6 | — | 0.947 | 0.001 | 0.992 [0.981, 0.997] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_global_null | 6/6/6 | 0.040 [0.026, 0.059] | 0.958 | 0.002 | — | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_partial_null_high | 6/6/6 | 0.055 [0.038, 0.076] | 0.945 | 0.003 | 0.982 [0.967, 0.991] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_partial_null_low | 6/6/6 | 0.038 [0.024, 0.057] | 0.952 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_alternative | 8/8/8 | — | 0.943 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_global_null | 8/8/8 | 0.060 [0.042, 0.082] | 0.945 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_partial_null_high | 8/8/8 | 0.048 [0.033, 0.069] | 0.952 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_partial_null_low | 8/8/8 | 0.047 [0.031, 0.067] | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_alternative | 6/6/6 | — | 0.940 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_partial_null_high | 6/6/6 | 0.043 [0.028, 0.063] | 0.952 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_partial_null_low | 6/6/6 | 0.062 [0.044, 0.084] | 0.938 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_alternative | 8/8/8 | — | 0.945 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_partial_null_high | 8/8/8 | 0.038 [0.024, 0.057] | 0.962 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_partial_null_low | 8/8/8 | 0.038 [0.024, 0.057] | 0.948 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_alternative | 6/6/6 | — | 0.958 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_global_null | 6/6/6 | 0.033 [0.020, 0.051] | 0.957 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_partial_null_high | 6/6/6 | 0.043 [0.028, 0.063] | 0.948 | 0.001 | 0.990 [0.978, 0.996] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_partial_null_low | 6/6/6 | 0.050 [0.034, 0.071] | 0.950 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_alternative | 8/8/8 | — | 0.955 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_global_null | 8/8/8 | 0.042 [0.027, 0.061] | 0.943 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_partial_null_high | 8/8/8 | 0.048 [0.033, 0.069] | 0.952 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_partial_null_low | 8/8/8 | 0.052 [0.035, 0.073] | 0.948 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_alternative | 6/6/6 | — | 0.972 | 0.001 | 0.987 [0.974, 0.994] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_global_null | 6/6/6 | 0.013 [0.006, 0.026] | 0.983 | 0.002 | — | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_partial_null_high | 6/6/6 | 0.030 [0.018, 0.047] | 0.963 | 0.003 | 0.968 [0.951, 0.981] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_partial_null_low | 6/6/6 | 0.015 [0.007, 0.028] | 0.975 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_alternative | 8/8/8 | — | 0.968 | 0.000 | 0.997 [0.988, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_global_null | 8/8/8 | 0.033 [0.020, 0.051] | 0.963 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_partial_null_high | 8/8/8 | 0.027 [0.015, 0.043] | 0.973 | 0.001 | 0.995 [0.985, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_partial_null_low | 8/8/8 | 0.028 [0.017, 0.045] | 0.972 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_alternative | 6/6/6 | — | 0.970 | 0.001 | 0.992 [0.981, 0.997] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_global_null | 6/6/6 | 0.012 [0.005, 0.024] | 0.983 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_partial_null_high | 6/6/6 | 0.023 [0.013, 0.039] | 0.967 | 0.003 | 0.968 [0.951, 0.981] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_partial_null_low | 6/6/6 | 0.012 [0.005, 0.024] | 0.978 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_alternative | 8/8/8 | — | 0.973 | 0.000 | 0.997 [0.988, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_global_null | 8/8/8 | 0.035 [0.022, 0.053] | 0.958 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_partial_null_high | 8/8/8 | 0.027 [0.015, 0.043] | 0.973 | 0.001 | 0.998 [0.991, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_partial_null_low | 8/8/8 | 0.030 [0.018, 0.047] | 0.970 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_alternative | 6/6/6 | — | 0.947 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_partial_null_high | 6/6/6 | 0.048 [0.033, 0.069] | 0.952 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_partial_null_low | 6/6/6 | 0.055 [0.038, 0.076] | 0.945 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_alternative | 8/8/8 | — | 0.942 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_partial_null_high | 8/8/8 | 0.042 [0.027, 0.061] | 0.958 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_partial_null_low | 8/8/8 | 0.037 [0.023, 0.055] | 0.960 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_alternative | 6/6/6 | — | 0.952 | 0.001 | 0.998 [0.991, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_partial_null_high | 6/6/6 | 0.042 [0.027, 0.061] | 0.958 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_partial_null_low | 6/6/6 | 0.050 [0.034, 0.071] | 0.950 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_alternative | 8/8/8 | — | 0.948 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_partial_null_high | 8/8/8 | 0.042 [0.027, 0.061] | 0.958 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_partial_null_low | 8/8/8 | 0.028 [0.017, 0.045] | 0.963 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_alternative | 6/6/6 | — | 0.963 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_global_null | 6/6/6 | 0.020 [0.010, 0.035] | 0.967 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_partial_null_high | 6/6/6 | 0.027 [0.015, 0.043] | 0.960 | 0.001 | 0.990 [0.978, 0.996] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_partial_null_low | 6/6/6 | 0.027 [0.015, 0.043] | 0.970 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_alternative | 8/8/8 | — | 0.962 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_global_null | 8/8/8 | 0.032 [0.019, 0.049] | 0.958 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_partial_null_high | 8/8/8 | 0.040 [0.026, 0.059] | 0.960 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_partial_null_low | 8/8/8 | 0.038 [0.024, 0.057] | 0.958 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_alternative | 6/6/6 | — | 0.970 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_global_null | 6/6/6 | 0.012 [0.005, 0.024] | 0.975 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_partial_null_high | 6/6/6 | 0.020 [0.010, 0.035] | 0.968 | 0.001 | 0.987 [0.974, 0.994] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_partial_null_low | 6/6/6 | 0.020 [0.010, 0.035] | 0.977 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_alternative | 8/8/8 | — | 0.962 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_global_null | 8/8/8 | 0.025 [0.014, 0.041] | 0.968 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_partial_null_high | 8/8/8 | 0.030 [0.018, 0.047] | 0.970 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_partial_null_low | 8/8/8 | 0.028 [0.017, 0.045] | 0.963 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_alternative | 6/6/6 | — | 0.947 | 0.001 | 0.995 [0.985, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_global_null | 6/6/6 | 0.048 [0.033, 0.069] | 0.952 | 0.002 | — | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_partial_null_high | 6/6/6 | 0.053 [0.037, 0.074] | 0.938 | 0.003 | 0.983 [0.970, 0.992] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_partial_null_low | 6/6/6 | 0.045 [0.030, 0.065] | 0.947 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_alternative | 8/8/8 | — | 0.942 | 0.000 | 0.997 [0.988, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_global_null | 8/8/8 | 0.060 [0.042, 0.082] | 0.937 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_partial_null_high | 8/8/8 | 0.043 [0.028, 0.063] | 0.957 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_partial_null_low | 8/8/8 | 0.047 [0.031, 0.067] | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_alternative | 6/6/6 | — | 0.947 | 0.001 | 0.995 [0.985, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_global_null | 6/6/6 | 0.048 [0.033, 0.069] | 0.952 | 0.002 | — | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_partial_null_high | 6/6/6 | 0.053 [0.037, 0.074] | 0.938 | 0.003 | 0.983 [0.970, 0.992] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_partial_null_low | 6/6/6 | 0.045 [0.030, 0.065] | 0.947 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_alternative | 8/8/8 | — | 0.942 | 0.000 | 0.997 [0.988, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_global_null | 8/8/8 | 0.060 [0.042, 0.082] | 0.937 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_partial_null_high | 8/8/8 | 0.043 [0.028, 0.063] | 0.957 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_partial_null_low | 8/8/8 | 0.047 [0.031, 0.067] | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_alternative | 6/6/6 | — | 0.935 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_partial_null_high | 6/6/6 | 0.057 [0.040, 0.078] | 0.943 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_partial_null_low | 6/6/6 | 0.067 [0.048, 0.090] | 0.933 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_alternative | 8/8/8 | — | 0.935 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_partial_null_high | 8/8/8 | 0.047 [0.031, 0.067] | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_partial_null_low | 8/8/8 | 0.045 [0.030, 0.065] | 0.948 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_alternative | 6/6/6 | — | 0.935 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_partial_null_high | 6/6/6 | 0.057 [0.040, 0.078] | 0.943 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_partial_null_low | 6/6/6 | 0.067 [0.048, 0.090] | 0.933 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_alternative | 8/8/8 | — | 0.935 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_partial_null_high | 8/8/8 | 0.047 [0.031, 0.067] | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_partial_null_low | 8/8/8 | 0.045 [0.030, 0.065] | 0.948 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_alternative | 6/6/6 | — | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_global_null | 6/6/6 | 0.040 [0.026, 0.059] | 0.948 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_partial_null_high | 6/6/6 | 0.043 [0.028, 0.063] | 0.945 | 0.001 | 0.992 [0.981, 0.997] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_partial_null_low | 6/6/6 | 0.052 [0.035, 0.073] | 0.948 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_alternative | 8/8/8 | — | 0.952 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_global_null | 8/8/8 | 0.055 [0.038, 0.076] | 0.942 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_partial_null_high | 8/8/8 | 0.055 [0.038, 0.076] | 0.945 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_partial_null_low | 8/8/8 | 0.060 [0.042, 0.082] | 0.940 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_alternative | 6/6/6 | — | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_global_null | 6/6/6 | 0.040 [0.026, 0.059] | 0.948 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_partial_null_high | 6/6/6 | 0.043 [0.028, 0.063] | 0.945 | 0.001 | 0.992 [0.981, 0.997] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_partial_null_low | 6/6/6 | 0.052 [0.035, 0.073] | 0.948 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_alternative | 8/8/8 | — | 0.952 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_global_null | 8/8/8 | 0.055 [0.038, 0.076] | 0.942 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_partial_null_high | 8/8/8 | 0.055 [0.038, 0.076] | 0.945 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_partial_null_low | 8/8/8 | 0.060 [0.042, 0.082] | 0.940 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |

## Interpretation (exploratory only)

- **A** (subject) inference tolerances: **6/6** mechanism×size cells.
- A met sourced inference tolerances on all three deferred mechanisms at both sizes.
- **B** (comparator) inference tolerances: **12/12**. A B miss does not restore GLS as a default.
- **C1** (comparator) inference tolerances: **10/12**. A C1 pass does not promote C1; a miss is not a retune.
- Meeting a tolerated FWER gate is not “FWER = 0.05.” This stream does not change production defaults.
- High alternative power reflects the −10%/−15% cells, not the deferred 5% effect.

## Constraints on use

- Do not treat this stream as locked confirmation (that would be independent seed `20270501`, B=2000).
- Do not retune C1 on these draws and then call a later stream confirmatory.
- Do not restore GLS as a production default.
- Do not change the ANCOVA default from this exploratory stream.
- Do not claim prior-outcome MAR validity. Missingness here is complete.
- Still deferred: 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2, MCAR on these stresses.

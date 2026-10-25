# A-default stress locked B=2000

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**
- **Locked B=2000 A-default stress. Independent of exploratory draws. Not a default-switch.**

- Seed `20270501`; 2000 replicates; complete data; 11 scenarios × 2 sizes × 5 analyses.
- Frozen matrix from `timing_pilot_20270301_a_stress` / `exploratory_20270401_a_stress`.
- Independent of exploratory seed `20270401`. C1 was not retuned.
- Mechanisms: covariance shift (`sigma_p` ×1.6, `ell` ×2), heavy tails (t(3) measurement ×1.5), delayed alternatives.
- A is the subject. B and C1 are comparators. Production defaults stay `[ancova]`.
- Fits recorded: **220000**. Study datasets: **44000**.
- Wall clock: **102.292 s**.
- Common random numbers: one weekly latent path per (scenario, size, replicate); biweekly is a visit subset.
- Working covariance: unstressed `HierarchicalCovariance.animal_cov` (historical plug-in stand-in).
- Failed analyses count as non-rejections and remain in the FWER/power denominator.
- Delayed FWER uses partial-null cells only.
- Replicate-level draws are not archived. This runner plus the seed reproduces them.

## Sourced tolerances (not exact 0.05 control)

From `configs/new_study.example.yaml` and `optimize.py::feasibility`:

- FWER point ceiling 0.065 (0.05+0.015); CI ceiling 0.095 (0.05+0.045).
- Coverage floor 0.92; bias fraction 0.15; failure rate 0.02.
- Power target 0.8 with CI floor 0.74 applies to a chosen design, not every small n.
- Passing a tolerated FWER gate is not “FWER = 0.05.”

## Review by mechanism (locked diagnostic, not qualification)

| Analysis | Mechanism | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference | Power if chosen |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | 6/6/6 | 0.050 [0.060] | 0.946 | 0.007 | 0.000 | 0.994 [0.989] | yes | yes |
| `B_rm_gls_quadratic` | covariance_shift | weekly | 6/6/6 | 0.025 [0.033] | 0.969 | 0.007 | 0.000 | 0.990 [0.984] | yes | yes |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | 6/6/6 | 0.026 [0.034] | 0.970 | 0.007 | 0.000 | 0.988 [0.982] | yes | yes |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | 6/6/6 | 0.053 [0.064] | 0.942 | 0.007 | 0.000 | 0.992 [0.986] | yes | yes |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | 6/6/6 | 0.053 [0.064] | 0.942 | 0.007 | 0.000 | 0.992 [0.986] | yes | yes |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | 8/8/8 | 0.053 [0.064] | 0.947 | 0.002 | 0.000 | 1.000 [0.998] | yes | yes |
| `B_rm_gls_quadratic` | covariance_shift | weekly | 8/8/8 | 0.027 [0.035] | 0.971 | 0.003 | 0.000 | 0.999 [0.996] | yes | yes |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | 8/8/8 | 0.029 [0.037] | 0.968 | 0.002 | 0.000 | 0.999 [0.996] | yes | yes |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | 8/8/8 | 0.054 [0.065] | 0.943 | 0.002 | 0.000 | 1.000 [0.998] | yes | yes |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | 8/8/8 | 0.054 [0.065] | 0.943 | 0.002 | 0.000 | 1.000 [0.998] | yes | yes |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | 6/6/6 | 0.049 [0.059] | 0.946 | 0.003 | 0.000 | 0.998 [0.995] | yes | yes |
| `B_rm_gls_quadratic` | heavy_tails | weekly | 6/6/6 | 0.033 [0.041] | 0.964 | 0.002 | 0.000 | 0.997 [0.993] | yes | yes |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | 6/6/6 | 0.041 [0.050] | 0.959 | 0.003 | 0.000 | 0.998 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | 6/6/6 | 0.056 [0.067] | 0.940 | 0.003 | 0.000 | 0.998 [0.995] | yes | yes |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | 6/6/6 | 0.056 [0.067] | 0.940 | 0.003 | 0.000 | 0.998 [0.995] | yes | yes |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | 8/8/8 | 0.055 [0.066] | 0.944 | 0.001 | 0.000 | 1.000 [0.998] | yes | yes |
| `B_rm_gls_quadratic` | heavy_tails | weekly | 8/8/8 | 0.035 [0.043] | 0.964 | 0.000 | 0.000 | 0.999 [0.996] | yes | yes |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | 8/8/8 | 0.046 [0.057] | 0.954 | 0.001 | 0.000 | 0.999 [0.996] | yes | yes |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | 8/8/8 | 0.061 [0.073] | 0.939 | 0.001 | 0.000 | 1.000 [0.998] | yes | yes |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | 8/8/8 | 0.061 [0.073] | 0.939 | 0.001 | 0.000 | 1.000 [0.998] | yes | yes |
| `A_endpoint_ancova` | delayed | endpoint_only | 6/6/6 | 0.050 [0.060] | 0.945 | 0.001 | 0.000 | 0.999 [0.996] | yes | yes |
| `B_rm_gls_quadratic` | delayed | weekly | 6/6/6 | 0.040 [0.050] | 0.951 | 0.005 | 0.000 | 0.998 [0.995] | yes | yes |
| `B_rm_gls_quadratic` | delayed | biweekly | 6/6/6 | 0.046 [0.056] | 0.945 | 0.001 | 0.000 | 0.999 [0.996] | yes | yes |
| `C1_mmrm_visit_contrast` | delayed | weekly | 6/6/6 | 0.060 [0.072] | 0.936 | 0.001 | 0.000 | 0.999 [0.996] | yes | yes |
| `C1_mmrm_visit_contrast` | delayed | biweekly | 6/6/6 | 0.060 [0.072] | 0.936 | 0.001 | 0.000 | 0.999 [0.996] | yes | yes |
| `A_endpoint_ancova` | delayed | endpoint_only | 8/8/8 | 0.049 [0.059] | 0.943 | 0.005 | 0.000 | 1.000 [0.998] | yes | yes |
| `B_rm_gls_quadratic` | delayed | weekly | 8/8/8 | 0.043 [0.052] | 0.955 | 0.008 | 0.000 | 1.000 [0.998] | yes | yes |
| `B_rm_gls_quadratic` | delayed | biweekly | 8/8/8 | 0.044 [0.054] | 0.941 | 0.005 | 0.000 | 1.000 [0.998] | yes | yes |
| `C1_mmrm_visit_contrast` | delayed | weekly | 8/8/8 | 0.050 [0.060] | 0.935 | 0.005 | 0.000 | 1.000 [0.998] | yes | yes |
| `C1_mmrm_visit_contrast` | delayed | biweekly | 8/8/8 | 0.050 [0.060] | 0.935 | 0.005 | 0.000 | 1.000 [0.998] | yes | yes |

“Inference” is FWER + coverage + alternative bias + failures within that mechanism. It is not a primary-analysis decision and not design qualification.

## Cells that miss a sourced inference tolerance

- None of the mechanism×analysis×schedule×size reviews missed FWER, coverage, bias, or failure tolerances.

## Agreement with exploratory B=600 inference decisions

Same pre-specified analyses and matrix; independent seed. **28/30** decisions match. A changed decision is reported, not patched.

| Analysis | Mechanism | Schedule | n/sex | Exploratory | Locked | Same decision | FWER expl → locked |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | 6/6/6 | met | met | yes | 0.055 → 0.050 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | 6/6/6 | met | met | yes | 0.023 → 0.025 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | 6/6/6 | met | met | yes | 0.030 → 0.026 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | 6/6/6 | met | met | yes | 0.053 → 0.053 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | 6/6/6 | met | met | yes | 0.053 → 0.053 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | 8/8/8 | met | met | yes | 0.060 → 0.053 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | 8/8/8 | met | met | yes | 0.035 → 0.027 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | 8/8/8 | met | met | yes | 0.033 → 0.029 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | 8/8/8 | met | met | yes | 0.060 → 0.054 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | 8/8/8 | met | met | yes | 0.060 → 0.054 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | 6/6/6 | met | met | yes | 0.050 → 0.049 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | 6/6/6 | met | met | yes | 0.020 → 0.033 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | 6/6/6 | met | met | yes | 0.027 → 0.041 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | 6/6/6 | met | met | yes | 0.052 → 0.056 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | 6/6/6 | met | met | yes | 0.052 → 0.056 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | 8/8/8 | met | met | yes | 0.052 → 0.055 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | 8/8/8 | met | met | yes | 0.030 → 0.035 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | 8/8/8 | met | met | yes | 0.040 → 0.046 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | 8/8/8 | met | met | yes | 0.060 → 0.061 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | 8/8/8 | met | met | yes | 0.060 → 0.061 |
| `A_endpoint_ancova` | delayed | endpoint_only | 6/6/6 | met | met | yes | 0.062 → 0.050 |
| `B_rm_gls_quadratic` | delayed | weekly | 6/6/6 | met | met | yes | 0.050 → 0.040 |
| `B_rm_gls_quadratic` | delayed | biweekly | 6/6/6 | met | met | yes | 0.055 → 0.046 |
| `C1_mmrm_visit_contrast` | delayed | weekly | 6/6/6 | missed | met | NO | 0.067 → 0.060 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | 6/6/6 | missed | met | NO | 0.067 → 0.060 |
| `A_endpoint_ancova` | delayed | endpoint_only | 8/8/8 | met | met | yes | 0.038 → 0.049 |
| `B_rm_gls_quadratic` | delayed | weekly | 8/8/8 | met | met | yes | 0.042 → 0.043 |
| `B_rm_gls_quadratic` | delayed | biweekly | 8/8/8 | met | met | yes | 0.042 → 0.044 |
| `C1_mmrm_visit_contrast` | delayed | weekly | 8/8/8 | met | met | yes | 0.047 → 0.050 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | 8/8/8 | met | met | yes | 0.047 → 0.050 |

## Complete-data identities (audit, not a gate)

- Successful complete-data triples compared: **44000**.
- Max |est A − est C1 weekly|: **1.141e-14**.
- Max |est C1 weekly − C1 biweekly|: **1.044e-14**.
- Max |SE C1 weekly − C1 biweekly|: **1.606e-15**.
- Holm disagreements A vs C1 weekly: **589**.
- Holm disagreements C1 weekly vs biweekly: **0**.

## Per-cell operating characteristics

Full rows are in `cell_oc.csv`.

| Analysis | Mechanism | Schedule | Scenario | n/sex | FWER | Coverage min | Bias max | Power-all | Fail |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_alternative | 6/6/6 | — | 0.946 | 0.001 | 0.994 [0.989, 0.997] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_global_null | 6/6/6 | 0.046 [0.038, 0.057] | 0.947 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_partial_null_high | 6/6/6 | 0.037 [0.030, 0.047] | 0.948 | 0.001 | 0.979 [0.972, 0.985] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_partial_null_low | 6/6/6 | 0.050 [0.040, 0.060] | 0.950 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_alternative | 8/8/8 | — | 0.950 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_global_null | 8/8/8 | 0.053 [0.044, 0.064] | 0.950 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_partial_null_high | 8/8/8 | 0.052 [0.043, 0.063] | 0.947 | 0.000 | 0.996 [0.992, 0.998] | 0.000 |
| `A_endpoint_ancova` | covariance_shift | endpoint_only | covshift_partial_null_low | 8/8/8 | 0.042 [0.034, 0.052] | 0.952 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_alternative | 6/6/6 | — | 0.952 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_partial_null_high | 6/6/6 | 0.050 [0.040, 0.060] | 0.945 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_partial_null_low | 6/6/6 | 0.046 [0.037, 0.056] | 0.945 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_alternative | 8/8/8 | — | 0.948 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_partial_null_high | 8/8/8 | 0.049 [0.040, 0.059] | 0.952 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | delayed | endpoint_only | delayed_partial_null_low | 8/8/8 | 0.046 [0.038, 0.057] | 0.943 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_alternative | 6/6/6 | — | 0.954 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_global_null | 6/6/6 | 0.049 [0.040, 0.059] | 0.948 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_partial_null_high | 6/6/6 | 0.045 [0.037, 0.056] | 0.953 | 0.000 | 0.995 [0.991, 0.998] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_partial_null_low | 6/6/6 | 0.048 [0.039, 0.058] | 0.946 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_alternative | 8/8/8 | — | 0.944 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_global_null | 8/8/8 | 0.040 [0.032, 0.050] | 0.953 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_partial_null_high | 8/8/8 | 0.041 [0.033, 0.051] | 0.953 | 0.001 | 0.999 [0.996, 1.000] | 0.000 |
| `A_endpoint_ancova` | heavy_tails | endpoint_only | heavytail_partial_null_low | 8/8/8 | 0.055 [0.045, 0.066] | 0.945 | 0.000 | 1.000 [0.997, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_alternative | 6/6/6 | — | 0.973 | 0.001 | 0.988 [0.982, 0.992] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_global_null | 6/6/6 | 0.021 [0.015, 0.028] | 0.972 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_partial_null_high | 6/6/6 | 0.018 [0.012, 0.024] | 0.977 | 0.001 | 0.966 [0.957, 0.974] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_partial_null_low | 6/6/6 | 0.026 [0.019, 0.034] | 0.970 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_alternative | 8/8/8 | — | 0.972 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_global_null | 8/8/8 | 0.029 [0.022, 0.037] | 0.968 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_partial_null_high | 8/8/8 | 0.027 [0.020, 0.035] | 0.972 | 0.000 | 0.996 [0.991, 0.998] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | biweekly | covshift_partial_null_low | 8/8/8 | 0.022 [0.016, 0.030] | 0.970 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_alternative | 6/6/6 | — | 0.969 | 0.001 | 0.990 [0.984, 0.993] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_global_null | 6/6/6 | 0.020 [0.014, 0.027] | 0.973 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_partial_null_high | 6/6/6 | 0.015 [0.010, 0.021] | 0.978 | 0.001 | 0.969 [0.960, 0.976] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_partial_null_low | 6/6/6 | 0.025 [0.019, 0.033] | 0.973 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_alternative | 8/8/8 | — | 0.975 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_global_null | 8/8/8 | 0.025 [0.018, 0.032] | 0.971 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_partial_null_high | 8/8/8 | 0.027 [0.020, 0.035] | 0.973 | 0.000 | 0.996 [0.992, 0.998] | 0.000 |
| `B_rm_gls_quadratic` | covariance_shift | weekly | covshift_partial_null_low | 8/8/8 | 0.022 [0.016, 0.029] | 0.973 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_alternative | 6/6/6 | — | 0.955 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_partial_null_high | 6/6/6 | 0.046 [0.037, 0.056] | 0.945 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_partial_null_low | 6/6/6 | 0.040 [0.031, 0.049] | 0.954 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_alternative | 8/8/8 | — | 0.953 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_partial_null_high | 8/8/8 | 0.044 [0.036, 0.054] | 0.955 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | biweekly | delayed_partial_null_low | 8/8/8 | 0.043 [0.035, 0.053] | 0.941 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_alternative | 6/6/6 | — | 0.963 | 0.001 | 0.998 [0.995, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_partial_null_high | 6/6/6 | 0.040 [0.032, 0.050] | 0.951 | 0.001 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_partial_null_low | 6/6/6 | 0.034 [0.026, 0.043] | 0.960 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_alternative | 8/8/8 | — | 0.956 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_partial_null_high | 8/8/8 | 0.043 [0.034, 0.052] | 0.958 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | delayed | weekly | delayed_partial_null_low | 8/8/8 | 0.037 [0.029, 0.046] | 0.955 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_alternative | 6/6/6 | — | 0.960 | 0.000 | 0.998 [0.994, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_global_null | 6/6/6 | 0.035 [0.027, 0.044] | 0.962 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_partial_null_high | 6/6/6 | 0.036 [0.029, 0.046] | 0.959 | 0.000 | 0.994 [0.989, 0.997] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_partial_null_low | 6/6/6 | 0.041 [0.032, 0.050] | 0.960 | 0.000 | 1.000 [0.997, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_alternative | 8/8/8 | — | 0.957 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_global_null | 8/8/8 | 0.026 [0.019, 0.034] | 0.965 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_partial_null_high | 8/8/8 | 0.030 [0.023, 0.038] | 0.964 | 0.001 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | biweekly | heavytail_partial_null_low | 8/8/8 | 0.046 [0.038, 0.057] | 0.954 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_alternative | 6/6/6 | — | 0.965 | 0.000 | 0.997 [0.993, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_global_null | 6/6/6 | 0.025 [0.019, 0.033] | 0.969 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_partial_null_high | 6/6/6 | 0.026 [0.019, 0.034] | 0.970 | 0.000 | 0.992 [0.987, 0.995] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_partial_null_low | 6/6/6 | 0.033 [0.025, 0.041] | 0.964 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_alternative | 8/8/8 | — | 0.964 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_global_null | 8/8/8 | 0.022 [0.016, 0.030] | 0.970 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_partial_null_high | 8/8/8 | 0.029 [0.022, 0.037] | 0.970 | 0.001 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | heavy_tails | weekly | heavytail_partial_null_low | 8/8/8 | 0.035 [0.027, 0.043] | 0.966 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_alternative | 6/6/6 | — | 0.944 | 0.001 | 0.992 [0.986, 0.995] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_global_null | 6/6/6 | 0.053 [0.044, 0.064] | 0.944 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_partial_null_high | 6/6/6 | 0.042 [0.034, 0.052] | 0.942 | 0.001 | 0.981 [0.973, 0.986] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_partial_null_low | 6/6/6 | 0.053 [0.044, 0.064] | 0.946 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_alternative | 8/8/8 | — | 0.947 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_global_null | 8/8/8 | 0.054 [0.045, 0.065] | 0.943 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_partial_null_high | 8/8/8 | 0.053 [0.044, 0.064] | 0.946 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | biweekly | covshift_partial_null_low | 8/8/8 | 0.049 [0.040, 0.059] | 0.944 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_alternative | 6/6/6 | — | 0.944 | 0.001 | 0.992 [0.986, 0.995] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_global_null | 6/6/6 | 0.053 [0.044, 0.064] | 0.944 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_partial_null_high | 6/6/6 | 0.042 [0.034, 0.052] | 0.942 | 0.001 | 0.981 [0.973, 0.986] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_partial_null_low | 6/6/6 | 0.053 [0.044, 0.064] | 0.946 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_alternative | 8/8/8 | — | 0.947 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_global_null | 8/8/8 | 0.054 [0.045, 0.065] | 0.943 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_partial_null_high | 8/8/8 | 0.053 [0.044, 0.064] | 0.946 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | covariance_shift | weekly | covshift_partial_null_low | 8/8/8 | 0.049 [0.040, 0.059] | 0.944 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_alternative | 6/6/6 | — | 0.946 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_partial_null_high | 6/6/6 | 0.060 [0.050, 0.072] | 0.936 | 0.000 | 0.998 [0.994, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_partial_null_low | 6/6/6 | 0.051 [0.041, 0.061] | 0.942 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_alternative | 8/8/8 | — | 0.943 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_partial_null_high | 8/8/8 | 0.050 [0.040, 0.060] | 0.951 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | biweekly | delayed_partial_null_low | 8/8/8 | 0.045 [0.036, 0.055] | 0.935 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_alternative | 6/6/6 | — | 0.946 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_partial_null_high | 6/6/6 | 0.060 [0.050, 0.072] | 0.936 | 0.000 | 0.998 [0.994, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_partial_null_low | 6/6/6 | 0.051 [0.041, 0.061] | 0.942 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_alternative | 8/8/8 | — | 0.943 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_partial_null_high | 8/8/8 | 0.050 [0.040, 0.060] | 0.951 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | delayed | weekly | delayed_partial_null_low | 8/8/8 | 0.045 [0.036, 0.055] | 0.935 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_alternative | 6/6/6 | — | 0.947 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_global_null | 6/6/6 | 0.055 [0.045, 0.066] | 0.940 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_partial_null_high | 6/6/6 | 0.049 [0.040, 0.059] | 0.948 | 0.000 | 0.993 [0.988, 0.996] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_partial_null_low | 6/6/6 | 0.056 [0.046, 0.067] | 0.941 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_alternative | 8/8/8 | — | 0.946 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_global_null | 8/8/8 | 0.042 [0.034, 0.052] | 0.947 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_partial_null_high | 8/8/8 | 0.044 [0.035, 0.054] | 0.948 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | biweekly | heavytail_partial_null_low | 8/8/8 | 0.061 [0.051, 0.073] | 0.939 | 0.000 | 1.000 [0.997, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_alternative | 6/6/6 | — | 0.947 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_global_null | 6/6/6 | 0.055 [0.045, 0.066] | 0.940 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_partial_null_high | 6/6/6 | 0.049 [0.040, 0.059] | 0.948 | 0.000 | 0.993 [0.988, 0.996] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_partial_null_low | 6/6/6 | 0.056 [0.046, 0.067] | 0.941 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_alternative | 8/8/8 | — | 0.946 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_global_null | 8/8/8 | 0.042 [0.034, 0.052] | 0.947 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_partial_null_high | 8/8/8 | 0.044 [0.035, 0.054] | 0.948 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | heavy_tails | weekly | heavytail_partial_null_low | 8/8/8 | 0.061 [0.051, 0.073] | 0.939 | 0.000 | 1.000 [0.997, 1.000] | 0.000 |

## Interpretation (locked diagnostic only)

- **A** (subject) inference tolerances: **6/6** mechanism×size cells.
- A met sourced inference tolerances on all three deferred mechanisms at both sizes.
- **B** (comparator) inference tolerances: **12/12**. A B miss does not restore GLS as a default.
- **C1** (comparator) inference tolerances: **12/12**. A C1 pass does not promote C1; a miss is not a retune.
- Meeting a tolerated FWER gate is not “FWER = 0.05.” This stream does not change production defaults.
- High alternative power reflects the −10%/−15% cells, not the deferred 5% effect.
- No A inference decision flipped versus exploratory B=600.
- B agreeing with A on these three stresses does not restore GLS (close-out was slope heterogeneity).
- C1 delayed n=6 flipped missed → met (FWER 0.067 → 0.060). That is Monte Carlo noise near the 0.065 ceiling, not a promotion and not a retune. The exploratory miss stays on the record.

## Constraints on use

- This locked stream is still a synthetic diagnostic on the deferred-mechanism complete-data matrix.
- It is not design qualification and not an operational recommendation.
- Do not change the ANCOVA default from this stream.
- Do not restore GLS as a production default.
- Do not retune C1.
- Do not claim prior-outcome MAR validity.
- Still deferred: 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2, MCAR on these stresses.

# C1 locked MCAR B=2000 diagnostic

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**
- **Locked MCAR B=2000. Independent of exploratory 20270101. Not prior-outcome MAR. Not a default-switch.**

- Seed `20270201`; 2000 replicates; 8 scenarios × 2 sizes × 5 analyses.
- Fits recorded: **160000**. Study datasets: **32000**.
- Wall clock: **105.436 s**.
- Missingness: `p_missed_visit=0.02`, `p_dropout=0.02` as in `paths.simulate_batch`.
- Observed endpoint fraction: **0.9605**. Observed visit fraction: **0.9678**.
- Absence is independent of Z and of zb/sex/group. This is **not** prior-outcome MAR.
- Independent of exploratory seed `20270101`. C1 was not retuned.
- Production defaults remain `[ancova, gls]`.
- Failed analyses count as non-rejections and remain in the FWER/power denominator.

## Sourced tolerances (not exact 0.05 control)

From `configs/new_study.example.yaml` and `optimize.py::feasibility`:

- FWER point ceiling 0.065 (0.05+0.015); CI ceiling 0.095 (0.05+0.045).
- Coverage floor 0.92; bias fraction 0.15; failure rate 0.02.
- Power target 0.8 with CI floor 0.74 applies to a chosen design, not every small n.
- Passing a tolerated FWER gate is not “FWER = 0.05.” C1 weekly sat on the 0.065 ceiling in exploratory MCAR; this stream checks whether that pass holds.

## Candidate review (locked MCAR, not qualification)

| Analysis | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference tolerances | Power if chosen |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `A_endpoint_ancova` | endpoint_only | 6/6/6 | 0.052 [0.063] | 0.944 | 0.002 | 0.000 | 0.998 [0.995] | yes | yes |
| `B_rm_gls_quadratic` | weekly | 6/6/6 | 0.115 [0.130] | 0.886 | 0.111 | 0.000 | 0.982 [0.975] | no | yes |
| `B_rm_gls_quadratic` | biweekly | 6/6/6 | 0.085 [0.098] | 0.914 | 0.002 | 0.000 | 0.999 [0.996] | no | yes |
| `C1_mmrm_visit_contrast` | weekly | 6/6/6 | 0.061 [0.073] | 0.939 | 0.002 | 0.000 | 0.999 [0.996] | yes | yes |
| `C1_mmrm_visit_contrast` | biweekly | 6/6/6 | 0.061 [0.072] | 0.939 | 0.002 | 0.000 | 0.999 [0.996] | yes | yes |
| `A_endpoint_ancova` | endpoint_only | 8/8/8 | 0.060 [0.072] | 0.940 | 0.002 | 0.000 | 1.000 [0.998] | yes | yes |
| `B_rm_gls_quadratic` | weekly | 8/8/8 | 0.141 [0.157] | 0.860 | 0.107 | 0.000 | 0.999 [0.996] | no | yes |
| `B_rm_gls_quadratic` | biweekly | 8/8/8 | 0.096 [0.110] | 0.904 | 0.002 | 0.000 | 1.000 [0.998] | no | yes |
| `C1_mmrm_visit_contrast` | weekly | 8/8/8 | 0.063 [0.075] | 0.937 | 0.002 | 0.000 | 1.000 [0.998] | yes | yes |
| `C1_mmrm_visit_contrast` | biweekly | 8/8/8 | 0.068 [0.079] | 0.932 | 0.002 | 0.000 | 1.000 [0.998] | no | yes |

“Inference tolerances” is FWER + coverage + alternative bias + failures. It is not a primary-analysis decision and not design qualification.

## Cells that miss a sourced inference tolerance

- `B_rm_gls_quadratic` weekly 6/6/6: FWER 0.115 (worst heterogeneity_global_null); coverage 0.886 (worst heterogeneity_global_null)
- `B_rm_gls_quadratic` biweekly 6/6/6: FWER 0.085 (worst heterogeneity_partial_null_low); coverage 0.914 (worst heterogeneity_partial_null_low)
- `B_rm_gls_quadratic` weekly 8/8/8: FWER 0.141 (worst heterogeneity_partial_null_low); coverage 0.860 (worst heterogeneity_partial_null_low)
- `B_rm_gls_quadratic` biweekly 8/8/8: FWER 0.096 (worst heterogeneity_partial_null_low); coverage 0.904 (worst heterogeneity_partial_null_low)
- `C1_mmrm_visit_contrast` biweekly 8/8/8: FWER 0.068 (worst heterogeneity_partial_null_low)

## Agreement with exploratory MCAR B=600 inference decisions

Same pre-specified analyses, matrix, and missingness; independent seed. A changed decision would be reported, not patched.

| Analysis | Schedule | n/sex | Exploratory MCAR | Locked MCAR | Same decision | FWER expl → locked |
| --- | --- | --- | --- | --- | --- | --- |
| `A_endpoint_ancova` | endpoint_only | 6/6/6 | met | met | yes | 0.052 → 0.052 |
| `B_rm_gls_quadratic` | weekly | 6/6/6 | missed | missed | yes | 0.115 → 0.115 |
| `B_rm_gls_quadratic` | biweekly | 6/6/6 | missed | missed | yes | 0.077 → 0.085 |
| `C1_mmrm_visit_contrast` | weekly | 6/6/6 | met | met | yes | 0.065 → 0.061 |
| `C1_mmrm_visit_contrast` | biweekly | 6/6/6 | met | met | yes | 0.060 → 0.061 |
| `A_endpoint_ancova` | endpoint_only | 8/8/8 | met | met | yes | 0.057 → 0.060 |
| `B_rm_gls_quadratic` | weekly | 8/8/8 | missed | missed | yes | 0.132 → 0.141 |
| `B_rm_gls_quadratic` | biweekly | 8/8/8 | missed | missed | yes | 0.105 → 0.096 |
| `C1_mmrm_visit_contrast` | weekly | 8/8/8 | met | met | yes | 0.065 → 0.063 |
| `C1_mmrm_visit_contrast` | biweekly | 8/8/8 | met | missed | NO | 0.062 → 0.068 |

## A versus C1 under complete data vs MCAR (audit)

- Complete-data pairs on these draws: **32000**. Max |est A − C1|: **1.074e-14**.
- MCAR pairs: **32000**. Max |est A − C1 weekly|: **2.418e-02**.
- Max |est C1 weekly − C1 biweekly| under MCAR: **1.690e-02**.
- Holm disagreements A vs C1 weekly under MCAR: **650**.
- Holm disagreements C1 weekly vs biweekly under MCAR: **170**.

## Per-cell operating characteristics

Full rows are in `cell_oc.csv`.

| Analysis | Schedule | Scenario | n/sex | FWER | Coverage min | Bias max | Power-all | Fail |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A_endpoint_ancova` | endpoint_only | alternative_transient | 6/6/6 | — | 0.946 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | global_null | 6/6/6 | 0.050 [0.040, 0.060] | 0.945 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_global_null | 6/6/6 | 0.046 [0.038, 0.057] | 0.950 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_partial_null_low | 6/6/6 | 0.046 [0.038, 0.057] | 0.944 | 0.001 | 0.997 [0.993, 0.999] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high | 6/6/6 | 0.050 [0.041, 0.060] | 0.949 | 0.001 | 0.996 [0.991, 0.998] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high_transient | 6/6/6 | 0.052 [0.043, 0.063] | 0.948 | 0.000 | 0.994 [0.989, 0.997] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low | 6/6/6 | 0.050 [0.040, 0.060] | 0.951 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low_transient | 6/6/6 | 0.052 [0.043, 0.063] | 0.948 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | alternative_transient | 8/8/8 | — | 0.949 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | global_null | 8/8/8 | 0.036 [0.028, 0.045] | 0.949 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_global_null | 8/8/8 | 0.039 [0.031, 0.048] | 0.948 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_partial_null_low | 8/8/8 | 0.060 [0.050, 0.072] | 0.940 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high | 8/8/8 | 0.050 [0.040, 0.060] | 0.945 | 0.001 | 1.000 [0.997, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high_transient | 8/8/8 | 0.058 [0.048, 0.069] | 0.942 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low | 8/8/8 | 0.048 [0.039, 0.058] | 0.953 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low_transient | 8/8/8 | 0.045 [0.037, 0.056] | 0.954 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | alternative_transient | 6/6/6 | — | 0.951 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | global_null | 6/6/6 | 0.040 [0.031, 0.049] | 0.949 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_global_null | 6/6/6 | 0.078 [0.067, 0.091] | 0.920 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_partial_null_low | 6/6/6 | 0.085 [0.073, 0.098] | 0.914 | 0.001 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high | 6/6/6 | 0.046 [0.038, 0.057] | 0.954 | 0.001 | 0.998 [0.994, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high_transient | 6/6/6 | 0.050 [0.041, 0.060] | 0.950 | 0.000 | 0.996 [0.991, 0.998] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low | 6/6/6 | 0.042 [0.034, 0.052] | 0.958 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low_transient | 6/6/6 | 0.043 [0.035, 0.053] | 0.953 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | alternative_transient | 8/8/8 | — | 0.952 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | global_null | 8/8/8 | 0.035 [0.027, 0.043] | 0.957 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_global_null | 8/8/8 | 0.084 [0.072, 0.096] | 0.914 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_partial_null_low | 8/8/8 | 0.096 [0.083, 0.110] | 0.904 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high | 8/8/8 | 0.045 [0.036, 0.055] | 0.951 | 0.001 | 1.000 [0.997, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high_transient | 8/8/8 | 0.052 [0.043, 0.063] | 0.948 | 0.001 | 1.000 [0.997, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low | 8/8/8 | 0.043 [0.035, 0.053] | 0.957 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low_transient | 8/8/8 | 0.042 [0.033, 0.051] | 0.959 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | alternative_transient | 6/6/6 | — | 0.932 | 0.017 | 0.982 [0.975, 0.987] | 0.000 |
| `B_rm_gls_quadratic` | weekly | global_null | 6/6/6 | 0.038 [0.030, 0.047] | 0.953 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_global_null | 6/6/6 | 0.115 [0.101, 0.130] | 0.886 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_partial_null_low | 6/6/6 | 0.114 [0.100, 0.129] | 0.886 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high | 6/6/6 | 0.043 [0.035, 0.053] | 0.957 | 0.001 | 0.998 [0.994, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high_transient | 6/6/6 | 0.033 [0.025, 0.041] | 0.940 | 0.011 | 0.976 [0.968, 0.982] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low | 6/6/6 | 0.041 [0.033, 0.051] | 0.959 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low_transient | 6/6/6 | 0.015 [0.010, 0.021] | 0.934 | 0.017 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | alternative_transient | 8/8/8 | — | 0.912 | 0.016 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | global_null | 8/8/8 | 0.030 [0.023, 0.039] | 0.960 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_global_null | 8/8/8 | 0.129 [0.115, 0.144] | 0.880 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_partial_null_low | 8/8/8 | 0.141 [0.126, 0.157] | 0.860 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high | 8/8/8 | 0.042 [0.034, 0.052] | 0.947 | 0.001 | 1.000 [0.997, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high_transient | 8/8/8 | 0.032 [0.024, 0.040] | 0.922 | 0.012 | 0.997 [0.993, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low | 8/8/8 | 0.044 [0.035, 0.054] | 0.956 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low_transient | 8/8/8 | 0.015 [0.010, 0.021] | 0.909 | 0.017 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | alternative_transient | 6/6/6 | — | 0.941 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | global_null | 6/6/6 | 0.056 [0.046, 0.067] | 0.941 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_global_null | 6/6/6 | 0.057 [0.047, 0.068] | 0.941 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_partial_null_low | 6/6/6 | 0.051 [0.042, 0.062] | 0.941 | 0.001 | 0.998 [0.994, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high | 6/6/6 | 0.052 [0.043, 0.063] | 0.947 | 0.001 | 0.996 [0.992, 0.998] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high_transient | 6/6/6 | 0.061 [0.051, 0.072] | 0.939 | 0.000 | 0.993 [0.988, 0.996] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low | 6/6/6 | 0.055 [0.045, 0.066] | 0.945 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low_transient | 6/6/6 | 0.056 [0.046, 0.067] | 0.942 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | alternative_transient | 8/8/8 | — | 0.944 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | global_null | 8/8/8 | 0.045 [0.037, 0.056] | 0.948 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_global_null | 8/8/8 | 0.051 [0.042, 0.062] | 0.945 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_partial_null_low | 8/8/8 | 0.068 [0.057, 0.079] | 0.932 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high | 8/8/8 | 0.050 [0.040, 0.060] | 0.940 | 0.001 | 1.000 [0.997, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high_transient | 8/8/8 | 0.059 [0.049, 0.070] | 0.941 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low | 8/8/8 | 0.049 [0.040, 0.059] | 0.952 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low_transient | 8/8/8 | 0.046 [0.037, 0.056] | 0.949 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | alternative_transient | 6/6/6 | — | 0.943 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | global_null | 6/6/6 | 0.054 [0.045, 0.065] | 0.942 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_global_null | 6/6/6 | 0.058 [0.048, 0.069] | 0.944 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_partial_null_low | 6/6/6 | 0.050 [0.041, 0.060] | 0.942 | 0.001 | 0.998 [0.994, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high | 6/6/6 | 0.052 [0.043, 0.063] | 0.944 | 0.001 | 0.995 [0.991, 0.998] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high_transient | 6/6/6 | 0.061 [0.051, 0.073] | 0.939 | 0.000 | 0.994 [0.990, 0.997] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low | 6/6/6 | 0.055 [0.045, 0.066] | 0.945 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low_transient | 6/6/6 | 0.057 [0.047, 0.068] | 0.943 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | alternative_transient | 8/8/8 | — | 0.943 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | global_null | 8/8/8 | 0.043 [0.035, 0.053] | 0.950 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_global_null | 8/8/8 | 0.051 [0.042, 0.062] | 0.945 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_partial_null_low | 8/8/8 | 0.063 [0.053, 0.075] | 0.937 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high | 8/8/8 | 0.051 [0.042, 0.062] | 0.942 | 0.001 | 1.000 [0.997, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high_transient | 8/8/8 | 0.059 [0.050, 0.071] | 0.941 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low | 8/8/8 | 0.050 [0.041, 0.060] | 0.950 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low_transient | 8/8/8 | 0.046 [0.037, 0.056] | 0.949 | 0.001 | 1.000 [0.998, 1.000] | 0.000 |

## Interpretation (locked MCAR only)

- **9 / 10** inference decisions match exploratory MCAR. One flipped and was not patched: C1 biweekly `8/8/8` FWER 0.0675 exceeds the 0.065 point ceiling (CI hi 0.079 still below 0.095). Coverage on that candidate remains 0.933.
- C1 weekly moved **off** the exploratory 0.065 ceiling (0.0615 / 0.063). That does not cancel the biweekly n=8 miss.
- A still meets both sizes. B still misses FWER and coverage under heterogeneity. Weekly B transient bias remains about 0.016.
- Under MCAR, A and C1 remain different estimators (max |Δest| 0.024; 650 Holm disagreements). Intermediate visits matter; A is still the analysis that meets every sourced inference tolerance on this matrix.
- C1 is not retuned. This flip is evidence against promoting C1, not a reason to change df or the sandwich.

## Constraints on use

- Not prior-outcome MAR.
- Not design qualification and not an operational recommendation.
- Do not switch production defaults.
- Do not retune C1 because a FWER point sat on or near 0.065.
- Deferred: covariance shift, heavy tails, delayed alternatives, 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2.

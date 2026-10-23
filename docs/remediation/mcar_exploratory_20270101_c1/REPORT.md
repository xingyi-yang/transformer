# C1 exploratory MCAR / design-dependent missingness

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**
- **Exploratory MCAR / design-dependent missingness. Not prior-outcome MAR. Not a default-switch.**

- Seed `20270101`; 600 replicates; 8 scenarios × 2 sizes × 5 analyses.
- Fits recorded: **48000**. Study datasets: **9600**.
- Wall clock: **31.639 s**.
- Missingness: `p_missed_visit=0.02`, `p_dropout=0.02` from `configs/new_study.example.yaml`, implemented as in `paths.simulate_batch`.
- Observed endpoint fraction: **0.9599**. Observed visit fraction: **0.9675**.
- Absence is independent of Z and of zb/sex/group. This is **not** prior-outcome MAR.
- C1 was not retuned. Production defaults remain `[ancova, gls]`.
- Failed analyses count as non-rejections and remain in the FWER/power denominator.

## Sourced tolerances (not exact 0.05 control)

From `configs/new_study.example.yaml` and `optimize.py::feasibility`:

- FWER point ceiling 0.065 (0.05+0.015); CI ceiling 0.095 (0.05+0.045).
- Coverage floor 0.92; bias fraction 0.15; failure rate 0.02.
- Power target 0.8 with CI floor 0.74 applies to a chosen design, not every small n.
- Passing a tolerated FWER gate is not “FWER = 0.05.”

## Candidate review (exploratory MCAR, not locked)

| Analysis | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference tolerances | Power if chosen |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `A_endpoint_ancova` | endpoint_only | 6/6/6 | 0.052 [0.073] | 0.935 | 0.003 | 0.000 | 0.998 [0.991] | yes | yes |
| `B_rm_gls_quadratic` | weekly | 6/6/6 | 0.115 [0.143] | 0.872 | 0.109 | 0.000 | 0.987 [0.974] | no | yes |
| `B_rm_gls_quadratic` | biweekly | 6/6/6 | 0.077 [0.101] | 0.910 | 0.003 | 0.000 | 0.998 [0.991] | no | yes |
| `C1_mmrm_visit_contrast` | weekly | 6/6/6 | 0.065 [0.088] | 0.928 | 0.003 | 0.000 | 0.998 [0.991] | yes | yes |
| `C1_mmrm_visit_contrast` | biweekly | 6/6/6 | 0.060 [0.082] | 0.925 | 0.003 | 0.000 | 0.998 [0.991] | yes | yes |
| `A_endpoint_ancova` | endpoint_only | 8/8/8 | 0.057 [0.078] | 0.937 | 0.003 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | weekly | 8/8/8 | 0.132 [0.161] | 0.873 | 0.105 | 0.000 | 0.998 [0.991] | no | yes |
| `B_rm_gls_quadratic` | biweekly | 8/8/8 | 0.105 [0.132] | 0.912 | 0.003 | 0.000 | 1.000 [0.994] | no | yes |
| `C1_mmrm_visit_contrast` | weekly | 8/8/8 | 0.065 [0.088] | 0.933 | 0.003 | 0.000 | 1.000 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | biweekly | 8/8/8 | 0.062 [0.084] | 0.932 | 0.003 | 0.000 | 1.000 [0.994] | yes | yes |

“Inference tolerances” is FWER + coverage + alternative bias + failures. It is not a primary-analysis decision and not design qualification.

## Cells that miss a sourced inference tolerance

- `B_rm_gls_quadratic` weekly 6/6/6: FWER 0.115 (worst heterogeneity_global_null); coverage 0.872 (worst heterogeneity_global_null)
- `B_rm_gls_quadratic` biweekly 6/6/6: FWER 0.077 (worst heterogeneity_global_null); coverage 0.910 (worst heterogeneity_partial_null_low)
- `B_rm_gls_quadratic` weekly 8/8/8: FWER 0.132 (worst heterogeneity_global_null); coverage 0.873 (worst heterogeneity_partial_null_low)
- `B_rm_gls_quadratic` biweekly 8/8/8: FWER 0.105 (worst heterogeneity_global_null); coverage 0.912 (worst heterogeneity_global_null)

## Qualitative comparison to locked complete-data decisions

Different seed. Use only to see whether a pass/fail flipped under MCAR.

| Analysis | Schedule | n/sex | Complete locked | MCAR exploratory | Same decision |
| --- | --- | --- | --- | --- | --- |
| `A_endpoint_ancova` | endpoint_only | 6/6/6 | met | met | yes |
| `B_rm_gls_quadratic` | weekly | 6/6/6 | missed | missed | yes |
| `B_rm_gls_quadratic` | biweekly | 6/6/6 | missed | missed | yes |
| `C1_mmrm_visit_contrast` | weekly | 6/6/6 | met | met | yes |
| `C1_mmrm_visit_contrast` | biweekly | 6/6/6 | met | met | yes |
| `A_endpoint_ancova` | endpoint_only | 8/8/8 | met | met | yes |
| `B_rm_gls_quadratic` | weekly | 8/8/8 | missed | missed | yes |
| `B_rm_gls_quadratic` | biweekly | 8/8/8 | missed | missed | yes |
| `C1_mmrm_visit_contrast` | weekly | 8/8/8 | met | met | yes |
| `C1_mmrm_visit_contrast` | biweekly | 8/8/8 | met | met | yes |

## A versus C1 under complete data vs MCAR (audit)

- Complete-data pairs on these draws: **9600**. Max |est A − C1|: **8.230e-15**.
- MCAR pairs: **9600**. Max |est A − C1 weekly|: **2.427e-02**.
- Max |est C1 weekly − C1 biweekly| under MCAR: **1.590e-02**.
- Max |SE C1 weekly − C1 biweekly| under MCAR: **3.576e-03**.
- Holm disagreements A vs C1 weekly under MCAR: **190**.
- Holm disagreements C1 weekly vs biweekly under MCAR: **78**.

On complete data the endpoint coefficients match. Under MCAR they need not. That difference is not by itself a reason to switch defaults.

## Per-cell operating characteristics

Full rows are in `cell_oc.csv`.

| Analysis | Schedule | Scenario | n/sex | FWER | Coverage min | Bias max | Power-all | Fail |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A_endpoint_ancova` | endpoint_only | alternative_transient | 6/6/6 | — | 0.947 | 0.001 | 0.998 [0.991, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | global_null | 6/6/6 | 0.045 [0.030, 0.065] | 0.950 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_global_null | 6/6/6 | 0.037 [0.023, 0.055] | 0.957 | 0.004 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_partial_null_low | 6/6/6 | 0.045 [0.030, 0.065] | 0.953 | 0.002 | 0.998 [0.991, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high | 6/6/6 | 0.042 [0.027, 0.061] | 0.958 | 0.002 | 0.997 [0.988, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high_transient | 6/6/6 | 0.048 [0.033, 0.069] | 0.948 | 0.000 | 0.992 [0.981, 0.997] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low | 6/6/6 | 0.042 [0.027, 0.061] | 0.935 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low_transient | 6/6/6 | 0.052 [0.035, 0.073] | 0.948 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | alternative_transient | 8/8/8 | — | 0.950 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | global_null | 8/8/8 | 0.038 [0.024, 0.057] | 0.937 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_global_null | 8/8/8 | 0.057 [0.040, 0.078] | 0.940 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_partial_null_low | 8/8/8 | 0.053 [0.037, 0.074] | 0.947 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high | 8/8/8 | 0.038 [0.024, 0.057] | 0.942 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high_transient | 8/8/8 | 0.040 [0.026, 0.059] | 0.960 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low | 8/8/8 | 0.042 [0.027, 0.061] | 0.958 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low_transient | 8/8/8 | 0.037 [0.023, 0.055] | 0.955 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | alternative_transient | 6/6/6 | — | 0.952 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | global_null | 6/6/6 | 0.045 [0.030, 0.065] | 0.952 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_global_null | 6/6/6 | 0.077 [0.057, 0.101] | 0.925 | 0.004 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_partial_null_low | 6/6/6 | 0.073 [0.054, 0.097] | 0.910 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high | 6/6/6 | 0.043 [0.028, 0.063] | 0.955 | 0.002 | 0.997 [0.988, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high_transient | 6/6/6 | 0.045 [0.030, 0.065] | 0.953 | 0.000 | 0.993 [0.983, 0.998] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low | 6/6/6 | 0.040 [0.026, 0.059] | 0.942 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low_transient | 6/6/6 | 0.040 [0.026, 0.059] | 0.955 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | alternative_transient | 8/8/8 | — | 0.948 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | global_null | 8/8/8 | 0.028 [0.017, 0.045] | 0.943 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_global_null | 8/8/8 | 0.105 [0.082, 0.132] | 0.912 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_partial_null_low | 8/8/8 | 0.088 [0.067, 0.114] | 0.912 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high | 8/8/8 | 0.033 [0.020, 0.051] | 0.945 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high_transient | 8/8/8 | 0.043 [0.028, 0.063] | 0.957 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low | 8/8/8 | 0.030 [0.018, 0.047] | 0.970 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low_transient | 8/8/8 | 0.038 [0.024, 0.057] | 0.958 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | alternative_transient | 6/6/6 | — | 0.925 | 0.016 | 0.987 [0.974, 0.994] | 0.000 |
| `B_rm_gls_quadratic` | weekly | global_null | 6/6/6 | 0.042 [0.027, 0.061] | 0.953 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_global_null | 6/6/6 | 0.115 [0.091, 0.143] | 0.872 | 0.004 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_partial_null_low | 6/6/6 | 0.115 [0.091, 0.143] | 0.880 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high | 6/6/6 | 0.038 [0.024, 0.057] | 0.962 | 0.002 | 0.997 [0.988, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high_transient | 6/6/6 | 0.028 [0.017, 0.045] | 0.935 | 0.011 | 0.967 [0.949, 0.980] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low | 6/6/6 | 0.032 [0.019, 0.049] | 0.957 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low_transient | 6/6/6 | 0.013 [0.006, 0.026] | 0.940 | 0.016 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | alternative_transient | 8/8/8 | — | 0.912 | 0.016 | 0.998 [0.991, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | global_null | 8/8/8 | 0.032 [0.019, 0.049] | 0.955 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_global_null | 8/8/8 | 0.132 [0.106, 0.161] | 0.875 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_partial_null_low | 8/8/8 | 0.127 [0.101, 0.156] | 0.873 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high | 8/8/8 | 0.032 [0.019, 0.049] | 0.952 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high_transient | 8/8/8 | 0.030 [0.018, 0.047] | 0.942 | 0.011 | 0.995 [0.985, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low | 8/8/8 | 0.033 [0.020, 0.051] | 0.967 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low_transient | 8/8/8 | 0.018 [0.009, 0.033] | 0.898 | 0.018 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | alternative_transient | 6/6/6 | — | 0.942 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | global_null | 6/6/6 | 0.057 [0.040, 0.078] | 0.925 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_global_null | 6/6/6 | 0.047 [0.031, 0.067] | 0.945 | 0.004 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_partial_null_low | 6/6/6 | 0.050 [0.034, 0.071] | 0.945 | 0.002 | 0.995 [0.985, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high | 6/6/6 | 0.060 [0.042, 0.082] | 0.940 | 0.002 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high_transient | 6/6/6 | 0.052 [0.035, 0.073] | 0.937 | 0.000 | 0.995 [0.985, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low | 6/6/6 | 0.048 [0.033, 0.069] | 0.938 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low_transient | 6/6/6 | 0.060 [0.042, 0.082] | 0.940 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | alternative_transient | 8/8/8 | — | 0.942 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | global_null | 8/8/8 | 0.042 [0.027, 0.061] | 0.932 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_global_null | 8/8/8 | 0.062 [0.044, 0.084] | 0.940 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_partial_null_low | 8/8/8 | 0.062 [0.044, 0.084] | 0.938 | 0.002 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high | 8/8/8 | 0.047 [0.031, 0.067] | 0.940 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high_transient | 8/8/8 | 0.043 [0.028, 0.063] | 0.957 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low | 8/8/8 | 0.045 [0.030, 0.065] | 0.955 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low_transient | 8/8/8 | 0.045 [0.030, 0.065] | 0.947 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | alternative_transient | 6/6/6 | — | 0.940 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | global_null | 6/6/6 | 0.057 [0.040, 0.078] | 0.928 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_global_null | 6/6/6 | 0.048 [0.033, 0.069] | 0.942 | 0.004 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_partial_null_low | 6/6/6 | 0.047 [0.031, 0.067] | 0.948 | 0.002 | 0.997 [0.988, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high | 6/6/6 | 0.065 [0.047, 0.088] | 0.935 | 0.002 | 0.997 [0.988, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high_transient | 6/6/6 | 0.057 [0.040, 0.078] | 0.933 | 0.001 | 0.993 [0.983, 0.998] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low | 6/6/6 | 0.048 [0.033, 0.069] | 0.937 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low_transient | 6/6/6 | 0.060 [0.042, 0.082] | 0.940 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | alternative_transient | 8/8/8 | — | 0.938 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | global_null | 8/8/8 | 0.037 [0.023, 0.055] | 0.933 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_global_null | 8/8/8 | 0.063 [0.045, 0.086] | 0.942 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_partial_null_low | 8/8/8 | 0.065 [0.047, 0.088] | 0.935 | 0.002 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high | 8/8/8 | 0.040 [0.026, 0.059] | 0.938 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high_transient | 8/8/8 | 0.045 [0.030, 0.065] | 0.955 | 0.000 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low | 8/8/8 | 0.043 [0.028, 0.063] | 0.957 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low_transient | 8/8/8 | 0.042 [0.027, 0.061] | 0.945 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |

## Interpretation (exploratory MCAR only)

- All 10 qualitative inference decisions match the locked complete-data stream. Light MCAR (about 4% missing at the endpoint) did not flip a pass/fail.
- B still misses FWER and coverage under slope heterogeneity. Weekly transient bias remains about 0.016.
- A and C1 still meet sourced inference tolerances. C1 weekly FWER sits on the 0.065 point ceiling at both sizes (CI hi 0.088). That is a pass of the tolerance, not “FWER = 0.05,” and not a reason to prefer C1 over A.
- Under MCAR, A and C1 are no longer the same estimator (max |Δest| 0.024; 190 Holm disagreements). C1 weekly and C1 biweekly also diverge (78 Holm disagreements). Intermediate visits now matter, but A remains valid, so C1 has not earned a default switch.
- This stream is not prior-outcome MAR and not locked confirmation.

## Constraints on use

- Not prior-outcome MAR. Do not describe these draws as outcome-dependent dropout.
- Not a locked confirmation (that would need a new independent seed and a larger B).
- Do not retune C1 on these draws.
- Do not switch production defaults.
- Deferred: covariance shift, heavy tails, delayed alternatives, 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2.

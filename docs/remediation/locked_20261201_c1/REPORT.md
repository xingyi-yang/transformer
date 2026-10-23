# C1 locked B=2000 diagnostic

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**
- **Locked B=2000 diagnostic. Independent of exploratory draws. Not a default-switch.**

- Seed `20261201`; 2000 replicates; complete data; 8 scenarios × 2 sizes × 5 analyses.
- Fits recorded: **160000**. Study datasets: **32000**.
- Wall clock: **72.545 s**.
- Independent of exploratory seed `20261101`. C1 was not retuned.
- Common random numbers: one weekly latent path per (scenario, size, replicate); biweekly is a visit subset.
- Working covariance: unstressed `HierarchicalCovariance.animal_cov` (historical plug-in stand-in).
- Failed analyses count as non-rejections and remain in the FWER/power denominator.
- C1 remains a **candidate**. Production defaults remain `[ancova, gls]`.
- Replicate-level draws are not archived. This runner plus the seed reproduces them.

## Sourced tolerances (not exact 0.05 control)

From `configs/new_study.example.yaml` and `optimize.py::feasibility`:

- FWER point ceiling 0.065 (0.05+0.015); CI ceiling 0.095 (0.05+0.045).
- Coverage floor 0.92; bias fraction 0.15; failure rate 0.02.
- Power target 0.8 with CI floor 0.74 applies to a chosen design, not every small n.
- Passing a tolerated FWER gate is not “FWER = 0.05.”

## Candidate review (locked diagnostic, not qualification)

| Analysis | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference tolerances | Power if chosen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `A_endpoint_ancova` | endpoint_only | 6/6/6 | 0.055 [0.066] | 0.943 | 0.001 | 0.000 | 0.999 [0.996] | yes | yes |
| `B_rm_gls_quadratic` | weekly | 6/6/6 | 0.129 [0.144] | 0.882 | 0.107 | 0.000 | 0.982 [0.975] | no | yes |
| `B_rm_gls_quadratic` | biweekly | 6/6/6 | 0.087 [0.101] | 0.912 | 0.001 | 0.000 | 0.999 [0.996] | no | yes |
| `C1_mmrm_visit_contrast` | weekly | 6/6/6 | 0.062 [0.074] | 0.938 | 0.001 | 0.000 | 0.998 [0.995] | yes | yes |
| `C1_mmrm_visit_contrast` | biweekly | 6/6/6 | 0.062 [0.074] | 0.938 | 0.001 | 0.000 | 0.998 [0.995] | yes | yes |
| `A_endpoint_ancova` | endpoint_only | 8/8/8 | 0.058 [0.069] | 0.943 | 0.003 | 0.000 | 1.000 [0.998] | yes | yes |
| `B_rm_gls_quadratic` | weekly | 8/8/8 | 0.128 [0.143] | 0.873 | 0.107 | 0.000 | 1.000 [0.998] | no | yes |
| `B_rm_gls_quadratic` | biweekly | 8/8/8 | 0.096 [0.110] | 0.904 | 0.003 | 0.000 | 1.000 [0.998] | no | yes |
| `C1_mmrm_visit_contrast` | weekly | 8/8/8 | 0.061 [0.073] | 0.939 | 0.003 | 0.000 | 1.000 [0.998] | yes | yes |
| `C1_mmrm_visit_contrast` | biweekly | 8/8/8 | 0.061 [0.073] | 0.939 | 0.003 | 0.000 | 1.000 [0.998] | yes | yes |

“Inference tolerances” is FWER + coverage + alternative bias + failures. It is not a primary-analysis decision and not design qualification.

## Cells that miss a sourced inference tolerance

- `B_rm_gls_quadratic` weekly 6/6/6: FWER 0.129 (worst heterogeneity_global_null); coverage 0.882 (worst heterogeneity_global_null)
- `B_rm_gls_quadratic` biweekly 6/6/6: FWER 0.087 (worst heterogeneity_partial_null_low); coverage 0.912 (worst heterogeneity_partial_null_low)
- `B_rm_gls_quadratic` weekly 8/8/8: FWER 0.128 (worst heterogeneity_partial_null_low); coverage 0.873 (worst heterogeneity_partial_null_low)
- `B_rm_gls_quadratic` biweekly 8/8/8: FWER 0.096 (worst heterogeneity_partial_null_low); coverage 0.904 (worst heterogeneity_partial_null_low)

## Agreement with exploratory B=600 inference decisions

Same pre-specified analyses and matrix; independent seed. A changed decision would be reported, not patched.

| Analysis | Schedule | n/sex | Exploratory | Locked | Same decision |
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

## Complete-data identities (audit, not a gate)

- Successful complete-data triples compared: **32000**.
- Max |est A − est C1 weekly|: **9.145e-15**.
- Max |est C1 weekly − C1 biweekly|: **8.979e-15**.
- Max |SE C1 weekly − C1 biweekly|: **1.978e-15**.
- Holm disagreements A vs C1 weekly: **501**.
- Holm disagreements C1 weekly vs biweekly: **0**.

## Per-cell operating characteristics

Full rows are in `cell_oc.csv`.

| Analysis | Schedule | Scenario | n/sex | FWER | Coverage min | Bias max | Power-all | Fail |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A_endpoint_ancova` | endpoint_only | alternative_transient | 6/6/6 | — | 0.946 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | global_null | 6/6/6 | 0.049 [0.040, 0.059] | 0.951 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_global_null | 6/6/6 | 0.043 [0.035, 0.053] | 0.947 | 0.002 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_partial_null_low | 6/6/6 | 0.043 [0.034, 0.052] | 0.954 | 0.000 | 0.996 [0.992, 0.998] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high | 6/6/6 | 0.052 [0.043, 0.063] | 0.948 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high_transient | 6/6/6 | 0.050 [0.040, 0.060] | 0.943 | 0.000 | 0.994 [0.990, 0.997] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low | 6/6/6 | 0.055 [0.045, 0.066] | 0.945 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low_transient | 6/6/6 | 0.041 [0.033, 0.051] | 0.950 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | alternative_transient | 8/8/8 | — | 0.946 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | global_null | 8/8/8 | 0.051 [0.041, 0.061] | 0.944 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_global_null | 8/8/8 | 0.038 [0.031, 0.048] | 0.950 | 0.000 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_partial_null_low | 8/8/8 | 0.058 [0.048, 0.069] | 0.943 | 0.000 | 1.000 [0.997, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high | 8/8/8 | 0.040 [0.032, 0.050] | 0.956 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high_transient | 8/8/8 | 0.044 [0.035, 0.054] | 0.956 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low | 8/8/8 | 0.050 [0.040, 0.060] | 0.948 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low_transient | 8/8/8 | 0.056 [0.046, 0.066] | 0.945 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | alternative_transient | 6/6/6 | — | 0.950 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | global_null | 6/6/6 | 0.037 [0.029, 0.046] | 0.955 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_global_null | 6/6/6 | 0.082 [0.070, 0.095] | 0.916 | 0.002 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_partial_null_low | 6/6/6 | 0.087 [0.075, 0.101] | 0.912 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high | 6/6/6 | 0.046 [0.038, 0.057] | 0.954 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high_transient | 6/6/6 | 0.041 [0.033, 0.051] | 0.947 | 0.000 | 0.997 [0.993, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low | 6/6/6 | 0.054 [0.045, 0.065] | 0.946 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low_transient | 6/6/6 | 0.037 [0.030, 0.047] | 0.956 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | alternative_transient | 8/8/8 | — | 0.952 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | global_null | 8/8/8 | 0.045 [0.037, 0.056] | 0.949 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_global_null | 8/8/8 | 0.084 [0.072, 0.096] | 0.913 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_partial_null_low | 8/8/8 | 0.096 [0.083, 0.110] | 0.904 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high | 8/8/8 | 0.039 [0.031, 0.048] | 0.957 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high_transient | 8/8/8 | 0.043 [0.035, 0.053] | 0.957 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low | 8/8/8 | 0.045 [0.036, 0.055] | 0.949 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low_transient | 8/8/8 | 0.050 [0.041, 0.060] | 0.950 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | alternative_transient | 6/6/6 | — | 0.933 | 0.016 | 0.982 [0.975, 0.987] | 0.000 |
| `B_rm_gls_quadratic` | weekly | global_null | 6/6/6 | 0.034 [0.026, 0.042] | 0.960 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_global_null | 6/6/6 | 0.129 [0.115, 0.144] | 0.882 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_partial_null_low | 6/6/6 | 0.117 [0.103, 0.131] | 0.883 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high | 6/6/6 | 0.036 [0.028, 0.045] | 0.963 | 0.000 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high_transient | 6/6/6 | 0.026 [0.019, 0.034] | 0.941 | 0.011 | 0.977 [0.969, 0.983] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low | 6/6/6 | 0.049 [0.040, 0.059] | 0.952 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low_transient | 6/6/6 | 0.010 [0.006, 0.015] | 0.949 | 0.016 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | alternative_transient | 8/8/8 | — | 0.911 | 0.016 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | global_null | 8/8/8 | 0.043 [0.035, 0.053] | 0.952 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_global_null | 8/8/8 | 0.127 [0.113, 0.142] | 0.880 | 0.000 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_partial_null_low | 8/8/8 | 0.128 [0.113, 0.143] | 0.873 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high | 8/8/8 | 0.037 [0.030, 0.047] | 0.957 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high_transient | 8/8/8 | 0.028 [0.021, 0.036] | 0.944 | 0.011 | 0.999 [0.996, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low | 8/8/8 | 0.048 [0.039, 0.058] | 0.952 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low_transient | 8/8/8 | 0.019 [0.014, 0.027] | 0.914 | 0.016 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | alternative_transient | 6/6/6 | — | 0.941 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | global_null | 6/6/6 | 0.058 [0.048, 0.069] | 0.945 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_global_null | 6/6/6 | 0.048 [0.039, 0.058] | 0.943 | 0.002 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_partial_null_low | 6/6/6 | 0.057 [0.047, 0.068] | 0.944 | 0.000 | 0.997 [0.993, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high | 6/6/6 | 0.056 [0.046, 0.067] | 0.943 | 0.000 | 0.998 [0.994, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high_transient | 6/6/6 | 0.052 [0.043, 0.063] | 0.941 | 0.000 | 0.993 [0.988, 0.996] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low | 6/6/6 | 0.062 [0.052, 0.074] | 0.938 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low_transient | 6/6/6 | 0.049 [0.040, 0.059] | 0.947 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | alternative_transient | 8/8/8 | — | 0.944 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | global_null | 8/8/8 | 0.058 [0.048, 0.069] | 0.939 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_global_null | 8/8/8 | 0.045 [0.036, 0.055] | 0.946 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_partial_null_low | 8/8/8 | 0.061 [0.051, 0.073] | 0.939 | 0.000 | 1.000 [0.997, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high | 8/8/8 | 0.041 [0.032, 0.050] | 0.948 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high_transient | 8/8/8 | 0.045 [0.037, 0.056] | 0.951 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low | 8/8/8 | 0.053 [0.044, 0.064] | 0.942 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low_transient | 8/8/8 | 0.060 [0.050, 0.072] | 0.940 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | alternative_transient | 6/6/6 | — | 0.941 | 0.000 | 0.998 [0.995, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | global_null | 6/6/6 | 0.058 [0.048, 0.069] | 0.945 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_global_null | 6/6/6 | 0.048 [0.039, 0.058] | 0.943 | 0.002 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_partial_null_low | 6/6/6 | 0.057 [0.047, 0.068] | 0.944 | 0.000 | 0.997 [0.993, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high | 6/6/6 | 0.056 [0.046, 0.067] | 0.943 | 0.000 | 0.998 [0.994, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high_transient | 6/6/6 | 0.052 [0.043, 0.063] | 0.941 | 0.000 | 0.993 [0.988, 0.996] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low | 6/6/6 | 0.062 [0.052, 0.074] | 0.938 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low_transient | 6/6/6 | 0.049 [0.040, 0.059] | 0.947 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | alternative_transient | 8/8/8 | — | 0.944 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | global_null | 8/8/8 | 0.058 [0.048, 0.069] | 0.939 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_global_null | 8/8/8 | 0.045 [0.036, 0.055] | 0.946 | 0.000 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_partial_null_low | 8/8/8 | 0.061 [0.051, 0.073] | 0.939 | 0.000 | 1.000 [0.997, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high | 8/8/8 | 0.041 [0.032, 0.050] | 0.948 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high_transient | 8/8/8 | 0.045 [0.037, 0.056] | 0.951 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low | 8/8/8 | 0.053 [0.044, 0.064] | 0.942 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low_transient | 8/8/8 | 0.060 [0.050, 0.072] | 0.940 | 0.000 | 1.000 [0.998, 1.000] | 0.000 |

## Interpretation (locked diagnostic only)

- All 10 pre-specified inference decisions agree with exploratory B=600. C1 was not retuned.
- B again misses FWER and coverage under slope heterogeneity ×1.6. Weekly transient bias remains 0.016 (fraction 0.107), matching the algebraic fixture. Weekly B at n=8 also has alternative coverage 0.912, below 0.92.
- A and C1 again meet sourced inference tolerances on this complete-data matrix. That is not a primary-analysis decision.
- Complete-data identities hold: A and C1 coefficients match; C1 weekly equals C1 biweekly for the endpoint contrast; sandwich SEs still differ from ANCOVA OLS (501 Holm disagreements).
- Meeting a tolerated FWER gate is not “FWER = 0.05.” This stream does not switch defaults and is not design qualification.

## Constraints on use

- This locked stream is still a synthetic diagnostic on the small complete-data matrix.
- It is not design qualification and not an operational recommendation.
- Do not switch production defaults.
- Do not claim prior-outcome MAR validity.
- Deferred cells remain deferred: covariance shift, heavy tails, delayed alternatives, 5% effect, control-heavy, twice-weekly, MAR, C2.

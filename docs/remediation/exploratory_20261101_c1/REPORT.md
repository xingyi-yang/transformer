# C1 exploratory B=600 stream

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**
- **Exploratory B=600. Not a locked evaluation.**

- Seed `20261101`; 600 replicates; complete data; 8 scenarios × 2 sizes × 5 analyses.
- Fits recorded: **48000**. Study datasets: **9600**.
- Wall clock: **22.292 s**.
- Common random numbers: one weekly latent path per (scenario, size, replicate); biweekly is a visit subset.
- Working covariance: unstressed `HierarchicalCovariance.animal_cov` (historical plug-in stand-in).
- Failed analyses count as non-rejections and remain in the FWER/power denominator.
- C1 remains a **candidate**. Production defaults remain `[ancova, gls]`.

## Sourced tolerances (not exact 0.05 control)

From `configs/new_study.example.yaml` and `optimize.py::feasibility`:

- FWER point ceiling 0.065 (0.05+0.015); CI ceiling 0.095 (0.05+0.045).
- Coverage floor 0.92; bias fraction 0.15; failure rate 0.02.
- Power target 0.8 with CI floor 0.74 applies to a chosen design, not every small n.
- Passing a tolerated FWER gate is not “FWER = 0.05.”

## Candidate review (exploratory, not locked)

| Analysis | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference tolerances | Power if chosen |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `A_endpoint_ancova` | endpoint_only | 6/6/6 | 0.057 [0.078] | 0.938 | 0.005 | 0.000 | 0.997 [0.988] | yes | yes |
| `B_rm_gls_quadratic` | weekly | 6/6/6 | 0.120 [0.149] | 0.875 | 0.101 | 0.000 | 0.985 [0.972] | no | yes |
| `B_rm_gls_quadratic` | biweekly | 6/6/6 | 0.090 [0.116] | 0.910 | 0.005 | 0.000 | 1.000 [0.994] | no | yes |
| `C1_mmrm_visit_contrast` | weekly | 6/6/6 | 0.062 [0.084] | 0.935 | 0.005 | 0.000 | 0.997 [0.988] | yes | yes |
| `C1_mmrm_visit_contrast` | biweekly | 6/6/6 | 0.062 [0.084] | 0.935 | 0.005 | 0.000 | 0.997 [0.988] | yes | yes |
| `A_endpoint_ancova` | endpoint_only | 8/8/8 | 0.057 [0.078] | 0.938 | 0.004 | 0.000 | 1.000 [0.994] | yes | yes |
| `B_rm_gls_quadratic` | weekly | 8/8/8 | 0.137 [0.167] | 0.860 | 0.108 | 0.000 | 1.000 [0.994] | no | yes |
| `B_rm_gls_quadratic` | biweekly | 8/8/8 | 0.092 [0.118] | 0.898 | 0.004 | 0.000 | 1.000 [0.994] | no | yes |
| `C1_mmrm_visit_contrast` | weekly | 8/8/8 | 0.063 [0.086] | 0.932 | 0.004 | 0.000 | 1.000 [0.994] | yes | yes |
| `C1_mmrm_visit_contrast` | biweekly | 8/8/8 | 0.063 [0.086] | 0.932 | 0.004 | 0.000 | 1.000 [0.994] | yes | yes |

“Inference tolerances” is FWER + coverage + alternative bias + failures. It is not a primary-analysis decision and not design qualification.

## Cells that miss a sourced inference tolerance

- `B_rm_gls_quadratic` weekly 6/6/6: FWER 0.120 (worst heterogeneity_global_null); coverage 0.875 (worst heterogeneity_global_null)
- `B_rm_gls_quadratic` biweekly 6/6/6: FWER 0.090 (worst heterogeneity_partial_null_low); coverage 0.910 (worst heterogeneity_global_null)
- `B_rm_gls_quadratic` weekly 8/8/8: FWER 0.137 (worst heterogeneity_global_null); coverage 0.860 (worst heterogeneity_global_null)
- `B_rm_gls_quadratic` biweekly 8/8/8: FWER 0.092 (worst heterogeneity_global_null); coverage 0.898 (worst heterogeneity_global_null)

## Per-cell operating characteristics

Full rows are in `cell_oc.csv`. Replicate-level fits are in `replicate_results.csv`.

| Analysis | Schedule | Scenario | n/sex | FWER | Coverage min | Bias max | Power-all | Fail |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A_endpoint_ancova` | endpoint_only | alternative_transient | 6/6/6 | — | 0.942 | 0.001 | 0.997 [0.988, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | global_null | 6/6/6 | 0.057 [0.040, 0.078] | 0.938 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_global_null | 6/6/6 | 0.047 [0.031, 0.067] | 0.942 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_partial_null_low | 6/6/6 | 0.055 [0.038, 0.076] | 0.943 | 0.003 | 0.998 [0.991, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high | 6/6/6 | 0.048 [0.033, 0.069] | 0.950 | 0.001 | 0.993 [0.983, 0.998] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high_transient | 6/6/6 | 0.037 [0.023, 0.055] | 0.960 | 0.001 | 0.997 [0.988, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low | 6/6/6 | 0.053 [0.037, 0.074] | 0.947 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low_transient | 6/6/6 | 0.055 [0.038, 0.076] | 0.945 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | alternative_transient | 8/8/8 | — | 0.958 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | global_null | 8/8/8 | 0.045 [0.030, 0.065] | 0.943 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_global_null | 8/8/8 | 0.047 [0.031, 0.067] | 0.938 | 0.001 | — | 0.000 |
| `A_endpoint_ancova` | endpoint_only | heterogeneity_partial_null_low | 8/8/8 | 0.050 [0.034, 0.071] | 0.950 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high | 8/8/8 | 0.048 [0.033, 0.069] | 0.952 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_high_transient | 8/8/8 | 0.040 [0.026, 0.059] | 0.952 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low | 8/8/8 | 0.053 [0.037, 0.074] | 0.947 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `A_endpoint_ancova` | endpoint_only | partial_null_low_transient | 8/8/8 | 0.057 [0.040, 0.078] | 0.943 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | alternative_transient | 6/6/6 | — | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | global_null | 6/6/6 | 0.043 [0.028, 0.063] | 0.935 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_global_null | 6/6/6 | 0.085 [0.064, 0.110] | 0.910 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_partial_null_low | 6/6/6 | 0.090 [0.068, 0.116] | 0.910 | 0.003 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high | 6/6/6 | 0.048 [0.033, 0.069] | 0.950 | 0.001 | 0.995 [0.985, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high_transient | 6/6/6 | 0.035 [0.022, 0.053] | 0.958 | 0.001 | 0.997 [0.988, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low | 6/6/6 | 0.048 [0.033, 0.069] | 0.952 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low_transient | 6/6/6 | 0.050 [0.034, 0.071] | 0.950 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | alternative_transient | 8/8/8 | — | 0.962 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | global_null | 8/8/8 | 0.035 [0.022, 0.053] | 0.953 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_global_null | 8/8/8 | 0.092 [0.070, 0.118] | 0.898 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | biweekly | heterogeneity_partial_null_low | 8/8/8 | 0.083 [0.062, 0.108] | 0.917 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high | 8/8/8 | 0.048 [0.033, 0.069] | 0.952 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_high_transient | 8/8/8 | 0.032 [0.019, 0.049] | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low | 8/8/8 | 0.048 [0.033, 0.069] | 0.952 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | biweekly | partial_null_low_transient | 8/8/8 | 0.045 [0.030, 0.065] | 0.955 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | alternative_transient | 6/6/6 | — | 0.933 | 0.015 | 0.985 [0.972, 0.993] | 0.000 |
| `B_rm_gls_quadratic` | weekly | global_null | 6/6/6 | 0.040 [0.026, 0.059] | 0.942 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_global_null | 6/6/6 | 0.120 [0.095, 0.149] | 0.875 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_partial_null_low | 6/6/6 | 0.120 [0.095, 0.149] | 0.875 | 0.003 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high | 6/6/6 | 0.050 [0.034, 0.071] | 0.947 | 0.000 | 0.995 [0.985, 0.999] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high_transient | 6/6/6 | 0.022 [0.012, 0.037] | 0.947 | 0.011 | 0.975 [0.959, 0.986] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low | 6/6/6 | 0.048 [0.033, 0.069] | 0.952 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low_transient | 6/6/6 | 0.018 [0.009, 0.033] | 0.930 | 0.016 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | alternative_transient | 8/8/8 | — | 0.920 | 0.016 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | global_null | 8/8/8 | 0.042 [0.027, 0.061] | 0.957 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_global_null | 8/8/8 | 0.137 [0.110, 0.167] | 0.860 | 0.001 | — | 0.000 |
| `B_rm_gls_quadratic` | weekly | heterogeneity_partial_null_low | 8/8/8 | 0.115 [0.091, 0.143] | 0.885 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high | 8/8/8 | 0.043 [0.028, 0.063] | 0.957 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_high_transient | 8/8/8 | 0.018 [0.009, 0.033] | 0.940 | 0.011 | 0.997 [0.988, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low | 8/8/8 | 0.043 [0.028, 0.063] | 0.957 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `B_rm_gls_quadratic` | weekly | partial_null_low_transient | 8/8/8 | 0.015 [0.007, 0.028] | 0.917 | 0.016 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | alternative_transient | 6/6/6 | — | 0.938 | 0.001 | 0.997 [0.988, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | global_null | 6/6/6 | 0.055 [0.038, 0.076] | 0.938 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_global_null | 6/6/6 | 0.053 [0.037, 0.074] | 0.942 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_partial_null_low | 6/6/6 | 0.053 [0.037, 0.074] | 0.935 | 0.003 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high | 6/6/6 | 0.062 [0.044, 0.084] | 0.938 | 0.001 | 0.995 [0.985, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high_transient | 6/6/6 | 0.052 [0.035, 0.073] | 0.948 | 0.001 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low | 6/6/6 | 0.055 [0.038, 0.076] | 0.945 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low_transient | 6/6/6 | 0.055 [0.038, 0.076] | 0.937 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | alternative_transient | 8/8/8 | — | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | global_null | 8/8/8 | 0.050 [0.034, 0.071] | 0.943 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_global_null | 8/8/8 | 0.052 [0.035, 0.073] | 0.932 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | heterogeneity_partial_null_low | 8/8/8 | 0.055 [0.038, 0.076] | 0.945 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high | 8/8/8 | 0.063 [0.045, 0.086] | 0.937 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_high_transient | 8/8/8 | 0.025 [0.014, 0.041] | 0.950 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low | 8/8/8 | 0.060 [0.042, 0.082] | 0.940 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | biweekly | partial_null_low_transient | 8/8/8 | 0.058 [0.041, 0.080] | 0.942 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | alternative_transient | 6/6/6 | — | 0.938 | 0.001 | 0.997 [0.988, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | global_null | 6/6/6 | 0.055 [0.038, 0.076] | 0.938 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_global_null | 6/6/6 | 0.053 [0.037, 0.074] | 0.942 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_partial_null_low | 6/6/6 | 0.053 [0.037, 0.074] | 0.935 | 0.003 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high | 6/6/6 | 0.062 [0.044, 0.084] | 0.938 | 0.001 | 0.995 [0.985, 0.999] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high_transient | 6/6/6 | 0.052 [0.035, 0.073] | 0.948 | 0.001 | 0.998 [0.991, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low | 6/6/6 | 0.055 [0.038, 0.076] | 0.945 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low_transient | 6/6/6 | 0.055 [0.038, 0.076] | 0.937 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | alternative_transient | 8/8/8 | — | 0.953 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | global_null | 8/8/8 | 0.050 [0.034, 0.071] | 0.943 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_global_null | 8/8/8 | 0.052 [0.035, 0.073] | 0.932 | 0.001 | — | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | heterogeneity_partial_null_low | 8/8/8 | 0.055 [0.038, 0.076] | 0.945 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high | 8/8/8 | 0.063 [0.045, 0.086] | 0.937 | 0.000 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_high_transient | 8/8/8 | 0.025 [0.014, 0.041] | 0.950 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low | 8/8/8 | 0.060 [0.042, 0.082] | 0.940 | 0.001 | 1.000 [0.994, 1.000] | 0.000 |
| `C1_mmrm_visit_contrast` | weekly | partial_null_low_transient | 8/8/8 | 0.058 [0.041, 0.080] | 0.942 | 0.002 | 1.000 [0.994, 1.000] | 0.000 |

## Interpretation (exploratory only)

- **B fails where it was predicted to fail.** Weekly restricted GLS exceeds the FWER point and CI ceilings and drops below the 0.92 coverage floor under slope heterogeneity ×1.6 (`heterogeneity_global_null` FWER 0.120 / 0.137). Transient weekly bias is 0.015–0.016 (fraction 0.10–0.11), matching the algebraic fixture 0.016. Biweekly GLS is unbiased on the two-visit schedule but still misses FWER and coverage under heterogeneity.
- **A meets the sourced inference tolerances** on this complete-data matrix at both sizes. Power is high because the alternative deltas (−0.10, −0.15) are larger than the deferred 5% meaningful-effect cell.
- **C1 also meets those tolerances here.** Complete-data point estimates match A (max |Δest| ≈ 10⁻¹⁴). Realized SEs do not: 156 Holm disagreements versus A. On complete data, C1 weekly and C1 biweekly are numerically identical for the endpoint contrast (est and CR1 SE); intermediate visits do not change complete-data C1. That is not evidence that C1 helps under missingness, which was not simulated.
- Meeting a tolerated FWER gate is not “FWER = 0.05.” This stream does not make C1 the primary analysis and does not justify switching defaults.

## Constraints on use

- Do not treat this stream as locked confirmation (that would be independent seed `20261201`, B=2000).
- Do not retune C1 (df, sandwich multiplier, working V0) on these draws and then call a later stream confirmatory.
- Do not switch production defaults.
- Do not claim prior-outcome MAR validity. Missingness here is complete.
- Deferred cells remain deferred: covariance shift, heavy tails, delayed alternatives, 5% effect, control-heavy, twice-weekly, MAR, C2.

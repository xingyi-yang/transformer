# Unresolved modeling limitations

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data validation. No operational recommendation.**

PR #7 and prior artifacts are preserved. Broad synthetic development stops here. Two modeling limitations remain unresolved. Neither was adjusted using exposed test outcomes.

## Status

- preserved PR: #7
- broad synthetic development: **stopped**
- next implementation task: `None`
- adjusted using exposed test: False
- reliability thresholds: `{'rhat_max': 1.01, 'ess_bulk_min': 400, 'ess_tail_min': 400}`
- record revision: `8d89add894d8ff6dabb9fc4c613a1f9f3ee8c418`

## 1. Excess increment variability

- status: **unresolved**
- adjusted using exposed test: False
- n_datasets: 2; cells: 64; study rows: 256
- matched pred/obs ratio, mean of dataset means: **1.41675** (sd of dataset means 0.14424)

After matching the animal→study→dataset increment statistic, predicted study-level increment SD still exceeds the observed study-level statistic (ratio > 1 on the mean of two dataset means). Excess increment variability remains an unresolved modeling limitation.

Not done: No residual scale was multiplied to match observed increment SD on test.; No increment-ratio target was used as a tuning objective.; Covariance bounds (including sigma_m at the optimizer floor) were not moved using exposed test outcomes.; Predictive widths were not calibrated to test coverage.

## 2. Unreliable Bayesian inference

- status: **unresolved**
- adjusted using exposed test: False
- explicitly unreliable: True; scored: False
- thresholds unchanged: `{'rhat_max': 1.01, 'ess_bulk_min': 400, 'ess_tail_min': 400}`

Executed Bayesian cells fail the unchanged reliability gate (rhat≤1.01, ESS≥400, divergences==0) on mean-component identifiability. Inference stays explicitly unreliable and is not counted.

| cell | class | rhat | ESS bulk/tail | counted |
|---|---|---|---|---|
| d0_t0_matched_context_specific_k0_full | mean_identifiability | 1.015 | 115.2 / 141.6 | False |
| d0_t0_both_context_specific_k0_full | mean_identifiability | 1.015 | 115.2 / 141.6 | False |

Not done: Thresholds were not relaxed.; The mean parameterization was not changed to chase ESS or R-hat.; Failed runs were not dropped or retuned on test.

## Phase B

- status: **ACCESS_BLOCKER**
- authorized: False
- created_ticket: False; self_approved: False
- source_opened: False
- endpoint_values_inspected: False
- models_trained: False
- design_recommendations: False

No genuine signed authorization was found. Phase B remains blocked until approval is supplied through the organization's accepted process. No source was opened. No further synthetic implementation task is generated.

If later authorized through the organization's accepted process: Metadata-only inspection: biological-support inventory, missingness, provenance, and candidate historical validation cohorts. Endpoint inspection, training, and design recommendations stay off unless separately authorized.

## Stop

No genuine signed authorization was found. Phase B remains blocked until approval is supplied through the organization's accepted process. No source was opened. No further synthetic implementation task is generated.

## Artifacts

- `runs/v3_limitations/limitations.json`
- `runs/v3_limitations/LIMITATIONS.md`


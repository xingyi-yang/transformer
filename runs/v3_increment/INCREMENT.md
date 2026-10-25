# Targeted increment diagnostic

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data validation. No operational recommendation.**

The synthetic closeout is the end of the implementation phase, subject to review. This note re-scores existing fitted models only. No architecture was added and no new training grid was launched.

## Labels and replication

- labels: ['Synthetic diagnostic.', 'Not design qualification.', 'No real-data validation.', 'No operational recommendation.', 'Exploratory.', 'Two independent dataset replicates.']
- exploratory: True
- independent dataset replicates: **2**
- cells scored: 64; study rows: 256
- train seeds per dataset: {0: 2, 1: 2}
- split: `test` (prepared test studies only)
- research freeze: `54095801f1d8d240bbf8b59e5a2506f66ddf5004`
- increment revision: `e62f9aec163085d79eec1fc78b1ec34c47020e68`

## Matched statistic

Observed and each simulated study draw use the same hierarchy:

1. Animal: SD of consecutive observed-visit log-weight increments.
2. Study: unweighted mean of those animal SDs.
3. Dataset: mean of study-level statistics (train seeds nested inside).

Each `sample_study` draw is one realized study (new study intercept/slope). The predicted study-level number is E[same statistic] over draws. The between-draw SD of that statistic is reported separately and is **not** folded into the increment SD.

- overall matched pred/obs ratio, mean of dataset means: **1.41675** (sd of dataset means 0.14424; n_datasets=2)

| model | sharing | bank | density | matched mean_ds (sd_ds) | obs study-stat mean_ds | pred E[study-stat] mean_ds | between-draw SD mean_ds | published pack ratio mean_ds | n_ds |
|---|---|---|---|---|---|---|---|---|---|
| spline | context_specific | both | full | 1.18609 (0.12331) | 0.02639 | 0.03113 | 0.00302 | 1.17162 | 2 |
| spline | context_specific | both | sparse | 1.27017 (0.13965) | 0.02639 | 0.03335 | 0.00356 | 1.25980 | 2 |
| spline | context_specific | matched | full | 1.18609 (0.12331) | 0.02639 | 0.03113 | 0.00302 | 1.17162 | 2 |
| spline | context_specific | matched | sparse | 1.27017 (0.13965) | 0.02639 | 0.03335 | 0.00356 | 1.25980 | 2 |
| spline | pooled | both | full | 1.59079 (0.49406) | 0.02639 | 0.04165 | 0.00475 | 1.52684 | 2 |
| spline | pooled | both | sparse | 1.39724 (0.21483) | 0.02639 | 0.03667 | 0.00612 | 1.34741 | 2 |
| spline | pooled | matched | full | 1.60546 (0.55710) | 0.02639 | 0.04205 | 0.00464 | 1.54155 | 2 |
| spline | pooled | matched | sparse | 1.43779 (0.25628) | 0.02639 | 0.03770 | 0.00577 | 1.38785 | 2 |
| spline | selective | both | full | 1.59929 (0.50302) | 0.02639 | 0.04187 | 0.00475 | 1.53568 | 2 |
| spline | selective | both | sparse | 1.38989 (0.21626) | 0.02639 | 0.03647 | 0.00610 | 1.33951 | 2 |
| spline | selective | matched | full | 1.61750 (0.56834) | 0.02639 | 0.04237 | 0.00464 | 1.55388 | 2 |
| spline | selective | matched | sparse | 1.43872 (0.25566) | 0.02639 | 0.03771 | 0.00577 | 1.38833 | 2 |
| spline | selective_regularized | both | full | 1.84746 (0.82042) | 0.02639 | 0.04842 | 0.00558 | 1.73521 | 2 |
| spline | selective_regularized | both | sparse | 1.63982 (0.50784) | 0.02639 | 0.04303 | 0.00945 | 1.55528 | 2 |
| spline | selective_regularized | matched | full | 1.85510 (0.86976) | 0.02639 | 0.04862 | 0.00577 | 1.73980 | 2 |
| spline | selective_regularized | matched | sparse | 1.67246 (0.50894) | 0.02639 | 0.04387 | 0.00954 | 1.58294 | 2 |
| transformer | context_specific | both | full | 1.21947 (0.15172) | 0.02639 | 0.03201 | 0.00238 | 1.21127 | 2 |
| transformer | context_specific | both | sparse | 1.37461 (0.11529) | 0.02639 | 0.03605 | 0.00253 | 1.36485 | 2 |
| transformer | context_specific | matched | full | 1.21947 (0.15172) | 0.02639 | 0.03201 | 0.00238 | 1.21127 | 2 |
| transformer | context_specific | matched | sparse | 1.37461 (0.11529) | 0.02639 | 0.03605 | 0.00253 | 1.36485 | 2 |
| transformer | pooled | both | full | 1.29217 (0.06363) | 0.02639 | 0.03397 | 0.00258 | 1.29131 | 2 |
| transformer | pooled | both | sparse | 1.34955 (0.01769) | 0.02639 | 0.03551 | 0.00277 | 1.35016 | 2 |
| transformer | pooled | matched | full | 1.33639 (0.12000) | 0.02639 | 0.03514 | 0.00261 | 1.33136 | 2 |
| transformer | pooled | matched | sparse | 1.40549 (0.04374) | 0.02639 | 0.03686 | 0.00274 | 1.39684 | 2 |
| transformer | selective | both | full | 1.38564 (0.00530) | 0.02639 | 0.03653 | 0.00261 | 1.38637 | 2 |
| transformer | selective | both | sparse | 1.26235 (0.13068) | 0.02639 | 0.03314 | 0.00252 | 1.25682 | 2 |
| transformer | selective | matched | full | 1.39253 (0.10677) | 0.02639 | 0.03672 | 0.00265 | 1.38977 | 2 |
| transformer | selective | matched | sparse | 1.33838 (0.12248) | 0.02639 | 0.03497 | 0.00257 | 1.31973 | 2 |
| transformer | selective_regularized | both | full | 1.35236 (0.00837) | 0.02639 | 0.03564 | 0.00270 | 1.34687 | 2 |
| transformer | selective_regularized | both | sparse | 1.32518 (0.07245) | 0.02639 | 0.03468 | 0.00276 | 1.31587 | 2 |
| transformer | selective_regularized | matched | full | 1.46131 (0.13920) | 0.02639 | 0.03853 | 0.00276 | 1.48637 | 2 |
| transformer | selective_regularized | matched | sparse | 1.24247 (0.04840) | 0.02639 | 0.03263 | 0.00260 | 1.23465 | 2 |

The published pack ratio is the previous animal-pooled `mean(sd_pred)/mean(sd_obs)` from `evaluate_prestudy`. It is shown only as the mismatched baseline. Widths were not calibrated to test coverage.

## Bayesian inference

- thresholds unchanged: `{'rhat_max': 1.01, 'ess_bulk_min': 400, 'ess_tail_min': 400}`
- explicitly unreliable: True
- scored: False
- cells recorded, not counted: 2

Unchanged gate: rhat≤1.01, ESS≥400, divergences==0. Failed runs are preserved. Bayesian increment is not counted until those pass.

| cell | class | rhat | ESS bulk/tail | counted |
|---|---|---|---|---|
| d0_t0_matched_context_specific_k0_full | mean_identifiability | 1.015 | 115.2 / 141.6 | False |
| d0_t0_both_context_specific_k0_full | mean_identifiability | 1.015 | 115.2 / 141.6 | False |

## Artifacts

- `runs/v3_increment/increment.json`
- `runs/v3_increment/study_rows.json`
- `runs/v3_increment/summary.json`
- `runs/v3_increment/bayesian.json`


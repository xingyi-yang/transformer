# Prefix-abstention diagnostic — closed (not confirmatory)

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Closed without retuning. Frozen prefix-abstention numbers in `runs/v3_share_abstain_run/` and `docs/v3/share_abstain_run_RESULTS.md` are unchanged. Two dataset seeds and one training seed are **not confirmatory**. Prefix donor choice remains a STUDYID sort.

- source: `runs/v3_share_abstain_run/SUMMARY.json`
- seconds_total: `326.66133666038513`; cap hit: `False`
- failures: `[]`
- protected unchanged: `True`
- prefix versus random equal on all slices: `True`
- retuned: `False`; confirmatory: False

## Abstention counts (validation-only)

- spline: `4/8` target-only, `4/8` share
- transformer: `5/8` target-only, `3/8` share
- overall: `9/16` target-only, `7/16` share

## Paired MAE/RMSE magnitudes by model and dataset

Δ = other − target-only. Δ>0 is harmful. Magnitudes are means over the four bank×density cells in that dataset (regularized: matched/full only).

| dataset | model | arm | n | MAE mean Δ | MAE mean \|Δ\| | MAE help/harm/null | RMSE mean Δ | RMSE mean \|Δ\| | RMSE help/harm/null |
|---|---|---|---|---|---|---|---|---|---|
| 0 | spline | selected_one | 4 | -0.00153 | 0.00153 | 4/0/0 | -0.00190 | 0.00190 | 4/0/0 |
| 0 | spline | selected_abstain | 4 | -0.00108 | 0.00108 | 3/0/1 | -0.00136 | 0.00136 | 3/0/1 |
| 0 | spline | random_one | 4 | -0.00153 | 0.00153 | 4/0/0 | -0.00190 | 0.00190 | 4/0/0 |
| 0 | spline | pooled_all | 4 | -0.00222 | 0.00222 | 4/0/0 | -0.00236 | 0.00236 | 4/0/0 |
| 0 | spline | selected_regularized | 1 | -0.00113 | 0.00113 | 1/0/0 | -0.00252 | 0.00252 | 1/0/0 |
| 0 | transformer | selected_one | 4 | +0.00524 | 0.00964 | 2/2/0 | +0.00498 | 0.01210 | 2/2/0 |
| 0 | transformer | selected_abstain | 4 | -0.00143 | 0.00143 | 1/0/3 | -0.00222 | 0.00222 | 1/0/3 |
| 0 | transformer | random_one | 4 | +0.00524 | 0.00964 | 2/2/0 | +0.00498 | 0.01210 | 2/2/0 |
| 0 | transformer | pooled_all | 4 | +0.00180 | 0.00259 | 1/3/0 | +0.00110 | 0.00511 | 3/1/0 |
| 0 | transformer | selected_regularized | 1 | -0.00481 | 0.00481 | 1/0/0 | -0.00772 | 0.00772 | 1/0/0 |
| 1 | spline | selected_one | 4 | +0.00158 | 0.00217 | 1/3/0 | +0.00014 | 0.00143 | 1/3/0 |
| 1 | spline | selected_abstain | 4 | -0.00030 | 0.00030 | 1/0/3 | -0.00064 | 0.00064 | 1/0/3 |
| 1 | spline | random_one | 4 | +0.00158 | 0.00217 | 1/3/0 | +0.00014 | 0.00143 | 1/3/0 |
| 1 | spline | pooled_all | 4 | +0.00219 | 0.00219 | 0/4/0 | +0.00127 | 0.00127 | 0/4/0 |
| 1 | spline | selected_regularized | 1 | +0.00315 | 0.00315 | 0/1/0 | +0.00152 | 0.00152 | 0/1/0 |
| 1 | transformer | selected_one | 4 | +0.00911 | 0.00911 | 0/4/0 | +0.01103 | 0.01103 | 0/4/0 |
| 1 | transformer | selected_abstain | 4 | +0.00196 | 0.00196 | 0/2/2 | +0.00304 | 0.00304 | 0/2/2 |
| 1 | transformer | random_one | 4 | +0.00911 | 0.00911 | 0/4/0 | +0.01103 | 0.01103 | 0/4/0 |
| 1 | transformer | pooled_all | 4 | +0.00544 | 0.00544 | 0/4/0 | +0.00635 | 0.00635 | 0/4/0 |
| 1 | transformer | selected_regularized | 1 | -0.00126 | 0.00126 | 1/0/0 | -0.00030 | 0.00030 | 1/0/0 |

### Cell-level selected_one and selected_abstain

| dataset | bank | density | model | arm | MAE Δ | MAE sign | RMSE Δ | RMSE sign |
|---|---|---|---|---|---|---|---|---|
| 0 | matched | full | spline | selected_one | -0.00160 | helpful | -0.00295 | helpful |
| 0 | matched | sparse | spline | selected_one | -0.00093 | helpful | -0.00044 | helpful |
| 0 | both | full | spline | selected_one | -0.00183 | helpful | -0.00216 | helpful |
| 0 | both | sparse | spline | selected_one | -0.00178 | helpful | -0.00205 | helpful |
| 0 | matched | full | spline | selected_abstain | -0.00160 | helpful | -0.00295 | helpful |
| 0 | matched | sparse | spline | selected_abstain | -0.00093 | helpful | -0.00044 | helpful |
| 0 | both | full | spline | selected_abstain | +0.00000 | null | +0.00000 | null |
| 0 | both | sparse | spline | selected_abstain | -0.00178 | helpful | -0.00205 | helpful |
| 0 | matched | full | transformer | selected_one | -0.00572 | helpful | -0.00887 | helpful |
| 0 | matched | sparse | transformer | selected_one | +0.01346 | harmful | +0.01492 | harmful |
| 0 | both | full | transformer | selected_one | -0.00308 | helpful | -0.00538 | helpful |
| 0 | both | sparse | transformer | selected_one | +0.01631 | harmful | +0.01923 | harmful |
| 0 | matched | full | transformer | selected_abstain | -0.00572 | helpful | -0.00887 | helpful |
| 0 | matched | sparse | transformer | selected_abstain | +0.00000 | null | +0.00000 | null |
| 0 | both | full | transformer | selected_abstain | +0.00000 | null | +0.00000 | null |
| 0 | both | sparse | transformer | selected_abstain | +0.00000 | null | +0.00000 | null |
| 1 | matched | full | spline | selected_one | +0.00273 | harmful | +0.00108 | harmful |
| 1 | matched | sparse | spline | selected_one | -0.00118 | helpful | -0.00257 | helpful |
| 1 | both | full | spline | selected_one | +0.00241 | harmful | +0.00054 | harmful |
| 1 | both | sparse | spline | selected_one | +0.00235 | harmful | +0.00151 | harmful |
| 1 | matched | full | spline | selected_abstain | +0.00000 | null | +0.00000 | null |
| 1 | matched | sparse | spline | selected_abstain | -0.00118 | helpful | -0.00257 | helpful |
| 1 | both | full | spline | selected_abstain | +0.00000 | null | +0.00000 | null |
| 1 | both | sparse | spline | selected_abstain | +0.00000 | null | +0.00000 | null |
| 1 | matched | full | transformer | selected_one | +0.00420 | harmful | +0.00596 | harmful |
| 1 | matched | sparse | transformer | selected_one | +0.01506 | harmful | +0.01533 | harmful |
| 1 | both | full | transformer | selected_one | +0.00364 | harmful | +0.00619 | harmful |
| 1 | both | sparse | transformer | selected_one | +0.01356 | harmful | +0.01665 | harmful |
| 1 | matched | full | transformer | selected_abstain | +0.00420 | harmful | +0.00596 | harmful |
| 1 | matched | sparse | transformer | selected_abstain | +0.00000 | null | +0.00000 | null |
| 1 | both | full | transformer | selected_abstain | +0.00364 | harmful | +0.00619 | harmful |
| 1 | both | sparse | transformer | selected_abstain | +0.00000 | null | +0.00000 | null |

## Reading (not a retune cue)

- Dataset 0 spline prefix-one was helpful on all 4/4 cells (mean MAE Δ −0.00153, mean RMSE Δ −0.00190).
- Dataset 0 transformer prefix-one was mixed (2 helpful / 2 harmful); sparse cells were harmful (MAE Δ up to +0.01631).
- Dataset 1 spline prefix-one was mostly harmful (3/4); transformer prefix-one was harmful on 4/4 (mean MAE Δ +0.00911, mean RMSE Δ +0.01103).
- Abstention (selected_abstain) was 5 helpful / 2 harmful / 9 null of 16. The two harmful shares are transformer on dataset 1 full.
- Prefix versus random was degenerate (SYNA-D001 on 8/8). That does not generalize beyond this 3-donor bank and seed 91001.
- These magnitudes are not a reason to change thresholds, seeds, or donor rules.

Stop. No automatic expansion.

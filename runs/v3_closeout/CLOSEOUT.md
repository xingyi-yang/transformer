# Synthetic implementation closeout

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

PR #5 completes the synthetic implementation milestone, subject to normal code review. This note uses those artifacts only. Unfavorable signs are preserved. No architecture was added and no new training grid was launched.

## Provenance

- research freeze: `54095801f1d8d240bbf8b59e5a2506f66ddf5004`
- research config hash: `438cb0640f9f9ff8`
- closeout revision: `5f5a669e81b500a7070d8996d547c36f1432778f`
- reliability thresholds: `{'rhat_max': 1.01, 'ess_bulk_min': 400, 'ess_tail_min': 400}`

## Nested seeds and splits

- Training seeds change target splits: **True**.
- Target-manifest hashes are STUDYID-set hashes. Training seeds reshuffle which of those IDs are validation vs test. Dataset seeds change the outcomes behind the same generator labels.
- Outer replicate: dataset / generate_seed (n=2).
- Nested replicate: train_seed (n=2 per dataset), which reshuffles validation/test STUDYIDs and refits.
- Published research tables used pooled n=4. That is not a dataset-level n.

- `d0`: 2 train seeds; validation hashes differ=True; test hashes differ=True.
- `d1`: 2 train seeds; validation hashes differ=True; test hashes differ=True.

## Sharing × density (RMSE and CRPS)

Sharing Δ = S_k − S_0 (context-specific). Sparse Δ = S_sparse − S_full. Interaction = sharing Δ under sparse minus sharing Δ under full. Δ>0 is harmful for RMSE/CRPS. `mean_ds` is the mean of the two dataset means; `sd_ds` is the dataset-level sd (n=2). `pooled` is the 2×2 iid summary.

| model | sharing | bank | metric | share full mean_ds (sd_ds) | share sparse mean_ds | sparse-vs-full mean_ds | interaction mean_ds |
|---|---|---|---|---|---|---|---|
| spline | context_specific | both | crps | — (—) | — | 0.00746 | — |
| spline | context_specific | both | rmse_log | — (—) | — | 0.01230 | — |
| spline | context_specific | matched | crps | — (—) | — | 0.00746 | — |
| spline | context_specific | matched | rmse_log | — (—) | — | 0.01230 | — |
| spline | pooled | both | crps | 0.00619 (0.00397) | 0.00134 | 0.00262 | -0.00484 |
| spline | pooled | both | rmse_log | 0.01097 (0.00761) | -0.00470 | -0.00337 | -0.01567 |
| spline | pooled | matched | crps | 0.00565 (0.00289) | 0.00086 | 0.00267 | -0.00479 |
| spline | pooled | matched | rmse_log | 0.01036 (0.00611) | -0.00394 | -0.00200 | -0.01431 |
| spline | selective | both | crps | 0.00551 (0.00355) | -0.00004 | 0.00191 | -0.00555 |
| spline | selective | both | rmse_log | 0.01074 (0.00740) | -0.00500 | -0.00344 | -0.01574 |
| spline | selective | matched | crps | 0.00511 (0.00268) | -0.00032 | 0.00203 | -0.00543 |
| spline | selective | matched | rmse_log | 0.01017 (0.00591) | -0.00428 | -0.00215 | -0.01445 |
| spline | selective_regularized | both | crps | 0.00830 (0.00266) | 0.00531 | 0.00446 | -0.00300 |
| spline | selective_regularized | both | rmse_log | 0.01722 (0.00612) | -0.00076 | -0.00568 | -0.01798 |
| spline | selective_regularized | matched | crps | 0.00849 (0.00258) | 0.00556 | 0.00453 | -0.00293 |
| spline | selective_regularized | matched | rmse_log | 0.01747 (0.00571) | 0.00188 | -0.00329 | -0.01559 |
| transformer | context_specific | both | crps | — (—) | — | 0.00194 | — |
| transformer | context_specific | both | rmse_log | — (—) | — | 0.00288 | — |
| transformer | context_specific | matched | crps | — (—) | — | 0.00194 | — |
| transformer | context_specific | matched | rmse_log | — (—) | — | 0.00288 | — |
| transformer | pooled | both | crps | -0.00438 (0.00287) | 0.01297 | 0.01929 | 0.01736 |
| transformer | pooled | both | rmse_log | -0.00907 (0.00440) | 0.02218 | 0.03413 | 0.03125 |
| transformer | pooled | matched | crps | -0.00426 (0.00314) | 0.01190 | 0.01810 | 0.01616 |
| transformer | pooled | matched | rmse_log | -0.00880 (0.00438) | 0.02001 | 0.03168 | 0.02880 |
| transformer | selective | both | crps | -0.00189 (0.00175) | 0.00978 | 0.01361 | 0.01167 |
| transformer | selective | both | rmse_log | -0.00277 (0.00166) | 0.01687 | 0.02252 | 0.01964 |
| transformer | selective | matched | crps | -0.00354 (0.00393) | 0.00986 | 0.01534 | 0.01340 |
| transformer | selective | matched | rmse_log | -0.00574 (0.00516) | 0.01563 | 0.02424 | 0.02137 |
| transformer | selective_regularized | both | crps | -0.00184 (0.00012) | 0.01872 | 0.02250 | 0.02056 |
| transformer | selective_regularized | both | rmse_log | -0.00357 (0.00137) | 0.02950 | 0.03595 | 0.03307 |
| transformer | selective_regularized | matched | crps | -0.00459 (0.00370) | 0.01474 | 0.02127 | 0.01933 |
| transformer | selective_regularized | matched | rmse_log | -0.00838 (0.00585) | 0.02237 | 0.03363 | 0.03075 |

Spline sharing remains harmful for RMSE at full density on both donor banks. Sparse target-study training makes spline sharing less harmful (negative interaction) and can reverse the sparse-vs-full sign once donors are present. Transformer donor-using sparse training stays harmful for RMSE and CRPS. Signs are published; they were not retuned.

## Contrast: row-weighted vs study-balanced

| model | sharing | bank | density | row-mean Δ | study-balanced Δ | n_ds |
|---|---|---|---|---|---|---|
| spline | pooled | both | full | -0.00694 | -0.00285 | 2 |
| spline | pooled | both | sparse | -0.00899 | -0.01197 | 2 |
| spline | pooled | matched | full | -0.00822 | -0.00479 | 2 |
| spline | pooled | matched | sparse | -0.00249 | -0.00920 | 2 |
| spline | selective | both | full | -0.00720 | -0.00315 | 2 |
| spline | selective | both | sparse | -0.00945 | -0.01207 | 2 |
| spline | selective | matched | full | -0.00854 | -0.00514 | 2 |
| spline | selective | matched | sparse | -0.00315 | -0.00951 | 2 |
| spline | selective_regularized | both | full | -0.00604 | -0.00178 | 2 |
| spline | selective_regularized | both | sparse | -0.01007 | -0.01124 | 2 |
| spline | selective_regularized | matched | full | -0.00879 | -0.00475 | 2 |
| spline | selective_regularized | matched | sparse | -0.00255 | -0.00955 | 2 |
| transformer | pooled | both | full | -0.01138 | -0.00899 | 2 |
| transformer | pooled | both | sparse | 0.04759 | 0.04461 | 2 |
| transformer | pooled | matched | full | -0.01088 | -0.01114 | 2 |
| transformer | pooled | matched | sparse | 0.03810 | 0.03263 | 2 |
| transformer | selective | both | full | -0.00677 | -0.00396 | 2 |
| transformer | selective | both | sparse | 0.02699 | 0.02291 | 2 |
| transformer | selective | matched | full | -0.00826 | -0.00543 | 2 |
| transformer | selective | matched | sparse | 0.02814 | 0.02243 | 2 |
| transformer | selective_regularized | both | full | -0.00371 | -0.00079 | 2 |
| transformer | selective_regularized | both | sparse | 0.05397 | 0.04368 | 2 |
| transformer | selective_regularized | matched | full | -0.00933 | -0.00904 | 2 |
| transformer | selective_regularized | matched | sparse | 0.03202 | 0.02679 | 2 |

Study-balanced contrast can differ in magnitude from the contrast-row mean because studies contribute unequal numbers of active arms. Neither series was substituted for the other in the frozen transfer tables.

## Overcoverage and increment variability

- packs: 128; cov80>0.80 in 100; cov95>0.95 in 86; increment ratio>1 in 128
- mean cov80/cov95: 0.86272 / 0.95651
- mean increment SD pred/obs: 1.38611
- covariance inspections with a bound: 12 / 64; params {'sigma_m': 9, 'sd_a0': 2, 'sd_s0': 1}

New-study draws add a shared study intercept/slope plus animal and OU terms. Increment SD is the mean, over draws, of per-animal consecutive-visit SDs; that includes the new study slope, so pred increment SD can exceed the observed within-animal increment SD. HierarchicalCovariance often parks sigma_m at its optimizer floor (train OOF, not a test-calibrated nugget). Wide predictive intervals then overcover. Widths were not shrunk to match exposed test coverage.

Not done: No residual scale was multiplied to hit nominal 80/95 coverage on test.; No increment-ratio target was used as a tuning objective.; Validation pre-study packs were not written by the research executor; compound-holdout coverage is a second exposed split and was not used to calibrate.

## Bayesian mean identifiability

- thresholds unchanged: `{'rhat_max': 1.01, 'ess_bulk_min': 400, 'ess_tail_min': 400}`
- failed runs preserved: True

| cell | class | rhat | ESS bulk/tail | dropped | counted |
|---|---|---|---|---|---|
| d0_t0_matched_context_specific_k0_full | mean_identifiability | 1.015 | 115.2 / 141.6 | ['ob_ld[0]'] | False |
| d0_t0_both_context_specific_k0_full | mean_identifiability | 1.015 | 115.2 / 141.6 | ['ob_ld[0]'] | False |

Both executed cells fail on theta/beta (mean_identifiability), not compile. The dropped column is the near-empty off-treatment basis (ob_ld[0]; OFF≈0). Remaining poorly mixed coordinates sit on strain×time bases: few studies identify strain-specific growth once ZeroSum study effects and the shared time spline are present. Doubling draws/tune did not meet ESS≥400. Parameterization and thresholds were not changed to chase reliability.

## Historical-data handoff

Synthetic implementation stops here. The historical-data audit is a separate gated workflow (`send-bw historical-audit`). It does not read real extracts until a signed authorization names the phase. Design recommendations stay off.

## Artifacts

- `runs/v3_closeout/closeout.json`
- `runs/v3_closeout/sharing_density_effects.json`
- `runs/v3_closeout/study_balanced_contrast.json`
- `runs/v3_closeout/coverage_increments.json`
- `runs/v3_closeout/bayesian_identifiability.json`


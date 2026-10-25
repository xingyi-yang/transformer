# Bounded official-sequence diagnostic

> **SUPERSEDED for the name `rmse_log` and for Chronos-2 fine-tune pooling.**
> `rmse_log` below is `|error|`. Dataset-0 Chronos LoRA failed and must not be
> averaged with dataset 1. Corrected metrics:
> [eval_correct_RESULTS.md](eval_correct_RESULTS.md). These tables are preserved.

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data validation. No operational recommendation.**

PR #10 (`runs/v3_official_seq/`) is frozen and was not rewritten. Early-history is scored separately from predose. Design stayed off.

## Provenance

- git: `a05b3b3c067277cf49812afc066147f5a03a6a99`
- generate_seed / train_seed: `72001` / `82001`
- n_datasets: `2`
- retuned_after_test: `False`
- official_seq_modified: `False`
- Chronos-2 SHA: `29ec3766d36d6f73f0696f85560a422f50e8498c` (huggingface_hub.model_info)
- official mean is median: `True`

## Phase A — correctness on the frozen official-seq holdout

Common quantile scoring. Point = median (q0.5) for quantile models; Gaussian MEAN for spline/transformer. `rmse_log` is the study-balanced mean of |point − LOGW|.

**Scoring alignment.** Official-seq prediction tables carried `STUDYID`. Merging them with observations produced `STUDYID_x` / `STUDYID_y`, so the published official_seq `n_studies=58` was animal-level. Common scoring now keeps the observed `STUDYID` (2 target test studies). Official-seq tables in `runs/v3_official_seq/` are not rewritten.

Last-value beats Chronos-2 on RMSE on this holdout (0.114 vs 0.128–0.136) and loses on pinball because its intervals have width 0. That is not a reason to change encoding, add covariates, or lengthen training.

| model | RMSE | pinball | cov80 | width80 | n |
|---|---|---|---|---|---|
| spline_target_only | 0.0463 | 0.0166 | 0.7817 | 0.1397 | 115 |
| spline_pooled | 0.0343 | 0.0144 | 1.0000 | 0.2233 | 115 |
| transformer_target_only | 0.0405 | 0.0142 | 0.7101 | 0.1083 | 115 |
| transformer_pooled | 0.0355 | 0.0124 | 0.7842 | 0.1171 | 115 |
| tft_target_only | 0.0455 | 0.0159 | 0.7332 | 0.1286 | 115 |
| tft_pooled | 0.0526 | 0.0197 | 0.6709 | 0.1881 | 115 |
| chronos2_zero_shot | 0.1355 | 0.0437 | 0.9535 | 0.7115 | 115 |
| chronos2_target_only | 0.1271 | 0.0422 | 0.9535 | 0.7014 | 115 |
| chronos2_pooled | 0.1287 | 0.0422 | 0.9535 | 0.6902 | 115 |
| last_value | 0.1140 | 0.0570 | 0.0000 | 0.0000 | 115 |

### Chronos transformation / time / covariates

- official encoding: `LOGW`; changed: `False`
- mean equals median on all series: `True`
- LOGW-input RMSE: `0.13549874475031248`
- exp(LOGW)-input logged-back RMSE: `0.13463477463358578` (diagnostic only)
- encoder animals: `58`; irregular: `0`; median gap: `7.0`
- Chronos time index: `visit_rank_unit_steps`; calendar time passed: `False`
- covariates unused: `['SEX', 'STRAIN', 'ROUTE', 'DOSE', 'SPECIES', 'DAY', 'COMPOUND_ID']`

## Phase B — prespecified fresh replicates (early-history only)

Predose was not evaluated. Harmful transfer is published as-is.

### dataset 0 (gen 72001 / train 82001)

- holdout hashes: `{'target_only': {'validation': 'ec72fd0e04ae6f71', 'test': 'dbf1e2c7d597db26', 'all_target': '9caa839ea3d54c92'}, 'pooled': {'validation': 'ec72fd0e04ae6f71', 'test': 'dbf1e2c7d597db26', 'all_target': '9caa839ea3d54c92'}}`

| model | RMSE | pinball | cov80 | width80 |
|---|---|---|---|---|
| last_value | 0.1163 | 0.0581 | 0.0000 | 0.0000 |
| spline_target_only | 0.0319 | 0.0110 | 0.8410 | 0.1136 |
| transformer_target_only | 0.0373 | 0.0126 | 0.7402 | 0.1049 |
| spline_pooled | 0.0406 | 0.0167 | 0.9888 | 0.2468 |
| transformer_pooled | 0.0357 | 0.0148 | 0.9775 | 0.2238 |
| tft_target_only | 0.0462 | 0.0162 | 0.6691 | 0.1135 |
| tft_pooled | 0.0405 | 0.0138 | 0.8113 | 0.1557 |
| chronos2_zero_shot | 0.1391 | 0.0456 | 0.9600 | 0.7475 |
| chronos2_target_only | failed | length < min_past + prediction_length (2+6) | — | — |
| chronos2_pooled | failed | length < min_past + prediction_length (2+6) | — | — |

| family | metric | target-only | pooled | Δ |
|---|---|---|---|---|
| spline | rmse_log | 0.03186304532472551 | 0.04062217702370737 | 0.008759131698981862 |
| spline | mean_quantile_loss | 0.011009011299395095 | 0.016691907104145164 | 0.005682895804750069 |
| transformer | rmse_log | 0.037278425394740904 | 0.035733657014941134 | -0.00154476837979977 |
| transformer | mean_quantile_loss | 0.012608465226420437 | 0.014787004220336505 | 0.002178538993916068 |
| tft | rmse_log | 0.04618120898768889 | 0.04051558993752243 | -0.005665619050166458 |
| tft | mean_quantile_loss | 0.01615362770781391 | 0.013765648449096165 | -0.0023879792587177436 |

### dataset 1 (gen 73010 / train 82018)

- holdout hashes: `{'target_only': {'validation': '5f4c5fe8d108dc8d', 'test': 'd4b3062a5b14b063', 'all_target': '93442f4925de67c9'}, 'pooled': {'validation': '5f4c5fe8d108dc8d', 'test': 'd4b3062a5b14b063', 'all_target': '93442f4925de67c9'}}`

| model | RMSE | pinball | cov80 | width80 |
|---|---|---|---|---|
| last_value | 0.1032 | 0.0516 | 0.0000 | 0.0000 |
| spline_target_only | 0.0297 | 0.0105 | 0.8854 | 0.1171 |
| transformer_target_only | 0.0434 | 0.0153 | 0.6802 | 0.1066 |
| spline_pooled | 0.0289 | 0.0121 | 0.9035 | 0.1636 |
| transformer_pooled | 0.0289 | 0.0101 | 0.8458 | 0.1059 |
| tft_target_only | 0.0634 | 0.0236 | 0.4875 | 0.1225 |
| tft_pooled | 0.0736 | 0.0244 | 0.5726 | 0.1333 |
| chronos2_zero_shot | 0.1104 | 0.0360 | 0.9551 | 0.5438 |
| chronos2_target_only | 0.1035 | 0.0345 | 0.9551 | 0.5321 |
| chronos2_pooled | 0.1036 | 0.0345 | 0.9551 | 0.5327 |

| family | metric | target-only | pooled | Δ |
|---|---|---|---|---|
| spline | rmse_log | 0.029744646886299107 | 0.028869204228532886 | -0.0008754426577662205 |
| spline | mean_quantile_loss | 0.010518640709306592 | 0.012117501999434295 | 0.0015988612901277031 |
| transformer | rmse_log | 0.043385867880032666 | 0.028924790747980395 | -0.014461077132052271 |
| transformer | mean_quantile_loss | 0.015320224996837526 | 0.010071169248990958 | -0.005249055747846568 |
| tft | rmse_log | 0.06339728580201573 | 0.07361999761191806 | 0.010222711809902335 |
| tft | mean_quantile_loss | 0.023633102190649763 | 0.024416461673702813 | 0.0007833594830530498 |
| chronos2_finetune | rmse_log | 0.1035157522770932 | 0.1036412493056296 | 0.00012549702853639766 |
| chronos2_finetune | mean_quantile_loss | 0.03448644066119095 | 0.03452067015294237 | 3.422949175142104e-05 |

### Pooled across replicates

| model | n | RMSE mean | RMSE sd | pinball mean |
|---|---|---|---|---|
| chronos2_pooled | 1 | 0.1036 | n/a | 0.0345 |
| chronos2_target_only | 1 | 0.1035 | n/a | 0.0345 |
| chronos2_zero_shot | 2 | 0.1248 | 0.0203 | 0.0408 |
| last_value | 2 | 0.1097 | 0.0093 | 0.0549 |
| spline_pooled | 2 | 0.0347 | 0.0083 | 0.0144 |
| spline_target_only | 2 | 0.0308 | 0.0015 | 0.0108 |
| tft_pooled | 2 | 0.0571 | 0.0234 | 0.0191 |
| tft_target_only | 2 | 0.0548 | 0.0122 | 0.0199 |
| transformer_pooled | 2 | 0.0323 | 0.0048 | 0.0124 |
| transformer_target_only | 2 | 0.0403 | 0.0043 | 0.0140 |

## Blockers and unsupported tasks

- blockers: []
- Chronos-2 LoRA on dataset 0 failed: every training series was shorter than `min_past + prediction_length` (2+6). Dataset 1 fine-tunes succeeded. Budgets were not reduced after seeing that failure.
- unsupported: {'predose': 'early-history is scored separately; TFT/Chronos/last-value refuse fabricated encoder', 'design_simulation': 'marginal quantiles are not complete trajectory simulators', 'crps_for_quantile_models': 'CRPS reported only for Gaussian spline/transformer'}
- real_data_accessed: False
- design_run: False
- predose_evaluated: False
- seconds_total: 181.82129383087158


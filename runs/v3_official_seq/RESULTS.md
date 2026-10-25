# Official TFT and Chronos-2 — executed early-history results

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data validation. No operational recommendation.**

This is **not** a spline or custom-transformer substitute. TFT is `pytorch_forecasting.TemporalFusionTransformer`. Chronos-2 is `chronos.Chronos2Pipeline` from `amazon/chronos-2`.

## Provenance

- git: `7dec702d773982bb502332f57c28cc6a3ae30c93`
- generate_seed / train_seed: `71001` / `81001`
- device: `cpu`; cuda=False
- TFT class: `pytorch_forecasting.models.temporal_fusion_transformer._tft.TemporalFusionTransformer`
- Chronos-2 class: `chronos.chronos2.pipeline.Chronos2Pipeline`

| package | version |
|---|---|
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| torch | 2.14.1+cu130 |
| pytorch_forecasting | 1.8.0 |
| lightning | 2.6.6 |
| chronos | 2.3.2 |
| transformers | 5.20.2 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |

## Holdouts

- encoder cutoff: DAY ≤ 14.0 (observed visits only)
- target holdout hashes: `{'target_only': {'validation': '37fb8b3fef2c9ca6', 'test': 'a60851220ed963ad', 'all_target': '911476c55c2cf369'}, 'pooled': {'validation': '37fb8b3fef2c9ca6', 'test': 'a60851220ed963ad', 'all_target': '911476c55c2cf369'}}`
- pooled training reuses the same target validation/test STUDYIDs
- encoder history is never fabricated

## Training commands

- `send-bw generate --config runs/v3_official_seq/datasets/d0/banks/matched/effective.yaml`
- `send-bw train --config runs/v3_official_seq/datasets/d0/trains/t0/cells/d0_t0_matched_target_only/effective.yaml --models spline transformer`
- `send-bw train --config runs/v3_official_seq/datasets/d0/trains/t0/cells/d0_t0_matched_pooled/effective.yaml --models spline transformer`
- `TFTEarlyHistoryModel.fit training_scope=target_only TemporalFusionTransformer.from_dataset`
- `TFTEarlyHistoryModel.fit training_scope=pooled TemporalFusionTransformer.from_dataset`
- `Chronos2Pipeline.from_pretrained('amazon/chronos-2')  # zero-shot`
- `Chronos2Pipeline.from_pretrained('amazon/chronos-2').fit finetune_mode=lora scope=target_only`
- `Chronos2Pipeline.from_pretrained('amazon/chronos-2').fit finetune_mode=lora scope=pooled`

## Metrics (study-balanced log-weight)

CRPS is Gaussian-only. TFT and Chronos-2 report pinball / coverage / widths from their quantile representation.

| model | status | RMSE | mean pinball | cov80 | width80 | CRPS | seconds |
|---|---|---|---|---|---|---|---|
| spline_target_only | ok | 0.0434 | 0.0155 | 0.8103 | 0.1398 | 0.0312 | 3.067136764526367 |
| transformer_target_only | ok | 0.0409 | 0.0143 | 0.6983 | 0.1081 | 0.0290 | 0.7572121620178223 |
| spline_pooled | ok | 0.0335 | 0.0149 | 1.0000 | 0.2371 | 0.0289 | 3.1625773906707764 |
| transformer_pooled | ok | 0.0358 | 0.0125 | 0.7845 | 0.1168 | 0.0255 | 0.8024265766143799 |
| tft_target_only | ok | 0.0453 | 0.0158 | 0.7328 | 0.1284 | n/a | 20.96111536026001 |
| tft_pooled | ok | 0.0528 | 0.0203 | 0.6293 | 0.1868 | n/a | 27.957898378372192 |
| chronos2_zero_shot | ok | 0.1412 | 0.0457 | 0.9655 | 0.7534 | n/a | 3.760669469833374 |
| chronos2_target_only | ok | 0.1322 | 0.0442 | 0.9655 | 0.7426 | n/a | 9.839610815048218 |
| chronos2_pooled | ok | 0.1340 | 0.0441 | 0.9655 | 0.7307 | n/a | 9.414591073989868 |

## Paired transfer (pooled − target-only; Δ>0 harmful)

| family | metric | target-only | pooled | Δ |
|---|---|---|---|---|
| spline | rmse_log | 0.04343930242637683 | 0.033481211391917634 | -0.009958091034459195 |
| spline | mean_quantile_loss | 0.01545871131360564 | 0.014863163690000031 | -0.0005955476236056082 |
| transformer | rmse_log | 0.04093284858624157 | 0.03582867920089618 | -0.005104169385345393 |
| transformer | mean_quantile_loss | 0.01425311727056803 | 0.012529612998654528 | -0.0017235042719135028 |
| tft | rmse_log | 0.04528301706341667 | 0.05280747718424303 | 0.0075244601208263595 |
| tft | mean_quantile_loss | 0.015824164327326726 | 0.020343781741419313 | 0.004519617414092587 |
| chronos2_finetune | rmse_log | 0.13224790921063823 | 0.13396545199049495 | 0.0017175427798567233 |
| chronos2_finetune | mean_quantile_loss | 0.04415222581655511 | 0.04408690100908479 | -6.532480747031832e-05 |

## Predictions

Scored future-day tables (DAY > 14) are in `runs/v3_official_seq/predictions/`. Each model has 115 scored rows (spline/transformer CSVs have 116 rows; one extra calendar time is dropped at merge). Reload-rescore RMSE matched the executed SUMMARY.

| file | n_rows | sha256_16 |
|---|---|---|
| spline_target_only.csv | 116 | 5db9812ac2c73385 |
| transformer_target_only.csv | 116 | 1b99cf3c08f46982 |
| spline_pooled.csv | 116 | 561bf3c084e41d2c |
| transformer_pooled.csv | 116 | b2127105b1b9bed8 |
| tft_target_only.csv | 115 | 544bdb675a563f3c |
| tft_pooled.csv | 115 | 055fb3748728d3e3 |
| chronos2_zero_shot.csv | 115 | e0c2080cf82d70c9 |
| chronos2_target_only.csv | 115 | 9746ea22a74b473e |
| chronos2_pooled.csv | 115 | 8fd24599f4132393 |

Zero-shot Chronos-2 full weights (`models/chronos2_zero_shot/pipeline/`, ~456M) and `hf_cache/` are not committed. Reload uses `amazon/chronos-2`. LoRA adapters and TFT checkpoints are committed.

## Serialization / reload

`{'tft_target_only': {'ok': True, 'max_abs_median_diff': 0.0, 'n_compared': 2}, 'tft_pooled': {'ok': True, 'max_abs_median_diff': 0.0, 'n_compared': 2}, 'chronos2_zero_shot': {'ok': True, 'max_abs_median_diff': 0.0, 'n_compared': 2}, 'chronos2_target_only': {'ok': True, 'max_abs_median_diff': 0.0, 'n_compared': 2}, 'chronos2_pooled': {'ok': True, 'max_abs_median_diff': 0.0, 'n_compared': 2}}`

## Blockers and unsupported tasks

- blockers: []
- unsupported: {'pre_study_baseline_only': 'TFT and Chronos-2 refuse fabricated encoder history', 'design_simulation': 'marginal quantiles are not complete trajectory simulators', 'crps_for_quantile_models': 'CRPS reported only for Gaussian spline/transformer'}
- real_data_accessed: False
- design_run: False

- seconds_total: 85.92211580276489


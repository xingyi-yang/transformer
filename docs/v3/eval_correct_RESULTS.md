# Evaluation correctness — named MAE and true RMSE

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data validation. No operational recommendation.**

No additional training. Prior artifacts are preserved and **superseded** for the name `rmse_log` (that quantity was `|error|`) and for Chronos-2 fine-tune pooling. Chronos-2 results here are **history-only / univariate**. A covariate-aware comparison has not been launched.

## Definitions

- Point: q0.5 for quantile models (Chronos official mean = median); MEAN for Gaussian.
- **MAE** (`mae_log`): study-balanced mean of `|point − LOGW|` (STUDYID).
- **RMSE** (`rmse_log`): per-study `sqrt(mean e²)`, then the unweighted mean of studies.
- Legacy published `rmse_log` = MAE, often grouped by USUBJID after a STUDYID merge collision.
- Horizon units: calendar DAY and observed future-visit rank. No fabricated days.

## Official-seq frozen holdout (rescored)

| model | MAE | RMSE | legacy animal MAE | pinball | n_studies | n | merge 1-1 |
|---|---|---|---|---|---|---|---|
| spline_target_only | 0.0463 | 0.0588 | 0.0434 | 0.0166 | 2 | 115 | True |
| spline_pooled | 0.0343 | 0.0420 | 0.0335 | 0.0144 | 2 | 115 | True |
| transformer_target_only | 0.0405 | 0.0513 | 0.0409 | 0.0142 | 2 | 115 | True |
| transformer_pooled | 0.0355 | 0.0453 | 0.0358 | 0.0124 | 2 | 115 | True |
| tft_target_only | 0.0455 | 0.0582 | 0.0453 | 0.0159 | 2 | 115 | True |
| tft_pooled | 0.0526 | 0.0648 | 0.0528 | 0.0197 | 2 | 115 | True |
| chronos2_zero_shot | 0.1355 | 0.1510 | 0.1412 | 0.0437 | 2 | 115 | True |
| chronos2_target_only | 0.1271 | 0.1429 | 0.1322 | 0.0422 | 2 | 115 | True |
| chronos2_pooled | 0.1287 | 0.1445 | 0.1340 | 0.0422 | 2 | 115 | True |
| last_value | 0.1140 | 0.1307 | 0.1184 | 0.0570 | 2 | 115 | True |

### Rankings (lower is better)

- MAE: `['spline_pooled', 'transformer_pooled', 'transformer_target_only', 'tft_target_only', 'spline_target_only', 'tft_pooled', 'last_value', 'chronos2_target_only', 'chronos2_pooled', 'chronos2_zero_shot']`
- RMSE: `['spline_pooled', 'transformer_pooled', 'transformer_target_only', 'tft_target_only', 'spline_target_only', 'tft_pooled', 'last_value', 'chronos2_target_only', 'chronos2_pooled', 'chronos2_zero_shot']`
- Legacy animal MAE: `['spline_pooled', 'transformer_pooled', 'transformer_target_only', 'spline_target_only', 'tft_target_only', 'tft_pooled', 'last_value', 'chronos2_target_only', 'chronos2_pooled', 'chronos2_zero_shot']`

### Paired transfer (pooled − target-only; Δ>0 harmful)

| family | metric | target-only | pooled | Δ | sign | published Δ | sign changed |
|---|---|---|---|---|---|---|---|
| spline | mae_log | 0.046286282755388375 | 0.034297829802250976 | -0.011988452953137399 | helpful | -0.009958091034459195 | False |
| spline | rmse_log | 0.05879186977035248 | 0.04195454498695002 | -0.01683732478340246 | helpful | -0.009958091034459195 | False |
| spline | mean_quantile_loss | 0.016618827899219025 | 0.01443948644115832 | -0.0021793414580607053 | helpful | -0.0005955476236056082 | False |
| transformer | mae_log | 0.040535755106582715 | 0.035522599392165205 | -0.00501315571441751 | helpful | -0.005104169385345393 | False |
| transformer | rmse_log | 0.051287312814853686 | 0.04530355609205898 | -0.005983756722794706 | helpful | -0.005104169385345393 | False |
| transformer | mean_quantile_loss | 0.014172849597787463 | 0.01236552710014258 | -0.0018073224976448837 | helpful | -0.0017235042719135028 | False |
| tft | mae_log | 0.045522275217933006 | 0.05257943213856495 | 0.007057156920631941 | harmful | 0.0075244601208263595 | False |
| tft | rmse_log | 0.05816230638802029 | 0.06483274991213397 | 0.006670443524113681 | harmful | 0.0075244601208263595 | False |
| tft | mean_quantile_loss | 0.01594368189446814 | 0.019728218885853972 | 0.0037845369913858326 | harmful | 0.004519617414092587 | False |
| chronos2_finetune | mae_log | 0.1270556677300616 | 0.12866709216199795 | 0.001611424431936348 | harmful | 0.0017175427798567233 | False |
| chronos2_finetune | rmse_log | 0.14293271116741207 | 0.14447782286906424 | 0.0015451117016521687 | harmful | 0.0017175427798567233 | False |
| chronos2_finetune | mean_quantile_loss | 0.04221019210121202 | 0.042167473972915176 | -4.271812829684252e-05 | helpful | -6.532480747031832e-05 | False |

## Seq-diag replicates (rescored; no retraining)

### dataset 0

| model | MAE | RMSE | pinball | status |
|---|---|---|---|---|
| spline_target_only | 0.0319 | 0.0394 | 0.0110 | ok |
| spline_pooled | 0.0406 | 0.0526 | 0.0167 | ok |
| transformer_target_only | 0.0373 | 0.0452 | 0.0126 | ok |
| transformer_pooled | 0.0357 | 0.0451 | 0.0148 | ok |
| tft_target_only | 0.0462 | 0.0565 | 0.0162 | ok |
| tft_pooled | 0.0405 | 0.0508 | 0.0138 | ok |
| chronos2_zero_shot | 0.1391 | 0.1547 | 0.0456 | ok |
| chronos2_target_only | — | — | — | failed: ValueError: The dataset is empty after filtering based on the length of the time |
| chronos2_pooled | — | — | — | failed: ValueError: The dataset is empty after filtering based on the length of the time |
| last_value | 0.1163 | 0.1326 | 0.0581 | ok |

- train eligibility (observed visits ≥ 8): 39/267 eligible; histogram `{3: 26, 4: 191, 5: 1, 6: 2, 7: 8, 8: 39}`; fabricated_history=False

### dataset 1

| model | MAE | RMSE | pinball | status |
|---|---|---|---|---|
| spline_target_only | 0.0297 | 0.0386 | 0.0105 | ok |
| spline_pooled | 0.0289 | 0.0371 | 0.0121 | ok |
| transformer_target_only | 0.0434 | 0.0559 | 0.0153 | ok |
| transformer_pooled | 0.0289 | 0.0365 | 0.0101 | ok |
| tft_target_only | 0.0634 | 0.0740 | 0.0236 | ok |
| tft_pooled | 0.0736 | 0.0854 | 0.0244 | ok |
| chronos2_zero_shot | 0.1104 | 0.1255 | 0.0360 | ok |
| chronos2_target_only | 0.1035 | 0.1184 | 0.0345 | ok |
| chronos2_pooled | 0.1036 | 0.1185 | 0.0345 | ok |
| last_value | 0.1032 | 0.1182 | 0.0516 | ok |

- train eligibility (observed visits ≥ 8): 41/244 eligible; histogram `{2: 2, 3: 11, 4: 185, 7: 5, 8: 41}`; fabricated_history=False

## Chronos-2 fine-tunes (not pooled across incomplete datasets)

- label: `history_only_univariate`
- complete: `1` / `2`
- aggregated_across_incomplete_datasets: `False`
- failures: `[{'dataset_index': 0, 'model': 'chronos2_target_only', 'reason': 'ValueError: The dataset is empty after filtering based on the length of the time series (length >= min_past + prediction_length). Please provide longer time series or reduce `min_past` or `prediction_length`.', 'scored': False}, {'dataset_index': 0, 'model': 'chronos2_pooled', 'reason': 'ValueError: The dataset is empty after filtering based on the length of the time series (length >= min_past + prediction_length). Please provide longer time series or reduce `min_past` or `prediction_length`.', 'scored': False}]`

## Eligibility and horizon units

- eligibility: `{'official_seq_target_only_train': {'min_past': 2, 'prediction_length': 6, 'required_observed_visits': 8, 'n_series': 214, 'n_eligible': 33, 'n_ineligible': 181, 'length_histogram': {1: 3, 2: 1, 3: 13, 4: 157, 5: 1, 7: 6, 8: 33}, 'fabricated_history': False, 'note': 'Chronos-2 fit keeps series with length >= min_past + prediction_length. Ineligible series are reported, not padded.'}, 'official_seq_pooled_train': {'min_past': 2, 'prediction_length': 6, 'required_observed_visits': 8, 'n_series': 313, 'n_eligible': 69, 'n_ineligible': 244, 'length_histogram': {1: 3, 2: 1, 3: 19, 4: 202, 5: 2, 7: 17, 8: 69}, 'fabricated_history': False, 'note': 'Chronos-2 fit keeps series with length >= min_past + prediction_length. Ineligible series are reported, not padded.'}, 'seq_diag_d0_target_only_train': {'min_past': 2, 'prediction_length': 6, 'required_observed_visits': 8, 'n_series': 267, 'n_eligible': 39, 'n_ineligible': 228, 'length_histogram': {3: 26, 4: 191, 5: 1, 6: 2, 7: 8, 8: 39}, 'fabricated_history': False, 'note': 'Chronos-2 fit keeps series with length >= min_past + prediction_length. Ineligible series are reported, not padded.'}, 'seq_diag_d1_target_only_train': {'min_past': 2, 'prediction_length': 6, 'required_observed_visits': 8, 'n_series': 244, 'n_eligible': 41, 'n_ineligible': 203, 'length_histogram': {2: 2, 3: 11, 4: 185, 7: 5, 8: 41}, 'fabricated_history': False, 'note': 'Chronos-2 fit keeps series with length >= min_past + prediction_length. Ineligible series are reported, not padded.'}}`
- horizon_units: `{'calendar_day': 'DAY', 'visit_step': 'observed future-visit rank', 'fabricated_history': False}`

## Guardrails

- real_data_accessed: False
- design_run: False
- training_run: False
- predose_evaluated: False
- covariate_aware_comparison: False


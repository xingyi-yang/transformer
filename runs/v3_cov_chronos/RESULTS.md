# Covariate-aware Chronos-2 — frozen matched-information comparison

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data validation. No operational recommendation.**

PR #12 evaluation correction is the scoring baseline. Prior artifacts are preserved. Chronos-2 history-only remains the comparator. Covariates are origin-only (SEX, STRAIN, SPECIES, ROUTE, PLANNED_DOSE, DAY).

## Defect trace

- Affected published metrics: `['runs/v3_official_seq', 'runs/v3_seq_diag']`
- Rescore still needed: `[]`
- Validation-selection used the defective quantile `rmse_log`: `False`
- Spline wrmse, transformer val_rmse, and TFT val_loss did not use the defective quantile rmse_log. Chronos-2 dataset-0 failed at validation length filtering before eval_loss selection.

## Fresh frozen holdout

- generate_seed / train_seed: `73001` / `83001`
- encoder cutoff: DAY ≤ 14.0 (observed visits only)
- holdout hashes: `{'target_only': {'validation': '7b7f4d224ea8042b', 'test': 'c978494c8169f581', 'all_target': '11ccd2a44fb8550f'}, 'pooled': {'validation': '7b7f4d224ea8042b', 'test': 'c978494c8169f581', 'all_target': '11ccd2a44fb8550f'}}`

| model | MAE | RMSE | pinball | n_studies | merge 1-1 | status |
|---|---|---|---|---|---|---|
| spline_target_only | 0.0382 | 0.0495 | 0.0135 | 2 | True | ok |
| transformer_target_only | 0.0372 | 0.0493 | 0.0133 | 2 | True | ok |
| spline_pooled | 0.0383 | 0.0484 | 0.0201 | 2 | True | ok |
| transformer_pooled | 0.0517 | 0.0693 | 0.0196 | 2 | True | ok |
| tft_target_only | 0.0385 | 0.0475 | 0.0130 | 2 | True | ok |
| tft_pooled | 0.0549 | 0.0680 | 0.0191 | 2 | True | ok |
| last_value | 0.1282 | 0.1458 | 0.0641 | 2 | True | ok |
| chronos2_zero_shot_history | 0.1525 | 0.1693 | 0.0482 | 2 | True | ok |
| chronos2_zero_shot_covariate | 0.1305 | 0.1478 | 0.0483 | 2 | True | ok |
| chronos2_target_only_history | 0.1422 | 0.1592 | 0.0462 | 2 | True | ok |
| chronos2_pooled_history | 0.1438 | 0.1608 | 0.0466 | 2 | True | ok |
| chronos2_target_only_covariate | 0.1159 | 0.1339 | 0.0478 | 2 | True | ok |
| chronos2_pooled_covariate | 0.1195 | 0.1372 | 0.0476 | 2 | True | ok |

### Paired transfer (pooled − target-only; Δ>0 harmful)

| family | metric | target-only | pooled | Δ | sign |
|---|---|---|---|---|---|
| spline | mae_log | 0.038199997901400135 | 0.03827418323797971 | 7.418533657957416e-05 | harmful |
| spline | rmse_log | 0.04952482020646163 | 0.0483757884085042 | -0.001149031797957431 | helpful |
| spline | mean_quantile_loss | 0.013467022103509165 | 0.02008689643894638 | 0.006619874335437214 | harmful |
| transformer | mae_log | 0.03717254939536637 | 0.0516803953947078 | 0.014507845999341427 | harmful |
| transformer | rmse_log | 0.04931831299484542 | 0.06932680313165647 | 0.020008490136811052 | harmful |
| transformer | mean_quantile_loss | 0.013289197306515716 | 0.019576714113939672 | 0.006287516807423956 | harmful |
| tft | mae_log | 0.038532115379403174 | 0.05493977860137429 | 0.016407663221971118 | harmful |
| tft | rmse_log | 0.04753561977495517 | 0.06797486576771226 | 0.020439245992757094 | harmful |
| tft | mean_quantile_loss | 0.012998855649483478 | 0.01914708655828883 | 0.0061482309088053515 | harmful |
| chronos2_finetune_history | mae_log | 0.14219707666433923 | 0.14379974598753115 | 0.0016026693231919187 | harmful |
| chronos2_finetune_history | rmse_log | 0.1592128154450581 | 0.1607721396678054 | 0.0015593242227472937 | harmful |
| chronos2_finetune_history | mean_quantile_loss | 0.04623519327444803 | 0.046566295872897234 | 0.00033110259844920664 | harmful |
| chronos2_finetune_covariate | mae_log | 0.11593680882245075 | 0.11945461329711696 | 0.0035178044746662085 | harmful |
| chronos2_finetune_covariate | rmse_log | 0.13391477251253786 | 0.1372454933265051 | 0.003330720813967236 | harmful |
| chronos2_finetune_covariate | mean_quantile_loss | 0.0478357683924351 | 0.04759970825346709 | -0.00023606013896800698 | helpful |

## Chronos-2 fine-tunes (not pooled when incomplete)

- history-only complete: `True`
- covariate-aware complete: `True`
- aggregated_across_incomplete: `False`
- failures: `[]`

## Guardrails

- real_data_accessed: False
- design_run: False
- predose_evaluated: False
- new_architectures: False


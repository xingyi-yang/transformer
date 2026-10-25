# Primary TFT early-history forecasting

Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.

Forecast windows were taken from observed `DAY` values. Unmeasured days were not created. Long-horizon chunking was not used. Official holdout test errors were not used to select hyperparameters.

## What ran

- output: `runs/tft_primary`
- data: read-only official_seq cells (generate_seed=71001)
- cutoff: DAY ≤ 14.0
- encoder visit cap: 8; prediction-length cap: 6 (no chunking)
- target normalizer: `species_or_global`
- window balance: `study`
- locked after search: lr=0.01, hidden=16, dropout=0.1, max_epochs=16
- eligible official test animals: 58
- correctness: True
- seconds: 400.9756245613098

## Official holdout (study-balanced)

| model | MAE log | RMSE log | mean pinball | cov80 | width80 | crossing | history |
|---|---:|---:|---:|---:|---:|---:|---|
| last_value | 0.1140 | 0.1307 | 0.0570 | 0.0000 | 0.0000 | 0.0 | yes |
| tft_target_only | 0.0499 | 0.0598 | 0.0168 | 0.6453 | 0.1129 | 0.0 | yes |
| tft_pooled | 0.0573 | 0.0746 | 0.0219 | 0.8309 | 0.2537 | 0.0 | yes |
| tft_finetune | 0.0472 | 0.0588 | 0.0165 | 0.8075 | 0.1631 | 0.0 | yes |

## Target-only stability (post-lock seeds)

- seeds: [82001, 82021, 82031]
- MAE mean: 0.0505118806406618; MAE sd: 0.012029655661770508; n=3
- used to retune: False

## Extra-split 33001 (same bank, different studies)

| model | MAE log | RMSE log | mean pinball | cov80 | crossing |
|---|---:|---:|---:|---:|---:|
| tft_target_only | 0.0356 | 0.0456 | 0.0125 | 0.7928 | 0.0 |
| tft_pooled | 0.0402 | 0.0497 | 0.0138 | 0.8034 | 0.0 |
| tft_finetune | 0.0390 | 0.0476 | 0.0133 | 0.7693 | 0.0 |
| locked_tft_pooled_reference | 0.5501 | 1.1689 | 0.2720 | 0.5521 | 0.1839080459770115 |

## Locked pooled-collapse diagnosis

- source: `runs/tft_early_hist_extend/banks/split_33001/extra_split_tft_pooled.csv` (read-only)
- study-balanced MAE on locked extra-split pooled: 0.5501027819408786
- flagged animals: 6
- control breakdown: {'n_control': 14, 'n_treated': 30, 'n_flagged_control': 6, 'n_flagged_treated': 0, 'mean_mae_control': 1.6270121195597191, 'mean_mae_treated': 0.04130967379948225}

- confirmed `global_group_normalizer_mixed_species`: True
- confirmed `predict_time_nan_compound_token`: True
- confirmed `failures_are_control_rats_on_dog_scale`: True
- hypothesis: Unknown COMPOUND_ID embedding plus the control flag failed to apply a RAT offset in globally normalized LOGW, so outputs sat near dog-typical inverse scale (~9.1).
- hypothesis: Extra-split vs official robustness difference is unexplained: official pooled used the same global normalizer and nan token but did not collapse to MAE 0.55.
- not confirmed: Checkpoint selection (last vs best) as the cause: extra-split val_loss was healthy.
- not confirmed: SPECIES encoder mis-mapping: locked encoder classes were DOG=1, RAT=2.

## Scope

Successful training and unit tests do not establish forecasting effectiveness. Numbers above are a synthetic-generator diagnostic on the evaluated settings only. They do not validate historical experimental studies and are not an operational recommendation.


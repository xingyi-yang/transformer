# Sharing replication — locked compute budget

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data validation. No operational recommendation.**

- n_dataset_seeds: `2`; n_train_seeds: `1`
- generate_seeds: `[74001, 75010]`; train_seeds: `[84001]`
- donor policy (not from test): `{'pooled': 3, 'selected_from_test': False, 'selective': 1, 'source': 'training_validation_protocol', 'target_only': 0}`
- Chronos: `history_only_univariate`; TFT/Chronos selective: not_implemented
- estimated_runtime_seconds: `{'lower': 240, 'point': 320, 'upper': 720}`
- seconds_total (actual): `274.0577144622803`

## Replicates

### dataset 0 / train 0

| model | MAE | RMSE | pinball | status |
|---|---|---|---|---|
| chronos2_pooled_history | 0.0831 | 0.1030 | 0.0297 | ok |
| chronos2_selective_history | — | — | — | not_implemented: Chronos-2 has no selective-sharing training path |
| chronos2_selective_regularized_history | — | — | — | not_implemented: Chronos-2 has no selective-sharing training path |
| chronos2_target_only_history | 0.0812 | 0.1010 | 0.0295 | ok |
| chronos2_zero_shot_history | 0.0874 | 0.1073 | 0.0306 | ok |
| last_value | 0.1059 | 0.1250 | 0.0530 | ok |
| spline_pooled | 0.0296 | 0.0385 | 0.0111 | ok |
| spline_selective | 0.0268 | 0.0355 | 0.0096 | ok |
| spline_selective_regularized | 0.0267 | 0.0354 | 0.0101 | ok |
| spline_target_only | 0.0274 | 0.0362 | 0.0100 | ok |
| tft_pooled | 0.1155 | 0.1282 | 0.0346 | ok |
| tft_selective | — | — | — | not_implemented: TFT has no selective-sharing training path |
| tft_selective_regularized | — | — | — | not_implemented: TFT has no selective-sharing training path |
| tft_target_only | 0.0626 | 0.0750 | 0.0231 | ok |
| transformer_pooled | 0.0447 | 0.0572 | 0.0159 | ok |
| transformer_selective | 0.0340 | 0.0444 | 0.0118 | ok |
| transformer_selective_regularized | 0.0366 | 0.0497 | 0.0133 | ok |
| transformer_target_only | 0.0287 | 0.0382 | 0.0102 | ok |

Paired transfer (sharing − target-only; Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| chronos2_finetune_history | mae_log | 0.08115034417639203 | 0.08312810864620904 | 0.0019777644698170116 | harmful |
| chronos2_finetune_history | rmse_log | 0.10095684446087268 | 0.10297728352298266 | 0.0020204390621099827 | harmful |
| chronos2_finetune_history | mean_quantile_loss | 0.02946621408699375 | 0.02968155582171443 | 0.00021534173472068097 | harmful |
| spline_pooled | mae_log | 0.027418126804387516 | 0.029581774844958837 | 0.0021636480405713213 | harmful |
| spline_pooled | rmse_log | 0.03623816900100929 | 0.03846197671025643 | 0.0022238077092471414 | harmful |
| spline_pooled | mean_quantile_loss | 0.010024264515448644 | 0.011138007039942468 | 0.0011137425244938243 | harmful |
| spline_selective | mae_log | 0.027418126804387516 | 0.02681164782693095 | -0.0006064789774565676 | helpful |
| spline_selective | rmse_log | 0.03623816900100929 | 0.03549778403147913 | -0.0007403849695301612 | helpful |
| spline_selective | mean_quantile_loss | 0.010024264515448644 | 0.009630390139144466 | -0.0003938743763041777 | helpful |
| spline_selective_regularized | mae_log | 0.027418126804387516 | 0.02672306117912723 | -0.0006950656252602862 | helpful |
| spline_selective_regularized | rmse_log | 0.03623816900100929 | 0.0353805221277278 | -0.0008576468732814924 | helpful |
| spline_selective_regularized | mean_quantile_loss | 0.010024264515448644 | 0.010118019139484508 | 9.375462403586436e-05 | harmful |
| tft_pooled | mae_log | 0.06261136134469747 | 0.11553434164428611 | 0.05292298029958864 | harmful |
| tft_pooled | rmse_log | 0.07499236393160072 | 0.12816852096952575 | 0.053176157037925026 | harmful |
| tft_pooled | mean_quantile_loss | 0.023149125577318783 | 0.03463052317086688 | 0.011481397593548097 | harmful |
| transformer_pooled | mae_log | 0.028749394955197866 | 0.04473605314911393 | 0.01598665819391606 | harmful |
| transformer_pooled | rmse_log | 0.03821678572300711 | 0.05722711461503102 | 0.019010328892023913 | harmful |
| transformer_pooled | mean_quantile_loss | 0.01019352222109229 | 0.015886311433005604 | 0.0056927892119133135 | harmful |
| transformer_selective | mae_log | 0.028749394955197866 | 0.033991211905910354 | 0.005241816950712488 | harmful |
| transformer_selective | rmse_log | 0.03821678572300711 | 0.04438635869261593 | 0.006169572969608822 | harmful |
| transformer_selective | mean_quantile_loss | 0.01019352222109229 | 0.011840256866516569 | 0.0016467346454242787 | harmful |
| transformer_selective_regularized | mae_log | 0.028749394955197866 | 0.036617736015450666 | 0.0078683410602528 | harmful |
| transformer_selective_regularized | rmse_log | 0.03821678572300711 | 0.0496986338602 | 0.011481848137192892 | harmful |
| transformer_selective_regularized | mean_quantile_loss | 0.01019352222109229 | 0.01325398041391572 | 0.0030604581928234294 | harmful |

### dataset 1 / train 0

| model | MAE | RMSE | pinball | status |
|---|---|---|---|---|
| chronos2_pooled_history | 0.1032 | 0.1197 | 0.0336 | ok |
| chronos2_selective_history | — | — | — | not_implemented: Chronos-2 has no selective-sharing training path |
| chronos2_selective_regularized_history | — | — | — | not_implemented: Chronos-2 has no selective-sharing training path |
| chronos2_target_only_history | 0.1026 | 0.1191 | 0.0335 | ok |
| chronos2_zero_shot_history | 0.1095 | 0.1260 | 0.0351 | ok |
| last_value | 0.1064 | 0.1233 | 0.0532 | ok |
| spline_pooled | 0.0403 | 0.0504 | 0.0223 | ok |
| spline_selective | 0.0333 | 0.0417 | 0.0126 | ok |
| spline_selective_regularized | 0.0310 | 0.0390 | 0.0111 | ok |
| spline_target_only | 0.0300 | 0.0379 | 0.0108 | ok |
| tft_pooled | 0.0576 | 0.0780 | 0.0214 | ok |
| tft_selective | — | — | — | not_implemented: TFT has no selective-sharing training path |
| tft_selective_regularized | — | — | — | not_implemented: TFT has no selective-sharing training path |
| tft_target_only | 0.0566 | 0.0679 | 0.0202 | ok |
| transformer_pooled | 0.0408 | 0.0501 | 0.0139 | ok |
| transformer_selective | 0.0351 | 0.0439 | 0.0121 | ok |
| transformer_selective_regularized | 0.0327 | 0.0409 | 0.0112 | ok |
| transformer_target_only | 0.0371 | 0.0478 | 0.0128 | ok |

Paired transfer (sharing − target-only; Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| chronos2_finetune_history | mae_log | 0.10263138490208294 | 0.10319112388267877 | 0.0005597389805958308 | harmful |
| chronos2_finetune_history | rmse_log | 0.11909690461138074 | 0.11966499097110209 | 0.0005680863597213454 | harmful |
| chronos2_finetune_history | mean_quantile_loss | 0.03349666408211812 | 0.03361443337612441 | 0.00011776929400628827 | harmful |
| spline_pooled | mae_log | 0.029950730564319877 | 0.0402676162850108 | 0.010316885720690921 | harmful |
| spline_pooled | rmse_log | 0.03794246172186816 | 0.0503746434786269 | 0.01243218175675874 | harmful |
| spline_pooled | mean_quantile_loss | 0.010767542163837941 | 0.02229667082085414 | 0.011529128657016198 | harmful |
| spline_selective | mae_log | 0.029950730564319877 | 0.03327711803598056 | 0.003326387471660683 | harmful |
| spline_selective | rmse_log | 0.03794246172186816 | 0.041684239878772116 | 0.0037417781569039563 | harmful |
| spline_selective | mean_quantile_loss | 0.010767542163837941 | 0.012598763115417078 | 0.0018312209515791364 | harmful |
| spline_selective_regularized | mae_log | 0.029950730564319877 | 0.03097840484322354 | 0.0010276742789036633 | harmful |
| spline_selective_regularized | rmse_log | 0.03794246172186816 | 0.038956175964867265 | 0.0010137142429991058 | harmful |
| spline_selective_regularized | mean_quantile_loss | 0.010767542163837941 | 0.011056983420015085 | 0.0002894412561771434 | harmful |
| tft_pooled | mae_log | 0.05661577065526238 | 0.05760632584558068 | 0.0009905551903183 | harmful |
| tft_pooled | rmse_log | 0.06793871899709125 | 0.0779689316744977 | 0.010030212677406447 | harmful |
| tft_pooled | mean_quantile_loss | 0.020158055844454348 | 0.02140704572366226 | 0.0012489898792079138 | harmful |
| transformer_pooled | mae_log | 0.03712396328945415 | 0.040795544618533935 | 0.0036715813290797844 | harmful |
| transformer_pooled | rmse_log | 0.04777820500496796 | 0.05005707533755431 | 0.0022788703325863496 | harmful |
| transformer_pooled | mean_quantile_loss | 0.012835422451412982 | 0.013882306042060321 | 0.0010468835906473385 | harmful |
| transformer_selective | mae_log | 0.03712396328945415 | 0.03514566370360517 | -0.0019782995858489832 | helpful |
| transformer_selective | rmse_log | 0.04777820500496796 | 0.043925985058452785 | -0.0038522199465151757 | helpful |
| transformer_selective | mean_quantile_loss | 0.012835422451412982 | 0.012132701832342819 | -0.0007027206190701636 | helpful |
| transformer_selective_regularized | mae_log | 0.03712396328945415 | 0.03265882609981775 | -0.004465137189636398 | helpful |
| transformer_selective_regularized | rmse_log | 0.04777820500496796 | 0.040891078995009725 | -0.006887126009958236 | helpful |
| transformer_selective_regularized | mean_quantile_loss | 0.012835422451412982 | 0.011229913495746922 | -0.0016055089556660605 | helpful |

## Failures (not aggregated as complete)

- chronos complete pairs: `2` / `2`
- aggregated_across_incomplete: False
- failures: `[]`

## Guardrails

- real_data_accessed: False
- design_run: False
- predose_evaluated: False
- new_architectures: False


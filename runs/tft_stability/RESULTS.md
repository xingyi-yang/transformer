# TFT stability and 80% intervals

Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.

Windows used observed `DAY` values. Official holdout test errors were not used to choose the training strategy or the CQR margin.

## Decision

- default strategy (official validation interval score): **tft_finetune** (mean 0.17804402298454083)
- CQR adopted for target-only: True (validation interval score decreased)
- confirm test studies (seed 34011): ['SYNA-003', 'SYNA-004']
- forbidden test studies: ['SYNA-007', 'SYNA-010', 'SYNA-011']
- seconds: 680.7007193565369

## Official validation (selection bank)

| variant | seed | MAE | pinball | IS80 | cov80 | width | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|
| tft_target_only | 82001 | 0.0460 | 0.0160 | 0.2071 | 0.7178 | 0.1085 | 0.0 |
| tft_pooled | 82001 | 0.0591 | 0.0235 | 0.3281 | 0.8483 | 0.2496 | 0.0 |
| tft_finetune | 82001 | 0.0426 | 0.0149 | 0.1931 | 0.7781 | 0.1308 | 0.0 |
| tft_target_only | 82021 | 0.0442 | 0.0166 | 0.2147 | 0.8110 | 0.1477 | 0.0 |
| tft_pooled | 82021 | 0.0402 | 0.0141 | 0.1772 | 0.7803 | 0.1223 | 0.0 |
| tft_finetune | 82021 | 0.0337 | 0.0115 | 0.1482 | 0.7585 | 0.0974 | 0.0 |
| tft_target_only | 82031 | 0.0418 | 0.0137 | 0.1702 | 0.7623 | 0.1064 | 0.0 |
| tft_pooled | 82031 | 0.0441 | 0.0165 | 0.2152 | 0.6786 | 0.1130 | 0.009345794392523364 |
| tft_finetune | 82031 | 0.0391 | 0.0141 | 0.1928 | 0.7586 | 0.1198 | 0.0 |
| tft_target_only_cqr | 82001 | 0.0460 | 0.0159 | 0.2040 | 0.8131 | 0.1347 | 0.0 |
| tft_target_only_cqr | 82021 | 0.0442 | 0.0166 | 0.2151 | 0.8208 | 0.1515 | 0.0 |
| tft_target_only_cqr | 82031 | 0.0418 | 0.0137 | 0.1686 | 0.8189 | 0.1273 | 0.0 |

## Official test (reference; not used to choose)

| variant | seed | MAE | pinball | IS80 | cov80 | width | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|
| tft_target_only | 82001 | 0.0499 | 0.0168 | 0.2061 | 0.6453 | 0.1129 | 0.0 |
| tft_pooled | 82001 | 0.0573 | 0.0219 | 0.3381 | 0.8309 | 0.2537 | 0.0 |
| tft_finetune | 82001 | 0.0472 | 0.0165 | 0.2051 | 0.8075 | 0.1631 | 0.0 |
| tft_target_only | 82021 | 0.0628 | 0.0219 | 0.2702 | 0.6823 | 0.1586 | 0.0 |
| tft_pooled | 82021 | 0.0566 | 0.0209 | 0.2969 | 0.7104 | 0.1404 | 0.0 |
| tft_finetune | 82021 | 0.0497 | 0.0181 | 0.2588 | 0.7011 | 0.1139 | 0.0 |
| tft_target_only | 82031 | 0.0388 | 0.0137 | 0.1792 | 0.7448 | 0.1113 | 0.0 |
| tft_pooled | 82031 | 0.0500 | 0.0157 | 0.1959 | 0.7544 | 0.1151 | 0.0 |
| tft_finetune | 82031 | 0.0460 | 0.0155 | 0.1899 | 0.7820 | 0.1244 | 0.0 |
| tft_target_only_cqr | 82001 | 0.0499 | 0.0166 | 0.1969 | 0.7776 | 0.1392 | 0.0 |
| tft_target_only_cqr | 82021 | 0.0628 | 0.0219 | 0.2682 | 0.7032 | 0.1624 | 0.0 |
| tft_target_only_cqr | 82031 | 0.0388 | 0.0136 | 0.1770 | 0.8330 | 0.1322 | 0.0 |

## Confirm holdout 34011 (untouched test studies)

| variant | seed | MAE | pinball | IS80 | cov80 | width | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|
| tft_target_only | 82001 | 0.0411 | 0.0143 | 0.1744 | 0.7903 | 0.1402 | 0.0 |
| tft_target_only_cqr | 82001 | 0.0411 | 0.0143 | 0.1772 | 0.8660 | 0.1575 | 0.0 |
| tft_pooled | 82001 | 0.0444 | 0.0156 | 0.2054 | 0.6742 | 0.1181 | 0.044534412955465584 |
| tft_finetune | 82001 | 0.0498 | 0.0180 | 0.2453 | 0.6034 | 0.1073 | 0.0 |
| tft_target_only | 82021 | 0.0360 | 0.0129 | 0.1722 | 0.8217 | 0.1320 | 0.0 |
| tft_target_only_cqr | 82021 | 0.0360 | 0.0129 | 0.1725 | 0.8217 | 0.1340 | 0.0 |
| tft_pooled | 82021 | 0.0441 | 0.0155 | 0.2071 | 0.6371 | 0.0940 | 0.0 |
| tft_finetune | 82021 | 0.0410 | 0.0145 | 0.1937 | 0.6728 | 0.0984 | 0.004048582995951417 |
| tft_target_only | 82031 | 0.0434 | 0.0151 | 0.1902 | 0.8112 | 0.1512 | 0.0 |
| tft_target_only_cqr | 82031 | 0.0434 | 0.0151 | 0.1913 | 0.8360 | 0.1614 | 0.0 |
| tft_pooled | 82031 | 0.0822 | 0.0322 | 0.4434 | 0.7320 | 0.1644 | 0.12955465587044535 |
| tft_finetune | 82031 | 0.0920 | 0.0379 | 0.5734 | 0.6103 | 0.1120 | 0.17408906882591094 |

## Recorded interpretation

On official validation, **fine-tune** had the lowest mean interval score (0.178 vs target-only 0.197, CQR 0.196, pooled 0.240) and is the predeclared default. Official test is a reference only: fine-tune MAE was the most seed-stable (mean 0.0476, sd 0.0019 vs target-only 0.0505 / 0.0120). CQR, fit on validation only, was adopted because mean validation interval score fell (0.19735 → 0.19590). Official target-only undercoverage is study- and lead-specific (SYNA-011 0.50/0.60/0.72 vs SYNA-010 0.79/0.77/0.77; longer leads worse; misses mostly above). CQR raised official-test coverage (0.691 → 0.771) without moving q0.5.

The untouched confirm holdout (seed 34011; test SYNA-003/004) did **not** support that default. Target-only was better and more stable (MAE 0.0402 sd 0.0038, coverage 0.808, IS 0.179, crossings 0) than fine-tune (MAE 0.0609 sd 0.0273, coverage 0.629, IS 0.337, crossings 0.059) or pooled. Seed 82031 pooled/fine-tune degraded (MAE 0.082/0.092, crossings 0.130/0.174). Confirm CQR slightly overcovered and did not lower interval score.

So: official-holdout point-forecast stability improved for fine-tune; official-holdout interval calibration improved for target-only + CQR; transfer did not improve on the new holdout. Extra-split 33001 remains a read-only reference (target-only MAE 0.0356 / cov 0.793; pooled 0.0402 / 0.803; fine-tune 0.0390 / 0.769). The validation-selected default is therefore not confirmed. No retuning after seeing confirmation scores.

## Scope

Successful training and unit tests do not establish forecasting effectiveness. Numbers above are a synthetic-generator diagnostic on the evaluated settings only.


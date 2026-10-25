# TFT early-history synthetic benchmark

Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.

Forecast windows were taken from observed `DAY` values. Unmeasured days were not created. Long-horizon chunking was not used.

## What ran

- output: `runs/tft_early_hist_bench`
- data: read-only official_seq cells (generate_seed=71001)
- TFT train_seed: 82001
- cutoff: DAY ≤ 14.0
- encoder visit cap: 8; prediction-length cap: 6 (no chunking)
- eligible test animals: 58
- excluded: 0 {}
- correctness: True
- seconds: 52.35309958457947

## Comparison (study-balanced, identical animal/day pairs)

| model | MAE log | RMSE log | mean pinball | cov80 | width80 | rel gram | crossing | history |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| last_value | 0.1140 | 0.1307 | 0.0570 | 0.0000 | 0.0000 | 0.1058 | 0.0 | yes |
| transformer_target_only | 0.0438 | 0.0557 | 0.0153 | 0.6869 | 0.1132 | 0.0427 | 0.0 | yes |
| tft_target_only | 0.0489 | 0.0610 | 0.0174 | 0.7726 | 0.1502 | 0.0484 | 0.0 | yes |
| tft_pooled | 0.0583 | 0.0674 | 0.0193 | 0.6773 | 0.1434 | 0.0576 | 0.33043478260869563 | yes |

## Data (observed, not invented)

Test studies `SYNA-010` and `SYNA-011` (RAT, 58 animals, 228 post-baseline rows). Observed `DAY` values on the bank are `{3, 7, 10, 14, 17, 21, 24, 28}`. Eligible examples: 58 animals, 115 target rows. No animal was excluded. Typical encoder is days 7 and 14; typical targets are 21 and 28, or 17/21/24/28 when twice-weekly visits exist. Missing visits stay missing.

FeatureSpec and compound references come from the official_seq prepare step (training studies only). TFT was retrained here (`train_seed` 82001). The custom transformer checkpoint was loaded from the same target-only cell and not refit.

## Correctness checks (4 representative animals)

Independent re-derivation of encoder/target `DAY`s matched the stored examples. `predict_quantiles()` returned those animal/day pairs. Single-horizon q0.5 matched the first multi-horizon step from the same history. Dummy decoder `LOGW=0` did not change quantiles. `LOGW` equals `ln(WEIGHT_G)`. Quantile order held on the checked target-only rows.

## Window construction

Target-only training built 1327 PF sliding windows across 6 studies. Row-level study-balanced `WEIGHT` does not equalize window counts (window-share range 0.30). That imbalance is recorded, not corrected.

## What the scores establish here

On this one Generator A holdout, last-value is worse than the three model-based forecasts on MAE/RMSE. The custom transformer (Gaussian early-history) has the lowest point error. Target-only TFT is close to it, with near-nominal 80% interval coverage and no quantile crossings. Pooled TFT is worse than target-only TFT on MAE/RMSE and has a 33% quantile-crossing rate, so adding donor studies did not help this compact fit.

These signs are not a sharing verdict and not a reason to prefer TFT. The transformer interval is a Gaussian residual layer, not a quantile head. Last-value intervals are degenerate (width 0) by construction.

## Untested on real SEND studies

No historical experimental body-weight studies were opened. Species transfer, assay error, protocol heterogeneity, longer irregular calendars, and any hyperparameter change remain untested. Successful training and passing unit tests are not forecasting effectiveness.


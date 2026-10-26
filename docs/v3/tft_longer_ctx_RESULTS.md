# Longer-context TFT comparison

Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.

Observed pretreatment / baseline visits (`DAY` −7 and 0 in Generator A) are reattached as encoder context. The cutoff stays `DAY ≤ 14`. Targets are the same later observed visits. Recovery rows are not used. Unmeasured days are not created. Locked TFT knobs are unchanged. Target-only remains the development reference.

## What ran

- output: `runs/tft_longer_ctx`
- source cells (read-only): `runs/tft_larger_bank/datasets/d0/trains/t0/cells/d0_t0_matched_target_only`
- generate_seed=81011; train_seed=83011
- cutoff: DAY ≤ 14.0
- encoder visit cap: 8
- locked knobs: lr=0.01, hidden=16, dropout=0.1, max_epochs=16
- search: False
- eligible animals: 256
- development reference: `tft_target_only`
- fine-tune promoted: False
- correctness: True
- seconds: 533.2915880680084

## Encoder length

- treatment-only (larger bank): mean 2.75 visits; counts {'2': 155, '3': 10, '4': 91}; days [3.0, 7.0, 10.0, 14.0]
- with pretreatment: mean 4.75 visits; counts {'4': 155, '5': 10, '6': 91}; days [-7.0, 0.0, 3.0, 7.0, 10.0, 14.0]

## Longer-context holdout (study-balanced)

| model | MAE log | RMSE log | mean pinball | cov80 | width80 | IS80 | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|
| last_value | 0.0908 | 0.1068 | 0.0454 | 0.0020 | 0.0000 | 0.9076 | 0.0 |
| tft_target_only (reference) | 0.0496 | 0.0610 | 0.0162 | 0.7813 | 0.1370 | 0.1974 | 0.0 |
| tft_pooled | 0.0489 | 0.0622 | 0.0171 | 0.8157 | 0.1684 | 0.2212 | 0.0 |
| tft_finetune | 0.0375 | 0.0474 | 0.0133 | 0.8103 | 0.1280 | 0.1758 | 0.0 |

- lowest study-balanced MAE: `tft_finetune`
- lowest MAE replaces reference: False

## Paired vs treatment-only encoder (same targets)

- shared pairs: 708; same_target_pairs=True

| model | MAE short | MAE long | Δ long−short |
|---|---:|---:|---:|
| last_value | 0.09075799857309412 | 0.09075799857309412 | 0.0 |
| tft_target_only | 0.03642823054468265 | 0.04961044780397316 | 0.013182217259290509 |
| tft_pooled | 0.05129906048151501 | 0.04887833122443339 | -0.0024207292570816163 |
| tft_finetune | 0.04716067118895563 | 0.03750763800098232 | -0.009653033187973309 |

## Scope

This is a synthetic-generator diagnostic. Including predose visits does not qualify the adapter on real SEND studies and is not an operational recommendation. Fine-tune stays experimental.


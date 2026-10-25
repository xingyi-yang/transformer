# TFT study-held-out evaluation

Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.

Target-only TFT is the development reference. Fine-tune and CQR are experimental.
The PR #22 fine-tune default failed confirmation and was not retuned.
Confirmation 34011 is not described as untouched: SYNA-007/010/011 were in its train set.

## Decision rules

- validation-preferred (IS80, not a new default): **tft_finetune**
- any experimental variant consistently beats target-only (MAE): True
- seconds: 2848.6159081459045

## Folds (split seed 35021)

- fold 0: test ['SYNA-001', 'SYNA-003']; val ['SYNA-005', 'SYNA-007']; cal ['SYNA-002', 'SYNA-010']; train ['SYNA-004', 'SYNA-006', 'SYNA-008', 'SYNA-009', 'SYNA-011', 'SYNA-012']
- fold 1: test ['SYNA-007', 'SYNA-011']; val ['SYNA-001', 'SYNA-005']; cal ['SYNA-006', 'SYNA-010']; train ['SYNA-002', 'SYNA-003', 'SYNA-004', 'SYNA-008', 'SYNA-009', 'SYNA-012']
- fold 2: test ['SYNA-004', 'SYNA-005']; val ['SYNA-007', 'SYNA-009']; cal ['SYNA-008', 'SYNA-010']; train ['SYNA-001', 'SYNA-002', 'SYNA-003', 'SYNA-006', 'SYNA-011', 'SYNA-012']
- fold 3: test ['SYNA-006', 'SYNA-009']; val ['SYNA-002', 'SYNA-005']; cal ['SYNA-004', 'SYNA-012']; train ['SYNA-001', 'SYNA-003', 'SYNA-007', 'SYNA-008', 'SYNA-010', 'SYNA-011']
- fold 4: test ['SYNA-010', 'SYNA-012']; val ['SYNA-006', 'SYNA-008']; cal ['SYNA-003', 'SYNA-005']; train ['SYNA-001', 'SYNA-002', 'SYNA-004', 'SYNA-007', 'SYNA-009', 'SYNA-011']
- fold 5: test ['SYNA-002', 'SYNA-008']; val ['SYNA-001', 'SYNA-005']; cal ['SYNA-003', 'SYNA-010']; train ['SYNA-004', 'SYNA-006', 'SYNA-007', 'SYNA-009', 'SYNA-011', 'SYNA-012']

## Test scores (raw quantiles except CQR rows)

| fold | variant | seed | MAE | pinball | IS80 | cov80 | width | crossing | table |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | tft_target_only | 82001 | 0.0456 | 0.0162 | 0.2195 | 0.6495 | 0.1182 | 0.0 | raw |
| 0 | tft_pooled | 82001 | 0.0484 | 0.0160 | 0.1902 | 0.8884 | 0.1822 | 0.0 | raw |
| 0 | tft_finetune | 82001 | 0.0338 | 0.0117 | 0.1459 | 0.7467 | 0.1061 | 0.0 | raw |
| 0 | tft_target_only_cqr | 82001 | 0.0456 | 0.0161 | 0.2130 | 0.7033 | 0.1284 | 0.0 | cqr_shifted_endpoints |
| 0 | tft_target_only | 82021 | 0.0371 | 0.0142 | 0.2104 | 0.6584 | 0.1134 | 0.2561576354679803 | raw |
| 0 | tft_pooled | 82021 | 0.0336 | 0.0128 | 0.1679 | 0.8618 | 0.1450 | 0.0 | raw |
| 0 | tft_finetune | 82021 | 0.0428 | 0.0148 | 0.1854 | 0.6825 | 0.1045 | 0.0 | raw |
| 0 | tft_target_only_cqr | 82021 | 0.0371 | 0.0140 | 0.2005 | 0.7532 | 0.1343 | 0.0 | cqr_shifted_endpoints |
| 0 | tft_target_only | 82031 | 0.0379 | 0.0132 | 0.1628 | 0.7485 | 0.1056 | 0.0 | raw |
| 0 | tft_pooled | 82031 | 0.0422 | 0.0145 | 0.1746 | 0.8263 | 0.1301 | 0.0 | raw |
| 0 | tft_finetune | 82031 | 0.0432 | 0.0146 | 0.1862 | 0.6802 | 0.1075 | 0.0 | raw |
| 0 | tft_target_only_cqr | 82031 | 0.0379 | 0.0133 | 0.1680 | 0.8916 | 0.1484 | 0.0 | cqr_shifted_endpoints |
| 1 | tft_target_only | 82001 | 0.0436 | 0.0153 | 0.1878 | 0.6806 | 0.1050 | 0.0 | raw |
| 1 | tft_pooled | 82001 | 0.0522 | 0.0179 | 0.2226 | 0.7266 | 0.1237 | 0.29310344827586204 | raw |
| 1 | tft_finetune | 82001 | 0.0469 | 0.0168 | 0.2172 | 0.6944 | 0.1264 | 0.0 | raw |
| 1 | tft_target_only_cqr | 82001 | 0.0436 | 0.0152 | 0.1813 | 0.7102 | 0.1173 | 0.0 | cqr_shifted_endpoints |
| 1 | tft_target_only | 82021 | 0.0405 | 0.0141 | 0.1819 | 0.7494 | 0.1247 | 0.0 | raw |
| 1 | tft_pooled | 82021 | 0.0397 | 0.0143 | 0.1933 | 0.8087 | 0.1491 | 0.0 | raw |
| 1 | tft_finetune | 82021 | 0.0430 | 0.0145 | 0.1768 | 0.8138 | 0.1370 | 0.0 | raw |
| 1 | tft_target_only_cqr | 82021 | 0.0405 | 0.0141 | 0.1818 | 0.7607 | 0.1251 | 0.0 | cqr_shifted_endpoints |
| 1 | tft_target_only | 82031 | 0.0564 | 0.0198 | 0.2624 | 0.6319 | 0.1379 | 0.0 | raw |
| 1 | tft_pooled | 82031 | 0.0413 | 0.0143 | 0.1834 | 0.7172 | 0.1022 | 0.0 | raw |
| 1 | tft_finetune | 82031 | 0.0380 | 0.0129 | 0.1604 | 0.7841 | 0.1133 | 0.0 | raw |
| 1 | tft_target_only_cqr | 82031 | 0.0564 | 0.0195 | 0.2484 | 0.8731 | 0.2092 | 0.0 | cqr_shifted_endpoints |
| 2 | tft_target_only | 82001 | 0.0550 | 0.0215 | 0.3242 | 0.6851 | 0.1137 | 0.0 | raw |
| 2 | tft_pooled | 82001 | 0.0427 | 0.0167 | 0.2381 | 0.5778 | 0.0968 | 0.03636363636363636 | raw |
| 2 | tft_finetune | 82001 | 0.0442 | 0.0160 | 0.2086 | 0.6782 | 0.1025 | 0.0 | raw |
| 2 | tft_target_only_cqr | 82001 | 0.0550 | 0.0213 | 0.3150 | 0.7809 | 0.1467 | 0.0 | cqr_shifted_endpoints |
| 2 | tft_target_only | 82021 | 0.0534 | 0.0194 | 0.2724 | 0.6595 | 0.1296 | 0.0 | raw |
| 2 | tft_pooled | 82021 | 0.0474 | 0.0171 | 0.2376 | 0.6557 | 0.0984 | 0.0 | raw |
| 2 | tft_finetune | 82021 | 0.0430 | 0.0162 | 0.2301 | 0.7281 | 0.1120 | 0.0 | raw |
| 2 | tft_target_only_cqr | 82021 | 0.0534 | 0.0191 | 0.2583 | 0.7865 | 0.1693 | 0.0 | cqr_shifted_endpoints |
| 2 | tft_target_only | 82031 | 0.0587 | 0.0203 | 0.2671 | 0.5652 | 0.1160 | 0.07727272727272727 | raw |
| 2 | tft_pooled | 82031 | 0.0972 | 0.0369 | 0.5569 | 0.5044 | 0.1442 | 0.0 | raw |
| 2 | tft_finetune | 82031 | 0.0507 | 0.0181 | 0.2466 | 0.6189 | 0.1180 | 0.0 | raw |
| 2 | tft_target_only_cqr | 82031 | 0.0587 | 0.0198 | 0.2424 | 0.7787 | 0.1633 | 0.0 | cqr_shifted_endpoints |
| 3 | tft_target_only | 82001 | 0.0577 | 0.0213 | 0.2812 | 0.5631 | 0.1372 | 0.0 | raw |
| 3 | tft_pooled | 82001 | 0.0590 | 0.0217 | 0.3060 | 0.4318 | 0.0937 | 0.0 | raw |
| 3 | tft_finetune | 82001 | 0.0463 | 0.0152 | 0.1755 | 0.7272 | 0.1202 | 0.014925373134328358 | raw |
| 3 | tft_target_only_cqr | 82001 | 0.0577 | 0.0208 | 0.2580 | 0.6374 | 0.1600 | 0.0 | cqr_shifted_endpoints |
| 3 | tft_target_only | 82021 | 0.0696 | 0.0248 | 0.3416 | 0.5314 | 0.1168 | 0.0 | raw |
| 3 | tft_pooled | 82021 | 0.0444 | 0.0151 | 0.1800 | 0.7734 | 0.1327 | 0.0 | raw |
| 3 | tft_finetune | 82021 | 0.0498 | 0.0176 | 0.2309 | 0.5468 | 0.1132 | 0.0 | raw |
| 3 | tft_target_only_cqr | 82021 | 0.0696 | 0.0237 | 0.2859 | 0.6639 | 0.1695 | 0.0 | cqr_shifted_endpoints |
| 3 | tft_target_only | 82031 | 0.0695 | 0.0232 | 0.2870 | 0.5777 | 0.1354 | 0.0 | raw |
| 3 | tft_pooled | 82031 | 0.0660 | 0.0222 | 0.2711 | 0.5812 | 0.1291 | 0.0 | raw |
| 3 | tft_finetune | 82031 | 0.0488 | 0.0180 | 0.2502 | 0.4970 | 0.1051 | 0.0 | raw |
| 3 | tft_target_only_cqr | 82031 | 0.0695 | 0.0232 | 0.2856 | 0.5777 | 0.1367 | 0.0 | cqr_shifted_endpoints |
| 4 | tft_target_only | 82001 | 0.0444 | 0.0158 | 0.2067 | 0.7151 | 0.1284 | 0.0 | raw |
| 4 | tft_pooled | 82001 | 0.0724 | 0.0261 | 0.3773 | 0.4709 | 0.1182 | 0.031007751937984496 | raw |
| 4 | tft_finetune | 82001 | 0.0717 | 0.0252 | 0.3515 | 0.6512 | 0.1452 | 0.0 | raw |
| 4 | tft_target_only_cqr | 82001 | 0.0444 | 0.0160 | 0.2174 | 0.9128 | 0.1899 | 0.0 | cqr_shifted_endpoints |
| 4 | tft_target_only | 82021 | 0.0506 | 0.0183 | 0.2600 | 0.6570 | 0.1324 | 0.10852713178294573 | raw |
| 4 | tft_pooled | 82021 | 0.0359 | 0.0130 | 0.1827 | 0.8547 | 0.1373 | 0.0 | raw |
| 4 | tft_finetune | 82021 | 0.0401 | 0.0146 | 0.1901 | 0.7791 | 0.1180 | 0.0 | raw |
| 4 | tft_target_only_cqr | 82021 | 0.0506 | 0.0181 | 0.2520 | 0.8953 | 0.2198 | 0.0 | cqr_shifted_endpoints |
| 4 | tft_target_only | 82031 | 0.0423 | 0.0151 | 0.1967 | 0.7384 | 0.1384 | 0.0 | raw |
| 4 | tft_pooled | 82031 | 0.4007 | 0.1910 | 3.6166 | 0.4942 | 0.1057 | 0.5581395348837209 | raw |
| 4 | tft_finetune | 82031 | 0.5920 | 0.2875 | 5.5634 | 0.3837 | 0.0801 | 0.5813953488372093 | raw |
| 4 | tft_target_only_cqr | 82031 | 0.0423 | 0.0151 | 0.1981 | 0.8721 | 0.1708 | 0.0 | cqr_shifted_endpoints |
| 5 | tft_target_only | 82001 | 0.0366 | 0.0124 | 0.1548 | 0.7878 | 0.1093 | 0.007246376811594203 | raw |
| 5 | tft_pooled | 82001 | 0.0459 | 0.0161 | 0.1850 | 0.7371 | 0.1171 | 0.0 | raw |
| 5 | tft_finetune | 82001 | 0.0423 | 0.0141 | 0.1745 | 0.7517 | 0.1149 | 0.0 | raw |
| 5 | tft_target_only_cqr | 82001 | 0.0366 | 0.0129 | 0.1762 | 0.9429 | 0.1601 | 0.0 | cqr_shifted_endpoints |
| 5 | tft_target_only | 82021 | 0.0422 | 0.0151 | 0.2060 | 0.7320 | 0.1259 | 0.0 | raw |
| 5 | tft_pooled | 82021 | 0.0458 | 0.0171 | 0.2102 | 0.7371 | 0.1277 | 0.0 | raw |
| 5 | tft_finetune | 82021 | 0.0371 | 0.0129 | 0.1698 | 0.7657 | 0.1066 | 0.0 | raw |
| 5 | tft_target_only_cqr | 82021 | 0.0422 | 0.0152 | 0.2087 | 0.6895 | 0.1202 | 0.0 | cqr_shifted_endpoints |
| 5 | tft_target_only | 82031 | 0.0484 | 0.0165 | 0.2004 | 0.7475 | 0.1324 | 0.0 | raw |
| 5 | tft_pooled | 82031 | 0.0435 | 0.0154 | 0.2260 | 0.7203 | 0.1442 | 0.0 | raw |
| 5 | tft_finetune | 82031 | 0.0318 | 0.0109 | 0.1313 | 0.7980 | 0.0998 | 0.0 | raw |
| 5 | tft_target_only_cqr | 82031 | 0.0484 | 0.0165 | 0.1998 | 0.8162 | 0.1536 | 0.0 | cqr_shifted_endpoints |

## Recorded interpretation

The PR #22 fine-tune default failed confirmation and was not retuned. Confirmation 34011 is not untouched: SYNA-007/010/011 were in its train set.

Official train is 214 RAT animals and 211 TFT series. Three animals have a single DAY=7 visit and fail `min_encoder_length + 1`. New histories store `epochs_completed` from logged `train_loss` (16 for a 16-epoch budget). Older files stored Lightning `current_epoch + 1` (17).

CQR pinball uses shifted q0.1/q0.9 and is not the TFT QuantileLoss. MAE (q0.5) matches the raw table. Margins were fit on the calibration split only.

### Does any experimental strategy improve consistently?

The predeclared vote (beat target-only test MAE on ≥5/6 folds and ≥10/12 studies) is met by **fine-tune** (5/6 folds, 10/12 studies). It is **not** a reason to replace target-only:

- Seed-mean test MAE is worse for fine-tune (0.075, sd 0.129) than target-only (0.049, sd 0.010) because fold 4 seed 82031 collapsed (MAE 0.592, crossings 0.58) on SYNA-012 (study MAE 0.42 vs 0.039).
- Fold 4 also collapsed pooled (MAE 0.401). Those test studies are SYNA-010/012 (official test / official validation).
- Several fine-tune “wins” are <0.01 MAE. SYNA-003 does not improve.
- Pooled is not consistent (2/6 folds, 6/12 studies).
- CQR cannot beat target-only on MAE (median unchanged). It raised coverage on many folds and zeroed crossings, but interval score improved on only 4/6 folds and 8/12 studies.

Validation again preferred fine-tune on interval score (0.2102 vs target-only 0.2103) by a negligible margin. Target-only remains the development reference. Fine-tune and CQR stay experimental.

## Scope

Successful training and unit tests do not establish forecasting effectiveness. CQR pinball uses shifted endpoints and is not the TFT QuantileLoss.


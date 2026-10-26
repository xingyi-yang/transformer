# Larger-bank TFT model comparison

Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.

This is a **new** Generator A bank (`generate_seed=81011`), not the official 12 RAT + 3 DOG holdout (`generate_seed=71001`). Forecast windows use observed `DAY` values. Unmeasured days were not created. Long-horizon chunking was not used. Official and new test errors were not used to select hyperparameters. Target-only remains the development reference. Fine-tune is experimental and is not promoted even if it has lower MAE here.

## What ran

- output: `runs/tft_larger_bank`
- generator: A; n_studies=36 RAT + 6 DOG; n_compounds=8; n_per_sex=[3, 4]
- generate_seed=81011; train_seed=83011
- donors: 6 (pooled only)
- cutoff: DAY ≤ 14.0
- encoder visit cap: 8; prediction-length cap: 6 (no chunking)
- locked knobs: lr=0.01, hidden=16, dropout=0.1, max_epochs=16, window_balance=study, target_normalizer=species_or_global
- search: False
- test studies: ['SYNA-001', 'SYNA-008', 'SYNA-015', 'SYNA-017', 'SYNA-021', 'SYNA-023', 'SYNA-034']
- validation studies: ['SYNA-004', 'SYNA-007', 'SYNA-012', 'SYNA-013', 'SYNA-016', 'SYNA-020', 'SYNA-028']
- target train studies: ['SYNA-002', 'SYNA-003', 'SYNA-005', 'SYNA-006', 'SYNA-009', 'SYNA-010', 'SYNA-011', 'SYNA-014', 'SYNA-018', 'SYNA-019', 'SYNA-022', 'SYNA-024', 'SYNA-025', 'SYNA-026', 'SYNA-027', 'SYNA-029', 'SYNA-030', 'SYNA-031', 'SYNA-033', 'SYNA-035', 'SYNA-036']
- eligible test animals: 256
- development reference: `tft_target_only`
- fine-tune promoted: False
- correctness: True
- seconds: 442.29331254959106

## Larger bank (study-balanced)

| model | MAE log | RMSE log | mean pinball | cov80 | width80 | IS80 | crossing | history |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| last_value | 0.0908 | 0.1068 | 0.0454 | 0.0020 | 0.0000 | 0.9076 | 0.0 | yes |
| tft_target_only (reference) | 0.0364 | 0.0462 | 0.0131 | 0.7995 | 0.1198 | 0.1730 | 0.0 | yes |
| tft_pooled | 0.0513 | 0.0737 | 0.0197 | 0.7172 | 0.1181 | 0.2873 | 0.012711864406779662 | yes |
| tft_finetune | 0.0472 | 0.0653 | 0.0178 | 0.7105 | 0.1123 | 0.2541 | 0.002824858757062147 | yes |

- lowest study-balanced MAE: `tft_target_only`
- lowest MAE replaces reference: False

## Official 12+3 bank (read-only reference, generate_seed=71001)

Different studies and a different generate seed. Same locked knobs. Not a paired animal comparison.

| model | MAE log | RMSE log | mean pinball | cov80 | width80 | n_scored |
|---|---:|---:|---:|---:|---:|---:|
| last_value | 0.1140 | 0.1307 | 0.0570 | 0.0000 | 0.0000 | 115 |
| tft_target_only | 0.0499 | 0.0598 | 0.0168 | 0.6453 | 0.1129 | 115 |
| tft_pooled | 0.0573 | 0.0746 | 0.0219 | 0.8309 | 0.2537 | 115 |
| tft_finetune | 0.0472 | 0.0588 | 0.0165 | 0.8075 | 0.1631 | 115 |

- official source: `runs/tft_primary/SUMMARY.json` (read_only=True)
- used to retune: False

## Scope

Successful training and unit tests do not establish forecasting effectiveness. Numbers above are a synthetic-generator diagnostic on the evaluated settings only. They do not validate historical experimental studies and are not an operational recommendation. Sharing and the transformer are not required to win. Fine-tune and CQR stay experimental.


# Forecast method audit

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation. Target-only TFT remains the development
reference. Locked run directories were not rewritten. No model was retrained.

Code: `src/send_bw/forecast_method_audit.py`. Artifacts:
`runs/forecast_method_audit/SUMMARY.json`,
`runs/forecast_method_audit/cqr_romano_rescored.json`.
Classified findings: [Forecast method audit canvas](https://cursor.com/canvas/store-d8a3e8c2-77c7-40bf-89fb-f761966eb659/662df9d3-4043-4885-b5a2-371a3619885c).

## Classification

### Confirmed bugs

1. **CQR order statistic (scoring, interval only).**
   `conformal_margin` used `np.quantile(k/n, method="higher")`. NumPy
   `higher` indexes `q(n−1)` and returns the next rank (`sorted[k]` when
   `k = ceil((n+1)(1−α))`). Romano wants `E_{(k)} = sorted[k-1]`.
   `method="inverted_cdf"` matches Romano. Official validation `n=214`,
   `k=172`. Locked CQR files were not overwritten. The helper now uses
   the k-th order statistic. MAE / q0.5 unchanged.

2. **CatBoost class docstring** said MultiQuantile; fit trains one
   `Quantile:alpha=q` head per level. Scoring was already correct.

3. **`docs/limitations.md`** still said Chronos-2 was unimplemented.
   Official_seq ran `HISTORY_ONLY` Chronos-2.

4. **PR #24 loss-scale note** said QuantileLoss is on the normalized
   target. One locked batch shows decoder `y` and `TFT.forward`
   predictions on raw `LOGW`. Corrected.

No leakage or point-forecast bug required a refit.

### Design limitations

- D1–D3 are curated synthetic descriptors, not SEND fields. Comparable
  columns must be supplied at prediction time.
- Hidden `A` / `ED50` / `tau_on` / `tau_off` never enter production features.
- Weight responds to actual dose deviations; models see planned regimen only.
- Missingness and death are MCAR-ish (constant p), not informative dropout.
- Official growth family is one saturating exponential (`tau_g=40`).
- Spline `sigma_m` sits at ~1e-4 (optimizer floor), not generator 0.010.
- Official Chronos-2 used visit-rank `LOGW`. Elapsed `DAY` is an adapter
  omission. `COVARIATE_AWARE` can pass `DAY` but still indexes by visit.
- Official test is a study holdout. SYNA-010/011 reuse train compounds
  CMP03/CMP06.
- CQR pair residuals are `(USUBJID, DAY)`, not study-balanced.

### Unresolved

- Whether a real SEND warehouse would have D1–D3-like descriptors.
- Whether a Chronos time index that used elapsed days (not just a
  covariate) would change official_seq errors — not measured.
- Study-level finite-sample CQR coverage is not guaranteed.

## Generator

D1–D3 are **production** (`FEATURE_REGISTRY`) and live in
`metadata/compound_reference.csv`. Hidden compound truth is only in
`truth/params.json`. `FeatureSpec.animal_features` raises
`UnsupportedContextError` if an active arm is missing a descriptor.

Official bank: Generator A saturating exponential, shared `tau_g=40`.
Generator B Gompertz is not this bank.

Trajectory uses the **actual** regimen after dose deviations
(`generate_dataset`: `gen.effect(grid, act, ct)`). 9 official-bank
animals have `DOSE_DEVIATION`; planned-vs-actual effect at day 28 is
nonzero (mean |Δ| ≈ 0.016). Production features use `PLANNED_DOSE`
(`ACTUAL_DOSE` is `truth_only`).

Death and missed visits are iid (`rng.random() < p_*`). Official-bank
death rate 0.0179 vs configured 0.015. Correlations vs treated / dose /
baseline LOGW are 0.087 / 0.101 / −0.022 (Monte Carlo noise, not MAR).

## Other methods

Spline mean target is `Y`; `predict_mean` reconstructs
`LOGW = baseline + Ŷ`. Covariance is L-BFGS-B on train OOF residuals
(n_obs=983, 6 studies), not generator truth:

| parameter | fitted | Generator A |
|---|---:|---:|
| sigma_p | 0.0271 | 0.022 |
| ell | 4.19 | 6.0 |
| sigma_m | 1.00e-4 | 0.010 |

Transformer uses that fitted covariance. Feature scales are train-only.
Official cell `compound_ids` = CMP03–06. Evaluation studies SYNA-010/011
were not in the checkpoint.

CatBoost features are encoder summaries plus known-future calendar
(`target_day`, elapsed). `y_logw` is the label. No future LOGW.

## Chronos-2

Official_seq omitted `covariate_mode` and therefore used
`history_only_univariate`: a raw LOGW ndarray, visit rank only. Two
series with the same visit count and different calendar spacing are
indistinguishable. That is an adapter choice. Chronos-2 can take `DAY`
as a covariate (`covariate_aware_origin`); the sequence index is still
unit steps.

## CQR Romano rescore (saved official-validation predictions)

| seed | stored margin | Romano margin | locked test IS | Romano test IS | MAE (unchanged) |
|---|---:|---:|---:|---:|---:|
| 82001 | 0.013130 | 0.012656 | 0.19692 | 0.19703 | 0.04993 |
| 82021 | 0.001902 | 0.000018 | 0.26822 | 0.27019 | 0.06282 |
| 82031 | 0.010473 | 0.010304 | 0.17698 | 0.17693 | 0.03879 |

Metric movement is small. Locked `runs/tft_stability/` scores were not
rewritten. Corrected tables: `cqr_romano_rescored.json`.

## TFT one-batch replay

Locked `runs/tft_primary/models/tft_target_only`, 12 official-train
animals, no refit.

| quantity | value |
|---|---:|
| GroupNormalizer center | 5.616 |
| GroupNormalizer scale | 0.245 |
| masked decoder y mean | 5.965 |
| masked prediction mean | 5.927 |
| loss target | raw LOGW |

`QuantileLoss` is `2 × pinball` on raw `LOGW` after
`transform_output`. Reported `mean_quantile_loss` is 1× pinball on raw
`LOGW`.

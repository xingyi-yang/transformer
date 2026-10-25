# Covariate-aware Chronos-2 (history-only comparator retained)

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

Accept PR #12 as the reported evaluation correction. Prior run directories stay
frozen. This milestone traces the MAE/RMSE and grouping defects, reconciles the
**seq-diag dataset-0** Chronos empty-dataset failure, and executes **one**
fresh frozen matched-information comparison (`n_datasets=1`, `n_train_seeds=1`).
It is not a replicated confirmation and does not add a new model family.

## Defects (do not assume every run is affected)

- `rmse_log` = `|error|` and STUDYID merge collision lived in
  `score_quantile_table` used by official-seq and seq-diag.
- `evaluation.forecast` predose/early-history RMSE was already
  `sqrt(mean e²)`.
- Spline `wrmse`, transformer `val_rmse`, and TFT `val_loss` did not select on
  the defective quantile `rmse_log`.

## Dataset-0 Chronos failure

Raw train visit counts (39 series ≥ 8) are not the Chronos filter. Chronos
keeps `context.shape[-1] >= min_past + prediction_length` after official
preprocess. On **seq-diag dataset 0**, train survives and **validation** does
not (all series length 2–4), so `pipeline.fit(..., validation_inputs=...)`
raises. That failure is not this comparison’s Chronos run (validation here
kept 31 series). Preflight records both splits. History is not padded. If
validation is empty, `eval_loss` selection is skipped instead of fabricating
visits.

## Covariates

Chronos-2 covariate-aware inputs use only origin-available fields:

`SEX`, `STRAIN`, `SPECIES`, `ROUTE`, `PLANNED_DOSE`, planned `DAY`.

Future `LOGW` / weights / actual dose are not covariates. History-only
univariate LOGW remains a published comparator.

## Comparison

One fresh matched synthetic holdout (seeds 73001 / 83001). Same encoder cutoff
(DAY ≤ 14, observed visits) and the same target-study val/test hashes for TFT,
spline, the custom transformer, Chronos history-only, and Chronos
covariate-aware. Corrected study-balanced MAE and true RMSE. Failures listed,
not averaged as if complete. No design, no real data, no new architectures.

`send-bw v3-cov-chronos --config configs/v3_cov_chronos.frozen.yaml`

# Evaluation correctness (no additional training)

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

Stop training. Rescore saved predictions only. Official-seq and seq-diag run
directories are frozen and **superseded** for the name `rmse_log`.

## What was wrong

`rmse_log` computed `sqrt((point − y)²)` = `|error|`, then averaged. That is
MAE, not RMSE. After a `STUDYID` merge collision, grouping often used
`USUBJID` (58 animals) instead of the two target test studies.

## What this protocol publishes

From saved prediction CSVs, with observation `STUDYID` as the grouping key:

- **MAE** (`mae_log`): study-balanced mean of `|point − LOGW|`
- **true RMSE** (`rmse_log`): per-study `sqrt(mean e²)`, then the mean of studies
- merge cardinality (1–1 USUBJID/DAY)
- paired transfer on MAE, RMSE, and pinball
- horizon units: calendar DAY and observed future-visit rank (no fabricated days)
- Chronos-2 training-series eligibility: observed visit count vs
  `min_past + prediction_length` (no padding)

Chronos-2 remains **history-only / univariate**. Fine-tune failures are listed
separately and are not averaged with completed datasets.

No real data, new architectures, design optimization, or covariate-aware
comparison in this step.

## Execution

`send-bw v3-eval-correct --config configs/v3_eval_correct.frozen.yaml`

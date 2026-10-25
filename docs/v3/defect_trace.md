# MAE/RMSE and grouping defect trace

> **Synthetic diagnostic. Not design qualification. No real-data validation.**

- Affected published metrics: `['runs/v3_official_seq', 'runs/v3_seq_diag']`
- Rescore needed: `[]`
- Validation-selection affected: `False`

Spline wrmse, transformer val_rmse, and TFT val_loss did not use the defective quantile rmse_log. Seq-diag Chronos-2 dataset-0 failed at validation length filtering before eval_loss selection. The covariate-aware comparison run is a separate single draw; its Chronos validation was usable.

Forecast.py runs (v3_eval through v3_objective) already published true RMSE grouped by STUDYID. Official-seq and seq-diag published `|error|` as `rmse_log` and often grouped by animal; those tables are preserved and were rescored in `runs/v3_eval_correct`.

# Bounded official-sequence diagnostic

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

PR #10 (`runs/v3_official_seq/`) is accepted and frozen. This protocol does not
overwrite it, does not retune TFT or Chronos-2, and does not add architectures.

## Scope

Two phases, both **early-history only** (observed DAY ≤ 14 encoder; score
DAY > 14). Predose / pre-study prediction is a different task and is not
mixed into these tables. Design recommendations stay off. Real SEND access
stays off.

| Phase | Data | Purpose |
|---|---|---|
| A. Correctness | Frozen official-seq holdout (read-only) | Last-value baseline; Chronos transformation / time / covariate checks; common quantile scoring with per-study and per-horizon breakdowns; point-prediction definition; exact checkpoint revision |
| B. Replicated comparison | Fresh synthetic datasets, prespecified seeds | Same models and settings as PR #10, plus last-value. Do not tune after viewing test metrics |

## Point-prediction definition (prespecified)

- Quantile models (TFT, Chronos-2, last-value): the point is **q0.5**.
- Chronos-2 `forecast_type` is QUANTILES. The official `mean` return is the
  **median** (`pipeline.py`: "the median is returned as the mean here").
- Gaussian comparators (spline, custom transformer): the point is `MEAN`.
- `rmse_log` is the study-balanced mean of `|point − LOGW|`, identical to
  official_seq. This definition is not changed after seeing results.

CRPS is reported only for the Gaussian residual layer.

## Last-value baseline

Last observed encoder LOGW, repeated at every future DAY. All reported
quantiles equal that last value (degenerate forecast). This is a baseline,
not a new architecture. Interval width is 0.

## Chronos checks (diagnostic only; not a retune)

1. **Transformation.** PR #10 feeds LOGW. Chronos-2 applies its own instance
   normalization. Phase A also scores zero-shot on `exp(LOGW)` and logs the
   quantiles back to LOGW. That comparison is recorded; the official encoding
   remains LOGW.
2. **Time.** Chronos receives visit-rank unit steps. Calendar DAY is used only
   to align scored rows. Irregular encoder gaps are counted, not interpolated.
3. **Covariates.** The official `predict` API accepts `past_covariates` /
   `future_covariates`. PR #10 passes a univariate LOGW array. SEX, STRAIN,
   ROUTE, dose, SPECIES, and DAY are unused. They are not added here.
4. **Checkpoint revision.** Record the Hugging Face SHA for `amazon/chronos-2`,
   package versions, and model class.

## Phase B freeze (declared before generate)

- `configs/v3_seq_diag.frozen.yaml`
- `n_datasets = 2`
- generate seeds: `72001`, `72001 + 1009`
- train seeds: `82001`, `82001 + 17`
- bank: `matched`; same study counts and TFT/Chronos budgets as official_seq
- models: last-value, spline, custom transformer, TFT (target-only / pooled),
  Chronos-2 (zero-shot / target-only / pooled)
- encoder window 14; identical target holdouts within each dataset
- `retune_after_test: false`

Negative or harmful transfer, and last-value beating Chronos or TFT, are
valid published findings. They are not a reason to change encoding, add
covariates, or lengthen training.

## Execution

`send-bw v3-seq-diag --config configs/v3_seq_diag.frozen.yaml`

Writes `runs/v3_seq_diag/` only.

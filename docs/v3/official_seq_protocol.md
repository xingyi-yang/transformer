# Official TFT and Chronos-2 early-history protocol

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

This protocol executes **PyTorch Forecasting’s `TemporalFusionTransformer`**
and the official **`chronos.Chronos2Pipeline` (`amazon/chronos-2`)**. It does
not substitute the custom baseline-conditioned transformer or the spline.

## Task

Common early-history forecast:

- Information cutoff: observed visits with **DAY ≤ 14**. Unobserved calendar
  days are not inserted. LOGW is never filled with 0.
- Score observed visits with **DAY > 14** on the **same target-species test
  studies** for every model.
- Target-only vs pooled training use the same frozen target validation/test
  STUDYID hashes. Donors join train only.

## Models

| Artifact | Class | Training |
|---|---|---|
| `tft_target_only` | `pytorch_forecasting.TemporalFusionTransformer` | target species |
| `tft_pooled` | same | target + donor studies |
| `chronos2_zero_shot` | `chronos.Chronos2Pipeline` | none (pretrained) |
| `chronos2_target_only` | same, LoRA `fit` | target species |
| `chronos2_pooled` | same, LoRA `fit` | target + donor studies |
| `spline_*` / `transformer_*` | existing comparators | same holdouts |

Zero-shot Chronos-2 is recorded separately from fine-tunes.

## Metrics

RMSE (median), mean pinball / quantile loss, 50/80% coverage and interval
widths, paired transfer Δ = pooled − target-only (Δ>0 harmful for RMSE/pinball).

CRPS only for the Gaussian spline/transformer residual layer. TFT and Chronos-2
emit a finite quantile set; that is not treated as a trajectory simulator and
does not justify sample or Gaussian CRPS.

## Refused

- Baseline-only / pre-study prediction (no encoder)
- Design simulation / `sample_study`
- Real SEND access
- Writing into preserved run directories

## Execution

`send-bw v3-official-seq --config configs/v3_official_seq.frozen.yaml`

# Locked sharing replication (compute-budgeted)

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

PR #13 frozen scores are unchanged. This protocol uses existing models and the
PR #12 corrected scorer. Checkpoints and donor counts are chosen from the
training/validation protocol only. Do not retune after viewing test metrics.

## Budget (locked before execution)

| quantity | value |
|---|---|
| `n_dataset_seeds` | **2** |
| `n_train_seeds` | **1** |
| bank | `matched` only |
| generate seeds | `74001`, `74001 + 1009 = 75010` |
| train seeds | `84001` (one per dataset) |
| sharing cells / (dataset, train) | 4 |
| total cells | **8** |

Sharing regimes (donor **k** frozen a priori, not from test):

| key | sharing | k | models trained |
|---|---|---|---|
| target_only | `context_specific` | 0 | spline, transformer, TFT, Chronos history-only LoRA |
| pooled | `pooled` | 3 | spline, transformer, TFT, Chronos history-only LoRA |
| selective | `selective` | 1 | spline, transformer |
| selective_regularized | `selective_regularized` | 1 | spline, transformer |

Comparators retained: spline, last-value, Chronos-2 **history-only** (zero-shot
and fine-tunes). TFT is scored on target-only and pooled only.

**Not implemented (published, not invented):**

- TFT selective / selective_regularized
- Chronos-2 selective / selective_regularized
- Chronos-2 covariate-aware (PR #13 single draw stays frozen; not re-run)
- Bayesian (off; unreliable on prior cells)
- Predose and design

Checkpoint selection (train/validation only):

- spline: ridge alpha by validation wrmse
- transformer: early-stop on validation RMSE
- TFT: Lightning `val_loss`
- Chronos LoRA: `eval_loss` when validation survives the length preflight;
  otherwise validation is skipped (no fabricated history)

Fits: 8 spline + 8 transformer + 4 TFT + 4 Chronos history-only LoRA +
2 Chronos zero-shot + 2 last-value.

## Estimated runtime (before execution)

Prior wall-clock on this CPU: official-seq 86s (1 dataset, 2 sharing cells),
seq-diag 182s (2 datasets), cov-chronos 139s (1 dataset, extra covariate
fine-tunes).

**Point estimate: 320 seconds (~5.3 minutes).** Bounded range 240–720 seconds
(4–12 minutes). Upper bound covers slower generate or Chronos preflight skips.

## Scoring

Corrected study-balanced MAE and true RMSE, STUDYID grouping, merge
cardinality. Paired Δ = S_sharing − S_target_only; Δ>0 harmful for MAE/RMSE/
pinball. Harmful, helpful, and null signs are all published. Failures are
listed and not averaged as if complete.

`send-bw v3-share-repl --config configs/v3_share_repl.frozen.yaml`

# Sharing replication — locked compute budget

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

**Not yet executed.** This file records the locked budget before the run.
PR #13 frozen scores in `runs/v3_cov_chronos/` are unchanged.

## Budget (stated before execution)

| quantity | value |
|---|---|
| `n_dataset_seeds` | **2** |
| `n_train_seeds` | **1** |
| generate seeds | `74001`, `75010` |
| train seeds | `84001` |
| bank | `matched` |
| sharing cells | 4 per (dataset, train) = **8** |
| estimated runtime | **320 s** (240–720 s; 4–12 min) |

Donor **k** (training/validation protocol only, not from test): target-only 0,
pooled 3, selective / selective_regularized 1.

TFT and Chronos history-only run on target-only and pooled. Selective cells
for those families are published as `not_implemented`. Chronos covariate-aware
is not re-run. Bayesian and predose stay off.

`send-bw v3-share-repl --config configs/v3_share_repl.frozen.yaml`

# Inner-fold donor ranking — locked, not executed

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

The executed prefix-abstention diagnostic is **closed without retuning**.
This lock is a separate, smaller ranking path. Prior run directories,
including `runs/v3_share_abstain/` and `runs/v3_share_abstain_run/`, are
not rewritten.

Scientific question: can one-donor candidates be ranked from inner
target-training study folds, then accepted or rejected on outer validation,
without using test outcomes or generator scenario labels?

## Ranking path

1. Eligible donors = donor-species QC studies (`n_eligible_donors` planned
   `3`).
2. One candidate per eligible donor (k=`1`).
3. Inner folds: sort target TRAIN STUDYIDs, split into
   `2` round-robin study folds. Fit on the other
   folds' target studies plus that one donor; score study-balanced true RMSE
   on the held-out target TRAIN studies (`inner_rmse_log`).
4. Rank by mean inner-fold RMSE. Ties break by STUDYID sort.
5. Outer-validation share-or-abstain: existing `decide_abstention` on the
   frozen target validation split (min_improvement 0, ties → target-only).

Rejected inputs: target test STUDYIDs, test metrics, generator
`donor_scenario` / bank labels. Banks remain experiment strata for data
generation; they are not ranking features.

Prefix (`studyid_sort_prefix`) stays as a non-learned comparator. It is not
replaced by this ranker until a later approved execution.

## Locked comparison (execution refused)

| quantity | value |
|---|---|
| `n_dataset_seeds` | **2** |
| `n_train_seeds` | **1** |
| generate seeds | `[78001, 79010]` |
| train seeds | `[86001]` |
| banks | `['matched', 'both']` |
| densities | `['full']` (full only; sparse already closed) |
| encoder window | DAY ≤ 14.0 |
| trained policy arms | target_only, prefix_one, ranked_one |
| abstention | reuse target_only vs ranked_one; 0 extra fits |
| candidate fits | **48** |
| candidate-fit formula | `n_dataset_seeds × n_train_seeds × n_banks × n_densities × n_models × n_eligible_donors × n_inner_folds` |
| final model fits | **24** |
| total fits if executed | **72** |
| estimated runtime | **240 s** (150–480) |
| compute cap | **600 s** |
| `execute` | **false** |

2×1 is **not confirmatory**. Training-seed variability is not estimated.

Stopping rules: refuse real data, design, TFT/Chronos, Bayesian, predose,
retune-after-test, writes into preserved dirs, and `execute: true`.

## Proposed later execution (not this PR)

Requires a separate approved config. This lock raises if `execute` is true.

`send-bw v3-donor-rank --config configs/v3_donor_rank.frozen.yaml`

# Share-or-abstain lock (not executed)

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

This protocol is separate from the fixed-donor PR #14 experiment
(`docs/v3/share_repl_protocol.md` at `da46768`). PR #12 remains the scoring
baseline. PR #13/#14 run directories, predictions, checkpoints, frozen
configs, and numerical tables are not rewritten.

Scientific question: can the existing framework identify when cross-species
sharing helps target-species prediction, and choose target-only when sharing
is unsupported? The selector is not optimized for a favorable transfer sign.

## Audit (read-only)

Existing “selected” donor inclusion is the STUDYID-sort prefix
(`donor_study_order` → first k). It is not a validation- or similarity-based
ranker. `selective` vs `selective_regularized` changes regularization, not
which donor IDs are chosen. Target-only is all target-species TRAIN studies
(not a single-study model). Zero donors is available (`context_specific`,
k=0). Target test outcomes do not enter donor inclusion, ridge/early-stop,
or the proposed abstention rule.

See `docs/v3/selective_sharing_audit.md` and `docs/v3/share_repl_ERRATUM.md`.

## Locked comparison (approval required to train)

| quantity | value |
|---|---|
| `n_dataset_seeds` | **2** |
| `n_train_seeds` | **1** |
| generate seeds | `76001`, `77010` |
| train seeds | `85001` |
| random-donor seed | `91001` (not the train seed) |
| banks | `matched` (compatible), `both` (incompatible) |
| densities | `full`, `sparse` (`n_keep=4` target TRAIN studies) |
| encoder window | DAY ≤ 14 (unchanged; sparsity is not a shorter encoder) |
| trained policy arms | target_only (k=0), random_one (k=1), selected_one (k=1 prefix, selective v1), pooled_all |
| parameterization check | selected_regularized (same prefix donor) on **matched + full only** |
| abstention | reuse target_only vs selected_one; 0 extra fits |
| trained cells | **34** |
| model fits | **68** (spline + transformer) |
| additional candidate fits | **0** |
| estimated runtime | **180 s** (120–480) |
| compute cap | **600 s** (stop if exceeded) |
| `execute` | **false** |

Sparsity: `compose_sparse_target_train` keeps the first 4 target TRAIN
STUDYIDs by sort. Validation and test study sets stay frozen. Distinct from
a sparse history encoder.

Selection rules are fixed across banks and densities. The selector does not
see generator donor-scenario labels.

### Abstention (prespecified)

- Metric: target-validation study-balanced **true RMSE** (`val_rmse_log`)
  on the early-history task (same cutoff as evaluation).
- Lower is better. `min_improvement = 0` (strict improvement required).
- Ties, missing scores, or `n_val_studies < 1` → **target-only**.
- Threshold is not chosen from test.

### Planned outcomes (when executed)

- Paired MAE/RMSE vs target-only (harmful if Δ>0).
- Harmful-transfer frequency and magnitude.
- Selected vs random one-donor (equal k=1).
- Abstention frequency (how often target-only is chosen).
- Per-study / horizon rows and every failure (not averaged as complete).
- Dataset variability from 2 generate seeds. Training-seed variability is
  **not estimated** (`n_train_seeds=1`).
- Any test-informed comparison is retrospective only and must not feed the
  deployed rule.

Stopping rules: refuse real data, design, TFT/Chronos, Bayesian, predose,
retune-after-test, and writes into preserved dirs. Stop at compute cap.
Fail cells are published; they are not filled in as completed.

## Proposed execution command (approval required)

1. Set `share_abstain.execute: true` in `configs/v3_share_abstain.frozen.yaml`.
2. `send-bw v3-share-abstain --config configs/v3_share_abstain.frozen.yaml`

Until that change is approved, `execute: false` writes the audit, erratum,
hashes, and lock only.

`send-bw v3-share-abstain --config configs/v3_share_abstain.frozen.yaml`

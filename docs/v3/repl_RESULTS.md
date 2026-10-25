# v3 replicated synthetic benchmark (executed)

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Signs are published as computed. The protocol was **not** retuned after
target-test metrics. Sharing is **not** required to improve. Negative transfer
is left as-is. Prior `runs/v3_eval/` and `runs/v3_pilot_k0k1/` were not
modified.

## Commands

```text
git rev-parse HEAD
# fbf219628c5606db3dc4c300e3f82d66a0b7a102  (training freeze; clean tree after isolation fix)
python3 -m send_bw.cli v3-repl --config configs/v3_repl.frozen.yaml
```

A later reporting-only patch (`88f0e1c`) groups paired-Δ uncertainty by frozen
`test` split. No model was refit.

## Effective configuration

- File: `configs/v3_repl.frozen.yaml`
- Config hash: `a9f9568910bf8669`
- Git revision locked before generate/train: `fbf219628c5606db3dc4c300e3f82d66a0b7a102`
- `generate_seed`: `20261026` (dataset only; replicate *i* uses `+ i * 1009`)
- `train_seed`: `4242` (splits and models only; replicate *i* uses `+ i * 17`)
- Bank: `both` (growth + treatment donor deviations)
- Replicates: 2
- Regimes: context-specific (k=0), pooled (k=1), selective v1 (k=1), selective_regularized (k=1)
- Sparse target-study training: `n_keep=4` on replicate 0, validation/test study sets frozen
- Sparse history encoders: W0/W1/Wlast14/Wall14 on replicate 0 context-specific (distinct kind)
- Models: spline; transformer `n_members=1`; Bayesian requested on replicate 0 (context, pooled, selective_regularized)
- Design / TFT / Chronos / DeepAR / real data: off

## Component status

| component | status | counted |
|---|---|---|
| selective v1 | **completed** | yes (signs published) |
| selective_regularized | **completed** | yes (signs published) |
| sparse target-study training | **completed** | yes |
| sparse history encoder | **completed** | yes (distinct from target-train thinning) |
| evaluation pack (individual / group-mean / contrast / temporal) | **completed** | yes |
| replicated transfer uncertainty | **completed** | yes |
| Bayesian (3 matched-data cells) | **unreliable** | **no** |
| Bayesian sensitivity variants | **not_run** | no (`sensitivity.enabled: false`) |
| extra families / design / real data | **not_run** | no (forbidden) |

Correctness / inference failures after the header-isolation fix: **none**.
Unfavorable transfer is not a failure.

## Hashes (replicate 0 target manifests)

Validation and test hashes are identical for full-train and `stt4` cells.

| Object | Hash (16 hex) |
|---|---|
| Frozen config | `a9f9568910bf8669` |
| Target train (pre-thinning) | `007718ba753abc1d` |
| Target validation | `37fb8b3fef2c9ca6` |
| Target test | `298a3ce08799bc77` |

`sparse_target_train.json` kind is `sparse_target_study_training`, not
`sparse_history_encoder`.

## Bayesian (executed, not counted)

Matched data: same prepare key / train-validation study hashes as spline and
transformer in that cell. Diagnostics vs gate `rhat≤1.01`, `ESS≥400`,
`divergences==0`:

| cell | rhat_max | ess_bulk_min | ess_tail_min | divergences | counted |
|---|---|---|---|---|---|
| context_specific | 1.084 | 29 | 63 | 0 | no |
| pooled | 1.040 | 65 | 38 | 0 | no |
| selective_regularized | 1.037 | 58 | 102 | 0 | no |

Bayesian Δ rows exist in `transfer_table.json` but are **not** a successful
benchmark. Thresholds were not relaxed.

## Primary predose transfer (frozen `test`, full target train)

Δ = S_k − S_0 across two generate/train seed pairs. RMSE/CRPS: Δ>0 harmful.
Contrast: Δ = |S_k| − |S_0|. Sharing is not required to improve.

| model | sharing | n | mean Δ RMSE | sd Δ RMSE | RMSE signs | CRPS signs |
|---|---|---|---|---|---|---|
| spline | pooled | 2 | −0.002753 | 0.001017 | helpful, helpful | harmful, harmful |
| spline | selective | 2 | −0.001322 | 0.001973 | harmful, helpful | harmful, harmful |
| spline | selective_regularized | 2 | +0.002313 | 0.006515 | harmful, helpful | harmful, harmful |
| transformer | pooled | 2 | +0.023860 | 0.019547 | harmful, harmful | harmful, harmful |
| transformer | selective | 2 | +0.029807 | 0.020552 | harmful, harmful | harmful, harmful |
| transformer | selective_regularized | 2 | +0.014773 | 0.007610 | harmful, harmful | harmful, harmful |

Transformer transfer is harmful on both replicates for RMSE and CRPS. Spline
RMSE is mixed or slightly helpful depending on regime; CRPS is harmful in every
full-train pair. These signs are published. They are not a license to pick the
favorable seed or to retune.

## Evaluation pack (example: r0 context-specific spline, frozen test)

| block | coverage | width | other |
|---|---|---|---|
| individual | cov80 0.780; cov95 0.945 | width80 0.185; width95 0.264 | RMSE 0.0825; CRPS 0.0434 |
| group-mean | cov90 0.825 | — | pred/obs between-animal var ratio 1.44 |
| contrast | cov90 1.00 | width90 0.139 | bias 0.0038; study-level RMSE 0.0318 (2 studies) |
| temporal | PIT / increments / extrema / gap corr written | — | see `*_evaluation_pack.json` |

Uncertainty limitation (fixed): coverage and widths are predictive-interval
diagnostics under HierarchicalCovariance. Parameter uncertainty is incompletely
estimated except for a **reliable** Bayesian posterior, which this run did not
produce.

## Artifact locations

- parent: `runs/v3_repl/`
- freeze lock: `runs/v3_repl/freeze_lock.json`
- component status: `runs/v3_repl/component_status.json`
- transfer: `runs/v3_repl/transfer_table.json`
- transfer uncertainty: `runs/v3_repl/transfer_uncertainty.json`
- per-cell packs: `runs/v3_repl/replicates/r*/cells/*/evaluation/*_evaluation_pack.json`

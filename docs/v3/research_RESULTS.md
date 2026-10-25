# v3 research evaluation protocol (executed)

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Signs are published as computed. Sharing is **not** required to improve.
Full-versus-sparse target-study training is the primary contrast. Negative
transfer is left as-is. Prior `runs/v3_repl/`, `runs/v3_eval/`, and
`runs/v3_pilot_k0k1/` were not modified.

## Commands

```text
git rev-parse HEAD
# 54095801f1d8d240bbf8b59e5a2506f66ddf5004  (protocol freeze)
python3 -m send_bw.cli v3-research --config configs/v3_research.frozen.yaml
```

A later reporting-only patch records temporal diagnostics on the evaluation
packs already written by that execution. No model was refit.

## Effective configuration

- File: `configs/v3_research.frozen.yaml`
- Config hash: `438cb0640f9f9ff8`
- Git revision locked before generate/train: `54095801f1d8d240bbf8b59e5a2506f66ddf5004`
- `generate_seed`: `20261027` (dataset *i* uses `+ i × 1009`)
- `train_seed`: `5150` (train *j* uses `+ j × 17`)
- Nested crossing: 2 dataset seeds × 2 train seeds
- Banks: `matched` and `both` (conflicting growth + treatment donor deviations)
- Sharing: context-specific (k=0), pooled, selective v1, selective_regularized (k=1)
- Sparse target-study training: `n_keep=4` on every (dataset, train, bank, sharing) cell
- Target splits: frozen once per (dataset, train) and reused across sharing and banks
- Models: spline; transformer `n_members=1`; Bayesian on d0/t0 context-specific only
- Bayesian budget check: draws 800, tune 800, chains 2; gate unchanged
- Design / TFT / Chronos / DeepAR / real data: off
- 64 spline+transformer cells; 0 correctness failures; 176 s wall time

## Component status

| component | status | counted |
|---|---|---|
| selective v1 | **completed** | yes (signs published) |
| selective_regularized | **completed** | yes (signs published) |
| sparse target-study training | **completed** | yes |
| full versus sparse (primary) | **completed** | yes |
| matched and conflicting donors | **completed** | yes |
| fixed target splits | **completed** | yes |
| evaluation pack | **completed** | yes |
| Monte Carlo uncertainty | **completed** | yes (n=4 paired Δ per group) |
| Bayesian (2 matched-data cells) | **unreliable** | **no** |
| extra families / design / real data | **not_run** | no (forbidden) |

Correctness / inference failures: **none**. Unfavorable transfer is not a
failure.

## Locked target-split hashes (STUDYID sets)

Hashes are of target STUDYID lists, not of outcomes. Generator study labels
are stable, so different dataset seeds can share the same ID-set hash while
still having different holdout observations. Within each (dataset, train)
pair, every sharing and bank cell reused the same validation and test IDs.

| pair | validation | test | all_target |
|---|---|---|---|
| d0 t0 | `51c7ec66a9c61a35` | `c638b57becb930c6` | `b42a27301470dc82` |
| d0 t1 | `1466a6d94c7e9b97` | `2e7d34cf23c01ab7` | `b42a27301470dc82` |
| d1 t0 | `51c7ec66a9c61a35` | `c638b57becb930c6` | `9caa839ea3d54c92` |
| d1 t1 | `1466a6d94c7e9b97` | `ec72fd0e04ae6f71` | `9caa839ea3d54c92` |

Context-specific cells are identical across `matched` and `both` because k=0
uses no donors. Bank contrast appears only in donor-using regimes.

## Bayesian sampling diagnosis (budget check; not counted)

Both requested cells sampled (not compile/backend). Thresholds stayed
`rhat≤1.01`, `ESS≥400`, `divergences==0`. Context-specific uses target-only
data, so the two banks produced the same posterior diagnostics.

| cell | draws×chains | rhat_max | ESS bulk/tail | divergences | class | counted |
|---|---|---|---|---|---|---|
| d0 t0 matched context | 800×2 | 1.015 | 115 / 142 | 0 | mean_identifiability | no |
| d0 t0 both context | 800×2 | 1.015 | 115 / 142 | 0 | mean_identifiability | no |

Worst rhat/ESS remain on `theta` / `beta`. Covariance families improved
relative to the 400×2 prototype (ell rhat 1.003, sd_s ESS 442) but the mean
coefficients still fail the gate. Doubling the posterior budget raised ESS
from 29 to 115; that is still below 400. Priors and parameterization were not
changed. Sensitivity variants were off.

## Primary: full versus sparse target-study training (frozen test)

Δ = S_sparse − S_full. RMSE/CRPS: Δ>0 means sparse is worse. n=4 is the
2×2 nested seed crossing. `sd` is the sample standard deviation of those
four paired Δ values; it is not a license to pick a favorable seed.

| model | sharing | bank | n | mean Δ RMSE | sd Δ | harmful | helpful |
|---|---|---|---|---|---|---|---|
| spline | context_specific | both / matched | 4 | +0.01230 | 0.00544 | 4 | 0 |
| spline | pooled | both | 4 | −0.00337 | 0.00207 | 0 | 4 |
| spline | pooled | matched | 4 | −0.00200 | 0.00508 | 1 | 3 |
| spline | selective | both | 4 | −0.00344 | 0.00141 | 0 | 4 |
| spline | selective | matched | 4 | −0.00215 | 0.00543 | 1 | 3 |
| spline | selective_regularized | both | 4 | −0.00568 | 0.00346 | 0 | 4 |
| spline | selective_regularized | matched | 4 | −0.00329 | 0.00673 | 1 | 3 |
| transformer | context_specific | both / matched | 4 | +0.00288 | 0.00858 | 3 | 1 |
| transformer | pooled | both | 4 | +0.03413 | 0.02655 | 4 | 0 |
| transformer | pooled | matched | 4 | +0.03168 | 0.02412 | 4 | 0 |
| transformer | selective | both | 4 | +0.02252 | 0.01865 | 4 | 0 |
| transformer | selective | matched | 4 | +0.02424 | 0.01969 | 4 | 0 |
| transformer | selective_regularized | both | 4 | +0.03595 | 0.02524 | 4 | 0 |
| transformer | selective_regularized | matched | 4 | +0.03363 | 0.02574 | 4 | 0 |

Spline context-specific RMSE is worse under sparse target-study training on
every seed pair. Transformer RMSE is worse under sparse training in every
donor-using regime. These signs are published.

## Sharing transfer (frozen test, full target train)

Δ = S_k − S_0 versus context-specific. RMSE: Δ>0 harmful.

| model | sharing | bank | n | mean Δ RMSE | sd Δ | harmful | helpful |
|---|---|---|---|---|---|---|---|
| spline | pooled | both | 4 | +0.01097 | 0.00772 | 4 | 0 |
| spline | pooled | matched | 4 | +0.01036 | 0.00712 | 4 | 0 |
| spline | selective | both | 4 | +0.01074 | 0.00711 | 4 | 0 |
| spline | selective | matched | 4 | +0.01017 | 0.00650 | 4 | 0 |
| spline | selective_regularized | both | 4 | +0.01722 | 0.00941 | 4 | 0 |
| spline | selective_regularized | matched | 4 | +0.01747 | 0.00925 | 4 | 0 |
| transformer | pooled | both | 4 | −0.00907 | 0.00893 | 1 | 3 |
| transformer | pooled | matched | 4 | −0.00880 | 0.00792 | 1 | 3 |
| transformer | selective | both | 4 | −0.00277 | 0.00720 | 1 | 3 |
| transformer | selective | matched | 4 | −0.00574 | 0.00811 | 1 | 3 |
| transformer | selective_regularized | both | 4 | −0.00357 | 0.00784 | 1 | 3 |
| transformer | selective_regularized | matched | 4 | −0.00838 | 0.01020 | 1 | 3 |

Spline sharing transfer is harmful for RMSE on every full-train seed pair in
both banks. Transformer sharing RMSE is mixed (3 helpful / 1 harmful). This
does not invert or replace the prototype result on its own seeds. It is a
second nested-seed draw, published as computed.

## Calibration, widths, contrast, temporal

Example: `d0_t0_matched_context_specific_k0_full` spline, frozen test.

| block | value | denominator |
|---|---|---|
| individual cov80 / cov95 | 0.979 / 0.994 | 2 studies, 56 animals, 218 obs (study-balanced) |
| individual width80 / width95 (log) | 0.195 / 0.275 | same |
| RMSE / CRPS (study-balanced) | 0.0467 / 0.0273 | 2 studies |
| group-mean cov90 | 1.00 | 64 SEX×ARM×DAY cells |
| pred/obs between-animal var ratio | 1.12 | 64 cells |
| contrast cov90 / width90 | 1.00 / 0.183 | 6 contrast rows at day 28 |
| contrast bias (row mean) | −0.0120 | 6 rows |
| contrast bias (study-balanced) | −0.0244 | 2 studies with contrast |
| contrast RMSE (study-balanced) | 0.0529 | 2 studies |
| increment sd ratio pred/obs | 1.34 | 56 increments |
| extrema max cov90 | 0.982 | 56 extrema |
| gap correlations (obs vs pred) | 0.888/0.866; 0.820/0.784; 0.769/0.681 | 6, 4, 2 pairs |
| PIT 10-bin counts | 0, 9, 28, 19, 35, 52, 40, 19, 12, 4 | 218 obs |

Uncertainty limitations: coverage and widths are predictive-interval
diagnostics under HierarchicalCovariance. Parameter uncertainty is
incompletely estimated (spline mean bootstrap only; transformer
`n_members=1`; covariance parameters not bootstrapped). The Bayesian
posterior is not counted. Transfer contrast Δ uses the contrast-row mean;
the study-balanced contrast is reported here and in pack denominators.
Monte Carlo uncertainty uses n=4 paired Δ; that sample sd is itself noisy.
Several temporal denominators are single-digit pair counts.

## Artifact locations

- parent: `runs/v3_research/`
- freeze lock: `runs/v3_research/freeze_lock.json`
- component status: `runs/v3_research/component_status.json`
- locked splits: `runs/v3_research/locked_target_splits.json`
- density uncertainty: `runs/v3_research/density_uncertainty.json`
- sharing uncertainty: `runs/v3_research/sharing_uncertainty.json`
- per-cell packs: `runs/v3_research/datasets/d*/trains/t*/cells/*/evaluation/*_evaluation_pack.json`

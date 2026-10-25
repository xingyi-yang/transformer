# v3 synthetic evaluation benchmark (executed)

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Signs are published as computed. The protocol was **not** retuned after
target-test metrics. Negative transfer is left as-is. The frozen k=0/k=1
pilot under `runs/v3_pilot_k0k1/` was not modified.

## Commands

```text
git rev-parse HEAD
# 68b9144a94bc4ca13b8609348ca1c052b961293d  (freeze; clean tree)
python3 -m send_bw.cli v3-eval --config configs/v3_eval.frozen.yaml
```

## Effective configuration

- File: `configs/v3_eval.frozen.yaml`
- Config hash: `fbf4c58c2e5715ae`
- Git revision locked before generate/train: `68b9144a94bc4ca13b8609348ca1c052b961293d`
- Seed: `20261025`
- Banks: `matched` and `both` (growth+treatment donor deviations)
- Regimes: context-specific (k=0), pooled (k=1), selective v1 (k=1)
- Models: spline; transformer `n_members=1`; Bayesian requested but **not_run** (PyMC extra missing) and **not counted**
- Tasks: predose-conditioned + sparse grid W0/W1/Wlast14/Wall14
- Also: fixed-schema spline pair; tight training-budget pair
- Design / TFT / Chronos / DeepAR / real data: off

## Hashes

| Object | Hash |
|---|---|
| Frozen config | `fbf4c58c2e5715ae` |
| Target train | `340dc8e79c1238b4` |
| Target validation | `4f8d1a95c3c18638` |
| Target test | `840a846f9dc59781` |
| Target all | `76d8ec6b17dae51a` |

Target manifests are identical across all nine cells (both banks share the RAT seed stream).

## Primary predose transfer (bank `both`, variable schema, primary budget, frozen target test)

Δ = S1 − S0. RMSE/CRPS: Δ>0 harmful. Contrast: Δ = |S1| − |S0|.

| model | sharing | S0 RMSE | S1 RMSE | Δ RMSE | RMSE sign | CRPS sign | contrast sign |
|---|---|---|---|---|---|---|---|
| spline | pooled | 0.042212 | 0.048860 | +0.006648 | **harmful** | **harmful** | helpful |
| spline | selective | 0.042212 | 0.048458 | +0.006246 | **harmful** | **harmful** | helpful |
| transformer | pooled | 0.040482 | 0.041850 | +0.001367 | **harmful** | **harmful** | **harmful** |
| transformer | selective | 0.040482 | 0.040368 | −0.000114 | helpful | **harmful** | **harmful** |

Matched-donor pooled RMSE is also harmful (spline +0.003170; transformer +0.007039). Tight-budget and fixed-schema pairs are harmful on RMSE/CRPS. Full rows: `runs/v3_eval/transfer_table.json`.

## Bayesian

Not run: `bayesian extra not installed`. Not counted as a successful benchmark.

## Covariance / uncertainty

Spline HierarchicalCovariance converged with **no** parameters at bound on the recorded cells. Covariance parameters are not bootstrapped.

- Predictive variability: HierarchicalCovariance draws in `sample_study`.
- Parameter uncertainty is incompletely estimated (spline mean bootstrap only; transformer `n_members=1`; no Bayesian posterior).

## Failures

None. Design directories were not written. `retuned_after_test: false`. Wall ~82 s.

## Artifact locations

- Freeze lock: `runs/v3_eval/freeze_lock.json`
- Banks: `runs/v3_eval/banks/{matched,both}/`
- Cells: `runs/v3_eval/cells/`
- Summary / transfer: `runs/v3_eval/SUMMARY.json`, `runs/v3_eval/transfer_table.json`
- Per-study CSVs and sparse JSON under each cell `evaluation/`

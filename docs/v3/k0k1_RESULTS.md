# v3 synthetic pilot k=0 → k=1 (executed)

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Stopped after k=1. Signs are published as computed. The protocol was **not**
retuned after target-test metrics. Transformer harmful transfer is left as-is.

## Commands

```text
git rev-parse HEAD
# 9f1da17dcc52f22154364c9caf751a6601b78760  (freeze commit; clean tree)
python3 -m send_bw.cli v3-pilot --config configs/v3_pilot.k0k1.frozen.yaml
```

## Effective configuration

- File: `configs/v3_pilot.k0k1.frozen.yaml`
- Config hash: `4112f7ae47000c25`
- Git revision locked before generate/train: `9f1da17dcc52f22154364c9caf751a6601b78760`
- Seed: `20261024`
- Task: predose-conditioned only
- Models: spline (`n_bootstrap=6`); transformer (`n_members=1`, `max_epochs=20`, `patience=6`)
- Tuning: target validation only (ridge α grid; transformer early stop)
- k=0: context-specific, 9 RAT train studies
- k=1: pooled, same 9 RAT train studies + `SYNA-D001`
- Design / Bayesian / TFT / Chronos / DeepAR / k=2,3 / real data: off

## Hashes

| Object | Hash |
|---|---|
| Frozen config | `4112f7ae47000c25` |
| Target train | `cbdb10a6c2f1bfdf` |
| Target validation | `509b5d25bd90188b` |
| Target test | `04d76ef2e097f9ff` |
| Target all (15 RAT in splits) | `d8aa4e2e093e11b9` |
| k=0 composed splits | `4797ef4589ea7e7f` |
| k=1 composed splits | `cabe2e0e5b0b6881` |

Target manifests are identical at k=0 and k=1. Donors selected at k=1:
`SYNA-D001` only. Validation/test remain `SYNA-009,012,016` /
`SYNA-002,005,007`.

## Study-balanced predose metrics (frozen target test)

| model | k | RMSE log | CRPS | contrast bias |
|---|---|---|---|---|
| spline | 0 | 0.061353 | 0.041406 | −0.010604 |
| spline | 1 | 0.060121 | 0.040259 | +0.004958 |
| transformer | 0 | 0.053812 | 0.036992 | +0.041213 |
| transformer | 1 | 0.057860 | 0.039478 | +0.042849 |

## Sign-defined transfer (frozen target test)

Δ = S1 − S0. RMSE/CRPS: Δ>0 harmful. Contrast: Δ = |S1| − |S0|.

| model | metric | S0 | S1 | delta | sign |
|---|---|---|---|---|---|
| spline | rmse_log | 0.061353 | 0.060121 | −0.001232 | helpful |
| spline | crps | 0.041406 | 0.040259 | −0.001147 | helpful |
| spline | contrast_bias | −0.010604 | 0.004958 | −0.005646 | helpful |
| transformer | rmse_log | 0.053812 | 0.057860 | +0.004048 | **harmful** |
| transformer | crps | 0.036992 | 0.039478 | +0.002486 | **harmful** |
| transformer | contrast_bias | 0.041213 | 0.042849 | +0.001636 | **harmful** |

Compound-holdout (one study; not the primary frozen test) is recorded in
`runs/v3_pilot_k0k1/transfer_table.json`. Transformer is also harmful there.
Spline RMSE/CRPS helpful, contrast harmful. Not used to retune.

## Per-study RMSE log (frozen target test)

| STUDYID | spline k=0 | spline k=1 | transformer k=0 | transformer k=1 |
|---|---|---|---|---|
| SYNA-002 | 0.046404 | 0.052925 | 0.045513 | 0.047302 |
| SYNA-005 | 0.084580 | 0.076239 | 0.052230 | 0.042109 |
| SYNA-007 | 0.053075 | 0.051198 | 0.063693 | 0.084170 |

## Recorded fit facts

- Study weighting: `study_balanced` (equal total weight per STUDYID).
- Spline α selected on target validation: 0.1 at both k.
- Transformer member-0 epochs: 10 (k=0), 20 (k=1). Approx AdamW steps 70 / 160.
- k=0 species schema: no `SPECIES_*`; no species token.
- k=1 species schema: `SPECIES_RAT` plus species token; levels DOG, RAT.
- Covariance: spline OOF HierarchicalCovariance, **not** bootstrapped; shared with the transformer.
- Runtime: ~11 s wall (generate+two fits). Spline ~0.5–0.6 s; transformer ~2.0–2.2 s each.
- Warnings (both k): 3 DATE_UNCERTAIN weights; 50 planned-regimen deviations.

## Uncertainty (fixed language; not result-dependent)

- **Predictive variability:** HierarchicalCovariance study/animal/residual draws in `sample_study`.
- **Parameter uncertainty is incompletely estimated:** spline bootstraps the mean only; covariance parameters are not bootstrapped; transformer `n_members=1` has no bootstrap spread. Predictive intervals are not complete parameter-uncertainty intervals.

## Failures

None in execution. Design directory was not written. No retune.

## Artifact locations

- Freeze lock: `runs/v3_pilot_k0k1/freeze_lock.json`
- Summary: `runs/v3_pilot_k0k1/SUMMARY.json`
- Transfer table: `runs/v3_pilot_k0k1/transfer_table.json`
- k=0 run: `runs/v3_pilot_k0k1/k0/`
- k=1 run: `runs/v3_pilot_k0k1/k1/`
- Per-study CSVs: `runs/v3_pilot_k0k1/k{0,1}/evaluation/{spline,transformer}_test_per_study.csv`
- Pilot records: `runs/v3_pilot_k0k1/k{0,1}/pilot_record.json`

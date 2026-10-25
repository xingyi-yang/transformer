# Pilot prespecification (stages 1–2)

Declared before any model-training wave. Source of truth in code:
`src/send_bw/pilot.py`. Proposed review-only config:
`configs/v3_pilot.smoke.yaml`. **Do not launch that config from this packet.**

## Task

First scored task: **predose-conditioned prediction** (v2 `evaluate_prestudy`;
alias `predose_conditioned`). Design-stage scoring and early-history /
sparse-history are later waves.

## Sharing

| Regime | Status |
|---|---|
| Context-specific | Implemented (v2 default; \(k=0\)) |
| Pooled | Implemented (explicit `SPECIES_*`; nested donor train additions) |
| Selective | **Required in the eventual benchmark. Deferred. Not implemented.** |

`sharing: selective` raises. Do not treat omission as a completed comparison.

## Study weighting

`study_balanced_weights`: each `STUDYID` contributes equal total weight;
weights average to 1. Donor studies in a pooled train set receive the same
per-study total as target studies. Validation stays target-only.

## Donor schedule

Order: `STUDYID` sort. Nested prefixes \(k \in \{0,1,2,3\}\). \(k=0\) is
zero-donor / context-specific equivalence. Target train/validation/test
manifests are frozen before any donor is added.

## Training budgets (proposed smoke, not executed)

| Model | Budget |
|---|---|
| Spline | `n_bootstrap=6`, `oof_folds=5`, `n_time_knots=3`, `n_expo_knots=2` |
| Transformer | `n_members=1`, `max_epochs=20`, `patience=6`, `d_model=32`, `n_heads=2` |
| Bayesian | `not_run` |
| TFT / Chronos / DeepAR | `not_run` |

## Tuning rules

- Spline: choose ridge \(\alpha\) from
  `[0.1, 1, 10, 100, 300, 1000, 3000, 10000]` by validation
  study-balanced WRMSE. No donor rows in validation.
- Transformer: early-stop on validation RMSE. Shared spline
  `HierarchicalCovariance` is required. Member 0 is the full-train mean;
  later members (if any) are study-bootstrap refits of the mean only.

## Random seeds

Root seed `20261024`. Named streams:

- compounds: `SeedStream("compounds")`
- target studies: `SeedStream("study", s)`
- donor studies: `SeedStream("donor", d)` (independent of `n_studies`)
- target splits: `run.seed`
- spline / transformer fit: model seed or `run.seed`
- predose evaluation: `run.seed + 17`

## Uncertainty safeguards (unchanged)

- Covariance: `HierarchicalCovariance` on study-level OOF residuals;
  covariance parameters are **not** bootstrapped.
- Parameter uncertainty: study-bootstrap of the **mean** only.
- Transformer residual: shared with the spline; required.
- No fabricated baseline-only series.
- Design recommendations stay off.

## Spline mean (implemented basis; do not call this “linear”)

Inspected in `src/send_bw/models/spline.py` and `models/design.py`:

- Ridge-penalized B-spline time basis (degree 3 when `rich`, else 2)
- Time basis interacted with sex, centered baseline, and strain
- Planned-exposure B-spline (degree 2) interacted with log-dose, log-dose
  squared, compound descriptors, inverse interval, route, and sex
- Post-dosing recovery basis (degree 1) interacted with log-dose and D1
- Pooled variant adds the same time-basis interactions for `SPECIES_*`

The mean is linear in the expanded coefficients after that basis. It is not
a linear-in-time model. The custom transformer is a separate neural mean.

## Labels (required on every artifact)

- Synthetic diagnostic.
- Not design qualification.
- No real-data validation.
- No operational recommendation.

# Frozen k=0 → k=1 synthetic pilot

**Status: frozen before execution.** Do not edit the effective YAML after the
freeze commit. Do not retune after viewing target-test metrics.

Effective configuration: `configs/v3_pilot.k0k1.frozen.yaml`

The executor writes `runs/v3_pilot_k0k1/freeze_lock.json` with the git
revision and config hash **before** generate/train. That lock is the
execution identity.

## Locked choices

| Item | Value |
|---|---|
| Task | predose-conditioned only |
| \(k\) | 0 then 1, then stop |
| Models | spline; transformer `n_members=1` |
| Hyperparameters / early stopping | target validation only |
| Donors | STUDYID sort; first donor only at \(k=1\) |
| Design | off |
| Bayesian / TFT / Chronos / DeepAR | off |
| Real data | off |
| \(k=2,3\) | off |
| Selective sharing | required later; not run |
| Retune after test | forbidden |

## Uncertainty language (fixed, not result-dependent)

- **Predictive variability:** `HierarchicalCovariance` study, animal, and
  residual draws inside `sample_study`.
- **Parameter uncertainty:** spline study-bootstrap of the **mean** only.
  Covariance parameters are **not** bootstrapped. The transformer has
  `n_members=1`, so it has **no** bootstrap member spread. That is
  incompletely estimated parameter uncertainty, not a claim that the
  predictive layer is complete.

## Labels

- Synthetic diagnostic.
- Not design qualification.
- No real-data validation.
- No operational recommendation.

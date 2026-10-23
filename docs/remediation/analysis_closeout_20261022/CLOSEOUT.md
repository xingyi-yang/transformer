# Analysis close-out (A default; B dropped from defaults; C1 frozen)

## Status

- **Synthetic diagnostic.**
- **Not design qualification.**
- **No real-data validation.**
- **No operational recommendation.**

This is a post-matrix software-default decision. It does not overwrite the locked
stream reports, does not retune any analysis, and does not qualify a study design.

## Decision

| Analysis | Status after this close-out |
| --- | --- |
| **A** `endpoint_ancova` | **Production default.** Survived locked complete and locked MCAR on the pre-specified matrix. |
| **B** `rm_gls` (quadratic weekly / biweekly) | **Implemented and requestable. Not a production default.** Missed FWER and coverage under slope heterogeneity ×1.6 in every locked stream. Weekly transient bias remains ~0.016. |
| **C1** `mmrm_visit_contrast` | **Frozen opt-in candidate.** Not promoted. Not retuned. Locked MCAR biweekly at \(n=8\) missed the FWER point ceiling (0.0675 > 0.065). |

Request default and shipped request YAMLs are now `analyses: [ancova]`.
`gls` and `mmrm` remain valid request names. Adding them is an explicit choice,
not the default search.

## Why this is not a rewrite of the locked reports

The locked complete (`docs/remediation/locked_20261201_c1/`) and locked MCAR
(`docs/remediation/mcar_locked_20270201_c1/`) reports said “do not switch
defaults” **as a constraint on those streams**: do not retune C1, do not promote
C1, and do not treat a passing cell as design qualification.

Those reports stay as written. This close-out is the later, authorized policy
step: after all four calibration streams finished, stop advertising B as a
peer of A in production defaults.

## Evidence used (already recorded; not re-run)

Sourced tolerances from `configs/new_study.example.yaml` and
`optimize.py::feasibility`: FWER point ceiling 0.065; FWER CI ceiling 0.095;
coverage floor 0.92; bias fraction 0.15; failure rate 0.02. Passing a tolerated
FWER gate is not “FWER = 0.05.”

### Locked complete, seed `20261201`, B=2000, 160,000 fits

- **A** met inference tolerances at both sizes.
- **B** missed at all four (schedule × size) cells. Worst weekly FWER 0.129
  (n=6, heterogeneity global null); weekly coverage down to 0.873 (n=8).
  Weekly transient bias 0.016 (fraction 0.107).
- **C1** met on complete data. That is not a promotion.

### Locked MCAR 2%/2%, seed `20270201`, B=2000, 160,000 fits

- Absence independent of \(Z\) (MCAR / design-dependent). **Not** prior-outcome MAR.
- **A** met both sizes.
- **B** missed all four cells again. Worst weekly FWER 0.141; coverage 0.860.
- **C1** weekly met; **C1 biweekly n=8 flipped** (exploratory met → locked missed).
  FWER point 0.0675. C1 was not retuned.

Exploratory streams (`20261101`, `20270101`) are supporting only. The lock is
the B=2000 streams.

## What changed in this revision

- `Request.analyses` default: `["ancova"]`.
- `configs/new_study.example.yaml` and `configs/new_study.smoke.yaml`:
  `analyses: [ancova]`.
- Status notes in `docs/methods.md`, `docs/limitations.md`, `README.md`,
  `HANDOFF.md`.

## What did not change

- `endpoint_ancova` and `rm_gls` formulas, df, and variance estimators.
- C1 specification (mean, sandwich, \(p_{\mathrm{visit}}\), df). No retune.
- Acceptance tolerances. No threshold was relaxed.
- Historical diagnostic artifacts and their “defaults remain `[ancova, gls]`”
  sentences.
- Smoke / Bayesian-smoke run artifacts.
- No real SEND data. No operational design. No `calibrate-analysis` CLI.

## Constraints that still hold

- Synthetic data only.
- Small matrix only (8 scenarios × 2 sizes; weekly / biweekly / endpoint).
- Deferred cells remain deferred: covariance shift, heavy tails, delayed
  alternatives, 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2.
- Unit tests are not scientific readiness.
- A surviving this matrix is not a claim that ANCOVA is valid on real studies.

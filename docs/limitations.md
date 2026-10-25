# Limitations

## Scope
- Body weight only, one species (RAT) and one study class. Other species are excluded, not extrapolated.
- Results cannot establish total animal requirements for a toxicology program. They cannot justify reducing
  welfare observations, replace concurrent controls, or support unplanned adaptation or regulatory claims.
- **No real SEND data has been used.** All metrics and designs are synthetic proofs of concept.

## Data
- Synthetic SEND-like tables are not certified SEND-conformant. The mapping profile is illustrative.
- TX/trial-set → arm joins are not automated. Planned regimens and compound identities require curated tables.
- The baseline rule assumes the last weight on or before elapsed day 0 is predose. This must be reviewed per study.
- define.xml parsing is not implemented. A reviewed mapping profile is required instead.
- Snowflake access is mock-tested only. Separate queries are not one consistent snapshot unless
  `as_of_timestamp` is set.

## Models
- Spline and transformer parameter uncertainty covers the mean only (study bootstrap). Covariance
  parameter uncertainty is not propagated.
- The covariance layer uses diagonal study and animal random-effect covariances and an OU residual. It was fitted by
  two-stage OOF residual maximum likelihood, not a joint likelihood.
- The Bayesian model has no separate white-noise nugget, uses a less flexible mean design than the spline, and was
  fitted on subsampled training animals (16 per study) in the smoke configuration. In the synthetic smoke
  run it **underestimated** the CMP03 effect (−7.7%/−14.6% vs oracle −15.5%/−22.2%). Its group-mean coverage
  on the compound-holdout study was poor (0.51 at the 90% level). The spline also underestimated the low-dose effect (−3.4%).
- Three of the four Bayesian sensitivity variants did not meet the diagnostic targets at 1500 draws.
- The transformer is a compact custom architecture, not a TFT. It overstates between-animal variance
  (predicted/observed ratio ≈ 1.47 in the smoke run) because it shares the spline covariance.
- Early-history conditioning for the required benchmarks is implemented only for the Gaussian-layer
  models (spline, transformer). The Bayesian model reports `not_run`.
- TFT is the documented early-history forecasting adapter and remains optional relative to the
  required spline / custom transformer / Bayesian models. It stays `not_run` unless
  `send-bw[tft]` is installed and the installed API verifies. It refuses baseline-only / pre-study
  prediction and design simulation. Quantile-spread variance is an approximation, not a fitted
  residual covariance. A global PF `GroupNormalizer` on mixed RAT+DOG `LOGW`, plus a train/predict
  `UNK` vs `"nan"` compound-token mismatch, produced a dog-scale collapse (MAE ≈ 0.55) on one
  extra split; the adapter now uses a species-or-global normalizer without `relu`. That repair
  is a synthetic diagnostic, not real-data qualification. Chronos-2 and DeepAR remain
  unimplemented (`not_run`).

## Statistical assumptions not validated
- A common treatment effect across sexes (pooled analysis).
- Any GLS gain from intermediate visits depends on the fixed quadratic ψ(t) basis. Under the transient-effect
  stress scenario, GLS bias reached 21% of δ and coverage fell to 0.87.
- Plug-in historical working covariance with model-based SEs and df = N − (3 + G·k), with no Kenward-Roger
  correction. Under the higher-heterogeneity stress scenario, GLS was **anti-conservative (FWER 0.10)**.
- An experimental visit-saturated analysis (`mmrm`) is implemented as an opt-in candidate. Its
  sandwich multiplier and endpoint-animal df are experimental. Complete-data streams (exploratory
  B=600, locked B=2000) met sourced inference tolerances on the small matrix. Under locked MCAR /
  design-dependent missingness (B=2000), C1 biweekly at \(n=8\) exceeded the FWER point ceiling
  (0.0675 > 0.065); ANCOVA still met. C1 was not retuned and is not the production default.
  It is not claimed valid under prior-outcome-dependent MAR when \(V_0\) is misspecified.
- After those locked streams, production defaults are **ANCOVA only**
  (`analyses: [ancova]`). GLS remains requestable; it is not a default because it
  missed FWER and coverage under slope heterogeneity on the same matrix. This default
  change is a synthetic software policy, not design qualification. See
  `docs/remediation/analysis_closeout_20261022/CLOSEOUT.md`.
- Exploratory A-default stress (B=600, seed `20270401`) and locked B=2000
  (seed `20270501`), complete data, on covariance shift, heavy tails, and
  delayed alternatives: ANCOVA met all six mechanism×size inference cells
  in both streams (no A flip). Exploratory C1 delayed at \(n=6\) missed
  FWER 0.067; locked C1 flipped to a pass (0.060). The flip was reported,
  not patched, and is not a promotion. B meeting those three stresses does
  not restore it as a default (the close-out was slope heterogeneity).
  Not design qualification.
- MAR missed visits and non-informative dropout only. Informative termination is not modelled, and no power
  claims are made for it.
- Generators A and B are illustrative and not biologically validated. Feasibility under them does not
  establish prospective power on real studies.
- Smoke designs used 400 proposal and 600 final replicates. This is a **nonconclusive software check**: FWER
  upper bounds are wide, and the final status depends on Monte Carlo noise near thresholds.

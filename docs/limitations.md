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
- Early-history conditioning is implemented only for the Gaussian-layer models (spline, transformer).
- TFT, Chronos-2 and DeepAR are not implemented (`not_run`).

## Statistical assumptions not validated
- A common treatment effect across sexes (pooled analysis).
- Any GLS gain from intermediate visits depends on the fixed quadratic ψ(t) basis. Under the transient-effect
  stress scenario, GLS bias reached 21% of δ and coverage fell to 0.87.
- Plug-in historical working covariance with model-based SEs and df = N − (3 + G·k), with no Kenward-Roger
  correction. Under the higher-heterogeneity stress scenario, GLS was **anti-conservative (FWER 0.10)**.
- An experimental visit-saturated analysis (`mmrm`) is implemented as an opt-in candidate. Its
  sandwich multiplier and endpoint-animal df are experimental. Complete-data streams (exploratory
  B=600, locked B=2000) and an exploratory MCAR / design-dependent stream (B=600) met sourced
  inference tolerances on the small matrix. That is not design qualification and not a reason to
  switch defaults. It is not claimed valid under prior-outcome-dependent MAR when \(V_0\) is
  misspecified. Production request configs remain `ancova` / `gls`.
- MAR missed visits and non-informative dropout only. Informative termination is not modelled, and no power
  claims are made for it.
- Generators A and B are illustrative and not biologically validated. Feasibility under them does not
  establish prospective power on real studies.
- Smoke designs used 400 proposal and 600 final replicates. This is a **nonconclusive software check**: FWER
  upper bounds are wide, and the final status depends on Monte Carlo noise near thresholds.

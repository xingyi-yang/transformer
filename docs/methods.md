# Methods

All results produced so far come from **synthetic** data and are a proof of concept only.

## Pipeline

Source (local CSV/Parquet/XPT or authorized read-only Snowflake) → raw domains and metadata →
version-aware mapping profile → canonical tables → QC/cohort → frozen study-level splits →
features → response models → joint trajectory simulator → prespecified final analysis → design report.
Synthetic and real inputs go through the same `Harmonizer`. Study splits (random or chronological,
plus an optional compound holdout) are frozen to `splits.csv`/`splits.json` with a hash before any
feature transform is fitted, and tampering is detected on load.

## Target and estimand

- Modelling target: `Y = log W(t) − log W_baseline` (observed baseline). Baseline log weight is also a
  covariate (centered on training data).
- Primary estimand: `Δ_g(T) = μ_g(T) − μ_0(T)`, the baseline-standardized active−control difference in
  mean log weight at fixed T. `100(exp Δ − 1)` is the geometric-mean-ratio percent difference.
- Secondary estimand (optional, request field `secondary_contrast`): `Σ_j a_j[μ_g(t_j) − μ_0(t_j)]`, with
  fixed query times and weights that are independent of the observation schedule. It is estimable only under the GLS
  basis, and it is reported (bias and coverage) but not used for feasibility.
- Sex strategy: pooled with a sex covariate. A common treatment effect across sexes is assumed, with equal
  allocation per sex.

## Response models (benchmarks)

All models implement `fit`, `predict_mean`, `predict_marginals`, `sample_study`, `save` and `load`. They
persist the feature specification (vocabularies, scaling, supported context and horizon) alongside their parameters.
Study and animal IDs are grouping keys only.

**A. Hierarchical spline (frequentist).** Ridge regression on a B-spline time basis interacted with
sex, baseline, compound descriptors and planned-exposure features. Training uses study-balanced weights, and alpha is
selected on the validation studies. The covariance layer is fitted by maximum likelihood (L-BFGS-B, batched
Woodbury) to **study-level out-of-fold residuals**:
`Cov = B(t)Ω_s B(t)' + B(t)Ω_a B(t)' + σ_p² exp(−|t−t'|/ℓ) + σ_m² I`, with `B(t) = [1, t/28]`
and diagonal Ω. Parameter uncertainty comes from study-bootstrap refits of the mean. **One member is applied
to a whole simulated study.** Covariance parameters are not bootstrapped (documented limitation).

**C. Baseline-conditioned transformer (custom; not a TFT).** Context tokens (biology, baseline,
compound descriptors, planned regimen) and query tokens (time and planned-exposure features at that
time). A boolean attention mask lets each query see only the context tokens and itself, so predictions
do not depend on which other times are queried (schedule invariance, which is tested). The defaults are d_model 32,
2 heads, 1 layer, dropout 0.1, AdamW, a loss on observed targets only with study-normalized weights,
validation early stopping, and bootstrap members. For the primary backbone comparison it **uses the spline's
covariance layer**.

**B. Bayesian hierarchical model (PyMC, NUTS via nutpie).**
`Y/s_y = Xβ + B(t)'u_s + B(t)'b_i + e_i(t)`. Design features:
- QR reparameterization, with the prior on θ = R*β.
- Non-identifiable design columns are dropped by pivoted QR and fixed at 0. These are recorded in the diagnostics.
- ZeroSumNormal non-centered study and animal effects.
- An OU residual via the exact conditional AR factorization.
- Priors (on the scaled-Y scale): θ ~ N(0, 0.35²); SDs ~ HalfNormal(0.3); σ_p ~ HalfNormal(0.3);
  ℓ ~ LogNormal(log 7, 0.75).

These priors were checked with a prior predictive check in **grams and percent change**. Further checks:
- A one-step posterior predictive check by time bin.
- A held-out new-study evaluation.
- Sensitivity runs with priors scaled ×0.5 and ×2, a Student-t likelihood, and the animal intercept removed.

Reliability requires R-hat ≤ 1.01, bulk and tail ESS ≥ 400 and zero divergences. Otherwise the model is labelled
unreliable and is not ranked. New-study prediction uses one posterior draw per simulated study,
with a new study effect drawn once and new animal effects. A documented simplification: there is no white-noise nugget separate from the OU term.

**Optional (TFT, Chronos-2, DeepAR):** `models/optional_adapters.py` only declares capabilities and checks
dependencies and download permission. All three report `not_run` in this build.

## Validation of predictions

These run on held-out test studies and a compound-holdout study:
- RMSE and MAE (log and grams; gram summaries come from sampled exp(Z), not exp of the mean).
- 80/95% coverage and width, CRPS, and the trajectory energy score.
- Group-mean and contrast coverage, the between-animal variance ratio, and increment SD ratio.
- Residual correlation by gap, and coverage of the extrema.
- PIT and sex-by-horizon tables.

Study-balanced and observation-weighted summaries and effective sample sizes are reported. The early-history
task (conditioning on DAY ≤ 14) is evaluated separately and only for the Gaussian-layer models.

## Design simulation (ADEMP)

- **Aims:** find group sizes, allocations and analytical schedules that meet prespecified power, FWER,
  coverage, bias, failure-rate and precision criteria.
- **Data-generating mechanisms:** Proposal uses each response model as the simulator. The final
  evaluation uses fresh batches from the proposing model, the **independent generator B** (Gompertz growth,
  Hill/tolerance effect, random-walk plus t noise, dose-dependent variance), stress scenarios (covariance shift,
  higher heterogeneity, delayed and transient effect shapes, heavy tails) and the synthetic oracle (generator A
  truth, labelled as a synthetic reference).
- **Estimand:** Δ_g(T), fixed across candidates.
- **Methods:**
  - Endpoint ANCOVA is the **production default** (always endpoint-only, so it cannot gain
    from intermediate visits). Locked complete and locked MCAR diagnostics on the small
    matrix are recorded under `docs/remediation/`; they are not design qualification.
    See `docs/remediation/analysis_closeout_20261022/CLOSEOUT.md`.
    An exploratory B=600 (seed `20270401`) and locked B=2000 (seed `20270501`)
    complete-data stress of that default on the deferred mechanisms
    (covariance shift, heavy tails, delayed alternatives) met sourced
    inference tolerances for ANCOVA at both sizes (6/6; no A flip).
    Exploratory C1 delayed at \(n=6\) missed FWER 0.067; locked C1 flipped
    to a pass (0.060). That flip was reported, not used to promote or retune
    C1. GLS was not restored. Those streams are not design qualification.
  - Repeated-measures GLS remains **implemented and requestable**, not a production default.
    Time-saturated intercept, sex and baseline terms, and a group basis
    ψ(t) = [t/T, (t/T)²] (so Δ_g(T) = Σγ_g). Historical plug-in working covariance
    (the spline's covariance), scaled by an estimated factor, model-based variance
    (CR1 is available) and t tests on df = N_animals − (3 + G·k). Locked streams
    missed FWER and coverage under slope heterogeneity ×1.6; weekly transient bias
    remains ~0.016.
  - Holm correction across active−control comparisons.
  - Opt-in experimental `mmrm` (`mmrm_visit_contrast`): visit-saturated treatment means and an
    explicit endpoint contrast, with historical plug-in \(V_0\) weights and animal-level CR1
    sandwich SEs. It is **not** a default production analysis and was **not promoted**.
    Synthetic complete-data (exploratory B=600, locked B=2000) and MCAR / design-dependent
    (exploratory B=600, locked B=2000) diagnostics are recorded under `docs/remediation/`.
    Locked MCAR found one C1 miss (biweekly \(n=8\) FWER point 0.0675). C1 was not retuned.
    Those streams are not design qualification. Secondary contrasts are formed only from
    scheduled visits; missing query days are marked not estimable rather than interpolated.
    See `docs/remediation/final_analysis_proposal_20261022_c1/DECISION_MEMO.md`.
- **Performance:** per-comparison, any and all-required power under the specified effects (conditional power),
  FWER under the global null and every partial null, bias, CI coverage and width, and failure rate. All use
  Clopper-Pearson intervals and MCSEs. "Assurance" uses model-predicted effects and is reported separately.

Complete latent paths are simulated on the union grid **before** schedules, sample sizes and missingness
(missed visits and dropout drawn independently of \(Z\); MCAR / design-dependent, not prior-outcome MAR)
are applied. Candidates use nested animal subsets and visit
subsets of the same paths (common random numbers). Specified-effect scenarios add δ·s_g(t) to control
paths, with s_g(T) = 1 exactly. Failed analyses count as non-rejections and are tabulated.

**Feasibility** requires all of the following:
- power ≥ target, with the Clopper-Pearson lower bound ≥ target − tolerance;
- FWER point ≤ α + tolerance, with the upper bound ≤ α + upper tolerance;
- coverage ≥ minimum;
- |bias| ≤ fraction × |δ|;
- failure rate ≤ maximum;
- precision (if set).

The Pareto frontier is over animals and analytical measurements. Finalists are the cheapest feasible designs plus the
cheapest feasible ANCOVA comparator. A design is **recommended only if it remains feasible on the fresh
proposing-model batch AND under generator B**. Stress and oracle failures are listed next to each
recommendation. A resource difference is reported only if the comparator also meets every requirement.
`no_feasible_design` (nothing meets the criteria) and `not_supported` (applicability gates fail) are
distinct, valid outcomes.

Welfare: schedules are analytical weighings only. Protocol-required welfare weighings and clinical
observations are outside the optimization and are never reduced.

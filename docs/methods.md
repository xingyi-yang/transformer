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
selected on the validation studies. The mean target is `Y = LOGW − BASELINE_LOGW`;
`predict_mean` reconstructs `LOGW = baseline + Ŷ`. The covariance layer is fitted by maximum likelihood (L-BFGS-B, batched
Woodbury) to **study-level out-of-fold residuals** on the training studies (not generator truth):
`Cov = B(t)Ω_s B(t)' + B(t)Ω_a B(t)' + σ_p² exp(−|t−t'|/ℓ) + σ_m² I`, with `B(t) = [1, t/28]`
and diagonal Ω. Official target-only fit: σ_p ≈ 0.027 vs generator 0.022, ℓ ≈ 4.19 vs 6,
σ_m ≈ 1×10⁻⁴ (optimizer floor; animal intercept absorbs measurement). Parameter uncertainty comes from study-bootstrap refits of the mean. **One member is applied
to a whole simulated study.** Covariance parameters are not bootstrapped (documented limitation).

**C. Baseline-conditioned transformer (custom; not a TFT).** Context tokens (biology, baseline,
compound descriptors, planned regimen) and query tokens (time and planned-exposure features at that
time). A boolean attention mask lets each query see only the context tokens and itself, so predictions
do not depend on which other times are queried (schedule invariance, which is tested). The defaults are d_model 32,
2 heads, 1 layer, dropout 0.1, AdamW, a loss on observed targets only with study-normalized weights,
validation early stopping, and bootstrap members. Feature scales are fit on **train only**.
`predict_mean` reconstructs `LOGW = baseline + y_scale · net`. For the primary backbone comparison it **uses the spline's
covariance layer**. Official cell `compound_ids` are the training compounds CMP03–06;
evaluation studies SYNA-010/011 were not in the checkpoint. Those test studies reuse
train compounds (CMP03 / CMP06), so official test is a study holdout, not a new-compound holdout.

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

**Optional (TFT, Chronos-2, DeepAR):** never required, never registered as `transformer`, and never
used as a design simulator. Chronos-2 and DeepAR still only declare capabilities and stay
`not_run`. TFT is the documented early-history forecasting adapter (`models/tft.py`,
`send-bw tft-forecast`) for the **secondary early-history task only**. It is built with
`TimeSeriesDataSet` (observed rows only; `allow_missing_timesteps=True` so irregular calendar
days are acceptable, but unobserved `LOGW` is never filled with 0) and
`TemporalFusionTransformer.from_dataset` on CPU. Missing static categoricals use the token
`UNK` at train and predict time (NaN is not the string `"nan"`). The default target scale is
a SPECIES-grouped `GroupNormalizer` when more than one species is present, otherwise a global
standardizer; `transformation=None` because `LOGW` is already `ln(grams)`. PyTorch Forecasting
only looks up group scales through `group_ids`, so mixed-species fits use
`group_ids=["SPECIES", "USUBJID"]` (each animal remains its own series). MEAN is the
0.5 quantile; VAR is a quantile-spread approximation. Pooled pretraining can continue on the
target species via `from_parameters` so donor SPECIES classes are kept.
An 80% Winkler interval score is reported alongside pinball and coverage.
CQR (experimental) adds a single margin to `q0.1`/`q0.9` and leaves `q0.5`
unchanged (Romano-style residual `max(q0.1−y, y−q0.9)`, finite-sample
`k = ceil((n+1)(1−α))`, `Q = E_{(k)}`; if `k = n+1` the code uses the
sample max instead of `+∞`). A previous helper used
`np.quantile(k/n, method='higher')`, which is one rank larger than
Romano (NumPy indexes `q(n−1)`). Locked CQR artifacts were left in
place; Romano-rescored metrics are saved under
`runs/forecast_method_audit/`. The margin is pair-level
`(USUBJID, DAY)`, not study-balanced. Study-held-out CV fits it on a
**calibration** split. The earlier stability run fit it on official
**validation**. Neither uses test outcomes. Pinball computed on CQR
endpoints is not the TFT QuantileLoss; raw and calibrated tables are
stored separately. PF training `QuantileLoss` is `2 × pinball` on **raw** `LOGW`
(confirmed by replaying one locked target-only batch: decoder `y` and
`transform_output(pred)` both sit near 5.8; `GroupNormalizer`
center ≈ 5.616 / scale ≈ 0.245 standardizes the head only).
A previous note that the loss used the normalized target was wrong.
Reported `mean_quantile_loss` is unscaled (1×) pinball on raw `LOGW`.
`FeatureSpec.check_context` is not enforced at TFT score
time (official target-only train never saw interval 7 or dose 1000;
validation SYNA-005 and eight SYNA-010 test animals still score).
Target-only is the development reference; fine-tune and CQR are not a
new default. Fine-tune keeps the pooled `FeatureSpec` and categorical
classes via `from_parameters`.
Confirmation split 34011 is not an untouched holdout: SYNA-007/010/011 were
in its training set. Lightning `epochs_run` in older histories is
`current_epoch + 1` (a 16-epoch budget was stored as 17). TFT series counts
exclude animals with fewer than `min_encoder_length + 1` observed visits
(official train: 214 animals, 211 series). Without `send-bw[tft]`
it stays `not_run`. It refuses baseline-only / pre-study prediction and `sample_study`.
Installing the extra is not part of the default environment and is not a scientific-readiness
claim.

Quantile CatBoost (extension baseline) fits **one `Quantile:alpha=q` head per
level**, not CatBoost MultiQuantile. Features are encoder summaries plus known
future calendar (`target_day`, elapsed days). `y_logw` is the label.

Chronos-2 official_seq used `history_only_univariate`: the model sees visit-rank
`LOGW` with no elapsed-day index. Two series with the same visit count and
different calendar spacing are indistinguishable. That is an **adapter**
limitation. `covariate_aware_origin` can pass `DAY` as a covariate; the
sequence index is still unit steps. Official_seq did not use that mode.

## Synthetic generator (official bank)

Official forecasting data are **Generator A** only. Growth is one saturating
exponential with shared `tau_g = 40`. Generator B (Gompertz, Hill/tolerance,
random walk, t-noise) is a design-stress mechanism, not the official bank.
The official holdout is 12 RAT + 3 DOG studies (`generate_seed=71001`).
`send-bw tft-larger-bank` generates a separate 36 RAT + 6 DOG bank
(`generate_seed=81011`) and reruns last_value / target-only / pooled /
fine-tune with the locked TFT knobs. That bank is not a replacement for
the official holdout and is not used to retune.
Default cohort selection keeps only post-baseline `TREATMENT` weights, so
the DAY≤14 encoder often has two weekly visits (7, 14) or four
twice-weekly visits (3, 7, 10, 14). Generator A still records predose
visits at `DAY` −7 and 0. `send-bw tft-longer-ctx` reattaches those
observed rows as encoder context on the same splits and cutoff; it does
not invent days and does not add recovery visits.

`send-bw rich-hist` is a **new** Generator A bank (`runs/rich_hist_v1/`,
`generate_seed=91021`), not a correction of the locked 71001 / 81011
experiments. Study duration, measurement schedule, treatment duration,
and recovery duration are configurable. 28-day studies keep the official
`t/28` slope basis. ~90-day studies use a saturating slope basis
`(1-exp(-t/τ_g))/(1-exp(-28/τ_g))` so random slopes do not grow
mechanically as `t/28`. Weekly / twice-weekly / dense calendars are
**experimental scenarios**. No signed historical authorization exists;
do not treat them as empirical protocol estimates.
`CohortConfig.include_pretreatment` (default false) can keep observed
predose rows in the modelling table. Forecast windows record origin,
last observed day, encoder count, encoder span, target day, lead from
origin, and lead from last observation. A short-history (last 2 visits)
table is written for later ablation. If a model cap truncates the
encoder, excluded visits are counted. Study-level splits are
train / validation / calibration / test; paired dense/sparse siblings
that share a latent path stay in one partition. FeatureSpec is fit on
training studies only. This command does not train models.

`send-bw rich-hist-cmp` is a **new** versioned experiment
(`runs/rich_hist_cmp/`, `train_seed=93041`) on the frozen bank. It does
not regenerate or retune trajectories. Preflight checks trajectory-family
independence (dense/sparse siblings share a latent path), reconciles the
1,338-vs-1,332 animal inventory, and records actual consumed encoder
lengths before any model is fit. Full-history uses
`max_encoder_length=40` so the bank's available visits are not silently
cut at 8. Variants A–D plus last-value and the existing spline share the
same eligible `(USUBJID, origin, target_day)` keys. B–D keep identical
baseline `ZB_C` and are separately trained TFTs; inference-only
truncation of one checkpoint is not the experiment. Primary metric is
study-balanced MAE on LOGW. Pair-level CQR is not a conformal guarantee.
A null is not proof that history is useless in real studies; a positive
is not real-data validation.

Synthetic methodological benchmark. Not real-data validation or
operational qualification.

`D1`, `D2`, `D3` are curated **production** descriptors in
`metadata/compound_reference.csv`. They are not SEND domains. A comparable
table must be supplied at prediction time; missing descriptors on an active arm
raise `UnsupportedContextError`. Hidden compound truth (`A`, `ED50`, `tau_on`,
`tau_off`, and the noisy link from D1–D3) lives only in `truth/params.json` and
is not a feature.

The latent weight path uses the **actual** regimen after dose deviations
(`p_dose_deviation = 0.04`: half dose or early stop). Production features use
the planned regimen only (`ACTUAL_DOSE` is `truth_only`; `EXDOSE_ACTUAL` is
optional / post-hoc).

Missed visits (`p_missing = 0.02` if `day > 0`) and death (`p_death = 0.015`,
uniform day) are iid and independent of treatment and weight (MCAR-ish, not
informative dropout).

## Validation of predictions

These run on held-out test studies and a compound-holdout study:
- RMSE and MAE (log and grams; gram summaries come from sampled exp(Z), not exp of the mean).
- 80/95% coverage and width, CRPS, and the trajectory energy score.
- Group-mean and contrast coverage, the between-animal variance ratio, and increment SD ratio.
- Residual correlation by gap, and coverage of the extrema.
- PIT and sex-by-horizon tables.

Study-balanced and observation-weighted summaries and effective sample sizes are reported. The early-history
task (conditioning on DAY ≤ window) is evaluated separately. Gaussian-layer models condition through
the shared covariance and also report a pre-study comparison. The optional TFT adapter, when the
extra is installed and trained, is scored on the conditioned task only; its pre-study comparison
is `not_supported` and is not fabricated.

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

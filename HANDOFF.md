# HANDOFF

Status: the required pipeline (synthetic generation → harmonization → QC/splits → spline, transformer and
Bayesian training → held-out evaluation → design proposal → independent finalist evaluation → report)
**has been executed end to end on synthetic data**. All findings are a **synthetic proof of concept**. The
design results come from low-replication smoke settings and are **nonconclusive software checks**.
Real-data qualification stage: **1** (implemented and mock-tested only).

## 1. Implemented

- **Data**:
  - `SendSource` interface, with `LocalSendSource` (CSV/Parquet; XPT through the optional pyreadstat; per-study or
    combined layout; alias and case normalization; ambiguity and collision detection).
  - `SnowflakeSendSource`: disabled unless authorized; `env:` credentials; allowlisted tables and columns; validated
    identifiers; parameterized filters; `AT(TIMESTAMP)` snapshots; fetchmany batching; query-ID extraction log.
  - Mapping profiles with study overrides.
  - `Harmonizer`: units, the no-day-zero rule, partial dates, SUPPBW fasting, dispositions, post-death quarantine,
    duplicates, curated regimens, baseline rule, phases, dosing deviations.
  - Canonical schemas with key and row-inflation checks.
  - QC and cohort selection; frozen study, chronological and compound-holdout splits; feature availability
    registry with leakage checks.
- **Synthetic data**: generator A (the spec's structure) and an independent generator B. The hidden truth is read only by
  `simulation/truth.py`.
- **Models**:
  - Hierarchical spline (ridge, OOF-residual covariance ML, study bootstrap).
  - Custom baseline-conditioned transformer (schedule-invariant masking, shared covariance).
  - PyMC Bayesian model (QR, ZeroSumNormal non-centered effects, OU residual, prior and posterior predictive checks,
    sensitivity runs, diagnostics-based reliability, new-study draws with new effects).
  - Optional TFT: verified PyTorch Forecasting early-history adapter (`models/tft.py`). Stays
    `not_run` without `send-bw[tft]`. Refuses baseline-only and design use; never registered as
    `transformer`. Chronos-2/DeepAR still only check capabilities and report `not_run`.
- **Design**:
  - Request schema with applicability gates (`not_supported`).
  - Production default: endpoint ANCOVA. Repeated-measures GLS remains requestable
    (model-based or CR1) and is not a default after locked A/B/C1 diagnostics.
    Opt-in experimental `mmrm`. Holm; the secondary weighted-average estimand
    (GLS/MMRM only, reported). See `docs/remediation/analysis_closeout_20261022/CLOSEOUT.md`.
  - Common-random-number trajectory simulator; scenarios (alternative, global and partial nulls, model-predicted
    assurance); Clopper-Pearson intervals and MCSEs.
  - Feasibility, Pareto frontier, finalists plus ANCOVA comparator, fresh, generator B, stress and oracle evaluation,
    recommendation rule, resource-claim rule, and `no_feasible_design`.
- **Delivery**: the CLI with all 12 commands, caching manifest, Markdown/HTML report with plots, and docs.

## 2. Executed and tested (exact results)

`pytest -q tests`: **106 passed** in 43 s. This includes the mock-Snowflake equivalence, messy-fixture, leakage, GLS
null-calibration (slow), integration CLI smoke and Bayesian smoke tests. `ruff check src tests`: clean.

**Smoke run** (`configs/smoke.yaml`, seed 20261017):
- **Data**:
  - 16 synthetic studies: 826 animals and 6,479 weights, with 0 quarantined.
  - 775 eligible animals and 3,941 modelling observations. Exclusions: 50 animals out of scope (DOG study) and 1 animal with too few post-baseline observations.
  - Recorded limitations: 3 weights with uncertain dates, and 31 EX deviations from plan.
  - Splits: train 9, validation 3, test 3, compound holdout 1 (CMP01).
- **Spline**: alpha 100; validation RMSE 0.0506 (log scale); out-of-fold RMSE 0.0701; covariance fit converged with no parameters at bounds.
- **Transformer**: 2 members, with validation RMSE 0.0498 and 0.0491.
- Held-out results on the 3 test studies:

| model | RMSE log | cov95 | group-mean cov90 | contrast cov90 | between-animal var ratio |
|---|---|---|---|---|---|
| spline | 0.084 | 0.914 | 0.68 | 0.75 | 1.30 |
| transformer | 0.065 | 0.963 | 0.89 | 0.58 | 1.47 |
| bayesian (bayesian_smoke run) | 0.072 | 0.859 | 0.69 | 0.67 | 1.08 |

  On the compound-holdout study (a single study): spline 0.044 / 0.992; transformer 0.066 / 0.955; Bayesian 0.086 / **0.654**.
  For early-history forecasting on the test split, conditioning reduced RMSE (log scale) from 0.100 to 0.038 for the spline and from 0.078 to 0.043 for the transformer.
- **Design** (`new_study.smoke.yaml`: −8% effects; 400 proposal and 600 final replicates):
  - Spline proposal: 6 of 8 candidates feasible. Transformer proposal: 5 of 8.
  - Both models recommend **endpoint ANCOVA with 10 animals per sex per group (60 animals)**. It was feasible on fresh batches, under generator B, under covariance shift, delayed-effect and heavy-tail stress, and against the oracle.
  - It was **not** feasible under higher heterogeneity (power 0.785) or a transient effect.
  - GLS weekly at 10 per sex per group was also feasible, but under the heterogeneity stress it was anti-conservative (FWER 0.10), and under the transient stress its bias was 21% of δ.
  - No resource-saving claim is made, because the recommended design is itself the endpoint comparator.
- **Model-predicted contrast at T vs the synthetic oracle (−15.5% / −22.2%)**: spline −3.4% / −9.0%; transformer
  −16.0% / −20.6%; Bayesian −7.7% / −14.6%.
- **Example request** (`new_study.example.yaml`, −5%, 40 candidates; run with `propose-design`): **no_feasible_design** under both
  models. The best power was 0.73 at 100 animals, against the 0.80 target within the 120-animal cap. All other criteria passed.

**Bayesian smoke** (`configs/bayesian_smoke.yaml`; 16 training animals per study; 4 chains; nutpie):

| attempt | settings | main R-hat max | bulk ESS min | tail ESS min | divergences | reliable |
|---|---|---|---|---|---|---|
| 1 | 1000 warmup / 1000 draws, target_accept 0.9 | 1.017 | 288 | 382 | 2 | no |
| 2 | 2000 / 2000, target_accept 0.95 (sensitivity runs 400/400) | 1.006 | 633 | n/r | 0 | yes* |
| 3 (final) | as attempt 2, sensitivity runs 1500/1500 | 1.006 | 633 | 963 | 0 | **yes** |

(*The attempt-2 main fit used the same settings and seed as attempt 3, which reproduced it exactly. Its tail ESS
was not recorded separately ("n/r"). Attempt-1 diagnostics are archived in
`runs/bayesian_smoke/models/bayesian_attempt1_diagnostics.json`.)
- The pivoted QR dropped 3 non-identifiable design columns.
- Prior predictive (2.5/50/97.5% change at the last observation): −45% / +9% / +116%, against observed +1% / +32% / +70%. Judged plausible.
- Posterior predictive: no extreme time bins.
- Sensitivity runs: prior ×0.5 was reliable; prior ×2 (R-hat 1.018), Student-t (1.011) and no-animal-intercept (1.021) were **not**. Predictions were stable across variants (mean day-28 change 0.294 to 0.295); removing the animal intercept doubled ℓ.
- **Bayesian design proposal**:
  - Proposal: 6 of 8 candidates feasible.
  - Its cheapest finalist (ANCOVA, 6 per sex per group) passed on fresh Bayesian batches but **failed under generator B**, so it was not recommended.
  - Recommended design: ANCOVA, 10 per sex per group.

## 3. Optional or incomplete

- TFT: adapter implemented (`models/tft.py`) but `not_run` here because `pytorch_forecasting` is
  not installed. Chronos-2 and DeepAR: `not_run` (capability declarations only; Chronos also needs
  download permission). The default environment must not install the TFT extra (smoke asserts
  optional status `not_run` while those deps are missing).
- XPT reading: the code path exists but was not exercised, because pyreadstat is not installed. A clear-error test exists.
- The Snowflake connector is not installed. The adapter is tested only with a mock connection.
- define.xml parsing: not implemented. A reviewed mapping profile is required instead.
- Research-scale configs (`research.yaml`, `bayesian_research.yaml`) were not executed. The Bayesian smoke run subsampled training animals.
- Early-history conditioning is not implemented for the Bayesian model (`NotImplementedError`, reported as not_run).
- Covariance-parameter uncertainty is not propagated for the spline and transformer. The GLS uses no Kenward-Roger correction.
- Informative termination: not modelled, and no claims are made.
- There is no git repository and git is not installed, so the manifest records the git revision as unavailable.

## 4. Synthetic findings (proof of concept only)

- Separating the proposal from independent evaluation matters. A Bayesian-proposed 36-animal design passed its
  own model but failed under generator B.
- In this setting, endpoint ANCOVA was as efficient as weekly GLS and more robust. The extra GLS visits did
  not buy power, and the fixed quadratic group basis plus the plug-in covariance were fragile under stress.
- The models disagree substantially about the effect of a known compound (the spline and Bayesian models
  under-predict it), so model-predicted "assurance" is not interchangeable with conditional power at the
  prespecified effects.

## 5. Real-data qualification still needed

Stages 2–5 in `docs/real_data_qualification.md`:
1. Authorized pilot extraction reconciled against source records.
2. Scientist review of the cohort, arm, regimen, compound and timing mappings, and of the baseline rule.
3. Held-out real-study evaluation.
4. Prospective shadow validation.

**No real data has been accessed and no real-SEND validation is claimed.**

## 6. Unvalidated statistical assumptions

These cover pooled-sex common effects, the GLS ψ(t) basis, plug-in working covariance and SE approximation,
MAR missingness and non-informative dropout, the illustrative generators, and Monte Carlo precision at smoke
replication. See `docs/limitations.md`.

## 7. Analysis close-out (synthetic; not design qualification)

Locked complete (seed `20261201`, B=2000) and locked MCAR 2%/2% (seed `20270201`, B=2000)
on the pre-specified 8×2 matrix:

- **A** (ANCOVA) met sourced inference tolerances. It is the production default.
- **B** (quadratic GLS) missed FWER and coverage under slope heterogeneity in every
  locked stream. It remains implemented and is **not** a production default.
- **C1** (visit-saturated MMRM) remains a frozen opt-in candidate. Locked MCAR
  biweekly at \(n=8\) missed the FWER point ceiling (0.0675). C1 was not retuned
  and was not promoted.

This is not design qualification and not an operational recommendation. Historical
smoke artifacts still searched `[ancova, gls]`; shipped request YAMLs now default
to `[ancova]`.

## 8. Suggested next steps

1. After authorization: a stage-2 pilot on a handful of studies with `validate-real`
   and record-level reconciliation. No real data has been accessed.
2. Deferred diagnostic cells remain deferred (covariance shift, heavy tails, delayed
   alternatives, 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2).
   Do not retune C1 or restore GLS as a default to chase a passing cell.
3. Propagate covariance-parameter uncertainty (bootstrap the covariance fit) for the
   spline and transformer if those generators are used in a later locked evaluation.
4. Research-scale configs (`research.yaml`, `bayesian_research.yaml`) remain unrun
   software-capacity work, not a substitute for real-data qualification.
5. TFT remains optional. Do not install `pytorch-forecasting` into the default
   environment, do not treat a local extra-enabled fit as design qualification, and
   do not implement Chronos-2/DeepAR unless they are named. The analysis chapter
   stays as written.
6. v3 stages 1–2 **software** is in the tree (frozen target manifests,
   nested donor-only train additions, explicit `SPECIES_*`, sign-defined
   transfer table, prespec in `src/send_bw/pilot.py`). Selective sharing
   remains required later and is deferred. The first scored task is
   predose-conditioned prediction. `configs/v3_pilot.smoke.yaml` is
   proposed for review and has **not** been executed. Do not enable design
   recommendations. Do not access real data. v2 defaults are unchanged.

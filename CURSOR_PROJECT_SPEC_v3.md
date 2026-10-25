# Cursor Project Specification v3 (PROPOSED)

## Nonlinear multi-species information sharing for body-weight prediction

**Status: proposed scientific text plus executed synthetic method checks.**
Stages 1–2 infrastructure and the frozen k=0/k=1 predose pilot are in the
repository. The v3 evaluation benchmark (selective v1, sparse history,
donor heterogeneity) has been executed under `runs/v3_eval/`. This file
does not replace `CURSOR_PROJECT_SPEC_v2.md`. Runtime defaults, shipped v2
configs, and the analysis close-out stay on v2.

**Do not** launch large training runs from this spec, add Chronos-2 / DeepAR,
or enable design recommendations as an acceptance criterion.

## 0. Relationship to v2

v2 remains the implemented system: SEND-like ingestion, one-species cohort
gates, spline / custom transformer / Bayesian benchmarks, pre-study and
early-history evaluation, and an ADEMP design engine whose production analysis
is endpoint ANCOVA.

v3 **reuses** that infrastructure. It changes the **primary scientific
question**, not the file layout.

| | v2 (in force) | v3 (proposed) |
|---|---|---|
| Primary objective | One-species design feasibility for \(\Delta_g(T)\) | Target-species prediction under context-specific, pooled, and selective sharing |
| Other species | Exploratory; excluded at cohort | Donor studies; never silently mapped to the target |
| Success | Feasible design under locked analysis tolerances | Helpful vs harmful transfer on frozen target holdouts |
| Design engine | Implemented | Frozen / unused for v3 acceptance |
| Optional TFT / Chronos / DeepAR | TFT adapter; others stubs | No new families; TFT stays optional early-history only |

## 1. Objective and boundaries

Learn whether **nonlinear** sharing of longitudinal body-weight information
across species improves prediction for a **declared target species**, and
detect **harmful transfer**.

Body weight only. One study class unless a later amendment says otherwise.
Both sexes with explicit estimands. No total-program animal counts, no
reduction of welfare observations, no replacement of concurrent controls, no
unplanned adaptation, no regulatory-grade claims.

Synthetic work is a method check. It is not biological validation and not
real-data qualification.

### 1.1 Estimands (prediction, not design feasibility)

Primary predictive estimand, per held-out **complete study** of the target
species:

- Animal-level log-weight path after the information set of the stage
  (design-stage / predose / early-history)
- Study-balanced RMSE / CRPS on log weight
- Active−control contrast at fixed \(T\), \(\Delta_g(T)=\mu_g(T)-\mu_0(T)\),
  scored for bias and contrast CRPS when the holdout study has ≥2 animals
  per compared group

Secondary: gram-scale summaries from sampled \(\exp(Z)\), not \(\exp(\mu)\).

The v2 design-feasibility estimand (power, FWER) is **not** a v3 acceptance
metric.

### 1.2 Sharing regimes (required comparison)

Every experiment reports all three, or records `not_run` with a reason:

1. **Context-specific.** Train only on the target species. This is the v2
   control (current `cohort.species` filter).
2. **Pooled.** Train on target + declared donor species. Species is an
   explicit feature / token. Strain is nested in species. Unseen species are
   refused, not mapped.
3. **Selective-sharing.** A named subset of structure is shared (see §5).
   The shared subset is a factor in the report, not a hidden default.
   **Required in the eventual benchmark. Deferred in stages 1–2**
   (`sharing: selective` raises).

### 1.3 Prediction stages (required comparison)

Information sets are nested. A later stage may use everything a earlier
stage used, plus the new observations. No stage may use future actual doses
or synthetic truth.

1. **Design-stage.** Planned context and a **baseline distribution** (sex-
   specific). No animal-level predose weight.
2. **Predose-conditioned.** Observed predose / baseline log weight for the
   holdout animals (v2 “pre-study” evaluation, renamed for clarity).
3. **Early-history.** Observed post-baseline weights with `DAY ≤ window`.
   Empty encoder ⇒ `not_supported` for that animal, not a fabricated series.

### 1.4 Holdouts and experiments (required)

- **Complete-study holdouts.** The atom is `STUDYID`. No animal from a
  holdout study enters training transforms.
- **Fixed target train set.** Target-species training studies are frozen
  before donor studies are added (pilot protocol).
- **Sparse-history.** Early-history stage is repeated on a prespecified
  encoder grid (cardinality, last-visit-only, gap), not only day 14.
- **Harmful-transfer assessment.** For each sharing regime and stage,
  \(\Delta = \mathrm{score}(\text{with donors}) - \mathrm{score}(\text{context-specific})\).
  Sign convention: on RMSE/CRPS, \(\Delta>0\) is harmful. Report interval
  and the number of holdout studies. A single smoke seed is not a claim.

## 2. What v3 inherits unchanged

From v2, unless a stage below amends it:

- Source contract, harmonizer, canonical tables, leakage registry
- Generators A and B as **illustrative** DGMs; hidden truth isolation
- Custom transformer architecture (not a TFT); spline; Bayesian interface
- Frozen split hashing; IDs as grouping keys
- Honest `not_run` / `not_supported` / `no_feasible_design` vocabulary
- Real-data qualification stages 1–5; stage 1 is not readiness
- Synthetic labels: `Synthetic diagnostic.` `Not design qualification.`
  `No real-data validation.` `No operational recommendation.`

## 3. Data amendments (proposed)

### 3.1 Cohort

Add, without removing the v2 single-species filter:

```yaml
cohort:
  species: RAT                    # target species (estimand)
  donor_species: [DOG]            # empty => context-specific only
  study_classes: [REPEAT_DOSE_TOX]
  sharing: context_specific       # or pooled | selective
```

`donor_species: []` and `sharing: context_specific` must reproduce v2
exclusions. Production shipped configs stay in that mode until a labeled
pilot config is added.

### 3.2 Features

- Classify `SPECIES_*` as **production** only when `sharing` is pooled or
  selective. Context-specific runs must not include donor-species levels.
- Refuse unseen species. Do not map DOG → RAT or BEAGLE → SPRAGUE-DAWLEY.
- Keep planned-regimen features production; actual doses optional / post-hoc.
- Scale residuals in a species-aware way if a shared covariance is used.
  A single RAT-fitted OU reused on DOG grams is `not_supported` unless a
  documented species-scale is applied.

### 3.3 Splits

Keep study-level train/validation/test and compound holdout.

Add optional `test_target_holdout` (complete target-species studies) that is
frozen **before** donor inclusion. Donor studies may be added only to
`train` (or a declared `donor` split that is merged into train for pooled /
selective runs).

Chronological splits remain available for later real-data work. They are not
required for the synthetic pilot.

## 4. Prediction-stage protocols (proposed)

All three stages score the same holdout observations after the information
cutoff. Metrics stay separate (no mixing into one “accuracy”).

### 4.1 Design-stage

Input: holdout study’s species, strain, planned arms, sex allocation, and a
baseline distribution fitted **only on target-species training baselines**
(or the request distribution when scoring a hypothetical study).

The model draws or predicts paths **without** the holdout animals’ observed
predose weights. This is the hardest stage and the closest to “we have not
enrolled animals yet.”

### 4.2 Predose-conditioned (first pilot task)

Input: design-stage context plus each holdout animal’s observed baseline log
weight. This is today’s `evaluate_prestudy`. It is the **first** scored
pilot task. Rename in reports so it is not confused with design-stage.

### 4.3 Early-history and sparse-history

Input: predose plus observed rows with `DAY ≤ window`.

Sparse-history grid (minimum for a locked synthetic run):

| Encoder | Rule |
|---|---|
| W0 | predose only (equals predose-conditioned) |
| W1 | predose + first post-baseline visit |
| Wlast14 | predose + last visit with `DAY ≤ 14` |
| Wall14 | predose + all visits `DAY ≤ 14` (v2 default) |

Animals with no encoder under a rule are omitted and counted
(`n_animals_without_encoder`), not filled.

## 5. Models (proposed use of existing families)

Do **not** add optional families.

| Role | Model | Sharing variant |
|---|---|---|
| Hierarchical spline mean | v2 spline (ridge on a B-spline time/exposure basis with sex, baseline, strain, dose, compound, interval, route, recovery interactions; **not** linear-in-time) | Context-specific; pooled adds explicit `SPECIES_*` time-basis interactions |
| Neural mean | v2 custom transformer | Context-specific control; pooled = add species context token when `SPECIES_*` exist; selective = freeze/share named token groups (**required later, deferred**) |
| Hierarchical generator | v2 Bayesian | Context-specific default. Any species random effect is a **new labeled variant**, not the MAIN v2 model |

TFT remains optional, early-history only, not a sharing vehicle, not a design
simulator.

Selective-sharing v1 (the only selective variant in the first locked
synthetic stream):

- Shared: compound descriptors + planned-dose query features
- Species-specific: growth / baseline token group and, if present, residual
  scale
- Record the split in `training_history.json`

Further selective variants are amendments, not implied.

## 6. Harmful-transfer assessment (proposed)

Let \(S_0\) be the context-specific score on the frozen target holdout
(study-balanced RMSE log, primary). Let \(S_k\) be the same score after
adding \(k\) donor studies to train.

- \(\Delta_k = S_k - S_0\)
- Harmful if \(\Delta_k > 0\) by more than a prespecified MC noise band
- Report per stage and per sharing regime
- Also report contrast \(\Delta_g(T)\) bias change (sign: larger abs bias
  is harmful)

Do not average away a harmful target-species result with improved donor-species
in-sample fit. Donor likelihood is a diagnostic, not a success metric.

## 7. Historical data (proposed; no access in this packet)

See `docs/v3/historical_data_audit.md`.

No Snowflake connection and no real extract may be opened without written
authorization naming data, storage, and purpose. Until that exists, every
real-inventory field is **unknown**.

v3 real-data work starts at v2 qualification stage 2. Stage 1 remains
“implemented / mock-tested only.”

## 8. Pilot (proposed)

See `docs/v3/pilot_protocol.md`.

Hold the target-species training studies fixed. Add other-species studies
in a declared order. Score only the frozen target holdout. Synthetic first
(generator already emits one DOG study). Real data only after the audit
clears authorization.

## 9. Design engine

v2 design / `propose-design` stays in the repository.

v3 **does not** use it for acceptance, does not emit recommendations, and
does not treat a feasible ANCOVA design as evidence of useful sharing.

## 10. Tests and acceptance (proposed)

New tests (when implementation starts, not in this packet):

- Context-specific run with `donor_species: []` matches v2 cohort exclusions
- Pooled run includes donor studies in train and still refuses unseen species
- Selective-sharing history records which token groups were shared
- Fixed target train hash does not change when donors are added
- Harmful-transfer table is produced even when \(\Delta_k>0\)
- Sparse-history does not fill missing encoder rows
- Design recommendations are not written by v3 pilot configs

Acceptance for a **synthetic method pilot** (not qualification):

- All three sharing regimes run or are `not_run` with reasons
- All three stages scored on the same target holdout
- Harmful-transfer sign is reported, not patched
- Labels present: synthetic diagnostic; not design qualification; no
  real-data validation; no operational recommendation

Acceptance for **real-data sharing claims** requires qualification stages
2–4 and a scientist-reviewed species/strain map. That is not this packet.

## 11. Delivery

Handoff for any future v3 implementation must list: target species, donor
list, frozen target train/holdout hashes, sharing regimes run, stages run,
\(\Delta_k\) table, skipped items, and qualification stage. Do not call
sharing “validated” from synthetic DOG studies.

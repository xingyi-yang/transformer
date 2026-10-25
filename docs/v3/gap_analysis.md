# Gap analysis: nonlinear multi-species information sharing

**New goal (this request):** the primary scientific question is whether a
nonlinear model can share body-weight information across species without
harmful transfer, and under which sharing rule.

**v2 goal (repository):** one-species (RAT) pre-study modeling plus
simulation-informed design. Other species are generated, then dropped.

This file is analysis, not an implementation. Distinctions:

- **Repository evidence** — what the tree does today
- **Gap** — what the new goal needs
- **Proposed** — v3 language only; not present in runtime code

## 1. Objective inversion

| | v2 (evidence) | New goal (proposed) |
|---|---|---|
| Primary question | Do allocations/schedules meet power, FWER, precision for \(\Delta_g(T)\)? | Does sharing other-species studies improve **target-species** prediction, and when does it hurt? |
| Species | `cohort.species: RAT`; DOG → `species_out_of_scope` (`qc.py`, smoke 50 animals) | Target species stays the estimand; other species are **donors**, not excluded by default |
| Design recommendations | Implemented ADEMP engine | **Out of scope for v3 work.** Do not enable or rank designs as the success criterion |
| Optional model families | TFT adapter; Chronos/DeepAR stubs | Do not add families. Reuse spline / custom transformer / Bayesian |

The design engine can stay compiled. It is not the acceptance surface for the
new objective.

## 2. Sharing regimes (all missing except context-specific)

The new goal needs three named comparators on the **same** target-species
holdout.

### Context-specific (exists)

Train only on the target species. This is today's pipeline:

- `CohortConfig.species` filters animals (`qc.py`)
- `FeatureSpec.species` is a single string; `check_context` refuses others
- `validate_request` refuses `req.species != spec.species`

This is the **control** arm of a sharing experiment, not a finished answer.

### Pooled (missing)

Train on target + donor species with an **explicit** species factor (and
strain nested in species). Nonlinear models may learn species-dependent
growth while sharing compound/regimen structure.

Gaps:

- `FeatureSpec.animal_cols` has no `SPECIES_*`
- Unseen species are refused rather than declared as a sharing experiment
- Splits never include donor studies in `train` while holding out target studies
- A pooled covariance that ignores species scale (grams, growth rate) would
  be scientifically false; no species-scaled residual layer exists

### Selective-sharing (missing)

Share a **named subset** of structure and keep the rest species-specific.
Examples to compare, not implement all at once:

1. Share compound-descriptor effect; species-specific growth
2. Share residual/OU dynamics; species-specific mean
3. Share nothing but the architecture (same net, separate weights)
4. Share a learned species embedding with a transfer-risk penalty

No such API exists. Bayesian v2 explicitly forbids automatic historical
borrowing in the MAIN benchmark (`CURSOR_PROJECT_SPEC_v2.md` §8). Selective
sharing is a **new**, separately labeled project, not a silent Bayesian prior.

## 3. Prediction stages (partial)

The new goal needs three **scored** stages on complete held-out studies.

| Stage | What the predictor may see | v2 evidence | Gap |
|---|---|---|---|
| Design-stage | Species, strain, sex mix, planned regimen, **baseline distribution** (no animal-level predose) | `sample_study` + request `baseline_distribution` for design Monte Carlo | Not scored as a predictive protocol on holdout studies. No table of design-stage RMSE/CRPS vs later stages |
| Predose-conditioned | The above + **observed predose / baseline log weight** for those animals | `evaluate_prestudy` (v2 “pre-study”) | Name collision: v2 “pre-study” mixes design intent with predose-conditioned eval. Need an explicit predose protocol |
| Early-history | Predose + observed `DAY ≤ window` | Gaussian `_condition`; TFT adapter (not_run) | No sparse-history grid (1 visit, last visit only, irregular gaps). Bayesian early-history missing |

Proposed rule: never fabricate encoder history. If a stage forbids a feature,
the model must refuse or use only allowed inputs.

## 4. Holdouts and experiments (partial / missing)

| Experiment | Evidence | Gap |
|---|---|---|
| Complete-study holdout | Whole `STUDYID` splits | Keep. Add **target-species** complete-study holdouts that stay fixed while donors are added |
| Compound holdout | Implemented | Orthogonal; keep. Do not treat compound holdout as species transfer |
| Chronological holdout | Code, untested | Needed later for real data; not the synthetic pilot’s first split |
| Sparse-history | Single window day 14 | Factorial: encoder cardinality × gap × window day, target-species holdout only |
| Harmful transfer | None | Primary metric: change in target-species holdout score when donor studies are added to a **fixed** target train set. Positive = helpful; negative = harmful |

## 5. Nonlinear models: reuse, do not replace

The custom transformer is the v2 nonlinear mean. It is baseline-conditioned
and schedule-invariant. It has **no species token**. Pooling it today would
either drop DOG (current) or silently treat BEAGLE as an unseen strain
(refused).

Gaps for a transformer sharing experiment:

- Add a species context token **only** in a labeled sharing variant
- Keep the v2 no-species transformer as the context-specific control
- Do not rename it TFT and do not route design through TFT
- Shared covariance with the spline is a v2 mean-function comparison. A
  pooled residual that is not species-scaled is a new risk; treat covariance
  sharing as its own factor

The implemented spline mean (`models/spline.py`, `models/design.py`) is
ridge-penalized and expanded on a B-spline time basis (degree 3 when
`rich`) interacted with sex, centered baseline, and strain, plus
planned-exposure and recovery bases interacted with log-dose, descriptors,
interval, route, and sex. After that expansion it is linear in
coefficients. It is **not** a linear-in-time model. Do not use it as a
stand-in for “the linear model.” The custom transformer is the neural
mean. Bayesian remains a hierarchical generator; any species hierarchy is a
new model, not a default.

## 6. Synthetic DGM vs real transfer

Repository DGM already has different RAT vs DOG `gmax` and baseline
(`generate.py`). That is enough for a **software-and-method** pilot
(see `pilot_protocol.md`). It is **not** biological transfer.

Gaps before any real-data claim:

- Authorization (unknown)
- Inventory of real species, strains, study class, SEND version (unknown)
- Whether donor species have overlapping compounds, routes, and horizons (unknown)
- Scientist-reviewed mappings (stage 3; not started)

## 7. What must not change in the first increment

- Do not overwrite `CURSOR_PROJECT_SPEC_v2.md`
- Do not relax v2 species refusal in **production** configs
- Do not enable `propose-design` recommendations as v3 acceptance
- Do not install Chronos/DeepAR or promote TFT to a required benchmark
- Do not treat unit tests or smoke metrics as scientific readiness
- Do not silently map unseen species to RAT

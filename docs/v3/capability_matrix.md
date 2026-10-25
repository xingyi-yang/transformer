# Capability matrix (repository evidence)

Statuses:

- **implemented** — code exists and is backed by a test and/or an executed artifact
- **partial** — code exists but is incomplete relative to v2 wording or the new goal
- **missing** — no implementation
- **unverified** — code or template exists, but this repository has no successful
  execution, no required extra, or no authorized real data

All findings below are from the current tree. No Snowflake connection and no
real SEND extracts were used. Synthetic smoke artifacts under `runs/` are
software checks, not scientific readiness.

## A. Pipeline and data contract

| Capability | Status | Evidence |
|---|---|---|
| Local SEND source (CSV/Parquet; per-study or combined) | implemented | `src/send_bw/data/sources.py` (`LocalSendSource`); `tests/unit/test_data_contract.py`; messy fixture `tests/fixtures/messy_send.py` |
| SAS XPT reader | unverified | Code path + `OptionalDependencyError` in `sources.py`; `test_xpt_optional_dependency_message` only asserts the missing-extra message. `pyreadstat` is not installed here |
| Mapping profiles + study overrides | implemented | `src/send_bw/data/metadata.py`; `configs/mappings/example_send.yaml`; mapping report in `runs/smoke/qc/` |
| define.xml parsing | missing | `docs/limitations.md`; no parser in `src/` |
| Harmonizer (units, no-day-zero, baseline, quarantine) | implemented | `src/send_bw/data/harmonize.py`; unit/day tests in `tests/unit/test_data_contract.py` |
| Canonical schemas + join-inflation checks | implemented | `src/send_bw/data/schema.py`; leakage/join tests in `tests/unit/test_generator_splits_leakage.py` |
| Snowflake adapter (read-only, allowlisted) | unverified | `SnowflakeSendSource` in `sources.py`; mock tests `tests/unit/test_snowflake_mock.py`; `configs/real_snowflake.example.yaml` has `authorized: false`. No live connection in this repo |
| Real-data qualification stage 1 | implemented | `docs/real_data_qualification.md`; `Run.validate_real`; HANDOFF stage **1** |
| Real-data stages 2–5 | missing | Checklist unchecked; HANDOFF: no real data accessed |
| Feature availability registry / leakage gate | implemented | `src/send_bw/data/features.py` `FEATURE_REGISTRY`; `tests/unit/test_generator_splits_leakage.py` |
| Species stored on canonical tables | implemented | `schema.py` requires `SPECIES` on `study_context` and `animal_context`; generator writes RAT + DOG |
| Cohort **excludes** non-target species | implemented | `src/send_bw/data/qc.py` `species_out_of_scope`; `pipeline.py` filters `SPECIES == ccfg.species`; smoke report 50 DOG animals excluded (`runs/smoke/report/report.md`) |
| Species as a model **feature** (pooling) | implemented | `SPECIES_*` in `FeatureSpec` when `sharing: pooled` and >1 training species; `MeanDesign` time-basis interactions; transformer `species` token only if those columns exist. Context-specific runs have no `SPECIES_*`. Unseen species still refused |
| Request refuses other species | implemented | `simulation/designs.py` `validate_request`; `tests/unit/test_design_workflow.py` DOG → `not_supported`; integration smoke same |

## B. Synthetic generators

| Capability | Status | Evidence |
|---|---|---|
| Generator A (spec structure) | implemented | `src/send_bw/data/generate.py` `GeneratorA`; smoke `synthetic.generator: A` |
| Independent generator B | implemented | `GeneratorB`; used in design finalist evaluation (`simulation/workflow.py`) |
| Hidden truth isolated from models | implemented | Truth under `synthetic/truth/`; `simulation/truth.py` is the only reader; leakage tests |
| Other-species studies generated | implemented | `n_other_species_studies` default 1; `species = "DOG"` after RAT studies; species-specific `baseline` / `gmax` / `STRAINS` |
| Other-species used in **fitting** | partial | Default v2 configs still drop DOG. Pooled cohort + nested donor-only train additions are implemented (`qc.py`, `splits.py`, `pipeline.prepare`). No training wave has been run |
| Biologically validated multi-species DGM | missing | Generators are labeled illustrative (`generate.py` docstring; `docs/limitations.md`) |
| Informative termination estimand | missing | MAR missed visits + non-informative dropout only; limitations document this |

## C. Splits and holdouts

| Capability | Status | Evidence |
|---|---|---|
| Frozen whole-study splits | implemented | `src/send_bw/data/splits.py`; tamper hash; smoke `train 9 / val 3 / test 3 / compound 1` |
| Compound holdout | implemented | `make_study_splits` + `compound_holdout_auto` in `pipeline.py`; `test_compound_holdout_split` |
| Chronological split | partial | `method: chronological` implemented in `splits.py`; example configs set it (`real_local.example.yaml`). **No unit test** named chronological |
| Complete-study holdout (study as atom) | implemented | Splits assign whole `STUDYID`s; `assert_disjoint` |
| Leave-one-species / cross-species holdout | partial | Frozen target train/validation/test manifests; donors may enter train only. No `test_species_holdout` split type |
| Sparse-history experimental protocol | missing | Early-history window is a single eval knob (`early_history_window_day: 14`). No encoder-sparsity design |

## D. Required models (v2)

| Capability | Status | Evidence |
|---|---|---|
| Hierarchical spline + OOF covariance | implemented | `models/spline.py`; `tests/unit/test_models.py`; `runs/smoke/models/spline/` |
| Custom baseline-conditioned transformer (not TFT) | implemented | `models/transformer.py`; schedule-invariance test; shared cov required; smoke members |
| Shared spline covariance for transformer | implemented | `pipeline.py` injects `shared_covariance`; `test_transformer_requires_shared_covariance` |
| Bayesian hierarchical (PyMC / nutpie) | implemented | `models/bayesian.py`; `tests/integration/test_bayesian_smoke.py`; `runs/bayesian_smoke/` (main reliable at 2000/2000; 3/4 sensitivity variants failed) |
| Bayesian research-scale fit | unverified | `configs/bayesian_research.yaml` **not executed** (README / HANDOFF) |
| IDs not predictors; unseen strain refused | implemented | `test_ids_are_not_predictors`; `test_unseen_biological_category_is_refused` |
| Historical borrowing / robust MAP | missing | v2 §8: not in MAIN benchmark. No MAP / power-prior / species-hierarchy code |

## E. Optional sequence models (v2)

| Capability | Status | Evidence |
|---|---|---|
| TFT early-history adapter | partial | `models/tft.py` + `verify_tft_api()`; unit gates in `tests/unit/test_optional_tft.py`. Default env: `not_run` (`runs/smoke/models/optional_models_status.json`). Fit path skipped without `pytorch_forecasting` |
| Chronos-2 | missing | Catalog + dep/download gate only (`optional_adapters.py`) |
| DeepAR | missing | Same |

v3 does **not** add these families. TFT remains optional and early-history only.

## F. Prediction tasks

| Capability | Status | Evidence |
|---|---|---|
| Design-stage simulation (sampled baselines, planned regimen) | implemented (as design engine) | `sample_study` + `simulation/workflow.py`; request `baseline_distribution` in `configs/new_study.*.yaml` |
| Design-stage **prediction scoring** as a holdout protocol | partial | Design scores power/FWER on simulated studies, not predictive scores of held-out real/synthetic studies under a sampled-baseline protocol |
| Predose-conditioned new-study prediction | implemented (as v2 pre-study eval) | `evaluate_prestudy` uses observed test-study baselines; no post-baseline outcomes in features |
| Distinct predose-only vs design-stage comparison table | missing | Both exist as different pipeline stages; they are not a paired scientific protocol |
| Early-history (Gaussian condition) | implemented | `gaussian_layer.py` `_condition`; `evaluate_early_history`; `test_early_history_conditioning_reduces_error`; smoke RMSE drop documented in HANDOFF |
| Early-history (Bayesian) | missing | `NotImplementedError` in `bayesian.py`; report `not_run` |
| Early-history (TFT) | unverified | Adapter present; extra not installed; no trained TFT artifact |
| Fabrication of baseline-only TFT | implemented (refused) | `TFTEarlyHistoryModel.predict_mean` / `sample_study` raise; eval marks pre-study `not_supported` |

## G. Analysis and design (v2; not the v3 objective)

| Capability | Status | Evidence |
|---|---|---|
| Estimand \(\Delta_g(T)\) | implemented | `docs/methods.md`; `simulation/analysis.py` |
| Endpoint ANCOVA default | implemented | Request YAMLs `analyses: [ancova]`; close-out `docs/remediation/analysis_closeout_20261022/CLOSEOUT.md`; locked A-stress 6/6 |
| GLS requestable, not default | implemented | Locked miss under slope heterogeneity; remains implemented |
| MMRM C1 opt-in, frozen | implemented | `mmrm_visit_contrast`; locked MCAR FWER miss 0.0675; not promoted |
| ADEMP design search + independent finalists | implemented | `simulation/workflow.py`; smoke design artifacts. **Nonconclusive** (400/600 reps) |
| Design recommendations as operational advice | implemented as software; **disabled for v3 work** | This packet does not turn them on or treat them as the new objective |

## H. New-goal capabilities (not in v2 as primary)

| Capability | Status | Evidence |
|---|---|---|
| Context-specific (target-species-only) fit | implemented | Current default: RAT-only cohort |
| Pooled multi-species fit | partial | Species feature + nested donor train composition implemented. No pooled model has been trained |
| Selective-sharing (share some layers, not others) | missing | **Required in the eventual benchmark. Deferred.** `sharing: selective` raises |
| Harmful-transfer assessment | implemented (table only) | `evaluation/transfer.py` sign-defined \(\Delta_k = S_k - S_0\); RMSE/CRPS \(\Delta>0\) harmful; contrast uses \(\|S_k\|-\|S_0\|\). No trained scores yet |
| Nonlinear multi-species sharing as **primary** objective | partial | v2 runtime still one-species default. v3 pilot infrastructure exists; training wave not launched |
| Pilot: fix target train set, add other-species studies | implemented (software) | Frozen target manifests + nested \(k\) donors. Proposed review-only config `configs/v3_pilot.smoke.yaml` is **not executed** |

## Count (this matrix)

| Status | Rows |
|---|---|
| implemented | 30 |
| partial | 8 |
| unverified | 5 |
| missing | 14 |

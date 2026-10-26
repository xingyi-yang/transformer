# send_bw: longitudinal body-weight modelling and simulation-informed design from SEND-organized data

> **All results in this repository come from SYNTHETIC SEND-like data: a proof of concept, not real-world
> validation.** Real-data qualification stage achieved: **1 of 5** (see `docs/real_data_qualification.md`).

The pipeline does the following:
- Harmonizes SEND-like extracts (local CSV/Parquet/XPT, or authorized read-only Snowflake) into
  canonical tables, with QC, provenance and quarantine.
- Fits three required response models: a hierarchical spline, a PyMC Bayesian hierarchical model and a
  baseline-conditioned transformer.
- Evaluates them on held-out studies.
- Simulates complete study trajectories to evaluate candidate group sizes and analytical schedules under a
  fixed estimand and a prespecified final analysis. The production default is endpoint
  ANCOVA with Holm correction. Repeated-measures GLS remains requestable but is not a
  default after locked synthetic diagnostics; visit-saturated MMRM is an opt-in candidate.

Body weight only. This cannot establish total animal requirements or justify reducing welfare
observations. See `docs/limitations.md`.

## Installation

Python ≥ 3.11 (developed on CPython 3.14.7, Windows). The `.venv` was prepared in the build session, but the
exact dependency-install command was not recorded. The equivalent setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[bayesian,dev]"     # core + PyMC/ArviZ/nutpie + pytest/ruff
# optional extras: ".[snowflake]" (connector), ".[xpt]" (pyreadstat), ".[tft]" (pytorch-forecasting; early-history only)
# do not install ".[tft]" in the default/smoke environment; optional adapters stay not_run without it
```

Versions used for the recorded runs: numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, torch 2.14.1+cpu,
pymc 6.3.2, arviz 1.3.0, nutpie 0.16.11, pytensor 3.3.4, pydantic 2.14.1, pyarrow 26.0.0. There is no C++
compiler, so PyTensor warns "g++ not available" (harmless) and NUTS runs through nutpie.

## Commands

`send-bw <command> --config <yaml>`: `generate`, `inspect-source`, `ingest`, `prepare`, `validate-real`, `train`,
`evaluate`, `design`, `report`, `run-all`, `validate-request`, `propose-design`,
`tft-forecast` (primary early-history TFT workflow), `tft-stability`,
`tft-study-cv`, `tft-larger-bank`, `tft-longer-ctx`, `tft-early-hist-bench`,
`tft-early-hist-extend`.
`validate-request` and `propose-design` also accept `--model <run dir | models dir | model dir>`, for
inference from serialized artifacts without a training config.

### Primary TFT early-history forecasting

TFT (`pytorch_forecasting.TemporalFusionTransformer`) is the documented forecasting
adapter for **observed early history**. It is still optional relative to the required
spline / custom transformer / Bayesian benchmark and is not a design simulator.

Shared knobs (use these unless a locked protocol says otherwise):

| knob | value | meaning |
|---|---|---|
| `encoder_window_day` | 14 | keep observed visits with `DAY ≤ 14` |
| `max_encoder_length` | 8 | cap on observed encoder visits; no invented days |
| `max_prediction_length` | 6 | multi-horizon from the same observed history |
| `min_encoder_length` | 1 | refuse animals with no observed encoder |
| `window_balance` | `study` | equal study probability over PF windows |
| `target_normalizer` | `species_or_global` | SPECIES-grouped scale when donors are present; no `relu` |
| `quantiles` | 0.1, 0.25, 0.5, 0.75, 0.9 | QuantileLoss head |

```powershell
# extra required once; default smoke env stays not_run without it
.\.venv\Scripts\pip install -e ".[tft]"
.\.venv\Scripts\send-bw tft-forecast --config configs/tft_forecast.yaml
# Linux / this environment:
#   python3 -m pip install -e ".[tft]"
#   send-bw tft-forecast --config configs/tft_forecast.yaml
```

The command writes `runs/tft_primary/` only. It does not overwrite `runs/v3_*`,
`runs/tft_early_hist_bench/`, or `runs/tft_early_hist_extend/`. Official holdout
test errors are not used to choose learning rate, hidden size, dropout, or duration.

Multi-seed interval work uses the same knobs:

```powershell
.\.venv\Scripts\send-bw tft-stability --config configs/tft_stability.yaml
#   send-bw tft-stability --config configs/tft_stability.yaml
```

That command writes `runs/tft_stability/` only. It reuses matching official fits
from `runs/tft_primary/`. Official validation selected fine-tune; that default
**failed** a later confirmation. Confirmation split 34011 used new *test*
studies (SYNA-003/004) but put previously reported holdouts SYNA-007/010/011
in **train**. Do not call that bank untouched.

Target-only TFT is the development reference. Fine-tune and CQR stay
experimental. Study-held-out evaluation:

```powershell
.\.venv\Scripts\send-bw tft-study-cv --config configs/tft_study_cv.yaml
#   send-bw tft-study-cv --config configs/tft_study_cv.yaml
```

That command writes `runs/tft_study_cv/` only. Folds, seeds, and the
calibration split are predeclared. CQR margins in that run are fit on
calibration studies, not validation or test. The earlier stability run
fit CQR on official validation. Raw and CQR metrics are stored separately.

A follow-up read-only method audit (generator observability, spline /
transformer / CatBoost information sets, Chronos visit-rank vs elapsed
day, CQR order statistic, one TFT batch) writes only
`runs/forecast_method_audit/` and does not retrain or overwrite locked
dirs. Target-only remains the development reference.

A larger Generator A bank (36 RAT + 6 DOG, `generate_seed=81011`) reruns
the locked-knob comparison without touching the official 71001 holdout:

```powershell
.\.venv\Scripts\send-bw tft-larger-bank --config configs/tft_larger_bank.yaml
#   send-bw tft-larger-bank --config configs/tft_larger_bank.yaml
```

That command writes `runs/tft_larger_bank/` only. Learning rate, hidden
size, dropout, and duration stay at the locked primary values. Official
and new test errors are not used to retune. Fine-tune remains
experimental even if it has lower MAE on the new bank.

Default cohort tables drop pretreatment / baseline rows, so a weekly
animal's DAY≤14 encoder is often only 7 and 14. Longer context reattaches
the observed predose visits (`DAY` −7 and 0) and reruns the same locked
comparison on the same larger-bank splits:

```powershell
.\.venv\Scripts\send-bw tft-longer-ctx --config configs/tft_longer_ctx.yaml
#   send-bw tft-longer-ctx --config configs/tft_longer_ctx.yaml
```

That command writes `runs/tft_longer_ctx/` only. The cutoff stays DAY≤14
and the targets stay the later observed visits. Recovery rows are not
added. Fine-tune remains experimental.

Configs: `configs/smoke.yaml`, `configs/bayesian_smoke.yaml` (executed); `configs/research.yaml` and
`configs/bayesian_research.yaml` (larger; **not executed**); `configs/real_local.example.yaml` and
`configs/real_snowflake.example.yaml` (templates; Snowflake is `authorized: false`, and credentials are only `env:` references);
and the requests `configs/new_study.example.yaml` (−5% effects) and `configs/new_study.smoke.yaml` (−8%, reduced candidate set).

## Commands actually executed (build session, Windows PowerShell, from the repo root)

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python -m pip install -e . --no-deps -q
.\.venv\Scripts\send-bw run-all --config configs/smoke.yaml              # 82 s (models cached), exit 0
.\.venv\Scripts\send-bw run-all --config configs/bayesian_smoke.yaml     # 3 attempts, see HANDOFF.md; last 107 s, exit 0
.\.venv\Scripts\send-bw validate-real --config configs/smoke.yaml        # stage 0 (synthetic extracts)
.\.venv\Scripts\send-bw inspect-source --config configs/real_local.example.yaml      # fails: output_dir not accessible (expected)
.\.venv\Scripts\send-bw inspect-source --config configs/real_snowflake.example.yaml  # refuses: not authorized (expected)
.\.venv\Scripts\send-bw propose-design --config configs/smoke.yaml --request configs/new_study.example.yaml --with-oracle --replicates 400 600
.\.venv\Scripts\python -m pytest -q tests -p no:cacheprovider            # 106 passed in 43 s
.\.venv\Scripts\python -m ruff check src tests                          # All checks passed
```

## Outputs (per run directory `runs/<run_name>/`)

| path | content |
|---|---|
| `manifest.json` | config hash, seed, software versions, git revision (unavailable: git not installed), stage keys, timings, SEND hash |
| `canonical/*.parquet`, `canonical/provenance.json` | canonical tables and lineage |
| `qc/` | data dictionary, mapping report, quarantine, exclusions, completeness, QC report, source inspection |
| `splits/` | frozen study splits (csv + hashed json) |
| `features/feature_spec.json`, `cohort/*.parquet` | feature specification and modelling cohort |
| `models/<name>/` | serialized models, training histories; `bayesian/diagnostics.json`; `optional_models_status.json` |
| `evaluation/` | `metrics.json`, per-study and sex×horizon tables |
| `design/` | request validation, per-model proposal results/feasibility/Pareto, `final_results.csv`, `final_feasibility.csv`, `design_summary.json` |
| `report/report.md`, `report/report.html`, `report/plots/` | the labelled report |
| `synthetic/` | SEND-like data, curated metadata, hidden `truth/` (read only by `simulation/truth.py`) |

`runs/` is git-ignored. Never commit outputs from real extracts.

## Modes

- **CPU smoke**: 16 small synthetic studies with 400 proposal and 600 final replicates. Software check only
  (nonconclusive Monte Carlo).
- **Research**: `configs/research.yaml` (40 studies, 1000/4000 replicates) and `configs/bayesian_research.yaml`
  (full data, 4 × 2000 draws). Not run in this session.

## Tests

`tests/unit`: data contract, messy fixtures, day and baseline conventions, joins, leakage, splits, generator,
covariance, GLS/ANCOVA, Holm, Monte Carlo intervals, models, design workflow and mock Snowflake.
`tests/integration`: CLI smoke run end to end and the Bayesian smoke test (markers `bayesian`, `slow`;
`pytest -m "not bayesian"` skips it).

See `docs/methods.md`, `docs/data_dictionary.md`, `docs/limitations.md`, `docs/real_data_qualification.md`
and `HANDOFF.md`.

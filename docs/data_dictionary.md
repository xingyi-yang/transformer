# Data dictionary

Every `send-bw ingest` writes the authoritative, code-generated version of this dictionary to
`<run>/qc/data_dictionary.md` (from `send_bw.data.schema.CANONICAL_SCHEMAS` and
`send_bw.data.features.FEATURE_REGISTRY`). This page adds the input-side conventions.

## Time conventions

- `DAY` = elapsed days = date − reference date (default `DM.RFSTDTC`, configurable in the mapping
  profile). The reference date is elapsed day 0 and SEND study day 1; **SEND has no day 0**.
- `dy_to_elapsed`: DY ≥ 1 → DY − 1; negative DY unchanged; DY = 0 is a hard mapping error.
- The source study day is kept in `bw_obs.SOURCE_DY`.
- Partial dates (`BWDTC` of the form `YYYY-MM` or `YYYY`): the rule `study_day_else_quarantine` uses
  BWDY when present and sets `DATE_UNCERTAIN = True`. Otherwise the record is quarantined. Dates are never completed silently.
- Baseline is the **last non-fasted weight in [max_day − window_days, max_day]** (default [−7, 0]
  elapsed). Ties go to the highest BWSEQ. When no eligible predose weight exists, the animal has no baseline and is
  excluded. A post-treatment weight is never substituted.

## Input domains (SEND-like; synthetic data is NOT certified SEND-conformant)

| domain | fields used | notes |
|---|---|---|
| DM | STUDYID, USUBJID, SUBJID, SEX, SPECIES, STRAIN, ARMCD, ARM, SETCD, RFSTDTC, RFENDTC | sex is never inferred from weight; unmapped SEX values become quarantine records |
| BW | STUDYID, USUBJID, BWSEQ, BWTESTCD, BWTEST, BWORRES, BWORRESU, BWSTRESC, BWSTRESN, BWSTRESU, BWDTC, BWDY | standardized value preferred, original as fallback (`value_priority`); units are converted via `units_to_grams`; an unknown unit is a hard quarantine |
| EX | STUDYID, USUBJID, EXSEQ, EXTRT, EXDOSE, EXDOSU, EXROUTE, EXSTDTC, EXENDTC, EXSTDY, EXENDY | **actual** administrations: used for deviation flags only, never as planned regimen or model input |
| DS | STUDYID, USUBJID, DSDECOD/DSTERM, DSSTDTC/DSSTDY | mapped through `disposition_terms` to SCHEDULED_SACRIFICE / RECOVERY_SACRIFICE / DEATH / UNSCHEDULED_REMOVAL; weights after disposition are quarantined |
| SUPPBW | IDVAR, IDVARVAL, QNAM, QVAL | `BWFAST` → `FASTING`; unmapped QNAMs are reported as warnings |
| TS, TX | illustrative subsets | written by the generator for realism; the harmonizer does not join TX to arms (assignment conventions vary) |

Domain file names may use aliases (`BODYWEIGHT` → BW). Column names are upper-cased explicitly, and a case
collision is an error. Two files claiming the same domain for a study are rejected as ambiguous.
Formats: CSV (read as text), Parquet, and SAS XPT (optional `pyreadstat` extra, with a clear error if missing).

## Curated metadata (required for pre-study prediction)

| table | key | columns | purpose |
|---|---|---|---|
| `study_metadata` | STUDYID | SPECIES, STRAIN, STUDY_CLASS, STUDY_START, SEND_VERSION, DURATION_DAYS | study context; cross-checked against DM |
| `arm_regimen` | STUDYID, ARMCD | COMPOUND_ID, IS_CONTROL, PLANNED_DOSE, DOSE_UNIT, ROUTE, INTERVAL_DAYS, PLANNED_START_DY, PLANNED_END_DY (SEND study days), COHORT | **planned** regimens; one row per arm (duplicates rejected). Arms without a row get `REGIMEN_AVAILABLE = False` and are excluded from modelling |
| `compound_reference` | COMPOUND_ID | D1, D2, D3 (numeric descriptors), CLASS | curated **production** descriptors (not SEND domains). Comparable columns must exist at prediction time or active-arm features fail. Hidden generator parameters `A`, `ED50`, `tau_on`, `tau_off` are **not** in this table |

## Canonical tables (summary)

- `study_context`, `animal_context` (ARM_KEY = STUDYID::ARMCD, COHORT main/recovery, baseline, eligibility),
  `arm_regimen`, `administrations`, `bw_obs`, `animal_status`, `quarantine`, `compound_reference`,
  plus `canonical/provenance.json` (source id, mapping version, extraction log).
- `bw_obs` keeps the original value and unit, the standardized value and unit, `WEIGHT_G`, `LOGW`, `SRC_FILE`/`SRC_ROW`,
  `MAPPING_VERSION`, `QC_FLAGS`, `DATE_UNCERTAIN`, `FASTING`, `IS_BASELINE` and `PHASE` (pretreatment / treatment / recovery).
- Modelling target: `Y = LOGW − BASELINE_LOGW` for post-baseline, non-fasted, non-quarantined weights.

## Feature availability classes

- **production**: available before a future study (sex, strain, route, centered baseline log weight,
  planned dose/interval/start/end, compound descriptors, and derived planned-exposure time features
  TAU/EXPO/OFF/LCUM).
- **optional**: available only for the secondary early-history task (observed early weights) or
  not available prospectively (actual dosing). These never enter pre-study features.
- **truth_only**: hidden generator quantities. They are readable only by `simulation/truth.py` for evaluation.
  `check_no_leakage` rejects any feature specification that contains them.

## Synthetic generator extensions

Synthetic studies deliberately include study-specific ARMCD styles, recovery cohorts, deaths,
missed visits, fasted terminal weights (SUPPBW), one kilogram-unit study, one lowercase-column study,
some Parquet files, partial dates, dosing deviations recorded in EX, and one out-of-scope DOG study.
The hidden truth (`truth/params.json`, `animals.parquet`, `studies.parquet`, `latent_paths.parquet`) is
written outside the SEND folder.

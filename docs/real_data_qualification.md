# Real-data qualification checklist

**Current stage achieved: 1 (implemented and mock-tested). No real data has been accessed.**
Stage 1 alone never certifies real-data readiness.

| stage | definition | status |
|---|---|---|
| 1 | Implemented; local adapter tested on messy fixtures; Snowflake adapter mock-tested (read-only SQL, allowlists, parameterization, redaction, local/mock equivalence) | done (synthetic and fixtures only) |
| 2 | Authorized pilot extraction reconciled against source records | not started (needs authorization) |
| 3 | Scientist-reviewed cohort, arm, regimen, compound and timing mappings | not started |
| 4 | Held-out real-study evaluation (chronological and compound holdouts) | not started |
| 5 | Prospective shadow validation (predictions frozen before study results) | not started |

## Before any access
- [ ] Written authorization naming the data, storage location and purpose.
- [ ] Approved storage location for `output_dir`. Runs contain record-level data; never commit `runs/`.
- [ ] Read-only Snowflake role. Credentials via `env:` references or an approved provider only.
- [ ] Explicit table and column allowlists in the config. `authorized: true` is set only after approval.

## Stage 2: pilot reconciliation (a few studies)
- [ ] `send-bw inspect-source` reviewed: domains, files, formats, duplicate or ambiguous files.
- [ ] `send-bw validate-real` run. Review `qc/mapping_report.json`, `qc/quarantine.csv`,
      `qc/exclusions.csv`, `qc/completeness.csv` and `qc/real_data_qualification.json`.
- [ ] Per study: animal counts by arm, sex and cohort match the study report.
- [ ] Spot-check at least 20 weights per study: original value, unit, converted grams and day against source records.
- [ ] Day convention: RFSTDTC reference, no day 0, and partial-date handling are confirmed.
- [ ] Baseline weights: confirm predose status and fasting conditions per study.
- [ ] Dispositions: deaths, unscheduled removals and scheduled sacrifices are mapped correctly. No post-death weights appear.
- [ ] Recovery and satellite cohorts are identified correctly.
- [ ] Extraction log query IDs and timestamps are archived. The snapshot consistency limitation is acknowledged.

## Stage 3: mapping review
- [ ] Mapping profile reviewed and versioned (`mapping_version`), including per-study overrides.
- [ ] Curated `arm_regimen` (planned dose, interval, start and end days, route) reviewed against protocols.
      EX was not used as a planned-regimen substitute.
- [ ] Curated `compound_reference` descriptors reviewed. The descriptor definitions are documented.
- [ ] Unmapped SUPPBW qualifiers and SEX/SPECIES values resolved.

## Stage 4: held-out evaluation
- [ ] Chronological split and compound holdout frozen before fitting.
- [ ] Coverage, contrast bias and correlation by gap reviewed by study. Effective sample sizes reported.
- [ ] Bayesian diagnostics meet the targets on full data (research configuration).
- [ ] GLS null calibration rechecked using real-data covariance estimates. Robust variance is considered.

## Stage 5: shadow validation
- [ ] For upcoming studies, the design recommendation and predictions are frozen and archived, then compared after completion.

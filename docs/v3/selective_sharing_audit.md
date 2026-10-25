# Selective sharing audit (no retraining)

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Inspected existing code and PR #14 artifacts. No training was rerun. Missing provenance is marked unavailable.

## Mechanism

Eligible donor STUDYIDs are donor-species studies that pass cohort QC. Order is sorted unique STUDYID (donor_study_order). Selected = first k. Not metadata similarity, not training/validation scores, not learned weights.

Code:

- `src/send_bw/data/splits.py:donor_study_order`
- `src/send_bw/data/splits.py:compose_train_with_donors`
- `src/send_bw/pipeline.py:Run.prepare`
- `src/send_bw/sharing.py:SELECTIVE_V1 / SELECTIVE_REGULARIZED`
- `src/send_bw/models/spline.py:SplineModel.fit`
- `src/send_bw/models/transformer.py:TransformerModel.fit`
- `src/send_bw/data/features.py:FeatureSpec.fit`

- Regularization vs inclusion: selective vs selective_regularized changes feature sharing and ridge/L2, not which donor IDs are chosen. Both used k=1 prefix SYNA-D001 on PR #14.
- Zero donors available: `True` via `context_specific + n_donor_studies=0`
- Target-only is not single-study: Target-only trains all target-species TRAIN studies (PR #14 d0: 7 studies). It is not a single-study model.
- Test outcomes used for selection: `False`

## Missing provenance

- PR #14 target_only donor_order.studyids is empty: eligible bank was not recorded when sharing=context_specific (donor_mode false).
- No stored similarity scores or learned donor weights exist to reconstruct.

## PR #14 donor cells

| cell | sharing | k | eligible | selected |
|---|---|---|---|---|
| `runs/v3_share_repl/datasets/d0/trains/t0/cells/d0_t0_matched_pooled` | pooled | 3 | ['SYNA-D001', 'SYNA-D002', 'SYNA-D003'] | ['SYNA-D001', 'SYNA-D002', 'SYNA-D003'] |
| `runs/v3_share_repl/datasets/d0/trains/t0/cells/d0_t0_matched_selective` | selective | 1 | ['SYNA-D001', 'SYNA-D002', 'SYNA-D003'] | ['SYNA-D001'] |
| `runs/v3_share_repl/datasets/d0/trains/t0/cells/d0_t0_matched_selective_regularized` | selective_regularized | 1 | ['SYNA-D001', 'SYNA-D002', 'SYNA-D003'] | ['SYNA-D001'] |
| `runs/v3_share_repl/datasets/d0/trains/t0/cells/d0_t0_matched_target_only` | context_specific | 0 | [] | [] |
| `runs/v3_share_repl/datasets/d1/trains/t0/cells/d1_t0_matched_pooled` | pooled | 3 | ['SYNA-D001', 'SYNA-D002', 'SYNA-D003'] | ['SYNA-D001', 'SYNA-D002', 'SYNA-D003'] |
| `runs/v3_share_repl/datasets/d1/trains/t0/cells/d1_t0_matched_selective` | selective | 1 | ['SYNA-D001', 'SYNA-D002', 'SYNA-D003'] | ['SYNA-D001'] |
| `runs/v3_share_repl/datasets/d1/trains/t0/cells/d1_t0_matched_selective_regularized` | selective_regularized | 1 | ['SYNA-D001', 'SYNA-D002', 'SYNA-D003'] | ['SYNA-D001'] |
| `runs/v3_share_repl/datasets/d1/trains/t0/cells/d1_t0_matched_target_only` | context_specific | 0 | [] | [] |

Target-only `studyids: []` means the eligible bank was not recorded on that cell; the bank is available on pooled/selective cells (SYNA-D001..D003) and is not reconstructed as a historical selection score.

Selective v1: `{'variant': 'v1', 'shared': ['compound', 'regimen'], 'species_specific': ['bio', 'baseline', 'species'], 'residual': 'target_species_only', 'note': 'Shared: compound descriptors and planned-dose/regimen query features. Species-specific: growth/baseline token groups and residual scale. Not a Bayesian species random-effect variant.'}`

Selective regularized: `{'variant': 'regularized', 'shared': ['compound', 'regimen'], 'species_specific': ['growth_deviation', 'treatment_deviation'], 'residual': 'target_species_only', 'regularization': {'growth': 'ridge inflation on SPECIES × time-basis coefficients (lambda_growth)', 'treatment': 'ridge inflation on SPECIES × exposure/dose coefficients (lambda_treatment)', 'transformer': 'phase1 target fit; phase2 extra L2 of species/growth tokens toward phase1 and treatment-deviation head toward 0'}, 'note': 'Selective v1 is retained. This variant adds regularized species-specific deviations in nonlinear growth and treatment response. Deviations shrink toward the shared mean; they are not unpenalized species RE.'}`


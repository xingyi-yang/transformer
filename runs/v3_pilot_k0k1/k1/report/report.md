# send_bw report: v3_pilot_k1

> **REAL-DATA RUN - qualification stage must be checked before any use**

Body weight is the only modelled endpoint. Results cannot establish total animal requirements for a toxicology program, cannot justify reducing welfare observations or protocol checks, and are not regulatory-grade.

## Run provenance

- config: `runs/v3_pilot_k0k1/k1/effective.yaml` (hash `e72ebebc95254669`), seed 20261024
- git revision: 9f1da17dcc52f22154364c9caf751a6601b78760
- software: matplotlib=3.11.2, numpy=2.5.4, pandas=3.1.0, platform=Linux-6.12.94+-x86_64-with-glibc2.39, pyarrow=26.0.0, pydantic=2.14.1, python=3.12.3, scikit-learn=1.9.1, scipy=1.18.1, torch=2.14.1

| stage | status | seconds | key |
|---|---|---|---|
| evaluate | NA | 0.9 | NA |
| ingest | NA | 0.6 | 2665b0bc959b5441 |
| optional_models | {'chronos2': 'not_run', 'deepar': 'not_run', 'tft': 'not_run'} | NA | NA |
| prepare | NA | 0.4 | 4e44789a1305c76d |
| train_spline | trained | 0.6 | 20af6834e1c77195 |
| train_transformer | trained | 2.0 | 6d216df54d92e627 |

## Data ingestion and QC

- weights: 7855, animals: 894, quarantined/flagged records: 0, mapping version: example-send-0.1
- eligible animals: 772, modelling observations: 4824
- QC checks: {'all_obs_have_animal': True, 'arm_keys_resolved': True, 'finite_log_weights': True, 'n_quarantined_hard': 0, 'n_quarantined_soft': 0, 'one_baseline_per_animal_max': True, 'passed': True, 'positive_weights': True, 'sex_values': ['F', 'M'], 'unique_animal_keys': True, 'unique_obs_keys': True}

Exclusion counts (animals):

| reason | n |
|---|---|
| too_few_post_baseline_obs | 3 |

Limitations recorded during ingestion:

- 3 weight(s) dated from SEND study day because BWDTC was partial (DATE_UNCERTAIN)
- 50 administration record(s) deviate from the planned regimen; models use planned regimens only, so outcomes include deviation effects

Mapping warnings by type: none

Frozen study-level splits: {'test': 3, 'test_compound_holdout': 1, 'train': 10, 'validation': 3} (hash `cabe2e0e5b0b6881`); compound holdout: ['CMP02']

Target-species manifests (frozen before donors): train n=9 hash `cbdb10a6c2f1bfdf`; validation n=3 hash `509b5d25bd90188b`; test n=3 hash `04d76ef2e097f9ff`. Sharing: pooled; donors selected: 1.

## Models

### spline

- ridge alpha 0.1 (validation-selected), train RMSE 0.0473, validation RMSE 0.0605, OOF RMSE 0.1620, 76 mean parameters, 7 study-bootstrap members
- covariance (OOF residual ML): ell=4.4351, sd_a0=0.0284, sd_a1=0.1218, sd_s0=0.0540, sd_s1=0.1162, sigma_m=0.0001, sigma_p=0.0393
- covariance fit: {'converged': True, 'message': 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH', 'n_animals': 470, 'n_obs': 2789, 'n_studies': 10, 'nll': -4499.342192542599, 'params_at_bound': ['sigma_m']}

### transformer

- architecture {'d_model': 32, 'dropout': 0.1, 'group_dims': {'baseline': 1, 'bio': 3, 'compound': 4, 'regimen': 5, 'species': 1}, 'n_heads': 2, 'n_layers': 1, 'q_dim': 5} (custom baseline-conditioned transformer; NOT a TFT)
- members (study-bootstrap except member 0): m0: best val RMSE 0.0498 after 20 epochs (seed 20261024)
- covariance layer shared with the spline model (primary mean-function comparison)

![transformer training](plots/transformer_training.png)

### bayesian: not_run



### Optional models (not required benchmarks)

| model | status | task | reason |
|---|---|---|---|
| chronos2 | not_run | secondary (early-history, zero-shot first; fine-tuning separate) | disabled in config |
| deepar | not_run | secondary (early-history forecasting) | disabled in config |
| tft | not_run | secondary (early-history forecasting) | disabled in config |

## Held-out evaluation

### Primary task: pre-study prediction of new studies

| model | split | studies | obs | RMSE log | RMSE g | MAE g | cov80 | cov95 | CRPS | SB RMSE | SB cov95 | energy | grp-mean cov90 | contrast cov90 | contrast bias | var ratio | incr ratio | max cov90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spline | test_compound_holdout | 1 | 389 | 0.102 | 35.246 | 26.467 | 0.979 | 1.000 | 0.063 | 0.102 | 1.000 | 0.201 | 1.000 | 0.000 | 0.258 | 2.960 | 3.238 | 0.920 |
| spline | test | 3 | 774 | 0.057 | 16.397 | 12.108 | 0.999 | 1.000 | 0.039 | 0.060 | 1.000 | 0.114 | 1.000 | 1.000 | 0.005 | 3.203 | 1.771 | 1.000 |
| transformer | test_compound_holdout | 1 | 389 | 0.059 | 17.531 | 13.762 | 0.982 | 1.000 | 0.040 | 0.059 | 1.000 | 0.125 | 1.000 | 0.750 | 0.076 | 2.945 | 1.592 | 1.000 |
| transformer | test | 3 | 774 | 0.068 | 22.202 | 15.815 | 0.982 | 1.000 | 0.042 | 0.058 | 1.000 | 0.116 | 1.000 | 0.889 | 0.043 | 3.299 | 1.681 | 0.992 |

SB = study-balanced. Coverage targets: cov80=0.80, cov95=0.95, group-mean/contrast/max cov90=0.90; variance and increment ratios target 1. Few test studies -> wide uncertainty.

Residual correlation by gap (observed vs predicted):

| model | split | gap_bin | corr_obs | corr_pred | n_pairs |
|---|---|---|---|---|---|
| spline | test_compound_holdout | (0,4] | 0.904 | 0.896 | 7 |
| spline | test_compound_holdout | (4,8] | 0.854 | 0.824 | 6 |
| spline | test_compound_holdout | (8,15] | 0.785 | 0.728 | 9 |
| spline | test_compound_holdout | (15,100] | 0.606 | 0.560 | 6 |
| spline | test | (0,4] | 0.901 | 0.887 | 14 |
| spline | test | (4,8] | 0.840 | 0.821 | 15 |
| spline | test | (8,15] | 0.749 | 0.721 | 20 |
| spline | test | (15,100] | 0.595 | 0.553 | 13 |
| transformer | test_compound_holdout | (0,4] | 0.904 | 0.895 | 7 |
| transformer | test_compound_holdout | (4,8] | 0.854 | 0.828 | 6 |
| transformer | test_compound_holdout | (8,15] | 0.785 | 0.739 | 9 |
| transformer | test_compound_holdout | (15,100] | 0.606 | 0.585 | 6 |
| transformer | test | (0,4] | 0.901 | 0.887 | 14 |
| transformer | test | (4,8] | 0.840 | 0.815 | 15 |
| transformer | test | (8,15] | 0.749 | 0.717 | 20 |
| transformer | test | (15,100] | 0.595 | 0.543 | 13 |

### Secondary task: early-history forecasting (kept separate)

(none)

Success on the secondary task does not establish success on the primary task.

![trajectories](plots/trajectories.png)

## Experimental design

Design stage not run.

## Statistical assumptions not yet validated

- Pooled-sex analysis assumes a common treatment effect across sexes.
- The production default final analysis is endpoint ANCOVA. GLS is requestable and is not a default after locked synthetic diagnostics; it is not design qualification.
- GLS benefit from intermediate visits relies on the fixed quadratic group-difference basis; bias is only checked under the simulated mechanisms.
- The working covariance is a historical plug-in from training studies; model-based SEs with df = N_animals - (3 + G k) are an approximation (no Kenward-Roger).
- Missingness is simulated as MAR missed visits and non-informative dropout; informative termination is not addressed and no power claims are made for it.
- Spline/transformer parameter uncertainty comes from study bootstrap of the mean only; covariance parameter uncertainty is not propagated for those models.
- Synthetic generators are illustrative and not biologically validated.


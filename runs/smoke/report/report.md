# send_bw report: smoke

> **SYNTHETIC PROOF OF CONCEPT - NOT REAL-WORLD VALIDATION**

Body weight is the only modelled endpoint. Results cannot establish total animal requirements for a toxicology program, cannot justify reducing welfare observations or protocol checks, and are not regulatory-grade.

## Run provenance

- config: `configs/smoke.yaml` (hash `acd2757a73831d6c`), seed 20261017
- git revision: not available (git not installed / not a repo)
- software: arviz=1.3.0, matplotlib=3.11.2, numpy=2.5.4, nutpie=0.16.11, pandas=3.0.6, platform=Windows-11-10.0.26300-SP0, pyarrow=26.0.0, pydantic=2.14.1, pymc=6.3.2, pytensor=3.3.4, python=3.14.7, scikit-learn=1.9.1, scipy=1.18.1, torch=2.14.1+cpu

| stage | status | seconds | key |
|---|---|---|---|
| design | evaluated | 57.9 | NA |
| evaluate | NA | 22.5 | NA |
| generate | NA | 0.2 | efd5df7e2d8f4af0 |
| ingest | NA | 0.5 | 137a14ffbade8ba6 |
| optional_models | {'chronos2': 'not_run', 'deepar': 'not_run', 'tft': 'not_run'} | NA | NA |
| prepare | NA | NA | feda14205081a545 |
| report | NA | NA | NA |
| train_spline | trained | 1.8 | 0b35b9892b6b941e |
| train_transformer | trained | 3.0 | 14cdab82a34a6dc5 |

## Data ingestion and QC

- weights: 6479, animals: 826, quarantined/flagged records: 0, mapping version: example-send-0.1
- eligible animals: 775, modelling observations: 3941
- QC checks: {'all_obs_have_animal': True, 'arm_keys_resolved': True, 'finite_log_weights': True, 'n_quarantined_hard': 0, 'n_quarantined_soft': 0, 'one_baseline_per_animal_max': True, 'passed': True, 'positive_weights': True, 'sex_values': ['F', 'M'], 'unique_animal_keys': True, 'unique_obs_keys': True}

Exclusion counts (animals):

| reason | n |
|---|---|
| species_out_of_scope | 50 |
| too_few_post_baseline_obs | 1 |

Limitations recorded during ingestion:

- 3 weight(s) dated from SEND study day because BWDTC was partial (DATE_UNCERTAIN)
- 31 administration record(s) deviate from the planned regimen; models use planned regimens only, so outcomes include deviation effects

Mapping warnings by type: none

Frozen study-level splits: {'test': 3, 'test_compound_holdout': 1, 'train': 9, 'validation': 3} (hash `47b9e5e7daf20bb0`); compound holdout: ['CMP01']

## Models

### spline

- ridge alpha 100.0 (validation-selected), train RMSE 0.0466, validation RMSE 0.0506, OOF RMSE 0.0701, 62 mean parameters, 7 study-bootstrap members
- covariance (OOF residual ML): ell=5.7184, sd_a0=0.0272, sd_a1=0.0575, sd_s0=0.0311, sd_s1=0.0375, sigma_m=0.0069, sigma_p=0.0255
- covariance fit: {'converged': True, 'message': 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH', 'n_animals': 431, 'n_obs': 1845, 'n_studies': 9, 'nll': -3575.8651116777655, 'params_at_bound': []}

### transformer

- architecture {'d_model': 32, 'dropout': 0.1, 'group_dims': {'baseline': 1, 'bio': 2, 'compound': 4, 'regimen': 5}, 'n_heads': 2, 'n_layers': 1, 'q_dim': 5} (custom baseline-conditioned transformer; NOT a TFT)
- members (study-bootstrap except member 0): m0: best val RMSE 0.0498 after 18 epochs (seed 20261017); m1: best val RMSE 0.0491 after 22 epochs (seed 20262017)
- covariance layer shared with the spline model (primary mean-function comparison)

![transformer training](plots/transformer_training.png)

### bayesian: not_run



### Optional models (not required benchmarks)

| model | status | task | reason |
|---|---|---|---|
| chronos2 | not_run | secondary (early-history, zero-shot first; fine-tuning separate) | optional dependency 'chronos' not installed |
| deepar | not_run | secondary (early-history forecasting) | optional dependency 'gluonts' not installed |
| tft | not_run | secondary (early-history forecasting) | optional dependency 'pytorch_forecasting' not installed |

## Held-out evaluation

### Primary task: pre-study prediction of new studies

| model | split | studies | obs | RMSE log | RMSE g | MAE g | cov80 | cov95 | CRPS | SB RMSE | SB cov95 | energy | grp-mean cov90 | contrast cov90 | contrast bias | var ratio | incr ratio | max cov90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spline | test_compound_holdout | 1 | 512 | 0.044 | 11.971 | 8.911 | 0.934 | 0.992 | 0.026 | 0.044 | 0.992 | 0.080 | 1.000 | 1.000 | -0.007 | 1.232 | 1.053 | 0.955 |
| spline | test | 3 | 573 | 0.084 | 26.493 | 19.988 | 0.728 | 0.914 | 0.047 | 0.084 | 0.909 | 0.101 | 0.680 | 0.750 | 0.014 | 1.302 | 1.058 | 0.774 |
| transformer | test_compound_holdout | 1 | 512 | 0.066 | 18.847 | 15.192 | 0.781 | 0.955 | 0.037 | 0.066 | 0.955 | 0.113 | 0.884 | 1.000 | -0.018 | 1.473 | 1.083 | 0.879 |
| transformer | test | 3 | 573 | 0.065 | 20.871 | 15.190 | 0.839 | 0.963 | 0.036 | 0.065 | 0.965 | 0.079 | 0.891 | 0.583 | -0.037 | 1.472 | 1.093 | 0.884 |

SB = study-balanced. Coverage targets: cov80=0.80, cov95=0.95, group-mean/contrast/max cov90=0.90; variance and increment ratios target 1. Few test studies -> wide uncertainty.

Residual correlation by gap (observed vs predicted):

| model | split | gap_bin | corr_obs | corr_pred | n_pairs |
|---|---|---|---|---|---|
| spline | test_compound_holdout | (0,4] | 0.920 | 0.926 | 7 |
| spline | test_compound_holdout | (4,8] | 0.885 | 0.875 | 6 |
| spline | test_compound_holdout | (8,15] | 0.800 | 0.806 | 9 |
| spline | test_compound_holdout | (15,100] | 0.681 | 0.693 | 6 |
| spline | test | (4,8] | 0.830 | 0.864 | 9 |
| spline | test | (8,15] | 0.772 | 0.774 | 6 |
| spline | test | (15,100] | 0.668 | 0.662 | 3 |
| transformer | test_compound_holdout | (0,4] | 0.920 | 0.937 | 7 |
| transformer | test_compound_holdout | (4,8] | 0.885 | 0.895 | 6 |
| transformer | test_compound_holdout | (8,15] | 0.800 | 0.838 | 9 |
| transformer | test_compound_holdout | (15,100] | 0.681 | 0.765 | 6 |
| transformer | test | (4,8] | 0.830 | 0.871 | 9 |
| transformer | test | (8,15] | 0.772 | 0.789 | 6 |
| transformer | test | (15,100] | 0.668 | 0.700 | 3 |

### Secondary task: early-history forecasting (kept separate)

| model | split | status | rmse_log | cov90 | crps_log | n_obs | note |
|---|---|---|---|---|---|---|---|
| spline | test_compound_holdout | conditioned | 0.035 | 0.946 | 0.019 | 259 | NA |
| spline | test_compound_holdout | pre_study | 0.052 | 0.985 | 0.030 | 259 | NA |
| spline | test | conditioned | 0.038 | 0.955 | 0.021 | 287 | NA |
| spline | test | pre_study | 0.100 | 0.801 | 0.058 | 287 | NA |
| transformer | test_compound_holdout | conditioned | 0.040 | 0.880 | 0.022 | 259 | NA |
| transformer | test_compound_holdout | pre_study | 0.077 | 0.873 | 0.044 | 259 | NA |
| transformer | test | conditioned | 0.043 | 0.899 | 0.024 | 287 | NA |
| transformer | test | pre_study | 0.078 | 0.906 | 0.044 | 287 | NA |

Success on the secondary task does not establish success on the primary task.

![trajectories](plots/trajectories.png)

## Experimental design

- request validation: {'problems': [], 'status': 'supported', 'warnings': []}
- overall status: **evaluated**
- power = conditional power at the prespecified meaningful effects (Holm, all required); 'assurance' columns use model-predicted effects and are a separate quantity
- secondary estimand (GLS only): fixed weighted average of contrasts at the request's query times; reported for bias/coverage, not used for feasibility
- replicates: proposal 400, final 600 -> **low replication: nonconclusive software check**

> SYNTHETIC PROOF OF CONCEPT
> conditional power at prespecified effects is reported separately from predictive assurance under model-predicted effects
> Analytical schedule only; protocol-required welfare weighings and clinical observations are performed regardless and are not optimized.

### Proposal under spline

- feasible candidates 6/8; status feasible_found; finalists ['ancova|endpoint_only|n=10/10/10|equal', 'gls|weekly|n=10/10/10|equal']; comparator ancova|endpoint_only|n=10/10/10|equal
- effect-shape notes: model-predicted shapes
- model-predicted contrast at T (%): {'high': -8.99, 'low': -3.39}; synthetic oracle: {'high': -22.19, 'low': -15.53}

| candidate | n_animals | n_measurements | power | power_ci_lo | fwer_max_point | fwer_max_ci_hi | coverage_min | bias_frac | failure_rate_max | feasible | assurance_all_required_model_effect | secondary_coverage_min | secondary_bias_max_abs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ancova|endpoint_only|n=6/6/6|equal | 36 | 72 | 0.605 | 0.555 | 0.050 | 0.076 | 0.945 | 0.035 | 0.000 | False | 0.412 | NA | NA |
| gls|weekly|n=6/6/6|equal | 36 | 180 | 0.565 | 0.515 | 0.037 | 0.061 | 0.943 | 0.110 | 0.000 | False | 0.388 | 0.948 | 0.008 |
| ancova|endpoint_only|n=10/10/10|equal | 60 | 120 | 0.922 | 0.892 | 0.055 | 0.082 | 0.945 | 0.013 | 0.000 | True | 0.613 | NA | NA |
| gls|weekly|n=10/10/10|equal | 60 | 300 | 0.877 | 0.841 | 0.050 | 0.076 | 0.930 | 0.089 | 0.000 | True | 0.618 | 0.930 | 0.007 |
| ancova|endpoint_only|n=15/15/15|equal | 90 | 180 | 0.975 | 0.955 | 0.060 | 0.088 | 0.938 | 0.017 | 0.000 | True | 0.735 | NA | NA |
| gls|weekly|n=15/15/15|equal | 90 | 450 | 0.973 | 0.951 | 0.055 | 0.082 | 0.915 | 0.088 | 0.000 | True | 0.743 | 0.912 | 0.006 |
| ancova|endpoint_only|n=20/20/20|equal | 120 | 240 | 0.995 | 0.982 | 0.060 | 0.088 | 0.940 | 0.016 | 0.000 | True | 0.818 | NA | NA |
| gls|weekly|n=20/20/20|equal | 120 | 600 | 0.995 | 0.982 | 0.048 | 0.073 | 0.915 | 0.089 | 0.000 | True | 0.825 | 0.905 | 0.006 |

### Proposal under transformer

- feasible candidates 5/8; status feasible_found; finalists ['ancova|endpoint_only|n=10/10/10|equal', 'gls|weekly|n=10/10/10|equal']; comparator ancova|endpoint_only|n=10/10/10|equal
- effect-shape notes: model-predicted shapes
- model-predicted contrast at T (%): {'high': -20.56, 'low': -16.04}; synthetic oracle: {'high': -22.19, 'low': -15.53}

| candidate | n_animals | n_measurements | power | power_ci_lo | fwer_max_point | fwer_max_ci_hi | coverage_min | bias_frac | failure_rate_max | feasible | assurance_all_required_model_effect | secondary_coverage_min | secondary_bias_max_abs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ancova|endpoint_only|n=6/6/6|equal | 36 | 72 | 0.657 | 0.609 | 0.050 | 0.076 | 0.932 | 0.023 | 0.000 | False | 1.000 | NA | NA |
| gls|weekly|n=6/6/6|equal | 36 | 180 | 0.693 | 0.645 | 0.040 | 0.064 | 0.935 | 0.045 | 0.000 | False | 1.000 | 0.948 | 0.004 |
| ancova|endpoint_only|n=10/10/10|equal | 60 | 120 | 0.910 | 0.878 | 0.052 | 0.079 | 0.943 | 0.008 | 0.000 | True | 1.000 | NA | NA |
| gls|weekly|n=10/10/10|equal | 60 | 300 | 0.920 | 0.889 | 0.045 | 0.070 | 0.938 | 0.033 | 0.000 | True | 1.000 | 0.958 | 0.003 |
| ancova|endpoint_only|n=15/15/15|equal | 90 | 180 | 0.990 | 0.975 | 0.072 | 0.102 | 0.925 | 0.020 | 0.000 | False | 1.000 | NA | NA |
| gls|weekly|n=15/15/15|equal | 90 | 450 | 0.995 | 0.982 | 0.055 | 0.082 | 0.930 | 0.045 | 0.000 | True | 1.000 | 0.930 | 0.002 |
| ancova|endpoint_only|n=20/20/20|equal | 120 | 240 | 1.000 | 0.991 | 0.055 | 0.082 | 0.945 | 0.014 | 0.000 | True | 1.000 | NA | NA |
| gls|weekly|n=20/20/20|equal | 120 | 600 | 1.000 | 0.991 | 0.050 | 0.076 | 0.900 | 0.035 | 0.000 | True | 1.000 | 0.945 | 0.002 |

![power](plots/power_vs_animals.png)

### Fresh independent evaluation of finalists

- `fresh_spline`: fresh batch from the proposing response model
- `fresh_transformer`: fresh batch from the proposing response model
- `generator_B`: independent generator B (different growth/effect/covariance families)
- `stress_covariance_shift`: generator A with sigma_p x1.6 and ell x2
- `stress_heterogeneity`: generator A with animal/study slope SD x1.6
- `stress_delayed_effect`: generator A with a delayed effect shape
- `stress_transient_effect`: generator A with a transient effect shape
- `stress_heavy_tails`: generator B with t(3) measurement noise x1.5
- `oracle_generator_A`: synthetic oracle (true data-generating parameters)

| simulator | candidate | n_animals | power | power_ci_lo | fwer_max_point | fwer_max_ci_hi | coverage_min | bias_frac | failure_rate_max | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| fresh_spline | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.933 | 0.910 | 0.048 | 0.069 | 0.947 | 0.004 | 0.000 | True |
| fresh_spline | gls|weekly|n=10/10/10|equal | 60 | 0.898 | 0.871 | 0.045 | 0.065 | 0.950 | 0.071 | 0.000 | True |
| fresh_transformer | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.907 | 0.881 | 0.040 | 0.059 | 0.948 | 0.018 | 0.000 | True |
| fresh_transformer | gls|weekly|n=10/10/10|equal | 60 | 0.928 | 0.905 | 0.042 | 0.061 | 0.952 | 0.053 | 0.000 | True |
| generator_B | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.913 | 0.888 | 0.053 | 0.074 | 0.947 | 0.011 | 0.000 | True |
| generator_B | gls|weekly|n=10/10/10|equal | 60 | 0.912 | 0.886 | 0.053 | 0.074 | 0.948 | 0.053 | 0.000 | True |
| stress_covariance_shift | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.910 | 0.884 | 0.053 | 0.074 | 0.943 | 0.043 | 0.000 | True |
| stress_covariance_shift | gls|weekly|n=10/10/10|equal | 60 | 0.848 | 0.817 | 0.040 | 0.059 | 0.945 | 0.103 | 0.000 | True |
| stress_heterogeneity | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.785 | 0.750 | 0.040 | 0.059 | 0.945 | 0.014 | 0.000 | False |
| stress_heterogeneity | gls|weekly|n=10/10/10|equal | 60 | 0.855 | 0.824 | 0.100 | 0.127 | 0.892 | 0.077 | 0.000 | False |
| stress_delayed_effect | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.990 | 0.978 | 0.050 | 0.071 | 0.947 | 0.029 | 0.000 | True |
| stress_delayed_effect | gls|weekly|n=10/10/10|equal | 60 | 0.985 | 0.972 | 0.038 | 0.057 | 0.962 | 0.012 | 0.000 | True |
| stress_transient_effect | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.983 | 0.970 | 0.068 | 0.092 | 0.932 | 0.011 | 0.000 | False |
| stress_transient_effect | gls|weekly|n=10/10/10|equal | 60 | 0.838 | 0.806 | 0.033 | 0.051 | 0.865 | 0.214 | 0.000 | False |
| stress_heavy_tails | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.887 | 0.859 | 0.062 | 0.084 | 0.940 | 0.034 | 0.000 | True |
| stress_heavy_tails | gls|weekly|n=10/10/10|equal | 60 | 0.840 | 0.808 | 0.043 | 0.063 | 0.953 | 0.050 | 0.000 | True |
| oracle_generator_A | ancova|endpoint_only|n=10/10/10|equal | 60 | 0.985 | 0.972 | 0.048 | 0.069 | 0.950 | 0.003 | 0.000 | True |
| oracle_generator_A | gls|weekly|n=10/10/10|equal | 60 | 0.965 | 0.947 | 0.040 | 0.059 | 0.947 | 0.062 | 0.000 | True |

### Recommendations

- **spline**: recommended -> `ancova|endpoint_only|n=10/10/10|equal` (60 animals, 120 analytical weighings)
  - criteria NOT met under (stress/oracle, reported not vetoing): ['stress_heterogeneity', 'stress_transient_effect']
- **transformer**: recommended -> `ancova|endpoint_only|n=10/10/10|equal` (60 animals, 120 analytical weighings)
  - criteria NOT met under (stress/oracle, reported not vetoing): ['stress_heterogeneity', 'stress_transient_effect']

## Statistical assumptions not yet validated

- Pooled-sex analysis assumes a common treatment effect across sexes.
- GLS benefit from intermediate visits relies on the fixed quadratic group-difference basis; bias is only checked under the simulated mechanisms.
- The working covariance is a historical plug-in from training studies; model-based SEs with df = N_animals - (3 + G k) are an approximation (no Kenward-Roger).
- Missingness is simulated as MAR missed visits and non-informative dropout; informative termination is not addressed and no power claims are made for it.
- Spline/transformer parameter uncertainty comes from study bootstrap of the mean only; covariance parameter uncertainty is not propagated for those models.
- Synthetic generators are illustrative and not biologically validated.


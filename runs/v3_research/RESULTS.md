# v3 research evaluation protocol

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Signs are published as computed. Sharing is not required to improve. Full-versus-sparse target-study training is a primary result. Prior prototype artifacts were not modified.

## Freeze lock

- git revision: `54095801f1d8d240bbf8b59e5a2506f66ddf5004`
- config hash: `438cb0640f9f9ff8`
- generate_seed: `20261027`
- train_seed: `5150`

## Component status

| component | status |
|---|---|
| selective_v1 | completed |
| selective_regularized | completed |
| sparse_target_study_training | completed |
| full_versus_sparse_primary | completed |
| matched_and_conflicting_donors | completed |
| fixed_target_splits | completed |
| evaluation_pack | completed |
| monte_carlo_uncertainty | completed |
| bayesian | unreliable |

### Bayesian sampling diagnosis

| cell | status | class | counted |
|---|---|---|---|
| d0_t0_matched_context_specific_k0_full | unreliable | mean_identifiability | False |
| d0_t0_both_context_specific_k0_full | unreliable | mean_identifiability | False |

## Primary: full versus sparse target-study training (frozen test)

| model | sharing | bank | n | mean Δ RMSE | sd Δ | harmful | helpful |
|---|---|---|---|---|---|---|---|
| spline | context_specific | both | 4 | 0.012301874631339462 | 0.005440156318663999 | 4 | 0 |
| spline | context_specific | matched | 4 | 0.012301874631339462 | 0.005440156318663999 | 4 | 0 |
| spline | pooled | both | 4 | -0.003367761160954816 | 0.0020746597661602363 | 0 | 4 |
| spline | pooled | matched | 4 | -0.0020049483321566697 | 0.005077172560691917 | 1 | 3 |
| spline | selective | both | 4 | -0.003436863494482115 | 0.0014129074690657484 | 0 | 4 |
| spline | selective | matched | 4 | -0.002150690170040756 | 0.00543340260171451 | 1 | 3 |
| spline | selective_regularized | both | 4 | -0.0056804450551066685 | 0.003457216195784293 | 0 | 4 |
| spline | selective_regularized | matched | 4 | -0.003290004261206035 | 0.00672711844940465 | 1 | 3 |
| transformer | context_specific | both | 4 | 0.0028756642113980137 | 0.00857652892192706 | 3 | 1 |
| transformer | context_specific | matched | 4 | 0.0028756642113980137 | 0.00857652892192706 | 3 | 1 |
| transformer | pooled | both | 4 | 0.03412654982307946 | 0.02654532194047662 | 4 | 0 |
| transformer | pooled | matched | 4 | 0.03168004264231011 | 0.02412075855489027 | 4 | 0 |
| transformer | selective | both | 4 | 0.022515297150725457 | 0.018654705235302757 | 4 | 0 |
| transformer | selective | matched | 4 | 0.02424440569764505 | 0.019688810806980338 | 4 | 0 |
| transformer | selective_regularized | both | 4 | 0.035948170689030556 | 0.02523633203033925 | 4 | 0 |
| transformer | selective_regularized | matched | 4 | 0.0336288170923795 | 0.025739633625547865 | 4 | 0 |

## Sharing transfer (frozen test, full train)

| model | sharing | bank | n | mean Δ RMSE | sd Δ | harmful | helpful |
|---|---|---|---|---|---|---|---|
| spline | pooled | both | 4 | 0.01096549604002349 | 0.007721444463581604 | 4 | 0 |
| spline | pooled | matched | 4 | 0.010362435200212237 | 0.007117214609861057 | 4 | 0 |
| spline | selective | both | 4 | 0.01073807963793145 | 0.007106621862550264 | 4 | 0 |
| spline | selective | matched | 4 | 0.010174446125259348 | 0.006501142048672989 | 4 | 0 |
| spline | selective_regularized | both | 4 | 0.017224627530036743 | 0.009410463014411187 | 4 | 0 |
| spline | selective_regularized | matched | 4 | 0.017474629497606895 | 0.00925407602218637 | 4 | 0 |
| transformer | pooled | both | 4 | -0.009070169223325429 | 0.008926124084319392 | 1 | 3 |
| transformer | pooled | matched | 4 | -0.008798284208441988 | 0.007919150754068785 | 1 | 3 |
| transformer | selective | both | 4 | -0.0027713068260012685 | 0.007201000315040466 | 1 | 3 |
| transformer | selective | matched | 4 | -0.005738989576090581 | 0.008106193572844018 | 1 | 3 |
| transformer | selective_regularized | both | 4 | -0.003570920210604832 | 0.007838323341873513 | 1 | 3 |
| transformer | selective_regularized | matched | 4 | -0.008381460185137826 | 0.010197455020147352 | 1 | 3 |

## Calibration, widths, contrast, temporal (example pack)

Cell `d0_t0_matched_context_specific_k0_full` spline pack. Coverage and widths are predictive-interval diagnostics. They are not complete parameter-uncertainty statements.

- denominators: n_studies=2, n_animals=56, n_obs=218, n_contrasts=6, n_studies_with_contrast=2, n_group_mean_cells=64, n_gap_pairs=12, n_increments=56, n_extrema=56
- weighting: RMSE/CRPS study_balanced; contrast bias contrast_row_mean; study-balanced contrast -0.024410259514845017
- individual: cov80=0.9789418267965254 cov95=0.9942528735632183 width80=0.19535685422948473 width95=0.2748478768601772
- group-mean: cov90=1.0 var_ratio=1.1174879927633148
- contrast: cov90=1.0 width90=0.1830927548948934 bias=-0.012044239853731625 n=6
- temporal: increment_sd_ratio=1.3375480003270157 extrema_max_cov90=0.9821428571428571


## Correctness / inference failures

None

## Artifact locations

- parent: `runs/v3_research`
- density uncertainty: `runs/v3_research/density_uncertainty.json`
- sharing uncertainty: `runs/v3_research/sharing_uncertainty.json`


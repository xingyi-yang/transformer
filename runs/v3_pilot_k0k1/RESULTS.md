# v3 synthetic pilot k=0 → k=1

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Stopped after k=1. Signs in the transfer table are published as computed. The protocol was not retuned after target-test metrics.

## Freeze lock

- git revision: `9f1da17dcc52f22154364c9caf751a6601b78760`
- config hash: `4112f7ae47000c25`
- k_schedule: [0, 1]

## Target manifests (must match)

- train: `cbdb10a6c2f1bfdf`
- validation: `509b5d25bd90188b`
- test: `04d76ef2e097f9ff`

## Uncertainty (fixed language)

- Predictive variability: HierarchicalCovariance draws in `sample_study`.
- Parameter uncertainty is incompletely estimated: spline bootstraps the mean only; covariance parameters are not bootstrapped; transformer `n_members=1` has no bootstrap spread.

## Transfer table (predose-conditioned, frozen target test)

| split | model | k | metric | S0 | Sk | delta | sign |
|---|---|---|---|---|---|---|---|
| test_compound_holdout | spline | 1 | rmse_log | 0.16402898712136052 | 0.10240654612223574 | -0.06162244099912478 | helpful |
| test_compound_holdout | spline | 1 | crps | 0.09996591147431404 | 0.06291797695519935 | -0.037047934519114686 | helpful |
| test_compound_holdout | spline | 1 | contrast_bias | 0.24160963591870238 | 0.25786036283196856 | 0.016250726913266178 | harmful |
| test | spline | 1 | rmse_log | 0.06135304836267915 | 0.06012068731753312 | -0.001232361045146034 | helpful |
| test | spline | 1 | crps | 0.041405723856932655 | 0.04025865452997673 | -0.0011470693269559282 | helpful |
| test | spline | 1 | contrast_bias | -0.010604365988220978 | 0.0049579710553760425 | -0.005646394932844936 | helpful |
| test_compound_holdout | transformer | 1 | rmse_log | 0.05161734619426325 | 0.05917268766641911 | 0.007555341472155856 | harmful |
| test_compound_holdout | transformer | 1 | crps | 0.036319945132309295 | 0.03967288505883519 | 0.003352939926525897 | harmful |
| test_compound_holdout | transformer | 1 | contrast_bias | 0.06582660208492513 | 0.0755650631192134 | 0.00973846103428827 | harmful |
| test | transformer | 1 | rmse_log | 0.05381178777560128 | 0.057860046461956784 | 0.004048258686355506 | harmful |
| test | transformer | 1 | crps | 0.03699239404039405 | 0.03947798690535312 | 0.0024855928649590714 | harmful |
| test | transformer | 1 | contrast_bias | 0.041213103830102356 | 0.042848697620298665 | 0.0016355937901963088 | harmful |

Primary scored split is `test` (frozen target holdout). `test_compound_holdout` is recorded and was not used to retune.

## Artifact locations

- parent: `runs/v3_pilot_k0k1`
- freeze lock: `runs/v3_pilot_k0k1/freeze_lock.json`
- k=0: `runs/v3_pilot_k0k1/k0`
- k=1: `runs/v3_pilot_k0k1/k1`
- transfer table: `runs/v3_pilot_k0k1/transfer_table.json`

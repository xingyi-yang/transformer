# Share-or-abstain — locked, not executed

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

PR #14 remains a fixed-donor experiment. This lock is a separate share-or-abstain policy proposal. Benchmark training has not been run.

- n_dataset_seeds: `2`; n_train_seeds: `1`
- generate_seeds: `[76001, 77010]`; train_seeds: `[85001]`
- donor_rng_seed: `91001`
- banks: `['matched', 'both']`; densities: `['full', 'sparse']`
- sparse_target_train_n: `4` (train studies only; val/test unchanged; not a shorter encoder)
- n_trained_cells: `34`; n_model_fits: `68`; additional_fits_for_abstention: `0`
- estimated_runtime_seconds: `{'point': 180, 'lower': 120, 'upper': 480}`
- compute_cap_seconds: `600`
- execute: `False`

Abstention metric: target-validation study-balanced true RMSE (`val_rmse_log`), min_improvement `0.0`, ties → target-only.

Protected artifacts unchanged: `True`

Proposed execution (requires flipping `share_abstain.execute` to true): `send-bw v3-share-abstain --config configs/v3_share_abstain.frozen.yaml`

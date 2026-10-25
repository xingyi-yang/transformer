# Inner-fold donor ranking — locked, not executed

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Prefix-abstention diagnostic closed without retuning. This lock records the smallest genuine donor-ranking path and **stops before benchmark execution**.

- generate_seeds: `[78001, 79010]`; train_seeds: `[86001]`
- n_eligible_donors: `3`; n_inner_folds: `2`
- n_candidate_fits: `48` (`n_dataset_seeds × n_train_seeds × n_banks × n_densities × n_models × n_eligible_donors × n_inner_folds`)
- n_final_model_fits: `24`; n_model_fits_if_executed: `72`
- estimated_runtime_seconds: `{'point': 240, 'lower': 150, 'upper': 480}`
- compute_cap_seconds: `600`
- execute: `False`
- rank: `inner target-training study-fold rank (not test-informed, not generator-scenario-informed)`

Closeout: `docs/v3/share_abstain_run_CLOSEOUT.md` (retuned=`False`).

Protected artifacts unchanged: `True`

This command refuses `execute: true`.

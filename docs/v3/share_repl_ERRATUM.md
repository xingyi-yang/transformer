# PR #14 erratum

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Generated from `runs/v3_share_repl/SUMMARY.json`. Frozen numerical tables in `runs/v3_share_repl/RESULTS.md` are unchanged.

## pr14_d0_spline_selective

On dataset 0, unregularized spline selective improves MAE, RMSE, and pinball; selective_regularized improves MAE/RMSE but worsens pinball.

- supported by saved numbers: `True`
- numbers: `{'target_only': {'mae': 0.027418126804387516, 'rmse': 0.03623816900100929, 'pinball': 0.010024264515448644}, 'selective': {'mae': 0.02681164782693095, 'rmse': 0.03549778403147913, 'pinball': 0.009630390139144466}, 'selective_regularized': {'mae': 0.02672306117912723, 'rmse': 0.0353805221277278, 'pinball': 0.010118019139484508}}`

## pr14_d1_chronos_vs_last_value

On PR #14 dataset 1, history-only Chronos fine-tunes beat last-value; only history-only zero-shot Chronos trails it.

- supported by saved numbers: `True`
- numbers: `{'last_value_mae': 0.10637670252259208, 'last_value_rmse': 0.12334966098175218, 'chronos2_zero_shot_history_mae': 0.10945143355554868, 'chronos2_zero_shot_history_rmse': 0.12595961648751774, 'chronos2_target_only_history_mae': 0.10263138490208294, 'chronos2_pooled_history_mae': 0.10319112388267877}`

# Share-or-abstain — executed (not confirmatory)

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Two dataset seeds and one training seed are **not confirmatory**. Prefix donor choice is a STUDYID sort, not learned / similarity / validation-ranked selection. The unexecuted frozen proposal is unchanged.

- proposal: `configs/v3_share_abstain.frozen.yaml` hash `9d0152f5a465cfa9`
- approved config hash: `1c4af4be93338539`
- science_fingerprint_match: `True`
- generate_seeds: `[76001, 77010]`; train_seeds: `[85001]`
- seconds_total: `326.66133666038513`; cap: `600.0`; hit: `False`
- status: `executed`; confirmatory: False
- prefix rule: `STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)`

## Abstention counts (validation-only decisions)

- spline: `4/8` target-only, `4/8` share
- transformer: `5/8` target-only, `3/8` share
- overall: `9/16` target-only, `7/16` share

## Prefix versus random one-donor overlap

| dataset | bank | density | prefix | random | intersection | equal |
|---|---|---|---|---|---|---|
| 0 | matched | full | ['SYNA-D001'] | ['SYNA-D001'] | ['SYNA-D001'] | True |
| 0 | matched | sparse | ['SYNA-D001'] | ['SYNA-D001'] | ['SYNA-D001'] | True |
| 0 | both | full | ['SYNA-D001'] | ['SYNA-D001'] | ['SYNA-D001'] | True |
| 0 | both | sparse | ['SYNA-D001'] | ['SYNA-D001'] | ['SYNA-D001'] | True |
| 1 | matched | full | ['SYNA-D001'] | ['SYNA-D001'] | ['SYNA-D001'] | True |
| 1 | matched | sparse | ['SYNA-D001'] | ['SYNA-D001'] | ['SYNA-D001'] | True |
| 1 | both | full | ['SYNA-D001'] | ['SYNA-D001'] | ['SYNA-D001'] | True |
| 1 | both | sparse | ['SYNA-D001'] | ['SYNA-D001'] | ['SYNA-D001'] | True |

## Replicates

### d0 t0 matched full

prefix-versus-random: `{'prefix_ids': ['SYNA-D001'], 'random_ids': ['SYNA-D001'], 'intersection': ['SYNA-D001'], 'overlap_count': 1, 'equal': True, 'prefix_rule': 'studyid_sort_prefix', 'prefix_rule_label': 'STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)', 'random_rule': 'rng_choice_without_replacement', 'learned': False, 'similarity_based': False, 'validation_ranked': False}`

| model | MAE | RMSE | status |
|---|---|---|---|
| spline_target_only | 0.0296 | 0.0384 | ok |
| transformer_target_only | 0.0397 | 0.0516 | ok |
| spline_random_one | 0.0280 | 0.0354 | ok |
| transformer_random_one | 0.0340 | 0.0427 | ok |
| spline_selected_one | 0.0280 | 0.0354 | ok |
| transformer_selected_one | 0.0340 | 0.0427 | ok |
| spline_pooled_all | 0.0278 | 0.0357 | ok |
| transformer_pooled_all | 0.0397 | 0.0483 | ok |
| spline_selected_regularized | 0.0284 | 0.0358 | ok |
| transformer_selected_regularized | 0.0349 | 0.0438 | ok |
| spline_selected_abstain | 0.0280 | 0.0354 | ok |
| transformer_selected_abstain | 0.0340 | 0.0427 | ok |

Validation decisions (saved before test):

`{'spline': {'candidate_val': 0.03573120394068861, 'chosen_arm': 'spline_selected_one', 'decision': 'share', 'delta_candidate_minus_target': -0.011071679547676544, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'validation_improvement', 'target_only_val': 0.04680288348836516, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d0_t0_matched_full/decisions/spline_abstain.json'}, 'transformer': {'candidate_val': 0.04211531614287627, 'chosen_arm': 'transformer_selected_one', 'decision': 'share', 'delta_candidate_minus_target': -0.004069576787044664, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'validation_improvement', 'target_only_val': 0.04618489292992094, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d0_t0_matched_full/decisions/transformer_abstain.json'}}`

Paired test Δ vs target-only (Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| spline_random_one | mae_log | 0.029562821530651234 | 0.027964031236226 | -0.0015987902944252334 | helpful |
| spline_random_one | rmse_log | 0.03836716327141584 | 0.035414944095051214 | -0.0029522191763646233 | helpful |
| spline_random_one | mean_quantile_loss | 0.011467132579195415 | 0.010034732743422054 | -0.001432399835773361 | helpful |
| spline_selected_one | mae_log | 0.029562821530651234 | 0.027964031236226 | -0.0015987902944252334 | helpful |
| spline_selected_one | rmse_log | 0.03836716327141584 | 0.035414944095051214 | -0.0029522191763646233 | helpful |
| spline_selected_one | mean_quantile_loss | 0.011467132579195415 | 0.010034732743422054 | -0.001432399835773361 | helpful |
| spline_pooled_all | mae_log | 0.029562821530651234 | 0.027833359255205702 | -0.001729462275445532 | helpful |
| spline_pooled_all | rmse_log | 0.03836716327141584 | 0.03574811139747462 | -0.0026190518739412152 | helpful |
| spline_pooled_all | mean_quantile_loss | 0.011467132579195415 | 0.010231269467729596 | -0.0012358631114658195 | helpful |
| spline_selected_abstain | mae_log | 0.029562821530651234 | 0.027964031236226 | -0.0015987902944252334 | helpful |
| spline_selected_abstain | rmse_log | 0.03836716327141584 | 0.035414944095051214 | -0.0029522191763646233 | helpful |
| spline_selected_abstain | mean_quantile_loss | 0.011467132579195415 | 0.010034732743422054 | -0.001432399835773361 | helpful |
| spline_selected_regularized | mae_log | 0.029562821530651234 | 0.028436449909330128 | -0.0011263716213211067 | helpful |
| spline_selected_regularized | rmse_log | 0.03836716327141584 | 0.03584390958405614 | -0.0025232536873597 | helpful |
| spline_selected_regularized | mean_quantile_loss | 0.011467132579195415 | 0.010112146597937724 | -0.001354985981257691 | helpful |
| transformer_random_one | mae_log | 0.03970978965408917 | 0.033990385697244115 | -0.0057194039568450525 | helpful |
| transformer_random_one | rmse_log | 0.05155887156115939 | 0.04269235308098435 | -0.008866518480175042 | helpful |
| transformer_random_one | mean_quantile_loss | 0.014348864530234418 | 0.011737387942878449 | -0.002611476587355969 | helpful |
| transformer_selected_one | mae_log | 0.03970978965408917 | 0.033990385697244115 | -0.0057194039568450525 | helpful |
| transformer_selected_one | rmse_log | 0.05155887156115939 | 0.04269235308098435 | -0.008866518480175042 | helpful |
| transformer_selected_one | mean_quantile_loss | 0.014348864530234418 | 0.011737387942878449 | -0.002611476587355969 | helpful |
| transformer_pooled_all | mae_log | 0.03970978965408917 | 0.039747219909945716 | 3.7430255856549244e-05 | harmful |
| transformer_pooled_all | rmse_log | 0.05155887156115939 | 0.0482576850038797 | -0.0033011865572796922 | helpful |
| transformer_pooled_all | mean_quantile_loss | 0.014348864530234418 | 0.013316440545613987 | -0.0010324239846204314 | helpful |
| transformer_selected_abstain | mae_log | 0.03970978965408917 | 0.033990385697244115 | -0.0057194039568450525 | helpful |
| transformer_selected_abstain | rmse_log | 0.05155887156115939 | 0.04269235308098435 | -0.008866518480175042 | helpful |
| transformer_selected_abstain | mean_quantile_loss | 0.014348864530234418 | 0.011737387942878449 | -0.002611476587355969 | helpful |
| transformer_selected_regularized | mae_log | 0.03970978965408917 | 0.03490457706342039 | -0.004805212590668774 | helpful |
| transformer_selected_regularized | rmse_log | 0.05155887156115939 | 0.04383441984196201 | -0.0077244517191973755 | helpful |
| transformer_selected_regularized | mean_quantile_loss | 0.014348864530234418 | 0.0119281457116577 | -0.002420718818576719 | helpful |

### d0 t0 matched sparse

prefix-versus-random: `{'prefix_ids': ['SYNA-D001'], 'random_ids': ['SYNA-D001'], 'intersection': ['SYNA-D001'], 'overlap_count': 1, 'equal': True, 'prefix_rule': 'studyid_sort_prefix', 'prefix_rule_label': 'STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)', 'random_rule': 'rng_choice_without_replacement', 'learned': False, 'similarity_based': False, 'validation_ranked': False}`

| model | MAE | RMSE | status |
|---|---|---|---|
| spline_target_only | 0.0344 | 0.0428 | ok |
| transformer_target_only | 0.0385 | 0.0503 | ok |
| spline_random_one | 0.0335 | 0.0424 | ok |
| transformer_random_one | 0.0519 | 0.0652 | ok |
| spline_selected_one | 0.0335 | 0.0424 | ok |
| transformer_selected_one | 0.0519 | 0.0652 | ok |
| spline_pooled_all | 0.0317 | 0.0407 | ok |
| transformer_pooled_all | 0.0468 | 0.0627 | ok |
| spline_selected_abstain | 0.0335 | 0.0424 | ok |
| transformer_selected_abstain | 0.0385 | 0.0503 | ok |

Validation decisions (saved before test):

`{'spline': {'candidate_val': 0.038185534292670575, 'chosen_arm': 'spline_selected_one', 'decision': 'share', 'delta_candidate_minus_target': -0.013852045478209389, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'validation_improvement', 'target_only_val': 0.052037579770879965, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d0_t0_matched_sparse/decisions/spline_abstain.json'}, 'transformer': {'candidate_val': 0.059325657090800524, 'chosen_arm': 'transformer_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.007185873358870581, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.05213978373192994, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d0_t0_matched_sparse/decisions/transformer_abstain.json'}}`

Paired test Δ vs target-only (Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| spline_random_one | mae_log | 0.03440728986011763 | 0.03348081740599086 | -0.0009264724541267685 | helpful |
| spline_random_one | rmse_log | 0.04283016276250061 | 0.042392024145874216 | -0.0004381386166263934 | helpful |
| spline_random_one | mean_quantile_loss | 0.011867561845054463 | 0.01151176666000176 | -0.0003557951850527031 | helpful |
| spline_selected_one | mae_log | 0.03440728986011763 | 0.03348081740599086 | -0.0009264724541267685 | helpful |
| spline_selected_one | rmse_log | 0.04283016276250061 | 0.042392024145874216 | -0.0004381386166263934 | helpful |
| spline_selected_one | mean_quantile_loss | 0.011867561845054463 | 0.01151176666000176 | -0.0003557951850527031 | helpful |
| spline_pooled_all | mae_log | 0.03440728986011763 | 0.03173529662065239 | -0.0026719932394652438 | helpful |
| spline_pooled_all | rmse_log | 0.04283016276250061 | 0.040705522928777826 | -0.002124639833722783 | helpful |
| spline_pooled_all | mean_quantile_loss | 0.011867561845054463 | 0.011170441918142423 | -0.0006971199269120397 | helpful |
| spline_selected_abstain | mae_log | 0.03440728986011763 | 0.03348081740599086 | -0.0009264724541267685 | helpful |
| spline_selected_abstain | rmse_log | 0.04283016276250061 | 0.042392024145874216 | -0.0004381386166263934 | helpful |
| spline_selected_abstain | mean_quantile_loss | 0.011867561845054463 | 0.01151176666000176 | -0.0003557951850527031 | helpful |
| transformer_random_one | mae_log | 0.03848645676964618 | 0.05194631307301191 | 0.013459856303365728 | harmful |
| transformer_random_one | rmse_log | 0.05032298938394944 | 0.06523844620604388 | 0.014915456822094442 | harmful |
| transformer_random_one | mean_quantile_loss | 0.014193172023644684 | 0.01881099627114024 | 0.004617824247495555 | harmful |
| transformer_selected_one | mae_log | 0.03848645676964618 | 0.05194631307301191 | 0.013459856303365728 | harmful |
| transformer_selected_one | rmse_log | 0.05032298938394944 | 0.06523844620604388 | 0.014915456822094442 | harmful |
| transformer_selected_one | mean_quantile_loss | 0.014193172023644684 | 0.01881099627114024 | 0.004617824247495555 | harmful |
| transformer_pooled_all | mae_log | 0.03848645676964618 | 0.046756320639068004 | 0.008269863869421822 | harmful |
| transformer_pooled_all | rmse_log | 0.05032298938394944 | 0.06274312943603716 | 0.01242014005208772 | harmful |
| transformer_pooled_all | mean_quantile_loss | 0.014193172023644684 | 0.017279375584245877 | 0.003086203560601193 | harmful |
| transformer_selected_abstain | mae_log | 0.03848645676964618 | 0.03848645676964618 | 0.0 | null |
| transformer_selected_abstain | rmse_log | 0.05032298938394944 | 0.05032298938394944 | 0.0 | null |
| transformer_selected_abstain | mean_quantile_loss | 0.014193172023644684 | 0.014193172023644684 | 0.0 | null |

### d0 t0 both full

prefix-versus-random: `{'prefix_ids': ['SYNA-D001'], 'random_ids': ['SYNA-D001'], 'intersection': ['SYNA-D001'], 'overlap_count': 1, 'equal': True, 'prefix_rule': 'studyid_sort_prefix', 'prefix_rule_label': 'STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)', 'random_rule': 'rng_choice_without_replacement', 'learned': False, 'similarity_based': False, 'validation_ranked': False}`

| model | MAE | RMSE | status |
|---|---|---|---|
| spline_target_only | 0.0296 | 0.0384 | ok |
| transformer_target_only | 0.0397 | 0.0516 | ok |
| spline_random_one | 0.0277 | 0.0362 | ok |
| transformer_random_one | 0.0366 | 0.0462 | ok |
| spline_selected_one | 0.0277 | 0.0362 | ok |
| transformer_selected_one | 0.0366 | 0.0462 | ok |
| spline_pooled_all | 0.0274 | 0.0354 | ok |
| transformer_pooled_all | 0.0402 | 0.0486 | ok |
| spline_selected_abstain | 0.0296 | 0.0384 | ok |
| transformer_selected_abstain | 0.0397 | 0.0516 | ok |

Validation decisions (saved before test):

`{'spline': {'candidate_val': 0.05634339476670581, 'chosen_arm': 'spline_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.00954051127834065, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.04680288348836516, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d0_t0_both_full/decisions/spline_abstain.json'}, 'transformer': {'candidate_val': 0.047802813522766247, 'chosen_arm': 'transformer_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.0016179205928453094, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.04618489292992094, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d0_t0_both_full/decisions/transformer_abstain.json'}}`

Paired test Δ vs target-only (Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| spline_random_one | mae_log | 0.029562821530651234 | 0.027732959664020236 | -0.0018298618666309982 | helpful |
| spline_random_one | rmse_log | 0.03836716327141584 | 0.036208089243304875 | -0.002159074028110962 | helpful |
| spline_random_one | mean_quantile_loss | 0.011467132579195415 | 0.011240595449427116 | -0.00022653712976829915 | helpful |
| spline_selected_one | mae_log | 0.029562821530651234 | 0.027732959664020236 | -0.0018298618666309982 | helpful |
| spline_selected_one | rmse_log | 0.03836716327141584 | 0.036208089243304875 | -0.002159074028110962 | helpful |
| spline_selected_one | mean_quantile_loss | 0.011467132579195415 | 0.011240595449427116 | -0.00022653712976829915 | helpful |
| spline_pooled_all | mae_log | 0.029562821530651234 | 0.027445576405545014 | -0.0021172451251062208 | helpful |
| spline_pooled_all | rmse_log | 0.03836716327141584 | 0.03538998800225299 | -0.0029771752691628503 | helpful |
| spline_pooled_all | mean_quantile_loss | 0.011467132579195415 | 0.010877634199183441 | -0.0005894983800119741 | helpful |
| spline_selected_abstain | mae_log | 0.029562821530651234 | 0.029562821530651234 | 0.0 | null |
| spline_selected_abstain | rmse_log | 0.03836716327141584 | 0.03836716327141584 | 0.0 | null |
| spline_selected_abstain | mean_quantile_loss | 0.011467132579195415 | 0.011467132579195415 | 0.0 | null |
| transformer_random_one | mae_log | 0.03970978965408917 | 0.03663433465319952 | -0.0030754550008896472 | helpful |
| transformer_random_one | rmse_log | 0.05155887156115939 | 0.046178319047760236 | -0.005380552513399153 | helpful |
| transformer_random_one | mean_quantile_loss | 0.014348864530234418 | 0.012619286963925854 | -0.0017295775663085638 | helpful |
| transformer_selected_one | mae_log | 0.03970978965408917 | 0.03663433465319952 | -0.0030754550008896472 | helpful |
| transformer_selected_one | rmse_log | 0.05155887156115939 | 0.046178319047760236 | -0.005380552513399153 | helpful |
| transformer_selected_one | mean_quantile_loss | 0.014348864530234418 | 0.012619286963925854 | -0.0017295775663085638 | helpful |
| transformer_pooled_all | mae_log | 0.03970978965408917 | 0.04018206606492399 | 0.00047227641083482513 | harmful |
| transformer_pooled_all | rmse_log | 0.05155887156115939 | 0.04863356601595838 | -0.0029253055452010124 | helpful |
| transformer_pooled_all | mean_quantile_loss | 0.014348864530234418 | 0.013349921084926938 | -0.0009989434453074805 | helpful |
| transformer_selected_abstain | mae_log | 0.03970978965408917 | 0.03970978965408917 | 0.0 | null |
| transformer_selected_abstain | rmse_log | 0.05155887156115939 | 0.05155887156115939 | 0.0 | null |
| transformer_selected_abstain | mean_quantile_loss | 0.014348864530234418 | 0.014348864530234418 | 0.0 | null |

### d0 t0 both sparse

prefix-versus-random: `{'prefix_ids': ['SYNA-D001'], 'random_ids': ['SYNA-D001'], 'intersection': ['SYNA-D001'], 'overlap_count': 1, 'equal': True, 'prefix_rule': 'studyid_sort_prefix', 'prefix_rule_label': 'STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)', 'random_rule': 'rng_choice_without_replacement', 'learned': False, 'similarity_based': False, 'validation_ranked': False}`

| model | MAE | RMSE | status |
|---|---|---|---|
| spline_target_only | 0.0344 | 0.0428 | ok |
| transformer_target_only | 0.0385 | 0.0503 | ok |
| spline_random_one | 0.0326 | 0.0408 | ok |
| transformer_random_one | 0.0548 | 0.0696 | ok |
| spline_selected_one | 0.0326 | 0.0408 | ok |
| transformer_selected_one | 0.0548 | 0.0696 | ok |
| spline_pooled_all | 0.0321 | 0.0411 | ok |
| transformer_pooled_all | 0.0369 | 0.0485 | ok |
| spline_selected_abstain | 0.0326 | 0.0408 | ok |
| transformer_selected_abstain | 0.0385 | 0.0503 | ok |

Validation decisions (saved before test):

`{'spline': {'candidate_val': 0.037181427310299574, 'chosen_arm': 'spline_selected_one', 'decision': 'share', 'delta_candidate_minus_target': -0.01485615246058039, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'validation_improvement', 'target_only_val': 0.052037579770879965, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d0_t0_both_sparse/decisions/spline_abstain.json'}, 'transformer': {'candidate_val': 0.06128580462266005, 'chosen_arm': 'transformer_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.009146020890730105, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.05213978373192994, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d0_t0_both_sparse/decisions/transformer_abstain.json'}}`

Paired test Δ vs target-only (Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| spline_random_one | mae_log | 0.03440728986011763 | 0.032627278861257596 | -0.0017800109988600349 | helpful |
| spline_random_one | rmse_log | 0.04283016276250061 | 0.04078141910264994 | -0.002048743659850666 | helpful |
| spline_random_one | mean_quantile_loss | 0.011867561845054463 | 0.011292001753243738 | -0.000575560091810725 | helpful |
| spline_selected_one | mae_log | 0.03440728986011763 | 0.032627278861257596 | -0.0017800109988600349 | helpful |
| spline_selected_one | rmse_log | 0.04283016276250061 | 0.04078141910264994 | -0.002048743659850666 | helpful |
| spline_selected_one | mean_quantile_loss | 0.011867561845054463 | 0.011292001753243738 | -0.000575560091810725 | helpful |
| spline_pooled_all | mae_log | 0.03440728986011763 | 0.03206250070094761 | -0.0023447891591700173 | helpful |
| spline_pooled_all | rmse_log | 0.04283016276250061 | 0.041091733849409263 | -0.0017384289130913458 | helpful |
| spline_pooled_all | mean_quantile_loss | 0.011867561845054463 | 0.01176406777006475 | -0.0001034940749897132 | helpful |
| spline_selected_abstain | mae_log | 0.03440728986011763 | 0.032627278861257596 | -0.0017800109988600349 | helpful |
| spline_selected_abstain | rmse_log | 0.04283016276250061 | 0.04078141910264994 | -0.002048743659850666 | helpful |
| spline_selected_abstain | mean_quantile_loss | 0.011867561845054463 | 0.011292001753243738 | -0.000575560091810725 | helpful |
| transformer_random_one | mae_log | 0.03848645676964618 | 0.05479216922860232 | 0.016305712458956137 | harmful |
| transformer_random_one | rmse_log | 0.05032298938394944 | 0.06955778064447402 | 0.019234791260524575 | harmful |
| transformer_random_one | mean_quantile_loss | 0.014193172023644684 | 0.020445905339955238 | 0.006252733316310554 | harmful |
| transformer_selected_one | mae_log | 0.03848645676964618 | 0.05479216922860232 | 0.016305712458956137 | harmful |
| transformer_selected_one | rmse_log | 0.05032298938394944 | 0.06955778064447402 | 0.019234791260524575 | harmful |
| transformer_selected_one | mean_quantile_loss | 0.014193172023644684 | 0.020445905339955238 | 0.006252733316310554 | harmful |
| transformer_pooled_all | mae_log | 0.03848645676964618 | 0.03690145329506807 | -0.0015850034745781144 | helpful |
| transformer_pooled_all | rmse_log | 0.05032298938394944 | 0.04851920688666418 | -0.0018037824972852584 | helpful |
| transformer_pooled_all | mean_quantile_loss | 0.014193172023644684 | 0.012826671131495757 | -0.0013665008921489273 | helpful |
| transformer_selected_abstain | mae_log | 0.03848645676964618 | 0.03848645676964618 | 0.0 | null |
| transformer_selected_abstain | rmse_log | 0.05032298938394944 | 0.05032298938394944 | 0.0 | null |
| transformer_selected_abstain | mean_quantile_loss | 0.014193172023644684 | 0.014193172023644684 | 0.0 | null |

### d1 t0 matched full

prefix-versus-random: `{'prefix_ids': ['SYNA-D001'], 'random_ids': ['SYNA-D001'], 'intersection': ['SYNA-D001'], 'overlap_count': 1, 'equal': True, 'prefix_rule': 'studyid_sort_prefix', 'prefix_rule_label': 'STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)', 'random_rule': 'rng_choice_without_replacement', 'learned': False, 'similarity_based': False, 'validation_ranked': False}`

| model | MAE | RMSE | status |
|---|---|---|---|
| spline_target_only | 0.0283 | 0.0368 | ok |
| transformer_target_only | 0.0340 | 0.0415 | ok |
| spline_random_one | 0.0310 | 0.0379 | ok |
| transformer_random_one | 0.0382 | 0.0475 | ok |
| spline_selected_one | 0.0310 | 0.0379 | ok |
| transformer_selected_one | 0.0382 | 0.0475 | ok |
| spline_pooled_all | 0.0308 | 0.0386 | ok |
| transformer_pooled_all | 0.0426 | 0.0514 | ok |
| spline_selected_regularized | 0.0315 | 0.0383 | ok |
| transformer_selected_regularized | 0.0327 | 0.0412 | ok |
| spline_selected_abstain | 0.0283 | 0.0368 | ok |
| transformer_selected_abstain | 0.0382 | 0.0475 | ok |

Validation decisions (saved before test):

`{'spline': {'candidate_val': 0.047886379208912214, 'chosen_arm': 'spline_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.008666102647484637, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.039220276561427576, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d1_t0_matched_full/decisions/spline_abstain.json'}, 'transformer': {'candidate_val': 0.04718931723498474, 'chosen_arm': 'transformer_selected_one', 'decision': 'share', 'delta_candidate_minus_target': -0.0023311408206339174, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'validation_improvement', 'target_only_val': 0.04952045805561866, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d1_t0_matched_full/decisions/transformer_abstain.json'}}`

Paired test Δ vs target-only (Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| spline_random_one | mae_log | 0.02830583003524118 | 0.031039376816366794 | 0.0027335467811256155 | harmful |
| spline_random_one | rmse_log | 0.03680558904761029 | 0.03788664678629484 | 0.0010810577386845474 | harmful |
| spline_random_one | mean_quantile_loss | 0.010226092283133264 | 0.012586907799890908 | 0.002360815516757644 | harmful |
| spline_selected_one | mae_log | 0.02830583003524118 | 0.031039376816366794 | 0.0027335467811256155 | harmful |
| spline_selected_one | rmse_log | 0.03680558904761029 | 0.03788664678629484 | 0.0010810577386845474 | harmful |
| spline_selected_one | mean_quantile_loss | 0.010226092283133264 | 0.012586907799890908 | 0.002360815516757644 | harmful |
| spline_pooled_all | mae_log | 0.02830583003524118 | 0.030842647184499326 | 0.0025368171492581477 | harmful |
| spline_pooled_all | rmse_log | 0.03680558904761029 | 0.03863075854524872 | 0.001825169497638432 | harmful |
| spline_pooled_all | mean_quantile_loss | 0.010226092283133264 | 0.011220274678014985 | 0.0009941823948817208 | harmful |
| spline_selected_abstain | mae_log | 0.02830583003524118 | 0.02830583003524118 | 0.0 | null |
| spline_selected_abstain | rmse_log | 0.03680558904761029 | 0.03680558904761029 | 0.0 | null |
| spline_selected_abstain | mean_quantile_loss | 0.010226092283133264 | 0.010226092283133264 | 0.0 | null |
| spline_selected_regularized | mae_log | 0.02830583003524118 | 0.03145591469504287 | 0.0031500846598016934 | harmful |
| spline_selected_regularized | rmse_log | 0.03680558904761029 | 0.038328324515511686 | 0.0015227354679013955 | harmful |
| spline_selected_regularized | mean_quantile_loss | 0.010226092283133264 | 0.01274083408965963 | 0.0025147418065263657 | harmful |
| transformer_random_one | mae_log | 0.03396140241991247 | 0.038158810080773105 | 0.004197407660860637 | harmful |
| transformer_random_one | rmse_log | 0.041492458547928976 | 0.04745275597819697 | 0.005960297430267991 | harmful |
| transformer_random_one | mean_quantile_loss | 0.011618943841529571 | 0.013888437932691998 | 0.002269494091162427 | harmful |
| transformer_selected_one | mae_log | 0.03396140241991247 | 0.038158810080773105 | 0.004197407660860637 | harmful |
| transformer_selected_one | rmse_log | 0.041492458547928976 | 0.04745275597819697 | 0.005960297430267991 | harmful |
| transformer_selected_one | mean_quantile_loss | 0.011618943841529571 | 0.013888437932691998 | 0.002269494091162427 | harmful |
| transformer_pooled_all | mae_log | 0.03396140241991247 | 0.042553729133406534 | 0.008592326713494067 | harmful |
| transformer_pooled_all | rmse_log | 0.041492458547928976 | 0.05140884317252324 | 0.009916384624594266 | harmful |
| transformer_pooled_all | mean_quantile_loss | 0.011618943841529571 | 0.014350683632935846 | 0.0027317397914062742 | harmful |
| transformer_selected_abstain | mae_log | 0.03396140241991247 | 0.038158810080773105 | 0.004197407660860637 | harmful |
| transformer_selected_abstain | rmse_log | 0.041492458547928976 | 0.04745275597819697 | 0.005960297430267991 | harmful |
| transformer_selected_abstain | mean_quantile_loss | 0.011618943841529571 | 0.013888437932691998 | 0.002269494091162427 | harmful |
| transformer_selected_regularized | mae_log | 0.03396140241991247 | 0.0327038332436832 | -0.0012575691762292673 | helpful |
| transformer_selected_regularized | rmse_log | 0.041492458547928976 | 0.04119426606262671 | -0.00029819248530226633 | helpful |
| transformer_selected_regularized | mean_quantile_loss | 0.011618943841529571 | 0.012743468727755308 | 0.001124524886225737 | harmful |

### d1 t0 matched sparse

prefix-versus-random: `{'prefix_ids': ['SYNA-D001'], 'random_ids': ['SYNA-D001'], 'intersection': ['SYNA-D001'], 'overlap_count': 1, 'equal': True, 'prefix_rule': 'studyid_sort_prefix', 'prefix_rule_label': 'STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)', 'random_rule': 'rng_choice_without_replacement', 'learned': False, 'similarity_based': False, 'validation_ranked': False}`

| model | MAE | RMSE | status |
|---|---|---|---|
| spline_target_only | 0.0293 | 0.0385 | ok |
| transformer_target_only | 0.0348 | 0.0462 | ok |
| spline_random_one | 0.0281 | 0.0359 | ok |
| transformer_random_one | 0.0499 | 0.0615 | ok |
| spline_selected_one | 0.0281 | 0.0359 | ok |
| transformer_selected_one | 0.0499 | 0.0615 | ok |
| spline_pooled_all | 0.0308 | 0.0387 | ok |
| transformer_pooled_all | 0.0380 | 0.0480 | ok |
| spline_selected_abstain | 0.0281 | 0.0359 | ok |
| transformer_selected_abstain | 0.0348 | 0.0462 | ok |

Validation decisions (saved before test):

`{'spline': {'candidate_val': 0.03754727566877343, 'chosen_arm': 'spline_selected_one', 'decision': 'share', 'delta_candidate_minus_target': -0.004308090802396011, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'validation_improvement', 'target_only_val': 0.04185536647116944, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d1_t0_matched_sparse/decisions/spline_abstain.json'}, 'transformer': {'candidate_val': 0.05371753091687874, 'chosen_arm': 'transformer_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.004274153872747419, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.04944337704413132, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d1_t0_matched_sparse/decisions/transformer_abstain.json'}}`

Paired test Δ vs target-only (Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| spline_random_one | mae_log | 0.02926835118708495 | 0.02808568861337237 | -0.0011826625737125784 | helpful |
| spline_random_one | rmse_log | 0.03852072858824568 | 0.03594962595692233 | -0.0025711026313233465 | helpful |
| spline_random_one | mean_quantile_loss | 0.010684697877904565 | 0.009966188457204796 | -0.0007185094206997693 | helpful |
| spline_selected_one | mae_log | 0.02926835118708495 | 0.02808568861337237 | -0.0011826625737125784 | helpful |
| spline_selected_one | rmse_log | 0.03852072858824568 | 0.03594962595692233 | -0.0025711026313233465 | helpful |
| spline_selected_one | mean_quantile_loss | 0.010684697877904565 | 0.009966188457204796 | -0.0007185094206997693 | helpful |
| spline_pooled_all | mae_log | 0.02926835118708495 | 0.030780840402464323 | 0.0015124892153793729 | harmful |
| spline_pooled_all | rmse_log | 0.03852072858824568 | 0.038698074744464434 | 0.00017734615621875505 | harmful |
| spline_pooled_all | mean_quantile_loss | 0.010684697877904565 | 0.011170497167713685 | 0.0004857992898091197 | harmful |
| spline_selected_abstain | mae_log | 0.02926835118708495 | 0.02808568861337237 | -0.0011826625737125784 | helpful |
| spline_selected_abstain | rmse_log | 0.03852072858824568 | 0.03594962595692233 | -0.0025711026313233465 | helpful |
| spline_selected_abstain | mean_quantile_loss | 0.010684697877904565 | 0.009966188457204796 | -0.0007185094206997693 | helpful |
| transformer_random_one | mae_log | 0.034825363054006825 | 0.049881614075124724 | 0.015056251021117899 | harmful |
| transformer_random_one | rmse_log | 0.046190330398974244 | 0.061515628537652856 | 0.015325298138678611 | harmful |
| transformer_random_one | mean_quantile_loss | 0.012605820203404346 | 0.01794100879237317 | 0.005335188588968823 | harmful |
| transformer_selected_one | mae_log | 0.034825363054006825 | 0.049881614075124724 | 0.015056251021117899 | harmful |
| transformer_selected_one | rmse_log | 0.046190330398974244 | 0.061515628537652856 | 0.015325298138678611 | harmful |
| transformer_selected_one | mean_quantile_loss | 0.012605820203404346 | 0.01794100879237317 | 0.005335188588968823 | harmful |
| transformer_pooled_all | mae_log | 0.034825363054006825 | 0.037954696857845346 | 0.0031293338038385216 | harmful |
| transformer_pooled_all | rmse_log | 0.046190330398974244 | 0.04801401039023373 | 0.0018236799912594859 | harmful |
| transformer_pooled_all | mean_quantile_loss | 0.012605820203404346 | 0.013072572307592845 | 0.00046675210418849894 | harmful |
| transformer_selected_abstain | mae_log | 0.034825363054006825 | 0.034825363054006825 | 0.0 | null |
| transformer_selected_abstain | rmse_log | 0.046190330398974244 | 0.046190330398974244 | 0.0 | null |
| transformer_selected_abstain | mean_quantile_loss | 0.012605820203404346 | 0.012605820203404346 | 0.0 | null |

### d1 t0 both full

prefix-versus-random: `{'prefix_ids': ['SYNA-D001'], 'random_ids': ['SYNA-D001'], 'intersection': ['SYNA-D001'], 'overlap_count': 1, 'equal': True, 'prefix_rule': 'studyid_sort_prefix', 'prefix_rule_label': 'STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)', 'random_rule': 'rng_choice_without_replacement', 'learned': False, 'similarity_based': False, 'validation_ranked': False}`

| model | MAE | RMSE | status |
|---|---|---|---|
| spline_target_only | 0.0283 | 0.0368 | ok |
| transformer_target_only | 0.0340 | 0.0415 | ok |
| spline_random_one | 0.0307 | 0.0373 | ok |
| transformer_random_one | 0.0376 | 0.0477 | ok |
| spline_selected_one | 0.0307 | 0.0373 | ok |
| transformer_selected_one | 0.0376 | 0.0477 | ok |
| spline_pooled_all | 0.0304 | 0.0382 | ok |
| transformer_pooled_all | 0.0411 | 0.0531 | ok |
| spline_selected_abstain | 0.0283 | 0.0368 | ok |
| transformer_selected_abstain | 0.0376 | 0.0477 | ok |

Validation decisions (saved before test):

`{'spline': {'candidate_val': 0.04832521856877027, 'chosen_arm': 'spline_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.009104942007342694, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.039220276561427576, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d1_t0_both_full/decisions/spline_abstain.json'}, 'transformer': {'candidate_val': 0.04601483716129534, 'chosen_arm': 'transformer_selected_one', 'decision': 'share', 'delta_candidate_minus_target': -0.0035056208943233177, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'validation_improvement', 'target_only_val': 0.04952045805561866, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d1_t0_both_full/decisions/transformer_abstain.json'}}`

Paired test Δ vs target-only (Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| spline_random_one | mae_log | 0.02830583003524118 | 0.030718927001623587 | 0.002413096966382408 | harmful |
| spline_random_one | rmse_log | 0.03680558904761029 | 0.037345624633185553 | 0.0005400355855752631 | harmful |
| spline_random_one | mean_quantile_loss | 0.010226092283133264 | 0.012676736650891652 | 0.002450644367758388 | harmful |
| spline_selected_one | mae_log | 0.02830583003524118 | 0.030718927001623587 | 0.002413096966382408 | harmful |
| spline_selected_one | rmse_log | 0.03680558904761029 | 0.037345624633185553 | 0.0005400355855752631 | harmful |
| spline_selected_one | mean_quantile_loss | 0.010226092283133264 | 0.012676736650891652 | 0.002450644367758388 | harmful |
| spline_pooled_all | mae_log | 0.02830583003524118 | 0.030415786894255958 | 0.002109956859014779 | harmful |
| spline_pooled_all | rmse_log | 0.03680558904761029 | 0.03815981559564812 | 0.0013542265480378304 | harmful |
| spline_pooled_all | mean_quantile_loss | 0.010226092283133264 | 0.01135460996553182 | 0.001128517682398555 | harmful |
| spline_selected_abstain | mae_log | 0.02830583003524118 | 0.02830583003524118 | 0.0 | null |
| spline_selected_abstain | rmse_log | 0.03680558904761029 | 0.03680558904761029 | 0.0 | null |
| spline_selected_abstain | mean_quantile_loss | 0.010226092283133264 | 0.010226092283133264 | 0.0 | null |
| transformer_random_one | mae_log | 0.03396140241991247 | 0.03760329936888528 | 0.0036418969489728156 | harmful |
| transformer_random_one | rmse_log | 0.041492458547928976 | 0.04768250283461882 | 0.006190044286689844 | harmful |
| transformer_random_one | mean_quantile_loss | 0.011618943841529571 | 0.014041028252730983 | 0.002422084411201411 | harmful |
| transformer_selected_one | mae_log | 0.03396140241991247 | 0.03760329936888528 | 0.0036418969489728156 | harmful |
| transformer_selected_one | rmse_log | 0.041492458547928976 | 0.04768250283461882 | 0.006190044286689844 | harmful |
| transformer_selected_one | mean_quantile_loss | 0.011618943841529571 | 0.014041028252730983 | 0.002422084411201411 | harmful |
| transformer_pooled_all | mae_log | 0.03396140241991247 | 0.041075800188390375 | 0.007114397768477908 | harmful |
| transformer_pooled_all | rmse_log | 0.041492458547928976 | 0.05314004856316075 | 0.011647590015231774 | harmful |
| transformer_pooled_all | mean_quantile_loss | 0.011618943841529571 | 0.014612228242270788 | 0.002993284400741217 | harmful |
| transformer_selected_abstain | mae_log | 0.03396140241991247 | 0.03760329936888528 | 0.0036418969489728156 | harmful |
| transformer_selected_abstain | rmse_log | 0.041492458547928976 | 0.04768250283461882 | 0.006190044286689844 | harmful |
| transformer_selected_abstain | mean_quantile_loss | 0.011618943841529571 | 0.014041028252730983 | 0.002422084411201411 | harmful |

### d1 t0 both sparse

prefix-versus-random: `{'prefix_ids': ['SYNA-D001'], 'random_ids': ['SYNA-D001'], 'intersection': ['SYNA-D001'], 'overlap_count': 1, 'equal': True, 'prefix_rule': 'studyid_sort_prefix', 'prefix_rule_label': 'STUDYID-sort prefix (not learned, not similarity-based, not validation-ranked)', 'random_rule': 'rng_choice_without_replacement', 'learned': False, 'similarity_based': False, 'validation_ranked': False}`

| model | MAE | RMSE | status |
|---|---|---|---|
| spline_target_only | 0.0293 | 0.0385 | ok |
| transformer_target_only | 0.0348 | 0.0462 | ok |
| spline_random_one | 0.0316 | 0.0400 | ok |
| transformer_random_one | 0.0484 | 0.0628 | ok |
| spline_selected_one | 0.0316 | 0.0400 | ok |
| transformer_selected_one | 0.0484 | 0.0628 | ok |
| spline_pooled_all | 0.0319 | 0.0402 | ok |
| transformer_pooled_all | 0.0378 | 0.0482 | ok |
| spline_selected_abstain | 0.0293 | 0.0385 | ok |
| transformer_selected_abstain | 0.0348 | 0.0462 | ok |

Validation decisions (saved before test):

`{'spline': {'candidate_val': 0.047843822016544385, 'chosen_arm': 'spline_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.005988455545374943, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.04185536647116944, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d1_t0_both_sparse/decisions/spline_abstain.json'}, 'transformer': {'candidate_val': 0.057479080836255594, 'chosen_arm': 'transformer_target_only', 'decision': 'target_only', 'delta_candidate_minus_target': 0.008035703792124274, 'lower_is_better': True, 'metric': 'val_rmse_log', 'min_improvement': 0.0, 'n_val_studies': 2, 'reason': 'candidate_worse', 'target_only_val': 0.04944337704413132, 'test_outcomes_used': False, 'threshold_from_test': False, 'tie_rule': 'target_only', 'saved_before_test': True, 'decision_path': 'runs/v3_share_abstain_run/repl/d1_t0_both_sparse/decisions/transformer_abstain.json'}}`

Paired test Δ vs target-only (Δ>0 harmful):

| family | metric | target-only | other | Δ | sign |
|---|---|---|---|---|---|
| spline_random_one | mae_log | 0.02926835118708495 | 0.031620618508939546 | 0.0023522673218545967 | harmful |
| spline_random_one | rmse_log | 0.03852072858824568 | 0.04003102331606055 | 0.0015102947278148732 | harmful |
| spline_random_one | mean_quantile_loss | 0.010684697877904565 | 0.011888242588552284 | 0.0012035447106477186 | harmful |
| spline_selected_one | mae_log | 0.02926835118708495 | 0.031620618508939546 | 0.0023522673218545967 | harmful |
| spline_selected_one | rmse_log | 0.03852072858824568 | 0.04003102331606055 | 0.0015102947278148732 | harmful |
| spline_selected_one | mean_quantile_loss | 0.010684697877904565 | 0.011888242588552284 | 0.0012035447106477186 | harmful |
| spline_pooled_all | mae_log | 0.02926835118708495 | 0.031866713916413325 | 0.0025983627293283754 | harmful |
| spline_pooled_all | rmse_log | 0.03852072858824568 | 0.04023082407716032 | 0.0017100954889146397 | harmful |
| spline_pooled_all | mean_quantile_loss | 0.010684697877904565 | 0.011910082013149602 | 0.001225384135245037 | harmful |
| spline_selected_abstain | mae_log | 0.02926835118708495 | 0.02926835118708495 | 0.0 | null |
| spline_selected_abstain | rmse_log | 0.03852072858824568 | 0.03852072858824568 | 0.0 | null |
| spline_selected_abstain | mean_quantile_loss | 0.010684697877904565 | 0.010684697877904565 | 0.0 | null |
| transformer_random_one | mae_log | 0.034825363054006825 | 0.048388715325329214 | 0.013563352271322389 | harmful |
| transformer_random_one | rmse_log | 0.046190330398974244 | 0.06283688748529731 | 0.016646557086323066 | harmful |
| transformer_random_one | mean_quantile_loss | 0.012605820203404346 | 0.017329320330050267 | 0.0047235001266459214 | harmful |
| transformer_selected_one | mae_log | 0.034825363054006825 | 0.048388715325329214 | 0.013563352271322389 | harmful |
| transformer_selected_one | rmse_log | 0.046190330398974244 | 0.06283688748529731 | 0.016646557086323066 | harmful |
| transformer_selected_one | mean_quantile_loss | 0.012605820203404346 | 0.017329320330050267 | 0.0047235001266459214 | harmful |
| transformer_pooled_all | mae_log | 0.034825363054006825 | 0.037751854437077556 | 0.002926491383070731 | harmful |
| transformer_pooled_all | rmse_log | 0.046190330398974244 | 0.04822247982221242 | 0.0020321494232381776 | harmful |
| transformer_pooled_all | mean_quantile_loss | 0.012605820203404346 | 0.0131542445249145 | 0.000548424321510154 | harmful |
| transformer_selected_abstain | mae_log | 0.034825363054006825 | 0.034825363054006825 | 0.0 | null |
| transformer_selected_abstain | rmse_log | 0.046190330398974244 | 0.046190330398974244 | 0.0 | null |
| transformer_selected_abstain | mean_quantile_loss | 0.012605820203404346 | 0.012605820203404346 | 0.0 | null |

## Failures (not aggregated as complete)

- aggregated_across_incomplete: False
- failures: `[]`

## Guardrails

- real_data_accessed: False
- design_run: False
- protected unchanged: `True`
- confirmatory: False

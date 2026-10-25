# v3 replicated synthetic benchmark

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Signs are published as computed. The protocol was not retuned after target-test metrics. Sharing is not required to improve. An unreliable Bayesian fit is not counted as a successful benchmark. Prior `runs/v3_eval` and `runs/v3_pilot_k0k1` artifacts were not modified.

## Freeze lock

- git revision: `fbf219628c5606db3dc4c300e3f82d66a0b7a102`
- config hash: `a9f9568910bf8669`
- generate_seed: `20261026` (dataset only)
- train_seed: `4242` (splits and models only)

## Component status

| component | status |
|---|---|
| selective_v1 | completed |
| selective_regularized | completed |
| sparse_target_study_training | completed |
| sparse_history_encoder | completed |
| evaluation_pack | completed |
| bayesian | unreliable |
| replicated_transfer_uncertainty | completed |

### Bayesian cells

| cell | status | counted | reason |
|---|---|---|---|
| r0_both_context_specific_k0 | unreliable | False | Bayesian diagnostics failed the reliability gate |
| r0_both_pooled_k1 | unreliable | False | Bayesian diagnostics failed the reliability gate |
| r0_both_selective_regularized_k1 | unreliable | False | Bayesian diagnostics failed the reliability gate |

## Uncertainty (fixed language)

- Predictive variability: HierarchicalCovariance draws in `sample_study`.
- Parameter uncertainty is incompletely estimated except for a reliable Bayesian posterior. Spline bootstraps the mean only; covariance is not bootstrapped; transformer `n_members=1` has no bootstrap spread.
- Replicate uncertainty is the spread of paired Δ across generate/train seed pairs. It is not a license to pick the favorable seed.

## Cells

| cell | sharing | k | stt | models | bayes |
|---|---|---|---|---|---|
| r0_both_context_specific_k0 | context_specific | 0 | None | spline,transformer,bayesian | unreliable/not counted |
| r0_both_pooled_k1 | pooled | 1 | None | spline,transformer,bayesian | unreliable/not counted |
| r0_both_selective_k1 | selective | 1 | None | spline,transformer | n/a |
| r0_both_selective_regularized_k1 | selective_regularized | 1 | None | spline,transformer,bayesian | unreliable/not counted |
| r0_both_context_specific_k0_stt4 | context_specific | 0 | 4 | spline,transformer | n/a |
| r0_both_selective_regularized_k1_stt4 | selective_regularized | 1 | 4 | spline,transformer | n/a |
| r1_both_context_specific_k0 | context_specific | 0 | None | spline,transformer | n/a |
| r1_both_pooled_k1 | pooled | 1 | None | spline,transformer | n/a |
| r1_both_selective_k1 | selective | 1 | None | spline,transformer | n/a |
| r1_both_selective_regularized_k1 | selective_regularized | 1 | None | spline,transformer | n/a |

## Transfer (predose, published as computed)

| tag | model | metric | S0 | Sk | delta | sign |
|---|---|---|---|---|---|---|
| r0|both|sttNone|pooled|bayesian|test_compound_holdout|pre_study | bayesian | rmse_log | 0.13533667707510694 | 0.1283064274127926 | -0.007030249662314331 | helpful |
| r0|both|sttNone|pooled|bayesian|test_compound_holdout|pre_study | bayesian | crps | 0.06767703474560612 | 0.06069708630985725 | -0.006979948435748867 | helpful |
| r0|both|sttNone|pooled|bayesian|test_compound_holdout|pre_study | bayesian | contrast_bias | 0.1418443716562261 | 0.1430561614513839 | 0.0012117897951577916 | harmful |
| r0|both|sttNone|pooled|bayesian|test|pre_study | bayesian | rmse_log | 0.05081486345969391 | 0.060404453326900275 | 0.009589589867206365 | harmful |
| r0|both|sttNone|pooled|bayesian|test|pre_study | bayesian | crps | 0.02923691750155281 | 0.034551096185998366 | 0.005314178684445555 | harmful |
| r0|both|sttNone|pooled|bayesian|test|pre_study | bayesian | contrast_bias | 0.007739834127800394 | -0.021484509100265066 | 0.013744674972464673 | harmful |
| r0|both|sttNone|pooled|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.052108221077582305 | 0.060390582176174205 | 0.0082823610985919 | harmful |
| r0|both|sttNone|pooled|spline|test_compound_holdout|pre_study | spline | crps | 0.02841098628350819 | 0.03352864668031945 | 0.0051176603968112605 | harmful |
| r0|both|sttNone|pooled|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.021231113812280038 | -0.0003179824686451553 | -0.020913131343634882 | helpful |
| r0|both|sttNone|pooled|spline|test|pre_study | spline | rmse_log | 0.08250039372675977 | 0.08046686065906555 | -0.0020335330676942154 | helpful |
| r0|both|sttNone|pooled|spline|test|pre_study | spline | crps | 0.04338474028138843 | 0.04426423710206441 | 0.000879496820675979 | harmful |
| r0|both|sttNone|pooled|spline|test|pre_study | spline | contrast_bias | 0.0038398401275500137 | -0.04499059899652559 | 0.041150758868975575 | harmful |
| r0|both|sttNone|pooled|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.051708385365070934 | 0.056969697955286766 | 0.005261312590215832 | harmful |
| r0|both|sttNone|pooled|transformer|test_compound_holdout|pre_study | transformer | crps | 0.027863732565752763 | 0.03135018133347095 | 0.0034864487677181873 | harmful |
| r0|both|sttNone|pooled|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.05022516406918243 | 0.02477760653381109 | -0.025447557535371344 | helpful |
| r0|both|sttNone|pooled|transformer|test|pre_study | transformer | rmse_log | 0.07983930507230506 | 0.1175207871480674 | 0.03768148207576233 | harmful |
| r0|both|sttNone|pooled|transformer|test|pre_study | transformer | crps | 0.04428351216849094 | 0.07280094405780489 | 0.02851743188931395 | harmful |
| r0|both|sttNone|pooled|transformer|test|pre_study | transformer | contrast_bias | -0.054414788341289946 | -0.12469403975674687 | 0.07027925141545692 | harmful |
| r0|both|sttNone|selective|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.052108221077582305 | 0.0629829625098738 | 0.010874741432291495 | harmful |
| r0|both|sttNone|selective|spline|test_compound_holdout|pre_study | spline | crps | 0.02841098628350819 | 0.03481656069414044 | 0.006405574410632255 | harmful |
| r0|both|sttNone|selective|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.021231113812280038 | -0.0004756390525452233 | -0.020755474759734813 | helpful |
| r0|both|sttNone|selective|spline|test|pre_study | spline | rmse_log | 0.08250039372675977 | 0.08257382464325613 | 7.343091649636424e-05 | harmful |
| r0|both|sttNone|selective|spline|test|pre_study | spline | crps | 0.04338474028138843 | 0.046540478391233056 | 0.0031557381098446247 | harmful |
| r0|both|sttNone|selective|spline|test|pre_study | spline | contrast_bias | 0.0038398401275500137 | -0.04446294365322303 | 0.04062310352567301 | harmful |
| r0|both|sttNone|selective|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.051708385365070934 | 0.06463393944076101 | 0.012925554075690078 | harmful |
| r0|both|sttNone|selective|transformer|test_compound_holdout|pre_study | transformer | crps | 0.027863732565752763 | 0.03575184120505406 | 0.007888108639301294 | harmful |
| r0|both|sttNone|selective|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.05022516406918243 | 0.020601793052154233 | -0.0296233710170282 | helpful |
| r0|both|sttNone|selective|transformer|test|pre_study | transformer | rmse_log | 0.07983930507230506 | 0.1241782287666677 | 0.04433892369436264 | harmful |
| r0|both|sttNone|selective|transformer|test|pre_study | transformer | crps | 0.04428351216849094 | 0.0794152315554778 | 0.035131719386986865 | harmful |
| r0|both|sttNone|selective|transformer|test|pre_study | transformer | contrast_bias | -0.054414788341289946 | -0.10312078072532915 | 0.04870599238403921 | harmful |
| r0|both|sttNone|selective_regularized|bayesian|test_compound_holdout|pre_study | bayesian | rmse_log | 0.13533667707510694 | 0.1387668265776119 | 0.0034301495025049755 | harmful |
| r0|both|sttNone|selective_regularized|bayesian|test_compound_holdout|pre_study | bayesian | crps | 0.06767703474560612 | 0.06827248843761442 | 0.0005954536920083076 | harmful |
| r0|both|sttNone|selective_regularized|bayesian|test_compound_holdout|pre_study | bayesian | contrast_bias | 0.1418443716562261 | 0.14556916155291572 | 0.0037247898966896176 | harmful |
| r0|both|sttNone|selective_regularized|bayesian|test|pre_study | bayesian | rmse_log | 0.05081486345969391 | 0.05632545919530828 | 0.005510595735614368 | harmful |
| r0|both|sttNone|selective_regularized|bayesian|test|pre_study | bayesian | crps | 0.02923691750155281 | 0.03164919537070088 | 0.002412277869148067 | harmful |
| r0|both|sttNone|selective_regularized|bayesian|test|pre_study | bayesian | contrast_bias | 0.007739834127800394 | -0.026191366161344376 | 0.018451532033543983 | harmful |
| r0|both|sttNone|selective_regularized|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.052108221077582305 | 0.06592010147071875 | 0.013811880393136446 | harmful |
| r0|both|sttNone|selective_regularized|spline|test_compound_holdout|pre_study | spline | crps | 0.02841098628350819 | 0.03637987488012641 | 0.00796888859661822 | harmful |
| r0|both|sttNone|selective_regularized|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.021231113812280038 | -0.004946840861651392 | -0.016284272950628648 | helpful |
| r0|both|sttNone|selective_regularized|spline|test|pre_study | spline | rmse_log | 0.08250039372675977 | 0.08942086434144514 | 0.00692047061468537 | harmful |
| r0|both|sttNone|selective_regularized|spline|test|pre_study | spline | crps | 0.04338474028138843 | 0.05093620617179184 | 0.007551465890403411 | harmful |
| r0|both|sttNone|selective_regularized|spline|test|pre_study | spline | contrast_bias | 0.0038398401275500137 | -0.05303259286746538 | 0.049192752739915364 | harmful |
| r0|both|sttNone|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.051708385365070934 | 0.06361801311186646 | 0.011909627746795527 | harmful |
| r0|both|sttNone|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | crps | 0.027863732565752763 | 0.03642757011895536 | 0.008563837553202596 | harmful |
| r0|both|sttNone|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.05022516406918243 | 0.033331062146146145 | -0.016894101923036288 | helpful |
| r0|both|sttNone|selective_regularized|transformer|test|pre_study | transformer | rmse_log | 0.07983930507230506 | 0.09999362203359669 | 0.02015431696129162 | harmful |
| r0|both|sttNone|selective_regularized|transformer|test|pre_study | transformer | crps | 0.04428351216849094 | 0.0610601527943336 | 0.016776640625842658 | harmful |
| r0|both|sttNone|selective_regularized|transformer|test|pre_study | transformer | contrast_bias | -0.054414788341289946 | -0.08035354094077159 | 0.02593875259948164 | harmful |
| r0|both|stt4|selective_regularized|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.05131093331017494 | 0.05914762615200936 | 0.007836692841834421 | harmful |
| r0|both|stt4|selective_regularized|spline|test_compound_holdout|pre_study | spline | crps | 0.028072290535094423 | 0.03302385466380454 | 0.004951564128710114 | harmful |
| r0|both|stt4|selective_regularized|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.02172921659307912 | 0.007446855485229841 | -0.014282361107849278 | helpful |
| r0|both|stt4|selective_regularized|spline|test|pre_study | spline | rmse_log | 0.08256298003365481 | 0.13306558936515164 | 0.050502609331496834 | harmful |
| r0|both|stt4|selective_regularized|spline|test|pre_study | spline | crps | 0.04789405034757579 | 0.0773200509957925 | 0.029426000648216717 | harmful |
| r0|both|stt4|selective_regularized|spline|test|pre_study | spline | contrast_bias | 0.025496561538507607 | -0.0355078955950626 | 0.010011334056554994 | harmful |
| r0|both|stt4|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.052200717750911216 | 0.05593925969670904 | 0.003738541945797827 | harmful |
| r0|both|stt4|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | crps | 0.028141492074896426 | 0.031277697485871225 | 0.003136205410974799 | harmful |
| r0|both|stt4|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.05357653044324612 | 0.029157996320216206 | -0.02441853412302991 | helpful |
| r0|both|stt4|selective_regularized|transformer|test|pre_study | transformer | rmse_log | 0.07465429222757403 | 0.08600079888985585 | 0.011346506662281819 | harmful |
| r0|both|stt4|selective_regularized|transformer|test|pre_study | transformer | crps | 0.04234683638957508 | 0.04936938743956491 | 0.007022551049989834 | harmful |
| r0|both|stt4|selective_regularized|transformer|test|pre_study | transformer | contrast_bias | -0.04516728124026472 | -0.0828970637987217 | 0.037729782558456985 | harmful |
| r1|both|sttNone|pooled|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.057913705174797374 | 0.05605302259866325 | -0.001860682576134122 | helpful |
| r1|both|sttNone|pooled|spline|test_compound_holdout|pre_study | spline | crps | 0.03136298318323936 | 0.03585978345148485 | 0.0044968002682454905 | harmful |
| r1|both|sttNone|pooled|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.027872016032803457 | -0.01582489928393287 | -0.012047116748870587 | helpful |
| r1|both|sttNone|pooled|spline|test|pre_study | spline | rmse_log | 0.05294195816527679 | 0.04947001350810938 | -0.0034719446571674084 | helpful |
| r1|both|sttNone|pooled|spline|test|pre_study | spline | crps | 0.028851722626313178 | 0.03325083513138946 | 0.004399112505076283 | harmful |
| r1|both|sttNone|pooled|spline|test|pre_study | spline | contrast_bias | 0.040813396774000574 | 0.03547787477916455 | -0.005335521994836022 | helpful |
| r1|both|sttNone|pooled|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.05995971098142095 | 0.0668273872959696 | 0.006867676314548647 | harmful |
| r1|both|sttNone|pooled|transformer|test_compound_holdout|pre_study | transformer | crps | 0.03330921523767649 | 0.04010400791984854 | 0.006794792682172049 | harmful |
| r1|both|sttNone|pooled|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.0009636669487844158 | -0.018866408553578247 | 0.017902741604793833 | harmful |
| r1|both|sttNone|pooled|transformer|test|pre_study | transformer | rmse_log | 0.053980288908543866 | 0.06401884558020571 | 0.010038556671661841 | harmful |
| r1|both|sttNone|pooled|transformer|test|pre_study | transformer | crps | 0.030121819482250255 | 0.03873590156534765 | 0.008614082083097392 | harmful |
| r1|both|sttNone|pooled|transformer|test|pre_study | transformer | contrast_bias | 0.0015602921687731203 | -0.0038372034154904263 | 0.002276911246717306 | harmful |
| r1|both|sttNone|selective|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.057913705174797374 | 0.05245219182799329 | -0.005461513346804087 | helpful |
| r1|both|sttNone|selective|spline|test_compound_holdout|pre_study | spline | crps | 0.03136298318323936 | 0.03433186430624862 | 0.002968881123009262 | harmful |
| r1|both|sttNone|selective|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.027872016032803457 | -0.016400911151690924 | -0.011471104881112533 | helpful |
| r1|both|sttNone|selective|spline|test|pre_study | spline | rmse_log | 0.05294195816527679 | 0.05022485604030118 | -0.002717102124975612 | helpful |
| r1|both|sttNone|selective|spline|test|pre_study | spline | crps | 0.028851722626313178 | 0.032561130017144536 | 0.0037094073908313582 | harmful |
| r1|both|sttNone|selective|spline|test|pre_study | spline | contrast_bias | 0.040813396774000574 | 0.03481868740444437 | -0.005994709369556206 | helpful |
| r1|both|sttNone|selective|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.05995971098142095 | 0.062393754868081165 | 0.0024340438866602143 | harmful |
| r1|both|sttNone|selective|transformer|test_compound_holdout|pre_study | transformer | crps | 0.03330921523767649 | 0.036966037333044224 | 0.0036568220953677316 | harmful |
| r1|both|sttNone|selective|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.0009636669487844158 | 0.01621181038687043 | 0.015248143438086016 | harmful |
| r1|both|sttNone|selective|transformer|test|pre_study | transformer | rmse_log | 0.053980288908543866 | 0.06925498780363655 | 0.015274698895092684 | harmful |
| r1|both|sttNone|selective|transformer|test|pre_study | transformer | crps | 0.030121819482250255 | 0.0406618639798228 | 0.010540044497572544 | harmful |
| r1|both|sttNone|selective|transformer|test|pre_study | transformer | contrast_bias | 0.0015602921687731203 | 0.005328738951565502 | 0.003768446782792382 | harmful |
| r1|both|sttNone|selective_regularized|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.057913705174797374 | 0.05275910504278679 | -0.005154600132010585 | helpful |
| r1|both|sttNone|selective_regularized|spline|test_compound_holdout|pre_study | spline | crps | 0.03136298318323936 | 0.034457951347782144 | 0.0030949681645427857 | harmful |
| r1|both|sttNone|selective_regularized|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.027872016032803457 | -0.016513931026651198 | -0.01135808500615226 | helpful |
| r1|both|sttNone|selective_regularized|spline|test|pre_study | spline | rmse_log | 0.05294195816527679 | 0.050648190219554676 | -0.002293767945722114 | helpful |
| r1|both|sttNone|selective_regularized|spline|test|pre_study | spline | crps | 0.028851722626313178 | 0.03276687070104148 | 0.0039151480747283 | harmful |
| r1|both|sttNone|selective_regularized|spline|test|pre_study | spline | contrast_bias | 0.040813396774000574 | 0.03597740732235966 | -0.004835989451640911 | helpful |
| r1|both|sttNone|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.05995971098142095 | 0.06076121913629868 | 0.0008015081548777281 | harmful |
| r1|both|sttNone|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | crps | 0.03330921523767649 | 0.03627283263294563 | 0.0029636173952691386 | harmful |
| r1|both|sttNone|selective_regularized|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.0009636669487844158 | 0.0015337962769002532 | 0.0005701293281158374 | harmful |
| r1|both|sttNone|selective_regularized|transformer|test|pre_study | transformer | rmse_log | 0.053980288908543866 | 0.06337212594853642 | 0.009391837039992551 | harmful |
| r1|both|sttNone|selective_regularized|transformer|test|pre_study | transformer | crps | 0.030121819482250255 | 0.03851463410120008 | 0.008392814618949827 | harmful |
| r1|both|sttNone|selective_regularized|transformer|test|pre_study | transformer | contrast_bias | 0.0015602921687731203 | 0.010511916893043002 | 0.008951624724269882 | harmful |

## Transfer uncertainty across replicates

| model | sharing | metric | n | mean Δ | sd Δ | harmful | helpful |
|---|---|---|---|---|---|---|---|
| bayesian | pooled | contrast_bias | 1 | 0.013744674972464673 | None | 1 | 0 |
| bayesian | pooled | crps | 1 | 0.005314178684445555 | None | 1 | 0 |
| bayesian | pooled | rmse_log | 1 | 0.009589589867206365 | None | 1 | 0 |
| bayesian | selective_regularized | contrast_bias | 1 | 0.018451532033543983 | None | 1 | 0 |
| bayesian | selective_regularized | crps | 1 | 0.002412277869148067 | None | 1 | 0 |
| bayesian | selective_regularized | rmse_log | 1 | 0.005510595735614368 | None | 1 | 0 |
| spline | pooled | contrast_bias | 2 | 0.017907618437069776 | 0.03287076443094362 | 1 | 1 |
| spline | pooled | crps | 2 | 0.002639304662876131 | 0.0024887441176099866 | 2 | 0 |
| spline | pooled | rmse_log | 2 | -0.002752738862430812 | 0.0010171105890538152 | 0 | 2 |
| spline | selective | contrast_bias | 2 | 0.017314197078058403 | 0.03296377162230226 | 1 | 1 |
| spline | selective | crps | 2 | 0.0034325727503379914 | 0.0003915033031203993 | 2 | 0 |
| spline | selective | rmse_log | 2 | -0.001321835604239624 | 0.001973204836749956 | 1 | 1 |
| spline | selective_regularized | contrast_bias | 2 | 0.022178381644137227 | 0.03820408998262917 | 1 | 1 |
| spline | selective_regularized | crps | 2 | 0.005733306982565856 | 0.002571264986013325 | 2 | 0 |
| spline | selective_regularized | rmse_log | 2 | 0.002313351334481628 | 0.006515450569534703 | 1 | 1 |
| transformer | pooled | contrast_bias | 2 | 0.03627808133108711 | 0.04808491586987013 | 2 | 0 |
| transformer | pooled | crps | 2 | 0.018565756986205672 | 0.014073793616303686 | 2 | 0 |
| transformer | pooled | rmse_log | 2 | 0.023860019373712085 | 0.01954650000507334 | 2 | 0 |
| transformer | selective | contrast_bias | 2 | 0.026237219583415794 | 0.03177564322452134 | 2 | 0 |
| transformer | selective | crps | 2 | 0.022835881942279704 | 0.017388940075039806 | 2 | 0 |
| transformer | selective | rmse_log | 2 | 0.02980681129472766 | 0.02055151044549401 | 2 | 0 |
| transformer | selective_regularized | contrast_bias | 2 | 0.017445188661875763 | 0.012011713313445263 | 2 | 0 |
| transformer | selective_regularized | crps | 2 | 0.012584727622396243 | 0.005928260221762055 | 2 | 0 |
| transformer | selective_regularized | rmse_log | 2 | 0.014773077000642086 | 0.007610222534734633 | 2 | 0 |

## Correctness / inference failures

None

## Artifact locations

- parent: `runs/v3_repl`
- freeze lock: `runs/v3_repl/freeze_lock.json`
- component status: `runs/v3_repl/component_status.json`
- transfer uncertainty: `runs/v3_repl/transfer_uncertainty.json`


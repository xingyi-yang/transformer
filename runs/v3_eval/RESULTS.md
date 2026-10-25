# v3 synthetic evaluation benchmark

> **Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.**

Signs are published as computed. The protocol was not retuned after target-test metrics. An unreliable Bayesian fit is not counted as a successful benchmark.

## Freeze lock

- git revision: `68b9144a94bc4ca13b8609348ca1c052b961293d`
- config hash: `fbf4c58c2e5715ae`

## Uncertainty (fixed language)

- Predictive variability: HierarchicalCovariance draws in `sample_study`.
- Parameter uncertainty is incompletely estimated except for a reliable Bayesian posterior. Spline bootstraps the mean only; covariance is not bootstrapped; transformer `n_members=1` has no bootstrap spread.

## Cells

| bank | sharing | k | schema | budget | models | bound | bayes |
|---|---|---|---|---|---|---|---|
| both | context_specific | 0 | variable | primary | spline,transformer,bayesian | [] | not_run (bayesian extra not installed) |
| both | pooled | 1 | variable | primary | spline,transformer,bayesian | [] | not_run (bayesian extra not installed) |
| both | selective | 1 | variable | primary | spline,transformer,bayesian | [] | not_run (bayesian extra not installed) |
| both | context_specific | 0 | fixed | primary | spline | [] | n/a |
| both | pooled | 1 | fixed | primary | spline | [] | n/a |
| both | context_specific | 0 | variable | tight | spline,transformer | [] | n/a |
| both | pooled | 1 | variable | tight | spline,transformer | [] | n/a |
| matched | pooled | 1 | variable | primary | spline,transformer | [] | n/a |
| matched | context_specific | 0 | variable | primary | spline,transformer | [] | n/a |

## Transfer (predose, published as computed)

| tag | model | metric | S0 | Sk | delta | sign |
|---|---|---|---|---|---|---|
| both|fix|primary|pooled|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.05264499081122549 | 0.08442548227024083 | 0.031780491459015334 | harmful |
| both|fix|primary|pooled|spline|test_compound_holdout|pre_study | spline | crps | 0.030927902160109513 | 0.04776868727180505 | 0.016840785111695535 | harmful |
| both|fix|primary|pooled|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.05635998852732042 | -0.09262936905926722 | 0.036269380531946796 | harmful |
| both|fix|primary|pooled|spline|test|pre_study | spline | rmse_log | 0.04334337933424674 | 0.04886048916272816 | 0.005517109828481415 | harmful |
| both|fix|primary|pooled|spline|test|pre_study | spline | crps | 0.026039688897486184 | 0.03130899109944836 | 0.005269302201962176 | harmful |
| both|fix|primary|pooled|spline|test|pre_study | spline | contrast_bias | 0.021373815609509253 | -0.010245873162694423 | -0.01112794244681483 | helpful |
| both|var|primary|pooled|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.04711933490367955 | 0.08442548227024083 | 0.03730614736656128 | harmful |
| both|var|primary|pooled|spline|test_compound_holdout|pre_study | spline | crps | 0.02874126315439209 | 0.04776868727180505 | 0.01902742411741296 | harmful |
| both|var|primary|pooled|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.021990490379103177 | -0.09262936905926722 | 0.07063887868016404 | harmful |
| both|var|primary|pooled|spline|test|pre_study | spline | rmse_log | 0.04221239918094281 | 0.04886048916272816 | 0.00664808998178535 | harmful |
| both|var|primary|pooled|spline|test|pre_study | spline | crps | 0.02592468047255086 | 0.03130899109944836 | 0.005384310626897499 | harmful |
| both|var|primary|pooled|spline|test|pre_study | spline | contrast_bias | 0.010618336697088698 | -0.010245873162694423 | -0.0003724635343942753 | helpful |
| both|var|primary|pooled|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.047926026131722615 | 0.059699213102242084 | 0.011773186970519468 | harmful |
| both|var|primary|pooled|transformer|test_compound_holdout|pre_study | transformer | crps | 0.028967565624924318 | 0.03741613718883709 | 0.008448571563912769 | harmful |
| both|var|primary|pooled|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.06380436433298094 | 0.054870733494124316 | -0.008933630838856621 | helpful |
| both|var|primary|pooled|transformer|test|pre_study | transformer | rmse_log | 0.04048220984972474 | 0.04184951582573417 | 0.0013673059760094322 | harmful |
| both|var|primary|pooled|transformer|test|pre_study | transformer | crps | 0.025145477706380398 | 0.02902314481010016 | 0.0038776671037197633 | harmful |
| both|var|primary|pooled|transformer|test|pre_study | transformer | contrast_bias | -0.0029571728084294042 | 0.005269598335082967 | 0.0023124255266535627 | harmful |
| both|var|primary|selective|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.04711933490367955 | 0.08395833368002828 | 0.03683899877634873 | harmful |
| both|var|primary|selective|spline|test_compound_holdout|pre_study | spline | crps | 0.02874126315439209 | 0.048052341902783104 | 0.019311078748391015 | harmful |
| both|var|primary|selective|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.021990490379103177 | -0.09274365239713966 | 0.07075316201803648 | harmful |
| both|var|primary|selective|spline|test|pre_study | spline | rmse_log | 0.04221239918094281 | 0.048458055869726305 | 0.006245656688783498 | harmful |
| both|var|primary|selective|spline|test|pre_study | spline | crps | 0.02592468047255086 | 0.02919539689183353 | 0.0032707164192826675 | harmful |
| both|var|primary|selective|spline|test|pre_study | spline | contrast_bias | 0.010618336697088698 | -0.0104709071590313 | -0.00014742953805739825 | helpful |
| both|var|primary|selective|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.047926026131722615 | 0.16685809999637582 | 0.11893207386465321 | harmful |
| both|var|primary|selective|transformer|test_compound_holdout|pre_study | transformer | crps | 0.028967565624924318 | 0.10717331016782082 | 0.0782057445428965 | harmful |
| both|var|primary|selective|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.06380436433298094 | -0.11883587708273223 | 0.055031512749751293 | harmful |
| both|var|primary|selective|transformer|test|pre_study | transformer | rmse_log | 0.04048220984972474 | 0.04036798254119747 | -0.0001142273085272677 | helpful |
| both|var|primary|selective|transformer|test|pre_study | transformer | crps | 0.025145477706380398 | 0.026179990261481087 | 0.001034512555100689 | harmful |
| both|var|primary|selective|transformer|test|pre_study | transformer | contrast_bias | -0.0029571728084294042 | 0.010586089706962062 | 0.0076289168985326575 | harmful |
| both|var|tight|pooled|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.046795094218656116 | 0.08671938722068079 | 0.03992429300202467 | harmful |
| both|var|tight|pooled|spline|test_compound_holdout|pre_study | spline | crps | 0.028356596233470636 | 0.04895412902650348 | 0.020597532793032843 | harmful |
| both|var|tight|pooled|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.013989704033985039 | -0.09483288446040772 | 0.08084318042642268 | harmful |
| both|var|tight|pooled|spline|test|pre_study | spline | rmse_log | 0.04143563821529969 | 0.04909749889774141 | 0.007661860682441721 | harmful |
| both|var|tight|pooled|spline|test|pre_study | spline | crps | 0.025612393633502817 | 0.03157961724320296 | 0.005967223609700145 | harmful |
| both|var|tight|pooled|spline|test|pre_study | spline | contrast_bias | 0.00818560280004891 | -0.007478666977829274 | -0.0007069358222196349 | helpful |
| both|var|tight|pooled|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.04903501993347049 | 0.051662873219731995 | 0.002627853286261507 | harmful |
| both|var|tight|pooled|transformer|test_compound_holdout|pre_study | transformer | crps | 0.029649984433834067 | 0.033366439031373776 | 0.0037164545975397097 | harmful |
| both|var|tight|pooled|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.04823474955247353 | 0.07120425553965813 | 0.0229695059871846 | harmful |
| both|var|tight|pooled|transformer|test|pre_study | transformer | rmse_log | 0.04610392394880446 | 0.05211851871430449 | 0.0060145947655000365 | harmful |
| both|var|tight|pooled|transformer|test|pre_study | transformer | crps | 0.027429433234060562 | 0.03260311737065409 | 0.0051736841365935295 | harmful |
| both|var|tight|pooled|transformer|test|pre_study | transformer | contrast_bias | -0.019383583935746962 | 0.029315759058499837 | 0.009932175122752875 | harmful |
| matched|var|primary|pooled|spline|test_compound_holdout|pre_study | spline | rmse_log | 0.04711933490367955 | 0.05113284993033951 | 0.004013515026659961 | harmful |
| matched|var|primary|pooled|spline|test_compound_holdout|pre_study | spline | crps | 0.02874126315439209 | 0.029351495821956265 | 0.0006102326675641763 | harmful |
| matched|var|primary|pooled|spline|test_compound_holdout|pre_study | spline | contrast_bias | 0.021990490379103177 | -0.006805650760445373 | -0.015184839618657804 | helpful |
| matched|var|primary|pooled|spline|test|pre_study | spline | rmse_log | 0.04221239918094281 | 0.04538221398427724 | 0.0031698148033344337 | harmful |
| matched|var|primary|pooled|spline|test|pre_study | spline | crps | 0.02592468047255086 | 0.027584072338849675 | 0.0016593918662988136 | harmful |
| matched|var|primary|pooled|spline|test|pre_study | spline | contrast_bias | 0.010618336697088698 | 0.009697545292732045 | -0.000920791404356653 | helpful |
| matched|var|primary|pooled|transformer|test_compound_holdout|pre_study | transformer | rmse_log | 0.047926026131722615 | 0.05354944288635857 | 0.005623416754635957 | harmful |
| matched|var|primary|pooled|transformer|test_compound_holdout|pre_study | transformer | crps | 0.028967565624924318 | 0.03206032341269272 | 0.003092757787768404 | harmful |
| matched|var|primary|pooled|transformer|test_compound_holdout|pre_study | transformer | contrast_bias | 0.06380436433298094 | 0.048515406424081746 | -0.015288957908899191 | helpful |
| matched|var|primary|pooled|transformer|test|pre_study | transformer | rmse_log | 0.04048220984972474 | 0.047521633598166496 | 0.007039423748441756 | harmful |
| matched|var|primary|pooled|transformer|test|pre_study | transformer | crps | 0.025145477706380398 | 0.028312428971978627 | 0.0031669512655982286 | harmful |
| matched|var|primary|pooled|transformer|test|pre_study | transformer | contrast_bias | -0.0029571728084294042 | -0.005083726441182258 | 0.0021265536327528534 | harmful |

## Failures

None

## Artifact locations

- parent: `runs/v3_eval`
- freeze lock: `runs/v3_eval/freeze_lock.json`
- banks: `runs/v3_eval/banks`
- cells: `runs/v3_eval/cells`
- transfer: `runs/v3_eval/transfer_table.json`


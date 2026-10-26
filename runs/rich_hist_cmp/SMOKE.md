# rich_hist_cmp smoke

Synthetic methodological benchmark. Not real-data validation or operational qualification.

Question: Does additional observed individual history improve future body-weight prediction under this synthetic generator?
Score split: validation  smoke=True
Families independent: True
Full-history keeps all visits: True

## Primary study-balanced MAE on LOGW

- population_only: 0.37391729219570813
- last_value: 0.10179522225227919
- spline: 0.0483574540893968
- latest_weight: 0.09969972752026537
- short_history: 0.0759360349141786
- full_history: 0.10383639625226153

## Paired study deltas (a − b; positive => a worse)

- latest_weight_minus_population_only: -0.2742175646754428 (n_studies=1)
- short_history_minus_latest_weight: -0.02376369260608678 (n_studies=1)
- full_history_minus_short_history: 0.027900361338082935 (n_studies=1)
- full_history_minus_latest_weight: 0.004136668731996154 (n_studies=1)
- full_history_minus_last_value: 0.0020411739999823403 (n_studies=1)
- full_history_minus_spline: 0.05547894216286473 (n_studies=1)

A null does not prove history is useless in real studies. A positive does not validate real data.

Synthetic methodological benchmark. Not real-data validation or operational qualification.

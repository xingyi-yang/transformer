# rich_hist_v1 results

Synthetic methodological benchmark. Not real-data validation or operational qualification.

Reproduction: `send-bw rich-hist --config configs/rich_hist_v1.yaml`
generate_seed=91021 split_seed=92031
send checksum `3b36f81a6da9a3af7d104ef6584d7eab5b4f37bcee881446cf27670923946a3e`
Studies=44 eligible animals=1338 raw obs=33014 cohort obs=28936
Durations=[28.0, 90.0] schedules=['dense', 'twice_weekly', 'weekly']

## Partitions (study-level; animals stay with their study)

- calibration: 6 studies, 188 animals {'RAT': 188} {'RAT': 6}
- test: 7 studies, 260 animals {'RAT': 260} {'RAT': 7}
- train: 20 studies, 748 animals {'RAT': 748} {'RAT': 20}
- validation: 5 studies, 142 animals {'RAT': 142} {'RAT': 5}
- other_species: 6 studies, 0 animals {} {'DOG': 6}

## Observations per animal

Before preprocessing: {'n_obs': 33014, 'n_animals': 1562, 'median': 16.0, 'mean': 21.13572343149808, 'min': 2, 'max': 56}
After preprocessing (include_pretreatment): {'n_obs': 28936, 'n_animals': 1338, 'median': 15.0, 'mean': 21.626307922272048, 'min': 3, 'max': 55}

## Encoder history (full, achieved not forced)

Mean available=10.586487142163998 mean used=10.586487142163998
Later-origin (DAY>=28) mean available=14.545494409120806 fraction>2=1.0

- duration 28.0 origin 7.0: mean available 4.23 (min 2, max 6, n_pairs 1079)
- duration 28.0 origin 14.0: mean available 6.45 (min 3, max 10, n_pairs 1085)
- duration 90.0 origin 14.0: mean available 6.17 (min 3, max 10, n_pairs 1519)
- duration 90.0 origin 28.0: mean available 10.35 (min 4, max 18, n_pairs 1517)
- duration 90.0 origin 42.0: mean available 14.55 (min 6, max 26, n_pairs 1527)
- duration 90.0 origin 56.0: mean available 18.74 (min 8, max 34, n_pairs 1517)

## Missingness and dropout

Dropout dead/moribund=31 recovery sacrifice=269
By dropout kind: {'challenging': {'n_animals': 52, 'n_dropout': 6}, 'ordinary': {'n_animals': 1510, 'n_dropout': 25}}

## Comparison with locked banks

Official 71001: {'generate_seed': 71001, 'n_rat_studies': 12, 'n_dog_studies': 3, 'duration_days': 28, 'schedules': 'random weekly vs twice-weekly', 'typical_encoder_at_day14': '1–2 weekly treatment visits after cohort drop of predose', 'source': 'locked official bank; not regenerated', 'read_only': True}
Larger 81011: {'generate_seed': 81011, 'status': 'documented', 'n_rat_studies': 36, 'n_dog_studies': 6, 'n_studies': 42, 'n_animals': 1360, 'n_weights': 10644, 'duration_days': 28, 'schedules': 'random weekly vs twice-weekly (official generator)', 'source': 'PR #26 documented bank; read-only', 'used_to_retune': False}
This bank: {'generate_seed': 91021, 'n_studies': 44, 'n_animals': 1338, 'n_obs_after': 28936, 'durations': [28.0, 90.0], 'schedules': ['dense', 'twice_weekly', 'weekly'], 'mean_encoder_available': 10.586487142163998}

## Integrity

passed=True

Models were not trained. Generation was not tuned for a model win.

## Proposed later comparison (not run)

Compare on this bank without retuning generation. Population-only uses planned covariates and no individual weights. Latest-weight uses the last observation at or before origin. Short-history uses the last two observed visits. Full-history uses every eligible observed visit through the origin. Score study-balanced MAE on the frozen test studies. Do not use test errors to retune.

Unresolved assumptions:
- Schedules are experimental, not authorized historical protocol aggregates.
- Saturating slope basis is a modelling choice to avoid linear t/28 blow-up at day 90.
- Challenging dropout is later-biased but still independent of weight (not MNAR).
- DOG studies are inventory-only (other_species); this bank is not a sharing experiment.
- 90-day saturating-exponential growth is still one family; not a claim of biological realism.


# rich_hist_v1 results

Synthetic methodological benchmark. Not real-data validation or operational
qualification.

New versioned Generator A bank. Not a correction of locked 71001 / 81011
experiments. Not real-data validation. Not evidence that any model is
superior. Models were not trained.

## Reproduction

```
send-bw rich-hist --config configs/rich_hist_v1.yaml
```

| item | value |
|---|---|
| generate_seed | 91021 |
| split_seed | 92031 |
| version | rich_hist_v1 |
| output | `runs/rich_hist_v1/` |
| send checksum | `3b36f81a6da9a3af7d104ef6584d7eab5b4f37bcee881446cf27670923946a3e` |
| metadata checksum | `8922b1459c8f742e15d1e5ca905e480aaf7944c4541c234fe1fba1e2f91aec9d` |
| splits checksum | `76e61d069bb161b2fc4b828a4eff48a8e674efaaf6edefb77c9629d3d3a8e728` |
| full windows checksum | `9ac125614c66a1daeb3f4b22eb31d1b707c89813c0dadc21b1b316f9c71695b0` |
| integrity | passed |
| models trained | false |
| tuned for a model win | false |

Study IDs are `SYNRH-001` … `SYNRH-044` (no collision with `SYNA-*`).

## Inspect: why official examples had 1–2 encoder visits

1. **Generator produces** predose `DAY` −7 and 0 plus weekly or twice-weekly
   treatment visits through T=28, optional recovery, optional fasted terminal.
2. **Preprocessing drops** baseline / `PRETREATMENT` (`include_pretreatment`
   default false) and fasted rows. Official `max_day=28` also drops later
   recovery.
3. **Encoder keeps** cohort rows with `DAY ≤ cutoff` (official 14), then the
   last `max_encoder_length` (official 8). Weekly animals therefore contribute
   `{7, 14}`. The cap of 8 was not binding.
4. **Audited examples exclude earlier visits** because of the cohort filter,
   not because the generator omitted them.

This bank keeps observed predose (`include_pretreatment=true` here only),
adds 90-day and dense calendars, and uses every observed visit through each
origin unless a model cap is declared.

## Studies and animals

44 studies (38 RAT + 6 DOG). Eligible modelling cohort is RAT only
(1,338 animals). 224 exclusions are the 6 DOG studies (species out of
scope for the target cohort). Raw bank: 1,562 animals, 33,014 weights.

| partition | studies | animals | species |
|---|---|---|---|
| train | 20 | 748 | RAT |
| validation | 5 | 142 | RAT |
| calibration | 6 | 188 | RAT |
| test | 7 | 260 | RAT |
| other_species | 6 | inventory only | DOG |

Paired dense/sparse siblings (6 latent pairs, 12 studies) stay in one
partition. FeatureSpec is fit on training studies only (`max_day=104`
from train, including recovery).

## Durations and schedules (experimental)

No authorized historical schedule metadata. These are experimental
scenarios, not verified real protocols.

| cell | RAT studies | DOG studies |
|---|---|---|
| 28-day weekly | 6 | 2 |
| 28-day twice-weekly | 4 | 0 |
| 28-day dense | 5 | 0 |
| 90-day weekly | 12 | 2 |
| 90-day twice-weekly | 4 | 2 |
| 90-day dense | 7 | 0 |

Dense = every-2-day even days ∪ weekly grid, so weekly is an exact subset
for paired studies. 90-day studies use the saturating slope basis;
28-day studies keep `t/28`. Treatment is 28, 56, or 90 days. Recovery
cohorts add `T+7`, `T+14` when drawn.

## Observations per animal

| | animals | n_obs | median | mean | min | max |
|---|---|---|---|---|---|---|
| before preprocessing | 1,562 | 33,014 | 16 | 21.1 | 2 | 56 |
| after preprocessing | 1,338 | 28,936 | 15 | 21.6 | 3 | 55 |

Official / larger 28-day cohort tables typically had about 4 weekly
treatment visits through day 28 after dropping predose.

## Available vs retained encoder history

Full-history windows use every observed visit with `DAY ≤ origin`.
Achieved, not forced. 8,244 pairs. 1,332 animals. 204 pairs skipped
because the exact `origin+horizon` day was unobserved. 12 animals were
already removed at the origin.

| duration | origin | mean available | min | max | n_pairs |
|---|---|---|---|---|---|
| 28 | 7 | 4.23 | 2 | 6 | 1,079 |
| 28 | 14 | 6.45 | 3 | 10 | 1,085 |
| 90 | 14 | 6.17 | 3 | 10 | 1,519 |
| 90 | 28 | 10.35 | 4 | 18 | 1,517 |
| 90 | 42 | 14.55 | 6 | 26 | 1,527 |
| 90 | 56 | 18.74 | 8 | 34 | 1,517 |

Later origins (DAY ≥ 28): mean 14.55 available visits; **fraction > 2 = 1.0**.

At origin 56 by schedule (achieved):

| schedule | mean | median | min | max |
|---|---|---|---|---|
| weekly | 9.83 | 10 | 8 | 10 |
| twice-weekly | 17.70 | 18 | 16 | 18 |
| dense | 33.27 | 33 | 31 | 34 |

Short-history ablation uses the last 2 visits (max used = 2). A model
cap of 8 truncates 3,866 / 8,244 pairs and reports
`n_encoder_excluded_by_limit` (mean excluded ≈ 4.0). Visit rank is
recorded; calendar `DAY` is retained.

## Missingness, dropout, recovery

- Ordinary missingness: `p_missing=0.02` on `day>0`. Predose −7 and 0
  are never MCAR-dropped.
- Canonical dropout: 19 `DEATH` + 12 `UNSCHEDULED_REMOVAL` (moribund) =
  31, matching hidden truth `DEATH_DAY` counts (19 ordinary-kind studies
  contributed 14; 2 challenging-kind studies, 52 animals, contributed 5).
- Recovery sacrifice: 269.
- Dose deviations in truth: 38. Planned regimen only is usable at origin.
- No post-death BW rows. 2,854 paired overlapping rows have identical
  weights.

## Comparison with locked banks

| | official 71001 | larger 81011 | rich_hist_v1 91021 |
|---|---|---|---|
| studies | 12 RAT + 3 DOG | 36 RAT + 6 DOG | 38 RAT + 6 DOG |
| duration | 28 | 28 | 28 and 90 |
| schedules | weekly / twice-weekly | weekly / twice-weekly | weekly / twice-weekly / dense |
| weights | short 28-day bank | 10,644 | 33,014 raw / 28,936 cohort |
| typical encoder at day 14 | 1–2 weekly treatment visits | 2.75 mean (treatment only) | 6.45 mean (28-day, predose kept) |
| later-origin encoder | not in design | not in design | 14.55 mean at DAY≥28 |
| overwritten? | no | no | new directory only |

## Integrity

Passed. Latent truth is separate. No interpolated window days. No future
rows in the encoder. Splits are disjoint at study level. Calibration is
not validation. FeatureSpec uses production features only. Pair-level
residual counts are **not** claimed to establish conformal validity.

## Unrealistic behavior / unresolved assumptions

- Schedules are experimental, not authorized historical protocol
  aggregates.
- Saturating slope basis is a modelling choice so `t/28` does not explode
  at day 90 (`tau(90)≈1.78` vs 3.21).
- Official saturating-exponential growth still lets some 90-day males
  approach ~600 g. That is the official family, not a claim of biological
  realism.
- Challenging dropout is later-biased but still independent of weight
  (not MNAR).
- DOG studies are inventory-only; this bank is not a sharing experiment.
- Fasted terminals exist in SEND-like files and are excluded from the
  modelling table (`exclude_fasted=true`).

## Proposed comparison (not run)

On the frozen test studies and the same targets, compare:

1. **population-only** — planned covariates, no individual weights
2. **latest-weight** — last observation at or before the origin
3. **short-history** — last two observed visits
4. **full-history** — every eligible observed visit through the origin

Score study-balanced MAE. Do not use test errors to retune generation or
models. Do not treat a win as operational qualification.

Synthetic methodological benchmark. Not real-data validation or
operational qualification.

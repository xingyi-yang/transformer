# Rich-history synthetic bank (v1)

Synthetic methodological benchmark. Not real-data validation or operational
qualification.

This is a **new** Generator A bank (`runs/rich_hist_v1/`,
`generate_seed=91021`). It is not a correction of the locked 71001 / 81011
experiments, not real-data validation, and not evidence that TFT or any
other model is superior.

## 1. Inspect: where visits are produced, dropped, and encoded

Four layers decide how much history a forecast actually sees. Changing the
generator alone does not lengthen an encoder if a later filter still drops
the extra rows.

### How many visits the generator produces

Official `generate_dataset` (no `longitudinal` block) always writes
acclimation/predose `DAY` −7 and 0, then a weekly grid
(`7, 14, …, T`) or a twice-weekly union (`3+7` grids). `T` is appended if
missing. Recovery cohorts add `T+7` and `T+14`. A fasted terminal may add
`T+1` or `T+15`. Visits with `day > 0` are dropped iid with
`p_missing=0.02`. Animals may die (`p_death=0.015`); no rows are written
on or after death. Official durations default to `[28]`.

The 71001 and 81011 banks therefore already contain predose history.
They do not lack earlier visits at generation time.

### Which visits survive preprocessing

`harmonize` marks the last non-fasted visit in the predose window as
`IS_BASELINE` and `PHASE=PRETREATMENT` for `DAY ≤ baseline` or `DAY < 0`.
`CohortConfig` default `include_pretreatment=False` then keeps only
`~IS_BASELINE` and `PHASE != PRETREATMENT`. Fasted terminals are dropped
when `exclude_fasted=True`. Official TFT configs also set `max_day=28`,
which drops recovery visits after day 28.

So the official modelling table starts at the first treatment visit
(weekly: 7; twice-weekly: 3), even though −7 and 0 were generated.

### Which visits enter the encoder

`animal_windows` / `construct_forecast_examples` take observed rows with
`DAY ≤ cutoff` from the **cohort** table, then keep the last
`max_encoder_length` of those rows. Official cutoff is 14;
`max_encoder_length=8`. On a weekly animal the cohort rows at or before
14 are `{7, 14}` — two visits. The cap of 8 is not binding. Twice-weekly
animals contribute `{3, 7, 10, 14}`.

TFT `time_idx` is visit rank among those retained rows. Calendar `DAY`
is still stored but the encoder length is the count of cohort rows, not
the generator calendar.

### Why audited examples exclude available earlier visits

Audited 1–2 visit encoders are a **pipeline filter stack**, not a
generator shortage:

1. Generator writes −7 and 0.
2. Cohort drops pretreatment / baseline.
3. Cutoff 14 on a weekly grid leaves 7 and 14.
4. `max_encoder_length=8` does not restore the dropped predose rows.

`send-bw tft-longer-ctx` reattached observed predose on the same 81011
splits and the same DAY≤14 / 708-pair targets (mean encoder 2.75 → 4.75).
That is a different information set on the **same** short studies. It is
not this benchmark.

This bank instead generates longer calendars, keeps pretreatment in the
cohort table (`include_pretreatment=true` for this run only; official
default remains false), and builds multi-origin windows that use every
observed visit through the origin unless a model cap is declared.

## 2. Design of the new bank

Configurable: study duration, measurement schedule, treatment duration,
recovery duration, ordinary missingness, challenging dropout, slope
basis. Proposed experimental scenarios (not verified real protocols):

- 28-day and ~90-day studies
- weekly / twice-weekly / dense (dense = every 2 days ∪ weekly grid)
- baseline/acclimation `DAY` −7 and 0
- treatment 28, 56, or 90 days; recovery visits `T+7`, `T+14` when the
  recovery cohort is drawn
- additional independent studies, not only more visits in old studies
- paired dense/sparse siblings drawn from one latent path and one
  measurement realization at overlapping days

No authorized historical schedule metadata were available. Assumptions
are labelled experimental.

28-day studies keep the official `t/28` random-slope basis. 90-day
studies use a saturating basis so `tau(28)=1` and `tau(90)≈1.78` instead
of `90/28≈3.21`. Growth remains the official saturating exponential
(`tau_g=40`). Trajectories are sampled from the process at configured
times. Rows are not interpolated or duplicated to fake density.

Ordinary missingness (`p_missing` on `day>0`) and challenging dropout
(later-biased death, still independent of weight) are separate knobs.
Post-death measurements are never written.

## 3. Forecasting windows

Each eligible animal and each origin appropriate to the study duration
yields pairs at explicit calendar horizons (7 and 14 days). Fields:

- forecast origin
- last observed measurement day
- encoder visit count (available and used)
- encoder history span
- target day
- lead from origin
- lead from last observation

Full history uses every observed visit with `DAY ≤ origin`. Short
history (ablation) uses the last two. A model cap of 8 is applied only
in the truncated table and reports `n_encoder_excluded_by_limit`.
Visit-rank `time_idx` is recorded and documented; calendar `DAY` is
retained. Future LOGW, future missingness, death after the origin, and
future dosing deviations are not origin features.

## 4. Leakage protection

Latent truth stays in `truth/`. Canonical SEND-like tables do not carry
`Z_LATENT` / `Z0_TRUE`. FeatureSpec is fit on training studies only and
uses production features (planned regimen, biology, baseline, time).
Study-level partitions are train / validation / calibration / test,
frozen before any later fit. Paired siblings stay in one partition.
Calibration is not validation. Animals and their windows follow their
study. Pair-level residual counts do not establish conformal validity.

## 5. Reproduction

```
send-bw rich-hist --config configs/rich_hist_v1.yaml
```

Seeds: `generate_seed=91021`, `split_seed=92031`. Output:
`runs/rich_hist_v1/` only. Locked `runs/v3_*`, TFT, and audit
directories are refused. Models are not trained.

Synthetic methodological benchmark. Not real-data validation or
operational qualification.

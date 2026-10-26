# rich_hist_cmp results

Synthetic methodological benchmark. Not real-data validation or
operational qualification.

Question: does additional observed individual history improve future
body-weight prediction under this synthetic generator?

A null is not proof that history is useless in real studies. A positive
is not real-data validation.

Reproduction:

```
send-bw rich-hist-cmp --config configs/rich_hist_cmp.yaml --smoke
send-bw rich-hist-cmp --config configs/rich_hist_cmp.yaml
send-bw rich-hist-cmp --config configs/rich_hist_cmp.yaml --score-only
```

Output: `runs/rich_hist_cmp/` only. Bank: frozen `runs/rich_hist_v1/`
(`generate_seed=91021`, `split_seed=92031`). Train seed: `93041`.
Bank checksum (SEND):
`3b36f81a6da9a3af7d104ef6584d7eab5b4f37bcee881446cf27670923946a3e`.
Full-window checksum:
`9ac125614c66a1daeb3f4b22eb31d1b707c89813c0dadc21b1b316f9c71695b0`.
Trajectories were not regenerated or tuned.

## Phase 1 — preflight

| Check | Result |
|---|---|
| Independent trajectory families | 32 (38 RAT studies). Train 16 / val 5 / cal 5 / test 6 |
| Crossing families | 0. Existing bank split copied (`split_source=bank_preserved`) |
| Cohort RAT animals | 1,338 |
| Window animals | 1,332 |
| Missing from windows | 6 (3 train, 3 test) |
| Missing reasons | already dead/removed at every origin (2 death-only; 4 death plus no observed target) |
| Full-history keeps all visits | yes. Mean available = consumed = 10.59 on 8,244 pairs; 0 truncated |
| Legacy `max_encoder_length=8` | would drop visits on 3,866 pairs; not used |
| Test encoder | 1,289 pairs; mean available 9.25; full consumed 9.25; truncated 0 |
| Loss scale | raw LOGW (decoder y mean 6.00; TFT prediction mean 5.98) |

Excluded animals: `SYNRH-003-0004` (train, unscheduled removal),
`SYNRH-029-0019` (train, death), `SYNRH-030-0019` (train, unscheduled
removal), `SYNRH-009-0012` (test, unscheduled removal),
`SYNRH-011-0020` (test, death), `SYNRH-022-0013` (test, death).

Protocol was written before test scores were inspected. Model selection
used validation only. Calibration was used only for the experimental CQR
margin. Test was not used to tune.

## Phase 2–3 — matched test scores

Primary metric: study-balanced MAE on LOGW. Same 1,289
`(USUBJID, origin, target_day)` keys for every variant. Dense schedules
are not given extra weight through row count.

| Variant | Study-balanced MAE | Family-balanced | Schedule-balanced | n | 80% cov (study-bal) | 80% width |
|---|---|---|---|---|---|---|
| A population-only | 0.8946 | 0.8210 | 0.9922 | 1289 | 0.00 (point) | 0.00 |
| last-value | 0.1014 | 0.1011 | 0.1033 | 1289 | — | 0.00 |
| spline | 0.0496 | 0.0488 | 0.0474 | 1289 | 0.927 | 0.242 |
| B latest-weight | **0.0346** | **0.0336** | **0.0354** | 1289 | 0.748 | 0.099 |
| C short-history | 0.0424 | 0.0432 | 0.0424 | 1289 | 0.764 | 0.122 |
| D full-history | 0.0467 | 0.0478 | 0.0430 | 1289 | 0.760 | 0.120 |
| D + pair-level CQR | 0.0467 | 0.0478 | 0.0430 | 1289 | 0.792 | 0.131 |

TFT failures: 0. B–D were trained separately (`max_encoder_length` 1 / 2
/ 40). Baseline `ZB_C` is identical across B–D. Predict clip was 1000, not
14. Full-history did not inherit `max_encoder_length=8`.

### Paired study deltas (a − b; positive => a worse)

| Pair | Mean Δ | Studies a better | Studies b better |
|---|---|---|---|
| latest − population | −0.8600 | 7 | 0 |
| short − latest | +0.0078 | 2 | 5 |
| full − short | +0.0043 | 2 | 5 |
| full − latest | +0.0121 | 2 | 5 |
| full − last-value | −0.0547 | 7 | 0 |
| full − spline | −0.0029 | 4 | 3 |

### Per-study MAE (LOGW)

| Study | Family | n | last-value | pop | spline | latest | short | full |
|---|---|---|---|---|---|---|---|---|
| SYNRH-006 | SYNRH-006 | 178 | 0.1127 | 1.2831 | 0.0350 | 0.0375 | 0.0451 | 0.0375 |
| SYNRH-009 | SYNRH-009 | 119 | 0.1049 | 1.2440 | 0.0313 | 0.0305 | 0.0464 | 0.0606 |
| SYNRH-011 | SYNRH-011 | 118 | 0.1260 | 0.2795 | 0.0521 | 0.0329 | 0.0537 | 0.0618 |
| SYNRH-012 | SYNRH-012 | 317 | 0.0876 | 0.4585 | 0.0616 | 0.0298 | 0.0423 | 0.0446 |
| SYNRH-022 | SYNRH-022 | 241 | 0.0721 | 0.3248 | 0.0580 | 0.0302 | 0.0349 | 0.0427 |
| SYNRH-027 | SYNRH-P01 | 158 | 0.1022 | 1.3362 | 0.0545 | 0.0405 | 0.0399 | 0.0472 |
| SYNRH-028 | SYNRH-P01 | 158 | 0.1042 | 1.3362 | 0.0545 | 0.0406 | 0.0344 | 0.0325 |

Test uncertainty is over 6 independent trajectory families (7 studies;
SYNRH-027/028 share a latent path).

### Full-history by origin, horizon, schedule, duration

| Slice | Study-balanced MAE |
|---|---|
| origin 7 / 14 / 28 / 42 / 56 | 0.0486 / 0.0449 / 0.0440 / 0.0369 / 0.0545 |
| elapsed 7 / 14 days | 0.0410 / 0.0524 |
| dense / twice-weekly / weekly | 0.0531 / 0.0375 / 0.0385 |
| 28-day / 90-day | 0.0479 / 0.0436 |
| available visits 1–2 / 3–4 / 5–8 / 9–16 / 17+ | 0.0641 / 0.0351 / 0.0506 / 0.0538 / 0.0426 |

Latest-weight remains lower than full-history in the 17+ available-visit
bin (0.0301 vs 0.0426). Extra consumed visits did not buy accuracy here.

CQR used 1,467 calibration pairs, Romano
`k=ceil((n+1)(1-α))`, margin 0.0054. Pair-level residuals. This does not
establish standard conformal coverage.

## Answer to the research question

Under this synthetic generator, adding individual weight information
beyond planned covariates helps: population-only is far worse than every
history-using method. Among matched TFT variants that already see
baseline `ZB_C` and the latest visit, using the last two visits or the
full encoder did **not** improve study-balanced, family-balanced, or
schedule-balanced MAE relative to latest-weight. Latest-weight also beat
last-value and the existing spline.

That is a result about this Generator A bank and these locked TFT knobs.
It is not proof that longitudinal history is useless in real studies, and
it is not real-data validation or operational qualification. The
generator is a smooth latent growth process plus small OU noise; a
strong latest-visit signal is expected. The result was not retuned.

Synthetic methodological benchmark. Not real-data validation or
operational qualification.

# rich_hist_cmp: matched history-length comparison

Synthetic methodological benchmark. Not real-data validation or
operational qualification.

This experiment answers one question on the **frozen** `rich_hist_v1`
bank (`generate_seed=91021`, `split_seed=92031`):

> Does additional observed individual history improve future body-weight
> prediction under this synthetic generator?

It does **not** regenerate or retune trajectories. It does **not**
overwrite locked `runs/v3_*`, TFT, audit, or `runs/rich_hist_v1/`
directories. A null result is not proof that history is useless in real
studies. A positive result is not real-data validation.

## Phase 1 — preflight before training

1. **Independence at the trajectory-family level.** Dense/sparse siblings
   share a latent path. `trajectory_family_id` is `PAIR_ID` when present
   and `STUDYID` otherwise. Study-ID separation alone is not enough.
   Every sibling family must sit in exactly one partition. On this bank
   that is 38 RAT studies / 32 independent families (16/5/5/6 by
   train/val/cal/test) and zero crossing families, so the existing
   manifest is copied rather than rewritten. If a family crosses, fitting
   on the leaked split stops; the bank manifest is preserved; a corrected
   family-grouped split is written only under `runs/rich_hist_cmp/`;
   FeatureSpec is refit on the corrected training partition.
2. **Inventory.** The bank lists 1,338 eligible RAT animals across
   partitions and 1,332 animals in forecasting windows. The six missing
   animals are already dead or removed at every origin (3 train, 3 test).
3. **Actual model inputs.** Full-history windows keep every eligible
   observed visit through the origin (`max_encoder_length=40`; bank max
   available = 34). A legacy cap of 8 would silently truncate thousands
   of pairs and is not used. Predict clip is raised to day 1000 so later
   origins are not recut at day 14.
4. **Frozen protocol** is written to `PROTOCOL.json` before test scores
   are inspected: origins, calendar-day horizons, matched targets, model
   knobs, training budget, seeds, metrics, and the validation-only
   selection rule.

## Phase 2 — matched history comparison

| Variant | Individual weight input | Baseline `ZB_C` |
|---|---|---|
| A population-only | none (planned biology / regimen / descriptors + time) | no |
| B latest-weight | last observed visit | yes, identical to C–D |
| C short-history | last two observed visits | yes, identical to B, D |
| D full-history | all eligible observed visits through the origin | yes, identical to B, C |
| last-value | last encoder LOGW, no covariates | n/a |
| spline | existing spline on Y; baseline + planned features; no visit history | yes |

B–D are **separately trained** TFT adapters with the locked knobs
(`hidden=16`, `lr=0.01`, `dropout=0.1`, 16 epochs, `window_balance=study`,
`target_normalizer=species_or_global`) and only `max_encoder_length`
changed (1 / 2 / 40). Inference-only truncation of one checkpoint is not
this experiment.

Eligible `(USUBJID, origin, target_day)` keys are identical across
variants. Dense schedules are not given extra evaluation weight just
because they contribute more rows: primary aggregation is study-balanced
MAE on LOGW; family-balanced and schedule-balanced MAE are also
reported.

## Phase 3 — training and evaluation

- Train on train; select on validation; calibrate intervals on
  calibration only; score test once.
- Loss scale is resolved by a one-batch trace of the pinned TFT
  (decoder y and `TFT.forward` on raw LOGW).
- CQR uses pair-level Romano residuals on calibration. That does not
  establish standard conformal coverage.
- Pipeline smoke (`--smoke`) scores a development slice of validation
  and never inspects test.

```
send-bw rich-hist-cmp --config configs/rich_hist_cmp.yaml --smoke
send-bw rich-hist-cmp --config configs/rich_hist_cmp.yaml
```

Output directory: `runs/rich_hist_cmp/` only. Train seed: `93041`.

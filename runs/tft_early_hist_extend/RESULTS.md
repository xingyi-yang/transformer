# TFT early-history extension

Synthetic diagnostic. Not design qualification. No real-data validation. No operational recommendation.

This extends the completed pilot `runs/tft_early_hist_bench` (Generator A, `generate_seed=71001`, RAT, TFT `train_seed=82001`). That directory and all locked `runs/v3_*` directories were not written. New artifacts are only in `runs/tft_early_hist_extend/`.

Forecast windows were taken from each generated dataset's observed `DAY` values. Unmeasured days were not created. A regular calendar was not imposed. Long-horizon chunking was not used.

## Predeclared stability plan

Locked in `configs/tft_early_hist_extend.yaml` and `predeclared.json` before new scores were examined:

- Official holdout: generate_seed 71001, test SYNA-010 / SYNA-011, TFT train_seed 82001 (pilot reference; its test errors were not used to tune later models).
- Extra TFT train seed: 82011 on the same official holdout.
- Extra whole-study split seed: 33001 of the same generated bank (FeatureSpec refit; spline / transformer / CatBoost / TFT refit). Donors remained train-only for pooled.
- Extra-split TFT seed 82001; extra-split mean-model seed 81001.
- Duration rule: if the diagnostic replay's validation loss was still best at the last 8-epoch point, new TFT fits used `max_epochs=16` / `patience=6`. That decision used validation only.

## Fair forecasting task

- Cutoff DAY ≤ 14. Encoder visit cap 8. Prediction-length cap 6. No chunking.
- Encoder and target days were derived separately for each animal from observed rows.
- Official holdout: 58 eligible animals, 115 scored (USUBJID, DAY) pairs. Zero exclusions.
- All methods on a bank were scored on identical eligible pairs.
- Configured caps vs exercised lengths (`history_lead_report.json`): encoder used 1–2 visits (cap 8 never bound); future used 1–2 visits (cap 6 never bound); calendar leads 7 / 14 / 21 days from the last encoder DAY. Typical encoder is days 7 and 14; typical targets are 21 and 28.

Information-set differences (`information_sets.json`):

| model | train targets | test history | covariates / intervals |
|---|---|---|---|
| last_value | none | last encoder LOGW | none; intervals unavailable |
| spline / GAMM | all observed train days | encoder LOGW via Gaussian residual conditioning | biology, planned regimen, B-spline time/exposure; Gaussian layer |
| custom transformer | all observed train days | same Gaussian residual layer | baseline-conditioned tokens; not TFT |
| CatBoost | one row per train animal × cutoff origin × observed future DAY | encoder summaries only | biology, planned regimen, history summaries, target DAY, elapsed times; q0.1/0.5/0.9 |
| TFT | PF sliding observed windows (not only the cutoff) | observed encoder visits DAY≤cutoff | static + known-future planned-dose features; QuantileLoss |

## Custom-transformer provenance

Loaded from `runs/v3_official_seq/.../d0_t0_matched_target_only/models/transformer/`. Not TFT.

- FeatureSpec: context_specific, species RAT, strains SPRAGUE-DAWLEY / WISTAR HAN, route ORAL GAVAGE; fit on official_seq training studies only.
- Split: frozen official_seq whole-study splits (train SYNA-002/003/004/006/007/009; val SYNA-005/012; test SYNA-010/011).
- Training: `train_seed=81001`, 1 member, 8 epochs, best validation RMSE 0.0654 at epoch 6. Checkpoint = lowest val RMSE among logged epochs, member 0.
- Label: pretrained official_seq reference. Not refit on TFT seeds. Refit only on the extra split (FeatureSpec and covariance refit there).

## Metrics

- `mae_log` / `rmse_log`: study-balanced |point − LOGW| and true RMSE. LOGW = ln(WEIGHT_G grams). RMSE is per-study sqrt(mean e²), then the mean of studies.
- `mae_gram`: study-balanced |exp(point) − WEIGHT_G| in grams.
- Pinball: quantile loss on LOGW and, separately, on grams at common levels 0.1 / 0.5 / 0.9.
- `coverage80` and `interval_width80` are reported together on LOGW (q0.1–q0.9).
- Persistence interval metrics are unavailable: last-value repeats the last encoder LOGW at every quantile. No interval method was defined or calibrated without test outcomes.
- Studies are averaged after within-study aggregation. Repeated visits within animals or studies are not independent replications.

## Official-holdout comparison (study-balanced; identical 115 pairs)

| model | MAE log | RMSE log | MAE g | pinball | cov80 | width80 | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|
| last_value (reference) | 0.1140 | 0.1307 | 34.45 | 0.0570 | n/a | n/a | 0 |
| spline / GAMM | 0.0450 | 0.0576 | 13.70 | 0.0159 | 0.803 | 0.144 | 0 |
| custom transformer (pretrained) | 0.0438 | 0.0557 | 13.80 | 0.0153 | 0.687 | 0.113 | 0 |
| CatBoost quantile | 0.0387 | 0.0484 | 11.77 | 0.0115 | 0.747 | 0.103 | 0.026 |
| TFT target-only 82001 (pilot) | 0.0489 | 0.0610 | 14.48 | 0.0174 | 0.773 | 0.150 | 0 |
| TFT pooled 82001 (pilot) | 0.0583 | 0.0674 | 17.00 | 0.0193 | 0.677 | 0.143 | 0.330 |
| TFT pooled 82001 rearranged | 0.0583 | 0.0674 | 17.00 | 0.0192 | 0.689 | 0.147 | 0 |
| TFT target-only balanced 82001 | 0.0468 | 0.0598 | 13.91 | 0.0169 | 0.659 | 0.125 | 0.052 |
| TFT pooled balanced 82001 | 0.0624 | 0.0856 | 20.55 | 0.0251 | 0.515 | 0.118 | 0.078 |
| TFT target-only seed 82011 | 0.0450 | 0.0573 | 13.52 | 0.0162 | 0.689 | 0.115 | 0 |
| TFT pooled seed 82011 | 0.0478 | 0.0593 | 14.07 | 0.0157 | 0.738 | 0.122 | 0.070 |

Paired study-level MAE-log deltas (SYNA-010, SYNA-011; Δ = other − TFT target-only; negative means the other model is better):

- CatBoost minus TFT TO: −0.0102 (both studies: −0.0016 and −0.0221).
- Spline minus TFT TO: −0.0039 (mixed sign: +0.0071 on SYNA-010, −0.0149 on SYNA-011).
- Transformer minus TFT TO: −0.0051 (both studies).
- Pooled TFT minus TFT TO: +0.0094 (both studies; pooling worse).
- Balanced pooled minus balanced TO: +0.0156 (both studies; pooling still worse).

## Effective study weighting

Intended objective: equal average of per-study forecast losses.

PF QuantileLoss reduces decoder-step `WEIGHT` after sliding-window construction. Raw window counts omit that.

**Pooled train, before (row-level study `WEIGHT` + uniform windows):** 2166 windows, 9 studies. Effective loss-share range 0.137. Window-share range 0.225. SYNA-D001 596 windows / 27.5% of windows; SYNA-003 502 windows / 23.2%. Target species: 1327 windows / decoder-weight sum 2893; donors: 839 windows / 1744.

**After the justified correction (`WEIGHT=1` + `WeightedRandomSampler` p ∝ 1/n_windows[study]):** each study has equal window-draw probability. Residual imbalance is decoder length (SYNA-003 and SYNA-D001 still ~18% of expected step mass vs ~9% for short studies). Double balancing was not applied.

**Target-only train:** 1327 windows, 6 studies. SYNA-003 dominates decoder steps (1816 / 3282 after WEIGHT=1). Imbalance confirmed; the same sampler was used on new TFT fits.

The correction did not make pooling helpful: balanced pooled MAE 0.0624 vs balanced TO 0.0468.

## Pooled TFT quantile crossings

Pilot pooled 82001 (raw):

- Any crossing: 0.330 (38/115 rows).
- Inner adjacent only: all 0.330 is `q0.1 > q0.25`. Other pairs, including `q0.1 > q0.9`, are 0.
- Endpoint reversed: 0. The 80% interval endpoints are not flipped.
- By lead: 0.29 at 7 days (n=56), 0.37 at 14 days (n=57), 0.50 at 21 days (n=2).
- Species: test is RAT only.

Replay curves (same seed, validation only): val_loss 0.102 → 0.032 over 8 epochs; last epoch was the best. Eight epochs were not a clear plateau, so new fits used 16 epochs / patience 6. That is not a claim that the official 82001 pilot was under-trained into a better test score; the official 82001 numbers stay the 8-epoch reference. The replay's test MAE matched the pilot (0.0583) and is labeled diagnostic, not a champion.

Quantile rearrangement (`np.sort` per row) is postprocessing. Raw and corrected scores are both stored. Rearrangement zeros crossings and changes coverage 0.677 → 0.689 / width 0.143 → 0.147; MAE is unchanged. That is not improved training.

New balanced/extra-seed pooled fits still show 5–8% inner crossings. Target-only 82001 remains crossing-free.

## Extra whole-study split (seed 33001)

Same generated bank. Test SYNA-007 / SYNA-010 (87 scored pairs). Train has 8 RAT studies (official holdout had 6). Donors SYNA-D001/D002/D003 train-only for pooled. FeatureSpec refit. Official SYNA-010/011 holdout was not reused as the scored test except that SYNA-010 appears again by chance.

| model | MAE log | RMSE log | MAE g | cov80 | width80 |
|---|---:|---:|---:|---:|---:|
| last_value | 0.0984 | 0.1133 | 29.63 | n/a | n/a |
| spline / GAMM (refit) | 0.0292 | 0.0370 | 8.73 | 0.966 | 0.378 |
| custom transformer (refit) | 0.0314 | 0.0402 | 9.43 | 0.874 | 0.123 |
| CatBoost (refit) | 0.0355 | 0.0417 | 10.31 | 0.701 | 0.106 |
| TFT target-only | 0.0488 | 0.0606 | 15.30 | 0.643 | 0.100 |
| TFT pooled | 0.5501 | 1.1689 | 1196 | 0.552 | 0.181 |

Do not compare these errors to the official holdout as if they share a test set. On this split, spline is best; TFT target-only is worse than all three learned baselines; pooled TFT collapsed on both test studies (MAE 0.65 / 0.45). Extra-split spline coverage 0.97 with width 0.38 (0.62 on SYNA-010) is over-wide, not a reason to prefer it as a calibrated interval.

## Stability

Target-only TFT MAE log on matching banks: 0.0489 (official 82001), 0.0468 (balanced 82001), 0.0450 (seed 82011), 0.0488 (extra split). Mean 0.0474, sd 0.0018 across those four fits. Stable and consistently worse than CatBoost / spline / transformer on the same bank.

Pooled TFT: 0.0583 (official), 0.0624 (balanced), 0.0478 (seed 82011), 0.550 (extra split). Not stable. The extra-split collapse is a pooled-training failure, not a scoring-pair mismatch (target-only TFT on the same pairs is 0.0488).

Training cost (TFT, CPU): replay 2166 windows / 9 epochs / 536 updates (unbalanced pooled). Official-holdout balanced TO: 1327 windows, 13–17 epochs, 492–656 updates. Balanced pooled: 2166 windows, 17 epochs, 1072 updates. Extra-split pooled: 2728 windows, 16 epochs, 1275 updates. CatBoost: ~3 s / fit, 200 trees, depth 4, three separate Quantile:alpha heads. Spline/transformer extra-split refits used official-seq budgets (8 epochs, 1 transformer member, 4 spline bootstrap members).

## Assessment

**Does target-only TFT improve over the new baselines on this generator?** No. On the official holdout, CatBoost (0.0387), the custom transformer (0.0438), and the spline/GAMM (0.0450) all beat target-only TFT (0.0489) on study-balanced MAE and RMSE. CatBoost also has the lowest pinball and a tighter 80% interval with near-nominal coverage. The extra split repeats the sign: spline 0.029, transformer 0.031, CatBoost 0.035, TFT TO 0.049. TFT's only advantage on the official holdout is a slightly closer-to-nominal 80% coverage than the transformer (0.77 vs 0.69) at a wider interval.

**Does pooling help the target species under balanced and comparable conditions?** No on the official holdout. Pilot pooled 0.0583 vs TO 0.0489; balanced pooled 0.0624 vs balanced TO 0.0468; both test studies agree. Extra seed 82011 shrinks the pooled gap (0.0478 vs 0.0450) but does not reverse it. The extra split is harmful (0.550 vs 0.049). Donor windows are a large share of pooled training (839/2166); the study sampler reduces but does not eliminate long-series dominance.

**Are the conclusions stable?** The target-only ranking (learned baselines ≥ TFT TO ≫ last-value; pooling not helpful) holds across the predeclared extra seed and the extra split. Absolute extra-split errors are not comparable to the official holdout. Pooled TFT is not stable. Two studies per holdout is a small study-level sample; within-study visits were not treated as replications.

## What this does not establish

Conclusions are limited to Generator A and the evaluated settings (RAT target, DAY≤14, encoder cap 8, horizon cap 6, compact TFT). Successful training and passing unit tests are not forecasting effectiveness. No historical experimental body-weight studies were opened. TFT remains optional and early-history only. This is not a sharing verdict, a design qualification, or an operational recommendation.

## Tests executed

- `pytest tests/unit/test_tft_early_hist_extend.py tests/unit/test_tft_early_hist_bench.py` — 19 passed (DAY-derived windows, CatBoost history summaries without invented slopes, crossing pair classification, rearrangement vs raw, decoder-weight audit vs window counts, persistence intervals unavailable, gram MAE, extra-split donors train-only, refuse pilot/v3 writes).
- `pytest tests/unit/test_optional_tft.py` — existing TFT adapter tests, including the extra-present fit (3 skipped when extras absent).
- `ruff check` on the edited modules.

CatBoost 1.2.10 required `loss_function='Quantile:alpha=q'` (not `quantile_alpha=`). The first extend run recorded that failure; CatBoost was then fit into the existing directory without refitting TFT. Official 82001 pilot scores were copied, not recomputed as champions.

## Artifacts

`runs/tft_early_hist_extend/`: freeze_lock, predeclared, versions, examples, encoder/target obs, predictions, per-study metrics, paired diffs, weight audit, crossing audit, replay curves, duration decision, extra-split bank, trajectory plots on actual DAY, SUMMARY.json.

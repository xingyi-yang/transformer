# Frozen v3 research evaluation protocol

**Status: frozen before generate/train.** Do not edit
`configs/v3_research.frozen.yaml` after the freeze commit. Do not retune after
viewing target-test metrics. Do not overwrite `runs/v3_repl/`, `runs/v3_eval/`,
or `runs/v3_pilot_k0k1/`.

> **Synthetic diagnostic. Not design qualification. No real-data validation.
> No operational recommendation.**

The executed v3 replicated benchmark is accepted as a synthetic prototype.
Negative transfer is a published result. This protocol strengthens evidence; it
does not add model families or redesign the generator to produce favorable
signs.

Effective configuration: `configs/v3_research.frozen.yaml`

The executor writes `runs/v3_research/freeze_lock.json` with the git revision
and config hash **before** generate/train. That lock is the execution identity.

## Independent verification (prototype, not retuned)

Sharing structures remain distinct:

- `context_specific` / `pooled` unchanged from stages 1–2.
- Selective v1: shared compound + regimen; species-specific bio / baseline /
  species tokens; residual `target_species_only`.
- Selective regularized retains v1 residual isolation and adds ridge-inflated
  SPECIES×time (growth) and SPECIES×exposure (treatment) deviations
  (`lambda_growth` / `lambda_treatment`, default 8). It is not an unpenalized
  species random-effect model.

Sparse target-study training is a different kind from sparse history encoders
(`sparse_target_study_training` ≠ `sparse_history_encoder`). It drops target
TRAIN studies by STUDYID sort and leaves validation/test hashes unchanged.
Prototype r0 cells share validation hash `37fb8b3fef2c9ca6` and test hash
`298a3ce08799bc77` across sharing regimes and the `stt4` cell.

Metric definitions used for transfer:

- RMSE / CRPS: study-balanced means; Δ = S_k − S_0; Δ>0 harmful.
- Contrast bias in transfer: contrast-row mean; Δ = |S_k| − |S_0|.
- Study-balanced contrast bias is reported in evaluation-pack denominators
  and is not substituted into the transfer Δ.

Provenance: prototype freeze lock `retune_after_test=false`; training revision
`fbf2196`; config hash `a9f9568910bf8669`; generate_seed `20261026`; train_seed
`4242`. Record commit `f9cee98`.

Slow-test coverage still marks Bayesian smoke
(`tests/integration/test_bayesian_smoke.py`) and covariance analysis
(`tests/unit/test_covariance_analysis.py`). Reliability in the Bayesian smoke
is a computed decision against `rhat≤1.01`, `ESS≥400`, `divergences==0`.

## Bayesian sampling diagnosis (prototype; thresholds unchanged)

Compile/backend is classified separately from sampling. The first prototype
attempt failed with a nutpie `CompileError` (`Python.h` missing). After the
header package was installed, three matched-data cells sampled and wrote
diagnostics. Those cells are **unreliable** and **not counted**.

| cell | draws×chains | rhat_max | ESS bulk/tail | divergences | worst rhat family | class |
|---|---|---|---|---|---|---|
| context_specific | 400×2 | 1.084 | 29 / 63 | 0 | theta / beta | mean_identifiability |
| pooled | 400×2 | 1.040 | 65 / 38 | 0 | sigma_p / ell / sd_* | covariance_mixing |
| selective_regularized | 400×2 | 1.037 | 58 / 102 | 0 | sigma_p / ell / sd_* | covariance_mixing |

Budget: 800 posterior draws cannot be declared `inadequate_budget` against an
ESS floor of 400; the failure is mixing / identifiability, not a missing
header. Priors were unchanged (`s_beta=0.35`, `s_sd=0.3`, `s_sigma=0.3`,
`m_ell=log 7`, `s_ell=0.75`). Parameterization remains QR+NUTS, ZeroSum
study/animal effects, OU residual, pivoted-QR drop of non-identifiable mean
columns. Prior predictive on the context-specific cell was marked plausible
(covers, does not match, observed percent change). This diagnosis does **not**
change parameterization, priors, or the reliability gate.

The research protocol requests two Bayesian cells at 800/800/2 as a **budget
check** on context-specific, matched and conflicting banks, dataset 0 / train
0 only. If still unreliable, the class is recorded and the cells are not
counted.

## Frozen research choices

| Item | Value |
|---|---|
| Task | predose-conditioned only |
| Dataset seeds | 2 (`generate_seed + i×1009`, base 20261027) |
| Train seeds | 2 per dataset (`train_seed + j×17`, base 5150) |
| Seed separation | generate ≠ train; nested crossing |
| Target splits | frozen once per (dataset, train); reused across sharing and banks |
| Banks | `matched` and `both` (conflicting growth+treatment donors) |
| Sharing | context_specific, pooled, selective v1, selective_regularized |
| Primary contrast | full versus sparse target-study training (`n_keep=4`) |
| Models | spline; transformer `n_members=1`; Bayesian only as above |
| Reliability gate | rhat≤1.01, ESS≥400, divergences==0 |
| Design / extra families / real data | off |
| Retune after test | forbidden |
| Sharing required to improve | no |

Paired transfer estimates are published with Monte Carlo uncertainty across
the nested seeds. Signs are not clipped. Harmful transfer is not a
correctness failure.

## Uncertainty language (fixed)

- **Predictive variability:** `HierarchicalCovariance` study, animal, and
  residual draws inside `sample_study`.
- **Parameter uncertainty:** spline study-bootstrap of the mean only.
  Transformer `n_members=1` has no member spread. Covariance parameters are
  not bootstrapped. A reliable Bayesian posterior is the only complete
  parameter+predictive statement.
- **Evaluation denominators:** n_studies, n_animals, n_obs, n_contrasts,
  n_group_mean_cells, n_gap_pairs, n_increments, n_extrema, and the stated
  coverage levels travel with every evaluation pack.

## Labels

- Synthetic diagnostic.
- Not design qualification.
- No real-data validation.
- No operational recommendation.

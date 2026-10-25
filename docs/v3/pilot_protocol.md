# Minimal pilot protocol

**Question.** Holding **target-species training studies fixed**, does adding
other-species studies help or harm target-species prediction?

**First execution surface:** synthetic generator A (already emits DOG).
Not biological validation. Not design qualification. No real data.

**Models.** Context-specific vs pooled on the **existing** hierarchical
spline (ridge-penalized B-spline time/exposure basis with sex, baseline,
strain, dose, compound, interval, route, and recovery interactions; pooled
adds the same time-basis interactions for `SPECIES_*`) and the custom
baseline-conditioned transformer (neural mean; pooled adds a species
context token when `SPECIES_*` columns exist). Do **not** call the spline
a linear-in-time model. Bayesian and TFT are `not_run` unless a later
amendment says otherwise. No new families.

**Selective sharing** is required in the eventual benchmark and is
**deferred** (not implemented in stages 1–2). `sharing: selective` must
raise.

**First scored task.** Predose-conditioned prediction (v2
`evaluate_prestudy`). Design-stage scoring and early-history are later
waves.

**Design engine.** Off. No `propose-design` in the pilot config.

## 1. Freeze the target-species experiment

1. Generate a synthetic library with `n_studies` RAT and
   `n_other_species_studies ≥ 3` DOG (today’s default is 1 DOG; the pilot
   config must raise this **without** changing the RAT seed stream for
   studies `0 … n_studies-1` — use a separate donor seed stream if the
   current loop would reshuffle RAT studies).
2. Apply the **v2 RAT cohort filter** to define the target population.
3. Freeze target-species splits **before** any donor is eligible for train:
   - `train_target` (fixed)
   - `validation_target` (fixed; used only for v2-style early stopping /
     alpha)
   - `test_target` (fixed complete-study holdout; the only scored set)
4. Persist hashes (`splits.json` style). Adding donors must not change
   these hashes.

If the current generator loop (`for s in range(n_studies + n_other_species)`)
cannot add DOG without changing later RAT draws, **do not** retune by
re-generating RAT. Amendment: generate RAT with the existing seed, then
generate donor studies with `SeedStream` namespace `"donor"`. That amendment
is implementation, not this protocol’s science.

## 2. Donor schedule

Add donor studies in a declared order (STUDYID sort). Levels:

| \(k\) | Train set |
|---|---|
| 0 | `train_target` only (context-specific; v2 equivalent) |
| 1 | `train_target` + first donor study |
| 2 | `train_target` + first two donor studies |
| \(K\) | `train_target` + all donor studies |

Validation stays `validation_target` (no donor leakage into early stopping).
Optional diagnostic: a `donor_internal` split that is never used to accept
a model.

## 3. Sharing regimes at each \(k\)

| Regime | \(k=0\) | \(k>0\) |
|---|---|---|
| Context-specific | Fit | Repeat the \(k=0\) artifact (do not refit). This is \(S_0\) |
| Pooled | `not_run` (no donors) | Fit with explicit species feature |
| Selective (v1) | `not_run` | **Required later; deferred.** Not implemented in stages 1–2. |

First locked synthetic wave: context-specific + pooled only. Selective
sharing stays required for the eventual benchmark and is not started here.

## 4. Score only `test_target`

**Wave 1 (this protocol’s first execution):** predose-conditioned only
(observed baselines; v2 `evaluate_prestudy`).

Later waves (same holdout observations; not started):

1. Design-stage (sampled baselines from **target-train** baseline
   distribution; no holdout predose)
2. Early-history `Wall14` (v2 window)

Sparse-history grid is a **later** wave. Do not block wave 1 on it.

Metrics (study-balanced): RMSE log, CRPS log, contrast bias at \(T=28\) if
estimable. Report \(n\) studies / animals / obs.

Harmful transfer: \(\Delta_k = S_k^{\mathrm{pooled}} - S_0\) on RMSE log and
on |contrast bias|. Publish the sign. Do not retune to remove harm.

## 5. Size (method pilot, not a locked claim)

- RAT: keep near smoke/research-small (e.g. 16–40 studies). Do not jump to
  “research.yaml 40 studies × full design” — design is off.
- DOG: at least 3 donor studies so \(k=1,2,3\) exists.
- Transformer: cap `max_epochs` / `n_members` at smoke-like values for the
  first compile check; a later locked wave may raise them.
- One seed for wave 1. A locked wave needs a second seed and a written
  “no flip / flip” rule — **after** wave 1 runs, not in this packet.

## 6. Labels (required on every artifact)

- Synthetic diagnostic.
- Not design qualification.
- No real-data validation.
- No operational recommendation.
- Donor DGM is illustrative (different `gmax` / baseline, not biology).

## 7. Stop rules

- If adding donors changes `train_target` membership or feature vocab in a
  way that remaps RAT strains, stop and fix the freeze.
- If pooled fit silently includes DOG without a species feature, fail the
  wave (that is not pooling; that is contamination).
- If anyone enables `propose-design` on the pilot config, the wave is
  invalid for v3 acceptance.
- Real extracts: do not start this protocol on real data until
  `historical_data_audit.md` Phase A+B pass.

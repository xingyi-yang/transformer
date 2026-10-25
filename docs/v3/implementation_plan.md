# Staged implementation plan

Preserve working v2 infrastructure. Implement only what the new goal needs,
in order. **Stages 1–2 software is in the repository.** The Stage 3 training wave has
not been launched. Selective sharing remains required later and is deferred.

Do not: overwrite v2, add optional model families, enable design
recommendations, launch large training runs, or treat tests as readiness.

## Stage 0 — planning (this packet)

**Done in-repo as documents only.**

- Capability matrix, gap analysis, proposed spec, audit plan, pilot protocol
- v2 spec untouched
- No runtime default changes

Acceptance: documents exist; shipped configs still `cohort.species: RAT`
with no `donor_species` key required.

## Stage 1 — freeze and include (infrastructure)

**Status: implemented (software).** Training wave not launched.

**Goal.** A donor study can enter **train** without changing the target
split hash, and can be left out (v2 behavior) with one config flag.

Work:

1. Split generation of target vs donor studies (separate `donor` seed namespace)
2. Cohort: `donor_species` + `sharing` (default empty / context_specific)
3. Split freeze: `train_target` / `validation_target` / `test_target` hashes independent of \(k\)
4. Tests: v2 exclusion reproduced; donor-in-train does not flip target hash;
   unseen species still refused; target records unchanged; donor isolation;
   stable donor generation

Out of scope: new models, design, real data, sparse-history grid.

Acceptance (software): unit tests green; no smoke default change; no
`propose-design` in any new pilot YAML.

## Stage 2 — pooled feature + scoring table

**Status: implemented (software).** Predose-conditioned is the first scored
task. Stage 3–4 software (selective v1, sparse grid, donor heterogeneity,
distributional / contrast-error, uncertainty inspection) is in the
evaluation benchmark. The frozen k=0/k=1 artifacts are unchanged.

**Goal.** Pooled spline and pooled transformer can fit with an explicit
species token/column; a sign-defined paired transfer table can be built
from \(S_0\) vs \(S_k\) on `test_target`.

Work:

1. `SPECIES_*` in `FeatureSpec` only when `sharing: pooled` and more than one
   species is in the training animals
2. Transformer: species context token only when `SPECIES_*` columns exist;
   default transformer remains v2 (no species token)
3. Evaluation: predose-conditioned (`evaluate_prestudy`) is the first pilot
   task; design-stage / `Wall14` are later
4. Harmful-transfer rows (`evaluation/transfer.py`) with published sign

Acceptance: unit tests for species features, serialization, zero-donor
equivalence, training-only preprocessing, and the transfer table. Do **not**
run the training wave here.

**Do not** run research-scale or locked B=2000 here.

## Stage 3 — synthetic pilot wave 1

The approved k=0 then k=1 predose wave is recorded under `runs/v3_pilot_k0k1/`.
Do not retune it. Harmful transformer transfer is a published result.

## Stage 4 — sparse-history + selective v1

Implemented in `src/send_bw/sharing.py`, `evaluation/sparse.py`, and
`v3_eval.py`. Frozen evaluation config: `configs/v3_eval.frozen.yaml`.

- Encoder grid W0 / W1 / Wlast14 / Wall14 (no encoder filling)
- Selective v1 (shared compound + regimen; species-specific growth/residual)
- Controlled donor scenarios: matched / growth_shift / treatment_shift / both
- Bayesian reliability gate: unreliable fits are not a successful benchmark

Acceptance: same holdout; no encoder filling; selective history names the
shared groups; design off.

## Stage 5 — historical audit, then stop or proceed

Execute `historical_data_audit.md` Phase A on paper. Phase B only with
authorization. **Unknown remains unknown.**

If the library is single-species or unauthorized, real v3 work is
`not_supported`. Synthetic stages still stand as method checks.

## Stage 6 — real-data sharing (not scheduled here)

Requires qualification stages 2–3, then a **new** protocol amendment. Not
implied by Stages 1–4. No design recommendations.

## What stays frozen from v2

| Asset | Action |
|---|---|
| `CURSOR_PROJECT_SPEC_v2.md` | Do not overwrite |
| Analysis close-out / ANCOVA default | Do not reopen |
| C1 / GLS locked streams | Do not retune |
| `available_models()` required-only | Keep |
| TFT optional / Chronos / DeepAR | Do not expand |
| `propose-design` | Do not call from v3 pilot configs |
| Smoke `not_run` optional extras | Keep default env free of `[tft]` unless a separate TFT ticket |

## Acceptance criteria (global)

A stage is accepted when:

1. Target-holdout hashes are stable
2. Sharing regime is named on every artifact
3. Harmful transfer can appear (no silent clip)
4. Synthetic / not-qualification / no-real-data / no-recommendation labels
   are present
5. v2 one-species smoke still passes without the new keys

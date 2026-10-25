# Synthetic objective validation — verdict

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

This is a **performance** verdict on fresh frozen synthetic datasets, not a
test-count or implementation-status list. Functionality scores start from
`docs/v3/objective_assessment.md` and are updated only where this run added a
capability. Sharing and the transformer are not required to win. Prior
checksummed run directories were not modified. No real data were accessed.

Freeze: generate seed 41001, train seed 61001, config hash `1e3b3a47eea0d107`,
revision `1241ab6`. Development changes used generate 31001 / train 51001 only.
Frozen test outcomes were not used to retune.

## Combined objective table

| ID | Objective | Functionality | Performance | Evidence |
|---|---|---|---|---|
| O1 | Target-species predose log-weight prediction | achieved | **partial** | 54/54 cells on 3 frozen datasets. Scores exist. Still synthetic / exploratory; no accuracy bar was pre-declared as “achieved.” |
| O2 | Context-specific vs pooled vs selective on a fixed holdout | achieved | **partial** | Paired holdouts reused across sharing. Spline RMSE Δ ≈ 0. Transformer mixed and small (about −0.006 to +0.009). Sharing is not required to win and did not win. |
| O3 | Detect and publish harmful transfer (Δ>0) | achieved | **achieved** | Signs were not clipped. Transformer sparse-donor and some pooled cells show harmful RMSE Δ; published as such. |
| O4 | Active−control contrast Δ_g(T) at T=28 | achieved | **partial** | Contrast packs scored on the frozen test. Not a design-feasibility claim. |
| O5 | Shared / partially shared / conflicting mechanisms | achieved | **partial** | `matched` / `growth_shift` / `both` all generated and scored (research had omitted `growth_shift`). No mechanism made sharing required or successful. |
| O6 | Full vs sparse target-study history | achieved | **partial** | Paired full vs `n_keep=4`. Spline sparse RMSE Δ ≈ +0.003 (slightly harmful). Transformer pooled sparse ~+0.01 harmful; context-specific sparse slightly helpful. Density is not required to win. |
| O7 | Sparse-history encoder grid (W0/W1/Wlast14/Wall14) | achieved | **not evaluated** | Not re-run. Prior `runs/v3_eval` stands. |
| O8 | Design-stage (no predose) holdout score | achieved (engine) | **not evaluated** | No paired design-stage vs predose table. |
| O9 | Early-history conditioning | partial | **not evaluated** | Not the primary frozen task. |
| O10 | Nested replication with Monte Carlo precision | achieved | **achieved** | Prior increment sd of dataset means 0.144; target SE ≤ 0.10 ⇒ n ≥ 2.07. Executed **n_datasets=3**, n_train=1. Observed SE of the primary increment mean = 0.034. |
| O11 | Temporal increment distribution | achieved | **partial** | Prior matched ratio **1.417**. After validation-only process scaling, spline primary slice **1.099** (sd_ds 0.058; n=3; SE 0.034) inside [0.80, 1.20] — distributional **gate passed**. Transformer increment remains **1.23–1.34**. Coverage / PIT / extrema were not re-qualified. |
| O12 | Reliable Bayesian inference | partial | **failed** | Thresholds unchanged (`rhat≤1.01`, `ESS≥400`, `divergences==0`). Development cell passed (rhat 1.006, ESS 635/763). Frozen eval cell **failed**: rhat 1.008, ESS bulk/tail **352/278**, divergences 0, class `covariance_mixing` (`sd_s[1]`). Not counted. |
| O13 | Designs vs independent generator B | achieved (engine) | **partial** | Increment gate passed, so the design workflow ran. Proposal: **0/4 feasible** (all failed FWER at B_proposal=80). No finalists, so generator B was not scored. `no_feasible_design` is a valid outcome. B<2000 is exploratory / nonconclusive. Not an operational recommendation. Sharing / transformer were not required to win and did not win. |
| O14 | Spline vs transformer; neither required to win | achieved | **partial** | Both trained on every cell. Transformer increment still high; some sharing/density RMSE Δ harmful. No family was added to invert signs. |
| O15 | Real SEND / historical validation | gated | **not evaluated** | No real data. Phase B remains ACCESS_BLOCKER. Correctly not opened. |

## What changed relative to the pre-evaluation assessment

| Item | Before (research / increment / limitations) | This frozen evaluation |
|---|---|---|
| O5 mechanisms | `matched` + `both` only | + `growth_shift` (partially shared) |
| O10 replication | n=2, not MC-justified | n=3 from SE≤0.10 plan; observed SE 0.034 |
| O11 increment | 1.417 (failed) | spline primary 1.099 (gate pass); transformer still ~1.26 |
| O12 Bayes | research `mean_identifiability`, rhat 1.015, ESS 115/142 | identifiability lever helped development (passed); frozen still fails ESS on `sd_s` |
| O13 design vs B | not evaluated | ran; no feasible proposal; B not reached |

Development increment moment (validation, not test): ratio_before 1.385, scale_factor 0.722, `used_test=false`.

## Increment slices (matched pred/obs ratio)

Primary slice — spline, matched / shared, context-specific, full:

- dataset means 1.144, 1.118, 1.033
- mean of dataset means **1.099** (sd_ds 0.058; n=3)

Spline ratios after the validation-only scale sit in about 1.02–1.10 across sharing, banks, and density. Transformer ratios stay above 1.20 (about 1.23–1.34). Target-species increment is largely insensitive to donor mechanism, as expected for a residual-process check.

## Bayesian (unchanged gate)

| cell | rhat | ESS bulk / tail | divergences | counted |
|---|---|---|---|---|
| development (gen 31001) | 1.006 | 635 / 763 | 0 | yes (development only) |
| frozen d0 matched context-specific full | 1.008 | 352 / 278 | 0 | **no** |

The frozen failure is `covariance_mixing` on `sd_s[1]`, not the old strain×time `mean_identifiability` class. Thresholds were not relaxed.

## Design vs generator B

Request supported. Spline proposal at B=80: four ANCOVA candidates, all `pass_power` / `pass_coverage` / `pass_bias` true, all `pass_fwer` false, **0 feasible**. Workflow stopped before independent-generator-B finalist evaluation. Status `no_feasible_design`. Not an operational recommendation.

## Sharing and density (not required to improve)

Spline sharing RMSE Δ is ~0 on this grid (unlike the earlier research full-train spline Δ ≈ +0.011). That is a different freeze and a smaller study count; it is not a retuned reversal and is not treated as sharing success.

Transformer pooling is slightly helpful on full history and harmful on sparse target training. Sparse target-study training is slightly harmful for spline.

Negative or near-zero transfer remains a valid published result.

## What this verdict does not claim

- Unit tests are not scientific readiness.
- Sharing and the transformer are not required to win and did not win.
- The increment gate is not a transformer or coverage qualification.
- Development Bayes success is not frozen-eval success.
- `no_feasible_design` at B=80 is not a locked design finding.
- No real extract was opened. No operational design is recommended.
- Prior artifacts in `runs/v3_eval`, `v3_pilot_k0k1`, `v3_repl`, `v3_research`, `v3_closeout`, `v3_audit`, `v3_increment`, `v3_phaseb`, and `v3_limitations` were not overwritten.

Artifacts: `runs/v3_objective/SUMMARY.json`, `runs/v3_objective/VERDICT.md`, `runs/v3_objective/objective_labels.json`.

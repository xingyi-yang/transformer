# Objective assessment (existing artifacts)

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

This note scores the **original v3 (and inherited v2) objectives** against
artifacts already in the tree (`runs/v3_pilot_k0k1`, `v3_eval`, `v3_repl`,
`v3_research`, `v3_closeout`, `v3_increment`, `v3_limitations`). It does not
use real SEND data. Unit tests are not treated as scientific readiness.

Two columns are kept separate:

- **Functionality** — is the method implemented and executable?
- **Performance** — did executed synthetic checks demonstrate the intended
  scientific behavior?

Verdicts: `achieved` | `partial` | `failed` | `not evaluated`.

| ID | Objective | Functionality | Performance | Evidence |
|---|---|---|---|---|
| O1 | Target-species predose log-weight prediction (study-balanced RMSE / CRPS) | achieved | partial | Research 64 cells, 2 datasets. Scores exist; n=2 is exploratory, not MC-justified |
| O2 | Context-specific vs pooled vs selective sharing on a **fixed** target holdout | achieved | partial | All three regimes ran. Spline full-train sharing RMSE harmful (~+0.011 mean of dataset means). Sharing is not required to win |
| O3 | Detect and publish harmful transfer (Δ = S_k − S_0; RMSE/CRPS Δ>0 harmful) | achieved | achieved | Negative transfer published in pilot/eval/research/closeout. Signs were not clipped |
| O4 | Active−control contrast Δ_g(T) at T=28 (bias, study-balanced beside row-weighted) | achieved | partial | Contrast packs and closeout table exist. Magnitudes differ by weighting; not a design-feasibility claim |
| O5 | Shared / partially shared / conflicting donor **mechanisms** | partial | partial | Research: `matched` and `both` only. `growth_shift` / `treatment_shift` exist in generator and v3_eval, not in the nested research crossing |
| O6 | Full vs sparse **target-study training** (not encoder sparsity) on paired holdouts | achieved | partial | Primary research contrast executed. Interaction often negative for spline; transformer sparse donor-using RMSE harmful |
| O7 | Sparse-history **encoder** grid (W0/W1/Wlast14/Wall14) | achieved | partial | Implemented and run in `runs/v3_eval`. Not re-scored in the research/closeout increment work |
| O8 | Design-stage prediction (baseline **distribution**, no animal predose) as a holdout score | implemented as design engine | not evaluated | `sample_study` + ADEMP exist. No paired design-stage vs predose holdout table on v3 cells |
| O9 | Early-history conditioning (DAY ≤ window) | partial | partial | Gaussian conditioning implemented; Bayesian early-history `not_run`. Not the research primary task |
| O10 | Nested replication with Monte Carlo precision justification | partial | failed | 2 datasets × 2 train seeds reported. Closeout labeled n=2 exploratory. SE was not used to choose n |
| O11 | Distributional / temporal checks (coverage, increment SD, PIT, extrema) | achieved | failed | 128 packs: mean cov80 0.863, cov95 0.957 (overcover). Matched increment ratio **1.417** (sd_ds 0.144). Limitation recorded, not test-calibrated |
| O12 | Reliable Bayesian inference (rhat≤1.01, ESS≥400, divergences==0) | partial | failed | Two research cells `mean_identifiability` (R-hat 1.015, ESS 115/142). Smoke at 2000/2000 was reliable; research-scale was not counted |
| O13 | Model-informed designs vs an **independent** truth generator (B) | achieved (software) | not evaluated | Workflow exists; v3 configs keep design off. Smoke design was nonconclusive (B<2000) |
| O14 | Nonlinear mean comparison (spline vs custom transformer); neither required to win | achieved | partial | Both trained. Transformer sparse donor RMSE harmful; spline sharing often harmful. No family added to invert signs |
| O15 | Real SEND / historical-data validation | gated | not evaluated | No real data. Phase B ACCESS_BLOCKER. Correctly not opened |

## Reading the scores

- **O3 achieved (performance)** means the *detection method* worked: harmful
  transfer appeared and was published. It does not mean sharing helped.
- **O11/O12 failed (performance)** are the unresolved modeling limitations.
  Functionality for both exists; the demonstrated layer does not yet meet
  the distributional or reliability bars.
- **O8/O13 not evaluated** are v2 design-engine capabilities that v3 did not
  accept as success, and that were not scored against independent truth on
  the research freeze.
- **O10 failed (performance)** because n=2 was a convenience crossing, not a
  precision calculation. Functionality for nested seeds exists.

## What this assessment does not do

- It does not retune on the exposed research test set.
- It does not open a real extract.
- It does not treat passing unit tests as objective achievement.

The focused synthetic evaluation that follows uses **development**
simulations for any model change and **fresh frozen** datasets for scoring.
Sharing and the transformer are not required to win. Executed performance
labels are in `docs/v3/objective_verdict.md`.

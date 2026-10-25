# v3 planning packet (proposed)

This directory, plus `CURSOR_PROJECT_SPEC_v3.md` at the repository root, is a
**planning packet**. It does not change v2 runtime defaults, does not overwrite
`CURSOR_PROJECT_SPEC_v2.md`, does not install optional model families, and does
not enable design recommendations.

**New study goal (this request, 2026-10-24):** make **nonlinear multi-species
information sharing** the primary scientific objective.

**v2 goal (repository spec):** learn one-species body-weight distributions and
evaluate whether model-informed allocations and schedules meet power / FWER /
precision. Additional species were labeled exploratory and are excluded at
cohort selection.

| Document | Contents |
|---|---|
| [capability_matrix.md](capability_matrix.md) | Implemented / partial / missing / unverified, with code paths |
| [gap_analysis.md](gap_analysis.md) | What must change for the new objective |
| [../CURSOR_PROJECT_SPEC_v3.md](../../CURSOR_PROJECT_SPEC_v3.md) | Proposed v3 specification |
| [historical_data_audit.md](historical_data_audit.md) | Authorization-first audit plan; unknowns marked |
| [pilot_protocol.md](pilot_protocol.md) | Fixed target-species train set; add other-species studies |
| [implementation_plan.md](implementation_plan.md) | Staged work that keeps v2 infrastructure |

**Evidence vs proposed.** Rows and claims that cite `src/`, `tests/`,
`configs/`, `runs/`, or `docs/` (other than this packet) are **repository
evidence**. Everything in the v3 spec and the later stages of the
implementation plan is **proposed functionality** unless an evidence citation
is attached.

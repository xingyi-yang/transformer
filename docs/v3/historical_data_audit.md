# Historical-data support audit plan

**Authorization first.** This plan does not connect to Snowflake, open real
SEND extracts, or request credentials. Fields that cannot be read from this
repository are **unknown**.

Purpose: decide whether historical multi-species SEND *could* support the
v3 question, before anyone is asked to share records.

## 0. Current knowledge (repository evidence)

| Item | Status |
|---|---|
| Real SEND used in this repo | No. HANDOFF and `docs/real_data_qualification.md`: stage **1** only |
| Snowflake | Adapter implemented; `authorized: false`; mock-tested; extra not required in default env |
| Local real template | `configs/real_local.example.yaml` — paths on `D:/approved_storage/...` (placeholder) |
| Mapping | Illustrative `configs/mappings/example_send.yaml` (`RAT`, `DOG` value map only) |
| define.xml | Not implemented |
| Curated regimens / compounds for real studies | Unknown (templates only) |
| Live warehouse inventory | **Unknown** |
| Written authorization | **Unknown** (none in repo) |
| Species mix, study counts, SEND versions | **Unknown** |
| Overlap of compounds / routes across species | **Unknown** |

Do not infer a real multi-species library from the synthetic DOG study.

## 1. Phase A — paper inventory (no system access)

Complete before any `authorized: true` or local extract copy.

1. Named data steward, legal owner, and purpose statement (sharing research,
   not design recommendations).
2. Whether a **read-only** inventory view exists (study-level metadata only:
   STUDYID, SPECIES, STRAIN, study class, start date, SEND version, domain
   list). If only row-level BW is available, stop and redesign the ask.
3. Approved storage for any later extract (`output_dir`). `runs/` must stay
   git-ignored.
4. Whether donor species are even present. If the library is RAT-only, v3
   real-data work is `not_supported` (synthetic pilot can still run).
5. Known mapping debt: TX/ARM convention, planned regimen source, compound
   identity (v2: do not split EXTRT).

Deliverable: a one-page **unknown / available / refused** table. No records.

## 2. Phase B — authorized metadata-only inspect

Requires written authorization that **explicitly** allows metadata inspect
and forbids model training on the same ticket if training is not approved.

Allowed:

- `send-bw inspect-source` on an **allowlisted metadata** extract, or a
  steward-produced study-level table
- Counts by SPECIES, STRAIN, STUDY_CLASS, SEND version, year
- Domain presence (DM/BW/EX/DS), not values

Forbidden without a separate authorization:

- Reading BWSTRESN / individual weights
- Fitting any model
- Enabling Snowflake `authorized: true` beyond the inspect ticket
- Uploading results to non-approved storage

If Snowflake is the source, keep `authorized: false` until the ticket is
signed. Then set it only for the inspect config, with column allowlists that
**omit** body-weight results if the ticket is metadata-only.

## 3. Support criteria (pass / fail / unknown)

A library **supports a synthetic-to-real follow-on** only if all pass or are
explicitly waived:

| Criterion | Pass rule | If unknown |
|---|---|---|
| ≥1 donor species with ≥ \(n_{\min}\) complete studies | Prespecify \(n_{\min}\) (pilot: 3) | Do not claim support |
| Target species has a frozen complete-study holdout of ≥ 2 studies | Required | Stop |
| Planned regimen curated (not EX-as-plan) | v2 rule | Stage 3 blocked |
| Compound identity curated | v2 rule | Stop pooling on compound |
| Compatible study class | Same class or a declared exception | Exclude |
| Horizon overlap with target \(T\) | Donor studies observe near \(T\) | Mark not_supported for contrast scores |
| Sex both present or single-sex estimand declared | Explicit | Stop |
| Baseline / predose rule reviewable | Per-study | Stage 2 incomplete |

Harmful-transfer analysis on real data additionally needs the **same**
target train set to remain identifiable after donors are added (stable
STUDYID list).

## 4. Phase C — record-level pilot (only after A+B pass)

This is v2 qualification stage 2, amended:

- Few studies, **target + one donor species**
- Reconcile counts and ≥20 weights/study against source
- Confirm predose, no post-death weights, recovery/satellite flags
- Do **not** run design recommendations
- Do **not** claim sharing benefit from this reconciliation

Stage 3 (scientist-reviewed species/strain/arm/regimen map) is required
before any pooled or selective **fit** on real data.

## 5. What this audit will not do

- Connect to Snowflake from this planning packet
- Browse real files if a path happens to exist on a developer machine
- Treat synthetic DOG parameters as historical support
- Treat stage 1 mock tests as inventory

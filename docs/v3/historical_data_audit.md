# Historical-data audit (ready to execute)

**Authorization first.** Phase A is paper inventory from this repository. It
does not connect to Snowflake, open real SEND extracts, or request credentials.
Phases B and C refuse unless a signed authorization file names the phase.

This is the validation workflow that follows the synthetic closeout. It is not
a redesign of the benchmark and does not issue design recommendations.

> **No real-data validation until authorized. No operational recommendation.**

## Commands

Phase A (safe; run now):

```text
python3 -m send_bw.cli historical-audit --config configs/v3_historical_audit.yaml --phase A
```

Phase B (blocked until a signed ticket exists). The runner is ready; it does
not create, sign, or self-approve a ticket. Without a genuine ticket it writes
`ACCESS_BLOCKER` and stops before opening any source.

```text
python3 -m send_bw.cli historical-audit --config configs/v3_historical_audit.yaml --phase B \
  --authorization /approved/tickets/metadata_inspect.yaml
```

Discovery looks at `--authorization`, `SEND_BW_AUTHORIZATION`, and
`/approved/tickets/*.yaml` only. It does not scan the repository for
documentation text. The in-repo example is unsigned and is rejected.

Once a steward-signed metadata-only ticket names `source.root`, Phase B
returns biological-support inventory, missingness, provenance, and feasible
candidate validation contexts. It does not inspect BWSTRESN/BWORRES, train
models, or offer design recommendations.

Copy `configs/historical_authorization.example.yaml` **outside the repo**,
complete it, and set `written_authorization: true` only after a steward signs.

## Gate

| Phase | What runs | Required ticket fields | Never does |
|---|---|---|---|
| A | Repo inventory | none | Open extract roots; Snowflake; BW columns |
| B | Metadata inspect only | `written_authorization`, steward, purpose, `allowed_phases: [B]`, `allows_metadata_only`, `forbids_design_recommendations` | BWSTRESN; training; design; `authorized: true` without the ticket |
| C | Record-level reconciliation | Phase B plus `allows_body_weight`; **not** `allows_model_training` | Fitting; pooling; design recommendations |

Snowflake stays `authorized: false` until a ticket that is metadata-only (B)
or reconciliation-only (C) is signed. Credentials remain `env:` references.

## Support criteria (unchanged)

A library supports a synthetic-to-real follow-on only if these pass or are
explicitly waived. Unknown is not a pass.

| Criterion | Pass rule |
|---|---|
| ≥1 donor species with ≥ 3 complete studies | Prespecified n_min |
| Target species frozen holdout ≥ 2 studies | Required |
| Planned regimen curated (not EX-as-plan) | v2 rule |
| Compound identity curated | v2 rule |
| Compatible study class | Same class or declared exception |
| Horizon overlap with T=28 | Required for contrast scores |
| Sex both present or single-sex estimand declared | Explicit |
| Baseline / predose rule reviewable | Per-study |

Harmful-transfer analysis additionally needs the same target train STUDYID list
after donors are added.

## What this workflow will not do

- Access real data without an explicit authorization file
- Treat synthetic DOG parameters as historical support
- Treat unit tests or stage-1 mocks as inventory
- Enable design recommendations
- Commit real extracts under `runs/`

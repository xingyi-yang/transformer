# Historical-data audit — Phase A (executed)

> **No real-data access. Not design qualification. No operational recommendation.**

- git revision: `5f5a669e81b500a7070d8996d547c36f1432778f`
- real_data_accessed: False
- support_for_synthetic_to_real: `not_claimed`

| item | status | evidence |
|---|---|---|
| real_SEND_in_repo | refused | docs/real_data_qualification.md stage 1 only; no real records committed |
| written_authorization | unknown | none in repository; required before phase B |
| named_steward | unknown | not present |
| snowflake_adapter | available | src/send_bw/data/sources.py; authorized defaults false |
| snowflake_authorized_flag | refused | configs/real_snowflake.example.yaml authorized=False |
| local_real_template | available | configs/real_local.example.yaml placeholder D:/approved_storage paths; not opened |
| local_real_extract_present | unknown | phase A does not stat placeholder extract roots |
| define.xml | unknown | not implemented |
| mapping_profile | available | configs/mappings/example_send.yaml |
| curated_arm_regimen_real | unknown | templates only |
| curated_compound_reference_real | unknown | templates only |
| donor_species_in_library | unknown | do not infer from synthetic DOG |
| target_holdout_of_two_studies | unknown | requires phase B metadata |
| horizon_overlap_day_28 | unknown | requires phase B metadata |
| qualification_doc | available | docs/real_data_qualification.md |

Phase B needs a signed authorization file naming steward, purpose, allows_metadata_only: true, allowed_phases: [B], and forbids_design_recommendations. Do not set Snowflake authorized: true on the same ticket as training.

Phase B/C commands stay gated:

```text
send-bw historical-audit --config configs/v3_historical_audit.yaml --phase B \
  --authorization /approved/tickets/metadata_inspect.yaml
```

The authorization file must set `written_authorization: true`, name the steward, list `allowed_phases`, and keep `forbids_design_recommendations: true`. Metadata-only tickets must omit BW result columns.


"""Ready-to-execute historical-data audit with a hard authorization gate.

Phase A runs from repository evidence only. Phases B and C refuse unless a
signed authorization file names the phase and forbids training/design when
those are not granted. Real extracts, Snowflake, and body-weight columns are
not opened without that ticket. This is a validation workflow, not a redesign.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from send_bw.sharing import LABELS
from send_bw.utils.artifacts import git_revision, write_json

REPO = Path(__file__).resolve().parents[2]
BLOCKED_ROOT_MARKERS = ("approved_storage", "send_extracts", "real_extract")
BW_COLUMNS = {"BWSTRESN", "BWORRES", "BWSTRESU", "BWORRESU"}


class AuditGateError(PermissionError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _looks_like_real_root(path: str | Path | None) -> bool:
    if path is None:
        return False
    s = str(path).replace("\\", "/").lower()
    return any(m in s for m in BLOCKED_ROOT_MARKERS) or s.startswith("d:/") or s.startswith("d:\\")


def validate_authorization(auth: dict[str, Any] | None, phase: str) -> dict[str, Any]:
    if not auth or not auth.get("written_authorization"):
        raise AuditGateError(
            f"phase {phase} requires a signed authorization file with written_authorization: true"
        )
    allowed = {str(p).upper() for p in (auth.get("allowed_phases") or [])}
    if phase.upper() not in allowed:
        raise AuditGateError(f"authorization does not allow phase {phase}; allowed={sorted(allowed)}")
    if phase.upper() == "B" and not auth.get("allows_metadata_only", False):
        raise AuditGateError("phase B requires allows_metadata_only: true")
    if phase.upper() == "C":
        if not auth.get("allows_body_weight", False):
            raise AuditGateError("phase C requires allows_body_weight: true")
        if auth.get("allows_model_training"):
            raise AuditGateError("phase C reconciliation must not enable model training on the same ticket")
    if auth.get("allows_model_training") and phase.upper() != "C":
        raise AuditGateError("model training is not part of the historical-data audit")
    if not auth.get("forbids_design_recommendations", True):
        raise AuditGateError("design recommendations must remain forbidden")
    if not auth.get("steward_name") or not auth.get("purpose"):
        raise AuditGateError("authorization must name steward_name and purpose")
    return auth


def refuse_real_access(cfg: dict[str, Any], auth: dict[str, Any] | None, phase: str) -> None:
    src = cfg.get("source") or {}
    snow = src.get("snowflake") or {}
    if src.get("type") == "snowflake" and not snow.get("authorized", False):
        raise AuditGateError("Snowflake authorized: false; refusing connection")
    if src.get("type") == "snowflake" and phase.upper() == "A":
        raise AuditGateError("phase A must not connect to Snowflake")
    if phase.upper() == "A" and _looks_like_real_root(src.get("root")):
        raise AuditGateError("phase A refuses to open a real extract root")
    cols = set()
    for v in (snow.get("columns") or {}).values():
        cols.update(str(c).upper() for c in v)
    if phase.upper() == "B" and (cols & BW_COLUMNS) and not (auth or {}).get("allows_body_weight"):
        raise AuditGateError("phase B authorization is metadata-only; BW result columns must be omitted")
    if phase.upper() in {"B", "C"} and src.get("type") == "snowflake" and snow.get("authorized"):
        if not auth or not auth.get("written_authorization"):
            raise AuditGateError("cannot enable Snowflake authorized: true without a signed ticket")


def phase_a_inventory(cfg: dict[str, Any]) -> dict[str, Any]:
    """Paper inventory from repository files. Does not open extract roots."""
    refuse_real_access(cfg, None, "A")
    real_local = load_yaml(REPO / "configs" / "real_local.example.yaml") if (REPO / "configs" / "real_local.example.yaml").exists() else {}
    real_sf = load_yaml(REPO / "configs" / "real_snowflake.example.yaml") if (REPO / "configs" / "real_snowflake.example.yaml").exists() else {}
    qual = REPO / "docs" / "real_data_qualification.md"
    mapping = REPO / "configs" / "mappings" / "example_send.yaml"
    rows = [
        {"item": "real_SEND_in_repo", "status": "refused",
         "evidence": "docs/real_data_qualification.md stage 1 only; no real records committed"},
        {"item": "written_authorization", "status": "unknown",
         "evidence": "none in repository; required before phase B"},
        {"item": "named_steward", "status": "unknown", "evidence": "not present"},
        {"item": "snowflake_adapter", "status": "available",
         "evidence": "src/send_bw/data/sources.py; authorized defaults false"},
        {"item": "snowflake_authorized_flag", "status": "refused",
         "evidence": f"configs/real_snowflake.example.yaml authorized="
                     f"{(real_sf.get('source') or {}).get('snowflake', {}).get('authorized')}"},
        {"item": "local_real_template", "status": "available",
         "evidence": "configs/real_local.example.yaml placeholder D:/approved_storage paths; not opened"},
        {"item": "local_real_extract_present", "status": "unknown",
         "evidence": "phase A does not stat placeholder extract roots"},
        {"item": "define.xml", "status": "unknown", "evidence": "not implemented"},
        {"item": "mapping_profile", "status": "available",
         "evidence": str(mapping.relative_to(REPO)) if mapping.exists() else "missing; illustrative RAT/DOG only"},
        {"item": "curated_arm_regimen_real", "status": "unknown", "evidence": "templates only"},
        {"item": "curated_compound_reference_real", "status": "unknown", "evidence": "templates only"},
        {"item": "donor_species_in_library", "status": "unknown",
         "evidence": "do not infer from synthetic DOG"},
        {"item": "target_holdout_of_two_studies", "status": "unknown", "evidence": "requires phase B metadata"},
        {"item": "horizon_overlap_day_28", "status": "unknown", "evidence": "requires phase B metadata"},
        {"item": "qualification_doc", "status": "available" if qual.exists() else "unknown",
         "evidence": "docs/real_data_qualification.md"},
    ]
    n = {s: sum(1 for r in rows if r["status"] == s) for s in ("available", "unknown", "refused")}
    support = "unknown"
    if n["unknown"] or n["refused"]:
        support = "not_claimed"
    return {
        "phase": "A",
        "labels": list(LABELS),
        "git_revision": git_revision(),
        "real_data_accessed": False,
        "snowflake_connected": False,
        "extract_root_opened": False,
        "counts": n,
        "inventory": rows,
        "support_for_synthetic_to_real": support,
        "next": (
            "Phase B needs a signed authorization file naming steward, purpose, "
            "allows_metadata_only: true, allowed_phases: [B], and forbids_design_recommendations. "
            "Do not set Snowflake authorized: true on the same ticket as training."
        ),
        "local_template_output_dir": (real_local.get("output_dir") if real_local else None),
        "note": "Placeholder paths were not opened. Synthetic DOG is not historical support.",
    }


def execute_historical_audit(config_path: str | Path, phase: str = "A",
                             authorization_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    phase = str(phase).upper()
    if (cfg.get("design") or {}).get("request"):
        raise AuditGateError("design operations are off for the historical-data audit")
    if cfg.get("synthetic", {}).get("enabled"):
        raise AuditGateError("historical-data audit is not a synthetic generate run")
    out_dir = Path(cfg.get("output_dir") or "runs/v3_audit")
    if phase != "A" and _looks_like_real_root(out_dir) is False:
        # real outputs must not be committed; still allow a repo draft dir only for gated refusals
        pass
    out_dir.mkdir(parents=True, exist_ok=True)
    auth = load_yaml(authorization_path) if authorization_path else None
    if phase == "A":
        refuse_real_access(cfg, None, "A")
        result = phase_a_inventory(cfg)
        write_json(out_dir / "phase_a_inventory.json", result)
        _write_phase_a_md(out_dir, result)
        if out_dir.name == "v3_audit":
            Path("docs/v3/historical_audit_phase_a.md").write_text(
                (out_dir / "PHASE_A.md").read_text(encoding="utf-8"), encoding="utf-8")
        return result
    validate_authorization(auth, phase)
    refuse_real_access(cfg, auth, phase)
    raise AuditGateError(
        f"phase {phase} is authorized in schema but this environment has no approved extract; "
        "refusing to open source.root or Snowflake. Re-run only after the steward places a "
        "metadata-only snapshot in approved storage and names it on the ticket."
    )


def _write_phase_a_md(parent: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Historical-data audit — Phase A (executed)",
        "",
        "> **No real-data access. Not design qualification. No operational recommendation.**",
        "",
        f"- git revision: `{result.get('git_revision')}`",
        f"- real_data_accessed: {result.get('real_data_accessed')}",
        f"- support_for_synthetic_to_real: `{result.get('support_for_synthetic_to_real')}`",
        "",
        "| item | status | evidence |",
        "|---|---|---|",
    ]
    for r in result.get("inventory") or []:
        lines.append(f"| {r['item']} | {r['status']} | {r['evidence']} |")
    lines += ["", result.get("next") or "", "",
              "Phase B/C commands stay gated:",
              "",
              "```text",
              "send-bw historical-audit --config configs/v3_historical_audit.yaml --phase B \\",
              "  --authorization /approved/tickets/metadata_inspect.yaml",
              "```",
              "",
              "The authorization file must set `written_authorization: true`, name the steward, "
              "list `allowed_phases`, and keep `forbids_design_recommendations: true`. "
              "Metadata-only tickets must omit BW result columns.",
              ""]
    (parent / "PHASE_A.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

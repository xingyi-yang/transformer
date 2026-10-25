"""Ready-to-execute historical-data audit with a hard authorization gate.

Phase A runs from repository evidence only. Phase B is metadata-only and
runs only after a genuine signed ticket is discovered. This module never
creates, signs, or self-approves authorization. Without a genuine ticket it
reports ACCESS_BLOCKER and stops before opening any source. Body-weight
endpoint columns are never read in Phase B. This is a validation workflow,
not a redesign.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from send_bw.sharing import LABELS
from send_bw.utils.artifacts import git_revision, write_json

REPO = Path(__file__).resolve().parents[2]
BLOCKED_ROOT_MARKERS = ("approved_storage", "send_extracts", "real_extract")
BW_COLUMNS = {"BWSTRESN", "BWORRES", "BWSTRESU", "BWORRESU"}
AUTH_ENV = "SEND_BW_AUTHORIZATION"
EXAMPLE_NAME = "historical_authorization.example.yaml"
APPROVED_TICKET_DIRS = (Path("/approved/tickets"),)
PHASE_B_DEFAULT_OUT = Path("runs/v3_phaseb")
PHASE_B_STUDY_FIELDS = (
    "STUDYID", "SPECIES", "STRAIN", "STUDY_CLASS", "STUDY_START", "SEND_VERSION", "DURATION_DAYS",
)
PHASE_B_FORBIDDEN_DOMAINS = {"BW", "SUPPBW"}
PHASE_B_TS_PARMS = {
    "SPECIES": "SPECIES",
    "STRAIN": "STRAIN",
    "SSTDTC": "STUDY_START",
    "STSTDTC": "STUDY_START",
    "SENDVER": "SEND_VERSION",
    "SSTYP": "STUDY_CLASS",
    "STYPE": "STUDY_CLASS",
}
FILE_EXTS = {".csv": "csv", ".parquet": "parquet", ".xpt": "xpt"}
DONOR_N_MIN = 3
TARGET_HOLDOUT_N_MIN = 2
HORIZON_DAY = 28.0


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
                             authorization_path: str | Path | None = None,
                             output_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    phase = str(phase).upper()
    if (cfg.get("design") or {}).get("request"):
        raise AuditGateError("design operations are off for the historical-data audit")
    if cfg.get("synthetic", {}).get("enabled"):
        raise AuditGateError("historical-data audit is not a synthetic generate run")
    if phase == "A":
        out_dir = Path(output_dir or cfg.get("output_dir") or "runs/v3_audit")
        out_dir.mkdir(parents=True, exist_ok=True)
        refuse_real_access(cfg, None, "A")
        result = phase_a_inventory(cfg)
        write_json(out_dir / "phase_a_inventory.json", result)
        _write_phase_a_md(out_dir, result)
        if out_dir.name == "v3_audit":
            Path("docs/v3/historical_audit_phase_a.md").write_text(
                (out_dir / "PHASE_A.md").read_text(encoding="utf-8"), encoding="utf-8")
        return result
    if phase == "C":
        discovery = discover_authorization(authorization_path)
        if not discovery["authorized"]:
            return write_access_blocker(output_dir or PHASE_B_DEFAULT_OUT, discovery, phase="C")
        validate_authorization(discovery["auth"], "C")
        refuse_real_access(cfg, discovery["auth"], "C")
        raise AuditGateError(
            "phase C is not executed here: this increment prepares Phase B only"
        )
    if phase != "B":
        raise AuditGateError(f"unknown audit phase {phase}")
    return execute_phase_b(cfg, authorization_path, out_dir=output_dir)


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


def _candidate_auth_paths(explicit: str | Path | None) -> list[Path]:
    out: list[Path] = []
    if explicit:
        out.append(Path(explicit))
    env = os.environ.get(AUTH_ENV)
    if env:
        out.append(Path(env))
    for d in APPROVED_TICKET_DIRS:
        if d.is_dir():
            out.extend(sorted(d.glob("*.yaml")))
            out.extend(sorted(d.glob("*.yml")))
        elif d.is_file():
            out.append(d)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _ticket_rejection(path: Path, data: dict[str, Any] | None) -> str | None:
    if path.name == EXAMPLE_NAME:
        return "in-repo example is unsigned and is not a steward ticket"
    if data is None:
        return "file could not be read"
    if not data.get("written_authorization"):
        return "written_authorization is not true"
    if not data.get("steward_name") or not data.get("purpose"):
        return "steward_name and purpose are required"
    if not data.get("allows_metadata_only"):
        return "allows_metadata_only is not true"
    if not data.get("forbids_design_recommendations", True):
        return "design recommendations must remain forbidden"
    if data.get("allows_model_training"):
        return "model training is not part of the historical-data audit"
    allowed = {str(p).upper() for p in (data.get("allowed_phases") or [])}
    if "B" not in allowed:
        return "allowed_phases does not include B"
    return None


def discover_authorization(explicit_path: str | Path | None = None) -> dict[str, Any]:
    """Find a genuine signed ticket. Never creates, signs, or self-approves one."""
    checked: list[dict[str, Any]] = []
    genuine: dict[str, Any] | None = None
    for p in _candidate_auth_paths(explicit_path):
        rec: dict[str, Any] = {"path": str(p), "exists": p.exists(), "genuine": False, "reason": None}
        if not p.exists():
            rec["reason"] = "path does not exist"
            checked.append(rec)
            continue
        try:
            data = load_yaml(p)
        except (OSError, yaml.YAMLError) as e:
            rec["reason"] = f"unreadable: {e}"
            checked.append(rec)
            continue
        reason = _ticket_rejection(p, data)
        if reason:
            rec["reason"] = reason
            checked.append(rec)
            continue
        rec["genuine"] = True
        rec["reason"] = "schema-valid signed ticket"
        checked.append(rec)
        genuine = {"path": str(p), "auth": data}
        break
    return {
        "authorized": genuine is not None,
        "ticket_path": None if genuine is None else genuine["path"],
        "auth": None if genuine is None else genuine["auth"],
        "candidates_checked": checked,
        "created_ticket": False,
        "self_approved": False,
        "unsigned_example": str(REPO / "configs" / EXAMPLE_NAME),
        "search": {
            "explicit": str(explicit_path) if explicit_path else None,
            "env": AUTH_ENV,
            "approved_dirs": [str(p) for p in APPROVED_TICKET_DIRS],
            "repo_scan": False,
        },
        "note": (
            "No ticket was created, signed, or self-approved. "
            "The in-repo example remains unsigned. Documentation text that "
            "mentions written_authorization: true is not a steward ticket."
        ),
    }


def access_blocker_report(discovery: dict[str, Any], phase: str = "B") -> dict[str, Any]:
    disc = {k: v for k, v in discovery.items() if k != "auth"}
    return {
        "status": "ACCESS_BLOCKER",
        "phase": phase,
        "labels": list(LABELS),
        "git_revision": git_revision(),
        "real_data_accessed": False,
        "source_opened": False,
        "endpoint_values_inspected": False,
        "models_trained": False,
        "design_recommendations": False,
        "created_ticket": False,
        "self_approved": False,
        "discovery": disc,
        "blocker": (
            "No genuine signed authorization was found. Phase B stops before "
            "opening any source. A steward must place a signed ticket (not the "
            "in-repo example) on SEND_BW_AUTHORIZATION or --authorization."
        ),
        "next": (
            "Copy configs/historical_authorization.example.yaml outside the repo, "
            "complete it, and set written_authorization: true only after a steward "
            "signs. This agent will not create or sign that ticket."
        ),
    }


def write_access_blocker(out_dir: str | Path, discovery: dict[str, Any],
                         phase: str = "B") -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = access_blocker_report(discovery, phase=phase)
    write_json(out_dir / "ACCESS_BLOCKER.json", result)
    write_json(out_dir / "authorization_discovery.json", result["discovery"])
    _write_phase_b_md(out_dir, result)
    if out_dir.name == "v3_phaseb":
        Path("docs/v3/phase_b_access.md").write_text(
            (out_dir / "PHASE_B.md").read_text(encoding="utf-8"), encoding="utf-8")
    return result


def _assert_no_bw_columns(columns: list[Any], where: str) -> None:
    cols = {str(c).upper() for c in columns}
    bad = cols & BW_COLUMNS
    if bad:
        raise AuditGateError(f"{where}: BW result columns {sorted(bad)} are forbidden in phase B")


def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext not in FILE_EXTS:
        raise AuditGateError(f"unsupported metadata table {path}")
    fmt = FILE_EXTS[ext]
    if fmt == "csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        raise AuditGateError(f"XPT is not opened in phase B metadata inspect: {path}")
    df.columns = [str(c).strip().upper() for c in df.columns]
    _assert_no_bw_columns(list(df.columns), str(path))
    return df


def list_domains_by_filename(root: Path, layout: str = "per_study") -> dict[str, list[str]]:
    """Filename stems only. Does not open file contents."""
    domains: dict[str, list[str]] = {}
    if not root.exists():
        return domains
    folders = [p for p in sorted(root.iterdir()) if p.is_dir()] if layout == "per_study" else [root]
    for folder in folders:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in FILE_EXTS:
                dom = f.stem.strip().upper()
                rel = str(f.relative_to(root)).replace("\\", "/")
                domains.setdefault(dom, []).append(rel)
    return domains


def _empty_study_row(studyid: str) -> dict[str, Any]:
    row = {f: None for f in PHASE_B_STUDY_FIELDS}
    row["STUDYID"] = studyid
    return row


def _merge_study_metadata(studies: dict[str, dict[str, Any]], frame: pd.DataFrame,
                          source: str, files_read: list[str]) -> None:
    if frame.empty:
        return
    _assert_no_bw_columns(list(frame.columns), source)
    keep = [c for c in PHASE_B_STUDY_FIELDS if c in frame.columns]
    if "STUDYID" not in keep:
        return
    files_read.append(source)
    for rec in frame[keep].to_dict("records"):
        sid = str(rec.get("STUDYID") or "").strip()
        if not sid:
            continue
        row = studies.setdefault(sid, _empty_study_row(sid))
        for f in PHASE_B_STUDY_FIELDS:
            if f == "STUDYID":
                continue
            val = rec.get(f)
            if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                continue
            if row.get(f) in (None, ""):
                row[f] = str(val).strip()


def _merge_ts(studies: dict[str, dict[str, Any]], frame: pd.DataFrame,
              source: str, files_read: list[str]) -> None:
    if frame.empty:
        return
    _assert_no_bw_columns(list(frame.columns), source)
    needed = {"STUDYID", "TSPARMCD", "TSVAL"}
    if not needed.issubset(set(frame.columns)):
        return
    files_read.append(source)
    for rec in frame.to_dict("records"):
        parm = str(rec.get("TSPARMCD") or "").strip().upper()
        field = PHASE_B_TS_PARMS.get(parm)
        if field is None:
            continue
        sid = str(rec.get("STUDYID") or "").strip()
        val = rec.get("TSVAL")
        if not sid or val is None or str(val).strip() == "":
            continue
        row = studies.setdefault(sid, _empty_study_row(sid))
        if row.get(field) in (None, ""):
            row[field] = str(val).strip()


def _missingness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    fields = {}
    for f in PHASE_B_STUDY_FIELDS:
        if f == "STUDYID":
            continue
        n_miss = sum(1 for r in rows if r.get(f) in (None, ""))
        fields[f] = {"n_present": n - n_miss, "n_missing": n_miss,
                     "frac_missing": (n_miss / n) if n else None}
    return {"n_studies": n, "fields": fields}


def _biological_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sp: dict[str, list[dict[str, Any]]] = {}
    by_class: dict[str, int] = {}
    for r in rows:
        sp = (r.get("SPECIES") or "UNKNOWN").upper()
        cl = (r.get("STUDY_CLASS") or "UNKNOWN")
        by_sp.setdefault(sp, []).append(r)
        by_class[str(cl).upper()] = by_class.get(str(cl).upper(), 0) + 1
    species = []
    for sp, items in sorted(by_sp.items()):
        n_h = 0
        n_h_unknown = 0
        for r in items:
            dur = r.get("DURATION_DAYS")
            if dur in (None, ""):
                n_h_unknown += 1
                continue
            try:
                n_h += int(float(dur) >= HORIZON_DAY)
            except (TypeError, ValueError):
                n_h_unknown += 1
        species.append({
            "SPECIES": sp,
            "n_studies": len(items),
            "n_strains": len({(r.get("STRAIN") or "UNKNOWN") for r in items}),
            "n_with_start": sum(1 for r in items if r.get("STUDY_START") not in (None, "")),
            "n_with_send_version": sum(1 for r in items if r.get("SEND_VERSION") not in (None, "")),
            "n_horizon_known_ge_28": n_h,
            "n_horizon_unknown": n_h_unknown,
            "meets_donor_n_min": len(items) >= DONOR_N_MIN,
            "meets_target_holdout_n_min": len(items) >= TARGET_HOLDOUT_N_MIN,
        })
    return {
        "n_studies": len(rows),
        "n_species": len(by_sp),
        "species": species,
        "study_class_counts": by_class,
        "horizon_day": HORIZON_DAY,
        "donor_n_min": DONOR_N_MIN,
        "target_holdout_n_min": TARGET_HOLDOUT_N_MIN,
        "sex": {"status": "unknown", "evidence": "DM was not opened in phase B"},
        "note": "Inventory is metadata-only. Unknown is not a pass. Synthetic DOG is not historical support.",
    }


def _candidate_contexts(rows: list[dict[str, Any]],
                        has_arm_regimen: bool, has_compound_ref: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = ((r.get("SPECIES") or "UNKNOWN").upper(), (r.get("STUDY_CLASS") or "UNKNOWN").upper())
        groups.setdefault(key, []).append(r)
    species_n = {}
    for (sp, _cl), items in groups.items():
        species_n[sp] = species_n.get(sp, 0) + len(items)
    out = []
    for (sp, cl), items in sorted(groups.items()):
        donors = [s for s, n in species_n.items() if s != sp and n >= DONOR_N_MIN]
        target_ok = len(items) >= TARGET_HOLDOUT_N_MIN and cl != "UNKNOWN"
        donor_ok = bool(donors)
        horizon_known = any(r.get("DURATION_DAYS") not in (None, "") for r in items)
        if target_ok and donor_ok:
            status = "candidate_metadata_only"
        elif any(v in (None, "", "UNKNOWN") for v in (sp, cl)) or not horizon_known:
            status = "unknown"
        else:
            status = "insufficient_metadata"
        out.append({
            "target_species": sp,
            "study_class": cl,
            "n_target_studies": len(items),
            "donor_species_meeting_n_min": donors,
            "planned_regimen_curated": has_arm_regimen,
            "compound_identity_curated": has_compound_ref,
            "horizon_overlap_day_28": "unknown" if not horizon_known else "metadata_duration_only",
            "sex_both_present": "unknown",
            "status": status,
            "note": (
                "Feasible candidate validation context from metadata presence only. "
                "Not a design recommendation. Not a claim that the library supports transfer."
            ),
        })
    return out


def _extract_blocker(reason: str, auth: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "EXTRACT_BLOCKER",
        "phase": "B",
        "labels": list(LABELS),
        "git_revision": git_revision(),
        "real_data_accessed": False,
        "source_opened": False,
        "endpoint_values_inspected": False,
        "models_trained": False,
        "design_recommendations": False,
        "ticket_path": discovery.get("ticket_path"),
        "steward_name": auth.get("steward_name"),
        "purpose": auth.get("purpose"),
        "blocker": reason,
        "note": "Authorization was schema-valid. No source was opened.",
    }


def phase_b_metadata_inventory(cfg: dict[str, Any], auth: dict[str, Any],
                               discovery: dict[str, Any]) -> dict[str, Any]:
    """Metadata-only inspect. Never reads BWSTRESN/BWORRES or trains models."""
    refuse_real_access(cfg, auth, "B")
    if auth.get("allows_model_training"):
        raise AuditGateError("phase B must not enable model training")
    ticket_src = auth.get("source") or {}
    if (ticket_src.get("type") or "").lower() == "snowflake":
        return _extract_blocker(
            "Snowflake is not opened in this environment; a local metadata-only "
            "snapshot must be named on the ticket",
            auth, discovery,
        )
    root = ticket_src.get("root")
    if not root:
        return _extract_blocker(
            "signed ticket does not name source.root; config source.root is not opened",
            auth, discovery,
        )
    root_p = Path(root)
    if not root_p.exists():
        return _extract_blocker(
            f"ticket source.root does not exist in this environment: {root}",
            auth, discovery,
        )
    layout = str(ticket_src.get("layout") or "per_study")
    metadata_dir = ticket_src.get("metadata_dir")
    domains = list_domains_by_filename(root_p, layout)
    refused_domains = sorted(d for d in domains if d in PHASE_B_FORBIDDEN_DOMAINS or d.startswith("BW"))
    files_read: list[str] = []
    files_refused = [p for d in refused_domains for p in domains[d]]
    studies: dict[str, dict[str, Any]] = {}
    has_arm = False
    has_cref = False
    if metadata_dir:
        md = Path(metadata_dir)
        if md.exists():
            for table in ("study_metadata", "arm_regimen", "compound_reference"):
                for ext in FILE_EXTS:
                    p = md / f"{table}{ext}"
                    if not p.exists():
                        continue
                    df = _read_table(p)
                    if table == "study_metadata":
                        _merge_study_metadata(studies, df, str(p), files_read)
                    elif table == "arm_regimen":
                        has_arm = True
                        files_read.append(str(p))
                    else:
                        has_cref = True
                        files_read.append(str(p))
                    break
    if "TS" in domains:
        for rel in domains["TS"]:
            p = root_p / rel
            _merge_ts(studies, _read_table(p), str(p), files_read)
    rows = [studies[k] for k in sorted(studies)]
    support = _biological_support(rows)
    missing = _missingness(rows)
    candidates = _candidate_contexts(rows, has_arm, has_cref)
    return {
        "status": "ok",
        "phase": "B",
        "labels": list(LABELS),
        "git_revision": git_revision(),
        "real_data_accessed": True,
        "source_opened": True,
        "endpoint_values_inspected": False,
        "body_weight_opened": False,
        "models_trained": False,
        "design_recommendations": False,
        "ticket_path": discovery.get("ticket_path"),
        "steward_name": auth.get("steward_name"),
        "purpose": auth.get("purpose"),
        "provenance": {
            "source_type": ticket_src.get("type") or "local",
            "root": str(root_p),
            "layout": layout,
            "metadata_dir": str(metadata_dir) if metadata_dir else None,
            "files_read": files_read,
            "files_refused": files_refused,
            "domains_present": sorted(domains),
            "domains_refused": refused_domains,
            "mapping": (cfg.get("mapping") if cfg else None),
        },
        "biological_support": support,
        "missingness": missing,
        "candidate_validation_contexts": candidates,
        "studies": rows,
        "support_for_synthetic_to_real": "not_claimed",
        "note": (
            "Phase B metadata only. Endpoint values were not inspected. "
            "No model was trained. No design recommendation is offered."
        ),
    }


def execute_phase_b(cfg: dict[str, Any], authorization_path: str | Path | None = None,
                    out_dir: str | Path | None = None) -> dict[str, Any]:
    discovery = discover_authorization(authorization_path)
    dest = Path(out_dir) if out_dir else PHASE_B_DEFAULT_OUT
    ticket_out = None
    if discovery.get("auth"):
        ticket_out = (discovery["auth"] or {}).get("output_dir")
    dest = Path(ticket_out or dest or PHASE_B_DEFAULT_OUT)
    if dest.resolve() in {Path(p).resolve() for p in ("runs/v3_eval", "runs/v3_pilot_k0k1",
                                                      "runs/v3_repl", "runs/v3_research",
                                                      "runs/v3_closeout", "runs/v3_audit")}:
        dest = PHASE_B_DEFAULT_OUT
    if not discovery["authorized"]:
        return write_access_blocker(dest, discovery, phase="B")
    auth = discovery["auth"]
    validate_authorization(auth, "B")
    refuse_real_access(cfg, auth, "B")
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "authorization_discovery.json", {k: v for k, v in discovery.items() if k != "auth"})
    result = phase_b_metadata_inventory(cfg, auth, discovery)
    write_json(dest / "phase_b_inventory.json", result)
    if result.get("status") == "EXTRACT_BLOCKER":
        write_json(dest / "EXTRACT_BLOCKER.json", result)
    _write_phase_b_md(dest, result)
    if dest.name == "v3_phaseb":
        Path("docs/v3/phase_b_access.md").write_text(
            (dest / "PHASE_B.md").read_text(encoding="utf-8"), encoding="utf-8")
    return result


def _write_phase_b_md(parent: Path, result: dict[str, Any]) -> None:
    status = result.get("status")
    lines = [
        "# Historical-data audit — Phase B",
        "",
        "> **No operational recommendation. Not design qualification.**",
        "",
        f"- status: `{status}`",
        f"- git revision: `{result.get('git_revision')}`",
        f"- real_data_accessed: {result.get('real_data_accessed')}",
        f"- source_opened: {result.get('source_opened')}",
        f"- endpoint_values_inspected: {result.get('endpoint_values_inspected')}",
        f"- models_trained: {result.get('models_trained')}",
        f"- design_recommendations: {result.get('design_recommendations')}",
        "",
    ]
    if status == "ACCESS_BLOCKER":
        lines += [
            "## Access blocker",
            "",
            result.get("blocker") or "",
            "",
            result.get("next") or "",
            "",
            "Candidates checked:",
            "",
        ]
        for c in (result.get("discovery") or {}).get("candidates_checked") or []:
            lines.append(f"- `{c.get('path')}` exists={c.get('exists')} genuine={c.get('genuine')}: {c.get('reason')}")
        if not (result.get("discovery") or {}).get("candidates_checked"):
            lines.append("- none (no --authorization, no SEND_BW_AUTHORIZATION, no /approved/tickets)")
        lines += ["", "No source was opened. No ticket was created or self-approved.", ""]
    elif status == "EXTRACT_BLOCKER":
        lines += ["## Extract blocker", "", result.get("blocker") or "", "",
                  "Authorization was accepted. The named extract was not opened.", ""]
    else:
        prov = result.get("provenance") or {}
        miss = result.get("missingness") or {}
        support = result.get("biological_support") or {}
        lines += [
            "## Provenance",
            "",
            f"- steward: {result.get('steward_name')}",
            f"- purpose: {result.get('purpose')}",
            f"- source type: {prov.get('source_type')}",
            f"- domains present: {prov.get('domains_present')}",
            f"- domains refused: {prov.get('domains_refused')}",
            f"- files read: {prov.get('files_read')}",
            "",
            "## Biological support (metadata only)",
            "",
            f"- n_studies: {support.get('n_studies')}; n_species: {support.get('n_species')}",
            f"- support_for_synthetic_to_real: `{result.get('support_for_synthetic_to_real')}`",
            "",
            "| species | n | strains | start | SEND ver | horizon≥28 | horizon unknown | donor n | holdout n |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in support.get("species") or []:
            lines.append(
                f"| {s.get('SPECIES')} | {s.get('n_studies')} | {s.get('n_strains')} | "
                f"{s.get('n_with_start')} | {s.get('n_with_send_version')} | "
                f"{s.get('n_horizon_known_ge_28')} | {s.get('n_horizon_unknown')} | "
                f"{s.get('meets_donor_n_min')} | {s.get('meets_target_holdout_n_min')} |"
            )
        lines += ["", "## Missingness", "",
                  f"- n_studies: {miss.get('n_studies')}", "",
                  "| field | present | missing | frac missing |",
                  "|---|---|---|---|"]
        for f, rec in (miss.get("fields") or {}).items():
            lines.append(f"| {f} | {rec.get('n_present')} | {rec.get('n_missing')} | {rec.get('frac_missing')} |")
        lines += ["", "## Candidate validation contexts", "",
                  "| target | class | n | donors | regimen | compound | horizon | sex | status |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in result.get("candidate_validation_contexts") or []:
            lines.append(
                f"| {c.get('target_species')} | {c.get('study_class')} | {c.get('n_target_studies')} | "
                f"{c.get('donor_species_meeting_n_min')} | {c.get('planned_regimen_curated')} | "
                f"{c.get('compound_identity_curated')} | {c.get('horizon_overlap_day_28')} | "
                f"{c.get('sex_both_present')} | {c.get('status')} |"
            )
        lines += ["", result.get("note") or "", ""]
    (parent / "PHASE_B.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

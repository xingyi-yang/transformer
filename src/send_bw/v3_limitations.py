"""Record unresolved modeling limitations from existing artifacts.

Reads the matched increment diagnostic and Bayesian gate only. Does not
refit, retune, calibrate to exposed test outcomes, add architectures, or
launch a grid. Broad synthetic development stops here.

Phase B is rediscovered, not executed. Without a genuine signed ticket this
module reports ACCESS_BLOCKER and does not open a source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS
from send_bw.historical_audit import discover_authorization
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import git_revision, read_json, write_json

PRESERVED = (
    "runs/v3_eval",
    "runs/v3_pilot_k0k1",
    "runs/v3_repl",
    "runs/v3_research",
    "runs/v3_closeout",
    "runs/v3_audit",
    "runs/v3_increment",
    "runs/v3_phaseb",
)


class LimitationsError(ValueError):
    pass


def execute_v3_limitations(increment_dir: str | Path = "runs/v3_increment",
                           phaseb_dir: str | Path = "runs/v3_phaseb",
                           output_dir: str | Path = "runs/v3_limitations") -> dict[str, Any]:
    increment_dir = Path(increment_dir)
    phaseb_dir = Path(phaseb_dir)
    output_dir = Path(output_dir)
    if output_dir.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise LimitationsError("refusing to write into a preserved artifact directory")
    inc_path = increment_dir / "increment.json"
    bay_path = increment_dir / "bayesian.json"
    if not inc_path.exists() or not bay_path.exists():
        raise LimitationsError(f"missing increment diagnostic under {increment_dir}")
    increment = read_json(inc_path)
    bayesian = read_json(bay_path)
    summary = increment.get("summary") or {}
    overall = summary.get("overall_matched_ratio") or {}
    discovery = discover_authorization(None)
    phaseb = {}
    if (phaseb_dir / "ACCESS_BLOCKER.json").exists():
        phaseb = read_json(phaseb_dir / "ACCESS_BLOCKER.json")
    authorized = bool(discovery.get("authorized"))
    if authorized:
        phase_b_status = "authorized_not_executed_here"
        blocker = None
        stop_reason = (
            "A genuine ticket was found. This recorder does not open a source; "
            "run historical-audit --phase B in the approved environment."
        )
    else:
        phase_b_status = "ACCESS_BLOCKER"
        blocker = (
            "No genuine signed authorization was found. Phase B remains blocked "
            "until approval is supplied through the organization's accepted "
            "process. No source was opened. No further synthetic implementation "
            "task is generated."
        )
        stop_reason = blocker
    increment_lim = {
        "id": "excess_increment_variability",
        "status": "unresolved",
        "adjusted_using_exposed_test": False,
        "statistic": (increment.get("statistic") or {}),
        "n_datasets": increment.get("n_datasets"),
        "n_cells_scored": increment.get("n_cells_scored"),
        "n_study_rows": summary.get("n_study_rows"),
        "matched_ratio_mean_of_dataset_means": overall.get("mean_of_group_means"),
        "matched_ratio_sd_of_dataset_means": overall.get("sd_of_group_means"),
        "source": str(inc_path),
        "finding": (
            "After matching the animal→study→dataset increment statistic, predicted "
            "study-level increment SD still exceeds the observed study-level "
            "statistic (ratio > 1 on the mean of two dataset means). Excess "
            "increment variability remains an unresolved modeling limitation."
        ),
        "not_done": [
            "No residual scale was multiplied to match observed increment SD on test.",
            "No increment-ratio target was used as a tuning objective.",
            "Covariance bounds (including sigma_m at the optimizer floor) were not "
            "moved using exposed test outcomes.",
            "Predictive widths were not calibrated to test coverage.",
        ],
    }
    bayes_lim = {
        "id": "unreliable_bayesian_inference",
        "status": "unresolved",
        "adjusted_using_exposed_test": False,
        "explicitly_unreliable": True,
        "scored": False,
        "counted_as_successful_benchmark": False,
        "thresholds_unchanged": dict(RELIABILITY_THRESHOLDS),
        "n_cells": bayesian.get("n_cells"),
        "cells": bayesian.get("cells") or [],
        "source": str(bay_path),
        "finding": (
            "Executed Bayesian cells fail the unchanged reliability gate "
            "(rhat≤1.01, ESS≥400, divergences==0) on mean-component identifiability. "
            "Inference stays explicitly unreliable and is not counted."
        ),
        "not_done": [
            "Thresholds were not relaxed.",
            "The mean parameterization was not changed to chase ESS or R-hat.",
            "Failed runs were not dropped or retuned on test.",
        ],
    }
    out = {
        "labels": list(LABELS) + [
            "Exploratory.",
            "Two independent dataset replicates.",
            "Broad synthetic development stopped.",
        ],
        "git_revision": git_revision(),
        "preserved_pr": 7,
        "retuned_after_test": False,
        "architectures_added": False,
        "grid_relaunched": False,
        "adjusted_using_exposed_test": False,
        "broad_synthetic_development": "stopped",
        "next_implementation_task": None,
        "limitations": [increment_lim, bayes_lim],
        "phase_b": {
            "status": phase_b_status,
            "authorized": authorized,
            "created_ticket": False,
            "self_approved": False,
            "source_opened": False,
            "endpoint_values_inspected": False,
            "models_trained": False,
            "design_recommendations": False,
            "blocker": blocker,
            "discovery": {k: v for k, v in discovery.items() if k != "auth"},
            "prior_blocker": {
                "dir": str(phaseb_dir),
                "status": phaseb.get("status"),
                "real_data_accessed": phaseb.get("real_data_accessed"),
            },
            "if_authorized": (
                "Metadata-only inspection: biological-support inventory, missingness, "
                "provenance, and candidate historical validation cohorts. Endpoint "
                "inspection, training, and design recommendations stay off unless "
                "separately authorized."
            ),
        },
        "stop_reason": stop_reason,
        "reliability_thresholds": dict(RELIABILITY_THRESHOLDS),
        "source_increment": {
            "dir": str(increment_dir),
            "research": increment.get("source_research"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "limitations.json", out)
    md = _write_md(output_dir, out)
    if output_dir.name == "v3_limitations":
        Path("docs/v3/unresolved_limitations.md").write_text(md, encoding="utf-8")
    return out


def _fmt(x: Any, nd: int = 5) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _write_md(parent: Path, out: dict[str, Any]) -> str:
    lims = {item["id"]: item for item in out.get("limitations") or []}
    inc = lims.get("excess_increment_variability") or {}
    bay = lims.get("unreliable_bayesian_inference") or {}
    phase = out.get("phase_b") or {}
    lines = [
        "# Unresolved modeling limitations",
        "",
        "> **Synthetic diagnostic. Exploratory. Not design qualification. "
        "No real-data validation. No operational recommendation.**",
        "",
        "PR #7 and prior artifacts are preserved. Broad synthetic development "
        "stops here. Two modeling limitations remain unresolved. Neither was "
        "adjusted using exposed test outcomes.",
        "",
        "## Status",
        "",
        f"- preserved PR: #{out.get('preserved_pr')}",
        f"- broad synthetic development: **{out.get('broad_synthetic_development')}**",
        f"- next implementation task: `{out.get('next_implementation_task')}`",
        f"- adjusted using exposed test: {out.get('adjusted_using_exposed_test')}",
        f"- reliability thresholds: `{out.get('reliability_thresholds')}`",
        f"- record revision: `{out.get('git_revision')}`",
        "",
        "## 1. Excess increment variability",
        "",
        f"- status: **{inc.get('status')}**",
        f"- adjusted using exposed test: {inc.get('adjusted_using_exposed_test')}",
        f"- n_datasets: {inc.get('n_datasets')}; cells: {inc.get('n_cells_scored')}; "
        f"study rows: {inc.get('n_study_rows')}",
        f"- matched pred/obs ratio, mean of dataset means: "
        f"**{_fmt(inc.get('matched_ratio_mean_of_dataset_means'))}** "
        f"(sd of dataset means {_fmt(inc.get('matched_ratio_sd_of_dataset_means'))})",
        "",
        inc.get("finding") or "",
        "",
        "Not done: " + "; ".join(inc.get("not_done") or []),
        "",
        "## 2. Unreliable Bayesian inference",
        "",
        f"- status: **{bay.get('status')}**",
        f"- adjusted using exposed test: {bay.get('adjusted_using_exposed_test')}",
        f"- explicitly unreliable: {bay.get('explicitly_unreliable')}; scored: {bay.get('scored')}",
        f"- thresholds unchanged: `{bay.get('thresholds_unchanged')}`",
        "",
        bay.get("finding") or "",
        "",
        "| cell | class | rhat | ESS bulk/tail | counted |",
        "|---|---|---|---|---|",
    ]
    for c in bay.get("cells") or []:
        lines.append(
            f"| {c.get('name')} | {c.get('failure_class')} | {_fmt(c.get('rhat_max'), 3)} | "
            f"{_fmt(c.get('ess_bulk_min'), 1)} / {_fmt(c.get('ess_tail_min'), 1)} | "
            f"{c.get('counted_as_successful_benchmark')} |"
        )
    lines += [
        "",
        "Not done: " + "; ".join(bay.get("not_done") or []),
        "",
        "## Phase B",
        "",
        f"- status: **{phase.get('status')}**",
        f"- authorized: {phase.get('authorized')}",
        f"- created_ticket: {phase.get('created_ticket')}; self_approved: {phase.get('self_approved')}",
        f"- source_opened: {phase.get('source_opened')}",
        f"- endpoint_values_inspected: {phase.get('endpoint_values_inspected')}",
        f"- models_trained: {phase.get('models_trained')}",
        f"- design_recommendations: {phase.get('design_recommendations')}",
        "",
        phase.get("blocker") or "",
        "",
        "If later authorized through the organization's accepted process: "
        + (phase.get("if_authorized") or ""),
        "",
        "## Stop",
        "",
        out.get("stop_reason") or "",
        "",
        "## Artifacts",
        "",
        f"- `{parent / 'limitations.json'}`",
        f"- `{parent / 'LIMITATIONS.md'}`",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (parent / "LIMITATIONS.md").write_text(text, encoding="utf-8")
    return text

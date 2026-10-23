"""Locked MCAR B=2000 diagnostic for A / B / C1.

Same small matrix and 2%/2% design-dependent missingness as the exploratory
MCAR stream. Independent seed 20270201. C1 is not retuned.
Not prior-outcome MAR, not design qualification, not a default-switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG))

from run_c1_exploratory import (  # noqa: E402
    GATES,
    SCENARIOS,
    SIZES,
    _finite,
    _fmt_rate,
    cell_table,
    gate_review,
)
from run_c1_mcar_exploratory import P_DROPOUT, P_MISSED_VISIT, simulate  # noqa: E402

OUT = ROOT / "docs/remediation/mcar_locked_20270201_c1"
EXPLORATORY_MCAR = ROOT / "docs/remediation/mcar_exploratory_20270101_c1/gate_review.json"
N_REP = 2000
MASTER_SEED = 20270201
STATUS = [
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
    "Locked MCAR B=2000. Independent of exploratory 20270101. Not prior-outcome MAR. Not a default-switch.",
]


def compare_reviews(locked: list[dict], prev_path: Path) -> list[dict]:
    if not prev_path.is_file():
        return []
    prev = {(r["analysis"], r["schedule"], r["n_per_sex"]): r
            for r in json.loads(prev_path.read_text())["gate_review"]}
    out = []
    for r in locked:
        e = prev.get((r["analysis"], r["schedule"], r["n_per_sex"]))
        if e is None:
            continue
        out.append({
            "analysis": r["analysis"],
            "schedule": r["schedule"],
            "n_per_sex": r["n_per_sex"],
            "exploratory_mcar_inference_tolerances_met": e["inference_tolerances_met"],
            "locked_mcar_inference_tolerances_met": r["inference_tolerances_met"],
            "same_inference_decision": e["inference_tolerances_met"] == r["inference_tolerances_met"],
            "exploratory_mcar_fwer_max_point": e["fwer_max_point"],
            "locked_mcar_fwer_max_point": r["fwer_max_point"],
            "exploratory_mcar_coverage_min": e["coverage_min"],
            "locked_mcar_coverage_min": r["coverage_min"],
        })
    return out


def write_report(path: Path, wall_s: float, n_fits: int, n_studies: int,
                 cells: pd.DataFrame, reviews: list[dict], audit: dict,
                 comparison: list[dict]) -> None:
    lines = [
        "# C1 locked MCAR B=2000 diagnostic",
        "",
        "## Status",
        "",
    ]
    lines += [f"- **{x}**" for x in STATUS]
    lines += [
        "",
        f"- Seed `{MASTER_SEED}`; {N_REP} replicates; 8 scenarios × 2 sizes × 5 analyses.",
        f"- Fits recorded: **{n_fits}**. Study datasets: **{n_studies}**.",
        f"- Wall clock: **{wall_s:.3f} s**.",
        f"- Missingness: `p_missed_visit={P_MISSED_VISIT}`, `p_dropout={P_DROPOUT}` as in `paths.simulate_batch`.",
        f"- Observed endpoint fraction: **{audit['observed_endpoint_fraction']:.4f}**. "
        f"Observed visit fraction: **{audit['observed_visit_fraction']:.4f}**.",
        "- Absence is independent of Z and of zb/sex/group. This is **not** prior-outcome MAR.",
        "- Independent of exploratory seed `20270101`. C1 was not retuned.",
        "- Production defaults remain `[ancova, gls]`.",
        "- Failed analyses count as non-rejections and remain in the FWER/power denominator.",
        "",
        "## Sourced tolerances (not exact 0.05 control)",
        "",
        "From `configs/new_study.example.yaml` and `optimize.py::feasibility`:",
        "",
        f"- FWER point ceiling {GATES['fwer_point_ceiling']} (0.05+0.015); CI ceiling {GATES['fwer_ci_ceiling']} (0.05+0.045).",
        f"- Coverage floor {GATES['coverage_min']}; bias fraction {GATES['max_abs_bias_fraction']}; failure rate {GATES['max_failure_rate']}.",
        f"- Power target {GATES['power_target']} with CI floor {GATES['power_ci_floor']} applies to a chosen design, not every small n.",
        "- Passing a tolerated FWER gate is not “FWER = 0.05.” C1 weekly sat on the 0.065 ceiling in exploratory MCAR; this stream checks whether that pass holds.",
        "",
        "## Candidate review (locked MCAR, not qualification)",
        "",
        "| Analysis | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference tolerances | Power if chosen |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in reviews:
        lines.append(
            f"| `{r['analysis']}` | {r['schedule']} | {r['n_per_sex']} | "
            f"{_fmt_rate(r['fwer_max_point'])} [{_fmt_rate(r['fwer_max_ci_hi'])}] | "
            f"{_fmt_rate(r['coverage_min'])} | {_fmt_rate(r['bias_frac_alternative'])} | "
            f"{_fmt_rate(r['failure_rate_max'])} | "
            f"{_fmt_rate(r['power_all_required_alternative'])}"
            + (f" [{r['power_all_required_ci_lo']:.3f}]" if _finite(r["power_all_required_ci_lo"]) else "")
            + " | "
            f"{'yes' if r['inference_tolerances_met'] else 'no'} | "
            f"{'yes' if r['pass_power_if_chosen_design'] else 'no'} |"
        )
    lines += [
        "",
        "“Inference tolerances” is FWER + coverage + alternative bias + failures. "
        "It is not a primary-analysis decision and not design qualification.",
        "",
        "## Cells that miss a sourced inference tolerance",
        "",
    ]
    misses = []
    for r in reviews:
        flags = []
        if not r["pass_fwer_tolerance"]:
            flags.append(f"FWER {r['fwer_max_point']:.3f} (worst {r['fwer_worst_scenario']})")
        if not r["pass_coverage"]:
            flags.append(f"coverage {r['coverage_min']:.3f} (worst {r['coverage_worst_scenario']})")
        if not r["pass_bias"]:
            flags.append(f"alt bias frac {r['bias_frac_alternative']}")
        if not r["pass_failures"]:
            flags.append(f"failure {r['failure_rate_max']:.3f}")
        if flags:
            misses.append(f"- `{r['analysis']}` {r['schedule']} {r['n_per_sex']}: " + "; ".join(flags))
    lines += misses or ["- None of the 10 analysis×schedule×size combinations missed FWER, coverage, bias, or failure tolerances."]
    if comparison:
        lines += [
            "",
            "## Agreement with exploratory MCAR B=600 inference decisions",
            "",
            "Same pre-specified analyses, matrix, and missingness; independent seed. A changed decision would be reported, not patched.",
            "",
            "| Analysis | Schedule | n/sex | Exploratory MCAR | Locked MCAR | Same decision | FWER expl → locked |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in comparison:
            lines.append(
                f"| `{c['analysis']}` | {c['schedule']} | {c['n_per_sex']} | "
                f"{'met' if c['exploratory_mcar_inference_tolerances_met'] else 'missed'} | "
                f"{'met' if c['locked_mcar_inference_tolerances_met'] else 'missed'} | "
                f"{'yes' if c['same_inference_decision'] else 'NO'} | "
                f"{c['exploratory_mcar_fwer_max_point']:.3f} → {c['locked_mcar_fwer_max_point']:.3f} |"
            )
    lines += [
        "",
        "## A versus C1 under complete data vs MCAR (audit)",
        "",
        f"- Complete-data pairs on these draws: **{audit['n_complete_pairs']}**. "
        f"Max |est A − C1|: **{audit['max_abs_est_A_vs_C1_complete']:.3e}**.",
        f"- MCAR pairs: **{audit['n_mcar_pairs']}**. "
        f"Max |est A − C1 weekly|: **{audit['max_abs_est_A_vs_C1_mcar']:.3e}**.",
        f"- Max |est C1 weekly − C1 biweekly| under MCAR: **{audit['max_abs_est_C1w_vs_C1b_mcar']:.3e}**.",
        f"- Holm disagreements A vs C1 weekly under MCAR: **{audit['holm_disagree_A_vs_C1_mcar']}**.",
        f"- Holm disagreements C1 weekly vs biweekly under MCAR: **{audit['holm_disagree_C1w_vs_C1b_mcar']}**.",
        "",
        "## Per-cell operating characteristics",
        "",
        "Full rows are in `cell_oc.csv`.",
        "",
        "| Analysis | Schedule | Scenario | n/sex | FWER | Coverage min | Bias max | Power-all | Fail |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    show = cells.sort_values(["analysis", "schedule", "n_per_sex", "scenario"])
    for r in show.itertuples(index=False):
        fwer = _fmt_rate(getattr(r, "fwer", None), getattr(r, "fwer_ci_lo", None), getattr(r, "fwer_ci_hi", None))
        power = _fmt_rate(getattr(r, "power_all_required", None),
                          getattr(r, "power_all_required_ci_lo", None),
                          getattr(r, "power_all_required_ci_hi", None))
        lines.append(
            f"| `{r.analysis}` | {r.schedule} | {r.scenario} | {r.n_per_sex} | "
            f"{fwer} | {_fmt_rate(r.coverage_min)} | {_fmt_rate(r.bias_max_abs)} | "
            f"{power} | {_fmt_rate(r.failure_rate)} |"
        )
    lines += [
        "",
        "## Constraints on use",
        "",
        "- Not prior-outcome MAR.",
        "- Not design qualification and not an operational recommendation.",
        "- Do not switch production defaults.",
        "- Do not retune C1 because a FWER point sat on or near 0.065.",
        "- Deferred: covariance shift, heavy tails, delayed alternatives, 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bundle(fits, audit: dict, wall_s: float, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cells = cell_table(fits)
    cells["missingness"] = "mcar_design_dependent"
    reviews = gate_review(cells)
    comparison = compare_reviews(reviews, EXPLORATORY_MCAR)
    n_fits = int(len(fits))
    n_studies = int(N_REP * len(SCENARIOS) * len(SIZES))
    payload = {
        "stream_id": "c1-mcar-locked-20270201",
        "status": "mcar_locked_completed_not_qualification",
        "labels": STATUS,
        "master_seed": MASTER_SEED,
        "n_replicates": N_REP,
        "n_study_datasets": n_studies,
        "n_analysis_fits": n_fits,
        "wall_seconds": wall_s,
        "c1_retuned": False,
        "independent_of_exploratory_mcar_seed": True,
        "missingness": {
            "name": "mcar_design_dependent",
            "p_missed_visit": P_MISSED_VISIT,
            "p_dropout": P_DROPOUT,
            "source": "configs/new_study.example.yaml missing_data; paths.simulate_batch",
            "independent_of_Z": True,
            "prior_outcome_mar": False,
        },
        "audit": audit,
        "gates": GATES,
        "gate_review": reviews,
        "exploratory_mcar_comparison": comparison,
        "forbidden_use": [
            "Not prior-outcome MAR.",
            "Not design qualification.",
            "Do not switch production defaults.",
            "Do not retune C1 on these draws.",
        ],
    }
    cells.to_csv(out / "cell_oc.csv", index=False)
    (out / "gate_review.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(out / "REPORT.md", wall_s, n_fits, n_studies, cells, reviews, audit, comparison)
    names = ("cell_oc.csv", "gate_review.json", "REPORT.md")
    manifest = {"bundle_dir": str(out.relative_to(ROOT)), "labels": STATUS, "files": {}}
    for name in names:
        p = out / name
        manifest["files"][name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "n_analysis_fits": n_fits, "wall_seconds": wall_s,
            "audit": audit, "gate_review": reviews, "exploratory_mcar_comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 replicates in memory; do not write the bundle.")
    args = parser.parse_args()
    n_rep = 2 if args.smoke else N_REP
    t0 = time.perf_counter()
    fits, audit = simulate(n_rep, MASTER_SEED)
    wall_s = time.perf_counter() - t0
    expected = n_rep * len(SCENARIOS) * len(SIZES) * 5
    if len(fits) != expected:
        raise SystemExit(f"fit count {len(fits)} != {expected}")
    if args.smoke:
        print(json.dumps({
            "smoke": True,
            "n_fits": int(len(fits)),
            "wall_seconds": wall_s,
            "failure_rate": float(fits["failed"].mean()),
            "audit": audit,
        }, indent=2))
        return
    payload = write_bundle(fits, audit, wall_s, OUT)
    print(json.dumps({
        "bundle_dir": payload["bundle_dir"],
        "n_analysis_fits": payload["n_analysis_fits"],
        "wall_seconds": payload["wall_seconds"],
        "files": payload["files"],
        "audit": payload["audit"],
        "exploratory_mcar_comparison": payload["exploratory_mcar_comparison"],
        "gate_review": payload["gate_review"],
    }, indent=2))


if __name__ == "__main__":
    main()

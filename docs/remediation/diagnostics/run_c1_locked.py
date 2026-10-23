"""Locked B=2000 diagnostic stream for A / B / C1.

Same small complete-data matrix as the exploratory stream. Independent seed
20261201. C1 is not retuned. Not design qualification and not a default-switch.
C2 and research.yaml are not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG))

from run_c1_exploratory import (  # noqa: E402
    BASE,
    BIWEEKLY_IDX,
    GATES,
    HORIZON,
    SCENARIOS,
    SIZES,
    WEEKLY,
    _finite,
    _fmt_rate,
    _py,
    animals,
    cell_table,
    confidence_interval,
    endpoint_ancova,
    gate_review,
    holm,
    mean_path,
    mmrm_visit_contrast,
    rm_gls,
    truth_cov,
)

OUT = ROOT / "docs/remediation/locked_20261201_c1"
EXPLORATORY_REVIEW = ROOT / "docs/remediation/exploratory_20261101_c1/gate_review.json"
N_REP = 2000
MASTER_SEED = 20261201
ALPHA = 0.05
CI_LEVEL = 0.95
STATUS = [
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
    "Locked B=2000 diagnostic. Independent of exploratory draws. Not a default-switch.",
]


def simulate(n_rep: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    V0_w = BASE.animal_cov(WEEKLY)
    V0_b = BASE.animal_cov(WEEKLY[BIWEEKLY_IDX])
    rows: list[dict] = []
    ident = {
        "n_pairs": 0,
        "max_abs_est_A_vs_C1": 0.0,
        "max_abs_est_C1w_vs_C1b": 0.0,
        "max_abs_se_C1w_vs_C1b": 0.0,
        "holm_disagree_A_vs_C1": 0,
        "holm_disagree_C1w_vs_C1b": 0,
    }
    for scen in SCENARIOS:
        dgp = truth_cov(scen["animal_slope"], scen["study_slope"])
        deltas = scen["deltas"]
        for n_per in SIZES:
            n_label = "/".join(map(str, n_per))
            n_animals = int(sum(n_per) * 2)
            for _rep in range(n_rep):
                zb, sex, group = animals(n_per, rng)
                resid = dgp.sample(rng, WEEKLY, len(zb), 1)[0]
                Z = mean_path(zb, sex, group, WEEKLY, deltas, scen["shape"]) + resid
                Zb = Z[:, BIWEEKLY_IDX]
                jobs = [
                    ("A_endpoint_ancova", "endpoint_only", endpoint_ancova,
                     (zb, sex, group, Z[:, -1], 2)),
                    ("B_rm_gls_quadratic", "weekly", rm_gls,
                     (zb, sex, group, Z, WEEKLY, HORIZON, V0_w, 2)),
                    ("B_rm_gls_quadratic", "biweekly", rm_gls,
                     (zb, sex, group, Zb, WEEKLY[BIWEEKLY_IDX], HORIZON, V0_b, 2)),
                    ("C1_mmrm_visit_contrast", "weekly", mmrm_visit_contrast,
                     (zb, sex, group, Z, WEEKLY, HORIZON, V0_w, 2)),
                    ("C1_mmrm_visit_contrast", "biweekly", mmrm_visit_contrast,
                     (zb, sex, group, Zb, WEEKLY[BIWEEKLY_IDX], HORIZON, V0_b, 2)),
                ]
                got = {}
                for analysis, schedule, fn, args in jobs:
                    res = fn(*args)
                    failed = bool(res.failed)
                    if failed:
                        est = se = p = lo = hi = np.full(2, np.nan)
                        rej = np.zeros(2, dtype=bool)
                    else:
                        est, se, p = res.est, res.se, res.p
                        lo, hi = confidence_interval(res, CI_LEVEL)
                        rej = holm(p, ALPHA)
                    got[(analysis, schedule)] = (failed, est, se, rej)
                    rows.append({
                        "scenario": scen["name"],
                        "n_per_sex": n_label,
                        "n_animals": n_animals,
                        "analysis": analysis,
                        "schedule": schedule,
                        "failed": failed,
                        "reason": res.reason if failed else "",
                        "est_low": _py(est[0]),
                        "est_high": _py(est[1]),
                        "se_low": _py(se[0]),
                        "se_high": _py(se[1]),
                        "lo_low": _py(lo[0]),
                        "hi_low": _py(hi[0]),
                        "lo_high": _py(lo[1]),
                        "hi_high": _py(hi[1]),
                        "rej_low": bool(rej[0]),
                        "rej_high": bool(rej[1]),
                    })
                a = got[("A_endpoint_ancova", "endpoint_only")]
                cw = got[("C1_mmrm_visit_contrast", "weekly")]
                cb = got[("C1_mmrm_visit_contrast", "biweekly")]
                if not (a[0] or cw[0] or cb[0]):
                    ident["n_pairs"] += 1
                    ident["max_abs_est_A_vs_C1"] = max(
                        ident["max_abs_est_A_vs_C1"], float(np.max(np.abs(a[1] - cw[1])))
                    )
                    ident["max_abs_est_C1w_vs_C1b"] = max(
                        ident["max_abs_est_C1w_vs_C1b"], float(np.max(np.abs(cw[1] - cb[1])))
                    )
                    ident["max_abs_se_C1w_vs_C1b"] = max(
                        ident["max_abs_se_C1w_vs_C1b"], float(np.max(np.abs(cw[2] - cb[2])))
                    )
                    ident["holm_disagree_A_vs_C1"] += int(np.any(a[3] != cw[3]))
                    ident["holm_disagree_C1w_vs_C1b"] += int(np.any(cw[3] != cb[3]))
    return pd.DataFrame(rows), ident


def compare_reviews(locked: list[dict], exploratory_path: Path) -> list[dict]:
    if not exploratory_path.is_file():
        return []
    prev = { (r["analysis"], r["schedule"], r["n_per_sex"]): r
             for r in json.loads(exploratory_path.read_text())["gate_review"] }
    out = []
    for r in locked:
        key = (r["analysis"], r["schedule"], r["n_per_sex"])
        e = prev.get(key)
        if e is None:
            continue
        out.append({
            "analysis": r["analysis"],
            "schedule": r["schedule"],
            "n_per_sex": r["n_per_sex"],
            "exploratory_inference_tolerances_met": e["inference_tolerances_met"],
            "locked_inference_tolerances_met": r["inference_tolerances_met"],
            "same_inference_decision": e["inference_tolerances_met"] == r["inference_tolerances_met"],
            "exploratory_fwer_max_point": e["fwer_max_point"],
            "locked_fwer_max_point": r["fwer_max_point"],
            "exploratory_coverage_min": e["coverage_min"],
            "locked_coverage_min": r["coverage_min"],
        })
    return out


def write_report(path: Path, wall_s: float, n_fits: int, n_studies: int,
                 cells: pd.DataFrame, reviews: list[dict], ident: dict,
                 comparison: list[dict]) -> None:
    lines = [
        "# C1 locked B=2000 diagnostic",
        "",
        "## Status",
        "",
    ]
    lines += [f"- **{x}**" for x in STATUS]
    lines += [
        "",
        f"- Seed `{MASTER_SEED}`; {N_REP} replicates; complete data; 8 scenarios × 2 sizes × 5 analyses.",
        f"- Fits recorded: **{n_fits}**. Study datasets: **{n_studies}**.",
        f"- Wall clock: **{wall_s:.3f} s**.",
        "- Independent of exploratory seed `20261101`. C1 was not retuned.",
        "- Common random numbers: one weekly latent path per (scenario, size, replicate); biweekly is a visit subset.",
        "- Working covariance: unstressed `HierarchicalCovariance.animal_cov` (historical plug-in stand-in).",
        "- Failed analyses count as non-rejections and remain in the FWER/power denominator.",
        "- C1 remains a **candidate**. Production defaults remain `[ancova, gls]`.",
        "- Replicate-level draws are not archived. This runner plus the seed reproduces them.",
        "",
        "## Sourced tolerances (not exact 0.05 control)",
        "",
        "From `configs/new_study.example.yaml` and `optimize.py::feasibility`:",
        "",
        f"- FWER point ceiling {GATES['fwer_point_ceiling']} (0.05+0.015); CI ceiling {GATES['fwer_ci_ceiling']} (0.05+0.045).",
        f"- Coverage floor {GATES['coverage_min']}; bias fraction {GATES['max_abs_bias_fraction']}; failure rate {GATES['max_failure_rate']}.",
        f"- Power target {GATES['power_target']} with CI floor {GATES['power_ci_floor']} applies to a chosen design, not every small n.",
        "- Passing a tolerated FWER gate is not “FWER = 0.05.”",
        "",
        "## Candidate review (locked diagnostic, not qualification)",
        "",
        "| Analysis | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference tolerances | Power if chosen |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
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
            "## Agreement with exploratory B=600 inference decisions",
            "",
            "Same pre-specified analyses and matrix; independent seed. A changed decision would be reported, not patched.",
            "",
            "| Analysis | Schedule | n/sex | Exploratory | Locked | Same decision |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for c in comparison:
            lines.append(
                f"| `{c['analysis']}` | {c['schedule']} | {c['n_per_sex']} | "
                f"{'met' if c['exploratory_inference_tolerances_met'] else 'missed'} | "
                f"{'met' if c['locked_inference_tolerances_met'] else 'missed'} | "
                f"{'yes' if c['same_inference_decision'] else 'NO'} |"
            )
    lines += [
        "",
        "## Complete-data identities (audit, not a gate)",
        "",
        f"- Successful complete-data triples compared: **{ident['n_pairs']}**.",
        f"- Max |est A − est C1 weekly|: **{ident['max_abs_est_A_vs_C1']:.3e}**.",
        f"- Max |est C1 weekly − C1 biweekly|: **{ident['max_abs_est_C1w_vs_C1b']:.3e}**.",
        f"- Max |SE C1 weekly − C1 biweekly|: **{ident['max_abs_se_C1w_vs_C1b']:.3e}**.",
        f"- Holm disagreements A vs C1 weekly: **{ident['holm_disagree_A_vs_C1']}**.",
        f"- Holm disagreements C1 weekly vs biweekly: **{ident['holm_disagree_C1w_vs_C1b']}**.",
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
        "- This locked stream is still a synthetic diagnostic on the small complete-data matrix.",
        "- It is not design qualification and not an operational recommendation.",
        "- Do not switch production defaults.",
        "- Do not claim prior-outcome MAR validity.",
        "- Deferred cells remain deferred: covariance shift, heavy tails, delayed alternatives, 5% effect, control-heavy, twice-weekly, MAR, C2.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bundle(fits: pd.DataFrame, ident: dict, wall_s: float, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cells = cell_table(fits)
    reviews = gate_review(cells)
    comparison = compare_reviews(reviews, EXPLORATORY_REVIEW)
    n_fits = int(len(fits))
    n_studies = int(N_REP * len(SCENARIOS) * len(SIZES))
    payload = {
        "stream_id": "c1-locked-20261201",
        "status": "locked_diagnostic_completed_not_qualification",
        "labels": STATUS,
        "master_seed": MASTER_SEED,
        "n_replicates": N_REP,
        "n_study_datasets": n_studies,
        "n_analysis_fits": n_fits,
        "wall_seconds": wall_s,
        "c1_retuned": False,
        "independent_of_exploratory_seed": True,
        "working_covariance": "unstressed HierarchicalCovariance.animal_cov (historical plug-in stand-in)",
        "dgp": "HierarchicalCovariance.sample plus specified mean path; complete data; CRN visit subsets",
        "analyses": ["A_endpoint_ancova", "B_rm_gls_quadratic weekly/biweekly", "C1_mmrm_visit_contrast weekly/biweekly"],
        "scenarios": [s["name"] for s in SCENARIOS],
        "sizes": [list(s) for s in SIZES],
        "gates": GATES,
        "complete_data_identities": ident,
        "gate_review": reviews,
        "exploratory_comparison": comparison,
        "forbidden_use": [
            "Not design qualification.",
            "Do not switch production defaults.",
            "Do not claim prior-outcome MAR validity.",
            "C1 remains a candidate.",
        ],
    }
    cells.to_csv(out / "cell_oc.csv", index=False)
    (out / "gate_review.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(out / "REPORT.md", wall_s, n_fits, n_studies, cells, reviews, ident, comparison)
    names = ("cell_oc.csv", "gate_review.json", "REPORT.md")
    manifest = {"bundle_dir": str(out.relative_to(ROOT)), "labels": STATUS, "files": {}}
    for name in names:
        p = out / name
        manifest["files"][name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "n_analysis_fits": n_fits, "wall_seconds": wall_s,
            "gate_review": reviews, "complete_data_identities": ident,
            "exploratory_comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 replicates in memory; do not write the locked bundle.")
    args = parser.parse_args()
    n_rep = 2 if args.smoke else N_REP
    t0 = time.perf_counter()
    fits, ident = simulate(n_rep, MASTER_SEED)
    wall_s = time.perf_counter() - t0
    expected = n_rep * len(SCENARIOS) * len(SIZES) * 5
    if len(fits) != expected:
        raise SystemExit(f"fit count {len(fits)} != {expected}")
    if args.smoke:
        cells = cell_table(fits)
        print(json.dumps({
            "smoke": True,
            "n_fits": int(len(fits)),
            "n_cells": int(len(cells)),
            "wall_seconds": wall_s,
            "failure_rate": float(fits["failed"].mean()),
            "identities": ident,
        }, indent=2))
        return
    payload = write_bundle(fits, ident, wall_s, OUT)
    print(json.dumps({
        "bundle_dir": payload["bundle_dir"],
        "n_analysis_fits": payload["n_analysis_fits"],
        "wall_seconds": payload["wall_seconds"],
        "files": payload["files"],
        "complete_data_identities": payload["complete_data_identities"],
        "exploratory_comparison": payload["exploratory_comparison"],
        "gate_review": payload["gate_review"],
    }, indent=2))


if __name__ == "__main__":
    main()

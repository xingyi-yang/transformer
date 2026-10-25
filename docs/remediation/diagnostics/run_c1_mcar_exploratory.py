"""Exploratory MCAR / design-dependent missingness stream for A / B / C1.

Same small matrix as the complete-data streams. Seed 20270101. B=600.
Missingness matches paths.simulate_batch: independent missed visits and
dropout, independent of Z. That is not prior-outcome MAR.

C1 is not retuned. Not design qualification and not a default-switch.
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

OUT = ROOT / "docs/remediation/mcar_exploratory_20270101_c1"
LOCKED_REVIEW = ROOT / "docs/remediation/locked_20261201_c1/gate_review.json"
N_REP = 600
MASTER_SEED = 20270101
P_MISSED_VISIT = 0.02
P_DROPOUT = 0.02
ALPHA = 0.05
CI_LEVEL = 0.95
STATUS = [
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
    "Exploratory MCAR / design-dependent missingness. Not prior-outcome MAR. Not a default-switch.",
]


def mcar_mask(times: np.ndarray, n_animals: int, rng: np.random.Generator,
              horizon: float, p_miss: float, p_drop: float) -> np.ndarray:
    """Observed-visit mask. Independent of outcomes and of zb/sex/group.

    Matches send_bw.simulation.paths.simulate_batch: each visit is missed
    independently, and dropout hides visits after a Uniform(0, T) day.
    """
    times = np.asarray(times, float)
    mask = rng.random((n_animals, len(times))) >= p_miss
    drop = rng.random(n_animals) < p_drop
    dday = rng.uniform(0.0, horizon, size=n_animals)
    return mask & ~(drop[:, None] & (times[None, :] > dday[:, None]))


def simulate(n_rep: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    V0_w = BASE.animal_cov(WEEKLY)
    V0_b = BASE.animal_cov(WEEKLY[BIWEEKLY_IDX])
    rows: list[dict] = []
    n_end_obs = 0
    n_end_slots = 0
    n_visit_obs = 0
    n_visit_slots = 0
    audit = {
        "n_complete_pairs": 0,
        "n_mcar_pairs": 0,
        "max_abs_est_A_vs_C1_complete": 0.0,
        "max_abs_est_A_vs_C1_mcar": 0.0,
        "max_abs_est_C1w_vs_C1b_mcar": 0.0,
        "max_abs_se_C1w_vs_C1b_mcar": 0.0,
        "holm_disagree_A_vs_C1_mcar": 0,
        "holm_disagree_C1w_vs_C1b_mcar": 0,
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
                a_c = endpoint_ancova(zb, sex, group, Z[:, -1], 2)
                c_c = mmrm_visit_contrast(zb, sex, group, Z, WEEKLY, HORIZON, V0_w, 2)
                if not (a_c.failed or c_c.failed):
                    audit["n_complete_pairs"] += 1
                    audit["max_abs_est_A_vs_C1_complete"] = max(
                        audit["max_abs_est_A_vs_C1_complete"],
                        float(np.max(np.abs(a_c.est - c_c.est))),
                    )
                mask = mcar_mask(WEEKLY, len(zb), rng, HORIZON, P_MISSED_VISIT, P_DROPOUT)
                n_end_obs += int(mask[:, -1].sum())
                n_end_slots += int(mask.shape[0])
                n_visit_obs += int(mask.sum())
                n_visit_slots += int(mask.size)
                Zm = np.where(mask, Z, np.nan)
                Zb = Zm[:, BIWEEKLY_IDX]
                jobs = [
                    ("A_endpoint_ancova", "endpoint_only", endpoint_ancova,
                     (zb, sex, group, Zm[:, -1], 2)),
                    ("B_rm_gls_quadratic", "weekly", rm_gls,
                     (zb, sex, group, Zm, WEEKLY, HORIZON, V0_w, 2)),
                    ("B_rm_gls_quadratic", "biweekly", rm_gls,
                     (zb, sex, group, Zb, WEEKLY[BIWEEKLY_IDX], HORIZON, V0_b, 2)),
                    ("C1_mmrm_visit_contrast", "weekly", mmrm_visit_contrast,
                     (zb, sex, group, Zm, WEEKLY, HORIZON, V0_w, 2)),
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
                        "missingness": "mcar_design_dependent",
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
                    audit["n_mcar_pairs"] += 1
                    audit["max_abs_est_A_vs_C1_mcar"] = max(
                        audit["max_abs_est_A_vs_C1_mcar"], float(np.max(np.abs(a[1] - cw[1])))
                    )
                    audit["max_abs_est_C1w_vs_C1b_mcar"] = max(
                        audit["max_abs_est_C1w_vs_C1b_mcar"], float(np.max(np.abs(cw[1] - cb[1])))
                    )
                    audit["max_abs_se_C1w_vs_C1b_mcar"] = max(
                        audit["max_abs_se_C1w_vs_C1b_mcar"], float(np.max(np.abs(cw[2] - cb[2])))
                    )
                    audit["holm_disagree_A_vs_C1_mcar"] += int(np.any(a[3] != cw[3]))
                    audit["holm_disagree_C1w_vs_C1b_mcar"] += int(np.any(cw[3] != cb[3]))
    audit["p_missed_visit"] = P_MISSED_VISIT
    audit["p_dropout"] = P_DROPOUT
    audit["observed_endpoint_fraction"] = n_end_obs / n_end_slots if n_end_slots else None
    audit["observed_visit_fraction"] = n_visit_obs / n_visit_slots if n_visit_slots else None
    audit["missingness_note"] = (
        "Independent of Z and of zb/sex/group. Matches paths.simulate_batch. "
        "Not prior-outcome MAR."
    )
    return pd.DataFrame(rows), audit


def compare_to_complete_locked(mcar: list[dict], locked_path: Path) -> list[dict]:
    if not locked_path.is_file():
        return []
    prev = {(r["analysis"], r["schedule"], r["n_per_sex"]): r
            for r in json.loads(locked_path.read_text())["gate_review"]}
    out = []
    for r in mcar:
        e = prev.get((r["analysis"], r["schedule"], r["n_per_sex"]))
        if e is None:
            continue
        out.append({
            "analysis": r["analysis"],
            "schedule": r["schedule"],
            "n_per_sex": r["n_per_sex"],
            "complete_locked_inference_tolerances_met": e["inference_tolerances_met"],
            "mcar_inference_tolerances_met": r["inference_tolerances_met"],
            "same_inference_decision": e["inference_tolerances_met"] == r["inference_tolerances_met"],
            "complete_locked_fwer_max_point": e["fwer_max_point"],
            "mcar_fwer_max_point": r["fwer_max_point"],
            "complete_locked_coverage_min": e["coverage_min"],
            "mcar_coverage_min": r["coverage_min"],
            "note": "Different seed from locked complete data. Qualitative pass/fail only.",
        })
    return out


def write_report(path: Path, wall_s: float, n_fits: int, n_studies: int,
                 cells: pd.DataFrame, reviews: list[dict], audit: dict,
                 comparison: list[dict]) -> None:
    lines = [
        "# C1 exploratory MCAR / design-dependent missingness",
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
        f"- Missingness: `p_missed_visit={P_MISSED_VISIT}`, `p_dropout={P_DROPOUT}` from "
        "`configs/new_study.example.yaml`, implemented as in `paths.simulate_batch`.",
        f"- Observed endpoint fraction: **{audit['observed_endpoint_fraction']:.4f}**. "
        f"Observed visit fraction: **{audit['observed_visit_fraction']:.4f}**.",
        "- Absence is independent of Z and of zb/sex/group. This is **not** prior-outcome MAR.",
        "- C1 was not retuned. Production defaults remain `[ancova, gls]`.",
        "- Failed analyses count as non-rejections and remain in the FWER/power denominator.",
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
        "## Candidate review (exploratory MCAR, not locked)",
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
            "## Qualitative comparison to locked complete-data decisions",
            "",
            "Different seed. Use only to see whether a pass/fail flipped under MCAR.",
            "",
            "| Analysis | Schedule | n/sex | Complete locked | MCAR exploratory | Same decision |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for c in comparison:
            lines.append(
                f"| `{c['analysis']}` | {c['schedule']} | {c['n_per_sex']} | "
                f"{'met' if c['complete_locked_inference_tolerances_met'] else 'missed'} | "
                f"{'met' if c['mcar_inference_tolerances_met'] else 'missed'} | "
                f"{'yes' if c['same_inference_decision'] else 'NO'} |"
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
        f"- Max |SE C1 weekly − C1 biweekly| under MCAR: **{audit['max_abs_se_C1w_vs_C1b_mcar']:.3e}**.",
        f"- Holm disagreements A vs C1 weekly under MCAR: **{audit['holm_disagree_A_vs_C1_mcar']}**.",
        f"- Holm disagreements C1 weekly vs biweekly under MCAR: **{audit['holm_disagree_C1w_vs_C1b_mcar']}**.",
        "",
        "On complete data the endpoint coefficients match. Under MCAR they need not. "
        "That difference is not by itself a reason to switch defaults.",
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
        "- Not prior-outcome MAR. Do not describe these draws as outcome-dependent dropout.",
        "- Not a locked confirmation (that would need a new independent seed and a larger B).",
        "- Do not retune C1 on these draws.",
        "- Do not switch production defaults.",
        "- Deferred: covariance shift, heavy tails, delayed alternatives, 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bundle(fits: pd.DataFrame, audit: dict, wall_s: float, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cells = cell_table(fits)
    cells["missingness"] = "mcar_design_dependent"
    reviews = gate_review(cells)
    comparison = compare_to_complete_locked(reviews, LOCKED_REVIEW)
    n_fits = int(len(fits))
    n_studies = int(N_REP * len(SCENARIOS) * len(SIZES))
    payload = {
        "stream_id": "c1-mcar-exploratory-20270101",
        "status": "mcar_exploratory_completed_not_locked_not_qualification",
        "labels": STATUS,
        "master_seed": MASTER_SEED,
        "n_replicates": N_REP,
        "n_study_datasets": n_studies,
        "n_analysis_fits": n_fits,
        "wall_seconds": wall_s,
        "c1_retuned": False,
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
        "complete_locked_comparison": comparison,
        "forbidden_use": [
            "Not prior-outcome MAR.",
            "Not a locked evaluation.",
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
            "audit": audit, "gate_review": reviews, "complete_locked_comparison": comparison}


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
        cells = cell_table(fits)
        print(json.dumps({
            "smoke": True,
            "n_fits": int(len(fits)),
            "n_cells": int(len(cells)),
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
        "complete_locked_comparison": payload["complete_locked_comparison"],
        "gate_review": payload["gate_review"],
    }, indent=2))


if __name__ == "__main__":
    main()

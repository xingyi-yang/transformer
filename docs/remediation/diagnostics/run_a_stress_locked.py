"""Locked B=2000 A-default stress diagnostic.

Same frozen complete-data matrix as exploratory_20270401_a_stress.
Independent seed 20270501. A is the subject. C1 is not retuned.
Not design qualification. Do not restore GLS as a default.
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

from run_a_stress_exploratory import (  # noqa: E402
    GATES,
    SCENARIOS,
    SIZES,
    _check_frozen_matrix,
    _finite,
    _fmt_rate,
    _interp,
    _miss_lines,
    cell_table,
    gate_review,
    simulate,
)

OUT = ROOT / "docs/remediation/locked_20270501_a_stress"
EXPLORATORY_REVIEW = ROOT / "docs/remediation/exploratory_20270401_a_stress/gate_review.json"
N_REP = 2000
MASTER_SEED = 20270501
STATUS = [
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
    "Locked B=2000 A-default stress. Independent of exploratory draws. Not a default-switch.",
]


def identities(fits: pd.DataFrame) -> dict:
    keys = ["scenario", "n_per_sex", "replicate"]
    a = fits[(fits["analysis"] == "A_endpoint_ancova") & (fits["schedule"] == "endpoint_only")].set_index(keys)
    cw = fits[(fits["analysis"] == "C1_mmrm_visit_contrast") & (fits["schedule"] == "weekly")].set_index(keys)
    cb = fits[(fits["analysis"] == "C1_mmrm_visit_contrast") & (fits["schedule"] == "biweekly")].set_index(keys)
    common = a.index.intersection(cw.index).intersection(cb.index)
    ok = ~(a.loc[common, "failed"] | cw.loc[common, "failed"] | cb.loc[common, "failed"]).to_numpy(bool)
    ea = a.loc[common, ["est_low", "est_high"]].to_numpy(float)[ok]
    ecw = cw.loc[common, ["est_low", "est_high"]].to_numpy(float)[ok]
    ecb = cb.loc[common, ["est_low", "est_high"]].to_numpy(float)[ok]
    scw = cw.loc[common, ["se_low", "se_high"]].to_numpy(float)[ok]
    scb = cb.loc[common, ["se_low", "se_high"]].to_numpy(float)[ok]
    ra = a.loc[common, ["rej_low", "rej_high"]].to_numpy(bool)[ok]
    rcw = cw.loc[common, ["rej_low", "rej_high"]].to_numpy(bool)[ok]
    rcb = cb.loc[common, ["rej_low", "rej_high"]].to_numpy(bool)[ok]
    return {
        "n_pairs": int(ok.sum()),
        "max_abs_est_A_vs_C1": float(np.max(np.abs(ea - ecw))) if len(ea) else 0.0,
        "max_abs_est_C1w_vs_C1b": float(np.max(np.abs(ecw - ecb))) if len(ecw) else 0.0,
        "max_abs_se_C1w_vs_C1b": float(np.max(np.abs(scw - scb))) if len(scw) else 0.0,
        "holm_disagree_A_vs_C1": int(np.any(ra != rcw, axis=1).sum()) if len(ra) else 0,
        "holm_disagree_C1w_vs_C1b": int(np.any(rcw != rcb, axis=1).sum()) if len(rcw) else 0,
    }


def compare_reviews(locked: list[dict], exploratory_path: Path) -> list[dict]:
    if not exploratory_path.is_file():
        return []
    prev = {
        (r["analysis"], r["schedule"], r["mechanism"], r["n_per_sex"]): r
        for r in json.loads(exploratory_path.read_text())["gate_review"]
    }
    out = []
    for r in locked:
        key = (r["analysis"], r["schedule"], r["mechanism"], r["n_per_sex"])
        e = prev.get(key)
        if e is None:
            continue
        out.append({
            "analysis": r["analysis"],
            "schedule": r["schedule"],
            "mechanism": r["mechanism"],
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
    n_same = sum(c["same_inference_decision"] for c in comparison)
    n_comp = len(comparison)
    a_flips = [
        c for c in comparison
        if c["analysis"] == "A_endpoint_ancova" and not c["same_inference_decision"]
    ]
    lines = [
        "# A-default stress locked B=2000",
        "",
        "## Status",
        "",
    ]
    lines += [f"- **{x}**" for x in STATUS]
    lines += [
        "",
        f"- Seed `{MASTER_SEED}`; {N_REP} replicates; complete data; 11 scenarios × 2 sizes × 5 analyses.",
        "- Frozen matrix from `timing_pilot_20270301_a_stress` / `exploratory_20270401_a_stress`.",
        "- Independent of exploratory seed `20270401`. C1 was not retuned.",
        "- Mechanisms: covariance shift (`sigma_p` ×1.6, `ell` ×2), heavy tails (t(3) measurement ×1.5), delayed alternatives.",
        "- A is the subject. B and C1 are comparators. Production defaults stay `[ancova]`.",
        f"- Fits recorded: **{n_fits}**. Study datasets: **{n_studies}**.",
        f"- Wall clock: **{wall_s:.3f} s**.",
        "- Common random numbers: one weekly latent path per (scenario, size, replicate); biweekly is a visit subset.",
        "- Working covariance: unstressed `HierarchicalCovariance.animal_cov` (historical plug-in stand-in).",
        "- Failed analyses count as non-rejections and remain in the FWER/power denominator.",
        "- Delayed FWER uses partial-null cells only.",
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
        "## Review by mechanism (locked diagnostic, not qualification)",
        "",
        "| Analysis | Mechanism | Schedule | n/sex | FWER max [CI hi] | Coverage min | Alt bias frac | Fail max | Alt power-all [CI lo] | Inference | Power if chosen |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in reviews:
        lines.append(
            f"| `{r['analysis']}` | {r['mechanism']} | {r['schedule']} | {r['n_per_sex']} | "
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
        "“Inference” is FWER + coverage + alternative bias + failures within that mechanism. "
        "It is not a primary-analysis decision and not design qualification.",
        "",
        "## Cells that miss a sourced inference tolerance",
        "",
    ]
    lines += _miss_lines(reviews) or ["- None of the mechanism×analysis×schedule×size reviews missed FWER, coverage, bias, or failure tolerances."]
    if comparison:
        lines += [
            "",
            "## Agreement with exploratory B=600 inference decisions",
            "",
            f"Same pre-specified analyses and matrix; independent seed. **{n_same}/{n_comp}** decisions match. "
            "A changed decision is reported, not patched.",
            "",
            "| Analysis | Mechanism | Schedule | n/sex | Exploratory | Locked | Same decision | FWER expl → locked |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in comparison:
            lines.append(
                f"| `{c['analysis']}` | {c['mechanism']} | {c['schedule']} | {c['n_per_sex']} | "
                f"{'met' if c['exploratory_inference_tolerances_met'] else 'missed'} | "
                f"{'met' if c['locked_inference_tolerances_met'] else 'missed'} | "
                f"{'yes' if c['same_inference_decision'] else 'NO'} | "
                f"{c['exploratory_fwer_max_point']:.3f} → {c['locked_fwer_max_point']:.3f} |"
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
        "| Analysis | Mechanism | Schedule | Scenario | n/sex | FWER | Coverage min | Bias max | Power-all | Fail |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    show = cells.sort_values(["analysis", "mechanism", "schedule", "n_per_sex", "scenario"])
    for r in show.itertuples(index=False):
        fwer = _fmt_rate(getattr(r, "fwer", None), getattr(r, "fwer_ci_lo", None), getattr(r, "fwer_ci_hi", None))
        power = _fmt_rate(getattr(r, "power_all_required", None),
                          getattr(r, "power_all_required_ci_lo", None),
                          getattr(r, "power_all_required_ci_hi", None))
        lines.append(
            f"| `{r.analysis}` | {r.mechanism} | {r.schedule} | {r.scenario} | {r.n_per_sex} | "
            f"{fwer} | {_fmt_rate(r.coverage_min)} | {_fmt_rate(r.bias_max_abs)} | "
            f"{power} | {_fmt_rate(r.failure_rate)} |"
        )
    lines += [
        "",
        "## Interpretation (locked diagnostic only)",
        "",
    ]
    lines += _interp(reviews)
    if a_flips:
        lines.append("- **A decision flipped versus exploratory (reported, not patched):**")
        for c in a_flips:
            lines.append(
                f"  - {c['mechanism']} {c['n_per_sex']}: "
                f"{'met' if c['exploratory_inference_tolerances_met'] else 'missed'} → "
                f"{'met' if c['locked_inference_tolerances_met'] else 'missed'}"
            )
    else:
        lines.append("- No A inference decision flipped versus exploratory B=600.")
    lines += [
        "- B agreeing with A on these three stresses does not restore GLS (close-out was slope heterogeneity).",
        "- C1 misses or flips are not a retune and not a promotion.",
        "",
        "## Constraints on use",
        "",
        "- This locked stream is still a synthetic diagnostic on the deferred-mechanism complete-data matrix.",
        "- It is not design qualification and not an operational recommendation.",
        "- Do not change the ANCOVA default from this stream.",
        "- Do not restore GLS as a production default.",
        "- Do not retune C1.",
        "- Do not claim prior-outcome MAR validity.",
        "- Still deferred: 5% effect, control-heavy, twice-weekly, prior-outcome MAR, C2, MCAR on these stresses.",
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
    a_reviews = [r for r in reviews if r["analysis"] == "A_endpoint_ancova"]
    payload = {
        "stream_id": "a-stress-locked-20270501",
        "status": "locked_diagnostic_completed_not_qualification",
        "labels": STATUS,
        "master_seed": MASTER_SEED,
        "n_replicates": N_REP,
        "n_study_datasets": n_studies,
        "n_analysis_fits": n_fits,
        "wall_seconds": wall_s,
        "c1_retuned": False,
        "independent_of_exploratory_seed": True,
        "subject": "A_endpoint_ancova",
        "comparators": ["B_rm_gls_quadratic", "C1_mmrm_visit_contrast"],
        "working_covariance": "unstressed HierarchicalCovariance.animal_cov (historical plug-in stand-in)",
        "dgp": (
            "Frozen A-default stress matrix. covariance_shift: sigma_p x1.6, ell x2. "
            "heavy_tails: t(3) measurement at sigma_m x1.5. delayed: fixed_shape delayed. "
            "Complete data; CRN visit subsets."
        ),
        "analyses": [
            "A_endpoint_ancova",
            "B_rm_gls_quadratic weekly/biweekly",
            "C1_mmrm_visit_contrast weekly/biweekly",
        ],
        "mechanisms": ["covariance_shift", "heavy_tails", "delayed"],
        "scenarios": [s["name"] for s in SCENARIOS],
        "sizes": [list(s) for s in SIZES],
        "gates": GATES,
        "subject_inference_met": all(r["inference_tolerances_met"] for r in a_reviews),
        "subject_inference_n": len(a_reviews),
        "complete_data_identities": ident,
        "gate_review": reviews,
        "exploratory_comparison": comparison,
        "forbidden_use": [
            "Not design qualification.",
            "Do not change production defaults.",
            "Do not restore GLS as a default.",
            "Do not retune C1.",
            "Do not claim prior-outcome MAR validity.",
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
    return {
        **manifest,
        "n_analysis_fits": n_fits,
        "wall_seconds": wall_s,
        "gate_review": reviews,
        "complete_data_identities": ident,
        "exploratory_comparison": comparison,
    }


def main() -> None:
    _check_frozen_matrix()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 replicates in memory; do not write the locked bundle.")
    args = parser.parse_args()
    n_rep = 2 if args.smoke else N_REP
    t0 = time.perf_counter()
    fits = simulate(n_rep, MASTER_SEED)
    wall_s = time.perf_counter() - t0
    expected = n_rep * len(SCENARIOS) * len(SIZES) * 5
    if len(fits) != expected:
        raise SystemExit(f"fit count {len(fits)} != {expected}")
    ident = identities(fits)
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
    flips = [c for c in payload["exploratory_comparison"] if not c["same_inference_decision"]]
    print(json.dumps({
        "bundle_dir": payload["bundle_dir"],
        "n_analysis_fits": payload["n_analysis_fits"],
        "wall_seconds": payload["wall_seconds"],
        "files": payload["files"],
        "complete_data_identities": payload["complete_data_identities"],
        "subject_inference": [
            {k: r[k] for k in (
                "mechanism", "n_per_sex", "fwer_max_point", "coverage_min",
                "bias_frac_alternative", "inference_tolerances_met",
            )}
            for r in payload["gate_review"] if r["analysis"] == "A_endpoint_ancova"
        ],
        "flips": [
            f"{c['analysis']} {c['mechanism']} {c['schedule']} {c['n_per_sex']}"
            for c in flips
        ],
        "misses": [
            f"{r['analysis']} {r['mechanism']} {r['schedule']} {r['n_per_sex']}"
            for r in payload["gate_review"] if not r["inference_tolerances_met"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

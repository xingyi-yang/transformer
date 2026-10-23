"""Exploratory B=600 operating-characteristic stream for A / B / C1.

Small diagnostic matrix only. Complete data. Seed 20261101.
Not a locked evaluation, not design qualification, not a default-switch.
C2, research.yaml, and B=2000 are not run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from send_bw.models.covariance import HierarchicalCovariance
from send_bw.simulation.analysis import (
    confidence_interval,
    endpoint_ancova,
    holm,
    mmrm_visit_contrast,
    rm_gls,
)
from send_bw.simulation.metrics import prop_summary
from send_bw.simulation.paths import fixed_shape

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/remediation/exploratory_20261101_c1"
HORIZON = 28.0
WEEKLY = np.array([7.0, 14.0, 21.0, 28.0])
BIWEEKLY_IDX = np.array([1, 3])
ALPHA = 0.05
CI_LEVEL = 0.95
STATUS = [
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
    "Exploratory B=600. Not a locked evaluation.",
]
SCENARIOS = [
    {"name": "global_null", "deltas": (0.0, 0.0), "shape": "linear", "animal_slope": 1.0, "study_slope": 1.0},
    {"name": "partial_null_low", "deltas": (0.0, -0.15), "shape": "linear", "animal_slope": 1.0, "study_slope": 1.0},
    {"name": "partial_null_high", "deltas": (-0.10, 0.0), "shape": "linear", "animal_slope": 1.0, "study_slope": 1.0},
    {"name": "heterogeneity_global_null", "deltas": (0.0, 0.0), "shape": "linear",
     "animal_slope": 1.6, "study_slope": 1.6},
    {"name": "heterogeneity_partial_null_low", "deltas": (0.0, -0.15), "shape": "linear",
     "animal_slope": 1.6, "study_slope": 1.6},
    {"name": "alternative_transient", "deltas": (-0.10, -0.15), "shape": "transient",
     "animal_slope": 1.0, "study_slope": 1.0},
    {"name": "partial_null_low_transient", "deltas": (0.0, -0.15), "shape": "transient",
     "animal_slope": 1.0, "study_slope": 1.0},
    {"name": "partial_null_high_transient", "deltas": (-0.10, 0.0), "shape": "transient",
     "animal_slope": 1.0, "study_slope": 1.0},
]
SIZES = [(6, 6, 6), (8, 8, 8)]
ARM_NAMES = ("low", "high")
N_REP = 600
MASTER_SEED = 20261101
BASE = HierarchicalCovariance()
GATES = {
    "nominal_alpha": 0.05,
    "fwer_max": 0.05,
    "fwer_tolerance": 0.015,
    "fwer_point_ceiling": 0.065,
    "fwer_upper_tolerance": 0.045,
    "fwer_ci_ceiling": 0.095,
    "coverage_min": 0.92,
    "max_abs_bias_fraction": 0.15,
    "max_failure_rate": 0.02,
    "power_target": 0.80,
    "power_lower_tolerance": 0.06,
    "power_ci_floor": 0.74,
    "source": "configs/new_study.example.yaml plus optimize.py::feasibility",
}
ALT_SCENARIO = "alternative_transient"
ALT_DELTA_SCALE = 0.15


def animals(n_per_sex: tuple[int, int, int], rng: np.random.Generator):
    groups, sexes, zbs = [], [], []
    for g, n in enumerate(n_per_sex):
        for s in (0, 1):
            groups.append(np.full(n, g, dtype=int))
            sexes.append(np.full(n, s, dtype=int))
            zbs.append(rng.normal(5.27 + 0.29 * s, 0.08, size=n))
    return np.concatenate(zbs), np.concatenate(sexes), np.concatenate(groups)


def mean_path(zb, sex, group, times, deltas, shape):
    u = times / HORIZON
    zb_c = zb - zb.mean()
    Z = 0.18 * u[None] - 0.03 * (u**2)[None] + (sex == 1)[:, None] * (0.015 + 0.025 * u[None])
    Z = Z + zb_c[:, None] * (1.0 - 0.05 * u[None])
    s = fixed_shape(shape, times, HORIZON)
    for g, d in enumerate(deltas, start=1):
        Z = Z + (group == g)[:, None] * (d * s[None])
    return Z


def truth_cov(animal_slope: float, study_slope: float) -> HierarchicalCovariance:
    return HierarchicalCovariance(
        sd_s0=BASE.sd_s0, sd_s1=BASE.sd_s1 * study_slope, sd_a0=BASE.sd_a0,
        sd_a1=BASE.sd_a1 * animal_slope, sigma_p=BASE.sigma_p, ell=BASE.ell, sigma_m=BASE.sigma_m,
    )


def _py(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _prop_fields(prefix: str, x: np.ndarray) -> dict:
    ps = prop_summary(x)
    return {f"{prefix}": _py(ps["est"]), f"{prefix}_k": _py(ps["k"]), f"{prefix}_n": _py(ps["n"]),
            f"{prefix}_mcse": _py(ps["mcse"]), f"{prefix}_ci_lo": _py(ps["ci_lo"]),
            f"{prefix}_ci_hi": _py(ps["ci_hi"])}


def summarize_cell(g: pd.DataFrame, deltas: tuple[float, float]) -> dict:
    truth = np.asarray(deltas, float)
    failed = g["failed"].to_numpy(bool)
    rej = g[["rej_low", "rej_high"]].to_numpy(bool)
    est = g[["est_low", "est_high"]].to_numpy(float)
    lo = g[["lo_low", "lo_high"]].to_numpy(float)
    hi = g[["hi_low", "hi_high"]].to_numpy(float)
    se = g[["se_low", "se_high"]].to_numpy(float)
    out = {
        "analysis": g["analysis"].iloc[0],
        "schedule": g["schedule"].iloc[0],
        "scenario": g["scenario"].iloc[0],
        "n_per_sex": g["n_per_sex"].iloc[0],
        "n_animals": int(g["n_animals"].iloc[0]),
        "missingness": "complete",
        "B": int(len(g)),
        "true_delta_low": float(truth[0]),
        "true_delta_high": float(truth[1]),
    }
    out.update(_prop_fields("failure_rate", failed))
    reasons = g.loc[failed, "reason"].value_counts().to_dict()
    out["failure_reasons"] = ";".join(f"{k}:{v}" for k, v in reasons.items())
    null = np.abs(truth) == 0.0
    alt = ~null
    if null.any():
        out.update(_prop_fields("fwer", rej[:, null].any(1)))
    if alt.any():
        out.update(_prop_fields("power_any", rej[:, alt].any(1)))
        out.update(_prop_fields("power_all_required", rej[:, alt].all(1)))
    for j, name in enumerate(ARM_NAMES):
        key = "type1" if null[j] else "power"
        out.update(_prop_fields(f"{key}_{name}", rej[:, j]))
    ok = ~failed
    n_ok = int(ok.sum())
    out["n_ok"] = n_ok
    if n_ok > 1:
        err = est[ok] - truth
        cover = (lo[ok] <= truth) & (truth <= hi[ok])
        out["bias_low"] = _py(np.nanmean(err[:, 0]))
        out["bias_high"] = _py(np.nanmean(err[:, 1]))
        out["bias_max_abs"] = _py(np.nanmax(np.abs(np.nanmean(err, 0))))
        out["bias_mcse_max"] = _py(np.nanmax(np.nanstd(err, 0) / np.sqrt(n_ok)))
        scale = float(np.max(np.abs(truth)))
        out["bias_frac"] = _py(out["bias_max_abs"] / scale) if scale > 0 else None
        for j, name in enumerate(ARM_NAMES):
            out.update(_prop_fields(f"coverage_{name}", cover[:, j]))
        out["coverage_min"] = _py(np.min(cover.mean(0)))
        out["ci_halfwidth_mean"] = _py(np.nanmean((hi[ok] - lo[ok]) / 2))
        emp = np.nanstd(est[ok], 0, ddof=1)
        se_mean = np.nanmean(se[ok], 0)
        out["emp_se_low"] = _py(emp[0])
        out["emp_se_high"] = _py(emp[1])
        out["se_mean_low"] = _py(se_mean[0])
        out["se_mean_high"] = _py(se_mean[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = se_mean / emp
        out["se_ratio_low"] = _py(ratio[0])
        out["se_ratio_high"] = _py(ratio[1])
    return out


def simulate(n_rep: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    V0_w = BASE.animal_cov(WEEKLY)
    V0_b = BASE.animal_cov(WEEKLY[BIWEEKLY_IDX])
    rows: list[dict] = []
    fit_idx = 0
    for scen in SCENARIOS:
        dgp = truth_cov(scen["animal_slope"], scen["study_slope"])
        deltas = scen["deltas"]
        for n_per in SIZES:
            n_label = "/".join(map(str, n_per))
            n_animals = int(sum(n_per) * 2)
            for rep in range(n_rep):
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
                for analysis, schedule, fn, args in jobs:
                    res = fn(*args)
                    failed = bool(res.failed)
                    if failed:
                        est = se = p = lo = hi = np.full(2, np.nan)
                        rej = np.zeros(2, dtype=bool)
                        df = float("nan")
                    else:
                        est, se, p, df = res.est, res.se, res.p, float(res.df)
                        lo, hi = confidence_interval(res, CI_LEVEL)
                        rej = holm(p, ALPHA)
                    rows.append({
                        "fit_idx": fit_idx,
                        "scenario": scen["name"],
                        "true_delta_low": float(deltas[0]),
                        "true_delta_high": float(deltas[1]),
                        "n_per_sex": n_label,
                        "n_animals": n_animals,
                        "replicate": rep,
                        "analysis": analysis,
                        "schedule": schedule,
                        "missingness": "complete",
                        "failed": failed,
                        "reason": res.reason if failed else "",
                        "df": df,
                        "est_low": _py(est[0]),
                        "est_high": _py(est[1]),
                        "se_low": _py(se[0]),
                        "se_high": _py(se[1]),
                        "p_low": _py(p[0]),
                        "p_high": _py(p[1]),
                        "lo_low": _py(lo[0]),
                        "hi_low": _py(hi[0]),
                        "lo_high": _py(lo[1]),
                        "hi_high": _py(hi[1]),
                        "rej_low": bool(rej[0]),
                        "rej_high": bool(rej[1]),
                    })
                    fit_idx += 1
    return pd.DataFrame(rows)


def cell_table(fits: pd.DataFrame) -> pd.DataFrame:
    scen_d = {s["name"]: s["deltas"] for s in SCENARIOS}
    keys = ["analysis", "schedule", "scenario", "n_per_sex"]
    rows = []
    for key, g in fits.groupby(keys, sort=False):
        rows.append(summarize_cell(g, scen_d[key[2]]))
    return pd.DataFrame(rows)


def gate_review(cells: pd.DataFrame) -> list[dict]:
    reviews = []
    for (analysis, schedule, n_per), g in cells.groupby(["analysis", "schedule", "n_per_sex"], sort=False):
        nulls = g[g["scenario"].str.contains("null")]
        alt = g[g["scenario"] == ALT_SCENARIO]
        alt_row = alt.iloc[0] if len(alt) else None
        fwer_point = float(nulls["fwer"].max()) if "fwer" in nulls and nulls["fwer"].notna().any() else float("nan")
        fwer_ci = float(nulls["fwer_ci_hi"].max()) if "fwer_ci_hi" in nulls and nulls["fwer_ci_hi"].notna().any() else float("nan")
        worst_fwer = nulls.loc[nulls["fwer"].idxmax(), "scenario"] if nulls["fwer"].notna().any() else ""
        coverage_min = float(g["coverage_min"].min()) if g["coverage_min"].notna().any() else float("nan")
        worst_cov = g.loc[g["coverage_min"].idxmin(), "scenario"] if g["coverage_min"].notna().any() else ""
        fail_max = float(g["failure_rate"].max())
        bias_frac = _py(alt_row["bias_frac"]) if alt_row is not None else None
        bias_abs = _py(alt_row["bias_max_abs"]) if alt_row is not None else None
        power = _py(alt_row["power_all_required"]) if alt_row is not None else None
        power_lo = _py(alt_row["power_all_required_ci_lo"]) if alt_row is not None else None
        n6 = str(n_per).startswith("6/")
        pass_fwer = bool(fwer_point <= GATES["fwer_point_ceiling"] and fwer_ci <= GATES["fwer_ci_ceiling"])
        pass_coverage = bool(coverage_min >= GATES["coverage_min"])
        pass_bias = bool(bias_frac is not None and bias_frac <= GATES["max_abs_bias_fraction"])
        pass_failures = bool(fail_max <= GATES["max_failure_rate"])
        pass_power = bool(
            power is not None and power_lo is not None
            and power >= GATES["power_target"] and power_lo >= GATES["power_ci_floor"]
        )
        inference_ok = pass_fwer and pass_coverage and pass_bias and pass_failures
        reviews.append({
            "analysis": analysis,
            "schedule": schedule,
            "n_per_sex": n_per,
            "n_animals": int(g["n_animals"].iloc[0]),
            "fwer_max_point": _py(fwer_point),
            "fwer_max_ci_hi": _py(fwer_ci),
            "fwer_worst_scenario": worst_fwer,
            "coverage_min": _py(coverage_min),
            "coverage_worst_scenario": worst_cov,
            "bias_max_abs_alternative": bias_abs,
            "bias_frac_alternative": bias_frac,
            "failure_rate_max": _py(fail_max),
            "power_all_required_alternative": power,
            "power_all_required_ci_lo": power_lo,
            "pass_fwer_tolerance": pass_fwer,
            "pass_coverage": pass_coverage,
            "pass_bias": pass_bias,
            "pass_failures": pass_failures,
            "pass_power_if_chosen_design": pass_power,
            "power_gate_applies": not n6,
            "inference_tolerances_met": inference_ok,
            "all_sourced_gates_if_chosen_design": inference_ok and pass_power,
            "note": (
                "n=6 is not required to meet the power target."
                if n6 else
                "Power gate applies only if this size were a chosen design."
            ),
        })
    return reviews


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and not np.isfinite(x))


def _fmt_rate(p, lo=None, hi=None) -> str:
    if not _finite(p):
        return "—"
    if not _finite(lo) or not _finite(hi):
        return f"{p:.3f}"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def write_report(path: Path, wall_s: float, n_fits: int, n_studies: int,
                 cells: pd.DataFrame, reviews: list[dict]) -> None:
    lines = [
        "# C1 exploratory B=600 stream",
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
        "- Common random numbers: one weekly latent path per (scenario, size, replicate); biweekly is a visit subset.",
        "- Working covariance: unstressed `HierarchicalCovariance.animal_cov` (historical plug-in stand-in).",
        "- Failed analyses count as non-rejections and remain in the FWER/power denominator.",
        "- C1 remains a **candidate**. Production defaults remain `[ancova, gls]`.",
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
        "## Candidate review (exploratory, not locked)",
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
            + (f" [{r['power_all_required_ci_lo']:.3f}]" if _finite(r['power_all_required_ci_lo']) else "")
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
    lines += [
        "",
        "## Per-cell operating characteristics",
        "",
        "Full rows are in `cell_oc.csv`. Replicate-level fits are in `replicate_results.csv`.",
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
        "- Do not treat this stream as locked confirmation (that would be independent seed `20261201`, B=2000).",
        "- Do not retune C1 (df, sandwich multiplier, working V0) on these draws and then call a later stream confirmatory.",
        "- Do not switch production defaults.",
        "- Do not claim prior-outcome MAR validity. Missingness here is complete.",
        "- Deferred cells remain deferred: covariance shift, heavy tails, delayed alternatives, 5% effect, control-heavy, twice-weekly, MAR, C2.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bundle(fits: pd.DataFrame, wall_s: float, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cells = cell_table(fits)
    reviews = gate_review(cells)
    n_fits = int(len(fits))
    n_studies = int(N_REP * len(SCENARIOS) * len(SIZES))
    payload = {
        "stream_id": "c1-exploratory-20261101",
        "status": "exploratory_completed_not_locked_not_qualification",
        "labels": STATUS,
        "master_seed": MASTER_SEED,
        "n_replicates": N_REP,
        "n_study_datasets": n_studies,
        "n_analysis_fits": n_fits,
        "wall_seconds": wall_s,
        "working_covariance": "unstressed HierarchicalCovariance.animal_cov (historical plug-in stand-in)",
        "dgp": "HierarchicalCovariance.sample plus specified mean path; complete data; CRN visit subsets",
        "analyses": ["A_endpoint_ancova", "B_rm_gls_quadratic weekly/biweekly", "C1_mmrm_visit_contrast weekly/biweekly"],
        "scenarios": [s["name"] for s in SCENARIOS],
        "sizes": [list(s) for s in SIZES],
        "gates": GATES,
        "gate_review": reviews,
        "forbidden_use": [
            "Not a locked evaluation.",
            "Not design qualification.",
            "Do not switch production defaults.",
            "Do not retune C1 on these draws.",
            "Do not claim prior-outcome MAR validity.",
        ],
    }
    fits.to_csv(out / "replicate_results.csv", index=False)
    cells.to_csv(out / "cell_oc.csv", index=False)
    (out / "gate_review.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(out / "REPORT.md", wall_s, n_fits, n_studies, cells, reviews)
    names = ("replicate_results.csv", "cell_oc.csv", "gate_review.json", "REPORT.md")
    manifest = {"bundle_dir": str(out.relative_to(ROOT)), "labels": STATUS, "files": {}}
    for name in names:
        p = out / name
        manifest["files"][name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "n_analysis_fits": n_fits, "wall_seconds": wall_s, "gate_review": reviews}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 replicates in memory; do not write the exploratory bundle.")
    args = parser.parse_args()
    n_rep = 2 if args.smoke else N_REP
    seed = MASTER_SEED
    t0 = time.perf_counter()
    fits = simulate(n_rep, seed)
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
        }, indent=2))
        return
    payload = write_bundle(fits, wall_s, OUT)
    print(json.dumps({
        "bundle_dir": payload["bundle_dir"],
        "n_analysis_fits": payload["n_analysis_fits"],
        "wall_seconds": payload["wall_seconds"],
        "files": payload["files"],
        "gate_review": payload["gate_review"],
    }, indent=2))


if __name__ == "__main__":
    main()

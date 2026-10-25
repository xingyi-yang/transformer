"""Timing-only pilot for A / B / C1. Not an operating-characteristic evaluation.

20 replicates × 8 scenarios × 2 sizes × 5 analyses = 1,600 fits.
Complete data only. Results must not be used for FWER, coverage, bias, or power.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from send_bw.models.covariance import HierarchicalCovariance
from send_bw.simulation.analysis import endpoint_ancova, mmrm_visit_contrast, rm_gls
from send_bw.simulation.paths import fixed_shape

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/remediation/timing_pilot_20261022_c1"
HORIZON = 28.0
WEEKLY = np.array([7.0, 14.0, 21.0, 28.0])
BIWEEKLY_IDX = np.array([1, 3])
STATUS = [
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
    "Timing-only pilot. Not an operating-characteristic evaluation.",
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
N_REP = 20
MASTER_SEED = 20261022
BASE = HierarchicalCovariance()


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


def time_call(fn, *args) -> tuple[float, bool, str]:
    t0 = time.perf_counter()
    res = fn(*args)
    ms = 1000.0 * (time.perf_counter() - t0)
    return ms, bool(res.failed), res.reason


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(MASTER_SEED)
    V0_w = BASE.animal_cov(WEEKLY)
    V0_b = BASE.animal_cov(WEEKLY[BIWEEKLY_IDX])
    rows: list[dict] = []
    fit_idx = 0
    wall0 = time.perf_counter()
    for scen in SCENARIOS:
        dgp = truth_cov(scen["animal_slope"], scen["study_slope"])
        for n_per in SIZES:
            for rep in range(N_REP):
                zb, sex, group = animals(n_per, rng)
                resid = dgp.sample(rng, WEEKLY, len(zb), 1)[0]
                Z = mean_path(zb, sex, group, WEEKLY, scen["deltas"], scen["shape"]) + resid
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
                    ms, failed, reason = time_call(fn, *args)
                    rows.append({
                        "fit_idx": fit_idx,
                        "scenario": scen["name"],
                        "n_per_sex": "/".join(map(str, n_per)),
                        "n_animals": int(sum(n_per) * 2),
                        "replicate": rep,
                        "analysis": analysis,
                        "schedule": schedule,
                        "missingness": "complete",
                        "elapsed_ms": ms,
                        "failed": failed,
                        "reason": reason,
                    })
                    fit_idx += 1
    wall_s = time.perf_counter() - wall0
    df = pd.DataFrame(rows)
    by = df.groupby("analysis", sort=False)["elapsed_ms"].agg(["count", "mean", "median", "max"]).reset_index()
    fail = df.groupby("analysis", sort=False)["failed"].mean().reset_index(name="failure_rate")
    summary = by.merge(fail, on="analysis")
    n_fits = len(df)
    mean_ms = float(df["elapsed_ms"].mean())
    p90_ms = float(df["elapsed_ms"].quantile(0.90))
    estimates = {
        "measured_wall_seconds_pilot": wall_s,
        "n_study_datasets": int(N_REP * len(SCENARIOS) * len(SIZES)),
        "n_analysis_fits": n_fits,
        "mean_ms_per_fit": mean_ms,
        "median_ms_per_fit": float(df["elapsed_ms"].median()),
        "p90_ms_per_fit": p90_ms,
        "failure_rate_all": float(df["failed"].mean()),
        "projected_exploratory_B600_seconds": 48000 * mean_ms / 1000.0,
        "projected_locked_B2000_seconds": 160000 * mean_ms / 1000.0,
        "projection_note": "Linear in mean_ms_per_fit from this pilot. Not a guarantee.",
    }
    payload = {
        "pilot_id": "c1-timing-pilot-20261022",
        "status": "timing_only_completed_not_operating_characteristics",
        "labels": STATUS,
        "master_seed": MASTER_SEED,
        "n_replicates": N_REP,
        "working_covariance": "unstressed HierarchicalCovariance.animal_cov (historical plug-in stand-in)",
        "dgp": "HierarchicalCovariance.sample plus specified mean path; complete data",
        "analyses": ["A_endpoint_ancova", "B_rm_gls_quadratic weekly/biweekly", "C1_mmrm_visit_contrast weekly/biweekly"],
        "scenarios": [s["name"] for s in SCENARIOS],
        "sizes": [list(s) for s in SIZES],
        "per_analysis": summary.to_dict(orient="records"),
        "estimates": estimates,
        "forbidden_use": [
            "Do not compute or cite FWER, coverage, bias, interval width, or power from this pilot.",
            "Do not tune C1 to these draws.",
        ],
    }
    (OUT / "fit_timings.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (OUT / "timing_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# C1 timing-only pilot",
        "",
        "## Status",
        "",
    ]
    lines += [f"- **{x}**" for x in STATUS]
    lines += [
        "",
        f"- Seed `{MASTER_SEED}`; {N_REP} replicates; complete data; 8 scenarios × 2 sizes × 5 analyses.",
        f"- Fits recorded: **{n_fits}**. Study datasets: **{estimates['n_study_datasets']}**.",
        f"- Wall clock: **{wall_s:.3f} s**. Mean **{mean_ms:.3f} ms/fit** (median {estimates['median_ms_per_fit']:.3f}; p90 {p90_ms:.3f}).",
        f"- Fit failure rate (software stability only): **{estimates['failure_rate_all']:.4f}**.",
        "",
        "## Per-analysis timing",
        "",
        "| Analysis | Fits | Mean ms | Median ms | Max ms | Failure rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| `{r.analysis}` | {int(r.count)} | {r.mean:.3f} | {r.median:.3f} | {r.max:.3f} | {r.failure_rate:.4f} |"
        )
    lines += [
        "",
        "## Linear projections from this pilot (not guaranteed)",
        "",
        f"- Exploratory B=600, 48,000 fits: about **{estimates['projected_exploratory_B600_seconds']:.1f} s**.",
        f"- Diagnostic locked B=2000, 160,000 fits: about **{estimates['projected_locked_B2000_seconds']:.1f} s**.",
        "",
        "These seconds are `n_fits × mean_ms_per_fit`. They ignore process startup and any later I/O.",
        "",
        "## Not computed",
        "",
        "FWER, coverage, bias, interval width, and power were not calculated.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"bundle_dir": str(OUT.relative_to(ROOT)), "labels": STATUS, "files": {}}
    for name in ("fit_timings.csv", "timing_summary.json", "REPORT.md"):
        p = OUT / name
        manifest["files"][name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "estimates": estimates}, indent=2))


if __name__ == "__main__":
    main()

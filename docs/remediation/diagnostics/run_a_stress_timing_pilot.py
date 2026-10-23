"""Timing-only A-default stress pilot. Not an operating-characteristic evaluation.

20 replicates x 11 scenarios x 2 sizes x 5 analyses = 2,200 fits.
Complete data. Deferred mechanisms only: covariance shift, heavy tails, delayed
alternatives. A is the subject; B and C1 are comparators. Do not use these draws
for FWER, coverage, bias, or power. Do not retune C1 or restore GLS as a default.
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
OUT = ROOT / "docs/remediation/timing_pilot_20270301_a_stress"
HORIZON = 28.0
WEEKLY = np.array([7.0, 14.0, 21.0, 28.0])
BIWEEKLY_IDX = np.array([1, 3])
STATUS = [
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
    "Timing-only A-default stress pilot. Not an operating-characteristic evaluation.",
]
# Named stresses match stress_simulators: sigma_p x1.6 and ell x2; t(3)
# measurement x1.5; delayed fixed shape. Working V0 stays the unstressed
# historical plug-in. Heterogeneity and transient are closed-chapter cells
# and are not re-run here.
SCENARIOS = [
    {"name": "covshift_global_null", "mechanism": "covariance_shift",
     "deltas": (0.0, 0.0), "shape": "linear", "noise": "gaussian",
     "sigma_p_mult": 1.6, "ell_mult": 2.0, "sigma_m_mult": 1.0},
    {"name": "covshift_partial_null_low", "mechanism": "covariance_shift",
     "deltas": (0.0, -0.15), "shape": "linear", "noise": "gaussian",
     "sigma_p_mult": 1.6, "ell_mult": 2.0, "sigma_m_mult": 1.0},
    {"name": "covshift_partial_null_high", "mechanism": "covariance_shift",
     "deltas": (-0.10, 0.0), "shape": "linear", "noise": "gaussian",
     "sigma_p_mult": 1.6, "ell_mult": 2.0, "sigma_m_mult": 1.0},
    {"name": "covshift_alternative", "mechanism": "covariance_shift",
     "deltas": (-0.10, -0.15), "shape": "linear", "noise": "gaussian",
     "sigma_p_mult": 1.6, "ell_mult": 2.0, "sigma_m_mult": 1.0},
    {"name": "heavytail_global_null", "mechanism": "heavy_tails",
     "deltas": (0.0, 0.0), "shape": "linear", "noise": "t3_measurement",
     "sigma_p_mult": 1.0, "ell_mult": 1.0, "sigma_m_mult": 1.5},
    {"name": "heavytail_partial_null_low", "mechanism": "heavy_tails",
     "deltas": (0.0, -0.15), "shape": "linear", "noise": "t3_measurement",
     "sigma_p_mult": 1.0, "ell_mult": 1.0, "sigma_m_mult": 1.5},
    {"name": "heavytail_partial_null_high", "mechanism": "heavy_tails",
     "deltas": (-0.10, 0.0), "shape": "linear", "noise": "t3_measurement",
     "sigma_p_mult": 1.0, "ell_mult": 1.0, "sigma_m_mult": 1.5},
    {"name": "heavytail_alternative", "mechanism": "heavy_tails",
     "deltas": (-0.10, -0.15), "shape": "linear", "noise": "t3_measurement",
     "sigma_p_mult": 1.0, "ell_mult": 1.0, "sigma_m_mult": 1.5},
    {"name": "delayed_alternative", "mechanism": "delayed",
     "deltas": (-0.10, -0.15), "shape": "delayed", "noise": "gaussian",
     "sigma_p_mult": 1.0, "ell_mult": 1.0, "sigma_m_mult": 1.0},
    {"name": "delayed_partial_null_low", "mechanism": "delayed",
     "deltas": (0.0, -0.15), "shape": "delayed", "noise": "gaussian",
     "sigma_p_mult": 1.0, "ell_mult": 1.0, "sigma_m_mult": 1.0},
    {"name": "delayed_partial_null_high", "mechanism": "delayed",
     "deltas": (-0.10, 0.0), "shape": "delayed", "noise": "gaussian",
     "sigma_p_mult": 1.0, "ell_mult": 1.0, "sigma_m_mult": 1.0},
]
SIZES = [(6, 6, 6), (8, 8, 8)]
N_REP = 20
MASTER_SEED = 20270301
BASE = HierarchicalCovariance()
N_SCEN = len(SCENARIOS)
N_ANALYSES = 5
N_EXPLORATORY_FITS = N_SCEN * len(SIZES) * 600 * N_ANALYSES
N_LOCKED_FITS = N_SCEN * len(SIZES) * 2000 * N_ANALYSES


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
    z = 0.18 * u[None] - 0.03 * (u**2)[None] + (sex == 1)[:, None] * (0.015 + 0.025 * u[None])
    z = z + zb_c[:, None] * (1.0 - 0.05 * u[None])
    s = fixed_shape(shape, times, HORIZON)
    for g, d in enumerate(deltas, start=1):
        z = z + (group == g)[:, None] * (d * s[None])
    return z


def truth_cov(scen: dict) -> HierarchicalCovariance:
    return HierarchicalCovariance(
        sd_s0=BASE.sd_s0,
        sd_s1=BASE.sd_s1,
        sd_a0=BASE.sd_a0,
        sd_a1=BASE.sd_a1,
        sigma_p=BASE.sigma_p * scen["sigma_p_mult"],
        ell=BASE.ell * scen["ell_mult"],
        sigma_m=BASE.sigma_m * scen["sigma_m_mult"],
    )


def sample_residuals(rng: np.random.Generator, times: np.ndarray, n_animals: int, scen: dict) -> np.ndarray:
    dgp = truth_cov(scen)
    if scen["noise"] == "gaussian":
        return dgp.sample(rng, times, n_animals, 1)[0]
    if scen["noise"] != "t3_measurement":
        raise ValueError(scen["noise"])
    # Same study / animal / OU family as HierarchicalCovariance; measurement is
    # t(3) scaled to the stressed sigma_m, matching stress_heavy_tails.
    latent = HierarchicalCovariance(
        sd_s0=dgp.sd_s0, sd_s1=dgp.sd_s1, sd_a0=dgp.sd_a0, sd_a1=dgp.sd_a1,
        sigma_p=dgp.sigma_p, ell=dgp.ell, sigma_m=0.0,
    ).sample(rng, times, n_animals, 1)[0]
    df = 3.0
    scale = dgp.sigma_m / np.sqrt(df / (df - 2.0))
    meas = rng.standard_t(df, size=(n_animals, len(times))) * scale
    return latent + meas


def time_call(fn, *args) -> tuple[float, bool, str]:
    t0 = time.perf_counter()
    res = fn(*args)
    ms = 1000.0 * (time.perf_counter() - t0)
    return ms, bool(res.failed), res.reason


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(MASTER_SEED)
    v0_w = BASE.animal_cov(WEEKLY)
    v0_b = BASE.animal_cov(WEEKLY[BIWEEKLY_IDX])
    rows: list[dict] = []
    fit_idx = 0
    wall0 = time.perf_counter()
    for scen in SCENARIOS:
        for n_per in SIZES:
            for _rep in range(N_REP):
                zb, sex, group = animals(n_per, rng)
                resid = sample_residuals(rng, WEEKLY, len(zb), scen)
                z = mean_path(zb, sex, group, WEEKLY, scen["deltas"], scen["shape"]) + resid
                zbwk = z[:, BIWEEKLY_IDX]
                jobs = [
                    ("A_endpoint_ancova", "endpoint_only", endpoint_ancova,
                     (zb, sex, group, z[:, -1], 2)),
                    ("B_rm_gls_quadratic", "weekly", rm_gls,
                     (zb, sex, group, z, WEEKLY, HORIZON, v0_w, 2)),
                    ("B_rm_gls_quadratic", "biweekly", rm_gls,
                     (zb, sex, group, zbwk, WEEKLY[BIWEEKLY_IDX], HORIZON, v0_b, 2)),
                    ("C1_mmrm_visit_contrast", "weekly", mmrm_visit_contrast,
                     (zb, sex, group, z, WEEKLY, HORIZON, v0_w, 2)),
                    ("C1_mmrm_visit_contrast", "biweekly", mmrm_visit_contrast,
                     (zb, sex, group, zbwk, WEEKLY[BIWEEKLY_IDX], HORIZON, v0_b, 2)),
                ]
                for analysis, schedule, fn, args in jobs:
                    ms, failed, reason = time_call(fn, *args)
                    rows.append({
                        "fit_idx": fit_idx,
                        "mechanism": scen["mechanism"],
                        "scenario": scen["name"],
                        "n_per_sex": "/".join(map(str, n_per)),
                        "n_animals": int(sum(n_per) * 2),
                        "replicate": _rep,
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
        "n_study_datasets": int(N_REP * N_SCEN * len(SIZES)),
        "n_analysis_fits": n_fits,
        "mean_ms_per_fit": mean_ms,
        "median_ms_per_fit": float(df["elapsed_ms"].median()),
        "p90_ms_per_fit": p90_ms,
        "failure_rate_all": float(df["failed"].mean()),
        "projected_exploratory_B600_fits": N_EXPLORATORY_FITS,
        "projected_exploratory_B600_seconds": N_EXPLORATORY_FITS * mean_ms / 1000.0,
        "projected_locked_B2000_fits": N_LOCKED_FITS,
        "projected_locked_B2000_seconds": N_LOCKED_FITS * mean_ms / 1000.0,
        "projection_note": "Linear in mean_ms_per_fit from this pilot. Not a guarantee.",
    }
    payload = {
        "pilot_id": "a-stress-timing-pilot-20270301",
        "status": "timing_only_completed_not_operating_characteristics",
        "labels": STATUS,
        "master_seed": MASTER_SEED,
        "n_replicates": N_REP,
        "subject": "A_endpoint_ancova (production default after analysis close-out)",
        "comparators": ["B_rm_gls_quadratic", "C1_mmrm_visit_contrast"],
        "working_covariance": "unstressed HierarchicalCovariance.animal_cov (historical plug-in stand-in)",
        "dgp": (
            "HierarchicalCovariance family plus specified mean path; complete data. "
            "covariance_shift: sigma_p x1.6, ell x2. "
            "heavy_tails: t(3) measurement at sigma_m x1.5, variance-normalized. "
            "delayed: fixed_shape delayed. Heterogeneity and transient not re-run."
        ),
        "analyses": [
            "A_endpoint_ancova",
            "B_rm_gls_quadratic weekly/biweekly",
            "C1_mmrm_visit_contrast weekly/biweekly",
        ],
        "mechanisms": ["covariance_shift", "heavy_tails", "delayed"],
        "scenarios": [s["name"] for s in SCENARIOS],
        "sizes": [list(s) for s in SIZES],
        "per_analysis": summary.to_dict(orient="records"),
        "estimates": estimates,
        "forbidden_use": [
            "Do not compute or cite FWER, coverage, bias, interval width, or power from this pilot.",
            "Do not tune C1 to these draws.",
            "Do not restore GLS as a production default from this pilot.",
            "Do not change the ANCOVA default from this pilot.",
        ],
    }
    (OUT / "fit_timings.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (OUT / "timing_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# A-default stress timing-only pilot",
        "",
        "## Status",
        "",
    ]
    lines += [f"- **{x}**" for x in STATUS]
    lines += [
        "",
        f"- Seed `{MASTER_SEED}`; {N_REP} replicates; complete data; {N_SCEN} scenarios × 2 sizes × 5 analyses.",
        "- Mechanisms: covariance shift (`sigma_p` ×1.6, `ell` ×2), heavy tails (t(3) measurement ×1.5), delayed alternatives.",
        "- A is the subject. B and C1 are comparators. Production defaults stay `[ancova]`. C1 is not retuned.",
        f"- Fits recorded: **{n_fits}**. Study datasets: **{estimates['n_study_datasets']}**.",
        f"- Wall clock: **{wall_s:.3f} s**. Mean **{mean_ms:.3f} ms/fit** (median {estimates['median_ms_per_fit']:.3f}; p90 {p90_ms:.3f}).",
        f"- Fit failure rate (software stability only): **{estimates['failure_rate_all']:.4f}**.",
        "- Working covariance: unstressed historical plug-in. Closed-chapter heterogeneity and transient cells were not re-run.",
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
        f"- Exploratory B=600, {N_EXPLORATORY_FITS:,} fits: about **{estimates['projected_exploratory_B600_seconds']:.1f} s**.",
        f"- Diagnostic locked B=2000, {N_LOCKED_FITS:,} fits: about **{estimates['projected_locked_B2000_seconds']:.1f} s**.",
        "",
        "These seconds are `n_fits × mean_ms_per_fit`. They ignore process startup and any later I/O.",
        "",
        "## Not computed",
        "",
        "FWER, coverage, bias, interval width, and power were not calculated.",
        "",
        "## Constraints on use",
        "",
        "- Do not cite this pilot as an operating-characteristic result.",
        "- Do not retune C1 or restore GLS as a default.",
        "- Do not treat a later pass on these mechanisms as design qualification.",
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

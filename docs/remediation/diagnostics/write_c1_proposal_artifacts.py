"""Write algebraic fixtures for the 2026-10-22 C1 proposal. No Monte Carlo calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from send_bw.models.covariance import HierarchicalCovariance
from send_bw.simulation.analysis import endpoint_ancova, mmrm_visit_contrast, rm_gls
from send_bw.simulation.metrics import clopper_pearson, mcse_prop
from send_bw.simulation.paths import fixed_shape

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/remediation/final_analysis_proposal_20261022_c1"
HORIZON = 28.0
SCHEDULES = {
    "biweekly": np.array([14.0, 28.0]),
    "weekly": np.array([7.0, 14.0, 21.0, 28.0]),
}
SHAPES = ("linear", "delayed", "transient")
SCENARIOS = {
    "global_null": (0.0, 0.0),
    "partial_null_low": (0.0, -0.15),
    "partial_null_high": (-0.10, 0.0),
    "alternative": (-0.10, -0.15),
}
STATUS = [
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
]


def fixture():
    groups, sexes, zbs = [], [], []
    for g in range(3):
        for s in (0, 1):
            u = (np.arange(1, 9) - 0.5) / 8.0
            z = 5.27 + 0.29 * s + 0.068 * (u - 0.5) * 2.0
            groups.append(np.full(8, g, int))
            sexes.append(np.full(8, s, int))
            zbs.append(z)
    return np.concatenate(zbs), np.concatenate(sexes), np.concatenate(groups)


def mean_Z(zb, sex, group, times, deltas, shape):
    u = times / HORIZON
    zb_c = zb - zb.mean()
    Z = 0.18 * u[None] - 0.03 * (u**2)[None] + (sex == 1)[:, None] * (0.015 + 0.025 * u[None])
    Z = Z + zb_c[:, None] * (1.0 - 0.05 * u[None])
    s = fixed_shape(shape, times, HORIZON)
    for g, d in enumerate(deltas, start=1):
        Z = Z + (group == g)[:, None] * (d * s[None])
    return Z


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zb, sex, grp = fixture()
    working = HierarchicalCovariance()
    rows = []
    idx = 0
    for sched, times in SCHEDULES.items():
        V0 = working.animal_cov(times)
        for shape in SHAPES:
            for scen, deltas in SCENARIOS.items():
                Z = mean_Z(zb, sex, grp, times, deltas, shape)
                a = endpoint_ancova(zb, sex, grp, Z[:, -1], 2)
                b = rm_gls(zb, sex, grp, Z, times, HORIZON, V0, 2)
                c = mmrm_visit_contrast(zb, sex, grp, Z, times, HORIZON, V0, 2)
                for cid, name, res in (("A", "endpoint_ancova", a), ("B", "rm_gls_quadratic", b),
                                       ("C1", "mmrm_visit_contrast", c)):
                    err = np.asarray(res.est) - np.asarray(deltas)
                    spill = 0.0
                    if deltas[0] == 0 and deltas[1] != 0:
                        spill = abs(err[0])
                    elif deltas[1] == 0 and deltas[0] != 0:
                        spill = abs(err[1])
                    elif deltas == (0.0, 0.0):
                        spill = float(np.max(np.abs(err)))
                    rows.append({
                        "fixture_row_idx": idx, "schedule": sched, "n_visits": len(times),
                        "shape": shape, "scenario": scen, "candidate_id": cid,
                        "candidate_name": name, "true_delta_g1": deltas[0],
                        "true_delta_g2": deltas[1], "est_delta_g1": float(res.est[0]),
                        "est_delta_g2": float(res.est[1]), "max_abs_bias": float(np.max(np.abs(err))),
                        "null_arm_abs_spillover": float(spill),
                        "recovers_exact_endpoint": bool(np.max(np.abs(err)) <= 1e-12),
                    })
                    idx += 1
    df = pd.DataFrame(rows)

    times = SCHEDULES["weekly"]
    raw = np.column_stack([
        0.020 * np.cos(np.arange(len(zb)) * 0.9 + j * 0.5)
        + 0.015 * ((np.arange(len(zb)) % 3 == 2) * (1.0 + 0.4 * j))
        for j in range(len(times))
    ])
    raw[grp == 2, -1] *= 1.8
    Z_se = 0.10 * (times / HORIZON)[None] + raw
    a = endpoint_ancova(zb, sex, grp, Z_se[:, -1], 2)
    c = mmrm_visit_contrast(zb, sex, grp, Z_se, times, HORIZON, working.animal_cov(times), 2)
    se_check = {
        "description": "Deterministic nonzero-residual complete-data check; not Monte Carlo.",
        "est_ancova": a.est.tolist(),
        "est_mmrm": c.est.tolist(),
        "max_abs_est_diff": float(np.max(np.abs(a.est - c.est))),
        "se_ancova_ols": a.se.tolist(),
        "se_mmrm_cr1": c.se.tolist(),
        "se_ratio_mmrm_over_ancova": (c.se / a.se).tolist(),
        "realized_se_equal": bool(np.allclose(a.se, c.se, rtol=1e-8, atol=1e-12)),
    }

    n_scen, n_size, n_miss = 8, 2, 1
    n_rep_expl, n_rep_final = 600, 2000
    n_unique_studies_expl = n_scen * n_size * n_miss * n_rep_expl
    n_unique_studies_final = n_scen * n_size * n_miss * n_rep_final
    fits_per_study = 5  # A + B weekly + B biweekly + C1 weekly + C1 biweekly under CRN
    proposal = {
        "proposal_id": "final-analysis-c1-diagnostic-20261022-v2",
        "status": "c1_implemented_opt_in_calibration_not_executed",
        "labels": STATUS,
        "implementation": {
            "function": "send_bw.simulation.analysis.mmrm_visit_contrast",
            "opt_in_analysis_name": "mmrm",
            "production_defaults_changed": False,
            "configs_still_ancova_gls": True,
            "cli_calibrate_analysis": False,
            "c2_included": False,
            "monte_carlo_executed": False,
        },
        "missingness": {
            "supported_now": {
                "complete": "All scheduled visits observed.",
                "mcar": "Visit absence independent of outcomes and of covariates except through the realized design.",
                "covariate_dependent": "Absence may depend on zb, sex, group already in the mean. Not prior outcomes.",
            },
            "not_supported_for_c1_claims": {
                "prior_outcome_mar_dropout": "Absence depends on previously observed Z. Historical V0 GLS plus sandwich is not claimed consistent when V0 is misspecified.",
            },
        },
        "diagnostic_matrix": {
            "analyses": ["endpoint_ancova", "rm_gls_quadratic", "mmrm_visit_contrast"],
            "scenarios": [
                {"name": "global_null", "deltas": [0.0, 0.0], "shape": "linear", "cov_scale": {}},
                {"name": "partial_null_low", "deltas": [0.0, -0.15], "shape": "linear", "cov_scale": {}},
                {"name": "partial_null_high", "deltas": [-0.10, 0.0], "shape": "linear", "cov_scale": {}},
                {"name": "heterogeneity_global_null", "deltas": [0.0, 0.0], "shape": "linear",
                 "cov_scale": {"animal_slope": 1.6, "study_slope": 1.6}},
                {"name": "heterogeneity_partial_null_low", "deltas": [0.0, -0.15], "shape": "linear",
                 "cov_scale": {"animal_slope": 1.6, "study_slope": 1.6}},
                {"name": "alternative_transient", "deltas": [-0.10, -0.15], "shape": "transient", "cov_scale": {}},
                {"name": "partial_null_low_transient", "deltas": [0.0, -0.15], "shape": "transient", "cov_scale": {}},
                {"name": "partial_null_high_transient", "deltas": [-0.10, 0.0], "shape": "transient", "cov_scale": {}},
            ],
            "n_per_sex": [[6, 6, 6], [8, 8, 8]],
            "allocation": "equal",
            "schedules": {"endpoint_only": [28.0], "biweekly": [14.0, 28.0], "weekly": [7.0, 14.0, 21.0, 28.0]},
            "missingness": ["complete"],
            "deferred_to_later_locked_evaluation": [
                "covariance_shift", "heavy_tails", "delayed_alternatives", "meaningful_5pct",
                "control_heavy allocation", "twice_weekly", "prior_outcome_mar", "C2/Satterthwaite",
            ],
        },
        "cell_inventory": {
            "unique_study_draws_per_replicate": n_scen * n_size * n_miss,
            "fits_per_unique_study": fits_per_study,
            "exploratory": {
                "B": n_rep_expl, "master_seed": 20261101,
                "n_study_datasets": n_unique_studies_expl,
                "n_analysis_fits": n_unique_studies_expl * fits_per_study,
            },
            "diagnostic_locked": {
                "B": n_rep_final, "master_seed": 20261201,
                "n_study_datasets": n_unique_studies_final,
                "n_analysis_fits": n_unique_studies_final * fits_per_study,
            },
            "timing_pilot_requested": {
                "purpose": "Measure ms/fit only. Discarded for operating-characteristic claims.",
                "n_replicates": 20,
                "n_study_datasets": n_scen * n_size * n_miss * 20,
                "n_analysis_fits": n_scen * n_size * n_miss * 20 * fits_per_study,
            },
        },
        "acceptance_sources": {
            "nominal_alpha": {"value": 0.05, "source": "configs/new_study.example.yaml:alpha"},
            "fwer_max": {"value": 0.05, "source": "configs/new_study.example.yaml:fwer_max"},
            "fwer_tolerance": {"value": 0.015, "source": "configs/new_study.example.yaml:fwer_tolerance"},
            "fwer_upper_tolerance": {"value": 0.045, "source": "configs/new_study.example.yaml:fwer_upper_tolerance"},
            "coverage_min": {"value": 0.92, "source": "configs/new_study.example.yaml:coverage_min"},
            "max_abs_bias_fraction": {"value": 0.15, "source": "configs/new_study.example.yaml:max_abs_bias_fraction"},
            "max_failure_rate": {"value": 0.02, "source": "configs/new_study.example.yaml:max_failure_rate"},
            "power_target": {"value": 0.80, "source": "configs/new_study.example.yaml:power_target"},
            "power_lower_tolerance": {"value": 0.06, "source": "configs/new_study.example.yaml:power_lower_tolerance"},
            "feasibility_application": {
                "source": "src/send_bw/simulation/optimize.py::feasibility",
                "note": "Power gates apply to a chosen design's alternative cell, not every small n.",
            },
            "clopper_pearson_at_B2000_p05": {
                "mcse": mcse_prop(0.05, 2000),
                "cp95_k100_n2000": list(clopper_pearson(100, 2000)),
            },
        },
        "algebraic_verification_complete_data": {
            name: {
                "max_abs_bias": float(df[df.candidate_name == name]["max_abs_bias"].max()),
                "max_null_spillover": float(df[df.candidate_name == name]["null_arm_abs_spillover"].max()),
                "all_unbiased": bool(df[df.candidate_name == name]["max_abs_bias"].max() <= 1e-12),
            }
            for name in ("endpoint_ancova", "rm_gls_quadratic", "mmrm_visit_contrast")
        },
        "complete_data_se_check": se_check,
    }
    (OUT / "algebraic_estimand_fixtures.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (OUT / "complete_data_se_check.json").write_text(json.dumps(se_check, indent=2) + "\n", encoding="utf-8")
    (OUT / "calibration_proposal.json").write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    files = ["algebraic_estimand_fixtures.csv", "complete_data_se_check.json", "calibration_proposal.json"]
    manifest = {"bundle_dir": str(OUT.relative_to(ROOT)), "labels": STATUS, "files": {}}
    for name in files:
        p = OUT / name
        manifest["files"][name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    (OUT / "manifest_partial.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

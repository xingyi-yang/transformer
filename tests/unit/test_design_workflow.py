"""Request validation, applicability gates, workflow outcomes and the secondary estimand."""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from send_bw.simulation.analysis import rm_gls, secondary_basis
from send_bw.simulation.designs import Request, validate_request
from send_bw.simulation.optimize import pareto, select_finalists
from send_bw.simulation.workflow import run_design_workflow

LOG = logging.getLogger("test")


def _request(trained, **over) -> dict:
    sm = trained["spline"]
    comp = sorted(sm.spec.compound_ids)[0]
    d = {"request_id": "T", "species": "RAT", "strain": sm.spec.strains[0], "sexes": ["M", "F"],
         "baseline_distribution": {"M": {"mean_g": 250.0, "sd_g": 15.0}, "F": {"mean_g": 190.0, "sd_g": 12.0}},
         "groups": [{"name": "control", "is_control": True, "end_dy": 28},
                    {"name": "low", "compound_id": comp, "dose": 30, "end_dy": 28},
                    {"name": "high", "compound_id": comp, "dose": 300, "end_dy": 28}],
         "horizon_day": 28, "meaningful_effects_pct": {"low": -8.0, "high": -8.0},
         "candidate_n_per_sex": [6, 10], "candidate_schedules": {"endpoint_only": [28], "weekly": [7, 14, 21, 28]},
         "constraints": {"min_per_sex_per_group": 3, "max_total_animals": 200},
         "secondary_contrast": {"times": [7, 14, 21, 28], "weights": [0.25, 0.25, 0.25, 0.25]}}
    d.update(over)
    return d


def _write(tmp_path, d) -> str:
    p = tmp_path / "req.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return str(p)


def test_request_requires_concurrent_control_and_effects(trained):
    d = _request(trained)
    bad = copy.deepcopy(d)
    bad["groups"][0]["is_control"] = False
    bad["groups"][0]["compound_id"], bad["groups"][0]["dose"] = "X", 1.0
    with pytest.raises(ValueError, match="control"):
        Request(**bad)
    bad = copy.deepcopy(d)
    bad["meaningful_effects_pct"] = {"low": -8.0}
    with pytest.raises(ValueError, match="meaningful"):
        Request(**bad)
    bad = copy.deepcopy(d)
    bad["candidate_schedules"] = {"x": [7, 14]}
    with pytest.raises(ValueError, match="horizon"):
        Request(**bad)
    bad = copy.deepcopy(d)
    bad["groups"][1]["start_dy"] = 0
    with pytest.raises(ValueError, match="day 0"):
        Request(**bad)


@pytest.mark.parametrize("change,needle", [
    ({"species": "DOG"}, "species"),
    ({"strain": "NEVER-SEEN"}, "strain"),
    ({"horizon_day": 90, "candidate_schedules": {"endpoint_only": [90]}}, "horizon"),
    ({"missing_feature_declarations": ["planned_regimen"]}, "planned_regimen"),
])
def test_applicability_gates_refuse_unsupported_context(trained, change, needle):
    sm = trained["spline"]
    req = Request(**_request(trained, **change))
    out = validate_request(req, sm.spec, sm.compound_ref)
    assert out["status"] == "not_supported"
    assert any(needle.lower() in p.lower() for p in out["problems"])


def test_unknown_compound_needs_curated_descriptors(trained):
    sm = trained["spline"]
    d = _request(trained)
    for g in d["groups"][1:]:
        g["compound_id"] = "NEWCMP"
    out = validate_request(Request(**d), sm.spec, sm.compound_ref)
    assert out["status"] == "not_supported"
    d["compound_descriptors"] = {"NEWCMP": {"D1": 0.1, "D2": -0.2, "D3": 0.3}}
    out = validate_request(Request(**d), sm.spec, sm.compound_ref)
    assert out["status"] == "supported" and any("extrapolation" in w for w in out["warnings"])


def test_workflow_not_supported_writes_summary(trained, tmp_path):
    path = _write(tmp_path, _request(trained, species="DOG"))
    out = run_design_workflow({"spline": trained["spline"]}, {"request": path, "replicates_proposal": 20},
                              tmp_path / "d", 1, None, LOG)
    assert out["status"] == "not_supported"
    assert (tmp_path / "d" / "design_summary.json").exists()


def test_workflow_no_feasible_design_is_a_valid_outcome(trained, tmp_path):
    d = _request(trained, meaningful_effects_pct={"low": -0.5, "high": -0.5}, candidate_n_per_sex=[4])
    path = _write(tmp_path, d)
    out = run_design_workflow({"spline": trained["spline"]}, {"request": path, "replicates_proposal": 40},
                              tmp_path / "d", 1, None, LOG)
    assert out["status"] == "no_feasible_design"
    assert out["proposals"]["spline"]["n_feasible"] == 0
    feas = pd.read_csv(tmp_path / "d" / "proposal_spline_feasibility.csv")
    assert not feas["feasible"].any() and (~feas["pass_power"]).all()
    assert set(feas["analysis"]) == {"ancova"}


def test_workflow_end_to_end_small(trained, tmp_path):
    path = _write(tmp_path, _request(trained, meaningful_effects_pct={"low": -15.0, "high": -15.0},
                                    analyses=["ancova", "gls"]))
    models = {"spline": trained["spline"], "transformer": trained["transformer"]}
    out = run_design_workflow(models, {"request": path, "replicates_proposal": 60, "replicates_final": 60,
                                       "n_finalists": 1}, tmp_path / "d", 2, None, LOG)
    assert out["status"] in {"evaluated", "no_feasible_design"}
    if out["status"] == "evaluated":
        sims = {s["name"] for s in out["simulators"]}
        assert {"fresh_spline", "fresh_transformer", "generator_B"} <= sims
        assert not any(s.startswith("oracle") for s in sims)  # no truth -> no oracle
        for rec in out["recommendations"].values():
            if rec["status"] == "recommended":
                checks = {c["candidate"]: c for c in rec["finalist_checks"]}
                assert checks[rec["design"]]["fresh_same_model"] and checks[rec["design"]]["generator_B"]
        assert out["monte_carlo_nonconclusive"]
    res = pd.read_csv(tmp_path / "d" / "proposal_spline_results.csv")
    assert {"global_null", "partial_null_low", "partial_null_high", "alternative", "model_predicted"} <= set(res["scenario"])
    gls = res[(res["analysis"] == "gls") & (res["scenario"] == "alternative")]
    assert gls["secondary_coverage_min"].notna().all()
    anc = res[res["analysis"] == "ancova"]
    assert "secondary_coverage_min" not in anc or anc["secondary_coverage_min"].isna().all()


def test_secondary_basis_and_gls_secondary_equals_primary_at_T():
    np.testing.assert_allclose(secondary_basis(np.array([28.0]), np.array([1.0]), 28.0, 2), [1.0, 1.0])
    np.testing.assert_allclose(secondary_basis(np.array([14.0, 28.0]), np.array([0.5, 0.5]), 28.0, 2),
                               [0.75, 0.625])
    rng = np.random.default_rng(0)
    N, times = 60, np.array([7.0, 14.0, 28.0])
    group, sex = np.repeat([0, 1, 2], 20), np.tile([0, 1], 30)
    zb = rng.normal(5.4, 0.05, N)
    Z = zb[:, None] + 0.01 * times[None] / 28 + rng.normal(0, 0.02, (N, 3))
    V0 = 0.0004 * np.exp(-np.abs(times[:, None] - times[None]) / 10) + 1e-5 * np.eye(3)
    r1 = rm_gls(zb, sex, group, Z, times, 28.0, V0, 2, secondary=(np.array([28.0]), np.array([1.0])))
    np.testing.assert_allclose(r1.sec_est, r1.est)
    np.testing.assert_allclose(r1.sec_se, r1.se)


def test_secondary_contrast_validation(trained):
    with pytest.raises(ValueError, match="secondary"):
        Request(**_request(trained, secondary_contrast={"times": [7, 40], "weights": [0.5, 0.5]}))
    with pytest.raises(ValueError, match="secondary"):
        Request(**_request(trained, secondary_contrast={"times": [7, 28], "weights": [0.5, 0.6]}))


def test_finalists_include_ancova_comparator_and_pareto():
    feas = pd.DataFrame({"candidate": ["g1", "g2", "a1", "a2"], "analysis": ["gls", "gls", "ancova", "ancova"],
                         "n_animals": [36, 60, 60, 90], "n_measurements": [180, 300, 120, 180],
                         "cost": [39.6, 66.0, 62.4, 93.6], "feasible": [True, True, False, True]})
    fin, info = select_finalists(feas, k=1)
    assert fin == ["g1", "a2"] and info["comparator"] == "a2"
    par = pareto(feas)
    assert set(par["candidate"]) == {"g1"}  # g1 dominates on both animals and measurements
    fin, info = select_finalists(feas.assign(feasible=False))
    assert fin == [] and info["status"] == "no_feasible_design"

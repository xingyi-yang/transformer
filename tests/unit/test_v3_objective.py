"""Objective validation: MC plan, increment-moment gate, unchanged Bayesian thresholds."""

from __future__ import annotations

import pytest

from send_bw.evaluation.increment_moment import FORBIDDEN_SPLITS, match_increment_moment
from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS
from send_bw.models.covariance import HierarchicalCovariance
from send_bw.v3_objective import (
    ObjectiveError,
    _distributional_pass,
    _mc_plan,
    _primary_increment_rows,
    execute_v3_objective,
)


def test_reliability_thresholds_still_unchanged():
    assert RELIABILITY_THRESHOLDS == {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400}


def test_mc_plan_uses_prior_precision():
    plan = _mc_plan(0.144, 0.10)
    assert plan["n_datasets"] == 3
    assert plan["n_train_seeds"] == 1
    assert "sd of dataset means" in plan["justification"]


def test_increment_moment_refuses_test_split():
    class _M:
        cov = HierarchicalCovariance()

    with pytest.raises(ValueError, match="test"):
        match_increment_moment(_M(), None, split_name="test")  # type: ignore[arg-type]
    assert "test" in FORBIDDEN_SPLITS


def test_process_scale_changes_sds_not_ell():
    cov = HierarchicalCovariance(sd_s1=0.04, sigma_p=0.02, ell=5.0)
    cov.scale_process(0.5)
    assert cov.sd_s1 == pytest.approx(0.02)
    assert cov.sigma_p == pytest.approx(0.01)
    assert cov.ell == pytest.approx(5.0)


def test_distributional_rule_does_not_require_sharing_win():
    rows = [
        {"matched_ratio": 0.95, "dataset_index": 0},
        {"matched_ratio": 1.05, "dataset_index": 1},
        {"matched_ratio": 1.00, "dataset_index": 2},
    ]
    out = _distributional_pass(rows, {"reliable": False})
    assert out["pass"] is True
    assert out["bayesian"]["pass"] is False
    assert out["increment"]["pass"] is True


def test_distributional_pass_uses_primary_spline_slice():
    rows = [
        {"matched_ratio": 1.00, "dataset_index": 0, "model": "spline",
         "sharing": "context_specific", "bank": "matched", "density": "full"},
        {"matched_ratio": 0.95, "dataset_index": 1, "model": "spline",
         "sharing": "context_specific", "bank": "matched", "density": "full"},
        {"matched_ratio": 1.05, "dataset_index": 2, "model": "spline",
         "sharing": "context_specific", "bank": "matched", "density": "full"},
        {"matched_ratio": 1.80, "dataset_index": 0, "model": "transformer",
         "sharing": "pooled", "bank": "both", "density": "sparse"},
    ]
    assert len(_primary_increment_rows(rows)) == 3
    out = _distributional_pass(rows, {"reliable": False})
    assert out["pass"] is True
    assert out["increment"]["used_primary_slice"] is True


def test_objective_refuses_preserved_dir(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("output_dir: runs/v3_research\ngenerate_seed: 1\ntrain_seed: 2\n", encoding="utf-8")
    with pytest.raises(ObjectiveError, match="preserved"):
        execute_v3_objective(cfg)

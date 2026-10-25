"""Unresolved limitations record: no test-set adjustment, no new synthetic work."""

from __future__ import annotations

from pathlib import Path

from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS
from send_bw.historical_audit import discover_authorization
from send_bw.v3_limitations import LimitationsError, execute_v3_limitations

ROOT = Path(__file__).resolve().parents[2]


def test_reliability_thresholds_still_unchanged():
    assert RELIABILITY_THRESHOLDS == {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400}


def test_limitations_refuses_preserved_increment_dir():
    try:
        execute_v3_limitations("runs/v3_increment", "runs/v3_phaseb", "runs/v3_increment")
    except LimitationsError as e:
        assert "preserved" in str(e)
    else:
        raise AssertionError("expected LimitationsError")


def test_record_from_existing_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("SEND_BW_AUTHORIZATION", raising=False)
    out = execute_v3_limitations("runs/v3_increment", "runs/v3_phaseb", tmp_path / "lim")
    assert out["adjusted_using_exposed_test"] is False
    assert out["retuned_after_test"] is False
    assert out["architectures_added"] is False
    assert out["grid_relaunched"] is False
    assert out["broad_synthetic_development"] == "stopped"
    assert out["next_implementation_task"] is None
    assert out["preserved_pr"] == 7
    ids = [item["id"] for item in out["limitations"]]
    assert ids == ["excess_increment_variability", "unreliable_bayesian_inference"]
    inc, bay = out["limitations"]
    assert inc["status"] == "unresolved"
    assert inc["adjusted_using_exposed_test"] is False
    assert inc["matched_ratio_mean_of_dataset_means"] > 1.0
    assert inc["n_datasets"] == 2
    assert bay["status"] == "unresolved"
    assert bay["explicitly_unreliable"] is True
    assert bay["scored"] is False
    assert bay["thresholds_unchanged"] == RELIABILITY_THRESHOLDS
    assert all(c.get("counted_as_successful_benchmark") is False for c in bay["cells"])
    phase = out["phase_b"]
    assert phase["status"] == "ACCESS_BLOCKER"
    assert phase["authorized"] is False
    assert phase["created_ticket"] is False
    assert phase["self_approved"] is False
    assert phase["source_opened"] is False
    assert phase["endpoint_values_inspected"] is False
    assert phase["models_trained"] is False
    assert phase["design_recommendations"] is False


def test_authorization_still_absent():
    d = discover_authorization(None)
    assert d["authorized"] is False
    assert d["created_ticket"] is False
    assert d["self_approved"] is False

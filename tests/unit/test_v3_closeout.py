"""Closeout analysis and historical-audit gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS
from send_bw.historical_audit import AuditGateError, execute_historical_audit, validate_authorization
from send_bw.v3_closeout import (
    CloseoutError,
    execute_v3_closeout,
    nested_delta_summary,
    sharing_density_effects,
    train_seeds_change_splits,
)

ROOT = Path(__file__).resolve().parents[2]


def test_reliability_thresholds_still_unchanged():
    assert RELIABILITY_THRESHOLDS == {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400}


def test_nested_summary_separates_dataset_from_train():
    rows = [
        {"dataset_index": 0, "train_index": 0, "generate_seed": 1, "train_seed": 10,
         "delta": 0.10, "sign": "harmful"},
        {"dataset_index": 0, "train_index": 1, "generate_seed": 1, "train_seed": 11,
         "delta": 0.00, "sign": "unchanged"},
        {"dataset_index": 1, "train_index": 0, "generate_seed": 2, "train_seed": 10,
         "delta": 0.20, "sign": "harmful"},
        {"dataset_index": 1, "train_index": 1, "generate_seed": 2, "train_seed": 11,
         "delta": 0.10, "sign": "harmful"},
    ]
    s = nested_delta_summary(rows)
    assert s["n_datasets"] == 2
    assert s["n_pairs_pooled"] == 4
    assert s["n_train_per_dataset"] == {0: 2, 1: 2}
    assert s["dataset_means"]["0"] == pytest.approx(0.05)
    assert s["dataset_means"]["1"] == pytest.approx(0.15)
    assert s["mean_of_dataset_means"] == pytest.approx(0.10)
    assert s["pooled_mean"] == pytest.approx(0.10)


def test_interaction_is_sparse_sharing_minus_full_sharing():
    paired = {
        "sharing": [
            {"model": "spline", "sharing": "pooled", "bank": "both", "metric": "rmse_log",
             "sparse_target_train": "full", "dataset_index": 0, "train_index": 0,
             "generate_seed": 1, "train_seed": 2, "delta": 0.04, "sign": "harmful"},
            {"model": "spline", "sharing": "pooled", "bank": "both", "metric": "rmse_log",
             "sparse_target_train": 4, "dataset_index": 0, "train_index": 0,
             "generate_seed": 1, "train_seed": 2, "delta": 0.01, "sign": "harmful"},
        ],
        "density": [
            {"model": "spline", "sharing": "pooled", "bank": "both", "metric": "rmse_log",
             "dataset_index": 0, "train_index": 0, "generate_seed": 1, "train_seed": 2,
             "delta": -0.02, "sign": "helpful"},
        ],
    }
    out = sharing_density_effects(paired)
    g = out["groups"][0]
    assert g["interaction_sparse_minus_full_sharing"]["pooled_mean"] == pytest.approx(-0.03)
    assert g["sharing_full"]["pooled_mean"] == pytest.approx(0.04)
    assert g["sparse_vs_full"]["pooled_mean"] == pytest.approx(-0.02)


def test_training_seeds_change_splits_on_executed_lock():
    locked = json.loads((ROOT / "runs" / "v3_research" / "locked_target_splits.json").read_text())
    info = train_seeds_change_splits(locked)
    assert info["training_seeds_change_splits"] is True
    assert any(d["validation_hashes_differ"] and d["test_hashes_differ"] for d in info["by_dataset"])


def test_closeout_refuses_preserved_dir():
    with pytest.raises(CloseoutError, match="preserved"):
        execute_v3_closeout("runs/v3_research", "runs/v3_research")


def test_closeout_from_existing_artifacts(tmp_path):
    out = execute_v3_closeout("runs/v3_research", tmp_path / "close")
    assert out["retuned_after_test"] is False
    assert out["grid_relaunched"] is False
    assert out["reliability_thresholds"]["ess_bulk_min"] == 400
    assert out["nested_seeds"]["training_seeds_change_splits"] is True
    assert out["sharing_density_effects"]["n_groups"] > 0
    assert out["study_balanced_contrast"]["groups"]
    assert out["coverage_increments"]["overall"]["n"] > 0
    bay = out["bayesian_identifiability"]
    assert bay["failed_runs_preserved"] is True
    assert all(c["counted_as_successful_benchmark"] is False for c in bay["cells"])
    assert all(c["failure_class"] == "mean_identifiability" for c in bay["cells"])


def test_audit_phase_a_does_not_touch_real_roots():
    out = execute_historical_audit(ROOT / "configs" / "v3_historical_audit.yaml", phase="A")
    assert out["real_data_accessed"] is False
    assert out["snowflake_connected"] is False
    assert out["extract_root_opened"] is False
    assert out["support_for_synthetic_to_real"] == "not_claimed"
    statuses = {r["item"]: r["status"] for r in out["inventory"]}
    assert statuses["written_authorization"] == "unknown"
    assert statuses["snowflake_authorized_flag"] == "refused"


def test_audit_phase_b_refuses_without_ticket():
    out = execute_historical_audit(ROOT / "configs" / "v3_historical_audit.yaml", phase="B")
    assert out["status"] == "ACCESS_BLOCKER"
    assert out["real_data_accessed"] is False
    assert out["source_opened"] is False
    assert out["created_ticket"] is False
    assert out["self_approved"] is False


def test_authorization_schema_rejects_training_and_unsigned_example():
    example = yaml.safe_load((ROOT / "configs" / "historical_authorization.example.yaml").read_text())
    with pytest.raises(AuditGateError, match="written_authorization"):
        validate_authorization(example, "B")
    with pytest.raises(AuditGateError, match="model training"):
        validate_authorization({
            "written_authorization": True, "steward_name": "x", "purpose": "y",
            "allowed_phases": ["B"], "allows_metadata_only": True,
            "allows_model_training": True, "forbids_design_recommendations": True,
        }, "B")

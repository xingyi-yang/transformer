"""Matched increment statistic and Phase B authorization gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from send_bw.evaluation.increment import (
    animal_increment_sd,
    compare_study,
    hierarchical_mean,
    mixed_draw_increment_sd,
    observed_study_increment,
    simulated_study_increments,
    study_increment_stat,
)
from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS
from send_bw.historical_audit import (
    AuditGateError,
    discover_authorization,
    execute_historical_audit,
    phase_b_metadata_inventory,
    validate_authorization,
)
from send_bw.v3_increment import IncrementError, execute_v3_increment

ROOT = Path(__file__).resolve().parents[2]


def test_reliability_thresholds_still_unchanged():
    assert RELIABILITY_THRESHOLDS == {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400}


def test_animal_and_study_stats_match_prestudy_std():
    inc = np.array([0.1, -0.2, 0.3])
    assert animal_increment_sd(inc) == pytest.approx(float(inc.std()))
    assert study_increment_stat([0.2, 0.4]) == pytest.approx(0.3)


def test_simulated_draw_is_one_study_and_does_not_mix_draws():
    # Equal spacing: a shared per-draw slope adds a constant increment and
    # must not change the study-level increment SD.
    times = np.array([0.0, 7.0, 14.0, 21.0])
    y = np.array([[0.0, 0.1, 0.15, 0.22],
                  [0.0, 0.08, 0.16, 0.20]])
    mask = np.ones_like(y, dtype=bool)
    slope = np.array([0.0, 0.4, -0.2])[:, None, None] * (times / 21.0)
    zs = y[None, :, :] + slope
    obs = observed_study_increment(y, mask)
    sim = simulated_study_increments(zs, mask)
    assert obs["study_stat"] == pytest.approx(sim["expected_study_stat"])
    assert sim["between_draw_sd"] == pytest.approx(0.0)
    mixed = mixed_draw_increment_sd(zs, mask)
    assert mixed is not None and mixed > obs["study_stat"]


def test_unequal_spacing_keeps_between_draw_sd_separate():
    times = np.array([0.0, 2.0, 20.0])
    y = np.array([[0.0, 0.05, 0.08],
                  [0.0, 0.02, 0.11]])
    mask = np.ones_like(y, dtype=bool)
    slope = np.array([0.0, 0.5])[:, None, None] * (times / 20.0)
    zs = y[None, :, :] + slope
    cmp = compare_study(y, zs, mask)
    assert cmp["simulated"]["between_draw_sd"] > 0
    assert cmp["matched_ratio_pred_over_obs"] == pytest.approx(
        cmp["simulated"]["expected_study_stat"] / cmp["observed"]["study_stat"]
    )
    # Mixed-draw SD is not the predicted study-level statistic.
    assert cmp["incorrect_mixed_draw_sd"] != pytest.approx(cmp["simulated"]["expected_study_stat"])


def test_hierarchical_mean_uses_dataset_as_outer_replicate():
    s = hierarchical_mean(
        [0.10, 0.00, 0.20, 0.10],
        group_keys=[0, 0, 1, 1],
    )
    assert s["n_groups"] == 2
    assert s["n_values"] == 4
    assert s["group_means"]["0"] == pytest.approx(0.05)
    assert s["group_means"]["1"] == pytest.approx(0.15)
    assert s["mean_of_group_means"] == pytest.approx(0.10)
    assert s["pooled_mean"] == pytest.approx(0.10)


def test_increment_refuses_preserved_dir():
    with pytest.raises(IncrementError, match="preserved"):
        execute_v3_increment("runs/v3_research", "runs/v3_research")


def test_discover_authorization_rejects_unsigned_example_and_creates_nothing():
    d = discover_authorization(ROOT / "configs" / "historical_authorization.example.yaml")
    assert d["authorized"] is False
    assert d["created_ticket"] is False
    assert d["self_approved"] is False
    assert d["auth"] is None
    reasons = " ".join(c["reason"] or "" for c in d["candidates_checked"])
    assert "unsigned" in reasons or "written_authorization" in reasons


def test_phase_b_without_ticket_is_access_blocker(tmp_path, monkeypatch):
    monkeypatch.delenv("SEND_BW_AUTHORIZATION", raising=False)
    out = execute_historical_audit(ROOT / "configs" / "v3_historical_audit.yaml",
                                   phase="B", output_dir=tmp_path / "phaseb")
    assert out["status"] == "ACCESS_BLOCKER"
    assert out["real_data_accessed"] is False
    assert out["source_opened"] is False
    assert out["endpoint_values_inspected"] is False
    assert out["models_trained"] is False
    assert out["design_recommendations"] is False
    assert (tmp_path / "phaseb" / "ACCESS_BLOCKER.json").exists()


def test_phase_b_metadata_inventory_ignores_bw_values(tmp_path):
    root = tmp_path / "extract"
    meta = tmp_path / "metadata"
    s1 = root / "S001"
    s2 = root / "S002"
    s3 = root / "S003"
    for p in (s1, s2, s3):
        p.mkdir(parents=True)
    (s1 / "ts.csv").write_text(
        "STUDYID,TSPARMCD,TSVAL\nS001,SPECIES,RAT\nS001,STRAIN,SD\nS001,SSTDTC,2020-01-01\nS001,SENDVER,3.1\n",
        encoding="utf-8")
    (s2 / "ts.csv").write_text(
        "STUDYID,TSPARMCD,TSVAL\nS002,SPECIES,RAT\nS002,SSTYP,REPEAT_DOSE_TOX\n",
        encoding="utf-8")
    (s3 / "ts.csv").write_text(
        "STUDYID,TSPARMCD,TSVAL\nS003,SPECIES,DOG\nS003,SSTYP,REPEAT_DOSE_TOX\n",
        encoding="utf-8")
    sentinel = "BW_SENTINEL_99173"
    (s1 / "bw.csv").write_text(
        f"STUDYID,USUBJID,BWSTRESN\nS001,A,{sentinel}\n", encoding="utf-8")
    meta.mkdir()
    (meta / "study_metadata.csv").write_text(
        "STUDYID,SPECIES,STRAIN,STUDY_CLASS,STUDY_START,SEND_VERSION,DURATION_DAYS\n"
        "S001,RAT,SD,REPEAT_DOSE_TOX,2020-01-01,3.1,28\n"
        "S002,RAT,WISTAR,REPEAT_DOSE_TOX,2020-02-01,3.1,28\n"
        "S003,DOG,BEAGLE,REPEAT_DOSE_TOX,2020-03-01,3.1,28\n"
        "S004,DOG,BEAGLE,REPEAT_DOSE_TOX,2020-04-01,3.1,28\n"
        "S005,DOG,BEAGLE,REPEAT_DOSE_TOX,2020-05-01,3.1,28\n",
        encoding="utf-8")
    (meta / "arm_regimen.csv").write_text("STUDYID,ARMCD\nS001,C\n", encoding="utf-8")
    (meta / "compound_reference.csv").write_text("COMPOUND_ID,D1\nC1,0\n", encoding="utf-8")
    auth = {
        "written_authorization": True,
        "steward_name": "test steward",
        "purpose": "unit test metadata inspect",
        "allowed_phases": ["B"],
        "allows_metadata_only": True,
        "forbids_design_recommendations": True,
        "allows_model_training": False,
        "source": {"type": "local", "root": str(root), "layout": "per_study",
                   "metadata_dir": str(meta)},
    }
    validate_authorization(auth, "B")
    out = phase_b_metadata_inventory({"source": {"type": "local", "snowflake": {"authorized": False}}},
                                     auth, {"ticket_path": str(tmp_path / "ticket.yaml")})
    dumped = yaml.safe_dump(out)
    assert sentinel not in dumped
    assert out["endpoint_values_inspected"] is False
    assert out["body_weight_opened"] is False
    assert out["models_trained"] is False
    assert out["design_recommendations"] is False
    assert "BW" in out["provenance"]["domains_refused"]
    assert out["biological_support"]["n_studies"] == 5
    assert out["missingness"]["n_studies"] == 5
    assert out["support_for_synthetic_to_real"] == "not_claimed"
    statuses = {c["status"] for c in out["candidate_validation_contexts"]}
    assert "candidate_metadata_only" in statuses


def test_phase_b_ticket_without_root_does_not_open_config_source(tmp_path):
    ticket = tmp_path / "signed.yaml"
    ticket.write_text(
        "written_authorization: true\n"
        "steward_name: steward\n"
        "purpose: metadata inspect\n"
        "allowed_phases: [B]\n"
        "allows_metadata_only: true\n"
        "forbids_design_recommendations: true\n"
        "allows_model_training: false\n"
        "source:\n  type: local\n  root: ''\n",
        encoding="utf-8",
    )
    out = execute_historical_audit(ROOT / "configs" / "v3_historical_audit.yaml",
                                   phase="B", authorization_path=ticket,
                                   output_dir=tmp_path / "phaseb")
    assert out["status"] == "EXTRACT_BLOCKER"
    assert out["source_opened"] is False
    assert out["real_data_accessed"] is False


def test_authorization_schema_still_rejects_training():
    with pytest.raises(AuditGateError, match="model training"):
        validate_authorization({
            "written_authorization": True, "steward_name": "x", "purpose": "y",
            "allowed_phases": ["B"], "allows_metadata_only": True,
            "allows_model_training": True, "forbids_design_recommendations": True,
        }, "B")

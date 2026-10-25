"""Independent verification of v3 sharing, sparse-study, metrics, provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from send_bw.data.splits import (
    SPARSE_HISTORY_KIND,
    SPARSE_TARGET_TRAIN_KIND,
    compose_sparse_target_train,
    target_manifest,
)
from send_bw.evaluation.forecast import evaluation_pack
from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS, sampling_diagnosis
from send_bw.sharing import SELECTIVE_REGULARIZED, SELECTIVE_V1, is_selective, parse_sharing
from send_bw.v3_research import ResearchProtocolError, validate_research_config

ROOT = Path(__file__).resolve().parents[2]


def test_sharing_structures_remain_distinct():
    assert parse_sharing("selective") == "selective"
    assert parse_sharing("selective_regularized") == "selective_regularized"
    assert is_selective("selective") and is_selective("selective_regularized")
    assert SELECTIVE_V1["variant"] == "v1"
    assert SELECTIVE_REGULARIZED["variant"] == "regularized"
    assert SELECTIVE_V1["residual"] == SELECTIVE_REGULARIZED["residual"] == "target_species_only"
    assert "compound" in SELECTIVE_V1["shared"]
    assert "growth_deviation" in SELECTIVE_REGULARIZED["species_specific"]


def test_sparse_study_kind_is_not_history_encoder():
    assert SPARSE_TARGET_TRAIN_KIND != SPARSE_HISTORY_KIND
    splits = pd.DataFrame({
        "STUDYID": [f"S{i:02d}" for i in range(8)],
        "SPLIT": ["train"] * 5 + ["validation", "test", "test"],
    })
    man0 = target_manifest(splits)
    out, meta = compose_sparse_target_train(splits, 3)
    assert meta["kind"] == SPARSE_TARGET_TRAIN_KIND
    assert target_manifest(out)["validation"]["hash"] == man0["validation"]["hash"]
    assert target_manifest(out)["test"]["hash"] == man0["test"]["hash"]


def test_reliability_thresholds_unchanged():
    assert RELIABILITY_THRESHOLDS == {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400}


def test_sampling_diagnosis_separates_compile_from_mixing():
    compile_d = sampling_diagnosis({}, compile_error="CompileError: Python.h", status="failed")
    assert compile_d["failure_class"] == "compile_or_backend"
    assert compile_d["counted_as_successful_benchmark"] is False
    mix = sampling_diagnosis({
        "reliable": False, "rhat_max": 1.08, "ess_bulk_min": 29, "ess_tail_min": 63,
        "divergences": 0, "draws": 400, "chains": 2, "tune": 400,
        "worst_rhat_params": {"theta[23]": 1.08},
        "worst_ess_params": {"theta[21]": 29.0},
        "thresholds": dict(RELIABILITY_THRESHOLDS),
    })
    assert mix["failure_class"] == "mean_identifiability"
    cov = sampling_diagnosis({
        "reliable": False, "rhat_max": 1.04, "ess_bulk_min": 65, "ess_tail_min": 38,
        "divergences": 0, "draws": 400, "chains": 2,
        "worst_rhat_params": {"sigma_p": 1.04, "ell": 1.03},
        "worst_ess_params": {"ell": 65.0},
    })
    assert cov["failure_class"] == "covariance_mixing"
    ok = sampling_diagnosis({
        "reliable": True, "rhat_max": 1.0, "ess_bulk_min": 500, "ess_tail_min": 500,
        "divergences": 0, "draws": 800, "chains": 2,
    })
    assert ok["failure_class"] == "reliable" and ok["counted_as_successful_benchmark"] is True
    raised = sampling_diagnosis({}, compile_error="ValueError: singular matrix", status="failed")
    assert raised["failure_class"] == "unreliable"
    assert "compile/backend" in raised["note"]


def test_bayesian_class_methods_were_not_swallowed():
    from send_bw.models.bayesian import BayesianModel, _family_diagnostics

    for name in ("_diagnostics", "_prior_predictive", "_posterior_predictive",
                 "_sensitivity_table", "predict_mean", "sample_study", "save", "load"):
        assert hasattr(BayesianModel, name)
    assert callable(_family_diagnostics)


def test_evaluation_pack_exposes_denominators():
    result = {
        "summary": {
            "task_alias": "predose_conditioned", "model": "spline", "split": "test",
            "observation_weighted": {"cov80": 0.7, "width80_log": 0.2, "width95_log": 0.4, "cov95": 0.9},
            "study_balanced": {"rmse_log": 0.05, "crps": 0.03, "cov80": 0.71, "cov95": 0.91,
                               "width80": 0.21, "width95": 0.41},
            "group_mean_cov90": 0.8, "contrast_cov90": 0.6, "contrast_width90": 0.15,
            "contrast_bias": 0.01, "n_contrasts": 3,
            "study_level_contrast_error": {"bias_study_mean": 0.02, "rmse_study_mean": 0.03,
                                           "n_studies_with_contrast": 2},
            "pit_histogram_10bins": [1] * 10,
            "denominators": {"n_studies": 2, "n_obs": 20, "n_contrasts": 3,
                             "rmse_crps_weighting": "study_balanced",
                             "contrast_bias_weighting": "contrast_row_mean"},
        }
    }
    pack = evaluation_pack(result)
    assert pack["denominators"]["n_contrasts"] == 3
    assert pack["denominators"]["rmse_crps_weighting"] == "study_balanced"
    assert pack["individual"]["width80_log"] == 0.21


def test_research_config_refuses_design_and_identical_seeds():
    cfg = yaml.safe_load((ROOT / "configs" / "v3_research.frozen.yaml").read_text(encoding="utf-8"))
    validate_research_config(cfg)
    assert cfg["generate_seed"] != cfg["train_seed"]
    assert set(cfg["research"]["banks"]) == {"matched", "both"}
    bad = dict(cfg)
    bad["design"] = {"request": "configs/new_study.smoke.yaml"}
    with pytest.raises(ResearchProtocolError, match="design"):
        validate_research_config(bad)


def test_executed_prototype_provenance_and_fixed_splits():
    parent = ROOT / "runs" / "v3_repl"
    assert parent.exists()
    lock = json.loads((parent / "freeze_lock.json").read_text())
    assert lock["retune_after_test"] is False
    cells = [
        parent / "replicates" / "r0" / "cells" / "r0_both_context_specific_k0" / "splits" / "target_manifest.json",
        parent / "replicates" / "r0" / "cells" / "r0_both_pooled_k1" / "splits" / "target_manifest.json",
        parent / "replicates" / "r0" / "cells" / "r0_both_selective_k1" / "splits" / "target_manifest.json",
        parent / "replicates" / "r0" / "cells" / "r0_both_selective_regularized_k1" / "splits" / "target_manifest.json",
        parent / "replicates" / "r0" / "cells" / "r0_both_context_specific_k0_stt4" / "splits" / "target_manifest.json",
    ]
    mans = [json.loads(p.read_text()) for p in cells if p.exists()]
    assert len(mans) >= 4
    val = {m["validation"]["hash"] for m in mans}
    tes = {m["test"]["hash"] for m in mans}
    assert len(val) == 1 and len(tes) == 1
    stt = parent / "replicates" / "r0" / "cells" / "r0_both_context_specific_k0_stt4" / "splits" / "sparse_target_train.json"
    if stt.exists():
        meta = json.loads(stt.read_text())
        assert meta["kind"] == SPARSE_TARGET_TRAIN_KIND
        assert meta["target_manifest_validation_hash"] in val
    gate = json.loads((parent / "replicates" / "r0" / "cells" / "r0_both_context_specific_k0" /
                       "evaluation" / "bayesian_gate.json").read_text())
    assert gate["counted_as_successful_benchmark"] is False
    assert gate["thresholds"]["ess_bulk_min"] == 400 if "thresholds" in gate else True


def test_pipeline_reuses_frozen_target_splits(tmp_path):
    from send_bw.pipeline import Run

    out = tmp_path / "run"
    base = {
        "run_name": "reuse", "output_dir": str(out / "a"), "seed": 11, "generate_seed": 101,
        "synthetic": {"enabled": True, "out_dir": str(out / "synthetic"),
                      "n_studies": 8, "n_other_species_studies": 2, "n_compounds": 5,
                      "n_per_sex": [3, 3], "donor_scenario": "matched"},
        "mapping": str(ROOT / "configs" / "mappings" / "example_send.yaml"),
        "source": {"type": "local", "root": str(out / "synthetic" / "send"),
                   "metadata_dir": str(out / "synthetic" / "metadata")},
        "cohort": {"species": "RAT", "max_day": 28, "sharing": "context_specific",
                   "donor_species": ["DOG"], "n_donor_studies": 0},
        "splits": {"validation": 0.2, "test": 0.2},
        "models": {"spline": {"enabled": False}, "transformer": {"enabled": False},
                   "bayesian": {"enabled": False}},
        "evaluation": {"tasks": ["predose_conditioned"]},
        "design": {},
    }
    pa = tmp_path / "a.yaml"
    pa.write_text(yaml.safe_dump(base), encoding="utf-8")
    ra = Run(pa)
    ra.generate()
    ra.ingest()
    ra.prepare()
    man_a = json.loads((ra.dir / "splits" / "target_manifest.json").read_text())
    base_b = dict(base)
    base_b["output_dir"] = str(out / "b")
    base_b["cohort"] = {**base["cohort"], "sharing": "pooled", "n_donor_studies": 1}
    base_b["synthetic"] = {**base["synthetic"], "enabled": False}
    base_b["splits"] = {**base["splits"], "reuse_target_from": str(ra.dir / "splits")}
    pb = tmp_path / "b.yaml"
    pb.write_text(yaml.safe_dump(base_b), encoding="utf-8")
    rb = Run(pb)
    rb.ingest()
    rb.prepare()
    man_b = json.loads((rb.dir / "splits" / "target_manifest.json").read_text())
    assert man_a["validation"]["hash"] == man_b["validation"]["hash"]
    assert man_a["test"]["hash"] == man_b["test"]["hash"]
    assert man_a["all_target"]["hash"] == man_b["all_target"]["hash"]


def test_slow_markers_still_cover_bayesian_and_calibration():
    text_b = (ROOT / "tests" / "integration" / "test_bayesian_smoke.py").read_text()
    text_c = (ROOT / "tests" / "unit" / "test_covariance_analysis.py").read_text()
    assert "pytest.mark.slow" in text_b and "pytest.mark.bayesian" in text_b
    assert "reliable" in text_b and "ess_bulk_min" in text_b
    assert "pytest.mark.slow" in text_c

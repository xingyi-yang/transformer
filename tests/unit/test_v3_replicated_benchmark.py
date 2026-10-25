"""v3 replicated benchmark: regularized selective, sparse target-train, packs, seeds."""

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
    compose_train_with_donors,
    target_manifest,
)
from send_bw.evaluation.forecast import evaluation_pack
from send_bw.evaluation.transfer import transfer_uncertainty
from send_bw.models.design import MeanDesign
from send_bw.models.spline import species_penalty
from send_bw.models.transformer import BaselineTransformer
from send_bw.sharing import SELECTIVE_REGULARIZED, SELECTIVE_V1, is_selective, parse_sharing, uses_donors
from send_bw.v3_repl import ReplProtocolError, validate_repl_config

ROOT = Path(__file__).resolve().parents[2]


def test_parse_sharing_retains_v1_and_adds_regularized():
    assert parse_sharing("selective") == "selective"
    assert parse_sharing("selective_regularized") == "selective_regularized"
    assert is_selective("selective") and is_selective("selective_regularized")
    assert uses_donors("selective_regularized")
    assert SELECTIVE_V1["variant"] == "v1"
    assert SELECTIVE_REGULARIZED["variant"] == "regularized"
    assert "growth_deviation" in SELECTIVE_REGULARIZED["species_specific"]
    assert "treatment_deviation" in SELECTIVE_REGULARIZED["species_specific"]


def test_species_penalty_inflates_growth_and_treatment():
    names = ["tb[0]", "tb_SPECIES_DOG[0]", "eb_ld[0]", "eb_ld_SPECIES_DOG[0]", "eb_ld_D1_SPECIES_RAT[1]"]
    pen = species_penalty(names, alpha=2.0, lambda_growth=8.0, lambda_treatment=5.0)
    assert list(pen) == [2.0, 16.0, 2.0, 10.0, 10.0]


def test_mean_design_species_treatment_columns():
    F = pd.DataFrame({
        "TAU": [0.5, 1.0], "EXPO": [0.2, 0.4], "OFF": [0.0, 0.0],
        "SEX_M": [1.0, 0.0], "ZB_C": [0.0, 0.1], "LD": [1.0, 2.0],
        "D1": [0.1, 0.2], "D2": [0.0, 0.0], "D3": [0.0, 0.0],
        "INV_INTERVAL": [1.0, 1.0],
        "SPECIES_DOG": [0.0, 1.0], "STRAIN_BEAGLE": [0.0, 1.0],
        "ROUTE_ORAL GAVAGE": [1.0, 1.0],
    })
    d0 = MeanDesign.from_features(F, species_treatment=False)
    d1 = MeanDesign.from_features(F, species_treatment=True)
    X0 = d0.matrix(F, fit_scale=True)
    X1 = d1.matrix(F, fit_scale=True)
    assert X1.shape[1] > X0.shape[1]
    assert any(n.startswith("eb_ld_SPECIES_") for n in d1.col_names)
    assert any("D1_SPECIES_" in n for n in d1.col_names)
    d2 = MeanDesign.from_dict(d1.to_dict())
    assert d2.species_treatment is True


def test_transformer_treat_dev_default_loads_without_key():
    net = BaselineTransformer(group_dims={"bio": 2, "regimen": 3}, q_dim=5, treat_dev_dim=0)
    assert net.treat_dev is None
    net2 = BaselineTransformer(group_dims={"bio": 2, "species": 2, "regimen": 3}, q_dim=5, treat_dev_dim=5)
    assert net2.treat_dev is not None


def test_sparse_target_train_keeps_val_test_and_is_not_history():
    splits = pd.DataFrame({
        "STUDYID": [f"S{i:02d}" for i in range(8)],
        "SPLIT": ["train"] * 5 + ["validation", "test", "test"],
    })
    man0 = target_manifest(splits)
    out, meta = compose_sparse_target_train(splits, n_keep=3)
    assert meta["kind"] == SPARSE_TARGET_TRAIN_KIND
    assert meta["not"] == SPARSE_HISTORY_KIND
    assert meta["kind"] != meta["not"]
    assert meta["n_train_after"] == 3
    assert meta["kept_train_studyids"] == ["S00", "S01", "S02"]
    assert meta["dropped_train_studyids"] == ["S03", "S04"]
    assert meta["validation_unchanged"] and meta["test_unchanged"]
    man1 = target_manifest(out)
    assert man1["validation"]["hash"] == man0["validation"]["hash"]
    assert man1["test"]["hash"] == man0["test"]["hash"]
    assert man1["train"]["hash"] != man0["train"]["hash"]
    composed = compose_train_with_donors(out, ["D01", "D02"], k=1)
    val_test = set(composed.loc[composed["SPLIT"].isin(["validation", "test"]), "STUDYID"])
    assert val_test == {"S05", "S06", "S07"}
    assert "D01" in set(composed.loc[composed["SPLIT"] == "train", "STUDYID"])
    with pytest.raises(ValueError, match="n_keep"):
        compose_sparse_target_train(splits, n_keep=1)


def test_evaluation_pack_reports_coverage_widths_contrast_temporal():
    result = {
        "summary": {
            "task_alias": "predose_conditioned", "model": "spline", "split": "test",
            "observation_weighted": {"cov80": 0.7, "cov95": 0.9, "width80_log": 0.2, "width95_log": 0.4},
            "study_balanced": {"rmse_log": 0.05, "crps": 0.03, "cov80": 0.71, "cov95": 0.91,
                               "width80": 0.21, "width95": 0.41},
            "group_mean_cov90": 0.8,
            "between_animal_var_ratio_pred_over_obs": 1.1,
            "contrast_cov90": 0.6, "contrast_width90": 0.15, "contrast_bias": 0.01, "n_contrasts": 3,
            "study_level_contrast_error": {"bias_study_mean": 0.02, "rmse_study_mean": 0.03,
                                           "n_studies_with_contrast": 2},
            "pit_histogram_10bins": [1] * 10,
            "increment_sd_ratio_pred_over_obs": 1.2,
            "extrema_max_cov90": 0.5,
            "correlation_by_gap": [{"gap_bin": "(0,4]", "corr_obs": 0.4, "corr_pred": 0.3, "n_pairs": 4}],
            "distributional": {"cov80": 0.71},
            "effective_sample_sizes": {"n_studies": 2, "n_animals": 8, "n_obs": 20},
        },
        "per_study": pd.DataFrame(),
    }
    pack = evaluation_pack(result)
    assert pack["status"] == "completed"
    assert pack["individual"]["coverage80"] == 0.71
    assert pack["individual"]["width80_log"] == 0.21
    assert pack["group_mean"]["coverage90"] == 0.8
    assert pack["contrast"]["study_level"]["rmse_study_mean"] == 0.03
    assert pack["temporal"]["pit_histogram_10bins"] == [1] * 10
    assert "parameter_uncertainty" in pack["uncertainty_limitations"]


def test_transfer_uncertainty_separates_seeds_and_allows_harmful():
    rows = [
        {"model": "spline", "sharing": "pooled", "metric": "rmse_log", "task": "predose_conditioned",
         "sparse_target_train": "full", "delta": 0.01, "sign": "harmful",
         "generate_seed": 1, "train_seed": 10},
        {"model": "spline", "sharing": "pooled", "metric": "rmse_log", "task": "predose_conditioned",
         "sparse_target_train": "full", "delta": -0.002, "sign": "helpful",
         "generate_seed": 2, "train_seed": 27},
    ]
    unc = transfer_uncertainty(rows)
    assert unc["n_groups"] == 1
    g = unc["groups"][0]
    assert g["n_harmful"] == 1 and g["n_helpful"] == 1
    assert g["sd_delta"] is not None
    assert g["generate_seeds"] == [1, 2]
    assert g["train_seeds"] == [10, 27]
    assert "not a license" in unc["note"].lower() or "favorable" in unc["note"]


def test_repl_config_refuses_design_real_data_and_identical_seeds():
    cfg = yaml.safe_load((ROOT / "configs" / "v3_repl.frozen.yaml").read_text(encoding="utf-8"))
    validate_repl_config(cfg)
    assert cfg["generate_seed"] != cfg["train_seed"]
    bad = dict(cfg)
    bad["design"] = {"request": "configs/new_study.smoke.yaml"}
    with pytest.raises(ReplProtocolError, match="design"):
        validate_repl_config(bad)
    snow = dict(cfg)
    snow["source"] = {"type": "snowflake"}
    with pytest.raises(ReplProtocolError, match="real-data"):
        validate_repl_config(snow)
    same = dict(cfg)
    same["train_seed"] = same["generate_seed"]
    with pytest.raises(ReplProtocolError, match="differ"):
        validate_repl_config(same)


def test_pipeline_sparse_target_train_fixes_val_test(tmp_path):
    from send_bw.pipeline import Run

    out = tmp_path / "run"
    cfg = {
        "run_name": "stt", "output_dir": str(out), "seed": 9, "generate_seed": 101,
        "synthetic": {"enabled": True, "out_dir": str(out / "synthetic"),
                      "n_studies": 10, "n_other_species_studies": 2, "n_compounds": 5,
                      "n_per_sex": [3, 3], "donor_scenario": "both"},
        "mapping": str(ROOT / "configs" / "mappings" / "example_send.yaml"),
        "source": {"type": "local", "root": str(out / "synthetic" / "send"),
                   "metadata_dir": str(out / "synthetic" / "metadata")},
        "cohort": {"species": "RAT", "max_day": 28, "sharing": "selective_regularized",
                   "donor_species": ["DOG"], "n_donor_studies": 1},
        "splits": {"validation": 0.2, "test": 0.2, "sparse_target_train_n": 3},
        "models": {"spline": {"enabled": False}, "transformer": {"enabled": False},
                   "bayesian": {"enabled": False}},
        "evaluation": {"tasks": ["predose_conditioned"]},
        "design": {},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run = Run(p)
    run.generate()
    run.ingest()
    run.prepare()
    man = json.loads((run.dir / "splits" / "target_manifest.json").read_text())
    stt = json.loads((run.dir / "splits" / "sparse_target_train.json").read_text())
    assert stt["kind"] == SPARSE_TARGET_TRAIN_KIND
    assert stt["not"] == SPARSE_HISTORY_KIND
    assert stt["n_train_after"] == 3
    assert stt["target_manifest_validation_hash"] == man["validation"]["hash"]
    assert stt["target_manifest_test_hash"] == man["test"]["hash"]
    splits = pd.read_csv(run.dir / "splits" / "splits.csv", dtype=str)
    val_test = set(splits.loc[splits["SPLIT"].isin(["validation", "test"]), "STUDYID"])
    assert set(stt["validation_studyids"] + stt["test_studyids"]) == val_test
    assert set(stt["dropped_train_studyids"]).isdisjoint(val_test)
    spec = json.loads((run.dir / "features" / "feature_spec.json").read_text())
    assert spec["sharing"] == "selective_regularized"


def test_prior_executed_artifacts_are_untouched():
    assert (ROOT / "runs" / "v3_eval" / "SUMMARY.json").exists()
    assert (ROOT / "runs" / "v3_pilot_k0k1").exists()
    eval_sum = json.loads((ROOT / "runs" / "v3_eval" / "SUMMARY.json").read_text())
    assert eval_sum.get("retuned_after_test") is False

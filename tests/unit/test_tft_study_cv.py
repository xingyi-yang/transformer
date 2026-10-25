"""Study-held-out protocol, confirm-34011 erratum, epoch and CQR counting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from send_bw.data.features import FeatureSpec
from send_bw.models.tft import count_completed_epochs
from send_bw.tft_audit import (
    CONFIRM_FORBIDDEN_PLACED_IN_TRAIN,
    confirm_34011_erratum,
    cqr_calibration_units,
    cqr_metric_policy,
    epoch_budget_versus_logged,
    quantile_loss_scale_note,
    series_versus_animals,
    tft_feature_applicability,
)
from send_bw.tft_early_hist_extend import load_yaml
from send_bw.tft_intervals import apply_cqr_margin, score_raw_and_calibrated
from send_bw.tft_study_cv import (
    LOCKED_DIRS,
    OUTPUT_DEFAULT,
    REFERENCE,
    StudyCVError,
    _refuse_locked,
    consistency_across_studies,
    make_cv_folds,
    select_on_validation,
    verify_fold_membership,
)


def test_config_predeclares_folds_and_reference():
    cfg = load_yaml("configs/tft_study_cv.yaml")
    ev = cfg["benchmark"]
    assert ev["stability_seeds"] == [82001, 82021, 82031]
    assert ev["cv_split_seed"] == 35021
    assert ev["n_folds"] == 6
    assert ev["n_cal_studies"] == 2
    assert cfg["output_dir"] == OUTPUT_DEFAULT
    assert cfg["output_dir"] != cfg["stability_dir"]
    assert cfg["selection"]["development_reference"] == "tft_target_only"
    assert cfg["calibration"]["split"] == "calibration"
    assert cfg["consistency"]["min_studies"] == 10


def test_refuses_locked_dirs(tmp_path):
    with pytest.raises(StudyCVError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_stability"))
    with pytest.raises(StudyCVError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_primary"))
    _refuse_locked(tmp_path / "tft_study_cv")
    assert "runs/tft_stability" in LOCKED_DIRS


def test_confirm_34011_is_not_untouched():
    err = confirm_34011_erratum()
    assert err["do_not_call_untouched"] is True
    assert set(err["forbidden_holdouts_in_train"]) == set(CONFIRM_FORBIDDEN_PLACED_IN_TRAIN)
    assert "SYNA-010" in err["splits"]["train"]
    assert "SYNA-003" in err["splits"]["test"]
    assert "untouched" in err["unsupported_claim"].lower()


def test_series_count_drops_single_visit_animals():
    animals = pd.DataFrame({
        "USUBJID": ["a", "b", "c"], "STUDYID": ["S1", "S1", "S2"],
    })
    obs = pd.DataFrame({
        "USUBJID": ["a", "a", "b", "c"],
        "STUDYID": ["S1", "S1", "S1", "S2"],
        "DAY": [7.0, 14.0, 7.0, 7.0],
        "LOGW": [5.0, 5.1, 5.0, 5.2],
    })
    rec = series_versus_animals(animals, obs, min_encoder_length=1)
    assert rec["n_animals_in_split"] == 3
    assert rec["n_tft_series"] == 1
    assert rec["n_dropped_short_series"] == 2
    assert {d["USUBJID"] for d in rec["dropped"]} == {"b", "c"}


def test_epoch_budget_is_not_current_epoch_plus_one():
    rows = [{"epoch": i, "train_loss": 0.1} for i in range(16)]
    rows = [{"epoch": 0, "sanity_check_loss": 1.0}] + rows
    rec = epoch_budget_versus_logged(rows, max_epochs=16, lightning_current_epoch=16)
    assert rec["epochs_completed"] == 16
    assert rec["epochs_run_legacy_plus_one"] == 17
    assert count_completed_epochs(rows, 16)["epochs_completed"] == 16


def test_cqr_pinball_uses_shifted_endpoints_and_keeps_raw():
    fut = pd.DataFrame({
        "USUBJID": ["a", "a"], "STUDYID": ["S1", "S1"],
        "DAY": [21.0, 28.0], "LOGW": [5.0, 5.2],
    })
    raw = pd.DataFrame({
        "USUBJID": ["a", "a"], "DAY": [21.0, 28.0],
        "q0.1": [4.9, 5.1], "q0.25": [4.95, 5.14], "q0.5": [5.0, 5.18],
        "q0.75": [5.04, 5.20], "q0.9": [5.08, 5.24],
    })
    cal = apply_cqr_margin(raw, 0.05)
    pack = score_raw_and_calibrated(fut, raw, cal)
    assert pack["raw"]["mae_log"] == pytest.approx(pack["cqr"]["mae_log"])
    assert pack["cqr"]["mean_quantile_loss"] != pack["raw"]["mean_quantile_loss"]
    assert pack["cqr"]["interval_width80"] == pytest.approx(
        pack["raw"]["interval_width80"] + 0.10)
    assert "shifted" in (pack["pinball_note"] or "").lower()
    assert cqr_metric_policy()["experimental"] is True


def test_folds_are_disjoint_and_each_study_is_test_once():
    ids = [f"SYNA-{i:03d}" for i in range(1, 13)]
    folds = make_cv_folds(ids, seed=35021, n_folds=6, n_val=2, n_cal=2)
    assert len(folds) == 6
    tests = [s for f in folds for s in f["test"]]
    assert sorted(tests) == ids
    assert folds[0]["test"] == ["SYNA-001", "SYNA-003"]
    for f in folds:
        verify_fold_membership(f)
        parts = [set(f[k]) for k in ("train", "validation", "calibration", "test")]
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                assert not (parts[i] & parts[j])


def test_select_on_validation_ignores_test_and_keeps_reference():
    rows = [
        {"bank": "validation", "variant": "tft_target_only", "interval_score80": 0.20},
        {"bank": "validation", "variant": "tft_finetune", "interval_score80": 0.10},
        {"bank": "test", "variant": "tft_finetune", "interval_score80": 0.99},
        {"bank": "test", "variant": "tft_target_only", "interval_score80": 0.01},
    ]
    dec = select_on_validation(rows)
    assert dec["lowest_validation"] == "tft_finetune"
    assert dec["reference"] == REFERENCE
    assert dec["test_scores_examined"] is False
    assert dec["validation_preferred_experimental"] == "tft_finetune"


def test_consistency_requires_majority_of_studies_and_folds():
    rows = []
    for fold, studies in enumerate((["A", "B"], ["C", "D"])):
        for sid in studies:
            rows.append({
                "bank": "test", "fold": fold, "variant": "tft_target_only",
                "mae_log": 0.04, "per_study": [{"STUDYID": sid, "mae_log": 0.04}],
            })
            rows.append({
                "bank": "test", "fold": fold, "variant": "tft_finetune",
                "mae_log": 0.08, "per_study": [{"STUDYID": sid, "mae_log": 0.08}],
            })
    rec = consistency_across_studies(rows, min_folds=2, min_studies=4)
    assert rec["previous_finetune_default_failed_confirmation"] is True
    assert rec["any_consistent_improvement"] is False
    assert rec["variants"]["tft_finetune"]["n_studies_beats_reference"] == 0


def test_audit_notes_distinguish_cqr_and_quantile_scales():
    pol = cqr_metric_policy()
    assert "validation" in pol["calibration_data"].lower()
    assert "pair-level" in pol["calibration_data"]
    units = cqr_calibration_units()
    assert units["study_balanced"] is False
    assert units["unit"].startswith("aligned")
    scale = quantile_loss_scale_note()
    assert scale["not_comparable_without_rescaling"] is True
    assert "2 * pinball" in scale["training"]


def test_tft_applicability_is_recorded_not_enforced():
    spec = FeatureSpec(
        species="RAT", strains=["SPRAGUE-DAWLEY"], routes=["ORAL GAVAGE"],
        sexes=["M", "F"], zb_mean={"M": 5.5, "F": 5.2}, max_dose=300.0,
        max_day=28.0, intervals=[1.0], compound_ids=["CMP03"],
        descriptor_ranges={"D1": [-1.0, 1.0], "D2": [-1.0, 1.0], "D3": [-1.0, 1.0]},
        animal_cols=["SEX_M", "ZB_C", "IS_CONTROL", "LD"],
        time_cols=["TAU", "EXPO", "OFF", "LCUM"],
    )
    animals = pd.DataFrame({
        "USUBJID": ["u1"], "STUDYID": ["SYNA-010"], "SEX": ["M"],
        "STRAIN": ["SPRAGUE-DAWLEY"], "ROUTE": ["ORAL GAVAGE"],
        "SPECIES": ["RAT"], "COMPOUND_ID": ["CMP03"], "IS_CONTROL": [False],
        "PLANNED_DOSE": [1000.0], "INTERVAL_DAYS": [7.0],
        "START_DAY": [0.0], "END_DAY": [27.0], "BASELINE_LOGW": [5.6],
    })
    cref = pd.DataFrame({"COMPOUND_ID": ["CMP03"], "D1": [0.0], "D2": [0.0], "D3": [0.0]})
    rec = tft_feature_applicability(animals, spec, cref)
    assert rec["enforced_by_tft_scoring"] is False
    assert rec["n_problems"] >= 2
    joined = " ".join(rec["problems"])
    assert "1000" in joined or "dose" in joined
    assert "7" in joined or "interval" in joined

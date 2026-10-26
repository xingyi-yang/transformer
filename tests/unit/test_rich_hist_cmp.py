"""History-length comparison: families, inventory, encoder, protocol, leakage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from send_bw.rich_hist import LABEL
from send_bw.rich_hist_cmp import (
    FULL_ENCODER_CAP,
    OUTPUT_DEFAULT,
    PopulationOnlyModel,
    _predict_last_value,
    _refuse_locked,
    assign_families,
    corrected_family_splits,
    encoder_consumption,
    execute_rich_hist_cmp,
    frozen_protocol,
    reconcile_inventory,
    verify_family_partitions,
)
from send_bw.tft_early_hist_extend import load_yaml

BANK = Path("runs/rich_hist_v1")


def test_config_freezes_protocol_and_does_not_regenerate():
    cfg = load_yaml("configs/rich_hist_cmp.yaml")
    assert cfg["bank_dir"] == "runs/rich_hist_v1"
    assert cfg["output_dir"] == OUTPUT_DEFAULT
    assert cfg["output_dir"] != cfg["bank_dir"]
    assert cfg["generate_seed"] == 91021
    assert cfg["split_seed"] == 92031
    assert cfg["train_seed"] == 93041
    assert cfg["train_seed"] not in {71001, 81011, 83011, 91021, 92031}
    assert cfg["history_cmp"]["tuned_from_test"] is False
    assert cfg["history_cmp"]["bank_regenerated"] is False
    assert cfg["history_cmp"]["full_encoder_cap"] == 40
    tft = cfg["models"]["optional"]["tft"]
    assert tft["enabled"] is True
    assert tft["hidden_size"] == 16
    assert tft["lr"] == 0.01
    assert tft["max_epochs"] == 16
    assert tft["window_balance"] == "study"
    assert "max_encoder_length" not in tft
    proto = frozen_protocol(cfg)
    assert proto["tuned_from_test"] is False
    assert proto["bank"]["regenerated"] is False
    assert proto["full_encoder_cap"] == FULL_ENCODER_CAP
    assert proto["matched_architecture_B_D"]["separately_trained"] is True
    assert proto["matched_architecture_B_D"]["not_inference_only_truncation"] is True
    assert proto["baseline_weight"]["isolates"] == "longitudinal history, not baseline access"
    assert LABEL.startswith("Synthetic methodological benchmark")


def test_refuses_locked_and_source_bank():
    with pytest.raises(Exception, match="preserved|overwrite|refusing"):
        _refuse_locked(Path("runs/rich_hist_v1"))
    with pytest.raises(Exception, match="preserved|overwrite|refusing"):
        _refuse_locked(Path("runs/tft_longer_ctx"))
    with pytest.raises(Exception, match="preserved|overwrite|refusing"):
        _refuse_locked(Path("runs/v3_share_repl"))
    _refuse_locked(Path("runs/rich_hist_cmp"))
    assert OUTPUT_DEFAULT == "runs/rich_hist_cmp"


def test_execute_refuses_colliding_train_seed(tmp_path):
    cfg = load_yaml("configs/rich_hist_cmp.yaml")
    cfg = dict(cfg)
    cfg["train_seed"] = 91021
    cfg["output_dir"] = str(tmp_path / "x")
    with pytest.raises(Exception, match="collides"):
        execute_rich_hist_cmp(cfg, force=True)


def test_families_do_not_cross_when_pairs_share_a_partition():
    scenarios = pd.DataFrame({
        "STUDYID": ["A", "B", "C", "D"],
        "PAIR_ID": ["P1", "P1", None, None],
    })
    splits = pd.DataFrame({
        "STUDYID": ["A", "B", "C", "D"],
        "SPLIT": ["train", "train", "validation", "test"],
    })
    assigned = assign_families(scenarios, splits)
    check = verify_family_partitions(assigned)
    assert check["passed"] is True
    assert check["n_independent_families"] == 3
    assert check["n_crossing_families"] == 0
    assert assigned.loc[assigned["STUDYID"] == "A", "trajectory_family_id"].iloc[0] == "P1"
    assert assigned.loc[assigned["STUDYID"] == "C", "trajectory_family_id"].iloc[0] == "C"


def test_crossing_families_are_detected_and_corrected():
    scenarios = pd.DataFrame({
        "STUDYID": [f"S{i}" for i in range(8)],
        "PAIR_ID": ["P1", "P1"] + [None] * 6,
    })
    splits = pd.DataFrame({
        "STUDYID": [f"S{i}" for i in range(8)],
        "SPLIT": ["train", "test", "train", "validation", "calibration", "test", "train", "validation"],
    })
    assigned = assign_families(scenarios, splits)
    check = verify_family_partitions(assigned)
    assert check["passed"] is False
    assert "P1" in check["crossing_families"]
    corr = corrected_family_splits(assigned, seed=92031)
    reassigned = assign_families(scenarios, corr)
    fixed = verify_family_partitions(reassigned)
    assert fixed["passed"] is True
    pair = reassigned.loc[reassigned["trajectory_family_id"] == "P1", "SPLIT"]
    assert pair.nunique() == 1


def test_full_encoder_keeps_every_available_visit():
    examples = pd.DataFrame({
        "USUBJID": ["U1", "U1"],
        "STUDYID": ["S1", "S1"],
        "origin": [14.0, 56.0],
        "target_day": [21.0, 63.0],
        "last_observed_day": [14.0, 56.0],
        "lead_from_origin": [7.0, 7.0],
        "lead_from_last_observation": [7.0, 7.0],
        "n_encoder_available": [5, 12],
        "encoder_days": ["[-7, 0, 7, 10, 14]",
                         "[-7, 0, 7, 14, 21, 28, 35, 42, 49, 52, 54, 56]"],
    })
    full = encoder_consumption(examples, history_kind="full", max_encoder_length=FULL_ENCODER_CAP)
    assert full["full_history_keeps_all"] is True
    assert full["n_truncated_pairs"] == 0
    assert full["mean_consumed"] == pytest.approx(8.5)
    short = encoder_consumption(examples, history_kind="short", max_encoder_length=2)
    assert short["mean_consumed"] == pytest.approx(2.0)
    latest = encoder_consumption(examples, history_kind="latest", max_encoder_length=1)
    assert latest["mean_consumed"] == pytest.approx(1.0)
    legacy = encoder_consumption(examples, history_kind="full", max_encoder_length=8)
    assert legacy["n_truncated_pairs"] == 1
    assert legacy["silent_full_truncation_pairs"] == 1


def test_inventory_reports_dead_animals_missing_from_windows():
    animals = pd.DataFrame({
        "USUBJID": ["A", "B", "C"],
        "STUDYID": ["S1", "S1", "S2"],
        "DISPOSITION": ["COMPLETED", "DEATH", "UNSCHEDULED_REMOVAL"],
    })
    examples = pd.DataFrame({"USUBJID": ["A"], "STUDYID": ["S1"]})
    exclusions = pd.DataFrame({
        "USUBJID": ["B", "B", "C"],
        "exclude_reason": ["already_dead_at_origin", "target_day_not_observed",
                           "already_dead_at_origin"],
    })
    splits = pd.DataFrame({"STUDYID": ["S1", "S2"], "SPLIT": ["train", "test"]})
    inv = reconcile_inventory(animals, examples, exclusions, splits)
    assert inv["n_cohort_animals"] == 3
    assert inv["n_window_animals"] == 1
    assert inv["n_missing_from_windows"] == 2
    assert inv["missing_by_split"] == {"train": 1, "test": 1}
    assert "already_dead_at_origin" in str(inv["missing_by_reason"])


def test_last_value_is_origin_specific_and_does_not_leak_later_visits():
    examples = pd.DataFrame({
        "USUBJID": ["U1", "U1"],
        "STUDYID": ["S1", "S1"],
        "origin": [14.0, 56.0],
        "target_day": [21.0, 63.0],
        "encoder_days": ["[0, 7, 14]", "[0, 7, 14, 28, 56]"],
        "encoder_logw": ["[5.0, 5.1, 5.2]", "[5.0, 5.1, 5.2, 5.4, 5.8]"],
    })
    pred = _predict_last_value(examples)
    early = pred.loc[pred["DAY"] == 21.0, "q0.5"].iloc[0]
    late = pred.loc[pred["DAY"] == 63.0, "q0.5"].iloc[0]
    assert early == pytest.approx(5.2)
    assert late == pytest.approx(5.8)
    assert early != late


def test_population_only_drops_individual_weight_features():
    pop = PopulationOnlyModel()
    animals = pd.DataFrame({
        "USUBJID": ["U1"], "STUDYID": ["S1"], "SEX": ["M"], "SPECIES": ["RAT"],
        "STRAIN": ["SD"], "ROUTE": ["ORAL GAVAGE"], "COMPOUND_ID": ["CMP01"],
        "IS_CONTROL": [True], "PLANNED_DOSE": [0.0], "INTERVAL_DAYS": [7.0],
        "START_DAY": [1.0], "END_DAY": [28.0], "BASELINE_LOGW": [5.4], "BASELINE_DAY": [0.0],
    })
    obs = pd.DataFrame({
        "STUDYID": ["S1"], "USUBJID": ["U1"], "DAY": [14.0], "LOGW": [5.5],
    })
    from send_bw.data.features import FeatureSpec

    spec = FeatureSpec(
        species="RAT", strains=["SD"], routes=["ORAL GAVAGE"], sexes=["M"],
        zb_mean={"M": 5.4}, max_dose=100.0, max_day=28.0,
        intervals=[7.0], compound_ids=["CMP01"],
        descriptor_ranges={"D1": [0.0, 1.0], "D2": [0.0, 1.0], "D3": [0.0, 1.0]},
        animal_cols=["SEX_M", "ZB_C", "IS_CONTROL", "LD", "INV_INTERVAL", "START", "END",
                     "D1", "D2", "D3"],
        time_cols=["TAU", "EXPO", "OFF", "LCUM"],
    )
    cref = pd.DataFrame({"COMPOUND_ID": ["CMP01"], "D1": [0.1], "D2": [0.2], "D3": [0.3]})
    X, y, _w = pop._X(animals, obs, spec, cref)
    assert "ZB_C" not in pop.cols
    assert "LOGW" not in pop.cols
    assert X.shape[1] == len(pop.cols)
    assert y[0] == pytest.approx(5.5)


@pytest.mark.skipif(not (BANK / "windows" / "full_examples.csv").is_file(),
                    reason="frozen rich_hist_v1 bank is not present")
def test_bank_families_and_inventory_match_preflight():
    scenarios = pd.read_csv(BANK / "study_scenarios.csv")
    splits = pd.read_csv(BANK / "splits" / "target_splits.csv")
    assigned = assign_families(scenarios, splits)
    check = verify_family_partitions(assigned)
    assert check["passed"] is True
    assert check["n_target_studies"] == 38
    assert check["n_independent_families"] == 32
    assert check["n_crossing_families"] == 0
    assert check["families_by_split"] == {
        "train": 16, "validation": 5, "calibration": 5, "test": 6,
    }
    animals = pd.read_csv(BANK / "cohort_animals.csv")
    examples = pd.read_csv(BANK / "windows" / "full_examples.csv")
    exclusions = pd.read_csv(BANK / "windows" / "full_exclusions.csv")
    inv = reconcile_inventory(animals, examples, exclusions, splits)
    assert inv["n_cohort_animals"] == 1338
    assert inv["n_window_animals"] == 1332
    assert inv["n_missing_from_windows"] == 6
    full = encoder_consumption(examples, history_kind="full", max_encoder_length=FULL_ENCODER_CAP)
    assert full["full_history_keeps_all"] is True
    legacy = encoder_consumption(examples, history_kind="full", max_encoder_length=8)
    assert legacy["n_truncated_pairs"] > 0

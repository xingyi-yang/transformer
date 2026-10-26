"""Larger-bank comparison: locked knobs, new generate seed, locked dirs."""

from __future__ import annotations

from pathlib import Path

import pytest

from send_bw.tft_early_hist_extend import load_yaml
from send_bw.tft_larger_bank import (
    EXPECTED_LOCKED,
    LOCKED_DIRS,
    OFFICIAL_GENERATE_SEED,
    OUTPUT_DEFAULT,
    REFERENCE,
    LargerBankError,
    _refuse_locked,
    assert_locked_knobs,
    official_reference,
)


def test_config_predeclares_larger_bank_and_locked_knobs():
    cfg = load_yaml("configs/tft_larger_bank.yaml")
    syn = cfg["synthetic"]
    ev = cfg["larger_bank"]
    tft = cfg["models"]["optional"]["tft"]
    assert cfg["generate_seed"] == 81011
    assert cfg["train_seed"] == 83011
    assert cfg["generate_seed"] != OFFICIAL_GENERATE_SEED
    assert cfg["generate_seed"] != cfg["train_seed"]
    assert syn["generator"] == "A"
    assert syn["n_studies"] == 36
    assert syn["n_other_species_studies"] == 6
    assert syn["n_compounds"] == 8
    assert syn["n_per_sex"] == [3, 4]
    assert ev["n_donor_studies"] == 6
    assert ev["search"] is False
    assert ev["development_reference"] == "tft_target_only"
    assert ev["official_generate_seed"] == 71001
    assert cfg.get("search") in (None, False, {})
    assert cfg["output_dir"] == OUTPUT_DEFAULT
    assert cfg["output_dir"] != cfg["official_primary_dir"]
    for key, expected in EXPECTED_LOCKED.items():
        assert tft[key] == expected
    assert tft["max_encoder_length"] == 8
    assert tft["max_prediction_length"] == 6
    assert tft["encoder_window_day"] == 14
    assert cfg["finetune"]["lr"] == 0.001
    assert cfg["finetune"]["max_epochs"] == 8
    assert cfg["labels"] == [
        "Synthetic diagnostic.",
        "Not design qualification.",
        "No real-data validation.",
        "No operational recommendation.",
    ]


def test_refuses_locked_and_official_dirs(tmp_path):
    with pytest.raises(LargerBankError, match="preserved"):
        _refuse_locked(Path("runs/v3_official_seq"))
    with pytest.raises(LargerBankError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_primary"))
    with pytest.raises(LargerBankError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_study_cv"))
    with pytest.raises(LargerBankError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/forecast_method_audit"))
    _refuse_locked(tmp_path / "tft_larger_bank")
    assert "runs/tft_primary" in LOCKED_DIRS
    assert "runs/tft_study_cv" in LOCKED_DIRS
    assert "runs/forecast_method_audit" in LOCKED_DIRS
    assert OUTPUT_DEFAULT not in LOCKED_DIRS


def test_assert_locked_knobs_rejects_search_and_retune():
    cfg = load_yaml("configs/tft_larger_bank.yaml")
    assert assert_locked_knobs(cfg)["lr"] == 0.01
    bad = dict(cfg)
    bad["search"] = {"order": [{"key": "lr", "values": [0.001]}]}
    with pytest.raises(LargerBankError, match="search"):
        assert_locked_knobs(bad)
    drifted = load_yaml("configs/tft_larger_bank.yaml")
    drifted["models"]["optional"]["tft"]["lr"] = 0.003
    with pytest.raises(LargerBankError, match="lr"):
        assert_locked_knobs(drifted)


def test_official_reference_is_read_only():
    rec = official_reference()
    assert rec["read_only"] is True
    assert rec.get("used_to_retune") is False
    if rec["status"] == "ok":
        assert rec["generate_seed"] == 71001
        assert "tft_target_only" in rec["results"]
        mae = rec["results"]["tft_target_only"]["mae_log"]
        assert abs(float(mae["study_balanced"]) - 0.0499) < 5e-4
    assert REFERENCE == "tft_target_only"

"""Longer-context encoder: pretreatment attach, locked knobs, locked dirs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from send_bw.data.features import FeatureSpec
from send_bw.models.base import ModelData
from send_bw.tft_early_hist_extend import load_yaml
from send_bw.tft_examples import construct_forecast_examples
from send_bw.tft_larger_bank import EXPECTED_LOCKED, assert_locked_knobs
from send_bw.tft_longer_ctx import (
    LOCKED_DIRS,
    OUTPUT_DEFAULT,
    LongerCtxError,
    _refuse_locked,
    attach_pretreatment_obs,
    encoder_day_inventory,
    pretreatment_mask,
)


def _spec() -> FeatureSpec:
    return FeatureSpec(
        species="RAT", strains=["SD"], routes=["ORAL GAVAGE"], sexes=["M"],
        zb_mean={"M": 5.5}, max_dose=100.0, max_day=28.0, intervals=[1.0],
        compound_ids=["CMP01"], descriptor_ranges={"D1": [0.0, 1.0], "D2": [0.0, 1.0], "D3": [0.0, 1.0]},
        sharing="context_specific", species_levels=["RAT"], zb_mean_by_species={},
        declared_species=["RAT"], fixed_schema=False,
    )


def test_config_predeclares_pretreatment_and_locked_knobs():
    cfg = load_yaml("configs/tft_longer_ctx.yaml")
    ev = cfg["longer_ctx"]
    tft = cfg["models"]["optional"]["tft"]
    assert ev["include_pretreatment_encoder"] is True
    assert ev["encoder_window_day"] == 14
    assert ev["search"] is False
    assert ev["development_reference"] == "tft_target_only"
    assert cfg["generate_seed"] == 81011
    assert cfg["train_seed"] == 83011
    assert cfg["output_dir"] == OUTPUT_DEFAULT
    assert cfg["output_dir"] != cfg["larger_bank_dir"]
    assert "runs/tft_larger_bank" in ev["source_cells"]["target_only"]
    for key, expected in EXPECTED_LOCKED.items():
        assert tft[key] == expected
    assert tft["max_encoder_length"] == 8
    assert_locked_knobs(cfg)


def test_refuses_locked_dirs(tmp_path):
    with pytest.raises(LongerCtxError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_larger_bank"))
    with pytest.raises(LongerCtxError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_primary"))
    _refuse_locked(tmp_path / "tft_longer_ctx")
    assert "runs/tft_larger_bank" in LOCKED_DIRS
    assert OUTPUT_DEFAULT not in LOCKED_DIRS


def test_attach_adds_observed_predose_not_recovery():
    animals = pd.DataFrame({
        "USUBJID": ["A"], "STUDYID": ["S1"], "SEX": ["M"], "SPECIES": ["RAT"],
        "STRAIN": ["SD"], "ROUTE": ["ORAL GAVAGE"], "COMPOUND_ID": ["CMP01"],
        "IS_CONTROL": [True], "PLANNED_DOSE": [0.0], "INTERVAL_DAYS": [1.0],
        "START_DAY": [0.0], "END_DAY": [28.0], "BASELINE_LOGW": [5.4], "BASELINE_DAY": [0.0],
    })
    obs = pd.DataFrame({
        "STUDYID": ["S1"] * 4, "USUBJID": ["A"] * 4, "DAY": [7.0, 14.0, 21.0, 28.0],
        "LOGW": [5.50, 5.55, 5.60, 5.62],
        "PHASE": ["TREATMENT"] * 4, "IS_BASELINE": [False] * 4,
        "BASELINE_LOGW": [5.4] * 4, "BASELINE_DAY": [0.0] * 4, "Y": [0.10, 0.15, 0.20, 0.22],
    })
    canonical = pd.DataFrame({
        "STUDYID": ["S1"] * 5, "USUBJID": ["A"] * 5,
        "DAY": [-7.0, 0.0, 7.0, 14.0, 35.0],
        "LOGW": [5.30, 5.40, 5.50, 5.55, 5.80],
        "PHASE": ["PRETREATMENT", "PRETREATMENT", "TREATMENT", "TREATMENT", "RECOVERY"],
        "IS_BASELINE": [False, True, False, False, False],
    })
    data = ModelData(animals, obs, pd.DataFrame({"COMPOUND_ID": ["CMP01"]}), _spec())
    attached, rec = attach_pretreatment_obs(data, canonical)
    days = sorted(attached.obs["DAY"].tolist())
    assert days == [-7.0, 0.0, 7.0, 14.0, 21.0, 28.0]
    assert rec["n_rows_added"] == 2
    assert rec["recovery_added"] is False
    assert rec["invented_days"] is False
    pack = construct_forecast_examples(
        attached.obs, attached.animals, cutoff=14.0,
        max_encoder_length=8, max_prediction_length=6)
    enc = pack["examples"].iloc[0]["encoder_days"]
    assert enc == [-7.0, 0.0, 7.0, 14.0]
    assert pack["examples"].iloc[0]["target_days"] == [21.0, 28.0]


def test_pretreatment_mask_excludes_recovery():
    obs = pd.DataFrame({
        "DAY": [-7.0, 14.0, 35.0],
        "PHASE": ["PRETREATMENT", "TREATMENT", "RECOVERY"],
        "IS_BASELINE": [False, False, False],
    })
    assert pretreatment_mask(obs).tolist() == [True, False, False]


def test_encoder_inventory_counts_patterns():
    examples = pd.DataFrame({
        "encoder_days": ["[7.0, 14.0]", "[-7.0, 0.0, 7.0, 14.0]"],
        "n_encoder_used": [2, 4],
    })
    inv = encoder_day_inventory(examples)
    assert inv["n_animals"] == 2
    assert inv["n_encoder_used"] == {"2": 1, "4": 1}
    assert -7.0 in inv["unique_encoder_days"]

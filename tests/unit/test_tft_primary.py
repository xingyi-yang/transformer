"""Primary TFT forecasting workflow: labels, normalizer, search, locked dirs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from send_bw.models.tft import (
    STATIC_CATS,
    build_target_normalizer,
    cat_label,
    uses_target_scales,
)
from send_bw.tft_early_hist_extend import load_yaml
from send_bw.tft_primary import (
    LOCKED_DIRS,
    OUTPUT_DEFAULT,
    ForecastError,
    _refuse_locked,
    diagnose_locked_pooled_collapse,
    select_by_val_loss,
)

PF_INSTALLED = importlib.util.find_spec("pytorch_forecasting") is not None


def test_config_predeclares_search_and_seeds():
    cfg = load_yaml("configs/tft_forecast.yaml")
    ev = cfg["benchmark"]
    search = cfg["search"]
    tft = cfg["models"]["optional"]["tft"]
    assert cfg["generate_seed"] == 71001
    assert cfg["train_seed"] == 82001
    assert ev["extra_split_seed"] == 33001
    assert ev["stability_seeds"] == [82001, 82021, 82031]
    assert [s["key"] for s in search["order"]] == ["lr", "hidden_size", "dropout"]
    assert search["order"][0]["values"] == [0.001, 0.003, 0.01]
    assert search["order"][1]["values"] == [16, 32]
    assert search["order"][2]["values"] == [0.1, 0.2]
    assert search["duration"]["base_max_epochs"] == 16
    assert search["duration"]["extended_max_epochs"] == 24
    assert tft["target_normalizer"] == "species_or_global"
    assert tft["window_balance"] == "study"
    assert tft["max_encoder_length"] == 8
    assert tft["max_prediction_length"] == 6
    assert tft["encoder_window_day"] == 14
    assert cfg["output_dir"] == OUTPUT_DEFAULT
    assert cfg["output_dir"] != cfg["pilot_dir"]
    assert cfg["output_dir"] != cfg["extend_dir"]


def test_refuses_locked_and_prior_tft_dirs(tmp_path):
    with pytest.raises(ForecastError, match="preserved"):
        _refuse_locked(Path("runs/v3_official_seq"),
                       pilot=Path("runs/tft_early_hist_bench"),
                       extend=Path("runs/tft_early_hist_extend"))
    with pytest.raises(ForecastError, match="preserved|bench/extend"):
        _refuse_locked(Path("runs/tft_early_hist_bench"),
                       pilot=Path("runs/tft_early_hist_bench"),
                       extend=Path("runs/tft_early_hist_extend"))
    _refuse_locked(tmp_path / "tft_primary",
                   pilot=Path("runs/tft_early_hist_bench"),
                   extend=Path("runs/tft_early_hist_extend"))
    assert "runs/tft_early_hist_bench" in LOCKED_DIRS
    assert "runs/tft_early_hist_extend" in LOCKED_DIRS


def test_cat_label_never_emits_nan_string():
    assert cat_label(None) == "UNK"
    assert cat_label(np.nan) == "UNK"
    assert cat_label(pd.NA) == "UNK"
    assert cat_label("nan") == "UNK"
    assert cat_label("NaN") == "UNK"
    assert cat_label("") == "UNK"
    assert cat_label("CMP03") == "CMP03"
    assert cat_label("RAT") == "RAT"
    legacy = str(pd.Series([np.nan]).astype(str).iloc[0])
    assert legacy.lower() == "nan"
    assert cat_label(np.nan) != legacy


def test_search_selects_validation_loss_not_test_mae():
    rows = [
        {"value": 0.001, "best_val_loss": 0.050, "test_mae": 0.01},
        {"value": 0.003, "best_val_loss": 0.040, "test_mae": 0.99},
        {"value": 0.01, "best_val_loss": 0.045, "test_mae": 0.02},
    ]
    assert select_by_val_loss(rows)["value"] == 0.003


@pytest.mark.skipif(not PF_INSTALLED, reason="pytorch_forecasting extra not installed")
def test_species_normalizer_skips_relu_and_groups_when_mixed():
    mixed = pd.DataFrame({"SPECIES": ["RAT", "DOG", "RAT"], "LOGW": [5.6, 9.1, 5.7]})
    one = pd.DataFrame({"SPECIES": ["RAT", "RAT"], "LOGW": [5.6, 5.7]})
    grouped = build_target_normalizer(mixed, {"target_normalizer": "species_or_global"})
    global_n = build_target_normalizer(one, {"target_normalizer": "species_or_global"})
    enc = build_target_normalizer(mixed, {"target_normalizer": "encoder"})
    assert type(grouped).__name__ == "GroupNormalizer"
    assert grouped.groups == ["SPECIES"]
    assert grouped.scale_by_group is True
    assert grouped.transformation is None
    assert type(global_n).__name__ == "GroupNormalizer"
    assert not global_n.groups
    assert global_n.transformation is None
    assert type(enc).__name__ == "EncoderNormalizer"
    assert enc.transformation is None
    assert uses_target_scales(mixed, {}) is True
    assert uses_target_scales(one, {}) is False


def test_diagnosis_reads_locked_extra_split_without_writing(tmp_path):
    src = Path("runs/tft_early_hist_extend/banks/split_33001")
    if not (src / "extra_split_tft_pooled.csv").is_file():
        pytest.skip("locked extra-split artifacts are not present")
    before = (src / "extra_split_tft_pooled.csv").read_bytes()
    rec = diagnose_locked_pooled_collapse(src)
    assert rec["status"] == "ok"
    assert rec["read_only"] is True
    assert rec["token_demo"]["missing_compound_cat_label"] == "UNK"
    assert rec["token_demo"]["confirmed_mismatch"] is True
    assert (src / "extra_split_tft_pooled.csv").read_bytes() == before
    assert not (tmp_path / "extra_split_tft_pooled.csv").exists()


def test_observed_panel_maps_missing_compound_to_unk(trained):
    from send_bw.models.tft import build_observed_panel

    data = trained["data"]["train"]
    animals = data.animals.copy()
    if "COMPOUND_ID" in animals.columns:
        animals.loc[animals.index[0], "COMPOUND_ID"] = np.nan
    from send_bw.models.base import ModelData

    patched = ModelData(animals, data.obs, data.compound_ref, data.spec)
    panel = build_observed_panel(patched)
    assert "COMPOUND_ID" in STATIC_CATS
    tokens = set(panel["COMPOUND_ID"].astype(str))
    assert "nan" not in {t.lower() for t in tokens}
    uid = str(animals.iloc[0]["USUBJID"])
    assert set(panel.loc[panel["USUBJID"].astype(str) == uid, "COMPOUND_ID"]) == {"UNK"}

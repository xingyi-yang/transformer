"""Stability, interval score, CQR, and predeclared confirmation holdout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from send_bw.tft_early_hist_extend import load_yaml
from send_bw.tft_intervals import (
    apply_cqr_margin,
    conformal_margin,
    conformal_nonconformity,
    fit_cqr_margin,
    interval_score80,
)
from send_bw.tft_stability import (
    LOCKED_DIRS,
    OUTPUT_DEFAULT,
    StabilityError,
    _refuse_locked,
    confirm_study_splits,
    official_reuse_map,
    select_default,
)


def test_config_predeclares_seeds_and_untouched_holdout():
    cfg = load_yaml("configs/tft_stability.yaml")
    ev = cfg["benchmark"]
    assert ev["stability_seeds"] == [82001, 82021, 82031]
    assert ev["confirm_split_seed"] == 34011
    assert ev["confirm_forbidden_test_studies"] == ["SYNA-007", "SYNA-010", "SYNA-011"]
    assert ev["extra_split_seed"] == 33001
    assert cfg["output_dir"] == OUTPUT_DEFAULT
    assert cfg["output_dir"] != cfg["primary_dir"]
    assert cfg["selection"]["bank"] == "official_validation"
    assert cfg["selection"]["metric"] == "interval_score80"
    assert cfg["calibration"]["method"] == "cqr_additive"
    assert cfg["models"]["optional"]["tft"]["lr"] == 0.01
    assert cfg["models"]["optional"]["tft"]["max_encoder_length"] == 8


def test_refuses_locked_and_primary_dirs(tmp_path):
    with pytest.raises(StabilityError, match="preserved"):
        _refuse_locked(Path("runs/tft_primary"))
    with pytest.raises(StabilityError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_early_hist_extend"))
    _refuse_locked(tmp_path / "tft_stability")
    assert "runs/tft_primary" in LOCKED_DIRS


def test_interval_score_is_width_plus_ten_times_miss():
    y = np.array([0.0, 0.0, 2.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    s = interval_score80(y, lo, hi)
    assert s[0] == pytest.approx(2.0)
    assert s[1] == pytest.approx(2.0)
    assert s[2] == pytest.approx(2.0 + 10.0 * 1.0)


def test_cqr_does_not_move_median_and_uses_validation_pairs():
    fut = pd.DataFrame({
        "USUBJID": ["a", "a", "b"], "STUDYID": ["S1", "S1", "S2"],
        "DAY": [21.0, 28.0, 21.0], "LOGW": [5.0, 5.2, 5.1],
    })
    pred = pd.DataFrame({
        "USUBJID": ["a", "a", "b"], "DAY": [21.0, 28.0, 21.0],
        "q0.1": [4.95, 5.15, 5.00], "q0.5": [5.00, 5.18, 5.08], "q0.9": [5.05, 5.22, 5.12],
    })
    rec = fit_cqr_margin(fut, pred)
    assert rec["status"] == "ok"
    assert rec["test_outcomes_used"] is False
    assert rec["n_cal"] == 3
    cal = apply_cqr_margin(pred, rec["margin"])
    assert list(cal["q0.5"]) == list(pred["q0.5"])
    assert (cal["q0.9"] - cal["q0.1"]).to_numpy() == pytest.approx(
        (pred["q0.9"] - pred["q0.1"]).to_numpy() + 2.0 * rec["margin"])
    scores = conformal_nonconformity(
        fut["LOGW"].to_numpy(), pred["q0.1"].to_numpy(), pred["q0.9"].to_numpy())
    assert rec["margin"] == pytest.approx(conformal_margin(scores))


def test_select_default_uses_validation_interval_score_not_test_mae():
    rows = [
        {"variant": "tft_target_only", "interval_score80": 0.20, "test_mae": 0.01},
        {"variant": "tft_target_only", "interval_score80": 0.22, "test_mae": 0.01},
        {"variant": "tft_finetune", "interval_score80": 0.18, "test_mae": 0.99},
        {"variant": "tft_finetune", "interval_score80": 0.16, "test_mae": 0.99},
        {"variant": "tft_pooled", "interval_score80": 0.30, "test_mae": 0.02},
        {"variant": "tft_pooled", "interval_score80": 0.28, "test_mae": 0.02},
    ]
    dec = select_default(rows, metric="interval_score80",
                         candidates=["tft_target_only", "tft_pooled", "tft_finetune"])
    assert dec["selected"] == "tft_finetune"
    assert dec["test_scores_examined"] is False


def test_confirm_split_keeps_reported_holdouts_out_of_test():
    animals = pd.DataFrame({
        "STUDYID": [f"SYNA-{i:03d}" for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11)],
        "REF_DATE": ["2000-01-01"] * 10,
        "SPECIES": ["RAT"] * 10,
        "USUBJID": [f"A{i}" for i in range(10)],
    })
    frame = animals
    class _Full:
        spec = type("S", (), {"species": "RAT"})()
        animals = frame
    pack = confirm_study_splits(_Full(), _Full(), seed=34011,
                                forbidden_test=["SYNA-007", "SYNA-010", "SYNA-011"])
    test = set(pack["target_splits"].loc[pack["target_splits"]["SPLIT"] == "test", "STUDYID"])
    assert test.isdisjoint({"SYNA-007", "SYNA-010", "SYNA-011"})
    train = set(pack["target_splits"].loc[pack["target_splits"]["SPLIT"] == "train", "STUDYID"])
    assert {"SYNA-007", "SYNA-010", "SYNA-011"} <= train


def test_reuse_map_points_at_completed_primary_fits():
    mp = official_reuse_map(Path("runs/tft_primary"))
    assert (mp[("target_only", 82001)] / "tft.ckpt").is_file()
    assert (mp[("target_only", 82021)] / "tft.ckpt").is_file()
    assert (mp[("target_only", 82031)] / "tft.ckpt").is_file()
    assert (mp[("pooled", 82001)] / "tft.ckpt").is_file()
    assert (mp[("finetune", 82001)] / "tft.ckpt").is_file()
    assert ("pooled", 82021) not in mp

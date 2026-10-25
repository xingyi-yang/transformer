"""Hand-checkable MAE/RMSE and merge-cardinality tests. No training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from send_bw.evaluation.quantile_forecast import (
    align_prediction_table,
    paired_transfer,
    score_quantile_table,
    series_eligibility,
)
from send_bw.v3_eval_correct import (
    EvalCorrectError,
    execute_v3_eval_correct,
    ranking_sign_changes,
)

Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def _qtable(rows: list[tuple[str, str, float, float, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.DataFrame([{"USUBJID": u, "STUDYID": s, "DAY": d, "LOGW": y} for u, s, d, y, _ in rows])
    pred = pd.DataFrame([{
        "USUBJID": u, "STUDYID": s, "DAY": d,
        "q0.1": p, "q0.25": p, "q0.5": p, "q0.75": p, "q0.9": p,
    } for u, s, d, _, p in rows])
    return obs, pred


def test_mae_is_abs_rmse_is_squared_hand_check():
    # One study, errors 0 and 4. MAE=2. RMSE=sqrt(8)=2*sqrt(2).
    obs, pred = _qtable([
        ("a", "S", 21.0, 0.0, 0.0),
        ("a", "S", 28.0, 4.0, 0.0),
    ])
    out = score_quantile_table(obs, pred, quantiles=Q, representation="tft_quantiles")
    assert out["mae_log"]["study_balanced"] == pytest.approx(2.0)
    assert out["rmse_log"]["study_balanced"] == pytest.approx(float(np.sqrt(8.0)))
    assert out["rmse_log"]["study_balanced"] != pytest.approx(out["mae_log"]["study_balanced"])
    assert out["grouping"] == "STUDYID"
    assert out["n_scored"] == 2


def test_study_balanced_mae_and_rmse_two_studies():
    # S1 errors 0,4 -> MAE 2, RMSE sqrt(8). S2 errors 0,0 -> MAE 0, RMSE 0.
    obs, pred = _qtable([
        ("a", "S1", 21.0, 0.0, 0.0),
        ("a", "S1", 28.0, 4.0, 0.0),
        ("b", "S2", 21.0, 1.0, 1.0),
        ("b", "S2", 28.0, 1.0, 1.0),
    ])
    out = score_quantile_table(obs, pred, quantiles=Q, representation="tft_quantiles")
    assert out["mae_log"]["n_studies"] == 2
    assert out["mae_log"]["study_balanced"] == pytest.approx(1.0)
    assert out["rmse_log"]["study_balanced"] == pytest.approx(float(np.sqrt(8.0)) / 2.0)
    assert out["merge"]["one_to_one"] is True


def test_merge_cardinality_one_to_one_and_extra_pred():
    obs = pd.DataFrame({"USUBJID": ["a"], "STUDYID": ["S"], "DAY": [21.0], "LOGW": [1.0]})
    pred = pd.DataFrame({
        "USUBJID": ["a", "a"], "STUDYID": ["S", "S"], "DAY": [21.0, 28.0],
        "q0.1": [1.0, 1.0], "q0.25": [1.0, 1.0], "q0.5": [1.0, 1.0],
        "q0.75": [1.0, 1.0], "q0.9": [1.0, 1.0],
    })
    mm, card = align_prediction_table(obs, pred)
    assert len(mm) == 1
    assert card["n_obs"] == 1
    assert card["n_pred"] == 2
    assert card["n_pred_only"] == 1
    assert card["n_obs_only"] == 0
    assert card["one_to_one"] is True
    assert card["inflated"] is False


def test_merge_detects_duplicate_pred_keys():
    obs = pd.DataFrame({"USUBJID": ["a"], "STUDYID": ["S"], "DAY": [21.0], "LOGW": [0.0]})
    pred = pd.DataFrame({
        "USUBJID": ["a", "a"], "DAY": [21.0, 21.0],
        "q0.5": [0.0, 1.0], "q0.1": [0.0, 1.0], "q0.25": [0.0, 1.0],
        "q0.75": [0.0, 1.0], "q0.9": [0.0, 1.0],
    })
    mm, card = align_prediction_table(obs, pred)
    assert card["duplicate_pred_keys"] == 1
    assert card["inflated"] is True
    assert len(mm) == 2


def test_studyid_from_obs_not_pred_suffix():
    obs = pd.DataFrame({"USUBJID": ["a"], "STUDYID": ["TRUE"], "DAY": [21.0], "LOGW": [0.0]})
    pred = pd.DataFrame({
        "USUBJID": ["a"], "STUDYID": ["WRONG"], "DAY": [21.0],
        "q0.1": [0.0], "q0.25": [0.0], "q0.5": [0.0], "q0.75": [0.0], "q0.9": [0.0],
    })
    out = score_quantile_table(obs, pred, quantiles=Q, representation="tft_quantiles")
    assert out["per_study"][0]["STUDYID"] == "TRUE"
    assert out["mae_log"]["n_studies"] == 1


def test_eligibility_does_not_fabricate_visits():
    obs = pd.DataFrame({
        "USUBJID": ["a"] * 5 + ["b"] * 8,
        "DAY": list(range(5)) + list(range(8)),
        "LOGW": [0.0] * 13,
    })
    el = series_eligibility(obs, min_past=2, prediction_length=6)
    assert el["required_observed_visits"] == 8
    assert el["n_eligible"] == 1
    assert el["n_ineligible"] == 1
    assert el["fabricated_history"] is False
    assert el["length_histogram"] == {5: 1, 8: 1}


def test_horizon_step_is_visit_rank_not_calendar_gap():
    obs, pred = _qtable([
        ("a", "S", 21.0, 0.0, 0.0),
        ("a", "S", 28.0, 1.0, 1.0),
    ])
    out = score_quantile_table(obs, pred, quantiles=Q, representation="chronos2_quantiles")
    steps = {r["horizon_step"] for r in out["per_horizon_step"]}
    assert steps == {1, 2}
    assert out["horizon_units"]["fabricated_calendar_steps"] is False
    assert out["chronos_label"] == "history_only_univariate"


def test_paired_transfer_accepts_scalars_and_dicts():
    tr = paired_transfer({"mae_log": 0.10, "rmse_log": {"study_balanced": 0.20}},
                         {"mae_log": 0.12, "rmse_log": {"study_balanced": 0.18}})
    mets = {r["metric"]: r["delta_pooled_minus_target"] for r in tr["rows"]}
    assert mets["mae_log"] == pytest.approx(0.02)
    assert mets["rmse_log"] == pytest.approx(-0.02)


def test_ranking_sign_changes_records_swap_not_sign():
    off = {
        "rank_mae": [{"model": "tft_target_only"}, {"model": "spline_target_only"}],
        "rank_rmse": [{"model": "tft_target_only"}, {"model": "spline_target_only"}],
        "rank_legacy_animal_mae": [
            {"model": "spline_target_only"}, {"model": "tft_target_only"},
        ],
        "sign_vs_published": {
            "spline": [{"metric": "mae_log", "sign": "helpful",
                        "published_sign": "helpful", "sign_changed": False}],
        },
    }
    note = ranking_sign_changes(off)
    assert note["mae_equals_rmse_ranking"] is True
    assert note["study_mae_equals_legacy_animal_mae_ranking"] is False
    assert note["any_transfer_sign_changed"] is False
    assert note["legacy_vs_study_mae_rank_swaps"][0]["rank"] == 1
    assert "tft_target_only vs spline_target_only" in note["grouping_note"]
    assert "do not change" in note["sign_note"]


def test_executor_refuses_preserved_train_and_real(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("output_dir: runs/v3_official_seq\ngenerate_seed: 1\ntrain_seed: 2\n",
                   encoding="utf-8")
    with pytest.raises(EvalCorrectError, match="preserved"):
        execute_v3_eval_correct(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "eval_correct: {train: true}\n", encoding="utf-8")
    with pytest.raises(EvalCorrectError, match="training"):
        execute_v3_eval_correct(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\nsource: {type: snowflake}\n",
        encoding="utf-8")
    with pytest.raises(EvalCorrectError, match="real-data"):
        execute_v3_eval_correct(cfg)

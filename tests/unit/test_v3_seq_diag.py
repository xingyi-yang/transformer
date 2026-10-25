"""Bounded official-seq diagnostic gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from send_bw.evaluation.quantile_forecast import (
    encoder_time_report,
    last_value_predictions,
    score_quantile_table,
)
from send_bw.v3_official_seq import OfficialSeqError, execute_v3_official_seq
from send_bw.v3_seq_diag import SeqDiagError, execute_v3_seq_diag


def test_last_value_is_degenerate_and_uses_encoder_tail():
    hist = pd.DataFrame({
        "USUBJID": ["a", "a"], "STUDYID": ["S", "S"], "DAY": [1.0, 14.0], "LOGW": [1.0, 2.0],
    })
    fut = pd.DataFrame({
        "USUBJID": ["a", "a"], "STUDYID": ["S", "S"], "DAY": [21.0, 28.0], "LOGW": [2.2, 2.4],
    })
    pred = last_value_predictions(hist, fut)
    assert (pred["q0.5"] == 2.0).all()
    assert (pred["q0.1"] == pred["q0.9"]).all()
    scored = score_quantile_table(fut, pred, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                                  representation="last_value")
    assert scored["crps_justified"] is False
    assert scored["point_definition"]["point"] == "median"
    assert scored["interval_width80"]["study_balanced"] == pytest.approx(0.0)
    assert scored["per_horizon"]
    assert scored["per_study"][0]["STUDYID"] == "S"


def test_common_score_has_point_and_breakdowns():
    obs = pd.DataFrame({"USUBJID": ["a", "a"], "STUDYID": ["S", "S"], "DAY": [21.0, 28.0],
                        "LOGW": [0.1, 0.2]})
    pred = pd.DataFrame({
        "USUBJID": ["a", "a"], "STUDYID": ["S", "S"], "DAY": [21.0, 28.0],
        "q0.1": [0.0, 0.05], "q0.25": [0.05, 0.1], "q0.5": [0.1, 0.2],
        "q0.75": [0.15, 0.25], "q0.9": [0.2, 0.3],
    })
    out = score_quantile_table(obs, pred, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                               representation="tft_quantiles")
    assert out["task"] == "early_history"
    assert out["point_definition"]["point"] == "median"
    assert {row["DAY"] for row in out["per_horizon"]} == {21.0, 28.0}


def test_encoder_time_does_not_invent_calendar_index():
    hist = pd.DataFrame({
        "USUBJID": ["a", "a"], "STUDYID": ["S", "S"], "DAY": [1.0, 14.0], "LOGW": [0.0, 0.1],
    })
    fut = pd.DataFrame({
        "USUBJID": ["a"], "STUDYID": ["S"], "DAY": [28.0], "LOGW": [0.2],
    })
    rep = encoder_time_report(hist, fut)
    assert rep["calendar_time_passed_to_chronos"] is False
    assert rep["chronos_time_index"] == "visit_rank_unit_steps"
    assert rep["rows"][0]["n_encoder"] == 2


def test_diag_refuses_preserved_predose_and_real(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("output_dir: runs/v3_official_seq\ngenerate_seed: 1\ntrain_seed: 2\n",
                   encoding="utf-8")
    with pytest.raises(SeqDiagError, match="preserved"):
        execute_v3_seq_diag(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "seq_diag: {predose: true}\n", encoding="utf-8")
    with pytest.raises(SeqDiagError, match="predose"):
        execute_v3_seq_diag(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\nsource: {type: snowflake}\n",
        encoding="utf-8")
    with pytest.raises(SeqDiagError, match="real-data"):
        execute_v3_seq_diag(cfg)


def test_official_seq_now_refuses_its_own_dir(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("output_dir: runs/v3_official_seq\ngenerate_seed: 1\ntrain_seed: 2\n",
                   encoding="utf-8")
    with pytest.raises(OfficialSeqError, match="preserved"):
        execute_v3_official_seq(cfg)


def test_pinball_unchanged():
    from send_bw.evaluation.quantile_forecast import pinball
    assert pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)

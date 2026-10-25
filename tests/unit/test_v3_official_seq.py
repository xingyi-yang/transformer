"""Official TFT / Chronos-2 gates: no fabricated encoder, no design, CRPS rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from send_bw.evaluation.quantile_forecast import paired_transfer, pinball, score_quantile_table
from send_bw.models.chronos2 import Chronos2EarlyHistoryModel
from send_bw.models.optional_adapters import CATALOG, check_optional, verify_tft_api
from send_bw.models.tft import TFTEarlyHistoryModel
from send_bw.v3_official_seq import OfficialSeqError, execute_v3_official_seq


def test_catalog_does_not_alias_official_models_as_transformer():
    assert CATALOG["tft"].name == "tft"
    assert CATALOG["chronos2"].name == "chronos2"
    assert "transformer" not in CATALOG


def test_tft_and_chronos_refuse_baseline_and_design():
    for cls, _name in ((TFTEarlyHistoryModel, "TFT"), (Chronos2EarlyHistoryModel, "Chronos")):
        m = cls()
        with pytest.raises(NotImplementedError):
            m.predict_mean(pd.DataFrame({"USUBJID": ["a"]}), np.array([28.0]))
        with pytest.raises(NotImplementedError):
            m.sample_study(pd.DataFrame({"USUBJID": ["a"]}), None, np.array([28.0]), 2, 0)
        with pytest.raises(ValueError, match="encoder window|baseline-only"):
            m.predict_quantiles(pd.DataFrame({"USUBJID": ["a"]}), np.array([28.0]), history=None)


def test_quantile_score_does_not_invent_crps_for_quantiles():
    obs = pd.DataFrame({"USUBJID": ["a", "a"], "STUDYID": ["S", "S"], "DAY": [21.0, 28.0],
                        "LOGW": [0.1, 0.2]})
    pred = pd.DataFrame({
        "USUBJID": ["a", "a"], "STUDYID": ["S", "S"], "DAY": [21.0, 28.0],
        "q0.1": [0.0, 0.05], "q0.25": [0.05, 0.1], "q0.5": [0.1, 0.2],
        "q0.75": [0.15, 0.25], "q0.9": [0.2, 0.3],
    })
    out = score_quantile_table(obs, pred, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                               representation="tft_quantiles")
    assert out["crps_justified"] is False
    assert out["crps_log"] is None
    assert out["trajectory_simulator"] is False
    assert out["rmse_log"]["study_balanced"] == pytest.approx(0.0)
    assert out["coverage80"]["study_balanced"] == pytest.approx(1.0)


def test_gaussian_crps_is_justified():
    obs = pd.DataFrame({"USUBJID": ["a"], "STUDYID": ["S"], "DAY": [28.0], "LOGW": [0.0]})
    pred = pd.DataFrame({"USUBJID": ["a"], "STUDYID": ["S"], "DAY": [28.0],
                         "MEAN": [0.0], "VAR": [0.04]})
    out = score_quantile_table(obs, pred, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                               representation="gaussian")
    assert out["crps_justified"] is True
    assert out["crps_log"]["study_balanced"] is not None


def test_pinball_and_paired_transfer_sign():
    assert pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)
    tr = paired_transfer(
        {"rmse_log": {"study_balanced": 0.10}},
        {"rmse_log": {"study_balanced": 0.13}},
    )
    assert tr["rows"][0]["delta_pooled_minus_target"] == pytest.approx(0.03)


def test_executor_refuses_preserved_and_real(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("output_dir: runs/v3_research\ngenerate_seed: 1\ntrain_seed: 2\n", encoding="utf-8")
    with pytest.raises(OfficialSeqError, match="preserved"):
        execute_v3_official_seq(cfg)
    cfg.write_text("output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\nsource: {type: snowflake}\n",
                   encoding="utf-8")
    with pytest.raises(OfficialSeqError, match="real-data"):
        execute_v3_official_seq(cfg)


def test_tft_api_verification_when_installed():
    v = verify_tft_api()
    if v["ok"]:
        assert "TemporalFusionTransformer" in v["reason"]
    rec = check_optional("chronos2", {"enabled": True, "download_permission": False})
    if rec.get("extra", {}).get("verify", {}).get("ok"):
        assert "download" in rec["reason"]

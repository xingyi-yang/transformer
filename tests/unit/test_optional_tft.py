"""Optional TFT adapter: honest gates without pytorch_forecasting in the default env."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from send_bw.models.optional_adapters import CATALOG, check_optional, optional_status, verify_tft_api
from send_bw.models.tft import TFTEarlyHistoryModel, build_observed_panel, quantiles_to_mean_var
from send_bw.pipeline import Run, model_class

PF_INSTALLED = importlib.util.find_spec("pytorch_forecasting") is not None


def test_tft_catalog_is_not_the_required_transformer():
    assert "tft" in CATALOG
    assert CATALOG["tft"].name == "tft"
    assert "transformer" not in CATALOG
    assert "not interchangeable with the custom baseline-conditioned transformer" in CATALOG["tft"].limitations


def test_verify_tft_api_reports_missing_without_extra():
    if PF_INSTALLED:
        pytest.skip("pytorch_forecasting is installed")
    v = verify_tft_api()
    assert v["ok"] is False
    assert "pytorch_forecasting" in v["reason"]


def test_check_optional_tft_not_run_without_dep():
    if PF_INSTALLED:
        pytest.skip("pytorch_forecasting is installed")
    rec = check_optional("tft", {"enabled": True})
    assert rec["status"] == "not_run"
    assert rec["name"] == "tft"
    assert "pytorch_forecasting" in rec["reason"]
    rec_off = check_optional("tft", {"enabled": False})
    assert rec_off["status"] == "not_run"
    assert rec_off["reason"] == "disabled in config"


def test_optional_status_does_not_claim_chronos_or_deepar():
    st = optional_status({"optional": {"tft": {"enabled": True}, "chronos2": {"enabled": True},
                                      "deepar": {"enabled": True}}})
    assert st["chronos2"]["status"] == "not_run"
    assert st["deepar"]["status"] == "not_run"
    if not PF_INSTALLED:
        assert st["tft"]["status"] == "not_run"


def test_model_class_does_not_register_tft():
    from send_bw.models.transformer import TransformerModel

    assert model_class("transformer") is TransformerModel
    with pytest.raises(KeyError, match="optional early-history"):
        model_class("tft")


def test_tft_refuses_baseline_only_and_design_use():
    m = TFTEarlyHistoryModel()
    assert m.name == "tft"
    assert m.task == "early_history"
    assert m.supports_pre_study is False
    with pytest.raises(NotImplementedError, match="encoder window|baseline-only"):
        m.predict_mean(pd.DataFrame({"USUBJID": ["a"]}), np.array([28.0]))
    with pytest.raises(NotImplementedError, match="design simulator|hypothetical"):
        m.sample_study(pd.DataFrame({"USUBJID": ["a"]}), None, np.array([28.0]), 2, 0)
    with pytest.raises(ValueError, match="encoder window|baseline-only"):
        m.predict_marginals(pd.DataFrame({"USUBJID": ["a"]}), np.array([28.0]), history=None)
    with pytest.raises(ValueError, match="encoder window|baseline-only"):
        m.predict_marginals(pd.DataFrame({"USUBJID": ["a"]}), np.array([28.0]), history=pd.DataFrame())


def test_tft_fit_refuses_without_dependency(trained):
    if PF_INSTALLED:
        pytest.skip("pytorch_forecasting is installed")
    with pytest.raises(RuntimeError, match="pytorch_forecasting"):
        TFTEarlyHistoryModel().fit(trained["data"]["train"], trained["data"]["validation"], {"max_epochs": 1})


def test_available_models_excludes_optional_tft(tmp_path):
    cfg = {
        "run_name": "opt_tft", "output_dir": str(tmp_path / "run"), "seed": 1,
        "source": {"type": "local", "root": str(tmp_path)},
        "mapping": "configs/mappings/example_send.yaml",
        "models": {"spline": {"enabled": True}, "optional": {"tft": {"enabled": True}}},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run = Run(p)
    (run.dir / "models" / "spline").mkdir(parents=True)
    (run.dir / "models" / "spline" / "model.json").write_text("{}", encoding="utf-8")
    (run.dir / "models" / "tft").mkdir(parents=True)
    (run.dir / "models" / "tft" / "model.json").write_text(json.dumps({"name": "tft"}), encoding="utf-8")
    assert run.available_models() == ["spline"]
    assert run.optional_trained() == ["tft"]
    assert "tft" not in run.available_models()


def test_evaluate_early_history_does_not_fabricate_pre_study(trained):
    from send_bw.evaluation.forecast import evaluate_early_history

    class _EarlyOnly:
        name = "tft"
        task = "early_history"
        supports_early_history = True
        supports_pre_study = False

        def predict_marginals(self, context, times, history=None):
            if history is None or not len(history):
                raise ValueError("TFT requires an observed encoder window")
            rows = [(u, t, 0.0, 0.01) for u in context["USUBJID"] for t in times]
            return pd.DataFrame(rows, columns=["USUBJID", "DAY", "MEAN", "VAR"])

    r = evaluate_early_history(_EarlyOnly(), trained["data"]["test"], 14.0)
    assert r["status"] == "ok"
    assert "conditioned" in r and r["conditioned"]["n_obs"] > 0
    assert r["pre_study"]["status"] == "not_supported"
    assert "not fabricated" in r["pre_study"]["reason"]


def test_observed_panel_does_not_fill_or_zero_target(trained):
    data = trained["data"]["test"]
    panel = build_observed_panel(data)
    assert len(panel) == len(data.obs)
    assert panel["LOGW"].notna().all()
    assert not (panel["LOGW"] == 0).all()
    for _, g in panel.groupby("USUBJID"):
        assert (g["time_idx"].to_numpy() == np.arange(len(g))).all()
        assert g["DAY"].is_monotonic_increasing


def test_quantiles_to_mean_var_uses_median_and_spread():
    from send_bw.models.tft import _Z_10_90

    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    q = np.array([[2.0 - _Z_10_90, 1.0, 2.0, 3.0, 2.0 + _Z_10_90]])
    mean, var = quantiles_to_mean_var(q, levels)
    assert mean[0] == pytest.approx(2.0)
    assert var[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.skipif(not PF_INSTALLED, reason="pytorch_forecasting extra not installed")
def test_tft_fit_and_early_history_when_extra_present(trained, tmp_path):
    m = TFTEarlyHistoryModel()
    hist = m.fit(trained["data"]["train"], trained["data"]["validation"],
                 {"max_epochs": 1, "hidden_size": 8, "attention_head_size": 1, "batch_size": 16,
                  "patience": 1, "max_encoder_length": 4, "max_prediction_length": 3, "seed": 0})
    assert hist["name"] == "tft"
    assert hist["filled_target_with_zero"] is False
    m.save(tmp_path / "tft")
    loaded = json.loads((tmp_path / "tft" / "model.json").read_text(encoding="utf-8"))
    assert loaded["name"] == "tft"
    assert loaded["not_the_required_transformer"] is True
    from send_bw.evaluation.forecast import evaluate_early_history

    r = evaluate_early_history(m, trained["data"]["test"], 14.0)
    assert r["pre_study"]["status"] == "not_supported"
    assert r["status"] in {"ok", "failed", "not_run"}

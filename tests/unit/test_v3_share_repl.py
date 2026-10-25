"""Locked sharing-replication budget, refuse gates, and not_implemented cells."""

from __future__ import annotations

import pytest

from send_bw.v3_research import load_yaml
from send_bw.v3_share_repl import (
    REGIMES,
    TFT_CHRONOS_KEYS,
    ShareReplError,
    _sign_rows,
    execute_v3_share_repl,
    locked_budget,
    tft_chronos_unsupported,
)

FROZEN = "configs/v3_share_repl.frozen.yaml"


def test_frozen_budget_states_seeds_and_runtime_before_execution():
    cfg = load_yaml(FROZEN)
    budget = locked_budget(cfg)
    assert budget["n_dataset_seeds"] == 2
    assert budget["n_train_seeds"] == 1
    assert budget["generate_seeds"] == [74001, 75010]
    assert budget["train_seeds"] == [84001]
    assert budget["n_cells"] == 8
    assert budget["bank"] == "matched"
    assert budget["donor_policy"]["target_only"] == 0
    assert budget["donor_policy"]["pooled"] == 3
    assert budget["donor_policy"]["selective"] == 1
    assert budget["donor_policy"]["selected_from_test"] is False
    assert budget["checkpoint_source"] == "training_validation"
    assert budget["chronos_mode"] == "history_only_univariate"
    assert budget["tft_selective"] == "not_implemented"
    assert budget["chronos_selective"] == "not_implemented"
    assert budget["covariate_chronos"] is False
    runtime = budget["estimated_runtime_seconds"]
    assert runtime["point"] == 320
    assert runtime["lower"] == 240
    assert runtime["upper"] == 720


def test_tft_chronos_selective_is_not_implemented():
    assert {r["key"] for r in REGIMES} == {
        "target_only", "pooled", "selective", "selective_regularized",
    }
    assert TFT_CHRONOS_KEYS == {"target_only", "pooled"}
    assert tft_chronos_unsupported("target_only", "TFT") is None
    assert tft_chronos_unsupported("pooled", "Chronos-2") is None
    tft = tft_chronos_unsupported("selective", "TFT")
    assert tft["status"] == "not_implemented"
    assert "TFT" in tft["reason"]
    ch = tft_chronos_unsupported("selective_regularized", "Chronos-2")
    assert ch["status"] == "not_implemented"
    assert "Chronos-2" in ch["reason"]


def test_sign_rows_publish_harmful_helpful_and_null():
    block = _sign_rows({
        "rows": [
            {"metric": "mae_log", "delta_pooled_minus_target": 0.01},
            {"metric": "rmse_log", "delta_pooled_minus_target": -0.02},
            {"metric": "mean_quantile_loss", "delta_pooled_minus_target": 0.0},
            {"metric": "missing", "delta_pooled_minus_target": None},
        ],
    })
    signs = {r["metric"]: r["sign"] for r in block["rows"]}
    assert signs["mae_log"] == "harmful"
    assert signs["rmse_log"] == "helpful"
    assert signs["mean_quantile_loss"] == "null"
    assert signs["missing"] == "null"


def test_executor_refuses_preserved_real_design_and_forbidden_flags(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("output_dir: runs/v3_cov_chronos\ngenerate_seed: 1\ntrain_seed: 2\n",
                   encoding="utf-8")
    with pytest.raises(ShareReplError, match="preserved"):
        execute_v3_share_repl(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\nsource: {type: snowflake}\n",
        encoding="utf-8")
    with pytest.raises(ShareReplError, match="real-data"):
        execute_v3_share_repl(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "design: {request: x.yaml}\n", encoding="utf-8")
    with pytest.raises(ShareReplError, match="design"):
        execute_v3_share_repl(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "share_repl: {retune_after_test: true}\n", encoding="utf-8")
    with pytest.raises(ShareReplError, match="retune"):
        execute_v3_share_repl(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "share_repl: {predose: true}\n", encoding="utf-8")
    with pytest.raises(ShareReplError, match="predose"):
        execute_v3_share_repl(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "share_repl: {bayesian: true}\n", encoding="utf-8")
    with pytest.raises(ShareReplError, match="Bayesian"):
        execute_v3_share_repl(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "share_repl: {covariate_chronos: true}\n", encoding="utf-8")
    with pytest.raises(ShareReplError, match="covariate Chronos"):
        execute_v3_share_repl(cfg)

"""Preflight eligibility, origin-only covariates, and defect-trace gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from send_bw.data.features import FeatureSpec
from send_bw.models.base import ModelData
from send_bw.models.chronos2 import (
    COVARIATE_AWARE,
    FORBIDDEN_COVARIATES,
    HISTORY_ONLY,
    KNOWN_FUTURE,
    build_chronos_fit_inputs,
    build_chronos_predict_item,
    chronos_fit_preflight,
    chronos_postprocess_lengths,
    observed_series,
)
from send_bw.v3_cov_chronos import CovChronosError, execute_v3_cov_chronos
from send_bw.v3_defect_trace import trace_metric_defects
from send_bw.v3_increment import load_cell_prepared


def _toy(n_visits: list[int]) -> ModelData:
    animals = []
    obs = []
    for i, n in enumerate(n_visits):
        u = f"A{i}"
        animals.append({
            "USUBJID": u, "STUDYID": "S1", "SEX": "M", "SPECIES": "RAT",
            "STRAIN": "WISTAR HAN", "ROUTE": "ORAL GAVAGE", "PLANNED_DOSE": 10.0,
            "BASELINE_LOGW": 5.0, "ARM_KEY": "S1::C", "IS_CONTROL": False,
            "COMPOUND_ID": "C1", "INTERVAL_DAYS": 1.0, "START_DAY": 0.0, "END_DAY": 27.0,
        })
        for d in range(n):
            obs.append({
                "USUBJID": u, "STUDYID": "S1", "DAY": float(d * 4),
                "LOGW": 5.0 + 0.01 * d, "Y": 0.01 * d,
            })
    a = pd.DataFrame(animals)
    o = pd.DataFrame(obs)
    spec = FeatureSpec.fit(a, o, pd.DataFrame({"COMPOUND_ID": ["C1"], "D1": [0.0],
                                               "D2": [0.0], "D3": [0.0]}), "RAT")
    return ModelData(a, o, pd.DataFrame(), spec)


def test_postprocess_lengths_are_chronos_context_lengths():
    arrays = [np.ones(k, float) for k in (4, 8, 3)]
    lengths = chronos_postprocess_lengths(arrays, prediction_length=6)
    assert lengths == [4, 8, 3]
    need = 2 + 6
    assert sum(L >= need for L in lengths) == 1


def test_preflight_empty_validation_not_train():
    train = _toy([8, 8, 4])
    val = _toy([4, 4, 3])
    pre = chronos_fit_preflight(
        train, val, min_past=2, prediction_length=6, covariate_mode=HISTORY_ONLY)
    assert pre["can_fit"] is True
    assert pre["validation_usable"] is False
    assert pre["train"]["n_kept"] == 2
    assert pre["validation"]["empty_after_filter"] is True
    assert pre["fabricated_history"] is False


def test_d0_reconciliation_validation_empty_after_postprocess():
    cell = Path("runs/v3_seq_diag/repl/d0/datasets/d0/trains/t0/cells/d0_t0_matched_target_only")
    if not cell.exists():
        pytest.skip("seq-diag d0 cell not present")
    prep = load_cell_prepared(cell)
    pre = chronos_fit_preflight(
        prep["data"]["train"], prep["data"]["validation"],
        min_past=2, prediction_length=6, covariate_mode=HISTORY_ONLY)
    assert pre["train"]["n_kept"] == 39
    assert pre["train"]["empty_after_filter"] is False
    assert pre["validation"]["n_kept"] == 0
    assert pre["validation"]["empty_after_filter"] is True
    assert pre["can_fit"] is True
    assert pre["validation_usable"] is False
    assert max(pre["validation"]["postprocess_length_histogram"] or {0: 0}) < 8


def test_covariate_inputs_are_origin_only():
    data = _toy([6])
    series = observed_series(data)[0]
    item = build_chronos_predict_item(series, np.array([21.0, 28.0]), COVARIATE_AWARE)
    assert set(item["past_covariates"]) <= set(KNOWN_FUTURE)
    assert set(item["future_covariates"]) <= set(KNOWN_FUTURE)
    assert "LOGW" not in item["future_covariates"]
    for name in FORBIDDEN_COVARIATES:
        assert name not in item["past_covariates"]
        assert name not in item["future_covariates"]
    fit_in = build_chronos_fit_inputs(data, COVARIATE_AWARE)
    assert "future_covariates" not in fit_in[0]
    assert set(fit_in[0]["past_covariates"]) <= set(KNOWN_FUTURE)


def test_defect_trace_marks_quantile_runs_only():
    tr = trace_metric_defects()
    assert "runs/v3_official_seq" in tr["affected_published_metrics"]
    assert "runs/v3_seq_diag" in tr["affected_published_metrics"]
    assert "runs/v3_eval" not in tr["affected_published_metrics"]
    assert "runs/v3_research" not in tr["affected_published_metrics"]
    assert tr["validation_selection_affected"] is False
    assert tr["rescore_needed"] == []


def test_executor_refuses_preserved_real_design(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("output_dir: runs/v3_eval_correct\ngenerate_seed: 1\ntrain_seed: 2\n",
                   encoding="utf-8")
    with pytest.raises(CovChronosError, match="preserved"):
        execute_v3_cov_chronos(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\nsource: {type: snowflake}\n",
        encoding="utf-8")
    with pytest.raises(CovChronosError, match="real-data"):
        execute_v3_cov_chronos(cfg)
    cfg.write_text(
        "output_dir: runs/tmpx\ngenerate_seed: 1\ntrain_seed: 2\n"
        "design: {request: x.yaml}\n", encoding="utf-8")
    with pytest.raises(CovChronosError, match="design"):
        execute_v3_cov_chronos(cfg)

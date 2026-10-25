"""Generator observability, CQR order statistic, and method information sets."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from send_bw.data.features import FEATURE_REGISTRY, FeatureSpec, UnsupportedContextError
from send_bw.data.generate import GeneratorA, Regimen, generate_dataset
from send_bw.forecast_method_audit import (
    OUTPUT_DEFAULT,
    catboost_information_set,
    chronos_visit_rank_note,
    classification,
    compare_cqr_order_statistic,
    descriptor_observability,
    official_growth_family,
    production_features_exclude_hidden,
)
from send_bw.models.chronos2 import COVARIATE_AWARE, HISTORY_ONLY, build_chronos_predict_item
from send_bw.tft_baselines import NUM_COLS, QuantileCatBoostModel
from send_bw.tft_intervals import conformal_margin, romano_cqr_k, romano_cqr_quantile


def test_d1_d3_are_production_hidden_pk_not_registered():
    rec = descriptor_observability()
    assert rec["d1_d2_d3"] == "production"
    assert rec["hidden_in_feature_registry"] == []
    assert rec["production_includes_descriptors"] is True
    prod = production_features_exclude_hidden()
    assert prod["ok"] is True
    assert "Z0_TRUE" in prod["truth_only"]
    assert "ACTUAL_DOSE" in prod["truth_only"]
    assert "A" not in FEATURE_REGISTRY


def test_missing_descriptors_fail_active_arms():
    animals = pd.DataFrame({
        "USUBJID": ["u1"], "STUDYID": ["S1"], "SEX": ["M"],
        "STRAIN": ["SPRAGUE-DAWLEY"], "ROUTE": ["ORAL GAVAGE"],
        "SPECIES": ["RAT"], "COMPOUND_ID": ["CMP99"], "IS_CONTROL": [False],
        "PLANNED_DOSE": [30.0], "INTERVAL_DAYS": [1.0],
        "START_DAY": [0.0], "END_DAY": [27.0], "BASELINE_LOGW": [5.6],
    })
    obs = pd.DataFrame({"USUBJID": ["u1"], "STUDYID": ["S1"], "DAY": [7.0], "LOGW": [5.7], "Y": [0.1]})
    cref = pd.DataFrame({"COMPOUND_ID": ["CMP03"], "D1": [0.0], "D2": [0.0], "D3": [0.0]})
    spec = FeatureSpec.fit(animals.assign(COMPOUND_ID="CMP03"), obs, cref, "RAT")
    with pytest.raises(UnsupportedContextError, match="descriptor"):
        spec.animal_features(animals, cref)


def test_official_growth_is_saturating_exponential_not_gompertz():
    rec = official_growth_family()
    assert rec["tau_g"] == 40.0
    assert rec["families_differ_at_t28"] is True
    assert rec["limited_family"] is True


def test_generate_uses_actual_regimen_and_iid_missingness():
    src = inspect.getsource(generate_dataset)
    assert "effect = gen.effect(grid, act, ct)" in src
    assert "if rng.random() < p_death:" in src
    assert "if day > 0 and rng.random() < p_missing:" in src
    assert "compound_ref.to_csv(meta_dir / \"compound_reference.csv\"" in src
    assert "write_json(truth_dir / \"params.json\"" in src
    a = GeneratorA()
    t = np.array([28.0])
    ct = {"A": 0.2, "ED50": 40.0, "tau_on": 6.0, "tau_off": 8.0}
    planned = Regimen(False, "CMP03", 100.0, 1.0, 0.0, 27.0, "ORAL GAVAGE")
    half = Regimen(False, "CMP03", 50.0, 1.0, 0.0, 27.0, "ORAL GAVAGE")
    ep = float(a.effect(t, planned, ct)[0])
    eh = float(a.effect(t, half, ct)[0])
    assert ep < eh < 0.0


def test_romano_order_statistic_matches_kth_and_numpy_higher():
    rng = np.random.default_rng(7)
    for n in (4, 20, 50, 172, 214):
        scores = rng.normal(size=n)
        k = romano_cqr_k(n, 0.2)
        assert k == int(np.ceil((n + 1) * 0.8))
        assert k <= n
        expect = float(np.sort(scores)[k - 1])
        assert romano_cqr_quantile(scores, 0.2) == pytest.approx(expect)
        assert conformal_margin(scores, 0.2) == pytest.approx(expect)
        cmp = compare_cqr_order_statistic(scores, 0.2)
        assert cmp["romano_matches_margin"] is True
        inverted = float(np.quantile(scores, cmp["q_used_by_legacy"], method="inverted_cdf"))
        assert inverted == pytest.approx(expect)
        if n >= 20:
            assert cmp["romano_matches_numpy_higher"] is False
    # n=3 at alpha=0.2 → k=n+1; practical rule uses the max
    tiny = np.array([0.1, -0.2, 0.4])
    assert romano_cqr_k(3, 0.2) == 4
    assert conformal_margin(tiny, 0.2) == pytest.approx(0.4)


def test_catboost_has_no_future_outcome_features():
    rec = catboost_information_set()
    assert rec["ok"] is True
    assert "y_logw" not in NUM_COLS
    assert "LOGW" not in NUM_COLS
    assert "target_day" in rec["known_future_calendar"]
    assert "Not MultiQuantile" in (QuantileCatBoostModel.__doc__ or "")


def test_chronos_history_only_drops_elapsed_day():
    series = {
        "USUBJID": "a", "DAY": np.array([0.0, 7.0, 21.0]),
        "LOGW": np.array([5.5, 5.6, 5.7]),
        "SEX": "M", "STRAIN": "WISTAR HAN", "SPECIES": "RAT",
        "ROUTE": "ORAL GAVAGE", "PLANNED_DOSE": 10.0,
    }
    hist = build_chronos_predict_item(series, np.array([28.0]), HISTORY_ONLY)
    assert isinstance(hist, np.ndarray)
    cov = build_chronos_predict_item(series, np.array([28.0]), COVARIATE_AWARE)
    assert cov["past_covariates"]["DAY"].tolist() == [0.0, 7.0, 21.0]
    assert cov["future_covariates"]["DAY"].tolist() == [28.0]
    note = chronos_visit_rank_note()
    assert note["official_seq_mode"] == HISTORY_ONLY
    assert note["not_a_chronos_inability"] is True


def test_classification_does_not_promote_finetune():
    rec = classification()
    assert rec["development_reference"] == "target-only TFT"
    assert rec["no_refit_required"] is True
    assert OUTPUT_DEFAULT == "runs/forecast_method_audit"
    assert rec["confirmed_bugs"]

"""Small Bayesian smoke (software check only; short chains are not adequate inference)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("nutpie")

pytestmark = [pytest.mark.bayesian, pytest.mark.slow]


@pytest.fixture(scope="module")
def bayes(trained):
    from send_bw.models.bayesian import BayesianModel

    m = BayesianModel()
    cfg = {"draws": 150, "tune": 150, "chains": 2, "seed": 3, "max_animals_per_study": 8,
           "sensitivity": {"enabled": True, "prior_scale_factors": [2.0], "student_t": False,
                           "reduced_random_effects": False, "draws": 80, "tune": 80}}
    m.fit(trained["data"]["train"], trained["data"]["validation"], cfg)
    return m


def test_diagnostics_recorded_and_reliability_is_a_decision(bayes):
    d = bayes.diagnostics["main"]
    for k in ["rhat_max", "ess_bulk_min", "ess_tail_min", "divergences", "thresholds", "reliable",
              "prior_predictive", "posterior_predictive", "sampler_warnings", "dropped_nonidentifiable_columns"]:
        assert k in d
    expected = (d["rhat_max"] <= 1.01 and d["ess_bulk_min"] >= 400 and d["ess_tail_min"] >= 400
                and d["divergences"] == 0)
    assert d["reliable"] == expected
    assert {r["variant"] for r in bayes.diagnostics["sensitivity_comparison"]} == {"main", "prior_scale_x2.0"}
    assert d["prior_predictive"]["grams_last_obs_q025_q50_q975"][0] > 0


def test_new_study_draws_use_new_effects(bayes, trained):
    ctx = trained["data"]["test"].animals.head(5).reset_index(drop=True)
    out = bayes.sample_study(ctx, None, np.array([7.0, 14.0, 28.0]), 300, seed=1)
    assert out["z"].shape == (300, 5, 3) and np.isfinite(out["z"]).all()
    dev = out["z"][:, :, -1] - out["mean"][:, :, -1]
    assert np.corrcoef(dev[:, 0], dev[:, 1])[0, 1] > 0.0  # shared NEW study effect
    assert dev.std() > 0


def test_save_load(bayes, trained, tmp_path):
    from send_bw.models.bayesian import BayesianModel

    bayes.save(tmp_path / "b")
    b2 = BayesianModel.load(tmp_path / "b")
    ctx = trained["data"]["test"].animals.head(3).reset_index(drop=True)
    t = np.array([28.0])
    np.testing.assert_allclose(bayes.predict_mean(ctx, t), b2.predict_mean(ctx, t))
    with pytest.raises(NotImplementedError):
        b2.predict_marginals(ctx, t, history=trained["data"]["test"].obs.head(3))

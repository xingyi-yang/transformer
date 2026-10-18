"""Response-model contracts: serialization, new-study inference, schedule invariance, uncertainty layers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from send_bw.models.spline import SplineModel
from send_bw.models.transformer import BaselineTransformer, TransformerModel


def _ctx(trained, n=4):
    return trained["data"]["test"].animals.head(n).reset_index(drop=True)


@pytest.mark.parametrize("name,cls", [("spline", SplineModel), ("transformer", TransformerModel)])
def test_save_load_reproduces_new_study_predictions(trained, tmp_path, name, cls):
    m = trained[name]
    ctx = _ctx(trained)
    times = np.array([3.0, 7.0, 14.0, 28.0])
    m.save(tmp_path / name)
    m2 = cls.load(tmp_path / name)
    np.testing.assert_allclose(m.predict_mean(ctx, times), m2.predict_mean(ctx, times), atol=1e-6)
    a = m.sample_study(ctx, None, times, 5, seed=11)["z"]
    b = m2.sample_study(ctx, None, times, 5, seed=11)["z"]
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert m2.spec.to_dict() == m.spec.to_dict()  # preprocessing / supported context persisted


def test_transformer_predictions_do_not_depend_on_query_schedule(trained):
    tm = trained["transformer"]
    ctx = _ctx(trained, 3)
    p1 = tm.predict_mean(ctx, np.array([7.0, 28.0]))
    p2 = tm.predict_mean(ctx, np.array([1.0, 3.0, 7.0, 10.0, 14.0, 28.0]))
    np.testing.assert_allclose(p1, p2[:, [2, 5]], atol=1e-5)


def test_transformer_rejects_incompatible_heads():
    with pytest.raises(ValueError):
        BaselineTransformer(group_dims={"bio": 2}, q_dim=3, d_model=30, n_heads=4, n_layers=1, dropout=0.0)


def test_transformer_requires_shared_covariance(trained):
    with pytest.raises(ValueError):
        TransformerModel().fit(trained["data"]["train"], trained["data"]["validation"], {})


def test_sample_study_uses_one_member_per_study_and_new_effects(trained):
    sm = trained["spline"]
    ctx = _ctx(trained, 6)
    times = np.array([7.0, 14.0, 28.0])
    out = sm.sample_study(ctx, None, times, 200, seed=5)
    assert out["z"].shape == (200, 6, 3) and len(out["members"]) == 200
    # the mean within a simulated study comes from a single member: identical across animals given features
    for k in range(3):
        mu = sm.predict_mean(ctx, times, member=int(out["members"][k]))
        np.testing.assert_allclose(out["mean"][k], mu, atol=1e-10)
    # new study effects -> animals in the same simulated study are positively correlated
    dev = out["z"][:, :, -1] - out["mean"][:, :, -1]
    c = np.corrcoef(dev[:, 0], dev[:, 1])[0, 1]
    assert c > 0.05


def test_predictive_variance_separates_layers(trained):
    sm = trained["spline"]
    ctx = _ctx(trained, 2)
    times = np.array([28.0])
    marg = sm.predict_marginals(ctx, times)
    bio = float(sm.cov.marginal_cov(times)[0, 0])
    assert (marg["VAR"] >= bio - 1e-12).all()  # biological + study + parameter (bootstrap) variance


def test_ids_are_not_predictors(trained):
    sm = trained["spline"]
    ctx = _ctx(trained, 2)
    ctx2 = ctx.assign(STUDYID="UNSEEN-STUDY", USUBJID=["X1", "X2"])
    np.testing.assert_allclose(sm.predict_mean(ctx, np.array([28.0])), sm.predict_mean(ctx2, np.array([28.0])))


def test_unseen_biological_category_is_refused(trained):
    sm = trained["spline"]
    ctx = _ctx(trained, 2).assign(STRAIN="UNSEEN-STRAIN")
    problems = sm.spec.check_context(ctx, sm.compound_ref, horizon=28)
    assert problems, "unseen strain must not be mapped silently to a familiar one"


def test_lognormal_mean_vs_median():
    from send_bw.evaluation.forecast import lognormal_mean

    mu, var = np.log(200.0), 0.04
    assert lognormal_mean(mu, var) == pytest.approx(200 * np.exp(0.02))
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(mu, np.sqrt(var), 400_000))
    assert s.mean() == pytest.approx(lognormal_mean(mu, var), rel=2e-3)
    assert np.median(s) == pytest.approx(200.0, rel=2e-3)


def test_early_history_conditioning_reduces_error(trained):
    from send_bw.evaluation.forecast import evaluate_early_history

    r = evaluate_early_history(trained["spline"], trained["data"]["test"], 14.0)
    assert r["status"] == "ok"
    assert r["conditioned"]["rmse_log"] <= r["pre_study"]["rmse_log"] + 1e-9


def test_check_context_horizon_gate(trained):
    sm = trained["spline"]
    ctx = _ctx(trained, 1)
    assert sm.spec.check_context(ctx, sm.compound_ref, horizon=28) == []
    assert sm.spec.check_context(ctx, sm.compound_ref, horizon=400)


def test_marginals_dataframe_contract(trained):
    out = trained["transformer"].predict_marginals(_ctx(trained, 2), np.array([7.0, 28.0]))
    assert isinstance(out, pd.DataFrame) and {"USUBJID", "DAY", "MEAN", "VAR"} <= set(out.columns)
    assert (out["VAR"] > 0).all()

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from send_bw.models.covariance import HierarchicalCovariance, PackedResiduals
from send_bw.simulation.analysis import endpoint_ancova, holm, rm_gls
from send_bw.simulation.designs import enumerate_candidates, load_request
from send_bw.simulation.metrics import clopper_pearson, prop_summary


def test_covariance_positive_definite_and_time_sensitive():
    c = HierarchicalCovariance(sd_s0=0.01, sd_s1=0.03, sd_a0=0.02, sd_a1=0.04, sigma_p=0.02, ell=6, sigma_m=0.01)
    t = np.array([3.0, 7, 14, 21, 28])
    S = c.marginal_cov(t)
    assert np.all(np.linalg.eigvalsh(S) > 0)
    assert np.all(np.diff(np.diag(S)) > 0)  # variance grows with time (slope effects)
    ou_only = HierarchicalCovariance(sd_s0=1e-6, sd_s1=1e-6, sd_a0=1e-6, sd_a1=1e-6, sigma_p=0.02, ell=6,
                                     sigma_m=0.01)
    A = ou_only.animal_cov(np.array([7.0, 8.0, 28.0]))
    assert A[0, 1] > A[0, 2] > 0
    assert A[0, 1] == pytest.approx(0.02**2 * np.exp(-1 / 6), rel=1e-6)


def test_sample_covariance_matches_analytic():
    c = HierarchicalCovariance(sd_s0=0.01, sd_s1=0.03, sd_a0=0.02, sd_a1=0.04, sigma_p=0.02, ell=6, sigma_m=0.01)
    t = np.array([7.0, 14, 28])
    x = c.sample(np.random.default_rng(1), t, 1, 40000)[:, 0, :]
    emp = np.cov(x.T)
    assert np.allclose(emp, c.marginal_cov(t), rtol=0.06, atol=2e-5)


def test_covariance_ml_recovers_parameters():
    true = HierarchicalCovariance(sd_s0=0.015, sd_s1=0.04, sd_a0=0.02, sd_a1=0.05, sigma_p=0.025, ell=5, sigma_m=0.01)
    rng = np.random.default_rng(3)
    rows = []
    t = np.array([3.0, 7, 10, 14, 17, 21, 24, 28])
    for s in range(40):
        x = true.sample(rng, t, 30, 1)[0]
        for i in range(30):
            for j, tt in enumerate(t):
                rows.append((f"S{s}", f"S{s}-{i}", tt, x[i, j]))
    df = pd.DataFrame(rows, columns=["STUDYID", "USUBJID", "DAY", "R"])
    fit = HierarchicalCovariance.fit(df)
    assert fit.fit_info["converged"]
    for p in ["sd_s1", "sd_a1", "sigma_p", "sd_a0"]:
        assert getattr(fit, p) == pytest.approx(getattr(true, p), rel=0.3), p
    # likelihood should favour the truth over a perturbed covariance
    packed = PackedResiduals.from_frame(df)
    worse = HierarchicalCovariance(**{**true.to_dict(), "sd_a1": 0.1, "fit_info": None})
    assert true.neg_loglik(packed) < worse.neg_loglik(packed)


def test_lognormal_mean_vs_median():
    mu, var = np.log(250.0), 0.05**2
    rng = np.random.default_rng(0)
    w = np.exp(rng.normal(mu, np.sqrt(var), 400000))
    assert np.median(w) == pytest.approx(np.exp(mu), rel=2e-3)
    assert w.mean() == pytest.approx(np.exp(mu + var / 2), rel=2e-3)


def _toy(rng, N=40, m=4, G=2, delta=(0.0, 0.0), noise=0.0):
    times = np.array([7.0, 14, 21, 28])[:m] if m <= 4 else np.linspace(4, 28, m)
    group = np.repeat(np.arange(G + 1), N // (G + 1) + 1)[:N]
    sex = np.tile([0, 1], N // 2 + 1)[:N]
    zb = rng.normal(5.4, 0.07, N)
    Z = zb[:, None] + 0.3 * times[None] / 28 + 0.1 * sex[:, None]
    for g in range(1, G + 1):
        Z[group == g] += delta[g - 1] * (times / 28)[None]
    Z = Z + noise * rng.standard_normal(Z.shape)
    return zb, sex, group, Z, times


def test_gls_recovers_known_contrast_exactly_without_noise():
    rng = np.random.default_rng(0)
    zb, sex, group, Z, times = _toy(rng, delta=(-0.05, -0.1), noise=0.0)
    Z = Z + 1e-6 * rng.standard_normal(Z.shape)
    V0 = HierarchicalCovariance().animal_cov(times)
    r = rm_gls(zb, sex, group, Z, times, 28.0, V0, 2)
    assert not r.failed
    assert np.allclose(r.est, [-0.05, -0.1], atol=1e-4)


def test_gls_requires_horizon_in_schedule():
    rng = np.random.default_rng(0)
    zb, sex, group, Z, times = _toy(rng)
    with pytest.raises(ValueError):
        rm_gls(zb, sex, group, Z[:, :3], times[:3], 28.0, np.eye(3), 2)


def test_ancova_ignores_intermediate_visits():
    rng = np.random.default_rng(1)
    zb, sex, group, Z, times = _toy(rng, delta=(-0.05, 0.0), noise=0.03)
    r1 = endpoint_ancova(zb, sex, group, Z[:, -1], 2)
    Z2 = Z.copy()
    Z2[:, :-1] = rng.normal(size=Z2[:, :-1].shape)  # scramble intermediate visits
    r2 = endpoint_ancova(zb, sex, group, Z2[:, -1], 2)
    assert np.allclose(r1.est, r2.est) and np.allclose(r1.se, r2.se)


def test_ancova_candidates_never_carry_intermediate_visits():
    req = load_request(ROOT / "configs" / "new_study.example.yaml")
    for c in enumerate_candidates(req):
        if c.analysis == "ancova":
            assert c.schedule == (req.horizon_day,)
        assert c.schedule[-1] == req.horizon_day


def test_holm():
    assert list(holm(np.array([0.01, 0.04, 0.03]), 0.05)) == [True, False, False]
    assert list(holm(np.array([0.01, 0.02, 0.03]), 0.05)) == [True, True, True]
    assert list(holm(np.array([0.2, 0.01]), 0.05)) == [False, True]
    assert not holm(np.array([0.03, 0.04]), 0.05).any()


def test_clopper_pearson_finite_at_extremes():
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0 and 0 < hi < 0.05
    lo, hi = clopper_pearson(100, 100)
    assert hi == 1 and 0.95 < lo < 1
    s = prop_summary(np.zeros(50, bool))
    assert s["est"] == 0 and np.isfinite(s["ci_hi"])


def test_analysis_failure_handling():
    rng = np.random.default_rng(0)
    zb, sex, group, Z, times = _toy(rng, N=6)
    r = endpoint_ancova(zb, sex, group, Z[:, -1], 2)
    assert r.failed and r.reason
    Z[group == 2] = np.nan
    r = rm_gls(zb, sex, group, Z, times, 28.0, np.eye(4) * 0.001, 2)
    assert r.failed and r.reason.startswith("empty_group")


@pytest.mark.slow
def test_gls_null_calibration_under_correct_covariance():
    cov = HierarchicalCovariance(sd_s0=0.0001, sd_s1=0.0001, sd_a0=0.02, sd_a1=0.04, sigma_p=0.02, ell=6, sigma_m=0.01)
    times = np.array([7.0, 14, 21, 28])
    rng = np.random.default_rng(11)
    rej = []
    for _ in range(1500):
        N = 36
        group = np.repeat([0, 1, 2], 12)
        sex = np.tile([0, 1], 18)
        zb = rng.normal(5.4, 0.07, N)
        Z = zb[:, None] + 0.3 * times / 28 + cov.sample(rng, times, N, 1)[0]
        r = rm_gls(zb, sex, group, Z, times, 28.0, cov.animal_cov(times), 2)
        rej.append(holm(r.p, 0.05).any())
    fwer = np.mean(rej)
    lo, hi = clopper_pearson(int(np.sum(rej)), len(rej))
    assert lo <= 0.05 + 0.01 and fwer < 0.075

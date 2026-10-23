"""Deterministic checks for experimental mmrm_visit_contrast (Candidate C1).

No Monte Carlo calibration. Production request configs default to ancova only.
gls and mmrm remain requestable; mmrm is not a default.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import ROOT

from send_bw.models.covariance import HierarchicalCovariance
from send_bw.simulation.analysis import (
    endpoint_ancova,
    mmrm_visit_contrast,
    rm_gls,
    secondary_visit_weights,
)
from send_bw.simulation.designs import Request, enumerate_candidates, load_request
from send_bw.simulation.paths import fixed_shape


def _fixture(n_per_group: int = 16, unbalanced: bool = False):
    groups, sexes, zbs = [], [], []
    for g in range(3):
        for s in (0, 1):
            u = (np.arange(1, n_per_group + 1, dtype=float) - 0.5) / n_per_group
            z = 5.27 + 0.29 * s + (0.04 * (g - 1) if unbalanced else 0.0) + 0.068 * (u - 0.5) * 2.0
            groups.append(np.full(n_per_group, g, dtype=int))
            sexes.append(np.full(n_per_group, s, dtype=int))
            zbs.append(z)
    return np.concatenate(zbs), np.concatenate(sexes), np.concatenate(groups)


def _mean_Z(zb, sex, group, times, deltas, shape: str):
    T = 28.0
    u = times / T
    zb_c = zb - zb.mean()
    Z = 0.18 * u[None] - 0.03 * (u**2)[None] + (sex == 1)[:, None] * (0.015 + 0.025 * u[None])
    Z = Z + zb_c[:, None] * (1.0 - 0.05 * u[None])
    s = fixed_shape(shape, times, T)
    for g, d in enumerate(deltas, start=1):
        Z = Z + (group == g)[:, None] * (d * s[None])
    return Z


def test_default_requests_are_ancova_only():
    bare = Request(
        request_id="T",
        species="RAT",
        strain="SPRAGUE-DAWLEY",
        baseline_distribution={"M": {"mean_g": 250.0, "sd_g": 15.0}, "F": {"mean_g": 190.0, "sd_g": 12.0}},
        groups=[{"name": "control", "is_control": True, "end_dy": 28},
                {"name": "low", "compound_id": "CMP03", "dose": 30, "end_dy": 28}],
        horizon_day=28,
        meaningful_effects_pct={"low": -8.0},
        candidate_schedules={"endpoint_only": [28], "weekly": [7, 14, 21, 28]},
    )
    assert bare.analyses == ["ancova"]
    assert {c.analysis for c in enumerate_candidates(bare)} == {"ancova"}

    for name in ("new_study.example.yaml", "new_study.smoke.yaml"):
        req = load_request(ROOT / "configs" / name)
        assert req.analyses == ["ancova"]
        assert "mmrm" not in req.analyses
        assert "gls" not in req.analyses
        assert {c.analysis for c in enumerate_candidates(req)} == {"ancova"}


def test_gls_and_mmrm_remain_requestable():
    req = load_request(ROOT / "configs" / "new_study.example.yaml")
    over = req.model_copy(update={"analyses": ["ancova", "gls", "mmrm"]})
    names = {c.analysis for c in enumerate_candidates(over)}
    assert names == {"ancova", "gls", "mmrm"}
    assert all(c.schedule == (req.horizon_day,) for c in enumerate_candidates(over) if c.analysis == "ancova")
    assert all(len(c.schedule) > 1 for c in enumerate_candidates(over) if c.analysis in {"gls", "mmrm"})


def test_mmrm_recovers_transient_endpoint_where_gls_does_not():
    zb, sex, group = _fixture()
    times = np.array([7.0, 14.0, 21.0, 28.0])
    deltas = (-0.10, -0.15)
    Z = _mean_Z(zb, sex, group, times, deltas, "transient")
    V0 = HierarchicalCovariance().animal_cov(times)
    a = endpoint_ancova(zb, sex, group, Z[:, -1], 2)
    b = rm_gls(zb, sex, group, Z, times, 28.0, V0, 2)
    c = mmrm_visit_contrast(zb, sex, group, Z, times, 28.0, V0, 2)
    assert not a.failed and not b.failed and not c.failed
    assert np.allclose(a.est, deltas, atol=1e-12)
    assert np.allclose(c.est, deltas, atol=1e-12)
    assert np.max(np.abs(b.est - np.asarray(deltas))) > 0.01


def test_complete_data_coefficients_match_ancova_any_v0():
    zb, sex, group = _fixture()
    times = np.array([7.0, 14.0, 21.0, 28.0])
    resid = np.column_stack([
        0.020 * np.cos(np.arange(len(zb)) * 0.9 + j * 0.5) for j in range(len(times))
    ])
    Z = _mean_Z(zb, sex, group, times, (-0.10, -0.15), "delayed") + resid
    a = endpoint_ancova(zb, sex, group, Z[:, -1], 2)
    for V0 in (HierarchicalCovariance().animal_cov(times), np.eye(len(times))):
        c = mmrm_visit_contrast(zb, sex, group, Z, times, 28.0, V0, 2)
        assert not c.failed
        assert np.allclose(c.est, a.est, atol=1e-12)


def test_realized_sandwich_se_is_not_ancova_ols_se():
    zb, sex, group = _fixture()
    times = np.array([7.0, 14.0, 21.0, 28.0])
    raw = np.column_stack([
        0.020 * np.cos(np.arange(len(zb)) * 0.9 + j * 0.5)
        + 0.015 * ((np.arange(len(zb)) % 3 == 2) * (1.0 + 0.4 * j))
        for j in range(len(times))
    ])
    raw[group == 2, -1] *= 1.8
    Z = 0.10 * (times / 28.0)[None] + raw
    a = endpoint_ancova(zb, sex, group, Z[:, -1], 2)
    c = mmrm_visit_contrast(zb, sex, group, Z, times, 28.0, HierarchicalCovariance().animal_cov(times), 2)
    assert np.allclose(c.est, a.est, atol=1e-12)
    assert np.max(np.abs(c.se / a.se - 1.0)) > 0.02


def test_df_matches_endpoint_animal_count():
    zb, sex, group = _fixture(n_per_group=8)
    times = np.array([14.0, 28.0])
    Z = _mean_Z(zb, sex, group, times, (0.0, 0.0), "linear")
    Z = Z + 0.01 * np.arange(Z.size).reshape(Z.shape)
    c = mmrm_visit_contrast(zb, sex, group, Z, times, 28.0, np.eye(2), 2)
    n_end = int(np.isfinite(Z[:, -1]).sum())
    assert c.df == pytest.approx(n_end - (3 + 2))


def test_secondary_only_on_observed_schedule_visits():
    zb, sex, group = _fixture()
    weekly = np.array([7.0, 14.0, 21.0, 28.0])
    biweekly = np.array([14.0, 28.0])
    Z = _mean_Z(zb, sex, group, weekly, (-0.10, -0.15), "linear")
    sec = (np.array([7.0, 14.0, 21.0, 28.0]), np.full(4, 0.25))
    assert secondary_visit_weights(*sec, weekly) is not None
    assert secondary_visit_weights(*sec, biweekly) is None
    V0 = HierarchicalCovariance().animal_cov(weekly)
    ok = mmrm_visit_contrast(zb, sex, group, Z, weekly, 28.0, V0, 2, secondary=sec)
    sparse = mmrm_visit_contrast(
        zb, sex, group, Z[:, [1, 3]], biweekly, 28.0, HierarchicalCovariance().animal_cov(biweekly), 2, secondary=sec
    )
    assert ok.sec_est is not None
    assert sparse.sec_est is None
    assert np.allclose(ok.sec_est, np.array([-0.10, -0.15]) * float((weekly / 28.0 * 0.25).sum()), atol=1e-12)


def test_mmrm_fails_on_empty_endpoint_group():
    zb, sex, group = _fixture(n_per_group=8)
    times = np.array([7.0, 14.0, 21.0, 28.0])
    Z = _mean_Z(zb, sex, group, times, (0.0, 0.0), "linear")
    Z[group == 2, -1] = np.nan
    r = mmrm_visit_contrast(zb, sex, group, Z, times, 28.0, np.eye(4), 2)
    assert r.failed and r.reason.startswith("empty_group")


def test_mmrm_requires_horizon():
    zb, sex, group = _fixture(n_per_group=8)
    times = np.array([7.0, 14.0, 21.0])
    Z = np.zeros((len(zb), 3))
    with pytest.raises(ValueError):
        mmrm_visit_contrast(zb, sex, group, Z, times, 28.0, np.eye(3), 2)

"""Hierarchical covariance for change-from-baseline log weight residuals.

r_sgi(t) = B(t)'u_s + B(t)'b_i + e_i(t),  B(t) = [1, t/28]
u_s ~ N(0, diag(sd_s0^2, sd_s1^2)),  b_i ~ N(0, diag(sd_a0^2, sd_a1^2))
Cov(e_i(t), e_i(t')) = sigma_p^2 exp(-|t-t'|/ell) + sigma_m^2 1[t = t']

The animal intercept absorbs baseline measurement error (Y is anchored on the observed
baseline). Fitted by maximum likelihood on study-level out-of-fold residuals, using a
Woodbury identity per study so studies with many animals stay cheap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

T_REF = 28.0
PARAMS = ["sd_s0", "sd_s1", "sd_a0", "sd_a1", "sigma_p", "ell", "sigma_m"]
JITTER = 1e-9


@dataclass
class HierarchicalCovariance:
    sd_s0: float = 0.01
    sd_s1: float = 0.03
    sd_a0: float = 0.01
    sd_a1: float = 0.04
    sigma_p: float = 0.02
    ell: float = 5.0
    sigma_m: float = 0.01
    fit_info: dict | None = None

    @staticmethod
    def basis(times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, float)
        return np.stack([np.ones_like(t), t / T_REF], axis=-1)

    def animal_cov(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, float)
        Z = self.basis(t)
        Oa = np.diag([self.sd_a0**2, self.sd_a1**2])
        K = self.sigma_p**2 * np.exp(-np.abs(t[:, None] - t[None, :]) / self.ell)
        return Z @ Oa @ Z.T + K + (self.sigma_m**2 + JITTER) * np.eye(len(t))

    def study_cov(self, times: np.ndarray) -> np.ndarray:
        Z = self.basis(times)
        return Z @ np.diag([self.sd_s0**2, self.sd_s1**2]) @ Z.T

    def marginal_cov(self, times: np.ndarray) -> np.ndarray:
        return self.animal_cov(times) + self.study_cov(times)

    # ---------------------------------------------------------------- simulation
    def sample(self, rng: np.random.Generator, times: np.ndarray, n_animals: int, n_draws: int,
               scale: float = 1.0) -> np.ndarray:
        """(n_draws, n_animals, n_times): one new study effect per draw, animal effects per animal."""
        times = np.asarray(times, float)
        Z = self.basis(times)
        La = _safe_chol(self.animal_cov(times))
        us = rng.standard_normal((n_draws, 2)) * np.array([self.sd_s0, self.sd_s1])
        study = us @ Z.T  # (n_draws, T)
        animal = rng.standard_normal((n_draws, n_animals, len(times))) @ La.T
        return scale * (study[:, None, :] + animal)

    # ---------------------------------------------------------------- likelihood
    def to_vector(self) -> np.ndarray:
        return np.log([getattr(self, p) for p in PARAMS])

    @staticmethod
    def from_vector(v: np.ndarray) -> HierarchicalCovariance:
        return HierarchicalCovariance(**dict(zip(PARAMS, np.exp(v).tolist(), strict=True)))

    def neg_loglik(self, packed: PackedResiduals) -> float:
        p = packed
        t = p.times
        n, L = t.shape
        Za = np.stack([np.ones_like(t), t / T_REF], axis=-1) * p.mask[..., None]
        Oa = np.array([self.sd_a0**2, self.sd_a1**2])
        A = np.einsum("nik,k,njk->nij", Za, Oa, Za)
        d = np.abs(t[:, :, None] - t[:, None, :])
        mm = p.mask[:, :, None] * p.mask[:, None, :]
        A = A + mm * self.sigma_p**2 * np.exp(-d / self.ell)
        eye = np.eye(L)[None]
        A = A + eye * (p.mask[:, :, None] * (self.sigma_m**2 + JITTER) + (1 - p.mask[:, :, None]))
        try:
            C = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return 1e12
        logdetA = 2.0 * np.log(np.diagonal(C, axis1=1, axis2=2)).sum(1)
        rhs = np.concatenate([p.r[..., None], Za], axis=2)
        sol = np.linalg.solve(A, rhs)
        Ainv_r, Ainv_Z = sol[..., 0], sol[..., 1:]
        q = np.einsum("ni,ni->n", p.r, Ainv_r)
        S = np.einsum("nik,nil->nkl", Za, Ainv_Z)
        v = np.einsum("nik,ni->nk", Za, Ainv_r)
        ns = p.n_studies
        Ss = np.zeros((ns, 2, 2))
        vs = np.zeros((ns, 2))
        qs = np.zeros(ns)
        lds = np.zeros(ns)
        np.add.at(Ss, p.study_idx, S)
        np.add.at(vs, p.study_idx, v)
        np.add.at(qs, p.study_idx, q)
        np.add.at(lds, p.study_idx, logdetA)
        Os = np.array([max(self.sd_s0**2, 1e-12), max(self.sd_s1**2, 1e-12)])
        M = Ss + np.diag(1.0 / Os)[None]
        sign, ldM = np.linalg.slogdet(M)
        if np.any(sign <= 0):
            return 1e12
        quad = qs - np.einsum("sk,sk->s", vs, np.linalg.solve(M, vs[..., None])[..., 0])
        ll = -0.5 * (quad + lds + np.log(Os).sum() + ldM + p.n_obs_study * np.log(2 * np.pi))
        return float(-ll.sum())

    @staticmethod
    def fit(resid: pd.DataFrame, init: HierarchicalCovariance | None = None) -> HierarchicalCovariance:
        """resid columns: STUDYID, USUBJID, DAY, R."""
        packed = PackedResiduals.from_frame(resid)
        x0 = (init or HierarchicalCovariance()).to_vector()
        lo = np.log([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.3, 1e-4])
        hi = np.log([0.5, 0.5, 0.5, 0.5, 0.5, 200.0, 0.5])
        res = minimize(lambda v: HierarchicalCovariance.from_vector(v).neg_loglik(packed), x0,
                       method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)))
        cov = HierarchicalCovariance.from_vector(res.x)
        at_bound = [PARAMS[i] for i in range(len(PARAMS)) if abs(res.x[i] - lo[i]) < 1e-3 or abs(res.x[i] - hi[i]) < 1e-3]
        cov.fit_info = {"converged": bool(res.success), "message": str(res.message), "nll": float(res.fun),
                        "n_obs": int(packed.mask.sum()), "n_animals": int(packed.r.shape[0]),
                        "n_studies": int(packed.n_studies), "params_at_bound": at_bound}
        return cov

    # ---------------------------------------------------------------- conditioning (early history)
    def condition_study(self, t_obs: list[np.ndarray], r_obs: list[np.ndarray], t_new: list[np.ndarray]
                        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Joint Gaussian conditioning within one study (shared study effect couples animals)."""
        n = len(t_obs)
        all_t = [np.concatenate([a, b]) for a, b in zip(t_obs, t_new, strict=True)]
        sizes = [len(x) for x in all_t]
        offs = np.cumsum([0] + sizes)
        N = offs[-1]
        Sigma = np.zeros((N, N))
        Zs = np.concatenate([self.basis(x) for x in all_t])
        Sigma += Zs @ np.diag([self.sd_s0**2, self.sd_s1**2]) @ Zs.T
        for i in range(n):
            sl = slice(offs[i], offs[i + 1])
            Sigma[sl, sl] += self.animal_cov(all_t[i])
        o_idx = np.concatenate([np.arange(offs[i], offs[i] + len(t_obs[i])) for i in range(n)]).astype(int)
        n_idx = np.concatenate([np.arange(offs[i] + len(t_obs[i]), offs[i + 1]) for i in range(n)]).astype(int)
        r = np.concatenate(r_obs) if n else np.zeros(0)
        Soo = Sigma[np.ix_(o_idx, o_idx)]
        Sno = Sigma[np.ix_(n_idx, o_idx)]
        Snn = Sigma[np.ix_(n_idx, n_idx)]
        if len(o_idx):
            Lc = _safe_chol(Soo)
            a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, r))
            B = np.linalg.solve(Lc, Sno.T)
            mean = Sno @ a
            cov = Snn - B.T @ B
        else:
            mean, cov = np.zeros(len(n_idx)), Snn
        means, covs, k = [], [], 0
        for i in range(n):
            m = len(t_new[i])
            means.append(mean[k:k + m])
            covs.append(cov[k:k + m, k:k + m])
            k += m
        return means, covs

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> HierarchicalCovariance:
        return HierarchicalCovariance(**d)


def _safe_chol(S: np.ndarray, max_tries: int = 6) -> np.ndarray:
    jitter = 0.0
    scale = max(float(np.mean(np.diag(S))), 1e-12)
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(S + jitter * np.eye(len(S)))
        except np.linalg.LinAlgError:
            jitter = scale * 10.0 ** (-10 + 2 * i)
    raise np.linalg.LinAlgError("covariance not positive definite after bounded jitter")


@dataclass
class PackedResiduals:
    times: np.ndarray  # (n_animals, L), padded
    r: np.ndarray
    mask: np.ndarray
    study_idx: np.ndarray
    n_studies: int
    n_obs_study: np.ndarray

    @staticmethod
    def from_frame(df: pd.DataFrame) -> PackedResiduals:
        df = df.sort_values(["STUDYID", "USUBJID", "DAY"])
        groups = list(df.groupby("USUBJID", sort=False))
        L = max(len(g) for _, g in groups)
        n = len(groups)
        times = np.zeros((n, L))
        r = np.zeros((n, L))
        mask = np.zeros((n, L))
        studies = []
        for i, (_, g) in enumerate(groups):
            k = len(g)
            times[i, :k] = g["DAY"].to_numpy(float)
            r[i, :k] = g["R"].to_numpy(float)
            mask[i, :k] = 1.0
            studies.append(g["STUDYID"].iloc[0])
        codes, uniq = pd.factorize(pd.Series(studies))
        nos = np.zeros(len(uniq))
        np.add.at(nos, codes, mask.sum(1))
        return PackedResiduals(times, r, mask, codes, len(uniq), nos)

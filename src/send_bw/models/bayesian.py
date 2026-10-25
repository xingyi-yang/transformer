"""Bayesian hierarchical longitudinal benchmark (benchmark B), PyMC + ArviZ.

Y_sgi(t) / s_y = X(t) beta + B(t)'u_s + B(t)'b_i + e_i(t),   B(t) = [1, t/28]
  beta_k ~ Normal(0, s_beta^2)        (X columns standardized on training data only)
  u_s = sd_s * z_s,  b_i = sd_a * z_i  (non-centered), z ~ ZeroSumNormal(1) over training groups
  (identifies the population mean through the fixed effects), sd ~ HalfNormal(s_sd)
  e_i(t): continuous-time Ornstein-Uhlenbeck residual with sigma_p ~ HalfNormal(s_sigma) and
          length scale ell ~ LogNormal(m_ell, s_ell) days, via the exact conditional factorization
          e_j | e_{j-1} ~ Normal(rho_j e_{j-1}, sigma_p^2 (1 - rho_j^2)),  rho_j = exp(-dt_j / ell).
  Optional heavy-tailed sensitivity: Student-t innovations.
The baseline anchor makes intercepts identifiable: there is no separate global intercept beyond
the spline basis, and the animal intercept absorbs baseline measurement error. No separate
white-noise nugget is modelled (it is not identifiable from the OU term with these schedules
without a latent process); this is a documented simplification.

New-study prediction draws ONE posterior draw per simulated study, a NEW study effect from
N(0, sd_s^2) and NEW animal effects; fitted training effects are never reused.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import FeatureSpec
from send_bw.models.base import ModelData, ResponseModel, merge_context
from send_bw.models.design import MeanDesign
from send_bw.models.spline import context_features
from send_bw.utils.artifacts import get_logger

PARAM_VARS = ["beta", "sd_s", "sd_a", "sigma_p", "ell"]
DIAG_VARS = ["theta", "beta", "sd_s", "sd_a", "sigma_p", "ell"]


def _require_pymc():
    try:
        import arviz as az  # noqa: F401
        import pymc as pm  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("The Bayesian benchmark requires the 'bayesian' extra: pip install 'send-bw[bayesian]'") from e


def _ordering(obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For obs sorted by (USUBJID, DAY): previous index (or -1) and gap in days."""
    u = obs["USUBJID"].to_numpy()
    d = obs["DAY"].to_numpy(float)
    prev = np.arange(len(obs)) - 1
    first = np.r_[True, u[1:] != u[:-1]]
    prev[first] = -1
    dt = np.where(first, 0.0, d - np.r_[d[0], d[:-1]])
    return prev, dt, first


class BayesianModel(ResponseModel):
    name = "bayesian"

    def __init__(self):
        self.spec: FeatureSpec | None = None
        self.design: MeanDesign | None = None
        self.compound_ref = pd.DataFrame()
        self.draws: dict[str, np.ndarray] = {}
        self.s_y = 1.0
        self.priors: dict[str, float] = {}
        self.diagnostics: dict[str, Any] = {}
        self.history: dict[str, Any] = {}

    # ---------------------------------------------------------------- model
    def _build(self, X, ys, study_idx, animal_idx, tau, prev, dt, first, n_s, n_a, priors, likelihood="normal",
               random_effects="full"):
        import pymc as pm
        import pytensor.tensor as pt

        with pm.Model() as model:
            # QR reparameterization: X = Q* R*, theta = R* beta; the prior is placed on theta
            theta = pm.Normal("theta", 0.0, priors["s_beta"], shape=X.shape[1])
            pm.Deterministic("beta", pt.dot(self._Rinv, theta))
            sd_s = pm.HalfNormal("sd_s", priors["s_sd"], shape=2)
            sd_a = pm.HalfNormal("sd_a", priors["s_sd"], shape=2)
            # zero-sum over groups: identifies the population mean through the fixed effects only
            z_s = pm.ZeroSumNormal("z_s", sigma=1.0, shape=(2, n_s)).T
            z_a = pm.ZeroSumNormal("z_a", sigma=1.0, shape=(2, n_a)).T
            sigma_p = pm.HalfNormal("sigma_p", priors["s_sigma"])
            ell = pm.LogNormal("ell", priors["m_ell"], priors["s_ell"])
            u = z_s * sd_s
            b = z_a * sd_a
            if random_effects == "no_animal_intercept":
                b = pt.concatenate([pt.zeros((n_a, 1)), b[:, 1:]], axis=1)
            mu = pt.dot(X, theta) + u[study_idx, 0] + u[study_idx, 1] * tau + b[animal_idx, 0] + b[animal_idx, 1] * tau
            rho = pt.exp(-dt / ell) * (1 - first)
            resid_prev = (ys[np.maximum(prev, 0)] - mu[np.maximum(prev, 0)]) * (1 - first)
            cmean = mu + rho * resid_prev
            csd = sigma_p * pt.sqrt(1 - rho**2 + 1e-9)
            if likelihood == "student_t":
                nu = pm.Gamma("nu", 2.0, 0.1)
                pm.StudentT("y", nu=nu, mu=cmean, sigma=csd, observed=ys)
            else:
                pm.Normal("y", mu=cmean, sigma=csd, observed=ys)
        return model

    def _arrays(self, d: ModelData, fit_scale: bool = False):
        obs = d.obs.sort_values(["USUBJID", "DAY"]).reset_index(drop=True)
        F = d.spec.observation_features(obs, d.animals, d.compound_ref)
        X = self.design.matrix(F, fit_scale=fit_scale)
        s_codes, s_uni = pd.factorize(obs["STUDYID"])
        a_codes, a_uni = pd.factorize(obs["USUBJID"])
        prev, dt, first = _ordering(obs)
        return obs, X, s_codes, a_codes, len(s_uni), len(a_uni), prev, dt, first.astype(float)

    def fit(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]:
        _require_pymc()
        import arviz as az
        import pymc as pm

        log = get_logger()
        self.spec, self.compound_ref = train.spec, train.compound_ref
        if config.get("max_animals_per_study"):
            rng = np.random.default_rng(config.get("seed", 0))
            keep = []
            for _, g in train.animals.groupby("STUDYID"):
                k = min(len(g), int(config["max_animals_per_study"]))
                keep += list(rng.choice(g["USUBJID"].to_numpy(), size=k, replace=False))
            train = ModelData(train.animals[train.animals["USUBJID"].isin(keep)].reset_index(drop=True),
                              train.obs[train.obs["USUBJID"].isin(keep)].reset_index(drop=True),
                              train.compound_ref, train.spec)
        F = train.spec.observation_features(train.obs, train.animals, train.compound_ref)
        sharing = getattr(train.spec, "sharing", "context_specific")
        regularized = sharing == "selective_regularized"
        self.design = MeanDesign.from_features(F, config.get("n_time_knots", 3), config.get("n_expo_knots", 2),
                                               rich=False, species_treatment=regularized)
        obs, X, s_idx, a_idx, n_s, n_a, prev, dt, first = self._arrays(train, fit_scale=True)
        from scipy.linalg import qr as pivoted_qr

        p_full = X.shape[1]
        _, Rp, piv = pivoted_qr(X, mode="economic", pivoting=True)
        d = np.abs(np.diag(Rp))
        rank = int((d > 1e-8 * d[0]).sum())
        keep = np.sort(piv[:rank])
        self._dropped = [int(j) for j in np.setdiff1d(np.arange(p_full), keep)]
        if self._dropped:
            log.info(f"bayesian: dropping {len(self._dropped)} non-identifiable design column(s) {self._dropped}")
        X = X[:, keep]
        n = X.shape[0]
        Q, R = np.linalg.qr(X)
        Rs = R / np.sqrt(n - 1)
        self._Rinv = np.linalg.inv(Rs)
        X = Q * np.sqrt(n - 1)  # mu = X_orig beta = Q* theta
        self.s_y = float(obs["Y"].std())
        ys = obs["Y"].to_numpy(float) / self.s_y
        tau = obs["DAY"].to_numpy(float) / 28.0
        # scales on the training-standardized Y/s_y scale; justified by the prior predictive check in grams
        self.priors = {"s_beta": 0.35, "s_sd": 0.3, "s_sigma": 0.3, "m_ell": float(np.log(7.0)), "s_ell": 0.75,
                       **config.get("priors", {})}
        variants = {"main": (self.priors, "normal", "full")}
        sens_cfg = config.get("sensitivity", {})
        if sens_cfg.get("enabled", True):
            for f in sens_cfg.get("prior_scale_factors", [0.5, 2.0]):
                variants[f"prior_scale_x{f}"] = ({**self.priors, "s_beta": self.priors["s_beta"] * f,
                                                  "s_sd": self.priors["s_sd"] * f,
                                                  "s_sigma": self.priors["s_sigma"] * f}, "normal", "full")
            if sens_cfg.get("student_t", True):
                variants["student_t"] = (self.priors, "student_t", "full")
            if sens_cfg.get("reduced_random_effects", True):
                variants["no_animal_intercept"] = (self.priors, "normal", "no_animal_intercept")
        results: dict[str, Any] = {}
        seed = int(config.get("seed", 0))
        for vname, (pri, lik, re) in variants.items():
            main = vname == "main"
            draws = int(config.get("draws", 500) if main else sens_cfg.get("draws", config.get("draws", 500)))
            tune = int(config.get("tune", 500) if main else sens_cfg.get("tune", config.get("tune", 500)))
            chains = int(config.get("chains", 2))
            model = self._build(X, ys, s_idx, a_idx, tau, prev, dt, first, n_s, n_a, pri, lik, re)
            t0 = time.time()
            with model, warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                idata = pm.sample(draws=draws, tune=tune, chains=chains, nuts_sampler=config.get("sampler", "nutpie"),
                                  random_seed=seed, progressbar=False, target_accept=config.get("target_accept", 0.9))
                if main:
                    prior = pm.sample_prior_predictive(int(config.get("prior_predictive_draws", 200)),
                                                       random_seed=seed + 1)
                    ppc = pm.sample_posterior_predictive(idata, random_seed=seed + 2, progressbar=False)
            elapsed = time.time() - t0
            diag = self._diagnostics(az, idata, [str(w.message)[:200] for w in wlist], config)
            diag.update({"draws": draws, "tune": tune, "chains": chains, "seed": seed, "seconds": elapsed,
                         "likelihood": lik, "random_effects": re, "priors": pri,
                         "dropped_nonidentifiable_columns": self._dropped})
            post = {v: idata.posterior[v].values.reshape(-1, *idata.posterior[v].shape[2:]) for v in PARAM_VARS}
            bfull = np.zeros((post["beta"].shape[0], p_full))
            bfull[:, keep] = post["beta"]
            post["beta"] = bfull  # non-identifiable columns fixed at 0
            diag["posterior_summary"] = {
                "sd_s": post["sd_s"].mean(0).tolist(), "sd_a": post["sd_a"].mean(0).tolist(),
                "sigma_p": float(post["sigma_p"].mean()), "ell": float(post["ell"].mean()),
                "sd_s_scaled_logw": (post["sd_s"].mean(0) * self.s_y).tolist(),
                "sd_a_scaled_logw": (post["sd_a"].mean(0) * self.s_y).tolist(),
                "sigma_p_scaled_logw": float(post["sigma_p"].mean() * self.s_y)}
            results[vname] = {"diagnostics": diag, "post": post}
            log.info(f"bayesian[{vname}] {elapsed:.0f}s rhat_max={diag['rhat_max']:.3f} "
                     f"ess_bulk_min={diag['ess_bulk_min']:.0f} divergences={diag['divergences']}")
            if main:
                results[vname]["prior_check"] = self._prior_predictive(prior, obs)
                results[vname]["ppc"] = self._posterior_predictive(ppc, obs, ys)
        self.draws = results["main"]["post"]
        self.diagnostics = {k: {**v["diagnostics"], **({"prior_predictive": v["prior_check"],
                                                         "posterior_predictive": v["ppc"]} if k == "main" else {})}
                            for k, v in results.items()}
        self.diagnostics["sensitivity_comparison"] = self._sensitivity_table(results, validation)
        self.history = {"main": self.diagnostics["main"], "n_obs": len(obs), "n_animals": n_a, "n_studies": n_s,
                        "n_beta": int(X.shape[1]), "s_y": self.s_y, "reliable": self.diagnostics["main"]["reliable"],
                        "species_treatment": regularized, "sharing": sharing}
        from send_bw.sharing import is_selective, selective_record

        if is_selective(sharing):
            self.history["selective_sharing"] = selective_record(
                "bayesian", "regularized" if regularized else "v1")
        return self.history

    def _diagnostics(self, az, idata, wmsgs, config) -> dict[str, Any]:
        summ = az.summary(idata, var_names=DIAG_VARS)
        div = int(np.asarray(idata.sample_stats["diverging"].values).sum())
        maxd = idata.sample_stats.get("maxdepth_reached") if hasattr(idata.sample_stats, "get") else None
        th = config.get("thresholds", {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400})
        d = {"rhat_max": float(summ["r_hat"].max()), "ess_bulk_min": float(summ["ess_bulk"].min()),
             "ess_tail_min": float(summ["ess_tail"].min()), "divergences": div,
             "maxdepth_hits": int(np.asarray(maxd.values).sum()) if maxd is not None else None,
             "sampler_warnings": wmsgs[:20], "thresholds": th,
             "worst_rhat_params": summ["r_hat"].sort_values(ascending=False).head(5).to_dict()}
        d["reliable"] = bool(d["rhat_max"] <= th["rhat_max"] and d["ess_bulk_min"] >= th["ess_bulk_min"]
                             and d["ess_tail_min"] >= th["ess_tail_min"] and div == 0)
        return d

    def _prior_predictive(self, prior, obs) -> dict[str, Any]:
        yp = np.asarray(prior["prior_predictive"]["y"].values).reshape(-1, len(obs)) * self.s_y
        last = obs.groupby("USUBJID").tail(1).index.to_numpy()
        pct = 100 * (np.exp(yp[:, last]) - 1)
        gm = np.exp(obs.loc[last, "BASELINE_LOGW"].to_numpy()[None] + yp[:, last])
        q = np.percentile(pct, [2.5, 50, 97.5])
        return {"pct_change_last_obs_q025_q50_q975": q.tolist(),
                "grams_last_obs_q025_q50_q975": np.percentile(gm, [2.5, 50, 97.5]).tolist(),
                "observed_pct_change_q025_q50_q975": np.percentile(100 * (np.exp(obs.loc[last, "Y"]) - 1),
                                                                   [2.5, 50, 97.5]).tolist(),
                "plausible": bool(q[0] > -90 and q[2] < 400),
                "note": "weakly informative: prior predictive should cover, not match, observed changes"}

    def _posterior_predictive(self, ppc, obs, ys) -> dict[str, Any]:
        yr = np.asarray(ppc["posterior_predictive"]["y"].values).reshape(-1, len(obs))
        bins = pd.cut(obs["DAY"], [0, 7, 14, 21, 28, 1e9], labels=["<=7", "8-14", "15-21", "22-28", ">28"])
        out = {}
        for b in bins.cat.categories:
            m = (bins == b).to_numpy()
            if m.sum() < 5:
                continue
            obs_mean, obs_sd = ys[m].mean(), ys[m].std()
            rep_mean, rep_sd = yr[:, m].mean(1), yr[:, m].std(1)
            out[str(b)] = {"p_mean": float((rep_mean >= obs_mean).mean()), "p_sd": float((rep_sd >= obs_sd).mean())}
        flagged = [k for k, v in out.items() if min(v["p_mean"], 1 - v["p_mean"], v["p_sd"], 1 - v["p_sd"]) < 0.025]
        return {"by_time_bin": out, "extreme_bins": flagged,
                "note": "one-step conditional replicates (likelihood factorization); full-trajectory "
                        "new-study checks are in the held-out evaluation"}

    def _sensitivity_table(self, results, validation: ModelData) -> list[dict[str, Any]]:
        rows = []
        a = validation.animals.reset_index(drop=True)
        for k, v in results.items():
            saved = self.draws
            self.draws = v["post"]
            pred = self.predict_mean(a, np.array([28.0])) - a["BASELINE_LOGW"].to_numpy()[:, None]
            self.draws = saved
            ps = v["diagnostics"]["posterior_summary"]
            rows.append({"variant": k, "mean_pred_change_day28_validation": float(pred.mean()),
                         "sd_s_logw": ps["sd_s_scaled_logw"], "sd_a_logw": ps["sd_a_scaled_logw"],
                         "sigma_p_logw": ps["sigma_p_scaled_logw"], "ell": ps["ell"],
                         "reliable": v["diagnostics"]["reliable"], "rhat_max": v["diagnostics"]["rhat_max"]})
        return rows

    # ---------------------------------------------------------------- prediction
    @property
    def n_draws(self) -> int:
        return self.draws["beta"].shape[0]

    def predict_mean(self, context: pd.DataFrame, times: np.ndarray, member: int | None = None) -> np.ndarray:
        times = np.asarray(times, float)
        F = context_features(self.spec, context, times, self.compound_ref)
        X = self.design.matrix(F)
        beta = self.draws["beta"].mean(0) if member is None else self.draws["beta"][member]
        y = (X @ beta).reshape(len(context), len(times)) * self.s_y
        return context["BASELINE_LOGW"].to_numpy(float)[:, None] + y

    def sample_study(self, context: pd.DataFrame, regimens: pd.DataFrame | None, times: np.ndarray, n_draws: int,
                     seed: int) -> dict[str, np.ndarray]:
        ctx = merge_context(context, regimens) if regimens is not None else context
        times = np.asarray(times, float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.n_draws, size=n_draws)  # one posterior draw per simulated study
        F = context_features(self.spec, ctx, times, self.compound_ref)
        X = self.design.matrix(F)
        n, T = len(ctx), len(times)
        Zb = np.stack([np.ones(T), times / 28.0], 1)
        d = np.abs(times[:, None] - times[None, :])
        zb = ctx["BASELINE_LOGW"].to_numpy(float)[:, None]
        z = np.empty((n_draws, n, T))
        mean = np.empty((n_draws, n, T))
        for k, j in enumerate(idx):
            mu = (X @ self.draws["beta"][j]).reshape(n, T)
            u = rng.standard_normal(2) * self.draws["sd_s"][j]  # NEW study effect
            b = rng.standard_normal((n, 2)) * self.draws["sd_a"][j]  # NEW animal effects
            K = self.draws["sigma_p"][j] ** 2 * np.exp(-d / self.draws["ell"][j]) + 1e-10 * np.eye(T)
            e = rng.standard_normal((n, T)) @ np.linalg.cholesky(K).T
            dev = (Zb @ u)[None] + b @ Zb.T + e
            mean[k] = zb + mu * self.s_y
            z[k] = mean[k] + dev * self.s_y
        return {"z": z, "mean": mean, "members": idx}

    def predict_marginals(self, context: pd.DataFrame, times: np.ndarray,
                          history: pd.DataFrame | None = None) -> pd.DataFrame:
        if history is not None and len(history):
            raise NotImplementedError("early-history conditioning is not implemented for the Bayesian model")
        s = self.sample_study(context, None, times, 400, seed=0)["z"]
        rows = []
        for i, u in enumerate(context["USUBJID"]):
            for j, t in enumerate(times):
                rows.append((u, t, s[:, i, j].mean(), s[:, i, j].var()))
        return pd.DataFrame(rows, columns=["USUBJID", "DAY", "MEAN", "VAR"])

    # ---------------------------------------------------------------- io
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "posterior_draws.npz", **self.draws)
        self.compound_ref.to_csv(path / "compound_reference.csv", index=False)
        (path / "model.json").write_text(json.dumps({
            "name": self.name, "spec": self.spec.to_dict(), "design": self.design.to_dict(), "s_y": self.s_y,
            "priors": self.priors, "history": {k: v for k, v in self.history.items() if k != "main"}},
            indent=1, default=float), encoding="utf-8")
        (path / "diagnostics.json").write_text(json.dumps(self.diagnostics, indent=1, default=float), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BayesianModel:
        path = Path(path)
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        m = cls()
        m.spec = FeatureSpec.from_dict(d["spec"])
        m.design = MeanDesign.from_dict(d["design"])
        m.s_y = d["s_y"]
        m.priors = d["priors"]
        m.history = d["history"]
        m.diagnostics = json.loads((path / "diagnostics.json").read_text(encoding="utf-8"))
        z = np.load(path / "posterior_draws.npz")
        m.draws = {k: z[k] for k in z.files}
        m.compound_ref = pd.read_csv(path / "compound_reference.csv")
        return m

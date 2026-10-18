"""Shared joint-trajectory layer: a mean function (with parameter members) + HierarchicalCovariance.

Used by the spline model and the baseline-conditioned transformer so that the primary benchmark
comparison isolates the mean function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from send_bw.models.base import merge_context
from send_bw.models.covariance import HierarchicalCovariance


class GaussianLayerMixin:
    cov: HierarchicalCovariance
    n_members: int

    def predict_mean(self, context: pd.DataFrame, times: np.ndarray, member: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def sample_study(self, context: pd.DataFrame, regimens: pd.DataFrame | None, times: np.ndarray, n_draws: int,
                     seed: int) -> dict[str, np.ndarray]:
        ctx = merge_context(context, regimens) if regimens is not None else context
        times = np.asarray(times, float)
        rng = np.random.default_rng(seed)
        members = rng.integers(0, self.n_members, size=n_draws)  # one parameter member per study
        means = {m: self.predict_mean(ctx, times, member=int(m)) for m in np.unique(members)}
        dev = self.cov.sample(rng, times, len(ctx), n_draws)
        mean = np.stack([means[int(m)] for m in members])
        return {"z": mean + dev, "mean": mean, "members": members}

    def predict_marginals(self, context: pd.DataFrame, times: np.ndarray,
                          history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Gaussian marginals of Z at `times`; with history, joint within-study conditioning.

        Parameter uncertainty enters through the spread of member means (law of total variance).
        """
        times = np.asarray(times, float)
        allm = np.stack([self.predict_mean(context, times, member=m) for m in range(self.n_members)])
        mean = allm.mean(0)
        pvar = allm.var(0)
        var = np.broadcast_to(np.diag(self.cov.marginal_cov(times)), mean.shape) + pvar
        if history is not None and len(history):
            mean, var = self._condition(context, times, history, mean, pvar)
        rows = []
        for i, u in enumerate(context["USUBJID"]):
            for j, t in enumerate(times):
                rows.append((u, t, mean[i, j], var[i, j]))
        return pd.DataFrame(rows, columns=["USUBJID", "DAY", "MEAN", "VAR"])

    def _condition(self, context, times, history, mean, pvar):
        mean = mean.copy()
        var = np.zeros_like(mean)
        h = history.set_index("USUBJID")
        for _sid, idx in context.groupby("STUDYID").groups.items():
            idx = list(idx)
            sub = context.loc[idx]
            t_obs, r_obs, t_new = [], [], []
            for i, u in zip(idx, sub["USUBJID"], strict=True):
                hh = h.loc[[u]] if u in h.index else pd.DataFrame(columns=["DAY", "LOGW"])
                to = hh["DAY"].to_numpy(float)
                if len(to):
                    mu_o = np.stack([self.predict_mean(context.loc[[i]], to, member=m)[0]
                                     for m in range(self.n_members)]).mean(0)
                    r_obs.append(hh["LOGW"].to_numpy(float) - mu_o)
                else:
                    r_obs.append(np.zeros(0))
                t_obs.append(to)
                t_new.append(times)
            cm, cc = self.cov.condition_study(t_obs, r_obs, t_new)
            pos = {i: k for k, i in enumerate(idx)}
            for i in idx:
                k = pos[i]
                mean[context.index.get_loc(i)] += cm[k]
                var[context.index.get_loc(i)] = np.diag(cc[k]) + pvar[context.index.get_loc(i)]
        return mean, var

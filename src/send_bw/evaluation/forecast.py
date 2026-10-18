"""Held-out new-study evaluation of joint trajectory predictions.

Primary task (pre-study): predictions use only baseline, biology and planned regimens; the
model samples complete studies with NEW study and animal effects.
Secondary task (early history): observations up to the window day condition the prediction of
later outcomes (Gaussian-layer models only). Metrics for the two tasks are kept separate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from send_bw.evaluation.calibration import crps_gaussian, crps_samples, energy_score, interval_hits, pit
from send_bw.models.base import ModelData, ResponseModel

GAP_BINS = [(0, 4), (4, 8), (8, 15), (15, 100)]


def lognormal_mean(mu, var):
    """Mean weight under a Gaussian log-weight marginal; exp(mu) alone is the median.

    Gram-scale metrics in this module use sampled exp(Z) instead, which does not rely on the
    Gaussian-marginal assumption (e.g. bootstrap-member mixtures)."""
    return np.exp(np.asarray(mu) + np.asarray(var) / 2.0)


def _study_draws(model: ResponseModel, data: ModelData, sid: str, n_draws: int, seed: int):
    a = data.animals[data.animals["STUDYID"] == sid].reset_index(drop=True)
    o = data.obs[data.obs["STUDYID"] == sid]
    times = np.array(sorted(o["DAY"].unique()), float)
    out = model.sample_study(a, None, times, n_draws, seed)
    Y = np.full((len(a), len(times)), np.nan)
    pos = {u: i for i, u in enumerate(a["USUBJID"])}
    tpos = {t: j for j, t in enumerate(times)}
    for r in o.itertuples(index=False):
        Y[pos[r.USUBJID], tpos[r.DAY]] = r.LOGW
    return a, times, out["z"], Y


def evaluate_prestudy(model: ResponseModel, data: ModelData, n_draws: int = 200, seed: int = 0,
                      contrast_day: float = 28.0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    per_obs, per_study, cells, contrasts, gaps, incr, extrema, es = [], [], [], [], [], [], [], []
    for k, sid in enumerate(data.studies):
        a, times, Zs, Y = _study_draws(model, data, sid, n_draws, seed + k)
        m = np.isfinite(Y)
        ii, jj = np.where(m)
        samp = Zs[:, ii, jj]
        y = Y[ii, jj]
        mean = samp.mean(0)
        h80, w80 = interval_hits(samp, y, 0.8)
        h95, w95 = interval_hits(samp, y, 0.95)
        crps = crps_samples(samp, y)
        df = pd.DataFrame({"STUDYID": sid, "SEX": a["SEX"].to_numpy()[ii], "ARM_KEY": a["ARM_KEY"].to_numpy()[ii],
                           "DAY": times[jj], "err_log": mean - y,
                           "err_g": np.exp(samp).mean(0) - np.exp(y),
                           "hit80": h80, "hit95": h95, "width95": w95, "crps": crps, "pit": pit(samp, y)})
        per_obs.append(df)
        per_study.append({"STUDYID": sid, "rmse_log": float(np.sqrt(np.mean((mean - y) ** 2))),
                          "mae_log": float(np.mean(np.abs(mean - y))), "cov80": float(h80.mean()),
                          "cov95": float(h95.mean()), "crps": float(crps.mean()), "n_obs": int(len(y)),
                          "n_animals": len(a)})
        for i in range(len(a)):
            mi = m[i]
            if mi.sum() >= 2:
                es.append(energy_score(Zs[:, i, mi], Y[i, mi], rng))
        # group-mean cells (arm x sex x day), using the same animals and observation mask
        for (arm, sex), idx in a.groupby(["ARM_KEY", "SEX"]).groups.items():
            idx = np.array(list(idx))
            for j, t in enumerate(times):
                mm = m[idx, j]
                if mm.sum() < 2:
                    continue
                sel = idx[mm]
                pred = Zs[:, sel, j].mean(1)
                obs_mean = Y[sel, j].mean()
                lo, hi = np.quantile(pred, [0.05, 0.95])
                vo = Y[sel, j].var(ddof=1)
                vp = Zs[:, sel, j].var(1, ddof=1).mean()
                cells.append({"STUDYID": sid, "ARM_KEY": arm, "SEX": sex, "DAY": t, "hit90": lo <= obs_mean <= hi,
                              "err": pred.mean() - obs_mean, "var_obs": vo, "var_pred": vp})
        # active-control contrast at contrast_day (main cohort, by sex pooled)
        if contrast_day in times:
            j = int(np.where(times == contrast_day)[0][0])
            ctrl = a.index[a["IS_CONTROL"].astype(bool) & m[:, j]].to_numpy()
            for arm, g in a[~a["IS_CONTROL"].astype(bool)].groupby("ARM_KEY"):
                act = g.index[m[g.index, j]].to_numpy()
                if len(act) < 2 or len(ctrl) < 2:
                    continue
                po = Zs[:, act, j].mean(1) - Zs[:, ctrl, j].mean(1)
                oo = Y[act, j].mean() - Y[ctrl, j].mean()
                lo, hi = np.quantile(po, [0.05, 0.95])
                contrasts.append({"STUDYID": sid, "ARM_KEY": arm, "obs": oo, "pred_mean": po.mean(),
                                  "hit90": lo <= oo <= hi, "width90": hi - lo})
        # dependence: residual correlation by gap (residual = deviation from arm-sex-day cell mean)
        cell_mean_obs = np.full_like(Y, np.nan)
        cell_mean_pred = np.zeros_like(Zs)
        for _, idx in a.groupby(["ARM_KEY", "SEX"]).groups.items():
            idx = np.array(list(idx))
            cell_mean_obs[idx] = np.nanmean(Y[idx], axis=0, keepdims=True)
            cell_mean_pred[:, idx] = Zs[:, idx].mean(1, keepdims=True)
        Ro = Y - cell_mean_obs
        Rp = Zs - cell_mean_pred
        for j1 in range(len(times)):
            for j2 in range(j1 + 1, len(times)):
                both = m[:, j1] & m[:, j2]
                if both.sum() < 6:
                    continue
                co = np.corrcoef(Ro[both, j1], Ro[both, j2])[0, 1]
                cp = np.mean([np.corrcoef(Rp[s, both, j1], Rp[s, both, j2])[0, 1] for s in range(0, n_draws, 10)])
                gaps.append({"gap": times[j2] - times[j1], "corr_obs": co, "corr_pred": cp})
        # increments between consecutive observed visits and per-animal extrema
        for i in range(len(a)):
            mi = np.where(m[i])[0]
            if len(mi) >= 2:
                do = np.diff(Y[i, mi])
                dp = np.diff(Zs[:, i, mi], axis=1)
                incr.append({"sd_obs": do.std(), "sd_pred": dp.std(1).mean()})
                mx_o = Y[i, mi].max()
                mx_p = Zs[:, i, mi].max(1)
                lo, hi = np.quantile(mx_p, [0.05, 0.95])
                extrema.append(lo <= mx_o <= hi)
    obs_df = pd.concat(per_obs, ignore_index=True)
    st = pd.DataFrame(per_study)
    cells = pd.DataFrame(cells)
    contrasts = pd.DataFrame(contrasts)
    gaps = pd.DataFrame(gaps)
    incr = pd.DataFrame(incr)
    gap_rows = []
    for lo, hi in GAP_BINS:
        g = gaps[(gaps["gap"] > lo) & (gaps["gap"] <= hi)] if len(gaps) else gaps
        if len(g):
            gap_rows.append({"gap_bin": f"({lo},{hi}]", "corr_obs": float(g["corr_obs"].mean()),
                             "corr_pred": float(g["corr_pred"].mean()), "n_pairs": len(g)})
    obs_df["horizon"] = pd.cut(obs_df["DAY"], [0, 7, 14, 28, 1e9], labels=["1-7", "8-14", "15-28", ">28"])
    by = (obs_df.groupby(["SEX", "horizon"], observed=True)
          .agg(rmse_log=("err_log", lambda e: float(np.sqrt(np.mean(e**2)))), cov95=("hit95", "mean"),
               crps=("crps", "mean"), n=("err_log", "size")).reset_index())
    pit_hist = np.histogram(obs_df["pit"], bins=10, range=(0, 1))[0].tolist()
    summary = {
        "task": "pre_study_new_study", "model": model.name,
        "effective_sample_sizes": {"n_studies": int(len(st)), "n_animals": int(st["n_animals"].sum()),
                                   "n_obs": int(len(obs_df))},
        "observation_weighted": {"rmse_log": float(np.sqrt(np.mean(obs_df["err_log"] ** 2))),
                                 "mae_log": float(np.mean(np.abs(obs_df["err_log"]))),
                                 "rmse_g": float(np.sqrt(np.mean(obs_df["err_g"] ** 2))),
                                 "mae_g": float(np.mean(np.abs(obs_df["err_g"]))),
                                 "cov80": float(obs_df["hit80"].mean()), "cov95": float(obs_df["hit95"].mean()),
                                 "width95_log": float(obs_df["width95"].mean()), "crps_log": float(obs_df["crps"].mean())},
        "study_balanced": {k: float(st[k].mean()) for k in ["rmse_log", "mae_log", "cov80", "cov95", "crps"]},
        "energy_score_mean": float(np.mean(es)) if es else None,
        "group_mean_cov90": float(cells["hit90"].mean()) if len(cells) else None,
        "between_animal_var_ratio_pred_over_obs": float(cells["var_pred"].mean() / cells["var_obs"].mean())
        if len(cells) else None,
        "contrast_day": contrast_day,
        "contrast_cov90": float(contrasts["hit90"].mean()) if len(contrasts) else None,
        "contrast_bias": float((contrasts["pred_mean"] - contrasts["obs"]).mean()) if len(contrasts) else None,
        "contrast_width90": float(contrasts["width90"].mean()) if len(contrasts) else None,
        "n_contrasts": int(len(contrasts)),
        "correlation_by_gap": gap_rows,
        "increment_sd_ratio_pred_over_obs": float(incr["sd_pred"].mean() / incr["sd_obs"].mean()) if len(incr) else None,
        "extrema_max_cov90": float(np.mean(extrema)) if extrema else None,
        "pit_histogram_10bins": pit_hist,
    }
    return {"summary": summary, "by_sex_horizon": by, "per_study": st}


def evaluate_early_history(model: ResponseModel, data: ModelData, window_day: float) -> dict[str, Any]:
    """Secondary task: condition on observations with DAY <= window_day, score later observations."""
    if not hasattr(model, "_condition"):
        return {"task": "early_history", "model": model.name, "status": "not_run",
                "reason": "early-history conditioning not implemented for this model"}
    rows = []
    for sid in data.studies:
        a = data.animals[data.animals["STUDYID"] == sid].reset_index(drop=True)
        o = data.obs[data.obs["STUDYID"] == sid]
        hist = o[o["DAY"] <= window_day][["USUBJID", "DAY", "LOGW"]]
        fut = o[o["DAY"] > window_day]
        if fut.empty:
            continue
        times = np.array(sorted(fut["DAY"].unique()), float)
        cond = model.predict_marginals(a, times, history=hist)
        pre = model.predict_marginals(a, times, history=None)
        for name, pm_ in [("conditioned", cond), ("pre_study", pre)]:
            mm = fut.merge(pm_, on=["USUBJID", "DAY"], how="left", validate="one_to_one")
            sd = np.sqrt(mm["VAR"].to_numpy())
            z = (mm["LOGW"] - mm["MEAN"]) / sd
            rows.append(pd.DataFrame({"which": name, "err": mm["MEAN"] - mm["LOGW"], "hit90": np.abs(z) <= 1.645,
                                      "crps": crps_gaussian(mm["MEAN"].to_numpy(), sd, mm["LOGW"].to_numpy())}))
    df = pd.concat(rows, ignore_index=True)
    res = {}
    for w, g in df.groupby("which"):
        res[w] = {"rmse_log": float(np.sqrt(np.mean(g["err"] ** 2))), "cov90": float(g["hit90"].mean()),
                  "crps_log": float(g["crps"].mean()), "n_obs": int(len(g))}
    return {"task": "early_history", "model": model.name, "status": "ok", "window_day": window_day,
            "note": "Gaussian marginals; secondary task only. Success here does not establish pre-study skill.",
            **res}

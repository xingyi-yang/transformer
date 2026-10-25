"""Early-history scoring for quantile and Gaussian predictive representations.

Do not treat marginal quantiles as joint trajectory simulators. CRPS is reported
only when the representation justifies it (Gaussian, or actual samples).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from send_bw.evaluation.calibration import crps_gaussian
from send_bw.models.base import ModelData, study_balanced_weights

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def pinball(y: np.ndarray, qhat: np.ndarray, level: float) -> np.ndarray:
    """Pinball / quantile loss at one level."""
    d = y - qhat
    return np.where(d >= 0, level * d, (level - 1.0) * d)


def _study_mean(values: np.ndarray, studies: np.ndarray) -> dict[str, float]:
    w = study_balanced_weights(studies)
    pooled = float(np.mean(values))
    by = pd.DataFrame({"v": values, "s": studies}).groupby("s")["v"].mean()
    return {
        "pooled": pooled,
        "study_balanced": float(by.mean()) if len(by) else float("nan"),
        "n_obs": int(len(values)),
        "n_studies": int(by.shape[0]),
        "weight_mean": float(w.mean()),
    }


def _study_rmse(err: np.ndarray, studies: np.ndarray) -> dict[str, Any]:
    """True RMSE: per-study sqrt(mean e^2), then unweighted mean of studies.

    Pooled is sqrt(mean e^2) over observations. This is not mean(|e|).
    """
    e2 = np.asarray(err, float) ** 2
    pooled = float(np.sqrt(np.mean(e2))) if len(e2) else float("nan")
    by = pd.DataFrame({"e2": e2, "s": studies}).groupby("s")["e2"].mean()
    study = float(np.sqrt(by).mean()) if len(by) else float("nan")
    return {
        "pooled": pooled,
        "study_balanced": study,
        "n_obs": int(len(err)),
        "n_studies": int(by.shape[0]),
        "definition": "per-study sqrt(mean e^2), then mean of studies; pooled=sqrt(mean e^2)",
    }


def align_prediction_table(obs: pd.DataFrame, pred: pd.DataFrame,
                           keys: tuple[str, ...] = ("USUBJID", "DAY")) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inner-join on USUBJID/DAY. Observation STUDYID wins. Report cardinality."""
    obs_k = obs.copy()
    pred_k = pred.copy()
    for k in keys:
        if k not in obs_k.columns or k not in pred_k.columns:
            raise ValueError(f"align keys missing {k}")
        if k == "DAY":
            obs_k[k] = pd.to_numeric(obs_k[k], errors="coerce")
            pred_k[k] = pd.to_numeric(pred_k[k], errors="coerce")
        else:
            obs_k[k] = obs_k[k].astype(str)
            pred_k[k] = pred_k[k].astype(str)
    key_list = list(keys)
    dup_obs = int(obs_k.duplicated(key_list).sum())
    dup_pred = int(pred_k.duplicated(key_list).sum())
    if "STUDYID" in pred_k.columns and "STUDYID" in obs_k.columns:
        pred_k = pred_k.drop(columns=["STUDYID"])
    mm = obs_k.merge(pred_k, on=key_list, how="inner")
    obs_keys = obs_k[key_list].drop_duplicates()
    pred_keys = pred_k[key_list].drop_duplicates()
    both = obs_keys.merge(pred_keys, on=key_list, how="inner")
    card = {
        "n_obs": int(len(obs_k)),
        "n_pred": int(len(pred_k)),
        "n_matched": int(len(mm)),
        "n_obs_key_unique": int(len(obs_keys)),
        "n_pred_key_unique": int(len(pred_keys)),
        "n_keys_in_both": int(len(both)),
        "n_obs_only": int(len(obs_keys) - len(both)),
        "n_pred_only": int(len(pred_keys) - len(both)),
        "duplicate_obs_keys": dup_obs,
        "duplicate_pred_keys": dup_pred,
        "inflated": bool(len(mm) != len(both)),
        "grouping_column": "STUDYID" if "STUDYID" in mm.columns else "USUBJID",
        "one_to_one": bool(dup_obs == 0 and dup_pred == 0 and len(mm) == len(both)),
    }
    return mm, card


def score_quantile_table(obs: pd.DataFrame, pred: pd.DataFrame, *, quantiles: list[float],
                         representation: str) -> dict[str, Any]:
    """Score aligned USUBJID/DAY rows. ``pred`` has columns q{level} or MEAN/VAR.

    ``mae_log`` is study-balanced mean |point − LOGW|.
    ``rmse_log`` is true RMSE (per-study sqrt(mean e^2), then mean of studies).
    The historical name ``rmse_log`` on official_seq/seq_diag tables was MAE.
    """
    mm, card = align_prediction_table(obs, pred)
    if mm.empty:
        return {"status": "not_run", "reason": "no overlapping USUBJID/DAY after encoder filter",
                "n_obs": 0, "merge": card}
    y = mm["LOGW"].to_numpy(float)
    studies = mm["STUDYID"].to_numpy() if "STUDYID" in mm.columns else mm["USUBJID"].to_numpy()
    animals = mm["USUBJID"].to_numpy() if "USUBJID" in mm.columns else studies
    levels = [float(q) for q in quantiles]
    qcols = {}
    for q in levels:
        name = f"q{q}"
        if name in mm.columns:
            qcols[q] = mm[name].to_numpy(float)
        elif "MEAN" in mm.columns and "VAR" in mm.columns:
            sd = np.sqrt(np.maximum(mm["VAR"].to_numpy(float), 1e-12))
            qcols[q] = mm["MEAN"].to_numpy(float) + sd * float(stats.norm.ppf(q))
        else:
            raise ValueError(f"prediction table missing quantile {q} and MEAN/VAR")
    if representation == "gaussian" and "MEAN" in mm.columns:
        point = mm["MEAN"].to_numpy(float)
        point_name = "MEAN"
        point_kind = "gaussian_mean"
    else:
        point = qcols[0.5] if 0.5 in qcols else qcols[min(levels, key=lambda x: abs(x - 0.5))]
        point_name = "q0.5"
        point_kind = "median"
    err = point - y
    pinballs = {q: pinball(y, qcols[q], q) for q in levels}
    mean_pinball = np.mean(np.stack(list(pinballs.values()), axis=0), axis=0)
    mm = mm.reset_index(drop=True)
    mm["_point"] = point
    mm["_y"] = y
    mm["_pin"] = mean_pinball
    mm["_HORIZON_STEP"] = mm.groupby(mm["USUBJID"].astype(str), sort=False).cumcount() + 1
    out: dict[str, Any] = {
        "status": "ok",
        "representation": representation,
        "quantiles": levels,
        "point_definition": {
            "point": point_kind,
            "column": point_name,
            "mae_log": "study-balanced mean of |point - LOGW|",
            "rmse_log": ("true RMSE: per-study sqrt(mean (point-LOGW)^2), then mean of studies. "
                         "Not |error|. Chronos-2 official mean is the median."),
            "legacy_rmse_log": "official_seq/seq_diag rmse_log was |error|, often grouped by USUBJID",
        },
        "mae_log": _study_mean(np.abs(err), studies),
        "rmse_log": _study_rmse(err, studies),
        "legacy_animal_balanced_mae": _study_mean(np.abs(err), animals),
        "quantile_loss": {str(q): _study_mean(pinballs[q], studies) for q in levels},
        "mean_quantile_loss": _study_mean(mean_pinball, studies),
        "n_scored": int(len(mm)),
        "n_obs_available": int(len(obs)),
        "n_pred_rows": int(len(pred)),
        "merge": card,
        "grouping": "STUDYID",
        "horizon_units": {
            "calendar_day": "DAY of the scored visit",
            "visit_step": "1-based rank among observed future visits for that animal",
            "fabricated_calendar_steps": False,
        },
        "trajectory_simulator": False,
        "task": "early_history",
    }
    if representation.startswith("chronos"):
        out["chronos_label"] = "history_only_univariate"
        out["covariates_used"] = False
    out["per_study"] = _breakdown(mm, qcols, key="STUDYID")
    out["per_horizon_day"] = _breakdown(mm, qcols, key="DAY")
    out["per_horizon_step"] = _breakdown(mm, qcols, key="_HORIZON_STEP")
    out["per_horizon"] = out["per_horizon_day"]
    if 0.1 in qcols and 0.9 in qcols:
        lo, hi = qcols[0.1], qcols[0.9]
        out["coverage80"] = _study_mean(((y >= lo) & (y <= hi)).astype(float), studies)
        out["interval_width80"] = _study_mean(hi - lo, studies)
    if 0.25 in qcols and 0.75 in qcols:
        lo, hi = qcols[0.25], qcols[0.75]
        out["coverage50"] = _study_mean(((y >= lo) & (y <= hi)).astype(float), studies)
        out["interval_width50"] = _study_mean(hi - lo, studies)
    if representation == "gaussian" and "MEAN" in mm.columns and "VAR" in mm.columns:
        sd = np.sqrt(np.maximum(mm["VAR"].to_numpy(float), 1e-12))
        out["crps_log"] = _study_mean(crps_gaussian(mm["MEAN"].to_numpy(float), sd, y), studies)
        out["crps_justified"] = True
        out["crps_note"] = "Gaussian CRPS from the hierarchical residual layer."
    else:
        out["crps_log"] = None
        out["crps_justified"] = False
        out["crps_note"] = (
            "CRPS not reported: predictive representation is a finite quantile set, "
            "not samples and not a fitted Gaussian. Marginal quantiles are not a "
            "complete trajectory simulator."
        )
    return out


def _breakdown(mm: pd.DataFrame, qcols: dict[float, np.ndarray],
               key: str) -> list[dict[str, Any]]:
    rows = []
    work = mm.reset_index(drop=True)
    if 0.1 in qcols and 0.9 in qcols:
        work["_lo80"] = qcols[0.1]
        work["_hi80"] = qcols[0.9]
    col = key if key in work.columns else key
    for label, g in work.groupby(col, sort=True):
        e = g["_point"].to_numpy(float) - g["_y"].to_numpy(float)
        rec: dict[str, Any] = {
            ("horizon_step" if key == "_HORIZON_STEP" else key): (
                int(label) if key == "_HORIZON_STEP" else (float(label) if key == "DAY" else str(label))),
            "n_obs": int(len(g)),
            "mae_log": float(np.mean(np.abs(e))),
            "rmse_log": float(np.sqrt(np.mean(e ** 2))),
            "mean_quantile_loss": float(g["_pin"].mean()),
        }
        if "_lo80" in g.columns:
            rec["coverage80"] = float(((g["_y"] >= g["_lo80"]) & (g["_y"] <= g["_hi80"])).mean())
            rec["interval_width80"] = float((g["_hi80"] - g["_lo80"]).mean())
        rows.append(rec)
    return rows


def series_eligibility(obs: pd.DataFrame, *, min_past: int, prediction_length: int) -> dict[str, Any]:
    """Count observed visits only. Do not insert unobserved calendar days."""
    n = obs.groupby(obs["USUBJID"].astype(str), sort=False).size()
    need = int(min_past) + int(prediction_length)
    hist = {int(k): int(v) for k, v in n.value_counts().sort_index().items()}
    return {
        "min_past": int(min_past),
        "prediction_length": int(prediction_length),
        "required_observed_visits": need,
        "n_series": int(len(n)),
        "n_eligible": int((n >= need).sum()),
        "n_ineligible": int((n < need).sum()),
        "length_histogram": hist,
        "fabricated_history": False,
        "note": ("Chronos-2 fit keeps series with length >= min_past + prediction_length. "
                 "Ineligible series are reported, not padded."),
    }


def last_value_predictions(hist: pd.DataFrame, fut: pd.DataFrame,
                           quantiles: list[float] | None = None) -> pd.DataFrame:
    """Last observed encoder LOGW, repeated at every future DAY. Degenerate quantiles."""
    levels = [float(q) for q in (quantiles or QUANTILES)]
    last = (hist.sort_values(["USUBJID", "DAY"])
            .groupby("USUBJID", as_index=False)
            .tail(1)[["USUBJID", "LOGW"]]
            .rename(columns={"LOGW": "last"}))
    mm = fut.merge(last, on="USUBJID", how="inner")
    if mm.empty:
        return pd.DataFrame(columns=["USUBJID", "STUDYID", "DAY", *[f"q{q}" for q in levels]])
    out = mm[["USUBJID", "DAY"]].copy()
    if "STUDYID" in mm.columns:
        out["STUDYID"] = mm["STUDYID"]
    val = mm["last"].to_numpy(float)
    for q in levels:
        out[f"q{q}"] = val
    out["point"] = val
    out["point_definition"] = "last_observed_encoder_LOGW"
    return out


def encoder_time_report(hist: pd.DataFrame, fut: pd.DataFrame) -> dict[str, Any]:
    """Visit-rank vs calendar-time diagnostic. Chronos receives unit steps, not DAY."""
    recs = []
    for u, hg in hist.groupby("USUBJID", sort=False):
        hg = hg.sort_values("DAY")
        fg = fut.loc[fut["USUBJID"] == u].sort_values("DAY")
        days = hg["DAY"].to_numpy(float)
        gaps = np.diff(days) if len(days) > 1 else np.array([])
        recs.append({
            "USUBJID": str(u),
            "STUDYID": str(hg["STUDYID"].iloc[0]) if "STUDYID" in hg.columns else "",
            "n_encoder": int(len(hg)),
            "n_future": int(len(fg)),
            "encoder_day_min": float(days.min()) if len(days) else None,
            "encoder_day_max": float(days.max()) if len(days) else None,
            "future_days": [float(x) for x in fg["DAY"].to_numpy(float)] if len(fg) else [],
            "median_encoder_gap": float(np.median(gaps)) if len(gaps) else None,
            "irregular_encoder": bool(len(gaps) > 1 and np.ptp(gaps) > 1e-9),
        })
    gaps = [r["median_encoder_gap"] for r in recs if r["median_encoder_gap"] is not None]
    return {
        "n_animals": len(recs),
        "n_irregular_encoder": int(sum(1 for r in recs if r["irregular_encoder"])),
        "median_encoder_gap": float(np.median(gaps)) if gaps else None,
        "chronos_time_index": "visit_rank_unit_steps",
        "calendar_time_passed_to_chronos": False,
        "note": ("Chronos-2 is given the observed LOGW sequence in visit order. "
                 "DAY values are used only to align scored rows, not as the time index."),
        "rows": recs,
    }


def split_encoder_future(data: ModelData, window_day: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed encoder (DAY <= window) and future (DAY > window). No filled rows."""
    o = data.obs[["USUBJID", "STUDYID", "DAY", "LOGW"]].copy()
    hist = o[o["DAY"] <= float(window_day)].copy()
    fut = o[o["DAY"] > float(window_day)].copy()
    have = set(hist["USUBJID"].astype(str))
    fut = fut[fut["USUBJID"].astype(str).isin(have)].reset_index(drop=True)
    hist = hist[hist["USUBJID"].astype(str).isin(set(fut["USUBJID"].astype(str)))].reset_index(drop=True)
    return hist, fut


def paired_transfer(target_only: dict[str, Any], pooled: dict[str, Any],
                    metrics: tuple[str, ...] = ("mae_log", "rmse_log", "mean_quantile_loss")) -> dict[str, Any]:
    """Δ = S_pooled − S_target_only. For MAE / RMSE / pinball, Δ>0 is harmful."""
    def _val(block: dict[str, Any], key: str) -> float | None:
        v = block.get(key)
        if isinstance(v, dict):
            v = v.get("study_balanced")
        return None if v is None else float(v)

    rows = []
    for m in metrics:
        a = _val(target_only, m)
        b = _val(pooled, m)
        if a is None or b is None:
            continue
        rows.append({
            "metric": m, "target_only": a, "pooled": b, "delta_pooled_minus_target": b - a,
            "harmful_if_positive": True,
        })
    return {"rows": rows, "note": "Paired on the same target-study holdout and encoder cutoff. Δ>0 harmful."}


def filter_species(data: ModelData, species: str) -> ModelData:
    keep = data.animals["SPECIES"].astype(str).str.upper() == str(species).upper()
    ids = set(data.animals.loc[keep, "USUBJID"].astype(str))
    return ModelData(
        data.animals[data.animals["USUBJID"].astype(str).isin(ids)].reset_index(drop=True),
        data.obs[data.obs["USUBJID"].astype(str).isin(ids)].reset_index(drop=True),
        data.compound_ref, data.spec,
    )

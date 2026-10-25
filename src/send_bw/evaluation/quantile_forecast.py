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


def score_quantile_table(obs: pd.DataFrame, pred: pd.DataFrame, *, quantiles: list[float],
                         representation: str) -> dict[str, Any]:
    """Score aligned USUBJID/DAY rows. ``pred`` has columns q{level} or MEAN/VAR."""
    mm = obs.merge(pred, on=["USUBJID", "DAY"], how="inner")
    if mm.empty:
        return {"status": "not_run", "reason": "no overlapping USUBJID/DAY after encoder filter",
                "n_obs": 0}
    y = mm["LOGW"].to_numpy(float)
    studies = mm["STUDYID"].to_numpy() if "STUDYID" in mm.columns else mm["USUBJID"].to_numpy()
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
    median = qcols[0.5] if 0.5 in qcols else qcols[min(levels, key=lambda x: abs(x - 0.5))]
    rmse = np.sqrt((median - y) ** 2)
    pinballs = {q: pinball(y, qcols[q], q) for q in levels}
    mean_pinball = np.mean(np.stack(list(pinballs.values()), axis=0), axis=0)
    out: dict[str, Any] = {
        "status": "ok",
        "representation": representation,
        "quantiles": levels,
        "rmse_log": _study_mean(rmse, studies),
        "quantile_loss": {str(q): _study_mean(pinballs[q], studies) for q in levels},
        "mean_quantile_loss": _study_mean(mean_pinball, studies),
        "n_scored": int(len(mm)),
        "n_obs_available": int(len(obs)),
        "n_pred_rows": int(len(pred)),
        "trajectory_simulator": False,
    }
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
                    metrics: tuple[str, ...] = ("rmse_log", "mean_quantile_loss")) -> dict[str, Any]:
    """Δ = S_pooled − S_target_only. For RMSE / pinball, Δ>0 is harmful."""
    rows = []
    for m in metrics:
        a = (target_only.get(m) or {}).get("study_balanced")
        b = (pooled.get(m) or {}).get("study_balanced")
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

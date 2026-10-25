"""80% interval scoring and validation-only conformal calibration.

Winkler interval score at alpha=0.2 is a proper score for the (q0.1, q0.9)
interval. Conformalized quantile regression (CQR) adds a single margin to both
ends. The margin is fit on calibration/validation pairs only.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from send_bw.evaluation.quantile_forecast import align_prediction_table
from send_bw.models.base import study_balanced_weights

ALPHA_80 = 0.2


def interval_score80(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     alpha: float = ALPHA_80) -> np.ndarray:
    """Winkler interval score. Width plus (2/alpha) times the miss distance."""
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return width + (2.0 / float(alpha)) * (below + above)


def conformal_nonconformity(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """CQR residual: how far y sits outside [lo, hi]. Negative means inside."""
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return np.maximum(lo - y, y - hi)


def conformal_margin(scores: np.ndarray, alpha: float = ALPHA_80) -> float:
    """Finite-sample (1-alpha) quantile of CQR residuals. Validation only."""
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite calibration scores")
    n = int(s.size)
    q = float(np.ceil((n + 1) * (1.0 - float(alpha))) / n)
    q = min(q, 1.0)
    return float(np.quantile(s, q, method="higher"))


def fit_cqr_margin(fut: pd.DataFrame, pred: pd.DataFrame, *,
                   lo_col: str = "q0.1", hi_col: str = "q0.9",
                   alpha: float = ALPHA_80) -> dict[str, Any]:
    """Estimate a CQR margin from calibration pairs. Does not read a test set."""
    mm, card = align_prediction_table(fut, pred)
    if mm.empty or lo_col not in mm.columns or hi_col not in mm.columns:
        return {"status": "not_run", "reason": "no overlapping calibration pairs", "merge": card}
    y = mm["LOGW"].to_numpy(float)
    lo = mm[lo_col].to_numpy(float)
    hi = mm[hi_col].to_numpy(float)
    scores = conformal_nonconformity(y, lo, hi)
    margin = conformal_margin(scores, alpha=alpha)
    covered = (y >= lo) & (y <= hi)
    return {
        "status": "ok",
        "method": "cqr_additive",
        "alpha": float(alpha),
        "margin": margin,
        "n_cal": int(len(mm)),
        "n_studies": int(mm["STUDYID"].nunique()) if "STUDYID" in mm.columns else None,
        "cal_coverage80": float(np.mean(covered)),
        "cal_mean_residual": float(np.mean(scores)),
        "quantile_rule": "ceil((n+1)(1-alpha))/n, method=higher",
        "test_outcomes_used": False,
        "merge": card,
        "note": ("Margin is fit on calibration/validation pairs only. "
                 "q0.5 is not moved. Inner quantiles are not rearranged."),
    }


def apply_cqr_margin(pred: pd.DataFrame, margin: float, *,
                     lo_col: str = "q0.1", hi_col: str = "q0.9") -> pd.DataFrame:
    """Widen (or shrink) the 80% interval by a constant. Point forecast unchanged."""
    out = pred.copy()
    if lo_col not in out.columns or hi_col not in out.columns:
        raise ValueError("prediction table missing 80% quantile columns")
    out[lo_col] = out[lo_col].to_numpy(float) - float(margin)
    out[hi_col] = out[hi_col].to_numpy(float) + float(margin)
    out["cqr_margin"] = float(margin)
    out["interval_calibrated"] = True
    return out


def coverage_slices(fut: pd.DataFrame, pred: pd.DataFrame, *,
                    lo_col: str = "q0.1", hi_col: str = "q0.9") -> dict[str, Any]:
    """Coverage, width, and interval score by study and forecast lead."""
    mm, card = align_prediction_table(fut, pred)
    if mm.empty or lo_col not in mm.columns or hi_col not in mm.columns:
        return {"status": "not_run", "reason": "no overlapping pairs", "merge": card}
    y = mm["LOGW"].to_numpy(float)
    lo = mm[lo_col].to_numpy(float)
    hi = mm[hi_col].to_numpy(float)
    hit = (y >= lo) & (y <= hi)
    width = hi - lo
    ins = interval_score80(y, lo, hi)
    below = y < lo
    above = y > hi
    work = mm.assign(_hit=hit, _width=width, _is=ins, _below=below, _above=above)
    if "LEAD_DAYS" in work.columns:
        work["_lead"] = np.round(work["LEAD_DAYS"].to_numpy(float), 6)

    def _pack(g: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_obs": int(len(g)),
            "coverage80": float(g["_hit"].mean()),
            "interval_width80": float(g["_width"].mean()),
            "interval_score80": float(g["_is"].mean()),
            "frac_below": float(g["_below"].mean()),
            "frac_above": float(g["_above"].mean()),
        }

    by_study = []
    if "STUDYID" in work.columns:
        for sid, g in work.groupby(work["STUDYID"].astype(str), sort=True):
            by_study.append({"STUDYID": str(sid), **_pack(g)})
    by_lead = []
    if "_lead" in work.columns:
        for lead, g in work.groupby("_lead", sort=True):
            by_lead.append({"lead_days": float(lead), **_pack(g)})
    studies = work["STUDYID"].to_numpy() if "STUDYID" in work.columns else work["USUBJID"].to_numpy()
    w = study_balanced_weights(studies)
    by = pd.DataFrame({"hit": hit.astype(float), "s": studies}).groupby("s")["hit"].mean()
    return {
        "status": "ok",
        "n_obs": int(len(work)),
        "n_studies": int(by.shape[0]),
        "coverage80_pooled": float(np.mean(hit)),
        "coverage80_study_balanced": float(by.mean()) if len(by) else None,
        "interval_width80_study_balanced": float(
            pd.DataFrame({"v": width, "s": studies}).groupby("s")["v"].mean().mean()),
        "interval_score80_study_balanced": float(
            pd.DataFrame({"v": ins, "s": studies}).groupby("s")["v"].mean().mean()),
        "frac_below": float(np.mean(below)),
        "frac_above": float(np.mean(above)),
        "weight_mean": float(w.mean()),
        "by_study": by_study,
        "by_lead_days": by_lead,
        "merge": card,
    }


def compact_interval_metrics(block: dict[str, Any]) -> dict[str, Any]:
    """Study-balanced MAE, pinball, interval score, coverage, width, crossings."""
    def _sb(key: str):
        v = block.get(key)
        if isinstance(v, dict):
            x = v.get("study_balanced")
            return None if x is None else float(x)
        return None if v is None else float(v)

    cr = block.get("crossing_rate")
    crossing = (cr or {}).get("pooled") if isinstance(cr, dict) else cr
    return {
        "mae_log": _sb("mae_log"),
        "mean_quantile_loss": _sb("mean_quantile_loss"),
        "interval_score80": _sb("interval_score80"),
        "coverage80": _sb("coverage80"),
        "interval_width80": _sb("interval_width80"),
        "crossing": None if crossing is None else float(crossing),
        "n_scored": block.get("n_scored"),
        "status": block.get("status"),
        "per_study": block.get("per_study"),
    }


def score_raw_and_calibrated(fut: pd.DataFrame, raw_pred: pd.DataFrame,
                             cal_pred: pd.DataFrame | None = None,
                             *, representation: str = "tft_quantiles") -> dict[str, Any]:
    """Score original quantiles and CQR endpoints as separate blocks.

    Calibrated pinball uses the shifted q0.1/q0.9. It is not the TFT training
    quantile loss. MAE (q0.5) must match the raw table.
    """
    from send_bw.tft_examples import score_aligned_quantiles

    raw_block = score_aligned_quantiles(fut, raw_pred, representation=representation)
    raw = compact_interval_metrics(raw_block)
    out: dict[str, Any] = {
        "raw": raw,
        "cqr": None,
        "pinball_note": cqr_pinball_note(),
        "test_outcomes_used": False,
    }
    if cal_pred is None:
        return out
    cal_block = score_aligned_quantiles(fut, cal_pred, representation=representation)
    cal = compact_interval_metrics(cal_block)
    cal["mean_quantile_loss_name"] = "pinball_on_cqr_shifted_endpoints"
    cal["q0_5_unchanged"] = True
    out["cqr"] = cal
    return out


def cqr_pinball_note() -> str:
    return (
        "Raw pinball is the mean QuantileLoss on the original head. "
        "CQR pinball uses the shifted 10% and 90% endpoints and is not "
        "the training objective. Report both; do not treat CQR pinball as "
        "an improvement of the TFT quantile fit."
    )


def attach_interval_score(block: dict[str, Any], fut: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    """Add Winkler 80% scores to an existing quantile score block."""
    mm, _card = align_prediction_table(fut, pred)
    if mm.empty or "q0.1" not in mm.columns or "q0.9" not in mm.columns:
        block["interval_score80"] = None
        return block
    y = mm["LOGW"].to_numpy(float)
    lo = mm["q0.1"].to_numpy(float)
    hi = mm["q0.9"].to_numpy(float)
    studies = mm["STUDYID"].to_numpy() if "STUDYID" in mm.columns else mm["USUBJID"].to_numpy()
    ins = interval_score80(y, lo, hi)
    by = pd.DataFrame({"v": ins, "s": studies}).groupby("s")["v"].mean()
    block["interval_score80"] = {
        "pooled": float(np.mean(ins)),
        "study_balanced": float(by.mean()) if len(by) else float("nan"),
        "n_obs": int(len(ins)),
        "n_studies": int(by.shape[0]),
    }
    block["interval_score80_definition"] = (
        "Winkler interval score at alpha=0.2 on LOGW: "
        "(q0.9-q0.1) + 10 * miss distance. Study-balanced mean of studies."
    )
    return block

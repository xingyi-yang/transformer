"""Forecast examples built only from observed DAY / LOGW rows.

No illustrative day arrays. No interpolation. No targets on unmeasured days.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from send_bw.evaluation.quantile_forecast import QUANTILES, align_prediction_table, score_quantile_table
from send_bw.models.base import ModelData, study_balanced_weights

EXCLUDE_NO_ENCODER = "no_observed_encoder_on_or_before_cutoff"
EXCLUDE_NO_FUTURE = "no_observed_target_after_cutoff"


def summarize_observation_schedules(obs: pd.DataFrame, animals: pd.DataFrame) -> dict[str, Any]:
    """Describe actual visit calendars. Does not invent days."""
    rows = []
    for sid, g in obs.groupby(obs["STUDYID"].astype(str), sort=True):
        days = [float(x) for x in sorted(g["DAY"].unique())]
        n_per = g.groupby(g["USUBJID"].astype(str)).size()
        gaps = []
        n_irreg = 0
        for _, ag in g.groupby("USUBJID"):
            gd = np.diff(np.sort(ag["DAY"].to_numpy(float)))
            if len(gd):
                gaps.extend(gd.tolist())
                if len(gd) > 1 and float(np.ptp(gd)) > 1e-9:
                    n_irreg += 1
        rows.append({
            "STUDYID": str(sid),
            "n_animals": int(g["USUBJID"].nunique()),
            "n_obs": int(len(g)),
            "unique_days": days,
            "median_visits_per_animal": float(n_per.median()) if len(n_per) else None,
            "min_visits": int(n_per.min()) if len(n_per) else 0,
            "max_visits": int(n_per.max()) if len(n_per) else 0,
            "n_animals_irregular_gaps": n_irreg,
            "median_gap_days": float(np.median(gaps)) if gaps else None,
        })
    species = {}
    if "SPECIES" in animals.columns:
        species = {str(k): int(v) for k, v in animals.groupby(animals["SPECIES"].astype(str))["USUBJID"].nunique().items()}
    return {
        "n_animals": int(animals["USUBJID"].nunique()) if len(animals) else 0,
        "n_obs": int(len(obs)),
        "n_studies": int(obs["STUDYID"].nunique()) if len(obs) else 0,
        "species_n_animals": species,
        "unique_days_all": [float(x) for x in sorted(obs["DAY"].unique())] if len(obs) else [],
        "day_min": float(obs["DAY"].min()) if len(obs) else None,
        "day_max": float(obs["DAY"].max()) if len(obs) else None,
        "weight_unit": "gram (WEIGHT_G); LOGW = ln(WEIGHT_G)",
        "by_study": rows,
        "fabricated_days": False,
        "interpolated": False,
    }


def animal_windows(obs: pd.DataFrame, usubjid: str, *, cutoff: float,
                   max_encoder_length: int, max_prediction_length: int) -> dict[str, Any]:
    """Independently derive encoder and target rows for one animal from the observed table."""
    g = obs.loc[obs["USUBJID"].astype(str) == str(usubjid)].sort_values("DAY")
    if g.empty:
        return {"USUBJID": str(usubjid), "eligible": False, "exclude_reason": "animal_absent_from_obs"}
    enc_all = g.loc[g["DAY"].to_numpy(float) <= float(cutoff)]
    fut_all = g.loc[g["DAY"].to_numpy(float) > float(cutoff)]
    if enc_all.empty:
        return {"USUBJID": str(usubjid), "eligible": False, "exclude_reason": EXCLUDE_NO_ENCODER,
                "STUDYID": str(g["STUDYID"].iloc[0]),
                "n_encoder_available": 0, "n_future_available": int(len(fut_all))}
    if fut_all.empty:
        return {"USUBJID": str(usubjid), "eligible": False, "exclude_reason": EXCLUDE_NO_FUTURE,
                "STUDYID": str(g["STUDYID"].iloc[0]),
                "n_encoder_available": int(len(enc_all)), "n_future_available": 0,
                "last_encoder_day": float(enc_all["DAY"].max())}
    enc = enc_all.tail(int(max_encoder_length))
    fut = fut_all.head(int(max_prediction_length))
    enc_days = [float(x) for x in enc["DAY"].to_numpy(float)]
    fut_days = [float(x) for x in fut["DAY"].to_numpy(float)]
    last = float(enc_days[-1])
    return {
        "USUBJID": str(usubjid),
        "STUDYID": str(g["STUDYID"].iloc[0]),
        "eligible": True,
        "exclude_reason": None,
        "cutoff": float(cutoff),
        "encoder_days": enc_days,
        "encoder_logw": [float(x) for x in enc["LOGW"].to_numpy(float)],
        "last_encoder_day": last,
        "target_days": fut_days,
        "target_logw": [float(x) for x in fut["LOGW"].to_numpy(float)],
        "lead_times": [float(d - last) for d in fut_days],
        "n_encoder_available": int(len(enc_all)),
        "n_encoder_used": int(len(enc)),
        "n_future_available": int(len(fut_all)),
        "n_future_used": int(len(fut)),
        "truncated_encoder": bool(len(enc_all) > len(enc)),
        "truncated_future": bool(len(fut_all) > len(fut)),
        "unused_future_days": [float(x) for x in fut_all["DAY"].to_numpy(float)[len(fut):]],
    }


def construct_forecast_examples(obs: pd.DataFrame, animals: pd.DataFrame, *, cutoff: float,
                                max_encoder_length: int, max_prediction_length: int) -> dict[str, Any]:
    """One example per eligible animal. Targets are that animal's measured days only."""
    recs = []
    excluded = []
    ids = sorted({str(u) for u in animals["USUBJID"].astype(str)} | {str(u) for u in obs["USUBJID"].astype(str)})
    meta = animals.drop_duplicates("USUBJID").set_index(animals.drop_duplicates("USUBJID")["USUBJID"].astype(str))
    for u in ids:
        w = animal_windows(obs, u, cutoff=cutoff, max_encoder_length=max_encoder_length,
                           max_prediction_length=max_prediction_length)
        if u in meta.index:
            row = meta.loc[u]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            w["SPECIES"] = str(row["SPECIES"]) if "SPECIES" in meta.columns else None
            w["SEX"] = str(row["SEX"]) if "SEX" in meta.columns else None
            if "STUDYID" not in w or w.get("STUDYID") is None:
                w["STUDYID"] = str(row["STUDYID"])
        if not w.get("eligible"):
            excluded.append(w)
            continue
        recs.append(w)
    examples = pd.DataFrame(recs)
    excl = pd.DataFrame(excluded)
    by_reason = excl["exclude_reason"].value_counts().to_dict() if len(excl) and "exclude_reason" in excl else {}
    return {
        "cutoff": float(cutoff),
        "max_encoder_length": int(max_encoder_length),
        "max_prediction_length": int(max_prediction_length),
        "n_eligible": int(len(examples)),
        "n_excluded": int(len(excl)),
        "exclude_counts": {str(k): int(v) for k, v in by_reason.items()},
        "examples": examples,
        "exclusions": excl,
        "note": ("Encoder = last <=max_encoder_length observed visits with DAY<=cutoff. "
                 "Targets = first <=max_prediction_length observed visits with DAY>cutoff. "
                 "No long-horizon chunking. No unmeasured target days."),
    }


def examples_to_tables(pack: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed encoder and future tables aligned to constructed examples."""
    hist_rows, fut_rows = [], []
    for rec in pack["examples"].to_dict("records"):
        u, sid = rec["USUBJID"], rec["STUDYID"]
        for d, z in zip(rec["encoder_days"], rec["encoder_logw"], strict=True):
            hist_rows.append({"USUBJID": u, "STUDYID": sid, "DAY": float(d), "LOGW": float(z)})
        for d, z, lead in zip(rec["target_days"], rec["target_logw"], rec["lead_times"], strict=True):
            fut_rows.append({"USUBJID": u, "STUDYID": sid, "DAY": float(d), "LOGW": float(z),
                             "LEAD_DAYS": float(lead), "LAST_ENCODER_DAY": float(rec["last_encoder_day"])})
    hist = pd.DataFrame(hist_rows)
    fut = pd.DataFrame(fut_rows)
    return hist, fut


def assert_windows_match_obs(obs: pd.DataFrame, pack: dict[str, Any], usubjid: str) -> dict[str, Any]:
    """Re-derive windows from obs and compare to a stored example."""
    ex = pack["examples"]
    hit = ex.loc[ex["USUBJID"].astype(str) == str(usubjid)]
    if hit.empty:
        raise AssertionError(f"{usubjid} is not an eligible example")
    rec = hit.iloc[0].to_dict()
    fresh = animal_windows(
        obs, usubjid, cutoff=float(pack["cutoff"]),
        max_encoder_length=int(pack["max_encoder_length"]),
        max_prediction_length=int(pack["max_prediction_length"]))
    for key in ("encoder_days", "target_days", "encoder_logw", "target_logw", "last_encoder_day"):
        a, b = rec[key], fresh[key]
        if isinstance(a, list):
            if [float(x) for x in a] != [float(x) for x in b]:
                raise AssertionError(f"{usubjid} {key} mismatch: {a} vs {b}")
        elif float(a) != float(b):
            raise AssertionError(f"{usubjid} {key} mismatch: {a} vs {b}")
    return {"USUBJID": str(usubjid), "ok": True, "encoder_days": fresh["encoder_days"],
            "target_days": fresh["target_days"]}


def gaussian_quantiles(pred: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
    """Map MEAN/VAR Gaussian marginals to quantile columns. Label as Gaussian, not a quantile head."""
    from scipy import stats

    levels = [float(q) for q in (quantiles or QUANTILES)]
    out = pred[["USUBJID", "DAY"]].copy()
    if "STUDYID" in pred.columns:
        out["STUDYID"] = pred["STUDYID"]
    mean = pred["MEAN"].to_numpy(float)
    sd = np.sqrt(np.maximum(pred["VAR"].to_numpy(float), 1e-12))
    for q in levels:
        out[f"q{q}"] = mean + sd * float(stats.norm.ppf(q))
    out["MEAN"] = mean
    out["VAR"] = pred["VAR"].to_numpy(float)
    out["point_definition"] = "gaussian_mean_conditioned_on_early_history"
    return out


def score_aligned_quantiles(fut: pd.DataFrame, pred: pd.DataFrame, *, representation: str,
                            quantiles: list[float] | None = None) -> dict[str, Any]:
    """Study-balanced scores plus relative error and quantile crossing on actual (USUBJID, DAY)."""
    levels = [float(q) for q in (quantiles or QUANTILES)]
    base = score_quantile_table(fut, pred, quantiles=levels, representation=representation)
    mm, card = align_prediction_table(fut, pred)
    extra: dict[str, Any] = {"merge": card, "n_matched": int(len(mm))}
    if mm.empty:
        extra.update({"relative_abs_log": None, "relative_abs_gram": None, "crossing_rate": None})
        base.update(extra)
        return base
    y = mm["LOGW"].to_numpy(float)
    q50 = mm["q0.5"].to_numpy(float) if "q0.5" in mm.columns else mm["MEAN"].to_numpy(float)
    studies = mm["STUDYID"].to_numpy() if "STUDYID" in mm.columns else mm["USUBJID"].to_numpy()
    rel_log = np.abs(q50 - y) / np.maximum(np.abs(y), 1e-8)
    yg, pg = np.exp(y), np.exp(q50)
    rel_g = np.abs(pg - yg) / np.maximum(yg, 1e-8)
    extra["relative_abs_log"] = _study_mean(rel_log, studies)
    extra["relative_abs_gram"] = _study_mean(rel_g, studies)
    qcols = [c for c in (f"q{q}" for q in levels) if c in mm.columns]
    if len(qcols) >= 2:
        arr = mm[qcols].to_numpy(float)
        extra["crossing_rate"] = {
            "pooled": float(np.mean(np.any(np.diff(arr, axis=1) < -1e-12, axis=1))),
            "n_obs": int(len(mm)),
        }
    else:
        extra["crossing_rate"] = {"pooled": None, "n_obs": int(len(mm))}
    if "LEAD_DAYS" in mm.columns:
        extra["per_lead_days"] = _lead_breakdown(mm, q50, y, levels)
    yg, pg = np.exp(y), np.exp(q50)
    extra["mae_gram"] = _study_mean(np.abs(pg - yg), studies)
    extra["mae_gram_definition"] = (
        "study-balanced mean |exp(point) - exp(LOGW)| in grams; "
        "exp(LOGW) = WEIGHT_G when LOGW = ln(WEIGHT_G)"
    )
    extra["quantile_loss_gram"] = {}
    gram_pins = []
    for q in levels:
        col = f"q{q}"
        if col not in mm.columns:
            continue
        qg = np.exp(mm[col].to_numpy(float))
        pin_g = np.where(yg - qg >= 0, q * (yg - qg), (q - 1.0) * (yg - qg))
        extra["quantile_loss_gram"][str(q)] = _study_mean(pin_g, studies)
        gram_pins.append(pin_g)
    extra["mean_quantile_loss_gram"] = (
        _study_mean(np.mean(np.stack(gram_pins, axis=0), axis=0), studies) if gram_pins else None)
    extra["units"] = {
        "mae_log": "ln(grams)",
        "rmse_log": "ln(grams)",
        "mae_gram": "grams",
        "quantile_loss": "pinball on LOGW = ln(WEIGHT_G)",
        "quantile_loss_gram": "pinball on grams = exp(quantile)",
        "interval_width80": "q0.9 - q0.1 on LOGW",
    }
    base.update(extra)
    return base


def _study_mean(values: np.ndarray, studies: np.ndarray) -> dict[str, float]:
    w = study_balanced_weights(studies)
    by = pd.DataFrame({"v": values, "s": studies}).groupby("s")["v"].mean()
    return {
        "pooled": float(np.mean(values)),
        "study_balanced": float(by.mean()) if len(by) else float("nan"),
        "n_obs": int(len(values)),
        "n_studies": int(by.shape[0]),
        "weight_mean": float(w.mean()),
    }


def _lead_breakdown(mm: pd.DataFrame, point: np.ndarray, y: np.ndarray,
                    levels: list[float]) -> list[dict[str, Any]]:
    work = mm.reset_index(drop=True)
    work["_point"] = point
    work["_y"] = y
    work["_lead"] = np.round(work["LEAD_DAYS"].to_numpy(float), 6)
    rows = []
    for lead, g in work.groupby("_lead", sort=True):
        e = g["_point"].to_numpy(float) - g["_y"].to_numpy(float)
        rec = {"lead_days": float(lead), "n_obs": int(len(g)),
               "mae_log": float(np.mean(np.abs(e))),
               "rmse_log": float(np.sqrt(np.mean(e ** 2)))}
        if "q0.1" in g.columns and "q0.9" in g.columns:
            rec["coverage80"] = float(((g["_y"] >= g["q0.1"]) & (g["_y"] <= g["q0.9"])).mean())
            rec["interval_width80"] = float((g["q0.9"] - g["q0.1"]).mean())
        rows.append(rec)
    return rows


def quantile_table_from_model(model, animals: pd.DataFrame, pack: dict[str, Any],
                              hist: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Predict per animal on that animal's actual target days. Shared times are never mixed."""
    failures: list[dict[str, Any]] = []
    recs = []
    ctx_all = animals.drop_duplicates("USUBJID").set_index(animals.drop_duplicates("USUBJID")["USUBJID"].astype(str))
    for rec in pack["examples"].to_dict("records"):
        u = rec["USUBJID"]
        if u not in ctx_all.index:
            failures.append({"USUBJID": u, "reason": "animal_missing_from_context"})
            continue
        a = ctx_all.loc[[u]].reset_index(drop=True)
        times = np.asarray(rec["target_days"], float)
        hh = hist.loc[hist["USUBJID"].astype(str) == str(u)]
        try:
            if hasattr(model, "predict_quantiles"):
                pred = model.predict_quantiles(a, times, history=hh)
            else:
                pred = gaussian_quantiles(model.predict_marginals(a, times, history=hh))
        except Exception as e:  # noqa: BLE001
            failures.append({"USUBJID": u, "reason": f"{type(e).__name__}: {e}"})
            continue
        if pred is None or pred.empty:
            failures.append({"USUBJID": u, "reason": "empty_prediction"})
            continue
        pred = pred.copy()
        pred["USUBJID"] = pred["USUBJID"].astype(str)
        pred["STUDYID"] = rec["STUDYID"]
        recs.append(pred)
    if not recs:
        return pd.DataFrame(), failures
    return pd.concat(recs, ignore_index=True), failures


def history_lead_report(pack: dict[str, Any]) -> dict[str, Any]:
    """Configured limits vs lengths actually exercised by observed DAY values."""
    ex = pack.get("examples")
    if ex is None or getattr(ex, "empty", True):
        return {"n_eligible": 0}
    n_enc = ex["n_encoder_used"].to_numpy(int)
    n_fut = ex["n_future_used"].to_numpy(int)
    n_enc_av = ex["n_encoder_available"].to_numpy(int)
    n_fut_av = ex["n_future_available"].to_numpy(int)
    leads = []
    for rec in ex.to_dict("records"):
        leads.extend(float(x) for x in rec.get("lead_times") or [])
    return {
        "configured": {
            "cutoff": pack.get("cutoff"),
            "max_encoder_length": pack.get("max_encoder_length"),
            "max_prediction_length": pack.get("max_prediction_length"),
        },
        "exercised": {
            "n_eligible_animals": int(len(ex)),
            "encoder_used_min": int(n_enc.min()),
            "encoder_used_median": float(np.median(n_enc)),
            "encoder_used_max": int(n_enc.max()),
            "encoder_available_max": int(n_enc_av.max()),
            "n_animals_truncated_encoder": int(ex["truncated_encoder"].sum()) if "truncated_encoder" in ex else 0,
            "future_used_min": int(n_fut.min()),
            "future_used_median": float(np.median(n_fut)),
            "future_used_max": int(n_fut.max()),
            "future_available_max": int(n_fut_av.max()),
            "n_animals_truncated_future": int(ex["truncated_future"].sum()) if "truncated_future" in ex else 0,
            "lead_days_min": float(min(leads)) if leads else None,
            "lead_days_median": float(np.median(leads)) if leads else None,
            "lead_days_max": float(max(leads)) if leads else None,
            "lead_days_unique": sorted({round(x, 6) for x in leads}),
        },
        "note": (
            "Lead times are target DAY minus last encoder DAY for that animal. "
            "Configured caps are not the lengths the data exercised unless they bind."
        ),
    }


def paired_study_differences(per_a: list[dict[str, Any]], per_b: list[dict[str, Any]],
                             *, metric: str, name_a: str, name_b: str) -> dict[str, Any]:
    """Study-level paired deltas. Observations within a study are not replications."""
    da = {str(r.get("STUDYID")): r for r in per_a if r.get("STUDYID") is not None}
    db = {str(r.get("STUDYID")): r for r in per_b if r.get("STUDYID") is not None}
    keys = sorted(set(da) & set(db))
    rows = []
    for k in keys:
        a, b = da[k].get(metric), db[k].get(metric)
        if a is None or b is None:
            continue
        rows.append({
            "STUDYID": k, "metric": metric, name_a: float(a), name_b: float(b),
            "delta": float(b) - float(a), "n_obs_a": da[k].get("n_obs"), "n_obs_b": db[k].get("n_obs"),
        })
    deltas = [r["delta"] for r in rows]
    return {
        "metric": metric, "a": name_a, "b": name_b, "n_studies": len(rows),
        "rows": rows,
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "sd_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
        "note": "One row per study. Within-study visits are not treated as independent replications.",
    }


def mark_persistence_intervals_unavailable(block: dict[str, Any]) -> dict[str, Any]:
    """Last-value quantiles are degenerate. Interval metrics are not a calibrated method."""
    reason = ("Persistence repeats the last encoder LOGW at every quantile. "
              "No interval method was defined or calibrated without test outcomes.")
    for key in ("coverage80", "interval_width80", "coverage50", "interval_width50"):
        block[key] = {"study_balanced": None, "pooled": None, "unavailable": True, "reason": reason}
    block["interval_metrics"] = "unavailable"
    return block


def inventory_model_data(data: ModelData) -> dict[str, Any]:
    return {
        "n_animals": int(len(data.animals)),
        "n_obs": int(len(data.obs)),
        "n_studies": int(data.animals["STUDYID"].nunique()) if len(data.animals) else 0,
        "studyids": sorted(str(s) for s in data.studies),
        "species": (
            {str(k): int(v) for k, v in
             data.animals.groupby(data.animals["SPECIES"].astype(str))["USUBJID"].nunique().items()}
            if "SPECIES" in data.animals.columns else {}
        ),
        "columns_animals": list(data.animals.columns),
        "columns_obs": list(data.obs.columns),
    }

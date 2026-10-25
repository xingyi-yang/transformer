"""Quantile-crossing audit for TFT QuantileLoss outputs.

QuantileLoss does not enforce monotonicity. Rearrangement (sorting predicted
quantiles per row) is postprocessing only. Raw and corrected scores are
reported separately. Removing crossings after training is not improved training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from send_bw.evaluation.quantile_forecast import QUANTILES, align_prediction_table


def quantile_columns(pred: pd.DataFrame, levels: list[float] | None = None) -> list[tuple[float, str]]:
    wanted = [float(q) for q in (levels or QUANTILES)]
    found = []
    for q in wanted:
        name = f"q{q}"
        if name in pred.columns:
            found.append((q, name))
    return found


def rearrange_quantiles(pred: pd.DataFrame, levels: list[float] | None = None) -> pd.DataFrame:
    """Sort predicted quantiles within each row. Does not change the fitted model."""
    cols = quantile_columns(pred, levels)
    out = pred.copy()
    if len(cols) < 2:
        out["rearranged"] = False
        return out
    names = [c for _, c in cols]
    arr = np.sort(out[names].to_numpy(float), axis=1)
    for j, name in enumerate(names):
        out[name] = arr[:, j]
    out["rearranged"] = True
    out["rearrangement"] = "np.sort_per_row_increasing"
    return out


def _pair_flags(arr: np.ndarray, levels: list[float]) -> dict[str, np.ndarray]:
    flags = {}
    for i in range(len(levels) - 1):
        for j in range(i + 1, len(levels)):
            flags[f"q{levels[i]}>q{levels[j]}"] = arr[:, i] > arr[:, j] + 1e-12
    return flags


def crossing_audit(pred: pd.DataFrame, fut: pd.DataFrame | None = None,
                   levels: list[float] | None = None,
                   animals: pd.DataFrame | None = None) -> dict[str, Any]:
    """Crossing rates by pair, species, and forecast lead. Inner vs endpoint."""
    cols = quantile_columns(pred, levels)
    if len(cols) < 2:
        return {"status": "not_run", "reason": "fewer than two quantile columns"}
    levels = [q for q, _ in cols]
    names = [c for _, c in cols]
    work = pred.copy()
    work["USUBJID"] = work["USUBJID"].astype(str)
    if fut is not None and len(fut) and "LEAD_DAYS" in fut.columns:
        mm, _card = align_prediction_table(fut, work)
        work = mm
    if animals is not None and "SPECIES" in animals.columns and "SPECIES" not in work.columns:
        spec = animals.drop_duplicates("USUBJID")[["USUBJID", "SPECIES"]].copy()
        spec["USUBJID"] = spec["USUBJID"].astype(str)
        work = work.merge(spec, on="USUBJID", how="left")
    arr = work[names].to_numpy(float)
    flags = _pair_flags(arr, levels)
    any_cross = np.zeros(len(work), dtype=bool)
    for f in flags.values():
        any_cross = any_cross | f
    lo, hi = levels[0], levels[-1]
    endpoint = flags.get(f"q{lo}>q{hi}", np.zeros(len(work), dtype=bool))
    # Inner: any adjacent inversion that is not only the endpoint pair.
    inner = np.zeros(len(work), dtype=bool)
    for i in range(len(levels) - 1):
        key = f"q{levels[i]}>q{levels[i + 1]}"
        if key in flags:
            inner = inner | flags[key]
    pair_rates = {k: float(np.mean(v)) for k, v in flags.items()}
    by_species = []
    if "SPECIES" in work.columns:
        tmp = work.assign(_any=any_cross, _end=endpoint, _inner=inner)
        for sp, g in tmp.groupby(tmp["SPECIES"].astype(str), sort=True):
            by_species.append({
                "SPECIES": str(sp),
                "n_obs": int(len(g)),
                "any_crossing": float(g["_any"].mean()),
                "endpoint_reversed": float(g["_end"].mean()),
                "inner_adjacent": float(g["_inner"].mean()),
            })
    by_lead = []
    if "LEAD_DAYS" in work.columns:
        tmp = work.assign(_any=any_cross, _end=endpoint, _inner=inner,
                          _lead=np.round(work["LEAD_DAYS"].to_numpy(float), 6))
        for lead, g in tmp.groupby("_lead", sort=True):
            by_lead.append({
                "lead_days": float(lead),
                "n_obs": int(len(g)),
                "any_crossing": float(g["_any"].mean()),
                "endpoint_reversed": float(g["_end"].mean()),
                "inner_adjacent": float(g["_inner"].mean()),
            })
    return {
        "status": "ok",
        "quantiles": levels,
        "n_obs": int(len(work)),
        "any_crossing": float(np.mean(any_cross)),
        "inner_adjacent_crossing": float(np.mean(inner)),
        "endpoint_reversed": float(np.mean(endpoint)),
        "endpoint_pair": f"q{lo}>q{hi}",
        "pair_rates": pair_rates,
        "by_species": by_species,
        "by_lead_days": by_lead,
        "rearrangement": "not_applied_in_this_block",
        "note": (
            "Inner = adjacent quantile inversions (e.g. q0.25>q0.5). "
            "Endpoint = q_min > q_max (reversed 80% interval if those are 0.1/0.9). "
            "A row may contribute to both."
        ),
    }


def curve_still_improving(rows: list[dict[str, Any]], key: str = "val_loss") -> dict[str, Any]:
    """Decide from validation only whether the last epoch was still the best."""
    vals = []
    for r in rows:
        if r.get(key) is None:
            continue
        try:
            vals.append((int(r["epoch"]), float(r[key])))
        except (TypeError, ValueError, KeyError):
            continue
    if not vals:
        return {"available": False, "still_improving": None, "reason": f"{key} missing"}
    vals.sort()
    best_epoch, best = min(vals, key=lambda x: x[1])
    last_epoch, last = vals[-1]
    return {
        "available": True,
        "key": key,
        "n_logged": len(vals),
        "best_epoch": int(best_epoch),
        "best_value": float(best),
        "last_epoch": int(last_epoch),
        "last_value": float(last),
        "last_is_best": bool(last_epoch == best_epoch),
        "still_improving": bool(last_epoch == best_epoch),
        "note": "Validation only. Test errors are not used.",
    }

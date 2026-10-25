"""Effective study weight after PF window construction.

Intended objective: equal average of per-study forecast losses.
Row-level study-balanced WEIGHT does not achieve that after sliding-window
multiplicity and decoder-step reduction. Do not infer balance from raw
window counts alone.

When correcting, set WEIGHT=1 and sample windows with p ∝ 1/n_windows[study].
Do not combine that sampler with the existing row WEIGHT (double balancing).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from send_bw.models.tft import STATIC_CATS, STATIC_REALS, TARGET, _enough_visits, _pf, build_observed_panel


def window_decoder_contributions(dataset, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per PF window with the decoder-step WEIGHT sum actually reduced."""
    decoded = dataset.decoded_index.copy()
    decoded["USUBJID"] = decoded["USUBJID"].astype(str)
    key = panel[["USUBJID", "STUDYID", "time_idx", "WEIGHT"]].copy()
    key["USUBJID"] = key["USUBJID"].astype(str)
    key["time_idx"] = key["time_idx"].astype(int)
    if "SPECIES" in panel.columns:
        spec = panel[["USUBJID", "SPECIES"]].drop_duplicates()
        spec["USUBJID"] = spec["USUBJID"].astype(str)
        decoded = decoded.merge(spec, on="USUBJID", how="left")
    sid = panel[["USUBJID", "STUDYID"]].drop_duplicates()
    sid["USUBJID"] = sid["USUBJID"].astype(str)
    decoded = decoded.merge(sid, on="USUBJID", how="left")
    recs = []
    for i, row in decoded.reset_index(drop=True).iterrows():
        u = str(row["USUBJID"])
        first = int(row["time_idx_first_prediction"])
        last = int(row["time_idx_last"])
        dec = key.loc[(key["USUBJID"] == u) & (key["time_idx"] >= first) & (key["time_idx"] <= last)]
        recs.append({
            "window_i": int(i),
            "USUBJID": u,
            "STUDYID": str(row["STUDYID"]),
            "SPECIES": str(row["SPECIES"]) if "SPECIES" in row.index and pd.notna(row.get("SPECIES")) else None,
            "n_decoder": int(last - first + 1),
            "decoder_weight_sum": float(dec["WEIGHT"].sum()) if len(dec) else 0.0,
            "encoder_length": int(first - int(row["time_idx_first"])),
        })
    return pd.DataFrame(recs)


def _shares(values: pd.Series) -> dict[str, float]:
    total = float(values.sum())
    if total <= 0:
        return {str(k): 0.0 for k in values.index}
    return {str(k): float(v / total) for k, v in values.items()}


def summarize_contributions(windows: pd.DataFrame, *, sampler: str) -> dict[str, Any]:
    if windows.empty:
        return {"n_windows": 0, "sampler": sampler}
    by = windows.groupby(windows["STUDYID"].astype(str), sort=True)
    n_win = by.size()
    dec_w = by["decoder_weight_sum"].sum()
    dec_len = by["n_decoder"].sum()
    n_studies = int(n_win.shape[0])
    # Uniform window sampling (PF default): P(window)=1/N.
    expected_uniform = dec_w / float(len(windows))
    # Study-balanced window sampler: P(study)=1/n_studies, then a uniform window of that study.
    mean_dec_w = by["decoder_weight_sum"].mean()
    expected_balanced = mean_dec_w / float(n_studies)
    species = {}
    if "SPECIES" in windows.columns and windows["SPECIES"].notna().any():
        species = {str(k): int(v) for k, v in windows.groupby(windows["SPECIES"].astype(str)).size().items()}
    return {
        "intended_objective": "equal average of per-study forecast losses",
        "sampler": sampler,
        "n_windows": int(len(windows)),
        "n_studies": n_studies,
        "windows_per_study": {str(k): int(v) for k, v in n_win.items()},
        "decoder_steps_per_study": {str(k): int(v) for k, v in dec_len.items()},
        "decoder_weight_sum_per_study": {str(k): float(v) for k, v in dec_w.items()},
        "window_count_share": _shares(n_win.astype(float)),
        "effective_loss_share_uniform_windows": _shares(dec_w),
        "expected_loss_share_if_study_sampler": _shares(expected_balanced),
        "max_minus_min_effective_share": float(
            max(_shares(dec_w).values()) - min(_shares(dec_w).values())) if n_studies else None,
        "max_minus_min_window_share": float(
            max(_shares(n_win.astype(float)).values()) - min(_shares(n_win.astype(float)).values())) if n_studies else None,
        "species_windows": species,
        "expected_uniform_window_contrib": {str(k): float(v) for k, v in expected_uniform.items()},
        "note": (
            "Effective share is the sum of decoder-step WEIGHT over windows of that study, "
            "divided by the grand total. This is the quantity PF QuantileLoss reduces. "
            "Window counts alone omit decoder length and WEIGHT."
        ),
    }


def audit_training_weights(data, *, max_encoder_length: int, max_prediction_length: int,
                           min_encoder_length: int = 1) -> dict[str, Any]:
    """Build the PF training index and report effective study contributions."""
    TimeSeriesDataSet, _, _ = _pf()
    from pytorch_forecasting.data import NaNLabelEncoder

    panel = build_observed_panel(data)
    panel = _enough_visits(panel, min_encoder_length + 1)
    if panel.empty:
        return {"n_windows": 0, "note": "empty panel"}
    cat_enc = {c: NaNLabelEncoder(add_nan=True) for c in ["USUBJID", *STATIC_CATS]}
    ds = TimeSeriesDataSet(
        panel,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["USUBJID"],
        weight="WEIGHT",
        max_encoder_length=int(max_encoder_length),
        min_encoder_length=int(min_encoder_length),
        max_prediction_length=int(max_prediction_length),
        min_prediction_length=1,
        static_categoricals=list(STATIC_CATS),
        static_reals=list(STATIC_REALS),
        time_varying_known_reals=["DAY", "TAU", "EXPO", "OFF", "LCUM"],
        time_varying_unknown_reals=[TARGET],
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        categorical_encoders=cat_enc,
    )
    windows = window_decoder_contributions(ds, panel)
    before = summarize_contributions(windows, sampler="uniform_windows_plus_row_study_WEIGHT")
    # Counterfactual: WEIGHT=1, then the same windows.
    flat = panel.copy()
    flat["WEIGHT"] = 1.0
    windows_flat = window_decoder_contributions(ds, flat)
    after = summarize_contributions(windows_flat, sampler="study_balanced_window_sampler_WEIGHT_1")
    imbalance = bool((before.get("max_minus_min_effective_share") or 0) > 0.05)
    target_vs_donor = _target_donor_split(windows, data)
    return {
        "imbalance_confirmed": imbalance,
        "threshold_max_minus_min_effective_share": 0.05,
        "before": before,
        "after_if_corrected": after,
        "correction": {
            "method": "WeightedRandomSampler on PF windows with p proportional to 1/n_windows[study]",
            "row_weight": 1.0,
            "double_balancing": False,
            "remaining_imbalance": (
                "Studies with longer decoder sequences still contribute more steps per drawn window. "
                "That residual is recorded, not removed by a second incompatible weight."
            ),
            "applied_when": "config window_balance=study",
        },
        "target_vs_donor": target_vs_donor,
        "n_train_series": int(panel["USUBJID"].nunique()),
        "n_train_obs": int(len(panel)),
    }


def _target_donor_split(windows: pd.DataFrame, data) -> dict[str, Any]:
    if windows.empty or "SPECIES" not in getattr(data, "animals", pd.DataFrame()).columns:
        return {"available": False}
    spec = data.animals.drop_duplicates("USUBJID")[["USUBJID", "SPECIES"]].copy()
    spec["USUBJID"] = spec["USUBJID"].astype(str)
    spec["SPECIES"] = spec["SPECIES"].astype(str).str.upper()
    w = windows.merge(spec, on="USUBJID", how="left", suffixes=("", "_ctx"))
    col = "SPECIES_ctx" if "SPECIES_ctx" in w.columns else "SPECIES"
    target = str(getattr(data.spec, "species", "RAT")).upper()
    is_tgt = w[col].astype(str).str.upper() == target
    def _block(mask: pd.Series) -> dict[str, Any]:
        sub = w.loc[mask]
        return {
            "n_windows": int(len(sub)),
            "n_studies": int(sub["STUDYID"].nunique()) if len(sub) else 0,
            "decoder_weight_sum": float(sub["decoder_weight_sum"].sum()) if len(sub) else 0.0,
        }
    return {
        "available": True,
        "target_species": target,
        "target": _block(is_tgt),
        "donor": _block(~is_tgt),
    }


def study_window_sampler(dataset, panel: pd.DataFrame):
    """WeightedRandomSampler: each study has equal probability of contributing a window."""
    from torch.utils.data import WeightedRandomSampler

    windows = window_decoder_contributions(dataset, panel)
    n = windows.groupby("STUDYID").size()
    w = windows["STUDYID"].map(lambda s: 1.0 / float(n[s])).to_numpy(float)
    return WeightedRandomSampler(weights=w, num_samples=int(len(windows)), replacement=True)


def apply_unit_row_weights(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["WEIGHT"] = 1.0
    return out

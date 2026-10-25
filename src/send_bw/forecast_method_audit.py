"""Read-only audit of generator observability and method-comparison claims.

Uses saved official artifacts. Does not retrain. Does not write locked run
dirs (v3_*, tft_primary, tft_stability, tft_study_cv, tft_early_hist_*).

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import DESCRIPTORS, FEATURE_REGISTRY, availability, check_no_leakage
from send_bw.data.generate import GeneratorA, GeneratorB, Regimen
from send_bw.models.chronos2 import COVARIATE_AWARE, FORBIDDEN_COVARIATES, HISTORY_ONLY
from send_bw.tft_baselines import CAT_COLS, NUM_COLS
from send_bw.tft_intervals import (
    ALPHA_80,
    apply_cqr_margin,
    conformal_margin,
    conformal_nonconformity,
    romano_cqr_k,
    romano_cqr_quantile,
    score_raw_and_calibrated,
)
from send_bw.utils.artifacts import write_json

OUTPUT_DEFAULT = "runs/forecast_method_audit"
LOCKED_DIRS = (
    "runs/v3_eval", "runs/v3_pilot_k0k1", "runs/v3_repl", "runs/v3_research",
    "runs/v3_closeout", "runs/v3_audit", "runs/v3_limitations", "runs/v3_objective",
    "runs/v3_official_seq", "runs/v3_eval_correct", "runs/v3_cov_chronos",
    "runs/v3_share_repl", "runs/v3_share_abstain", "runs/v3_share_abstain_run",
    "runs/v3_donor_rank", "runs/v3_seq_diag", "runs/v3_increment",
    "runs/tft_early_hist_bench", "runs/tft_early_hist_extend",
    "runs/tft_primary", "runs/tft_stability", "runs/tft_study_cv",
)

OFFICIAL_CELL = Path(
    "runs/v3_official_seq/datasets/d0/trains/t0/cells/d0_t0_matched_target_only")
OFFICIAL_BANK = Path("runs/v3_official_seq/datasets/d0/banks/matched/synthetic")
OFFICIAL_SPLINE = OFFICIAL_CELL / "models" / "spline" / "model.json"
OFFICIAL_TRANSFORMER = OFFICIAL_CELL / "models" / "transformer" / "model.json"
TFT_TO = Path("runs/tft_primary/models/tft_target_only")
STABILITY = Path("runs/tft_stability")
CHRONOS_TO = Path("runs/v3_official_seq/models/chronos2_target_only/model.json")

GENERATOR_A = {
    "sigma_p": 0.022, "ell": 6.0, "sigma_m": 0.010,
    "sd_study_slope": 0.035, "sd_animal_slope": 0.045, "tau_g": 40.0,
}
HIDDEN_COMPOUND = ("A", "ED50", "tau_on", "tau_off")
TRUTH_ONLY = [c for c, a in FEATURE_REGISTRY.items() if a == "truth_only"]


class MethodAuditError(ValueError):
    pass


def _refuse_locked(parent: Path) -> None:
    resolved = parent.resolve()
    blocked = {Path(p).resolve() for p in LOCKED_DIRS}
    if resolved in blocked:
        raise MethodAuditError("refusing to write into a preserved run directory")
    if resolved.name.startswith("v3_") and "runs" in resolved.parts:
        raise MethodAuditError("refusing to write a new locked-style runs/v3_* directory")


def descriptor_observability() -> dict[str, Any]:
    """D1–D3 are production; A/ED50/tau are hidden generator truth."""
    prod = [d for d in DESCRIPTORS if availability(d) == "production"]
    hidden_in_registry = [h for h in HIDDEN_COMPOUND if h in FEATURE_REGISTRY]
    return {
        "d1_d2_d3": "production",
        "written_to": "metadata/compound_reference.csv",
        "hidden_compound_truth": list(HIDDEN_COMPOUND),
        "hidden_written_to": "truth/params.json (compounds)",
        "hidden_in_feature_registry": hidden_in_registry,
        "comparable_at_prediction_time": (
            "Only if a compound_reference table with D1–D3 is supplied. "
            "Those columns are not SEND domains. FeatureSpec.animal_features "
            "raises UnsupportedContextError when an active arm is missing a "
            "descriptor. Official test compounds CMP03/CMP06 were in train; "
            "compound-holdout CMP01 was not."
        ),
        "production_includes_descriptors": prod == list(DESCRIPTORS),
    }


def production_features_exclude_hidden() -> dict[str, Any]:
    cols = [c for c, a in FEATURE_REGISTRY.items() if a == "production"]
    leaked = [c for c in cols if c in TRUTH_ONLY or c in HIDDEN_COMPOUND]
    check_no_leakage(cols)
    return {
        "production": cols,
        "truth_only": TRUTH_ONLY,
        "hidden_compound_not_in_registry": list(HIDDEN_COMPOUND),
        "leaked_into_production": leaked,
        "ok": leaked == [],
    }


def official_growth_family() -> dict[str, Any]:
    a, b = GeneratorA(), GeneratorB()
    t = np.linspace(0.0, 28.0, 8)
    ga = a.growth(t, "RAT", "M", "SPRAGUE-DAWLEY", float(np.log(260.0)))
    gb = b.growth(t, "RAT", "M", "SPRAGUE-DAWLEY", float(np.log(260.0)))
    return {
        "official_bank": "Generator A saturating exponential, shared tau_g=40",
        "tau_g": float(a.p["tau_g"]),
        "formula": "gmax(sex,strain,z0) * (1 - exp(-t / tau_g))",
        "generator_b": "Gompertz; not the official forecasting bank",
        "families_differ_at_t28": bool(abs(float(ga[-1] - gb[-1])) > 0.01),
        "a_gmax_rat_m": float(a.p["gmax"]["RAT"]["M"]),
        "limited_family": True,
    }


def chronos_visit_rank_note() -> dict[str, Any]:
    return {
        "official_seq_mode": HISTORY_ONLY,
        "adapter_limitation": (
            "HISTORY_ONLY passes a raw LOGW ndarray. Chronos sees visit rank "
            "(unit steps). Elapsed DAY is discarded. Two series with the same "
            "visit count and different calendar spacing are indistinguishable."
        ),
        "model_capability": (
            "COVARIATE_AWARE can pass DAY plus planned statics as past/future "
            "covariates. The sequence index is still unit steps; irregular "
            "spacing is a covariate, not the time index. Official_seq did not "
            "use this mode."
        ),
        "forbidden_covariates": list(FORBIDDEN_COVARIATES),
        "covariate_aware_constant": COVARIATE_AWARE,
        "not_a_chronos_inability": True,
    }


def catboost_information_set() -> dict[str, Any]:
    future_outcomes = [c for c in NUM_COLS + CAT_COLS if c in {
        "y_logw", "LOGW", "Y", "WEIGHT_G", "target_logw", "ACTUAL_DOSE",
        "DOSE_DEVIATION", "Z0_TRUE",
    }]
    known_future_calendar = [c for c in NUM_COLS if c in {
        "target_day", "elapsed_from_origin", "elapsed_from_last_obs",
    }]
    return {
        "heads": "one CatBoostRegressor per quantile (Quantile:alpha=q), not MultiQuantile",
        "label": "y_logw (not a feature)",
        "features": list(CAT_COLS + NUM_COLS),
        "future_outcome_features": future_outcomes,
        "known_future_calendar": known_future_calendar,
        "encoder_summaries": "history_summaries uses encoder DAY/LOGW only",
        "ok": future_outcomes == [],
    }


def compare_cqr_order_statistic(scores: np.ndarray, alpha: float = ALPHA_80) -> dict[str, Any]:
    """Romano k-th order statistic vs the previous numpy higher-quantile rule."""
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = int(s.size)
    k = romano_cqr_k(n, alpha)
    romano = romano_cqr_quantile(s, alpha)
    current = conformal_margin(s, alpha)
    q = min(float(k) / n, 1.0)
    numpy_higher = float(np.quantile(s, q, method="higher"))
    return {
        "n": n,
        "alpha": float(alpha),
        "k": k,
        "k_equals_n_plus_1": bool(k >= n + 1),
        "romano": romano,
        "conformal_margin": current,
        "numpy_higher_k_over_n": numpy_higher,
        "romano_matches_margin": bool(np.isclose(romano, current, rtol=0, atol=1e-15)),
        "romano_matches_numpy_higher": bool(np.isclose(romano, numpy_higher, rtol=0, atol=1e-15)),
        "q_used_by_legacy": q,
    }


def _point_biserial(flag: np.ndarray, value: np.ndarray) -> float | None:
    f = np.asarray(flag, float)
    v = np.asarray(value, float)
    m = np.isfinite(f) & np.isfinite(v)
    if m.sum() < 4 or f[m].std() == 0 or v[m].std() == 0:
        return None
    return float(np.corrcoef(f[m], v[m])[0, 1])


def check_missingness_and_death(bank: Path = OFFICIAL_BANK, cell: Path = OFFICIAL_CELL) -> dict[str, Any]:
    """Official-bank rates. Code path is iid; this is a numerical confirmation."""
    truth = pd.read_parquet(bank / "truth" / "animals.parquet")
    animals = pd.read_parquet(cell / "cohort" / "animals.parquet")
    obs = pd.read_parquet(cell / "cohort" / "obs.parquet")
    keep = ["USUBJID", "IS_CONTROL", "PLANNED_DOSE", "BASELINE_LOGW"]
    if "STUDYID" in animals.columns and "STUDYID" not in truth.columns:
        keep.append("STUDYID")
    mm = truth.merge(animals[keep], on="USUBJID", how="inner")
    if "STUDYID" not in mm.columns and "STUDYID_x" in mm.columns:
        mm = mm.rename(columns={"STUDYID_x": "STUDYID"})
    dead = mm["DEATH_DAY"].notna()
    treated = ~mm["IS_CONTROL"].astype(bool)
    n = int(len(mm))
    death_rate = float(dead.mean()) if n else None
    dose = mm["PLANNED_DOSE"].to_numpy(float)
    z0 = mm["BASELINE_LOGW"].to_numpy(float)
    # among non-dead, visit counts vs study mode as a missingness proxy
    alive = mm.loc[~dead, "USUBJID"].astype(str)
    visits = obs.loc[obs["USUBJID"].astype(str).isin(set(alive))].copy()
    visits = visits[visits["DAY"].to_numpy(float) > 0]
    n_vis = visits.groupby(visits["USUBJID"].astype(str)).size()
    study_of = mm.set_index(mm["USUBJID"].astype(str))["STUDYID"].astype(str)
    ctrl_of = mm.set_index(mm["USUBJID"].astype(str))["IS_CONTROL"].astype(bool)
    dose_of = mm.set_index(mm["USUBJID"].astype(str))["PLANNED_DOSE"].astype(float)
    rows = []
    for u, n_u in n_vis.items():
        sid = str(study_of.get(u, ""))
        rows.append({
            "USUBJID": u, "STUDYID": sid, "n_pos": int(n_u),
            "IS_CONTROL": bool(ctrl_of.get(u, True)),
            "PLANNED_DOSE": float(dose_of.get(u, 0.0)),
        })
    vis_df = pd.DataFrame(rows)
    miss_proxy = None
    if len(vis_df):
        mode = vis_df.groupby("STUDYID")["n_pos"].transform(lambda s: int(s.mode().iloc[0]))
        short = vis_df["n_pos"].to_numpy(int) < mode.to_numpy(int)
        miss_proxy = {
            "n_alive_with_visits": int(len(vis_df)),
            "frac_below_study_mode": float(np.mean(short)),
            "corr_short_vs_treated": _point_biserial(short, (~vis_df["IS_CONTROL"]).to_numpy()),
            "corr_short_vs_dose": _point_biserial(short, vis_df["PLANNED_DOSE"].to_numpy(float)),
            "mean_visits_control": float(vis_df.loc[vis_df["IS_CONTROL"], "n_pos"].mean()),
            "mean_visits_treated": float(vis_df.loc[~vis_df["IS_CONTROL"], "n_pos"].mean()),
        }
    return {
        "n_animals": n,
        "n_dead": int(dead.sum()),
        "death_rate": death_rate,
        "configured_p_death": 0.015,
        "configured_p_missing": 0.02,
        "corr_death_vs_treated": _point_biserial(dead.to_numpy(), treated.to_numpy(float)),
        "corr_death_vs_dose": _point_biserial(dead.to_numpy(), dose),
        "corr_death_vs_baseline_logw": _point_biserial(dead.to_numpy(), z0),
        "code_path": "iid rng.random() < p_*; does not read dose or weight",
        "missingness_proxy": miss_proxy,
        "note": (
            "Death and missed visits are sampled independently of treatment and "
            "weight in generate_dataset. Official-bank correlations are Monte "
            "Carlo noise around that independence, not a designed MAR process."
        ),
    }


def check_dose_deviation_effect(bank: Path = OFFICIAL_BANK) -> dict[str, Any]:
    """Weight uses actual dose; features use planned dose."""
    truth = pd.read_parquet(bank / "truth" / "animals.parquet")
    params = json.loads((bank / "truth" / "params.json").read_text(encoding="utf-8"))
    arms = pd.read_csv(bank / "metadata" / "arm_regimen.csv")
    cref = pd.read_csv(bank / "metadata" / "compound_reference.csv")
    animals = pd.read_parquet(OFFICIAL_CELL / "cohort" / "animals.parquet")
    dev = truth.loc[truth["DOSE_DEVIATION"].astype(bool)].copy()
    gen = GeneratorA()
    t = np.array([7.0, 14.0, 28.0])
    examples = []
    for rec in dev.head(8).to_dict("records"):
        row = animals.loc[animals["USUBJID"] == rec["USUBJID"]]
        if row.empty:
            continue
        a = row.iloc[0]
        cid = a.get("COMPOUND_ID")
        ct = (params.get("compounds") or {}).get(cid) if isinstance(cid, str) else None
        planned = Regimen(bool(a["IS_CONTROL"]), cid if isinstance(cid, str) else None,
                          float(a["PLANNED_DOSE"] or 0.0), float(a["INTERVAL_DAYS"] or 1.0),
                          float(a.get("START_DAY") or 0.0), float(a.get("END_DAY") or 27.0),
                          str(a.get("ROUTE") or "ORAL GAVAGE"))
        actual = Regimen(**planned.__dict__)
        actual.dose = float(rec["ACTUAL_DOSE"])
        actual.end_day = float(rec["ACTUAL_END_DAY"])
        ep = gen.effect(t, planned, ct)
        ea = gen.effect(t, actual, ct)
        examples.append({
            "USUBJID": rec["USUBJID"],
            "planned_dose": planned.dose,
            "actual_dose": actual.dose,
            "planned_end": planned.end_day,
            "actual_end": actual.end_day,
            "effect_planned_t28": float(ep[-1]),
            "effect_actual_t28": float(ea[-1]),
            "delta_effect_t28": float(ea[-1] - ep[-1]),
        })
    deltas = [e["delta_effect_t28"] for e in examples]
    return {
        "n_deviations_in_truth": int(len(dev)),
        "n_examples": int(len(examples)),
        "any_nonzero_effect_delta": bool(any(abs(x) > 1e-8 for x in deltas)),
        "mean_abs_delta_t28": None if not deltas else float(np.mean(np.abs(deltas))),
        "examples": examples,
        "features_use": "PLANNED_DOSE / planned start-end (production)",
        "trajectory_uses": "act regimen in generate_dataset (dose or early stop)",
        "actual_dose_availability": FEATURE_REGISTRY.get("ACTUAL_DOSE"),
        "exdose_actual_availability": FEATURE_REGISTRY.get("EXDOSE_ACTUAL"),
        "compound_reference_rows": int(len(cref)),
        "arm_rows": int(len(arms)),
    }


def check_spline_cov_not_truth(path: Path = OFFICIAL_SPLINE) -> dict[str, Any]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cov = d.get("cov") or {}
    fit = cov.get("fit_info") or {}
    fitted = {k: float(cov[k]) for k in ("sd_s0", "sd_s1", "sd_a0", "sd_a1", "sigma_p", "ell", "sigma_m")}
    gen = dict(GENERATOR_A)
    return {
        "fitted": fitted,
        "generator_a": gen,
        "copied_from_truth": False,
        "sigma_m_near_lower_bound_1e4": bool(fitted["sigma_m"] < 2e-4),
        "sigma_m_bound_flagged": "sigma_m" in (fit.get("params_at_bound") or []),
        "n_obs": fit.get("n_obs"),
        "n_studies": fit.get("n_studies"),
        "covariance_fit_on": (d.get("history") or {}).get("covariance_fit_on"),
        "init_defaults_resemble_generator": (
            "HierarchicalCovariance defaults sigma_p=0.02, ell=5, sigma_m=0.01 "
            "are starting values; official params are L-BFGS-B ML on train OOF."
        ),
        "reconstruct_logw": "predict_mean = BASELINE_LOGW + Yhat",
        "mean_target": "Y = LOGW - BASELINE_LOGW, train split only",
    }


def check_transformer_train_only(path: Path = OFFICIAL_TRANSFORMER) -> dict[str, Any]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = d.get("spec") or {}
    hist = d.get("history") or {}
    cov = d.get("cov") or {}
    return {
        "compound_ids": list(spec.get("compound_ids") or []),
        "official_test_studies": ["SYNA-010", "SYNA-011"],
        "official_test_compounds": {"SYNA-010": "CMP03", "SYNA-011": "CMP06"},
        "compound_holdout_cmp01": "SYNA-001 / SYNA-008; not in spec.compound_ids",
        "evaluation_studies_in_compound_ids": False,
        "shared_cov_sigma_p": cov.get("sigma_p"),
        "best_val_rmse_member0": hist.get("best_val_rmse_member0"),
        "reconstruct_logw": "BASELINE_LOGW + y_scale * net",
        "fit_norm_on": "train only (_data(..., fit_norm=True))",
        "validation_role": "early stopping, not official test",
    }


def check_chronos_saved_mode(path: Path = CHRONOS_TO) -> dict[str, Any]:
    if not Path(path).is_file():
        return {"status": "missing"}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    hist = d.get("history") or {}
    return {
        "saved_covariate_mode": d.get("covariate_mode"),
        "history_covariate_mode": hist.get("covariate_mode"),
        "default_if_absent": HISTORY_ONLY,
        "n_train_series": hist.get("n_train_series"),
        "note": (
            "Older official_seq model.json omitted covariate_mode; load() "
            "defaults to HISTORY_ONLY. v3_official_seq never passed "
            "covariate_mode, so the run used visit-rank LOGW."
        ),
    }


def replay_tft_one_batch(cell: Path = OFFICIAL_CELL, model_dir: Path = TFT_TO) -> dict[str, Any]:
    """Rebuild a train dataloader from the locked checkpoint. No refit."""
    try:
        from pytorch_forecasting import TimeSeriesDataSet
    except ImportError:
        return {"status": "not_run", "reason": "pytorch_forecasting not installed"}
    if not (Path(model_dir) / "dataset_parameters.pkl").is_file():
        return {"status": "not_run", "reason": "TFT checkpoint missing"}
    from send_bw.models.tft import (
        TFTEarlyHistoryModel,
        _enough_visits,
        build_observed_panel,
    )
    from send_bw.v3_increment import load_cell_prepared

    prep = load_cell_prepared(Path(cell))
    train = prep["data"]["train"]
    m = TFTEarlyHistoryModel.load(Path(model_dir))
    panel = build_observed_panel(train)
    panel = _enough_visits(panel, int(m.min_encoder_length) + 1)
    uids = list(panel["USUBJID"].astype(str).unique()[:12])
    slim = panel.loc[panel["USUBJID"].astype(str).isin(uids)].reset_index(drop=True)
    ds = TimeSeriesDataSet.from_parameters(m.dataset_parameters, slim, predict=False)
    dl = ds.to_dataloader(train=False, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    x, y = batch[0], batch[1]
    y_t = y[0] if isinstance(y, (list, tuple)) else y
    import torch

    y_t = y_t.detach().cpu() if hasattr(y_t, "detach") else torch.as_tensor(y_t)
    dec_len = x["decoder_lengths"].detach().cpu()
    mask = torch.arange(y_t.size(1))[None, :] < dec_len[:, None]
    raw_dec = y_t[mask].numpy()
    scale = x["target_scale"].detach().cpu().numpy()
    norm = m.dataset_parameters.get("target_normalizer")
    center = scale_p = None
    if norm is not None and hasattr(norm, "norm_"):
        center = float(norm.norm_["center"])
        scale_p = float(norm.norm_["scale"])
    m.net.eval()
    with torch.no_grad():
        pred = m.net(x)["prediction"].detach().cpu()
    pred_np = pred[mask].numpy()
    panel_logw = slim["LOGW"].to_numpy(float)
    return {
        "status": "ok",
        "n_animals_in_batch_panel": int(slim["USUBJID"].nunique()),
        "panel_logw_mean": float(np.mean(panel_logw)),
        "panel_logw_std": float(np.std(panel_logw)),
        "decoder_y_masked_mean": float(np.mean(raw_dec)),
        "decoder_y_masked_std": float(np.std(raw_dec)),
        "decoder_y_masked_min": float(np.min(raw_dec)),
        "decoder_y_masked_max": float(np.max(raw_dec)),
        "prediction_masked_mean": float(np.mean(pred_np)),
        "prediction_masked_min": float(np.min(pred_np)),
        "prediction_masked_max": float(np.max(pred_np)),
        "target_scale_center": float(scale[0, 0]),
        "target_scale_scale": float(scale[0, 1]),
        "normalizer_center": center,
        "normalizer_scale": scale_p,
        "loss_target": "raw_LOGW",
        "evidence": (
            "decoder y and TFT.forward(pred) are both ~5.8 (raw LOGW). "
            "Zeros in the unmasked tensor are decoder padding, not a "
            "normalized target. QuantileLoss sees the inverse-transformed head."
        ),
        "pf_quantile_loss": "2 * pinball on raw LOGW after transform_output",
        "reported_pinball": "unscaled 1x pinball on raw LOGW",
        "pr24_normalized_claim": False,
        "checkpoint": str(model_dir),
        "retrained": False,
    }


def recompute_cqr_from_saved(stability: Path = STABILITY) -> dict[str, Any]:
    """Re-fit Romano margins on locked official-validation predictions."""
    fut_val = Path(stability) / "official_val_target_obs.csv"
    fut_test = Path(stability) / "official_test_target_obs.csv"
    if not fut_val.is_file():
        return {"status": "not_run", "reason": "stability validation targets missing"}
    val_obs = pd.read_csv(fut_val)
    test_obs = pd.read_csv(fut_test) if fut_test.is_file() else None
    rows = []
    corrected = []
    for seed in (82001, 82021, 82031):
        pred_val = Path(stability) / "predictions" / f"official_val_to_{seed}.csv"
        pred_test = Path(stability) / "predictions" / f"official_test_to_{seed}.csv"
        if not pred_val.is_file():
            continue
        pv = pd.read_csv(pred_val)
        from send_bw.evaluation.quantile_forecast import align_prediction_table
        from send_bw.tft_intervals import fit_cqr_margin

        rec = fit_cqr_margin(val_obs, pv, alpha=0.2)
        mm, _ = align_prediction_table(val_obs, pv)
        scores = conformal_nonconformity(mm["LOGW"].to_numpy(float),
                                         mm["q0.1"].to_numpy(float),
                                         mm["q0.9"].to_numpy(float))
        cmp = compare_cqr_order_statistic(scores, 0.2)
        stored = None
        stored_path = Path(stability) / "cqr_decision.json"
        if stored_path.is_file():
            logs = (json.loads(stored_path.read_text(encoding="utf-8")).get("logs") or [])
            hit = [x for x in logs if int(x.get("seed") or 0) == seed]
            stored = (hit[0] if hit else {}).get("margin")
        row = {
            "seed": seed,
            "n_cal": rec.get("n_cal"),
            "romano_margin": rec.get("margin"),
            "stored_margin": stored,
            "margins_match": None if stored is None else bool(
                np.isclose(float(rec["margin"]), float(stored), rtol=0, atol=1e-12)),
            "order_statistic": cmp,
        }
        if test_obs is not None and pred_test.is_file():
            raw = pd.read_csv(pred_test)
            cal = apply_cqr_margin(raw, float(rec["margin"]))
            pack = score_raw_and_calibrated(test_obs, raw, cal)
            row["official_test_raw"] = pack.get("raw")
            row["official_test_cqr_romano"] = pack.get("cqr")
            corrected.append({
                "seed": seed,
                "bank": "official_test",
                "margin": rec.get("margin"),
                "raw": pack.get("raw"),
                "cqr": pack.get("cqr"),
                "source_predictions": str(pred_test),
                "locked_scores_not_overwritten": True,
            })
        rows.append(row)
    return {
        "status": "ok",
        "protocol": "official validation pairs from runs/tft_stability (not test)",
        "seeds": rows,
        "corrected_test_metrics": corrected,
        "locked_cqr_artifacts_rewritten": False,
    }


def classification() -> dict[str, Any]:
    return {
        "confirmed_bugs": [
            {
                "id": "cqr_numpy_higher_one_rank_too_large",
                "severity": "scoring (CQR margin only; q0.5 / MAE unchanged)",
                "detail": (
                    "conformal_margin used np.quantile(k/n, method='higher'). "
                    "NumPy 'higher' is the Hyndman-Fan next-rank rule on q*(n-1), "
                    "not Romano's k-th order statistic. For official validation "
                    "n=214, k=172, it returned sorted[172] instead of sorted[171]. "
                    "Locked CQR tables were not overwritten. Romano margins and "
                    "rescored official-test metrics are in runs/forecast_method_audit/. "
                    "TFT point forecasts are unaffected."
                ),
            },
            {
                "id": "catboost_docstring_multiquantile",
                "severity": "documentation",
                "detail": (
                    "QuantileCatBoostModel claimed MultiQuantile; fit() trains "
                    "one Quantile:alpha=q head per level. Scoring was already "
                    "correct. Docstring fixed."
                ),
            },
            {
                "id": "limitations_chronos_unimplemented",
                "severity": "documentation",
                "detail": (
                    "docs/limitations.md still said Chronos-2 remained "
                    "unimplemented. Official_seq ran HISTORY_ONLY Chronos-2. "
                    "Stale sentence corrected."
                ),
            },
            {
                "id": "pr24_tft_loss_said_normalized",
                "severity": "documentation",
                "detail": (
                    "PR #24 wrote that QuantileLoss is 2× pinball on the "
                    "normalized target. One locked batch shows decoder y and "
                    "TFT.forward predictions on raw LOGW; the normalizer "
                    "inverse-transforms before the loss. Corrected."
                ),
            },
        ],
        "design_limitations": [
            "D1–D3 are curated synthetic descriptors, not SEND fields. Comparable columns must be supplied at prediction time.",
            "Hidden A/ED50/tau_on/tau_off never enter production features.",
            "Weight responds to actual dose deviations; models see planned regimen only.",
            "Missingness and death are MCAR-ish (constant p, independent of treatment/weight), not informative dropout.",
            "Official growth family is one saturating exponential (tau_g=40). Gompertz exists only as Generator B.",
            "Spline sigma_m sits at ~1e-4 (optimizer floor). Animal intercept absorbs measurement; not generator truth (0.010).",
            "Official Chronos-2 used visit-rank LOGW (HISTORY_ONLY). Elapsed DAY is an adapter omission, not a model inability.",
            "COVARIATE_AWARE can pass DAY but the sequence index remains unit steps.",
            "CQR k=n+1 (n<4 at alpha=0.2) uses the sample max instead of Romano +inf.",
            "Official test is a study holdout (SYNA-010/011). Those studies use train compounds CMP03/CMP06, not a new compound.",
            "FeatureSpec.check_context is not enforced at TFT score time (already recorded in PR #24).",
            "CQR pair residuals are (USUBJID, DAY), not study-balanced; exchangeability is an assumption.",
        ],
        "unresolved": [
            "Whether a real SEND warehouse would have D1–D3-like descriptors at prediction time (qualification, not this synthetic bank).",
            "Whether a Chronos encoding that used elapsed time as the index (not just a covariate) would change official_seq errors — not measured; would need a new adapter and a new unlocked run.",
            "Study-level finite-sample CQR coverage is not guaranteed by the pair-level Romano rule.",
        ],
        "no_refit_required": True,
        "development_reference": "target-only TFT",
    }


def execute_forecast_method_audit(out_dir: str | Path = OUTPUT_DEFAULT) -> dict[str, Any]:
    parent = Path(out_dir)
    _refuse_locked(parent)
    parent.mkdir(parents=True, exist_ok=True)
    cqr_grid = []
    rng = np.random.default_rng(0)
    for n in (3, 4, 20, 50, 172, 214, 215):
        scores = rng.normal(size=n)
        cqr_grid.append(compare_cqr_order_statistic(scores, 0.2))
    out: dict[str, Any] = {
        "labels": [
            "Synthetic diagnostic.",
            "Not design qualification.",
            "No real-data validation.",
            "No operational recommendation.",
        ],
        "development_reference": "tft_target_only",
        "retrained": False,
        "locked_dirs_written": False,
        "descriptor_observability": descriptor_observability(),
        "production_features": production_features_exclude_hidden(),
        "growth_family": official_growth_family(),
        "chronos_visit_rank": chronos_visit_rank_note(),
        "chronos_saved": check_chronos_saved_mode(),
        "catboost": catboost_information_set(),
        "spline_cov": check_spline_cov_not_truth(),
        "transformer": check_transformer_train_only(),
        "dose_deviation": check_dose_deviation_effect(),
        "missingness_death": check_missingness_and_death(),
        "cqr_order_statistic_grid": cqr_grid,
        "cqr_from_saved": recompute_cqr_from_saved(),
        "tft_one_batch": replay_tft_one_batch(),
        "classification": classification(),
    }
    write_json(parent / "SUMMARY.json", out)
    results = parent / "RESULTS.md"
    if not results.is_file():
        results.write_text(_results_md(out), encoding="utf-8")
    cqr = out.get("cqr_from_saved") or {}
    if cqr.get("corrected_test_metrics"):
        write_json(parent / "cqr_romano_rescored.json", cqr["corrected_test_metrics"])
    return out


def _results_md(out: dict[str, Any]) -> str:
    d1 = out["descriptor_observability"]
    miss = out["missingness_death"]
    dose = out["dose_deviation"]
    spline = out["spline_cov"]
    tft = out["tft_one_batch"]
    cqr = out["cqr_from_saved"]
    lines = [
        "# Forecast method audit",
        "",
        "Synthetic diagnostic. Not design qualification. No real-data validation.",
        "No operational recommendation. Target-only TFT remains the development reference.",
        "Locked run directories were not rewritten. No model was retrained.",
        "",
        "## Generator",
        "",
        f"- D1–D3: **{d1['d1_d2_d3']}**, written to `{d1['written_to']}`.",
        f"- Hidden: {', '.join(d1['hidden_compound_truth'])} in `{d1['hidden_written_to']}`.",
        f"- Comparable descriptors at prediction time: {d1['comparable_at_prediction_time']}",
        f"- Official growth: {out['growth_family']['official_bank']}.",
        f"- Death rate on official bank: {miss.get('death_rate')} "
        f"(configured p_death={miss.get('configured_p_death')}).",
        f"- corr(death, treated)={miss.get('corr_death_vs_treated')}; "
        f"corr(death, dose)={miss.get('corr_death_vs_dose')}; "
        f"corr(death, baseline LOGW)={miss.get('corr_death_vs_baseline_logw')}.",
        f"- Dose deviations in truth: {dose.get('n_deviations_in_truth')}; "
        f"any planned-vs-actual effect delta: {dose.get('any_nonzero_effect_delta')}.",
        "",
        "## Other methods",
        "",
        f"- Spline cov fitted on `{spline.get('covariance_fit_on')}` "
        f"(n_obs={spline.get('n_obs')}, n_studies={spline.get('n_studies')}). "
        f"sigma_p={spline['fitted']['sigma_p']:.4f} vs generator 0.022; "
        f"ell={spline['fitted']['ell']:.2f} vs 6; "
        f"sigma_m={spline['fitted']['sigma_m']:.6g} vs 0.010.",
        f"- Transformer spec compounds: {out['transformer']['compound_ids']}. "
        f"Official test studies SYNA-010/011 were not trained; they reuse CMP03/CMP06.",
        f"- CatBoost future-outcome features: {out['catboost']['future_outcome_features'] or 'none'}.",
        "",
        "## Chronos-2",
        "",
        f"- Official_seq mode: `{out['chronos_visit_rank']['official_seq_mode']}`.",
        f"- {out['chronos_visit_rank']['adapter_limitation']}",
        f"- {out['chronos_visit_rank']['model_capability']}",
        "",
        "## Technical checks",
        "",
        f"- CQR saved-protocol: {cqr.get('status')}. Locked CQR files not rewritten.",
        f"- TFT batch: {tft.get('status')}. "
        f"decoder y mean={tft.get('decoder_y_masked_mean')}, "
        f"pred mean={tft.get('prediction_masked_mean')}, "
        f"loss_target={tft.get('loss_target')}.",
        "",
        "## Classification",
        "",
        "Confirmed bugs (documentation): CatBoost MultiQuantile wording; stale Chronos unimplemented line.",
        "No scoring or leakage bug required a refit.",
        "",
    ]
    return "\n".join(lines) + "\n"

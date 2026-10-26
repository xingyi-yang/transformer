"""Matched history-length forecasting on the frozen rich_hist_v1 bank.

Does not regenerate or retune trajectories. Does not overwrite locked runs.
Synthetic methodological benchmark. Not real-data validation or operational
qualification.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.features import FeatureSpec, check_no_leakage
from send_bw.data.harmonize import harmonize
from send_bw.data.metadata import load_mapping
from send_bw.data.qc import CohortConfig, select_cohort
from send_bw.data.sources import LocalSendSource
from send_bw.data.splits import (
    assert_disjoint,
    freeze_splits,
    load_frozen_splits,
    make_study_splits_calibrated,
)
from send_bw.models.base import ModelData, study_balanced_weights
from send_bw.models.spline import SplineModel, ridge, wrmse
from send_bw.rich_hist import LABEL, LOCKED_DIRS
from send_bw.rich_hist import OUTPUT_DEFAULT as BANK_DEFAULT
from send_bw.tft_early_hist_bench import load_yaml
from send_bw.tft_examples import gaussian_quantiles, score_aligned_quantiles
from send_bw.utils.artifacts import file_hash, git_revision, read_json, write_json
from send_bw.v3_official_seq import collect_versions

OUTPUT_DEFAULT = "runs/rich_hist_cmp"
BANK_GENERATE_SEED = 91021
BANK_SPLIT_SEED = 92031
OFFICIAL_GENERATE_SEED = 71001
LARGER_GENERATE_SEED = 81011
COMMON_Q = [0.1, 0.25, 0.5, 0.75, 0.9]
FULL_ENCODER_CAP = 40
VAL_ENCODER_DAY = 14.0
PREDICT_CLIP_DAY = 1000.0


class HistCmpError(ValueError):
    pass


def _refuse_locked(parent: Path) -> None:
    resolved = parent.resolve()
    blocked = {Path(p).resolve() for p in LOCKED_DIRS + (BANK_DEFAULT,)}
    if resolved in blocked:
        raise HistCmpError("refusing to write into a preserved run directory")
    if resolved.name.startswith("v3_") and "runs" in resolved.parts:
        raise HistCmpError("refusing to write a new locked-style runs/v3_* directory")
    if resolved.name in {
        "rich_hist_v1", "tft_larger_bank", "tft_longer_ctx", "tft_primary",
        "tft_stability", "tft_study_cv", "forecast_method_audit",
    }:
        raise HistCmpError("refusing to overwrite a locked or source-bank directory")


def _parse_list(val: Any) -> list[float]:
    if isinstance(val, list):
        return [float(x) for x in val]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    return [float(x) for x in ast.literal_eval(str(val))]


def trajectory_family_id(studyid: str, pair_id: Any) -> str:
    if pair_id is not None and str(pair_id) not in {"", "nan", "None"}:
        return str(pair_id)
    return str(studyid)


def assign_families(scenarios: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    sc = scenarios.copy()
    sc["trajectory_family_id"] = [
        trajectory_family_id(str(r["STUDYID"]), r.get("PAIR_ID")) for r in sc.to_dict("records")
    ]
    out = sc.merge(splits, on="STUDYID", how="left")
    return out


def verify_family_partitions(assigned: pd.DataFrame) -> dict[str, Any]:
    """Every sibling family must sit in exactly one partition. Study-ID is not enough."""
    target = assigned.loc[assigned["SPLIT"].isin(["train", "validation", "calibration", "test"])]
    n_split = target.groupby("trajectory_family_id")["SPLIT"].nunique()
    crossing = sorted(n_split[n_split > 1].index.astype(str).tolist())
    by_split = target.groupby("SPLIT")["trajectory_family_id"].nunique().to_dict()
    n_fam = int(target["trajectory_family_id"].nunique())
    n_studies = int(target["STUDYID"].nunique())
    return {
        "n_target_studies": n_studies,
        "n_independent_families": n_fam,
        "families_by_split": {str(k): int(v) for k, v in by_split.items()},
        "n_crossing_families": len(crossing),
        "crossing_families": crossing,
        "passed": len(crossing) == 0,
        "note": (
            "trajectory_family_id is PAIR_ID for dense/sparse siblings that share "
            "a latent path, otherwise STUDYID. Study-ID separation alone is not "
            "independence when siblings share realizations."
        ),
        "label": LABEL,
    }


def corrected_family_splits(assigned: pd.DataFrame, *, seed: int,
                            fractions: dict[str, float] | None = None) -> pd.DataFrame:
    """Whole-family re-split. Writes never touch the bank manifest."""
    target = assigned.loc[assigned["SPLIT"].isin(
        ["train", "validation", "calibration", "test"])].copy()
    ctx = target.drop_duplicates("STUDYID")[["STUDYID"]].copy()
    ctx["STUDYID"] = ctx["STUDYID"].astype(str)
    ctx["STUDY_START"] = 0
    groups = {
        str(r["STUDYID"]): str(r["trajectory_family_id"])
        for r in target.drop_duplicates("STUDYID").to_dict("records")
    }
    fr = dict(fractions or {"validation": 0.15, "calibration": 0.15, "test": 0.20})
    out = make_study_splits_calibrated(ctx, fr, seed, groups=groups)
    other = assigned.loc[~assigned["SPLIT"].isin(
        ["train", "validation", "calibration", "test"]), ["STUDYID", "SPLIT"]].copy()
    if len(other):
        extra = other.drop_duplicates("STUDYID")
        extra["STUDYID"] = extra["STUDYID"].astype(str)
        extra["GROUP"] = extra["STUDYID"]
        out = pd.concat([out, extra], ignore_index=True)
    return out.reset_index(drop=True)


def reconcile_inventory(animals: pd.DataFrame, examples: pd.DataFrame,
                        exclusions: pd.DataFrame, splits: pd.DataFrame) -> dict[str, Any]:
    cohort = set(animals["USUBJID"].astype(str))
    win = set(examples["USUBJID"].astype(str))
    missing = sorted(cohort - win)
    an = animals.drop_duplicates("USUBJID").set_index(animals.drop_duplicates("USUBJID")["USUBJID"].astype(str))
    sp = splits.set_index(splits["STUDYID"].astype(str))
    rows = []
    for u in missing:
        sid = str(an.loc[u, "STUDYID"]) if u in an.index else None
        reasons = []
        if len(exclusions):
            hit = exclusions.loc[exclusions["USUBJID"].astype(str) == u]
            reasons = sorted({str(x) for x in hit["exclude_reason"].dropna()})
        rows.append({
            "USUBJID": u, "STUDYID": sid,
            "SPLIT": None if sid is None or sid not in sp.index else str(sp.loc[sid, "SPLIT"]),
            "DISPOSITION": None if u not in an.index or "DISPOSITION" not in an.columns else str(an.loc[u, "DISPOSITION"]),
            "exclude_reasons": reasons,
        })
    miss = pd.DataFrame(rows)
    by_split = miss.groupby("SPLIT").size().to_dict() if len(miss) and "SPLIT" in miss else {}
    by_reason = {}
    for rec in rows:
        key = ";".join(rec["exclude_reasons"]) or "unlisted"
        by_reason[key] = by_reason.get(key, 0) + 1
    return {
        "n_cohort_animals": int(len(cohort)),
        "n_window_animals": int(len(win)),
        "n_missing_from_windows": int(len(missing)),
        "missing_by_split": {str(k): int(v) for k, v in by_split.items()},
        "missing_by_reason": {str(k): int(v) for k, v in by_reason.items()},
        "missing_animals": rows,
        "note": (
            "Cohort count includes every eligible RAT animal. Window count keeps "
            "animals with at least one observed target after an eligible origin. "
            "Animals already dead or removed at every origin do not enter windows."
        ),
        "label": LABEL,
    }


def encoder_consumption(examples: pd.DataFrame, *, history_kind: str,
                        max_encoder_length: int | None) -> dict[str, Any]:
    """What a model would consume from the frozen window table."""
    rows = []
    n_silent = 0
    for r in examples.to_dict("records"):
        avail = _parse_list(r.get("encoder_days"))
        n_avail = int(r.get("n_encoder_available") or len(avail))
        if history_kind == "latest":
            used = avail[-1:]
        elif history_kind == "short":
            used = avail[-2:]
        elif max_encoder_length is None:
            used = avail
        else:
            used = avail[-int(max_encoder_length):]
        n_used = len(used)
        n_excl = n_avail - n_used
        if history_kind == "full" and n_excl:
            n_silent += 1
        rows.append({
            "USUBJID": r["USUBJID"], "STUDYID": r["STUDYID"],
            "origin": float(r["origin"]), "target_day": float(r["target_day"]),
            "last_observed_day": float(r["last_observed_day"]),
            "n_encoder_available": n_avail, "n_encoder_consumed": n_used,
            "n_encoder_excluded": n_excl,
            "encoder_days_consumed": used,
            "encoder_span": float(used[-1] - used[0]) if len(used) > 1 else 0.0,
            "lead_from_origin": float(r["lead_from_origin"]),
            "lead_from_last_observation": float(r["lead_from_last_observation"]),
            "history_kind": history_kind,
            "max_encoder_length": max_encoder_length,
        })
    tab = pd.DataFrame(rows)
    return {
        "history_kind": history_kind,
        "max_encoder_length": max_encoder_length,
        "n_pairs": int(len(tab)),
        "mean_available": float(tab["n_encoder_available"].mean()) if len(tab) else None,
        "mean_consumed": float(tab["n_encoder_consumed"].mean()) if len(tab) else None,
        "mean_excluded": float(tab["n_encoder_excluded"].mean()) if len(tab) else None,
        "n_truncated_pairs": int((tab["n_encoder_excluded"] > 0).sum()) if len(tab) else 0,
        "silent_full_truncation_pairs": int(n_silent),
        "full_history_keeps_all": bool(history_kind != "full" or n_silent == 0),
        "table": tab,
        "label": LABEL,
    }


def frozen_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    ev = cfg.get("history_cmp") or {}
    tft = ((cfg.get("models") or {}).get("optional") or {}).get("tft") or {}
    return {
        "version": "rich_hist_cmp_v1",
        "label": LABEL,
        "research_question": (
            "Does additional observed individual history improve future "
            "body-weight prediction under this synthetic generator?"
        ),
        "not": [
            "real-data validation",
            "operational qualification",
            "proof that history is useless in real studies if the result is null",
        ],
        "bank": {
            "source": BANK_DEFAULT,
            "generate_seed": BANK_GENERATE_SEED,
            "split_seed": BANK_SPLIT_SEED,
            "regenerated": False,
            "tuned_for_model_win": False,
        },
        "origins_by_duration": ev.get("origins_by_duration") or {"28": [7, 14], "90": [14, 28, 42, 56]},
        "horizons_calendar_days": ev.get("horizons") or [7, 14],
        "eligible_targets": (
            "Frozen rich_hist_v1 full-history example rows. The same "
            "(USUBJID, origin, target_day) keys are scored for every variant."
        ),
        "baseline_weight": {
            "population_only": "none (no ZB_C, no individual LOGW)",
            "latest_short_full": "identical observed BASELINE_LOGW / ZB_C",
            "isolates": "longitudinal history, not baseline access",
        },
        "variants": {
            "population_only": "planned biology/regimen/descriptors + time; no individual weights",
            "latest_weight": "TFT, max_encoder_length=1, last observed visit + common covariates + baseline",
            "short_history": "TFT, max_encoder_length=2, last two visits + common covariates + baseline",
            "full_history": "TFT, max_encoder_length=40, all eligible visits through origin + common covariates + baseline",
            "last_value": "benchmark: last encoder LOGW, no covariates",
            "spline": "benchmark: existing spline on Y, uses baseline + planned features, no visit history",
        },
        "matched_architecture_B_D": {
            "hidden_size": tft.get("hidden_size", 16),
            "dropout": tft.get("dropout", 0.1),
            "lr": tft.get("lr", 0.01),
            "max_epochs": tft.get("max_epochs", 16),
            "patience": tft.get("patience", 6),
            "batch_size": tft.get("batch_size", 32),
            "window_balance": tft.get("window_balance", "study"),
            "target_normalizer": tft.get("target_normalizer", "species_or_global"),
            "separately_trained": True,
            "not_inference_only_truncation": True,
        },
        "training_validation_filter": (
            f"TFT validation windows use encoder DAY<={VAL_ENCODER_DAY} and "
            f"decoder DAY>{VAL_ENCODER_DAY}. Predict clip is raised to "
            f"{PREDICT_CLIP_DAY} so origin-specific history is not recut at day 14."
        ),
        "full_encoder_cap": FULL_ENCODER_CAP,
        "max_available_encoder_visits_in_bank": 34,
        "seeds": {
            "generate_seed": BANK_GENERATE_SEED,
            "split_seed": BANK_SPLIT_SEED,
            "train_seed": int(cfg.get("train_seed") or cfg.get("seed")),
        },
        "splits": {
            "train": "fit",
            "validation": "model selection (val loss / val RMSE / val MAE)",
            "calibration": "interval CQR margin only; not checkpoint selection",
            "test": "final evaluation only; not used to tune",
        },
        "model_selection_rule": (
            "Lowest validation loss among the candidate's own validation metric. "
            "Test errors are not inspected for selection."
        ),
        "metrics": {
            "primary": "study-balanced MAE on LOGW",
            "also": [
                "family-balanced MAE (independent trajectory families)",
                "schedule-balanced MAE (unweighted mean of schedule-level study-balanced MAE)",
                "by origin, calendar horizon, schedule, duration, available-history bin",
                "per-study MAE",
                "80% interval coverage and width (TFT native; CQR experimental)",
                "consumed encoder length and truncation rate",
            ],
        },
        "aggregation": {
            "study_balanced": "mean |e| within study, then unweighted mean of studies",
            "family_balanced": "mean |e| within trajectory_family_id, then unweighted mean of families",
            "schedule_balanced": "study-balanced MAE within schedule, then unweighted mean of schedules",
            "dense_not_overweighted": (
                "Dense studies have more target rows; study-balanced and "
                "schedule-balanced prevent extra row count from dominating."
            ),
        },
        "conformal": (
            "CQR uses pair-level residuals on calibration. Animals contribute "
            "multiple pairs. This does not establish standard conformal coverage."
        ),
        "loss_scale": "resolved by a one-batch trace of the pinned TFT after fit",
        "tuned_from_test": False,
    }


def family_balanced_mae(err: np.ndarray, families: np.ndarray) -> dict[str, float]:
    by = pd.DataFrame({"e": np.abs(err), "f": families}).groupby("f")["e"].mean()
    return {
        "family_balanced": float(by.mean()) if len(by) else float("nan"),
        "n_families": int(by.shape[0]),
        "n_obs": int(len(err)),
    }


def schedule_balanced_mae(err: np.ndarray, studies: np.ndarray, schedules: np.ndarray) -> dict[str, Any]:
    df = pd.DataFrame({"e": np.abs(err), "s": studies, "sch": schedules})
    per = df.groupby(["sch", "s"])["e"].mean().groupby("sch").mean()
    return {
        "schedule_balanced": float(per.mean()) if len(per) else float("nan"),
        "by_schedule": {str(k): float(v) for k, v in per.items()},
        "n_schedules": int(per.shape[0]),
    }


def score_matched(examples: pd.DataFrame, pred: pd.DataFrame, *,
                  families: pd.DataFrame, representation: str) -> dict[str, Any]:
    """Score one row per (USUBJID, origin, target_day). USUBJID+DAY is not unique."""
    fut = examples.rename(columns={"target_day": "DAY", "target_logw": "LOGW"}).copy()
    fut["DAY"] = fut["DAY"].astype(float)
    fut["USUBJID"] = fut["USUBJID"].astype(str)
    fut["origin"] = fut["origin"].astype(float)
    if "LEAD_DAYS" not in fut.columns:
        fut["LEAD_DAYS"] = fut["lead_from_origin"]
    pr = pred.copy()
    pr["USUBJID"] = pr["USUBJID"].astype(str)
    pr["DAY"] = pr["DAY"].astype(float)
    keys = ["USUBJID", "DAY"]
    if "origin" in pr.columns:
        pr["origin"] = pr["origin"].astype(float)
        keys = ["USUBJID", "origin", "DAY"]
    pr = pr.drop_duplicates(keys, keep="first")
    # score_aligned_quantiles keys on USUBJID+DAY; make origin-specific rows unique.
    fut_s = fut.copy()
    pr_s = pr.copy()
    if "origin" in pr.columns:
        fut_s["USUBJID"] = fut_s["USUBJID"] + "::" + fut_s["origin"].map(lambda x: f"{float(x):g}")
        pr_s["USUBJID"] = pr_s["USUBJID"] + "::" + pr_s["origin"].map(lambda x: f"{float(x):g}")
    block = score_aligned_quantiles(fut_s, pr_s, representation=representation)
    mm = examples.copy()
    mm["USUBJID"] = mm["USUBJID"].astype(str)
    mm["DAY"] = mm["target_day"].astype(float)
    mm["origin"] = mm["origin"].astype(float)
    hit = mm.merge(pr, on=keys, how="inner", suffixes=("", "_pred"))
    if hit.empty:
        block["family_balanced_mae"] = None
        block["schedule_balanced_mae"] = None
        return block
    point = hit["q0.5"].to_numpy(float) if "q0.5" in hit.columns else hit["MEAN"].to_numpy(float)
    y = hit["target_logw"].to_numpy(float)
    err = point - y
    fam_map = (families.assign(STUDYID=lambda d: d["STUDYID"].astype(str))
               .drop_duplicates("STUDYID")
               .set_index("STUDYID")["trajectory_family_id"])
    fam = hit["STUDYID"].astype(str).map(fam_map).fillna(hit["STUDYID"].astype(str)).to_numpy()
    sch = hit["SCHEDULE"].to_numpy() if "SCHEDULE" in hit.columns else np.array(["unknown"] * len(hit))
    block["family_balanced_mae"] = family_balanced_mae(err, fam)
    block["schedule_balanced_mae"] = schedule_balanced_mae(err, hit["STUDYID"].to_numpy(), sch)
    block["n_matched_pairs"] = int(len(hit))
    block["by_origin"] = _group_mae(hit, err, "origin")
    block["by_horizon"] = _group_mae(hit, err, "horizon")
    block["by_duration"] = _group_mae(hit, err, "duration_days")
    if "SCHEDULE" in hit.columns:
        block["by_schedule"] = _group_mae(hit, err, "SCHEDULE")
    if "n_encoder_available" in hit.columns:
        bins = pd.cut(hit["n_encoder_available"], bins=[0, 2, 4, 8, 16, 64],
                      labels=["1-2", "3-4", "5-8", "9-16", "17+"])
        block["by_available_history"] = _group_mae(hit.assign(_bin=bins), err, "_bin")
    per_study = []
    for sid, g in hit.groupby(hit["STUDYID"].astype(str)):
        e = (g["q0.5"] if "q0.5" in g.columns else g["MEAN"]).to_numpy(float) - g["target_logw"].to_numpy(float)
        per_study.append({
            "STUDYID": sid,
            "trajectory_family_id": str(fam_map.get(sid, sid)),
            "n": int(len(g)),
            "mae_log": float(np.mean(np.abs(e))),
        })
    block["per_study"] = per_study
    block["label"] = LABEL
    return block


def _group_mae(hit: pd.DataFrame, err: np.ndarray, col: str) -> list[dict[str, Any]]:
    work = hit.reset_index(drop=True)
    work["_e"] = np.abs(err)
    out = []
    for key, g in work.groupby(col, sort=True, dropna=False):
        by = g.groupby(g["STUDYID"].astype(str))["_e"].mean()
        out.append({
            "key": None if pd.isna(key) else (float(key) if isinstance(key, (int, float, np.floating)) else str(key)),
            "n": int(len(g)),
            "n_studies": int(by.shape[0]),
            "mae_log_study_balanced": float(by.mean()) if len(by) else float("nan"),
            "mae_log_pooled": float(g["_e"].mean()),
        })
    return out


def paired_delta(a: dict[str, Any], b: dict[str, Any], *, name_a: str, name_b: str) -> dict[str, Any]:
    pa = {r["STUDYID"]: r["mae_log"] for r in a.get("per_study") or []}
    pb = {r["STUDYID"]: r["mae_log"] for r in b.get("per_study") or []}
    keys = sorted(set(pa) & set(pb))
    deltas = [pa[k] - pb[k] for k in keys]
    return {
        "name_a": name_a, "name_b": name_b,
        "n_studies": len(keys),
        "mean_delta_a_minus_b": float(np.mean(deltas)) if deltas else None,
        "n_studies_a_better": int(sum(d < 0 for d in deltas)),
        "n_studies_b_better": int(sum(d > 0 for d in deltas)),
        "definition": "positive => a worse than b (higher study MAE)",
        "label": LABEL,
    }


class PopulationOnlyModel:
    """Ridge on planned features predicting LOGW. No individual weight features."""

    name = "population_only"

    def __init__(self) -> None:
        self.spec: FeatureSpec | None = None
        self.compound_ref = pd.DataFrame()
        self.cols: list[str] = []
        self.beta: np.ndarray | None = None
        self.history: dict[str, Any] = {}

    def _X(self, animals: pd.DataFrame, obs: pd.DataFrame, spec: FeatureSpec,
           cref: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        F = spec.observation_features(obs, animals, cref)
        drop = {"USUBJID", "STUDYID", "DAY", "ZB_C"}
        cols = [c for c in F.columns if c not in drop]
        self.cols = cols
        return F[cols].to_numpy(float), obs["LOGW"].to_numpy(float), study_balanced_weights(obs["STUDYID"].to_numpy())

    def fit(self, train: ModelData, validation: ModelData, config: dict[str, Any]) -> dict[str, Any]:
        self.spec, self.compound_ref = train.spec, train.compound_ref
        X, y, w = self._X(train.animals, train.obs, train.spec, train.compound_ref)
        Xv, yv, wv = self._X(validation.animals, validation.obs, train.spec, train.compound_ref)
        grid = config.get("ridge_alphas", [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0])
        scores = {a: wrmse(yv, Xv @ ridge(X, y, w, a), wv) for a in grid}
        alpha = min(scores, key=scores.get)
        self.beta = ridge(X, y, w, alpha)
        self.history = {
            "alpha": alpha,
            "validation_rmse_by_alpha": {str(k): float(v) for k, v in scores.items()},
            "validation_rmse": float(scores[alpha]),
            "dropped_individual_weight_features": ["ZB_C", "LOGW history"],
            "label": LABEL,
        }
        return self.history

    def predict_logw(self, animals: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
        days = np.asarray(days, float)
        obs = pd.DataFrame({
            "STUDYID": np.repeat(animals["STUDYID"].to_numpy(), len(days)),
            "USUBJID": np.repeat(animals["USUBJID"].to_numpy(), len(days)),
            "DAY": np.tile(days, len(animals)),
        })
        F = self.spec.observation_features(obs, animals.reset_index(drop=True), self.compound_ref)
        X = F[self.cols].to_numpy(float)
        hat = X @ self.beta
        out = obs[["USUBJID", "STUDYID", "DAY"]].copy()
        out["MEAN"] = hat
        out["q0.5"] = hat
        for q in COMMON_Q:
            if q != 0.5:
                out[f"q{q}"] = hat
        return out


def _load_bank(bank: Path, mapping_path: str) -> dict[str, Any]:
    syn = bank / "synthetic"
    mapping = load_mapping(mapping_path)
    src = LocalSendSource(syn / "send", metadata_dir=syn / "metadata",
                          domain_aliases=mapping.domain_aliases)
    tables, _ = harmonize(src, mapping)
    cohort = select_cohort(tables, CohortConfig(include_pretreatment=True, species="RAT"))
    scenarios = pd.read_csv(bank / "study_scenarios.csv")
    splits = load_frozen_splits(bank / "splits" / "target_splits")
    examples = pd.read_csv(bank / "windows" / "full_examples.csv")
    exclusions = pd.read_csv(bank / "windows" / "full_exclusions.csv")
    return {
        "tables": tables, "cohort": cohort, "scenarios": scenarios, "splits": splits,
        "examples": examples, "exclusions": exclusions,
        "compound_ref": tables["compound_reference"],
    }


def _attach_schedule(examples: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    meta = scenarios[["STUDYID", "SCHEDULE", "DURATION_DAYS", "PAIR_ID"]].drop_duplicates("STUDYID")
    out = examples.merge(meta, on="STUDYID", how="left")
    if "duration_days" not in out.columns and "DURATION_DAYS" in out.columns:
        out["duration_days"] = out["DURATION_DAYS"]
    return out


def _split_ids(splits: pd.DataFrame, name: str) -> list[str]:
    return splits.loc[splits["SPLIT"] == name, "STUDYID"].astype(str).tolist()


def _model_data(cohort, spec: FeatureSpec, cref: pd.DataFrame, ids: list[str]) -> ModelData:
    a = cohort.animals[cohort.animals["STUDYID"].astype(str).isin(set(ids))].reset_index(drop=True)
    o = cohort.obs[cohort.obs["STUDYID"].astype(str).isin(set(ids))].reset_index(drop=True)
    return ModelData(a, o, cref, spec)


def _history_table(examples: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    for r in examples.to_dict("records"):
        days = _parse_list(r.get("encoder_days"))
        logw = _parse_list(r.get("encoder_logw"))
        if kind == "latest":
            days, logw = days[-1:], logw[-1:]
        elif kind == "short":
            days, logw = days[-2:], logw[-2:]
        for d, z in zip(days, logw, strict=True):
            rows.append({"USUBJID": r["USUBJID"], "STUDYID": r["STUDYID"],
                         "DAY": float(d), "LOGW": float(z), "origin": float(r["origin"])})
    return pd.DataFrame(rows).drop_duplicates(["USUBJID", "DAY", "origin"])


def _predict_last_value(examples: pd.DataFrame) -> pd.DataFrame:
    """Last encoder LOGW at this origin. Not the animal's latest visit across origins."""
    recs = []
    for r in examples.to_dict("records"):
        logw = _parse_list(r.get("encoder_logw"))
        if not logw:
            continue
        val = float(logw[-1])
        row = {
            "USUBJID": str(r["USUBJID"]),
            "STUDYID": r["STUDYID"],
            "DAY": float(r["target_day"]),
            "origin": float(r["origin"]),
            "MEAN": val,
            "point": val,
            "point_definition": "last_observed_encoder_LOGW_at_origin",
        }
        for q in COMMON_Q:
            row[f"q{q}"] = val
        recs.append(row)
    return pd.DataFrame(recs)


def _predict_pop(model: PopulationOnlyModel, examples: pd.DataFrame, animals: pd.DataFrame) -> pd.DataFrame:
    recs = []
    ctx = animals.drop_duplicates("USUBJID").set_index(animals.drop_duplicates("USUBJID")["USUBJID"].astype(str))
    for r in examples.to_dict("records"):
        u = str(r["USUBJID"])
        if u not in ctx.index:
            continue
        pred = model.predict_logw(ctx.loc[[u]].reset_index(drop=True), np.array([float(r["target_day"])]))
        pred = pred.copy()
        pred["origin"] = float(r["origin"])
        recs.append(pred)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def _predict_spline(model: SplineModel, examples: pd.DataFrame, animals: pd.DataFrame) -> pd.DataFrame:
    recs = []
    ctx = animals.drop_duplicates("USUBJID").set_index(animals.drop_duplicates("USUBJID")["USUBJID"].astype(str))
    for r in examples.to_dict("records"):
        u = str(r["USUBJID"])
        if u not in ctx.index:
            continue
        a = ctx.loc[[u]].reset_index(drop=True)
        t = np.array([float(r["target_day"])])
        try:
            marg = model.predict_marginals(a, t)
            pred = gaussian_quantiles(marg, quantiles=COMMON_Q)
        except Exception:  # noqa: BLE001
            mu = model.predict_mean(a, t)
            pred = pd.DataFrame({"USUBJID": [u], "DAY": t, "q0.5": mu.ravel(), "MEAN": mu.ravel()})
            for q in COMMON_Q:
                pred[f"q{q}"] = mu.ravel()
        pred["STUDYID"] = r["STUDYID"]
        pred["origin"] = float(r["origin"])
        recs.append(pred)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def _predict_tft(model, examples: pd.DataFrame, animals: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, list]:
    """One Lightning session per origin. History stays origin-specific per animal."""
    model.encoder_window_day = PREDICT_CLIP_DAY
    animals = animals.copy()
    animals["USUBJID"] = animals["USUBJID"].astype(str)
    ctx_all = animals.drop_duplicates("USUBJID").set_index(
        animals.drop_duplicates("USUBJID")["USUBJID"].astype(str))
    recs = []
    fails: list = []
    for origin, g in examples.groupby("origin", sort=False):
        hist_rows: list[dict[str, Any]] = []
        ctx_rows: list[pd.DataFrame] = []
        seen: set[str] = set()
        target_days = sorted({float(x) for x in g["target_day"]})
        for r in g.to_dict("records"):
            u = str(r["USUBJID"])
            if u not in ctx_all.index:
                fails.append({"USUBJID": u, "reason": "animal_missing_from_context"})
                continue
            if u in seen:
                continue
            seen.add(u)
            days = _parse_list(r.get("encoder_days"))
            logw = _parse_list(r.get("encoder_logw"))
            if kind == "latest":
                days, logw = days[-1:], logw[-1:]
            elif kind == "short":
                days, logw = days[-2:], logw[-2:]
            if not days:
                fails.append({"USUBJID": u, "reason": "empty_encoder"})
                continue
            ctx_rows.append(ctx_all.loc[[u]].reset_index(drop=True))
            for d, z in zip(days, logw, strict=True):
                hist_rows.append({"USUBJID": u, "STUDYID": r["STUDYID"],
                                  "DAY": float(d), "LOGW": float(z)})
        if not ctx_rows:
            continue
        ctx = pd.concat(ctx_rows, ignore_index=True)
        hist = pd.DataFrame(hist_rows)
        try:
            pred = model.predict_quantiles(ctx, np.asarray(target_days, float), history=hist)
        except Exception as e:  # noqa: BLE001
            fails.append({"origin": float(origin), "reason": f"{type(e).__name__}: {e}"})
            continue
        if pred is None or pred.empty:
            fails.append({"origin": float(origin), "reason": "empty_prediction"})
            continue
        pred = pred.copy()
        pred["USUBJID"] = pred["USUBJID"].astype(str)
        sid = (g.assign(USUBJID=g["USUBJID"].astype(str))
               .drop_duplicates("USUBJID")
               .set_index("USUBJID")["STUDYID"])
        pred["STUDYID"] = pred["USUBJID"].map(sid)
        pred["origin"] = float(origin)
        recs.append(pred)
    return (pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()), fails


def _cqr_margin(cal_ex: pd.DataFrame, cal_pred: pd.DataFrame, alpha: float = 0.2) -> dict[str, Any]:
    mm = cal_ex.copy()
    mm["USUBJID"] = mm["USUBJID"].astype(str)
    mm["DAY"] = mm["target_day"].astype(float)
    mm["origin"] = mm["origin"].astype(float)
    pr = cal_pred.copy()
    pr["USUBJID"] = pr["USUBJID"].astype(str)
    pr["DAY"] = pr["DAY"].astype(float)
    keys = ["USUBJID", "DAY"]
    if "origin" in pr.columns:
        pr["origin"] = pr["origin"].astype(float)
        keys = ["USUBJID", "origin", "DAY"]
    hit = mm.merge(pr, on=keys, how="inner")
    if hit.empty or "q0.1" not in hit.columns:
        return {"status": "skipped", "reason": "no calibration quantile rows"}
    y = hit["target_logw"].to_numpy(float)
    scores = np.maximum(hit["q0.1"].to_numpy(float) - y, y - hit["q0.9"].to_numpy(float))
    n = int(len(scores))
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    margin = float(np.max(scores)) if k >= n else float(np.sort(scores)[k - 1])
    return {
        "status": "ok", "alpha": alpha, "n": n, "k": k, "margin": margin,
        "rule": "Romano k=ceil((n+1)(1-a)); pair-level; not study-balanced",
        "conformal_claim": False,
        "label": LABEL,
    }


def _apply_cqr(pred: pd.DataFrame, margin: float) -> pd.DataFrame:
    out = pred.copy()
    if "q0.1" in out.columns:
        out["q0.1"] = out["q0.1"] - margin
    if "q0.9" in out.columns:
        out["q0.9"] = out["q0.9"] + margin
    return out


def _loss_trace(model, train: ModelData) -> dict[str, Any]:
    try:
        from pytorch_forecasting import TimeSeriesDataSet

        from send_bw.models.tft import _enough_visits, build_observed_panel
    except Exception as e:  # noqa: BLE001
        return {"status": "not_run", "reason": str(e)}
    if model.net is None or model.dataset_parameters is None:
        return {"status": "not_run", "reason": "model not fitted"}
    panel = _enough_visits(build_observed_panel(train), int(model.min_encoder_length) + 1)
    uids = list(panel["USUBJID"].astype(str).unique()[:8])
    slim = panel.loc[panel["USUBJID"].astype(str).isin(uids)].reset_index(drop=True)
    ds = TimeSeriesDataSet.from_parameters(model.dataset_parameters, slim, predict=False)
    dl = ds.to_dataloader(train=False, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    x, y = batch[0], batch[1]
    import torch

    y_t = y[0] if isinstance(y, (list, tuple)) else y
    y_t = y_t.detach().cpu()
    dec_len = x["decoder_lengths"].detach().cpu()
    mask = torch.arange(y_t.size(1))[None, :] < dec_len[:, None]
    raw_dec = y_t[mask].numpy()
    model.net.eval()
    with torch.no_grad():
        pred = model.net(x)["prediction"].detach().cpu()
    pred_np = pred[mask].numpy()
    return {
        "status": "ok",
        "decoder_y_masked_mean": float(np.mean(raw_dec)),
        "prediction_masked_mean": float(np.mean(pred_np)),
        "panel_logw_mean": float(panel["LOGW"].mean()),
        "loss_target": "raw_LOGW",
        "evidence": (
            "decoder y and TFT.forward prediction sit on raw LOGW (~5–6), "
            "not a standardized residual. QuantileLoss is 2× pinball after "
            "transform_output."
        ),
        "pf_quantile_loss": "2 * pinball on raw LOGW after transform_output",
        "label": LABEL,
    }


def _compact(block: dict[str, Any]) -> dict[str, Any]:
    out = {k: block.get(k) for k in (
        "status", "n_scored", "n_matched_pairs", "n_failures",
        "mae_log", "rmse_log", "coverage80", "interval_width80",
        "family_balanced_mae", "schedule_balanced_mae",
        "by_origin", "by_horizon", "by_schedule", "by_duration",
        "by_available_history", "per_study",
    )}
    out["label"] = LABEL
    return out


def execute_rich_hist_cmp(config_path: str | Path, force: bool = False,
                          smoke: bool = False, score_only: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path) if not isinstance(config_path, dict) else config_path
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    _refuse_locked(parent)
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise HistCmpError("real-data access is off")
    bank = Path(cfg.get("bank_dir") or BANK_DEFAULT)
    if not (bank / "windows" / "full_examples.csv").is_file():
        raise HistCmpError(f"frozen bank missing at {bank}")
    train_seed = int(cfg.get("train_seed") or cfg.get("seed"))
    if train_seed in {OFFICIAL_GENERATE_SEED, LARGER_GENERATE_SEED, BANK_GENERATE_SEED, BANK_SPLIT_SEED}:
        raise HistCmpError("train_seed collides with a locked or bank seed")
    if parent.exists() and not force and (parent / "SUMMARY.json").is_file() and not smoke and not score_only:
        return read_json(parent / "SUMMARY.json")
    parent.mkdir(parents=True, exist_ok=True)
    for sub in ("preflight", "splits", "preprocess", "models", "predictions"):
        (parent / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    packed = _load_bank(bank, cfg["mapping"])
    assigned = assign_families(packed["scenarios"], packed["splits"])
    fam_check = verify_family_partitions(assigned)
    write_json(parent / "preflight" / "family_independence.json",
               {k: v for k, v in fam_check.items() if k != "table"})
    assigned.to_csv(parent / "preflight" / "trajectory_families.csv", index=False)
    split_source = "bank_preserved"
    if not fam_check["passed"]:
        # Stop before fitting the leaked split. Keep the bank manifest.
        packed["splits"].to_csv(parent / "preflight" / "preserved_bank_splits.csv", index=False)
        write_json(parent / "preflight" / "crossing_stop.json", {
            "stopped_before_fit_on_leaked_split": True,
            "crossing_families": fam_check["crossing_families"],
            "bank_manifest_rewritten": False,
            "action": "corrected_family_grouped_split",
            "label": LABEL,
        })
        corr = corrected_family_splits(assigned, seed=BANK_SPLIT_SEED)
        freeze_splits(corr, parent / "splits" / "target_splits")
        packed["splits"] = corr
        assigned = assign_families(packed["scenarios"], packed["splits"])
        fam_check = verify_family_partitions(assigned)
        fam_check["corrected_after_crossing"] = True
        write_json(parent / "preflight" / "family_independence_corrected.json",
                   {k: v for k, v in fam_check.items() if k != "table"})
        if not fam_check["passed"]:
            raise HistCmpError(
                "corrected family split still crosses partitions; refusing to fit. "
                f"crossing={fam_check['crossing_families']}")
        split_source = "corrected_family_grouped"
    else:
        # Existing manifest is valid at the family level. Copy, do not rewrite bank.
        freeze_splits(packed["splits"], parent / "splits" / "target_splits")
    assert_disjoint(packed["splits"])
    packed["splits"].to_csv(parent / "splits" / "splits_with_families.csv", index=False)

    inv = reconcile_inventory(packed["cohort"].animals, packed["examples"],
                              packed["exclusions"], packed["splits"])
    write_json(parent / "preflight" / "inventory.json",
               {k: v for k, v in inv.items() if k != "missing_animals"})
    pd.DataFrame(inv["missing_animals"]).to_csv(parent / "preflight" / "window_exclusions.csv", index=False)

    examples = _attach_schedule(packed["examples"], packed["scenarios"])
    fam_map = (assigned.drop_duplicates("STUDYID")
               .assign(STUDYID=lambda d: d["STUDYID"].astype(str))
               .set_index("STUDYID")["trajectory_family_id"])
    examples["trajectory_family_id"] = examples["STUDYID"].astype(str).map(fam_map)
    enc_full = encoder_consumption(examples, history_kind="full", max_encoder_length=FULL_ENCODER_CAP)
    enc_short = encoder_consumption(examples, history_kind="short", max_encoder_length=2)
    enc_latest = encoder_consumption(examples, history_kind="latest", max_encoder_length=1)
    enc_legacy = encoder_consumption(examples, history_kind="full", max_encoder_length=8)
    for name, pack in (("full", enc_full), ("short", enc_short), ("latest", enc_latest),
                       ("legacy_cap8", enc_legacy)):
        pack["table"].to_csv(parent / "preflight" / f"encoder_{name}.csv", index=False)
        write_json(parent / "preflight" / f"encoder_{name}.json",
                   {k: v for k, v in pack.items() if k != "table"})
    if not enc_full["full_history_keeps_all"]:
        raise HistCmpError("full-history would still drop visits; raise FULL_ENCODER_CAP")

    protocol = frozen_protocol(cfg)
    protocol["split_source"] = split_source
    protocol["family_independence"] = {k: v for k, v in fam_check.items()}
    protocol["inventory"] = {k: v for k, v in inv.items() if k != "missing_animals"}
    protocol["encoder_audit"] = {
        "full": {k: enc_full[k] for k in enc_full if k != "table"},
        "short": {k: enc_short[k] for k in enc_short if k != "table"},
        "latest": {k: enc_latest[k] for k in enc_latest if k != "table"},
        "legacy_cap8_would_truncate_pairs": enc_legacy["n_truncated_pairs"],
        "legacy_cap8_not_used_for_full": True,
    }
    write_json(parent / "PROTOCOL.json", protocol)
    if protocol.get("tuned_from_test") is not False:
        raise HistCmpError("protocol must freeze tuned_from_test=false before training")

    train_ids = _split_ids(packed["splits"], "train")
    val_ids = _split_ids(packed["splits"], "validation")
    cal_ids = _split_ids(packed["splits"], "calibration")
    test_ids = _split_ids(packed["splits"], "test")
    if smoke:
        train_ids, val_ids = train_ids[:3], val_ids[:1]
        cal_ids, test_ids = [], []
    spec = FeatureSpec.fit(
        packed["cohort"].animals[packed["cohort"].animals["STUDYID"].astype(str).isin(set(train_ids))],
        packed["cohort"].obs[packed["cohort"].obs["STUDYID"].astype(str).isin(set(train_ids))],
        packed["compound_ref"], "RAT")
    check_no_leakage(spec.animal_cols + spec.time_cols)
    write_json(parent / "preprocess" / "feature_spec.json", spec.to_dict())
    train = _model_data(packed["cohort"], spec, packed["compound_ref"], train_ids)
    val = _model_data(packed["cohort"], spec, packed["compound_ref"], val_ids)
    animals_all = packed["cohort"].animals

    def _ex_for(ids: list[str]) -> pd.DataFrame:
        return examples[examples["STUDYID"].astype(str).isin(set(ids))].reset_index(drop=True)

    ex_val, ex_cal, ex_test = _ex_for(val_ids), _ex_for(cal_ids), _ex_for(test_ids)
    if smoke:
        ex_val = ex_val.head(40)
        score_split, score_ex = "validation", ex_val
    else:
        score_split, score_ex = "test", ex_test

    tft_base = dict(((cfg.get("models") or {}).get("optional") or {}).get("tft") or {})
    tft_base.setdefault("hidden_size", 16)
    tft_base.setdefault("dropout", 0.1)
    tft_base.setdefault("lr", 0.01)
    tft_base.setdefault("max_epochs", 2 if smoke else 16)
    tft_base.setdefault("patience", 2 if smoke else 6)
    tft_base.setdefault("batch_size", 32)
    tft_base.setdefault("window_balance", "study")
    tft_base.setdefault("target_normalizer", "species_or_global")
    tft_base.setdefault("max_prediction_length", 6)
    tft_base["seed"] = train_seed
    tft_base["encoder_window_day"] = VAL_ENCODER_DAY
    tft_base["quantiles"] = COMMON_Q
    if smoke:
        tft_base["max_epochs"] = 2
        tft_base["patience"] = 2

    results: dict[str, Any] = {}
    preds: dict[str, pd.DataFrame] = {}
    failures: dict[str, Any] = {}

    tft_specs = {
        "latest_weight": {"max_encoder_length": 1, "min_encoder_length": 1, "kind": "latest"},
        "short_history": {"max_encoder_length": 2, "min_encoder_length": 1, "kind": "short"},
        "full_history": {"max_encoder_length": FULL_ENCODER_CAP, "min_encoder_length": 1, "kind": "full"},
    }
    loss_trace = None
    if score_only:
        if smoke:
            raise HistCmpError("score_only is not a smoke path")
        from send_bw.models.tft import TFTEarlyHistoryModel as _T

        pop = PopulationOnlyModel()
        spline = SplineModel.load(parent / "models" / "spline")
        preds["last_value"] = _predict_last_value(score_ex)
        results["last_value"] = score_matched(score_ex, preds["last_value"], families=assigned,
                                              representation="last_value")
        results["last_value"]["uses_individual_history"] = "latest_only"
        results["last_value"]["trained"] = False
        results["last_value"]["rescored"] = True
        # population-only has no serializer; refit on train only (same ridge grid, no test use)
        pop.fit(train, val, {"ridge_alphas": [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]})
        preds["population_only"] = _predict_pop(pop, score_ex, animals_all)
        results["population_only"] = score_matched(score_ex, preds["population_only"], families=assigned,
                                                   representation="population_ridge")
        results["population_only"]["validation_rmse"] = pop.history.get("validation_rmse")
        results["population_only"]["uses_individual_history"] = False
        results["population_only"]["uses_baseline_weight"] = False
        preds["spline"] = _predict_spline(spline, score_ex, animals_all)
        results["spline"] = score_matched(score_ex, preds["spline"], families=assigned,
                                          representation="gaussian")
        results["spline"]["validation_rmse"] = spline.history.get("validation_rmse")
        results["spline"]["uses_baseline_weight"] = True
        results["spline"]["uses_visit_history"] = False
        for name, spec_t in tft_specs.items():
            model = _T.load(parent / "models" / name)
            if name == "full_history":
                loss_trace = _loss_trace(model, train)
            pred, fails = _predict_tft(model, score_ex, animals_all, spec_t["kind"])
            preds[name] = pred
            failures[name] = fails
            block = score_matched(score_ex, pred, families=assigned, representation="tft_quantiles")
            block["max_encoder_length"] = spec_t["max_encoder_length"]
            block["n_failures"] = len(fails)
            block["uses_baseline_weight"] = True
            block["uses_visit_history"] = spec_t["kind"]
            block["separately_trained"] = True
            block["rescored_from_checkpoint"] = True
            results[name] = block
    else:
        # --- last value (no fit)
        preds["last_value"] = _predict_last_value(score_ex)
        results["last_value"] = score_matched(score_ex, preds["last_value"], families=assigned,
                                              representation="last_value")
        results["last_value"]["uses_individual_history"] = "latest_only"
        results["last_value"]["trained"] = False

        # --- population only
        pop = PopulationOnlyModel()
        pop.fit(train, val, {"ridge_alphas": [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]})
        preds["population_only"] = _predict_pop(pop, score_ex, animals_all)
        results["population_only"] = score_matched(score_ex, preds["population_only"], families=assigned,
                                                   representation="population_ridge")
        results["population_only"]["validation_rmse"] = pop.history.get("validation_rmse")
        results["population_only"]["uses_individual_history"] = False
        results["population_only"]["uses_baseline_weight"] = False

        # --- spline benchmark
        spline = SplineModel()
        spline.fit(train, val, {"n_bootstrap": 3 if smoke else 5, "seed": train_seed,
                                "ridge_alphas": [1.0, 10.0, 100.0, 1000.0]})
        preds["spline"] = _predict_spline(spline, score_ex, animals_all)
        results["spline"] = score_matched(score_ex, preds["spline"], families=assigned,
                                          representation="gaussian")
        results["spline"]["validation_rmse"] = spline.history.get("validation_rmse")
        results["spline"]["uses_baseline_weight"] = True
        results["spline"]["uses_visit_history"] = False
        spline.save(parent / "models" / "spline")

        # --- TFT variants (separately trained)
        from send_bw.models.tft import TFTEarlyHistoryModel

        for name, spec_t in tft_specs.items():
            t1 = time.time()
            cfg_t = dict(tft_base)
            cfg_t.update({k: spec_t[k] for k in ("max_encoder_length", "min_encoder_length")})
            model = TFTEarlyHistoryModel()
            hist = model.fit(train, val, cfg_t)
            if name == "full_history":
                loss_trace = _loss_trace(model, train)
            pred, fails = _predict_tft(model, score_ex, animals_all, spec_t["kind"])
            preds[name] = pred
            failures[name] = fails
            block = score_matched(score_ex, pred, families=assigned, representation="tft_quantiles")
            block["seconds"] = time.time() - t1
            block["max_encoder_length"] = spec_t["max_encoder_length"]
            block["n_failures"] = len(fails)
            block["best_val_loss"] = None
            for row in reversed(hist.get("curves") or []):
                if row.get("val_loss") is not None:
                    block["best_val_loss"] = float(row["val_loss"])
                    break
            block["uses_baseline_weight"] = True
            block["uses_visit_history"] = spec_t["kind"]
            block["separately_trained"] = True
            results[name] = block
            model.save(parent / "models" / name)

    # calibration CQR for full-history intervals (not model selection)
    cqr = {"status": "skipped", "reason": "smoke or empty calibration"}
    if not smoke and len(ex_cal) and "full_history" in preds:
        from send_bw.models.tft import TFTEarlyHistoryModel as _T

        full_m = _T.load(parent / "models" / "full_history")
        cal_pred, _ = _predict_tft(full_m, ex_cal, animals_all, "full")
        cqr = _cqr_margin(ex_cal, cal_pred)
        if cqr.get("status") == "ok":
            cqr_pred = _apply_cqr(preds["full_history"], float(cqr["margin"]))
            results["full_history_cqr"] = score_matched(score_ex, cqr_pred, families=assigned,
                                                        representation="tft_cqr")
            results["full_history_cqr"]["experimental"] = True
            results["full_history_cqr"]["conformal_claim"] = False
            cqr_pred.to_csv(parent / "predictions" / "test_full_history_cqr.csv", index=False)

    dest = parent / "predictions"
    dest.mkdir(parents=True, exist_ok=True)
    for name, df in preds.items():
        df.to_csv(dest / f"{'smoke_' if smoke else 'test_'}{name}.csv", index=False)

    score_enc = {
        "full": encoder_consumption(score_ex, history_kind="full", max_encoder_length=FULL_ENCODER_CAP),
        "short": encoder_consumption(score_ex, history_kind="short", max_encoder_length=2),
        "latest": encoder_consumption(score_ex, history_kind="latest", max_encoder_length=1),
    }
    write_json(parent / "preflight" / f"encoder_{score_split}.json", {
        name: {k: pack[k] for k in pack if k != "table"} for name, pack in score_enc.items()
    })

    deltas = {}
    order = ("population_only", "last_value", "spline", "latest_weight", "short_history", "full_history")
    pairs = [("latest_weight", "population_only"), ("short_history", "latest_weight"),
             ("full_history", "short_history"), ("full_history", "latest_weight"),
             ("full_history", "last_value"), ("full_history", "spline")]
    for a, b in pairs:
        if a in results and b in results:
            deltas[f"{a}_minus_{b}"] = paired_delta(results[a], results[b], name_a=a, name_b=b)

    primary = {name: ((results[name].get("mae_log") or {}).get("study_balanced")
                      if isinstance(results[name].get("mae_log"), dict)
                      else results[name].get("mae_log"))
               for name in order if name in results}

    summary = {
        "status": "smoke" if smoke else "ok",
        "label": LABEL,
        "version": "rich_hist_cmp_v1",
        "research_question": protocol["research_question"],
        "output_dir": str(parent),
        "bank_dir": str(bank),
        "smoke": bool(smoke),
        "score_split": score_split,
        "generate_seed": BANK_GENERATE_SEED,
        "split_seed": BANK_SPLIT_SEED,
        "train_seed": train_seed,
        "family_independence": fam_check,
        "split_source": split_source,
        "inventory": {k: v for k, v in inv.items() if k != "missing_animals"},
        "encoder_full_keeps_all": enc_full["full_history_keeps_all"],
        "legacy_cap8_would_truncate_pairs": enc_legacy["n_truncated_pairs"],
        "score_encoder": {name: {k: pack[k] for k in pack if k != "table"}
                          for name, pack in score_enc.items()},
        "n_scored_pairs": int(len(score_ex)),
        "primary_mae_log_study_balanced": primary,
        "results": {k: _compact(v) for k, v in results.items()},
        "paired_deltas": deltas,
        "loss_scale_trace": loss_trace,
        "cqr": cqr,
        "models_trained": [k for k in ("population_only", "spline", "latest_weight",
                                       "short_history", "full_history") if k in results],
        "tuned_from_test": False,
        "score_only": bool(score_only),
        "bank_regenerated": False,
        "locked_dirs_written": False,
        "interpretation": {
            "null_is_not": "proof that history is useless in real studies",
            "positive_is_not": "real-data validation or operational qualification",
        },
        "seconds_total": time.time() - t0,
        "git_revision": git_revision(),
        "versions": collect_versions(),
        "reproduction": (
            f"send-bw rich-hist-cmp --config {config_path}"
            + (" --smoke" if smoke else "")
        ),
        "checksums": {
            "protocol": file_hash(parent / "PROTOCOL.json"),
            "bank_full_windows": file_hash(bank / "windows" / "full_examples.csv"),
        },
    }
    write_json(parent / ("SMOKE.json" if smoke else "SUMMARY.json"), summary)
    (parent / "LABEL.txt").write_text(LABEL + "\n", encoding="utf-8")
    _write_md(parent, summary, smoke=smoke)
    return summary


def _write_md(parent: Path, summary: dict[str, Any], *, smoke: bool) -> None:
    lines = [
        "# rich_hist_cmp " + ("smoke" if smoke else "results"),
        "", LABEL, "",
        f"Question: {summary.get('research_question')}",
        f"Score split: {summary.get('score_split')}  smoke={smoke}",
        f"Families independent: {(summary.get('family_independence') or {}).get('passed')}",
        f"Full-history keeps all visits: {summary.get('encoder_full_keeps_all')}",
        "",
        "## Primary study-balanced MAE on LOGW",
        "",
    ]
    for k, v in (summary.get("primary_mae_log_study_balanced") or {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Paired study deltas (a − b; positive => a worse)", ""]
    for k, v in (summary.get("paired_deltas") or {}).items():
        lines.append(f"- {k}: {v.get('mean_delta_a_minus_b')} "
                     f"(n_studies={v.get('n_studies')})")
    lines += [
        "",
        "A null does not prove history is useless in real studies. "
        "A positive does not validate real data.",
        "", LABEL, "",
    ]
    (parent / ("SMOKE.md" if smoke else "RESULTS.md")).write_text("\n".join(lines), encoding="utf-8")

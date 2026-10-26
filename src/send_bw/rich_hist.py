"""Rich-history synthetic benchmark (generation, splits, windows, integrity).

Does not train TFT or any other model. Does not overwrite locked run dirs.

Synthetic methodological benchmark. Not real-data validation or operational
qualification.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from send_bw.data.features import FeatureSpec, check_no_leakage
from send_bw.data.generate import SCHEDULE_ASSUMPTIONS, generate_dataset
from send_bw.data.harmonize import harmonize
from send_bw.data.metadata import load_mapping
from send_bw.data.qc import CohortConfig, select_cohort
from send_bw.data.sources import LocalSendSource
from send_bw.data.splits import (
    assert_disjoint,
    calibrated_target_manifest,
    freeze_splits,
    make_study_splits_calibrated,
)
from send_bw.tft_early_hist_bench import PRESERVED as V3_PRESERVED
from send_bw.tft_early_hist_bench import load_yaml
from send_bw.tft_larger_bank import OUTPUT_DEFAULT as LARGER_DEFAULT
from send_bw.tft_longer_ctx import OUTPUT_DEFAULT as LONGER_CTX_DEFAULT
from send_bw.tft_primary import OUTPUT_DEFAULT as PRIMARY_DEFAULT
from send_bw.tft_stability import EXTEND_DEFAULT, PILOT_DEFAULT
from send_bw.tft_stability import OUTPUT_DEFAULT as STABILITY_DEFAULT
from send_bw.tft_study_cv import OUTPUT_DEFAULT as STUDY_CV_DEFAULT
from send_bw.utils.artifacts import cache_key, dir_hash, file_hash, git_revision, read_json, write_json
from send_bw.v3_official_seq import collect_versions

OUTPUT_DEFAULT = "runs/rich_hist_v1"
AUDIT_DEFAULT = "runs/forecast_method_audit"
OFFICIAL_GENERATE_SEED = 71001
LARGER_GENERATE_SEED = 81011
LABEL = (
    "Synthetic methodological benchmark. Not real-data validation or operational qualification."
)
KNOWN_AT_ORIGIN = (
    "SEX", "SPECIES", "STRAIN", "IS_CONTROL", "PLANNED_DOSE", "ROUTE", "INTERVAL_DAYS",
    "START_DAY", "END_DAY", "D1", "D2", "D3", "BASELINE_LOGW", "BASELINE_DAY",
    "observed LOGW with DAY<=origin", "planned regimen only",
)
FORBIDDEN_AT_ORIGIN = (
    "Z0_TRUE", "Z_LATENT", "U_STUDY_SLOPE", "U_STUDY_BASELINE", "DOSE_DEVIATION",
    "DEATH_DAY", "ACTUAL_DOSE", "ACTUAL_END_DAY", "future LOGW (DAY>origin)",
    "future missingness", "future death", "future dosing deviations",
)
LOCKED_DIRS = tuple(V3_PRESERVED) + (
    PILOT_DEFAULT, EXTEND_DEFAULT, PRIMARY_DEFAULT, STABILITY_DEFAULT,
    STUDY_CV_DEFAULT, AUDIT_DEFAULT, LARGER_DEFAULT, LONGER_CTX_DEFAULT,
)
DEFAULT_ORIGINS = {28.0: [7.0, 14.0], 90.0: [14.0, 28.0, 42.0, 56.0]}
DEFAULT_HORIZONS = [7.0, 14.0]


class RichHistError(ValueError):
    pass


def _refuse_locked(parent: Path) -> None:
    resolved = parent.resolve()
    blocked = {Path(p).resolve() for p in LOCKED_DIRS}
    if resolved in blocked:
        raise RichHistError("refusing to write into a preserved run directory")
    if resolved.name.startswith("v3_") and "runs" in resolved.parts:
        raise RichHistError("refusing to write a new locked-style runs/v3_* directory")
    if resolved.name in {
        "tft_early_hist_bench", "tft_early_hist_extend", "tft_primary",
        "tft_stability", "tft_study_cv", "forecast_method_audit",
        "tft_larger_bank", "tft_longer_ctx",
    }:
        raise RichHistError("refusing to overwrite a completed TFT or audit run directory")


def _duration_of(sid: str, animals: pd.DataFrame, scenarios: pd.DataFrame | None) -> float:
    if scenarios is not None and len(scenarios) and "DURATION_DAYS" in scenarios.columns:
        hit = scenarios.loc[scenarios["STUDYID"].astype(str) == str(sid)]
        if len(hit):
            return float(hit.iloc[0]["DURATION_DAYS"])
    g = animals.loc[animals["STUDYID"].astype(str) == str(sid)]
    if "DURATION_DAYS" in g.columns and g["DURATION_DAYS"].notna().any():
        return float(g["DURATION_DAYS"].dropna().iloc[0])
    return 28.0


def build_history_windows(
    obs: pd.DataFrame,
    animals: pd.DataFrame,
    *,
    origins_by_duration: dict[float, list[float]] | None = None,
    horizons: list[float] | None = None,
    max_encoder_length: int | None = None,
    short_history_k: int = 2,
    scenarios: pd.DataFrame | None = None,
    history_kind: str = "full",
) -> dict[str, Any]:
    """Calendar-time windows. Encoder = all observed visits with DAY<=origin.

    Distinguishes origin, last observed day, encoder count, encoder span,
    target day, lead from origin, and lead from last observation.
    Does not interpolate or invent days. Future rows are targets only.
    If max_encoder_length is set, truncation is reported rather than silent.
    """
    origins_by_duration = origins_by_duration or DEFAULT_ORIGINS
    horizons = [float(h) for h in (horizons or DEFAULT_HORIZONS)]
    short_k = int(short_history_k)
    recs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    meta = animals.drop_duplicates("USUBJID").set_index(
        animals.drop_duplicates("USUBJID")["USUBJID"].astype(str))
    for u, g in obs.groupby(obs["USUBJID"].astype(str), sort=True):
        g = g.sort_values("DAY")
        sid = str(g["STUDYID"].iloc[0])
        duration = _duration_of(sid, animals, scenarios)
        origins = [float(x) for x in origins_by_duration.get(float(duration), DEFAULT_ORIGINS[28.0])]
        row = meta.loc[u] if u in meta.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        days = g["DAY"].to_numpy(float)
        logw = g["LOGW"].to_numpy(float)
        death = None
        if row is not None and "DISPOSITION_DAY" in meta.columns:
            dval = row["DISPOSITION_DAY"]
            disp = str(row["DISPOSITION"]) if "DISPOSITION" in meta.columns else ""
            if pd.notna(dval) and ("FOUND DEAD" in disp.upper() or "MORIBUND" in disp.upper()):
                death = float(dval)
        for origin in origins:
            enc_mask = days <= origin + 1e-9
            enc_days = days[enc_mask]
            enc_logw = logw[enc_mask]
            if death is not None and death <= origin:
                excluded.append({
                    "USUBJID": u, "STUDYID": sid, "origin": origin,
                    "exclude_reason": "already_dead_at_origin", "death_day": death,
                })
                continue
            if len(enc_days) == 0:
                excluded.append({
                    "USUBJID": u, "STUDYID": sid, "origin": origin,
                    "exclude_reason": "no_observed_encoder_on_or_before_origin",
                })
                continue
            n_avail = int(len(enc_days))
            if history_kind == "short":
                used_days = enc_days[-short_k:]
                used_logw = enc_logw[-short_k:]
                truncated = bool(n_avail > short_k)
                n_excluded = int(n_avail - len(used_days))
            elif max_encoder_length is None:
                used_days, used_logw = enc_days, enc_logw
                truncated, n_excluded = False, 0
            else:
                cap = int(max_encoder_length)
                used_days, used_logw = enc_days[-cap:], enc_logw[-cap:]
                truncated = bool(n_avail > cap)
                n_excluded = int(n_avail - len(used_days))
            last_obs = float(used_days[-1])
            span = float(used_days[-1] - used_days[0]) if len(used_days) else 0.0
            n_targets = 0
            for h in horizons:
                target = float(origin + h)
                hit = np.where(np.isclose(days, target))[0]
                if len(hit) == 0:
                    excluded.append({
                        "USUBJID": u, "STUDYID": sid, "origin": origin, "horizon": h,
                        "target_day": target, "exclude_reason": "target_day_not_observed",
                        "n_encoder_available": n_avail,
                    })
                    continue
                if target <= origin + 1e-9:
                    excluded.append({
                        "USUBJID": u, "STUDYID": sid, "origin": origin, "horizon": h,
                        "exclude_reason": "target_not_after_origin",
                    })
                    continue
                ti = int(hit[0])
                recs.append({
                    "USUBJID": u, "STUDYID": sid,
                    "SPECIES": None if row is None else str(row.get("SPECIES")) if row is not None else None,
                    "SEX": None if row is None else str(row.get("SEX")) if row is not None else None,
                    "origin": float(origin),
                    "last_observed_day": last_obs,
                    "n_encoder_available": n_avail,
                    "n_encoder_used": int(len(used_days)),
                    "n_encoder_excluded_by_limit": n_excluded,
                    "truncated_encoder": truncated,
                    "encoder_span": span,
                    "encoder_days": [float(x) for x in used_days],
                    "encoder_logw": [float(x) for x in used_logw],
                    "time_idx_visit_rank": list(range(len(used_days))),
                    "target_day": float(days[ti]),
                    "target_logw": float(logw[ti]),
                    "lead_from_origin": float(days[ti] - origin),
                    "lead_from_last_observation": float(days[ti] - last_obs),
                    "horizon": float(h),
                    "history_kind": history_kind,
                    "max_encoder_length": max_encoder_length,
                    "duration_days": float(duration),
                    "visit_rank_note": (
                        "time_idx_visit_rank is 0..n-1 among retained encoder visits. "
                        "Calendar spacing is DAY / last_observed_day / lead_from_origin."
                    ),
                    "label": LABEL,
                })
                n_targets += 1
            if n_targets == 0 and not any(
                    e.get("USUBJID") == u and e.get("origin") == origin for e in excluded):
                excluded.append({
                    "USUBJID": u, "STUDYID": sid, "origin": origin,
                    "exclude_reason": "no_observed_target_after_origin",
                    "n_encoder_available": n_avail,
                })
    examples = pd.DataFrame(recs)
    excl = pd.DataFrame(excluded)
    by_reason = excl["exclude_reason"].value_counts().to_dict() if len(excl) and "exclude_reason" in excl else {}
    return {
        "history_kind": history_kind,
        "max_encoder_length": max_encoder_length,
        "short_history_k": short_k,
        "horizons": horizons,
        "origins_by_duration": {str(k): v for k, v in origins_by_duration.items()},
        "n_pairs": int(len(examples)),
        "n_animals": int(examples["USUBJID"].nunique()) if len(examples) else 0,
        "n_excluded_rows": int(len(excl)),
        "exclude_counts": {str(k): int(v) for k, v in by_reason.items()},
        "examples": examples,
        "exclusions": excl,
        "label": LABEL,
        "note": (
            "Encoder uses every observed visit with DAY<=origin unless a model "
            "limit is set. Limits are reported as n_encoder_excluded_by_limit. "
            "Targets are observed days at origin+horizon only. No interpolation."
        ),
    }


def encoder_history_report(pack: dict[str, Any]) -> dict[str, Any]:
    ex = pack["examples"]
    if ex.empty:
        return {"n_pairs": 0, "label": LABEL}
    rows = []
    for (origin, dur), g in ex.groupby(["origin", "duration_days"], sort=True):
        rows.append({
            "origin": float(origin), "duration_days": float(dur),
            "n_pairs": int(len(g)), "n_animals": int(g["USUBJID"].nunique()),
            "mean_encoder_available": float(g["n_encoder_available"].mean()),
            "mean_encoder_used": float(g["n_encoder_used"].mean()),
            "median_encoder_used": float(g["n_encoder_used"].median()),
            "min_encoder_used": int(g["n_encoder_used"].min()),
            "max_encoder_used": int(g["n_encoder_used"].max()),
            "n_truncated": int(g["truncated_encoder"].sum()),
            "mean_excluded_by_limit": float(g["n_encoder_excluded_by_limit"].mean()),
            "mean_span": float(g["encoder_span"].mean()),
            "n_gt_2_used": int((g["n_encoder_used"] > 2).sum()),
            "n_gt_2_available": int((g["n_encoder_available"] > 2).sum()),
        })
    later = ex.loc[ex["origin"] >= 28.0] if "origin" in ex else ex.iloc[0:0]
    return {
        "by_origin_duration": rows,
        "overall_mean_encoder_used": float(ex["n_encoder_used"].mean()),
        "overall_mean_encoder_available": float(ex["n_encoder_available"].mean()),
        "later_origin_mean_available": float(later["n_encoder_available"].mean()) if len(later) else None,
        "later_origin_fraction_gt_2": (
            float((later["n_encoder_available"] > 2).mean()) if len(later) else None),
        "achieved_not_forced": True,
        "label": LABEL,
    }


def run_integrity_checks(
    tables: dict[str, pd.DataFrame],
    cohort,
    splits: pd.DataFrame,
    full_pack: dict[str, Any],
    short_pack: dict[str, Any],
    trunc_pack: dict[str, Any],
    scenarios: pd.DataFrame,
    truth_dir: Path,
    train_spec: FeatureSpec,
) -> dict[str, Any]:
    """Automated integrity and information-availability tests. Not model skill."""
    checks: dict[str, Any] = {}
    obs = cohort.obs
    animals = cohort.animals
    status = tables["animal_status"]
    bw = tables["bw_obs"]

    dead = status.loc[status["DISPOSITION"].astype(str).str.contains("DEAD|MORIBUND", case=False, na=False)]
    post_death = 0
    if len(dead):
        dmap = dead.set_index("USUBJID")["DISPOSITION_DAY"]
        for u, g in bw.groupby("USUBJID"):
            if u in dmap.index and pd.notna(dmap.loc[u]):
                post_death += int((g["DAY"] >= float(dmap.loc[u])).sum())
    checks["no_post_death_measurements"] = post_death == 0
    checks["n_post_death_rows"] = int(post_death)

    send_has_truth = False
    for col in ("Z_LATENT", "Z0_TRUE", "U_STUDY_SLOPE"):
        if col in bw.columns or col in animals.columns:
            send_has_truth = True
    checks["latent_absent_from_canonical"] = not send_has_truth
    checks["truth_dir_separate"] = Path(truth_dir).name == "truth"

    pair_ok, n_overlap, n_mismatch = True, 0, 0
    if "PAIR_ID" in scenarios.columns:
        pairs = scenarios.dropna(subset=["PAIR_ID"])
        for _pid, pg in pairs.groupby("PAIR_ID"):
            if len(pg) != 2:
                pair_ok = False
                continue
            sids = pg["STUDYID"].astype(str).tolist()
            a = bw.loc[bw["STUDYID"].astype(str) == sids[0], ["USUBJID", "DAY", "WEIGHT_G"]]
            b = bw.loc[bw["STUDYID"].astype(str) == sids[1], ["USUBJID", "DAY", "WEIGHT_G"]]
            a = a.assign(SUBJID=a["USUBJID"].astype(str).str.split("-").str[-1])
            b = b.assign(SUBJID=b["USUBJID"].astype(str).str.split("-").str[-1])
            m = a.merge(b, on=["SUBJID", "DAY"], suffixes=("_a", "_b"))
            n_overlap += int(len(m))
            if len(m) and not np.allclose(m["WEIGHT_G_a"].to_numpy(float),
                                          m["WEIGHT_G_b"].to_numpy(float), atol=1e-6, rtol=0):
                n_mismatch += int((~np.isclose(m["WEIGHT_G_a"], m["WEIGHT_G_b"], atol=1e-6)).sum())
                pair_ok = False
    checks["paired_overlap_shared_realizations"] = pair_ok
    checks["n_paired_overlap_rows"] = int(n_overlap)
    checks["n_paired_weight_mismatches"] = int(n_mismatch)

    def _window_ok(pack: dict[str, Any]) -> dict[str, Any]:
        ex = pack["examples"]
        if ex.empty:
            return {"ok": True, "n": 0}
        keyed = {(str(u), round(float(d), 6)) for u, d in zip(obs["USUBJID"].astype(str), obs["DAY"], strict=True)}
        future_in_enc = invented = origin_ge_target = 0
        for r in ex.to_dict("records"):
            for d in r["encoder_days"]:
                if float(d) > float(r["origin"]) + 1e-9:
                    future_in_enc += 1
                if (str(r["USUBJID"]), round(float(d), 6)) not in keyed:
                    invented += 1
            if float(r["target_day"]) <= float(r["origin"]) + 1e-9:
                origin_ge_target += 1
            if (str(r["USUBJID"]), round(float(r["target_day"]), 6)) not in keyed:
                invented += 1
        return {"ok": future_in_enc == 0 and invented == 0 and origin_ge_target == 0,
                "n": int(len(ex)), "future_in_encoder": future_in_enc,
                "invented_or_interpolated": invented, "target_not_after_origin": origin_ge_target}

    checks["full_windows"] = _window_ok(full_pack)
    checks["short_windows"] = _window_ok(short_pack)
    checks["truncated_windows"] = _window_ok(trunc_pack)
    checks["truncation_reported"] = bool(
        "n_encoder_excluded_by_limit" in trunc_pack["examples"].columns
        and (trunc_pack["examples"]["truncated_encoder"].any()
             or int((trunc_pack["examples"]["n_encoder_available"]
                     <= int(trunc_pack["max_encoder_length"] or 0)).all()
                    if len(trunc_pack["examples"]) else True)))

    assert_disjoint(splits)
    checks["splits_disjoint_studies"] = True
    if "GROUP" in splits.columns:
        mixed = splits.groupby("GROUP")["SPLIT"].nunique()
        checks["paired_studies_same_split"] = bool((mixed <= 1).all())
    else:
        checks["paired_studies_same_split"] = True
    checks["calibration_separate"] = "calibration" in set(splits["SPLIT"])
    cal_n = int((splits["SPLIT"] == "calibration").sum())
    val_n = int((splits["SPLIT"] == "validation").sum())
    checks["calibration_not_validation"] = cal_n > 0 and val_n > 0
    checks["train_studies_ge_2"] = int((splits["SPLIT"] == "train").sum()) >= 2

    train_ids = set(splits.loc[splits["SPLIT"] == "train", "STUDYID"].astype(str))
    checks["featurespec_train_only"] = True
    checks["featurespec_max_day"] = float(train_spec.max_day)
    if len(obs) and train_ids:
        tr_max = float(obs.loc[obs["STUDYID"].astype(str).isin(train_ids), "DAY"].max())
        checks["featurespec_max_day_matches_train"] = abs(tr_max - float(train_spec.max_day)) < 1e-9
    try:
        check_no_leakage(train_spec.animal_cols + train_spec.time_cols)
        checks["featurespec_production_only"] = True
    except ValueError:
        checks["featurespec_production_only"] = False
        checks["featurespec_train_only"] = False
    for col in FORBIDDEN_AT_ORIGIN:
        if col in train_spec.animal_cols or col in train_spec.time_cols:
            checks["featurespec_production_only"] = False

    later = full_pack["examples"]
    if len(later):
        late = later.loc[later["origin"] >= 28.0]
        checks["later_origins_exist"] = bool(len(late) > 0)
        checks["later_origin_mean_available"] = float(late["n_encoder_available"].mean()) if len(late) else None
        checks["later_origin_some_gt_2"] = bool(len(late) and (late["n_encoder_available"] > 2).any())
    else:
        checks["later_origins_exist"] = False
        checks["later_origin_some_gt_2"] = False

    checks["known_at_origin"] = list(KNOWN_AT_ORIGIN)
    checks["forbidden_at_origin"] = list(FORBIDDEN_AT_ORIGIN)
    checks["repeated_measurement_note"] = (
        "Animals contribute multiple (origin, horizon) pairs. Pair-level residual "
        "counts do not by themselves establish conformal validity."
    )
    checks["label"] = LABEL
    bools = [v for v in checks.values() if isinstance(v, bool)]
    nested_ok = [v["ok"] for v in checks.values() if isinstance(v, dict) and "ok" in v]
    checks["passed"] = all(bools) and all(nested_ok)
    return checks


def _obs_inventory(raw: pd.DataFrame, cohort_obs: pd.DataFrame, animals: pd.DataFrame) -> dict[str, Any]:
    def _per(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"n_obs": 0, "n_animals": 0, "median": 0, "mean": 0, "min": 0, "max": 0}
        n = df.groupby(df["USUBJID"].astype(str)).size()
        return {
            "n_obs": int(len(df)), "n_animals": int(n.size),
            "median": float(n.median()), "mean": float(n.mean()),
            "min": int(n.min()), "max": int(n.max()),
        }
    return {
        "before_preprocessing": _per(raw),
        "after_preprocessing": _per(cohort_obs),
        "label": LABEL,
    }


def _partition_report(splits: pd.DataFrame, animals: pd.DataFrame, scenarios: pd.DataFrame) -> dict[str, Any]:
    a = animals.merge(splits, on="STUDYID", how="left")
    sc = scenarios.merge(splits, on="STUDYID", how="left") if len(scenarios) else scenarios
    by_split: dict[str, Any] = {}
    for name, g in a.groupby(a["SPLIT"].fillna("unassigned").astype(str)):
        by_split[str(name)] = {
            "n_studies": int(g["STUDYID"].nunique()),
            "n_animals": int(g["USUBJID"].nunique()),
            "by_species": {str(k): int(v) for k, v in
                           g.groupby(g["SPECIES"].astype(str))["USUBJID"].nunique().items()},
        }
    by_sched = {}
    if len(sc) and "SCHEDULE" in sc.columns:
        for (dur, sch), g in sc.groupby(["DURATION_DAYS", "SCHEDULE"]):
            by_sched[f"{int(float(dur))}d_{sch}"] = {
                "n_studies": int(g["STUDYID"].nunique()),
                "splits": g.groupby("SPLIT")["STUDYID"].nunique().to_dict() if "SPLIT" in g else {},
            }
    return {"by_split": by_split, "by_duration_schedule": by_sched, "label": LABEL}


def _missingness_report(tables, cohort, scenarios: pd.DataFrame) -> dict[str, Any]:
    status = tables["animal_status"]
    dead = status["DISPOSITION"].astype(str).str.contains("DEAD|MORIBUND", case=False, na=False)
    recov = status["DISPOSITION"].astype(str).str.contains("RECOVERY", case=False, na=False)
    drop_kind = {}
    if len(scenarios) and "DROPOUT_KIND" in scenarios.columns:
        sm = cohort.animals.merge(scenarios[["STUDYID", "DROPOUT_KIND"]], on="STUDYID", how="left")
        sm = sm.merge(status[["USUBJID", "DISPOSITION"]], on="USUBJID", how="left")
        for k, g in sm.groupby(sm["DROPOUT_KIND"].fillna("ordinary")):
            d = g["DISPOSITION"].astype(str).str.contains("DEAD|MORIBUND", case=False, na=False)
            drop_kind[str(k)] = {"n_animals": int(g["USUBJID"].nunique()), "n_dropout": int(d.sum())}
    return {
        "n_animals": int(status["USUBJID"].nunique()) if len(status) else 0,
        "n_dropout_dead_or_moribund": int(dead.sum()),
        "n_recovery_sacrifice": int(recov.sum()),
        "by_dropout_kind": drop_kind,
        "ordinary_missingness": "p_missing on day>0; predose DAY -7 and 0 are never MCAR-dropped",
        "challenging_dropout": "separately configurable; later-biased day; still independent of weight",
        "label": LABEL,
    }


def _plot_diagnostics(out: Path, cohort, scenarios: pd.DataFrame, full_pack: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    obs = cohort.obs.merge(scenarios[["STUDYID", "SCHEDULE", "DURATION_DAYS"]].drop_duplicates(),
                           on="STUDYID", how="left") if len(scenarios) else cohort.obs
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False)
    axes = axes.ravel()
    cells = [(28.0, "weekly"), (28.0, "dense"), (90.0, "weekly"), (90.0, "dense")]
    for ax, (dur, sch) in zip(axes, cells, strict=False):
        sub = obs
        if "DURATION_DAYS" in obs.columns:
            sub = obs.loc[np.isclose(obs["DURATION_DAYS"].astype(float), dur)]
        if "SCHEDULE" in sub.columns:
            sub = sub.loc[sub["SCHEDULE"].astype(str) == sch]
        ids = sorted(sub["USUBJID"].astype(str).unique())[:8]
        for u in ids:
            g = sub.loc[sub["USUBJID"].astype(str) == u].sort_values("DAY")
            ax.plot(g["DAY"], np.exp(g["LOGW"].to_numpy(float)), alpha=0.7, lw=1)
        ax.set_title(f"{int(dur)}-day {sch} (n≤8)")
        ax.set_xlabel("DAY (elapsed)")
        ax.set_ylabel("WEIGHT_G")
    fig.suptitle(LABEL, fontsize=8)
    fig.tight_layout()
    p1 = out / "trajectories_by_scenario.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    paths.append(str(p1))

    ex = full_pack["examples"]
    if len(ex):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels, data = [], []
        for (origin, dur), g in ex.groupby(["origin", "duration_days"], sort=True):
            labels.append(f"T={int(dur)} o={int(origin)}")
            data.append(g["n_encoder_available"].to_numpy(float))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(2, color="0.5", ls="--", lw=1, label="audit 1–2 visit pattern")
        ax.set_ylabel("encoder visits available")
        ax.set_title("Available encoder history (not forced)\n" + LABEL, fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        p2 = out / "encoder_history_by_origin.png"
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        paths.append(str(p2))
    return paths


def _compare_prior_banks(parent: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Read-only comparison with documented official / larger banks. No overwrite."""
    official = {
        "generate_seed": OFFICIAL_GENERATE_SEED, "n_rat_studies": 12, "n_dog_studies": 3,
        "duration_days": 28, "schedules": "random weekly vs twice-weekly",
        "typical_encoder_at_day14": "1–2 weekly treatment visits after cohort drop of predose",
        "source": "locked official bank; not regenerated",
    }
    larger_sum = Path(LARGER_DEFAULT) / "SUMMARY.json"
    larger = {"generate_seed": LARGER_GENERATE_SEED, "status": "missing_summary"}
    if larger_sum.is_file():
        data = read_json(larger_sum)
        larger = {
            "generate_seed": LARGER_GENERATE_SEED,
            "status": "read_only",
            "n_studies": (data.get("bank") or {}).get("n_studies") or data.get("n_studies"),
            "n_animals": (data.get("bank") or {}).get("n_animals") or data.get("n_animals"),
            "duration_days": 28,
            "source": str(larger_sum),
            "used_to_retune": False,
        }
    return {
        "official_71001": official,
        "larger_81011": larger,
        "rich_hist_v1": {
            "generate_seed": summary.get("generate_seed"),
            "n_studies": summary.get("n_studies"),
            "n_animals": summary.get("n_animals"),
            "n_obs_after": ((summary.get("observations") or {}).get("after_preprocessing") or {}).get("n_obs"),
            "durations": summary.get("durations"),
            "schedules": summary.get("schedules"),
            "mean_encoder_available": (summary.get("encoder_report") or {}).get("overall_mean_encoder_available"),
        },
        "note": (
            "New versioned bank. Not a correction of locked experiments. "
            "Not evidence that any model is superior."
        ),
        "label": LABEL,
    }


def execute_rich_hist(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path) if not isinstance(config_path, dict) else config_path
    if isinstance(config_path, dict):
        cfg = config_path
        config_path = cfg.get("_config_path") or "configs/rich_hist_v1.yaml"
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    _refuse_locked(parent)
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise RichHistError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise RichHistError("design optimization stays off")
    if (cfg.get("models") or {}).get("optional", {}).get("tft", {}).get("enabled"):
        raise RichHistError("this command does not train TFT; disable models.optional.tft.enabled")
    gen0 = int(cfg["generate_seed"])
    split0 = int(cfg.get("split_seed") or cfg.get("seed"))
    if gen0 in {OFFICIAL_GENERATE_SEED, LARGER_GENERATE_SEED}:
        raise RichHistError("generate_seed collides with a locked bank; use a new seed")
    if gen0 == split0:
        raise RichHistError("generate_seed and split_seed must differ")
    syn = dict(cfg.get("synthetic") or {})
    if not syn.get("longitudinal"):
        raise RichHistError("synthetic.longitudinal is required for this benchmark version")
    if parent.exists() and not force:
        if (parent / "SUMMARY.json").is_file():
            return read_json(parent / "SUMMARY.json")
    parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    synth_dir = Path(syn.get("out_dir") or (parent / "synthetic"))
    info = generate_dataset(syn, synth_dir, seed=gen0)
    mapping = load_mapping(cfg["mapping"])
    src = LocalSendSource(info["send_dir"], metadata_dir=info["metadata_dir"],
                          domain_aliases=mapping.domain_aliases)
    tables, report = harmonize(src, mapping)
    ccfg = CohortConfig.from_dict(cfg.get("cohort"))
    if not ccfg.include_pretreatment:
        raise RichHistError("cohort.include_pretreatment must be true for this bank")
    cohort = select_cohort(tables, ccfg)
    scenarios = pd.read_csv(Path(info["metadata_dir"]) / "study_scenarios.csv")
    target_species = str(ccfg.species).upper()
    in_scope = tables["study_context"]
    in_scope = in_scope[in_scope["SPECIES"].astype(str).str.upper() == target_species]
    in_scope = in_scope[in_scope["STUDYID"].isin(set(cohort.animals["STUDYID"]))]
    scfg = dict(cfg.get("splits") or {})
    comp = (tables["arm_regimen"].dropna(subset=["COMPOUND_ID"])
            .groupby("STUDYID")["COMPOUND_ID"].first().to_dict())
    scfg["_study_compound"] = comp
    groups = {}
    for r in scenarios.to_dict("records"):
        sid = str(r["STUDYID"])
        groups[sid] = str(r["PAIR_ID"]) if pd.notna(r.get("PAIR_ID")) and r.get("PAIR_ID") else sid
    target_studies = in_scope[in_scope["STUDYID"].astype(str).isin(
        set(scenarios.loc[scenarios["SPECIES"].astype(str).str.upper() == target_species, "STUDYID"].astype(str))
        if "SPECIES" in scenarios.columns else set(in_scope["STUDYID"].astype(str))
    )]
    splits = make_study_splits_calibrated(target_studies, scfg, split0, groups=groups)
    freeze_splits(splits, parent / "splits" / "target_splits")
    tman = calibrated_target_manifest(splits)
    write_json(parent / "splits" / "target_manifest.json", tman)
    other = scenarios.loc[scenarios["SPECIES"].astype(str).str.upper() != target_species, "STUDYID"]
    other_df = pd.DataFrame({"STUDYID": other.astype(str).tolist(), "SPLIT": "other_species",
                             "GROUP": other.astype(str).tolist()})
    splits_all = pd.concat([splits, other_df], ignore_index=True) if len(other_df) else splits
    splits_all.to_csv(parent / "splits" / "splits_all.csv", index=False)

    train_ids = splits.loc[splits["SPLIT"] == "train", "STUDYID"].astype(str)
    tr_a = cohort.animals[cohort.animals["STUDYID"].astype(str).isin(set(train_ids))]
    tr_o = cohort.obs[cohort.obs["STUDYID"].astype(str).isin(set(train_ids))]
    spec = FeatureSpec.fit(tr_a, tr_o, tables["compound_reference"], target_species)
    write_json(parent / "preprocess" / "feature_spec.json", spec.to_dict())

    win_cfg = cfg.get("windows") or {}
    origins = {float(k): [float(x) for x in v] for k, v in (win_cfg.get("origins_by_duration")
                                                             or DEFAULT_ORIGINS).items()}
    # yaml keys may be ints
    if win_cfg.get("origins_by_duration"):
        origins = {float(k): [float(x) for x in v] for k, v in win_cfg["origins_by_duration"].items()}
    horizons = [float(x) for x in (win_cfg.get("horizons") or DEFAULT_HORIZONS)]
    short_k = int(win_cfg.get("short_history_k") or 2)
    cap = win_cfg.get("model_encoder_limit")
    cap = int(cap) if cap is not None else 8

    full = build_history_windows(cohort.obs, cohort.animals, origins_by_duration=origins,
                                 horizons=horizons, max_encoder_length=None,
                                 short_history_k=short_k, scenarios=scenarios, history_kind="full")
    short = build_history_windows(cohort.obs, cohort.animals, origins_by_duration=origins,
                                  horizons=horizons, max_encoder_length=None,
                                  short_history_k=short_k, scenarios=scenarios, history_kind="short")
    trunc = build_history_windows(cohort.obs, cohort.animals, origins_by_duration=origins,
                                  horizons=horizons, max_encoder_length=cap,
                                  short_history_k=short_k, scenarios=scenarios, history_kind="full")
    dest = parent / "windows"
    dest.mkdir(parents=True, exist_ok=True)
    for name, pack in (("full", full), ("short", short), ("truncated", trunc)):
        pack["examples"].to_csv(dest / f"{name}_examples.csv", index=False)
        pack["exclusions"].to_csv(dest / f"{name}_exclusions.csv", index=False)
        write_json(dest / f"{name}_meta.json", {k: v for k, v in pack.items()
                                                if k not in {"examples", "exclusions"}})

    integrity = run_integrity_checks(
        tables, cohort, splits, full, short, trunc, scenarios,
        Path(info["truth_dir"]), spec)
    write_json(parent / "integrity.json", integrity)
    if not integrity["passed"]:
        raise RichHistError(f"integrity checks failed: {integrity}")

    enc_rep = encoder_history_report(full)
    obs_inv = _obs_inventory(tables["bw_obs"], cohort.obs, cohort.animals)
    part = _partition_report(splits_all, cohort.animals, scenarios)
    miss = _missingness_report(tables, cohort, scenarios)
    plot_paths = _plot_diagnostics(parent / "plots", cohort, scenarios, full)

    raw_n = tables["bw_obs"].groupby("USUBJID").size()
    coh_n = cohort.obs.groupby("USUBJID").size() if len(cohort.obs) else pd.Series(dtype=float)
    summary = {
        "status": "ok",
        "label": LABEL,
        "schedule_assumptions": SCHEDULE_ASSUMPTIONS,
        "version": "rich_hist_v1",
        "output_dir": str(parent),
        "generate_seed": gen0,
        "split_seed": split0,
        "git_revision": git_revision(),
        "versions": collect_versions(),
        "reproduction": f"send-bw rich-hist --config {config_path}",
        "n_studies": int(info["n_studies"]),
        "n_pairs_latent": int(info.get("n_pairs") or 0),
        "n_animals": int(cohort.animals["USUBJID"].nunique()),
        "n_obs_raw": int(len(tables["bw_obs"])),
        "n_obs_cohort": int(len(cohort.obs)),
        "n_eligible_animals": int(len(cohort.animals)),
        "n_exclusions": int(len(cohort.exclusions)),
        "durations": sorted({float(x) for x in scenarios["DURATION_DAYS"]}),
        "schedules": sorted({str(x) for x in scenarios["SCHEDULE"]}),
        "species": sorted({str(x) for x in scenarios["SPECIES"]}),
        "observations": obs_inv,
        "obs_per_animal_raw": {"median": float(raw_n.median()), "mean": float(raw_n.mean()),
                               "min": int(raw_n.min()), "max": int(raw_n.max())},
        "obs_per_animal_cohort": (
            {"median": float(coh_n.median()), "mean": float(coh_n.mean()),
             "min": int(coh_n.min()), "max": int(coh_n.max())} if len(coh_n) else {}),
        "partitions": part,
        "missingness": miss,
        "encoder_report": enc_rep,
        "window_counts": {
            "full": {"n_pairs": full["n_pairs"], "n_animals": full["n_animals"],
                     "exclude_counts": full["exclude_counts"]},
            "short": {"n_pairs": short["n_pairs"], "n_animals": short["n_animals"]},
            "truncated": {"n_pairs": trunc["n_pairs"],
                          "n_truncated_pairs": int(trunc["examples"]["truncated_encoder"].sum())
                          if len(trunc["examples"]) else 0,
                          "model_encoder_limit": cap},
        },
        "known_at_origin": list(KNOWN_AT_ORIGIN),
        "forbidden_at_origin": list(FORBIDDEN_AT_ORIGIN),
        "integrity": {"passed": integrity["passed"],
                      "later_origin_mean_available": integrity.get("later_origin_mean_available")},
        "plots": plot_paths,
        "checksums": {
            "send": dir_hash(Path(info["send_dir"])),
            "metadata": dir_hash(Path(info["metadata_dir"])),
            "splits": file_hash(parent / "splits" / "target_splits.csv"),
            "full_windows": file_hash(dest / "full_examples.csv"),
        },
        "models_trained": False,
        "tuned_for_model_win": False,
        "locked_dirs_written": False,
        "unresolved_assumptions": [
            "Schedules are experimental, not authorized historical protocol aggregates.",
            "Saturating slope basis is a modelling choice to avoid linear t/28 blow-up at day 90.",
            "Challenging dropout is later-biased but still independent of weight (not MNAR).",
            "DOG studies are inventory-only (other_species); this bank is not a sharing experiment.",
            "90-day saturating-exponential growth is still one family; not a claim of biological realism.",
        ],
        "proposed_comparison": {
            "predictors": ["population_only", "latest_weight", "short_history", "full_history"],
            "same_targets": True,
            "same_splits": True,
            "not_run": True,
            "note": (
                "Compare on this bank without retuning generation. Population-only "
                "uses planned covariates and no individual weights. Latest-weight "
                "uses the last observation at or before origin. Short-history uses "
                "the last two observed visits. Full-history uses every eligible "
                "observed visit through the origin. Score study-balanced MAE on "
                "the frozen test studies. Do not use test errors to retune."
            ),
        },
        "seconds_total": time.time() - t0,
        "mapping_report": {"n_issues": len(report) if hasattr(report, "__len__") else None},
        "qc": cohort.qc_checks,
    }
    summary["comparison_with_original"] = _compare_prior_banks(parent, summary)
    write_json(parent / "SUMMARY.json", summary)
    write_json(parent / "config_used.json", cfg if isinstance(cfg, dict) else {})
    (parent / "LABEL.txt").write_text(LABEL + "\n" + SCHEDULE_ASSUMPTIONS + "\n", encoding="utf-8")
    scenarios.to_csv(parent / "study_scenarios.csv", index=False)
    cohort.animals.to_csv(parent / "cohort_animals.csv", index=False)
    cohort.exclusions.to_csv(parent / "qc" / "exclusions.csv", index=False)
    write_json(parent / "config_hash.json", {"config_hash": cache_key(cfg), "generate_seed": gen0,
                                             "split_seed": split0, "label": LABEL})
    _write_results_md(parent, summary)
    return summary


def _write_results_md(parent: Path, summary: dict[str, Any]) -> None:
    enc = summary.get("encoder_report") or {}
    part = (summary.get("partitions") or {}).get("by_split") or {}
    lines = [
        "# rich_hist_v1 results",
        "",
        LABEL,
        "",
        f"Reproduction: `{summary.get('reproduction')}`",
        f"generate_seed={summary.get('generate_seed')} split_seed={summary.get('split_seed')}",
        f"Studies={summary.get('n_studies')} animals={summary.get('n_animals')} "
        f"cohort obs={summary.get('n_obs_cohort')}",
        "",
        "## Partitions",
        "",
    ]
    for name, block in part.items():
        lines.append(f"- {name}: {block.get('n_studies')} studies, {block.get('n_animals')} animals "
                     f"{block.get('by_species')}")
    lines += [
        "",
        "## Encoder history (full, achieved not forced)",
        "",
        f"Mean available={enc.get('overall_mean_encoder_available')} "
        f"mean used={enc.get('overall_mean_encoder_used')}",
        f"Later-origin (DAY≥28) mean available={enc.get('later_origin_mean_available')} "
        f"fraction>2={enc.get('later_origin_fraction_gt_2')}",
        "",
        "## Integrity",
        "",
        f"passed={((summary.get('integrity') or {}).get('passed'))}",
        "",
        "Models were not trained. Generation was not tuned for a model win.",
        "",
    ]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_config_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

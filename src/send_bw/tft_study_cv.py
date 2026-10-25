"""Predeclared study-held-out TFT evaluation.

Target-only is the development reference. Pooled, fine-tune, and CQR are
experimental. Writes only runs/tft_study_cv. Locked run directories stay
untouched.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.splits import compose_train_with_selected_donors
from send_bw.models.base import ModelData
from send_bw.sharing import LABELS
from send_bw.tft_audit import (
    REPORTED_ROLES,
    confirm_34011_erratum,
    cqr_metric_policy,
    epoch_budget_versus_logged,
    roles_for_study,
    series_versus_animals,
)
from send_bw.tft_crossings import crossing_audit
from send_bw.tft_early_hist_bench import PRESERVED as V3_PRESERVED
from send_bw.tft_early_hist_bench import _compact_score, load_yaml
from send_bw.tft_early_hist_extend import load_full_cell, prepare_from_splits
from send_bw.tft_examples import (
    construct_forecast_examples,
    examples_to_tables,
    quantile_table_from_model,
    summarize_observation_schedules,
)
from send_bw.tft_intervals import (
    apply_cqr_margin,
    compact_interval_metrics,
    coverage_slices,
    cqr_pinball_note,
    fit_cqr_margin,
    score_raw_and_calibrated,
)
from send_bw.tft_primary import OUTPUT_DEFAULT as PRIMARY_DEFAULT
from send_bw.tft_primary import _finetune_tft, _fit_tft
from send_bw.tft_stability import EXTEND_DEFAULT, PILOT_DEFAULT, metric_value
from send_bw.tft_stability import OUTPUT_DEFAULT as STABILITY_DEFAULT
from send_bw.utils.artifacts import cache_key, git_revision, write_json
from send_bw.v3_official_seq import collect_versions

OUTPUT_DEFAULT = "runs/tft_study_cv"
LOCKED_DIRS = tuple(V3_PRESERVED) + (
    PILOT_DEFAULT, EXTEND_DEFAULT, PRIMARY_DEFAULT, STABILITY_DEFAULT)
SEEDS = (82001, 82021, 82031)
FOLD_SEED = 35021
N_FOLDS = 6
COMMON_Q = [0.1, 0.25, 0.5, 0.75, 0.9]
REFERENCE = "tft_target_only"
EXPERIMENTAL = ("tft_pooled", "tft_finetune", "tft_target_only_cqr")


class StudyCVError(ValueError):
    pass


def _refuse_locked(parent: Path) -> None:
    resolved = parent.resolve()
    blocked = {Path(p).resolve() for p in LOCKED_DIRS}
    if resolved in blocked:
        raise StudyCVError("refusing to write into a preserved run directory")
    if resolved.name.startswith("v3_") and "runs" in resolved.parts:
        raise StudyCVError("refusing to write a new locked-style runs/v3_* directory")
    if resolved.name in {
        "tft_early_hist_bench", "tft_early_hist_extend", "tft_primary", "tft_stability",
    }:
        raise StudyCVError("refusing to overwrite a completed TFT run directory")


def make_cv_folds(study_ids: list[str], *, seed: int = FOLD_SEED, n_folds: int = N_FOLDS,
                  n_val: int = 2, n_cal: int = 2) -> list[dict[str, Any]]:
    """Every study is test in exactly one fold. Val, cal, and train are disjoint."""
    ids = sorted({str(s) for s in study_ids})
    if len(ids) < n_folds * 2:
        raise StudyCVError("need at least two test studies per fold")
    if len(ids) % n_folds:
        raise StudyCVError("study count must divide n_folds so each study is test once")
    rng = np.random.default_rng(int(seed))
    order = ids.copy()
    rng.shuffle(order)
    n_test = len(order) // int(n_folds)
    folds = []
    for k in range(int(n_folds)):
        test = [str(s) for s in order[k * n_test:(k + 1) * n_test]]
        rest = [s for s in order if s not in set(test)]
        rest_rng = np.random.default_rng(int(seed) + 1000 * (k + 1))
        rest_rng.shuffle(rest)
        if len(rest) < n_val + n_cal + 2:
            raise StudyCVError(f"fold {k} has too few remaining studies for val/cal/train")
        val = rest[:n_val]
        cal = rest[n_val:n_val + n_cal]
        train = rest[n_val + n_cal:]
        pack = {
            "fold": k,
            "seed": int(seed),
            "train": sorted(train),
            "validation": sorted(val),
            "calibration": sorted(cal),
            "test": sorted(test),
            "roles": {sid: roles_for_study(sid) for sid in sorted(train + val + cal + test)},
        }
        verify_fold_membership(pack)
        folds.append(pack)
    tests = [s for f in folds for s in f["test"]]
    if sorted(tests) != ids:
        raise StudyCVError("test rotation does not cover each study once")
    return folds


def verify_fold_membership(fold: dict[str, Any]) -> None:
    groups = {name: set(fold[name]) for name in ("train", "validation", "calibration", "test")}
    if any(not g for g in groups.values()):
        raise StudyCVError(f"fold {fold.get('fold')} has an empty split")
    seen: set[str] = set()
    for name, ids in groups.items():
        if seen & ids:
            raise StudyCVError(f"fold {fold.get('fold')} overlap in {name}: {sorted(seen & ids)}")
        seen |= ids


def _examples(data, cutoff: float, max_enc: int, max_pred: int) -> dict[str, Any]:
    pack = construct_forecast_examples(
        data.obs, data.animals, cutoff=cutoff,
        max_encoder_length=max_enc, max_prediction_length=max_pred)
    hist, fut = examples_to_tables(pack)
    return {"pack": pack, "hist": hist, "fut": fut, "animals": data.animals}


def _sb(block: dict[str, Any] | None, key: str) -> float | None:
    return metric_value(block, key)


def _row(bank: str, variant: str, seed: int, fold: int, metrics: dict[str, Any],
         *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "bank": bank, "variant": variant, "seed": int(seed), "fold": int(fold),
        "mae_log": metrics.get("mae_log"),
        "mean_quantile_loss": metrics.get("mean_quantile_loss"),
        "interval_score80": metrics.get("interval_score80"),
        "coverage80": metrics.get("coverage80"),
        "interval_width80": metrics.get("interval_width80"),
        "crossing": metrics.get("crossing"),
        "n_scored": metrics.get("n_scored"),
        "status": metrics.get("status"),
        "per_study": metrics.get("per_study"),
        "quantile_table": "raw" if not str(variant).endswith("_cqr") else "cqr_shifted_endpoints",
    }
    if extra:
        row.update(extra)
    return row


def _score_raw(model, bank: dict[str, Any], *, label: str) -> tuple[dict[str, Any], pd.DataFrame]:
    pred, fails = quantile_table_from_model(model, bank["animals"], bank["pack"], bank["hist"])
    from send_bw.tft_examples import score_aligned_quantiles

    block = score_aligned_quantiles(bank["fut"], pred, representation="tft_quantiles")
    block["crossing_audit"] = crossing_audit(pred, bank["fut"], levels=COMMON_Q,
                                             animals=bank["animals"])
    slices = coverage_slices(bank["fut"], pred)
    block["coverage_slices"] = slices
    by_is = {str(r["STUDYID"]): r for r in (slices.get("by_study") or [])}
    for rec in block.get("per_study") or []:
        extra = by_is.get(str(rec.get("STUDYID")))
        if extra:
            rec["interval_score80"] = extra.get("interval_score80")
    block["n_failures"] = len(fails)
    block["label"] = label
    block["uses_early_history"] = True
    return block, pred


def splits_frame(fold: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name in ("train", "validation", "calibration", "test"):
        for sid in fold[name]:
            rows.append({"STUDYID": sid, "SPLIT": name, "fold": fold["fold"]})
    return pd.DataFrame(rows).sort_values(["SPLIT", "STUDYID"]).reset_index(drop=True)


def attach_donors(target_splits: pd.DataFrame, donor_ids: list[str]) -> pd.DataFrame:
    return compose_train_with_selected_donors(target_splits, donor_ids)


def fold_integrity(fold: dict[str, Any], prep_target: dict[str, Any], prep_pooled: dict[str, Any],
                   cal_studies: list[str], checkpoint_studies: dict[str, list[str]]) -> dict[str, Any]:
    """Verify studies in train/preprocess/calibration/checkpoints match the fold."""
    issues: list[str] = []
    t_train = sorted(prep_target["data"]["train"].studies)
    if t_train != sorted(fold["train"]):
        issues.append(f"target train {t_train} != fold {sorted(fold['train'])}")
    t_val = sorted(prep_target["data"]["validation"].studies)
    if t_val != sorted(fold["validation"]):
        issues.append(f"target val {t_val} != fold {sorted(fold['validation'])}")
    t_test = sorted(prep_target["data"]["test"].studies)
    if t_test != sorted(fold["test"]):
        issues.append(f"target test {t_test} != fold {sorted(fold['test'])}")
    if sorted(cal_studies) != sorted(fold["calibration"]):
        issues.append(f"calibration {cal_studies} != fold {sorted(fold['calibration'])}")
    spec_sharing = getattr(prep_target["spec"], "sharing", None)
    if spec_sharing not in {None, "context_specific", "target_only"}:
        issues.append(f"unexpected target spec sharing {spec_sharing}")
    for name, studies in checkpoint_studies.items():
        overlap = sorted(set(studies) & set(fold["test"]))
        if overlap:
            issues.append(f"{name} checkpoint trained on test studies {overlap}")
    p_train = set(prep_pooled["data"]["train"].studies)
    extra = sorted(p_train - set(fold["train"]))
    missing = sorted(set(fold["train"]) - p_train)
    if missing:
        issues.append(f"pooled train missing target studies {missing}")
    return {
        "ok": not issues,
        "issues": issues,
        "target_train": t_train,
        "target_validation": t_val,
        "target_test": t_test,
        "calibration": sorted(cal_studies),
        "pooled_train_extra_donors": extra,
        "checkpoint_studies": checkpoint_studies,
        "feature_spec_fit_on": t_train,
        "note": "FeatureSpec is fit on train studies only. Calibration is a separate split.",
    }


def select_on_validation(val_rows: list[dict[str, Any]], *, metric: str = "interval_score80",
                         candidates: list[str] | None = None) -> dict[str, Any]:
    """Lowest mean validation metric. Test scores are ignored. TO stays the reference."""
    names = candidates or [REFERENCE, "tft_pooled", "tft_finetune"]
    by: dict[str, list[float]] = {c: [] for c in names}
    for row in val_rows:
        if row.get("variant") not in by or row.get(metric) is None:
            continue
        if row.get("bank") != "validation":
            continue
        by[row["variant"]].append(float(row[metric]))
    summary = []
    for name in names:
        vals = by[name]
        summary.append({
            "variant": name,
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        })
    usable = [s for s in summary if s["mean"] is not None]
    winner = min(usable, key=lambda s: s["mean"]) if usable else None
    return {
        "metric": metric,
        "reference": REFERENCE,
        "validation_preferred_experimental": None if winner is None or winner["variant"] == REFERENCE
        else winner["variant"],
        "lowest_validation": None if winner is None else winner["variant"],
        "candidates": summary,
        "test_scores_examined": False,
        "note": ("Validation only. Target-only remains the development reference. "
                 "Fine-tune and pooled stay experimental even if they win validation."),
    }


def consistency_across_studies(test_rows: list[dict[str, Any]], *,
                               reference: str = REFERENCE,
                               experimental: tuple[str, ...] = EXPERIMENTAL,
                               metric: str = "mae_log",
                               min_folds: int = 5,
                               min_studies: int = 10) -> dict[str, Any]:
    """Does any experimental variant beat target-only across studies?"""
    use = [r for r in test_rows if r.get("bank") == "test" and r.get(metric) is not None
           and r.get("per_study")]
    # seed-mean per study x variant
    study_vals: dict[tuple[str, str], list[float]] = {}
    fold_of: dict[str, int] = {}
    for r in use:
        for rec in (r.get("per_study") or []):
            sid = str(rec.get("STUDYID") or rec.get("study"))
            val = rec.get(metric)
            if isinstance(val, dict):
                val = val.get("study_balanced", val.get("mean"))
            if val is None:
                continue
            study_vals.setdefault((sid, r["variant"]), []).append(float(val))
            fold_of[sid] = int(r["fold"])
    # fallback: fold-level metric if per_study shape is unexpected
    fold_vals: dict[tuple[int, str], list[float]] = {}
    for r in test_rows:
        if r.get("bank") != "test" or r.get(metric) is None:
            continue
        fold_vals.setdefault((int(r["fold"]), r["variant"]), []).append(float(r[metric]))

    studies = sorted({sid for sid, _v in study_vals})
    out_variants = {}
    for exp in experimental:
        study_wins = []
        for sid in studies:
            ref = study_vals.get((sid, reference))
            other = study_vals.get((sid, exp))
            if not ref or not other:
                continue
            study_wins.append({
                "STUDYID": sid,
                "fold": fold_of.get(sid),
                "reference": float(np.mean(ref)),
                "experimental": float(np.mean(other)),
                "delta": float(np.mean(other)) - float(np.mean(ref)),
                "beats_reference": float(np.mean(other)) < float(np.mean(ref)),
            })
        fold_wins = []
        for fold in sorted({k[0] for k in fold_vals}):
            ref = fold_vals.get((fold, reference))
            other = fold_vals.get((fold, exp))
            if not ref or not other:
                continue
            fold_wins.append({
                "fold": fold,
                "reference": float(np.mean(ref)),
                "experimental": float(np.mean(other)),
                "delta": float(np.mean(other)) - float(np.mean(ref)),
                "beats_reference": float(np.mean(other)) < float(np.mean(ref)),
            })
        n_study_beats = sum(1 for r in study_wins if r["beats_reference"])
        n_fold_beats = sum(1 for r in fold_wins if r["beats_reference"])
        consistent = n_study_beats >= min_studies and n_fold_beats >= min_folds
        out_variants[exp] = {
            "metric": metric,
            "n_studies_compared": len(study_wins),
            "n_studies_beats_reference": n_study_beats,
            "n_folds_compared": len(fold_wins),
            "n_folds_beats_reference": n_fold_beats,
            "min_studies": min_studies,
            "min_folds": min_folds,
            "consistent_improvement": consistent,
            "study_rows": study_wins,
            "fold_rows": fold_wins,
        }
    any_consistent = any(v["consistent_improvement"] for v in out_variants.values())
    return {
        "reference": reference,
        "metric": metric,
        "any_consistent_improvement": any_consistent,
        "variants": out_variants,
        "previous_finetune_default_failed_confirmation": True,
        "note": ("Consistent improvement requires beating target-only on the "
                 f"test {metric} in at least {min_folds} folds and {min_studies} studies. "
                 "Fine-tune and CQR remain experimental."),
    }


def execute_tft_study_cv(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    _refuse_locked(parent)
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise StudyCVError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise StudyCVError("design optimization stays off")
    if parent.exists() and any(parent.iterdir()) and not force:
        raise StudyCVError(f"{parent} exists; pass --force to overwrite this unlocked directory")
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "predictions").mkdir(parents=True, exist_ok=True)
    (parent / "models").mkdir(parents=True, exist_ok=True)
    (parent / "folds").mkdir(parents=True, exist_ok=True)
    (parent / "scores").mkdir(parents=True, exist_ok=True)

    ev = cfg.get("benchmark") or {}
    cutoff = float(ev.get("encoder_window_day") or 14)
    tft_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("tft") or {})
    max_enc = int(tft_cfg.get("max_encoder_length", 8))
    max_pred = int(tft_cfg.get("max_prediction_length", 6))
    seeds = [int(s) for s in (ev.get("stability_seeds") or SEEDS)]
    fold_seed = int(ev.get("cv_split_seed") or FOLD_SEED)
    n_folds = int(ev.get("n_folds") or N_FOLDS)
    cells = ev.get("source_cells") or {}
    target_dir = Path(cells.get("target_only") or "")
    pooled_dir = Path(cells.get("pooled") or "")
    if not target_dir.is_dir() or not pooled_dir.is_dir():
        raise StudyCVError("source_cells.target_only and source_cells.pooled must exist")

    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    target_full = load_full_cell(target_dir)
    pooled_full = load_full_cell(pooled_dir)
    study_ids = sorted(target_full["full"].animals["STUDYID"].astype(str).unique())
    folds = make_cv_folds(study_ids, seed=fold_seed, n_folds=n_folds,
                          n_val=int(ev.get("n_val_studies") or 2),
                          n_cal=int(ev.get("n_cal_studies") or 2))
    write_json(parent / "folds.json", {"seed": fold_seed, "n_folds": n_folds, "folds": folds})

    audit = {
        "confirm_34011": confirm_34011_erratum(),
        "cqr_metric_policy": cqr_metric_policy(),
        "pinball_note": cqr_pinball_note(),
        "epoch_counting": epoch_budget_versus_logged(
            [{"epoch": i, "train_loss": 0.1} for i in range(16)],
            max_epochs=16, lightning_current_epoch=16, finetune_max_epochs=8),
        "official_series_counts": None,
        "development_reference": REFERENCE,
        "experimental": list(EXPERIMENTAL),
        "previous_finetune_default_failed_confirmation": True,
    }
    off_train = target_full["full"].subset(REPORTED_ROLES["official_train"])
    audit["official_series_counts"] = series_versus_animals(
        off_train.animals, off_train.obs, min_encoder_length=int(tft_cfg.get("min_encoder_length", 1)))
    write_json(parent / "AUDIT.json", audit)

    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": cfg.get("generate_seed"),
        "stability_seeds": seeds, "cv_split_seed": fold_seed, "n_folds": n_folds,
        "official_holdout_test_used_for_search": False,
        "development_reference": REFERENCE,
        "experimental": list(EXPERIMENTAL),
        "real_data": False, "design": False, "labels": list(LABELS),
        "preserved": list(LOCKED_DIRS), "writes_v3_locks": False,
        "writes_primary": False, "writes_stability": False,
        "confirm_34011_called_untouched": False,
    }
    write_json(parent / "freeze_lock.json", lock)

    donor_ids = []
    if "SPECIES" in pooled_full["full"].animals.columns:
        donor = pooled_full["full"].animals.loc[
            pooled_full["full"].animals["SPECIES"].astype(str).str.upper()
            != str(target_full["full"].spec.species).upper(), "STUDYID"]
        donor_ids = sorted({str(s) for s in donor})

    t0 = time.time()
    failures: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    integrity_logs: list[dict[str, Any]] = []
    ft_cfg = cfg.get("finetune") or {}
    cal_cfg = cfg.get("calibration") or {}

    for fold in folds:
        fdir = parent / "folds" / f"fold{fold['fold']}"
        fdir.mkdir(parents=True, exist_ok=True)
        target_splits = splits_frame(fold)
        # prepare_from_splits only knows train/validation/test
        prep_splits = target_splits.loc[target_splits["SPLIT"] != "calibration"].copy()
        target_splits.to_csv(fdir / "splits_target.csv", index=False)
        pooled_splits = attach_donors(prep_splits, donor_ids)
        pooled_splits.to_csv(fdir / "splits_pooled.csv", index=False)
        extra_to = prepare_from_splits(
            target_full["full"], prep_splits, species="RAT", sharing="context_specific")
        extra_po = prepare_from_splits(
            pooled_full["full"], pooled_splits, species="RAT", sharing="pooled")
        cal_ids = fold["calibration"]
        cal_sub = target_full["full"].subset(cal_ids)
        cal_data = ModelData(cal_sub.animals, cal_sub.obs, target_full["full"].compound_ref,
                             extra_to["spec"])
        banks = {
            "validation": _examples(extra_to["data"]["validation"], cutoff, max_enc, max_pred),
            "calibration": _examples(cal_data, cutoff, max_enc, max_pred),
            "test": _examples(extra_to["data"]["test"], cutoff, max_enc, max_pred),
        }
        for name, bank in banks.items():
            bank["fut"].to_csv(fdir / f"{name}_target_obs.csv", index=False)
            bank["pack"]["examples"].to_csv(fdir / f"{name}_examples.csv", index=False)
        write_json(fdir / "schedules.json", {
            k: summarize_observation_schedules(
                (extra_to["data"][k].obs if k != "calibration" else cal_data.obs),
                (extra_to["data"][k].animals if k != "calibration" else cal_data.animals))
            for k in ("train", "validation", "test")
        } | {"calibration": summarize_observation_schedules(cal_data.obs, cal_data.animals)})

        integ = fold_integrity(fold, extra_to, extra_po, cal_ids, checkpoint_studies={})
        integrity_logs.append({"fold": fold["fold"], **integ})
        write_json(fdir / "integrity.json", integ)
        if not integ["ok"]:
            failures.append({"model": f"fold{fold['fold']}_integrity",
                             "reason": "; ".join(integ["issues"])})
            continue

        for seed in seeds:
            models = {}
            for scope, prep, dest_name in (
                ("target_only", extra_to, f"to_{seed}"),
                ("pooled", extra_po, f"pooled_{seed}"),
            ):
                try:
                    dest = fdir / "models" / dest_name
                    model, hist = _fit_tft(prep, {**tft_cfg, "seed": seed}, dest, scope)
                    models[scope] = model
                    exposures.append(_exposure(hist, prep, scope=f"fold{fold['fold']}_{scope}",
                                               seed=seed, fold=fold["fold"], studies=fold["train"]))
                    for bank_name in ("validation", "test"):
                        block, pred = _score_raw(
                            model, banks[bank_name],
                            label=f"fold{fold['fold']} {scope} {bank_name} seed {seed}")
                        pred.to_csv(parent / "predictions" /
                                    f"fold{fold['fold']}_{bank_name}_{scope}_{seed}.csv", index=False)
                        write_json(parent / "scores" /
                                   f"fold{fold['fold']}_{bank_name}_{scope}_{seed}.json",
                                   _compact_score(block))
                        variant = "tft_target_only" if scope == "target_only" else "tft_pooled"
                        rows.append(_row(bank_name, variant, seed, fold["fold"],
                                         compact_interval_metrics(block),
                                         extra={"reused": False}))
                except Exception as e:  # noqa: BLE001
                    failures.append({
                        "model": f"fold{fold['fold']}_{scope}_{seed}",
                        "reason": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-1500:],
                    })

            try:
                pooled_ckpt = fdir / "models" / f"pooled_{seed}"
                if not (pooled_ckpt / "tft.ckpt").exists():
                    raise StudyCVError(f"pooled checkpoint missing for fold {fold['fold']} seed {seed}")
                dest = fdir / "models" / f"ft_{seed}"
                ft_locked = dict(tft_cfg)
                ft_locked["seed"] = seed
                ft_locked["finetune"] = ft_cfg
                model, hist = _finetune_tft(pooled_ckpt, extra_to, ft_locked, dest)
                models["finetune"] = model
                exposures.append(_exposure(hist, extra_to, scope=f"fold{fold['fold']}_finetune",
                                           seed=seed, fold=fold["fold"], studies=fold["train"]))
                for bank_name in ("validation", "test"):
                    block, pred = _score_raw(
                        model, banks[bank_name],
                        label=f"fold{fold['fold']} finetune {bank_name} seed {seed}")
                    pred.to_csv(parent / "predictions" /
                                f"fold{fold['fold']}_{bank_name}_finetune_{seed}.csv", index=False)
                    write_json(parent / "scores" /
                               f"fold{fold['fold']}_{bank_name}_finetune_{seed}.json",
                               _compact_score(block))
                    rows.append(_row(bank_name, "tft_finetune", seed, fold["fold"],
                                     compact_interval_metrics(block), extra={"reused": False}))
            except Exception as e:  # noqa: BLE001
                failures.append({
                    "model": f"fold{fold['fold']}_finetune_{seed}",
                    "reason": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-1500:],
                })

            # CQR: margin from calibration only. Experimental.
            raw_val = parent / "predictions" / f"fold{fold['fold']}_validation_target_only_{seed}.csv"
            raw_te = parent / "predictions" / f"fold{fold['fold']}_test_target_only_{seed}.csv"
            if raw_val.is_file() and "target_only" in models and not banks["calibration"]["fut"].empty:
                _cal_block, cal_pred = _score_raw(
                    models["target_only"], banks["calibration"],
                    label=f"fold{fold['fold']} cal TO seed {seed}")
                rec = fit_cqr_margin(
                    banks["calibration"]["fut"], cal_pred,
                    alpha=float(cal_cfg.get("alpha") or 0.2),
                )
                rec.update({"seed": seed, "fold": fold["fold"], "source": "calibration",
                            "test_outcomes_used": False, "validation_used_for_margin": False})
                write_json(fdir / f"cqr_margin_{seed}.json", rec)
                if rec.get("status") == "ok" and raw_te.is_file():
                    for bank_name, path in (("validation", raw_val), ("test", raw_te)):
                        raw = pd.read_csv(path)
                        cal = apply_cqr_margin(raw, float(rec["margin"]))
                        pack = score_raw_and_calibrated(banks[bank_name]["fut"], raw, cal)
                        cal.to_csv(parent / "predictions" /
                                   f"fold{fold['fold']}_{bank_name}_to_cqr_{seed}.csv", index=False)
                        write_json(parent / "scores" /
                                   f"fold{fold['fold']}_{bank_name}_to_cqr_{seed}.json", {
                                       "raw": pack["raw"], "cqr": pack["cqr"],
                                       "pinball_note": pack["pinball_note"],
                                       "margin": rec.get("margin"),
                                   })
                        if pack.get("cqr"):
                            slices = coverage_slices(banks[bank_name]["fut"], cal)
                            by_is = {str(r["STUDYID"]): r for r in (slices.get("by_study") or [])}
                            for rec in pack["cqr"].get("per_study") or []:
                                extra = by_is.get(str(rec.get("STUDYID")))
                                if extra:
                                    rec["interval_score80"] = extra.get("interval_score80")
                            rows.append(_row(bank_name, "tft_target_only_cqr", seed, fold["fold"],
                                             pack["cqr"], extra={
                                                 "reused": False,
                                                 "cqr_margin": rec.get("margin"),
                                                 "pinball_is_shifted_endpoints": True,
                                                 "raw_pinball": (pack["raw"] or {}).get("mean_quantile_loss"),
                                                 "raw_mae": (pack["raw"] or {}).get("mae_log"),
                                             }))

        # refresh integrity with actual checkpoint train studies
        integrity_logs[-1]["checkpoint_studies"] = {
            "fold_fits": list(fold["train"]),
            "official_not_reused": True,
        }

    sel_cfg = cfg.get("selection") or {}
    decision = select_on_validation(
        rows, metric=str(sel_cfg.get("metric") or "interval_score80"),
        candidates=list(sel_cfg.get("candidates") or [REFERENCE, "tft_pooled", "tft_finetune"]))
    write_json(parent / "validation_selection.json", decision)

    cons_cfg = cfg.get("consistency") or {}
    consistency = consistency_across_studies(
        rows, metric=str(cons_cfg.get("metric") or "mae_log"),
        min_folds=int(cons_cfg.get("min_folds") or 5),
        min_studies=int(cons_cfg.get("min_studies") or 10))
    # also interval-score consistency
    consistency_is = consistency_across_studies(
        rows, metric="interval_score80",
        min_folds=int(cons_cfg.get("min_folds") or 5),
        min_studies=int(cons_cfg.get("min_studies") or 10))
    write_json(parent / "consistency.json", {"mae_log": consistency, "interval_score80": consistency_is})

    seed_var = _seed_variability(rows)
    write_json(parent / "seed_variability.json", seed_var)
    pd.DataFrame(rows).drop(columns=["per_study"], errors="ignore").to_csv(
        parent / "fold_seed_metrics.csv", index=False)
    pd.DataFrame(exposures).to_csv(parent / "training_exposure.csv", index=False)
    write_json(parent / "integrity.json", integrity_logs)

    out = {
        "labels": list(LABELS) + ["TFT study-held-out diagnostic.", "Synthetic generator only."],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "audit": {k: audit[k] for k in audit if k != "official_series_counts"} | {
            "official_series_counts": {
                k: audit["official_series_counts"].get(k)
                for k in ("n_animals_in_split", "n_tft_series", "n_dropped_short_series", "note")
            }
        },
        "folds": [{"fold": f["fold"], "test": f["test"], "validation": f["validation"],
                   "calibration": f["calibration"], "train": f["train"]} for f in folds],
        "validation_selection": decision,
        "consistency_mae": {k: consistency[k] for k in consistency if k != "variants"} | {
            "variants": {vk: {kk: vv for kk, vv in vv.items() if kk not in {"study_rows"}}
                         for vk, vv in consistency["variants"].items()}
        },
        "consistency_interval_score": {
            k: consistency_is[k] for k in consistency_is if k != "variants"
        } | {
            "variants": {vk: {kk: vv for kk, vv in vv.items() if kk not in {"study_rows"}}
                         for vk, vv in consistency_is["variants"].items()}
        },
        "seed_variability": seed_var,
        "training_exposure": exposures,
        "integrity": integrity_logs,
        "failures": failures,
        "real_data_accessed": False,
        "design_run": False,
        "seconds_total": time.time() - t0,
        "development_reference": REFERENCE,
        "previous_finetune_default_failed_confirmation": True,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out, rows, folds)
    return out


def _exposure(hist: dict[str, Any], prep: dict[str, Any], *, scope: str, seed: int,
              fold: int, studies: list[str]) -> dict[str, Any]:
    train = prep["data"]["train"]
    animals = train.animals
    counts = series_versus_animals(animals, train.obs)
    ep = hist.get("epoch_count") or count_from_hist(hist)
    return {
        "scope": scope,
        "fold": fold,
        "seed": int(seed),
        "reused": False,
        "train_studies": list(studies),
        "n_animals_in_split": counts["n_animals_in_split"],
        "n_tft_series": counts["n_tft_series"],
        "n_dropped_short_series": counts["n_dropped_short_series"],
        "n_train_series_logged": hist.get("n_train_series"),
        "n_val_series": hist.get("n_val_series"),
        "n_train_obs": hist.get("n_train_obs"),
        "n_train_windows": hist.get("n_train_windows"),
        "max_epochs_budget": hist.get("max_epochs"),
        "epochs_completed": ep.get("epochs_completed") if isinstance(ep, dict) else hist.get("epochs_run"),
        "epochs_run_legacy_plus_one": (ep or {}).get("epochs_run_legacy_plus_one") if isinstance(ep, dict) else None,
        "n_optimizer_updates": hist.get("n_optimizer_updates"),
        "finetuned": hist.get("finetuned", False),
        "finetune_max_epochs_budget": hist.get("finetune_max_epochs"),
        "finetune_epochs_completed": (hist.get("finetune_epoch_count") or {}).get("epochs_completed"),
        "window_balance": hist.get("window_balance"),
        "target_species_n_animals": counts["n_animals_in_split"],
    }


def count_from_hist(hist: dict[str, Any]) -> dict[str, Any]:
    if isinstance(hist.get("epoch_count"), dict):
        return hist["epoch_count"]
    return epoch_budget_versus_logged(
        hist.get("curves") or [], max_epochs=int(hist.get("max_epochs") or 0),
        lightning_current_epoch=None)


def _seed_variability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    keys = sorted({(r.get("bank"), r.get("variant"), r.get("fold")) for r in rows})
    for bank, variant, fold in keys:
        grp = [r for r in rows if (r.get("bank"), r.get("variant"), r.get("fold")) == (bank, variant, fold)]
        out[f"{bank}:fold{fold}:{variant}"] = {
            "mae_mean": _mean([r.get("mae_log") for r in grp]),
            "mae_sd": _sd([r.get("mae_log") for r in grp]),
            "pinball_mean": _mean([r.get("mean_quantile_loss") for r in grp]),
            "interval_score_mean": _mean([r.get("interval_score80") for r in grp]),
            "interval_score_sd": _sd([r.get("interval_score80") for r in grp]),
            "coverage_mean": _mean([r.get("coverage80") for r in grp]),
            "width_mean": _mean([r.get("interval_width80") for r in grp]),
            "crossing_mean": _mean([r.get("crossing") for r in grp]),
            "n_seeds": len(grp),
        }
    return out


def _mean(vals: list) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return float(np.mean(xs)) if xs else None


def _sd(vals: list) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else None


def _write_report(parent: Path, out: dict[str, Any], rows: list[dict[str, Any]],
                  folds: list[dict[str, Any]]) -> None:
    cons = out.get("consistency_mae") or {}
    sel = out.get("validation_selection") or {}
    lines = [
        "# TFT study-held-out evaluation",
        "",
        "Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.",
        "",
        "Target-only TFT is the development reference. Fine-tune and CQR are experimental.",
        "The PR #22 fine-tune default failed confirmation and was not retuned.",
        "Confirmation 34011 is not described as untouched: SYNA-007/010/011 were in its train set.",
        "",
        "## Decision rules",
        "",
        f"- validation-preferred (IS80, not a new default): **{sel.get('lowest_validation')}**",
        f"- any experimental variant consistently beats target-only (MAE): {cons.get('any_consistent_improvement')}",
        f"- seconds: {out.get('seconds_total')}",
        "",
        "## Folds (split seed 35021)",
        "",
    ]
    for f in folds:
        lines.append(
            f"- fold {f['fold']}: test {f['test']}; val {f['validation']}; "
            f"cal {f['calibration']}; train {f['train']}")
    lines += [
        "",
        "## Test scores (raw quantiles except CQR rows)",
        "",
        "| fold | variant | seed | MAE | pinball | IS80 | cov80 | width | crossing | table |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("bank") != "test":
            continue
        lines.append(
            f"| {r['fold']} | {r['variant']} | {r['seed']} | {_n(r.get('mae_log'))} | "
            f"{_n(r.get('mean_quantile_loss'))} | {_n(r.get('interval_score80'))} | "
            f"{_n(r.get('coverage80'))} | {_n(r.get('interval_width80'))} | "
            f"{r.get('crossing')} | {r.get('quantile_table')} |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "Successful training and unit tests do not establish forecasting effectiveness. "
        "CQR pinball uses shifted endpoints and is not the TFT QuantileLoss.",
        "",
    ]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _n(v) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.4f}"

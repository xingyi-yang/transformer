"""TFT multi-seed stability, interval scores, and a validation-led default.

Reuses matching official runs from runs/tft_primary. Writes only
runs/tft_stability. Locked v3, bench, extend, and primary directories stay
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

from send_bw.data.splits import make_study_splits
from send_bw.sharing import LABELS
from send_bw.tft_crossings import crossing_audit
from send_bw.tft_early_hist_bench import (
    PRESERVED as V3_PRESERVED,
)
from send_bw.tft_early_hist_bench import (
    _compact_score,
    load_yaml,
)
from send_bw.tft_early_hist_extend import load_full_cell, prepare_from_splits
from send_bw.tft_examples import (
    construct_forecast_examples,
    examples_to_tables,
    paired_study_differences,
    quantile_table_from_model,
    score_aligned_quantiles,
    summarize_observation_schedules,
)
from send_bw.tft_intervals import apply_cqr_margin, coverage_slices, fit_cqr_margin
from send_bw.tft_primary import (
    OUTPUT_DEFAULT as PRIMARY_DEFAULT,
)
from send_bw.tft_primary import (
    _finetune_tft,
    _fit_tft,
)
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_official_seq import collect_versions

OUTPUT_DEFAULT = "runs/tft_stability"
PILOT_DEFAULT = "runs/tft_early_hist_bench"
EXTEND_DEFAULT = "runs/tft_early_hist_extend"
LOCKED_DIRS = tuple(V3_PRESERVED) + (PILOT_DEFAULT, EXTEND_DEFAULT, PRIMARY_DEFAULT)
SEEDS = (82001, 82021, 82031)
COMMON_Q = [0.1, 0.25, 0.5, 0.75, 0.9]


class StabilityError(ValueError):
    pass


def _refuse_locked(parent: Path, extras: list[Path] | None = None) -> None:
    resolved = parent.resolve()
    blocked = {Path(p).resolve() for p in LOCKED_DIRS}
    for extra in extras or []:
        blocked.add(Path(extra).resolve())
    if resolved in blocked:
        raise StabilityError("refusing to write into a preserved run directory")
    if resolved.name.startswith("v3_") and "runs" in resolved.parts:
        raise StabilityError("refusing to write a new locked-style runs/v3_* directory")
    if resolved.name in {"tft_early_hist_bench", "tft_early_hist_extend", "tft_primary"}:
        raise StabilityError("refusing to overwrite a completed TFT run directory")


def metric_value(block: dict[str, Any] | None, key: str) -> float | None:
    if not block or block.get("status") not in {"ok", None}:
        return None
    v = block.get(key)
    if isinstance(v, dict):
        x = v.get("study_balanced")
        return None if x is None else float(x)
    return None if v is None else float(v)


def official_reuse_map(primary: Path) -> dict[tuple[str, int], Path]:
    """Completed official fits whose locked architecture matches this run."""
    return {
        ("target_only", 82001): primary / "models" / "tft_target_only",
        ("target_only", 82021): primary / "models" / "tft_target_only_seed82021",
        ("target_only", 82031): primary / "models" / "tft_target_only_seed82031",
        ("pooled", 82001): primary / "models" / "tft_pooled",
        ("finetune", 82001): primary / "models" / "tft_finetune",
    }


def confirm_study_splits(target_full, pooled_full, *, seed: int,
                         forbidden_test: list[str]) -> dict[str, Any]:
    """Random split of studies that have never been a reported test holdout."""
    animals = target_full.animals
    base = animals.drop_duplicates("STUDYID")
    ctx = pd.DataFrame({
        "STUDYID": base["STUDYID"].to_numpy(),
        "STUDY_START": base["REF_DATE"].to_numpy() if "REF_DATE" in base.columns else "2000-01-01",
    })
    forbidden = {str(s) for s in forbidden_test}
    allowed = ctx.loc[~ctx["STUDYID"].astype(str).isin(forbidden)].copy()
    if allowed["STUDYID"].nunique() < 4:
        raise StabilityError("too few unused studies for a confirmation holdout")
    cfg = {"method": "random", "validation": 0.2, "test": 0.2}
    target_splits = make_study_splits(allowed, cfg, seed)
    reserved = ctx.loc[ctx["STUDYID"].astype(str).isin(forbidden), ["STUDYID"]].copy()
    reserved["SPLIT"] = "train"
    target_splits = pd.concat([target_splits, reserved], ignore_index=True).sort_values("STUDYID")
    test_ids = set(target_splits.loc[target_splits["SPLIT"] == "test", "STUDYID"].astype(str))
    if test_ids & forbidden:
        raise StabilityError(f"confirmation test overlaps a reported holdout: {sorted(test_ids & forbidden)}")
    donor_ids = []
    if "SPECIES" in pooled_full.animals.columns:
        donor = pooled_full.animals.loc[
            pooled_full.animals["SPECIES"].astype(str).str.upper() != str(target_full.spec.species).upper(),
            "STUDYID"]
        donor_ids = sorted({str(s) for s in donor})
    from send_bw.data.splits import compose_train_with_selected_donors

    pooled_splits = compose_train_with_selected_donors(target_splits, donor_ids)
    return {
        "seed": int(seed),
        "target_splits": target_splits.reset_index(drop=True),
        "pooled_splits": pooled_splits,
        "donor_ids": donor_ids,
        "forbidden_test_studies": sorted(forbidden),
        "test_studies": sorted(test_ids),
        "note": ("Test studies are unused as test. Previously reported holdouts "
                 "SYNA-007/010/011 were placed in TRAIN. Confirmation training "
                 "was therefore not holdout-free. Do not call this untouched."),
    }


def _examples(data, cutoff: float, max_enc: int, max_pred: int) -> dict[str, Any]:
    pack = construct_forecast_examples(
        data.obs, data.animals, cutoff=cutoff,
        max_encoder_length=max_enc, max_prediction_length=max_pred)
    hist, fut = examples_to_tables(pack)
    return {"pack": pack, "hist": hist, "fut": fut, "animals": data.animals}


def exposure_record(hist: dict[str, Any], prep: dict[str, Any], *,
                    scope: str, seed: int, reused: bool) -> dict[str, Any]:
    train = prep["data"]["train"]
    animals = train.animals
    species = {}
    if "SPECIES" in animals.columns:
        species = {str(k): int(v) for k, v in
                   animals.groupby(animals["SPECIES"].astype(str))["USUBJID"].nunique().items()}
    target = str(train.spec.species).upper() if getattr(train, "spec", None) else "RAT"
    n_target = 0
    n_donor = 0
    if "SPECIES" in animals.columns:
        sp = animals["SPECIES"].astype(str).str.upper()
        n_target = int(animals.loc[sp == target, "USUBJID"].nunique())
        n_donor = int(animals.loc[sp != target, "USUBJID"].nunique())
    else:
        n_target = int(animals["USUBJID"].nunique())
    return {
        "scope": scope,
        "seed": int(seed),
        "reused": reused,
        "n_train_series": hist.get("n_train_series"),
        "n_val_series": hist.get("n_val_series"),
        "n_train_obs": hist.get("n_train_obs"),
        "n_train_windows": hist.get("n_train_windows"),
        "max_epochs": hist.get("max_epochs"),
        "epochs_run": hist.get("epochs_run"),
        "n_optimizer_updates": hist.get("n_optimizer_updates"),
        "finetuned": hist.get("finetuned", False),
        "finetune_epochs_run": hist.get("finetune_epochs_run"),
        "finetune_n_train_series": hist.get("finetune_n_train_series"),
        "finetune_n_optimizer_updates": hist.get("finetune_n_optimizer_updates"),
        "window_balance": hist.get("window_balance"),
        "target_normalizer": (hist.get("target_normalizer") or {}).get("class")
        if isinstance(hist.get("target_normalizer"), dict) else hist.get("target_normalizer"),
        "species_n_animals": species,
        "target_species": target,
        "target_species_n_animals": n_target,
        "donor_n_animals": n_donor,
        "target_species_n_studies": int(
            animals.loc[animals["SPECIES"].astype(str).str.upper() == target, "STUDYID"].nunique()
            if "SPECIES" in animals.columns else animals["STUDYID"].nunique()),
    }


def _score(name: str, model, bank: dict[str, Any], *, label: str,
           extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    t1 = time.time()
    pred, fails = quantile_table_from_model(model, bank["animals"], bank["pack"], bank["hist"])
    block = score_aligned_quantiles(bank["fut"], pred, representation="tft_quantiles")
    audit = crossing_audit(pred, bank["fut"], levels=COMMON_Q, animals=bank["animals"])
    slices = coverage_slices(bank["fut"], pred)
    block.update({
        "seconds": time.time() - t1,
        "uses_early_history": True,
        "label": label,
        "n_failures": len(fails),
        "failures": fails,
        "crossing_audit": audit,
        "coverage_slices": slices,
    })
    if extra:
        block.update(extra)
    return block, pred


def _load_or_fit(scope: str, seed: int, prep: dict[str, Any], cfg: dict[str, Any],
                 dest: Path, reuse: dict[tuple[str, int], Path], *,
                 pooled_dir: Path | None = None, finetune_cfg: dict[str, Any] | None = None):
    from send_bw.models.tft import TFTEarlyHistoryModel

    src = reuse.get((scope, seed))
    if src is not None and (src / "tft.ckpt").exists():
        model = TFTEarlyHistoryModel.load(src)
        hist = read_json(src / "training_history.json") if (src / "training_history.json").exists() else (
            model.history or {})
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest / "reuse.json", {"reused_from": str(src), "scope": scope, "seed": seed})
        return model, hist, True
    dest.mkdir(parents=True, exist_ok=True)
    scfg = {**cfg, "seed": seed}
    if scope == "finetune":
        if pooled_dir is None or not (pooled_dir / "tft.ckpt").exists():
            raise StabilityError(f"pooled checkpoint missing for fine-tune seed {seed}")
        ft = dict(scfg)
        ft["finetune"] = finetune_cfg or {}
        model, hist = _finetune_tft(pooled_dir, prep, ft, dest)
        return model, hist, False
    model, hist = _fit_tft(prep, scfg, dest, scope)
    return model, hist, False


def select_default(val_rows: list[dict[str, Any]], *, metric: str,
                   candidates: list[str]) -> dict[str, Any]:
    """Lowest mean validation metric across seeds. Test scores are ignored."""
    by: dict[str, list[float]] = {c: [] for c in candidates}
    for row in val_rows:
        name = row.get("variant")
        if name not in by:
            continue
        if row.get(metric) is None:
            continue
        by[name].append(float(row[metric]))
    summary = []
    for name in candidates:
        vals = by[name]
        summary.append({
            "variant": name,
            "n_seeds": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "values": vals,
        })
    usable = [s for s in summary if s["mean"] is not None]
    if not usable:
        raise StabilityError("no validation scores available for strategy selection")
    winner = min(usable, key=lambda s: s["mean"])
    return {
        "metric": metric,
        "selected": winner["variant"],
        "selected_mean": winner["mean"],
        "candidates": summary,
        "test_scores_examined": False,
        "note": "Official validation only. Official test and the confirm holdout were not used.",
    }


def _sb(block: dict[str, Any], key: str) -> float | None:
    return metric_value(block, key)


def execute_tft_stability(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    primary = Path(cfg.get("primary_dir") or PRIMARY_DEFAULT)
    _refuse_locked(parent, extras=[primary, Path(cfg.get("pilot_dir") or PILOT_DEFAULT),
                                   Path(cfg.get("extend_dir") or EXTEND_DEFAULT)])
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise StabilityError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise StabilityError("design optimization stays off")
    if parent.exists() and any(parent.iterdir()) and not force:
        raise StabilityError(f"{parent} exists; pass --force to overwrite this unlocked directory")
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "predictions").mkdir(parents=True, exist_ok=True)
    (parent / "models").mkdir(parents=True, exist_ok=True)
    ev = cfg.get("benchmark") or {}
    cutoff = float(ev.get("encoder_window_day") or 14)
    tft_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("tft") or {})
    max_enc = int(tft_cfg.get("max_encoder_length", 8))
    max_pred = int(tft_cfg.get("max_prediction_length", 6))
    seeds = [int(s) for s in (ev.get("stability_seeds") or SEEDS)]
    cells = ev.get("source_cells") or {}
    target_dir = Path(cells.get("target_only") or "")
    pooled_dir = Path(cells.get("pooled") or "")
    if not target_dir.is_dir() or not pooled_dir.is_dir():
        raise StabilityError("source_cells.target_only and source_cells.pooled must exist")
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    reuse = official_reuse_map(primary)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": cfg.get("generate_seed"),
        "stability_seeds": seeds,
        "confirm_split_seed": ev.get("confirm_split_seed"),
        "confirm_forbidden_test_studies": ev.get("confirm_forbidden_test_studies"),
        "selection": cfg.get("selection"),
        "calibration": cfg.get("calibration"),
        "official_holdout_test_used_for_search": False,
        "reused_from": str(primary),
        "real_data": False, "design": False, "labels": list(LABELS),
        "preserved": list(LOCKED_DIRS), "writes_v3_locks": False,
        "writes_primary": False,
    }
    write_json(parent / "freeze_lock.json", lock)

    target = load_cell_prepared(target_dir)
    pooled = load_cell_prepared(pooled_dir)
    official_test = _examples(target["data"]["test"], cutoff, max_enc, max_pred)
    official_val = _examples(target["data"]["validation"], cutoff, max_enc, max_pred)
    official_test["pack"]["examples"].to_csv(parent / "official_test_examples.csv", index=False)
    official_val["pack"]["examples"].to_csv(parent / "official_val_examples.csv", index=False)
    official_test["fut"].to_csv(parent / "official_test_target_obs.csv", index=False)
    official_val["fut"].to_csv(parent / "official_val_target_obs.csv", index=False)

    t0 = time.time()
    failures: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    official_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    models_by: dict[tuple[str, int], Any] = {}
    preds: dict[str, pd.DataFrame] = {}

    def _record(bank_name: str, variant: str, seed: int, block: dict[str, Any],
                pred: pd.DataFrame, dest_name: str) -> None:
        pred.to_csv(parent / "predictions" / f"{dest_name}.csv", index=False)
        preds[dest_name] = pred
        row = {
            "bank": bank_name, "variant": variant, "seed": seed,
            "mae_log": _sb(block, "mae_log"),
            "mean_quantile_loss": _sb(block, "mean_quantile_loss"),
            "interval_score80": _sb(block, "interval_score80"),
            "coverage80": _sb(block, "coverage80"),
            "interval_width80": _sb(block, "interval_width80"),
            "crossing": (block.get("crossing_rate") or {}).get("pooled")
            if isinstance(block.get("crossing_rate"), dict) else block.get("crossing_rate"),
            "n_scored": block.get("n_scored"),
            "n_failures": block.get("n_failures"),
            "status": block.get("status"),
            "reused": block.get("reused"),
            "per_study": block.get("per_study"),
            "coverage_slices": block.get("coverage_slices"),
        }
        if bank_name.endswith("validation") or bank_name.startswith("official_val"):
            val_rows.append(row)
        else:
            official_rows.append(row)
        write_json(parent / "scores" / f"{dest_name}.json", _compact_score(block))

    (parent / "scores").mkdir(parents=True, exist_ok=True)
    ft_cfg = cfg.get("finetune") or {}

    for seed in seeds:
        try:
            m_to, h_to, ru_to = _load_or_fit(
                "target_only", seed, target, tft_cfg,
                parent / "models" / f"official_to_{seed}", reuse)
            models_by[("target_only", seed)] = m_to
            exposures.append(exposure_record(h_to, target, scope="target_only", seed=seed, reused=ru_to))
            b_val, p_val = _score(
                f"to_{seed}_val", m_to, official_val,
                label=f"official validation target-only seed {seed}",
                extra={"reused": ru_to, "seed": seed})
            _record("official_validation", "tft_target_only", seed, b_val, p_val, f"official_val_to_{seed}")
            b_te, p_te = _score(
                f"to_{seed}_test", m_to, official_test,
                label=f"official test target-only seed {seed} (reference)",
                extra={"reused": ru_to, "seed": seed, "reference": True})
            _record("official_test_reference", "tft_target_only", seed, b_te, p_te, f"official_test_to_{seed}")
        except Exception as e:  # noqa: BLE001
            failures.append({"model": f"official_to_{seed}", "reason": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-1500:]})

        try:
            m_po, h_po, ru_po = _load_or_fit(
                "pooled", seed, pooled, tft_cfg,
                parent / "models" / f"official_pooled_{seed}", reuse)
            models_by[("pooled", seed)] = m_po
            exposures.append(exposure_record(h_po, pooled, scope="pooled", seed=seed, reused=ru_po))
            b_val, p_val = _score(
                f"po_{seed}_val", m_po, official_val,
                label=f"official validation pooled seed {seed}",
                extra={"reused": ru_po, "seed": seed})
            _record("official_validation", "tft_pooled", seed, b_val, p_val, f"official_val_pooled_{seed}")
            b_te, p_te = _score(
                f"po_{seed}_test", m_po, official_test,
                label=f"official test pooled seed {seed} (reference)",
                extra={"reused": ru_po, "seed": seed, "reference": True})
            _record("official_test_reference", "tft_pooled", seed, b_te, p_te, f"official_test_pooled_{seed}")
        except Exception as e:  # noqa: BLE001
            failures.append({"model": f"official_pooled_{seed}", "reason": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-1500:]})
            m_po = None

        try:
            pooled_src = reuse.get(("pooled", seed))
            if pooled_src is None or not (pooled_src / "tft.ckpt").exists():
                pooled_src = parent / "models" / f"official_pooled_{seed}"
            m_ft, h_ft, ru_ft = _load_or_fit(
                "finetune", seed, target, tft_cfg,
                parent / "models" / f"official_ft_{seed}", reuse,
                pooled_dir=pooled_src, finetune_cfg=ft_cfg)
            models_by[("finetune", seed)] = m_ft
            exposures.append(exposure_record(h_ft, target, scope="finetune", seed=seed, reused=ru_ft))
            b_val, p_val = _score(
                f"ft_{seed}_val", m_ft, official_val,
                label=f"official validation fine-tune seed {seed}",
                extra={"reused": ru_ft, "seed": seed})
            _record("official_validation", "tft_finetune", seed, b_val, p_val, f"official_val_ft_{seed}")
            b_te, p_te = _score(
                f"ft_{seed}_test", m_ft, official_test,
                label=f"official test fine-tune seed {seed} (reference)",
                extra={"reused": ru_ft, "seed": seed, "reference": True})
            _record("official_test_reference", "tft_finetune", seed, b_te, p_te, f"official_test_ft_{seed}")
        except Exception as e:  # noqa: BLE001
            failures.append({"model": f"official_ft_{seed}", "reason": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-1500:]})

    cal_cfg = cfg.get("calibration") or {}
    cqr_logs: list[dict[str, Any]] = []
    for seed in seeds:
        val_pred_path = parent / "predictions" / f"official_val_to_{seed}.csv"
        if not val_pred_path.is_file():
            continue
        val_pred = pd.read_csv(val_pred_path)
        rec = fit_cqr_margin(official_val["fut"], val_pred, alpha=float(cal_cfg.get("alpha") or 0.2))
        rec["seed"] = seed
        rec["source"] = "official_validation"
        cqr_logs.append(rec)
        if rec.get("status") != "ok":
            continue
        for bank_name, bank, tag in (
            ("official_validation", official_val, "val"),
            ("official_test_reference", official_test, "test"),
        ):
            raw_path = parent / "predictions" / f"official_{tag}_to_{seed}.csv"
            if not raw_path.is_file():
                continue
            raw = pd.read_csv(raw_path)
            cal = apply_cqr_margin(raw, float(rec["margin"]))
            block = score_aligned_quantiles(bank["fut"], cal, representation="tft_quantiles")
            block["crossing_audit"] = crossing_audit(cal, bank["fut"], levels=COMMON_Q,
                                                     animals=bank["animals"])
            block["coverage_slices"] = coverage_slices(bank["fut"], cal)
            block.update({"uses_early_history": True, "cqr": rec, "seed": seed,
                          "label": f"target-only + CQR ({tag}) seed {seed}"})
            _record(bank_name, "tft_target_only_cqr", seed, block, cal,
                    f"official_{tag}_to_cqr_{seed}")

    adopt = False
    to_is = [r["interval_score80"] for r in val_rows if r["variant"] == "tft_target_only"
             and r["interval_score80"] is not None]
    cqr_is = [r["interval_score80"] for r in val_rows if r["variant"] == "tft_target_only_cqr"
              and r["interval_score80"] is not None]
    if to_is and cqr_is and float(np.mean(cqr_is)) < float(np.mean(to_is)):
        adopt = True
    remedy = {
        "method": cal_cfg.get("method"),
        "applied_to": cal_cfg.get("apply_to"),
        "adopted": adopt,
        "reason": ("validation interval score decreased" if adopt
                   else "validation interval score did not decrease; CQR is reported but not the default"),
        "val_interval_score_target_only_mean": float(np.mean(to_is)) if to_is else None,
        "val_interval_score_cqr_mean": float(np.mean(cqr_is)) if cqr_is else None,
        "logs": cqr_logs,
        "test_outcomes_used": False,
    }
    write_json(parent / "cqr_decision.json", remedy)

    undercover = []
    for seed in seeds:
        raw_path = parent / "predictions" / f"official_test_to_{seed}.csv"
        val_path = parent / "predictions" / f"official_val_to_{seed}.csv"
        if raw_path.is_file():
            undercover.append({
                "seed": seed, "bank": "official_test_reference",
                **coverage_slices(official_test["fut"], pd.read_csv(raw_path)),
            })
        if val_path.is_file():
            undercover.append({
                "seed": seed, "bank": "official_validation",
                **coverage_slices(official_val["fut"], pd.read_csv(val_path)),
            })
    write_json(parent / "undercoverage.json", {
        "note": ("Target-only 80% coverage by study and lead. Official test is a "
                 "reference diagnosis. The CQR margin used validation only."),
        "rows": undercover,
    })

    sel_cfg = cfg.get("selection") or {}
    decision = select_default(
        val_rows, metric=str(sel_cfg.get("metric") or "interval_score80"),
        candidates=list(sel_cfg.get("candidates") or [
            "tft_target_only", "tft_target_only_cqr", "tft_pooled", "tft_finetune"]))
    write_json(parent / "strategy_decision.json", decision)

    target_full = load_full_cell(target_dir)
    pooled_full = load_full_cell(pooled_dir)
    confirm_pack = confirm_study_splits(
        target_full["full"], pooled_full["full"],
        seed=int(ev.get("confirm_split_seed") or 34011),
        forbidden_test=list(ev.get("confirm_forbidden_test_studies") or ["SYNA-007", "SYNA-010", "SYNA-011"]))
    extra_to = prepare_from_splits(
        target_full["full"], confirm_pack["target_splits"], species="RAT", sharing="context_specific")
    extra_po = prepare_from_splits(
        pooled_full["full"], confirm_pack["pooled_splits"], species="RAT", sharing="pooled")
    confirm_dir = parent / "confirm_34011"
    confirm_dir.mkdir(parents=True, exist_ok=True)
    confirm_pack["target_splits"].to_csv(confirm_dir / "splits_target.csv", index=False)
    confirm_pack["pooled_splits"].to_csv(confirm_dir / "splits_pooled.csv", index=False)
    write_json(confirm_dir / "split_summary.json", {
        k: confirm_pack[k] for k in confirm_pack if k not in {"target_splits", "pooled_splits"}
    })
    confirm_test = _examples(extra_to["data"]["test"], cutoff, max_enc, max_pred)
    confirm_val = _examples(extra_to["data"]["validation"], cutoff, max_enc, max_pred)
    confirm_test["fut"].to_csv(confirm_dir / "target_obs.csv", index=False)
    confirm_val["fut"].to_csv(confirm_dir / "val_target_obs.csv", index=False)
    confirm_test["pack"]["examples"].to_csv(confirm_dir / "examples.csv", index=False)
    write_json(confirm_dir / "example_summary.json", {
        k: confirm_test["pack"][k] for k in confirm_test["pack"] if k not in {"examples", "exclusions"}
    })
    write_json(confirm_dir / "schedules.json", {
        "test": summarize_observation_schedules(extra_to["data"]["test"].obs, extra_to["data"]["test"].animals),
        "val": summarize_observation_schedules(extra_to["data"]["validation"].obs,
                                               extra_to["data"]["validation"].animals),
        "train_target": summarize_observation_schedules(extra_to["data"]["train"].obs,
                                                        extra_to["data"]["train"].animals),
    })

    confirm_rows: list[dict[str, Any]] = []
    empty_reuse: dict[tuple[str, int], Path] = {}
    for seed in seeds:
        try:
            m_to, h_to, _ = _load_or_fit(
                "target_only", seed, extra_to, tft_cfg,
                confirm_dir / "models" / f"to_{seed}", empty_reuse)
            exposures.append(exposure_record(h_to, extra_to, scope="confirm_target_only",
                                             seed=seed, reused=False))
            b_val, p_val = _score("c_to_val", m_to, confirm_val,
                                  label=f"confirm val target-only seed {seed}", extra={"seed": seed})
            p_val.to_csv(confirm_dir / f"val_to_{seed}.csv", index=False)
            b_te, p_te = _score("c_to_te", m_to, confirm_test,
                                label=f"confirm test target-only seed {seed}", extra={"seed": seed})
            p_te.to_csv(confirm_dir / f"test_to_{seed}.csv", index=False)
            confirm_rows.append({"bank": "confirm_val", "variant": "tft_target_only", "seed": seed,
                                 "interval_score80": _sb(b_val, "interval_score80"),
                                 "mae_log": _sb(b_val, "mae_log"),
                                 "coverage80": _sb(b_val, "coverage80"),
                                 "mean_quantile_loss": _sb(b_val, "mean_quantile_loss"),
                                 "interval_width80": _sb(b_val, "interval_width80"),
                                 "crossing": (b_val.get("crossing_rate") or {}).get("pooled")})
            confirm_rows.append({"bank": "confirm_test", "variant": "tft_target_only", "seed": seed,
                                 "interval_score80": _sb(b_te, "interval_score80"),
                                 "mae_log": _sb(b_te, "mae_log"),
                                 "coverage80": _sb(b_te, "coverage80"),
                                 "mean_quantile_loss": _sb(b_te, "mean_quantile_loss"),
                                 "interval_width80": _sb(b_te, "interval_width80"),
                                 "crossing": (b_te.get("crossing_rate") or {}).get("pooled"),
                                 "per_study": b_te.get("per_study"),
                                 "coverage_slices": b_te.get("coverage_slices")})
            if adopt:
                rec = fit_cqr_margin(confirm_val["fut"], p_val, alpha=float(cal_cfg.get("alpha") or 0.2))
                rec["seed"] = seed
                rec["source"] = "confirm_validation"
                cqr_logs.append(rec)
                if rec.get("status") == "ok":
                    cal_val = apply_cqr_margin(p_val, float(rec["margin"]))
                    cal_te = apply_cqr_margin(p_te, float(rec["margin"]))
                    bv = score_aligned_quantiles(confirm_val["fut"], cal_val, representation="tft_quantiles")
                    bt = score_aligned_quantiles(confirm_test["fut"], cal_te, representation="tft_quantiles")
                    bt["crossing_audit"] = crossing_audit(cal_te, confirm_test["fut"], levels=COMMON_Q)
                    bt["coverage_slices"] = coverage_slices(confirm_test["fut"], cal_te)
                    cal_te.to_csv(confirm_dir / f"test_to_cqr_{seed}.csv", index=False)
                    confirm_rows.append({"bank": "confirm_val", "variant": "tft_target_only_cqr",
                                         "seed": seed, "interval_score80": _sb(bv, "interval_score80"),
                                         "mae_log": _sb(bv, "mae_log"),
                                         "coverage80": _sb(bv, "coverage80"),
                                         "mean_quantile_loss": _sb(bv, "mean_quantile_loss"),
                                         "interval_width80": _sb(bv, "interval_width80")})
                    confirm_rows.append({"bank": "confirm_test", "variant": "tft_target_only_cqr",
                                         "seed": seed, "interval_score80": _sb(bt, "interval_score80"),
                                         "mae_log": _sb(bt, "mae_log"),
                                         "coverage80": _sb(bt, "coverage80"),
                                         "mean_quantile_loss": _sb(bt, "mean_quantile_loss"),
                                         "interval_width80": _sb(bt, "interval_width80"),
                                         "crossing": (bt.get("crossing_rate") or {}).get("pooled"),
                                         "per_study": bt.get("per_study"),
                                         "coverage_slices": bt.get("coverage_slices")})
        except Exception as e:  # noqa: BLE001
            failures.append({"model": f"confirm_to_{seed}", "reason": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-1500:]})

        try:
            m_po, h_po, _ = _load_or_fit(
                "pooled", seed, extra_po, tft_cfg,
                confirm_dir / "models" / f"pooled_{seed}", empty_reuse)
            exposures.append(exposure_record(h_po, extra_po, scope="confirm_pooled",
                                             seed=seed, reused=False))
            b_te, p_te = _score("c_po", m_po, confirm_test,
                                label=f"confirm test pooled seed {seed}", extra={"seed": seed})
            p_te.to_csv(confirm_dir / f"test_pooled_{seed}.csv", index=False)
            confirm_rows.append({"bank": "confirm_test", "variant": "tft_pooled", "seed": seed,
                                 "interval_score80": _sb(b_te, "interval_score80"),
                                 "mae_log": _sb(b_te, "mae_log"),
                                 "coverage80": _sb(b_te, "coverage80"),
                                 "mean_quantile_loss": _sb(b_te, "mean_quantile_loss"),
                                 "interval_width80": _sb(b_te, "interval_width80"),
                                 "crossing": (b_te.get("crossing_rate") or {}).get("pooled"),
                                 "per_study": b_te.get("per_study")})
            ft_locked = dict(tft_cfg)
            ft_locked["seed"] = seed
            ft_locked["finetune"] = ft_cfg
            m_ft, h_ft = _finetune_tft(
                confirm_dir / "models" / f"pooled_{seed}", extra_to, ft_locked,
                confirm_dir / "models" / f"ft_{seed}")
            exposures.append(exposure_record(h_ft, extra_to, scope="confirm_finetune",
                                             seed=seed, reused=False))
            b_ft, p_ft = _score("c_ft", m_ft, confirm_test,
                                label=f"confirm test fine-tune seed {seed}", extra={"seed": seed})
            p_ft.to_csv(confirm_dir / f"test_ft_{seed}.csv", index=False)
            confirm_rows.append({"bank": "confirm_test", "variant": "tft_finetune", "seed": seed,
                                 "interval_score80": _sb(b_ft, "interval_score80"),
                                 "mae_log": _sb(b_ft, "mae_log"),
                                 "coverage80": _sb(b_ft, "coverage80"),
                                 "mean_quantile_loss": _sb(b_ft, "mean_quantile_loss"),
                                 "interval_width80": _sb(b_ft, "interval_width80"),
                                 "crossing": (b_ft.get("crossing_rate") or {}).get("pooled"),
                                 "per_study": b_ft.get("per_study")})
        except Exception as e:  # noqa: BLE001
            failures.append({"model": f"confirm_share_{seed}", "reason": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-1500:]})

    paired = _paired_tables(official_rows, confirm_rows)
    seed_var = _seed_variability(official_rows + confirm_rows)
    pd.DataFrame(official_rows).drop(columns=["per_study", "coverage_slices"], errors="ignore").to_csv(
        parent / "official_seed_metrics.csv", index=False)
    pd.DataFrame(val_rows).drop(columns=["per_study", "coverage_slices"], errors="ignore").to_csv(
        parent / "validation_seed_metrics.csv", index=False)
    pd.DataFrame(confirm_rows).drop(columns=["per_study", "coverage_slices"], errors="ignore").to_csv(
        parent / "confirm_seed_metrics.csv", index=False)
    pd.DataFrame(exposures).to_csv(parent / "training_exposure.csv", index=False)
    write_json(parent / "paired_differences.json", paired)
    write_json(parent / "seed_variability.json", seed_var)

    extra_ref = _reference_extra_split(Path(ev.get("extra_split_dir") or
                                            "runs/tft_early_hist_extend/banks/split_33001"),
                                       parent)
    out = {
        "labels": list(LABELS) + ["TFT stability and interval diagnostic.", "Synthetic generator only."],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "strategy_decision": decision,
        "cqr_decision": {k: remedy.get(k) for k in remedy if k != "logs"},
        "confirm_split": {k: confirm_pack[k] for k in confirm_pack if k not in {"target_splits", "pooled_splits"}},
        "official_validation": _summarize_bank(val_rows),
        "official_test_reference": _summarize_bank(official_rows),
        "confirm_34011": _summarize_bank(confirm_rows),
        "seed_variability": seed_var,
        "paired_differences": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                               for k, v in paired.items()},
        "training_exposure": exposures,
        "extra_split_33001_reference": extra_ref,
        "failures": failures,
        "real_data_accessed": False,
        "design_run": False,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out, val_rows, official_rows, confirm_rows)
    return out


def _summarize_bank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in sorted({r["variant"] for r in rows}):
        use = [r for r in rows if r["variant"] == variant]
        banks: dict[str, list] = {}
        for r in use:
            banks.setdefault(str(r["bank"]), []).append(r)
        out[variant] = {}
        for bank, rs in banks.items():
            out[variant][bank] = {
                "n": len(rs),
                "mae_mean": _mean([r.get("mae_log") for r in rs]),
                "mae_sd": _sd([r.get("mae_log") for r in rs]),
                "pinball_mean": _mean([r.get("mean_quantile_loss") for r in rs]),
                "interval_score_mean": _mean([r.get("interval_score80") for r in rs]),
                "interval_score_sd": _sd([r.get("interval_score80") for r in rs]),
                "coverage_mean": _mean([r.get("coverage80") for r in rs]),
                "width_mean": _mean([r.get("interval_width80") for r in rs]),
                "crossing_mean": _mean([r.get("crossing") for r in rs]),
                "seeds": [r.get("seed") for r in rs],
            }
    return out


def _mean(vals: list) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return float(np.mean(xs)) if xs else None


def _sd(vals: list) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else None


def _seed_variability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for (bank, variant), grp in _group(rows):
        out[f"{bank}:{variant}"] = {
            "mae_mean": _mean([r.get("mae_log") for r in grp]),
            "mae_sd": _sd([r.get("mae_log") for r in grp]),
            "pinball_mean": _mean([r.get("mean_quantile_loss") for r in grp]),
            "pinball_sd": _sd([r.get("mean_quantile_loss") for r in grp]),
            "interval_score_mean": _mean([r.get("interval_score80") for r in grp]),
            "interval_score_sd": _sd([r.get("interval_score80") for r in grp]),
            "coverage_mean": _mean([r.get("coverage80") for r in grp]),
            "coverage_sd": _sd([r.get("coverage80") for r in grp]),
            "width_mean": _mean([r.get("interval_width80") for r in grp]),
            "n_seeds": len(grp),
        }
    return out


def _group(rows: list[dict[str, Any]]):
    keys = sorted({(r.get("bank"), r.get("variant")) for r in rows})
    for k in keys:
        yield k, [r for r in rows if (r.get("bank"), r.get("variant")) == k]


def _paired_tables(official_rows: list[dict[str, Any]], confirm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    pairs = [("tft_target_only", "tft_pooled"), ("tft_target_only", "tft_finetune"),
             ("tft_pooled", "tft_finetune"), ("tft_target_only", "tft_target_only_cqr")]
    for bank, rows in (("official_test_reference", official_rows), ("confirm_test",
                        [r for r in confirm_rows if r.get("bank") == "confirm_test"])):
        for a, b in pairs:
            for metric in ("mae_log", "mean_quantile_loss", "interval_score80", "coverage80"):
                seed_rows = []
                for seed in SEEDS:
                    ra = next((r for r in rows if r.get("variant") == a and r.get("seed") == seed), None)
                    rb = next((r for r in rows if r.get("variant") == b and r.get("seed") == seed), None)
                    if not ra or not rb or ra.get(metric) is None or rb.get(metric) is None:
                        continue
                    seed_rows.append({
                        "seed": seed, "a": ra[metric], "b": rb[metric],
                        "delta": float(rb[metric]) - float(ra[metric]),
                    })
                if not seed_rows:
                    continue
                deltas = [r["delta"] for r in seed_rows]
                out[f"{bank}:{a}_vs_{b}:{metric}"] = {
                    "a": a, "b": b, "metric": metric, "bank": bank,
                    "n": len(seed_rows),
                    "mean_delta": float(np.mean(deltas)),
                    "sd_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
                    "rows": seed_rows,
                    "note": "Paired by seed. Positive delta means b is larger than a.",
                }
                # study-level pairing for the first available seed with per_study
                ra = next((r for r in rows if r.get("variant") == a and r.get("per_study")), None)
                rb = next((r for r in rows if r.get("variant") == b and r.get("per_study")), None)
                if ra and rb:
                    key = "mae_log" if metric == "mae_log" else (
                        "mean_quantile_loss" if metric == "mean_quantile_loss" else None)
                    if key:
                        out[f"{bank}:{a}_vs_{b}:{metric}:studies"] = paired_study_differences(
                            ra["per_study"], rb["per_study"], metric=key, name_a=a, name_b=b)
    return out


def _reference_extra_split(src: Path, parent: Path) -> dict[str, Any]:
    """Re-score locked extra-split predictions with interval score. Read-only."""
    fut = src / "target_obs.csv"
    if not fut.is_file():
        return {"status": "not_run", "reason": "extra-split target_obs missing"}
    fut_df = pd.read_csv(fut)
    out = {}
    primary_extra = Path("runs/tft_primary/extra_split_33001")
    for name, fname in (
        ("tft_target_only", "tft_target_only.csv"),
        ("tft_pooled", "tft_pooled.csv"),
        ("tft_finetune", "tft_finetune.csv"),
    ):
        cand = [
            primary_extra / fname,
            src / f"extra_split_{name}.csv",
            src / fname,
        ]
        path = next((p for p in cand if p.is_file()), None)
        if path is None:
            continue
        pred = pd.read_csv(path)
        block = score_aligned_quantiles(fut_df, pred, representation="tft_quantiles")
        dest = parent / "predictions" / f"ref33001_{name}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != dest.resolve():
            pred.to_csv(dest, index=False)
        out[name] = {
            "source": str(path),
            "read_only": True,
            **_compact_score(block),
        }
    return out


def _write_report(parent: Path, out: dict[str, Any], val_rows, official_rows, confirm_rows) -> None:
    dec = out.get("strategy_decision") or {}
    cqr = out.get("cqr_decision") or {}
    confirm = out.get("confirm_split") or {}
    lines = [
        "# TFT stability and 80% intervals",
        "",
        "Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.",
        "",
        "Windows used observed `DAY` values. Official holdout test errors were not "
        "used to choose the training strategy or the CQR margin.",
        "",
        "## Decision",
        "",
        f"- default strategy (official validation interval score): **{dec.get('selected')}** "
        f"(mean {dec.get('selected_mean')})",
        f"- CQR adopted for target-only: {cqr.get('adopted')} ({cqr.get('reason')})",
        f"- confirm test studies (seed 34011): {confirm.get('test_studies')}",
        f"- forbidden test studies: {confirm.get('forbidden_test_studies')}",
        f"- seconds: {out.get('seconds_total')}",
        "",
        "## Official validation (selection bank)",
        "",
        "| variant | seed | MAE | pinball | IS80 | cov80 | width | crossing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in val_rows:
        lines.append(
            f"| {r['variant']} | {r['seed']} | {_n(r.get('mae_log'))} | "
            f"{_n(r.get('mean_quantile_loss'))} | {_n(r.get('interval_score80'))} | "
            f"{_n(r.get('coverage80'))} | {_n(r.get('interval_width80'))} | {r.get('crossing')} |"
        )
    lines += [
        "",
        "## Official test (reference; not used to choose)",
        "",
        "| variant | seed | MAE | pinball | IS80 | cov80 | width | crossing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in official_rows:
        lines.append(
            f"| {r['variant']} | {r['seed']} | {_n(r.get('mae_log'))} | "
            f"{_n(r.get('mean_quantile_loss'))} | {_n(r.get('interval_score80'))} | "
            f"{_n(r.get('coverage80'))} | {_n(r.get('interval_width80'))} | {r.get('crossing')} |"
        )
    lines += [
        "",
        "## Confirm holdout 34011 (untouched test studies)",
        "",
        "| variant | seed | MAE | pinball | IS80 | cov80 | width | crossing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in confirm_rows:
        if r.get("bank") != "confirm_test":
            continue
        lines.append(
            f"| {r['variant']} | {r['seed']} | {_n(r.get('mae_log'))} | "
            f"{_n(r.get('mean_quantile_loss'))} | {_n(r.get('interval_score80'))} | "
            f"{_n(r.get('coverage80'))} | {_n(r.get('interval_width80'))} | {r.get('crossing')} |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "Successful training and unit tests do not establish forecasting effectiveness. "
        "Numbers above are a synthetic-generator diagnostic on the evaluated settings only.",
        "",
    ]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _n(v) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.4f}"

"""Primary TFT early-history forecasting workflow.

Improves and documents TemporalFusionTransformer as the forecasting adapter
for observed early history. Does not overwrite locked runs/v3_*, the completed
pilot, or the extension directory.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

import pickle
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.evaluation.quantile_forecast import QUANTILES, last_value_predictions
from send_bw.models.base import ModelData
from send_bw.models.tft import (
    cat_label,
    normalizer_record,
)
from send_bw.sharing import LABELS
from send_bw.tft_crossings import crossing_audit, curve_still_improving
from send_bw.tft_early_hist_bench import (
    PRESERVED as V3_PRESERVED,
)
from send_bw.tft_early_hist_bench import (
    _compact_score,
    _fmt,
    _plot_trajectories,
    _verify_adapter,
    load_yaml,
)
from send_bw.tft_early_hist_extend import load_full_cell, prepare_from_splits
from send_bw.tft_examples import (
    construct_forecast_examples,
    examples_to_tables,
    inventory_model_data,
    quantile_table_from_model,
    score_aligned_quantiles,
    summarize_observation_schedules,
)
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_official_seq import collect_versions

OUTPUT_DEFAULT = "runs/tft_primary"
PILOT_DEFAULT = "runs/tft_early_hist_bench"
EXTEND_DEFAULT = "runs/tft_early_hist_extend"
LOCKED_DIRS = tuple(V3_PRESERVED) + (PILOT_DEFAULT, EXTEND_DEFAULT)
COMMON_Q = [0.1, 0.25, 0.5, 0.75, 0.9]


class ForecastError(ValueError):
    pass


def _refuse_locked(parent: Path, *, pilot: Path, extend: Path) -> None:
    resolved = parent.resolve()
    blocked = {Path(p).resolve() for p in LOCKED_DIRS}
    blocked.update({pilot.resolve(), extend.resolve()})
    if resolved in blocked:
        raise ForecastError("refusing to write into a preserved run directory")
    if resolved.name.startswith("v3_") and "runs" in resolved.parts:
        raise ForecastError("refusing to write a new locked-style runs/v3_* directory")
    if resolved.name in {"tft_early_hist_bench", "tft_early_hist_extend"}:
        raise ForecastError("refusing to overwrite completed TFT bench/extend artifacts")


def best_val_loss(hist: dict[str, Any], key: str = "val_loss") -> float | None:
    vals = []
    for row in hist.get("curves") or []:
        if row.get(key) is None:
            continue
        try:
            vals.append(float(row[key]))
        except (TypeError, ValueError):
            continue
    return min(vals) if vals else None


def select_by_val_loss(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the lowest validation loss. Test errors are ignored even if present."""
    usable = [c for c in candidates if c.get("best_val_loss") is not None]
    if not usable:
        raise ForecastError("no candidate reported a validation loss")
    return min(usable, key=lambda r: float(r["best_val_loss"]))


def _slug(value: Any) -> str:
    text = str(value).replace(".", "p")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "p"} else "_" for ch in text)


def _fit_tft(prep: dict[str, Any], cfg: dict[str, Any], dest: Path, scope: str):
    from send_bw.models.tft import TFTEarlyHistoryModel

    dest.mkdir(parents=True, exist_ok=True)
    model = TFTEarlyHistoryModel()
    hist = model.fit(prep["data"]["train"], prep["data"]["validation"], {**cfg, "training_scope": scope})
    model.save(dest)
    write_json(dest / "training_history.json", hist)
    return model, hist


def _finetune_tft(pooled_dir: Path, target_prep: dict[str, Any], cfg: dict[str, Any], dest: Path):
    from send_bw.models.tft import TFTEarlyHistoryModel

    dest.mkdir(parents=True, exist_ok=True)
    model = TFTEarlyHistoryModel.load(pooled_dir)
    ft_cfg = dict(cfg)
    ft_cfg["finetune_lr"] = float((cfg.get("finetune") or {}).get("lr", cfg.get("lr", 1e-3)))
    ft_cfg["finetune_max_epochs"] = int((cfg.get("finetune") or {}).get("max_epochs", 8))
    ft_cfg["finetune_patience"] = int((cfg.get("finetune") or {}).get("patience", 4))
    ft_cfg["training_scope"] = "pooled_then_target_finetune"
    hist = model.finetune(target_prep["data"]["train"], target_prep["data"]["validation"], ft_cfg)
    model.save(dest)
    write_json(dest / "training_history.json", hist)
    return model, hist


def _score_model(name: str, model, animals: pd.DataFrame, pack: dict[str, Any],
                 hist: pd.DataFrame, fut: pd.DataFrame, *, label: str,
                 extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame, list]:
    t1 = time.time()
    pred, fails = quantile_table_from_model(model, animals, pack, hist)
    block = score_aligned_quantiles(fut, pred, representation="tft_quantiles")
    audit = crossing_audit(pred, fut, levels=COMMON_Q, animals=animals)
    block.update({
        "seconds": time.time() - t1,
        "uses_early_history": True,
        "label": label,
        "n_failures": len(fails),
        "failures": fails,
        "crossing_audit": audit,
    })
    if extra:
        block.update(extra)
    return block, pred, fails


def diagnose_locked_pooled_collapse(extend_split: Path, animals: pd.DataFrame | None = None,
                                    *, mae_flag: float = 1.0, dog_scale: float = 8.0) -> dict[str, Any]:
    """Read-only audit of the extra-split pooled TFT that reached MAE ≈ 0.550."""
    pred_path = extend_split / "extra_split_tft_pooled.csv"
    fut_path = extend_split / "target_obs.csv"
    ex_path = extend_split / "examples.csv"
    if not pred_path.is_file() or not fut_path.is_file():
        return {"status": "not_run", "reason": "locked extra-split predictions are missing"}
    pred = pd.read_csv(pred_path)
    fut = pd.read_csv(fut_path)
    mm = fut.merge(pred, on=["USUBJID", "DAY"], how="inner", suffixes=("", "_pred"))
    if "STUDYID" not in mm.columns and "STUDYID" in pred.columns:
        mm["STUDYID"] = pred["STUDYID"]
    y = mm["LOGW"].to_numpy(float)
    q50 = mm["q0.5"].to_numpy(float)
    mm = mm.assign(abs_err=np.abs(q50 - y), q50=q50, true=y)
    per_animal = mm.groupby(mm["USUBJID"].astype(str), sort=True).agg(
        mae=("abs_err", "mean"), q50_mean=("q50", "mean"), true_mean=("true", "mean"),
        n=("abs_err", "size")).reset_index()
    flagged = per_animal.loc[(per_animal["mae"] >= mae_flag) | (per_animal["q50_mean"] >= dog_scale)]
    examples = pd.read_csv(ex_path) if ex_path.is_file() else pd.DataFrame()
    if animals is not None and len(animals) and "USUBJID" in animals.columns:
        meta = animals.drop_duplicates("USUBJID").copy()
        meta["USUBJID"] = meta["USUBJID"].astype(str)
        keep = [c for c in ("USUBJID", "IS_CONTROL", "COMPOUND_ID", "SPECIES", "SEX", "STRAIN") if c in meta.columns]
        flagged = flagged.merge(meta[keep], on="USUBJID", how="left")
        per_animal = per_animal.merge(meta[keep], on="USUBJID", how="left")
    elif not examples.empty:
        keep = [c for c in ("USUBJID", "SPECIES", "SEX") if c in examples.columns]
        flagged = flagged.merge(examples[keep].drop_duplicates("USUBJID"), on="USUBJID", how="left")
    by_control = None
    if "IS_CONTROL" in per_animal.columns:
        tmp = per_animal.assign(ctrl=per_animal["IS_CONTROL"].astype(bool))
        by_control = {
            "n_control": int(tmp["ctrl"].sum()),
            "n_treated": int((~tmp["ctrl"]).sum()),
            "n_flagged_control": int(((tmp["ctrl"]) & (tmp["mae"] >= mae_flag)).sum()),
            "n_flagged_treated": int(((~tmp["ctrl"]) & (tmp["mae"] >= mae_flag)).sum()),
            "mean_mae_control": float(tmp.loc[tmp["ctrl"], "mae"].mean()) if tmp["ctrl"].any() else None,
            "mean_mae_treated": float(tmp.loc[~tmp["ctrl"], "mae"].mean()) if (~tmp["ctrl"]).any() else None,
        }
    norm_rec = None
    ckpt_params = extend_split / "models" / "extra_split_tft_pooled" / "dataset_parameters.pkl"
    if ckpt_params.is_file():
        try:
            import pytorch_forecasting  # noqa: F401
            params = pickle.loads(ckpt_params.read_bytes())
            norm_rec = normalizer_record(params.get("target_normalizer"))
        except Exception as e:  # noqa: BLE001
            norm_rec = {"status": "unreadable", "reason": f"{type(e).__name__}: {e}"}
    token_demo = {
        "missing_compound_cat_label": cat_label(np.nan),
        "missing_compound_legacy_astype_str": str(pd.Series([np.nan]).astype(str).iloc[0]),
        "confirmed_mismatch": cat_label(np.nan) != str(pd.Series([np.nan]).astype(str).iloc[0]),
    }
    mae_study = None
    if "STUDYID" in mm.columns:
        mae_study = float(mm.groupby(mm["STUDYID"].astype(str))["abs_err"].mean().mean())
    confirmed = [
        ("global_group_normalizer_mixed_species",
         bool(norm_rec and norm_rec.get("class") == "GroupNormalizer"
              and not norm_rec.get("groups"))),
        ("predict_time_nan_compound_token", token_demo["confirmed_mismatch"]),
        ("failures_are_control_rats_on_dog_scale",
         bool(by_control and by_control["n_flagged_control"] > 0
              and by_control["n_flagged_treated"] == 0
              and float(flagged["q50_mean"].median()) >= dog_scale if len(flagged) else False)),
    ]
    return {
        "status": "ok",
        "source": str(pred_path),
        "read_only": True,
        "n_pairs": int(len(mm)),
        "n_animals": int(per_animal["USUBJID"].nunique()),
        "study_balanced_mae": mae_study,
        "pooled_mae": float(mm["abs_err"].mean()) if len(mm) else None,
        "mean_q50": float(np.mean(q50)) if len(mm) else None,
        "mean_true": float(np.mean(y)) if len(mm) else None,
        "n_flagged_animals": int(len(flagged)),
        "flagged_animals": flagged.to_dict("records"),
        "by_control": by_control,
        "locked_target_normalizer": norm_rec,
        "token_demo": token_demo,
        "confirmed": [{"claim": k, "supported": bool(v)} for k, v in confirmed],
        "hypotheses": [
            "Unknown COMPOUND_ID embedding plus the control flag failed to apply a RAT offset "
            "in globally normalized LOGW, so outputs sat near dog-typical inverse scale (~9.1).",
            "Extra-split vs official robustness difference is unexplained: official pooled used "
            "the same global normalizer and nan token but did not collapse to MAE 0.55.",
        ],
        "not_confirmed": [
            "Checkpoint selection (last vs best) as the cause: extra-split val_loss was healthy.",
            "SPECIES encoder mis-mapping: locked encoder classes were DOG=1, RAT=2.",
        ],
        "note": "Diagnosis reads locked artifacts only. It does not rewrite them.",
    }


def run_validation_search(prep: dict[str, Any], base_cfg: dict[str, Any],
                          search_cfg: dict[str, Any], dest: Path) -> dict[str, Any]:
    """Sequential validation-led search. Does not score the official holdout test."""
    dest.mkdir(parents=True, exist_ok=True)
    chosen = dict(base_cfg)
    chosen["seed"] = int(search_cfg.get("seed", chosen.get("seed", 82001)))
    duration = search_cfg.get("duration") or {}
    chosen["max_epochs"] = int(duration.get("base_max_epochs", chosen.get("max_epochs", 16)))
    stages = []
    for spec in search_cfg.get("order") or []:
        key = str(spec["key"])
        rows = []
        for value in spec.get("values") or []:
            cfg = {**chosen, key: value}
            name = f"{key}_{_slug(value)}"
            t1 = time.time()
            _model, hist = _fit_tft(prep, cfg, dest / name, "target_only_search")
            row = {
                "key": key, "value": value, "path": str(dest / name),
                "best_val_loss": best_val_loss(hist),
                "curve": curve_still_improving(hist.get("curves") or []),
                "epochs_run": hist.get("epochs_run"),
                "seconds": time.time() - t1,
                "test_scores_examined": False,
            }
            rows.append(row)
        winner = select_by_val_loss(rows)
        chosen[key] = winner["value"]
        stages.append({"key": key, "candidates": rows, "selected": winner["value"],
                       "selected_val_loss": winner["best_val_loss"]})
    dur_row = None
    if duration.get("extend_if_val_still_improving"):
        probe = dest / f"duration_{int(chosen['max_epochs'])}"
        if not (probe / "training_history.json").exists():
            _model, hist = _fit_tft(prep, chosen, probe, "target_only_search")
        else:
            hist = read_json(probe / "training_history.json")
        decision = curve_still_improving(hist.get("curves") or [])
        extended = int(duration.get("extended_max_epochs", 24))
        dur_row = {"base": chosen["max_epochs"], "decision": decision, "extended": None}
        if decision.get("still_improving"):
            cfg = {**chosen, "max_epochs": extended, "patience": max(int(chosen.get("patience", 6)), 8)}
            _model, hist_ext = _fit_tft(prep, cfg, dest / f"duration_{extended}", "target_only_search")
            base_val = best_val_loss(hist)
            ext_val = best_val_loss(hist_ext)
            pick = extended if ext_val is not None and (base_val is None or ext_val < base_val) else chosen["max_epochs"]
            chosen["max_epochs"] = pick
            dur_row["extended"] = {
                "max_epochs": extended, "best_val_loss": ext_val,
                "base_val_loss": base_val, "selected": pick, "test_scores_examined": False,
            }
        else:
            dur_row["reason"] = "validation no longer improving; keep base duration"
    out = {
        "seed": chosen["seed"],
        "stages": stages,
        "duration": dur_row,
        "locked_config": {k: chosen.get(k) for k in (
            "lr", "hidden_size", "dropout", "max_epochs", "patience",
            "target_normalizer", "window_balance", "batch_size")},
        "test_scores_examined": False,
        "note": "Selection used validation loss only. Official holdout test was not read.",
    }
    write_json(dest / "search_log.json", out)
    return out


def _examples_for(test: ModelData, cutoff: float, max_enc: int, max_pred: int) -> dict[str, Any]:
    pack = construct_forecast_examples(
        test.obs, test.animals, cutoff=cutoff,
        max_encoder_length=max_enc, max_prediction_length=max_pred)
    hist, fut = examples_to_tables(pack)
    return {"pack": pack, "hist": hist, "fut": fut}


def _write_bank(dest: Path, pack: dict[str, Any], hist: pd.DataFrame, fut: pd.DataFrame,
                splits_target: pd.DataFrame, splits_pooled: pd.DataFrame | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    pack["examples"].to_csv(dest / "examples.csv", index=False)
    pack["exclusions"].to_csv(dest / "exclusions.csv", index=False)
    hist.to_csv(dest / "encoder_obs.csv", index=False)
    fut.to_csv(dest / "target_obs.csv", index=False)
    splits_target.to_csv(dest / "splits_target.csv", index=False)
    if splits_pooled is not None:
        splits_pooled.to_csv(dest / "splits_pooled.csv", index=False)


def execute_tft_forecast(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    pilot = Path(cfg.get("pilot_dir") or PILOT_DEFAULT)
    extend = Path(cfg.get("extend_dir") or EXTEND_DEFAULT)
    _refuse_locked(parent, pilot=pilot, extend=extend)
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise ForecastError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise ForecastError("design optimization stays off")
    if parent.exists() and any(parent.iterdir()) and not force:
        raise ForecastError(f"{parent} exists; pass --force to overwrite this unlocked directory")
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "predictions").mkdir(parents=True, exist_ok=True)
    (parent / "models").mkdir(parents=True, exist_ok=True)
    ev = cfg.get("benchmark") or {}
    cutoff = float(ev.get("encoder_window_day") or cfg.get("evaluation", {}).get("early_history_window_day") or 14)
    tft_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("tft") or {})
    max_enc = int(tft_cfg.get("max_encoder_length", 8))
    max_pred = int(tft_cfg.get("max_prediction_length", 6))
    train_seed = int(cfg.get("train_seed") or ev.get("train_seed") or 82001)
    cells = ev.get("source_cells") or {}
    target_dir = Path(cells.get("target_only") or "")
    pooled_dir = Path(cells.get("pooled") or "")
    extra_dir = Path(ev.get("extra_split_dir") or (extend / "banks" / "split_33001"))
    if not target_dir.is_dir() or not pooled_dir.is_dir():
        raise ForecastError("source_cells.target_only and source_cells.pooled must exist (read-only)")
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": cfg.get("generate_seed"),
        "train_seed": train_seed, "data_source": "existing_official_seq_cells_read_only",
        "target_only_cell": str(target_dir), "pooled_cell": str(pooled_dir),
        "encoder_window_day": cutoff, "max_encoder_length": max_enc,
        "max_prediction_length": max_pred, "chunk_long_horizons": False,
        "target_normalizer": tft_cfg.get("target_normalizer", "species_or_global"),
        "window_balance": tft_cfg.get("window_balance", "study"),
        "search_order": (cfg.get("search") or {}).get("order"),
        "stability_seeds": ev.get("stability_seeds"),
        "extra_split_seed": ev.get("extra_split_seed"),
        "official_holdout_test_used_for_search": False,
        "real_data": False, "design": False, "labels": list(LABELS),
        "preserved": list(LOCKED_DIRS), "writes_v3_locks": False,
        "writes_pilot": False, "writes_extend": False,
    }
    write_json(parent / "freeze_lock.json", lock)

    target = load_cell_prepared(target_dir)
    pooled = load_cell_prepared(pooled_dir)
    test = target["data"]["test"]
    tgt_test_ids = set(target["splits"].loc[target["splits"]["SPLIT"] == "test", "STUDYID"].astype(str))
    poo_test_ids = set(pooled["splits"].loc[pooled["splits"]["SPLIT"] == "test", "STUDYID"].astype(str))
    if tgt_test_ids != poo_test_ids:
        raise ForecastError(f"test STUDYIDs differ: {sorted(tgt_test_ids)} vs {sorted(poo_test_ids)}")

    inventory = {
        "target_only": {k: inventory_model_data(v) for k, v in target["data"].items()},
        "pooled": {k: inventory_model_data(v) for k, v in pooled["data"].items()},
        "schedules_test": summarize_observation_schedules(test.obs, test.animals),
        "schedules_target_train": summarize_observation_schedules(
            target["data"]["train"].obs, target["data"]["train"].animals),
        "schedules_pooled_train": summarize_observation_schedules(
            pooled["data"]["train"].obs, pooled["data"]["train"].animals),
    }
    write_json(parent / "data_inventory.json", inventory)

    official = _examples_for(test, cutoff, max_enc, max_pred)
    _write_bank(parent / "official", official["pack"], official["hist"], official["fut"],
                target["splits"], pooled["splits"])
    write_json(parent / "official" / "example_summary.json", {
        k: official["pack"][k] for k in official["pack"] if k not in {"examples", "exclusions"}
    })

    diagnosis = diagnose_locked_pooled_collapse(
        extra_dir, pd.concat([target["data"]["test"].animals, pooled["data"]["train"].animals],
                             ignore_index=True))
    write_json(parent / "diagnosis_extra_split_pooled.json", diagnosis)

    tft_cfg.setdefault("seed", train_seed)
    tft_cfg.setdefault("encoder_window_day", cutoff)
    tft_cfg.setdefault("target_normalizer", "species_or_global")
    tft_cfg.setdefault("window_balance", "study")
    search_cfg = cfg.get("search") or {}
    t0 = time.time()
    failures: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    preds: dict[str, pd.DataFrame] = {}
    train_info: dict[str, Any] = {}

    t1 = time.time()
    lv = last_value_predictions(official["hist"], official["fut"], QUANTILES)
    lv.to_csv(parent / "predictions" / "last_value.csv", index=False)
    results["last_value"] = score_aligned_quantiles(official["fut"], lv, representation="last_value")
    results["last_value"].update({
        "seconds": time.time() - t1, "uses_early_history": True,
        "label": "last observed encoder LOGW; degenerate quantiles; no covariates",
        "n_failures": 0,
    })
    preds["last_value"] = lv

    search = run_validation_search(target, tft_cfg, search_cfg, parent / "search")
    locked = dict(tft_cfg)
    locked.update({k: v for k, v in (search.get("locked_config") or {}).items() if v is not None})
    write_json(parent / "locked_tft_config.json", locked)

    def _eval(name: str, model, hist: dict[str, Any], dest: Path, label: str) -> None:
        nonlocal results, preds, failures, train_info
        train_info[name] = hist
        extra = {
            "checkpoint": str(dest / "tft.ckpt"),
            "training_history": {k: hist.get(k) for k in (
                "n_train_series", "n_val_series", "n_train_obs", "max_epochs",
                "encoder_window_day", "validation_windows", "training_windows",
                "target_normalizer", "add_target_scales", "cat_label",
                "window_balance", "finetuned", "finetune_lr",
                "finetune_species_classes_kept")},
        }
        block, pred, fails = _score_model(
            name, model, test.animals, official["pack"], official["hist"], official["fut"],
            label=label, extra=extra)
        pred.to_csv(parent / "predictions" / f"{name}.csv", index=False)
        results[name] = block
        preds[name] = pred
        failures.extend({"model": name, **f} for f in fails)

    try:
        m_to, h_to = _fit_tft(target, locked, parent / "models" / "tft_target_only", "target_only")
        _eval("tft_target_only", m_to, h_to, parent / "models" / "tft_target_only",
              "target-only TFT after validation-led search; SPECIES constant")
        verify = _verify_adapter(m_to, test.animals, test.obs, official["pack"],
                                 official["hist"], official["fut"])
        write_json(parent / "correctness.json", verify)
    except Exception as e:  # noqa: BLE001
        verify = {"ok": False, "reason": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-2000:]}
        write_json(parent / "correctness.json", verify)
        results["tft_target_only"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }
        failures.append({"model": "tft_target_only", "reason": str(e)})
        m_to = None

    try:
        m_po, h_po = _fit_tft(pooled, locked, parent / "models" / "tft_pooled", "pooled")
        _eval("tft_pooled", m_po, h_po, parent / "models" / "tft_pooled",
              "pooled TFT with species-or-global normalizer and UNK cat labels")
    except Exception as e:  # noqa: BLE001
        results["tft_pooled"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }
        failures.append({"model": "tft_pooled", "reason": str(e)})
        m_po = None

    try:
        if not (parent / "models" / "tft_pooled" / "tft.ckpt").exists():
            raise RuntimeError("pooled checkpoint missing; cannot fine-tune")
        ft_locked = dict(locked)
        ft_locked["finetune"] = cfg.get("finetune") or {}
        m_ft, h_ft = _finetune_tft(
            parent / "models" / "tft_pooled", target, ft_locked,
            parent / "models" / "tft_finetune")
        _eval("tft_finetune", m_ft, h_ft, parent / "models" / "tft_finetune",
              "pooled pretrain then target-species fine-tune; SPECIES classes kept")
    except Exception as e:  # noqa: BLE001
        results["tft_finetune"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }
        failures.append({"model": "tft_finetune", "reason": str(e)})

    stability_seeds = [int(s) for s in (ev.get("stability_seeds") or [82001, 82021, 82031])]
    stability_rows = []
    for seed in stability_seeds:
        try:
            scfg = {**locked, "seed": seed}
            dest = parent / "models" / f"tft_target_only_seed{seed}"
            if seed == int(locked.get("seed", train_seed)) and m_to is not None:
                model, hist = m_to, h_to
                dest = parent / "models" / "tft_target_only"
            else:
                model, hist = _fit_tft(target, scfg, dest, "target_only_stability")
            block, pred, fails = _score_model(
                f"tft_target_only_seed{seed}", model, test.animals, official["pack"],
                official["hist"], official["fut"],
                label=f"target-only TFT seed {seed} (post-lock stability)",
                extra={"seed": seed, "best_val_loss": best_val_loss(hist)})
            pred.to_csv(parent / "predictions" / f"tft_target_only_seed{seed}.csv", index=False)
            results[f"tft_target_only_seed{seed}"] = block
            failures.extend({"model": f"tft_target_only_seed{seed}", **f} for f in fails)
            stability_rows.append({
                "seed": seed,
                "best_val_loss": best_val_loss(hist),
                "mae_log": (block.get("mae_log") or {}).get("study_balanced"),
                "rmse_log": (block.get("rmse_log") or {}).get("study_balanced"),
                "mean_quantile_loss": (block.get("mean_quantile_loss") or {}).get("study_balanced"),
                "coverage80": (block.get("coverage80") or {}).get("study_balanced"),
                "crossing": (block.get("crossing_rate") or {}).get("pooled"),
            })
        except Exception as e:  # noqa: BLE001
            stability_rows.append({"seed": seed, "status": "failed", "reason": str(e)})
            failures.append({"model": f"tft_target_only_seed{seed}", "reason": str(e)})
    mae_vals = [r["mae_log"] for r in stability_rows if r.get("mae_log") is not None]
    stability = {
        "seeds": stability_seeds,
        "rows": stability_rows,
        "mae_mean": float(np.mean(mae_vals)) if mae_vals else None,
        "mae_sd": float(np.std(mae_vals, ddof=1)) if len(mae_vals) > 1 else None,
        "n_ok": len(mae_vals),
        "used_to_retune": False,
        "note": "Post-lock stability. Seeds were predeclared. Test errors did not change the locked config.",
    }
    write_json(parent / "stability.json", stability)

    extra_out: dict[str, Any] = {}
    extra_results: dict[str, Any] = {}
    try:
        extra_out = _run_extra_split(
            parent, target_dir, pooled_dir, extra_dir, locked, cfg, cutoff, max_enc, max_pred)
        extra_results = extra_out.get("results") or {}
    except Exception as e:  # noqa: BLE001
        extra_out = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()[-2000:]}
        failures.append({"model": "extra_split", "reason": str(e)})
    write_json(parent / "extra_split.json", extra_out)

    plots = []
    if preds:
        try:
            plot_preds = {k: v for k, v in preds.items() if k in {
                "last_value", "tft_target_only", "tft_pooled", "tft_finetune"}}
            plots = _plot_trajectories(parent, official["pack"], official["hist"], official["fut"],
                                       plot_preds, test.animals)
        except Exception as e:  # noqa: BLE001
            failures.append({"model": "plots", "reason": str(e)})

    comparison = [{"model": name, **_compact_score(block)} for name, block in results.items()]
    pd.DataFrame(comparison).to_csv(parent / "model_comparison.csv", index=False)
    per_study_rows = []
    for name, block in results.items():
        for rec in block.get("per_study") or []:
            per_study_rows.append({"model": name, "bank": "official", **rec})
    for name, block in extra_results.items():
        if isinstance(block, dict):
            for rec in block.get("per_study") or []:
                per_study_rows.append({"model": name, "bank": "extra_split_33001", **rec})
    if per_study_rows:
        pd.DataFrame(per_study_rows).to_csv(parent / "per_study_metrics.csv", index=False)

    out = {
        "labels": list(LABELS) + ["Primary TFT early-history forecasting workflow.",
                                  "Synthetic generator only."],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "example_summary": {k: official["pack"][k] for k in official["pack"]
                            if k not in {"examples", "exclusions"}},
        "inventory": {
            "test_studies": inventory["target_only"]["test"]["studyids"],
            "target_train_studies": inventory["target_only"]["train"]["studyids"],
            "pooled_train_studies": inventory["pooled"]["train"]["studyids"],
            "test_unique_days": inventory["schedules_test"]["unique_days_all"],
        },
        "diagnosis_extra_split_pooled": diagnosis,
        "search": {k: search.get(k) for k in search if k != "stages"} | {
            "n_stages": len(search.get("stages") or []),
            "locked_config": search.get("locked_config"),
        },
        "locked_config": locked,
        "correctness": read_json(parent / "correctness.json") if (parent / "correctness.json").exists() else {},
        "results": {k: _compact_score(v) if isinstance(v, dict) else v for k, v in results.items()},
        "stability": stability,
        "extra_split": {k: extra_out.get(k) for k in extra_out if k != "results"} | {
            "results": {k: _compact_score(v) if isinstance(v, dict) else v
                        for k, v in extra_results.items()},
        },
        "training": {k: {kk: vv for kk, vv in (v.items() if isinstance(v, dict) else [])
                         if kk != "curves"} for k, v in train_info.items()},
        "failures": failures,
        "plots": plots,
        "real_data_accessed": False,
        "design_run": False,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out, results, extra_results)
    return out


def _run_extra_split(parent: Path, target_dir: Path, pooled_dir: Path, extra_dir: Path,
                     locked: dict[str, Any], cfg: dict[str, Any], cutoff: float,
                     max_enc: int, max_pred: int) -> dict[str, Any]:
    target_full = load_full_cell(target_dir)
    pooled_full = load_full_cell(pooled_dir)
    target_splits = pd.read_csv(extra_dir / "splits_target.csv")
    pooled_splits = pd.read_csv(extra_dir / "splits_pooled.csv")
    extra_to = prepare_from_splits(
        target_full["full"], target_splits, species="RAT", sharing="context_specific")
    extra_po = prepare_from_splits(
        pooled_full["full"], pooled_splits, species="RAT", sharing="pooled")
    test = extra_to["data"]["test"]
    bank = _examples_for(test, cutoff, max_enc, max_pred)
    dest = parent / "extra_split_33001"
    _write_bank(dest, bank["pack"], bank["hist"], bank["fut"], target_splits, pooled_splits)
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    def _eval(name: str, model, hist: dict[str, Any], model_dir: Path, label: str) -> None:
        extra = {
            "checkpoint": str(model_dir / "tft.ckpt"),
            "best_val_loss": best_val_loss(hist),
            "target_normalizer": hist.get("target_normalizer"),
            "cat_label": hist.get("cat_label"),
            "finetuned": hist.get("finetuned"),
        }
        block, pred, fails = _score_model(
            name, model, test.animals, bank["pack"], bank["hist"], bank["fut"],
            label=label, extra=extra)
        pred.to_csv(dest / f"{name}.csv", index=False)
        results[name] = block
        failures.extend({"model": name, **f} for f in fails)

    m_to, h_to = _fit_tft(extra_to, locked, dest / "models" / "tft_target_only", "target_only")
    _eval("tft_target_only", m_to, h_to, dest / "models" / "tft_target_only",
          "extra-split target-only TFT with locked architecture")
    m_po, h_po = _fit_tft(extra_po, locked, dest / "models" / "tft_pooled", "pooled")
    _eval("tft_pooled", m_po, h_po, dest / "models" / "tft_pooled",
          "extra-split pooled TFT with species-or-global normalizer")
    ft_locked = dict(locked)
    ft_locked["finetune"] = cfg.get("finetune") or {}
    m_ft, h_ft = _finetune_tft(dest / "models" / "tft_pooled", extra_to, ft_locked,
                               dest / "models" / "tft_finetune")
    _eval("tft_finetune", m_ft, h_ft, dest / "models" / "tft_finetune",
          "extra-split pooled pretrain + target fine-tune")
    locked_pred = extra_dir / "extra_split_tft_pooled.csv"
    if locked_pred.is_file():
        old = pd.read_csv(locked_pred)
        old_block = score_aligned_quantiles(bank["fut"], old, representation="tft_quantiles")
        old_block.update({
            "label": "locked extra-split pooled TFT (read-only; MAE ~0.550 reference)",
            "uses_early_history": True,
            "read_only": True,
        })
        results["locked_tft_pooled_reference"] = old_block
    return {
        "status": "ok",
        "seed": 33001,
        "test_studies": sorted(set(test.animals["STUDYID"].astype(str))),
        "n_eligible": bank["pack"].get("n_eligible"),
        "results": results,
        "failures": failures,
        "splits_source": str(extra_dir),
        "note": "Same Generator A bank, predeclared extra split. Official holdout was not reused.",
    }


def _write_report(parent: Path, out: dict[str, Any], results: dict[str, Any],
                  extra_results: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    locked = out.get("locked_config") or {}
    diag = out.get("diagnosis_extra_split_pooled") or {}
    lines = [
        "# Primary TFT early-history forecasting",
        "",
        "Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.",
        "",
        "Forecast windows were taken from observed `DAY` values. Unmeasured days were "
        "not created. Long-horizon chunking was not used. Official holdout test errors "
        "were not used to select hyperparameters.",
        "",
        "## What ran",
        "",
        f"- output: `{out.get('output_dir')}`",
        f"- data: read-only official_seq cells (generate_seed={lock.get('generate_seed')})",
        f"- cutoff: DAY ≤ {lock.get('encoder_window_day')}",
        f"- encoder visit cap: {lock.get('max_encoder_length')}; "
        f"prediction-length cap: {lock.get('max_prediction_length')} (no chunking)",
        f"- target normalizer: `{lock.get('target_normalizer')}`",
        f"- window balance: `{lock.get('window_balance')}`",
        f"- locked after search: lr={locked.get('lr')}, hidden={locked.get('hidden_size')}, "
        f"dropout={locked.get('dropout')}, max_epochs={locked.get('max_epochs')}",
        f"- eligible official test animals: {(out.get('example_summary') or {}).get('n_eligible')}",
        f"- correctness: {(out.get('correctness') or {}).get('ok')}",
        f"- seconds: {out.get('seconds_total')}",
        "",
        "## Official holdout (study-balanced)",
        "",
        "| model | MAE log | RMSE log | mean pinball | cov80 | width80 | crossing | history |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, block in results.items():
        if str(name).startswith("tft_target_only_seed"):
            continue
        hist = "yes" if block.get("uses_early_history") else "no"
        cr = block.get("crossing_rate")
        cross = cr.get("pooled") if isinstance(cr, dict) else cr
        lines.append(
            f"| {name} | {_fmt(block, 'mae_log')} | {_fmt(block, 'rmse_log')} | "
            f"{_fmt(block, 'mean_quantile_loss')} | {_fmt(block, 'coverage80')} | "
            f"{_fmt(block, 'interval_width80')} | {cross} | {hist} |"
        )
    stab = out.get("stability") or {}
    lines += [
        "",
        "## Target-only stability (post-lock seeds)",
        "",
        f"- seeds: {stab.get('seeds')}",
        f"- MAE mean: {stab.get('mae_mean')}; MAE sd: {stab.get('mae_sd')}; n={stab.get('n_ok')}",
        f"- used to retune: {stab.get('used_to_retune')}",
        "",
        "## Extra-split 33001 (same bank, different studies)",
        "",
        "| model | MAE log | RMSE log | mean pinball | cov80 | crossing |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, block in extra_results.items():
        cr = block.get("crossing_rate") if isinstance(block, dict) else None
        cross = cr.get("pooled") if isinstance(cr, dict) else cr
        lines.append(
            f"| {name} | {_fmt(block, 'mae_log')} | {_fmt(block, 'rmse_log')} | "
            f"{_fmt(block, 'mean_quantile_loss')} | {_fmt(block, 'coverage80')} | {cross} |"
        )
    lines += [
        "",
        "## Locked pooled-collapse diagnosis",
        "",
        f"- source: `{diag.get('source')}` (read-only)",
        f"- study-balanced MAE on locked extra-split pooled: {diag.get('study_balanced_mae')}",
        f"- flagged animals: {diag.get('n_flagged_animals')}",
        f"- control breakdown: {diag.get('by_control')}",
        "",
    ]
    for rec in diag.get("confirmed") or []:
        lines.append(f"- confirmed `{rec.get('claim')}`: {rec.get('supported')}")
    for h in diag.get("hypotheses") or []:
        lines.append(f"- hypothesis: {h}")
    for h in diag.get("not_confirmed") or []:
        lines.append(f"- not confirmed: {h}")
    lines += [
        "",
        "## Scope",
        "",
        "Successful training and unit tests do not establish forecasting effectiveness. "
        "Numbers above are a synthetic-generator diagnostic on the evaluated settings only. "
        "They do not validate historical experimental studies and are not an operational recommendation.",
        "",
    ]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

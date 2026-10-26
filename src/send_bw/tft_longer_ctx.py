"""Longer TFT encoder context: observed pretreatment visits.

Default cohort selection drops PHASE=PRETREATMENT and baseline rows, so the
DAY<=14 encoder often has only weekly treatment visits (7, 14). This command
reattaches observed predose / baseline weights from canonical tables and
reruns the locked-knob comparison on the same larger-bank splits.

Same cutoff (DAY<=14) and the same post-cutoff targets. Recovery visits are
not added. Unmeasured days are not created. Official and new test errors are
not used to retune. Target-only remains the development reference.

Writes only runs/tft_longer_ctx. Locked run directories stay untouched.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

import ast
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.evaluation.quantile_forecast import QUANTILES, last_value_predictions
from send_bw.models.base import ModelData
from send_bw.sharing import LABELS
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
from send_bw.tft_examples import (
    inventory_model_data,
    score_aligned_quantiles,
    summarize_observation_schedules,
)
from send_bw.tft_larger_bank import (
    AUDIT_DEFAULT,
    COMPARISON_MODELS,
    REFERENCE,
    _examples_for,
    _rank_mae,
    _score_model,
    _write_bank,
    assert_locked_knobs,
)
from send_bw.tft_larger_bank import (
    OUTPUT_DEFAULT as LARGER_DEFAULT,
)
from send_bw.tft_primary import (
    OUTPUT_DEFAULT as PRIMARY_DEFAULT,
)
from send_bw.tft_primary import (
    _finetune_tft,
    _fit_tft,
)
from send_bw.tft_stability import EXTEND_DEFAULT, PILOT_DEFAULT
from send_bw.tft_stability import OUTPUT_DEFAULT as STABILITY_DEFAULT
from send_bw.tft_study_cv import OUTPUT_DEFAULT as STUDY_CV_DEFAULT
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_official_seq import collect_versions

OUTPUT_DEFAULT = "runs/tft_longer_ctx"
LOCKED_DIRS = tuple(V3_PRESERVED) + (
    PILOT_DEFAULT, EXTEND_DEFAULT, PRIMARY_DEFAULT, STABILITY_DEFAULT,
    STUDY_CV_DEFAULT, AUDIT_DEFAULT, LARGER_DEFAULT,
)
COMMON_Q = [0.1, 0.25, 0.5, 0.75, 0.9]


class LongerCtxError(ValueError):
    pass


def _refuse_locked(parent: Path) -> None:
    resolved = parent.resolve()
    blocked = {Path(p).resolve() for p in LOCKED_DIRS}
    if resolved in blocked:
        raise LongerCtxError("refusing to write into a preserved run directory")
    if resolved.name.startswith("v3_") and "runs" in resolved.parts:
        raise LongerCtxError("refusing to write a new locked-style runs/v3_* directory")
    if resolved.name in {
        "tft_early_hist_bench", "tft_early_hist_extend", "tft_primary",
        "tft_stability", "tft_study_cv", "forecast_method_audit", "tft_larger_bank",
    }:
        raise LongerCtxError("refusing to overwrite a completed TFT run directory")


def pretreatment_mask(obs: pd.DataFrame) -> pd.Series:
    """Observed predose / baseline rows only. Recovery is excluded."""
    phase = obs["PHASE"].astype(str).str.upper() if "PHASE" in obs.columns else pd.Series("", index=obs.index)
    baseline = obs["IS_BASELINE"].astype(bool) if "IS_BASELINE" in obs.columns else pd.Series(False, index=obs.index)
    day = obs["DAY"].to_numpy(float) if "DAY" in obs.columns else np.full(len(obs), np.nan)
    return (phase.eq("PRETREATMENT") | baseline | (day <= 0)) & phase.ne("RECOVERY")


def attach_pretreatment_obs(data: ModelData, canonical_obs: pd.DataFrame) -> tuple[ModelData, dict[str, Any]]:
    """Add observed pretreatment/baseline rows for animals already in `data`.

    Does not invent days, does not add recovery, does not change splits or animals.
    """
    if canonical_obs.empty or data.animals.empty:
        return data, {"n_rows_added": 0, "n_animals_with_extra": 0, "extra_days": []}
    ids = set(data.animals["USUBJID"].astype(str))
    extra = canonical_obs.copy()
    extra["USUBJID"] = extra["USUBJID"].astype(str)
    extra = extra.loc[extra["USUBJID"].isin(ids) & pretreatment_mask(extra)].copy()
    if extra.empty:
        return data, {"n_rows_added": 0, "n_animals_with_extra": 0, "extra_days": []}
    cur = data.obs.copy()
    cur["USUBJID"] = cur["USUBJID"].astype(str)
    have = set(zip(cur["USUBJID"].astype(str), cur["DAY"].to_numpy(float).round(6), strict=True))
    extra = extra.loc[[
        (str(u), float(d)) not in have
        for u, d in zip(extra["USUBJID"].astype(str), extra["DAY"].to_numpy(float).round(6), strict=True)
    ]].copy()
    if extra.empty:
        return data, {"n_rows_added": 0, "n_animals_with_extra": 0, "extra_days": []}
    meta = data.animals.drop_duplicates("USUBJID").copy()
    meta["USUBJID"] = meta["USUBJID"].astype(str)
    extra = extra.merge(meta[["USUBJID", "BASELINE_LOGW", "BASELINE_DAY"]], on="USUBJID", how="left")
    extra["Y"] = extra["LOGW"].to_numpy(float) - extra["BASELINE_LOGW"].to_numpy(float)
    cols = list(cur.columns)
    for c in cols:
        if c not in extra.columns:
            extra[c] = np.nan
    extra = extra[cols]
    out = pd.concat([cur, extra], ignore_index=True)
    out = out.drop_duplicates(["USUBJID", "DAY"], keep="first")
    out = out.sort_values(["USUBJID", "DAY"], kind="stable").reset_index(drop=True)
    rec = {
        "n_rows_added": int(len(extra)),
        "n_animals_with_extra": int(extra["USUBJID"].nunique()),
        "extra_days": [float(x) for x in sorted(extra["DAY"].unique())],
        "recovery_added": False,
        "invented_days": False,
    }
    return ModelData(data.animals, out, data.compound_ref, data.spec), rec


def augment_prep_with_pretreatment(prep: dict[str, Any], canonical_obs: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    data = {}
    notes = {}
    for split, md in (prep.get("data") or {}).items():
        if not isinstance(md, ModelData):
            data[split] = md
            continue
        attached, rec = attach_pretreatment_obs(md, canonical_obs)
        data[split] = attached
        notes[str(split)] = rec
    out = dict(prep)
    out["data"] = data
    out["pretreatment_encoder"] = True
    return out, notes


def encoder_day_inventory(examples: pd.DataFrame) -> dict[str, Any]:
    def _days(val: Any) -> tuple[float, ...]:
        if isinstance(val, (list, tuple)):
            return tuple(float(x) for x in val)
        text = str(val).strip()
        if text.startswith("["):
            return tuple(float(x) for x in ast.literal_eval(text))
        return (float(text),)

    if examples.empty or "encoder_days" not in examples.columns:
        return {"n_animals": 0, "n_encoder_used": {}, "patterns": []}
    used = examples["n_encoder_used"].value_counts().sort_index()
    patterns = Counter(_days(v) for v in examples["encoder_days"])
    return {
        "n_animals": int(len(examples)),
        "n_encoder_used": {str(int(k)): int(v) for k, v in used.items()},
        "mean_encoder_used": float(examples["n_encoder_used"].mean()),
        "patterns": [
            {"days": list(k), "n_animals": int(v)}
            for k, v in patterns.most_common()
        ],
        "unique_encoder_days": sorted({d for pat in patterns for d in pat}),
    }


def execute_tft_longer_ctx(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    _refuse_locked(parent)
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise LongerCtxError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise LongerCtxError("design optimization stays off")
    knobs = assert_locked_knobs(cfg)
    ev = cfg.get("longer_ctx") or {}
    if not ev.get("include_pretreatment_encoder", True):
        raise LongerCtxError("include_pretreatment_encoder must stay true for this command")
    if str(ev.get("development_reference") or REFERENCE) != REFERENCE:
        raise LongerCtxError("development_reference must stay tft_target_only")
    if ev.get("search"):
        raise LongerCtxError("hyperparameter search is off")
    if parent.exists() and any(parent.iterdir()) and not force:
        raise LongerCtxError(f"{parent} exists; pass --force to overwrite this unlocked directory")
    cells = ev.get("source_cells") or {}
    target_dir = Path(cells.get("target_only") or "")
    pooled_dir = Path(cells.get("pooled") or "")
    if not target_dir.is_dir() or not pooled_dir.is_dir():
        raise LongerCtxError("source_cells.target_only and source_cells.pooled must exist (read-only)")
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "predictions").mkdir(parents=True, exist_ok=True)
    (parent / "models").mkdir(parents=True, exist_ok=True)

    cutoff = float(ev.get("encoder_window_day") or 14)
    tft_cfg = dict(((cfg.get("models") or {}).get("optional") or {}).get("tft") or {})
    train_seed = int(cfg.get("train_seed") or 83011)
    tft_cfg.setdefault("seed", train_seed)
    tft_cfg.setdefault("encoder_window_day", cutoff)
    max_enc = int(tft_cfg.get("max_encoder_length", 8))
    max_pred = int(tft_cfg.get("max_prediction_length", 6))
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(),
        "generate_seed": cfg.get("generate_seed"), "train_seed": train_seed,
        "data_source": "larger_bank_cells_plus_observed_pretreatment",
        "include_pretreatment_encoder": True,
        "encoder_window_day": cutoff, "max_encoder_length": max_enc,
        "max_prediction_length": max_pred, "chunk_long_horizons": False,
        "locked_knobs": knobs,
        "search": False,
        "official_holdout_test_used_for_search": False,
        "new_test_used_for_search": False,
        "development_reference": REFERENCE,
        "finetune_promoted": False,
        "same_cutoff_and_targets": True,
        "real_data": False, "design": False, "labels": list(LABELS),
        "preserved": list(LOCKED_DIRS), "writes_locked_dirs": False,
        "target_only_cell": str(target_dir), "pooled_cell": str(pooled_dir),
    }
    write_json(parent / "freeze_lock.json", lock)
    write_json(parent / "locked_tft_config.json", {**tft_cfg, **knobs, "seed": train_seed})

    t0 = time.time()
    failures: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    preds: dict[str, pd.DataFrame] = {}
    train_info: dict[str, Any] = {}

    target = load_cell_prepared(target_dir)
    pooled = load_cell_prepared(pooled_dir)
    target_can = pd.read_parquet(target_dir / "canonical" / "bw_obs.parquet")
    pooled_can = pd.read_parquet(pooled_dir / "canonical" / "bw_obs.parquet")
    target, target_note = augment_prep_with_pretreatment(target, target_can)
    pooled, pooled_note = augment_prep_with_pretreatment(pooled, pooled_can)
    test = target["data"]["test"]
    tgt_test = set(target["splits"].loc[target["splits"]["SPLIT"] == "test", "STUDYID"].astype(str))
    poo_test = set(pooled["splits"].loc[pooled["splits"]["SPLIT"] == "test", "STUDYID"].astype(str))
    if tgt_test != poo_test:
        raise LongerCtxError(f"test STUDYIDs differ: {sorted(tgt_test)} vs {sorted(poo_test)}")

    short_ex = Path(ev.get("short_context_examples") or (Path(LARGER_DEFAULT) / "holdout" / "examples.csv"))
    short_inv = encoder_day_inventory(pd.read_csv(short_ex)) if short_ex.is_file() else {}

    examples = _examples_for(test, cutoff, max_enc, max_pred)
    long_inv = encoder_day_inventory(examples["pack"]["examples"])
    _write_bank(parent / "holdout", examples["pack"], examples["hist"], examples["fut"],
                target["splits"], pooled["splits"])
    write_json(parent / "holdout" / "example_summary.json", {
        k: examples["pack"][k] for k in examples["pack"] if k not in {"examples", "exclusions"}
    })
    write_json(parent / "encoder_inventory.json", {
        "short_treatment_only": short_inv,
        "long_with_pretreatment": long_inv,
        "attach_target_only": target_note,
        "attach_pooled": pooled_note,
        "note": "Same DAY<=14 cutoff. Extra encoder rows are observed pretreatment/baseline only.",
    })

    inventory = {
        "target_only": {k: inventory_model_data(v) for k, v in target["data"].items()},
        "pooled": {k: inventory_model_data(v) for k, v in pooled["data"].items()},
        "schedules_test": summarize_observation_schedules(test.obs, test.animals),
        "schedules_target_train": summarize_observation_schedules(
            target["data"]["train"].obs, target["data"]["train"].animals),
        "encoder": {"short": short_inv, "long": long_inv},
    }
    write_json(parent / "data_inventory.json", inventory)

    t1 = time.time()
    lv = last_value_predictions(examples["hist"], examples["fut"], QUANTILES)
    lv.to_csv(parent / "predictions" / "last_value.csv", index=False)
    results["last_value"] = score_aligned_quantiles(examples["fut"], lv, representation="last_value")
    results["last_value"].update({
        "seconds": time.time() - t1, "uses_early_history": True,
        "label": "last observed encoder LOGW after attaching pretreatment; usually still DAY 14",
        "n_failures": 0,
    })
    preds["last_value"] = lv

    def _eval(name: str, model, hist: dict[str, Any], dest: Path, label: str) -> None:
        nonlocal results, preds, failures, train_info
        train_info[name] = hist
        extra = {
            "checkpoint": str(dest / "tft.ckpt"),
            "training_history": {k: hist.get(k) for k in (
                "n_train_series", "n_val_series", "n_train_obs", "max_epochs",
                "encoder_window_day", "validation_windows", "training_windows",
                "target_normalizer", "window_balance", "finetuned")},
        }
        block, pred, fails = _score_model(
            model, test.animals, examples["pack"], examples["hist"], examples["fut"],
            label=label, extra=extra)
        pred.to_csv(parent / "predictions" / f"{name}.csv", index=False)
        results[name] = block
        preds[name] = pred
        failures.extend({"model": name, **f} for f in fails)

    try:
        m_to, h_to = _fit_tft(target, tft_cfg, parent / "models" / "tft_target_only", "target_only")
        _eval("tft_target_only", m_to, h_to, parent / "models" / "tft_target_only",
              "target-only TFT with observed pretreatment encoder; development reference")
        verify = _verify_adapter(m_to, test.animals, test.obs, examples["pack"],
                                 examples["hist"], examples["fut"])
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

    try:
        m_po, h_po = _fit_tft(pooled, tft_cfg, parent / "models" / "tft_pooled", "pooled")
        _eval("tft_pooled", m_po, h_po, parent / "models" / "tft_pooled",
              "pooled TFT with observed pretreatment encoder")
    except Exception as e:  # noqa: BLE001
        results["tft_pooled"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }
        failures.append({"model": "tft_pooled", "reason": str(e)})

    try:
        if not (parent / "models" / "tft_pooled" / "tft.ckpt").exists():
            raise RuntimeError("pooled checkpoint missing; cannot fine-tune")
        ft_locked = dict(tft_cfg)
        ft_locked["finetune"] = cfg.get("finetune") or {}
        m_ft, h_ft = _finetune_tft(
            parent / "models" / "tft_pooled", target, ft_locked,
            parent / "models" / "tft_finetune")
        _eval("tft_finetune", m_ft, h_ft, parent / "models" / "tft_finetune",
              "pooled pretrain then target fine-tune; pretreatment encoder; experimental")
    except Exception as e:  # noqa: BLE001
        results["tft_finetune"] = {
            "status": "failed", "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }
        failures.append({"model": "tft_finetune", "reason": str(e)})

    plots = []
    if preds:
        try:
            plot_preds = {k: v for k, v in preds.items() if k in COMPARISON_MODELS}
            plots = _plot_trajectories(parent, examples["pack"], examples["hist"], examples["fut"],
                                       plot_preds, test.animals)
        except Exception as e:  # noqa: BLE001
            failures.append({"model": "plots", "reason": str(e)})

    comparison = [{"model": name, **_compact_score(block)} for name, block in results.items()]
    pd.DataFrame(comparison).to_csv(parent / "model_comparison.csv", index=False)
    per_study_rows = []
    for name, block in results.items():
        for rec in block.get("per_study") or []:
            per_study_rows.append({"model": name, "bank": "longer_ctx_81011", **rec})
    if per_study_rows:
        pd.DataFrame(per_study_rows).to_csv(parent / "per_study_metrics.csv", index=False)

    paired = _paired_short_context(Path(ev.get("short_context_dir") or LARGER_DEFAULT), examples["fut"], results)
    write_json(parent / "paired_short_context.json", paired)

    compact = {k: _compact_score(v) if isinstance(v, dict) else v for k, v in results.items()}
    ranking = _rank_mae(compact)
    out = {
        "labels": list(LABELS) + [
            "Longer TFT encoder: observed pretreatment visits.",
            "Same DAY<=14 cutoff and same post-cutoff targets.",
            "Target-only is the development reference.",
        ],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "example_summary": {k: examples["pack"][k] for k in examples["pack"]
                            if k not in {"examples", "exclusions"}},
        "encoder_inventory": {"short": short_inv, "long": long_inv},
        "inventory": {
            "test_studies": inventory["target_only"]["test"]["studyids"],
            "test_n_animals": inventory["target_only"]["test"].get("n_animals"),
        },
        "locked_config": {**tft_cfg, **knobs, "seed": train_seed},
        "correctness": read_json(parent / "correctness.json") if (parent / "correctness.json").exists() else {},
        "results": compact,
        "ranking_mae_study_balanced": ranking,
        "paired_short_context": paired,
        "development_reference": REFERENCE,
        "finetune_promoted": False,
        "training": {k: {kk: vv for kk, vv in (v.items() if isinstance(v, dict) else [])
                         if kk != "curves"} for k, v in train_info.items()},
        "failures": failures,
        "plots": plots,
        "real_data_accessed": False,
        "design_run": False,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out, results)
    docs = Path("docs/v3/tft_longer_ctx_RESULTS.md")
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text((parent / "RESULTS.md").read_text(encoding="utf-8"), encoding="utf-8")
    return out


def _paired_short_context(short_dir: Path, fut: pd.DataFrame, results: dict[str, Any]) -> dict[str, Any]:
    """Compare this run to the treatment-only encoder on the same target pairs."""
    short_fut = short_dir / "holdout" / "target_obs.csv"
    if not short_fut.is_file():
        return {"status": "missing", "source": str(short_fut), "read_only": True}
    old = pd.read_csv(short_fut)
    keys = ["USUBJID", "DAY"]
    a = fut[keys].astype({"USUBJID": str, "DAY": float})
    b = old[keys].astype({"USUBJID": str, "DAY": float})
    merged = a.merge(b, on=keys, how="inner")
    rows = []
    for name in COMPARISON_MODELS:
        pred_path = short_dir / "predictions" / f"{name}.csv"
        block = results.get(name) or {}
        mae = (block.get("mae_log") or {}).get("study_balanced") if isinstance(block, dict) else None
        short_mae = None
        if pred_path.is_file():
            pred = pd.read_csv(pred_path)
            short_block = score_aligned_quantiles(old, pred, representation="tft_quantiles")
            short_mae = (short_block.get("mae_log") or {}).get("study_balanced")
        delta = None
        if mae is not None and short_mae is not None:
            delta = float(mae) - float(short_mae)
        rows.append({
            "model": name,
            "mae_long": mae,
            "mae_short": short_mae,
            "delta_long_minus_short": delta,
        })
    return {
        "status": "ok",
        "source": str(short_dir),
        "read_only": True,
        "n_shared_pairs": int(len(merged)),
        "n_long_pairs": int(len(a)),
        "n_short_pairs": int(len(b)),
        "same_target_pairs": int(len(merged)) == int(len(a)) == int(len(b)),
        "rows": rows,
        "used_to_retune": False,
        "note": "Negative delta means the longer encoder has lower study-balanced MAE.",
    }


def _write_report(parent: Path, out: dict[str, Any], results: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    locked = out.get("locked_config") or {}
    enc = out.get("encoder_inventory") or {}
    paired = out.get("paired_short_context") or {}
    rank = out.get("ranking_mae_study_balanced") or {}
    short = enc.get("short") or {}
    long = enc.get("long") or {}
    lines = [
        "# Longer-context TFT comparison",
        "",
        "Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.",
        "",
        "Observed pretreatment / baseline visits (`DAY` −7 and 0 in Generator A) "
        "are reattached as encoder context. The cutoff stays `DAY ≤ 14`. Targets "
        "are the same later observed visits. Recovery rows are not used. "
        "Unmeasured days are not created. Locked TFT knobs are unchanged. "
        "Target-only remains the development reference.",
        "",
        "## What ran",
        "",
        f"- output: `{out.get('output_dir')}`",
        f"- source cells (read-only): `{lock.get('target_only_cell')}`",
        f"- generate_seed={lock.get('generate_seed')}; train_seed={lock.get('train_seed')}",
        f"- cutoff: DAY ≤ {lock.get('encoder_window_day')}",
        f"- encoder visit cap: {lock.get('max_encoder_length')}",
        f"- locked knobs: lr={locked.get('lr')}, hidden={locked.get('hidden_size')}, "
        f"dropout={locked.get('dropout')}, max_epochs={locked.get('max_epochs')}",
        f"- search: {lock.get('search')}",
        f"- eligible animals: {(out.get('example_summary') or {}).get('n_eligible')}",
        f"- development reference: `{out.get('development_reference')}`",
        f"- fine-tune promoted: {out.get('finetune_promoted')}",
        f"- correctness: {(out.get('correctness') or {}).get('ok')}",
        f"- seconds: {out.get('seconds_total')}",
        "",
        "## Encoder length",
        "",
        f"- treatment-only (larger bank): mean {short.get('mean_encoder_used')} visits; "
        f"counts {short.get('n_encoder_used')}; days {short.get('unique_encoder_days')}",
        f"- with pretreatment: mean {long.get('mean_encoder_used')} visits; "
        f"counts {long.get('n_encoder_used')}; days {long.get('unique_encoder_days')}",
        "",
        "## Longer-context holdout (study-balanced)",
        "",
        "| model | MAE log | RMSE log | mean pinball | cov80 | width80 | IS80 | crossing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in COMPARISON_MODELS:
        block = results.get(name) or {}
        cr = block.get("crossing_rate")
        cross = cr.get("pooled") if isinstance(cr, dict) else cr
        role = " (reference)" if name == REFERENCE else ""
        lines.append(
            f"| {name}{role} | {_fmt(block, 'mae_log')} | {_fmt(block, 'rmse_log')} | "
            f"{_fmt(block, 'mean_quantile_loss')} | {_fmt(block, 'coverage80')} | "
            f"{_fmt(block, 'interval_width80')} | {_fmt(block, 'interval_score80')} | {cross} |"
        )
    lines += [
        "",
        f"- lowest study-balanced MAE: `{rank.get('lowest_mae')}`",
        f"- lowest MAE replaces reference: {rank.get('lowest_mae_replaces_reference')}",
        "",
        "## Paired vs treatment-only encoder (same targets)",
        "",
        f"- shared pairs: {paired.get('n_shared_pairs')}; same_target_pairs={paired.get('same_target_pairs')}",
        "",
        "| model | MAE short | MAE long | Δ long−short |",
        "|---|---:|---:|---:|",
    ]
    for rec in paired.get("rows") or []:
        lines.append(
            f"| {rec.get('model')} | {rec.get('mae_short')} | {rec.get('mae_long')} | "
            f"{rec.get('delta_long_minus_short')} |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "This is a synthetic-generator diagnostic. Including predose visits does not "
        "qualify the adapter on real SEND studies and is not an operational "
        "recommendation. Fine-tune stays experimental.",
        "",
    ]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

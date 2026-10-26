"""Larger Generator A bank and locked-knob TFT comparison.

Generates a new synthetic bank (not the official 71001 holdout) and reruns
last_value / target-only / pooled / fine-tune with the locked TFT knobs.
Writes only runs/tft_larger_bank. Locked run directories stay untouched.
Official and new test errors are not used to retune.

Target-only remains the development reference. Fine-tune is experimental
and is not promoted even if it wins on this bank.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from send_bw.evaluation.quantile_forecast import QUANTILES, last_value_predictions
from send_bw.sharing import LABELS
from send_bw.tft_crossings import crossing_audit
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
    construct_forecast_examples,
    examples_to_tables,
    inventory_model_data,
    quantile_table_from_model,
    score_aligned_quantiles,
    summarize_observation_schedules,
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
from send_bw.v3_research import _bank_cfg, _cell_cfg, _run_from_cfg

OUTPUT_DEFAULT = "runs/tft_larger_bank"
OFFICIAL_PRIMARY = Path(PRIMARY_DEFAULT)
AUDIT_DEFAULT = "runs/forecast_method_audit"
LOCKED_DIRS = tuple(V3_PRESERVED) + (
    PILOT_DEFAULT, EXTEND_DEFAULT, PRIMARY_DEFAULT, STABILITY_DEFAULT,
    STUDY_CV_DEFAULT, AUDIT_DEFAULT,
)
LOCKED_KNOB_KEYS = (
    "lr", "hidden_size", "dropout", "max_epochs", "patience", "batch_size",
    "window_balance", "target_normalizer",
)
EXPECTED_LOCKED = {
    "lr": 0.01,
    "hidden_size": 16,
    "dropout": 0.1,
    "max_epochs": 16,
    "patience": 6,
    "batch_size": 32,
    "window_balance": "study",
    "target_normalizer": "species_or_global",
}
COMPARISON_MODELS = ("last_value", "tft_target_only", "tft_pooled", "tft_finetune")
REFERENCE = "tft_target_only"
COMMON_Q = [0.1, 0.25, 0.5, 0.75, 0.9]
OFFICIAL_GENERATE_SEED = 71001


class LargerBankError(ValueError):
    pass


def _refuse_locked(parent: Path) -> None:
    resolved = parent.resolve()
    blocked = {Path(p).resolve() for p in LOCKED_DIRS}
    if resolved in blocked:
        raise LargerBankError("refusing to write into a preserved run directory")
    if resolved.name.startswith("v3_") and "runs" in resolved.parts:
        raise LargerBankError("refusing to write a new locked-style runs/v3_* directory")
    if resolved.name in {
        "tft_early_hist_bench", "tft_early_hist_extend", "tft_primary",
        "tft_stability", "tft_study_cv", "forecast_method_audit",
    }:
        raise LargerBankError("refusing to overwrite a completed TFT run directory")


def locked_tft_knobs(cfg: dict[str, Any]) -> dict[str, Any]:
    tft = ((cfg.get("models") or {}).get("optional") or {}).get("tft") or {}
    return {k: tft.get(k) for k in LOCKED_KNOB_KEYS}


def assert_locked_knobs(cfg: dict[str, Any]) -> dict[str, Any]:
    got = locked_tft_knobs(cfg)
    for key, expected in EXPECTED_LOCKED.items():
        if got.get(key) != expected:
            raise LargerBankError(
                f"TFT knob {key}={got.get(key)!r} differs from locked {expected!r}; "
                "this command does not search or retune")
    if (cfg.get("larger_bank") or {}).get("search"):
        raise LargerBankError("hyperparameter search is off for the larger-bank comparison")
    if cfg.get("search"):
        raise LargerBankError("search block is forbidden; knobs stay locked")
    return got


def official_reference(path: Path | None = None) -> dict[str, Any]:
    """Read-only official 12+3 comparison. Missing file is not an error."""
    src = path or (OFFICIAL_PRIMARY / "SUMMARY.json")
    if not src.is_file():
        return {"status": "missing", "source": str(src), "read_only": True, "results": {}}
    data = read_json(src)
    results = data.get("results") or {}
    return {
        "status": "ok",
        "source": str(src),
        "read_only": True,
        "generate_seed": OFFICIAL_GENERATE_SEED,
        "n_scored": ((results.get("tft_target_only") or {}).get("n_scored")),
        "results": {k: results[k] for k in COMPARISON_MODELS if k in results},
        "used_to_retune": False,
        "note": "Official 12 RAT + 3 DOG bank, generate_seed=71001. Different studies.",
    }


def _examples_for(test, cutoff: float, max_enc: int, max_pred: int) -> dict[str, Any]:
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


def _score_model(model, animals: pd.DataFrame, pack: dict[str, Any],
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


def execute_tft_larger_bank(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or OUTPUT_DEFAULT)
    _refuse_locked(parent)
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise LargerBankError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise LargerBankError("design optimization stays off")
    gen0 = int(cfg["generate_seed"])
    tr0 = int(cfg["train_seed"])
    if gen0 == OFFICIAL_GENERATE_SEED:
        raise LargerBankError("generate_seed 71001 is the official bank; use a new seed")
    if gen0 == tr0:
        raise LargerBankError("generate_seed and train_seed must differ")
    knobs = assert_locked_knobs(cfg)
    ev = cfg.get("larger_bank") or {}
    if str(ev.get("development_reference") or REFERENCE) != REFERENCE:
        raise LargerBankError("development_reference must stay tft_target_only")
    if parent.exists() and any(parent.iterdir()) and not force:
        raise LargerBankError(f"{parent} exists; pass --force to overwrite this unlocked directory")
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "predictions").mkdir(parents=True, exist_ok=True)
    (parent / "models").mkdir(parents=True, exist_ok=True)

    cutoff = float(ev.get("encoder_window_day") or 14)
    tft_cfg = dict(((cfg.get("models") or {}).get("optional") or {}).get("tft") or {})
    tft_cfg.setdefault("seed", tr0)
    tft_cfg.setdefault("encoder_window_day", cutoff)
    max_enc = int(tft_cfg.get("max_encoder_length", 8))
    max_pred = int(tft_cfg.get("max_prediction_length", 6))
    n_donor = int(ev.get("n_donor_studies") or 6)
    bank_name = str(ev.get("bank") or "matched")
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": gen0, "train_seed": tr0,
        "official_generate_seed": OFFICIAL_GENERATE_SEED,
        "data_source": "new_generator_a_bank",
        "n_studies": (cfg.get("synthetic") or {}).get("n_studies"),
        "n_other_species_studies": (cfg.get("synthetic") or {}).get("n_other_species_studies"),
        "n_compounds": (cfg.get("synthetic") or {}).get("n_compounds"),
        "n_per_sex": (cfg.get("synthetic") or {}).get("n_per_sex"),
        "n_donor_studies": n_donor,
        "encoder_window_day": cutoff, "max_encoder_length": max_enc,
        "max_prediction_length": max_pred, "chunk_long_horizons": False,
        "locked_knobs": knobs,
        "locked_knob_source": str(cfg.get("locked_tft_config") or (OFFICIAL_PRIMARY / "locked_tft_config.json")),
        "search": False,
        "official_holdout_test_used_for_search": False,
        "new_test_used_for_search": False,
        "development_reference": REFERENCE,
        "finetune_promoted": False,
        "real_data": False, "design": False, "labels": list(LABELS),
        "preserved": list(LOCKED_DIRS), "writes_locked_dirs": False,
    }
    write_json(parent / "freeze_lock.json", lock)
    write_json(parent / "locked_tft_config.json", {**tft_cfg, **knobs, "seed": tr0})

    t0 = time.time()
    failures: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    preds: dict[str, pd.DataFrame] = {}
    train_info: dict[str, Any] = {}

    bcfg = _bank_cfg(cfg, parent, 0, gen0, bank_name)
    br = _run_from_cfg(bcfg)
    br.generate(force=force)

    cells: dict[str, Any] = {}
    hashes: dict[str, Any] = {}
    reuse = None
    for scope, sharing, k in (("target_only", "context_specific", 0),
                              ("pooled", "pooled", n_donor)):
        spec = {
            "name": f"d0_t0_matched_{scope}", "dataset_index": 0, "train_index": 0,
            "bank": bank_name, "sharing": sharing, "k": k,
            "sparse_target_train_n": None, "generate_seed": gen0, "train_seed": tr0,
            "models": [],
        }
        ccfg = _cell_cfg(cfg, parent, spec, generate_seed=gen0, train_seed=tr0,
                         models=[], sparse_target_train_n=None, reuse_target_from=reuse)
        ccfg["evaluation"]["tasks"] = ["early_history"]
        ccfg["evaluation"]["early_history_window_day"] = cutoff
        ccfg["design"] = {}
        run = _run_from_cfg(ccfg)
        run.ingest(force=force)
        run.prepare(force=force)
        cells[scope] = run
        man = read_json(run.p("splits", "target_manifest.json"))
        hashes[scope] = {n: (man.get(n) or {}).get("hash") for n in ("validation", "test", "all_target")}
        if reuse is None:
            reuse = str(run.p("splits", "x").parent)

    if hashes["target_only"]["test"] != hashes["pooled"]["test"]:
        raise LargerBankError("target-study test holdout hashes differ between scopes")
    if hashes["target_only"]["validation"] != hashes["pooled"]["validation"]:
        raise LargerBankError("target-study validation holdout hashes differ between scopes")

    target = load_cell_prepared(Path(cells["target_only"].dir))
    pooled = load_cell_prepared(Path(cells["pooled"].dir))
    test = target["data"]["test"]
    inventory = {
        "target_only": {k: inventory_model_data(v) for k, v in target["data"].items()},
        "pooled": {k: inventory_model_data(v) for k, v in pooled["data"].items()},
        "schedules_test": summarize_observation_schedules(test.obs, test.animals),
        "schedules_target_train": summarize_observation_schedules(
            target["data"]["train"].obs, target["data"]["train"].animals),
        "schedules_pooled_train": summarize_observation_schedules(
            pooled["data"]["train"].obs, pooled["data"]["train"].animals),
        "split_hashes": hashes,
    }
    write_json(parent / "data_inventory.json", inventory)

    examples = _examples_for(test, cutoff, max_enc, max_pred)
    _write_bank(parent / "holdout", examples["pack"], examples["hist"], examples["fut"],
                target["splits"], pooled["splits"])
    write_json(parent / "holdout" / "example_summary.json", {
        k: examples["pack"][k] for k in examples["pack"] if k not in {"examples", "exclusions"}
    })

    t1 = time.time()
    lv = last_value_predictions(examples["hist"], examples["fut"], QUANTILES)
    lv.to_csv(parent / "predictions" / "last_value.csv", index=False)
    results["last_value"] = score_aligned_quantiles(examples["fut"], lv, representation="last_value")
    results["last_value"].update({
        "seconds": time.time() - t1, "uses_early_history": True,
        "label": "last observed encoder LOGW; degenerate quantiles; no covariates",
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
                "target_normalizer", "add_target_scales", "cat_label",
                "window_balance", "finetuned", "finetune_lr",
                "finetune_species_classes_kept")},
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
              "target-only TFT with locked knobs; SPECIES constant; development reference")
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
              "pooled TFT with species-or-global normalizer and UNK cat labels")
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
              "pooled pretrain then target-species fine-tune; experimental; not promoted")
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
            per_study_rows.append({"model": name, "bank": "larger_81011", **rec})
    if per_study_rows:
        pd.DataFrame(per_study_rows).to_csv(parent / "per_study_metrics.csv", index=False)

    official = official_reference(Path(cfg.get("official_primary_dir") or PRIMARY_DEFAULT) / "SUMMARY.json")
    write_json(parent / "official_reference.json", official)

    compact = {k: _compact_score(v) if isinstance(v, dict) else v for k, v in results.items()}
    ranking = _rank_mae(compact)
    out = {
        "labels": list(LABELS) + [
            "Larger Generator A bank TFT comparison.",
            "Synthetic generator only.",
            "Target-only is the development reference.",
            "Fine-tune is experimental and not promoted.",
        ],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "example_summary": {k: examples["pack"][k] for k in examples["pack"]
                            if k not in {"examples", "exclusions"}},
        "inventory": {
            "test_studies": inventory["target_only"]["test"]["studyids"],
            "target_train_studies": inventory["target_only"]["train"]["studyids"],
            "pooled_train_studies": inventory["pooled"]["train"]["studyids"],
            "validation_studies": inventory["target_only"]["validation"]["studyids"],
            "test_n_animals": inventory["target_only"]["test"].get("n_animals"),
            "test_n_obs": inventory["target_only"]["test"].get("n_obs"),
            "test_unique_days": inventory["schedules_test"]["unique_days_all"],
        },
        "locked_config": {**tft_cfg, **knobs, "seed": tr0},
        "correctness": read_json(parent / "correctness.json") if (parent / "correctness.json").exists() else {},
        "results": compact,
        "ranking_mae_study_balanced": ranking,
        "development_reference": REFERENCE,
        "finetune_promoted": False,
        "official_reference": official,
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
    docs = Path("docs/v3/tft_larger_bank_RESULTS.md")
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text((parent / "RESULTS.md").read_text(encoding="utf-8"), encoding="utf-8")
    return out


def _rank_mae(results: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for name in COMPARISON_MODELS:
        block = results.get(name) or {}
        mae = block.get("mae_log")
        val = mae.get("study_balanced") if isinstance(mae, dict) else None
        if val is not None:
            rows.append({"model": name, "mae_log_study_balanced": float(val)})
    rows.sort(key=lambda r: r["mae_log_study_balanced"])
    winner = rows[0]["model"] if rows else None
    return {
        "order": rows,
        "lowest_mae": winner,
        "development_reference": REFERENCE,
        "lowest_mae_replaces_reference": False,
        "note": "Lowest MAE does not change the development reference.",
    }


def _write_report(parent: Path, out: dict[str, Any], results: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    locked = out.get("locked_config") or {}
    official = out.get("official_reference") or {}
    inv = out.get("inventory") or {}
    rank = out.get("ranking_mae_study_balanced") or {}
    lines = [
        "# Larger-bank TFT model comparison",
        "",
        "Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.",
        "",
        "This is a **new** Generator A bank (`generate_seed=81011`), not the official "
        "12 RAT + 3 DOG holdout (`generate_seed=71001`). Forecast windows use observed "
        "`DAY` values. Unmeasured days were not created. Long-horizon chunking was not "
        "used. Official and new test errors were not used to select hyperparameters. "
        "Target-only remains the development reference. Fine-tune is experimental and "
        "is not promoted even if it has lower MAE here.",
        "",
        "## What ran",
        "",
        f"- output: `{out.get('output_dir')}`",
        f"- generator: A; n_studies={lock.get('n_studies')} RAT + "
        f"{lock.get('n_other_species_studies')} DOG; n_compounds={lock.get('n_compounds')}; "
        f"n_per_sex={lock.get('n_per_sex')}",
        f"- generate_seed={lock.get('generate_seed')}; train_seed={lock.get('train_seed')}",
        f"- donors: {lock.get('n_donor_studies')} (pooled only)",
        f"- cutoff: DAY ≤ {lock.get('encoder_window_day')}",
        f"- encoder visit cap: {lock.get('max_encoder_length')}; "
        f"prediction-length cap: {lock.get('max_prediction_length')} (no chunking)",
        f"- locked knobs: lr={locked.get('lr')}, hidden={locked.get('hidden_size')}, "
        f"dropout={locked.get('dropout')}, max_epochs={locked.get('max_epochs')}, "
        f"window_balance={locked.get('window_balance')}, "
        f"target_normalizer={locked.get('target_normalizer')}",
        f"- search: {lock.get('search')}",
        f"- test studies: {inv.get('test_studies')}",
        f"- validation studies: {inv.get('validation_studies')}",
        f"- target train studies: {inv.get('target_train_studies')}",
        f"- eligible test animals: {(out.get('example_summary') or {}).get('n_eligible')}",
        f"- development reference: `{out.get('development_reference')}`",
        f"- fine-tune promoted: {out.get('finetune_promoted')}",
        f"- correctness: {(out.get('correctness') or {}).get('ok')}",
        f"- seconds: {out.get('seconds_total')}",
        "",
        "## Larger bank (study-balanced)",
        "",
        "| model | MAE log | RMSE log | mean pinball | cov80 | width80 | IS80 | crossing | history |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in COMPARISON_MODELS:
        block = results.get(name) or {}
        hist = "yes" if block.get("uses_early_history") else "no"
        cr = block.get("crossing_rate")
        cross = cr.get("pooled") if isinstance(cr, dict) else cr
        role = " (reference)" if name == REFERENCE else ""
        lines.append(
            f"| {name}{role} | {_fmt(block, 'mae_log')} | {_fmt(block, 'rmse_log')} | "
            f"{_fmt(block, 'mean_quantile_loss')} | {_fmt(block, 'coverage80')} | "
            f"{_fmt(block, 'interval_width80')} | {_fmt(block, 'interval_score80')} | "
            f"{cross} | {hist} |"
        )
    lines += [
        "",
        f"- lowest study-balanced MAE: `{rank.get('lowest_mae')}`",
        f"- lowest MAE replaces reference: {rank.get('lowest_mae_replaces_reference')}",
        "",
        "## Official 12+3 bank (read-only reference, generate_seed=71001)",
        "",
        "Different studies and a different generate seed. Same locked knobs. "
        "Not a paired animal comparison.",
        "",
        "| model | MAE log | RMSE log | mean pinball | cov80 | width80 | n_scored |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    off_res = official.get("results") or {}
    for name in COMPARISON_MODELS:
        block = off_res.get(name) or {}
        n = block.get("n_scored")
        lines.append(
            f"| {name} | {_fmt(block, 'mae_log')} | {_fmt(block, 'rmse_log')} | "
            f"{_fmt(block, 'mean_quantile_loss')} | {_fmt(block, 'coverage80')} | "
            f"{_fmt(block, 'interval_width80')} | {n} |"
        )
    lines += [
        "",
        f"- official source: `{official.get('source')}` (read_only={official.get('read_only')})",
        f"- used to retune: {official.get('used_to_retune')}",
        "",
        "## Scope",
        "",
        "Successful training and unit tests do not establish forecasting effectiveness. "
        "Numbers above are a synthetic-generator diagnostic on the evaluated settings only. "
        "They do not validate historical experimental studies and are not an operational "
        "recommendation. Sharing and the transformer are not required to win. Fine-tune "
        "and CQR stay experimental.",
        "",
    ]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

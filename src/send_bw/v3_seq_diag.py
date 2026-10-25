"""Bounded official-sequence diagnostic.

Phase A reads the frozen official-seq holdout. Phase B trains on fresh
synthetic replicates. Early-history only. No predose mix-in. No design.
PR #10 artifacts are never overwritten. Settings are not retuned after test.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.evaluation.quantile_forecast import (
    QUANTILES,
    encoder_time_report,
    last_value_predictions,
    paired_transfer,
    score_quantile_table,
    split_encoder_future,
)
from send_bw.historical_audit import discover_authorization
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_official_seq import (
    _reload_check,
    _score_gaussian,
    _score_quantiles,
    collect_versions,
)
from send_bw.v3_research import _bank_cfg, _cell_cfg, _run_from_cfg, load_yaml

PRESERVED = (
    "runs/v3_eval", "runs/v3_pilot_k0k1", "runs/v3_repl", "runs/v3_research",
    "runs/v3_closeout", "runs/v3_audit", "runs/v3_increment", "runs/v3_phaseb",
    "runs/v3_limitations", "runs/v3_objective", "runs/v3_official_seq",
)
WINDOW = 14.0
GEN_STRIDE = 1009
TRAIN_STRIDE = 17


class SeqDiagError(ValueError):
    pass


def chronos2_checkpoint_revision(pretrained_id: str = "amazon/chronos-2") -> dict[str, Any]:
    """Exact Hugging Face revision. Network optional; local config is a fallback."""
    out: dict[str, Any] = {
        "pretrained_id": pretrained_id,
        "sha": None,
        "last_modified": None,
        "source": None,
        "forecast_type": "QUANTILES",
        "official_mean_is_median": True,
        "source_note": "chronos2/pipeline.py: 'the median is returned as the mean here'",
    }
    try:
        from huggingface_hub import model_info

        info = model_info(pretrained_id)
        out["sha"] = getattr(info, "sha", None)
        out["last_modified"] = str(getattr(info, "lastModified", None))
        out["pipeline_tag"] = getattr(info, "pipeline_tag", None)
        out["source"] = "huggingface_hub.model_info"
    except Exception as e:  # noqa: BLE001
        out["hub_error"] = f"{type(e).__name__}: {e}"
    local = Path("runs/v3_official_seq/models/chronos2_zero_shot/pipeline/config.json")
    if not local.exists():
        local = Path("runs/v3_official_seq/hf_cache")
    try:
        if Path("runs/v3_official_seq/models/chronos2_zero_shot/pipeline/config.json").exists():
            cfg = json.loads(
                Path("runs/v3_official_seq/models/chronos2_zero_shot/pipeline/config.json")
                .read_text(encoding="utf-8"))
            out["local_config_keys"] = sorted(cfg.keys())[:30]
            chronos_cfg = cfg.get("chronos_config") or cfg
            out["local_context_length"] = chronos_cfg.get("context_length")
            out["local_quantiles"] = chronos_cfg.get("quantiles")
            if out["source"] is None:
                out["source"] = "local_pipeline_config"
    except Exception as e:  # noqa: BLE001
        out["local_config_error"] = f"{type(e).__name__}: {e}"
    return out


def _chronos_covariate_check() -> dict[str, Any]:
    return {
        "api_accepts_past_covariates": True,
        "api_accepts_future_covariates": True,
        "official_seq_input": "univariate LOGW ndarray",
        "unused": ["SEX", "STRAIN", "ROUTE", "DOSE", "SPECIES", "DAY", "COMPOUND_ID"],
        "added_in_this_diagnostic": False,
        "note": ("Chronos-2 can take covariate dicts. PR #10 and this diagnostic "
                 "do not add them. That is a recorded limitation, not a retune target."),
    }


def _score_last_value(data, window_day: float, pred_path: Path | None = None) -> dict[str, Any]:
    hist, fut = split_encoder_future(data, window_day)
    pred = last_value_predictions(hist, fut, QUANTILES)
    if pred_path is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(pred_path, index=False)
    out = score_quantile_table(fut, pred, quantiles=QUANTILES, representation="last_value")
    out["n_skipped_no_encoder"] = 0
    out["encoder_fabricated"] = False
    out["model_class"] = "last_value_baseline"
    out["point_definition"]["point"] = "last_observed_encoder_LOGW"
    out["point_definition"]["column"] = "last_observed_encoder_LOGW"
    return out


def _chronos_scale_check(test, window_day: float) -> dict[str, Any]:
    """Zero-shot LOGW vs exp(LOGW)→log-back. Diagnostic only; official encoding stays LOGW."""
    from send_bw.models.chronos2 import Chronos2EarlyHistoryModel

    dest = Path("runs/v3_official_seq/models/chronos2_zero_shot")
    if not dest.exists():
        return {"status": "not_run", "reason": "frozen zero-shot checkpoint missing"}
    model = Chronos2EarlyHistoryModel.load(dest)
    hist, fut = split_encoder_future(test, window_day)
    log_pred = []
    raw_pred = []
    skipped = 0
    mean_is_median = []
    for u, fg in fut.groupby("USUBJID", sort=False):
        hh = hist.loc[hist["USUBJID"] == u].sort_values("DAY")
        if hh.empty:
            skipped += 1
            continue
        times = fg.sort_values("DAY")["DAY"].to_numpy(float)
        enc_log = hh["LOGW"].to_numpy(float)
        qs_log, mean_log = model.pipeline.predict_quantiles(
            [enc_log], prediction_length=int(len(times)), quantile_levels=list(QUANTILES))
        arr_log = np.asarray(qs_log[0].detach().cpu() if hasattr(qs_log[0], "detach") else qs_log[0], float)
        if arr_log.ndim == 3:
            arr_log = arr_log[0]
        m_log = np.asarray(mean_log[0].detach().cpu() if hasattr(mean_log[0], "detach") else mean_log[0], float)
        if m_log.ndim == 2:
            m_log = m_log[0]
        q50_idx = list(QUANTILES).index(0.5)
        n = min(len(times), arr_log.shape[0], len(m_log))
        mean_is_median.append(bool(np.allclose(m_log[:n], arr_log[:n, q50_idx], atol=1e-6)))
        for j in range(n):
            row = {"USUBJID": u, "DAY": float(times[j]), "STUDYID": fg["STUDYID"].iloc[0]}
            for k, q in enumerate(QUANTILES):
                row[f"q{q}"] = float(arr_log[j, k])
            log_pred.append(row)
        enc_raw = np.exp(enc_log)
        qs_raw, _ = model.pipeline.predict_quantiles(
            [enc_raw], prediction_length=int(len(times)), quantile_levels=list(QUANTILES))
        arr_raw = np.asarray(qs_raw[0].detach().cpu() if hasattr(qs_raw[0], "detach") else qs_raw[0], float)
        if arr_raw.ndim == 3:
            arr_raw = arr_raw[0]
        n = min(len(times), arr_raw.shape[0])
        for j in range(n):
            row = {"USUBJID": u, "DAY": float(times[j]), "STUDYID": fg["STUDYID"].iloc[0]}
            for k, q in enumerate(QUANTILES):
                row[f"q{q}"] = float(np.log(np.maximum(arr_raw[j, k], 1e-12)))
            raw_pred.append(row)
    scored_log = score_quantile_table(
        fut, pd.DataFrame(log_pred), quantiles=QUANTILES, representation="chronos2_quantiles")
    scored_raw = score_quantile_table(
        fut, pd.DataFrame(raw_pred), quantiles=QUANTILES, representation="chronos2_quantiles")
    return {
        "status": "ok",
        "n_skipped_no_encoder": skipped,
        "official_encoding": "LOGW",
        "official_encoding_changed": False,
        "mean_equals_median_all_series": bool(mean_is_median) and all(mean_is_median),
        "n_series_checked": len(mean_is_median),
        "logw_input": {
            "rmse_log": (scored_log.get("rmse_log") or {}).get("study_balanced"),
            "mean_quantile_loss": (scored_log.get("mean_quantile_loss") or {}).get("study_balanced"),
        },
        "exp_logw_input_logged_back": {
            "rmse_log": (scored_raw.get("rmse_log") or {}).get("study_balanced"),
            "mean_quantile_loss": (scored_raw.get("mean_quantile_loss") or {}).get("study_balanced"),
            "note": "Diagnostic only. Not used to replace the official LOGW encoding.",
        },
    }


def _rescore_saved(official: Path, test, window: float, dest: Path) -> dict[str, Any]:
    hist, fut = split_encoder_future(test, window)
    pred_dir = official / "predictions"
    out: dict[str, Any] = {}
    mapping = {
        "spline_target_only": "gaussian",
        "spline_pooled": "gaussian",
        "transformer_target_only": "gaussian",
        "transformer_pooled": "gaussian",
        "tft_target_only": "tft_quantiles",
        "tft_pooled": "tft_quantiles",
        "chronos2_zero_shot": "chronos2_quantiles",
        "chronos2_target_only": "chronos2_quantiles",
        "chronos2_pooled": "chronos2_quantiles",
    }
    dest.mkdir(parents=True, exist_ok=True)
    for key, rep in mapping.items():
        path = pred_dir / f"{key}.csv"
        if not path.exists():
            out[key] = {"status": "not_run", "reason": f"missing {path}"}
            continue
        pred = pd.read_csv(path)
        sc = score_quantile_table(fut, pred, quantiles=QUANTILES, representation=rep)
        write_json(dest / f"{key}.json", sc)
        out[key] = {
            "status": sc.get("status"),
            "rmse_log": (sc.get("rmse_log") or {}).get("study_balanced"),
            "mean_quantile_loss": (sc.get("mean_quantile_loss") or {}).get("study_balanced"),
            "coverage80": (sc.get("coverage80") or {}).get("study_balanced"),
            "interval_width80": (sc.get("interval_width80") or {}).get("study_balanced"),
            "crps_justified": sc.get("crps_justified"),
            "point_definition": sc.get("point_definition"),
            "per_horizon": sc.get("per_horizon"),
            "n_scored": sc.get("n_scored"),
            "n_studies": len(sc.get("per_study") or []),
        }
    lv = _score_last_value(test, window, dest / "last_value.csv")
    write_json(dest / "last_value.json", lv)
    out["last_value"] = {
        "status": lv.get("status"),
        "rmse_log": (lv.get("rmse_log") or {}).get("study_balanced"),
        "mean_quantile_loss": (lv.get("mean_quantile_loss") or {}).get("study_balanced"),
        "coverage80": (lv.get("coverage80") or {}).get("study_balanced"),
        "interval_width80": (lv.get("interval_width80") or {}).get("study_balanced"),
        "crps_justified": lv.get("crps_justified"),
        "point_definition": lv.get("point_definition"),
        "per_horizon": lv.get("per_horizon"),
        "n_scored": lv.get("n_scored"),
        "n_studies": len(lv.get("per_study") or []),
    }
    time_rep = encoder_time_report(hist, fut)
    write_json(dest / "encoder_time.json", time_rep)
    return out


def _run_one_replicate(cfg: dict[str, Any], parent: Path, dataset_index: int,
                       gen0: int, tr0: int, window: float, force: bool) -> dict[str, Any]:
    from send_bw.models.chronos2 import OFFICIAL_ID, Chronos2EarlyHistoryModel, verify_chronos2_api
    from send_bw.models.tft import TFTEarlyHistoryModel
    from send_bw.pipeline import model_class

    ev = cfg.get("seq_diag") or {}
    bank = str(ev.get("bank") or "matched")
    bcfg = _bank_cfg(cfg, parent, dataset_index, gen0, bank)
    br = _run_from_cfg(bcfg)
    br.generate(force=force)
    cells = {}
    hashes = {}
    reuse = None
    commands = [f"send-bw generate --config {br.dir}/effective.yaml"]
    for scope, sharing, k in (("target_only", "context_specific", 0),
                              ("pooled", "pooled", int(ev.get("n_donor_studies") or 3))):
        spec = {
            "name": f"d{dataset_index}_t0_{bank}_{scope}", "dataset_index": dataset_index,
            "train_index": 0, "bank": bank, "sharing": sharing, "k": k,
            "sparse_target_train_n": None, "generate_seed": gen0, "train_seed": tr0,
            "models": ["spline", "transformer"],
        }
        ccfg = _cell_cfg(cfg, parent, spec, generate_seed=gen0, train_seed=tr0,
                         models=spec["models"], sparse_target_train_n=None,
                         reuse_target_from=reuse)
        ccfg["evaluation"]["tasks"] = ["early_history"]
        ccfg["evaluation"]["early_history_window_day"] = window
        ccfg["design"] = {}
        run = _run_from_cfg(ccfg)
        run.ingest(force=force)
        run.prepare(force=force)
        run.train(["spline", "transformer"], force=force)
        cells[scope] = run
        man = read_json(run.p("splits", "target_manifest.json"))
        hashes[scope] = {n: (man.get(n) or {}).get("hash") for n in ("validation", "test", "all_target")}
        if reuse is None:
            reuse = str(run.p("splits", "x").parent)
        commands.append(f"send-bw train --config {run.dir}/effective.yaml --models spline transformer")
    if hashes["target_only"]["test"] != hashes["pooled"]["test"]:
        raise SeqDiagError(f"d{dataset_index}: target test hashes differ")
    test = load_cell_prepared(Path(cells["target_only"].dir))["data"]["test"]
    pred_dir = parent / "predictions"
    results: dict[str, Any] = {}
    reloads: dict[str, Any] = {}
    results["last_value"] = _score_last_value(test, window, pred_dir / "last_value.csv")
    results["last_value"]["seconds"] = 0.0
    for scope, run in cells.items():
        for name in ("spline", "transformer"):
            key = f"{name}_{scope}"
            t1 = time.time()
            model = model_class(name).load(run.p("models", name))
            sc = _score_gaussian(model, test, window, pred_dir / f"{key}.csv")
            sc["seconds"] = time.time() - t1
            sc["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
            sc["training_scope"] = scope
            results[key] = sc
    tft_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("tft") or {})
    tft_cfg.setdefault("seed", tr0)
    tft_cfg.setdefault("encoder_window_day", window)
    for scope, run in cells.items():
        key = f"tft_{scope}"
        t1 = time.time()
        try:
            prep = load_cell_prepared(Path(run.dir))
            model = TFTEarlyHistoryModel()
            hist = model.fit(prep["data"]["train"], prep["data"]["validation"],
                             {**tft_cfg, "training_scope": scope})
            dest = parent / "models" / key
            model.save(dest)
            write_json(dest / "training_history.json", hist)
            sc = _score_quantiles(model, test, window, "tft_quantiles", pred_dir / f"{key}.csv")
            sc["seconds"] = time.time() - t1
            sc["model_class"] = "pytorch_forecasting.TemporalFusionTransformer"
            sc["training_scope"] = scope
            sc["training_history"] = {k: hist.get(k) for k in
                                      ("n_train_series", "n_val_series", "max_epochs",
                                       "library", "training_scope", "official_class")}
            results[key] = sc
            reloads[key] = _reload_check(TFTEarlyHistoryModel.load, dest, model, test, window)
            commands.append(f"TFTEarlyHistoryModel.fit training_scope={scope}")
        except Exception as e:  # noqa: BLE001
            results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                            "seconds": time.time() - t1, "traceback": traceback.format_exc()[-1500:]}
    ch_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("chronos2") or {})
    ch_cfg.setdefault("pretrained_id", OFFICIAL_ID)
    ch_cfg.setdefault("device_map", "cpu")
    ch_cfg.setdefault("encoder_window_day", window)
    ch_cfg.setdefault("download_permission", True)
    cver = verify_chronos2_api()
    if cver.get("ok") and ch_cfg.get("download_permission"):
        t1 = time.time()
        try:
            zs = Chronos2EarlyHistoryModel()
            zh = zs.load_pretrained(ch_cfg)
            dest = parent / "models" / "chronos2_zero_shot"
            zs.save(dest)
            write_json(dest / "training_history.json", zh)
            sc = _score_quantiles(zs, test, window, "chronos2_quantiles",
                                  pred_dir / "chronos2_zero_shot.csv")
            sc["seconds"] = time.time() - t1
            sc["model_class"] = "chronos.Chronos2Pipeline"
            sc["mode"] = "zero_shot"
            sc["pretrained_id"] = ch_cfg["pretrained_id"]
            sc["training_history"] = zh
            results["chronos2_zero_shot"] = sc
            reloads["chronos2_zero_shot"] = _reload_check(
                Chronos2EarlyHistoryModel.load, dest, zs, test, window)
        except Exception as e:  # noqa: BLE001
            results["chronos2_zero_shot"] = {
                "status": "failed", "reason": f"{type(e).__name__}: {e}",
                "seconds": time.time() - t1}
        for scope, run in cells.items():
            key = f"chronos2_{scope}"
            t1 = time.time()
            try:
                prep = load_cell_prepared(Path(run.dir))
                model = Chronos2EarlyHistoryModel()
                ft_dir = parent / "models" / key / "finetune"
                hist = model.fit(prep["data"]["train"], prep["data"]["validation"], {
                    **ch_cfg, "mode": f"finetune_{scope}", "finetune_dir": str(ft_dir),
                })
                dest = parent / "models" / key
                model.save(dest)
                write_json(dest / "training_history.json", hist)
                sc = _score_quantiles(model, test, window, "chronos2_quantiles",
                                      pred_dir / f"{key}.csv")
                sc["seconds"] = time.time() - t1
                sc["model_class"] = "chronos.Chronos2Pipeline"
                sc["mode"] = f"finetune_{scope}"
                sc["pretrained_id"] = ch_cfg["pretrained_id"]
                sc["training_history"] = hist
                results[key] = sc
                reloads[key] = _reload_check(Chronos2EarlyHistoryModel.load, dest, model, test, window)
            except Exception as e:  # noqa: BLE001
                results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                                "seconds": time.time() - t1}
    else:
        reason = cver.get("reason") if not cver.get("ok") else "download not authorized"
        for key in ("chronos2_zero_shot", "chronos2_target_only", "chronos2_pooled"):
            results[key] = {"status": "not_run", "reason": reason}
    transfer = {
        "spline": paired_transfer(results.get("spline_target_only") or {}, results.get("spline_pooled") or {}),
        "transformer": paired_transfer(results.get("transformer_target_only") or {},
                                       results.get("transformer_pooled") or {}),
        "tft": paired_transfer(results.get("tft_target_only") or {}, results.get("tft_pooled") or {}),
        "chronos2_finetune": paired_transfer(results.get("chronos2_target_only") or {},
                                             results.get("chronos2_pooled") or {}),
    }
    compact = {k: _compact_metrics(v) for k, v in results.items()}
    return {
        "dataset_index": dataset_index,
        "generate_seed": gen0,
        "train_seed": tr0,
        "target_holdout_hashes": hashes,
        "results": compact,
        "paired_transfer": transfer,
        "reload_checks": reloads,
        "training_commands": commands,
        "task": "early_history",
        "predose_evaluated": False,
    }


def _compact_metrics(block: dict[str, Any]) -> dict[str, Any]:
    if not block:
        return {"status": "missing"}
    keep = ("status", "reason", "representation", "model_class", "mode",
            "training_scope", "seconds", "n_scored", "crps_justified",
            "point_definition", "per_horizon", "pretrained_id")
    out = {k: block[k] for k in keep if k in block}
    for m in ("rmse_log", "mean_quantile_loss", "coverage80", "interval_width80", "coverage50",
              "interval_width50", "crps_log"):
        if isinstance(block.get(m), dict):
            out[m] = block[m].get("study_balanced")
        elif m in block:
            out[m] = block[m]
    return out


def execute_v3_seq_diag(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or "runs/v3_seq_diag")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise SeqDiagError("refusing to write into a preserved artifact directory")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise SeqDiagError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise SeqDiagError("design optimization stays off")
    ev = cfg.get("seq_diag") or {}
    if ev.get("retune_after_test"):
        raise SeqDiagError("retune_after_test is forbidden")
    if ev.get("predose"):
        raise SeqDiagError("predose is off; early-history stays separate")
    tasks = list(ev.get("tasks") or (cfg.get("evaluation") or {}).get("tasks") or [])
    if "predose_conditioned" in tasks or "pre_study" in tasks:
        raise SeqDiagError("predose/pre_study must not be mixed into this diagnostic")
    parent.mkdir(parents=True, exist_ok=True)
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    window = float(ev.get("encoder_window_day") or WINDOW)
    gen0 = int(cfg["generate_seed"])
    tr0 = int(cfg["train_seed"])
    n_ds = int(ev.get("n_datasets") or 2)
    gstride = int(ev.get("generate_seed_stride") or GEN_STRIDE)
    tstride = int(ev.get("train_seed_stride") or TRAIN_STRIDE)
    official = Path(ev.get("official_seq_dir") or "runs/v3_official_seq")
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": gen0, "train_seed": tr0,
        "n_datasets": n_ds, "encoder_window_day": window,
        "official_seq_dir": str(official), "official_seq_frozen": True,
        "retune_after_test": False, "fabricate_encoder_history": False,
        "real_data": False, "design": False, "predose": False,
        "task": "early_history",
        "labels": list(LABELS),
        "not_a_new_architecture": True,
        "last_value_is_baseline": True,
    }
    write_json(parent / "freeze_lock.json", lock)
    t0 = time.time()
    auth = discover_authorization(None)
    correctness: dict[str, Any] = {}
    if not official.exists():
        correctness = {"status": "not_run", "reason": f"missing {official}"}
    else:
        test = load_cell_prepared(
            official / "datasets/d0/trains/t0/cells/d0_t0_matched_target_only")["data"]["test"]
        cdir = parent / "correctness"
        rescored = _rescore_saved(official, test, window, cdir)
        revision = chronos2_checkpoint_revision(
            ((cfg.get("models") or {}).get("optional") or {}).get("chronos2", {}).get(
                "pretrained_id") or "amazon/chronos-2")
        write_json(cdir / "checkpoint_revision.json", revision)
        scale = _chronos_scale_check(test, window)
        write_json(cdir / "chronos_scale.json", scale)
        covariates = _chronos_covariate_check()
        write_json(cdir / "chronos_covariates.json", covariates)
        official_summary = read_json(official / "SUMMARY.json") if (official / "SUMMARY.json").exists() else {}
        correctness = {
            "status": "ok",
            "official_seq_git": (official_summary.get("freeze_lock") or {}).get("git_revision"),
            "holdout_hashes": official_summary.get("target_holdout_hashes"),
            "rescored": rescored,
            "checkpoint_revision": revision,
            "chronos_scale": {k: scale.get(k) for k in scale if k != "rows"},
            "chronos_covariates": covariates,
            "encoder_time": {k: (read_json(cdir / "encoder_time.json") or {}).get(k)
                             for k in ("n_animals", "n_irregular_encoder", "median_encoder_gap",
                                       "chronos_time_index", "calendar_time_passed_to_chronos",
                                       "note")} if (cdir / "encoder_time.json").exists() else {},
            "predose_evaluated": False,
        }
    replicates = []
    blockers: list[dict[str, Any]] = []
    for i in range(n_ds):
        gseed = gen0 + i * gstride
        tseed = tr0 + i * tstride
        rparent = parent / "repl" / f"d{i}"
        try:
            rec = _run_one_replicate(cfg, rparent, i, gseed, tseed, window, force)
            write_json(rparent / "replicate.json", rec)
            replicates.append(rec)
        except Exception as e:  # noqa: BLE001
            blockers.append({"dataset_index": i, "reason": f"{type(e).__name__}: {e}"})
            replicates.append({"dataset_index": i, "status": "failed",
                               "reason": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-2000:]})
    pooled = _pool_replicates(replicates)
    out = {
        "labels": list(LABELS) + ["Official-seq diagnostic.", "Early-history only.",
                                  "Last-value baseline.", "Not retuned after test."],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "correctness": correctness,
        "replicates": [_replicate_public(r) for r in replicates],
        "pooled_replicates": pooled,
        "blockers": blockers,
        "unsupported": {
            "predose": "early-history is scored separately; TFT/Chronos/last-value refuse fabricated encoder",
            "design_simulation": "marginal quantiles are not complete trajectory simulators",
            "crps_for_quantile_models": "CRPS reported only for Gaussian spline/transformer",
        },
        "real_data_accessed": False,
        "authorization_present": bool(auth.get("authorized")),
        "design_run": False,
        "predose_evaluated": False,
        "retuned_after_test": False,
        "official_seq_modified": False,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out)
    return out


def _replicate_public(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec.get(k) for k in
            ("dataset_index", "generate_seed", "train_seed", "target_holdout_hashes",
             "results", "paired_transfer", "reload_checks", "status", "reason",
             "task", "predose_evaluated")}


def _pool_replicates(replicates: list[dict[str, Any]]) -> dict[str, Any]:
    keys = set()
    for rec in replicates:
        keys.update((rec.get("results") or {}).keys())
    out: dict[str, Any] = {}
    for key in sorted(keys):
        rmses, pins = [], []
        for rec in replicates:
            block = (rec.get("results") or {}).get(key) or {}
            if block.get("rmse_log") is not None:
                rmses.append(float(block["rmse_log"]))
            if block.get("mean_quantile_loss") is not None:
                pins.append(float(block["mean_quantile_loss"]))
        out[key] = {
            "n": len(rmses),
            "rmse_log_mean": float(np.mean(rmses)) if rmses else None,
            "rmse_log_sd": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else None,
            "mean_quantile_loss_mean": float(np.mean(pins)) if pins else None,
        }
    return out


def _fmt(block: dict[str, Any] | None, key: str) -> str:
    if not block:
        return "n/a"
    if block.get("status") not in {"ok", None} and key not in block:
        return str(block.get("status"))
    v = block.get(key)
    if isinstance(v, dict):
        v = v.get("study_balanced")
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(parent: Path, out: dict[str, Any]) -> None:
    corr = out.get("correctness") or {}
    rev = corr.get("checkpoint_revision") or {}
    scale = corr.get("chronos_scale") or {}
    lines = [
        "# Bounded official-sequence diagnostic",
        "",
        "> **Synthetic diagnostic. Exploratory. Not design qualification. "
        "No real-data validation. No operational recommendation.**",
        "",
        "PR #10 (`runs/v3_official_seq/`) is frozen and was not rewritten. "
        "Early-history is scored separately from predose. Design stayed off.",
        "",
        "## Provenance",
        "",
        f"- git: `{out.get('freeze_lock', {}).get('git_revision')}`",
        f"- generate_seed / train_seed: `{out.get('freeze_lock', {}).get('generate_seed')}` / "
        f"`{out.get('freeze_lock', {}).get('train_seed')}`",
        f"- n_datasets: `{out.get('freeze_lock', {}).get('n_datasets')}`",
        f"- retuned_after_test: `{out.get('retuned_after_test')}`",
        f"- official_seq_modified: `{out.get('official_seq_modified')}`",
        f"- Chronos-2 SHA: `{rev.get('sha')}` ({rev.get('source')})",
        f"- official mean is median: `{rev.get('official_mean_is_median')}`",
        "",
        "## Phase A — correctness on the frozen official-seq holdout",
        "",
        "Common quantile scoring. Point = median (q0.5) for quantile models; "
        "Gaussian MEAN for spline/transformer. `rmse_log` is the study-balanced "
        "mean of |point − LOGW|, the official_seq definition.",
        "",
        "| model | RMSE | pinball | cov80 | width80 | n |",
        "|---|---|---|---|---|---|",
    ]
    for key, block in (corr.get("rescored") or {}).items():
        lines.append(
            f"| {key} | {_fmt(block, 'rmse_log')} | {_fmt(block, 'mean_quantile_loss')} | "
            f"{_fmt(block, 'coverage80')} | {_fmt(block, 'interval_width80')} | "
            f"{block.get('n_scored')} |"
        )
    et = corr.get("encoder_time") or {}
    lines += [
        "",
        "### Chronos transformation / time / covariates",
        "",
        f"- official encoding: `{scale.get('official_encoding')}`; "
        f"changed: `{scale.get('official_encoding_changed')}`",
        f"- mean equals median on all series: `{scale.get('mean_equals_median_all_series')}`",
        f"- LOGW-input RMSE: `{(scale.get('logw_input') or {}).get('rmse_log')}`",
        f"- exp(LOGW)-input logged-back RMSE: "
        f"`{(scale.get('exp_logw_input_logged_back') or {}).get('rmse_log')}` "
        "(diagnostic only)",
        f"- encoder animals: `{et.get('n_animals')}`; irregular: `{et.get('n_irregular_encoder')}`; "
        f"median gap: `{et.get('median_encoder_gap')}`",
        f"- Chronos time index: `{et.get('chronos_time_index')}`; "
        f"calendar time passed: `{et.get('calendar_time_passed_to_chronos')}`",
        f"- covariates unused: `{(corr.get('chronos_covariates') or {}).get('unused')}`",
        "",
        "## Phase B — prespecified fresh replicates (early-history only)",
        "",
        "Predose was not evaluated. Harmful transfer is published as-is.",
        "",
    ]
    for rec in out.get("replicates") or []:
        lines.append(f"### dataset {rec.get('dataset_index')} "
                     f"(gen {rec.get('generate_seed')} / train {rec.get('train_seed')})")
        lines.append("")
        lines.append(f"- holdout hashes: `{rec.get('target_holdout_hashes')}`")
        lines.append("")
        lines.append("| model | RMSE | pinball | cov80 | width80 |")
        lines.append("|---|---|---|---|---|")
        for key, block in (rec.get("results") or {}).items():
            lines.append(
                f"| {key} | {_fmt(block, 'rmse_log')} | {_fmt(block, 'mean_quantile_loss')} | "
                f"{_fmt(block, 'coverage80')} | {_fmt(block, 'interval_width80')} |"
            )
        lines.append("")
        lines.append("| family | metric | target-only | pooled | Δ |")
        lines.append("|---|---|---|---|---|")
        for fam, block in (rec.get("paired_transfer") or {}).items():
            for row in block.get("rows") or []:
                lines.append(
                    f"| {fam} | {row.get('metric')} | {row.get('target_only')} | "
                    f"{row.get('pooled')} | {row.get('delta_pooled_minus_target')} |"
                )
        lines.append("")
    lines += [
        "### Pooled across replicates",
        "",
        "| model | n | RMSE mean | RMSE sd | pinball mean |",
        "|---|---|---|---|---|",
    ]
    for key, block in (out.get("pooled_replicates") or {}).items():
        rmse_sd = block.get("rmse_log_sd")
        lines.append(
            f"| {key} | {block.get('n')} | {_fmt(block, 'rmse_log_mean')} | "
            f"{'n/a' if rmse_sd is None else f'{rmse_sd:.4f}'} | "
            f"{_fmt(block, 'mean_quantile_loss_mean')} |"
        )
    lines += [
        "",
        "## Blockers and unsupported tasks",
        "",
        f"- blockers: {out.get('blockers')}",
        f"- unsupported: {out.get('unsupported')}",
        f"- real_data_accessed: {out.get('real_data_accessed')}",
        f"- design_run: {out.get('design_run')}",
        f"- predose_evaluated: {out.get('predose_evaluated')}",
        f"- seconds_total: {out.get('seconds_total')}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (parent / "RESULTS.md").write_text(text, encoding="utf-8")
    Path("docs/v3/seq_diag_RESULTS.md").write_text(text, encoding="utf-8")

"""Official TFT + Chronos-2 early-history execution on one frozen synthetic holdout.

Spline and the custom transformer are comparators on the same cutoff and target
test studies. They are not substitutes for TFT or Chronos-2.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.evaluation.quantile_forecast import (
    QUANTILES,
    paired_transfer,
    score_quantile_table,
    split_encoder_future,
)
from send_bw.historical_audit import discover_authorization
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_research import _bank_cfg, _cell_cfg, _run_from_cfg, load_yaml

PRESERVED = (
    "runs/v3_eval", "runs/v3_pilot_k0k1", "runs/v3_repl", "runs/v3_research",
    "runs/v3_closeout", "runs/v3_audit", "runs/v3_increment", "runs/v3_phaseb",
    "runs/v3_limitations", "runs/v3_objective", "runs/v3_official_seq",
)
WINDOW = 14.0


class OfficialSeqError(ValueError):
    pass


def collect_versions() -> dict[str, Any]:
    import importlib
    import sys

    out: dict[str, Any] = {"python": sys.version.split()[0], "packages": {}}
    for name in ("numpy", "pandas", "torch", "pytorch_forecasting", "lightning",
                 "chronos", "transformers", "peft", "accelerate"):
        try:
            mod = importlib.import_module(name)
            out["packages"][name] = getattr(mod, "__version__", "present")
        except Exception as e:  # noqa: BLE001
            out["packages"][name] = f"NOT_INSTALLED: {type(e).__name__}: {e}"
    try:
        import torch
        out["torch_cuda"] = bool(torch.cuda.is_available())
        out["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        out["torch_cuda"] = False
        out["torch_device"] = "cpu"
    try:
        from pytorch_forecasting import TemporalFusionTransformer
        out["tft_class"] = f"{TemporalFusionTransformer.__module__}.{TemporalFusionTransformer.__name__}"
    except Exception as e:  # noqa: BLE001
        out["tft_class"] = f"unavailable: {e}"
    try:
        from chronos import Chronos2Pipeline
        out["chronos2_class"] = f"{Chronos2Pipeline.__module__}.{Chronos2Pipeline.__name__}"
    except Exception as e:  # noqa: BLE001
        out["chronos2_class"] = f"unavailable: {e}"
    return out


def _write_pred_table(pred: pd.DataFrame, pred_path: Path | None) -> None:
    if pred_path is None:
        return
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pred_path, index=False)


def _score_gaussian(model, data, window_day: float,
                    pred_path: Path | None = None) -> dict[str, Any]:
    hist, fut = split_encoder_future(data, window_day)
    ctx = data.animals[data.animals["USUBJID"].isin(set(hist["USUBJID"]))]
    times = np.array(sorted(fut["DAY"].unique()), float)
    pred = model.predict_marginals(ctx, times, history=hist)
    if "STUDYID" not in pred.columns:
        pred = pred.merge(fut[["USUBJID", "STUDYID", "DAY"]], on=["USUBJID", "DAY"], how="left")
    _write_pred_table(pred, pred_path)
    return score_quantile_table(fut, pred, quantiles=QUANTILES, representation="gaussian")


def _score_quantiles(model, data, window_day: float, representation: str,
                     pred_path: Path | None = None) -> dict[str, Any]:
    hist, fut = split_encoder_future(data, window_day)
    recs = []
    skipped = 0
    for u, fg in fut.groupby("USUBJID", sort=False):
        hh = hist.loc[hist["USUBJID"] == u]
        if hh.empty:
            skipped += 1
            continue
        a = data.animals.loc[data.animals["USUBJID"] == u]
        times = fg.sort_values("DAY")["DAY"].to_numpy(float)
        pred = model.predict_quantiles(a, times, history=hh)
        if pred is None or pred.empty:
            skipped += 1
            continue
        recs.append(pred)
    if not recs:
        return {"status": "not_run", "reason": "no animal had observed encoder history",
                "n_skipped_no_encoder": skipped}
    pred = pd.concat(recs, ignore_index=True)
    if "STUDYID" not in pred.columns:
        pred = pred.merge(fut[["USUBJID", "STUDYID", "DAY"]], on=["USUBJID", "DAY"], how="left")
    _write_pred_table(pred, pred_path)
    out = score_quantile_table(fut, pred, quantiles=QUANTILES, representation=representation)
    out["n_skipped_no_encoder"] = skipped
    out["encoder_fabricated"] = False
    return out


def _reload_check(load_fn, path: Path, model, data, window_day: float) -> dict[str, Any]:
    hist, fut = split_encoder_future(data, window_day)
    if hist.empty or fut.empty:
        return {"ok": False, "reason": "empty encoder/future"}
    u = fut["USUBJID"].iloc[0]
    a = data.animals.loc[data.animals["USUBJID"] == u]
    times = fut.loc[fut["USUBJID"] == u, "DAY"].sort_values().to_numpy(float)[:2]
    hh = hist.loc[hist["USUBJID"] == u]
    if hh.empty or not len(times):
        return {"ok": False, "reason": "no overlapping animal"}
    before = model.predict_quantiles(a, times, history=hh)
    loaded = load_fn(path)
    after = loaded.predict_quantiles(a, times, history=hh)
    if before.empty or after.empty:
        return {"ok": False, "reason": "empty predictions"}
    col = "q0.5" if "q0.5" in before.columns else None
    if col is None:
        return {"ok": False, "reason": "no median column"}
    b = before.sort_values("DAY")[col].to_numpy(float)
    c = after.sort_values("DAY")[col].to_numpy(float)
    n = min(len(b), len(c))
    max_abs = float(np.max(np.abs(b[:n] - c[:n]))) if n else None
    return {"ok": bool(max_abs is not None and max_abs < 1e-5), "max_abs_median_diff": max_abs,
            "n_compared": n}


def execute_v3_official_seq(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or "runs/v3_official_seq")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise OfficialSeqError("refusing to write into a preserved artifact directory")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise OfficialSeqError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise OfficialSeqError("design optimization stays off")
    parent.mkdir(parents=True, exist_ok=True)
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    ev = cfg.get("official_seq") or {}
    window = float(ev.get("encoder_window_day") or WINDOW)
    gen0 = int(cfg["generate_seed"])
    tr0 = int(cfg["train_seed"])
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": gen0, "train_seed": tr0,
        "encoder_window_day": window,
        "identical_target_holdouts": True, "identical_information_cutoff": True,
        "fabricate_encoder_history": False, "real_data": False, "design": False,
        "labels": list(LABELS),
        "official_tft": "pytorch_forecasting.TemporalFusionTransformer",
        "official_chronos2": "chronos.Chronos2Pipeline / amazon/chronos-2",
        "not_a_custom_transformer_substitute": True,
    }
    write_json(parent / "freeze_lock.json", lock)
    t0 = time.time()
    blockers: list[dict[str, Any]] = []
    commands: list[str] = []

    # --- generate one synthetic bank ---
    bcfg = _bank_cfg(cfg, parent, 0, gen0, str(ev.get("bank") or "matched"))
    br = _run_from_cfg(bcfg)
    commands.append(f"send-bw generate --config {br.dir}/effective.yaml")
    br.generate(force=force)

    cells = {}
    hashes = {}
    reuse = None
    for scope, sharing, k in (("target_only", "context_specific", 0),
                              ("pooled", "pooled", int(ev.get("n_donor_studies") or 3))):
        spec = {
            "name": f"d0_t0_matched_{scope}", "dataset_index": 0, "train_index": 0,
            "bank": ev.get("bank") or "matched", "sharing": sharing, "k": k,
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
        raise OfficialSeqError("target-study test holdout hashes differ between scopes")
    if hashes["target_only"]["validation"] != hashes["pooled"]["validation"]:
        raise OfficialSeqError("target-study validation holdout hashes differ between scopes")

    test = load_cell_prepared(Path(cells["target_only"].dir))["data"]["test"]
    results: dict[str, Any] = {}
    reloads: dict[str, Any] = {}

    # --- spline / custom transformer comparators ---
    from send_bw.pipeline import model_class

    for scope, run in cells.items():
        for name in ("spline", "transformer"):
            key = f"{name}_{scope}"
            t1 = time.time()
            try:
                model = model_class(name).load(run.p("models", name))
                sc = _score_gaussian(model, test, window,
                                     parent / "predictions" / f"{key}.csv")
                sc["seconds"] = time.time() - t1
                sc["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
                sc["training_scope"] = scope
                sc["checkpoint"] = str(run.p("models", name))
                results[key] = sc
            except Exception as e:  # noqa: BLE001
                results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                                "seconds": time.time() - t1}

    # --- official TFT ---
    from send_bw.models.tft import TFTEarlyHistoryModel

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
            sc = _score_quantiles(model, test, window, "tft_quantiles",
                                 parent / "predictions" / f"{key}.csv")
            sc["seconds"] = time.time() - t1
            sc["model_class"] = "pytorch_forecasting.TemporalFusionTransformer"
            sc["training_scope"] = scope
            sc["checkpoint"] = str(dest / "tft.ckpt")
            sc["training_history"] = {k: hist.get(k) for k in
                                      ("n_train_series", "n_val_series", "max_epochs",
                                       "library", "training_scope", "official_class")}
            results[key] = sc
            reloads[key] = _reload_check(TFTEarlyHistoryModel.load, dest, model, test, window)
            commands.append(f"TFTEarlyHistoryModel.fit training_scope={scope} TemporalFusionTransformer.from_dataset")
        except Exception as e:  # noqa: BLE001
            results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                            "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:]}
            blockers.append({"model": key, "kind": "tft_fit_or_score", "reason": str(e)})

    # --- official Chronos-2 ---
    from send_bw.models.chronos2 import OFFICIAL_ID, Chronos2EarlyHistoryModel, verify_chronos2_api

    cver = verify_chronos2_api()
    ch_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("chronos2") or {})
    ch_cfg.setdefault("pretrained_id", OFFICIAL_ID)
    ch_cfg.setdefault("device_map", "cpu" if not versions.get("torch_cuda") else "cuda")
    ch_cfg.setdefault("encoder_window_day", window)
    ch_cfg.setdefault("download_permission", True)
    if not cver.get("ok"):
        blockers.append({"model": "chronos2", "kind": "dependency", "reason": cver.get("reason")})
        for key in ("chronos2_zero_shot", "chronos2_target_only", "chronos2_pooled"):
            results[key] = {"status": "not_run", "reason": cver.get("reason")}
    elif not ch_cfg.get("download_permission"):
        blockers.append({"model": "chronos2", "kind": "download",
                         "reason": "download_permission is false"})
        for key in ("chronos2_zero_shot", "chronos2_target_only", "chronos2_pooled"):
            results[key] = {"status": "not_run", "reason": "download not authorized"}
    else:
        # zero-shot
        t1 = time.time()
        try:
            zs = Chronos2EarlyHistoryModel()
            zh = zs.load_pretrained(ch_cfg)
            dest = parent / "models" / "chronos2_zero_shot"
            zs.save(dest)
            write_json(dest / "training_history.json", zh)
            sc = _score_quantiles(zs, test, window, "chronos2_quantiles",
                                 parent / "predictions" / "chronos2_zero_shot.csv")
            sc["seconds"] = time.time() - t1
            sc["model_class"] = "chronos.Chronos2Pipeline"
            sc["mode"] = "zero_shot"
            sc["pretrained_id"] = ch_cfg["pretrained_id"]
            sc["checkpoint"] = str(dest / "pipeline")
            sc["training_history"] = zh
            results["chronos2_zero_shot"] = sc
            reloads["chronos2_zero_shot"] = _reload_check(
                Chronos2EarlyHistoryModel.load, dest, zs, test, window)
            commands.append(f"Chronos2Pipeline.from_pretrained({ch_cfg['pretrained_id']!r})  # zero-shot")
        except Exception as e:  # noqa: BLE001
            results["chronos2_zero_shot"] = {
                "status": "failed", "reason": f"{type(e).__name__}: {e}",
                "seconds": time.time() - t1, "traceback": traceback.format_exc()[-2000:]}
            blockers.append({"model": "chronos2_zero_shot", "kind": "download_or_infer", "reason": str(e)})

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
                                     parent / "predictions" / f"{key}.csv")
                sc["seconds"] = time.time() - t1
                sc["model_class"] = "chronos.Chronos2Pipeline"
                sc["mode"] = f"finetune_{scope}"
                sc["pretrained_id"] = ch_cfg["pretrained_id"]
                sc["checkpoint"] = str(dest / "pipeline")
                sc["training_history"] = hist
                results[key] = sc
                reloads[key] = _reload_check(Chronos2EarlyHistoryModel.load, dest, model, test, window)
                commands.append(
                    f"Chronos2Pipeline.from_pretrained({ch_cfg['pretrained_id']!r}).fit "
                    f"finetune_mode={ch_cfg.get('finetune_mode', 'lora')} scope={scope}"
                )
            except Exception as e:  # noqa: BLE001
                results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                                "seconds": time.time() - t1,
                                "traceback": traceback.format_exc()[-2000:]}
                blockers.append({"model": key, "kind": "finetune_or_score", "reason": str(e)})

    transfer = {
        "spline": paired_transfer(results.get("spline_target_only") or {}, results.get("spline_pooled") or {}),
        "transformer": paired_transfer(results.get("transformer_target_only") or {},
                                       results.get("transformer_pooled") or {}),
        "tft": paired_transfer(results.get("tft_target_only") or {}, results.get("tft_pooled") or {}),
        "chronos2_finetune": paired_transfer(results.get("chronos2_target_only") or {},
                                             results.get("chronos2_pooled") or {}),
    }
    auth = discover_authorization(None)
    out = {
        "labels": list(LABELS) + ["Official TFT and Chronos-2.", "Early-history only."],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "training_commands": commands,
        "target_holdout_hashes": hashes,
        "encoder_window_day": window,
        "results": results,
        "paired_transfer": transfer,
        "reload_checks": reloads,
        "blockers": blockers,
        "unsupported": {
            "pre_study_baseline_only": "TFT and Chronos-2 refuse fabricated encoder history",
            "design_simulation": "marginal quantiles are not complete trajectory simulators",
            "crps_for_quantile_models": "CRPS reported only for Gaussian spline/transformer",
        },
        "real_data_accessed": False,
        "authorization_present": bool(auth.get("authorized")),
        "design_run": False,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out)
    return out


def _fmt_metric(block: dict[str, Any] | None, key: str) -> str:
    if not block or block.get("status") not in {"ok", None}:
        if block:
            return f"{block.get('status')} ({(block.get('reason') or '')[:80]})"
        return "n/a"
    m = block.get(key)
    if isinstance(m, dict):
        v = m.get("study_balanced")
        return "n/a" if v is None else f"{v:.4f}"
    if m is None:
        return "n/a"
    return str(m)


def _write_report(parent: Path, out: dict[str, Any]) -> None:
    vers = out.get("versions") or {}
    pkgs = vers.get("packages") or {}
    lines = [
        "# Official TFT and Chronos-2 — executed early-history results",
        "",
        "> **Synthetic diagnostic. Exploratory. Not design qualification. "
        "No real-data validation. No operational recommendation.**",
        "",
        "This is **not** a spline or custom-transformer substitute. "
        "TFT is `pytorch_forecasting.TemporalFusionTransformer`. "
        "Chronos-2 is `chronos.Chronos2Pipeline` from `amazon/chronos-2`.",
        "",
        "## Provenance",
        "",
        f"- git: `{out.get('freeze_lock', {}).get('git_revision')}`",
        f"- generate_seed / train_seed: `{out.get('freeze_lock', {}).get('generate_seed')}` / "
        f"`{out.get('freeze_lock', {}).get('train_seed')}`",
        f"- device: `{vers.get('torch_device')}`; cuda={vers.get('torch_cuda')}",
        f"- TFT class: `{vers.get('tft_class')}`",
        f"- Chronos-2 class: `{vers.get('chronos2_class')}`",
        "",
        "| package | version |",
        "|---|---|",
    ]
    for k, v in pkgs.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Holdouts",
        "",
        f"- encoder cutoff: DAY ≤ {out.get('encoder_window_day')} (observed visits only)",
        f"- target holdout hashes: `{out.get('target_holdout_hashes')}`",
        "- pooled training reuses the same target validation/test STUDYIDs",
        "- encoder history is never fabricated",
        "",
        "## Training commands",
        "",
    ]
    for c in out.get("training_commands") or []:
        lines.append(f"- `{c}`")
    lines += [
        "",
        "## Metrics (study-balanced log-weight)",
        "",
        "CRPS is Gaussian-only. TFT and Chronos-2 report pinball / coverage / widths "
        "from their quantile representation.",
        "",
        "| model | status | RMSE | mean pinball | cov80 | width80 | CRPS | seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, block in (out.get("results") or {}).items():
        lines.append(
            f"| {key} | {block.get('status')} | {_fmt_metric(block, 'rmse_log')} | "
            f"{_fmt_metric(block, 'mean_quantile_loss')} | {_fmt_metric(block, 'coverage80')} | "
            f"{_fmt_metric(block, 'interval_width80')} | {_fmt_metric(block, 'crps_log')} | "
            f"{block.get('seconds')} |"
        )
    lines += [
        "",
        "## Paired transfer (pooled − target-only; Δ>0 harmful)",
        "",
        "| family | metric | target-only | pooled | Δ |",
        "|---|---|---|---|---|",
    ]
    for fam, block in (out.get("paired_transfer") or {}).items():
        for row in block.get("rows") or []:
            lines.append(
                f"| {fam} | {row.get('metric')} | {row.get('target_only')} | "
                f"{row.get('pooled')} | {row.get('delta_pooled_minus_target')} |"
            )
    lines += [
        "",
        "## Serialization / reload",
        "",
        f"`{out.get('reload_checks')}`",
        "",
        "## Blockers and unsupported tasks",
        "",
        f"- blockers: {out.get('blockers')}",
        f"- unsupported: {out.get('unsupported')}",
        f"- real_data_accessed: {out.get('real_data_accessed')}",
        f"- design_run: {out.get('design_run')}",
        "",
        f"- seconds_total: {out.get('seconds_total')}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (parent / "RESULTS.md").write_text(text, encoding="utf-8")
    Path("docs/v3/official_seq_RESULTS.md").write_text(text, encoding="utf-8")

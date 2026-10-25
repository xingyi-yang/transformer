"""Fresh frozen comparison: covariate-aware Chronos-2 vs history-only and TFT.

PR #12 metric correction is the scoring baseline. Prior run dirs stay frozen.
No fabricated history, no real data, no design, no new architectures.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from send_bw.evaluation.quantile_forecast import (
    QUANTILES,
    last_value_predictions,
    paired_transfer,
    score_quantile_table,
    split_encoder_future,
)
from send_bw.historical_audit import discover_authorization
from send_bw.models.chronos2 import COVARIATE_AWARE, HISTORY_ONLY, chronos_fit_preflight
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_defect_trace import trace_metric_defects
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
    "runs/v3_seq_diag", "runs/v3_eval_correct",
)
WINDOW = 14.0


class CovChronosError(ValueError):
    pass


def _compact(sc: dict[str, Any]) -> dict[str, Any]:
    if sc.get("status") != "ok":
        return {k: sc.get(k) for k in ("status", "reason") if k in sc}
    out = {
        "status": "ok",
        "mae_log": (sc.get("mae_log") or {}).get("study_balanced"),
        "rmse_log": (sc.get("rmse_log") or {}).get("study_balanced"),
        "mean_quantile_loss": (sc.get("mean_quantile_loss") or {}).get("study_balanced"),
        "coverage80": (sc.get("coverage80") or {}).get("study_balanced"),
        "interval_width80": (sc.get("interval_width80") or {}).get("study_balanced"),
        "n_studies": (sc.get("mae_log") or {}).get("n_studies"),
        "n_scored": sc.get("n_scored"),
        "grouping": sc.get("grouping"),
        "merge": sc.get("merge"),
        "chronos_label": sc.get("chronos_label"),
        "covariates_used": sc.get("covariates_used"),
        "point_definition": (sc.get("point_definition") or {}).get("point"),
        "crps_justified": sc.get("crps_justified"),
        "seconds": sc.get("seconds"),
        "preflight": sc.get("preflight"),
    }
    return out


def _score_last_value(data, window_day: float, pred_path: Path | None = None) -> dict[str, Any]:
    hist, fut = split_encoder_future(data, window_day)
    pred = last_value_predictions(hist, fut, QUANTILES)
    if pred_path is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(pred_path, index=False)
    out = score_quantile_table(fut, pred, quantiles=QUANTILES, representation="last_value")
    out["encoder_fabricated"] = False
    return out


def execute_v3_cov_chronos(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or "runs/v3_cov_chronos")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise CovChronosError("refusing to write into a preserved artifact directory")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise CovChronosError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise CovChronosError("design optimization stays off")
    ev = cfg.get("cov_chronos") or {}
    if ev.get("train") is False:
        raise CovChronosError("this milestone trains a fresh frozen comparison")
    parent.mkdir(parents=True, exist_ok=True)
    window = float(ev.get("encoder_window_day") or WINDOW)
    gen0 = int(cfg["generate_seed"])
    tr0 = int(cfg["train_seed"])
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), "generate_seed": gen0, "train_seed": tr0,
        "encoder_window_day": window,
        "identical_target_holdouts": True, "identical_information_cutoff": True,
        "fabricate_encoder_history": False, "real_data": False, "design": False,
        "new_architectures": False,
        "history_only_comparator_retained": True,
        "covariate_information": "origin_only",
        "metrics": "corrected study-balanced MAE and true RMSE",
        "labels": list(LABELS),
        "preserved": list(PRESERVED),
    }
    write_json(parent / "freeze_lock.json", lock)

    defect = trace_metric_defects(ev.get("eval_correct_dir") or "runs/v3_eval_correct")
    write_json(parent / "defect_trace.json", defect)
    if defect.get("rescore_needed"):
        raise CovChronosError(
            "affected prediction tables still need rescoring; run v3-eval-correct first: "
            + ", ".join(defect["rescore_needed"])
        )

    t0 = time.time()
    blockers: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    commands: list[str] = []

    bcfg = _bank_cfg(cfg, parent, 0, gen0, str(ev.get("bank") or "matched"))
    br = _run_from_cfg(bcfg)
    commands.append(f"send-bw generate --config {br.dir}/effective.yaml")
    br.generate(force=force)

    cells: dict[str, Any] = {}
    hashes: dict[str, Any] = {}
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
        raise CovChronosError("target-study test holdout hashes differ between scopes")
    if hashes["target_only"]["validation"] != hashes["pooled"]["validation"]:
        raise CovChronosError("target-study validation holdout hashes differ between scopes")

    test = load_cell_prepared(Path(cells["target_only"].dir))["data"]["test"]
    pred_dir = parent / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    reloads: dict[str, Any] = {}
    preflights: dict[str, Any] = {}

    from send_bw.pipeline import model_class

    for scope, run in cells.items():
        for name in ("spline", "transformer"):
            key = f"{name}_{scope}"
            t1 = time.time()
            try:
                model = model_class(name).load(run.p("models", name))
                sc = _score_gaussian(model, test, window, pred_dir / f"{key}.csv")
                sc["seconds"] = time.time() - t1
                results[key] = sc
            except Exception as e:  # noqa: BLE001
                results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
                failures.append({"model": key, "reason": str(e), "scored": False})

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
            sc = _score_quantiles(model, test, window, "tft_quantiles", pred_dir / f"{key}.csv")
            sc["seconds"] = time.time() - t1
            results[key] = sc
            reloads[key] = _reload_check(TFTEarlyHistoryModel.load, dest, model, test, window)
        except Exception as e:  # noqa: BLE001
            results[key] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-2000:]}
            failures.append({"model": key, "reason": str(e), "scored": False})
            blockers.append({"model": key, "kind": "tft_fit_or_score", "reason": str(e)})

    t1 = time.time()
    lv = _score_last_value(test, window, pred_dir / "last_value.csv")
    lv["seconds"] = time.time() - t1
    results["last_value"] = lv

    from send_bw.models.chronos2 import OFFICIAL_ID, Chronos2EarlyHistoryModel, verify_chronos2_api

    cver = verify_chronos2_api()
    ch_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("chronos2") or {})
    ch_cfg.setdefault("pretrained_id", OFFICIAL_ID)
    ch_cfg.setdefault("device_map", "cpu" if not versions.get("torch_cuda") else "cuda")
    ch_cfg.setdefault("encoder_window_day", window)
    ch_cfg.setdefault("download_permission", True)
    if not cver.get("ok"):
        blockers.append({"model": "chronos2", "kind": "dependency", "reason": cver.get("reason")})
        for key in ("chronos2_zero_shot_history", "chronos2_zero_shot_covariate",
                    "chronos2_target_only_history", "chronos2_pooled_history",
                    "chronos2_target_only_covariate", "chronos2_pooled_covariate"):
            results[key] = {"status": "not_run", "reason": cver.get("reason")}
            failures.append({"model": key, "reason": cver.get("reason"), "scored": False})
    else:
        t1 = time.time()
        try:
            zs = Chronos2EarlyHistoryModel()
            zh = zs.load_pretrained({**ch_cfg, "covariate_mode": HISTORY_ONLY})
            dest = parent / "models" / "chronos2_zero_shot"
            zs.save(dest)
            write_json(dest / "training_history.json", zh)
            sc_h = _score_quantiles(zs, test, window, "chronos2_quantiles",
                                    pred_dir / "chronos2_zero_shot_history.csv",
                                    chronos_label=HISTORY_ONLY)
            sc_h["seconds"] = time.time() - t1
            sc_h["chronos_label"] = HISTORY_ONLY
            results["chronos2_zero_shot_history"] = sc_h
            zs.covariate_mode = COVARIATE_AWARE
            t2 = time.time()
            sc_c = _score_quantiles(zs, test, window, "chronos2_quantiles",
                                    pred_dir / "chronos2_zero_shot_covariate.csv",
                                    chronos_label=COVARIATE_AWARE)
            sc_c["seconds"] = time.time() - t2
            sc_c["chronos_label"] = COVARIATE_AWARE
            results["chronos2_zero_shot_covariate"] = sc_c
            reloads["chronos2_zero_shot"] = _reload_check(
                Chronos2EarlyHistoryModel.load, dest, zs, test, window)
        except Exception as e:  # noqa: BLE001
            results["chronos2_zero_shot_history"] = {
                "status": "failed", "reason": f"{type(e).__name__}: {e}"}
            results["chronos2_zero_shot_covariate"] = {
                "status": "failed", "reason": f"{type(e).__name__}: {e}"}
            failures.append({"model": "chronos2_zero_shot", "reason": str(e), "scored": False})
            blockers.append({"model": "chronos2_zero_shot", "kind": "infer", "reason": str(e)})

        for cov_mode, tag in ((HISTORY_ONLY, "history"), (COVARIATE_AWARE, "covariate")):
            for scope, run in cells.items():
                key = f"chronos2_{scope}_{tag}"
                t1 = time.time()
                try:
                    prep = load_cell_prepared(Path(run.dir))
                    pre = chronos_fit_preflight(
                        prep["data"]["train"], prep["data"]["validation"],
                        min_past=int(ch_cfg.get("min_past", 2)),
                        prediction_length=int(ch_cfg.get("prediction_length", 6)),
                        covariate_mode=cov_mode)
                    preflights[key] = pre
                    write_json(parent / "preflight" / f"{key}.json", pre)
                    if not pre["can_fit"]:
                        raise RuntimeError(
                            "preflight: train empty after Chronos length filter "
                            f"(kept {pre['train']['n_kept']})"
                        )
                    model = Chronos2EarlyHistoryModel()
                    ft_dir = parent / "models" / key / "finetune"
                    hist = model.fit(prep["data"]["train"], prep["data"]["validation"], {
                        **ch_cfg, "mode": f"finetune_{scope}", "finetune_dir": str(ft_dir),
                        "covariate_mode": cov_mode,
                    })
                    dest = parent / "models" / key
                    model.save(dest)
                    write_json(dest / "training_history.json", hist)
                    sc = _score_quantiles(
                        model, test, window, "chronos2_quantiles",
                        pred_dir / f"{key}.csv", chronos_label=cov_mode)
                    sc["seconds"] = time.time() - t1
                    sc["preflight"] = pre
                    sc["chronos_label"] = cov_mode
                    results[key] = sc
                    reloads[key] = _reload_check(Chronos2EarlyHistoryModel.load, dest, model, test, window)
                except Exception as e:  # noqa: BLE001
                    results[key] = {
                        "status": "failed", "reason": f"{type(e).__name__}: {e}",
                        "preflight": preflights.get(key),
                        "traceback": traceback.format_exc()[-2000:],
                    }
                    failures.append({
                        "model": key, "reason": str(e), "scored": False,
                        "preflight": preflights.get(key),
                    })
                    blockers.append({"model": key, "kind": "finetune_or_score", "reason": str(e)})

    transfer = {
        "spline": paired_transfer(results.get("spline_target_only") or {},
                                  results.get("spline_pooled") or {}),
        "transformer": paired_transfer(results.get("transformer_target_only") or {},
                                       results.get("transformer_pooled") or {}),
        "tft": paired_transfer(results.get("tft_target_only") or {},
                               results.get("tft_pooled") or {}),
        "chronos2_finetune_history": paired_transfer(
            results.get("chronos2_target_only_history") or {},
            results.get("chronos2_pooled_history") or {}),
        "chronos2_finetune_covariate": paired_transfer(
            results.get("chronos2_target_only_covariate") or {},
            results.get("chronos2_pooled_covariate") or {}),
    }
    for _name, block in transfer.items():
        for row in block.get("rows") or []:
            row["sign"] = "harmful" if (row.get("delta_pooled_minus_target") or 0) > 0 else "helpful"

    ft_hist_ok = all(
        (results.get(k) or {}).get("status") == "ok"
        for k in ("chronos2_target_only_history", "chronos2_pooled_history"))
    ft_cov_ok = all(
        (results.get(k) or {}).get("status") == "ok"
        for k in ("chronos2_target_only_covariate", "chronos2_pooled_covariate"))
    auth = discover_authorization(None)
    compact = {k: _compact(v) if isinstance(v, dict) else v for k, v in results.items()}
    out = {
        "labels": list(LABELS) + [
            "Covariate-aware Chronos-2.",
            "History-only comparator retained.",
            "Corrected MAE and true RMSE.",
        ],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "defect_trace": {
            "affected_published_metrics": defect.get("affected_published_metrics"),
            "rescore_needed": defect.get("rescore_needed"),
            "validation_selection_affected": defect.get("validation_selection_affected"),
        },
        "training_commands": commands,
        "target_holdout_hashes": hashes,
        "encoder_window_day": window,
        "results": compact,
        "paired_transfer": transfer,
        "reload_checks": reloads,
        "preflights": preflights,
        "failures": failures,
        "blockers": blockers,
        "chronos_finetune_pool": {
            "history_only_complete": bool(ft_hist_ok),
            "covariate_aware_complete": bool(ft_cov_ok),
            "aggregated_across_incomplete": False,
            "label_history": HISTORY_ONLY,
            "label_covariate": COVARIATE_AWARE,
        },
        "real_data_accessed": False,
        "authorization_present": bool(auth.get("authorized")),
        "design_run": False,
        "predose_evaluated": False,
        "new_architectures": False,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", out)
    write_json(parent / "results_full.json", {k: results[k] for k in results})
    _write_report(parent, out, defect)
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(parent: Path, out: dict[str, Any], defect: dict[str, Any]) -> None:
    lines = [
        "# Covariate-aware Chronos-2 — frozen matched-information comparison",
        "",
        "> **Synthetic diagnostic. Exploratory. Not design qualification. "
        "No real-data validation. No operational recommendation.**",
        "",
        "PR #12 evaluation correction is the scoring baseline. Prior artifacts "
        "are preserved. Chronos-2 history-only remains the comparator. "
        "Covariates are origin-only (SEX, STRAIN, SPECIES, ROUTE, PLANNED_DOSE, DAY).",
        "",
        "## Defect trace",
        "",
        f"- Affected published metrics: `{defect.get('affected_published_metrics')}`",
        f"- Rescore still needed: `{defect.get('rescore_needed')}`",
        f"- Validation-selection used the defective quantile `rmse_log`: "
        f"`{defect.get('validation_selection_affected')}`",
        f"- {defect.get('validation_selection_note')}",
        "",
        "## Official-seq frozen holdout (this run)",
        "",
        f"- generate_seed / train_seed: `{out.get('freeze_lock', {}).get('generate_seed')}` / "
        f"`{out.get('freeze_lock', {}).get('train_seed')}`",
        f"- encoder cutoff: DAY ≤ {out.get('encoder_window_day')} (observed visits only)",
        f"- holdout hashes: `{out.get('target_holdout_hashes')}`",
        "",
        "| model | MAE | RMSE | pinball | n_studies | merge 1-1 | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, b in (out.get("results") or {}).items():
        if b.get("status") != "ok":
            lines.append(
                f"| {key} | — | — | — | — | — | {b.get('status')}: "
                f"{(b.get('reason') or '')[:80]} |"
            )
            continue
        mg = b.get("merge") or {}
        lines.append(
            f"| {key} | {_fmt(b.get('mae_log'))} | {_fmt(b.get('rmse_log'))} | "
            f"{_fmt(b.get('mean_quantile_loss'))} | {b.get('n_studies')} | "
            f"{mg.get('one_to_one')} | ok |"
        )
    lines += [
        "",
        "### Paired transfer (pooled − target-only; Δ>0 harmful)",
        "",
        "| family | metric | target-only | pooled | Δ | sign |",
        "|---|---|---|---|---|---|",
    ]
    for fam, block in (out.get("paired_transfer") or {}).items():
        rows = block.get("rows") or []
        if not rows:
            lines.append(f"| {fam} | — | — | — | — | not paired (incomplete) |")
            continue
        for r in rows:
            lines.append(
                f"| {fam} | {r.get('metric')} | {r.get('target_only')} | {r.get('pooled')} | "
                f"{r.get('delta_pooled_minus_target')} | {r.get('sign')} |"
            )
    lines += [
        "",
        "## Chronos-2 fine-tunes (not pooled when incomplete)",
        "",
        f"- history-only complete: `{(out.get('chronos_finetune_pool') or {}).get('history_only_complete')}`",
        f"- covariate-aware complete: `{(out.get('chronos_finetune_pool') or {}).get('covariate_aware_complete')}`",
        f"- aggregated_across_incomplete: `{(out.get('chronos_finetune_pool') or {}).get('aggregated_across_incomplete')}`",
        f"- failures: `{out.get('failures')}`",
        "",
        "## Guardrails",
        "",
        f"- real_data_accessed: {out.get('real_data_accessed')}",
        f"- design_run: {out.get('design_run')}",
        f"- predose_evaluated: {out.get('predose_evaluated')}",
        f"- new_architectures: {out.get('new_architectures')}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (parent / "RESULTS.md").write_text(text, encoding="utf-8")
    Path("docs/v3/cov_chronos_RESULTS.md").write_text(text, encoding="utf-8")
    Path("docs/v3/defect_trace.md").write_text(
        "# MAE/RMSE and grouping defect trace\n\n"
        "> **Synthetic diagnostic. Not design qualification. No real-data validation.**\n\n"
        f"- Affected published metrics: `{defect.get('affected_published_metrics')}`\n"
        f"- Rescore needed: `{defect.get('rescore_needed')}`\n"
        f"- Validation-selection affected: `{defect.get('validation_selection_affected')}`\n\n"
        f"{defect.get('validation_selection_note')}\n\n"
        "Forecast.py runs (v3_eval through v3_objective) already published true RMSE "
        "grouped by STUDYID. Official-seq and seq-diag published `|error|` as `rmse_log` "
        "and often grouped by animal; those tables are preserved and were rescored in "
        "`runs/v3_eval_correct`.\n",
        encoding="utf-8",
    )

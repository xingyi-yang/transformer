"""Locked sharing replication: target-only vs pooled vs implemented selective.

Uses existing models and the PR #12 corrected scorer. Chronos history-only is
the Chronos comparator. TFT/Chronos selective paths are published as
not_implemented. No fabricated history, real data, design, or new families.
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
from send_bw.models.chronos2 import HISTORY_ONLY, chronos_fit_preflight
from send_bw.sharing import LABELS, SELECTIVE_REGULARIZED, SELECTIVE_V1
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_official_seq import (
    _score_gaussian,
    _score_quantiles,
    collect_versions,
)
from send_bw.v3_research import _bank_cfg, _cell_cfg, _run_from_cfg, load_yaml

PRESERVED = (
    "runs/v3_eval", "runs/v3_pilot_k0k1", "runs/v3_repl", "runs/v3_research",
    "runs/v3_closeout", "runs/v3_audit", "runs/v3_increment", "runs/v3_phaseb",
    "runs/v3_limitations", "runs/v3_objective", "runs/v3_official_seq",
    "runs/v3_seq_diag", "runs/v3_eval_correct", "runs/v3_cov_chronos",
)
WINDOW = 14.0
REGIMES = (
    {"key": "target_only", "sharing": "context_specific", "k_field": "n_donor_target_only"},
    {"key": "pooled", "sharing": "pooled", "k_field": "n_donor_pooled"},
    {"key": "selective", "sharing": "selective", "k_field": "n_donor_selective"},
    {"key": "selective_regularized", "sharing": "selective_regularized",
     "k_field": "n_donor_selective"},
)
TFT_CHRONOS_KEYS = {"target_only", "pooled"}


class ShareReplError(ValueError):
    pass


def locked_budget(cfg: dict[str, Any]) -> dict[str, Any]:
    """Seed counts and runtime estimate. Call before execution."""
    ev = cfg.get("share_repl") or {}
    n_ds = int(ev.get("n_dataset_seeds") or 2)
    n_tr = int(ev.get("n_train_seeds") or 1)
    g0 = int(cfg["generate_seed"])
    t0s = int(cfg["train_seed"])
    g_stride = int(ev.get("generate_seed_stride") or 1009)
    t_stride = int(ev.get("train_seed_stride") or 17)
    return {
        "n_dataset_seeds": n_ds,
        "n_train_seeds": n_tr,
        "generate_seeds": [g0 + i * g_stride for i in range(n_ds)],
        "train_seeds": [t0s + j * t_stride for j in range(n_tr)],
        "generate_seed_stride": g_stride,
        "train_seed_stride": t_stride,
        "bank": str(ev.get("bank") or "matched"),
        "n_cells": n_ds * n_tr * len(REGIMES),
        "donor_policy": {
            "target_only": int(ev.get("n_donor_target_only") or 0),
            "pooled": int(ev.get("n_donor_pooled") or 3),
            "selective": int(ev.get("n_donor_selective") or 1),
            "source": ev.get("donor_policy_source") or "training_validation_protocol",
            "selected_from_test": False,
        },
        "checkpoint_source": ev.get("checkpoint_source") or "training_validation",
        "chronos_mode": ev.get("chronos_mode") or HISTORY_ONLY,
        "tft_selective": ev.get("tft_selective") or "not_implemented",
        "chronos_selective": ev.get("chronos_selective") or "not_implemented",
        "covariate_chronos": bool(ev.get("covariate_chronos")),
        "estimated_runtime_seconds": ev.get("estimated_runtime_seconds"),
    }


def tft_chronos_unsupported(key: str, family: str) -> dict[str, Any] | None:
    if key in TFT_CHRONOS_KEYS:
        return None
    return {
        "status": "not_implemented",
        "reason": f"{family} has no selective-sharing training path",
    }


def _compact(sc: dict[str, Any]) -> dict[str, Any]:
    if sc.get("status") != "ok":
        return {k: sc.get(k) for k in ("status", "reason") if k in sc}
    mg = sc.get("merge") or {}
    return {
        "status": "ok",
        "mae_log": (sc.get("mae_log") or {}).get("study_balanced"),
        "rmse_log": (sc.get("rmse_log") or {}).get("study_balanced"),
        "mean_quantile_loss": (sc.get("mean_quantile_loss") or {}).get("study_balanced"),
        "n_studies": (sc.get("mae_log") or {}).get("n_studies"),
        "n_scored": sc.get("n_scored"),
        "grouping": sc.get("grouping"),
        "merge_one_to_one": mg.get("one_to_one"),
        "chronos_label": sc.get("chronos_label"),
        "seconds": sc.get("seconds"),
    }


def _score_last_value(data, window_day: float, pred_path: Path | None = None) -> dict[str, Any]:
    hist, fut = split_encoder_future(data, window_day)
    pred = last_value_predictions(hist, fut, QUANTILES)
    if pred_path is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(pred_path, index=False)
    out = score_quantile_table(fut, pred, quantiles=QUANTILES, representation="last_value")
    out["encoder_fabricated"] = False
    return out


def _sign_rows(block: dict[str, Any]) -> dict[str, Any]:
    for row in block.get("rows") or []:
        d = row.get("delta_pooled_minus_target")
        if d is None:
            row["sign"] = "null"
        elif d > 0:
            row["sign"] = "harmful"
        elif d < 0:
            row["sign"] = "helpful"
        else:
            row["sign"] = "null"
    return block


def execute_v3_share_repl(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or "runs/v3_share_repl")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise ShareReplError("refusing to write into a preserved artifact directory")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise ShareReplError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise ShareReplError("design optimization stays off")
    ev = cfg.get("share_repl") or {}
    if ev.get("retune_after_test"):
        raise ShareReplError("retune_after_test is forbidden")
    if ev.get("predose"):
        raise ShareReplError("predose stays off")
    if ev.get("bayesian"):
        raise ShareReplError("Bayesian stays off")
    if ev.get("covariate_chronos"):
        raise ShareReplError("covariate Chronos is not part of this lock")
    parent.mkdir(parents=True, exist_ok=True)
    window = float(ev.get("encoder_window_day") or WINDOW)
    budget = locked_budget(cfg)
    n_ds = int(budget["n_dataset_seeds"])
    n_tr = int(budget["n_train_seeds"])
    g0 = int(cfg["generate_seed"])
    t0s = int(cfg["train_seed"])
    g_stride = int(budget["generate_seed_stride"])
    t_stride = int(budget["train_seed_stride"])
    bank = str(budget["bank"])
    versions = collect_versions()
    write_json(parent / "versions.json", versions)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(),
        **budget,
        "retune_after_test": False,
        "chronos_mode": HISTORY_ONLY,
        "tft_selective": "not_implemented",
        "chronos_selective": "not_implemented",
        "selective_v1": SELECTIVE_V1,
        "selective_regularized": SELECTIVE_REGULARIZED,
        "fabricate_encoder_history": False, "real_data": False, "design": False,
        "new_architectures": False, "predose": False,
        "labels": list(LABELS),
        "preserved": list(PRESERVED),
    }
    write_json(parent / "freeze_lock.json", lock)

    t_wall = time.time()
    failures: list[dict[str, Any]] = []
    replicates: list[dict[str, Any]] = []

    from send_bw.models.chronos2 import OFFICIAL_ID, Chronos2EarlyHistoryModel, verify_chronos2_api
    from send_bw.models.tft import TFTEarlyHistoryModel
    from send_bw.pipeline import model_class

    cver = verify_chronos2_api()
    tft_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("tft") or {})
    ch_cfg = dict((cfg.get("models") or {}).get("optional", {}).get("chronos2") or {})
    ch_cfg.setdefault("pretrained_id", OFFICIAL_ID)
    ch_cfg.setdefault("device_map", "cpu")
    ch_cfg.setdefault("encoder_window_day", window)
    ch_cfg.setdefault("download_permission", True)
    tft_cfg.setdefault("encoder_window_day", window)

    for i in range(n_ds):
        gen = g0 + i * g_stride
        for j in range(n_tr):
            trn = t0s + j * t_stride
            bcfg = _bank_cfg(cfg, parent, i, gen, bank)
            br = _run_from_cfg(bcfg)
            br.generate(force=force)
            cells: dict[str, Any] = {}
            hashes: dict[str, Any] = {}
            reuse = None
            for reg in REGIMES:
                k = int(ev.get(reg["k_field"]) or 0)
                spec = {
                    "name": f"d{i}_t{j}_{bank}_{reg['key']}",
                    "dataset_index": i, "train_index": j, "bank": bank,
                    "sharing": reg["sharing"], "k": k,
                    "sparse_target_train_n": None,
                    "generate_seed": gen, "train_seed": trn,
                    "models": ["spline", "transformer"],
                }
                ccfg = _cell_cfg(cfg, parent, spec, generate_seed=gen, train_seed=trn,
                                 models=spec["models"], sparse_target_train_n=None,
                                 reuse_target_from=reuse)
                ccfg["evaluation"]["tasks"] = ["early_history"]
                ccfg["evaluation"]["early_history_window_day"] = window
                ccfg["design"] = {}
                run = _run_from_cfg(ccfg)
                run.ingest(force=force)
                run.prepare(force=force)
                run.train(["spline", "transformer"], force=force)
                cells[reg["key"]] = run
                man = read_json(run.p("splits", "target_manifest.json"))
                hashes[reg["key"]] = {
                    n: (man.get(n) or {}).get("hash")
                    for n in ("validation", "test", "all_target")
                }
                if reuse is None:
                    reuse = str(run.p("splits", "x").parent)
            test_hash = hashes["target_only"]["test"]
            val_hash = hashes["target_only"]["validation"]
            if any(h.get("test") != test_hash or h.get("validation") != val_hash
                   for h in hashes.values()):
                raise ShareReplError("target holdout hashes differ across sharing cells")

            test = load_cell_prepared(Path(cells["target_only"].dir))["data"]["test"]
            pred_dir = parent / f"repl/d{i}_t{j}/predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            results: dict[str, Any] = {}

            for key, run in cells.items():
                for name in ("spline", "transformer"):
                    mk = f"{name}_{key}"
                    t1 = time.time()
                    try:
                        model = model_class(name).load(run.p("models", name))
                        sc = _score_gaussian(model, test, window, pred_dir / f"{mk}.csv")
                        sc["seconds"] = time.time() - t1
                        results[mk] = sc
                    except Exception as e:  # noqa: BLE001
                        results[mk] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
                        failures.append({"dataset_index": i, "model": mk, "reason": str(e),
                                         "scored": False})

            t1 = time.time()
            lv = _score_last_value(test, window, pred_dir / "last_value.csv")
            lv["seconds"] = time.time() - t1
            results["last_value"] = lv

            for key in ("target_only", "pooled", "selective", "selective_regularized"):
                mk = f"tft_{key}"
                skipped = tft_chronos_unsupported(key, "TFT")
                if skipped is not None:
                    results[mk] = skipped
                    continue
                t1 = time.time()
                try:
                    prep = load_cell_prepared(Path(cells[key].dir))
                    model = TFTEarlyHistoryModel()
                    hist = model.fit(prep["data"]["train"], prep["data"]["validation"],
                                     {**tft_cfg, "seed": trn, "training_scope": key})
                    dest = parent / "models" / f"d{i}_{mk}"
                    model.save(dest)
                    write_json(dest / "training_history.json", hist)
                    sc = _score_quantiles(model, test, window, "tft_quantiles",
                                          pred_dir / f"{mk}.csv")
                    sc["seconds"] = time.time() - t1
                    results[mk] = sc
                except Exception as e:  # noqa: BLE001
                    results[mk] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                                    "traceback": traceback.format_exc()[-1500:]}
                    failures.append({"dataset_index": i, "model": mk, "reason": str(e),
                                     "scored": False})

            if not cver.get("ok"):
                for key in ("zero_shot_history", "target_only_history", "pooled_history"):
                    results[f"chronos2_{key}"] = {"status": "not_run", "reason": cver.get("reason")}
                    failures.append({"dataset_index": i, "model": f"chronos2_{key}",
                                     "reason": cver.get("reason"), "scored": False})
            else:
                t1 = time.time()
                try:
                    zs = Chronos2EarlyHistoryModel()
                    zh = zs.load_pretrained({**ch_cfg, "covariate_mode": HISTORY_ONLY})
                    dest = parent / "models" / f"d{i}_chronos2_zero_shot"
                    zs.save(dest)
                    write_json(dest / "training_history.json", zh)
                    sc = _score_quantiles(zs, test, window, "chronos2_quantiles",
                                          pred_dir / "chronos2_zero_shot_history.csv",
                                          chronos_label=HISTORY_ONLY)
                    sc["seconds"] = time.time() - t1
                    results["chronos2_zero_shot_history"] = sc
                except Exception as e:  # noqa: BLE001
                    results["chronos2_zero_shot_history"] = {
                        "status": "failed", "reason": f"{type(e).__name__}: {e}"}
                    failures.append({"dataset_index": i, "model": "chronos2_zero_shot_history",
                                     "reason": str(e), "scored": False})
                for key in ("target_only", "pooled", "selective", "selective_regularized"):
                    mk = f"chronos2_{key}_history"
                    skipped = tft_chronos_unsupported(key, "Chronos-2")
                    if skipped is not None:
                        results[mk] = skipped
                        continue
                    t1 = time.time()
                    try:
                        prep = load_cell_prepared(Path(cells[key].dir))
                        pre = chronos_fit_preflight(
                            prep["data"]["train"], prep["data"]["validation"],
                            min_past=int(ch_cfg.get("min_past", 2)),
                            prediction_length=int(ch_cfg.get("prediction_length", 6)),
                            covariate_mode=HISTORY_ONLY)
                        write_json(parent / "preflight" / f"d{i}_{mk}.json", pre)
                        if not pre["can_fit"]:
                            raise RuntimeError(
                                "preflight: train empty after Chronos length filter "
                                f"(kept {pre['train']['n_kept']})"
                            )
                        model = Chronos2EarlyHistoryModel()
                        ft_dir = parent / "models" / f"d{i}_{mk}" / "finetune"
                        hist = model.fit(prep["data"]["train"], prep["data"]["validation"], {
                            **ch_cfg, "mode": f"finetune_{key}", "finetune_dir": str(ft_dir),
                            "covariate_mode": HISTORY_ONLY,
                        })
                        dest = parent / "models" / f"d{i}_{mk}"
                        model.save(dest)
                        write_json(dest / "training_history.json", hist)
                        sc = _score_quantiles(
                            model, test, window, "chronos2_quantiles",
                            pred_dir / f"{mk}.csv", chronos_label=HISTORY_ONLY)
                        sc["seconds"] = time.time() - t1
                        sc["preflight"] = pre
                        results[mk] = sc
                    except Exception as e:  # noqa: BLE001
                        results[mk] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                                        "traceback": traceback.format_exc()[-1500:]}
                        failures.append({"dataset_index": i, "model": mk, "reason": str(e),
                                         "scored": False})

            transfer = {
                "spline_pooled": _sign_rows(paired_transfer(
                    results.get("spline_target_only") or {}, results.get("spline_pooled") or {})),
                "spline_selective": _sign_rows(paired_transfer(
                    results.get("spline_target_only") or {}, results.get("spline_selective") or {})),
                "spline_selective_regularized": _sign_rows(paired_transfer(
                    results.get("spline_target_only") or {},
                    results.get("spline_selective_regularized") or {})),
                "transformer_pooled": _sign_rows(paired_transfer(
                    results.get("transformer_target_only") or {},
                    results.get("transformer_pooled") or {})),
                "transformer_selective": _sign_rows(paired_transfer(
                    results.get("transformer_target_only") or {},
                    results.get("transformer_selective") or {})),
                "transformer_selective_regularized": _sign_rows(paired_transfer(
                    results.get("transformer_target_only") or {},
                    results.get("transformer_selective_regularized") or {})),
                "tft_pooled": _sign_rows(paired_transfer(
                    results.get("tft_target_only") or {}, results.get("tft_pooled") or {})),
                "chronos2_finetune_history": _sign_rows(paired_transfer(
                    results.get("chronos2_target_only_history") or {},
                    results.get("chronos2_pooled_history") or {})),
            }
            replicates.append({
                "dataset_index": i, "train_index": j,
                "generate_seed": gen, "train_seed": trn,
                "target_holdout_hashes": hashes,
                "results": {k: _compact(v) if isinstance(v, dict) else v
                            for k, v in results.items()},
                "paired_transfer": transfer,
            })

    auth = discover_authorization(None)
    chronos_ok = [
        r for r in replicates
        if (r.get("results") or {}).get("chronos2_target_only_history", {}).get("status") == "ok"
        and (r.get("results") or {}).get("chronos2_pooled_history", {}).get("status") == "ok"
    ]
    out = {
        "labels": list(LABELS) + [
            "Sharing replication.", "Corrected MAE and true RMSE.",
            "Chronos-2 is history-only/univariate.",
        ],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "versions": versions,
        "replicates": replicates,
        "failures": failures,
        "chronos_finetune_pool": {
            "n_dataset_train_pairs": len(replicates),
            "n_complete": len(chronos_ok),
            "aggregated_across_incomplete": False,
            "label": HISTORY_ONLY,
        },
        "real_data_accessed": False,
        "authorization_present": bool(auth.get("authorized")),
        "design_run": False,
        "predose_evaluated": False,
        "new_architectures": False,
        "seconds_total": time.time() - t_wall,
    }
    write_json(parent / "SUMMARY.json", out)
    _write_report(parent, out)
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(parent: Path, out: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    lines = [
        "# Sharing replication — locked compute budget",
        "",
        "> **Synthetic diagnostic. Exploratory. Not design qualification. "
        "No real-data validation. No operational recommendation.**",
        "",
        f"- n_dataset_seeds: `{lock.get('n_dataset_seeds')}`; "
        f"n_train_seeds: `{lock.get('n_train_seeds')}`",
        f"- generate_seeds: `{lock.get('generate_seeds')}`; "
        f"train_seeds: `{lock.get('train_seeds')}`",
        f"- donor policy (not from test): `{lock.get('donor_policy')}`",
        f"- Chronos: `{lock.get('chronos_mode')}`; "
        f"TFT/Chronos selective: not_implemented",
        f"- estimated_runtime_seconds: `{lock.get('estimated_runtime_seconds')}`",
        f"- seconds_total (actual): `{out.get('seconds_total')}`",
        "",
        "## Replicates",
        "",
    ]
    for rec in out.get("replicates") or []:
        lines += [
            f"### dataset {rec.get('dataset_index')} / train {rec.get('train_index')}",
            "",
            "| model | MAE | RMSE | pinball | status |",
            "|---|---|---|---|---|",
        ]
        for key, b in (rec.get("results") or {}).items():
            if b.get("status") != "ok":
                lines.append(
                    f"| {key} | — | — | — | {b.get('status')}: {(b.get('reason') or '')[:90]} |"
                )
                continue
            lines.append(
                f"| {key} | {_fmt(b.get('mae_log'))} | {_fmt(b.get('rmse_log'))} | "
                f"{_fmt(b.get('mean_quantile_loss'))} | ok |"
            )
        lines += [
            "",
            "Paired transfer (sharing − target-only; Δ>0 harmful):",
            "",
            "| family | metric | target-only | other | Δ | sign |",
            "|---|---|---|---|---|---|",
        ]
        for fam, block in (rec.get("paired_transfer") or {}).items():
            rows = block.get("rows") or []
            if not rows:
                lines.append(f"| {fam} | — | — | — | — | not paired |")
                continue
            for r in rows:
                lines.append(
                    f"| {fam} | {r.get('metric')} | {r.get('target_only')} | {r.get('pooled')} | "
                    f"{r.get('delta_pooled_minus_target')} | {r.get('sign')} |"
                )
        lines.append("")
    lines += [
        "## Failures (not aggregated as complete)",
        "",
        f"- chronos complete pairs: `{(out.get('chronos_finetune_pool') or {}).get('n_complete')}` / "
        f"`{(out.get('chronos_finetune_pool') or {}).get('n_dataset_train_pairs')}`",
        "- aggregated_across_incomplete: False",
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
    Path("docs/v3/share_repl_RESULTS.md").write_text(text, encoding="utf-8")

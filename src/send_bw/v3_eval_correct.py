"""Evaluation-correctness rescore from saved predictions. No training.

Official-seq and seq-diag artifacts stay frozen. ``rmse_log`` on those tables
was |error|. This executor publishes named study-balanced MAE and true RMSE
with STUDYID grouping, merge cardinality, horizon units, and Chronos
eligibility. Fine-tune failures are not pooled with completed scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from send_bw.evaluation.quantile_forecast import (
    QUANTILES,
    last_value_predictions,
    paired_transfer,
    score_quantile_table,
    series_eligibility,
    split_encoder_future,
)
from send_bw.historical_audit import discover_authorization
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_research import load_yaml

PRESERVED = (
    "runs/v3_eval", "runs/v3_pilot_k0k1", "runs/v3_repl", "runs/v3_research",
    "runs/v3_closeout", "runs/v3_audit", "runs/v3_increment", "runs/v3_phaseb",
    "runs/v3_limitations", "runs/v3_objective", "runs/v3_official_seq",
    "runs/v3_seq_diag",
)
WINDOW = 14.0
REPR = {
    "spline_target_only": "gaussian",
    "spline_pooled": "gaussian",
    "transformer_target_only": "gaussian",
    "transformer_pooled": "gaussian",
    "tft_target_only": "tft_quantiles",
    "tft_pooled": "tft_quantiles",
    "chronos2_zero_shot": "chronos2_quantiles",
    "chronos2_target_only": "chronos2_quantiles",
    "chronos2_pooled": "chronos2_quantiles",
    "last_value": "last_value",
}


class EvalCorrectError(ValueError):
    pass


def _compact(sc: dict[str, Any]) -> dict[str, Any]:
    if sc.get("status") != "ok":
        return {k: sc.get(k) for k in ("status", "reason") if k in sc}
    out = {
        "status": "ok",
        "mae_log": (sc.get("mae_log") or {}).get("study_balanced"),
        "rmse_log": (sc.get("rmse_log") or {}).get("study_balanced"),
        "mae_log_pooled": (sc.get("mae_log") or {}).get("pooled"),
        "rmse_log_pooled": (sc.get("rmse_log") or {}).get("pooled"),
        "legacy_animal_balanced_mae": (sc.get("legacy_animal_balanced_mae") or {}).get("study_balanced"),
        "n_studies": (sc.get("mae_log") or {}).get("n_studies"),
        "n_scored": sc.get("n_scored"),
        "mean_quantile_loss": (sc.get("mean_quantile_loss") or {}).get("study_balanced"),
        "coverage80": (sc.get("coverage80") or {}).get("study_balanced"),
        "interval_width80": (sc.get("interval_width80") or {}).get("study_balanced"),
        "crps_justified": sc.get("crps_justified"),
        "point_definition": (sc.get("point_definition") or {}).get("point"),
        "grouping": sc.get("grouping"),
        "merge": sc.get("merge"),
        "per_horizon_day": sc.get("per_horizon_day"),
        "per_horizon_step": sc.get("per_horizon_step"),
        "chronos_label": sc.get("chronos_label"),
        "covariates_used": sc.get("covariates_used"),
    }
    return out


def _score_csv(obs: pd.DataFrame, path: Path, representation: str) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_run", "reason": f"missing {path}"}
    pred = pd.read_csv(path)
    return score_quantile_table(obs, pred, quantiles=QUANTILES, representation=representation)


def _score_holdout(name: str, obs: pd.DataFrame, pred_dir: Path, hist: pd.DataFrame | None,
                   dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for key, rep in REPR.items():
        if key == "last_value":
            if hist is None:
                continue
            pred = last_value_predictions(hist, obs, QUANTILES)
            pred.to_csv(dest / "last_value.csv", index=False)
            sc = score_quantile_table(obs, pred, quantiles=QUANTILES, representation=rep)
        else:
            sc = _score_csv(obs, pred_dir / f"{key}.csv", rep)
        if key.startswith("chronos2") and sc.get("status") == "ok":
            sc["chronos_label"] = "history_only_univariate"
            sc["covariates_used"] = False
        write_json(dest / f"{key}.json", sc)
        results[key] = _compact(sc)
    transfer = {
        "spline": paired_transfer(results.get("spline_target_only") or {},
                                  results.get("spline_pooled") or {}),
        "transformer": paired_transfer(results.get("transformer_target_only") or {},
                                       results.get("transformer_pooled") or {}),
        "tft": paired_transfer(results.get("tft_target_only") or {},
                               results.get("tft_pooled") or {}),
        "chronos2_finetune": paired_transfer(results.get("chronos2_target_only") or {},
                                             results.get("chronos2_pooled") or {}),
    }
    return {"name": name, "results": results, "paired_transfer": transfer}


def _eligibility(cell: Path, min_past: int, prediction_length: int) -> dict[str, Any]:
    prep = load_cell_prepared(cell)
    train = prep["data"]["train"].obs
    return series_eligibility(train, min_past=min_past, prediction_length=prediction_length)


def _rank(results: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    rows = []
    for k, b in results.items():
        if b.get("status") != "ok" or b.get(metric) is None:
            continue
        rows.append({"model": k, metric: b[metric]})
    rows.sort(key=lambda r: r[metric])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def _sign_table(old_rows: list[dict[str, Any]] | None, new_block: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    old_map = {r.get("metric"): r for r in (old_rows or [])}
    for row in (new_block.get("rows") or []):
        rec = {
            "metric": row.get("metric"),
            "target_only": row.get("target_only"),
            "pooled": row.get("pooled"),
            "delta": row.get("delta_pooled_minus_target"),
            "sign": "harmful" if (row.get("delta_pooled_minus_target") or 0) > 0 else "helpful",
        }
        old = old_map.get("rmse_log") if row.get("metric") in {"mae_log", "rmse_log"} else old_map.get(row.get("metric"))
        if old is not None:
            rec["published_delta"] = old.get("delta_pooled_minus_target")
            rec["published_sign"] = (
                "harmful" if (old.get("delta_pooled_minus_target") or 0) > 0 else "helpful")
            rec["sign_changed"] = rec["sign"] != rec["published_sign"]
        out.append(rec)
    return out


def execute_v3_eval_correct(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    del force
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or "runs/v3_eval_correct")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise EvalCorrectError("refusing to write into a preserved artifact directory")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise EvalCorrectError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise EvalCorrectError("design optimization stays off")
    if (cfg.get("eval_correct") or {}).get("train"):
        raise EvalCorrectError("training is off; rescore saved predictions only")
    parent.mkdir(parents=True, exist_ok=True)
    ev = cfg.get("eval_correct") or {}
    official = Path(ev.get("official_seq_dir") or "runs/v3_official_seq")
    diag = Path(ev.get("seq_diag_dir") or "runs/v3_seq_diag")
    window = float(ev.get("encoder_window_day") or WINDOW)
    min_past = int(ev.get("chronos_min_past") or 2)
    pred_len = int(ev.get("chronos_prediction_length") or 6)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(),
        "training": False, "real_data": False, "design": False,
        "predose": False, "new_architectures": False,
        "official_seq_dir": str(official), "seq_diag_dir": str(diag),
        "official_seq_rewritten": False, "seq_diag_rewritten": False,
        "labels": list(LABELS),
        "rmse_log_definition": "true RMSE; prior published rmse_log was |error|",
        "grouping": "STUDYID",
    }
    write_json(parent / "freeze_lock.json", lock)

    off_cell = official / "datasets/d0/trains/t0/cells/d0_t0_matched_target_only"
    off_test = load_cell_prepared(off_cell)["data"]["test"]
    hist, fut = split_encoder_future(off_test, window)
    official_block = _score_holdout(
        "official_seq_frozen_holdout", fut, official / "predictions", hist,
        parent / "official_seq")
    official_summary = read_json(official / "SUMMARY.json") if (official / "SUMMARY.json").exists() else {}
    official_block["published_paired_transfer"] = official_summary.get("paired_transfer")
    official_block["sign_vs_published"] = {
        fam: _sign_table((official_summary.get("paired_transfer") or {}).get(fam, {}).get("rows"),
                         official_block["paired_transfer"].get(fam) or {})
        for fam in ("spline", "transformer", "tft", "chronos2_finetune")
    }
    official_block["rank_mae"] = _rank(official_block["results"], "mae_log")
    official_block["rank_rmse"] = _rank(official_block["results"], "rmse_log")
    official_block["rank_legacy_animal_mae"] = _rank(official_block["results"], "legacy_animal_balanced_mae")

    eligibility = {
        "official_seq_target_only_train": _eligibility(off_cell, min_past, pred_len),
        "official_seq_pooled_train": _eligibility(
            official / "datasets/d0/trains/t0/cells/d0_t0_matched_pooled", min_past, pred_len),
    }

    replicates = []
    ft_failures = []
    for i in (0, 1):
        cell = diag / f"repl/d{i}/datasets/d{i}/trains/t0/cells/d{i}_t0_matched_target_only"
        pred_dir = diag / f"repl/d{i}/predictions"
        if not cell.exists():
            replicates.append({"dataset_index": i, "status": "not_run", "reason": f"missing {cell}"})
            continue
        test = load_cell_prepared(cell)["data"]["test"]
        h, f = split_encoder_future(test, window)
        block = _score_holdout(f"seq_diag_d{i}", f, pred_dir, h, parent / f"seq_diag_d{i}")
        block["dataset_index"] = i
        block["eligibility_target_only_train"] = _eligibility(cell, min_past, pred_len)
        pooled_cell = diag / f"repl/d{i}/datasets/d{i}/trains/t0/cells/d{i}_t0_matched_pooled"
        if pooled_cell.exists():
            block["eligibility_pooled_train"] = _eligibility(pooled_cell, min_past, pred_len)
        repl = read_json(diag / f"repl/d{i}/replicate.json") if (diag / f"repl/d{i}/replicate.json").exists() else {}
        for key in ("chronos2_target_only", "chronos2_pooled"):
            raw = ((repl.get("results") or {}).get(key) or {})
            if raw.get("status") == "failed" or block["results"].get(key, {}).get("status") != "ok":
                ft_failures.append({
                    "dataset_index": i, "model": key,
                    "reason": raw.get("reason") or (block["results"].get(key) or {}).get("reason"),
                    "scored": False,
                })
                block["results"][key] = {
                    "status": "failed",
                    "reason": raw.get("reason") or (block["results"].get(key) or {}).get("reason"),
                    "excluded_from_pooled_finetune": True,
                }
        block["rank_mae"] = _rank(block["results"], "mae_log")
        block["rank_rmse"] = _rank(block["results"], "rmse_log")
        replicates.append(block)
        eligibility[f"seq_diag_d{i}_target_only_train"] = block.get("eligibility_target_only_train")

    ft_ok = [r for r in replicates if (r.get("results") or {}).get("chronos2_target_only", {}).get("status") == "ok"]
    chronos_pool_note = (
        "Chronos-2 LoRA scores are not averaged across datasets. "
        f"{len(ft_ok)} of {len(replicates)} datasets completed fine-tunes."
    )

    auth = discover_authorization(None)
    out = {
        "labels": list(LABELS) + [
            "Evaluation correctness.", "No additional training.",
            "Chronos-2 is history-only/univariate.",
        ],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "superseded": {
            "runs/v3_official_seq": "rmse_log was |error|; grouping often USUBJID after STUDYID merge collision",
            "runs/v3_seq_diag": "same metric name; Chronos LoRA pooled as if both datasets completed",
            "docs/v3/official_seq_RESULTS.md": "tables preserved; banner marks superseded",
            "docs/v3/seq_diag_RESULTS.md": "tables preserved; banner marks superseded",
        },
        "official_seq": official_block,
        "replicates": replicates,
        "chronos_finetune_failures": ft_failures,
        "chronos_finetune_pool": {
            "n_datasets": len(replicates),
            "n_finetune_complete": len(ft_ok),
            "aggregated_across_incomplete_datasets": False,
            "note": chronos_pool_note,
            "label": "history_only_univariate",
        },
        "eligibility": eligibility,
        "horizon_units": {
            "calendar_day": "DAY",
            "visit_step": "observed future-visit rank",
            "fabricated_history": False,
        },
        "real_data_accessed": False,
        "authorization_present": bool(auth.get("authorized")),
        "design_run": False,
        "training_run": False,
        "predose_evaluated": False,
        "covariate_aware_comparison": False,
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
    off = out.get("official_seq") or {}
    lines = [
        "# Evaluation correctness — named MAE and true RMSE",
        "",
        "> **Synthetic diagnostic. Exploratory. Not design qualification. "
        "No real-data validation. No operational recommendation.**",
        "",
        "No additional training. Prior artifacts are preserved and **superseded** "
        "for the name `rmse_log` (that quantity was `|error|`) and for Chronos-2 "
        "fine-tune pooling. Chronos-2 results here are **history-only / univariate**. "
        "A covariate-aware comparison has not been launched.",
        "",
        "## Definitions",
        "",
        "- Point: q0.5 for quantile models (Chronos official mean = median); MEAN for Gaussian.",
        "- **MAE** (`mae_log`): study-balanced mean of `|point − LOGW|` (STUDYID).",
        "- **RMSE** (`rmse_log`): per-study `sqrt(mean e²)`, then the unweighted mean of studies.",
        "- Legacy published `rmse_log` = MAE, often grouped by USUBJID after a STUDYID merge collision.",
        "- Horizon units: calendar DAY and observed future-visit rank. No fabricated days.",
        "",
        "## Official-seq frozen holdout (rescored)",
        "",
        "| model | MAE | RMSE | legacy animal MAE | pinball | n_studies | n | merge 1-1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, b in (off.get("results") or {}).items():
        if b.get("status") != "ok":
            lines.append(f"| {key} | {b.get('status')} | — | — | — | — | — | — |")
            continue
        mg = b.get("merge") or {}
        lines.append(
            f"| {key} | {_fmt(b.get('mae_log'))} | {_fmt(b.get('rmse_log'))} | "
            f"{_fmt(b.get('legacy_animal_balanced_mae'))} | {_fmt(b.get('mean_quantile_loss'))} | "
            f"{b.get('n_studies')} | {b.get('n_scored')} | {mg.get('one_to_one')} |"
        )
    lines += [
        "",
        "### Rankings (lower is better)",
        "",
        f"- MAE: `{[r['model'] for r in off.get('rank_mae') or []]}`",
        f"- RMSE: `{[r['model'] for r in off.get('rank_rmse') or []]}`",
        f"- Legacy animal MAE: `{[r['model'] for r in off.get('rank_legacy_animal_mae') or []]}`",
        "",
        "### Paired transfer (pooled − target-only; Δ>0 harmful)",
        "",
        "| family | metric | target-only | pooled | Δ | sign | published Δ | sign changed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fam, rows in (off.get("sign_vs_published") or {}).items():
        for r in rows:
            lines.append(
                f"| {fam} | {r.get('metric')} | {r.get('target_only')} | {r.get('pooled')} | "
                f"{r.get('delta')} | {r.get('sign')} | {r.get('published_delta')} | "
                f"{r.get('sign_changed')} |"
            )
    lines += ["", "## Seq-diag replicates (rescored; no retraining)", ""]
    for rec in out.get("replicates") or []:
        lines.append(f"### dataset {rec.get('dataset_index')}")
        lines.append("")
        lines.append("| model | MAE | RMSE | pinball | status |")
        lines.append("|---|---|---|---|---|")
        for key, b in (rec.get("results") or {}).items():
            if b.get("status") != "ok":
                lines.append(f"| {key} | — | — | — | {b.get('status')}: {(b.get('reason') or '')[:80]} |")
                continue
            lines.append(
                f"| {key} | {_fmt(b.get('mae_log'))} | {_fmt(b.get('rmse_log'))} | "
                f"{_fmt(b.get('mean_quantile_loss'))} | ok |"
            )
        el = rec.get("eligibility_target_only_train") or {}
        lines.append("")
        lines.append(
            f"- train eligibility (observed visits ≥ {el.get('required_observed_visits')}): "
            f"{el.get('n_eligible')}/{el.get('n_series')} eligible; "
            f"histogram `{el.get('length_histogram')}`; fabricated_history={el.get('fabricated_history')}"
        )
        lines.append("")
    lines += [
        "## Chronos-2 fine-tunes (not pooled across incomplete datasets)",
        "",
        f"- label: `{(out.get('chronos_finetune_pool') or {}).get('label')}`",
        f"- complete: `{(out.get('chronos_finetune_pool') or {}).get('n_finetune_complete')}` / "
        f"`{(out.get('chronos_finetune_pool') or {}).get('n_datasets')}`",
        f"- aggregated_across_incomplete_datasets: "
        f"`{(out.get('chronos_finetune_pool') or {}).get('aggregated_across_incomplete_datasets')}`",
        f"- failures: `{out.get('chronos_finetune_failures')}`",
        "",
        "## Eligibility and horizon units",
        "",
        f"- eligibility: `{out.get('eligibility')}`",
        f"- horizon_units: `{out.get('horizon_units')}`",
        "",
        "## Guardrails",
        "",
        f"- real_data_accessed: {out.get('real_data_accessed')}",
        f"- design_run: {out.get('design_run')}",
        f"- training_run: {out.get('training_run')}",
        f"- predose_evaluated: {out.get('predose_evaluated')}",
        f"- covariate_aware_comparison: {out.get('covariate_aware_comparison')}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (parent / "RESULTS.md").write_text(text, encoding="utf-8")
    Path("docs/v3/eval_correct_RESULTS.md").write_text(text, encoding="utf-8")

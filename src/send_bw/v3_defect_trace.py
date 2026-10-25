"""Trace MAE/RMSE and grouping defects to specific runs and selection paths.

Do not assume every earlier result is affected. Forecast.py predose RMSE was
already true RMSE. The |error| name and STUDYID merge collision lived in the
quantile scorer used by official-seq and seq-diag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from send_bw.utils.artifacts import read_json

FORECAST_RUNS = (
    "runs/v3_eval",
    "runs/v3_pilot_k0k1",
    "runs/v3_repl",
    "runs/v3_research",
    "runs/v3_closeout",
    "runs/v3_objective",
    "runs/v3_increment",
)
QUANTILE_AFFECTED = (
    "runs/v3_official_seq",
    "runs/v3_seq_diag",
)


def _mae_rmse_pair(block: dict[str, Any]) -> tuple[float | None, float | None]:
    mae = rmse = None
    sb = block.get("study_balanced") if isinstance(block, dict) else None
    if isinstance(sb, dict):
        mae = sb.get("mae_log")
        rmse = sb.get("rmse_log")
    if mae is None and isinstance(block, dict):
        mae = block.get("mae_log")
        rmse = block.get("rmse_log")
        if isinstance(mae, dict):
            mae = mae.get("study_balanced")
        if isinstance(rmse, dict):
            rmse = rmse.get("study_balanced")
    try:
        mae = None if mae is None else float(mae)
        rmse = None if rmse is None else float(rmse)
    except (TypeError, ValueError):
        return None, None
    return mae, rmse


def _forecast_evidence(run_dir: Path) -> dict[str, Any]:
    pairs = []
    for path in list(run_dir.glob("**/metrics.json"))[:8]:
        try:
            doc = read_json(path)
        except Exception:  # noqa: BLE001
            continue
        blocks = doc if isinstance(doc, list) else [doc]
        if isinstance(doc, dict):
            for v in doc.values():
                if isinstance(v, dict):
                    blocks.append(v)
                if isinstance(v, list):
                    blocks.extend([x for x in v if isinstance(x, dict)])
        for b in blocks:
            if not isinstance(b, dict):
                continue
            mae, rmse = _mae_rmse_pair(b)
            if mae is None or rmse is None:
                continue
            pairs.append({
                "path": str(path),
                "mae_log": mae,
                "rmse_log": rmse,
                "rmse_equals_mae": abs(rmse - mae) < 1e-12,
                "rmse_gt_mae": rmse > mae + 1e-12,
            })
            if len(pairs) >= 4:
                break
        if len(pairs) >= 4:
            break
    affected = bool(pairs) and all(p["rmse_equals_mae"] for p in pairs)
    return {
        "scorer": "send_bw.evaluation.forecast.evaluate / evaluate_early_history",
        "rmse_definition": "sqrt(mean e^2) per study, then mean of studies (true RMSE)",
        "grouping": "STUDYID from ModelData, not a prediction merge",
        "validation_selection": {
            "spline": "weighted RMSE of residuals (wrmse); not quantile rmse_log",
            "transformer": "weighted sqrt(mean w*(p-y)^2); not quantile rmse_log",
            "tft": "Lightning val_loss (quantile); not used on these runs",
            "chronos2": "not trained on these runs",
        },
        "published_metric_affected": False if pairs and any(p["rmse_gt_mae"] for p in pairs) else affected,
        "rescore_needed": False,
        "evidence": pairs[:4],
        "note": "mae_log and rmse_log already differ; this is not the |error| defect.",
    }


def _quantile_evidence(run_dir: Path, already_rescored: bool) -> dict[str, Any]:
    summary = run_dir / "SUMMARY.json"
    n_studies = None
    if summary.exists():
        doc = read_json(summary)
        results = doc.get("results") or {}
        first = next((v for v in results.values() if isinstance(v, dict) and v.get("n_studies")), None)
        if first:
            n_studies = first.get("n_studies")
        if n_studies is None:
            for rec in doc.get("replicates") or []:
                for v in (rec.get("results") or {}).values():
                    if isinstance(v, dict) and v.get("n_studies"):
                        n_studies = v.get("n_studies")
                        break
    grouping_collision = n_studies is not None and int(n_studies) > 10
    return {
        "scorer": "send_bw.evaluation.quantile_forecast.score_quantile_table (pre-PR #12)",
        "rmse_definition": "sqrt((point-y)^2) = |error| (MAE), published as rmse_log",
        "grouping": (
            "USUBJID after STUDYID_x/STUDYID_y merge collision"
            if grouping_collision else "STUDYID if merge kept the column"
        ),
        "published_n_studies": n_studies,
        "validation_selection": {
            "spline": "not affected (wrmse on Gaussian residuals)",
            "transformer": "not affected (true weighted val RMSE)",
            "tft": "not affected (Lightning val_loss)",
            "chronos2": (
                "eval_loss only when validation survived Chronos length filter; "
                "empty validation raises before any selection"
            ),
        },
        "published_metric_affected": True,
        "rescore_needed": not already_rescored,
        "already_rescored_in": "runs/v3_eval_correct" if already_rescored else None,
        "note": "Tables in this run dir are preserved. Corrected MAE/true RMSE live in eval_correct.",
    }


def trace_metric_defects(eval_correct_dir: str | Path = "runs/v3_eval_correct") -> dict[str, Any]:
    """Identify which prior runs and validation-selection paths were affected."""
    runs: dict[str, Any] = {}
    for p in FORECAST_RUNS:
        path = Path(p)
        if not path.exists():
            runs[p] = {"status": "absent", "published_metric_affected": False, "rescore_needed": False}
            continue
        runs[p] = _forecast_evidence(path)
    already = Path(eval_correct_dir).exists()
    for p in QUANTILE_AFFECTED:
        path = Path(p)
        if not path.exists():
            runs[p] = {"status": "absent", "published_metric_affected": True, "rescore_needed": True}
            continue
        runs[p] = _quantile_evidence(path, already_rescored=already)
    runs["runs/v3_eval_correct"] = {
        "scorer": "corrected score_quantile_table",
        "rmse_definition": "true RMSE; mae_log named separately",
        "grouping": "STUDYID from observations",
        "published_metric_affected": False,
        "rescore_needed": False,
        "note": "PR #12 correction. Not a training run.",
    }
    affected = [k for k, v in runs.items() if v.get("published_metric_affected")]
    rescore = [k for k, v in runs.items() if v.get("rescore_needed")]
    return {
        "defects": {
            "rmse_log_was_abs_error": {
                "where": "quantile_forecast.score_quantile_table before PR #12",
                "formula": "sqrt((point - y)^2) == |error|",
            },
            "studyid_merge_collision": {
                "where": "same scorer, merge of obs and pred both carrying STUDYID",
                "effect": "STUDYID_x/STUDYID_y; grouping fell back to USUBJID (animals, not studies)",
            },
        },
        "runs": runs,
        "affected_published_metrics": affected,
        "rescore_needed": rescore,
        "validation_selection_affected": False,
        "validation_selection_note": (
            "Spline wrmse, transformer val_rmse, and TFT val_loss did not use the "
            "defective quantile rmse_log. Chronos-2 dataset-0 failed at validation "
            "length filtering before eval_loss selection."
        ),
        "rescored": "runs/v3_eval_correct" if already else None,
    }

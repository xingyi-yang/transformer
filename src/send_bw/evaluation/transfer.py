"""Sign-defined paired transfer table.

Δ_k = S_k − S_0 on the frozen target holdout. Signs are published; they are never
clipped or retuned away. Selective sharing v1 is a scored regime in the v3
evaluation benchmark; the frozen k=0/k=1 executor still omits it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

TASK = "predose_conditioned"
HARMFUL_ON_INCREASE = ("rmse_log", "mae_log", "crps", "crps_log")
DEFAULT_METRICS = ("rmse_log", "crps", "contrast_bias")
LABELS = (
    "Synthetic diagnostic.",
    "Not design qualification.",
    "No real-data validation.",
    "No operational recommendation.",
)


def _get(summary: dict[str, Any], metric: str) -> Any:
    sb = summary.get("study_balanced") or {}
    if metric in sb:
        return sb[metric]
    if metric == "crps" and "crps" not in sb and "crps_log" in (summary.get("observation_weighted") or {}):
        return sb.get("crps", summary.get("crps"))
    return summary.get(metric)


def _convention(metric: str) -> str:
    if metric == "contrast_bias":
        return "delta = |Sk| - |S0| on contrast_bias; delta>0 is harmful"
    return "delta = Sk - S0; delta>0 is harmful for RMSE/CRPS"


def paired_transfer_row(s0: dict[str, Any], sk: dict[str, Any], metric: str, k: int,
                        model: str | None = None, task: str = TASK) -> dict[str, Any]:
    v0, vk = _get(s0, metric), _get(sk, metric)
    model = model or s0.get("model") or sk.get("model")
    if v0 is None or vk is None:
        return {"task": task, "model": model, "k": int(k), "metric": metric,
                "S0": v0, "Sk": vk, "delta": None, "sign": "not_estimable",
                "convention": _convention(metric)}
    v0, vk = float(v0), float(vk)
    if metric == "contrast_bias":
        delta = abs(vk) - abs(v0)
    else:
        delta = vk - v0
    if delta > 0:
        sign = "harmful"
    elif delta < 0:
        sign = "helpful"
    else:
        sign = "unchanged"
    return {"task": task, "model": model, "k": int(k), "metric": metric,
            "S0": v0, "Sk": vk, "delta": delta, "sign": sign,
            "convention": _convention(metric)}


def build_transfer_table(s0_summary: dict[str, Any],
                         sk_by_k: Iterable[tuple[int, dict[str, Any]]],
                         metrics: tuple[str, ...] = DEFAULT_METRICS,
                         model: str | None = None,
                         task: str = TASK,
                         sharing_regimes: list[str] | None = None,
                         selective_sharing: str = "required_deferred") -> dict[str, Any]:
    model = model or s0_summary.get("model")
    rows = []
    for k, sk in sk_by_k:
        for metric in metrics:
            row = paired_transfer_row(s0_summary, sk, metric, k, model, task)
            if isinstance(sk, dict) and sk.get("sharing"):
                row["sharing"] = sk.get("sharing")
            rows.append(row)
    return {
        "task": task,
        "model": model,
        "labels": list(LABELS),
        "sharing_regimes": sharing_regimes or ["context_specific (S0)", "pooled (Sk)"],
        "selective_sharing": selective_sharing,
        "rows": rows,
    }


def transfer_uncertainty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Uncertainty around paired transfer differences across replicates.

    Dataset-generation seeds and model-training seeds are recorded separately.
    Sharing is not required to improve performance. Signs are published.
    """
    if not rows:
        return {
            "n_rows": 0, "n_groups": 0, "groups": [],
            "seed_separation": {
                "generate_seed": "dataset generation only",
                "train_seed": "splits and model training only",
            },
            "note": "no paired transfer rows",
            "labels": list(LABELS),
        }
    df = pd.DataFrame(rows)
    keys = [c for c in ("model", "sharing", "metric", "task", "split", "sparse_target_train") if c in df.columns]
    groups: list[dict[str, Any]] = []
    if not keys:
        keys = ["metric"] if "metric" in df.columns else []
    grouped = df.groupby(keys, dropna=False) if keys else [((), df)]
    for key, g in grouped:
        key_t = key if isinstance(key, tuple) else (key,)
        deltas = pd.to_numeric(g["delta"], errors="coerce") if "delta" in g else pd.Series(dtype=float)
        deltas = deltas.dropna()
        signs = [str(s) for s in (g["sign"].tolist() if "sign" in g else [])]
        groups.append({
            "key": dict(zip(keys, key_t, strict=True)) if keys else {},
            "n": int(len(g)),
            "generate_seeds": sorted({int(x) for x in g["generate_seed"].tolist()}) if "generate_seed" in g else [],
            "train_seeds": sorted({int(x) for x in g["train_seed"].tolist()}) if "train_seed" in g else [],
            "mean_delta": float(deltas.mean()) if len(deltas) else None,
            "sd_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
            "min_delta": float(deltas.min()) if len(deltas) else None,
            "max_delta": float(deltas.max()) if len(deltas) else None,
            "signs": signs,
            "n_harmful": int(sum(s == "harmful" for s in signs)),
            "n_helpful": int(sum(s == "helpful" for s in signs)),
            "note": "Sharing is not required to improve. Signs are published as computed.",
        })
    return {
        "n_rows": int(len(df)),
        "n_groups": len(groups),
        "groups": groups,
        "seed_separation": {
            "generate_seed": "dataset generation only",
            "train_seed": "splits and model training only",
        },
        "labels": list(LABELS),
        "note": (
            "Uncertainty is the replicate-to-replicate spread of paired Δ. "
            "It is not a license to pick the favorable seed."
        ),
    }

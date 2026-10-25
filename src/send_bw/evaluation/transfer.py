"""Sign-defined paired transfer table.

Δ_k = S_k − S_0 on the frozen target holdout. Signs are published; they are never
clipped or retuned away. Selective sharing v1 is a scored regime in the v3
evaluation benchmark; the frozen k=0/k=1 executor still omits it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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

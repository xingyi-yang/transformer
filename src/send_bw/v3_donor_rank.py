"""Close the prefix-abstention diagnostic and lock inner-fold donor ranking.

Does not retune the executed prefix run. Does not overwrite prior artifacts.
Benchmark training is refused: this lock stops before execution.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from send_bw.donor_rank import (
    K_DONORS,
    N_INNER_FOLDS,
    RANK_LABEL,
    RANK_METRIC,
    RANK_RULE,
)
from send_bw.historical_audit import discover_authorization
from send_bw.protected_hashes import compare_snapshots, snapshot_protected
from send_bw.share_policy import PREFIX_LABEL, PREFIX_RULE
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_official_seq import collect_versions
from send_bw.v3_research import load_yaml
from send_bw.v3_share_abstain import PRESERVED as SHARE_ABSTAIN_PRESERVED

PRESERVED = tuple(SHARE_ABSTAIN_PRESERVED) + (
    "runs/v3_share_abstain", "runs/v3_share_abstain_run",
)
LOCK_DIR = "runs/v3_donor_rank"
LOCK_CONFIG = "configs/v3_donor_rank.frozen.yaml"
SHARE_ABSTAIN_RUN_SUMMARY = Path("runs/v3_share_abstain_run/SUMMARY.json")
CLOSEOUT_PATH = Path("docs/v3/share_abstain_run_CLOSEOUT.md")
PROTOCOL_PATH = Path("docs/v3/donor_rank_protocol.md")

POLICY_ARMS = (
    {"key": "target_only", "sharing": "context_specific", "k": 0, "donor_rule": "zero"},
    {"key": "prefix_one", "sharing": "selective", "k": 1, "donor_rule": "prefix"},
    {"key": "ranked_one", "sharing": "selective", "k": 1, "donor_rule": "ranked"},
)

TRANSFER_ARMS = (
    "selected_one", "selected_abstain", "random_one", "pooled_all", "selected_regularized",
)


class DonorRankLockError(ValueError):
    pass


def _fmt(v: float | None, digits: int = 5) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{digits}f}"


def _mean(xs: list[float]) -> float | None:
    return None if not xs else sum(xs) / len(xs)


def prefix_abstention_closeout(summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize executed prefix-abstention magnitudes. No retune."""
    src = summary or (read_json(SHARE_ABSTAIN_RUN_SUMMARY)
                      if SHARE_ABSTAIN_RUN_SUMMARY.exists() else None)
    if not src:
        return {"status": "unavailable", "reason": "runs/v3_share_abstain_run/SUMMARY.json missing"}
    by: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    cells: list[dict[str, Any]] = []
    for rec in src.get("replicates") or []:
        d = int(rec.get("dataset_index"))
        bank, dens = rec.get("bank"), rec.get("density")
        for model in ("spline", "transformer"):
            for arm in TRANSFER_ARMS:
                fam = f"{model}_{arm}"
                rows = ((rec.get("paired_transfer") or {}).get(fam) or {}).get("rows") or []
                if not rows:
                    continue
                mae = next((r for r in rows if r.get("metric") == "mae_log"), None)
                rmse = next((r for r in rows if r.get("metric") == "rmse_log"), None)
                if not mae or not rmse:
                    continue
                row = {
                    "dataset_index": d, "model": model, "arm": arm,
                    "bank": bank, "density": dens,
                    "mae_delta": mae.get("delta_pooled_minus_target"),
                    "mae_sign": mae.get("sign"),
                    "rmse_delta": rmse.get("delta_pooled_minus_target"),
                    "rmse_sign": rmse.get("sign"),
                }
                by[(d, model, arm)].append(row)
                cells.append(row)
    groups: list[dict[str, Any]] = []
    for d in (0, 1):
        for model in ("spline", "transformer"):
            for arm in TRANSFER_ARMS:
                xs = by[(d, model, arm)]
                if not xs:
                    continue
                mae_ds = [float(x["mae_delta"]) for x in xs]
                rmse_ds = [float(x["rmse_delta"]) for x in xs]
                groups.append({
                    "dataset_index": d, "model": model, "arm": arm, "n": len(xs),
                    "mae_delta_mean": _mean(mae_ds),
                    "mae_abs_mean": _mean([abs(v) for v in mae_ds]),
                    "mae_delta_min": min(mae_ds), "mae_delta_max": max(mae_ds),
                    "mae_signs": {s: sum(1 for x in xs if x["mae_sign"] == s)
                                  for s in ("helpful", "harmful", "null")},
                    "rmse_delta_mean": _mean(rmse_ds),
                    "rmse_abs_mean": _mean([abs(v) for v in rmse_ds]),
                    "rmse_delta_min": min(rmse_ds), "rmse_delta_max": max(rmse_ds),
                    "rmse_signs": {s: sum(1 for x in xs if x["rmse_sign"] == s)
                                   for s in ("helpful", "harmful", "null")},
                    "cells": xs,
                })
    return {
        "status": "ok",
        "source": str(SHARE_ABSTAIN_RUN_SUMMARY),
        "retuned": False,
        "confirmatory": False,
        "seconds_total": src.get("seconds_total"),
        "compute_cap_hit": src.get("compute_cap_hit"),
        "failures": src.get("failures") or [],
        "protected": src.get("protected"),
        "abstention_counts": src.get("abstention_counts"),
        "prefix_versus_random_all_equal": all(
            (r.get("equal") is True) for r in (src.get("prefix_versus_random") or [])
        ),
        "groups": groups,
        "n_cells": len(cells),
        "labels": list(LABELS) + [
            "Prefix-abstention diagnostic closed without retuning.",
            "Two dataset seeds and one training seed are not confirmatory.",
        ],
    }


def _write_closeout(close: dict[str, Any]) -> None:
    lines = [
        "# Prefix-abstention diagnostic — closed (not confirmatory)",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data "
        "validation. No operational recommendation.**",
        "",
        "Closed without retuning. Frozen prefix-abstention numbers in "
        "`runs/v3_share_abstain_run/` and `docs/v3/share_abstain_run_RESULTS.md` "
        "are unchanged. Two dataset seeds and one training seed are **not "
        "confirmatory**. Prefix donor choice remains a STUDYID sort.",
        "",
        f"- source: `{close.get('source')}`",
        f"- seconds_total: `{close.get('seconds_total')}`; cap hit: `{close.get('compute_cap_hit')}`",
        f"- failures: `{close.get('failures')}`",
        f"- protected unchanged: `{(close.get('protected') or {}).get('unchanged')}`",
        f"- prefix versus random equal on all slices: `{close.get('prefix_versus_random_all_equal')}`",
        f"- retuned: `{close.get('retuned')}`; confirmatory: False",
        "",
        "## Abstention counts (validation-only)",
        "",
    ]
    counts = close.get("abstention_counts") or {}
    lines += [
        f"- spline: `{(counts.get('spline') or {}).get('target_only_over_n')}` target-only, "
        f"`{(counts.get('spline') or {}).get('share_over_n')}` share",
        f"- transformer: `{(counts.get('transformer') or {}).get('target_only_over_n')}` target-only, "
        f"`{(counts.get('transformer') or {}).get('share_over_n')}` share",
        f"- overall: `{(counts.get('overall') or {}).get('target_only_over_n')}` target-only, "
        f"`{(counts.get('overall') or {}).get('share_over_n')}` share",
        "",
        "## Paired MAE/RMSE magnitudes by model and dataset",
        "",
        "Δ = other − target-only. Δ>0 is harmful. Magnitudes are means over the "
        "four bank×density cells in that dataset (regularized: matched/full only).",
        "",
        "| dataset | model | arm | n | MAE mean Δ | MAE mean \\|Δ\\| | MAE help/harm/null | "
        "RMSE mean Δ | RMSE mean \\|Δ\\| | RMSE help/harm/null |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in close.get("groups") or []:
        ms, rs = g["mae_signs"], g["rmse_signs"]
        lines.append(
            f"| {g['dataset_index']} | {g['model']} | {g['arm']} | {g['n']} | "
            f"{_fmt(g['mae_delta_mean'])} | {_fmt(g['mae_abs_mean']).lstrip('+')} | "
            f"{ms['helpful']}/{ms['harmful']}/{ms['null']} | "
            f"{_fmt(g['rmse_delta_mean'])} | {_fmt(g['rmse_abs_mean']).lstrip('+')} | "
            f"{rs['helpful']}/{rs['harmful']}/{rs['null']} |"
        )
    lines += [
        "",
        "### Cell-level selected_one and selected_abstain",
        "",
        "| dataset | bank | density | model | arm | MAE Δ | MAE sign | RMSE Δ | RMSE sign |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in close.get("groups") or []:
        if g["arm"] not in {"selected_one", "selected_abstain"}:
            continue
        for x in g["cells"]:
            lines.append(
                f"| {x['dataset_index']} | {x['bank']} | {x['density']} | {x['model']} | "
                f"{x['arm']} | {_fmt(x['mae_delta'])} | {x['mae_sign']} | "
                f"{_fmt(x['rmse_delta'])} | {x['rmse_sign']} |"
            )
    lines += [
        "",
        "## Reading (not a retune cue)",
        "",
        "- Dataset 0 spline prefix-one was helpful on all 4/4 cells "
        "(mean MAE Δ −0.00153, mean RMSE Δ −0.00190).",
        "- Dataset 0 transformer prefix-one was mixed (2 helpful / 2 harmful); "
        "sparse cells were harmful (MAE Δ up to +0.01631).",
        "- Dataset 1 spline prefix-one was mostly harmful (3/4); transformer "
        "prefix-one was harmful on 4/4 (mean MAE Δ +0.00911, mean RMSE Δ +0.01103).",
        "- Abstention (selected_abstain) was 5 helpful / 2 harmful / 9 null of 16. "
        "The two harmful shares are transformer on dataset 1 full.",
        "- Prefix versus random was degenerate (SYNA-D001 on 8/8). That does not "
        "generalize beyond this 3-donor bank and seed 91001.",
        "- These magnitudes are not a reason to change thresholds, seeds, or donor rules.",
        "",
        "Stop. No automatic expansion.",
        "",
    ]
    CLOSEOUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def locked_budget(cfg: dict[str, Any]) -> dict[str, Any]:
    ev = cfg.get("donor_rank") or {}
    n_ds = int(ev.get("n_dataset_seeds") or 2)
    n_tr = int(ev.get("n_train_seeds") or 1)
    g0 = int(cfg["generate_seed"])
    t0s = int(cfg["train_seed"])
    g_stride = int(ev.get("generate_seed_stride") or 1009)
    t_stride = int(ev.get("train_seed_stride") or 17)
    banks = list(ev.get("banks") or ["matched", "both"])
    densities = list(ev.get("densities") or ["full"])
    n_donors = int(ev.get("n_eligible_donors") or 3)
    n_folds = int(ev.get("n_inner_folds") or N_INNER_FOLDS)
    n_models = 2
    n_slices = n_ds * n_tr * len(banks) * len(densities)
    n_candidate_fits = n_slices * n_models * n_donors * n_folds
    n_cells = n_slices * len(POLICY_ARMS)
    n_final_fits = n_cells * n_models
    return {
        "n_dataset_seeds": n_ds,
        "n_train_seeds": n_tr,
        "generate_seeds": [g0 + i * g_stride for i in range(n_ds)],
        "train_seeds": [t0s + j * t_stride for j in range(n_tr)],
        "n_eligible_donors": n_donors,
        "n_inner_folds": n_folds,
        "k_donors": int(ev.get("k_donors") or K_DONORS),
        "banks": banks,
        "densities": densities,
        "sparse_target_train_n": ev.get("sparse_target_train_n"),
        "encoder_window_day": float(ev.get("encoder_window_day") or 14),
        "policy_arms": [a["key"] for a in POLICY_ARMS] + ["ranked_abstain"],
        "n_slices": n_slices,
        "n_trained_cells": n_cells,
        "n_candidate_fits": n_candidate_fits,
        "n_final_model_fits": n_final_fits,
        "additional_fits_for_abstention": 0,
        "n_model_fits_if_executed": n_candidate_fits + n_final_fits,
        "candidate_fit_formula": (
            "n_dataset_seeds × n_train_seeds × n_banks × n_densities × "
            "n_models × n_eligible_donors × n_inner_folds"
        ),
        "rank_rule": RANK_RULE,
        "rank_rule_label": RANK_LABEL,
        "rank_metric": RANK_METRIC,
        "prefix_rule": PREFIX_RULE,
        "prefix_rule_label": PREFIX_LABEL,
        "estimated_runtime_seconds": ev.get("estimated_runtime_seconds") or {
            "point": 240, "lower": 150, "upper": 480,
        },
        "compute_cap_seconds": int(ev.get("compute_cap_seconds") or 600),
        "execute": bool(ev.get("execute")),
        "training_seed_variability_estimated": n_tr > 1,
        "dataset_variability_estimated": n_ds > 1,
        "confirmatory": False,
    }


def execute_v3_donor_rank(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    del force
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or LOCK_DIR)
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise DonorRankLockError("refusing to write into a preserved artifact directory")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise DonorRankLockError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise DonorRankLockError("design optimization stays off")
    ev = cfg.get("donor_rank") or {}
    if ev.get("retune_after_test"):
        raise DonorRankLockError("retune_after_test is forbidden")
    if ev.get("predose") or ev.get("bayesian") or ev.get("tft") or ev.get("chronos"):
        raise DonorRankLockError("TFT, Chronos, Bayesian, and predose stay off")
    budget = locked_budget(cfg)
    if budget["execute"]:
        raise DonorRankLockError("this lock stops before benchmark execution")
    parent.mkdir(parents=True, exist_ok=True)
    hashes_before = snapshot_protected()
    write_json(parent / "protected_hashes_before.json", hashes_before)
    close = prefix_abstention_closeout()
    _write_closeout(close)
    write_json(parent / "prefix_abstention_closeout.json", close)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), **budget,
        "retune_after_test": False, "real_data": False, "design": False,
        "new_architectures": False, "fabricate_encoder_history": False,
        "selector_sees_generator_labels": False,
        "selector_sees_test_outcomes": False,
        "prefix_diagnostic_closed": True,
        "prefix_diagnostic_retuned": False,
        "labels": list(LABELS), "preserved": list(PRESERVED),
    }
    write_json(parent / "freeze_lock.json", lock)
    write_json(parent / "versions.json", collect_versions())
    _write_protocol(lock)
    out = {
        "status": "locked_not_executed",
        "labels": list(LABELS),
        "output_dir": str(parent),
        "freeze_lock": lock,
        "closeout": {
            "path": str(CLOSEOUT_PATH),
            "status": close.get("status"),
            "retuned": False,
            "n_groups": len(close.get("groups") or []),
        },
        "n_candidate_fits": budget["n_candidate_fits"],
        "n_final_model_fits": budget["n_final_model_fits"],
        "n_model_fits_if_executed": budget["n_model_fits_if_executed"],
        "execute_command": f"send-bw v3-donor-rank --config {LOCK_CONFIG}",
        "execute_requires": "separate approval; this lock refuses execute:true",
        "real_data_accessed": False,
        "authorization_present": bool(discover_authorization(None).get("authorized")),
        "design_run": False,
        "new_architectures": False,
    }
    hashes_after = snapshot_protected()
    cmp = compare_snapshots(hashes_before, hashes_after)
    write_json(parent / "protected_hashes_after.json", hashes_after)
    write_json(parent / "protected_hashes_compare.json", cmp)
    out["protected"] = cmp
    write_json(parent / "SUMMARY.json", out)
    _write_lock_report(parent, out)
    if not cmp["unchanged"]:
        raise DonorRankLockError("protected artifacts changed: " + ", ".join(cmp["changed"]))
    return out


def _write_protocol(lock: dict[str, Any]) -> None:
    text = f"""# Inner-fold donor ranking — locked, not executed

> **Synthetic diagnostic. Exploratory. Not design qualification. No real-data
> validation. No operational recommendation.**

The executed prefix-abstention diagnostic is **closed without retuning**.
This lock is a separate, smaller ranking path. Prior run directories,
including `runs/v3_share_abstain/` and `runs/v3_share_abstain_run/`, are
not rewritten.

Scientific question: can one-donor candidates be ranked from inner
target-training study folds, then accepted or rejected on outer validation,
without using test outcomes or generator scenario labels?

## Ranking path

1. Eligible donors = donor-species QC studies (`n_eligible_donors` planned
   `{lock.get('n_eligible_donors')}`).
2. One candidate per eligible donor (k=`{lock.get('k_donors')}`).
3. Inner folds: sort target TRAIN STUDYIDs, split into
   `{lock.get('n_inner_folds')}` round-robin study folds. Fit on the other
   folds' target studies plus that one donor; score study-balanced true RMSE
   on the held-out target TRAIN studies (`{RANK_METRIC}`).
4. Rank by mean inner-fold RMSE. Ties break by STUDYID sort.
5. Outer-validation share-or-abstain: existing `decide_abstention` on the
   frozen target validation split (min_improvement 0, ties → target-only).

Rejected inputs: target test STUDYIDs, test metrics, generator
`donor_scenario` / bank labels. Banks remain experiment strata for data
generation; they are not ranking features.

Prefix (`{PREFIX_RULE}`) stays as a non-learned comparator. It is not
replaced by this ranker until a later approved execution.

## Locked comparison (execution refused)

| quantity | value |
|---|---|
| `n_dataset_seeds` | **{lock.get('n_dataset_seeds')}** |
| `n_train_seeds` | **{lock.get('n_train_seeds')}** |
| generate seeds | `{lock.get('generate_seeds')}` |
| train seeds | `{lock.get('train_seeds')}` |
| banks | `{lock.get('banks')}` |
| densities | `{lock.get('densities')}` (full only; sparse already closed) |
| encoder window | DAY ≤ {lock.get('encoder_window_day')} |
| trained policy arms | target_only, prefix_one, ranked_one |
| abstention | reuse target_only vs ranked_one; 0 extra fits |
| candidate fits | **{lock.get('n_candidate_fits')}** |
| candidate-fit formula | `{lock.get('candidate_fit_formula')}` |
| final model fits | **{lock.get('n_final_model_fits')}** |
| total fits if executed | **{lock.get('n_model_fits_if_executed')}** |
| estimated runtime | **{((lock.get('estimated_runtime_seconds') or {}).get('point'))} s** (150–480) |
| compute cap | **{lock.get('compute_cap_seconds')} s** |
| `execute` | **false** |

2×1 is **not confirmatory**. Training-seed variability is not estimated.

Stopping rules: refuse real data, design, TFT/Chronos, Bayesian, predose,
retune-after-test, writes into preserved dirs, and `execute: true`.

## Proposed later execution (not this PR)

Requires a separate approved config. This lock raises if `execute` is true.

`send-bw v3-donor-rank --config {LOCK_CONFIG}`
"""
    PROTOCOL_PATH.write_text(text, encoding="utf-8")


def _write_lock_report(parent: Path, out: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    lines = [
        "# Inner-fold donor ranking — locked, not executed",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data "
        "validation. No operational recommendation.**",
        "",
        "Prefix-abstention diagnostic closed without retuning. This lock "
        "records the smallest genuine donor-ranking path and **stops before "
        "benchmark execution**.",
        "",
        f"- generate_seeds: `{lock.get('generate_seeds')}`; train_seeds: `{lock.get('train_seeds')}`",
        f"- n_eligible_donors: `{lock.get('n_eligible_donors')}`; n_inner_folds: `{lock.get('n_inner_folds')}`",
        f"- n_candidate_fits: `{lock.get('n_candidate_fits')}` "
        f"(`{lock.get('candidate_fit_formula')}`)",
        f"- n_final_model_fits: `{lock.get('n_final_model_fits')}`; "
        f"n_model_fits_if_executed: `{lock.get('n_model_fits_if_executed')}`",
        f"- estimated_runtime_seconds: `{lock.get('estimated_runtime_seconds')}`",
        f"- compute_cap_seconds: `{lock.get('compute_cap_seconds')}`",
        f"- execute: `{lock.get('execute')}`",
        f"- rank: `{RANK_LABEL}`",
        "",
        f"Closeout: `{((out.get('closeout') or {}).get('path'))}` "
        f"(retuned=`{((out.get('closeout') or {}).get('retuned'))}`).",
        "",
        f"Protected artifacts unchanged: `{(out.get('protected') or {}).get('unchanged')}`",
        "",
        "This command refuses `execute: true`.",
        "",
    ]
    text = "\n".join(lines)
    (parent / "RESULTS.md").write_text(text, encoding="utf-8")
    Path("docs/v3/donor_rank_RESULTS.md").write_text(text, encoding="utf-8")

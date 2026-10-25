"""Targeted increment diagnostic on executed v3 research cells.

Re-scores existing fitted spline/transformer models and prepared test splits.
Does not refit, add architectures, or write into preserved run directories.
Bayesian cells are recorded as unreliable and are not scored.
Findings are exploratory: two independent dataset replicates.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from send_bw.evaluation.forecast import _study_draws
from send_bw.evaluation.increment import compare_study, hierarchical_mean
from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS
from send_bw.models.base import ModelData
from send_bw.pipeline import model_class
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import git_revision, read_json, write_json

PRESERVED = (
    "runs/v3_eval",
    "runs/v3_pilot_k0k1",
    "runs/v3_repl",
    "runs/v3_research",
    "runs/v3_closeout",
    "runs/v3_audit",
)
SCORED_MODELS = ("spline", "transformer")


class IncrementError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_cell_prepared(cell_dir: Path) -> dict[str, Any]:
    """Read prepared splits and cohort tables without constructing Run.

    Run.__init__ writes into the cell directory. Preserved research cells must
    stay untouched.
    """
    from send_bw.data.features import FeatureSpec
    from send_bw.data.splits import load_frozen_splits

    splits = load_frozen_splits(cell_dir / "splits" / "splits")
    spec = FeatureSpec.from_dict(read_json(cell_dir / "features" / "feature_spec.json"))
    cref = pd.read_parquet(cell_dir / "canonical" / "compound_reference.parquet")
    full = ModelData(
        pd.read_parquet(cell_dir / "cohort" / "animals.parquet"),
        pd.read_parquet(cell_dir / "cohort" / "obs.parquet"),
        cref,
        spec,
    )
    data = {s: full.subset(splits.loc[splits["SPLIT"] == s, "STUDYID"]) for s in splits["SPLIT"].unique()}
    return {"splits": splits, "spec": spec, "data": data, "compound_ref": cref}


def _density(rec: dict[str, Any]) -> str:
    return "full" if rec.get("sparse_target_train_n") in (None, "full") else "sparse"


def _pack_ratio(rec: dict[str, Any], model: str) -> float | None:
    pack = (rec.get("evaluation_packs") or {}).get(model) or {}
    tmp = pack.get("temporal") or {}
    val = tmp.get("increment_sd_ratio_pred_over_obs")
    return float(val) if val is not None else None


def score_model_on_test(model: Any, data: ModelData, n_draws: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for k, sid in enumerate(data.studies):
        _a, _times, zs, y = _study_draws(model, data, sid, n_draws, seed + k)
        mask = np.isfinite(y)
        cmp = compare_study(y, zs, mask)
        rows.append({
            "STUDYID": sid,
            "n_animals": cmp["observed"]["n_animals"],
            "n_draws": cmp["simulated"]["n_draws"],
            "obs_study_stat": cmp["observed"]["study_stat"],
            "pred_expected_study_stat": cmp["simulated"]["expected_study_stat"],
            "pred_between_draw_sd": cmp["simulated"]["between_draw_sd"],
            "matched_ratio": cmp["matched_ratio_pred_over_obs"],
            "mismatched_animal_pooled_pred": cmp["mismatched_animal_pooled_pred"],
        })
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_study_rows": 0, "n_datasets": 0}
    df = pd.DataFrame(rows)
    by_key: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("model"), r.get("sharing"), r.get("bank"), r.get("density"))
        by_key[key].append(r)
    groups = []
    for key, g in sorted(by_key.items()):
        ratios = [x["matched_ratio"] for x in g if x.get("matched_ratio") is not None]
        ds_keys = [x["dataset_index"] for x in g if x.get("matched_ratio") is not None]
        pred = [x["pred_expected_study_stat"] for x in g if x.get("pred_expected_study_stat") is not None]
        obs = [x["obs_study_stat"] for x in g if x.get("obs_study_stat") is not None]
        btw = [x["pred_between_draw_sd"] for x in g if x.get("pred_between_draw_sd") is not None]
        pack = [x["published_pack_ratio"] for x in g if x.get("published_pack_ratio") is not None]
        groups.append({
            "key": dict(zip(("model", "sharing", "bank", "density"), key, strict=True)),
            "n_study_rows": len(g),
            "matched_ratio": hierarchical_mean(ratios, group_keys=ds_keys),
            "obs_study_stat": hierarchical_mean(obs, group_keys=[x["dataset_index"] for x in g if x.get("obs_study_stat") is not None]),
            "pred_expected_study_stat": hierarchical_mean(
                pred, group_keys=[x["dataset_index"] for x in g if x.get("pred_expected_study_stat") is not None]),
            "pred_between_draw_sd": hierarchical_mean(
                btw, group_keys=[x["dataset_index"] for x in g if x.get("pred_between_draw_sd") is not None]),
            "published_pack_ratio": hierarchical_mean(
                pack, group_keys=[x["dataset_index"] for x in g if x.get("published_pack_ratio") is not None]),
        })
    overall_ratio = hierarchical_mean(
        [r["matched_ratio"] for r in rows if r.get("matched_ratio") is not None],
        group_keys=[r["dataset_index"] for r in rows if r.get("matched_ratio") is not None],
    )
    return {
        "n_study_rows": int(len(df)),
        "n_cells": int(df.groupby("name").ngroups) if "name" in df else 0,
        "n_datasets": int(df["dataset_index"].nunique()) if "dataset_index" in df else 0,
        "n_train_per_dataset": (
            {int(k): int(v) for k, v in df.groupby("dataset_index")["train_index"].nunique().to_dict().items()}
            if "dataset_index" in df else {}),
        "overall_matched_ratio": overall_ratio,
        "groups": groups,
    }


def execute_v3_increment(research_dir: str | Path = "runs/v3_research",
                         output_dir: str | Path = "runs/v3_increment",
                         split: str = "test") -> dict[str, Any]:
    research_dir = Path(research_dir)
    output_dir = Path(output_dir)
    if output_dir.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise IncrementError("refusing to write into a preserved artifact directory")
    summary_path = research_dir / "SUMMARY.json"
    if not summary_path.exists():
        raise IncrementError(f"missing executed research artifacts under {research_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(summary_path)
    cells = list(summary.get("cells") or [])
    study_rows: list[dict[str, Any]] = []
    bayes_rows: list[dict[str, Any]] = []
    scored_cells = 0
    for rec in cells:
        if rec.get("failed"):
            continue
        cell_dir = Path(rec["dir"])
        if not cell_dir.exists():
            continue
        cfg = load_yaml(cell_dir / "effective.yaml") if (cell_dir / "effective.yaml").exists() else {}
        n_draws = int((cfg.get("evaluation") or {}).get("n_draws") or 40)
        seed = int(cfg.get("seed") or rec.get("train_seed") or 0) + 17
        prep = load_cell_prepared(cell_dir)
        if split not in prep["data"]:
            continue
        data = prep["data"][split]
        if not len(data.obs):
            continue
        dens = _density(rec)
        scored_cells += 1
        for name in SCORED_MODELS:
            model_path = cell_dir / "models" / name / "model.json"
            if not model_path.exists():
                continue
            model = model_class(name).load(cell_dir / "models" / name)
            for row in score_model_on_test(model, data, n_draws, seed):
                study_rows.append({
                    **row,
                    "name": rec.get("name"),
                    "model": name,
                    "sharing": rec.get("sharing"),
                    "bank": rec.get("bank"),
                    "density": dens,
                    "dataset_index": rec.get("dataset_index"),
                    "train_index": rec.get("train_index"),
                    "generate_seed": rec.get("generate_seed"),
                    "train_seed": rec.get("train_seed"),
                    "n_draws_config": n_draws,
                    "eval_seed": seed,
                    "split": split,
                    "published_pack_ratio": _pack_ratio(rec, name),
                })
        if "bayesian" in (rec.get("models") or []):
            gate = rec.get("bayesian_gate") or {}
            samp = rec.get("sampling_diagnosis") or {}
            bayes_rows.append({
                "name": rec.get("name"),
                "scored": False,
                "reliable": False,
                "counted_as_successful_benchmark": False,
                "failure_class": samp.get("failure_class"),
                "rhat_max": samp.get("rhat_max") or gate.get("rhat_max"),
                "ess_bulk_min": samp.get("ess_bulk_min") or gate.get("ess_bulk_min"),
                "ess_tail_min": samp.get("ess_tail_min") or gate.get("ess_tail_min"),
                "divergences": samp.get("divergences") or gate.get("divergences"),
                "thresholds": dict(RELIABILITY_THRESHOLDS),
                "reason": "Bayesian inference remains explicitly unreliable until the unchanged diagnostics pass.",
            })
    nested = _summarize_rows(study_rows)
    out = {
        "labels": list(LABELS) + ["Exploratory.", "Two independent dataset replicates."],
        "git_revision": git_revision(),
        "source_research": {
            "dir": str(research_dir),
            "config_hash": (summary.get("freeze_lock") or {}).get("config_hash"),
            "freeze_revision": (summary.get("freeze_lock") or {}).get("git_revision"),
        },
        "retuned_after_test": False,
        "architectures_added": False,
        "grid_relaunched": False,
        "refit": False,
        "split": split,
        "n_cells_scored": scored_cells,
        "exploratory": True,
        "n_datasets": nested.get("n_datasets"),
        "replication": {
            "outer": "dataset / generate_seed",
            "n_datasets": nested.get("n_datasets"),
            "nested": "train_seed then study then animal",
            "n_train_per_dataset": nested.get("n_train_per_dataset"),
            "note": (
                "Findings are exploratory. n_datasets=2 is the independent dataset "
                "replicate count. Training seeds also reshuffle target splits."
            ),
        },
        "statistic": {
            "animal": "SD of consecutive observed-visit log-weight increments (ddof=0)",
            "study": "unweighted mean of animal increment SDs",
            "simulated": (
                "each sample_study draw is one realized study; the same animal→study "
                "statistic is computed per draw"
            ),
            "reported_separately": "E[study-level statistic] and between-draw SD of that statistic",
            "not_done": (
                "between-draw study-effect variability is not mixed into the increment SD"
            ),
        },
        "summary": nested,
        "study_rows": study_rows,
        "bayesian": {
            "thresholds_unchanged": dict(RELIABILITY_THRESHOLDS),
            "explicitly_unreliable": True,
            "scored": False,
            "n_cells": len(bayes_rows),
            "cells": bayes_rows,
            "note": (
                "Unchanged gate: rhat≤1.01, ESS≥400, divergences==0. Failed runs are "
                "preserved. Bayesian increment is not counted until those pass."
            ),
        },
        "reliability_thresholds": dict(RELIABILITY_THRESHOLDS),
    }
    write_json(output_dir / "increment.json", out)
    write_json(output_dir / "study_rows.json", study_rows)
    write_json(output_dir / "summary.json", nested)
    write_json(output_dir / "bayesian.json", out["bayesian"])
    md = _write_md(output_dir, out)
    if output_dir.name == "v3_increment":
        Path("docs/v3/increment_diagnostic.md").write_text(md, encoding="utf-8")
    return out


def _fmt(x: Any, nd: int = 5) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _write_md(parent: Path, out: dict[str, Any]) -> str:
    src = out.get("source_research") or {}
    nested = out.get("summary") or {}
    overall = nested.get("overall_matched_ratio") or {}
    lines = [
        "# Targeted increment diagnostic",
        "",
        "> **Synthetic diagnostic. Exploratory. Not design qualification. "
        "No real-data validation. No operational recommendation.**",
        "",
        "The synthetic closeout is the end of the implementation phase, subject to "
        "review. This note re-scores existing fitted models only. No architecture "
        "was added and no new training grid was launched.",
        "",
        "## Labels and replication",
        "",
        f"- labels: {out.get('labels')}",
        f"- exploratory: {out.get('exploratory')}",
        f"- independent dataset replicates: **{out.get('n_datasets')}**",
        f"- cells scored: {out.get('n_cells_scored')}; study rows: {nested.get('n_study_rows')}",
        f"- train seeds per dataset: {nested.get('n_train_per_dataset')}",
        f"- split: `{out.get('split')}` (prepared test studies only)",
        f"- research freeze: `{src.get('freeze_revision')}`",
        f"- increment revision: `{out.get('git_revision')}`",
        "",
        "## Matched statistic",
        "",
        "Observed and each simulated study draw use the same hierarchy:",
        "",
        "1. Animal: SD of consecutive observed-visit log-weight increments.",
        "2. Study: unweighted mean of those animal SDs.",
        "3. Dataset: mean of study-level statistics (train seeds nested inside).",
        "",
        "Each `sample_study` draw is one realized study (new study intercept/slope). "
        "The predicted study-level number is E[same statistic] over draws. The "
        "between-draw SD of that statistic is reported separately and is **not** "
        "folded into the increment SD.",
        "",
        f"- overall matched pred/obs ratio, mean of dataset means: "
        f"**{_fmt(overall.get('mean_of_group_means'))}** "
        f"(sd of dataset means {_fmt(overall.get('sd_of_group_means'))}; "
        f"n_datasets={overall.get('n_groups')})",
        "",
        "| model | sharing | bank | density | matched mean_ds (sd_ds) | "
        "obs study-stat mean_ds | pred E[study-stat] mean_ds | "
        "between-draw SD mean_ds | published pack ratio mean_ds | n_ds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in nested.get("groups") or []:
        k = g.get("key") or {}
        mr = g.get("matched_ratio") or {}
        obs = g.get("obs_study_stat") or {}
        pred = g.get("pred_expected_study_stat") or {}
        btw = g.get("pred_between_draw_sd") or {}
        pack = g.get("published_pack_ratio") or {}
        lines.append(
            f"| {k.get('model')} | {k.get('sharing')} | {k.get('bank')} | {k.get('density')} | "
            f"{_fmt(mr.get('mean_of_group_means'))} ({_fmt(mr.get('sd_of_group_means'))}) | "
            f"{_fmt(obs.get('mean_of_group_means'))} | {_fmt(pred.get('mean_of_group_means'))} | "
            f"{_fmt(btw.get('mean_of_group_means'))} | {_fmt(pack.get('mean_of_group_means'))} | "
            f"{mr.get('n_groups')} |"
        )
    bay = out.get("bayesian") or {}
    lines += [
        "",
        "The published pack ratio is the previous animal-pooled "
        "`mean(sd_pred)/mean(sd_obs)` from `evaluate_prestudy`. It is shown only as "
        "the mismatched baseline. Widths were not calibrated to test coverage.",
        "",
        "## Bayesian inference",
        "",
        f"- thresholds unchanged: `{bay.get('thresholds_unchanged')}`",
        f"- explicitly unreliable: {bay.get('explicitly_unreliable')}",
        f"- scored: {bay.get('scored')}",
        f"- cells recorded, not counted: {bay.get('n_cells')}",
        "",
        bay.get("note") or "",
        "",
        "| cell | class | rhat | ESS bulk/tail | counted |",
        "|---|---|---|---|---|",
    ]
    for c in bay.get("cells") or []:
        lines.append(
            f"| {c.get('name')} | {c.get('failure_class')} | {_fmt(c.get('rhat_max'), 3)} | "
            f"{_fmt(c.get('ess_bulk_min'), 1)} / {_fmt(c.get('ess_tail_min'), 1)} | "
            f"{c.get('counted_as_successful_benchmark')} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- `{parent / 'increment.json'}`",
        f"- `{parent / 'study_rows.json'}`",
        f"- `{parent / 'summary.json'}`",
        f"- `{parent / 'bayesian.json'}`",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (parent / "INCREMENT.md").write_text(text, encoding="utf-8")
    return text

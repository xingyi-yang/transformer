"""Synthetic closeout from executed v3 research artifacts.

Reads existing outputs only. Does not refit, retune, add families, or launch a
grid. Training-seed vs generate-seed nesting is reported explicitly. Study-balanced
contrast sits beside contrast-row transfer. Overcoverage is described without
calibrating widths to exposed test outcomes. Bayesian failed runs are preserved.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import git_revision, read_json, write_json

PRESERVED = ("runs/v3_eval", "runs/v3_pilot_k0k1", "runs/v3_repl", "runs/v3_research")
METRICS = ("rmse_log", "crps")


class CloseoutError(ValueError):
    pass


def nested_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Account for dataset seeds as the outer replicate and train seeds as nested.

    Training seeds reshuffle target splits (and refit). They are not independent
    dataset replicates. Pooling all pairs as iid overstates the dataset-level n.
    """
    if not rows:
        return {"n_datasets": 0, "n_train_per_dataset": 0, "n_pairs_pooled": 0,
                "pooled_mean": None, "pooled_sd": None, "mean_of_dataset_means": None,
                "sd_of_dataset_means": None, "mean_within_dataset_sd": None,
                "dataset_means": {}, "note": "no rows"}
    df = pd.DataFrame(rows)
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df = df.dropna(subset=["delta"])
    by = df.groupby("dataset_index")["delta"]
    means = by.mean()
    within = by.std(ddof=1)
    n_train = {int(k): int(v) for k, v in by.size().to_dict().items()}
    signs = [str(s) for s in df["sign"].tolist()] if "sign" in df else []
    return {
        "n_datasets": int(means.size),
        "n_train_per_dataset": n_train,
        "n_pairs_pooled": int(len(df)),
        "generate_seeds": sorted({int(x) for x in df["generate_seed"].tolist()}) if "generate_seed" in df else [],
        "train_seeds": sorted({int(x) for x in df["train_seed"].tolist()}) if "train_seed" in df else [],
        "pooled_mean": float(df["delta"].mean()),
        "pooled_sd": float(df["delta"].std(ddof=1)) if len(df) > 1 else None,
        "mean_of_dataset_means": float(means.mean()),
        "sd_of_dataset_means": float(means.std(ddof=1)) if len(means) > 1 else None,
        "mean_within_dataset_sd": float(within.mean()) if within.notna().any() else None,
        "dataset_means": {str(int(k)): float(v) for k, v in means.items()},
        "n_harmful": int(sum(s == "harmful" for s in signs)),
        "n_helpful": int(sum(s == "helpful" for s in signs)),
        "note": (
            "n_datasets is the outer replication count. Train seeds are nested: they "
            "also change target splits. pooled_sd treats the 2×2 crossing as iid and "
            "is not the dataset-level standard error."
        ),
    }


def _group_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def sharing_density_effects(paired: dict[str, Any]) -> dict[str, Any]:
    share = list(paired.get("sharing") or [])
    dens = list(paired.get("density") or [])
    full, sparse = [], []
    for r in share:
        if r.get("metric") not in METRICS:
            continue
        rec = {**r, "density": "full" if r.get("sparse_target_train") in ("full", None) else "sparse"}
        (full if rec["density"] == "full" else sparse).append(rec)
    share_full = _group_rows(full, "model", "sharing", "bank", "metric")
    share_sparse = _group_rows(sparse, "model", "sharing", "bank", "metric")
    dens_g = _group_rows([r for r in dens if r.get("metric") in METRICS],
                         "model", "sharing", "bank", "metric")
    groups = []
    for key in sorted(set(share_full) | set(share_sparse) | set(dens_g)):
        f, s, d = share_full.get(key, []), share_sparse.get(key, []), dens_g.get(key, [])
        inter_rows = []
        fmap = {(r.get("dataset_index"), r.get("train_index")): r for r in f}
        smap = {(r.get("dataset_index"), r.get("train_index")): r for r in s}
        for pair, fr in fmap.items():
            sr = smap.get(pair)
            if not sr or fr.get("delta") is None or sr.get("delta") is None:
                continue
            inter_rows.append({
                **{k: fr.get(k) for k in ("dataset_index", "train_index", "generate_seed",
                                          "train_seed", "model", "sharing", "bank", "metric")},
                "delta_share_full": fr["delta"],
                "delta_share_sparse": sr["delta"],
                "delta": float(sr["delta"]) - float(fr["delta"]),
                "sign": ("harmful" if (float(sr["delta"]) - float(fr["delta"])) > 0
                         else "helpful" if (float(sr["delta"]) - float(fr["delta"])) < 0
                         else "unchanged"),
            })
        groups.append({
            "key": dict(zip(("model", "sharing", "bank", "metric"), key, strict=True)),
            "sharing_full": nested_delta_summary(f),
            "sharing_sparse": nested_delta_summary(s),
            "sparse_vs_full": nested_delta_summary(d),
            "interaction_sparse_minus_full_sharing": nested_delta_summary(inter_rows),
        })
    return {"n_groups": len(groups), "groups": groups,
            "interaction_definition": (
                "interaction = (S_sparse,k − S_sparse,0) − (S_full,k − S_full,0). "
                "Positive means sharing is more harmful (or less helpful) under sparse "
                "target-study training than under full target-study training."
            )}


def train_seeds_change_splits(locked: dict[str, Any]) -> dict[str, Any]:
    """Training seeds reshuffle validation/test STUDYID assignments within a dataset."""
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, hashes in (locked or {}).items():
        parts = str(name).split("_")
        by_ds[parts[0]].append({"pair": name, **hashes})
    changed = []
    for ds, rows in by_ds.items():
        vals = {r.get("validation") for r in rows}
        tes = {r.get("test") for r in rows}
        changed.append({
            "dataset": ds,
            "n_train_seeds": len(rows),
            "validation_hashes_differ": len(vals) > 1,
            "test_hashes_differ": len(tes) > 1,
            "pairs": rows,
        })
    return {
        "training_seeds_change_splits": all(c["validation_hashes_differ"] or c["test_hashes_differ"] for c in changed)
        if changed else False,
        "generate_seed_changes_study_contents": True,
        "hash_note": (
            "Target-manifest hashes are STUDYID-set hashes. Training seeds reshuffle "
            "which of those IDs are validation vs test. Dataset seeds change the "
            "outcomes behind the same generator labels."
        ),
        "by_dataset": changed,
    }


def _pack_contrast(pack: dict[str, Any]) -> dict[str, Any]:
    den = pack.get("denominators") or {}
    con = pack.get("contrast") or {}
    st = con.get("study_level") or {}
    return {
        "contrast_row_bias": con.get("bias"),
        "contrast_row_n": con.get("n_contrasts") or den.get("n_contrasts"),
        "contrast_study_balanced_bias": den.get("contrast_bias_study_balanced", st.get("bias_study_mean")),
        "contrast_study_balanced_rmse": st.get("rmse_study_mean"),
        "n_studies_with_contrast": den.get("n_studies_with_contrast", st.get("n_studies_with_contrast")),
    }


def study_balanced_contrast(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    by: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for rec in cells:
        if rec.get("failed"):
            continue
        dens = "full" if rec.get("sparse_target_train_n") in (None, "full") else "sparse"
        packs = rec.get("evaluation_packs") or {}
        for model, pack in packs.items():
            if model == "bayesian":
                continue
            key = (rec.get("dataset_index"), rec.get("train_index"), rec.get("bank"), dens, model)
            by[key][rec.get("sharing")] = {**_pack_contrast(pack), "rec": rec, "model": model}
    for key, regs in by.items():
        s0 = regs.get("context_specific")
        if not s0:
            continue
        for sharing, sk in regs.items():
            if sharing == "context_specific":
                continue
            for kind, field in (("contrast_row", "contrast_row_bias"),
                                ("study_balanced", "contrast_study_balanced_bias")):
                v0, vk = s0.get(field), sk.get(field)
                if v0 is None or vk is None:
                    continue
                delta = abs(float(vk)) - abs(float(v0))
                rec = sk["rec"]
                rows.append({
                    "dataset_index": key[0], "train_index": key[1], "bank": key[2],
                    "density": key[3], "model": key[4], "sharing": sharing,
                    "metric": kind, "S0": float(v0), "Sk": float(vk), "delta": delta,
                    "sign": "harmful" if delta > 0 else "helpful" if delta < 0 else "unchanged",
                    "generate_seed": rec.get("generate_seed"), "train_seed": rec.get("train_seed"),
                    "n_contrasts": sk.get("contrast_row_n"),
                    "n_studies_with_contrast": sk.get("n_studies_with_contrast"),
                })
    grouped = []
    for key, g in _group_rows(rows, "model", "sharing", "bank", "density", "metric").items():
        grouped.append({
            "key": dict(zip(("model", "sharing", "bank", "density", "metric"), key, strict=True)),
            **nested_delta_summary(g),
        })
    return {
        "rows": rows,
        "groups": grouped,
        "note": (
            "Transfer contrast Δ remains |Sk|−|S0|. contrast_row is the contrast-row mean "
            "used in the published transfer tables. study_balanced is the unweighted mean "
            "of per-study contrast biases (denominators.contrast_bias_study_balanced)."
        ),
    }


def coverage_increment_investigation(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe overcoverage and increment SD ratio without calibrating to test."""
    rows = []
    bounds = []
    for rec in cells:
        if rec.get("failed"):
            continue
        dens = "full" if rec.get("sparse_target_train_n") in (None, "full") else "sparse"
        cov = rec.get("covariance") or {}
        if cov:
            bounds.append({
                "name": rec.get("name"), "params_at_bound": cov.get("params_at_bound"),
                "n_params_at_bound": cov.get("n_params_at_bound"),
                "boundary_flag": cov.get("boundary_flag"),
            })
        for model, pack in (rec.get("evaluation_packs") or {}).items():
            if model == "bayesian":
                continue
            ind = pack.get("individual") or {}
            tmp = pack.get("temporal") or {}
            den = pack.get("denominators") or {}
            rows.append({
                "name": rec.get("name"), "model": model, "sharing": rec.get("sharing"),
                "bank": rec.get("bank"), "density": dens,
                "dataset_index": rec.get("dataset_index"), "train_index": rec.get("train_index"),
                "cov80": ind.get("coverage80"), "cov95": ind.get("coverage95"),
                "width80": ind.get("width80_log"), "width95": ind.get("width95_log"),
                "increment_sd_ratio": tmp.get("increment_sd_ratio_pred_over_obs"),
                "extrema_max_cov90": tmp.get("extrema_max_cov90"),
                "n_studies": den.get("n_studies"), "n_obs": den.get("n_obs"),
                "n_increments": den.get("n_increments"),
            })
    df = pd.DataFrame(rows)
    def _summ(sub: pd.DataFrame) -> dict[str, Any]:
        if sub.empty:
            return {"n": 0}
        out = {"n": int(len(sub))}
        for c in ("cov80", "cov95", "width80", "width95", "increment_sd_ratio", "extrema_max_cov90"):
            s = pd.to_numeric(sub[c], errors="coerce").dropna()
            out[c] = {"mean": float(s.mean()) if len(s) else None,
                      "min": float(s.min()) if len(s) else None,
                      "max": float(s.max()) if len(s) else None,
                      "n": int(len(s))}
        out["n_over_cov80"] = int((pd.to_numeric(sub["cov80"], errors="coerce") > 0.80).sum())
        out["n_over_cov95"] = int((pd.to_numeric(sub["cov95"], errors="coerce") > 0.95).sum())
        out["n_increment_gt_1"] = int((pd.to_numeric(sub["increment_sd_ratio"], errors="coerce") > 1.0).sum())
        return out
    bound_counts: dict[str, int] = defaultdict(int)
    for b in bounds:
        for p in b.get("params_at_bound") or []:
            bound_counts[str(p)] += 1
    return {
        "n_cells": int(len(rows)),
        "by_model": {m: _summ(df[df["model"] == m]) for m in sorted(df["model"].unique())} if len(df) else {},
        "overall": _summ(df),
        "covariance_bounds": {
            "n_inspections": len(bounds),
            "n_with_boundary": int(sum(1 for b in bounds if b.get("boundary_flag"))),
            "param_counts": dict(bound_counts),
        },
        "mechanism": (
            "New-study draws add a shared study intercept/slope plus animal and OU terms. "
            "Increment SD is the mean, over draws, of per-animal consecutive-visit SDs; "
            "that includes the new study slope, so pred increment SD can exceed the "
            "observed within-animal increment SD. HierarchicalCovariance often parks "
            "sigma_m at its optimizer floor (train OOF, not a test-calibrated nugget). "
            "Wide predictive intervals then overcover. Widths were not shrunk to match "
            "exposed test coverage."
        ),
        "not_done": [
            "No residual scale was multiplied to hit nominal 80/95 coverage on test.",
            "No increment-ratio target was used as a tuning objective.",
            "Validation pre-study packs were not written by the research executor; "
            "compound-holdout coverage is a second exposed split and was not used to calibrate.",
        ],
        "denominators_note": "Coverage uses study-balanced means unless noted. Increment n is per-animal consecutive-visit series.",
    }


def bayesian_identifiability(cells: list[dict[str, Any]], research_root: Path) -> dict[str, Any]:
    reports = []
    for rec in cells:
        if "bayesian" not in (rec.get("models") or []) or rec.get("failed"):
            continue
        samp = rec.get("sampling_diagnosis") or {}
        gate = rec.get("bayesian_gate") or {}
        model_json = {}
        diag = {}
        if rec.get("dir"):
            mp = Path(rec["dir"]) / "models" / "bayesian" / "model.json"
            dp = Path(rec["dir"]) / "models" / "bayesian" / "diagnostics.json"
            if mp.exists():
                model_json = read_json(mp)
            if dp.exists():
                diag = read_json(dp).get("main") or {}
        cols = (model_json.get("design") or {}).get("col_names") or []
        dropped = list(diag.get("dropped_nonidentifiable_columns") or samp.get("dropped_nonidentifiable_columns") or [])
        dropped_names = [cols[i] for i in dropped if isinstance(i, int) and 0 <= i < len(cols)]
        reports.append({
            "name": rec.get("name"),
            "failure_class": samp.get("failure_class"),
            "counted_as_successful_benchmark": bool(gate.get("counted_as_successful_benchmark")),
            "reliable": bool(gate.get("reliable") or samp.get("reliable")),
            "rhat_max": samp.get("rhat_max") or gate.get("rhat_max"),
            "ess_bulk_min": samp.get("ess_bulk_min") or gate.get("ess_bulk_min"),
            "ess_tail_min": samp.get("ess_tail_min") or gate.get("ess_tail_min"),
            "divergences": samp.get("divergences") or gate.get("divergences"),
            "draws": samp.get("draws"), "tune": samp.get("tune"), "chains": samp.get("chains"),
            "thresholds": dict(RELIABILITY_THRESHOLDS),
            "priors": samp.get("priors") or diag.get("priors"),
            "n_mean_columns": len(cols),
            "dropped_columns": dropped,
            "dropped_names": dropped_names,
            "worst_rhat_params": samp.get("worst_rhat_params") or diag.get("worst_rhat_params"),
            "worst_ess_params": samp.get("worst_ess_params") or diag.get("worst_ess_params"),
            "family_diagnostics": samp.get("family_diagnostics") or diag.get("family_diagnostics"),
        })
    return {
        "thresholds_unchanged": dict(RELIABILITY_THRESHOLDS),
        "failed_runs_preserved": True,
        "n_cells": len(reports),
        "cells": reports,
        "parameterization": {
            "mean": "QR: prior on theta = R* beta; beta = Rinv theta; pivoted QR drops rank-deficient columns",
            "random_effects": "non-centered ZeroSumNormal study/animal intercept and slope (identifies mean via fixed effects)",
            "residual": "OU without a free white-noise nugget (documented non-identifiability vs sigma_p)",
            "rich": False,
        },
        "interpretation": (
            "Both executed cells fail on theta/beta (mean_identifiability), not compile. "
            "The dropped column is the near-empty off-treatment basis (ob_ld[0]; OFF≈0). "
            "Remaining poorly mixed coordinates sit on strain×time bases: few studies "
            "identify strain-specific growth once ZeroSum study effects and the shared "
            "time spline are present. Doubling draws/tune did not meet ESS≥400. "
            "Parameterization and thresholds were not changed to chase reliability."
        ),
        "source": str(research_root),
    }


def execute_v3_closeout(research_dir: str | Path = "runs/v3_research",
                        output_dir: str | Path = "runs/v3_closeout") -> dict[str, Any]:
    research_dir = Path(research_dir)
    output_dir = Path(output_dir)
    if output_dir.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise CloseoutError("refusing to write into a preserved artifact directory")
    if not (research_dir / "paired_rows.json").exists():
        raise CloseoutError(f"missing executed research artifacts under {research_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = read_json(research_dir / "paired_rows.json")
    locked = read_json(research_dir / "locked_target_splits.json")
    summary = read_json(research_dir / "SUMMARY.json")
    cells = list(summary.get("cells") or [])
    effects = sharing_density_effects(paired)
    splits = train_seeds_change_splits(locked)
    contrast = study_balanced_contrast(cells)
    coverage = coverage_increment_investigation(cells)
    bayes = bayesian_identifiability(cells, research_dir)
    out = {
        "labels": list(LABELS),
        "git_revision": git_revision(),
        "source_research": {
            "dir": str(research_dir),
            "config_hash": (summary.get("freeze_lock") or {}).get("config_hash"),
            "freeze_revision": (summary.get("freeze_lock") or {}).get("git_revision"),
        },
        "retuned_after_test": False,
        "architectures_added": False,
        "grid_relaunched": False,
        "nested_seeds": splits,
        "sharing_density_effects": effects,
        "study_balanced_contrast": contrast,
        "coverage_increments": coverage,
        "bayesian_identifiability": bayes,
        "reliability_thresholds": dict(RELIABILITY_THRESHOLDS),
    }
    write_json(output_dir / "closeout.json", out)
    write_json(output_dir / "sharing_density_effects.json", effects)
    write_json(output_dir / "study_balanced_contrast.json", contrast)
    write_json(output_dir / "coverage_increments.json", coverage)
    write_json(output_dir / "bayesian_identifiability.json", bayes)
    md = _write_md(output_dir, out)
    if output_dir.name == "v3_closeout":
        Path("docs/v3/synthetic_closeout.md").write_text(md, encoding="utf-8")
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
    splits = out.get("nested_seeds") or {}
    lines = [
        "# Synthetic implementation closeout",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.**",
        "",
        "PR #5 completes the synthetic implementation milestone, subject to normal "
        "code review. This note uses those artifacts only. Unfavorable signs are "
        "preserved. No architecture was added and no new training grid was launched.",
        "",
        "## Provenance",
        "",
        f"- research freeze: `{src.get('freeze_revision')}`",
        f"- research config hash: `{src.get('config_hash')}`",
        f"- closeout revision: `{out.get('git_revision')}`",
        f"- reliability thresholds: `{out.get('reliability_thresholds')}`",
        "",
        "## Nested seeds and splits",
        "",
        f"- Training seeds change target splits: **{splits.get('training_seeds_change_splits')}**.",
        f"- {splits.get('hash_note')}",
        "- Outer replicate: dataset / generate_seed (n=2).",
        "- Nested replicate: train_seed (n=2 per dataset), which reshuffles validation/test STUDYIDs and refits.",
        "- Published research tables used pooled n=4. That is not a dataset-level n.",
        "",
    ]
    for ds in splits.get("by_dataset") or []:
        lines.append(
            f"- `{ds.get('dataset')}`: {ds.get('n_train_seeds')} train seeds; "
            f"validation hashes differ={ds.get('validation_hashes_differ')}; "
            f"test hashes differ={ds.get('test_hashes_differ')}."
        )
    lines += [
        "",
        "## Sharing × density (RMSE and CRPS)",
        "",
        "Sharing Δ = S_k − S_0 (context-specific). Sparse Δ = S_sparse − S_full. "
        "Interaction = sharing Δ under sparse minus sharing Δ under full. "
        "Δ>0 is harmful for RMSE/CRPS. `mean_ds` is the mean of the two dataset "
        "means; `sd_ds` is the dataset-level sd (n=2). `pooled` is the 2×2 iid summary.",
        "",
        "| model | sharing | bank | metric | share full mean_ds (sd_ds) | share sparse mean_ds | sparse-vs-full mean_ds | interaction mean_ds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in (out.get("sharing_density_effects") or {}).get("groups") or []:
        k = g.get("key") or {}
        sf, ss, d, inter = (g.get("sharing_full") or {}, g.get("sharing_sparse") or {},
                            g.get("sparse_vs_full") or {}, g.get("interaction_sparse_minus_full_sharing") or {})
        lines.append(
            f"| {k.get('model')} | {k.get('sharing')} | {k.get('bank')} | {k.get('metric')} | "
            f"{_fmt(sf.get('mean_of_dataset_means'))} ({_fmt(sf.get('sd_of_dataset_means'))}) | "
            f"{_fmt(ss.get('mean_of_dataset_means'))} | {_fmt(d.get('mean_of_dataset_means'))} | "
            f"{_fmt(inter.get('mean_of_dataset_means'))} |"
        )
    lines += [
        "",
        "Spline sharing remains harmful for RMSE at full density on both donor banks. "
        "Sparse target-study training makes spline sharing less harmful (negative "
        "interaction) and can reverse the sparse-vs-full sign once donors are present. "
        "Transformer donor-using sparse training stays harmful for RMSE and CRPS. "
        "Signs are published; they were not retuned.",
        "",
        "## Contrast: row-weighted vs study-balanced",
        "",
        "| model | sharing | bank | density | row-mean Δ | study-balanced Δ | n_ds |",
        "|---|---|---|---|---|---|---|",
    ]
    contrast_groups = (out.get("study_balanced_contrast") or {}).get("groups") or []
    pair = {}
    for g in contrast_groups:
        k = g.get("key") or {}
        key = (k.get("model"), k.get("sharing"), k.get("bank"), k.get("density"))
        pair.setdefault(key, {})[k.get("metric")] = g
    for key in sorted(pair):
        row, sb = pair[key].get("contrast_row") or {}, pair[key].get("study_balanced") or {}
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | "
            f"{_fmt(row.get('mean_of_dataset_means'))} | {_fmt(sb.get('mean_of_dataset_means'))} | "
            f"{row.get('n_datasets') or sb.get('n_datasets')} |"
        )
    lines += [
        "",
        "Study-balanced contrast can differ in magnitude from the contrast-row mean "
        "because studies contribute unequal numbers of active arms. Neither series was "
        "substituted for the other in the frozen transfer tables.",
        "",
        "## Overcoverage and increment variability",
        "",
    ]
    cov = out.get("coverage_increments") or {}
    ov = cov.get("overall") or {}
    lines += [
        f"- packs: {ov.get('n')}; cov80>0.80 in {ov.get('n_over_cov80')}; "
        f"cov95>0.95 in {ov.get('n_over_cov95')}; increment ratio>1 in {ov.get('n_increment_gt_1')}",
        f"- mean cov80/cov95: {_fmt((ov.get('cov80') or {}).get('mean'))} / {_fmt((ov.get('cov95') or {}).get('mean'))}",
        f"- mean increment SD pred/obs: {_fmt((ov.get('increment_sd_ratio') or {}).get('mean'))}",
        f"- covariance inspections with a bound: {(cov.get('covariance_bounds') or {}).get('n_with_boundary')} / "
        f"{(cov.get('covariance_bounds') or {}).get('n_inspections')}; "
        f"params { (cov.get('covariance_bounds') or {}).get('param_counts') }",
        "",
        cov.get("mechanism") or "",
        "",
        "Not done: " + "; ".join(cov.get("not_done") or []),
        "",
        "## Bayesian mean identifiability",
        "",
    ]
    bay = out.get("bayesian_identifiability") or {}
    lines += [
        f"- thresholds unchanged: `{bay.get('thresholds_unchanged')}`",
        f"- failed runs preserved: {bay.get('failed_runs_preserved')}",
        "",
        "| cell | class | rhat | ESS bulk/tail | dropped | counted |",
        "|---|---|---|---|---|---|",
    ]
    for c in bay.get("cells") or []:
        lines.append(
            f"| {c.get('name')} | {c.get('failure_class')} | {_fmt(c.get('rhat_max'), 3)} | "
            f"{_fmt(c.get('ess_bulk_min'), 1)} / {_fmt(c.get('ess_tail_min'), 1)} | "
            f"{c.get('dropped_names')} | {c.get('counted_as_successful_benchmark')} |"
        )
    lines += ["", bay.get("interpretation") or "", "",
              "## Historical-data handoff", "",
              "Synthetic implementation stops here. The historical-data audit is a separate "
              "gated workflow (`send-bw historical-audit`). It does not read real extracts "
              "until a signed authorization names the phase. Design recommendations stay off.",
              "",
              "## Artifacts", "",
              f"- `{parent / 'closeout.json'}`",
              f"- `{parent / 'sharing_density_effects.json'}`",
              f"- `{parent / 'study_balanced_contrast.json'}`",
              f"- `{parent / 'coverage_increments.json'}`",
              f"- `{parent / 'bayesian_identifiability.json'}`",
              ""]
    text = "\n".join(lines) + "\n"
    (parent / "CLOSEOUT.md").write_text(text, encoding="utf-8")
    return text

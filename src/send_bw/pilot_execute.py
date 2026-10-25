"""Execute the frozen k=0 → k=1 synthetic pilot. No retune. No design. No k>1.

The freeze lock (git revision + config hash) is written before generate/train.
Target manifests must match across k. Hyperparameters are taken from the frozen
YAML; early stopping uses target validation only. Transfer signs are published
even when harmful.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from send_bw.evaluation.transfer import build_transfer_table
from send_bw.models.base import study_balanced_weights
from send_bw.models.transformer import token_groups
from send_bw.pilot import LABELS, SELECTIVE_SHARING, STUDY_WEIGHTING, UNCERTAINTY_SAFEGUARDS, as_dict
from send_bw.pipeline import Run
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json

ALLOWED_K = (0, 1)
FORBIDDEN_MODELS = ("bayesian", "tft", "chronos2", "deepar")


class PilotProtocolError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_frozen_config(cfg: dict[str, Any]) -> None:
    pilot = cfg.get("pilot") or {}
    sched = list(pilot.get("k_schedule") or [])
    if sched != [0, 1]:
        raise PilotProtocolError(f"k_schedule must be [0, 1]; got {sched}")
    if int(pilot.get("stop_after_k", 1)) != 1:
        raise PilotProtocolError("stop_after_k must be 1")
    if pilot.get("retune_after_test", False):
        raise PilotProtocolError("retune_after_test is forbidden")
    if (cfg.get("design") or {}).get("request"):
        raise PilotProtocolError("design operations are off for this pilot")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise PilotProtocolError("real-data access is off")
    models = cfg.get("models") or {}
    if models.get("bayesian", {}).get("enabled"):
        raise PilotProtocolError("bayesian is off")
    opt = models.get("optional") or {}
    for name in ("tft", "chronos2", "deepar"):
        if (opt.get(name) or {}).get("enabled"):
            raise PilotProtocolError(f"{name} is off")
    tr = models.get("transformer") or {}
    if int(tr.get("n_members", 0)) != 1:
        raise PilotProtocolError("transformer n_members must be 1")
    if not (models.get("spline") or {}).get("enabled"):
        raise PilotProtocolError("spline must be enabled")
    tasks = list((cfg.get("evaluation") or {}).get("tasks") or [])
    if tasks != ["predose_conditioned"]:
        raise PilotProtocolError(f"evaluation.tasks must be [predose_conditioned]; got {tasks}")
    if (cfg.get("cohort") or {}).get("sharing") == "selective":
        raise PilotProtocolError("selective sharing is deferred")


def git_tracked_dirty() -> list[str]:
    import subprocess

    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                             timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    dirty = []
    for line in out.stdout.splitlines():
        if line.startswith("??") or line.startswith("!!"):
            continue
        dirty.append(line)
    return dirty


def write_freeze_lock(parent: Path, cfg: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    dirty = git_tracked_dirty()
    if dirty:
        raise PilotProtocolError("refuse to execute with dirty tracked files: " + "; ".join(dirty))
    lock = {
        "frozen_before": "generate_train_evaluate",
        "config_path": str(config_path),
        "config_hash": cache_key(cfg),
        "git_revision": git_revision(),
        "k_schedule": [0, 1],
        "stop_after_k": 1,
        "retune_after_test": False,
        "design": False,
        "real_data": False,
        "extra_families": False,
        "selective_sharing": SELECTIVE_SHARING,
        "prespec": as_dict(),
        "labels": list(LABELS),
        "uncertainty": {
            "predictive_variability": (
                "HierarchicalCovariance study, animal, and residual draws in sample_study. "
                "This is predictive variability of new-study trajectories under the fitted residual layer."
            ),
            "parameter_uncertainty": (
                "Spline: study-bootstrap of the mean only (n_bootstrap includes member 0). "
                "Transformer: n_members=1, so there is no bootstrap member spread. "
                "Covariance parameters are not bootstrapped. Parameter uncertainty is "
                "incompletely estimated; do not treat predictive intervals as complete "
                "parameter-uncertainty intervals."
            ),
            "safeguards": UNCERTAINTY_SAFEGUARDS,
        },
    }
    write_json(parent / "freeze_lock.json", lock)
    return lock


def k_config(base: dict[str, Any], k: int, parent: Path) -> dict[str, Any]:
    if k not in ALLOWED_K:
        raise PilotProtocolError(f"k={k} is off")
    cfg = copy.deepcopy(base)
    cfg["run_name"] = f"v3_pilot_k{k}"
    cfg["output_dir"] = str(parent / f"k{k}")
    cfg["synthetic"] = dict(cfg.get("synthetic") or {})
    cfg["synthetic"]["enabled"] = False
    cfg["synthetic"]["out_dir"] = str(parent / "synthetic")
    cfg["source"] = dict(cfg.get("source") or {})
    cfg["source"]["type"] = "local"
    cfg["source"]["root"] = str(parent / "synthetic" / "send")
    cfg["source"]["metadata_dir"] = str(parent / "synthetic" / "metadata")
    cfg["design"] = {}
    cohort = dict(cfg.get("cohort") or {})
    if k == 0:
        cohort["sharing"] = "context_specific"
        cohort["n_donor_studies"] = 0
    else:
        cohort["sharing"] = "pooled"
        cohort["n_donor_studies"] = int(k)
    cfg["cohort"] = cohort
    return cfg


def _species_counts(animals: pd.DataFrame) -> dict[str, int]:
    if "SPECIES" not in animals.columns:
        return {}
    return animals.groupby(animals["SPECIES"].astype(str).str.upper())["STUDYID"].nunique().to_dict()


def _weight_totals(study_ids) -> dict[str, float]:
    sid = pd.Series(study_ids).astype(str)
    w = study_balanced_weights(sid.to_numpy())
    totals = pd.Series(w, index=sid).groupby(level=0).sum()
    return {str(k): float(v) for k, v in totals.items()}


def record_pilot_fit(run: Run, k: int) -> dict[str, Any]:
    prep = run.load_prepared()
    spec = prep["spec"]
    data = prep["data"]
    train, val = data["train"], data["validation"]
    donor = read_json(run.p("splits", "donor_order.json")) if run.p("splits", "donor_order.json").exists() else {}
    target = read_json(run.p("splits", "target_manifest.json"))
    tr_ids = sorted(train.studies)
    selected = list(donor.get("selected") or [])
    hist_s = read_json(run.p("models", "spline", "training_history.json")) if run.p("models", "spline", "training_history.json").exists() else {}
    hist_t = read_json(run.p("models", "transformer", "training_history.json")) if run.p("models", "transformer", "training_history.json").exists() else {}
    t_members = hist_t.get("members") or []
    epochs = int(t_members[0]["epochs_run"]) if t_members else None
    n_animals_tr = len(train.animals)
    bs = int((run.cfg.get("models") or {}).get("transformer", {}).get("batch_size", 64))
    steps = None
    if epochs is not None:
        steps = int(epochs * max(1, math.ceil(n_animals_tr / max(bs, 1))))
    species_cols = [c for c in spec.animal_cols if c.startswith("SPECIES_")]
    man = run.manifest.data
    warnings = list((read_json(run.p("qc", "qc_report.json")) or {}).get("limitations") or [])
    rec = {
        "k": int(k),
        "sharing": (run.cfg.get("cohort") or {}).get("sharing"),
        "task": "predose_conditioned",
        "labels": list(LABELS),
        "target_manifest": target,
        "donor_order": donor,
        "training_composition": {
            "train_studies": tr_ids,
            "validation_studies": sorted(val.studies),
            "test_studies": sorted(data["test"].studies) if "test" in data else [],
            "donor_studies_in_train": selected,
            "n_train_studies": len(tr_ids),
            "n_train_animals": int(len(train.animals)),
            "n_train_obs": int(len(train.obs)),
            "n_validation_animals": int(len(val.animals)),
            "n_validation_obs": int(len(val.obs)),
            "species_counts_train_studies": _species_counts(train.animals),
            "study_weighting": STUDY_WEIGHTING,
        },
        "study_weights": {
            "rule": STUDY_WEIGHTING,
            "per_study_total_train": _weight_totals(train.obs["STUDYID"]) if len(train.obs) else {},
            "note": "each STUDYID has equal total weight; totals average toward n_obs/n_studies",
        },
        "hyperparameters": {
            "selected_on": "validation_target",
            "spline_alpha": hist_s.get("alpha"),
            "spline_validation_rmse_by_alpha": hist_s.get("validation_rmse_by_alpha"),
            "transformer_best_val_rmse_member0": hist_t.get("best_val_rmse_member0"),
            "retuned_after_test": False,
        },
        "optimizer_steps": {
            "spline": {
                "method": "ridge_closed_form",
                "alphas_evaluated": len((run.cfg.get("models") or {}).get("spline", {}).get("ridge_alphas") or []),
                "oof_folds": hist_s.get("covariance", {}).get("n_studies") or (run.cfg.get("models") or {}).get("spline", {}).get("oof_folds"),
                "n_bootstrap_members": hist_s.get("n_members"),
            },
            "transformer": {
                "member0_epochs_run": epochs,
                "max_epochs": (run.cfg.get("models") or {}).get("transformer", {}).get("max_epochs"),
                "patience": (run.cfg.get("models") or {}).get("transformer", {}).get("patience"),
                "approx_adamw_steps": steps,
                "n_members": 1,
                "batch_size": bs,
                "n_train_animals": n_animals_tr,
            },
        },
        "species_schema": {
            "sharing": spec.sharing,
            "species": spec.species,
            "species_levels": spec.species_levels,
            "animal_cols": spec.animal_cols,
            "species_feature_columns": species_cols,
            "transformer_token_groups": {g: cols for g, cols in token_groups(spec).items()},
        },
        "covariance_provenance": {
            "layer": "HierarchicalCovariance",
            "fitted_on": "spline study-level out-of-fold residuals (train studies)",
            "covariance_parameters_bootstrapped": False,
            "shared_with_transformer": True,
            "spline_covariance": hist_s.get("covariance"),
        },
        "uncertainty": {
            "predictive_variability": (
                "New-study sampling uses HierarchicalCovariance draws (study + animal + residual). "
                "That is predictive variability under the fitted residual layer."
            ),
            "parameter_uncertainty": (
                "Spline mean members are a study bootstrap (incompletely: covariance not "
                "bootstrapped). Transformer n_members=1: no bootstrap spread, so transformer "
                "parameter uncertainty is incompletely estimated."
            ),
            "incompletely_estimated": [
                "covariance parameters are not bootstrapped",
                "transformer n_members=1 (no bootstrap members)",
            ],
        },
        "runtime_seconds": {name: (man.get("stages") or {}).get(name, {}).get("seconds")
                            for name in ("prepare", "train_spline", "train_transformer", "evaluate")},
        "warnings": warnings,
    }
    write_json(run.p("pilot_record.json"), rec)
    return rec


def assert_target_manifests_match(k0: Path, k1: Path) -> dict[str, Any]:
    a = read_json(k0 / "splits" / "target_manifest.json")
    b = read_json(k1 / "splits" / "target_manifest.json")
    if a != b:
        raise PilotProtocolError("target manifests changed between k=0 and k=1")
    d0 = read_json(k0 / "splits" / "donor_order.json")
    d1 = read_json(k1 / "splits" / "donor_order.json")
    if d1.get("selected") and set(d1["selected"]) & set(a["validation"]["studyids"] + a["test"]["studyids"]):
        raise PilotProtocolError("donor isolation failed: donor in target validation/test")
    if d0.get("selected"):
        raise PilotProtocolError("k=0 must have no selected donors")
    return {"target_manifest": a, "k0_donors": d0, "k1_donors": d1}


def assemble_transfer(k0_metrics: dict[str, Any], k1_metrics: dict[str, Any]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for key, s0 in k0_metrics.items():
        if not str(key).endswith("pre_study"):
            continue
        sk = k1_metrics.get(key)
        if not sk:
            continue
        tables[key] = build_transfer_table(s0, [(1, sk)], model=s0.get("model") or sk.get("model"))
    return tables


def execute_k0k1(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    base = load_yaml(config_path)
    validate_frozen_config(base)
    parent = Path(base.get("output_dir") or "runs/v3_pilot_k0k1")
    parent.mkdir(parents=True, exist_ok=True)
    (parent / ".send_bw_generated").write_text("generated by send_bw\n", encoding="utf-8")
    lock_path = parent / "freeze_lock.json"
    if lock_path.exists() and not force:
        lock = read_json(lock_path)
    else:
        lock = write_freeze_lock(parent, base, config_path)
    t0 = time.time()
    gen_cfg = copy.deepcopy(base)
    gen_cfg["output_dir"] = str(parent)
    gen_cfg["synthetic"]["out_dir"] = str(parent / "synthetic")
    gen_cfg["source"]["root"] = str(parent / "synthetic" / "send")
    gen_cfg["source"]["metadata_dir"] = str(parent / "synthetic" / "metadata")
    gen_yaml = parent / "generate.yaml"
    gen_yaml.write_text(yaml.safe_dump(gen_cfg, sort_keys=False), encoding="utf-8")
    gen_run = Run(gen_yaml)
    gen_run.generate(force)
    records = {}
    for k in (0, 1):
        kcfg = k_config(base, k, parent)
        k_yaml = parent / f"k{k}" / "effective.yaml"
        k_yaml.parent.mkdir(parents=True, exist_ok=True)
        k_yaml.write_text(yaml.safe_dump(kcfg, sort_keys=False), encoding="utf-8")
        run = Run(k_yaml)
        run.inspect_source()
        run.ingest(force)
        run.prepare(force)
        enabled = [m for m in ("spline", "transformer") if (kcfg.get("models") or {}).get(m, {}).get("enabled")]
        run.train(enabled, force)
        run.evaluate(force)
        if (run.cfg.get("design") or {}).get("request"):
            raise PilotProtocolError("design request appeared during execution")
        design_status = run.design(force)
        if design_status.get("status") not in (None, "not_run"):
            raise PilotProtocolError(f"design must stay not_run; got {design_status}")
        run.report()
        records[k] = record_pilot_fit(run, k)
    check = assert_target_manifests_match(parent / "k0", parent / "k1")
    m0 = read_json(parent / "k0" / "evaluation" / "metrics.json")
    m1 = read_json(parent / "k1" / "evaluation" / "metrics.json")
    tables = assemble_transfer(m0, m1)
    write_json(parent / "transfer_table.json", tables)
    # also copy into k1 so the report path can find it
    write_json(parent / "k1" / "evaluation" / "transfer_table.json", tables)
    summary = {
        "labels": list(LABELS),
        "freeze_lock": lock,
        "target_manifest_check": {name: check["target_manifest"][name]["hash"]
                                  for name in ("train", "validation", "test")},
        "donor_isolation": {
            "k0_selected": check["k0_donors"].get("selected"),
            "k1_selected": check["k1_donors"].get("selected"),
            "k1_available": check["k1_donors"].get("studyids"),
        },
        "k_records": {str(k): {"sharing": records[k]["sharing"],
                               "train_studies": records[k]["training_composition"]["train_studies"],
                               "donor_studies_in_train": records[k]["training_composition"]["donor_studies_in_train"],
                               "species_schema": records[k]["species_schema"],
                               "hyperparameters": records[k]["hyperparameters"],
                               "optimizer_steps": records[k]["optimizer_steps"],
                               "runtime_seconds": records[k]["runtime_seconds"],
                               "warnings": records[k]["warnings"]}
                      for k in (0, 1)},
        "metrics_test": {
            key: {"k0": m0.get(key, {}).get("study_balanced"),
                  "k1": m1.get(key, {}).get("study_balanced"),
                  "contrast_bias": {"k0": m0.get(key, {}).get("contrast_bias"),
                                    "k1": m1.get(key, {}).get("contrast_bias")}}
            for key in m0 if str(key).endswith("pre_study") and str(key).split("|")[1] == "test"
        },
        "transfer_table": tables,
        "seconds_total": time.time() - t0,
        "retuned_after_test": False,
        "failures": [],
    }
    write_json(parent / "SUMMARY.json", summary)
    _write_results_md(parent, summary, m0, m1)
    return summary


def _write_results_md(parent: Path, summary: dict[str, Any], m0: dict, m1: dict) -> None:
    lock = summary.get("freeze_lock") or {}
    lines = [
        "# v3 synthetic pilot k=0 → k=1",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.**",
        "",
        "Stopped after k=1. Signs in the transfer table are published as computed. "
        "The protocol was not retuned after target-test metrics.",
        "",
        "## Freeze lock",
        "",
        f"- git revision: `{lock.get('git_revision')}`",
        f"- config hash: `{lock.get('config_hash')}`",
        f"- k_schedule: {lock.get('k_schedule')}",
        "",
        "## Target manifests (must match)",
        "",
    ]
    for name, h in (summary.get("target_manifest_check") or {}).items():
        lines.append(f"- {name}: `{h}`")
    lines += ["", "## Uncertainty (fixed language)", "",
              "- Predictive variability: HierarchicalCovariance draws in `sample_study`.",
              "- Parameter uncertainty is incompletely estimated: spline bootstraps the mean only; "
              "covariance parameters are not bootstrapped; transformer `n_members=1` has no "
              "bootstrap spread.",
              "", "## Transfer table (predose-conditioned, frozen target test)", ""]
    rows = []
    for block in (summary.get("transfer_table") or {}).values():
        rows.extend(block.get("rows") or [])
    if rows:
        lines += ["| model | k | metric | S0 | Sk | delta | sign |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            if r.get("metric") in ("rmse_log", "crps", "contrast_bias"):
                lines.append(
                    f"| {r.get('model')} | {r.get('k')} | {r.get('metric')} | "
                    f"{r.get('S0')} | {r.get('Sk')} | {r.get('delta')} | {r.get('sign')} |"
                )
    else:
        lines.append("(no rows)")
    lines += ["", "## Artifact locations", "",
              f"- parent: `{parent}`",
              f"- freeze lock: `{parent / 'freeze_lock.json'}`",
              f"- k=0: `{parent / 'k0'}`",
              f"- k=1: `{parent / 'k1'}`",
              f"- transfer table: `{parent / 'transfer_table.json'}`",
              ""]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

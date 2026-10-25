"""Frozen v3 synthetic evaluation benchmark.

Uses the stages 1–2 pilot machinery. Writes a freeze lock and the synthetic
evaluation bank before any comparison fit. Design recommendations and real
data stay off. Signs are published; the protocol is not retuned after test.
An unreliable Bayesian fit is recorded and is not counted as a successful
benchmark.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import yaml

from send_bw.evaluation.transfer import build_transfer_table
from send_bw.evaluation.uncertainty import bayesian_gate, inspect_covariance
from send_bw.pipeline import Run
from send_bw.sharing import LABELS, SELECTIVE_V1
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json

BANKS = ("matched", "both")


class EvalProtocolError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_eval_config(cfg: dict[str, Any]) -> None:
    ev = cfg.get("eval") or {}
    if ev.get("retune_after_test", False):
        raise EvalProtocolError("retune_after_test is forbidden")
    if (cfg.get("design") or {}).get("request"):
        raise EvalProtocolError("design operations are off for this evaluation")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise EvalProtocolError("real-data access is off")
    opt = (cfg.get("models") or {}).get("optional") or {}
    for name in ("tft", "chronos2", "deepar"):
        if (opt.get(name) or {}).get("enabled"):
            raise EvalProtocolError(f"{name} is off")
    tasks = list((cfg.get("evaluation") or {}).get("tasks") or [])
    if "predose_conditioned" not in tasks:
        raise EvalProtocolError("evaluation.tasks must include predose_conditioned")
    banks = list(ev.get("banks") or BANKS)
    if any(b not in BANKS for b in banks):
        raise EvalProtocolError(f"eval.banks must be subset of {BANKS}")


def write_freeze_lock(parent: Path, cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "config_path": str(config_path),
        "config_hash": cache_key(cfg),
        "git_revision": git_revision(),
        "frozen_before": "generate_banks_train_evaluate",
        "retune_after_test": False,
        "design": False,
        "real_data": False,
        "selective_sharing": SELECTIVE_V1,
        "labels": list(LABELS),
        "uncertainty": {
            "predictive_variability": (
                "HierarchicalCovariance study, animal, and residual draws in sample_study."
            ),
            "parameter_uncertainty": (
                "Incompletely estimated except for a reliable Bayesian posterior. "
                "Spline bootstraps the mean only; covariance is not bootstrapped; "
                "transformer n_members=1 has no bootstrap spread."
            ),
        },
    }
    write_json(parent / "freeze_lock.json", lock)
    return lock


def _bank_cfg(base: dict[str, Any], parent: Path, scenario: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    out = parent / "banks" / scenario
    cfg["run_name"] = f"v3_eval_bank_{scenario}"
    cfg["output_dir"] = str(out)
    cfg["synthetic"] = dict(cfg.get("synthetic") or {})
    cfg["synthetic"]["enabled"] = True
    cfg["synthetic"]["out_dir"] = str(out / "synthetic")
    cfg["synthetic"]["donor_scenario"] = scenario
    cfg["source"] = dict(cfg.get("source") or {})
    cfg["source"]["type"] = "local"
    cfg["source"]["root"] = str(out / "synthetic" / "send")
    cfg["source"]["metadata_dir"] = str(out / "synthetic" / "metadata")
    cfg["design"] = {}
    return cfg


def _cell_cfg(base: dict[str, Any], parent: Path, *, bank: str, sharing: str, k: int,
              fixed_schema: bool, budget_name: str, models: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    name = f"{bank}_{sharing}_k{k}_{'fix' if fixed_schema else 'var'}_{budget_name}"
    bank_syn = parent / "banks" / bank / "synthetic"
    cell = parent / "cells" / name
    cfg["run_name"] = f"v3_eval_{name}"
    cfg["output_dir"] = str(cell)
    cfg["synthetic"] = dict(cfg.get("synthetic") or {})
    cfg["synthetic"]["enabled"] = False
    cfg["synthetic"]["out_dir"] = str(bank_syn)
    cfg["synthetic"]["donor_scenario"] = bank
    cfg["source"] = dict(cfg.get("source") or {})
    cfg["source"]["type"] = "local"
    cfg["source"]["root"] = str(bank_syn / "send")
    cfg["source"]["metadata_dir"] = str(bank_syn / "metadata")
    cfg["cohort"] = dict(cfg.get("cohort") or {})
    cfg["cohort"]["sharing"] = sharing
    cfg["cohort"]["n_donor_studies"] = int(k)
    cfg["features"] = {"fixed_schema": bool(fixed_schema)}
    cfg["design"] = {}
    budgets = ((cfg.get("eval") or {}).get("budgets") or {}).get(budget_name) or {}
    models_cfg = dict(cfg.get("models") or {})
    for fam in ("spline", "transformer", "bayesian"):
        models_cfg[fam] = dict(models_cfg.get(fam) or {})
        models_cfg[fam]["enabled"] = fam in models
        if fam in budgets:
            models_cfg[fam].update(budgets[fam])
    opt = dict(models_cfg.get("optional") or {})
    for extra in ("tft", "chronos2", "deepar"):
        opt[extra] = {"enabled": False}
    models_cfg["optional"] = opt
    cfg["models"] = models_cfg
    return cfg


def _record_cell(run: Run, spec: dict[str, Any]) -> dict[str, Any]:
    man = run.manifest.data
    spline_hist = {}
    cov_insp = {}
    if run.p("models", "spline", "training_history.json").exists():
        spline_hist = read_json(run.p("models", "spline", "training_history.json"))
        cov_insp = inspect_covariance(spline_hist.get("covariance"))
        write_json(run.p("evaluation", "covariance_inspection.json"), cov_insp)
    bayes = None
    train_b = man.get("stages", {}).get("train_bayesian") or {}
    if train_b:
        bayes = bayesian_gate({
            "reliable": train_b.get("reliable"),
            **(train_b.get("diagnostics_summary") or {}),
        })
        write_json(run.p("evaluation", "bayesian_gate.json"), bayes)
    tman = read_json(run.p("splits", "target_manifest.json")) if run.p("splits", "target_manifest.json").exists() else {}
    metrics = read_json(run.p("evaluation", "metrics.json")) if run.p("evaluation", "metrics.json").exists() else {}
    return {
        **spec,
        "dir": str(run.dir),
        "target_manifest": {k: (v or {}).get("hash") for k, v in tman.items() if isinstance(v, dict)},
        "covariance": cov_insp,
        "bayesian_gate": bayes,
        "selective_sharing": spline_hist.get("selective_sharing"),
        "metrics_keys": list(metrics),
        "warnings": (read_json(run.p("qc", "qc_report.json")).get("limitations")
                     if run.p("qc", "qc_report.json").exists() else []),
    }


def execute_v3_eval(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    validate_eval_config(cfg)
    parent = Path(cfg.get("output_dir") or "runs/v3_eval")
    ev = cfg.get("eval") or {}
    t0 = time.time()
    lock = write_freeze_lock(parent, cfg, config_path)
    banks = list(ev.get("banks") or BANKS)
    primary = str(ev.get("primary_bank") or "both")
    bank_manifest: dict[str, Any] = {}
    for scenario in banks:
        bcfg = _bank_cfg(cfg, parent, scenario)
        run = _run_from_cfg(bcfg)
        run.generate(force=force)
        truth = read_json(Path(bcfg["synthetic"]["out_dir"]) / "truth" / "params.json")
        bank_manifest[scenario] = {
            "synthetic_dir": bcfg["synthetic"]["out_dir"],
            "donor_scenario": scenario,
            "donor_deviations": truth.get("donor_deviations"),
            "seed": cfg.get("seed"),
        }
    write_json(parent / "bank_manifest.json", bank_manifest)

    cells_spec = [
        {"bank": primary, "sharing": "context_specific", "k": 0, "fixed_schema": False,
         "budget": "primary", "models": ["spline", "transformer", "bayesian"], "sparse": True},
        {"bank": primary, "sharing": "pooled", "k": 1, "fixed_schema": False,
         "budget": "primary", "models": ["spline", "transformer", "bayesian"], "sparse": True},
        {"bank": primary, "sharing": "selective", "k": 1, "fixed_schema": False,
         "budget": "primary", "models": ["spline", "transformer", "bayesian"], "sparse": True},
        {"bank": primary, "sharing": "context_specific", "k": 0, "fixed_schema": True,
         "budget": "primary", "models": ["spline"], "sparse": False},
        {"bank": primary, "sharing": "pooled", "k": 1, "fixed_schema": True,
         "budget": "primary", "models": ["spline"], "sparse": False},
        {"bank": primary, "sharing": "context_specific", "k": 0, "fixed_schema": False,
         "budget": "tight", "models": ["spline", "transformer"], "sparse": False},
        {"bank": primary, "sharing": "pooled", "k": 1, "fixed_schema": False,
         "budget": "tight", "models": ["spline", "transformer"], "sparse": False},
        {"bank": "matched", "sharing": "pooled", "k": 1, "fixed_schema": False,
         "budget": "primary", "models": ["spline", "transformer"], "sparse": False},
        {"bank": "matched", "sharing": "context_specific", "k": 0, "fixed_schema": False,
         "budget": "primary", "models": ["spline", "transformer"], "sparse": False},
    ]
    # drop cells whose bank was not generated
    cells_spec = [c for c in cells_spec if c["bank"] in banks]
    records = []
    failures: list[str] = []
    for spec in cells_spec:
        cell_cfg = _cell_cfg(cfg, parent, bank=spec["bank"], sharing=spec["sharing"], k=spec["k"],
                             fixed_schema=spec["fixed_schema"], budget_name=spec["budget"],
                             models=spec["models"])
        if not spec.get("sparse"):
            tasks = [t for t in (cell_cfg.get("evaluation") or {}).get("tasks") or [] if t != "sparse_history"]
            cell_cfg.setdefault("evaluation", {})
            cell_cfg["evaluation"]["tasks"] = tasks or ["predose_conditioned"]
        run = _run_from_cfg(cell_cfg)
        try:
            run.ingest(force=force)
            run.prepare(force=force)
            run.train(force=force)
            run.evaluate(force=force)
            run.design(force=force)
            rec = _record_cell(run, spec)
            records.append(rec)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{spec}: {type(e).__name__}: {e}")
            records.append({**spec, "failed": True, "error": f"{type(e).__name__}: {e}"})

    transfer = _assemble_transfer(parent, records)
    write_json(parent / "transfer_table.json", transfer)
    summary = {
        "freeze_lock": lock,
        "labels": list(LABELS),
        "bank_manifest": bank_manifest,
        "cells": records,
        "transfer_table": transfer,
        "retuned_after_test": False,
        "failures": failures,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", summary)
    _write_results_md(parent, summary)
    return summary


def _run_from_cfg(cfg: dict[str, Any]) -> Run:
    path = Path(cfg["output_dir"]) / "effective.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return Run(path)


def _cell_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    metrics_path = Path(rec["dir"]) / "evaluation" / "metrics.json" if rec.get("dir") else None
    if not metrics_path or not metrics_path.exists():
        return {}
    return read_json(metrics_path)


def _assemble_transfer(parent: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {}
    for rec in records:
        if rec.get("failed"):
            continue
        key = (rec["bank"], rec["fixed_schema"], rec["budget"])
        by_key.setdefault(key, {})[rec["sharing"]] = rec
    tables: dict[str, Any] = {}
    for (bank, fixed, budget), regs in by_key.items():
        s0_rec = regs.get("context_specific")
        if not s0_rec:
            continue
        s0_all = _cell_metrics(s0_rec)
        for sharing, rec in regs.items():
            if sharing == "context_specific":
                continue
            sk_all = _cell_metrics(rec)
            for mkey, sk in sk_all.items():
                if not str(mkey).endswith("pre_study"):
                    continue
                s0 = s0_all.get(mkey)
                if not s0:
                    continue
                tag = f"{bank}|{'fix' if fixed else 'var'}|{budget}|{sharing}|{mkey}"
                tables[tag] = build_transfer_table(
                    s0, [(int(rec["k"]), {**sk, "sharing": sharing})],
                    model=sk.get("model"),
                    sharing_regimes=["context_specific (S0)", f"{sharing} (Sk)"],
                    selective_sharing="v1" if sharing == "selective" else "not_this_row",
                )
    return tables


def _write_results_md(parent: Path, summary: dict[str, Any]) -> None:
    lock = summary.get("freeze_lock") or {}
    lines = [
        "# v3 synthetic evaluation benchmark",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.**",
        "",
        "Signs are published as computed. The protocol was not retuned after target-test "
        "metrics. An unreliable Bayesian fit is not counted as a successful benchmark.",
        "",
        "## Freeze lock",
        "",
        f"- git revision: `{lock.get('git_revision')}`",
        f"- config hash: `{lock.get('config_hash')}`",
        "",
        "## Uncertainty (fixed language)",
        "",
        "- Predictive variability: HierarchicalCovariance draws in `sample_study`.",
        "- Parameter uncertainty is incompletely estimated except for a reliable Bayesian "
        "posterior. Spline bootstraps the mean only; covariance is not bootstrapped; "
        "transformer `n_members=1` has no bootstrap spread.",
        "",
        "## Cells",
        "",
        "| bank | sharing | k | schema | budget | models | bound | bayes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in summary.get("cells") or []:
        cov = rec.get("covariance") or {}
        bayes = rec.get("bayesian_gate")
        btxt = "n/a"
        if bayes:
            btxt = "reliable" if bayes.get("reliable") else "unreliable/not counted"
        lines.append(
            f"| {rec.get('bank')} | {rec.get('sharing')} | {rec.get('k')} | "
            f"{'fixed' if rec.get('fixed_schema') else 'variable'} | {rec.get('budget')} | "
            f"{','.join(rec.get('models') or [])} | {cov.get('params_at_bound')} | {btxt} |"
        )
    lines += ["", "## Transfer (predose, published as computed)", "",
              "| tag | model | metric | S0 | Sk | delta | sign |",
              "|---|---|---|---|---|---|---|"]
    for tag, block in (summary.get("transfer_table") or {}).items():
        for r in block.get("rows") or []:
            if r.get("metric") in ("rmse_log", "crps", "contrast_bias"):
                lines.append(
                    f"| {tag} | {r.get('model')} | {r.get('metric')} | {r.get('S0')} | "
                    f"{r.get('Sk')} | {r.get('delta')} | {r.get('sign')} |"
                )
    fails = summary.get("failures") or []
    lines += ["", "## Failures", "", *(fails or ["None"]), "",
              "## Artifact locations", "",
              f"- parent: `{parent}`",
              f"- freeze lock: `{parent / 'freeze_lock.json'}`",
              f"- banks: `{parent / 'banks'}`",
              f"- cells: `{parent / 'cells'}`",
              f"- transfer: `{parent / 'transfer_table.json'}`",
              ""]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

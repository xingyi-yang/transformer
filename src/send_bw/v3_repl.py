"""Frozen replicated v3 synthetic benchmark.

Preserves `runs/v3_eval` and `runs/v3_pilot_k0k1`. Dataset-generation seeds are
separated from model-training seeds. Selective v1 is retained; selective
regularized is added. Sparse target-study training is distinct from sparse
history encoders. Bayesian is executed with a reliability gate and matched-data
comparisons. Sharing is not required to improve. Signs are published.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import yaml

from send_bw.evaluation.transfer import build_transfer_table, paired_transfer_row, transfer_uncertainty
from send_bw.evaluation.uncertainty import bayesian_gate, inspect_covariance
from send_bw.pipeline import Run
from send_bw.sharing import LABELS, SELECTIVE_REGULARIZED, SELECTIVE_V1
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json

PRESERVED = ("runs/v3_eval", "runs/v3_pilot_k0k1")
REGIMES = (
    {"sharing": "context_specific", "k": 0},
    {"sharing": "pooled", "k": 1},
    {"sharing": "selective", "k": 1},
    {"sharing": "selective_regularized", "k": 1},
)


class ReplProtocolError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_repl_config(cfg: dict[str, Any]) -> None:
    ev = cfg.get("repl") or {}
    if ev.get("retune_after_test", False):
        raise ReplProtocolError("retune_after_test is forbidden")
    if (cfg.get("design") or {}).get("request"):
        raise ReplProtocolError("design operations are off for this evaluation")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise ReplProtocolError("real-data access is off")
    opt = (cfg.get("models") or {}).get("optional") or {}
    for name in ("tft", "chronos2", "deepar"):
        if (opt.get(name) or {}).get("enabled"):
            raise ReplProtocolError(f"{name} is off")
    tasks = list((cfg.get("evaluation") or {}).get("tasks") or [])
    if "predose_conditioned" not in tasks:
        raise ReplProtocolError("evaluation.tasks must include predose_conditioned")
    if int(ev.get("n_replicates") or 0) < 1:
        raise ReplProtocolError("repl.n_replicates must be >= 1")
    if cfg.get("generate_seed") is None or cfg.get("train_seed") is None:
        raise ReplProtocolError("generate_seed and train_seed must both be set")
    if int(cfg["generate_seed"]) == int(cfg["train_seed"]):
        raise ReplProtocolError("generate_seed and train_seed must differ")


def write_freeze_lock(parent: Path, cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "config_path": str(config_path),
        "config_hash": cache_key(cfg),
        "git_revision": git_revision(),
        "frozen_before": "generate_replicates_train_evaluate",
        "retune_after_test": False,
        "design": False,
        "real_data": False,
        "preserved_artifacts": list(PRESERVED),
        "generate_seed": int(cfg["generate_seed"]),
        "train_seed": int(cfg["train_seed"]),
        "seed_separation": {
            "generate_seed": "dataset generation only; replicate i uses generate_seed + i * 1009",
            "train_seed": "splits and model training only; replicate i uses train_seed + i * 17",
        },
        "selective_sharing": SELECTIVE_V1,
        "selective_regularized": SELECTIVE_REGULARIZED,
        "sparse_target_train": {
            "kind": "sparse_target_study_training",
            "not": "sparse_history_encoder",
            "n_keep": (cfg.get("repl") or {}).get("sparse_target_train_n"),
        },
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
        "sharing_not_required_to_improve": True,
    }
    write_json(parent / "freeze_lock.json", lock)
    return lock


def _run_from_cfg(cfg: dict[str, Any]) -> Run:
    path = Path(cfg["output_dir"]) / "effective.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return Run(path)


def _bank_cfg(base: dict[str, Any], parent: Path, replicate: int, generate_seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    bank = str((base.get("repl") or {}).get("bank") or "both")
    out = parent / "replicates" / f"r{replicate}" / "bank"
    cfg["run_name"] = f"v3_repl_r{replicate}_bank"
    cfg["output_dir"] = str(out)
    cfg["seed"] = int(generate_seed)
    cfg["generate_seed"] = int(generate_seed)
    cfg["train_seed"] = int(base["train_seed"])
    cfg["synthetic"] = dict(cfg.get("synthetic") or {})
    cfg["synthetic"]["enabled"] = True
    cfg["synthetic"]["out_dir"] = str(out / "synthetic")
    cfg["synthetic"]["donor_scenario"] = bank
    cfg["source"] = dict(cfg.get("source") or {})
    cfg["source"]["type"] = "local"
    cfg["source"]["root"] = str(out / "synthetic" / "send")
    cfg["source"]["metadata_dir"] = str(out / "synthetic" / "metadata")
    cfg["design"] = {}
    return cfg


def _cell_cfg(base: dict[str, Any], parent: Path, spec: dict[str, Any], *,
              generate_seed: int, train_seed: int, models: list[str],
              sparse_target_train_n: int | None, sparse_history: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    name = spec["name"]
    bank_syn = parent / "replicates" / f"r{spec['replicate']}" / "bank" / "synthetic"
    cell = parent / "replicates" / f"r{spec['replicate']}" / "cells" / name
    cfg["run_name"] = f"v3_repl_{name}"
    cfg["output_dir"] = str(cell)
    cfg["seed"] = int(train_seed)
    cfg["generate_seed"] = int(generate_seed)
    cfg["train_seed"] = int(train_seed)
    cfg["synthetic"] = dict(cfg.get("synthetic") or {})
    cfg["synthetic"]["enabled"] = False
    cfg["synthetic"]["out_dir"] = str(bank_syn)
    cfg["synthetic"]["donor_scenario"] = spec["bank"]
    cfg["source"] = dict(cfg.get("source") or {})
    cfg["source"]["type"] = "local"
    cfg["source"]["root"] = str(bank_syn / "send")
    cfg["source"]["metadata_dir"] = str(bank_syn / "metadata")
    cfg["cohort"] = dict(cfg.get("cohort") or {})
    cfg["cohort"]["sharing"] = spec["sharing"]
    cfg["cohort"]["n_donor_studies"] = int(spec["k"])
    cfg["splits"] = dict(cfg.get("splits") or {})
    if sparse_target_train_n is not None:
        cfg["splits"]["sparse_target_train_n"] = int(sparse_target_train_n)
    else:
        cfg["splits"].pop("sparse_target_train_n", None)
    cfg["design"] = {}
    budgets = ((cfg.get("repl") or {}).get("budgets") or {}).get("primary") or {}
    models_cfg = dict(cfg.get("models") or {})
    for fam in ("spline", "transformer", "bayesian"):
        models_cfg[fam] = dict(models_cfg.get(fam) or {})
        models_cfg[fam]["enabled"] = fam in models
        models_cfg[fam]["seed"] = int(train_seed)
        if fam in budgets:
            models_cfg[fam].update(budgets[fam])
            models_cfg[fam]["seed"] = int(train_seed)
    opt = dict(models_cfg.get("optional") or {})
    for extra in ("tft", "chronos2", "deepar"):
        opt[extra] = {"enabled": False}
    models_cfg["optional"] = opt
    cfg["models"] = models_cfg
    tasks = ["predose_conditioned"]
    if sparse_history:
        tasks.append("sparse_history")
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["tasks"] = tasks
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
    if "bayesian" in (spec.get("models") or []):
        if not train_b or train_b.get("status") in ("not_run", "failed"):
            bayes = bayesian_gate({}, status=train_b.get("status") or "not_run",
                                  reason=train_b.get("reason") or "Bayesian was not run")
        else:
            bayes = bayesian_gate({
                "reliable": train_b.get("reliable"),
                **(train_b.get("diagnostics_summary") or {}),
            }, status="trained")
        write_json(run.p("evaluation", "bayesian_gate.json"), bayes)
    tman = read_json(run.p("splits", "target_manifest.json")) if run.p("splits", "target_manifest.json").exists() else {}
    sparse_meta = None
    if run.p("splits", "sparse_target_train.json").exists():
        sparse_meta = read_json(run.p("splits", "sparse_target_train.json"))
    packs = {}
    for name in spec.get("models") or []:
        p = run.p("evaluation", f"{name}_test_evaluation_pack.json")
        if p.exists():
            packs[name] = read_json(p)
    metrics = read_json(run.p("evaluation", "metrics.json")) if run.p("evaluation", "metrics.json").exists() else {}
    return {
        **spec,
        "dir": str(run.dir),
        "target_manifest": {k: (v or {}).get("hash") for k, v in tman.items() if isinstance(v, dict)},
        "sparse_target_train": sparse_meta,
        "covariance": cov_insp,
        "bayesian_gate": bayes,
        "matched_data": train_b.get("matched_data"),
        "selective_sharing": spline_hist.get("selective_sharing"),
        "evaluation_packs": {k: {"status": v.get("status"), "individual": v.get("individual"),
                                 "group_mean": v.get("group_mean"), "contrast": v.get("contrast")}
                             for k, v in packs.items()},
        "metrics_keys": list(metrics),
        "warnings": (read_json(run.p("qc", "qc_report.json")).get("limitations")
                     if run.p("qc", "qc_report.json").exists() else []),
    }


def _cell_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    metrics_path = Path(rec["dir"]) / "evaluation" / "metrics.json" if rec.get("dir") else None
    if not metrics_path or not metrics_path.exists():
        return {}
    return read_json(metrics_path)


def _assemble_transfer(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tables: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    for rec in records:
        if rec.get("failed"):
            continue
        key = (rec["replicate"], rec["bank"], rec.get("sparse_target_train_n"))
        by_key.setdefault(key, {})[rec["sharing"]] = rec
    for (rep, bank, stt), regs in by_key.items():
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
                tag = f"r{rep}|{bank}|stt{stt}|{sharing}|{mkey}"
                sel = "regularized" if sharing == "selective_regularized" else (
                    "v1" if sharing == "selective" else "not_this_row")
                tables[tag] = build_transfer_table(
                    s0, [(int(rec["k"]), {**sk, "sharing": sharing})],
                    model=sk.get("model"),
                    sharing_regimes=["context_specific (S0)", f"{sharing} (Sk)"],
                    selective_sharing=sel,
                )
                for metric in ("rmse_log", "crps", "contrast_bias"):
                    row = paired_transfer_row(s0, sk, metric, int(rec["k"]), model=sk.get("model"))
                    split = str(mkey).split("|")[1] if "|" in str(mkey) else sk.get("split")
                    row.update({
                        "replicate": int(rep),
                        "bank": bank,
                        "sharing": sharing,
                        "split": split,
                        "sparse_target_train": "full" if stt is None else stt,
                        "generate_seed": rec.get("generate_seed"),
                        "train_seed": rec.get("train_seed"),
                        "metric_key": mkey,
                    })
                    pair_rows.append(row)
    return tables, pair_rows


def _component_status(records: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    def _bayes_label(rec: dict[str, Any]) -> str:
        g = rec.get("bayesian_gate")
        if not g:
            return "not_run"
        if g.get("status") in ("not_run", "failed"):
            return str(g.get("status"))
        if g.get("reliable"):
            return "completed"
        return "unreliable"

    bayes_cells = [r for r in records if "bayesian" in (r.get("models") or []) and not r.get("failed")]
    bayes_states = [_bayes_label(r) for r in bayes_cells]
    if any(s == "completed" for s in bayes_states):
        bayes_overall = "completed"
    elif any(s == "unreliable" for s in bayes_states):
        bayes_overall = "unreliable"
    elif any(s == "failed" for s in bayes_states):
        bayes_overall = "failed"
    else:
        bayes_overall = "not_run"
    packs_ok = any((r.get("evaluation_packs") or {}) and not r.get("failed") for r in records)
    stt_ok = any(r.get("sparse_target_train") and not r.get("failed") for r in records)
    return {
        "selective_v1": "completed" if any(r.get("sharing") == "selective" and not r.get("failed") for r in records)
        else "not_run",
        "selective_regularized": (
            "completed" if any(r.get("sharing") == "selective_regularized" and not r.get("failed") for r in records)
            else "not_run"),
        "sparse_target_study_training": "completed" if stt_ok else "not_run",
        "sparse_history_encoder": (
            "completed" if any(r.get("sparse_history") and not r.get("failed") for r in records) else "not_run"),
        "evaluation_pack": "completed" if packs_ok else "not_run",
        "bayesian": bayes_overall,
        "bayesian_cells": [
            {"name": r.get("name"), "status": _bayes_label(r),
             "reliable": bool((r.get("bayesian_gate") or {}).get("reliable")),
             "counted_as_successful_benchmark": bool((r.get("bayesian_gate") or {}).get("counted_as_successful_benchmark")),
             "reason": (r.get("bayesian_gate") or {}).get("reason")}
            for r in bayes_cells
        ],
        "replicated_transfer_uncertainty": "completed",
        "correctness_failures": list(failures),
        "note": (
            "Unfavorable transfer is a published result, not a failure. "
            "Unreliable Bayesian fits are not counted as a successful benchmark."
        ),
    }


def execute_v3_repl(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    validate_repl_config(cfg)
    parent = Path(cfg.get("output_dir") or "runs/v3_repl")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise ReplProtocolError("refusing to write into a preserved artifact directory")
    ev = cfg.get("repl") or {}
    t0 = time.time()
    lock = write_freeze_lock(parent, cfg, config_path)
    n_rep = int(ev.get("n_replicates") or 2)
    bank = str(ev.get("bank") or "both")
    gen0 = int(cfg["generate_seed"])
    tr0 = int(cfg["train_seed"])
    stt_n = ev.get("sparse_target_train_n")
    stt_rep = int(ev.get("sparse_target_train_on_replicate", 0))
    bayes_rep = int(ev.get("bayesian_on_replicate", 0))
    bayes_regimes = list(ev.get("bayesian_regimes") or ["context_specific", "pooled", "selective_regularized"])
    hist_rep = int(ev.get("sparse_history_on_replicate", 0))

    bank_manifest: dict[str, Any] = {}
    for rep in range(n_rep):
        gen_seed = gen0 + rep * 1009
        bcfg = _bank_cfg(cfg, parent, rep, gen_seed)
        run = _run_from_cfg(bcfg)
        run.generate(force=force)
        truth = read_json(Path(bcfg["synthetic"]["out_dir"]) / "truth" / "params.json")
        bank_manifest[f"r{rep}"] = {
            "synthetic_dir": bcfg["synthetic"]["out_dir"],
            "donor_scenario": bank,
            "donor_deviations": truth.get("donor_deviations"),
            "generate_seed": gen_seed,
            "train_seed": tr0 + rep * 17,
        }
    write_json(parent / "bank_manifest.json", bank_manifest)

    cells_spec: list[dict[str, Any]] = []
    for rep in range(n_rep):
        for regime in REGIMES:
            cells_spec.append({
                "name": f"r{rep}_{bank}_{regime['sharing']}_k{regime['k']}",
                "replicate": rep, "bank": bank, **regime,
                "sparse_target_train_n": None,
                "sparse_history": rep == hist_rep and regime["sharing"] == "context_specific",
                "generate_seed": gen0 + rep * 1009,
                "train_seed": tr0 + rep * 17,
                "models": ["spline", "transformer"] + (
                    ["bayesian"] if rep == bayes_rep and regime["sharing"] in bayes_regimes else []
                ),
            })
        if stt_n is not None and rep == stt_rep:
            for sharing in ("context_specific", "selective_regularized"):
                k = 0 if sharing == "context_specific" else 1
                cells_spec.append({
                    "name": f"r{rep}_{bank}_{sharing}_k{k}_stt{int(stt_n)}",
                    "replicate": rep, "bank": bank, "sharing": sharing, "k": k,
                    "sparse_target_train_n": int(stt_n),
                    "sparse_history": False,
                    "generate_seed": gen0 + rep * 1009,
                    "train_seed": tr0 + rep * 17,
                    "models": ["spline", "transformer"],
                })

    records = []
    failures: list[str] = []
    for spec in cells_spec:
        cell_cfg = _cell_cfg(
            cfg, parent, spec,
            generate_seed=spec["generate_seed"], train_seed=spec["train_seed"],
            models=spec["models"], sparse_target_train_n=spec.get("sparse_target_train_n"),
            sparse_history=bool(spec.get("sparse_history")),
        )
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
            failures.append(f"{spec['name']}: {type(e).__name__}: {e}")
            records.append({**spec, "failed": True, "error": f"{type(e).__name__}: {e}"})

    transfer, pair_rows = _assemble_transfer(records)
    write_json(parent / "transfer_table.json", transfer)
    unc = transfer_uncertainty(pair_rows)
    write_json(parent / "transfer_uncertainty.json", unc)
    write_json(parent / "paired_transfer_rows.json", pair_rows)
    components = _component_status(records, failures)
    write_json(parent / "component_status.json", components)
    summary = {
        "freeze_lock": lock,
        "labels": list(LABELS),
        "bank_manifest": bank_manifest,
        "cells": records,
        "transfer_table": transfer,
        "transfer_uncertainty": unc,
        "component_status": components,
        "retuned_after_test": False,
        "sharing_required_to_improve": False,
        "failures": failures,
        "seconds_total": time.time() - t0,
    }
    write_json(parent / "SUMMARY.json", summary)
    _write_results_md(parent, summary)
    return summary


def _write_results_md(parent: Path, summary: dict[str, Any]) -> None:
    lock = summary.get("freeze_lock") or {}
    comp = summary.get("component_status") or {}
    lines = [
        "# v3 replicated synthetic benchmark",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.**",
        "",
        "Signs are published as computed. The protocol was not retuned after target-test "
        "metrics. Sharing is not required to improve. An unreliable Bayesian fit is not "
        "counted as a successful benchmark. Prior `runs/v3_eval` and `runs/v3_pilot_k0k1` "
        "artifacts were not modified.",
        "",
        "## Freeze lock",
        "",
        f"- git revision: `{lock.get('git_revision')}`",
        f"- config hash: `{lock.get('config_hash')}`",
        f"- generate_seed: `{lock.get('generate_seed')}` (dataset only)",
        f"- train_seed: `{lock.get('train_seed')}` (splits and models only)",
        "",
        "## Component status",
        "",
        "| component | status |",
        "|---|---|",
    ]
    for k in ("selective_v1", "selective_regularized", "sparse_target_study_training",
              "sparse_history_encoder", "evaluation_pack", "bayesian",
              "replicated_transfer_uncertainty"):
        lines.append(f"| {k} | {comp.get(k, 'not_run')} |")
    lines += [
        "",
        "### Bayesian cells",
        "",
        "| cell | status | counted | reason |",
        "|---|---|---|---|",
    ]
    for b in comp.get("bayesian_cells") or []:
        lines.append(
            f"| {b.get('name')} | {b.get('status')} | {b.get('counted_as_successful_benchmark')} | "
            f"{b.get('reason')} |"
        )
    if not (comp.get("bayesian_cells") or []):
        lines.append("| — | not_run | False | no Bayesian cell |")
    lines += [
        "",
        "## Uncertainty (fixed language)",
        "",
        "- Predictive variability: HierarchicalCovariance draws in `sample_study`.",
        "- Parameter uncertainty is incompletely estimated except for a reliable Bayesian "
        "posterior. Spline bootstraps the mean only; covariance is not bootstrapped; "
        "transformer `n_members=1` has no bootstrap spread.",
        "- Replicate uncertainty is the spread of paired Δ across generate/train seed pairs. "
        "It is not a license to pick the favorable seed.",
        "",
        "## Cells",
        "",
        "| cell | sharing | k | stt | models | bayes |",
        "|---|---|---|---|---|---|",
    ]
    for rec in summary.get("cells") or []:
        bayes = rec.get("bayesian_gate")
        btxt = "n/a"
        if rec.get("failed"):
            btxt = f"failed ({rec.get('error')})"
        elif bayes:
            if bayes.get("status") in ("not_run", "failed"):
                btxt = f"{bayes.get('status')} ({bayes.get('reason')})"
            elif bayes.get("reliable"):
                btxt = "reliable"
            else:
                btxt = "unreliable/not counted"
        lines.append(
            f"| {rec.get('name')} | {rec.get('sharing')} | {rec.get('k')} | "
            f"{rec.get('sparse_target_train_n')} | {','.join(rec.get('models') or [])} | {btxt} |"
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
    lines += ["", "## Transfer uncertainty across replicates", "",
              "| model | sharing | metric | n | mean Δ | sd Δ | harmful | helpful |",
              "|---|---|---|---|---|---|---|---|"]
    for g in (summary.get("transfer_uncertainty") or {}).get("groups") or []:
        key = g.get("key") or {}
        if key.get("sparse_target_train") != "full":
            continue
        if key.get("split") not in (None, "test"):
            continue
        if key.get("metric") not in ("rmse_log", "crps", "contrast_bias"):
            continue
        lines.append(
            f"| {key.get('model')} | {key.get('sharing')} | {key.get('metric')} | {g.get('n')} | "
            f"{g.get('mean_delta')} | {g.get('sd_delta')} | {g.get('n_harmful')} | {g.get('n_helpful')} |"
        )
    fails = summary.get("failures") or []
    lines += ["", "## Correctness / inference failures", "", *(fails or ["None"]), "",
              "## Artifact locations", "",
              f"- parent: `{parent}`",
              f"- freeze lock: `{parent / 'freeze_lock.json'}`",
              f"- component status: `{parent / 'component_status.json'}`",
              f"- transfer uncertainty: `{parent / 'transfer_uncertainty.json'}`",
              ""]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

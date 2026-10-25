"""Frozen v3 research evaluation protocol.

Preserves executed prototype artifacts. Nested dataset seeds and training seeds.
Target splits are frozen once per (dataset, train) pair and reused across sharing
and donor scenarios. Full-versus-sparse target-study learning is a primary result.
Sharing is not required to improve. Bayesian sampling is diagnosed separately
from compile/backend errors. Reliability thresholds are unchanged.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import yaml

from send_bw.evaluation.transfer import build_transfer_table, paired_transfer_row, transfer_uncertainty
from send_bw.evaluation.uncertainty import bayesian_gate, inspect_covariance, sampling_diagnosis
from send_bw.pipeline import Run
from send_bw.sharing import LABELS, SELECTIVE_REGULARIZED, SELECTIVE_V1
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json

PRESERVED = ("runs/v3_eval", "runs/v3_pilot_k0k1", "runs/v3_repl")
REGIMES = (
    {"sharing": "context_specific", "k": 0},
    {"sharing": "pooled", "k": 1},
    {"sharing": "selective", "k": 1},
    {"sharing": "selective_regularized", "k": 1},
)
BANKS = ("matched", "both")


class ResearchProtocolError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_research_config(cfg: dict[str, Any]) -> None:
    ev = cfg.get("research") or {}
    if ev.get("retune_after_test", False):
        raise ResearchProtocolError("retune_after_test is forbidden")
    if (cfg.get("design") or {}).get("request"):
        raise ResearchProtocolError("design operations are off for this evaluation")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise ResearchProtocolError("real-data access is off")
    opt = (cfg.get("models") or {}).get("optional") or {}
    for name in ("tft", "chronos2", "deepar"):
        if (opt.get(name) or {}).get("enabled"):
            raise ResearchProtocolError(f"{name} is off")
    if "predose_conditioned" not in list((cfg.get("evaluation") or {}).get("tasks") or []):
        raise ResearchProtocolError("evaluation.tasks must include predose_conditioned")
    if int(ev.get("n_dataset_seeds") or 0) < 1 or int(ev.get("n_train_seeds") or 0) < 1:
        raise ResearchProtocolError("n_dataset_seeds and n_train_seeds must be >= 1")
    if cfg.get("generate_seed") is None or cfg.get("train_seed") is None:
        raise ResearchProtocolError("generate_seed and train_seed must both be set")
    if int(cfg["generate_seed"]) == int(cfg["train_seed"]):
        raise ResearchProtocolError("generate_seed and train_seed must differ")
    banks = list(ev.get("banks") or BANKS)
    if any(b not in BANKS for b in banks):
        raise ResearchProtocolError(f"research.banks must be subset of {BANKS}")


def write_freeze_lock(parent: Path, cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    parent.mkdir(parents=True, exist_ok=True)
    ev = cfg.get("research") or {}
    lock = {
        "config_path": str(config_path),
        "config_hash": cache_key(cfg),
        "git_revision": git_revision(),
        "frozen_before": "generate_datasets_train_evaluate",
        "retune_after_test": False,
        "design": False,
        "real_data": False,
        "preserved_artifacts": list(PRESERVED),
        "generate_seed": int(cfg["generate_seed"]),
        "train_seed": int(cfg["train_seed"]),
        "seed_separation": {
            "generate_seed": "dataset generation only; dataset i uses generate_seed + i * 1009",
            "train_seed": "splits and model training only; train j uses train_seed + j * 17",
            "nested": "every dataset seed is crossed with every train seed",
        },
        "target_splits": "frozen once per (dataset, train) pair and reused across sharing and banks",
        "primary_result": "full_versus_sparse_target_study_training",
        "selective_sharing": SELECTIVE_V1,
        "selective_regularized": SELECTIVE_REGULARIZED,
        "reliability_thresholds": {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400},
        "labels": list(LABELS),
        "sharing_not_required_to_improve": True,
        "n_dataset_seeds": int(ev.get("n_dataset_seeds") or 1),
        "n_train_seeds": int(ev.get("n_train_seeds") or 1),
        "banks": list(ev.get("banks") or BANKS),
    }
    write_json(parent / "freeze_lock.json", lock)
    return lock


def _run_from_cfg(cfg: dict[str, Any]) -> Run:
    path = Path(cfg["output_dir"]) / "effective.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return Run(path)


def _bank_cfg(base: dict[str, Any], parent: Path, dataset_index: int, generate_seed: int,
              bank: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    out = parent / "datasets" / f"d{dataset_index}" / "banks" / bank
    cfg["run_name"] = f"v3_research_d{dataset_index}_{bank}"
    cfg["output_dir"] = str(out)
    cfg["seed"] = int(generate_seed)
    cfg["generate_seed"] = int(generate_seed)
    cfg["synthetic"] = dict(cfg.get("synthetic") or {})
    cfg["synthetic"]["enabled"] = True
    cfg["synthetic"]["out_dir"] = str(out / "synthetic")
    cfg["synthetic"]["donor_scenario"] = bank
    cfg["source"] = {"type": "local", "root": str(out / "synthetic" / "send"),
                     "metadata_dir": str(out / "synthetic" / "metadata")}
    cfg["design"] = {}
    return cfg


def _cell_cfg(base: dict[str, Any], parent: Path, spec: dict[str, Any], *,
              generate_seed: int, train_seed: int, models: list[str],
              sparse_target_train_n: int | None, reuse_target_from: str | None) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    name = spec["name"]
    bank_syn = parent / "datasets" / f"d{spec['dataset_index']}" / "banks" / spec["bank"] / "synthetic"
    cell = (parent / "datasets" / f"d{spec['dataset_index']}" / "trains" / f"t{spec['train_index']}"
            / "cells" / name)
    cfg["run_name"] = f"v3_research_{name}"
    cfg["output_dir"] = str(cell)
    cfg["seed"] = int(train_seed)
    cfg["generate_seed"] = int(generate_seed)
    cfg["train_seed"] = int(train_seed)
    cfg["synthetic"] = dict(cfg.get("synthetic") or {})
    cfg["synthetic"]["enabled"] = False
    cfg["synthetic"]["out_dir"] = str(bank_syn)
    cfg["synthetic"]["donor_scenario"] = spec["bank"]
    cfg["source"] = {"type": "local", "root": str(bank_syn / "send"),
                     "metadata_dir": str(bank_syn / "metadata")}
    cfg["cohort"] = dict(cfg.get("cohort") or {})
    cfg["cohort"]["sharing"] = spec["sharing"]
    cfg["cohort"]["n_donor_studies"] = int(spec["k"])
    cfg["splits"] = dict(cfg.get("splits") or {})
    if sparse_target_train_n is not None:
        cfg["splits"]["sparse_target_train_n"] = int(sparse_target_train_n)
    else:
        cfg["splits"].pop("sparse_target_train_n", None)
    if reuse_target_from:
        cfg["splits"]["reuse_target_from"] = reuse_target_from
    else:
        cfg["splits"].pop("reuse_target_from", None)
    cfg["design"] = {}
    budgets = ((cfg.get("research") or {}).get("budgets") or {}).get("primary") or {}
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
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["tasks"] = ["predose_conditioned"]
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
    samp = None
    train_b = man.get("stages", {}).get("train_bayesian") or {}
    if "bayesian" in (spec.get("models") or []):
        if not train_b or train_b.get("status") in ("not_run", "failed"):
            bayes = bayesian_gate({}, status=train_b.get("status") or "not_run",
                                  reason=train_b.get("reason") or "Bayesian was not run")
            samp = sampling_diagnosis({}, compile_error=train_b.get("reason"),
                                      status=train_b.get("status") or "not_run")
        else:
            bayes = bayesian_gate({
                "reliable": train_b.get("reliable"),
                **(train_b.get("diagnostics_summary") or {}),
            }, status="trained")
            samp = train_b.get("sampling_diagnosis") or sampling_diagnosis(
                read_json(run.p("models", "bayesian", "diagnostics.json")).get("main")
                if run.p("models", "bayesian", "diagnostics.json").exists() else {})
        write_json(run.p("evaluation", "bayesian_gate.json"), bayes)
        if samp:
            write_json(run.p("evaluation", "sampling_diagnosis.json"), samp)
    tman = read_json(run.p("splits", "target_manifest.json")) if run.p("splits", "target_manifest.json").exists() else {}
    sparse_meta = (read_json(run.p("splits", "sparse_target_train.json"))
                   if run.p("splits", "sparse_target_train.json").exists() else None)
    packs = {}
    for name in spec.get("models") or []:
        p = run.p("evaluation", f"{name}_test_evaluation_pack.json")
        if p.exists():
            packs[name] = read_json(p)
    return {
        **spec,
        "dir": str(run.dir),
        "target_manifest": {k: (v or {}).get("hash") for k, v in tman.items() if isinstance(v, dict)},
        "sparse_target_train": sparse_meta,
        "covariance": cov_insp,
        "bayesian_gate": bayes,
        "sampling_diagnosis": samp,
        "matched_data": train_b.get("matched_data"),
        "selective_sharing": spline_hist.get("selective_sharing"),
        "evaluation_packs": {
            k: {"status": v.get("status"), "individual": v.get("individual"),
                "group_mean": v.get("group_mean"), "contrast": v.get("contrast"),
                "denominators": v.get("denominators")}
            for k, v in packs.items()},
    }


def _cell_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    p = Path(rec["dir"]) / "evaluation" / "metrics.json" if rec.get("dir") else None
    if not p or not p.exists():
        return {}
    return read_json(p)


def _prestudy_test(metrics: dict[str, Any], model: str) -> dict[str, Any] | None:
    for k, v in metrics.items():
        if k == f"{model}|test|pre_study":
            return v
    return None


def _assemble(records: list[dict[str, Any]]) -> dict[str, Any]:
    share_tables: dict[str, Any] = {}
    share_rows: list[dict[str, Any]] = []
    dens_tables: dict[str, Any] = {}
    dens_rows: list[dict[str, Any]] = []
    by_share: dict[tuple, dict[str, dict]] = {}
    by_dens: dict[tuple, dict[str, dict]] = {}
    for rec in records:
        if rec.get("failed"):
            continue
        dens = rec.get("sparse_target_train_n")
        by_share.setdefault((rec["dataset_index"], rec["train_index"], rec["bank"], dens), {})[rec["sharing"]] = rec
        by_dens.setdefault((rec["dataset_index"], rec["train_index"], rec["bank"], rec["sharing"]), {})[
            "sparse" if dens else "full"] = rec
    for key, regs in by_share.items():
        s0 = regs.get("context_specific")
        if not s0:
            continue
        s0m = _cell_metrics(s0)
        for sharing, rec in regs.items():
            if sharing == "context_specific":
                continue
            skm = _cell_metrics(rec)
            for model in ("spline", "transformer", "bayesian"):
                s0s, sks = _prestudy_test(s0m, model), _prestudy_test(skm, model)
                if not s0s or not sks:
                    continue
                tag = f"d{key[0]}|t{key[1]}|{key[2]}|stt{key[3]}|{sharing}|{model}|test"
                sel = ("regularized" if sharing == "selective_regularized"
                       else "v1" if sharing == "selective" else "not_this_row")
                share_tables[tag] = build_transfer_table(
                    s0s, [(int(rec["k"]), {**sks, "sharing": sharing})], model=model,
                    sharing_regimes=["context_specific (S0)", f"{sharing} (Sk)"],
                    selective_sharing=sel)
                for metric in ("rmse_log", "crps", "contrast_bias"):
                    row = paired_transfer_row(s0s, sks, metric, int(rec["k"]), model=model)
                    row.update({
                        "dataset_index": key[0], "train_index": key[1], "bank": key[2],
                        "sharing": sharing, "split": "test",
                        "sparse_target_train": "full" if key[3] is None else key[3],
                        "generate_seed": rec.get("generate_seed"),
                        "train_seed": rec.get("train_seed"),
                        "contrast": "sharing",
                    })
                    share_rows.append(row)
    for key, dens in by_dens.items():
        full, sparse = dens.get("full"), dens.get("sparse")
        if not full or not sparse:
            continue
        fm, sm = _cell_metrics(full), _cell_metrics(sparse)
        for model in ("spline", "transformer", "bayesian"):
            fs, ss = _prestudy_test(fm, model), _prestudy_test(sm, model)
            if not fs or not ss:
                continue
            tag = f"d{key[0]}|t{key[1]}|{key[2]}|{key[3]}|{model}|full_vs_sparse|test"
            dens_tables[tag] = build_transfer_table(
                fs, [(int(sparse.get("sparse_target_train_n") or 0), {**ss, "sharing": "sparse_target_train"})],
                model=model, sharing_regimes=["full_target_train (S0)", "sparse_target_train (Sk)"],
                selective_sharing="not_this_row")
            for metric in ("rmse_log", "crps", "contrast_bias"):
                row = paired_transfer_row(fs, ss, metric, int(sparse.get("sparse_target_train_n") or 0), model=model)
                row.update({
                    "dataset_index": key[0], "train_index": key[1], "bank": key[2],
                    "sharing": key[3], "split": "test",
                    "sparse_target_train": sparse.get("sparse_target_train_n"),
                    "generate_seed": sparse.get("generate_seed"),
                    "train_seed": sparse.get("train_seed"),
                    "contrast": "full_versus_sparse",
                })
                dens_rows.append(row)
    return {
        "sharing_transfer": share_tables,
        "density_transfer": dens_tables,
        "sharing_rows": share_rows,
        "density_rows": dens_rows,
        "sharing_uncertainty": transfer_uncertainty(share_rows),
        "density_uncertainty": transfer_uncertainty(dens_rows),
    }


def _component_status(records: list[dict[str, Any]], failures: list[str], assembled: dict[str, Any]) -> dict[str, Any]:
    def _bayes(rec):
        g = rec.get("bayesian_gate") or {}
        if g.get("status") in ("not_run", "failed"):
            return g.get("status")
        if g.get("reliable"):
            return "completed"
        return "unreliable"

    bayes = [r for r in records if "bayesian" in (r.get("models") or []) and not r.get("failed")]
    states = [_bayes(r) for r in bayes]
    if any(s == "completed" for s in states):
        bstat = "completed"
    elif any(s == "unreliable" for s in states):
        bstat = "unreliable"
    elif any(s == "failed" for s in states):
        bstat = "failed"
    else:
        bstat = "not_run"
    return {
        "selective_v1": "completed" if any(r.get("sharing") == "selective" and not r.get("failed") for r in records) else "not_run",
        "selective_regularized": "completed" if any(r.get("sharing") == "selective_regularized" and not r.get("failed") for r in records) else "not_run",
        "sparse_target_study_training": "completed" if any(r.get("sparse_target_train") and not r.get("failed") for r in records) else "not_run",
        "full_versus_sparse_primary": "completed" if assembled.get("density_rows") else "not_run",
        "matched_and_conflicting_donors": "completed" if {r.get("bank") for r in records if not r.get("failed")} >= {"matched", "both"} else "not_run",
        "fixed_target_splits": "completed",
        "evaluation_pack": "completed" if any(r.get("evaluation_packs") for r in records if not r.get("failed")) else "not_run",
        "monte_carlo_uncertainty": "completed" if assembled.get("sharing_rows") else "not_run",
        "bayesian": bstat,
        "bayesian_cells": [
            {"name": r.get("name"), "status": _bayes(r),
             "failure_class": (r.get("sampling_diagnosis") or {}).get("failure_class"),
             "counted_as_successful_benchmark": bool((r.get("bayesian_gate") or {}).get("counted_as_successful_benchmark")),
             "reason": (r.get("bayesian_gate") or {}).get("reason")}
            for r in bayes
        ],
        "correctness_failures": list(failures),
        "note": "Unfavorable transfer is a published result, not a failure.",
    }


def execute_v3_research(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    validate_research_config(cfg)
    parent = Path(cfg.get("output_dir") or "runs/v3_research")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise ResearchProtocolError("refusing to write into a preserved artifact directory")
    ev = cfg.get("research") or {}
    t0 = time.time()
    lock = write_freeze_lock(parent, cfg, config_path)
    n_ds = int(ev.get("n_dataset_seeds") or 2)
    n_ts = int(ev.get("n_train_seeds") or 2)
    banks = list(ev.get("banks") or BANKS)
    gen0 = int(cfg["generate_seed"])
    tr0 = int(cfg["train_seed"])
    stt_n = ev.get("sparse_target_train_n")
    bayes_on = ev.get("bayesian_on") or {}
    bank_manifest: dict[str, Any] = {}
    for i in range(n_ds):
        gen_seed = gen0 + i * 1009
        for bank in banks:
            bcfg = _bank_cfg(cfg, parent, i, gen_seed, bank)
            _run_from_cfg(bcfg).generate(force=force)
            truth = read_json(Path(bcfg["synthetic"]["out_dir"]) / "truth" / "params.json")
            bank_manifest[f"d{i}_{bank}"] = {
                "synthetic_dir": bcfg["synthetic"]["out_dir"],
                "donor_scenario": bank,
                "donor_deviations": truth.get("donor_deviations"),
                "generate_seed": gen_seed,
            }
    write_json(parent / "bank_manifest.json", bank_manifest)

    records = []
    failures: list[str] = []
    split_hashes: dict[tuple, dict[str, str]] = {}
    for i in range(n_ds):
        gen_seed = gen0 + i * 1009
        for j in range(n_ts):
            train_seed = tr0 + j * 17
            reuse = None
            for bank in banks:
                for regime in REGIMES:
                    for dens in (None, stt_n):
                        if dens is None and stt_n is None:
                            pass
                        want_bayes = (
                            i == int(bayes_on.get("dataset_index", -1))
                            and j == int(bayes_on.get("train_index", -1))
                            and bank in list(bayes_on.get("banks") or [])
                            and regime["sharing"] in list(bayes_on.get("sharings") or [])
                            and dens is None
                        )
                        dens_tag = "full" if dens is None else f"stt{int(dens)}"
                        spec = {
                            "name": f"d{i}_t{j}_{bank}_{regime['sharing']}_k{regime['k']}_{dens_tag}",
                            "dataset_index": i, "train_index": j, "bank": bank,
                            **regime, "sparse_target_train_n": None if dens is None else int(dens),
                            "generate_seed": gen_seed, "train_seed": train_seed,
                            "models": ["spline", "transformer"] + (["bayesian"] if want_bayes else []),
                        }
                        cell_cfg = _cell_cfg(
                            cfg, parent, spec, generate_seed=gen_seed, train_seed=train_seed,
                            models=spec["models"], sparse_target_train_n=spec["sparse_target_train_n"],
                            reuse_target_from=reuse)
                        run = _run_from_cfg(cell_cfg)
                        try:
                            run.ingest(force=force)
                            run.prepare(force=force)
                            run.train(force=force)
                            run.evaluate(force=force)
                            run.design(force=force)
                            rec = _record_cell(run, spec)
                            records.append(rec)
                            h = rec.get("target_manifest") or {}
                            key = (i, j)
                            if key not in split_hashes:
                                split_hashes[key] = {n: h.get(n) for n in ("validation", "test", "all_target")}
                            elif any(split_hashes[key].get(n) != h.get(n) for n in ("validation", "test", "all_target")):
                                raise ResearchProtocolError(
                                    f"target splits drifted for dataset {i} train {j} cell {spec['name']}")
                            if reuse is None:
                                reuse = str(run.p("splits", "x").parent)
                        except Exception as e:  # noqa: BLE001
                            failures.append(f"{spec['name']}: {type(e).__name__}: {e}")
                            records.append({**spec, "failed": True, "error": f"{type(e).__name__}: {e}"})
                            if reuse is None and Path(run.dir, "splits", "target_manifest.json").exists():
                                reuse = str(run.p("splits", "x").parent)

    assembled = _assemble(records)
    write_json(parent / "sharing_transfer.json", assembled["sharing_transfer"])
    write_json(parent / "density_transfer.json", assembled["density_transfer"])
    write_json(parent / "sharing_uncertainty.json", assembled["sharing_uncertainty"])
    write_json(parent / "density_uncertainty.json", assembled["density_uncertainty"])
    write_json(parent / "paired_rows.json", {
        "sharing": assembled["sharing_rows"], "density": assembled["density_rows"]})
    write_json(parent / "locked_target_splits.json", {f"d{i}_t{j}": h for (i, j), h in split_hashes.items()})
    components = _component_status(records, failures, assembled)
    write_json(parent / "component_status.json", components)
    summary = {
        "freeze_lock": lock,
        "labels": list(LABELS),
        "bank_manifest": bank_manifest,
        "cells": records,
        "sharing_transfer": assembled["sharing_transfer"],
        "density_transfer": assembled["density_transfer"],
        "sharing_uncertainty": assembled["sharing_uncertainty"],
        "density_uncertainty": assembled["density_uncertainty"],
        "locked_target_splits": {f"d{i}_t{j}": h for (i, j), h in split_hashes.items()},
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
        "# v3 research evaluation protocol",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data validation. "
        "No operational recommendation.**",
        "",
        "Signs are published as computed. Sharing is not required to improve. "
        "Full-versus-sparse target-study training is a primary result. "
        "Prior prototype artifacts were not modified.",
        "",
        "## Freeze lock",
        "",
        f"- git revision: `{lock.get('git_revision')}`",
        f"- config hash: `{lock.get('config_hash')}`",
        f"- generate_seed: `{lock.get('generate_seed')}`",
        f"- train_seed: `{lock.get('train_seed')}`",
        "",
        "## Component status",
        "",
        "| component | status |",
        "|---|---|",
    ]
    for k in ("selective_v1", "selective_regularized", "sparse_target_study_training",
              "full_versus_sparse_primary", "matched_and_conflicting_donors",
              "fixed_target_splits", "evaluation_pack", "monte_carlo_uncertainty", "bayesian"):
        lines.append(f"| {k} | {comp.get(k, 'not_run')} |")
    lines += ["", "### Bayesian sampling diagnosis", "",
              "| cell | status | class | counted |", "|---|---|---|---|"]
    for b in comp.get("bayesian_cells") or []:
        lines.append(f"| {b.get('name')} | {b.get('status')} | {b.get('failure_class')} | "
                     f"{b.get('counted_as_successful_benchmark')} |")
    if not (comp.get("bayesian_cells") or []):
        lines.append("| — | not_run | not_run | False |")
    lines += ["", "## Primary: full versus sparse target-study training (frozen test)", "",
              "| model | sharing | bank | n | mean Δ RMSE | sd Δ | harmful | helpful |",
              "|---|---|---|---|---|---|---|---|"]
    for g in (summary.get("density_uncertainty") or {}).get("groups") or []:
        key = g.get("key") or {}
        if key.get("metric") != "rmse_log" or key.get("split") not in (None, "test"):
            continue
        lines.append(
            f"| {key.get('model')} | {key.get('sharing')} | {key.get('bank', '')} | {g.get('n')} | "
            f"{g.get('mean_delta')} | {g.get('sd_delta')} | {g.get('n_harmful')} | {g.get('n_helpful')} |"
        )
    lines += ["", "## Sharing transfer (frozen test, full train)", "",
              "| model | sharing | bank | n | mean Δ RMSE | sd Δ | harmful | helpful |",
              "|---|---|---|---|---|---|---|---|"]
    for g in (summary.get("sharing_uncertainty") or {}).get("groups") or []:
        key = g.get("key") or {}
        if key.get("metric") != "rmse_log" or key.get("sparse_target_train") not in ("full", None):
            continue
        if key.get("split") not in (None, "test"):
            continue
        lines.append(
            f"| {key.get('model')} | {key.get('sharing')} | {key.get('bank', '')} | {g.get('n')} | "
            f"{g.get('mean_delta')} | {g.get('sd_delta')} | {g.get('n_harmful')} | {g.get('n_helpful')} |"
        )
    example = None
    for rec in summary.get("cells") or []:
        pack = (rec.get("evaluation_packs") or {}).get("spline")
        if pack and not rec.get("failed"):
            example = (rec, pack)
            break
    lines += ["", "## Calibration, widths, contrast, temporal (example pack)", ""]
    if example:
        rec, pack = example
        den = pack.get("denominators") or {}
        ind = pack.get("individual") or {}
        grp = pack.get("group_mean") or {}
        con = pack.get("contrast") or {}
        tmp = pack.get("temporal") or {}
        lines += [
            f"Cell `{rec.get('name')}` spline pack. Coverage and widths are predictive-interval "
            "diagnostics. They are not complete parameter-uncertainty statements.",
            "",
            f"- denominators: n_studies={den.get('n_studies')}, n_animals={den.get('n_animals')}, "
            f"n_obs={den.get('n_obs')}, n_contrasts={den.get('n_contrasts')}, "
            f"n_studies_with_contrast={den.get('n_studies_with_contrast')}, "
            f"n_group_mean_cells={den.get('n_group_mean_cells')}, n_gap_pairs={den.get('n_gap_pairs')}, "
            f"n_increments={den.get('n_increments')}, n_extrema={den.get('n_extrema')}",
            f"- weighting: RMSE/CRPS {den.get('rmse_crps_weighting')}; contrast bias "
            f"{den.get('contrast_bias_weighting')}; study-balanced contrast "
            f"{den.get('contrast_bias_study_balanced')}",
            f"- individual: cov80={ind.get('coverage80')} cov95={ind.get('coverage95')} "
            f"width80={ind.get('width80_log')} width95={ind.get('width95_log')}",
            f"- group-mean: cov90={grp.get('coverage90')} var_ratio="
            f"{grp.get('between_animal_var_ratio_pred_over_obs')}",
            f"- contrast: cov90={con.get('coverage90')} width90={con.get('width90')} "
            f"bias={con.get('bias')} n={con.get('n_contrasts')}",
            f"- temporal: increment_sd_ratio={tmp.get('increment_sd_ratio_pred_over_obs')} "
            f"extrema_max_cov90={tmp.get('extrema_max_cov90')}",
            "",
        ]
    else:
        lines += ["No completed evaluation pack was available.", ""]
    fails = summary.get("failures") or []
    lines += ["", "## Correctness / inference failures", "", *(fails or ["None"]), "",
              "## Artifact locations", "",
              f"- parent: `{parent}`",
              f"- density uncertainty: `{parent / 'density_uncertainty.json'}`",
              f"- sharing uncertainty: `{parent / 'sharing_uncertainty.json'}`",
              ""]
    (parent / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

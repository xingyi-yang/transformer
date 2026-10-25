"""Focused synthetic objective validation on fresh frozen datasets.

Development simulations may change covariance increment-moment scaling and
Bayesian identifiable-mean settings. Evaluation uses a new generate-seed
namespace. Prior run directories are not overwritten. Sharing and the
transformer are not required to win.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from send_bw.evaluation.increment import hierarchical_mean
from send_bw.evaluation.uncertainty import RELIABILITY_THRESHOLDS
from send_bw.historical_audit import discover_authorization
from send_bw.sharing import LABELS
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared, score_model_on_test
from send_bw.v3_research import (
    _assemble,
    _bank_cfg,
    _cell_cfg,
    _record_cell,
    _run_from_cfg,
    load_yaml,
)

PRESERVED = (
    "runs/v3_eval",
    "runs/v3_pilot_k0k1",
    "runs/v3_repl",
    "runs/v3_research",
    "runs/v3_closeout",
    "runs/v3_audit",
    "runs/v3_increment",
    "runs/v3_phaseb",
    "runs/v3_limitations",
)
REGIMES = (
    {"sharing": "context_specific", "k": 0},
    {"sharing": "pooled", "k": 1},
    {"sharing": "selective", "k": 1},
)
MECHANISMS = {
    "matched": "shared",
    "growth_shift": "partially_shared",
    "both": "conflicting",
}


class ObjectiveError(ValueError):
    pass


def _mc_plan(prior_sd: float = 0.144, target_se: float = 0.10) -> dict[str, Any]:
    n = int(np.ceil((prior_sd / target_se) ** 2)) if target_se > 0 else 3
    n = max(n, 3)
    return {
        "prior_sd_of_dataset_means": prior_sd,
        "target_se_of_mean": target_se,
        "n_datasets": n,
        "n_train_seeds": 1,
        "justification": (
            f"Prior matched increment-ratio sd of dataset means was {prior_sd}. "
            f"SE of the mean ≤ {target_se} requires n ≥ ({prior_sd}/{target_se})^2 "
            f"= {(prior_sd / target_se) ** 2:.2f}. Use n_datasets={n}. "
            "Train-seed split reshuffling is a nested nuisance, not the outer DGM replicate."
        ),
    }


def _score_increment_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec.get("failed") or not rec.get("dir"):
            continue
        cell = Path(rec["dir"])
        cfg = load_yaml(cell / "effective.yaml") if (cell / "effective.yaml").exists() else {}
        n_draws = int((cfg.get("evaluation") or {}).get("n_draws") or 40)
        seed = int(cfg.get("seed") or rec.get("train_seed") or 0) + 17
        prep = load_cell_prepared(cell)
        if "test" not in prep["data"] or not len(prep["data"]["test"].obs):
            continue
        from send_bw.pipeline import model_class

        for name in ("spline", "transformer"):
            if not (cell / "models" / name / "model.json").exists():
                continue
            model = model_class(name).load(cell / "models" / name)
            for row in score_model_on_test(model, prep["data"]["test"], n_draws, seed):
                rows.append({
                    **row,
                    "name": rec.get("name"), "model": name,
                    "sharing": rec.get("sharing"), "bank": rec.get("bank"),
                    "mechanism": MECHANISMS.get(rec.get("bank"), rec.get("bank")),
                    "density": "full" if rec.get("sparse_target_train_n") in (None, "full") else "sparse",
                    "dataset_index": rec.get("dataset_index"),
                    "generate_seed": rec.get("generate_seed"),
                    "train_seed": rec.get("train_seed"),
                })
    return rows


def _primary_increment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spline / matched / context-specific / full — the development increment target."""
    return [
        r for r in rows
        if r.get("model") == "spline"
        and r.get("sharing") == "context_specific"
        and r.get("bank") == "matched"
        and r.get("density") == "full"
        and r.get("matched_ratio") is not None
    ]


def _increment_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("matched_ratio") is None:
            continue
        key = (r.get("model"), r.get("sharing"), r.get("bank"), r.get("density"))
        by.setdefault(key, []).append(r)
    out = []
    for key, g in sorted(by.items()):
        summ = hierarchical_mean(
            [x["matched_ratio"] for x in g],
            group_keys=[x["dataset_index"] for x in g],
        )
        out.append({
            "model": key[0], "sharing": key[1], "bank": key[2], "density": key[3],
            "mechanism": MECHANISMS.get(key[2], key[2]),
            "primary": key == ("spline", "context_specific", "matched", "full"),
            **summ,
        })
    return out


def _distributional_pass(inc_rows: list[dict[str, Any]], bayes: dict[str, Any]) -> dict[str, Any]:
    primary = _primary_increment_rows(inc_rows)
    use = primary or [r for r in inc_rows if r.get("matched_ratio") is not None]
    ratios = [r["matched_ratio"] for r in use]
    keys = [r["dataset_index"] for r in use]
    summ = hierarchical_mean(ratios, group_keys=keys)
    mean = summ.get("mean_of_group_means")
    sd = summ.get("sd_of_group_means")
    n = summ.get("n_groups") or 0
    se = (sd / np.sqrt(n)) if sd is not None and n else None
    inc_ok = mean is not None and 0.80 <= mean <= 1.20
    bay_ok = bool(bayes.get("reliable"))
    return {
        "increment": {
            "summary": summ, "se_of_mean": float(se) if se is not None else None,
            "pass": inc_ok, "n_primary_rows": len(primary),
            "used_primary_slice": bool(primary),
            "rule": (
                "mean of dataset means of matched increment ratio on the primary "
                "slice (spline, matched/shared, context_specific, full target history) "
                "in [0.80, 1.20]"
            ),
        },
        "bayesian": {"reliable": bay_ok, "pass": bay_ok,
                     "rule": "unchanged gate rhat≤1.01, ESS≥400, divergences==0"},
        "pass": inc_ok,  # design may run on increment pass even if Bayesian is still unreliable
        "note": "Sharing and transformer are not required to win. Bayesian is scored separately.",
    }


def execute_v3_objective(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or "runs/v3_objective")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise ObjectiveError("refusing to write into a preserved artifact directory")
    if (cfg.get("design") or {}).get("request") and not (cfg.get("objective") or {}).get("allow_design_eval"):
        raise ObjectiveError("operational design recommendations stay off")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise ObjectiveError("real-data access is off")
    parent.mkdir(parents=True, exist_ok=True)
    ev = cfg.get("objective") or {}
    mc = _mc_plan(float(ev.get("prior_increment_sd", 0.144)), float(ev.get("target_se", 0.10)))
    n_ds = int(ev.get("n_dataset_seeds") or mc["n_datasets"])
    n_ts = int(ev.get("n_train_seeds") or 1)
    banks = list(ev.get("banks") or list(MECHANISMS))
    gen0 = int(cfg["generate_seed"])
    tr0 = int(cfg["train_seed"])
    lock = {
        "config_path": str(config_path),
        "config_hash": cache_key(cfg),
        "git_revision": git_revision(),
        "generate_seed": gen0,
        "train_seed": tr0,
        "mc_plan": mc,
        "n_dataset_seeds": n_ds,
        "n_train_seeds": n_ts,
        "banks": banks,
        "mechanisms": {b: MECHANISMS.get(b, b) for b in banks},
        "retune_after_test": False,
        "real_data": False,
        "sharing_required_to_improve": False,
        "transformer_required_to_win": False,
        "labels": list(LABELS),
        "reliability_thresholds": dict(RELIABILITY_THRESHOLDS),
        "development_then_frozen_eval": True,
    }
    write_json(parent / "freeze_lock.json", lock)
    t0 = time.time()

    # --- development: one small context-specific cell for increment + Bayes ---
    dev_parent = parent / "development"
    dev_base = copy.deepcopy(cfg)
    dev_base["objective"] = ev
    dev_gen = int(ev.get("development_generate_seed") or (gen0 - 10007))
    dev_train = int(ev.get("development_train_seed") or (tr0 - 1009))
    bank_cfg = _bank_cfg(dev_base, dev_parent, 0, dev_gen, "matched")
    bank_run = _run_from_cfg(bank_cfg)
    bank_run.generate(force=force)
    spec = {
        "name": "dev_matched_context_specific_k0_full",
        "dataset_index": 0, "train_index": 0, "bank": "matched",
        "sharing": "context_specific", "k": 0, "sparse_target_train_n": None,
        "generate_seed": dev_gen, "train_seed": dev_train,
        "models": ["spline", "transformer", "bayesian"],
    }
    cell_cfg = _cell_cfg(dev_base, dev_parent, spec, generate_seed=dev_gen, train_seed=dev_train,
                         models=spec["models"], sparse_target_train_n=None, reuse_target_from=None)
    cell_cfg["models"]["spline"]["increment_moment_split"] = "validation"
    cell_cfg["models"]["bayesian"]["identifiable_mean"] = True
    cell_cfg["models"]["bayesian"]["sensitivity"] = {"enabled": False}
    dev_run = _run_from_cfg(cell_cfg)
    dev_run.ingest(force=force)
    dev_run.prepare(force=force)
    dev_run.train(force=force)
    dev_run.evaluate(force=force)
    dev_rec = _record_cell(dev_run, spec)
    write_json(parent / "development_cell.json", {
        "record": {k: v for k, v in dev_rec.items() if k != "evaluation_packs"},
        "increment_moment": ((read_json(dev_run.p("models", "spline", "training_history.json"))
                              if dev_run.p("models", "spline", "training_history.json").exists() else {})
                             .get("increment_moment")),
        "bayesian_gate": dev_rec.get("bayesian_gate"),
        "sampling_diagnosis": dev_rec.get("sampling_diagnosis"),
    })

    # --- frozen evaluation datasets ---
    eval_parent = parent / "eval"
    records = []
    failures: list[str] = []
    split_hashes: dict[tuple, dict[str, str]] = {}
    bank_manifest = {}
    for i in range(n_ds):
        gen_seed = gen0 + i * 1009
        for bank in banks:
            bcfg = _bank_cfg(cfg, eval_parent, i, gen_seed, bank)
            br = _run_from_cfg(bcfg)
            br.generate(force=force)
            bank_manifest[f"d{i}_{bank}"] = {
                "generate_seed": gen_seed, "donor_scenario": bank,
                "mechanism": MECHANISMS.get(bank, bank),
                "synthetic_dir": str(Path(bcfg["synthetic"]["out_dir"])),
            }
        for j in range(n_ts):
            train_seed = tr0 + j * 17
            reuse = None
            for bank in banks:
                for regime in REGIMES:
                    for dens in (None, int(ev.get("sparse_target_train_n") or 4)):
                        want_bayes = (
                            i == 0 and j == 0 and bank == "matched"
                            and regime["sharing"] == "context_specific" and dens is None
                        )
                        dens_tag = "full" if dens is None else f"stt{int(dens)}"
                        spec = {
                            "name": f"d{i}_t{j}_{bank}_{regime['sharing']}_k{regime['k']}_{dens_tag}",
                            "dataset_index": i, "train_index": j, "bank": bank,
                            **regime, "sparse_target_train_n": None if dens is None else int(dens),
                            "generate_seed": gen_seed, "train_seed": train_seed,
                            "models": ["spline", "transformer"] + (["bayesian"] if want_bayes else []),
                            "mechanism": MECHANISMS.get(bank, bank),
                        }
                        ccfg = _cell_cfg(cfg, eval_parent, spec, generate_seed=gen_seed,
                                         train_seed=train_seed, models=spec["models"],
                                         sparse_target_train_n=spec["sparse_target_train_n"],
                                         reuse_target_from=reuse)
                        ccfg["models"]["spline"]["increment_moment_split"] = "validation"
                        if want_bayes:
                            ccfg["models"]["bayesian"]["identifiable_mean"] = True
                            ccfg["models"]["bayesian"]["sensitivity"] = {"enabled": False}
                        run = _run_from_cfg(ccfg)
                        try:
                            run.ingest(force=force)
                            run.prepare(force=force)
                            run.train(force=force)
                            run.evaluate(force=force)
                            rec = _record_cell(run, spec)
                            records.append(rec)
                            h = rec.get("target_manifest") or {}
                            key = (i, j)
                            if key not in split_hashes:
                                split_hashes[key] = {n: h.get(n) for n in ("validation", "test", "all_target")}
                            if reuse is None:
                                reuse = str(run.p("splits", "x").parent)
                        except Exception as e:  # noqa: BLE001
                            failures.append(f"{spec['name']}: {type(e).__name__}: {e}")
                            records.append({**spec, "failed": True, "error": f"{type(e).__name__}: {e}"})
                            if reuse is None and Path(run.dir, "splits", "target_manifest.json").exists():
                                reuse = str(run.p("splits", "x").parent)

    assembled = _assemble(records)
    inc_rows = _score_increment_cells(records)
    inc_slices = _increment_slices(inc_rows)
    bayes_eval = None
    for rec in records:
        if rec.get("bayesian_gate"):
            bayes_eval = rec.get("bayesian_gate")
            break
    dist = _distributional_pass(inc_rows, bayes_eval or {})
    design = {"status": "not_run", "reason": "distributional increment check did not pass"}
    if dist.get("pass") and ev.get("allow_design_eval"):
        design = _run_design_vs_b(eval_parent, records, ev)
    auth = discover_authorization(None)
    out = {
        "labels": list(LABELS) + ["Exploratory.", "Fresh frozen evaluation."],
        "output_dir": str(parent),
        "git_revision": git_revision(),
        "freeze_lock": lock,
        "mc_plan": mc,
        "development": read_json(parent / "development_cell.json"),
        "bank_manifest": bank_manifest,
        "n_cells": len(records),
        "failures": failures,
        "locked_target_splits": {f"d{i}_t{j}": h for (i, j), h in split_hashes.items()},
        "sharing_uncertainty": assembled.get("sharing_uncertainty"),
        "density_uncertainty": assembled.get("density_uncertainty"),
        "increment_rows": inc_rows,
        "increment_slices": inc_slices,
        "increment_hierarchical": hierarchical_mean(
            [r["matched_ratio"] for r in _primary_increment_rows(inc_rows)],
            group_keys=[r["dataset_index"] for r in _primary_increment_rows(inc_rows)],
        ),
        "distributional": dist,
        "bayesian_eval": bayes_eval,
        "design_vs_generator_b": design,
        "real_data_accessed": False,
        "authorization_present": bool(auth.get("authorized")),
        "sharing_required_to_improve": False,
        "transformer_required_to_win": False,
        "retuned_after_test": False,
        "seconds_total": time.time() - t0,
        "cells": records,
    }
    write_json(parent / "SUMMARY.json", out)
    write_json(parent / "paired_rows.json", {
        "sharing": assembled.get("sharing_rows"), "density": assembled.get("density_rows")})
    _write_verdict(parent, out)
    return out


def _run_design_vs_b(eval_parent: Path, records: list[dict[str, Any]], ev: dict[str, Any]) -> dict[str, Any]:
    from send_bw.pipeline import model_class
    from send_bw.simulation.workflow import run_design_workflow
    from send_bw.utils.artifacts import setup_logging

    rec = next((r for r in records if r.get("sharing") == "context_specific"
                and r.get("bank") == "matched" and not r.get("failed")
                and r.get("sparse_target_train_n") in (None, "full")), None)
    if not rec or not rec.get("dir"):
        return {"status": "not_run", "reason": "no context-specific matched cell"}
    cell = Path(rec["dir"])
    models = {}
    for name in ("spline", "transformer"):
        if (cell / "models" / name / "model.json").exists():
            models[name] = model_class(name).load(cell / "models" / name)
    if "spline" not in models:
        return {"status": "not_run", "reason": "spline missing"}
    truth = None
    syn = cell.parent.parent.parent / "banks" / "matched" / "synthetic" / "truth"
    # datasets/d0/trains/t0/cells/... -> banks at datasets/d0/banks
    parts = cell.parts
    try:
        di = parts.index("datasets")
        truth = Path(*parts[: di + 2]) / "banks" / "matched" / "synthetic" / "truth"
    except ValueError:
        truth = syn
    request = ev.get("design_request") or "configs/new_study.smoke.yaml"
    out_dir = eval_parent / "design_vs_b"
    log = setup_logging(out_dir / "design.jsonl")
    dcfg = {"request": request, "replicates_proposal": int(ev.get("design_B_proposal") or 120),
            "replicates_final": int(ev.get("design_B_final") or 200),
            "proposal_models": ["spline"]}
    res = run_design_workflow(models, dcfg, out_dir, seed=int(ev.get("design_seed") or 17),
                              truth_dir=str(truth) if Path(truth).exists() else None, log=log)
    recs = res.get("recommendations") or {}
    spline_rec = recs.get("spline") or {}
    b_checks = []
    for name, rec_m in recs.items():
        for chk in rec_m.get("finalist_checks") or []:
            b_checks.append({"model": name, **chk})
    return {
        "status": res.get("status"),
        "cell": rec.get("name"),
        "truth_dir": str(truth) if Path(truth).exists() else None,
        "independent_generator": "B",
        "replicates_proposal": dcfg["replicates_proposal"],
        "replicates_final": dcfg["replicates_final"],
        "nonconclusive": dcfg["replicates_final"] < 2000,
        "recommendations_emitted": bool(recs),
        "spline_status": spline_rec.get("status"),
        "generator_b_finalist_checks": b_checks,
        "monte_carlo_nonconclusive": bool(res.get("monte_carlo_nonconclusive")),
        "note": (
            "Synthetic design-vs-B score only. Not an operational recommendation. "
            "B<2000 is exploratory / nonconclusive for locked design claims. "
            "Sharing and the transformer are not required to win."
        ),
        "summary": {
            "status": res.get("status"),
            "spline": spline_rec.get("status"),
            "design": spline_rec.get("design"),
            "n_animals": spline_rec.get("n_animals"),
        },
    }


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _performance_labels(out: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Map executed evidence to objective performance labels (not functionality)."""
    dist = out.get("distributional") or {}
    inc = out.get("increment_hierarchical") or {}
    bay = out.get("bayesian_eval") or {}
    design = out.get("design_vs_generator_b") or {}
    n_ok = sum(1 for r in (out.get("cells") or []) if not r.get("failed"))
    n_fail = len(out.get("failures") or [])
    banks = {r.get("bank") for r in (out.get("cells") or []) if not r.get("failed")}
    dens = {r.get("sparse_target_train_n") for r in (out.get("cells") or []) if not r.get("failed")}
    n_ds = (inc.get("n_groups") or 0) or len({r.get("dataset_index") for r in (out.get("cells") or [])})
    inc_pass = bool((dist.get("increment") or {}).get("pass"))
    mean = inc.get("mean_of_group_means")
    bay_ok = bool(bay.get("reliable"))
    design_ran = design.get("status") not in (None, "not_run")
    labels: dict[str, dict[str, str]] = {}

    def put(oid: str, perf: str, evidence: str) -> None:
        labels[oid] = {"performance": perf, "evidence": evidence}

    if n_ok and n_ds >= 3:
        put("O1", "partial",
            f"{n_ok} cells on {n_ds} frozen datasets. Scores exist; still synthetic / exploratory.")
    elif n_ok:
        put("O1", "partial", f"{n_ok} cells; dataset count {n_ds} below the MC plan.")
    else:
        put("O1", "failed" if n_fail else "not evaluated", f"failures={n_fail}")

    share = out.get("sharing_uncertainty") or {}
    if share.get("groups"):
        put("O2", "partial",
            "Paired fixed-holdout sharing contrasts executed. Sharing is not required to win.")
    else:
        put("O2", "failed" if n_fail else "not evaluated", "No sharing contrast rows.")

    put("O3", "achieved" if share.get("groups") or n_ok else "not evaluated",
        "Harmful-transfer sign (Δ>0) is published when it appears; signs are not clipped.")

    put("O4", "partial" if n_ok else "not evaluated",
        "Active−control contrast at T=28 is scored on the frozen test packs.")

    if {"matched", "growth_shift", "both"} <= banks:
        put("O5", "partial",
            "Shared / partially shared / conflicting banks all generated and scored.")
    elif banks:
        put("O5", "partial", f"Executed banks: {sorted(banks)}")
    else:
        put("O5", "not evaluated", "No mechanism bank completed.")

    if None in dens and any(d not in (None, "full") for d in dens):
        put("O6", "partial",
            "Full vs sparse target-study training on paired holdouts. Density is not required to win.")
    else:
        put("O6", "not evaluated", f"densities seen: {sorted(str(d) for d in dens)}")

    put("O7", "not evaluated",
        "Encoder-sparsity grid (W0/W1/Wlast14/Wall14) was not re-run; prior v3_eval stands.")

    put("O8", "not evaluated",
        "No paired design-stage vs predose holdout table. Design engine exists but is not this score.")

    put("O9", "not evaluated",
        "Early-history conditioning was not the primary frozen task.")

    mc = out.get("mc_plan") or {}
    if n_ds >= int(mc.get("n_datasets") or 3) and n_ok:
        put("O10", "achieved",
            (mc.get("justification") or "") + f" Executed n_datasets={n_ds}.")
    elif n_ok:
        put("O10", "partial", f"Cells ran but n_datasets={n_ds} did not meet the MC plan.")
    else:
        put("O10", "failed" if n_fail else "not evaluated", "Replication did not complete.")

    if mean is None:
        put("O11", "failed" if n_ok else "not evaluated", "No matched increment ratios on the primary slice.")
    elif inc_pass:
        put("O11", "achieved",
            f"Primary matched increment mean-of-dataset-means {_fmt(mean)} "
            f"(sd_ds={_fmt(inc.get('sd_of_group_means'))}; n={inc.get('n_groups')}) inside [0.80, 1.20].")
    else:
        put("O11", "failed",
            f"Primary matched increment mean-of-dataset-means {_fmt(mean)} "
            f"(sd_ds={_fmt(inc.get('sd_of_group_means'))}; n={inc.get('n_groups')}) outside [0.80, 1.20].")

    if not bay:
        put("O12", "not evaluated", "No Bayesian cell recorded.")
    elif bay_ok:
        put("O12", "achieved",
            f"Unchanged gate passed (rhat={_fmt(bay.get('rhat_max'))}, "
            f"ESS bulk/tail={_fmt(bay.get('ess_bulk_min'))}/{_fmt(bay.get('ess_tail_min'))}, "
            f"divergences={bay.get('divergences')}).")
    else:
        put("O12", "failed",
            f"Unchanged gate failed (rhat={_fmt(bay.get('rhat_max'))}, "
            f"ESS bulk/tail={_fmt(bay.get('ess_bulk_min'))}/{_fmt(bay.get('ess_tail_min'))}, "
            f"divergences={bay.get('divergences')}; reason={bay.get('reason')}).")

    if not design_ran:
        put("O13", "not evaluated",
            design.get("reason") or "Design vs generator B was not run.")
    elif design.get("nonconclusive"):
        put("O13", "partial",
            f"Ran vs independent generator B; B_final={design.get('replicates_final')} < 2000, "
            f"exploratory/nonconclusive. spline_status={design.get('spline_status')}.")
    else:
        put("O13", "partial",
            f"Ran vs independent generator B. status={design.get('status')}; "
            f"spline_status={design.get('spline_status')}. Not an operational recommendation.")

    put("O14", "partial" if n_ok else "not evaluated",
        "Spline and transformer both trained. Neither family is required to win.")

    put("O15", "not evaluated",
        "No real SEND data. Phase B remains ACCESS_BLOCKER. Correctly not opened.")
    return labels


def _write_verdict(parent: Path, out: dict[str, Any]) -> None:
    dist = out.get("distributional") or {}
    inc = out.get("increment_hierarchical") or {}
    bay = out.get("bayesian_eval") or {}
    design = out.get("design_vs_generator_b") or {}
    mc = out.get("mc_plan") or {}
    labels = _performance_labels(out)
    n_ok = sum(1 for r in (out.get("cells") or []) if not r.get("failed"))
    lines = [
        "# Synthetic objective validation — verdict",
        "",
        "> **Synthetic diagnostic. Exploratory. Not design qualification. "
        "No real-data validation. No operational recommendation.**",
        "",
        "This is a **performance** verdict on fresh frozen synthetic datasets, "
        "not an implementation-status list. Functionality scores stay in "
        "`docs/v3/objective_assessment.md`. Sharing and the transformer are not "
        "required to win. Prior artifacts were not modified. No real data were accessed.",
        "",
        "## Monte Carlo plan",
        "",
        mc.get("justification") or "",
        f"- n_datasets={mc.get('n_datasets')}; n_train_seeds={mc.get('n_train_seeds')}",
        f"- cells completed: {n_ok}; failures: {out.get('failures')}",
        "",
        "## Distributional checks (fresh frozen test)",
        "",
        "Primary slice: spline, matched (shared mechanism), context-specific, full target history.",
        f"- matched increment ratio, mean of dataset means: {_fmt(inc.get('mean_of_group_means'))} "
        f"(sd_ds={_fmt(inc.get('sd_of_group_means'))}; n={inc.get('n_groups')}; "
        f"SE={_fmt((dist.get('increment') or {}).get('se_of_mean'))})",
        f"- increment pass: {(dist.get('increment') or {}).get('pass')}",
        f"- Bayesian reliable: {bay.get('reliable')}; counted={bay.get('counted_as_successful_benchmark')}",
        f"- rhat={_fmt(bay.get('rhat_max'))}; ESS bulk/tail="
        f"{_fmt(bay.get('ess_bulk_min'))}/{_fmt(bay.get('ess_tail_min'))}; "
        f"divergences={bay.get('divergences')}",
        f"- distributional pass (increment rule; Bayes scored separately): {dist.get('pass')}",
        "",
        "## Increment slices (matched pred/obs ratio)",
        "",
        "| model | sharing | bank | mechanism | density | mean of ds means | sd_ds | n_ds | primary |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in out.get("increment_slices") or []:
        lines.append(
            f"| {s.get('model')} | {s.get('sharing')} | {s.get('bank')} | {s.get('mechanism')} | "
            f"{s.get('density')} | {_fmt(s.get('mean_of_group_means'))} | "
            f"{_fmt(s.get('sd_of_group_means'))} | {s.get('n_groups')} | {s.get('primary')} |"
        )
    lines += [
        "",
        "## Design vs independent generator B",
        "",
        f"- status: `{design.get('status')}`",
        f"- spline_status: `{design.get('spline_status')}`",
        f"- B_proposal={design.get('replicates_proposal')}; B_final={design.get('replicates_final')}; "
        f"nonconclusive={design.get('nonconclusive')}",
        f"- generator_B finalist checks: {design.get('generator_b_finalist_checks')}",
        f"- {design.get('reason') or design.get('note') or ''}",
        "",
        "## Sharing / density RMSE (not required to improve)",
        "",
        "| contrast | model | sharing | bank | mean Δ | n |",
        "|---|---|---|---|---|---|",
    ]
    for kind, block in (("sharing", out.get("sharing_uncertainty") or {}),
                        ("density", out.get("density_uncertainty") or {})):
        for g in block.get("groups") or []:
            key = g.get("key") or {}
            if key.get("metric") != "rmse_log":
                continue
            lines.append(
                f"| {kind} | {key.get('model')} | {key.get('sharing')} | {key.get('bank', '')} | "
                f"{_fmt(g.get('mean_delta'))} | {g.get('n')} |"
            )
    lines += [
        "",
        "## Objective-by-objective performance",
        "",
        "Pre-evaluation functionality is unchanged from `docs/v3/objective_assessment.md`. "
        "The **performance** column below is this frozen evaluation.",
        "",
        "| ID | Performance | Evidence |",
        "|---|---|---|",
    ]
    titles = {
        "O1": "Target-species predose prediction",
        "O2": "Sharing on a fixed holdout",
        "O3": "Detect harmful transfer",
        "O4": "Active−control contrast at T=28",
        "O5": "Shared / partial / conflicting mechanisms",
        "O6": "Full vs sparse target-study history",
        "O7": "Sparse-history encoder grid",
        "O8": "Design-stage (no predose) holdout",
        "O9": "Early-history conditioning",
        "O10": "MC-justified nested replication",
        "O11": "Temporal increment distribution",
        "O12": "Reliable Bayesian inference",
        "O13": "Designs vs independent generator B",
        "O14": "Spline vs transformer (neither required to win)",
        "O15": "Real SEND / historical validation",
    }
    for oid, title in titles.items():
        lab = labels.get(oid) or {}
        lines.append(f"| {oid} {title} | **{lab.get('performance', 'not evaluated')}** | {lab.get('evidence', '')} |")
    lines += [
        "",
        "## What this verdict does not claim",
        "",
        "- It does not treat unit tests as scientific readiness.",
        "- It does not require sharing or the transformer to win.",
        "- It does not open a real extract or recommend an operational design.",
        "- It does not overwrite `runs/v3_eval`, `v3_pilot_k0k1`, `v3_repl`, "
        "`v3_research`, `v3_closeout`, `v3_audit`, `v3_increment`, `v3_phaseb`, "
        "or `v3_limitations`.",
        "",
        f"- seconds: {out.get('seconds_total')}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (parent / "VERDICT.md").write_text(text, encoding="utf-8")
    Path("docs/v3/objective_verdict.md").write_text(text, encoding="utf-8")
    write_json(parent / "objective_labels.json", labels)

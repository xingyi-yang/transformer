"""Command-line interface: `send-bw <command> --config <yaml>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from send_bw.utils.security import redact

MODEL_NAMES = ("spline", "transformer", "bayesian")


def _run(args):
    from send_bw.pipeline import Run

    return Run(args.config)


def load_artifacts(path: str | Path) -> dict:
    """Load serialized models from a run directory, its models/ folder, or a single model folder."""
    from send_bw.pipeline import model_class

    p = Path(path)
    if (p / "models").is_dir():
        p = p / "models"
    if (p / "model.json").exists():
        name = json.loads((p / "model.json").read_text(encoding="utf-8"))["name"]
        return {name: model_class(name).load(p)}
    out = {m: model_class(m).load(p / m) for m in MODEL_NAMES if (p / m / "model.json").exists()}
    if not out:
        raise FileNotFoundError(f"no serialized models found under {path}")
    return out


def _models(args):
    if args.model:
        return load_artifacts(args.model), None
    run = _run(args)
    return {m: run.load_model(m) for m in run.available_models()}, run


def cmd_validate_request(args) -> int:
    from send_bw.simulation.designs import load_request, validate_request

    models, run = _models(args)
    path = args.request or (run.cfg["design"]["request"] if run else None)
    if path is None:
        raise ValueError("--request is required")
    req = load_request(path)
    ref = models.get("spline") or next(iter(models.values()))
    out = validate_request(req, ref.spec, req.compound_reference(ref.compound_ref))
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "supported" else 2


def cmd_propose_design(args) -> int:
    """Inference only: serialized models + a new request -> design workflow (no retraining)."""
    from send_bw.simulation.workflow import run_design_workflow

    models, run = _models(args)
    dcfg = dict(run.cfg.get("design", {})) if run else {"replicates_proposal": 400, "replicates_final": 1000}
    if args.request:
        dcfg["request"] = args.request
    if args.replicates:
        dcfg["replicates_proposal"], dcfg["replicates_final"] = args.replicates
    if "request" not in dcfg:
        raise ValueError("--request is required")
    if args.out:
        out_dir = Path(args.out)
    elif run:
        out_dir = run.p("propose", Path(dcfg["request"]).stem, "x").parent
    else:
        out_dir = Path("runs") / "propose" / Path(dcfg["request"]).stem
    truth = None
    if args.with_oracle and run:
        truth = run.manifest.data["stages"].get("generate", {}).get("truth_dir")
    log = run.log if run else logging.getLogger("send_bw")
    seed = run.seed if run else args.seed
    res = run_design_workflow(models, dcfg, out_dir, seed, truth, log)
    print(json.dumps({"status": res["status"], "out_dir": str(out_dir),
                      "recommendations": {k: {kk: v.get(kk) for kk in ("status", "design", "n_animals")}
                                          for k, v in res.get("recommendations", {}).items()}}, indent=2))
    return 0 if res["status"] != "not_supported" else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="send-bw", description="Body-weight modelling and design simulation")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["generate", "inspect-source", "ingest", "prepare", "validate-real", "train", "evaluate",
                 "design", "report", "run-all", "validate-request", "propose-design", "v3-pilot",
                 "v3-eval", "v3-repl", "v3-research", "v3-closeout", "v3-increment",
                 "v3-limitations", "v3-objective", "v3-official-seq", "v3-seq-diag",
                 "v3-eval-correct", "v3-cov-chronos", "v3-share-repl",
                 "v3-share-abstain", "v3-donor-rank", "historical-audit",
                 "tft-early-hist-bench", "tft-early-hist-extend", "tft-forecast"]:
        s = sub.add_parser(name)
        artifact_cmd = name in ("validate-request", "propose-design", "v3-closeout",
                                "v3-increment", "v3-limitations")
        s.add_argument("--config", required=not artifact_cmd and name != "v3-closeout")
        s.add_argument("--force", action="store_true")
        if name == "train":
            s.add_argument("--models", nargs="*")
        if artifact_cmd:
            s.add_argument("--request")
            s.add_argument("--model", help="run directory, its models/ folder, or one model folder")
        if name == "propose-design":
            s.add_argument("--out")
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--replicates", type=int, nargs=2, metavar=("PROPOSAL", "FINAL"))
            s.add_argument("--with-oracle", action="store_true",
                           help="synthetic runs only (needs --config): include the oracle reference")
        if name == "v3-closeout":
            s.add_argument("--research-dir", default="runs/v3_research")
            s.add_argument("--output-dir", default="runs/v3_closeout")
        if name == "v3-increment":
            s.add_argument("--research-dir", default="runs/v3_research")
            s.add_argument("--output-dir", default="runs/v3_increment")
        if name == "v3-limitations":
            s.add_argument("--increment-dir", default="runs/v3_increment")
            s.add_argument("--phaseb-dir", default="runs/v3_phaseb")
            s.add_argument("--output-dir", default="runs/v3_limitations")
        if name == "historical-audit":
            s.add_argument("--phase", default="A")
            s.add_argument("--authorization", help="signed ticket; required for phases B and C")
    args = p.parse_args(argv)
    if args.cmd not in ("v3-closeout", "v3-increment", "v3-limitations") and getattr(args, "model", None) is None and args.config is None:
        p.error("--config or --model is required")
    try:
        if args.cmd == "v3-pilot":
            from send_bw.pilot_execute import execute_k0k1, load_yaml

            out = execute_k0k1(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "git_revision": (out.get("freeze_lock") or {}).get("git_revision"),
                              "config_hash": (out.get("freeze_lock") or {}).get("config_hash"),
                              "retuned_after_test": False,
                              "parent": load_yaml(args.config).get("output_dir")}, indent=2, default=str))
            return 0
        if args.cmd == "v3-eval":
            from send_bw.v3_eval import execute_v3_eval
            from send_bw.v3_eval import load_yaml as load_eval_yaml

            out = execute_v3_eval(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "git_revision": (out.get("freeze_lock") or {}).get("git_revision"),
                              "config_hash": (out.get("freeze_lock") or {}).get("config_hash"),
                              "retuned_after_test": False,
                              "failures": out.get("failures"),
                              "parent": load_eval_yaml(args.config).get("output_dir")}, indent=2, default=str))
            return 0
        if args.cmd == "v3-research":
            from send_bw.v3_research import execute_v3_research
            from send_bw.v3_research import load_yaml as load_research_yaml

            out = execute_v3_research(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "git_revision": (out.get("freeze_lock") or {}).get("git_revision"),
                              "config_hash": (out.get("freeze_lock") or {}).get("config_hash"),
                              "retuned_after_test": False,
                              "component_status": out.get("component_status"),
                              "failures": out.get("failures"),
                              "parent": load_research_yaml(args.config).get("output_dir")}, indent=2, default=str))
            return 0
        if args.cmd == "v3-closeout":
            from send_bw.v3_closeout import execute_v3_closeout

            out = execute_v3_closeout(args.research_dir, args.output_dir)
            print(json.dumps({"status": "ok",
                              "retuned_after_test": False,
                              "grid_relaunched": False,
                              "source": out.get("source_research"),
                              "parent": args.output_dir}, indent=2, default=str))
            return 0
        if args.cmd == "v3-increment":
            from send_bw.v3_increment import execute_v3_increment

            out = execute_v3_increment(args.research_dir, args.output_dir)
            print(json.dumps({"status": "ok",
                              "retuned_after_test": False,
                              "grid_relaunched": False,
                              "exploratory": out.get("exploratory"),
                              "n_datasets": out.get("n_datasets"),
                              "n_cells_scored": out.get("n_cells_scored"),
                              "bayesian_unreliable": (out.get("bayesian") or {}).get("explicitly_unreliable"),
                              "parent": args.output_dir}, indent=2, default=str))
            return 0
        if args.cmd == "v3-limitations":
            from send_bw.v3_limitations import execute_v3_limitations

            out = execute_v3_limitations(args.increment_dir, args.phaseb_dir, args.output_dir)
            print(json.dumps({"status": "ok",
                              "broad_synthetic_development": out.get("broad_synthetic_development"),
                              "next_implementation_task": out.get("next_implementation_task"),
                              "adjusted_using_exposed_test": out.get("adjusted_using_exposed_test"),
                              "phase_b": (out.get("phase_b") or {}).get("status"),
                              "parent": args.output_dir}, indent=2, default=str))
            return 0
        if args.cmd == "v3-official-seq":
            from send_bw.v3_official_seq import execute_v3_official_seq

            out = execute_v3_official_seq(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "retuned_after_test": False,
                              "n_results": len(out.get("results") or {}),
                              "blockers": len(out.get("blockers") or []),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "v3-seq-diag":
            from send_bw.v3_seq_diag import execute_v3_seq_diag

            out = execute_v3_seq_diag(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "retuned_after_test": False,
                              "official_seq_modified": out.get("official_seq_modified"),
                              "predose_evaluated": out.get("predose_evaluated"),
                              "n_replicates": len(out.get("replicates") or []),
                              "blockers": len(out.get("blockers") or []),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "v3-eval-correct":
            from send_bw.v3_eval_correct import execute_v3_eval_correct

            out = execute_v3_eval_correct(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "training_run": out.get("training_run"),
                              "official_seq_rewritten": False,
                              "covariate_aware_comparison": out.get("covariate_aware_comparison"),
                              "n_finetune_failures": len(out.get("chronos_finetune_failures") or []),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "v3-cov-chronos":
            from send_bw.v3_cov_chronos import execute_v3_cov_chronos

            out = execute_v3_cov_chronos(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "n_failures": len(out.get("failures") or []),
                              "history_only_complete": (out.get("chronos_finetune_pool") or {}).get(
                                  "history_only_complete"),
                              "covariate_aware_complete": (out.get("chronos_finetune_pool") or {}).get(
                                  "covariate_aware_complete"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "v3-share-repl":
            from send_bw.v3_share_repl import execute_v3_share_repl

            out = execute_v3_share_repl(args.config, force=args.force)
            lock = out.get("freeze_lock") or {}
            print(json.dumps({"status": "ok",
                              "n_dataset_seeds": lock.get("n_dataset_seeds"),
                              "n_train_seeds": lock.get("n_train_seeds"),
                              "generate_seeds": lock.get("generate_seeds"),
                              "train_seeds": lock.get("train_seeds"),
                              "n_failures": len(out.get("failures") or []),
                              "chronos_complete": (out.get("chronos_finetune_pool") or {}).get(
                                  "n_complete"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "v3-share-abstain":
            from send_bw.v3_share_abstain import execute_v3_share_abstain

            out = execute_v3_share_abstain(args.config, force=args.force)
            lock = out.get("freeze_lock") or {}
            print(json.dumps({"status": out.get("status"),
                              "n_dataset_seeds": lock.get("n_dataset_seeds"),
                              "n_train_seeds": lock.get("n_train_seeds"),
                              "generate_seeds": lock.get("generate_seeds"),
                              "train_seeds": lock.get("train_seeds"),
                              "execute": lock.get("execute"),
                              "n_model_fits": lock.get("n_model_fits"),
                              "estimated_runtime_seconds": lock.get("estimated_runtime_seconds"),
                              "seconds_total": out.get("seconds_total"),
                              "compute_cap_hit": out.get("compute_cap_hit"),
                              "protected_unchanged": (out.get("protected") or {}).get("unchanged"),
                              "proposal_hash": (lock.get("proposal") or {}).get("config_hash"),
                              "science_fingerprint_match": (lock.get("proposal") or {}).get(
                                  "science_fingerprint_match"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "v3-donor-rank":
            from send_bw.v3_donor_rank import execute_v3_donor_rank

            out = execute_v3_donor_rank(args.config, force=args.force)
            lock = out.get("freeze_lock") or {}
            print(json.dumps({"status": out.get("status"),
                              "n_dataset_seeds": lock.get("n_dataset_seeds"),
                              "n_train_seeds": lock.get("n_train_seeds"),
                              "generate_seeds": lock.get("generate_seeds"),
                              "train_seeds": lock.get("train_seeds"),
                              "execute": lock.get("execute"),
                              "n_candidate_fits": out.get("n_candidate_fits"),
                              "n_final_model_fits": out.get("n_final_model_fits"),
                              "n_model_fits_if_executed": out.get("n_model_fits_if_executed"),
                              "estimated_runtime_seconds": lock.get("estimated_runtime_seconds"),
                              "protected_unchanged": (out.get("protected") or {}).get("unchanged"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "v3-objective":
            from send_bw.v3_objective import execute_v3_objective

            out = execute_v3_objective(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "retuned_after_test": False,
                              "distributional_pass": (out.get("distributional") or {}).get("pass"),
                              "n_cells": out.get("n_cells"),
                              "design": (out.get("design_vs_generator_b") or {}).get("status"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "tft-early-hist-bench":
            from send_bw.tft_early_hist_bench import execute_tft_early_hist_bench

            out = execute_tft_early_hist_bench(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "eligible": (out.get("example_summary") or {}).get("n_eligible"),
                              "correctness": (out.get("correctness") or {}).get("ok"),
                              "n_failures": len(out.get("failures") or []),
                              "seconds_total": out.get("seconds_total"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "tft-early-hist-extend":
            from send_bw.tft_early_hist_extend import execute_tft_early_hist_extend

            out = execute_tft_early_hist_extend(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "eligible": (out.get("example_summary") or {}).get("n_eligible"),
                              "correctness": (out.get("correctness") or {}).get("ok"),
                              "n_failures": len(out.get("failures") or []),
                              "seconds_total": out.get("seconds_total"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "tft-forecast":
            from send_bw.tft_primary import execute_tft_forecast

            out = execute_tft_forecast(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "eligible": (out.get("example_summary") or {}).get("n_eligible"),
                              "correctness": (out.get("correctness") or {}).get("ok"),
                              "locked": (out.get("locked_config") or {}),
                              "n_failures": len(out.get("failures") or []),
                              "seconds_total": out.get("seconds_total"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "parent": out.get("output_dir")},
                             indent=2, default=str))
            return 0
        if args.cmd == "historical-audit":
            from send_bw.historical_audit import execute_historical_audit

            out = execute_historical_audit(args.config, phase=args.phase,
                                           authorization_path=args.authorization)
            print(json.dumps({"status": out.get("status") or "ok", "phase": out.get("phase"),
                              "real_data_accessed": out.get("real_data_accessed"),
                              "source_opened": out.get("source_opened"),
                              "endpoint_values_inspected": out.get("endpoint_values_inspected"),
                              "support_for_synthetic_to_real": out.get("support_for_synthetic_to_real"),
                              "blocker": out.get("blocker"),
                              "counts": out.get("counts")}, indent=2, default=str))
            if out.get("status") in {"ACCESS_BLOCKER", "EXTRACT_BLOCKER"}:
                return 2
            return 0
        if args.cmd == "v3-repl":
            from send_bw.v3_repl import execute_v3_repl
            from send_bw.v3_repl import load_yaml as load_repl_yaml

            out = execute_v3_repl(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "git_revision": (out.get("freeze_lock") or {}).get("git_revision"),
                              "config_hash": (out.get("freeze_lock") or {}).get("config_hash"),
                              "retuned_after_test": False,
                              "component_status": out.get("component_status"),
                              "failures": out.get("failures"),
                              "parent": load_repl_yaml(args.config).get("output_dir")}, indent=2, default=str))
            return 0
        if args.cmd == "validate-request":
            return cmd_validate_request(args)
        if args.cmd == "propose-design":
            return cmd_propose_design(args)
        run = _run(args)
        f = args.force
        if args.cmd == "generate":
            run.generate(f)
        elif args.cmd == "inspect-source":
            print(json.dumps(run.inspect_source(), indent=2, default=str))
        elif args.cmd == "ingest":
            run.ingest(f)
        elif args.cmd == "prepare":
            run.prepare(f)
        elif args.cmd == "validate-real":
            print(json.dumps(run.validate_real(), indent=2, default=str))
        elif args.cmd == "train":
            print(json.dumps(run.train(args.models, f), indent=2, default=str))
        elif args.cmd == "evaluate":
            run.evaluate(f)
        elif args.cmd == "design":
            print(json.dumps({"status": run.design(f).get("status")}))
        elif args.cmd == "report":
            print(run.report())
        elif args.cmd == "run-all":
            run.run_all(f)
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"send-bw {args.cmd} failed: {redact(f'{type(e).__name__}: {e}')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

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
                 "v3-eval"]:
        s = sub.add_parser(name)
        artifact_cmd = name in ("validate-request", "propose-design")
        s.add_argument("--config", required=not artifact_cmd)
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
    args = p.parse_args(argv)
    if getattr(args, "model", None) is None and args.config is None:
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
            from send_bw.v3_eval import execute_v3_eval, load_yaml as load_eval_yaml

            out = execute_v3_eval(args.config, force=args.force)
            print(json.dumps({"status": "ok",
                              "git_revision": (out.get("freeze_lock") or {}).get("git_revision"),
                              "config_hash": (out.get("freeze_lock") or {}).get("config_hash"),
                              "retuned_after_test": False,
                              "failures": out.get("failures"),
                              "parent": load_eval_yaml(args.config).get("output_dir")}, indent=2, default=str))
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

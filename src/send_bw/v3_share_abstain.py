"""Audit selective sharing and lock a share-or-abstain experiment.

Does not overwrite PR #12/#13/#14 artifacts. Benchmark training runs only when
the frozen config sets ``share_abstain.execute: true``. The locked proposal
leaves that false.

Synthetic diagnostic. Not design qualification. No real-data validation.
No operational recommendation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from send_bw.evaluation.quantile_forecast import paired_transfer
from send_bw.historical_audit import discover_authorization
from send_bw.protected_hashes import compare_snapshots, snapshot_protected
from send_bw.share_policy import (
    ABSTAIN_METRIC,
    ABSTAIN_MIN_IMPROVEMENT,
    PREFIX_LABEL,
    PREFIX_RULE,
    decide_abstention,
    prefix_random_overlap,
    val_score,
)
from send_bw.sharing import LABELS, SELECTIVE_REGULARIZED, SELECTIVE_V1
from send_bw.utils.artifacts import cache_key, git_revision, read_json, write_json
from send_bw.v3_increment import load_cell_prepared
from send_bw.v3_official_seq import _score_gaussian, collect_versions
from send_bw.v3_research import _bank_cfg, _cell_cfg, _run_from_cfg, load_yaml
from send_bw.v3_share_repl import PRESERVED as SHARE_REPL_PRESERVED
from send_bw.v3_share_repl import WINDOW, _compact, _sign_rows

PRESERVED = tuple(SHARE_REPL_PRESERVED) + ("runs/v3_share_repl",)
PROPOSAL_DIR = "runs/v3_share_abstain"
PROPOSAL_CONFIG = "configs/v3_share_abstain.frozen.yaml"
SHARE_REPL_SUMMARY = Path("runs/v3_share_repl/SUMMARY.json")
SHARE_REPL_CELLS = Path("runs/v3_share_repl/datasets")


class RuntimeCapExceeded(Exception):
    def __init__(self, where: str, elapsed: float, cap: float):
        super().__init__(f"compute cap {cap}s exceeded at {where} ({elapsed:.1f}s)")
        self.where = where
        self.elapsed = elapsed
        self.cap = cap


class RuntimeCap:
    def __init__(self, seconds: float):
        self.cap = float(seconds)
        self.t0 = time.time()
        self.hit = False
        self.where: str | None = None

    def elapsed(self) -> float:
        return time.time() - self.t0

    def check(self, where: str) -> None:
        if self.hit or self.elapsed() > self.cap:
            self.hit = True
            self.where = where
            raise RuntimeCapExceeded(where, self.elapsed(), self.cap)

POLICY_ARMS = (
    {"key": "target_only", "sharing": "context_specific", "k": 0, "donor_rule": "zero"},
    {"key": "random_one", "sharing": "selective", "k": 1, "donor_rule": "random"},
    {"key": "selected_one", "sharing": "selective", "k": 1, "donor_rule": "prefix"},
    {"key": "pooled_all", "sharing": "pooled", "k": 99, "donor_rule": "all"},
)
REGULARIZED_ARM = {
    "key": "selected_regularized", "sharing": "selective_regularized",
    "k": 1, "donor_rule": "prefix", "matched_full_only": True,
}


class ShareAbstainError(ValueError):
    pass


def _mae(block: dict[str, Any]) -> float | None:
    v = block.get("mae_log")
    return v if isinstance(v, (int, float)) else None


def _rmse(block: dict[str, Any]) -> float | None:
    v = block.get("rmse_log")
    return v if isinstance(v, (int, float)) else None


def _pinball(block: dict[str, Any]) -> float | None:
    v = block.get("mean_quantile_loss")
    return v if isinstance(v, (int, float)) else None


def erratum_from_share_repl(summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Narrative corrections generated from saved PR #14 SUMMARY numbers."""
    src = summary or (read_json(SHARE_REPL_SUMMARY) if SHARE_REPL_SUMMARY.exists() else None)
    if not src:
        return {"status": "unavailable", "reason": "runs/v3_share_repl/SUMMARY.json missing"}
    claims: list[dict[str, Any]] = []
    for rec in src.get("replicates") or []:
        i = rec.get("dataset_index")
        res = rec.get("results") or {}
        lv, zs = res.get("last_value") or {}, res.get("chronos2_zero_shot_history") or {}
        to, po = res.get("chronos2_target_only_history") or {}, res.get("chronos2_pooled_history") or {}
        if i == 1 and all(b.get("status") == "ok" for b in (lv, zs, to, po)):
            claims.append({
                "id": "pr14_d1_chronos_vs_last_value",
                "dataset_index": 1,
                "statement": (
                    "On PR #14 dataset 1, history-only Chronos fine-tunes beat last-value; "
                    "only history-only zero-shot Chronos trails it."
                ),
                "numbers": {
                    "last_value_mae": _mae(lv), "last_value_rmse": _rmse(lv),
                    "chronos2_zero_shot_history_mae": _mae(zs),
                    "chronos2_zero_shot_history_rmse": _rmse(zs),
                    "chronos2_target_only_history_mae": _mae(to),
                    "chronos2_pooled_history_mae": _mae(po),
                },
                "supported": (
                    _mae(to) < _mae(lv) and _mae(po) < _mae(lv) and _mae(zs) > _mae(lv)
                    and _rmse(to) < _rmse(lv) and _rmse(po) < _rmse(lv) and _rmse(zs) > _rmse(lv)
                ),
            })
        sto = res.get("spline_target_only") or {}
        ssel = res.get("spline_selective") or {}
        sreg = res.get("spline_selective_regularized") or {}
        if i == 0 and all(b.get("status") == "ok" for b in (sto, ssel, sreg)):
            claims.append({
                "id": "pr14_d0_spline_selective",
                "dataset_index": 0,
                "statement": (
                    "On dataset 0, unregularized spline selective improves MAE, RMSE, "
                    "and pinball; selective_regularized improves MAE/RMSE but worsens pinball."
                ),
                "numbers": {
                    "target_only": {"mae": _mae(sto), "rmse": _rmse(sto), "pinball": _pinball(sto)},
                    "selective": {"mae": _mae(ssel), "rmse": _rmse(ssel), "pinball": _pinball(ssel)},
                    "selective_regularized": {
                        "mae": _mae(sreg), "rmse": _rmse(sreg), "pinball": _pinball(sreg),
                    },
                },
                "supported": (
                    _mae(ssel) < _mae(sto) and _rmse(ssel) < _rmse(sto)
                    and _pinball(ssel) < _pinball(sto)
                    and _mae(sreg) < _mae(sto) and _rmse(sreg) < _rmse(sto)
                    and _pinball(sreg) > _pinball(sto)
                ),
            })
    return {
        "status": "ok",
        "source": str(SHARE_REPL_SUMMARY),
        "labels": list(LABELS),
        "claims": claims,
        "all_supported": all(c.get("supported") for c in claims) and len(claims) == 2,
    }


def audit_existing_selection(root: Path | None = None) -> dict[str, Any]:
    """Read-only audit of the existing selective-sharing mechanism. No training."""
    base = Path(root or ".")
    cells: list[dict[str, Any]] = []
    for path in sorted((base / SHARE_REPL_CELLS).glob("d*/trains/t0/cells/*/splits/donor_order.json")):
        cell = path.parents[1]
        donor = read_json(path)
        man_p = cell / "splits" / "target_manifest.json"
        hist_p = cell / "models" / "spline" / "training_history.json"
        cells.append({
            "cell": str(cell.relative_to(base)),
            "donor_order": donor,
            "target_manifest": read_json(man_p) if man_p.exists() else None,
            "spline_history": (
                {
                    "alpha": (read_json(hist_p) or {}).get("alpha"),
                    "validation_rmse": (read_json(hist_p) or {}).get("validation_rmse"),
                    "covariance_fit_on": (read_json(hist_p) or {}).get("covariance_fit_on"),
                    "lambda_growth": (read_json(hist_p) or {}).get("lambda_growth"),
                    "lambda_treatment": (read_json(hist_p) or {}).get("lambda_treatment"),
                    "n_params": (read_json(hist_p) or {}).get("n_params"),
                    "selective_sharing": (read_json(hist_p) or {}).get("selective_sharing"),
                } if hist_p.exists() else None
            ),
        })
    return {
        "labels": list(LABELS),
        "no_training": True,
        "mechanism": {
            "donor_inclusion": (
                "Eligible donor STUDYIDs are donor-species studies that pass cohort QC. "
                "Order is sorted unique STUDYID (donor_study_order). Selected = first k. "
                "Not metadata similarity, not training/validation scores, not learned weights."
            ),
            "code": [
                "src/send_bw/data/splits.py:donor_study_order",
                "src/send_bw/data/splits.py:compose_train_with_donors",
                "src/send_bw/pipeline.py:Run.prepare",
                "src/send_bw/sharing.py:SELECTIVE_V1 / SELECTIVE_REGULARIZED",
                "src/send_bw/models/spline.py:SplineModel.fit",
                "src/send_bw/models/transformer.py:TransformerModel.fit",
                "src/send_bw/data/features.py:FeatureSpec.fit",
            ],
            "regularization_not_inclusion": (
                "selective vs selective_regularized changes feature sharing and ridge/L2, "
                "not which donor IDs are chosen. Both used k=1 prefix SYNA-D001 on PR #14."
            ),
            "zero_donors_available": True,
            "zero_donors_path": "context_specific + n_donor_studies=0",
            "target_only_is_not_single_study": (
                "Target-only trains all target-species TRAIN studies (PR #14 d0: 7 studies). "
                "It is not a single-study model."
            ),
            "test_outcomes_used_for_selection": False,
            "missing_provenance": [
                "PR #14 target_only donor_order.studyids is empty: eligible bank was not "
                "recorded when sharing=context_specific (donor_mode false).",
                "No stored similarity scores or learned donor weights exist to reconstruct.",
            ],
            "selective_v1": SELECTIVE_V1,
            "selective_regularized": SELECTIVE_REGULARIZED,
        },
        "pr14_cells": cells,
        "erratum": erratum_from_share_repl(),
    }


def locked_budget(cfg: dict[str, Any]) -> dict[str, Any]:
    ev = cfg.get("share_abstain") or {}
    n_ds = int(ev.get("n_dataset_seeds") or 2)
    n_tr = int(ev.get("n_train_seeds") or 1)
    g0 = int(cfg["generate_seed"])
    t0s = int(cfg["train_seed"])
    g_stride = int(ev.get("generate_seed_stride") or 1009)
    t_stride = int(ev.get("train_seed_stride") or 17)
    banks = list(ev.get("banks") or ["matched", "both"])
    densities = list(ev.get("densities") or ["full", "sparse"])
    sparse_n = int(ev.get("sparse_target_train_n") or 4)
    n_core = n_ds * n_tr * len(banks) * len(densities) * len(POLICY_ARMS)
    n_reg = n_ds * n_tr  # matched + full only
    n_cells = n_core + n_reg
    n_model_fits = n_cells * 2  # spline + transformer; abstain reuses fits
    return {
        "n_dataset_seeds": n_ds,
        "n_train_seeds": n_tr,
        "generate_seeds": [g0 + i * g_stride for i in range(n_ds)],
        "train_seeds": [t0s + j * t_stride for j in range(n_tr)],
        "donor_rng_seed": int(ev.get("donor_rng_seed") or 91001),
        "banks": banks,
        "densities": densities,
        "sparse_target_train_n": sparse_n,
        "sparse_kind": "sparse_target_study_training",
        "not_sparse_encoder": True,
        "encoder_window_day": float(ev.get("encoder_window_day") or WINDOW),
        "policy_arms": [a["key"] for a in POLICY_ARMS] + ["selected_abstain", "selected_regularized"],
        "n_trained_cells": n_cells,
        "n_model_fits": n_model_fits,
        "additional_fits_for_abstention": 0,
        "abstention_reuses": ["target_only", "selected_one"],
        "abstention": {
            "metric": ABSTAIN_METRIC,
            "min_improvement": ABSTAIN_MIN_IMPROVEMENT,
            "tie": "target_only",
            "threshold_from_test": False,
        },
        "estimated_runtime_seconds": ev.get("estimated_runtime_seconds") or {
            "point": 180, "lower": 120, "upper": 480,
        },
        "compute_cap_seconds": int(ev.get("compute_cap_seconds") or 600),
        "execute": bool(ev.get("execute")),
        "training_seed_variability_estimated": n_tr > 1,
        "dataset_variability_estimated": n_ds > 1,
        "prefix_rule": PREFIX_RULE,
        "prefix_rule_label": PREFIX_LABEL,
        "prefix_is_learned": False,
        "prefix_is_similarity_based": False,
        "prefix_is_validation_ranked": False,
        "confirmatory": False,
    }


def science_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """Scientific settings only. execute / output_dir / run_name are excluded."""
    ev = dict(cfg.get("share_abstain") or {})
    for drop in ("execute", "proposal_config"):
        ev.pop(drop, None)
    models = cfg.get("models") or {}
    return {
        "generate_seed": cfg.get("generate_seed"),
        "train_seed": cfg.get("train_seed"),
        "share_abstain": ev,
        "synthetic": {k: (cfg.get("synthetic") or {}).get(k) for k in
                      ("generator", "n_studies", "n_other_species_studies",
                       "n_compounds", "n_per_sex")},
        "cohort": {k: (cfg.get("cohort") or {}).get(k) for k in
                   ("species", "donor_species", "study_classes", "max_day")},
        "models": {
            "spline": {k: (models.get("spline") or {}).get(k)
                       for k in ("n_bootstrap", "oof_folds")},
            "transformer": {k: (models.get("transformer") or {}).get(k)
                            for k in ("n_members", "max_epochs", "patience",
                                      "d_model", "n_heads", "n_layers")},
            "optional": models.get("optional"),
            "bayesian_enabled": (models.get("bayesian") or {}).get("enabled"),
        },
        "evaluation": dict(cfg.get("evaluation") or {}),
        "budget": {k: locked_budget(cfg).get(k) for k in
                   ("n_trained_cells", "n_model_fits", "additional_fits_for_abstention",
                    "generate_seeds", "train_seeds", "banks", "densities",
                    "sparse_target_train_n", "compute_cap_seconds",
                    "estimated_runtime_seconds", "donor_rng_seed")},
    }


def execute_v3_share_abstain(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = Path(cfg.get("output_dir") or "runs/v3_share_abstain")
    if parent.resolve() in {Path(p).resolve() for p in PRESERVED}:
        raise ShareAbstainError("refusing to write into a preserved artifact directory")
    if (cfg.get("source") or {}).get("type") == "snowflake":
        raise ShareAbstainError("real-data access is off")
    if (cfg.get("design") or {}).get("request"):
        raise ShareAbstainError("design optimization stays off")
    ev = cfg.get("share_abstain") or {}
    if ev.get("retune_after_test"):
        raise ShareAbstainError("retune_after_test is forbidden")
    if ev.get("predose") or ev.get("bayesian") or ev.get("tft") or ev.get("chronos"):
        raise ShareAbstainError("TFT, Chronos, Bayesian, and predose stay off")
    budget = locked_budget(cfg)
    if budget["execute"] and parent.resolve() == Path(PROPOSAL_DIR).resolve():
        raise ShareAbstainError(
            "execution must not overwrite the unexecuted proposal directory "
            f"{PROPOSAL_DIR}; use a separate approved output_dir"
        )
    parent.mkdir(parents=True, exist_ok=True)
    hashes_before = snapshot_protected()
    write_json(parent / "protected_hashes_before.json", hashes_before)
    proposal_cfg = load_yaml(PROPOSAL_CONFIG) if Path(PROPOSAL_CONFIG).exists() else None
    proposal_hash = cache_key(proposal_cfg) if proposal_cfg is not None else None
    science = science_fingerprint(cfg)
    proposal_science = science_fingerprint(proposal_cfg) if proposal_cfg is not None else None
    science_match = science == proposal_science
    if budget["execute"] and proposal_cfg is not None and not science_match:
        raise ShareAbstainError(
            "approved execution settings differ from the frozen proposal; stop for review"
        )
    audit = audit_existing_selection()
    write_json(parent / "selective_audit.json", audit)
    erratum = audit.get("erratum") or {}
    if not budget["execute"]:
        _write_erratum(erratum)
        _write_audit_md(audit)
    lock = {
        "config_path": str(config_path), "config_hash": cache_key(cfg),
        "git_revision": git_revision(), **budget,
        "proposal": {
            "path": PROPOSAL_CONFIG,
            "config_hash": proposal_hash,
            "output_dir": PROPOSAL_DIR,
            "execute": False,
            "relationship": (
                "approved execution of the unexecuted frozen proposal"
                if budget["execute"] else "unexecuted frozen proposal"
            ),
            "science_fingerprint_match": science_match,
        },
        "science_fingerprint": science,
        "retune_after_test": False, "real_data": False, "design": False,
        "new_architectures": False, "fabricate_encoder_history": False,
        "selector_sees_generator_labels": False,
        "prefix_rule_label": PREFIX_LABEL,
        "labels": list(LABELS), "preserved": list(PRESERVED) + [PROPOSAL_DIR],
    }
    write_json(parent / "freeze_lock.json", lock)
    write_json(parent / "versions.json", collect_versions())

    if not budget["execute"]:
        out = {
            "status": "locked_not_executed",
            "labels": list(LABELS),
            "output_dir": str(parent),
            "freeze_lock": lock,
            "audit": {"n_pr14_cells": len(audit.get("pr14_cells") or [])},
            "erratum": erratum,
            "execute_command": (
                "send-bw v3-share-abstain --config configs/v3_share_abstain.approved.yaml"
            ),
            "execute_requires": (
                "separate approved config with execute:true; do not edit the frozen proposal"
            ),
            "real_data_accessed": False,
            "authorization_present": bool(discover_authorization(None).get("authorized")),
            "design_run": False,
            "new_architectures": False,
        }
        return _finalize(parent, hashes_before, out, report="lock")

    return _run_locked(cfg, parent, lock, hashes_before, force)


def _finalize(parent: Path, hashes_before: dict[str, Any], out: dict[str, Any],
              report: str) -> dict[str, Any]:
    hashes_after = snapshot_protected()
    cmp = compare_snapshots(hashes_before, hashes_after)
    write_json(parent / "protected_hashes_after.json", hashes_after)
    write_json(parent / "protected_hashes_compare.json", cmp)
    out["protected"] = cmp
    write_json(parent / "SUMMARY.json", out)
    if report == "lock":
        _write_lock_report(parent, out)
    else:
        _write_exec_report(parent, out)
    if not cmp["unchanged"]:
        raise ShareAbstainError("protected artifacts changed: " + ", ".join(cmp["changed"]))
    return out


def _run_locked(cfg: dict[str, Any], parent: Path, lock: dict[str, Any],
                hashes_before: dict[str, Any], force: bool) -> dict[str, Any]:
    window = float(lock["encoder_window_day"])
    cap = RuntimeCap(int(lock["compute_cap_seconds"]))
    failures: list[dict[str, Any]] = []
    replicates: list[dict[str, Any]] = []
    status = "executed"
    try:
        for i, gen in enumerate(lock["generate_seeds"]):
            for j, trn in enumerate(lock["train_seeds"]):
                for bank in lock["banks"]:
                    cap.check(f"before_generate_d{i}_{bank}")
                    bcfg = _bank_cfg(cfg, parent, i, gen, bank)
                    _run_from_cfg(bcfg).generate(force=force)
                    cap.check(f"after_generate_d{i}_{bank}")
                    for dens in lock["densities"]:
                        cap.check(f"before_slice_d{i}_{bank}_{dens}")
                        sparse_n = (int(lock["sparse_target_train_n"]) if dens == "sparse" else None)
                        rec = _run_slice(
                            cfg, parent, i, j, gen, trn, bank, dens, sparse_n, window,
                            lock, force, failures, cap)
                        replicates.append(rec)
                        cap.check(f"after_slice_d{i}_{bank}_{dens}")
    except RuntimeCapExceeded as e:
        status = "timeout"
        failures.append({
            "dataset_index": None, "bank": None, "density": None,
            "model": None, "reason": str(e), "scored": False, "timeout": True,
            "where": e.where,
        })
    out = {
        "status": status,
        "labels": list(LABELS) + [
            "Share-or-abstain diagnostic.",
            "Prefix donor choice is STUDYID sort, not learned ranking.",
            "Two dataset seeds and one training seed are not confirmatory.",
        ],
        "output_dir": str(parent),
        "freeze_lock": lock,
        "replicates": replicates,
        "failures": failures,
        "aggregated_across_incomplete": False,
        "confirmatory": False,
        "real_data_accessed": False,
        "design_run": False,
        "predose_evaluated": False,
        "new_architectures": False,
        "seconds_total": cap.elapsed(),
        "compute_cap_seconds": cap.cap,
        "compute_cap_hit": cap.hit,
        "compute_cap_where": cap.where,
        "prefix_versus_random": [
            {"dataset_index": r.get("dataset_index"), "bank": r.get("bank"),
             "density": r.get("density"), **(r.get("prefix_versus_random") or {})}
            for r in replicates
        ],
        "abstention_counts": _abstention_counts(replicates),
    }
    return _finalize(parent, hashes_before, out, report="exec")


def _checkpoint_ref(run, name: str) -> dict[str, Any]:
    dest = run.p("models", name)
    hist_p = dest / "training_history.json"
    hist = read_json(hist_p) if hist_p.exists() else {}
    return {
        "dir": str(dest),
        "training_history": str(hist_p) if hist_p.exists() else None,
        "ridge_alpha": hist.get("alpha"),
        "validation_rmse": hist.get("validation_rmse") or hist.get("best_val_rmse_member0"),
        "source": "training_validation",
        "test_outcomes_used": False,
    }


def _n_val_studies(block: dict[str, Any]) -> int | None:
    if block.get("n_studies") is not None:
        return int(block["n_studies"])
    mae = block.get("mae_log")
    if isinstance(mae, dict) and mae.get("n_studies") is not None:
        return int(mae["n_studies"])
    return None


def _run_slice(cfg, parent, i, j, gen, trn, bank, dens, sparse_n, window,
               lock, force, failures, cap: RuntimeCap) -> dict[str, Any]:
    from send_bw.pipeline import model_class

    cells: dict[str, Any] = {}
    hashes: dict[str, Any] = {}
    reuse = None
    arms = list(POLICY_ARMS)
    if bank == "matched" and dens == "full":
        arms = arms + [REGULARIZED_ARM]
    for arm in arms:
        cap.check(f"before_train_d{i}_{bank}_{dens}_{arm['key']}")
        spec = {
            "name": f"d{i}_t{j}_{bank}_{dens}_{arm['key']}",
            "dataset_index": i, "train_index": j, "bank": bank,
            "sharing": arm["sharing"], "k": int(arm["k"]),
            "sparse_target_train_n": sparse_n,
            "generate_seed": gen, "train_seed": trn,
            "models": ["spline", "transformer"],
            "donor_rule": arm["donor_rule"],
            "donor_rng_seed": lock["donor_rng_seed"] if arm["donor_rule"] == "random" else None,
        }
        ccfg = _cell_cfg(cfg, parent, spec, generate_seed=gen, train_seed=trn,
                         models=spec["models"], sparse_target_train_n=sparse_n,
                         reuse_target_from=reuse)
        ccfg["evaluation"]["tasks"] = ["early_history"]
        ccfg["evaluation"]["early_history_window_day"] = window
        ccfg["design"] = {}
        run = _run_from_cfg(ccfg)
        run.ingest(force=force)
        run.prepare(force=force)
        run.train(["spline", "transformer"], force=force)
        cells[arm["key"]] = run
        man = read_json(run.p("splits", "target_manifest.json"))
        hashes[arm["key"]] = {n: (man.get(n) or {}).get("hash") for n in
                              ("validation", "test", "all_target")}
        if reuse is None:
            reuse = str(run.p("splits", "x").parent)
        cap.check(f"after_train_d{i}_{bank}_{dens}_{arm['key']}")
    test_hash = hashes["target_only"]["test"]
    val_hash = hashes["target_only"]["validation"]
    if any(h.get("test") != test_hash or h.get("validation") != val_hash for h in hashes.values()):
        raise ShareAbstainError("target holdout hashes differ across sharing arms")

    prep = load_cell_prepared(Path(cells["target_only"].dir))
    test = prep["data"]["test"]
    val = prep["data"]["validation"]
    pred_dir = parent / f"repl/d{i}_t{j}_{bank}_{dens}/predictions"
    dec_dir = parent / f"repl/d{i}_t{j}_{bank}_{dens}/decisions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    val_results: dict[str, Any] = {}
    donor_records: dict[str, Any] = {
        key: read_json(run.p("splits", "donor_order.json")) for key, run in cells.items()
    }

    # Validation-only abstention decisions are written before any test scoring.
    for name in ("spline", "transformer"):
        cap.check(f"before_val_{name}_d{i}_{bank}_{dens}")
        for key in ("target_only", "selected_one"):
            mk = f"{name}_{key}"
            try:
                model = model_class(name).load(cells[key].p("models", name))
                val_results[mk] = _score_gaussian(model, val, window, None)
            except Exception as e:  # noqa: BLE001
                val_results[mk] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
                failures.append({"dataset_index": i, "bank": bank, "density": dens,
                                 "model": mk, "reason": str(e), "scored": False,
                                 "split": "validation"})
        to_key, sel_key = f"{name}_target_only", f"{name}_selected_one"
        v_to, v_sel = val_results.get(to_key) or {}, val_results.get(sel_key) or {}
        dec = decide_abstention(
            val_score(v_to), val_score(v_sel), n_val_studies=_n_val_studies(v_to))
        chosen = to_key if dec["decision"] == "target_only" else sel_key
        record = {
            "model": name,
            "saved_before_test": True,
            "test_scored": False,
            "donor_rule_candidate": PREFIX_RULE,
            "donor_rule_label": PREFIX_LABEL,
            "learned": False,
            "similarity_based": False,
            "validation_ranked_donors": False,
            "target_only_donors": (donor_records.get("target_only") or {}).get("selected") or [],
            "candidate_donors": (donor_records.get("selected_one") or {}).get("selected") or [],
            "checkpoint_target_only": _checkpoint_ref(cells["target_only"], name),
            "checkpoint_candidate": _checkpoint_ref(cells["selected_one"], name),
            "validation": {
                "metric": ABSTAIN_METRIC,
                "target_only": _compact(v_to) if v_to.get("status") == "ok" else v_to,
                "candidate": _compact(v_sel) if v_sel.get("status") == "ok" else v_sel,
                "n_val_studies": _n_val_studies(v_to),
            },
            "decision": {**dec, "chosen_arm": chosen},
        }
        write_json(dec_dir / f"{name}_abstain.json", record)

    n_rand = len((donor_records.get("random_one") or {}).get("selected") or [])
    n_sel = len((donor_records.get("selected_one") or {}).get("selected") or [])
    if n_rand != n_sel:
        raise ShareAbstainError(f"random/prefix donor counts differ: {n_rand} vs {n_sel}")
    overlap = prefix_random_overlap(
        (donor_records.get("selected_one") or {}).get("selected") or [],
        (donor_records.get("random_one") or {}).get("selected") or [],
    )

    for key, run in cells.items():
        for name in ("spline", "transformer"):
            mk = f"{name}_{key}"
            cap.check(f"before_test_{mk}_d{i}_{bank}_{dens}")
            t1 = time.time()
            try:
                model = model_class(name).load(run.p("models", name))
                sc = _score_gaussian(model, test, window, pred_dir / f"{mk}.csv")
                sc["seconds"] = time.time() - t1
                results[mk] = sc
            except Exception as e:  # noqa: BLE001
                results[mk] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
                failures.append({"dataset_index": i, "bank": bank, "density": dens,
                                 "model": mk, "reason": str(e), "scored": False,
                                 "split": "test"})

    abstain: dict[str, Any] = {}
    for name in ("spline", "transformer"):
        rec_path = dec_dir / f"{name}_abstain.json"
        record = read_json(rec_path)
        chosen = (record.get("decision") or {}).get("chosen_arm")
        src = results.get(chosen) or {"status": "failed", "reason": "chosen arm missing"}
        chosen_block = _compact(src) if src.get("status") == "ok" else src
        results[f"{name}_selected_abstain"] = {
            **chosen_block, "chosen_arm": chosen,
            "abstention": record.get("decision"),
            "decision_path": str(rec_path),
        }
        abstain[name] = {**(record.get("decision") or {}), "chosen_arm": chosen,
                         "saved_before_test": True, "decision_path": str(rec_path)}
        if src.get("status") == "ok":
            src_csv = pred_dir / f"{chosen}.csv"
            dest = pred_dir / f"{name}_selected_abstain.csv"
            if src_csv.exists():
                dest.write_bytes(src_csv.read_bytes())
        record["test_scored"] = True
        write_json(rec_path.with_name(f"{name}_abstain_after_test.json"), record)

    transfer = {}
    for name in ("spline", "transformer"):
        to = results.get(f"{name}_target_only") or {}
        for other in ("random_one", "selected_one", "pooled_all", "selected_abstain",
                      "selected_regularized"):
            mk = f"{name}_{other}"
            if mk not in results:
                continue
            transfer[mk] = _sign_rows(paired_transfer(to, results.get(mk) or {}))

    return {
        "dataset_index": i, "train_index": j, "generate_seed": gen, "train_seed": trn,
        "bank": bank, "density": dens, "sparse_target_train_n": sparse_n,
        "target_holdout_hashes": hashes,
        "donor_records": donor_records,
        "random_selected_equal_k": n_rand == n_sel and n_rand == 1,
        "prefix_versus_random": overlap,
        "prefix_rule_label": PREFIX_LABEL,
        "abstention": abstain,
        "results": {k: _compact(v) if isinstance(v, dict) and "mae_log" in v
                    and isinstance(v.get("mae_log"), dict) else v
                    for k, v in results.items()},
        "paired_transfer": transfer,
    }


def _abstention_counts(replicates: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("spline", "transformer"):
        rows = [(r.get("abstention") or {}).get(name) or {} for r in replicates]
        n = len(rows)
        n_to = sum(1 for d in rows if d.get("decision") == "target_only")
        n_share = sum(1 for d in rows if d.get("decision") == "share")
        out[name] = {
            "n_decisions": n,
            "n_target_only": n_to,
            "n_share": n_share,
            "target_only_over_n": f"{n_to}/{n}",
            "share_over_n": f"{n_share}/{n}",
        }
    n = 2 * len(replicates)
    n_to = sum(out[m]["n_target_only"] for m in ("spline", "transformer"))
    n_share = sum(out[m]["n_share"] for m in ("spline", "transformer"))
    out["overall"] = {
        "n_decisions": n,
        "n_target_only": n_to,
        "n_share": n_share,
        "target_only_over_n": f"{n_to}/{n}",
        "share_over_n": f"{n_share}/{n}",
    }
    return out


def _write_erratum(erratum: dict[str, Any]) -> None:
    lines = [
        "# PR #14 erratum",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data "
        "validation. No operational recommendation.**",
        "",
        "Generated from `runs/v3_share_repl/SUMMARY.json`. Frozen numerical tables "
        "in `runs/v3_share_repl/RESULTS.md` are unchanged.",
        "",
    ]
    if erratum.get("status") != "ok":
        lines += [f"Unavailable: {erratum.get('reason')}", ""]
        Path("docs/v3/share_repl_ERRATUM.md").write_text("\n".join(lines), encoding="utf-8")
        return
    for c in erratum.get("claims") or []:
        lines += [
            f"## {c.get('id')}",
            "",
            c.get("statement") or "",
            "",
            f"- supported by saved numbers: `{c.get('supported')}`",
            f"- numbers: `{c.get('numbers')}`",
            "",
        ]
    Path("docs/v3/share_repl_ERRATUM.md").write_text("\n".join(lines), encoding="utf-8")


def _write_audit_md(audit: dict[str, Any]) -> None:
    mech = audit.get("mechanism") or {}
    lines = [
        "# Selective sharing audit (no retraining)",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data "
        "validation. No operational recommendation.**",
        "",
        "Inspected existing code and PR #14 artifacts. No training was rerun. "
        "Missing provenance is marked unavailable.",
        "",
        "## Mechanism",
        "",
        mech.get("donor_inclusion") or "",
        "",
        "Code:",
        "",
    ]
    for c in mech.get("code") or []:
        lines.append(f"- `{c}`")
    lines += [
        "",
        f"- Regularization vs inclusion: {mech.get('regularization_not_inclusion')}",
        f"- Zero donors available: `{mech.get('zero_donors_available')}` via `{mech.get('zero_donors_path')}`",
        f"- Target-only is not single-study: {mech.get('target_only_is_not_single_study')}",
        f"- Test outcomes used for selection: `{mech.get('test_outcomes_used_for_selection')}`",
        "",
        "## Missing provenance",
        "",
    ]
    for m in mech.get("missing_provenance") or []:
        lines.append(f"- {m}")
    lines += [
        "",
        "## PR #14 donor cells",
        "",
        "| cell | sharing | k | eligible | selected |",
        "|---|---|---|---|---|",
    ]
    for cell in audit.get("pr14_cells") or []:
        d = cell.get("donor_order") or {}
        lines.append(
            f"| `{cell.get('cell')}` | {d.get('sharing')} | {d.get('k')} | "
            f"{d.get('studyids')} | {d.get('selected')} |"
        )
    lines += [
        "",
        "Target-only `studyids: []` means the eligible bank was not recorded on that cell; "
        "the bank is available on pooled/selective cells (SYNA-D001..D003) and is not reconstructed "
        "as a historical selection score.",
        "",
        f"Selective v1: `{mech.get('selective_v1')}`",
        "",
        f"Selective regularized: `{mech.get('selective_regularized')}`",
        "",
    ]
    Path("docs/v3/selective_sharing_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("runs/v3_share_abstain").mkdir(parents=True, exist_ok=True)
    Path("runs/v3_share_abstain/AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_lock_report(parent: Path, out: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    lines = [
        "# Share-or-abstain — locked, not executed",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data "
        "validation. No operational recommendation.**",
        "",
        "PR #14 remains a fixed-donor experiment. This lock is a separate "
        "share-or-abstain policy proposal. Benchmark training has not been run.",
        "",
        f"- n_dataset_seeds: `{lock.get('n_dataset_seeds')}`; "
        f"n_train_seeds: `{lock.get('n_train_seeds')}`",
        f"- generate_seeds: `{lock.get('generate_seeds')}`; train_seeds: `{lock.get('train_seeds')}`",
        f"- donor_rng_seed: `{lock.get('donor_rng_seed')}`",
        f"- banks: `{lock.get('banks')}`; densities: `{lock.get('densities')}`",
        f"- sparse_target_train_n: `{lock.get('sparse_target_train_n')}` "
        f"(train studies only; val/test unchanged; not a shorter encoder)",
        f"- n_trained_cells: `{lock.get('n_trained_cells')}`; "
        f"n_model_fits: `{lock.get('n_model_fits')}`; "
        f"additional_fits_for_abstention: `{lock.get('additional_fits_for_abstention')}`",
        f"- estimated_runtime_seconds: `{lock.get('estimated_runtime_seconds')}`",
        f"- compute_cap_seconds: `{lock.get('compute_cap_seconds')}`",
        f"- execute: `{lock.get('execute')}`",
        "",
        "Abstention metric: target-validation study-balanced true RMSE "
        f"(`{ABSTAIN_METRIC}`), min_improvement "
        f"`{ABSTAIN_MIN_IMPROVEMENT}`, ties → target-only.",
        "",
        f"Protected artifacts unchanged: `{(out.get('protected') or {}).get('unchanged')}`",
        "",
        f"Proposed execution (requires flipping `share_abstain.execute` to true): "
        f"`{out.get('execute_command')}`",
        "",
    ]
    text = "\n".join(lines)
    (parent / "RESULTS.md").write_text(text, encoding="utf-8")
    Path("docs/v3/share_abstain_RESULTS.md").write_text(text, encoding="utf-8")


def _write_exec_report(parent: Path, out: dict[str, Any]) -> None:
    lock = out.get("freeze_lock") or {}
    counts = out.get("abstention_counts") or {}
    lines = [
        "# Share-or-abstain — executed (not confirmatory)",
        "",
        "> **Synthetic diagnostic. Not design qualification. No real-data "
        "validation. No operational recommendation.**",
        "",
        "Two dataset seeds and one training seed are **not confirmatory**. "
        "Prefix donor choice is a STUDYID sort, not learned / similarity / "
        "validation-ranked selection. The unexecuted frozen proposal is unchanged.",
        "",
        f"- proposal: `{(lock.get('proposal') or {}).get('path')}` "
        f"hash `{(lock.get('proposal') or {}).get('config_hash')}`",
        f"- approved config hash: `{lock.get('config_hash')}`",
        f"- science_fingerprint_match: `{(lock.get('proposal') or {}).get('science_fingerprint_match')}`",
        f"- generate_seeds: `{lock.get('generate_seeds')}`; train_seeds: `{lock.get('train_seeds')}`",
        f"- seconds_total: `{out.get('seconds_total')}`; "
        f"cap: `{out.get('compute_cap_seconds')}`; hit: `{out.get('compute_cap_hit')}`",
        f"- status: `{out.get('status')}`; confirmatory: False",
        f"- prefix rule: `{PREFIX_LABEL}`",
        "",
        "## Abstention counts (validation-only decisions)",
        "",
        f"- spline: `{(counts.get('spline') or {}).get('target_only_over_n')}` target-only, "
        f"`{(counts.get('spline') or {}).get('share_over_n')}` share",
        f"- transformer: `{(counts.get('transformer') or {}).get('target_only_over_n')}` target-only, "
        f"`{(counts.get('transformer') or {}).get('share_over_n')}` share",
        f"- overall: `{(counts.get('overall') or {}).get('target_only_over_n')}` target-only, "
        f"`{(counts.get('overall') or {}).get('share_over_n')}` share",
        "",
        "## Prefix versus random one-donor overlap",
        "",
        "| dataset | bank | density | prefix | random | intersection | equal |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in out.get("prefix_versus_random") or []:
        lines.append(
            f"| {row.get('dataset_index')} | {row.get('bank')} | {row.get('density')} | "
            f"{row.get('prefix_ids')} | {row.get('random_ids')} | "
            f"{row.get('intersection')} | {row.get('equal')} |"
        )
    lines += ["", "## Replicates", ""]
    for rec in out.get("replicates") or []:
        lines += [
            f"### d{rec.get('dataset_index')} t{rec.get('train_index')} "
            f"{rec.get('bank')} {rec.get('density')}",
            "",
            f"prefix-versus-random: `{rec.get('prefix_versus_random')}`",
            "",
            "| model | MAE | RMSE | status |",
            "|---|---|---|---|",
        ]
        for key, b in (rec.get("results") or {}).items():
            if not isinstance(b, dict):
                continue
            if b.get("status") not in {None, "ok"} and "mae_log" not in b:
                lines.append(f"| {key} | — | — | {b.get('status')}: {(b.get('reason') or '')[:80]} |")
                continue
            lines.append(
                f"| {key} | {_fmt_num(b.get('mae_log'))} | {_fmt_num(b.get('rmse_log'))} | "
                f"{b.get('status') or 'ok'} |"
            )
        lines += [
            "",
            "Validation decisions (saved before test):",
            "",
            f"`{rec.get('abstention')}`",
            "",
            "Paired test Δ vs target-only (Δ>0 harmful):",
            "",
            "| family | metric | target-only | other | Δ | sign |",
            "|---|---|---|---|---|---|",
        ]
        for fam, block in (rec.get("paired_transfer") or {}).items():
            for r in block.get("rows") or []:
                lines.append(
                    f"| {fam} | {r.get('metric')} | {r.get('target_only')} | {r.get('pooled')} | "
                    f"{r.get('delta_pooled_minus_target')} | {r.get('sign')} |"
                )
        lines.append("")
    lines += [
        "## Failures (not aggregated as complete)",
        "",
        "- aggregated_across_incomplete: False",
        f"- failures: `{out.get('failures')}`",
        "",
        "## Guardrails",
        "",
        f"- real_data_accessed: {out.get('real_data_accessed')}",
        f"- design_run: {out.get('design_run')}",
        f"- protected unchanged: `{(out.get('protected') or {}).get('unchanged')}`",
        "- confirmatory: False",
        "",
    ]
    text = "\n".join(lines)
    (parent / "RESULTS.md").write_text(text, encoding="utf-8")
    Path("docs/v3/share_abstain_run_RESULTS.md").write_text(text, encoding="utf-8")


def _fmt_num(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if v is None:
        return "n/a"
    return str(v)

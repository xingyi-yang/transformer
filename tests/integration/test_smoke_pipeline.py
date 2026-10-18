"""Complete CPU smoke through the CLI with real spline/transformer training (tiny, nonconclusive)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from send_bw.cli import main

ROOT = Path(__file__).resolve().parents[2]


def _config(tmp: Path, **models) -> Path:
    out = tmp / "run"
    cfg = {
        "run_name": "it_smoke", "output_dir": str(out), "seed": 5,
        "synthetic": {"enabled": True, "out_dir": str(out / "synthetic"), "n_studies": 12, "n_compounds": 6,
                      "n_per_sex": [3, 4]},
        "mapping": str(ROOT / "configs" / "mappings" / "example_send.yaml"),
        "source": {"type": "local", "root": str(out / "synthetic" / "send"),
                   "metadata_dir": str(out / "synthetic" / "metadata")},
        "cohort": {"species": "RAT", "max_day": 28},
        "splits": {"validation": 0.2, "test": 0.2, "compound_holdout_auto": 1},
        "models": {"spline": {"enabled": True, "n_bootstrap": 2},
                   "transformer": {"enabled": True, "n_members": 1, "max_epochs": 8, "patience": 3},
                   "bayesian": {"enabled": False}, **models},
        "evaluation": {"n_draws": 40},
        "design": {"request": str(ROOT / "configs" / "new_study.smoke.yaml"), "proposal_models": ["spline"],
                   "replicates_proposal": 30, "replicates_final": 30, "n_finalists": 1},
    }
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("it")
    cfg = _config(tmp)
    assert main(["run-all", "--config", str(cfg)]) == 0
    return tmp, cfg, tmp / "run"


def test_artifacts_written(smoke_run):
    _, _, run = smoke_run
    for rel in ["manifest.json", "canonical/bw_obs.parquet", "canonical/provenance.json", "qc/data_dictionary.md",
                "qc/mapping_report.json", "qc/quarantine.csv", "qc/exclusions.csv", "qc/completeness.csv",
                "splits/splits.csv", "splits/splits.json", "features/feature_spec.json", "cohort/obs.parquet",
                "models/spline/model.json", "models/transformer/members.pt", "models/spline/training_history.json",
                "models/optional_models_status.json", "evaluation/metrics.json", "design/design_summary.json",
                "design/request_validation.json", "report/report.md", "report/report.html"]:
        assert (run / rel).exists(), rel


def test_report_labels_and_honest_status(smoke_run):
    _, _, run = smoke_run
    md = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC PROOF OF CONCEPT" in md
    assert "nonconclusive" in md.lower()
    assert "### bayesian: not_run" in md
    opt = json.loads((run / "models" / "optional_models_status.json").read_text(encoding="utf-8"))
    assert all(v["status"] == "not_run" for v in opt.values())


def test_manifest_has_provenance(smoke_run):
    _, _, run = smoke_run
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert man["seed"] == 5 and man["software"]["numpy"]
    assert {"generate", "ingest", "prepare", "train_spline", "train_transformer", "evaluate", "design"} <= set(man["stages"])
    assert man["stages"]["generate"]["send_hash"]


def test_truth_not_in_model_inputs(smoke_run):
    _, _, run = smoke_run
    spec = json.loads((run / "features" / "feature_spec.json").read_text(encoding="utf-8"))
    text = json.dumps(spec).lower()
    assert "truth" not in text and "exdose" not in text


def test_rerun_uses_cache_and_validate_request(smoke_run, capsys):
    _, cfg, run = smoke_run
    assert main(["prepare", "--config", str(cfg)]) == 0
    assert main(["validate-request", "--model", str(run), "--request",
                 str(ROOT / "configs" / "new_study.smoke.yaml")]) == 0
    assert '"supported"' in capsys.readouterr().out


def test_propose_design_from_serialized_artifacts(smoke_run, tmp_path):
    _, _, run = smoke_run
    out = tmp_path / "prop"
    rc = main(["propose-design", "--model", str(run / "models"), "--request",
               str(ROOT / "configs" / "new_study.smoke.yaml"), "--out", str(out), "--replicates", "20", "20"])
    assert rc == 0 and (out / "design_summary.json").exists()


def test_unsupported_request_refused(smoke_run, tmp_path):
    _, _, run = smoke_run
    req = yaml.safe_load((ROOT / "configs" / "new_study.smoke.yaml").read_text(encoding="utf-8"))
    req["species"] = "DOG"
    p = tmp_path / "dog.yaml"
    p.write_text(yaml.safe_dump(req), encoding="utf-8")
    assert main(["validate-request", "--model", str(run), "--request", str(p)]) == 2

"""v3 core benchmark: selective sharing, sparse history, donor hetero, uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from send_bw.data.features import FeatureSpec
from send_bw.data.generate import generate_dataset
from send_bw.data.qc import CohortConfig
from send_bw.evaluation.sparse import encoder_history
from send_bw.evaluation.transfer import build_transfer_table
from send_bw.evaluation.uncertainty import bayesian_gate, inspect_covariance
from send_bw.sharing import SELECTIVE_V1, parse_sharing, resolve_donor_scenario, uses_donors
from send_bw.v3_eval import EvalProtocolError, validate_eval_config

ROOT = Path(__file__).resolve().parents[2]


def test_parse_sharing_accepts_selective():
    assert parse_sharing("selective") == "selective"
    assert uses_donors("selective")
    assert uses_donors("pooled")
    assert not uses_donors("context_specific")
    with pytest.raises(ValueError, match="unknown"):
        parse_sharing("magic")


def test_selective_cohort_includes_declared_donors():
    cfg = CohortConfig.from_dict({"species": "RAT", "donor_species": ["DOG"], "sharing": "selective"})
    assert cfg.allowed_species() == {"RAT", "DOG"}


def test_selective_v1_contract():
    assert SELECTIVE_V1["shared"] == ["compound", "regimen"]
    assert "bio" in SELECTIVE_V1["species_specific"]
    assert SELECTIVE_V1["residual"] == "target_species_only"


def test_feature_spec_selective_and_fixed_schema():
    animals = pd.DataFrame({
        "STUDYID": ["T1", "T1", "D1"], "USUBJID": ["a", "b", "c"],
        "SEX": ["M", "F", "M"], "SPECIES": ["RAT", "RAT", "DOG"],
        "STRAIN": ["SPRAGUE-DAWLEY", "SPRAGUE-DAWLEY", "BEAGLE"],
        "ROUTE": ["ORAL GAVAGE", "ORAL GAVAGE", "ORAL GAVAGE"],
        "COMPOUND_ID": ["CMP01", "CMP01", "CMP01"], "IS_CONTROL": [True, False, False],
        "PLANNED_DOSE": [0.0, 10.0, 10.0], "INTERVAL_DAYS": [1.0, 1.0, 1.0],
        "START_DAY": [0.0, 0.0, 0.0], "END_DAY": [28.0, 28.0, 28.0],
        "BASELINE_LOGW": [5.5, 5.3, 9.1],
    })
    obs = pd.DataFrame({"STUDYID": ["T1", "T1", "D1"], "USUBJID": ["a", "b", "c"],
                        "DAY": [7.0, 7.0, 7.0], "LOGW": [5.6, 5.4, 9.2]})
    cref = pd.DataFrame({"COMPOUND_ID": ["CMP01"], "D1": [0.1], "D2": [0.0], "D3": [0.0]})
    sel = FeatureSpec.fit(animals, obs, cref, "RAT", sharing="selective")
    assert sel.sharing == "selective"
    assert "SPECIES_RAT" in sel.animal_cols or "SPECIES_DOG" in sel.animal_cols
    rat_only = animals[animals["SPECIES"] == "RAT"]
    fixed = FeatureSpec.fit(rat_only, obs[obs["SPECIES"] == "RAT"] if "SPECIES" in obs.columns else obs[obs["STUDYID"] == "T1"],
                            cref, "RAT", sharing="context_specific",
                            declared_species=["DOG", "RAT"], fixed_schema=True)
    assert fixed.fixed_schema
    assert sorted(fixed.species_levels) == ["DOG", "RAT"]
    assert any(c.startswith("SPECIES_") for c in fixed.animal_cols)


def test_donor_scenario_does_not_change_target_studies(tmp_path):
    common = {"n_studies": 4, "n_other_species_studies": 2, "n_compounds": 4, "n_per_sex": [3, 3]}
    a = generate_dataset({**common, "donor_scenario": "matched"}, tmp_path / "m", seed=21)
    b = generate_dataset({**common, "donor_scenario": "both"}, tmp_path / "b", seed=21)
    ta = pd.read_parquet(Path(a["truth_dir"]) / "animals.parquet")
    tb = pd.read_parquet(Path(b["truth_dir"]) / "animals.parquet")
    ra = ta[~ta["STUDYID"].astype(str).str.contains("-D")]
    rb = tb[~tb["STUDYID"].astype(str).str.contains("-D")]
    pd.testing.assert_frame_equal(ra.reset_index(drop=True), rb.reset_index(drop=True))
    da = ta[ta["STUDYID"].astype(str).str.contains("-D")]
    db = tb[tb["STUDYID"].astype(str).str.contains("-D")]
    assert not np.allclose(da["Z0_TRUE"].to_numpy(), db["Z0_TRUE"].to_numpy())
    params_b = json.loads((Path(b["truth_dir"]) / "params.json").read_text())
    assert params_b["donor_scenario"] == "both"
    assert params_b["donor_deviations"]["effect_scale"] == 2.0


def test_encoder_rules_do_not_fill():
    obs = pd.DataFrame({
        "USUBJID": ["u1", "u1", "u1", "u2"],
        "DAY": [0.0, 7.0, 14.0, 21.0],
        "LOGW": [5.0, 5.1, 5.2, 5.3],
        "IS_BASELINE": [True, False, False, False],
    })
    h0, n0 = encoder_history(obs, "W0")
    assert len(h0) == 0 and n0 == 2
    h1, n1 = encoder_history(obs, "W1")
    assert set(h1["USUBJID"]) == {"u1", "u2"}
    assert n1 == 0
    assert list(h1.loc[h1["USUBJID"] == "u1", "DAY"]) == [7.0]
    hlast, nlast = encoder_history(obs, "Wlast14")
    assert list(hlast.loc[hlast["USUBJID"] == "u1", "DAY"]) == [14.0]
    assert "u2" not in set(hlast["USUBJID"])
    assert nlast == 1
    hall, nall = encoder_history(obs, "Wall14")
    assert set(hall["DAY"]) == {7.0, 14.0}
    assert nall == 1
    empty = pd.DataFrame({"USUBJID": ["u3"], "DAY": [21.0], "LOGW": [5.4], "IS_BASELINE": [False]})
    hmiss, nmiss = encoder_history(empty, "Wall14")
    assert len(hmiss) == 0 and nmiss == 1


def test_inspect_covariance_and_bayesian_gate():
    insp = inspect_covariance({"fit_info": {"params_at_bound": ["sigma_m"], "converged": True}})
    assert insp["boundary_flag"] is True
    assert insp["params_at_bound"] == ["sigma_m"]
    assert insp["covariance_parameters_bootstrapped"] is False
    assert "predictive_variability" in insp["components"]
    gate = bayesian_gate({"reliable": False, "rhat_max": 1.2, "divergences": 3})
    assert gate["counted_as_successful_benchmark"] is False
    assert gate["status"] == "trained"
    skipped = bayesian_gate({}, status="not_run", reason="bayesian extra not installed")
    assert skipped["status"] == "not_run"
    assert skipped["counted_as_successful_benchmark"] is False
    ok = bayesian_gate({"reliable": True, "rhat_max": 1.0, "divergences": 0,
                        "ess_bulk_min": 500, "ess_tail_min": 500})
    assert ok["counted_as_successful_benchmark"] is True


def test_transfer_table_names_selective_regime():
    s0 = {"model": "spline", "study_balanced": {"rmse_log": 0.1, "crps": 0.05}, "contrast_bias": 0.01}
    sk = {"model": "spline", "study_balanced": {"rmse_log": 0.12, "crps": 0.06}, "contrast_bias": 0.02,
          "sharing": "selective"}
    table = build_transfer_table(s0, [(1, sk)], sharing_regimes=["context_specific (S0)", "selective (Sk)"],
                                 selective_sharing="v1")
    assert table["selective_sharing"] == "v1"
    assert table["rows"][0]["sign"] == "harmful"
    assert table["rows"][0]["sharing"] == "selective"


def test_eval_config_refuses_design_and_real_data():
    cfg = yaml.safe_load((ROOT / "configs" / "v3_eval.frozen.yaml").read_text(encoding="utf-8"))
    validate_eval_config(cfg)
    bad = dict(cfg)
    bad["design"] = {"request": "configs/new_study.smoke.yaml"}
    with pytest.raises(EvalProtocolError, match="design"):
        validate_eval_config(bad)
    snow = dict(cfg)
    snow["source"] = {"type": "snowflake"}
    with pytest.raises(EvalProtocolError, match="real-data"):
        validate_eval_config(snow)


def test_pipeline_selective_prepare_keeps_target_manifest(tmp_path):
    from send_bw.pipeline import Run

    out = tmp_path / "run"
    cfg = {
        "run_name": "sel", "output_dir": str(out), "seed": 7,
        "synthetic": {"enabled": True, "out_dir": str(out / "synthetic"),
                      "n_studies": 8, "n_other_species_studies": 3, "n_compounds": 5,
                      "n_per_sex": [3, 4], "donor_scenario": "growth_shift"},
        "mapping": str(ROOT / "configs" / "mappings" / "example_send.yaml"),
        "source": {"type": "local", "root": str(out / "synthetic" / "send"),
                   "metadata_dir": str(out / "synthetic" / "metadata")},
        "cohort": {"species": "RAT", "max_day": 28, "sharing": "selective",
                   "donor_species": ["DOG"], "n_donor_studies": 1},
        "splits": {"validation": 0.2, "test": 0.2},
        "models": {"spline": {"enabled": False}, "transformer": {"enabled": False},
                   "bayesian": {"enabled": False}},
        "evaluation": {"tasks": ["predose_conditioned"]},
        "design": {},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run = Run(p)
    run.generate()
    run.ingest()
    run.prepare()
    man = json.loads((run.dir / "splits" / "target_manifest.json").read_text())
    donors = json.loads((run.dir / "splits" / "donor_order.json").read_text())
    spec = FeatureSpec.from_dict(json.loads((run.dir / "features" / "feature_spec.json").read_text()))
    assert donors["sharing"] == "selective"
    assert donors["k"] == 1
    assert man["test"]["n"] >= 1
    assert spec.sharing == "selective"
    assert any(c.startswith("SPECIES_") for c in spec.animal_cols)
    animals = pd.read_parquet(run.dir / "cohort" / "animals.parquet")
    assert set(animals["SPECIES"].str.upper()) == {"RAT", "DOG"}
    splits = pd.read_csv(run.dir / "splits" / "splits.csv", dtype=str)
    val_test = set(splits.loc[splits["SPLIT"].isin(["validation", "test"]), "STUDYID"])
    assert set(donors["selected"]).isdisjoint(val_test)


def test_resolve_donor_scenarios():
    assert resolve_donor_scenario("matched")["gmax_scale"] == 1.0
    assert resolve_donor_scenario("both")["effect_scale"] == 2.0
    with pytest.raises(ValueError):
        resolve_donor_scenario("alien")

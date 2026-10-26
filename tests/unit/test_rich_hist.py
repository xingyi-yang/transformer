"""Rich-history generator, cohort, windows, and leakage guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from send_bw.data.generate import (
    generate_dataset,
    planned_visit_days,
    recovery_visit_days,
    slope_tau,
)
from send_bw.data.harmonize import harmonize
from send_bw.data.metadata import load_mapping
from send_bw.data.qc import CohortConfig, select_cohort
from send_bw.data.sources import LocalSendSource
from send_bw.data.splits import make_study_splits, make_study_splits_calibrated
from send_bw.rich_hist import (
    LABEL,
    LOCKED_DIRS,
    OUTPUT_DEFAULT,
    RichHistError,
    _refuse_locked,
    build_history_windows,
    execute_rich_hist,
)
from send_bw.tft_early_hist_extend import load_yaml

MAPPING = Path("configs/mappings/example_send.yaml")


def test_slope_tau_matches_official_at_day_28_and_bounds_day_90():
    t = np.array([0.0, 28.0, 90.0])
    lin = slope_tau(t, "linear_tref")
    sat = slope_tau(t, "saturating", tau_g=40.0)
    assert lin[1] == pytest.approx(1.0)
    assert sat[1] == pytest.approx(1.0)
    assert sat[0] == pytest.approx(0.0)
    assert sat[2] < lin[2]
    assert sat[2] < 2.0
    with pytest.raises(ValueError, match="unknown slope_basis"):
        slope_tau(t, "linear")


def test_planned_schedules_are_experimental_and_nested():
    w = planned_visit_days(28, "weekly")
    t = planned_visit_days(28, "twice_weekly")
    d = planned_visit_days(28, "dense")
    assert w[:2] == [-7.0, 0.0]
    assert 14.0 in w and 28.0 in w
    assert set(w) <= set(d)
    assert 3.0 in t and 7.0 in t
    rec = recovery_visit_days(90, 14)
    assert rec == [97.0, 104.0]


def test_official_generate_path_unchanged_without_longitudinal(tmp_path):
    info = generate_dataset({"n_studies": 2, "n_other_species_studies": 0, "n_per_sex": [2, 2],
                             "n_compounds": 2}, tmp_path / "off", seed=11)
    assert info["n_studies"] == 2
    assert not info.get("longitudinal")
    ids = [p.name for p in Path(info["send_dir"]).iterdir() if p.is_dir()]
    assert all(i.startswith("SYNA-") for i in ids)
    assert not any(i.startswith("SYNRH-") for i in ids)


def test_include_pretreatment_default_drops_predose(synth):
    c0 = select_cohort(synth["tables"], CohortConfig())
    assert not bool(CohortConfig().include_pretreatment)
    assert (c0.obs["PHASE"] != "PRETREATMENT").all()
    assert not c0.obs["IS_BASELINE"].any()
    c1 = select_cohort(synth["tables"], CohortConfig(include_pretreatment=True))
    assert (c1.obs["DAY"] <= 0).any()
    assert len(c1.obs) >= len(c0.obs)


def _tiny_long_cfg(_out: Path | None = None) -> dict:
    return {
        "n_compounds": 3,
        "n_per_sex": [2, 2],
        "p_missing": 0.02,
        "p_death": 0.02,
        "p_dose_deviation": 0.04,
        "longitudinal": {
            "study_prefix": "SYNRH",
            "p_dropout_challenging": 0.10,
            "scenarios": [
                {"id": "r28w", "species": "RAT", "n_studies": 3, "duration": 28,
                 "schedule": "weekly", "treatment_days": 28, "recovery_days": 14,
                 "recovery_prob": 0.5, "slope_basis": "linear_tref", "dropout_kind": "ordinary"},
                {"id": "r90d", "species": "RAT", "n_studies": 2, "duration": 90,
                 "schedule": "dense", "treatment_days": 56, "recovery_days": 14,
                 "recovery_prob": 0.0, "slope_basis": "saturating", "dropout_kind": "ordinary"},
                {"id": "pair", "species": "RAT", "paired": True, "n_pairs": 1, "duration": 28,
                 "dense_schedule": "dense", "sparse_schedule": "weekly", "treatment_days": 28,
                 "recovery_days": 14, "recovery_prob": 0.0, "slope_basis": "linear_tref",
                 "dropout_kind": "ordinary"},
                {"id": "drop", "species": "RAT", "n_studies": 1, "duration": 90,
                 "schedule": "weekly", "treatment_days": 56, "recovery_days": 14,
                 "recovery_prob": 0.0, "slope_basis": "saturating", "dropout_kind": "challenging",
                 "p_dropout": 1.0},
                {"id": "dog", "species": "DOG", "n_studies": 1, "duration": 28,
                 "schedule": "weekly", "treatment_days": 28, "recovery_days": 14,
                 "recovery_prob": 0.0, "slope_basis": "linear_tref", "dropout_kind": "ordinary"},
            ],
        },
    }


def test_longitudinal_bank_pairing_no_postdeath_and_windows(tmp_path):
    info = generate_dataset(_tiny_long_cfg(tmp_path), tmp_path / "bank", seed=91021)
    assert info["n_studies"] == 3 + 2 + 2 + 1 + 1
    mapping = load_mapping(MAPPING)
    src = LocalSendSource(info["send_dir"], metadata_dir=info["metadata_dir"],
                          domain_aliases=mapping.domain_aliases)
    tables, _ = harmonize(src, mapping)
    scn = pd.read_csv(Path(info["metadata_dir"]) / "study_scenarios.csv")
    assert set(scn["SCHEDULE"]) >= {"weekly", "dense"}
    assert (scn["SPECIES"] == "DOG").any()

    status = tables["animal_status"]
    dead = status[status["DISPOSITION"].astype(str).str.contains("DEAD|MORIBUND", case=False, na=False)]
    for u, day in zip(dead["USUBJID"], dead["DISPOSITION_DAY"], strict=True):
        g = tables["bw_obs"].loc[tables["bw_obs"]["USUBJID"] == u]
        assert (g["DAY"] < float(day)).all()

    pairs = scn.dropna(subset=["PAIR_ID"])
    assert len(pairs) == 2
    sids = pairs["STUDYID"].astype(str).tolist()
    a = tables["bw_obs"].loc[tables["bw_obs"]["STUDYID"] == sids[0]].copy()
    b = tables["bw_obs"].loc[tables["bw_obs"]["STUDYID"] == sids[1]].copy()
    a["SUBJID"] = a["USUBJID"].astype(str).str.split("-").str[-1]
    b["SUBJID"] = b["USUBJID"].astype(str).str.split("-").str[-1]
    m = a.merge(b, on=["SUBJID", "DAY"], suffixes=("_a", "_b"))
    assert len(m) > 0
    assert np.allclose(m["WEIGHT_G_a"].to_numpy(float), m["WEIGHT_G_b"].to_numpy(float))

    cohort = select_cohort(tables, CohortConfig(include_pretreatment=True, species="RAT"))
    assert (cohort.obs["DAY"] <= 0).any()
    pack = build_history_windows(cohort.obs, cohort.animals, scenarios=scn, history_kind="full")
    assert pack["n_pairs"] > 0
    late = pack["examples"].loc[pack["examples"]["origin"] >= 28]
    assert len(late)
    assert float(late["n_encoder_available"].max()) > 2
    for r in pack["examples"].itertuples():
        assert max(r.encoder_days) <= r.origin + 1e-9
        assert r.target_day > r.origin
        assert r.lead_from_origin == pytest.approx(r.target_day - r.origin)
        assert r.lead_from_last_observation == pytest.approx(r.target_day - r.last_observed_day)
    short = build_history_windows(cohort.obs, cohort.animals, scenarios=scn, history_kind="short",
                                  short_history_k=2)
    assert short["examples"]["n_encoder_used"].max() <= 2
    trunc = build_history_windows(cohort.obs, cohort.animals, scenarios=scn,
                                  max_encoder_length=8, history_kind="full")
    if (trunc["examples"]["n_encoder_available"] > 8).any():
        assert bool(trunc["examples"]["truncated_encoder"].any())
        assert (trunc["examples"]["n_encoder_excluded_by_limit"] >= 0).all()

    groups = {}
    for r in scn.to_dict("records"):
        groups[str(r["STUDYID"])] = str(r["PAIR_ID"]) if pd.notna(r.get("PAIR_ID")) else str(r["STUDYID"])
    rat = tables["study_context"]
    rat = rat[rat["SPECIES"] == "RAT"]
    sp = make_study_splits_calibrated(rat, {"validation": 0.2, "calibration": 0.2, "test": 0.2},
                                      seed=92031, groups=groups)
    assert set(sp["SPLIT"]) >= {"train", "validation", "calibration", "test"}
    mixed = sp.groupby("GROUP")["SPLIT"].nunique()
    assert (mixed <= 1).all()
    official = make_study_splits(rat, {"validation": 0.2, "test": 0.2}, seed=3)
    assert "calibration" not in set(official["SPLIT"])


def test_config_and_locked_dirs():
    cfg = load_yaml("configs/rich_hist_v1.yaml")
    assert cfg["generate_seed"] == 91021
    assert cfg["split_seed"] == 92031
    assert cfg["generate_seed"] not in {71001, 81011}
    assert cfg["cohort"]["include_pretreatment"] is True
    assert cfg["models"]["optional"]["tft"]["enabled"] is False
    assert cfg["output_dir"] == OUTPUT_DEFAULT
    assert LABEL.startswith("Synthetic methodological benchmark")
    with pytest.raises(RichHistError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_larger_bank"))
    with pytest.raises(RichHistError, match="preserved|overwrite"):
        _refuse_locked(Path("runs/tft_longer_ctx"))
    assert OUTPUT_DEFAULT not in LOCKED_DIRS
    assert "runs/tft_longer_ctx" in LOCKED_DIRS


def test_execute_refuses_locked_generate_seed(tmp_path):
    cfg = load_yaml("configs/rich_hist_v1.yaml")
    cfg = dict(cfg)
    cfg["generate_seed"] = 71001
    cfg["output_dir"] = str(tmp_path / "x")
    with pytest.raises(RichHistError, match="locked bank"):
        execute_rich_hist(cfg, force=True)

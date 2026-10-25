"""Stages 1–2 v3 pilot infrastructure. No model-training wave. Synthetic only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from send_bw.data.features import FeatureSpec, availability
from send_bw.data.generate import generate_dataset
from send_bw.data.harmonize import harmonize
from send_bw.data.qc import SELECTIVE_SHARING_DEFERRED, CohortConfig, select_cohort
from send_bw.data.sources import LocalSendSource
from send_bw.data.splits import (
    compose_train_with_donors,
    donor_study_order,
    freeze_target_manifests,
    make_study_splits,
    target_manifest,
)
from send_bw.evaluation.transfer import build_transfer_table, paired_transfer_row
from send_bw.models.design import MeanDesign
from send_bw.models.spline import SplineModel
from send_bw.models.transformer import token_groups
from send_bw.pilot import (
    PILOT_TASK,
    RIDGE_ALPHAS,
    SELECTIVE_SHARING,
    STUDY_WEIGHTING,
    UNCERTAINTY_SAFEGUARDS,
    as_dict,
)
from send_bw.pipeline import Run

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "configs" / "mappings" / "example_send.yaml"


@pytest.fixture(scope="module")
def multi_tables(tmp_path_factory, mapping):
    root = tmp_path_factory.mktemp("v3ms")
    info = generate_dataset(
        {"n_studies": 8, "n_other_species_studies": 3, "n_compounds": 5, "n_per_sex": [3, 4]},
        root, seed=11,
    )
    src = LocalSendSource(info["send_dir"], metadata_dir=info["metadata_dir"],
                          domain_aliases=mapping.domain_aliases)
    tables, _ = harmonize(src, mapping)
    return {"root": root, "tables": tables, "info": info}


def _target_splits(tables, seed=3):
    target = select_cohort(tables, CohortConfig(max_day=28))
    sc = tables["study_context"]
    in_scope = sc[(sc["SPECIES"] == "RAT") & sc["STUDYID"].isin(set(target.animals["STUDYID"]))]
    splits = make_study_splits(in_scope, {"validation": 0.2, "test": 0.2}, seed=seed)
    return target, splits


def test_species_prefix_is_production():
    assert availability("SPECIES_DOG") == "production"
    assert availability("SPECIES_RAT") == "production"


def test_selective_sharing_is_deferred():
    with pytest.raises(ValueError, match="deferred"):
        CohortConfig.from_dict({"sharing": "selective", "donor_species": ["DOG"]})
    assert SELECTIVE_SHARING == "required_deferred"
    assert "deferred" in SELECTIVE_SHARING_DEFERRED


def test_v2_cohort_still_excludes_donors(multi_tables):
    t = multi_tables["tables"]
    c = select_cohort(t, CohortConfig(max_day=28))
    assert set(c.animals["SPECIES"].str.upper()) == {"RAT"}
    assert (t["study_context"]["SPECIES"] == "DOG").any()
    assert c.exclusions["EXCLUSION_REASON"].str.contains("species_out_of_scope").any()


def test_pooled_cohort_includes_declared_donors_only(multi_tables):
    t = multi_tables["tables"]
    c = select_cohort(t, CohortConfig.from_dict(
        {"species": "RAT", "donor_species": ["DOG"], "sharing": "pooled", "max_day": 28}))
    assert set(c.animals["SPECIES"].str.upper()) == {"RAT", "DOG"}
    ctx = t["animal_context"].copy()
    ctx["SPECIES"] = ctx["SPECIES"].replace({"DOG": "MOUSE"})
    # unseen species stay out of scope even when pooling declared donors
    c2 = select_cohort(t, CohortConfig.from_dict(
        {"species": "RAT", "donor_species": ["MOUSE"], "sharing": "pooled", "max_day": 28}))
    assert "DOG" not in set(c2.animals["SPECIES"].str.upper())


def test_stable_donor_generation_independent_of_target_count(tmp_path, mapping):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_dataset({"n_studies": 6, "n_other_species_studies": 2, "n_compounds": 4, "n_per_sex": [3, 3]}, a, seed=21)
    generate_dataset({"n_studies": 10, "n_other_species_studies": 2, "n_compounds": 4, "n_per_sex": [3, 3]}, b, seed=21)
    ma = pd.read_csv(a / "metadata" / "study_metadata.csv")
    mb = pd.read_csv(b / "metadata" / "study_metadata.csv")
    assert list(ma.loc[ma["SPECIES"] == "RAT", "STUDYID"]) != list(mb.loc[mb["SPECIES"] == "RAT", "STUDYID"])
    da = ma.loc[ma["SPECIES"] == "DOG"].sort_values("STUDYID")
    db = mb.loc[mb["SPECIES"] == "DOG"].sort_values("STUDYID")
    assert list(da["STUDYID"]) == list(db["STUDYID"]) == ["SYNA-D001", "SYNA-D002"]
    src_a = LocalSendSource(a / "send", metadata_dir=a / "metadata", domain_aliases=mapping.domain_aliases)
    src_b = LocalSendSource(b / "send", metadata_dir=b / "metadata", domain_aliases=mapping.domain_aliases)
    ta, _ = harmonize(src_a, mapping)
    tb, _ = harmonize(src_b, mapping)
    for sid in ["SYNA-D001", "SYNA-D002"]:
        aa = ta["animal_context"][ta["animal_context"]["STUDYID"] == sid].sort_values("USUBJID")
        bb = tb["animal_context"][tb["animal_context"]["STUDYID"] == sid].sort_values("USUBJID")
        pd.testing.assert_series_equal(aa["BASELINE_LOGW"].reset_index(drop=True),
                                       bb["BASELINE_LOGW"].reset_index(drop=True))


def test_target_generation_unchanged_when_donors_added(tmp_path):
    a = tmp_path / "one"
    b = tmp_path / "three"
    generate_dataset({"n_studies": 6, "n_other_species_studies": 1, "n_compounds": 4, "n_per_sex": [3, 3]}, a, seed=21)
    generate_dataset({"n_studies": 6, "n_other_species_studies": 3, "n_compounds": 4, "n_per_sex": [3, 3]}, b, seed=21)
    ma = pd.read_csv(a / "metadata" / "study_metadata.csv")
    mb = pd.read_csv(b / "metadata" / "study_metadata.csv")
    rat_a = ma.loc[ma["SPECIES"] == "RAT"].sort_values("STUDYID").reset_index(drop=True)
    rat_b = mb.loc[mb["SPECIES"] == "RAT"].sort_values("STUDYID").reset_index(drop=True)
    pd.testing.assert_frame_equal(rat_a, rat_b)
    ta = pd.read_parquet(a / "truth" / "animals.parquet")
    tb = pd.read_parquet(b / "truth" / "animals.parquet")
    rat_ids = set(rat_a["STUDYID"])
    pd.testing.assert_frame_equal(
        ta[ta["STUDYID"].isin(rat_ids)].sort_values("USUBJID").reset_index(drop=True),
        tb[tb["STUDYID"].isin(rat_ids)].sort_values("USUBJID").reset_index(drop=True),
    )


def test_target_manifests_frozen_and_nested_donors(multi_tables, tmp_path):
    t = multi_tables["tables"]
    target, splits = _target_splits(t)
    man0 = target_manifest(splits)
    freeze_target_manifests(splits, tmp_path / "target_manifest.json")
    donors = donor_study_order(t["study_context"].loc[t["study_context"]["SPECIES"] == "DOG", "STUDYID"])
    assert len(donors) == 3
    composed = compose_train_with_donors(splits, donors, 2)
    man1 = target_manifest(splits)
    assert man0 == man1
    assert man0["train"]["hash"] == man1["train"]["hash"]
    train = set(composed.loc[composed["SPLIT"] == "train", "STUDYID"])
    val = set(composed.loc[composed["SPLIT"] == "validation", "STUDYID"])
    test = set(composed.loc[composed["SPLIT"] == "test", "STUDYID"])
    assert set(donors[:2]) <= train
    assert set(donors).isdisjoint(val)
    assert set(donors).isdisjoint(test)
    assert set(donors[2:]).isdisjoint(train)
    target_ids = set(target.animals["STUDYID"].astype(str))
    assert (val | test) <= target_ids


def test_unchanged_target_records_when_donors_added(multi_tables):
    t = multi_tables["tables"]
    target, splits = _target_splits(t)
    donors = donor_study_order(t["study_context"].loc[t["study_context"]["SPECIES"] == "DOG", "STUDYID"])
    c0 = compose_train_with_donors(splits, donors, 0)
    c2 = compose_train_with_donors(splits, donors, 2)
    t0 = set(c0["STUDYID"])
    extra = set(c2["STUDYID"]) - t0
    assert extra == set(donors[:2])
    a0 = target.animals.sort_values("USUBJID").reset_index(drop=True)
    o0 = target.obs.sort_values(["USUBJID", "DAY"]).reset_index(drop=True)
    # selecting donors does not mutate the target cohort object
    _ = select_cohort(t, CohortConfig.from_dict(
        {"species": "RAT", "donor_species": ["DOG"], "sharing": "pooled", "max_day": 28}))
    a1 = target.animals.sort_values("USUBJID").reset_index(drop=True)
    o1 = target.obs.sort_values(["USUBJID", "DAY"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a0, a1)
    pd.testing.assert_frame_equal(o0, o1)


def test_zero_donor_equivalence(multi_tables):
    t = multi_tables["tables"]
    target, splits = _target_splits(t)
    donors = donor_study_order(t["study_context"].loc[t["study_context"]["SPECIES"] == "DOG", "STUDYID"])
    composed = compose_train_with_donors(splits, donors, 0)
    left = composed.sort_values("STUDYID").reset_index(drop=True).astype(str)
    right = splits[["STUDYID", "SPLIT"]].astype(str).sort_values("STUDYID").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    tr = set(composed.loc[composed["SPLIT"] == "train", "STUDYID"])
    spec_v2 = FeatureSpec.fit(target.animals[target.animals.STUDYID.isin(tr)],
                              target.obs[target.obs.STUDYID.isin(tr)], t["compound_reference"], "RAT")
    spec_p0 = FeatureSpec.fit(target.animals[target.animals.STUDYID.isin(tr)],
                              target.obs[target.obs.STUDYID.isin(tr)], t["compound_reference"], "RAT",
                              sharing="pooled")
    assert spec_v2.animal_cols == spec_p0.animal_cols
    assert not any(c.startswith("SPECIES_") for c in spec_v2.animal_cols)
    assert spec_v2.zb_mean == spec_p0.zb_mean


def test_training_only_preprocessing(multi_tables):
    t = multi_tables["tables"]
    target, splits = _target_splits(t)
    tr = set(splits.loc[splits["SPLIT"] == "train", "STUDYID"])
    va = set(splits.loc[splits["SPLIT"] == "validation", "STUDYID"])
    spec_tr = FeatureSpec.fit(target.animals[target.animals.STUDYID.isin(tr)],
                              target.obs[target.obs.STUDYID.isin(tr)], t["compound_reference"], "RAT")
    spec_all = FeatureSpec.fit(target.animals[target.animals.STUDYID.isin(tr | va)],
                               target.obs[target.obs.STUDYID.isin(tr | va)], t["compound_reference"], "RAT")
    train_zb = {s: float(g["BASELINE_LOGW"].mean())
                for s, g in target.animals[target.animals.STUDYID.isin(tr)].groupby("SEX")}
    assert spec_tr.zb_mean == pytest.approx(train_zb)
    assert spec_tr.strains == sorted(target.animals.loc[target.animals.STUDYID.isin(tr), "STRAIN"].dropna().unique())
    assert spec_tr.zb_mean != spec_all.zb_mean or spec_tr.max_day != spec_all.max_day or spec_tr.strains != spec_all.strains


def test_explicit_species_features_and_unseen_refused(multi_tables):
    t = multi_tables["tables"]
    target = select_cohort(t, CohortConfig(max_day=28))
    donors = select_cohort(t, CohortConfig(species="DOG", max_day=28))
    train = pd.concat([target.animals.head(20), donors.animals.head(12)], ignore_index=True)
    obs = pd.concat([target.obs[target.obs.USUBJID.isin(train.USUBJID)],
                     donors.obs[donors.obs.USUBJID.isin(train.USUBJID)]], ignore_index=True)
    spec = FeatureSpec.fit(train, obs, t["compound_reference"], "RAT", sharing="pooled")
    assert spec.sharing == "pooled"
    assert "DOG" in spec.species_levels and "RAT" in spec.species_levels
    assert any(c.startswith("SPECIES_") for c in spec.animal_cols)
    F = spec.observation_features(obs, train, t["compound_reference"])
    species_cols = [c for c in F.columns if c.startswith("SPECIES_")]
    assert species_cols
    design = MeanDesign.from_features(F)
    assert design.species_cols == species_cols
    X = design.matrix(F, fit_scale=True)
    assert X.shape[1] == len(design.col_names)
    assert any(n.startswith("tb_SPECIES_") for n in design.col_names)
    groups = token_groups(spec)
    assert "species" in groups
    ctx = train.head(2).assign(SPECIES="MOUSE")
    problems = spec.check_context(ctx, t["compound_reference"], horizon=28)
    assert any("MOUSE" in p for p in problems)


def test_v2_transformer_token_groups_have_no_species(synth):
    from send_bw.data.qc import CohortConfig as CC
    c = select_cohort(synth["tables"], CC(max_day=28))
    spec = FeatureSpec.fit(c.animals, c.obs, synth["tables"]["compound_reference"], "RAT")
    assert "species" not in token_groups(spec)
    assert not any(c.startswith("SPECIES_") for c in spec.animal_cols)


def test_serialization_roundtrip_species_features(multi_tables, tmp_path):
    t = multi_tables["tables"]
    target = select_cohort(t, CohortConfig(max_day=28))
    donors = select_cohort(t, CohortConfig(species="DOG", max_day=28))
    _, splits = _target_splits(t)
    tr = set(splits.loc[splits["SPLIT"] == "train", "STUDYID"])
    animals = pd.concat([target.animals[target.animals.STUDYID.isin(tr)], donors.animals], ignore_index=True)
    obs = pd.concat([target.obs[target.obs.STUDYID.isin(tr)], donors.obs], ignore_index=True)
    spec = FeatureSpec.fit(animals, obs, t["compound_reference"], "RAT", sharing="pooled")
    spec2 = FeatureSpec.from_dict(spec.to_dict())
    assert spec2.to_dict() == spec.to_dict()
    assert spec2.species_levels == spec.species_levels
    assert any(c.startswith("SPECIES_") for c in spec2.animal_cols)
    F = spec.observation_features(obs, animals, t["compound_reference"])
    design = MeanDesign.from_features(F)
    design.matrix(F, fit_scale=True)
    design2 = MeanDesign.from_dict(design.to_dict())
    assert design2.species_cols == design.species_cols
    from send_bw.models.base import ModelData
    data = ModelData(animals, obs, t["compound_reference"], spec)
    # tiny held-out slice of target studies for validation
    va_ids = set(splits.loc[splits["SPLIT"] == "validation", "STUDYID"])
    va = ModelData(target.animals[target.animals.STUDYID.isin(va_ids)],
                   target.obs[target.obs.STUDYID.isin(va_ids)], t["compound_reference"], spec)
    sm = SplineModel()
    sm.fit(data, va, {"n_bootstrap": 1, "seed": 1, "ridge_alphas": [10.0, 100.0]})
    sm.save(tmp_path / "spline")
    sm2 = SplineModel.load(tmp_path / "spline")
    assert any(c.startswith("SPECIES_") for c in sm2.spec.animal_cols)
    ctx = target.animals[target.animals.STUDYID.isin(tr)].head(3).reset_index(drop=True)
    np.testing.assert_allclose(sm.predict_mean(ctx, np.array([7.0, 28.0])),
                               sm2.predict_mean(ctx, np.array([7.0, 28.0])), atol=1e-6)


def test_old_feature_spec_dict_loads_without_species_fields():
    spec = FeatureSpec.from_dict({
        "species": "RAT", "strains": ["SPRAGUE-DAWLEY"], "routes": ["ORAL GAVAGE"],
        "sexes": ["F", "M"], "zb_mean": {"F": 5.0, "M": 5.5}, "max_dose": 100.0,
        "max_day": 28.0, "intervals": [1.0], "compound_ids": [],
        "descriptor_ranges": {"D1": [0.0, 1.0], "D2": [0.0, 1.0], "D3": [0.0, 1.0]},
        "animal_cols": ["SEX_M", "ZB_C"], "time_cols": ["TAU", "EXPO", "OFF", "LCUM"],
    })
    assert spec.sharing == "context_specific"
    assert spec.species_levels == []
    assert spec.zb_mean_by_species == {}


def test_sign_defined_paired_transfer_table():
    s0 = {"model": "spline", "study_balanced": {"rmse_log": 0.10, "crps": 0.08}, "contrast_bias": 0.02}
    sk_help = {"model": "spline", "study_balanced": {"rmse_log": 0.08, "crps": 0.07}, "contrast_bias": 0.01}
    sk_harm = {"model": "spline", "study_balanced": {"rmse_log": 0.14, "crps": 0.11}, "contrast_bias": -0.06}
    table = build_transfer_table(s0, [(1, sk_help), (2, sk_harm)])
    assert table["task"] == PILOT_TASK
    assert table["selective_sharing"] == "required_deferred"
    by = {(r["k"], r["metric"]): r for r in table["rows"]}
    assert by[(1, "rmse_log")]["sign"] == "helpful"
    assert by[(1, "rmse_log")]["delta"] == pytest.approx(-0.02)
    assert by[(2, "rmse_log")]["sign"] == "harmful"
    assert by[(2, "contrast_bias")]["sign"] == "harmful"
    assert by[(2, "contrast_bias")]["delta"] == pytest.approx(0.04)
    missing = paired_transfer_row(s0, {"study_balanced": {}}, "rmse_log", 3, "spline")
    assert missing["sign"] == "not_estimable"


def test_prespec_declares_weighting_budgets_tuning_seeds():
    p = as_dict()
    assert p["task"] == "predose_conditioned"
    assert p["study_weighting"] == STUDY_WEIGHTING == "study_balanced"
    assert p["spline_budget"]["ridge_alphas"] == RIDGE_ALPHAS
    assert p["transformer_budget_smoke"]["n_members"] == 1
    assert "validation" in p["tuning_rules"]["spline"]
    assert p["seed_namespaces"]["generate_donor_study"] == ("donor", "{d}")
    assert p["uncertainty_safeguards"]["no_fabricated_baseline_only"] is True
    assert p["uncertainty_safeguards"] == UNCERTAINTY_SAFEGUARDS
    assert p["selective_sharing"] == "required_deferred"


def _prepare_cfg(tmp: Path, **cohort) -> Path:
    out = tmp / "run"
    cfg = {
        "run_name": "v3_prep", "output_dir": str(out), "seed": 5,
        "synthetic": {"enabled": True, "out_dir": str(out / "synthetic"),
                      "n_studies": 8, "n_other_species_studies": 3, "n_compounds": 5, "n_per_sex": [3, 4]},
        "mapping": str(MAPPING),
        "source": {"type": "local", "root": str(out / "synthetic" / "send"),
                   "metadata_dir": str(out / "synthetic" / "metadata")},
        "cohort": {"species": "RAT", "max_day": 28, **cohort},
        "splits": {"validation": 0.2, "test": 0.2},
        "models": {"spline": {"enabled": False}, "transformer": {"enabled": False}, "bayesian": {"enabled": False}},
        "evaluation": {"tasks": ["predose_conditioned"]},
    }
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_pipeline_v2_defaults_and_donor_isolation(tmp_path):
    cfg = _prepare_cfg(tmp_path)
    run = Run(cfg)
    run.generate()
    run.ingest()
    run.prepare()
    splits = pd.read_csv(run.dir / "splits" / "splits.csv", dtype=str)
    animals = pd.read_parquet(run.dir / "cohort" / "animals.parquet")
    spec = FeatureSpec.from_dict(json.loads((run.dir / "features" / "feature_spec.json").read_text()))
    assert set(animals["SPECIES"].str.upper()) == {"RAT"}
    assert not any(c.startswith("SPECIES_") for c in spec.animal_cols)
    donor_order = json.loads((run.dir / "splits" / "donor_order.json").read_text())
    assert donor_order["k"] == 0 and donor_order["selected"] == []
    man = json.loads((run.dir / "splits" / "target_manifest.json").read_text())
    assert man["train"]["n"] >= 2
    assert set(splits["STUDYID"]) == set(man["all_target"]["studyids"])

    pooled = _prepare_cfg(tmp_path / "pooled", sharing="pooled", donor_species=["DOG"], n_donor_studies=2)
    # reuse the same generated SEND library
    pcfg = yaml.safe_load(pooled.read_text())
    pcfg["synthetic"]["enabled"] = False
    pcfg["source"]["root"] = str(run.dir / "synthetic" / "send")
    pcfg["source"]["metadata_dir"] = str(run.dir / "synthetic" / "metadata")
    pooled.write_text(yaml.safe_dump(pcfg), encoding="utf-8")
    run2 = Run(pooled)
    run2.ingest()
    run2.prepare()
    man2 = json.loads((run2.dir / "splits" / "target_manifest.json").read_text())
    assert man["train"]["hash"] == man2["train"]["hash"]
    assert man["validation"]["hash"] == man2["validation"]["hash"]
    assert man["test"]["hash"] == man2["test"]["hash"]
    a1 = pd.read_parquet(run.dir / "cohort" / "animals.parquet").sort_values("USUBJID").reset_index(drop=True)
    a2 = pd.read_parquet(run2.dir / "cohort" / "animals.parquet")
    target2 = a2[a2["SPECIES"].str.upper() == "RAT"].sort_values("USUBJID").reset_index(drop=True)
    pd.testing.assert_frame_equal(a1, target2)
    splits2 = pd.read_csv(run2.dir / "splits" / "splits.csv", dtype=str)
    donors = json.loads((run2.dir / "splits" / "donor_order.json").read_text())
    assert donors["k"] == 2
    val_test = set(splits2.loc[splits2["SPLIT"].isin(["validation", "test", "test_compound_holdout"]), "STUDYID"])
    assert set(donors["selected"]).isdisjoint(val_test)
    spec2 = FeatureSpec.from_dict(json.loads((run2.dir / "features" / "feature_spec.json").read_text()))
    assert any(c.startswith("SPECIES_") for c in spec2.animal_cols)
    assert spec2.sharing == "pooled"


def test_proposed_smoke_config_is_review_only():
    cfg = yaml.safe_load((ROOT / "configs" / "v3_pilot.smoke.yaml").read_text(encoding="utf-8"))
    assert cfg["evaluation"]["tasks"] == ["predose_conditioned"]
    assert not cfg.get("design", {}).get("request")
    assert cfg["cohort"]["sharing"] == "pooled"
    assert cfg["synthetic"]["n_other_species_studies"] >= 3
    assert cfg["models"]["bayesian"]["enabled"] is False
    assert "propose-design" not in yaml.dump(cfg)

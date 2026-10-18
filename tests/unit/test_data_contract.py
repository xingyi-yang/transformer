from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from send_bw.data.harmonize import dy_to_elapsed, harmonize, parse_date
from send_bw.data.qc import CohortConfig, select_cohort
from send_bw.data.schema import SchemaError, safe_merge
from send_bw.data.sources import AmbiguousSourceError, LocalSendSource, OptionalDependencyError, _read_file


def q(tables, reason):
    qt = tables["quarantine"]
    return qt[qt["REASON"] == reason]


# ------------------------------------------------------------------ day conventions
def test_no_day_zero_convention():
    assert dy_to_elapsed(1) == 0.0
    assert dy_to_elapsed(8) == 7.0
    assert dy_to_elapsed(-1) == -1.0
    assert dy_to_elapsed(-7) == -7.0
    with pytest.raises(ValueError):
        dy_to_elapsed(0)
    assert dy_to_elapsed(0, no_day_zero=False) == 0.0


def test_parse_date_kinds():
    assert parse_date("2021-03-01")[1] == "full"
    assert parse_date("2021-03-01T08:30")[1] == "full"
    assert parse_date("2021-03")[1] == "partial"
    assert parse_date("2021")[1] == "partial"
    assert parse_date(None)[1] == "missing"
    assert parse_date("03/01/2021")[1] == "invalid"


# ------------------------------------------------------------------ messy fixture
def test_units_converted_and_unknown_unit_quarantined(messy):
    obs = messy["tables"]["bw_obs"]
    a2 = obs[obs["USUBJID"] == "MSY-01-002"].sort_values("DAY")
    assert np.allclose(a2["WEIGHT_G"].to_numpy()[:2], [195.0, 210.0])
    assert (a2["ORIG_UNIT"] == "kg").all()
    assert "from_original_result" in a2["QC_FLAGS"].iloc[0]
    assert len(q(messy["tables"], "no_convertible_value")) == 1


def test_partial_dates(messy):
    obs = messy["tables"]["bw_obs"]
    r = obs[(obs["USUBJID"] == "MSY-01-001") & (obs["BWSEQ"] == 4)].iloc[0]
    assert r["DAY"] == 14.0 and r["DATE_UNCERTAIN"]
    assert len(q(messy["tables"], "date_partial_unresolvable")) == 1


def test_day_zero_record_quarantined(messy):
    assert len(q(messy["tables"], "invalid_study_day_zero")) == 1


def test_duplicates(messy):
    t = messy["tables"]
    assert len(q(t, "duplicate_record_dropped")) == 1
    assert len(q(t, "conflicting_duplicate_weight")) == 2
    obs = t["bw_obs"]
    a3 = obs[obs["USUBJID"] == "MSY-01-003"]
    assert 14.0 not in set(a3["DAY"])
    assert len(q(t, "nonpositive_weight")) == 1


def test_post_death_weights_quarantined(messy):
    t = messy["tables"]
    obs = t["bw_obs"]
    assert obs[obs["USUBJID"] == "MSY-01-004"]["DAY"].max() == 7.0
    assert len(q(t, "post_disposition_weight")) == 1
    st = t["animal_status"].set_index("USUBJID")
    assert st.loc["MSY-01-004", "DISPOSITION"] == "DEATH"
    assert st.loc["MSY-01-004", "DISPOSITION_DAY"] == 11.0
    assert st.loc["MSY-01-003", "DISPOSITION"] == "ONGOING_OR_UNKNOWN"


def test_fasting_from_supplemental_and_unmapped_qnam_reported(messy):
    obs = messy["tables"]["bw_obs"]
    f = obs[(obs["USUBJID"] == "MSY-01-001") & obs["FASTING"]]
    assert list(f["DAY"]) == [29.0]
    types = {w["type"] for w in messy["report"]["warnings"]}
    assert "unmapped_supplemental_qualifier" in types
    assert "unmapped_disposition_term" in types


def test_baseline_rule_last_predose_no_post_treatment_substitution(messy):
    a = messy["tables"]["animal_context"].set_index("USUBJID")
    assert a.loc["MSY-01-001", "BASELINE_DAY"] == 0.0
    assert a.loc["MSY-01-001", "BASELINE_G"] == pytest.approx(262.0)
    assert np.isnan(a.loc["MSY-01-008", "BASELINE_LOGW"])
    assert "no_eligible_baseline" in a.loc["MSY-01-008", "EXCLUSION_REASON"]
    obs = messy["tables"]["bw_obs"]
    assert obs.groupby("USUBJID")["IS_BASELINE"].sum().max() == 1


def test_sex_never_inferred_and_missing_regimen_flagged(messy):
    a = messy["tables"]["animal_context"].set_index("USUBJID")
    assert a.loc["MSY-01-006", "SEX"] == "U"
    assert "sex_not_mapped" in a.loc["MSY-01-006", "EXCLUSION_REASON"]
    reg = messy["tables"]["arm_regimen"].set_index("ARM_KEY")
    assert not reg.loc["MSY-01::9", "REGIMEN_AVAILABLE"]
    c = select_cohort(messy["tables"], CohortConfig())
    ex = c.exclusions.set_index("USUBJID")
    assert "planned_regimen_unavailable" in ex.loc["MSY-01-007", "EXCLUSION_REASON"]


def test_recovery_phase_and_cohort(messy):
    t = messy["tables"]
    obs = t["bw_obs"]
    r = obs[obs["USUBJID"] == "MSY-01-005"].set_index("DAY")
    assert r.loc[42.0, "PHASE"] == "RECOVERY"
    assert r.loc[28.0, "PHASE"] == "TREATMENT"
    assert t["animal_context"].set_index("USUBJID").loc["MSY-01-005", "COHORT"] == "RECOVERY"


def test_arm_labels_study_specific(messy):
    reg = messy["tables"]["arm_regimen"].set_index("ARM_KEY")
    assert reg.loc["MSY-01::1", "IS_CONTROL"] and not reg.loc["MSY-02::1", "IS_CONTROL"]
    assert reg.loc["MSY-02::2", "IS_CONTROL"]


def test_administration_deviation_flagged_not_used_as_plan(messy):
    adm = messy["tables"]["administrations"].set_index("USUBJID")
    assert adm.loc["MSY-01-004", "DEVIATION"] == "dose_differs_from_plan"
    reg = messy["tables"]["arm_regimen"].set_index("ARM_KEY")
    assert reg.loc["MSY-01::2", "PLANNED_DOSE"] == 100.0


def test_alias_and_mixed_case_and_parquet(messy):
    obs = messy["tables"]["bw_obs"]
    assert set(obs["STUDYID"]) == {"MSY-01", "MSY-02"}
    assert set(obs.loc[obs["STUDYID"] == "MSY-02", "SRC_FILE"]) == {"MSY-02/BODYWEIGHT.csv"}


def test_provenance_preserved(messy):
    obs = messy["tables"]["bw_obs"]
    assert obs["SRC_FILE"].notna().all() and (obs["SRC_ROW"] >= 0).all()
    assert (obs["MAPPING_VERSION"] == "example-send-0.1").all()


# ------------------------------------------------------------------ joins and sources
def test_safe_merge_rejects_row_inflation():
    left = pd.DataFrame({"k": [1, 2]})
    right = pd.DataFrame({"k": [1, 1], "v": [3, 4]})
    with pytest.raises(pd.errors.MergeError):  # validate="many_to_one"
        safe_merge(left, right, on=["k"])


def test_duplicate_curated_regimen_rows_rejected(tmp_path, mapping):
    from fixtures import messy_send

    p = messy_send.build(tmp_path)
    ar = pd.read_csv(p["metadata"] / "arm_regimen.csv")
    pd.concat([ar, ar.iloc[[0]]]).to_csv(p["metadata"] / "arm_regimen.csv", index=False)
    src = LocalSendSource(p["send"], metadata_dir=p["metadata"], domain_aliases=mapping.domain_aliases)
    with pytest.raises(SchemaError):
        harmonize(src, mapping)


def test_ambiguous_domain_files_rejected(tmp_path, mapping):
    from fixtures import messy_send

    p = messy_send.build(tmp_path)
    shutil.copy(p["send"] / "MSY-01" / "dm.csv", p["send"] / "MSY-01" / "DM.parquet")
    with pytest.raises(AmbiguousSourceError):
        LocalSendSource(p["send"], metadata_dir=p["metadata"])


def test_column_collision_rejected(tmp_path):
    d = tmp_path / "S1"
    d.mkdir()
    (d / "dm.csv").write_text("STUDYID,usubjid,USUBJID\nS1,a,b\n")
    src = LocalSendSource(tmp_path)
    with pytest.raises(AmbiguousSourceError):
        src.read_domain("DM")


def test_xpt_optional_dependency_message(tmp_path):
    try:
        import pyreadstat  # noqa: F401
        pytest.skip("pyreadstat installed")
    except ImportError:
        pass
    f = tmp_path / "bw.xpt"
    f.write_bytes(b"not really xpt")
    with pytest.raises(OptionalDependencyError, match="xpt"):
        _read_file(f, "xpt")


def test_combined_layout_equivalent_to_per_study(tmp_path, mapping, messy):
    """Same records laid out as combined domain files yield the same canonical outputs."""
    root = tmp_path / "combined"
    root.mkdir()
    src = messy["source"]
    for dom in ["DM", "BW", "DS", "EX", "SUPPBW"]:
        df = src.read_domain(dom).drop(columns=["_SRC_FILE", "_SRC_ROW"])
        df.to_csv(root / f"{dom.lower()}.csv", index=False)
    comb = LocalSendSource(root, layout="combined", metadata_dir=messy["paths"]["metadata"])
    t2, _ = harmonize(comb, mapping)
    a = messy["tables"]["bw_obs"].drop(columns=["SRC_FILE", "SRC_ROW"]).sort_values(["USUBJID", "DAY", "FASTING"])
    b = t2["bw_obs"].drop(columns=["SRC_FILE", "SRC_ROW"]).sort_values(["USUBJID", "DAY", "FASTING"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

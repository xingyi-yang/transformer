"""Independently hand-authored messy SEND-like fixture (does NOT use the synthetic generator).

Contents (study MSY-01, rat, per-study layout):
- lowercase/mixed-case column headers in BW, 'BODYWEIGHT' file name alias for BW in study MSY-02
- one subject reported in kg without standardized value; one record with an unmapped unit 'lb'
- partial date with BWDY (resolved, flagged) and partial date without BWDY (quarantined)
- SEND study day 0 (invalid under no-day-zero)
- fasted terminal weight via SUPPBW, plus an unmapped supplemental QNAM
- recovery cohort arm, arm with missing curated planned regimen, ARMCD meaning differs by study
- an exact duplicate record (duplicate extract) and a conflicting duplicate weight
- a weight after death (must be quarantined) and an animal without a predose weight
- an animal with sex 'U' (must not be inferred from weight)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DM_COLS = ["STUDYID", "DOMAIN", "USUBJID", "SUBJID", "SEX", "SPECIES", "STRAIN", "ARMCD", "ARM", "RFSTDTC",
           "RFENDTC"]


def _dm(study, rows):
    return pd.DataFrame([dict(zip(DM_COLS, [study, "DM", *r], strict=True)) for r in rows])


def build(root: Path) -> dict[str, Path]:
    send = root / "send"
    meta = root / "metadata"
    (send / "MSY-01").mkdir(parents=True, exist_ok=True)
    (send / "MSY-02").mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    dm1 = _dm("MSY-01", [
        ["MSY-01-001", "001", "M", "RAT", "SPRAGUE-DAWLEY", "1", "Control", "2021-03-01", "2021-03-30"],
        ["MSY-01-002", "002", "F", "RAT", "SPRAGUE-DAWLEY", "1", "Control", "2021-03-01", "2021-03-30"],
        ["MSY-01-003", "003", "M", "RAT", "SPRAGUE-DAWLEY", "2", "High", "2021-03-01", "2021-03-30"],
        ["MSY-01-004", "004", "F", "RAT", "SPRAGUE-DAWLEY", "2", "High", "2021-03-01", "2021-03-12"],
        ["MSY-01-005", "005", "M", "RAT", "SPRAGUE-DAWLEY", "2R", "High recovery", "2021-03-01", "2021-04-13"],
        ["MSY-01-006", "006", "U", "RAT", "SPRAGUE-DAWLEY", "1", "Control", "2021-03-01", "2021-03-30"],
        ["MSY-01-007", "007", "M", "RAT", "SPRAGUE-DAWLEY", "9", "Unplanned arm", "2021-03-01", "2021-03-30"],
        ["MSY-01-008", "008", "F", "RAT", "SPRAGUE-DAWLEY", "1", "Control", "2021-03-01", "2021-03-30"],
    ])
    bw = [
        # USUBJID, BWSEQ, TESTCD, ORRES, ORRESU, STRESN, STRESU, DTC, DY
        ["MSY-01-001", 1, "BW", "250.1", "g", 250.1, "g", "2021-02-22", -7],
        ["MSY-01-001", 2, "BW", "262.0", "g", 262.0, "g", "2021-03-01", 1],
        ["MSY-01-001", 3, "BW", "301.5", "g", 301.5, "g", "2021-03-08", 8],
        ["MSY-01-001", 4, "BW", "350.2", "g", 350.2, "g", "2021-03", 15],  # partial date + DY
        ["MSY-01-001", 5, "BW", "380.0", "g", 380.0, "g", "2021-03-29", 29],
        ["MSY-01-001", 6, "BW", "365.0", "g", 365.0, "g", "2021-03-30", 30],  # fasted terminal (SUPPBW)
        ["MSY-01-002", 1, "BW", "0.1950", "kg", None, None, "2021-03-01", 1],  # kg, no standardized
        ["MSY-01-002", 2, "BW", "0.2100", "kg", None, None, "2021-03-08", 8],
        ["MSY-01-002", 3, "BW", "0.47", "lb", None, None, "2021-03-15", 15],  # unmapped unit
        ["MSY-01-002", 4, "BW", "230.0", "g", 230.0, "g", "2021-03", None],  # partial, no DY
        ["MSY-01-003", 1, "BW", "255.0", "g", 255.0, "g", "2021-03-01", 1],
        ["MSY-01-003", 2, "BW", "280.0", "g", 280.0, "g", "2021-03-08", 8],
        ["MSY-01-003", 2, "BW", "280.0", "g", 280.0, "g", "2021-03-08", 8],  # exact duplicate (extract)
        ["MSY-01-003", 3, "BW", "300.0", "g", 300.0, "g", "2021-03-15", 15],
        ["MSY-01-003", 4, "BW", "310.0", "g", 310.0, "g", "2021-03-15", 15],  # conflicting duplicate day
        ["MSY-01-003", 5, "BW", "-5", "g", -5.0, "g", "2021-03-22", 22],  # nonpositive
        ["MSY-01-004", 1, "BW", "190.0", "g", 190.0, "g", "2021-03-01", 1],
        ["MSY-01-004", 2, "BW", "185.0", "g", 185.0, "g", "2021-03-08", 8],
        ["MSY-01-004", 3, "BW", "170.0", "g", 170.0, "g", "2021-03-15", 15],  # after death on day 12
        ["MSY-01-005", 1, "BW", "251.0", "g", 251.0, "g", "2021-03-01", 1],
        ["MSY-01-005", 2, "BW", "290.0", "g", 290.0, "g", "2021-03-29", 29],
        ["MSY-01-005", 3, "BW", "330.0", "g", 330.0, "g", "2021-04-12", 43],  # recovery phase
        ["MSY-01-006", 1, "BW", "220.0", "g", 220.0, "g", "2021-03-01", 1],
        ["MSY-01-006", 2, "BW", "240.0", "g", 240.0, "g", "2021-03-08", 8],
        ["MSY-01-007", 1, "BW", "260.0", "g", 260.0, "g", "2021-03-01", 1],
        ["MSY-01-007", 2, "BW", "290.0", "g", 290.0, "g", "2021-03-08", 8],
        ["MSY-01-008", 1, "BW", "201.0", "g", 201.0, "g", "2021-03-08", 8],  # no predose weight
        ["MSY-01-008", 2, "BW", "222.0", "g", 222.0, "g", "2021-03-15", 15],
        ["MSY-01-008", 3, "BW", "230.0", "g", 230.0, "g", "2021-03-16", 0],  # DY 0 invalid
        ["MSY-01-008", 4, "BW", "50", "g", 50.0, "g", "2021-03-17", 17],
    ]
    bw_df = pd.DataFrame(bw, columns=["USUBJID", "BWSEQ", "BWTESTCD", "BWORRES", "BWORRESU", "BWSTRESN",
                                      "BWSTRESU", "BWDTC", "BWDY"])
    bw_df.insert(0, "STUDYID", "MSY-01")
    bw_df["BWTEST"] = "Body Weight"
    bw_df.loc[len(bw_df) - 1, "BWTESTCD"] = "BWGAIN"  # different test code -> not mapped
    bw_df.columns = [c.lower() if i % 2 else c for i, c in enumerate(bw_df.columns)]  # mixed case headers
    bw_df.to_csv(send / "MSY-01" / "bw.csv", index=False)
    dm1.to_csv(send / "MSY-01" / "dm.csv", index=False)
    pd.DataFrame([
        ["MSY-01", "BW", "MSY-01-001", "BWSEQ", "6", "BWFAST", "Fasting Status", "Y"],
        ["MSY-01", "BW", "MSY-01-001", "BWSEQ", "2", "BWNOTE", "Note", "odd"],
    ], columns=["STUDYID", "RDOMAIN", "USUBJID", "IDVAR", "IDVARVAL", "QNAM", "QLABEL", "QVAL"]).to_csv(
        send / "MSY-01" / "suppbw.csv", index=False)
    pd.DataFrame([
        ["MSY-01", "DS", "MSY-01-001", 1, "TERMINAL SACRIFICE", "TERMINAL SACRIFICE", "2021-03-30", 30],
        ["MSY-01", "DS", "MSY-01-004", 1, "FOUND DEAD", "FOUND DEAD", "2021-03-12", 12],
        ["MSY-01", "DS", "MSY-01-005", 1, "RECOVERY SACRIFICE", "RECOVERY SACRIFICE", "2021-04-13", 44],
        ["MSY-01", "DS", "MSY-01-003", 1, "ACCIDENTAL", "ACCIDENTAL", "2021-03-30", 30],  # unmapped term
    ], columns=["STUDYID", "DOMAIN", "USUBJID", "DSSEQ", "DSTERM", "DSDECOD", "DSSTDTC", "DSSTDY"]).to_csv(
        send / "MSY-01" / "ds.csv", index=False)
    pd.DataFrame([
        ["MSY-01", "EX", "MSY-01-003", 1, "CMPX", 100, "mg/kg", "ORAL GAVAGE", "2021-03-01", "2021-03-28"],
        ["MSY-01", "EX", "MSY-01-004", 1, "CMPX", 50, "mg/kg", "ORAL GAVAGE", "2021-03-01", "2021-03-11"],
    ], columns=["STUDYID", "DOMAIN", "USUBJID", "EXSEQ", "EXTRT", "EXDOSE", "EXDOSU", "EXROUTE", "EXSTDTC",
                "EXENDTC"]).to_csv(send / "MSY-01" / "ex.csv", index=False)

    # second study: control is ARMCD "2" here (labels mean different things across studies)
    dm2 = _dm("MSY-02", [
        ["MSY-02-001", "001", "M", "RAT", "SPRAGUE-DAWLEY", "2", "Vehicle", "2022-05-02", "2022-05-31"],
        ["MSY-02-002", "002", "F", "RAT", "SPRAGUE-DAWLEY", "1", "Low", "2022-05-02", "2022-05-31"],
    ])
    dm2.to_parquet(send / "MSY-02" / "dm.parquet", index=False)
    pd.DataFrame([
        ["MSY-02", "MSY-02-001", 1, "BW", "240", "g", 240.0, "g", "2022-05-02", 1, "Body Weight"],
        ["MSY-02", "MSY-02-001", 2, "BW", "270", "g", 270.0, "g", "2022-05-09", 8, "Body Weight"],
        ["MSY-02", "MSY-02-002", 1, "BW", "190", "g", 190.0, "g", "2022-05-02", 1, "Body Weight"],
        ["MSY-02", "MSY-02-002", 2, "BW", "200", "g", 200.0, "g", "2022-05-09", 8, "Body Weight"],
    ], columns=["STUDYID", "USUBJID", "BWSEQ", "BWTESTCD", "BWORRES", "BWORRESU", "BWSTRESN", "BWSTRESU",
                "BWDTC", "BWDY", "BWTEST"]).to_csv(send / "MSY-02" / "BODYWEIGHT.csv", index=False)

    pd.DataFrame([
        ["MSY-01", "RAT", "SPRAGUE-DAWLEY", "REPEAT_DOSE_TOX", "2021-03-01", "3.1"],
        ["MSY-02", "RAT", "SPRAGUE-DAWLEY", "REPEAT_DOSE_TOX", "2022-05-02", "3.1"],
    ], columns=["STUDYID", "SPECIES", "STRAIN", "STUDY_CLASS", "STUDY_START", "SEND_VERSION"]).to_csv(
        meta / "study_metadata.csv", index=False)
    pd.DataFrame([
        ["MSY-01", "1", "", True, 0, "mg/kg", "ORAL GAVAGE", 1, 1, 28, "MAIN"],
        ["MSY-01", "2", "CMPX", False, 100, "mg/kg", "ORAL GAVAGE", 1, 1, 28, "MAIN"],
        ["MSY-01", "2R", "CMPX", False, 100, "mg/kg", "ORAL GAVAGE", 1, 1, 28, "RECOVERY"],
        ["MSY-02", "2", "", True, 0, "mg/kg", "ORAL GAVAGE", 1, 1, 28, "MAIN"],
        ["MSY-02", "1", "CMPX", False, 10, "mg/kg", "ORAL GAVAGE", 1, 1, 28, "MAIN"],
        # ARMCD 9 of MSY-01 intentionally missing (no curated planned regimen)
    ], columns=["STUDYID", "ARMCD", "COMPOUND_ID", "IS_CONTROL", "PLANNED_DOSE", "DOSE_UNIT", "ROUTE",
                "INTERVAL_DAYS", "PLANNED_START_DY", "PLANNED_END_DY", "COHORT"]).to_csv(
        meta / "arm_regimen.csv", index=False)
    pd.DataFrame([["CMPX", 0.1, -0.2, 0.3, "CLASS_A"]], columns=["COMPOUND_ID", "D1", "D2", "D3", "CLASS"]).to_csv(
        meta / "compound_reference.csv", index=False)
    return {"send": send, "metadata": meta}

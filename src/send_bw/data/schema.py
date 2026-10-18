"""Canonical analytical tables (source independent).

All fitting code consumes these tables; nothing downstream may read raw
vendor/SEND structures. Day convention: DAY is elapsed days from the selected
reference date (reference date = elapsed day 0 = SEND study day 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# column -> (kind, required). kinds: str, float, int, bool
CANONICAL_SCHEMAS: dict[str, dict[str, tuple[str, bool]]] = {
    "study_context": {
        "STUDYID": ("str", True),
        "SPECIES": ("str", True),
        "STRAIN": ("str", False),
        "STUDY_CLASS": ("str", True),
        "SEND_VERSION": ("str", False),
        "STUDY_START": ("str", False),
        "MAPPING_VERSION": ("str", True),
        "SOURCE_ID": ("str", True),
    },
    "animal_context": {
        "STUDYID": ("str", True),
        "USUBJID": ("str", True),
        "SUBJID": ("str", False),
        "SEX": ("str", True),
        "SPECIES": ("str", True),
        "STRAIN": ("str", False),
        "ARMCD": ("str", True),
        "ARM": ("str", False),
        "ARM_KEY": ("str", True),
        "COHORT": ("str", True),
        "REF_DATE": ("str", False),
        "BASELINE_DAY": ("float", False),
        "BASELINE_G": ("float", False),
        "BASELINE_LOGW": ("float", False),
        "ELIGIBLE": ("bool", True),
        "EXCLUSION_REASON": ("str", False),
    },
    "arm_regimen": {
        "STUDYID": ("str", True),
        "ARMCD": ("str", True),
        "ARM_KEY": ("str", True),
        "COMPOUND_ID": ("str", False),
        "IS_CONTROL": ("bool", True),
        "PLANNED_DOSE": ("float", False),
        "DOSE_UNIT": ("str", False),
        "ROUTE": ("str", False),
        "INTERVAL_DAYS": ("float", False),
        "START_DAY": ("float", False),
        "END_DAY": ("float", False),
        "COHORT": ("str", True),
        "REGIMEN_AVAILABLE": ("bool", True),
    },
    "administrations": {
        "STUDYID": ("str", True),
        "USUBJID": ("str", True),
        "EXSEQ": ("float", False),
        "EXTRT": ("str", False),
        "EXDOSE": ("float", False),
        "EXDOSU": ("str", False),
        "EXROUTE": ("str", False),
        "START_DAY": ("float", False),
        "END_DAY": ("float", False),
        "SRC_FILE": ("str", True),
        "SRC_ROW": ("int", True),
    },
    "bw_obs": {
        "STUDYID": ("str", True),
        "USUBJID": ("str", True),
        "BWSEQ": ("float", False),
        "DAY": ("float", True),
        "SOURCE_DY": ("float", False),
        "PHASE": ("str", True),
        "WEIGHT_G": ("float", True),
        "LOGW": ("float", True),
        "ORIG_VALUE": ("str", False),
        "ORIG_UNIT": ("str", False),
        "STD_VALUE": ("float", False),
        "STD_UNIT": ("str", False),
        "DATE_UNCERTAIN": ("bool", True),
        "FASTING": ("bool", True),
        "IS_BASELINE": ("bool", True),
        "QC_FLAGS": ("str", False),
        "SRC_FILE": ("str", True),
        "SRC_ROW": ("int", True),
        "MAPPING_VERSION": ("str", True),
    },
    "animal_status": {
        "STUDYID": ("str", True),
        "USUBJID": ("str", True),
        "DISPOSITION": ("str", True),
        "DISPOSITION_DAY": ("float", False),
        "SOURCE_TERM": ("str", False),
    },
    "quarantine": {
        "TABLE": ("str", True),
        "STUDYID": ("str", False),
        "USUBJID": ("str", False),
        "REASON": ("str", True),
        "SEVERITY": ("str", True),
        "SRC_FILE": ("str", False),
        "SRC_ROW": ("float", False),
        "DETAIL": ("str", False),
    },
}

DISPOSITIONS = {
    "SCHEDULED_SACRIFICE",
    "RECOVERY_SACRIFICE",
    "DEATH",
    "UNSCHEDULED_REMOVAL",
    "ONGOING_OR_UNKNOWN",
}
COHORTS = {"MAIN", "RECOVERY", "SATELLITE", "UNKNOWN"}
PHASES = {"PRETREATMENT", "TREATMENT", "RECOVERY"}


class SchemaError(ValueError):
    pass


@dataclass
class CanonicalTables:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, k: str) -> pd.DataFrame:
        return self.tables[k]

    def __setitem__(self, k: str, v: pd.DataFrame) -> None:
        self.tables[k] = v


def empty_table(name: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=_pd_dtype(k)) for c, (k, _) in CANONICAL_SCHEMAS[name].items()})


def _pd_dtype(kind: str) -> str:
    return {"str": "object", "float": "float64", "int": "int64", "bool": "bool"}[kind]


def coerce_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Order columns, add optional columns as missing, coerce types, verify required."""
    schema = CANONICAL_SCHEMAS[name]
    out = pd.DataFrame(index=df.index)
    for col, (kind, required) in schema.items():
        if col not in df.columns:
            if required:
                raise SchemaError(f"{name}: required column {col} missing")
            out[col] = pd.Series([None] * len(df), index=df.index, dtype=_pd_dtype(kind) if kind != "bool" else "object")
            if kind == "float":
                out[col] = out[col].astype("float64")
            continue
        s = df[col]
        if kind == "float":
            s = pd.to_numeric(s, errors="raise").astype("float64")
        elif kind == "int":
            s = pd.to_numeric(s, errors="raise").astype("int64")
        elif kind == "bool":
            if s.isna().any():
                raise SchemaError(f"{name}.{col}: missing values in boolean column")
            s = s.astype(bool)
        else:
            s = s.astype("object").where(s.notna(), None)
        if required and s.isna().any():
            raise SchemaError(f"{name}.{col}: missing values in required column")
        out[col] = s
    extra = [c for c in df.columns if c not in schema]
    for c in extra:
        out[c] = df[c]
    return out.reset_index(drop=True)


def validate_keys(df: pd.DataFrame, keys: list[str], table: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        raise SchemaError(f"{table}: duplicate keys {keys}: {df.loc[dup, keys].head(5).to_dict('records')}")


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: list[str], how: str = "left",
               validate: str = "many_to_one", name: str = "merge") -> pd.DataFrame:
    """Merge that refuses many-to-many inflation and verifies row counts."""
    n0 = len(left)
    out = left.merge(right, on=on, how=how, validate=validate)
    if how == "left" and len(out) != n0:
        raise SchemaError(f"{name}: row inflation {n0} -> {len(out)}")
    return out

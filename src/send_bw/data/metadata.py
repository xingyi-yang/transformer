"""Version-aware mapping profiles.

A mapping profile is a reviewed YAML document that declares how a sponsor's
SEND extracts map to canonical tables: accepted test codes, unit factors,
timing reference and day convention, supplemental qualifiers, disposition
terms and study-specific overrides. Nothing is guessed: unmapped values are
quarantined or reported.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class BaselineRule(BaseModel):
    max_day: float = 0.0  # last eligible elapsed day (0 == SEND day 1, assumed predose)
    window_days: float = 7.0  # earliest eligible = max_day - window_days


class TimingRule(BaseModel):
    reference: str = "RFSTDTC"
    no_day_zero: bool = True
    partial_date_rule: str = "study_day_else_quarantine"
    baseline: BaselineRule = Field(default_factory=BaselineRule)

    @field_validator("partial_date_rule")
    @classmethod
    def _rule(cls, v: str) -> str:
        allowed = {"study_day_else_quarantine", "quarantine"}
        if v not in allowed:
            raise ValueError(f"partial_date_rule must be one of {allowed}")
        return v


class BodyWeightRule(BaseModel):
    testcd: list[str] = Field(default_factory=lambda: ["BW"])
    value_priority: list[str] = Field(default_factory=lambda: ["standardized", "original"])
    units_to_grams: dict[str, float] = Field(default_factory=lambda: {"g": 1.0, "kg": 1000.0})
    exclude_fasted: bool = True

    def unit_factor(self, unit: Any) -> float | None:
        if unit is None or (isinstance(unit, float) and unit != unit):
            return None
        key = str(unit).strip().lower()
        lowered = {k.lower(): v for k, v in self.units_to_grams.items()}
        return lowered.get(key)


class MappingProfile(BaseModel):
    mapping_version: str
    send_version: str
    description: str = ""
    domain_aliases: dict[str, str] = Field(default_factory=dict)
    column_renames: dict[str, dict[str, str]] = Field(default_factory=dict)
    body_weight: BodyWeightRule = Field(default_factory=BodyWeightRule)
    timing: TimingRule = Field(default_factory=TimingRule)
    supplemental: dict[str, dict[str, str]] = Field(default_factory=dict)
    supplemental_true_values: list[str] = Field(default_factory=lambda: ["Y", "YES", "TRUE", "1"])
    disposition_terms: dict[str, str] = Field(default_factory=dict)
    sex_values: dict[str, str] = Field(default_factory=lambda: {"M": "M", "F": "F"})
    species_values: dict[str, str] = Field(default_factory=dict)
    regimen_source: str = "curated"
    study_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("regimen_source")
    @classmethod
    def _regimen_source(cls, v: str) -> str:
        if v != "curated":
            raise ValueError(
                "Only curated planned-regimen metadata is supported; deriving planned regimens from "
                "actual EX administrations or TX strings is not allowed."
            )
        return v

    def for_study(self, studyid: str) -> MappingProfile:
        ov = self.study_overrides.get(studyid)
        if not ov:
            return self
        base = self.model_dump()
        merged = _deep_merge(base, ov)
        merged["study_overrides"] = {}
        return MappingProfile(**merged)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_mapping(path: str | Path) -> MappingProfile:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return MappingProfile(**raw)

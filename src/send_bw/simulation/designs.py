"""Future-study requests, applicability gates and candidate designs.

Candidate d = (n per sex per group [unequal allocation allowed], analytical schedule, T, analysis).
Every schedule implicitly includes the baseline (day 0) and must end at T. Schedules are
ANALYTICAL measurement schedules only: protocol-required welfare weighings/observations are
outside the optimization and are never reduced by it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from send_bw.data.features import DESCRIPTORS, FeatureSpec


class GroupSpec(BaseModel):
    name: str
    is_control: bool = False
    compound_id: str | None = None
    dose: float = 0.0
    dose_unit: str = "mg/kg"
    route: str = "ORAL GAVAGE"
    interval_days: float = 1.0
    start_dy: int = 1
    end_dy: int

    @model_validator(mode="after")
    def _check(self):
        if not self.is_control and (self.compound_id is None or self.dose <= 0):
            raise ValueError(f"active group {self.name} needs compound_id and a positive planned dose")
        if self.start_dy == 0 or self.end_dy == 0:
            raise ValueError("SEND study day 0 does not exist")
        return self


class Request(BaseModel):
    request_id: str
    species: str
    strain: str
    study_class: str = "REPEAT_DOSE_TOX"
    sexes: list[str] = Field(default_factory=lambda: ["M", "F"])
    sex_strategy: str = "pooled_with_sex_covariate"
    baseline_distribution: dict[str, dict[str, float]]
    groups: list[GroupSpec]
    compound_descriptors: dict[str, dict[str, float]] = Field(default_factory=dict)
    horizon_day: float
    estimand: str = "fixed_time_log_weight_difference"
    meaningful_effects_pct: dict[str, float]
    alpha: float = 0.05
    multiplicity: str = "holm"
    power_target: float = 0.8
    power_requirement: str = "all_required"
    required_comparisons: list[str] | None = None
    fwer_max: float = 0.05
    fwer_tolerance: float = 0.01
    fwer_upper_tolerance: float = 0.04
    power_lower_tolerance: float = 0.05
    coverage_min: float = 0.92
    max_abs_bias_fraction: float = 0.15
    max_failure_rate: float = 0.02
    precision_max_halfwidth: float | None = None
    ci_level: float = 0.95
    constraints: dict[str, Any] = Field(default_factory=lambda: {"min_per_sex_per_group": 3,
                                                                  "max_total_animals": 200})
    candidate_n_per_sex: list[int] = Field(default_factory=lambda: [4, 6, 8, 10])
    candidate_allocations: dict[str, dict[str, float]] = Field(default_factory=lambda: {"equal": {}})
    candidate_schedules: dict[str, list[float]]
    analyses: list[str] = Field(default_factory=lambda: ["ancova", "gls"])
    gls_variance: str = "model"
    missing_data: dict[str, float] = Field(default_factory=lambda: {"p_missed_visit": 0.02, "p_dropout": 0.02})
    costs: dict[str, float] = Field(default_factory=lambda: {"per_animal": 1.0, "per_measurement": 0.02,
                                                              "per_day": 0.0})
    missing_feature_declarations: list[str] = Field(default_factory=list)
    # secondary estimand: fixed query times/weights, independent of the observation schedule
    secondary_contrast: dict[str, list[float]] | None = None
    welfare_note: str = ("Analytical schedule only; protocol-required welfare weighings and clinical "
                         "observations are performed regardless and are not optimized.")

    @model_validator(mode="after")
    def _check(self):
        ctrl = [g for g in self.groups if g.is_control]
        if len(ctrl) != 1:
            raise ValueError("exactly one concurrent control group is required")
        active = [g.name for g in self.groups if not g.is_control]
        missing = set(active) - set(self.meaningful_effects_pct)
        if missing:
            raise ValueError(f"meaningful effects missing for {sorted(missing)}")
        for name, s in self.candidate_schedules.items():
            if not s or max(s) != self.horizon_day or min(s) <= 0 or sorted(s) != list(s):
                raise ValueError(f"schedule {name} must be increasing, > 0 and end at the horizon")
        if self.multiplicity != "holm":
            raise ValueError("only Holm multiplicity is implemented")
        if self.sex_strategy != "pooled_with_sex_covariate":
            raise ValueError("only pooled_with_sex_covariate is implemented")
        for a in self.analyses:
            if a not in {"ancova", "gls", "mmrm"}:
                raise ValueError(f"unknown analysis {a}")
        if self.secondary_contrast is not None:
            t, w = self.secondary_contrast.get("times", []), self.secondary_contrast.get("weights", [])
            if not t or len(t) != len(w) or min(t) <= 0 or max(t) > self.horizon_day or abs(sum(w) - 1) > 1e-8:
                raise ValueError("secondary_contrast needs times in (0, T] and weights summing to 1")
        return self

    @property
    def secondary(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.secondary_contrast is None:
            return None
        return (np.array(self.secondary_contrast["times"], float), np.array(self.secondary_contrast["weights"], float))

    @property
    def control(self) -> GroupSpec:
        return next(g for g in self.groups if g.is_control)

    @property
    def active(self) -> list[GroupSpec]:
        return [g for g in self.groups if not g.is_control]

    @property
    def deltas(self) -> np.ndarray:
        return np.log1p(np.array([self.meaningful_effects_pct[g.name] for g in self.active]) / 100.0)

    @property
    def required(self) -> list[int]:
        names = [g.name for g in self.active]
        req = self.required_comparisons or names
        return [names.index(r) for r in req]

    def regimens(self, all_control: bool = False) -> pd.DataFrame:
        rows = []
        for g in self.groups:
            ctrl = g.is_control or all_control
            rows.append({"ARM_KEY": g.name, "IS_CONTROL": ctrl, "COMPOUND_ID": None if ctrl else g.compound_id,
                         "PLANNED_DOSE": 0.0 if ctrl else g.dose, "ROUTE": g.route.upper(),
                         "INTERVAL_DAYS": g.interval_days, "START_DAY": float(g.start_dy - 1 if g.start_dy > 0
                                                                               else g.start_dy),
                         "END_DAY": float(g.end_dy - 1 if g.end_dy > 0 else g.end_dy)})
        return pd.DataFrame(rows)

    def compound_reference(self, base: pd.DataFrame) -> pd.DataFrame:
        extra = [{"COMPOUND_ID": c, **v} for c, v in self.compound_descriptors.items()]
        if not extra:
            return base
        ex = pd.DataFrame(extra)
        return pd.concat([base[~base["COMPOUND_ID"].isin(ex["COMPOUND_ID"])], ex], ignore_index=True)


def load_request(path: str | Path) -> Request:
    with open(path, encoding="utf-8") as f:
        return Request(**yaml.safe_load(f))


def validate_request(req: Request, spec: FeatureSpec, compound_ref: pd.DataFrame) -> dict[str, Any]:
    """Applicability gates. Returns status 'supported' or 'not_supported' (never extrapolates silently)."""
    problems: list[str] = []
    warnings: list[str] = []
    cref = req.compound_reference(compound_ref)
    ctx = reference_context(req, 1, np.random.default_rng(0)).merge(req.regimens(), on="ARM_KEY")
    problems += spec.check_context(ctx, cref, horizon=req.horizon_day)
    for g in req.active:
        if g.compound_id not in set(cref["COMPOUND_ID"]):
            problems.append(f"compound {g.compound_id}: no curated descriptors (provide compound_descriptors)")
        elif g.compound_id not in spec.compound_ids:
            warnings.append(f"compound {g.compound_id} not in training; prediction relies on descriptors "
                            f"{DESCRIPTORS} (new-compound extrapolation)")
    if req.species.upper() != spec.species:
        problems.append(f"species {req.species} not supported")
    for s in req.sexes:
        if s not in req.baseline_distribution:
            problems.append(f"baseline distribution missing for sex {s}")
    for decl in req.missing_feature_declarations:
        if decl in {"planned_regimen", "dose", "compound_identity"}:
            problems.append(f"declared missing feature '{decl}' is required for pre-study prediction")
        else:
            warnings.append(f"declared missing feature '{decl}'")
    return {"status": "supported" if not problems else "not_supported", "problems": problems,
            "warnings": warnings}


@dataclass(frozen=True)
class Candidate:
    n_per_sex: tuple[int, ...]  # per group, control first
    schedule_name: str
    schedule: tuple[float, ...]
    analysis: str
    allocation: str = "equal"

    @property
    def label(self) -> str:
        return f"{self.analysis}|{self.schedule_name}|n={'/'.join(map(str, self.n_per_sex))}|{self.allocation}"

    def n_animals(self, n_sexes: int) -> int:
        return int(sum(self.n_per_sex) * n_sexes)

    def n_measurements(self, n_sexes: int) -> int:
        """Analytical weighings: baseline + scheduled visits for every animal (ignoring dropouts)."""
        return self.n_animals(n_sexes) * (1 + len(self.schedule))


def enumerate_candidates(req: Request) -> list[Candidate]:
    out = []
    G = len(req.groups)
    mn = req.constraints.get("min_per_sex_per_group", 3)
    mx_total = req.constraints.get("max_total_animals", 10**9)
    names = [g.name for g in req.groups]
    for n, (aname, alloc), (sname, sched), an in itertools.product(
            req.candidate_n_per_sex, req.candidate_allocations.items(), req.candidate_schedules.items(),
            req.analyses):
        if an == "ancova" and len(sched) > 1:
            # ANCOVA ignores intermediate visits: only evaluated on the endpoint-only form of a schedule,
            # so it can never appear to benefit from extra visits.
            sched = [req.horizon_day]
            sname = "endpoint_only"
        if an in {"gls", "mmrm"} and len(sched) == 1:
            continue  # single-visit repeated measures duplicates endpoint ANCOVA
        per = tuple(int(round(n * alloc.get(nm, 1.0))) for nm in names)
        if min(per) < mn or sum(per) * len(req.sexes) > mx_total:
            continue
        out.append(Candidate(per, sname, tuple(float(x) for x in sched), an, aname))
    uniq = list(dict.fromkeys(out))
    assert all(len(c.n_per_sex) == G for c in uniq)
    return uniq


def reference_context(req: Request, n_per_sex: int | tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    """Animals ordered group -> sex -> index; the first n per (group, sex) form nested subsets."""
    if isinstance(n_per_sex, int):
        n_per_sex = tuple([n_per_sex] * len(req.groups))
    rows = []
    for gi, (g, n) in enumerate(zip(req.groups, n_per_sex, strict=True)):
        for s in req.sexes:
            bd = req.baseline_distribution[s]
            sd_log = np.sqrt(np.log1p((bd["sd_g"] / bd["mean_g"]) ** 2))
            mu_log = np.log(bd["mean_g"]) - sd_log**2 / 2
            zb = rng.normal(mu_log, sd_log, size=n)
            for j in range(n):
                rows.append({"STUDYID": "SIM", "USUBJID": f"SIM-{g.name}-{s}-{j:03d}", "ARM_KEY": g.name,
                             "GROUP": gi, "SEX": s, "IDX": j, "BASELINE_LOGW": float(zb[j]),
                             "SPECIES": req.species.upper(), "STRAIN": req.strain.upper()})
    return pd.DataFrame(rows)

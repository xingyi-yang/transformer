"""QC, cohort selection, exclusion and completeness reporting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.schema import CanonicalTables
from send_bw.sharing import parse_sharing, uses_donors

# Historical stages 1–2 constant. Selective v1 is implemented; the frozen k=0/k=1
# executor still refuses it. Do not use this string as a runtime raise.
SELECTIVE_SHARING_DEFERRED = (
    "selective sharing v1 is implemented in the v3 evaluation benchmark; "
    "the frozen k=0/k=1 executor still refuses it"
)


@dataclass
class CohortConfig:
    species: str = "RAT"
    study_classes: list[str] = field(default_factory=lambda: ["REPEAT_DOSE_TOX"])
    sexes: list[str] = field(default_factory=lambda: ["M", "F"])
    require_regimen: bool = True
    exclude_fasted: bool = True
    min_post_baseline_obs: int = 1
    max_day: float | None = None  # restrict modeled outcomes to DAY <= max_day (e.g. analysis horizon)
    exclude_uncertain_dates: bool = False
    donor_species: list[str] = field(default_factory=list)
    sharing: str = "context_specific"
    n_donor_studies: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> CohortConfig:
        raw = dict(d or {})
        raw["sharing"] = parse_sharing(raw.get("sharing"))
        raw["donor_species"] = [str(s).upper() for s in (raw.get("donor_species") or [])]
        raw["n_donor_studies"] = int(raw.get("n_donor_studies") or 0)
        return CohortConfig(**raw)

    def allowed_species(self) -> set[str]:
        out = {self.species.upper()}
        if uses_donors(self.sharing):
            out.update(s.upper() for s in self.donor_species)
        return out

    def target_only(self) -> CohortConfig:
        return replace(self, sharing="context_specific", donor_species=[], n_donor_studies=0)


@dataclass
class Cohort:
    animals: pd.DataFrame
    obs: pd.DataFrame  # post-baseline modelling observations of eligible animals
    exclusions: pd.DataFrame
    completeness: pd.DataFrame
    limitations: list[str]
    qc_checks: dict[str, Any]


def select_cohort(t: CanonicalTables, cfg: CohortConfig) -> Cohort:
    animals = t["animal_context"].copy()
    studies = t["study_context"].set_index("STUDYID")
    reg = t["arm_regimen"].set_index("ARM_KEY")
    status = t["animal_status"].set_index("USUBJID")
    obs = t["bw_obs"].copy()

    reasons: dict[str, list[str]] = {u: ([r] if r else []) for u, r in
                                     zip(animals["USUBJID"], animals["EXCLUSION_REASON"].fillna(""), strict=True)}
    for r in animals.to_dict("records"):
        u = r["USUBJID"]
        reasons[u] = [x for x in ";".join(reasons[u]).split(";") if x]
        st = studies.loc[r["STUDYID"]] if r["STUDYID"] in studies.index else None
        if st is None or str(st["SPECIES"]).upper() not in cfg.allowed_species():
            reasons[u].append("species_out_of_scope")
        if st is None or st["STUDY_CLASS"] not in cfg.study_classes:
            reasons[u].append("study_class_out_of_scope")
        if r["SEX"] not in cfg.sexes:
            reasons[u].append("sex_out_of_scope")
        if cfg.require_regimen and (r["ARM_KEY"] not in reg.index or not bool(reg.loc[r["ARM_KEY"], "REGIMEN_AVAILABLE"])):
            reasons[u].append("planned_regimen_unavailable")
    if cfg.exclude_fasted:
        obs = obs[~obs["FASTING"]]
    if cfg.exclude_uncertain_dates:
        obs = obs[~obs["DATE_UNCERTAIN"]]
    post = obs[(~obs["IS_BASELINE"]) & (obs["PHASE"] != "PRETREATMENT")]
    if cfg.max_day is not None:
        post = post[post["DAY"] <= cfg.max_day]
    n_post = post.groupby("USUBJID").size()
    for u in animals["USUBJID"]:
        if n_post.get(u, 0) < cfg.min_post_baseline_obs:
            reasons[u].append("too_few_post_baseline_obs")
    animals["EXCLUSION_REASON"] = animals["USUBJID"].map(lambda u: ";".join(dict.fromkeys(reasons[u])) or None)
    animals["ELIGIBLE"] = animals["EXCLUSION_REASON"].isna()
    elig = animals[animals["ELIGIBLE"]].copy()
    elig = elig.merge(reg[["COMPOUND_ID", "IS_CONTROL", "PLANNED_DOSE", "ROUTE", "INTERVAL_DAYS", "START_DAY",
                           "END_DAY"]].reset_index(), on="ARM_KEY", how="left", validate="many_to_one",
                      suffixes=("", "_REG"))
    elig["DISPOSITION"] = elig["USUBJID"].map(status["DISPOSITION"])
    elig["DISPOSITION_DAY"] = elig["USUBJID"].map(status["DISPOSITION_DAY"])
    mobs = post[post["USUBJID"].isin(set(elig["USUBJID"]))].copy()
    mobs = mobs.merge(elig[["USUBJID", "BASELINE_LOGW", "BASELINE_DAY"]], on="USUBJID", how="left",
                      validate="many_to_one")
    mobs["Y"] = mobs["LOGW"] - mobs["BASELINE_LOGW"]

    exclusions = (animals[~animals["ELIGIBLE"]][["STUDYID", "USUBJID", "SEX", "ARM_KEY", "EXCLUSION_REASON"]]
                  .reset_index(drop=True))
    completeness = _completeness(animals, t["bw_obs"], reg)
    limitations = []
    q = t["quarantine"]
    if len(q):
        for (reason, sev), n in q.groupby(["REASON", "SEVERITY"]).size().items():
            limitations.append(f"{n} record(s) quarantined/flagged: {reason} ({sev})")
    n_unc = int(t["bw_obs"]["DATE_UNCERTAIN"].sum())
    if n_unc:
        limitations.append(f"{n_unc} weight(s) dated from SEND study day because BWDTC was partial (DATE_UNCERTAIN)")
    adm = t["administrations"]
    if "DEVIATION" in adm.columns and adm["DEVIATION"].notna().any():
        limitations.append(f"{int(adm['DEVIATION'].notna().sum())} administration record(s) deviate from the "
                           "planned regimen; models use planned regimens only, so outcomes include deviation effects")
    checks = run_qc_checks(t)
    return Cohort(elig.reset_index(drop=True), mobs.reset_index(drop=True), exclusions, completeness, limitations,
                  checks)


def _completeness(animals: pd.DataFrame, obs: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    o = obs.merge(animals[["USUBJID", "ARM_KEY", "SEX", "COHORT"]], on="USUBJID", how="left",
                  validate="many_to_one")
    g = o.groupby(["STUDYID", "ARM_KEY", "COHORT", "SEX", "PHASE"])
    comp = g.agg(n_animals=("USUBJID", "nunique"), n_obs=("DAY", "size"), n_days=("DAY", "nunique")).reset_index()
    comp["expected_obs"] = comp["n_animals"] * comp["n_days"]
    comp["completeness"] = comp["n_obs"] / comp["expected_obs"]
    return comp


def run_qc_checks(t: CanonicalTables) -> dict[str, Any]:
    obs, animals = t["bw_obs"], t["animal_context"]
    checks = {
        "positive_weights": bool((obs["WEIGHT_G"] > 0).all()),
        "finite_log_weights": bool(np.isfinite(obs["LOGW"]).all()),
        "unique_animal_keys": not animals["USUBJID"].duplicated().any(),
        "unique_obs_keys": not obs.duplicated(["USUBJID", "DAY", "FASTING"]).any(),
        "all_obs_have_animal": bool(obs["USUBJID"].isin(set(animals["USUBJID"])).all()),
        "arm_keys_resolved": bool(animals["ARM_KEY"].isin(set(t["arm_regimen"]["ARM_KEY"])).all()),
        "one_baseline_per_animal_max": bool(obs[obs["IS_BASELINE"]].groupby("USUBJID").size().max() <= 1)
        if obs["IS_BASELINE"].any() else True,
        "sex_values": sorted(animals["SEX"].unique().tolist()),
        "n_quarantined_hard": int((t["quarantine"]["SEVERITY"] == "hard").sum()),
        "n_quarantined_soft": int((t["quarantine"]["SEVERITY"] == "soft").sum()),
    }
    checks["passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks

"""Pre-study features from canonical tables only (planned regimens, biology, baseline, time).

Every feature is classified in FEATURE_REGISTRY. Pre-study (primary task) models may use only
'production' features. Actual administrations are 'optional' (post-hoc only) and never used for
pre-study prediction; synthetic generator quantities are 'truth_only' and never available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

T_REF = 28.0
DESCRIPTORS = ["D1", "D2", "D3"]

FEATURE_REGISTRY: dict[str, str] = {
    "SEX_M": "production", "ZB_C": "production", "IS_CONTROL": "production", "LD": "production",
    "INV_INTERVAL": "production", "D1": "production", "D2": "production", "D3": "production",
    "TAU": "production", "EXPO": "production", "OFF": "production", "LCUM": "production",
    "START": "production", "END": "production",
    "EXDOSE_ACTUAL": "optional", "ACTUAL_END_DAY": "optional", "EARLY_HISTORY": "optional",
    "Z0_TRUE": "truth_only", "Z_LATENT": "truth_only", "U_STUDY_SLOPE": "truth_only",
    "U_STUDY_BASELINE": "truth_only", "DOSE_DEVIATION": "truth_only", "DEATH_DAY": "truth_only",
    "ACTUAL_DOSE": "truth_only",
}
PREFIX_REGISTRY = {"STRAIN_": "production", "ROUTE_": "production"}


def availability(col: str) -> str:
    if col in FEATURE_REGISTRY:
        return FEATURE_REGISTRY[col]
    for p, a in PREFIX_REGISTRY.items():
        if col.startswith(p):
            return a
    raise KeyError(f"feature {col} is not classified in the registry")


def check_no_leakage(cols: list[str], allowed: tuple[str, ...] = ("production",)) -> None:
    bad = [c for c in cols if availability(c) not in allowed]
    if bad:
        raise ValueError(f"features not available pre-study: {bad}")


class UnsupportedContextError(ValueError):
    pass


@dataclass
class FeatureSpec:
    """Fitted on training studies only; persisted with every model."""

    species: str
    strains: list[str]
    routes: list[str]
    sexes: list[str]
    zb_mean: dict[str, float]
    max_dose: float
    max_day: float
    intervals: list[float]
    compound_ids: list[str]
    descriptor_ranges: dict[str, list[float]]
    t_ref: float = T_REF
    version: str = "features-v1"
    animal_cols: list[str] = field(default_factory=list)
    time_cols: list[str] = field(default_factory=list)

    @staticmethod
    def fit(animals: pd.DataFrame, obs: pd.DataFrame, compound_ref: pd.DataFrame, species: str) -> FeatureSpec:
        cref = compound_ref.set_index("COMPOUND_ID")
        cids = sorted(set(animals["COMPOUND_ID"].dropna()))
        rng = {d: [float(cref.loc[cids, d].astype(float).min()), float(cref.loc[cids, d].astype(float).max())]
               for d in DESCRIPTORS} if cids else {d: [0.0, 0.0] for d in DESCRIPTORS}
        spec = FeatureSpec(
            species=species, strains=sorted(animals["STRAIN"].dropna().unique().tolist()),
            routes=sorted(animals["ROUTE"].dropna().unique().tolist()),
            sexes=sorted(animals["SEX"].unique().tolist()),
            zb_mean={s: float(g["BASELINE_LOGW"].mean()) for s, g in animals.groupby("SEX")},
            max_dose=float(animals["PLANNED_DOSE"].max()), max_day=float(obs["DAY"].max()),
            intervals=sorted({float(x) for x in animals["INTERVAL_DAYS"].dropna()}),
            compound_ids=cids, descriptor_ranges=rng)
        spec.animal_cols = (["SEX_M", "ZB_C", "IS_CONTROL", "LD", "INV_INTERVAL", "START", "END"] + DESCRIPTORS
                            + [f"STRAIN_{s}" for s in spec.strains[1:]] + [f"ROUTE_{r}" for r in spec.routes[1:]])
        spec.time_cols = ["TAU", "EXPO", "OFF", "LCUM"]
        check_no_leakage(spec.animal_cols + spec.time_cols)
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FeatureSpec:
        return FeatureSpec(**d)

    # ------------------------------------------------------------- applicability gates
    def check_context(self, animals: pd.DataFrame, compound_ref: pd.DataFrame, horizon: float | None = None,
                      dose_extrapolation_factor: float = 1.0) -> list[str]:
        problems = []
        cref = compound_ref.set_index("COMPOUND_ID") if len(compound_ref) else pd.DataFrame()
        for s in set(animals.get("SPECIES", pd.Series([self.species]))):
            if str(s).upper() != self.species:
                problems.append(f"species {s} not supported (trained on {self.species})")
        for st in set(animals["STRAIN"].dropna()):
            if st not in self.strains:
                problems.append(f"strain {st} not seen in training; no silent mapping to a familiar strain")
        for r in set(animals["ROUTE"].dropna()):
            if r not in self.routes:
                problems.append(f"route {r} not seen in training")
        for sx in set(animals["SEX"]):
            if sx not in self.sexes:
                problems.append(f"sex {sx} not supported")
        if animals["PLANNED_DOSE"].max() > self.max_dose * dose_extrapolation_factor:
            problems.append(f"planned dose {animals['PLANNED_DOSE'].max()} exceeds training max {self.max_dose}")
        for iv in set(animals["INTERVAL_DAYS"].dropna()):
            if float(iv) not in self.intervals:
                problems.append(f"dosing interval {iv} not seen in training")
        for c in set(animals["COMPOUND_ID"].dropna()):
            if c not in cref.index or cref.loc[c, DESCRIPTORS].isna().any():
                problems.append(f"compound {c}: curated descriptors missing")
                continue
            for d in DESCRIPTORS:
                lo, hi = self.descriptor_ranges[d]
                v = float(cref.loc[c, d])
                if v < lo - 0.25 * (hi - lo) or v > hi + 0.25 * (hi - lo):
                    problems.append(f"compound {c}: descriptor {d}={v:.2f} outside training range [{lo:.2f},{hi:.2f}]")
        if horizon is not None and horizon > self.max_day:
            problems.append(f"horizon {horizon} beyond training maximum day {self.max_day}")
        return problems

    # ------------------------------------------------------------- feature matrices
    def animal_features(self, animals: pd.DataFrame, compound_ref: pd.DataFrame) -> pd.DataFrame:
        cref = compound_ref.set_index("COMPOUND_ID")
        a = animals
        out = pd.DataFrame(index=a.index)
        out["SEX_M"] = (a["SEX"] == "M").astype(float)
        out["ZB_C"] = a["BASELINE_LOGW"].astype(float) - a["SEX"].map(self.zb_mean).astype(float)
        ctrl = a["IS_CONTROL"].astype(bool)
        dose = a["PLANNED_DOSE"].astype(float).where(~ctrl, 0.0).fillna(0.0)
        out["IS_CONTROL"] = ctrl.astype(float)
        out["LD"] = np.log1p(dose / 10.0)
        out["INV_INTERVAL"] = (1.0 / a["INTERVAL_DAYS"].astype(float).fillna(1.0)).where(~ctrl, 0.0)
        out["START"] = a["START_DAY"].astype(float).fillna(0.0) / self.t_ref
        out["END"] = a["END_DAY"].astype(float).fillna(0.0) / self.t_ref
        for d in DESCRIPTORS:
            vals = a["COMPOUND_ID"].map(lambda c, d=d: float(cref.loc[c, d]) if isinstance(c, str) and c in cref.index
                                        else np.nan)
            if (~ctrl & vals.isna()).any():
                raise UnsupportedContextError(f"missing curated descriptor {d} for an active arm")
            out[d] = vals.where(~ctrl, 0.0).fillna(0.0)
        for s in self.strains[1:]:
            out[f"STRAIN_{s}"] = (a["STRAIN"] == s).astype(float)
        for r in self.routes[1:]:
            out[f"ROUTE_{r}"] = (a["ROUTE"] == r).astype(float)
        return out[self.animal_cols]

    def time_features(self, day: np.ndarray, animals_rows: pd.DataFrame) -> pd.DataFrame:
        """Planned-exposure features at `day` for each row (rows aligned with day)."""
        day = np.asarray(day, float)
        ctrl = animals_rows["IS_CONTROL"].astype(bool).to_numpy()
        start = animals_rows["START_DAY"].astype(float).fillna(0.0).to_numpy()
        end = animals_rows["END_DAY"].astype(float).fillna(0.0).to_numpy()
        iv = animals_rows["INTERVAL_DAYS"].astype(float).fillna(1.0).to_numpy()
        dose = np.where(ctrl, 0.0, animals_rows["PLANNED_DOSE"].astype(float).fillna(0.0).to_numpy())
        expo = np.clip(day - start, 0.0, np.maximum(end + 1.0 - start, 0.0))
        off = np.clip(day - end - 1.0, 0.0, None)
        ndose = np.where(day >= start, np.floor(np.clip(np.minimum(day, end) - start, 0, None) / iv) + 1.0, 0.0)
        out = pd.DataFrame({"TAU": day / self.t_ref, "EXPO": np.where(ctrl, 0.0, expo / self.t_ref),
                            "OFF": np.where(ctrl, 0.0, off / self.t_ref),
                            "LCUM": np.log1p(ndose * dose / 100.0)}, index=animals_rows.index)
        return out[self.time_cols]

    def observation_features(self, obs: pd.DataFrame, animals: pd.DataFrame,
                             compound_ref: pd.DataFrame) -> pd.DataFrame:
        af = self.animal_features(animals, compound_ref)
        af.index = animals["USUBJID"].to_numpy()
        a_idx = animals.set_index("USUBJID")
        rows = a_idx.loc[obs["USUBJID"]].reset_index()
        tf = self.time_features(obs["DAY"].to_numpy(), rows)
        out = pd.concat([af.loc[obs["USUBJID"]].reset_index(drop=True), tf.reset_index(drop=True)], axis=1)
        out.insert(0, "DAY", obs["DAY"].to_numpy())
        out.insert(0, "USUBJID", obs["USUBJID"].to_numpy())
        out.insert(0, "STUDYID", obs["STUDYID"].to_numpy())
        return out

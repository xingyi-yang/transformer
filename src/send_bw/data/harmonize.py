"""Source-independent harmonizer: raw SEND-like domains + curated metadata -> canonical tables.

Rules (all explicit, all configurable through the mapping profile):
- Elapsed DAY = calendar difference from the reference date (default DM.RFSTDTC).
  The SEND study day (no day 0) is preserved in SOURCE_DY; elapsed = DY - 1 for DY >= 1, DY for DY < 0.
- Partial dates: if the rule is study_day_else_quarantine, use SOURCE_DY and flag DATE_UNCERTAIN;
  otherwise quarantine. Never silently complete dates.
- Units are converted only through the profile's unit table; unknown units are quarantined.
- Baseline = last eligible non-fasted measurement in [max_day - window_days, max_day]; ties -> highest BWSEQ.
  Post-treatment measurements are never substituted.
- Planned regimens come only from curated metadata keyed by (STUDYID, ARMCD). EX is kept as
  actual administrations for QC; it is never used as planned information.
- Weights after an animal's disposition day are quarantined, never kept as outcomes.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.metadata import MappingProfile
from send_bw.data.schema import CanonicalTables, coerce_table, safe_merge, validate_keys
from send_bw.data.sources import SendSource

_FULL = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_PARTIAL = re.compile(r"^(\d{4})(-(\d{2}))?$")


def _clean(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s or None


def parse_date(v: Any) -> tuple[dt.date | None, str]:
    s = _clean(v)
    if s is None:
        return None, "missing"
    m = _FULL.match(s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "full"
        except ValueError:
            return None, "invalid"
    if _PARTIAL.match(s):
        return None, "partial"
    return None, "invalid"


def dy_to_elapsed(dy: float | None, no_day_zero: bool = True) -> float | None:
    if dy is None or (isinstance(dy, float) and np.isnan(dy)):
        return None
    dy = float(dy)
    if no_day_zero:
        if dy == 0:
            raise ValueError("study day 0 does not exist under the no-day-zero convention")
        return dy - 1.0 if dy >= 1 else dy
    return dy


def _num(v: Any) -> float | None:
    s = _clean(v)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


class Harmonizer:
    def __init__(self, mapping: MappingProfile, source: SendSource):
        self.mapping = mapping
        self.source = source
        self.quarantine: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def _q(self, table: str, reason: str, severity: str, row: dict | None = None, detail: str = "") -> None:
        row = row or {}
        self.quarantine.append({"TABLE": table, "STUDYID": _clean(row.get("STUDYID")),
                                "USUBJID": _clean(row.get("USUBJID")), "REASON": reason, "SEVERITY": severity,
                                "SRC_FILE": _clean(row.get("_SRC_FILE")), "SRC_ROW": _num(row.get("_SRC_ROW")),
                                "DETAIL": detail})

    def _domain(self, name: str) -> pd.DataFrame:
        df = self.source.read_domain(name)
        ren = {k.upper(): v.upper() for k, v in self.mapping.column_renames.get(name, {}).items()}
        if ren:
            df = df.rename(columns=ren)
        return df

    # ------------------------------------------------------------------ main entry
    def run(self) -> tuple[CanonicalTables, dict[str, Any]]:
        meta = self.source.read_metadata()
        dm = self._domain("DM")
        bw = self._domain("BW")
        ex = self._domain("EX")
        ds = self._domain("DS")
        supp = self._domain("SUPPBW")
        if dm.empty or bw.empty:
            raise ValueError("DM and BW domains are required")

        dm = self._dedupe_exact(dm, "DM", key=["STUDYID", "USUBJID"])
        study_ctx = self._study_context(dm, meta)
        regimen = self._arm_regimen(dm, meta)
        animals = self._animals(dm, regimen)
        status = self._status(ds, animals)
        obs = self._bw(bw, supp, animals, status)
        animals, obs = self._baseline(animals, obs)
        obs = self._phase(obs, animals, regimen)
        admin = self._administrations(ex, animals, regimen)

        compound_ref = meta.get("compound_reference")
        if compound_ref is None:
            compound_ref = pd.DataFrame(columns=["COMPOUND_ID"])
            self.warnings.append({"type": "missing_compound_reference"})
        quarantine = pd.DataFrame(self.quarantine) if self.quarantine else pd.DataFrame(
            columns=["TABLE", "STUDYID", "USUBJID", "REASON", "SEVERITY", "SRC_FILE", "SRC_ROW", "DETAIL"])
        tables = CanonicalTables({
            "study_context": coerce_table("study_context", study_ctx),
            "animal_context": coerce_table("animal_context", animals),
            "arm_regimen": coerce_table("arm_regimen", regimen),
            "administrations": coerce_table("administrations", admin),
            "bw_obs": coerce_table("bw_obs", obs),
            "animal_status": coerce_table("animal_status", status),
            "quarantine": coerce_table("quarantine", quarantine),
            "compound_reference": compound_ref.copy(),
        })
        validate_keys(tables["animal_context"], ["USUBJID"], "animal_context")
        validate_keys(tables["arm_regimen"], ["ARM_KEY"], "arm_regimen")
        report = {"mapping_version": self.mapping.mapping_version, "send_version": self.mapping.send_version,
                  "warnings": self.warnings,
                  "quarantine_counts": quarantine.groupby(["REASON", "SEVERITY"]).size().rename("n")
                  .reset_index().to_dict("records") if len(quarantine) else []}
        return tables, report

    # ------------------------------------------------------------------ helpers
    def _dedupe_exact(self, df: pd.DataFrame, table: str, key: list[str]) -> pd.DataFrame:
        content = [c for c in df.columns if not c.startswith("_")]
        exact = df.duplicated(content, keep="first")
        for r in df[exact].to_dict("records"):
            self._q(table, "duplicate_record_dropped", "soft", r, "exact duplicate of an earlier record")
        df = df[~exact]
        conflict = df.duplicated(key, keep=False)
        for r in df[conflict].to_dict("records"):
            self._q(table, "conflicting_duplicate_key", "hard", r, f"duplicate {key} with differing content")
        return df[~conflict].copy()

    def _study_context(self, dm: pd.DataFrame, meta: dict) -> pd.DataFrame:
        studies = pd.DataFrame({"STUDYID": sorted(dm["STUDYID"].astype(str).unique())})
        sm = meta.get("study_metadata")
        if sm is not None and len(sm):
            sm = sm.copy()
            sm["STUDYID"] = sm["STUDYID"].astype(str)
            validate_keys(sm, ["STUDYID"], "study_metadata")
            keep = [c for c in ["STUDYID", "SPECIES", "STRAIN", "STUDY_CLASS", "STUDY_START", "SEND_VERSION",
                                "DURATION_DAYS"] if c in sm.columns]
            studies = safe_merge(studies, sm[keep], on=["STUDYID"], name="study_metadata")
        for c in ["SPECIES", "STRAIN", "STUDY_CLASS", "SEND_VERSION", "STUDY_START"]:
            if c not in studies:
                studies[c] = None
        dm_species = dm.groupby("STUDYID")["SPECIES"].agg(lambda s: sorted({str(x).upper() for x in s.dropna()}))
        rows = []
        for r in studies.to_dict("records"):
            sid = r["STUDYID"]
            sp_dm = dm_species.get(sid, [])
            sp = _clean(r.get("SPECIES"))
            if sp is None and len(sp_dm) == 1:
                sp = sp_dm[0]
                self.warnings.append({"type": "species_from_dm", "study": sid})
            elif sp is not None and sp_dm and sp.upper() not in sp_dm:
                self._q("study_context", "species_conflict_metadata_vs_dm", "hard", {"STUDYID": sid},
                        f"curated={sp} dm={sp_dm}")
                sp = "CONFLICT"
            sp = self.mapping.species_values.get(sp, sp) if sp else "UNKNOWN"
            r["SPECIES"] = sp.upper()
            r["STUDY_CLASS"] = _clean(r.get("STUDY_CLASS")) or "UNKNOWN"
            r["SEND_VERSION"] = _clean(r.get("SEND_VERSION")) or self.mapping.send_version
            r["MAPPING_VERSION"] = self.mapping.for_study(sid).mapping_version
            r["SOURCE_ID"] = self.source.source_id
            rows.append(r)
        return pd.DataFrame(rows)

    def _arm_regimen(self, dm: pd.DataFrame, meta: dict) -> pd.DataFrame:
        arms = dm[["STUDYID", "ARMCD"]].astype(str).drop_duplicates().reset_index(drop=True)
        ar = meta.get("arm_regimen")
        cols = ["COMPOUND_ID", "IS_CONTROL", "PLANNED_DOSE", "DOSE_UNIT", "ROUTE", "INTERVAL_DAYS",
                "PLANNED_START_DY", "PLANNED_END_DY", "COHORT"]
        if ar is None or not len(ar):
            self.warnings.append({"type": "missing_planned_regimen_metadata",
                                  "detail": "all arms flagged REGIMEN_AVAILABLE=False"})
            ar = pd.DataFrame(columns=["STUDYID", "ARMCD"] + cols)
        ar = ar.copy()
        ar["STUDYID"] = ar["STUDYID"].astype(str)
        ar["ARMCD"] = ar["ARMCD"].astype(str)
        validate_keys(ar, ["STUDYID", "ARMCD"], "arm_regimen (curated)")
        out = safe_merge(arms, ar[["STUDYID", "ARMCD"] + [c for c in cols if c in ar.columns]],
                         on=["STUDYID", "ARMCD"], name="arm_regimen")
        rows = []
        for r in out.to_dict("records"):
            sid = r["STUDYID"]
            nd = self.mapping.for_study(sid).timing.no_day_zero
            avail = _clean(r.get("IS_CONTROL")) is not None
            is_ctrl = str(_clean(r.get("IS_CONTROL"))).upper() in {"TRUE", "1", "Y", "YES"}
            rec = {"STUDYID": sid, "ARMCD": r["ARMCD"], "ARM_KEY": f"{sid}::{r['ARMCD']}",
                   "COMPOUND_ID": None if is_ctrl else _clean(r.get("COMPOUND_ID")), "IS_CONTROL": is_ctrl,
                   "PLANNED_DOSE": _num(r.get("PLANNED_DOSE")), "DOSE_UNIT": _clean(r.get("DOSE_UNIT")),
                   "ROUTE": (_clean(r.get("ROUTE")) or "").upper() or None,
                   "INTERVAL_DAYS": _num(r.get("INTERVAL_DAYS")),
                   "START_DAY": None, "END_DAY": None,
                   "COHORT": (_clean(r.get("COHORT")) or "UNKNOWN").upper()}
            try:
                rec["START_DAY"] = dy_to_elapsed(_num(r.get("PLANNED_START_DY")), nd)
                rec["END_DAY"] = dy_to_elapsed(_num(r.get("PLANNED_END_DY")), nd)
            except ValueError as e:
                self._q("arm_regimen", "invalid_planned_study_day", "hard", rec, str(e))
                avail = False
            needed = ["PLANNED_DOSE", "INTERVAL_DAYS", "START_DAY", "END_DAY", "ROUTE"]
            if not is_ctrl:
                needed.append("COMPOUND_ID")
            missing = [k for k in needed if rec.get(k) is None]
            if not avail or missing:
                self.warnings.append({"type": "planned_regimen_unavailable", "arm": rec["ARM_KEY"],
                                      "missing": missing if avail else ["curated_row"]})
            rec["REGIMEN_AVAILABLE"] = bool(avail and not missing)
            rows.append(rec)
        return pd.DataFrame(rows)

    def _animals(self, dm: pd.DataFrame, regimen: pd.DataFrame) -> pd.DataFrame:
        rows = []
        reg = regimen.set_index("ARM_KEY")
        for r in dm.to_dict("records"):
            sid = str(r["STUDYID"])
            m = self.mapping.for_study(sid)
            sex_raw = (_clean(r.get("SEX")) or "").upper()
            sex = m.sex_values.get(sex_raw)
            ref_col = m.timing.reference.upper()
            ref, kind = parse_date(r.get(ref_col))
            key = f"{sid}::{_clean(r.get('ARMCD'))}"
            reasons = []
            if sex is None:
                reasons.append("sex_not_mapped")  # never inferred from weight
            if ref is None:
                reasons.append(f"reference_date_{kind}")
            cohort = reg.loc[key, "COHORT"] if key in reg.index else "UNKNOWN"
            sp = (_clean(r.get("SPECIES")) or "UNKNOWN").upper()
            rows.append({"STUDYID": sid, "USUBJID": str(r["USUBJID"]), "SUBJID": _clean(r.get("SUBJID")),
                         "SEX": sex or "U", "SPECIES": m.species_values.get(sp, sp),
                         "STRAIN": (_clean(r.get("STRAIN")) or "").upper() or None,
                         "ARMCD": str(_clean(r.get("ARMCD"))), "ARM": _clean(r.get("ARM")), "ARM_KEY": key,
                         "COHORT": cohort, "REF_DATE": ref.isoformat() if ref else None,
                         "ELIGIBLE": not reasons, "EXCLUSION_REASON": ";".join(reasons) or None})
        return pd.DataFrame(rows)

    def _status(self, ds: pd.DataFrame, animals: pd.DataFrame) -> pd.DataFrame:
        ref = animals.set_index("USUBJID")["REF_DATE"].to_dict()
        rows = {}
        if not ds.empty:
            for r in ds.to_dict("records"):
                u = str(r["USUBJID"])
                term = (_clean(r.get("DSDECOD")) or _clean(r.get("DSTERM")) or "").upper()
                m = self.mapping.for_study(str(r["STUDYID"]))
                disp = {k.upper(): v for k, v in m.disposition_terms.items()}.get(term)
                if disp is None:
                    self.warnings.append({"type": "unmapped_disposition_term", "term": term})
                    disp = "ONGOING_OR_UNKNOWN"
                d, kind = parse_date(r.get("DSSTDTC"))
                day = None
                if d is not None and ref.get(u):
                    day = float((d - dt.date.fromisoformat(ref[u])).days)
                elif _num(r.get("DSSTDY")) is not None:
                    day = dy_to_elapsed(_num(r.get("DSSTDY")), m.timing.no_day_zero)
                if u in rows and rows[u]["DISPOSITION"] != "ONGOING_OR_UNKNOWN":
                    self._q("animal_status", "multiple_disposition_records", "soft", r, "kept first mapped")
                    continue
                rows[u] = {"STUDYID": str(r["STUDYID"]), "USUBJID": u, "DISPOSITION": disp,
                           "DISPOSITION_DAY": day, "SOURCE_TERM": term}
        for a in animals.to_dict("records"):
            rows.setdefault(a["USUBJID"], {"STUDYID": a["STUDYID"], "USUBJID": a["USUBJID"],
                                           "DISPOSITION": "ONGOING_OR_UNKNOWN", "DISPOSITION_DAY": None,
                                           "SOURCE_TERM": None})
        return pd.DataFrame(list(rows.values()))

    def _bw(self, bw: pd.DataFrame, supp: pd.DataFrame, animals: pd.DataFrame,
            status: pd.DataFrame) -> pd.DataFrame:
        ref = animals.set_index("USUBJID")["REF_DATE"].to_dict()
        known = set(animals["USUBJID"])
        disp_day = status.set_index("USUBJID")["DISPOSITION_DAY"].to_dict()
        supp_map: dict[tuple[str, str], dict[str, bool]] = {}
        if not supp.empty:
            for r in supp.to_dict("records"):
                if (_clean(r.get("RDOMAIN")) or "BW").upper() != "BW":
                    continue
                m = self.mapping.for_study(str(r["STUDYID"]))
                qmap = {k.upper(): v for k, v in m.supplemental.get("SUPPBW", {}).items()}
                qnam = (_clean(r.get("QNAM")) or "").upper()
                if qnam not in qmap:
                    self.warnings.append({"type": "unmapped_supplemental_qualifier", "qnam": qnam})
                    continue
                if (_clean(r.get("IDVAR")) or "").upper() != "BWSEQ":
                    self._q("SUPPBW", "unsupported_idvar", "soft", r, str(r.get("IDVAR")))
                    continue
                key = (str(r["USUBJID"]), str(int(float(r["IDVARVAL"]))))
                val = (_clean(r.get("QVAL")) or "").upper() in {v.upper() for v in m.supplemental_true_values}
                supp_map.setdefault(key, {})[qmap[qnam]] = val
        rows = []
        for r in bw.to_dict("records"):
            sid = str(r["STUDYID"])
            m = self.mapping.for_study(sid)
            u = str(r.get("USUBJID"))
            tcd = (_clean(r.get("BWTESTCD")) or "").upper()
            if tcd not in {t.upper() for t in m.body_weight.testcd}:
                self._q("BW", "test_code_not_mapped", "soft", r, tcd)
                continue
            if u not in known:
                self._q("BW", "subject_not_in_dm", "hard", r)
                continue
            flags = []
            value = factor = None
            orig_v, orig_u = _clean(r.get("BWORRES")), _clean(r.get("BWORRESU"))
            std_v, std_u = _num(r.get("BWSTRESN")), _clean(r.get("BWSTRESU"))
            for pr in m.body_weight.value_priority:
                if pr == "standardized" and std_v is not None:
                    value, factor = std_v, m.body_weight.unit_factor(std_u)
                    if factor is None:
                        value = None
                        flags.append(f"std_unit_unmapped:{std_u}")
                        continue
                    break
                if pr == "original" and _num(orig_v) is not None:
                    value, factor = _num(orig_v), m.body_weight.unit_factor(orig_u)
                    if factor is None:
                        value = None
                        flags.append(f"orig_unit_unmapped:{orig_u}")
                        continue
                    flags.append("from_original_result")
                    break
            if value is None:
                self._q("BW", "no_convertible_value", "hard", r, ";".join(flags) or "missing value")
                continue
            w = value * factor
            if not np.isfinite(w) or w <= 0:
                self._q("BW", "nonpositive_weight", "hard", r, str(value))
                continue
            d, kind = parse_date(r.get("BWDTC"))
            dy = _num(r.get("BWDY"))
            try:
                dy_el = dy_to_elapsed(dy, m.timing.no_day_zero)
            except ValueError as e:
                self._q("BW", "invalid_study_day_zero", "hard", r, str(e))
                continue
            uncertain = False
            if kind == "full" and ref.get(u):
                day = float((d - dt.date.fromisoformat(ref[u])).days)
                if dy_el is not None and dy_el != day:
                    flags.append("dy_date_mismatch")
            elif kind in {"partial", "missing"} and m.timing.partial_date_rule == "study_day_else_quarantine" \
                    and dy_el is not None:
                day = dy_el
                uncertain = kind == "partial"
                flags.append(f"{kind}_date_used_study_day")
            else:
                self._q("BW", f"date_{kind}_unresolvable", "hard", r)
                continue
            seq = _num(r.get("BWSEQ"))
            fast = supp_map.get((u, str(int(seq)) if seq is not None else ""), {}).get("FASTING", False)
            ddd = disp_day.get(u)
            if ddd is not None and not pd.isna(ddd) and day > ddd:
                self._q("BW", "post_disposition_weight", "hard", r, f"day {day} > disposition {ddd}")
                continue
            rows.append({"STUDYID": sid, "USUBJID": u, "BWSEQ": seq, "DAY": day, "SOURCE_DY": dy,
                         "WEIGHT_G": w, "LOGW": float(np.log(w)), "ORIG_VALUE": orig_v, "ORIG_UNIT": orig_u,
                         "STD_VALUE": std_v, "STD_UNIT": std_u, "DATE_UNCERTAIN": uncertain,
                         "FASTING": bool(fast), "QC_FLAGS": ";".join(flags) or None,
                         "SRC_FILE": r.get("_SRC_FILE"), "SRC_ROW": int(r.get("_SRC_ROW", -1)),
                         "MAPPING_VERSION": m.mapping_version})
        obs = pd.DataFrame(rows)
        if obs.empty:
            raise ValueError("no usable body-weight records after mapping")
        # duplicate handling (e.g. duplicate extracts): exact -> drop extras; conflicting -> quarantine all
        key = ["USUBJID", "DAY", "FASTING"]
        exact = obs.duplicated(key + ["WEIGHT_G"], keep="first")
        for r in obs[exact].to_dict("records"):
            self._q("BW", "duplicate_record_dropped", "soft", {**r, "_SRC_FILE": r["SRC_FILE"],
                                                               "_SRC_ROW": r["SRC_ROW"]})
        obs = obs[~exact]
        conflict = obs.duplicated(key, keep=False)
        for r in obs[conflict].to_dict("records"):
            self._q("BW", "conflicting_duplicate_weight", "hard", {**r, "_SRC_FILE": r["SRC_FILE"],
                                                                   "_SRC_ROW": r["SRC_ROW"]})
        return obs[~conflict].sort_values(["STUDYID", "USUBJID", "DAY"]).reset_index(drop=True)

    def _baseline(self, animals: pd.DataFrame, obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        obs = obs.copy()
        obs["IS_BASELINE"] = False
        base = {}
        for sid, g in obs.groupby("STUDYID"):
            rule = self.mapping.for_study(str(sid)).timing.baseline
            elig = g[(~g["FASTING"]) & (g["DAY"] <= rule.max_day) & (g["DAY"] >= rule.max_day - rule.window_days)]
            elig = elig.assign(_SEQ=elig["BWSEQ"].fillna(-1)).sort_values(["USUBJID", "DAY", "_SEQ"])
            last = elig.groupby("USUBJID").tail(1)
            obs.loc[last.index, "IS_BASELINE"] = True
            for r in last.to_dict("records"):
                base[r["USUBJID"]] = (r["DAY"], r["WEIGHT_G"], r["LOGW"])
        animals = animals.copy()
        animals["BASELINE_DAY"] = animals["USUBJID"].map(lambda u: base.get(u, (np.nan,) * 3)[0])
        animals["BASELINE_G"] = animals["USUBJID"].map(lambda u: base.get(u, (np.nan,) * 3)[1])
        animals["BASELINE_LOGW"] = animals["USUBJID"].map(lambda u: base.get(u, (np.nan,) * 3)[2])
        nob = animals["BASELINE_LOGW"].isna()
        animals.loc[nob, "ELIGIBLE"] = False
        animals.loc[nob, "EXCLUSION_REASON"] = animals.loc[nob, "EXCLUSION_REASON"].fillna("").map(
            lambda s: ";".join([x for x in [s, "no_eligible_baseline"] if x]))
        return animals, obs

    def _phase(self, obs: pd.DataFrame, animals: pd.DataFrame, regimen: pd.DataFrame) -> pd.DataFrame:
        a = animals.set_index("USUBJID")
        end = regimen.set_index("ARM_KEY")["END_DAY"]
        bday = obs["USUBJID"].map(a["BASELINE_DAY"])
        eday = obs["USUBJID"].map(a["ARM_KEY"]).map(end)
        phase = np.where(obs["IS_BASELINE"] | (obs["DAY"] <= bday.fillna(0)) | (obs["DAY"] < 0), "PRETREATMENT",
                         np.where(eday.notna() & (obs["DAY"] > eday + 1.0), "RECOVERY", "TREATMENT"))
        obs = obs.copy()
        obs["PHASE"] = phase
        return obs

    def _administrations(self, ex: pd.DataFrame, animals: pd.DataFrame, regimen: pd.DataFrame) -> pd.DataFrame:
        if ex.empty:
            return pd.DataFrame(columns=["STUDYID", "USUBJID", "SRC_FILE", "SRC_ROW"])
        ref = animals.set_index("USUBJID")["REF_DATE"].to_dict()
        arm = animals.set_index("USUBJID")["ARM_KEY"].to_dict()
        reg = regimen.set_index("ARM_KEY")
        rows = []
        for r in ex.to_dict("records"):
            u = str(r["USUBJID"])
            if u not in ref:
                self._q("EX", "subject_not_in_dm", "hard", r)
                continue
            days = []
            for c in ("EXSTDTC", "EXENDTC"):
                d, _ = parse_date(r.get(c))
                days.append(float((d - dt.date.fromisoformat(ref[u])).days) if d is not None and ref[u] else None)
            rec = {"STUDYID": str(r["STUDYID"]), "USUBJID": u, "EXSEQ": _num(r.get("EXSEQ")),
                   "EXTRT": _clean(r.get("EXTRT")), "EXDOSE": _num(r.get("EXDOSE")),
                   "EXDOSU": _clean(r.get("EXDOSU")), "EXROUTE": _clean(r.get("EXROUTE")),
                   "START_DAY": days[0], "END_DAY": days[1], "SRC_FILE": r.get("_SRC_FILE"),
                   "SRC_ROW": int(r.get("_SRC_ROW", -1))}
            k = arm.get(u)
            if k in reg.index and reg.loc[k, "REGIMEN_AVAILABLE"]:
                pd_ = reg.loc[k, "PLANNED_DOSE"]
                if rec["EXDOSE"] is not None and pd_ is not None and abs(rec["EXDOSE"] - pd_) > 1e-9:
                    rec["DEVIATION"] = "dose_differs_from_plan"
                elif rec["END_DAY"] is not None and reg.loc[k, "END_DAY"] is not None and \
                        rec["END_DAY"] < reg.loc[k, "END_DAY"]:
                    rec["DEVIATION"] = "ended_before_plan"
            rows.append(rec)
        return pd.DataFrame(rows)


def harmonize(source: SendSource, mapping: MappingProfile) -> tuple[CanonicalTables, dict[str, Any]]:
    return Harmonizer(mapping, source).run()

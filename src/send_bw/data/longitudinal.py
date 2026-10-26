"""Versioned rich-history synthetic studies.

Synthetic methodological benchmark. Not real-data validation or operational
qualification. Official generate_dataset defaults do not use this module.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.generate import (
    ARMCD_STYLES,
    ROUTES,
    SCHEDULE_ASSUMPTIONS,
    SCHEDULE_KINDS,
    SLOPE_BASES,
    STRAINS,
    GeneratorA,
    Regimen,
    _bw_record,
    _dy,
    _iso,
    _write_study,
    ou_cov,
    planned_visit_days,
    recovery_visit_days,
    slope_tau,
)
from send_bw.utils.artifacts import mark_generated, write_json, write_parquet
from send_bw.utils.seeds import SeedStream

LABEL = (
    "Synthetic methodological benchmark. Not real-data validation or operational qualification."
)


def _as_float_list(xs: Any) -> list[float]:
    return [float(x) for x in xs]


def _scenario_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    s = dict(raw)
    s.setdefault("species", "RAT")
    s.setdefault("duration", 28)
    s.setdefault("schedule", "weekly")
    s.setdefault("treatment_days", s["duration"])
    s.setdefault("recovery_days", 14)
    s.setdefault("recovery_prob", 0.5)
    s.setdefault("slope_basis", "linear_tref" if float(s["duration"]) <= 28 else "saturating")
    s.setdefault("dropout_kind", "ordinary")
    s.setdefault("p_missing", None)
    s.setdefault("p_dropout", None)
    s.setdefault("p_dose_deviation", None)
    s.setdefault("paired", False)
    s.setdefault("n_studies", 0)
    s.setdefault("n_pairs", 0)
    s.setdefault("dense_schedule", "dense")
    s.setdefault("sparse_schedule", "weekly")
    s.setdefault("pretreatment_days", [-7.0, 0.0])
    if s["slope_basis"] not in SLOPE_BASES:
        raise ValueError(f"unknown slope_basis {s['slope_basis']!r}")
    if s["dropout_kind"] not in {"ordinary", "challenging"}:
        raise ValueError(f"unknown dropout_kind {s['dropout_kind']!r}")
    return s


def _draw_death(rng: np.random.Generator, T: int, kind: str, p_ordinary: float,
                p_challenging: float) -> float | None:
    """Ordinary vs challenging dropout. Independent of weight and treatment.

    Ordinary: iid Bernoulli then uniform day in [2, T). Same spirit as official p_death.
    Challenging: higher rate, later-biased day in [0.4 T, T). Still not MNAR on Y.
    """
    if kind == "challenging":
        if rng.random() >= float(p_challenging):
            return None
        lo = max(2, int(0.4 * T))
        hi = max(lo + 1, int(T))
        return float(rng.integers(lo, hi))
    if rng.random() >= float(p_ordinary):
        return None
    return float(rng.integers(2, max(3, int(T))))


def _draw_study_animals(
    *,
    gen: GeneratorA,
    rng: np.random.Generator,
    species: str,
    strain: str,
    studyid: str,
    T: float,
    treatment_days: float,
    interval: float,
    route: str,
    cid: str,
    arms: list[dict[str, Any]],
    n_per_sex: list[int],
    n_ctrl_per_sex: int,
    grid: np.ndarray,
    slope_basis: str,
    p_missing: float,
    p_dose_deviation: float,
    dropout_kind: str,
    p_death: float,
    p_dropout_challenging: float,
    start_date: dt.date,
    fasted_terminal: bool,
    sched_by_cohort: dict[str, list[float]],
    last_by_cohort: dict[str, float],
    donor_dev: dict[str, float] | None,
    comp_truth: dict[str, dict],
) -> dict[str, Any]:
    """Draw one study's animals on a shared grid. Observation masks applied later."""
    u_s_base = float(rng.normal(0, gen.p["study_baseline_sd"]))
    u_slope = float(rng.normal(0, gen.p["sd_study_slope"]))
    tau = slope_tau(grid, slope_basis, gen.p["tau_g"])
    study_dev = u_slope * tau
    animals: list[dict[str, Any]] = []
    animal_counter = 0
    for arm in arms:
        n_sex = n_ctrl_per_sex if arm["IS_CONTROL"] else int(rng.integers(n_per_sex[0], n_per_sex[1] + 1))
        if arm["COHORT"] == "RECOVERY":
            n_sex = max(3, n_sex // 2)
        for sex in ("M", "F"):
            for _ in range(n_sex):
                animal_counter += 1
                subjid = f"{animal_counter:04d}"
                usubjid = f"{studyid}-{subjid}"
                mu0, sd0 = gen.p["baseline"][species][sex]
                if donor_dev:
                    mu0 = float(mu0) + float(donor_dev["baseline_shift"])
                z0 = float(rng.normal(mu0 + u_s_base, sd0))
                planned_end = float(treatment_days) - 1.0
                reg = Regimen(arm["IS_CONTROL"], arm["COMPOUND_ID"], arm["PLANNED_DOSE"], interval, 0.0,
                              planned_end, route)
                act = Regimen(**reg.__dict__)
                deviation = (not reg.is_control) and rng.random() < p_dose_deviation
                if deviation:
                    if rng.random() < 0.5:
                        act.end_day = float(rng.integers(5, max(6, int(treatment_days) - 3)))
                    else:
                        act.dose = reg.dose * 0.5
                ct = comp_truth.get(arm["COMPOUND_ID"]) if arm["COMPOUND_ID"] else None
                death_day = _draw_death(rng, int(T), dropout_kind, p_death, p_dropout_challenging)
                b = float(rng.normal(0, gen.p["sd_animal_slope"]))
                K = ou_cov(grid, gen.p["sigma_p"], gen.p["ell"]) + 1e-10 * np.eye(len(grid))
                ou = rng.standard_normal(len(grid)) @ np.linalg.cholesky(K).T
                scale = gen.variance_multiplier(act)
                meas = rng.normal(0, gen.p["sigma_m"], size=len(grid)) * scale
                growth = gen.growth(grid, species, sex, strain, z0)
                effect = gen.effect(grid, act, ct)
                if donor_dev:
                    growth = growth * float(donor_dev["gmax_scale"])
                    effect = effect * float(donor_dev["effect_scale"])
                z_lat = z0 + growth + effect + (b * tau) + ou + study_dev
                z_obs = z_lat + meas
                miss = {}
                for day in sched_by_cohort[arm["COHORT"]]:
                    if death_day is not None and day >= death_day:
                        miss[float(day)] = True
                    elif day > 0 and rng.random() < p_missing:
                        miss[float(day)] = True
                    else:
                        miss[float(day)] = False
                animals.append({
                    "SUBJID": subjid, "USUBJID": usubjid, "SEX": sex, "ARM": arm,
                    "Z0_TRUE": z0, "DOSE_DEVIATION": bool(deviation),
                    "ACTUAL_END_DAY": act.end_day, "ACTUAL_DOSE": act.dose,
                    "DEATH_DAY": death_day, "z_lat": z_lat, "z_obs": z_obs,
                    "miss": miss, "last": last_by_cohort[arm["COHORT"]],
                })
    return {
        "animals": animals,
        "u_s_base": u_s_base,
        "u_slope": u_slope,
        "n_animals": animal_counter,
    }


def _emit_frames(studyid: str, species: str, strain: str, start_date: dt.date, T: float,
                 route: str, cid: str, arms: list[dict[str, Any]], grid: np.ndarray,
                 drawn: dict[str, Any], sched_by_cohort: dict[str, list[float]],
                 fasted_terminal: bool, gen: GeneratorA, rng: np.random.Generator,
                 usubjid_fn=None) -> dict[str, Any]:
    """Write SEND frames from a drawn latent. Optional USUBJID rewrite for siblings."""
    dm, bw, ex, ds, supp = [], [], [], [], []
    truth_animals, latent_rows = [], []
    for a in drawn["animals"]:
        arm = a["ARM"]
        subjid = a["SUBJID"]
        usubjid = usubjid_fn(a) if usubjid_fn is not None else a["USUBJID"]
        death_day = a["DEATH_DAY"]
        last = a["last"]
        truth_animals.append({
            "STUDYID": studyid, "USUBJID": usubjid, "PAIR_USUBJID": a["USUBJID"],
            "SUBJID": subjid, "Z0_TRUE": a["Z0_TRUE"],
            "DOSE_DEVIATION": a["DOSE_DEVIATION"], "ACTUAL_END_DAY": a["ACTUAL_END_DAY"],
            "ACTUAL_DOSE": a["ACTUAL_DOSE"], "DEATH_DAY": death_day,
        })
        dm.append({"STUDYID": studyid, "DOMAIN": "DM", "USUBJID": usubjid, "SUBJID": subjid,
                   "SEX": a["SEX"], "SPECIES": species, "STRAIN": strain, "ARMCD": arm["ARMCD"],
                   "ARM": arm["ARM"], "RFSTDTC": _iso(start_date),
                   "RFENDTC": _iso(start_date + dt.timedelta(days=int(death_day or last)))})
        ex.append({"STUDYID": studyid, "DOMAIN": "EX", "USUBJID": usubjid, "EXSEQ": 1,
                   "EXTRT": arm["COMPOUND_ID"] or "VEHICLE", "EXDOSE": a["ACTUAL_DOSE"],
                   "EXDOSU": "mg/kg", "EXROUTE": route,
                   "EXSTDTC": _iso(start_date),
                   "EXENDTC": _iso(start_date + dt.timedelta(
                       days=int(min(a["ACTUAL_END_DAY"], (death_day or 1e9) - 1))))})
        seq = 0
        for day in sched_by_cohort[arm["COHORT"]]:
            if death_day is not None and day >= death_day:
                break
            if a["miss"].get(float(day), False):
                continue
            gi = int(np.where(np.isclose(grid, day))[0][0])
            seq += 1
            w = float(np.exp(a["z_obs"][gi]))
            latent_rows.append({"USUBJID": usubjid, "DAY": float(day), "Z_LATENT": float(a["z_lat"][gi])})
            bw.append(_bw_record(studyid, usubjid, seq, start_date, day, w))
        term_day = None if death_day is not None else last
        if term_day is not None and fasted_terminal:
            gi = int(np.where(np.isclose(grid, term_day))[0][0])
            seq += 1
            w = float(np.exp(a["z_obs"][gi] + gen.p["fasting_shift"]))
            bw.append(_bw_record(studyid, usubjid, seq, start_date, term_day, w))
            supp.append({"STUDYID": studyid, "RDOMAIN": "BW", "USUBJID": usubjid, "IDVAR": "BWSEQ",
                         "IDVARVAL": str(seq), "QNAM": "BWFAST", "QLABEL": "Fasting Status",
                         "QVAL": "Y"})
        if death_day is not None:
            term = str(rng.choice(["FOUND DEAD", "MORIBUND SACRIFICE"]))
            dday = death_day
        else:
            term = "RECOVERY SACRIFICE" if arm["COHORT"] == "RECOVERY" else "TERMINAL SACRIFICE"
            dday = term_day
        ds.append({"STUDYID": studyid, "DOMAIN": "DS", "USUBJID": usubjid, "DSSEQ": 1,
                   "DSTERM": term, "DSDECOD": term,
                   "DSSTDTC": _iso(start_date + dt.timedelta(days=int(dday))), "DSSTDY": _dy(dday)})
    ts = [{"STUDYID": studyid, "DOMAIN": "TS", "TSSEQ": i + 1, "TSPARMCD": k, "TSVAL": v}
          for i, (k, v) in enumerate([("SPECIES", species), ("STRAIN", strain), ("SSTDTC", _iso(start_date)),
                                      ("SNDIGVER", "SEND Implementation Guide Version 3.1"),
                                      ("DOSDUR", f"P{int(T)}D"), ("ROUTE", route)])]
    tx = [{"STUDYID": studyid, "DOMAIN": "TX", "SETCD": a["ARMCD"], "SET": a["ARM"], "TXSEQ": j + 1,
           "TXPARMCD": "TRTDOS", "TXVAL": str(a["PLANNED_DOSE"])} for j, a in enumerate(arms)]
    return {"dm": dm, "bw": bw, "ex": ex, "ds": ds, "ts": ts, "tx": tx, "suppbw": supp,
            "truth_animals": truth_animals, "latent_rows": latent_rows}


def _build_arms(studyid: str, cid: str, n_active: int, doses: np.ndarray, route: str,
                interval: float, treatment_days: float, codes: list[str],
                has_recovery: bool) -> list[dict[str, Any]]:
    n_arms = n_active + 1
    arms = []
    for a in range(n_arms):
        is_ctrl = a == 0
        for cohort in (["MAIN", "RECOVERY"] if has_recovery and a in (0, n_arms - 1) else ["MAIN"]):
            code = codes[a] + ("R" if cohort == "RECOVERY" else "")
            dose = 0.0 if is_ctrl else float(doses[a - 1])
            arm_label = ("Vehicle control" if is_ctrl else f"{cid} {dose:g} mg/kg") + (
                " recovery" if cohort == "RECOVERY" else "")
            arms.append({
                "STUDYID": studyid, "ARMCD": code, "ARM": arm_label,
                "COMPOUND_ID": None if is_ctrl else cid, "IS_CONTROL": is_ctrl,
                "PLANNED_DOSE": dose, "DOSE_UNIT": "mg/kg", "ROUTE": route, "INTERVAL_DAYS": interval,
                "PLANNED_START_DY": 1, "PLANNED_END_DY": int(treatment_days), "COHORT": cohort,
            })
    return arms


def _design_bits(rng: np.random.Generator, compound_ref: pd.DataFrame, n_per_sex: list[int],
                 style_index: int) -> dict[str, Any]:
    n_active = int(rng.integers(2, 5))
    cid = str(rng.choice(compound_ref["COMPOUND_ID"]))
    ladder = np.array([3, 10, 30, 100, 300, 1000], float)
    lo = int(rng.integers(0, len(ladder) - n_active + 1))
    return {
        "n_active": n_active,
        "cid": cid,
        "doses": ladder[lo:lo + n_active],
        "interval": float(rng.choice([1.0, 1.0, 1.0, 7.0])),
        "route": str(rng.choice(ROUTES, p=[0.8, 0.2])),
        "codes": ARMCD_STYLES[style_index % len(ARMCD_STYLES)](style_index, n_active + 1),
        "start_date": dt.date(2014, 1, 6) + dt.timedelta(days=int(rng.integers(0, 3650))),
        "fasted_terminal": bool(rng.random() < 0.5),
        "n_ctrl_per_sex": int(rng.integers(n_per_sex[0], n_per_sex[1] + 1)),
    }


def generate_longitudinal_dataset(cfg: dict[str, Any], out_dir: str | Path, seed: int) -> dict[str, Any]:
    """Independent studies with configurable duration, schedule, treatment, recovery."""
    out = Path(out_dir)
    mark_generated(out)
    send_dir, meta_dir, truth_dir = out / "send", out / "metadata", out / "truth"
    for d in (send_dir, meta_dir, truth_dir):
        d.mkdir(parents=True, exist_ok=True)
    gen = GeneratorA()
    ss = SeedStream(seed)
    long_cfg = dict(cfg["longitudinal"])
    scenarios = [_scenario_defaults(s) for s in (long_cfg.get("scenarios") or [])]
    if not scenarios:
        raise ValueError("longitudinal.scenarios is required")
    n_compounds = int(cfg.get("n_compounds", 8))
    n_per_sex = cfg.get("n_per_sex", [3, 4])
    p_missing_def = float(cfg.get("p_missing", 0.02))
    p_death_def = float(cfg.get("p_death", 0.015))
    p_drop_ch_def = float(long_cfg.get("p_dropout_challenging", 0.08))
    p_dose_def = float(cfg.get("p_dose_deviation", 0.04))
    study_prefix = str(long_cfg.get("study_prefix", "SYNRH"))

    crng = ss.rng("compounds")
    compounds, comp_truth = [], {}
    for c in range(n_compounds):
        cid = f"CMP{c + 1:02d}"
        desc = {"D1": float(crng.normal()), "D2": float(crng.normal()), "D3": float(crng.normal())}
        compounds.append({"COMPOUND_ID": cid, **{k: round(v, 4) for k, v in desc.items()},
                          "CLASS": f"CLASS_{'ABC'[c % 3]}"})
        comp_truth[cid] = gen.compound_truth(desc, crng)
    compound_ref = pd.DataFrame(compounds)

    study_rows, arm_rows, scenario_rows = [], [], []
    truth_animals, latent_rows, study_truth = [], [], []
    study_counter = 0
    pair_counter = 0

    def next_studyid() -> str:
        nonlocal study_counter
        study_counter += 1
        return f"{study_prefix}-{study_counter:03d}"

    def write_study_bundle(studyid: str, species: str, strain: str, T: float, treatment_days: float,
                           schedule: str, route: str, cid: str, start_date: dt.date,
                           arms: list[dict[str, Any]], frames: dict[str, Any],
                           drawn: dict[str, Any], slope_basis: str, style_index: int,
                           extra_meta: dict[str, Any]) -> None:
        for arm in arms:
            row = dict(arm)
            row["STUDYID"] = studyid
            arm_rows.append(row)
        study_rows.append({"STUDYID": studyid, "SPECIES": species, "STRAIN": strain,
                           "STUDY_CLASS": "REPEAT_DOSE_TOX", "STUDY_START": _iso(start_date),
                           "SEND_VERSION": "3.1", "DURATION_DAYS": T})
        scenario_rows.append({
            "STUDYID": studyid, "SPECIES": species, "STRAIN": strain, "DURATION_DAYS": T,
            "TREATMENT_DAYS": treatment_days, "SCHEDULE": schedule,
            "SLOPE_BASIS": slope_basis, "COMPOUND_ID": cid, "ROUTE": route,
            **extra_meta,
        })
        study_truth.append({"STUDYID": studyid, "U_STUDY_SLOPE": drawn["u_slope"],
                            "U_STUDY_BASELINE": drawn["u_s_base"], "COMPOUND_ID": cid,
                            "SLOPE_BASIS": slope_basis, **{k: extra_meta.get(k) for k in
                                                          ("PAIR_ID", "SCENARIO_ID", "DROPOUT_KIND")}})
        truth_animals.extend(frames["truth_animals"])
        latent_rows.extend(frames["latent_rows"])
        _write_study(send_dir / studyid, {k: frames[k] for k in
                                         ("dm", "bw", "ex", "ds", "ts", "tx", "suppbw")},
                     style_index, kg=False, lower=False, parquet=False, partial=0, rng=ss.rng("write", studyid))

    for s_idx, scn in enumerate(scenarios):
        species = str(scn["species"]).upper()
        T = float(scn["duration"])
        treatment_days = float(scn["treatment_days"])
        rec_days = float(scn["recovery_days"])
        slope_basis = str(scn["slope_basis"])
        pretreat = tuple(_as_float_list(scn["pretreatment_days"]))
        p_missing = float(scn["p_missing"] if scn["p_missing"] is not None else p_missing_def)
        p_dose = float(scn["p_dose_deviation"] if scn["p_dose_deviation"] is not None else p_dose_def)
        p_death = float(scn["p_dropout"] if scn["p_dropout"] is not None else p_death_def)
        dropout_kind = str(scn["dropout_kind"])
        rec_sched = recovery_visit_days(T, rec_days)
        is_donor = species != "RAT"

        if scn["paired"]:
            dense_kind, sparse_kind = str(scn["dense_schedule"]), str(scn["sparse_schedule"])
            if dense_kind not in SCHEDULE_KINDS or sparse_kind not in SCHEDULE_KINDS:
                raise ValueError("paired schedules must be weekly/twice_weekly/dense")
            n_pairs = int(scn["n_pairs"])
            for p in range(n_pairs):
                pair_counter += 1
                pair_id = f"{study_prefix}-P{pair_counter:02d}"
                rng = ss.rng("longitudinal", s_idx, "pair", p)
                strain = str(rng.choice(STRAINS[species]))
                bits = _design_bits(rng, compound_ref, n_per_sex, style_index=study_counter + p)
                dense_id, sparse_id = next_studyid(), next_studyid()
                has_recovery = bool(rng.random() < float(scn["recovery_prob"]))
                dense_days = planned_visit_days(T, dense_kind, pretreat)
                sparse_days = planned_visit_days(T, sparse_kind, pretreat)
                term_main, term_rec = T + 1.0, T + rec_days + 1.0
                grid = np.array(sorted(set(dense_days + sparse_days
                                           + (rec_sched if has_recovery else [])
                                           + [term_main, term_rec])), float)
                sched_dense = {"MAIN": dense_days,
                               "RECOVERY": dense_days + rec_sched if has_recovery else dense_days}
                sched_sparse = {"MAIN": sparse_days,
                                "RECOVERY": sparse_days + rec_sched if has_recovery else sparse_days}
                last_by = {"MAIN": term_main, "RECOVERY": term_rec}
                arms_template = _build_arms(pair_id, bits["cid"], bits["n_active"], bits["doses"],
                                            bits["route"], bits["interval"], treatment_days,
                                            bits["codes"], has_recovery)
                drawn = _draw_study_animals(
                    gen=gen, rng=rng, species=species, strain=strain, studyid=pair_id, T=T,
                    treatment_days=treatment_days, interval=bits["interval"], route=bits["route"],
                    cid=bits["cid"], arms=arms_template, n_per_sex=n_per_sex,
                    n_ctrl_per_sex=bits["n_ctrl_per_sex"], grid=grid, slope_basis=slope_basis,
                    p_missing=p_missing, p_dose_deviation=p_dose, dropout_kind=dropout_kind,
                    p_death=p_death, p_dropout_challenging=p_drop_ch_def,
                    start_date=bits["start_date"], fasted_terminal=bits["fasted_terminal"],
                    sched_by_cohort={"MAIN": sorted(set(dense_days + sparse_days)),
                                     "RECOVERY": sorted(set(dense_days + sparse_days + rec_sched))},
                    last_by_cohort=last_by, donor_dev=None, comp_truth=comp_truth,
                )
                for sid, kind, sched in ((dense_id, dense_kind, sched_dense),
                                         (sparse_id, sparse_kind, sched_sparse)):
                    arms = _build_arms(sid, bits["cid"], bits["n_active"], bits["doses"],
                                       bits["route"], bits["interval"], treatment_days,
                                       bits["codes"], has_recovery)
                    frames = _emit_frames(
                        sid, species, strain, bits["start_date"], T, bits["route"], bits["cid"],
                        arms, grid, drawn, sched, bits["fasted_terminal"], gen, rng,
                        usubjid_fn=lambda a, sid=sid: f"{sid}-{a['SUBJID']}")
                    write_study_bundle(
                        sid, species, strain, T, treatment_days, kind, bits["route"], bits["cid"],
                        bits["start_date"], arms, frames, drawn, slope_basis, study_counter,
                        extra_meta={
                            "SCENARIO_ID": scn.get("id", f"s{s_idx}"),
                            "PAIR_ID": pair_id, "PAIRED_ROLE": kind, "DROPOUT_KIND": dropout_kind,
                            "RECOVERY": has_recovery, "TREATMENT_DAYS": treatment_days,
                            "P_MISSING": p_missing, "DONOR": is_donor,
                            "LABEL": LABEL,
                        })
            continue

        n_st = int(scn["n_studies"])
        kind = str(scn["schedule"])
        if kind not in SCHEDULE_KINDS:
            raise ValueError(f"unknown schedule {kind!r}")
        for j in range(n_st):
            rng = ss.rng("longitudinal", s_idx, "study", j)
            strain = str(rng.choice(STRAINS[species]))
            bits = _design_bits(rng, compound_ref, n_per_sex, style_index=study_counter)
            sid = next_studyid()
            has_recovery = bool(rng.random() < float(scn["recovery_prob"]))
            days = planned_visit_days(T, kind, pretreat)
            term_main, term_rec = T + 1.0, T + rec_days + 1.0
            grid = np.array(sorted(set(days + (rec_sched if has_recovery else [])
                                       + [term_main, term_rec])), float)
            sched = {"MAIN": days, "RECOVERY": days + rec_sched if has_recovery else days}
            last_by = {"MAIN": term_main, "RECOVERY": term_rec}
            arms = _build_arms(sid, bits["cid"], bits["n_active"], bits["doses"],
                               bits["route"], bits["interval"], treatment_days,
                               bits["codes"], has_recovery)
            drawn = _draw_study_animals(
                gen=gen, rng=rng, species=species, strain=strain, studyid=sid, T=T,
                treatment_days=treatment_days, interval=bits["interval"], route=bits["route"],
                cid=bits["cid"], arms=arms, n_per_sex=n_per_sex,
                n_ctrl_per_sex=bits["n_ctrl_per_sex"], grid=grid, slope_basis=slope_basis,
                p_missing=p_missing, p_dose_deviation=p_dose, dropout_kind=dropout_kind,
                p_death=p_death, p_dropout_challenging=p_drop_ch_def,
                start_date=bits["start_date"], fasted_terminal=bits["fasted_terminal"],
                sched_by_cohort=sched, last_by_cohort=last_by, donor_dev=None,
                comp_truth=comp_truth,
            )
            frames = _emit_frames(sid, species, strain, bits["start_date"], T, bits["route"],
                                  bits["cid"], arms, grid, drawn, sched,
                                  bits["fasted_terminal"], gen, rng)
            write_study_bundle(
                sid, species, strain, T, treatment_days, kind, bits["route"], bits["cid"],
                bits["start_date"], arms, frames, drawn, slope_basis, study_counter,
                extra_meta={
                    "SCENARIO_ID": scn.get("id", f"s{s_idx}"),
                    "PAIR_ID": None, "PAIRED_ROLE": None, "DROPOUT_KIND": dropout_kind,
                    "RECOVERY": has_recovery, "TREATMENT_DAYS": treatment_days,
                    "P_MISSING": p_missing, "DONOR": is_donor, "LABEL": LABEL,
                })

    pd.DataFrame(study_rows).to_csv(meta_dir / "study_metadata.csv", index=False)
    pd.DataFrame(arm_rows).to_csv(meta_dir / "arm_regimen.csv", index=False)
    compound_ref.to_csv(meta_dir / "compound_reference.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(meta_dir / "study_scenarios.csv", index=False)
    write_json(truth_dir / "params.json", {
        "generator": gen.name, "params": gen.p, "compounds": comp_truth, "seed": seed,
        "longitudinal": True, "schedule_assumptions": SCHEDULE_ASSUMPTIONS,
        "label": LABEL, "warning": "TRUTH-ONLY. Not for model fitting.",
    })
    write_parquet(truth_dir / "animals.parquet", pd.DataFrame(truth_animals))
    write_parquet(truth_dir / "studies.parquet", pd.DataFrame(study_truth))
    write_parquet(truth_dir / "latent_paths.parquet", pd.DataFrame(latent_rows))
    return {
        "send_dir": str(send_dir), "metadata_dir": str(meta_dir), "truth_dir": str(truth_dir),
        "n_studies": study_counter, "generator": gen.name, "longitudinal": True,
        "n_pairs": pair_counter, "label": LABEL,
        "schedule_assumptions": SCHEDULE_ASSUMPTIONS,
    }

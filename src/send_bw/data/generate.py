"""Synthetic SEND-like data generators A and B (illustrative, NOT biologically validated).

Generator A (log weight Z = log W, t = elapsed days, tau = t / 28):
    Z(t) = z0 + growth_A(t, sex, strain, z0) + effect_A(t, regimen, compound)
           + u_s * tau + b_i * tau + OU(t) + m(t)
    growth_A = gmax * (1 - exp(-t / tau_g)), gmax shrinks for heavier animals
    effect_A = -A(c) * d_eff / (ED50(c) + d_eff) * H(t),  d_eff = dose * route_factor / sqrt(interval)
    H(t)     = onset 1 - exp(-(t - start) / tau_on) while dosing, exponential recovery
               exp(-(t - end) / tau_off) after the last planned (or actual) dose
    Cov(OU(t), OU(t')) = sigma_p^2 exp(-|t - t'| / ell);  m(t) ~ N(0, sigma_m^2) iid

Official bank (Generator A only): one saturating-exponential growth family with
shared tau_g=40. Generator B (Gompertz) exists for design-stress evaluation and
is not the official forecasting bank.

Observability:
    D1, D2, D3 are curated production descriptors written to
    metadata/compound_reference.csv. They are not SEND domains. A comparable
    table would have to be supplied at prediction time; missing D1–D3 on an
    active arm raises UnsupportedContextError. Hidden compound truth
    (A, ED50, tau_on, tau_off, and the link+noise that maps D1–D3 onto them)
    is written only to truth/params.json.

The trajectory uses the *actual* regimen after dose deviations (act.dose /
act.end_day). Production features use the planned regimen only.

Missing visits (p_missing if day>0) and death (p_death) are iid Bernoulli /
uniform-day draws. They do not depend on treatment, dose, or weight (MCAR-ish,
not informative dropout).

Truth (parameters and latent paths) is written to a separate directory that no
fitting or feature code reads.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.sharing import resolve_donor_scenario
from send_bw.utils.artifacts import mark_generated, write_json, write_parquet
from send_bw.utils.seeds import SeedStream

T_REF = 28.0
STRAINS = {"RAT": ["SPRAGUE-DAWLEY", "WISTAR HAN"], "DOG": ["BEAGLE"]}
ROUTES = ["ORAL GAVAGE", "INTRAVENOUS"]
SCHEDULE_KINDS = ("weekly", "twice_weekly", "dense")
SLOPE_BASES = ("linear_tref", "saturating")

# Experimental schedule labels. Not verified representations of real in-vivo protocols.
# No authorized historical schedule metadata were available.
SCHEDULE_ASSUMPTIONS = (
    "Synthetic methodological benchmark. Not real-data validation or operational qualification. "
    "Weekly / twice-weekly / dense calendars are experimental scenarios. "
    "No signed historical authorization; schedules are not empirical protocol estimates."
)


def slope_tau(times: np.ndarray, basis: str = "linear_tref", tau_g: float = 40.0) -> np.ndarray:
    """Time basis for study/animal random slopes.

    linear_tref: t / 28 (official short-study basis).
    saturating: (1-exp(-t/tau_g)) / (1-exp(-28/tau_g)) so tau(28)=1 and long
    horizons do not grow linearly past the official 28-day scale.
    """
    t = np.asarray(times, dtype=float)
    if basis in (None, "linear_tref"):
        return t / T_REF
    if basis == "saturating":
        denom = 1.0 - np.exp(-T_REF / float(tau_g))
        return (1.0 - np.exp(-t / float(tau_g))) / denom
    raise ValueError(f"unknown slope_basis {basis!r}; expected one of {SLOPE_BASES}")


def planned_visit_days(duration: float, schedule: str,
                       pretreatment_days: tuple[float, ...] = (-7.0, 0.0),
                       include_terminal: bool = True) -> list[float]:
    """Configured measurement days. Dense is the weekly grid union every-2-day even days.

    Dense includes weekly days so paired sparse weekly observations are an exact
    subset of the dense calendar. Days are generated from the process, not interpolated.
    """
    T = float(duration)
    if schedule == "weekly":
        treat = list(np.arange(7.0, T + 1e-9, 7.0))
    elif schedule == "twice_weekly":
        treat = sorted(set(np.concatenate([
            np.arange(3.0, T + 1e-9, 7.0),
            np.arange(7.0, T + 1e-9, 7.0),
        ]).tolist()))
    elif schedule == "dense":
        even = list(np.arange(2.0, T + 1e-9, 2.0))
        weekly = list(np.arange(7.0, T + 1e-9, 7.0))
        treat = sorted(set(even + weekly))
    else:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULE_KINDS}")
    days = [float(x) for x in pretreatment_days] + [float(x) for x in treat]
    if include_terminal and T not in days:
        days.append(T)
    return sorted({float(x) for x in days})


def recovery_visit_days(duration: float, recovery_days: float, interval: float = 7.0) -> list[float]:
    """Post-duration recovery visits at interval spacing, e.g. T+7, T+14."""
    T = float(duration)
    rec = float(recovery_days)
    if rec <= 0:
        return []
    n = max(1, int(round(rec / float(interval))))
    return [T + float(interval) * (i + 1) for i in range(n)]


def ou_cov(times: np.ndarray, sigma: float, ell: float) -> np.ndarray:
    d = np.abs(times[:, None] - times[None, :])
    return sigma**2 * np.exp(-d / ell)


def exposure_kernel(t: np.ndarray, start: float, end: float, tau_on: float, tau_off: float,
                    lag: float = 0.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    s = start + lag
    on = np.clip(t - s, 0.0, None)
    h_on = 1.0 - np.exp(-on / tau_on)
    h_end = 1.0 - np.exp(-max(end + 1.0 - s, 0.0) / tau_on)
    after = t > end + 1.0
    h = np.where(after, h_end * np.exp(-(t - end - 1.0) / tau_off), h_on)
    return np.where(t < s, 0.0, h)


@dataclass
class Regimen:
    is_control: bool
    compound_id: str | None
    dose: float
    interval_days: float
    start_day: float
    end_day: float
    route: str

    @staticmethod
    def from_row(r: Any) -> Regimen:
        get = r.get if hasattr(r, "get") else lambda k, d=None: getattr(r, k, d)
        return Regimen(bool(get("IS_CONTROL")), get("COMPOUND_ID"), float(get("PLANNED_DOSE") or 0.0),
                       float(get("INTERVAL_DAYS") or 1.0), float(get("START_DAY") or 0.0),
                       float(get("END_DAY") or 0.0), str(get("ROUTE") or "ORAL GAVAGE"))


# --------------------------------------------------------------------------- generator A

class GeneratorA:
    name = "A"

    def __init__(self, params: dict[str, Any] | None = None):
        self.p = params or self.default_params()

    @staticmethod
    def default_params() -> dict[str, Any]:
        return {
            "baseline": {"RAT": {"M": [np.log(260.0), 0.07], "F": [np.log(195.0), 0.065]},
                         "DOG": {"M": [np.log(10000.0), 0.08], "F": [np.log(8500.0), 0.08]}},
            "study_baseline_sd": 0.04,
            "gmax": {"RAT": {"M": 0.95, "F": 0.52}, "DOG": {"M": 0.08, "F": 0.06}},
            "strain_factor": {"SPRAGUE-DAWLEY": 1.0, "WISTAR HAN": 0.88, "BEAGLE": 1.0},
            "tau_g": 40.0,
            "gmax_baseline_slope": 1.2,
            "sd_study_slope": 0.035,
            "sd_animal_slope": 0.045,
            "sigma_p": 0.022,
            "ell": 6.0,
            "sigma_m": 0.010,
            "route_factor": {"ORAL GAVAGE": 1.0, "INTRAVENOUS": 1.4},
            "compound_link": {"a0": -1.0, "a1": 0.9, "a2": -0.6, "amax": 0.30, "ed50_0": np.log(40.0),
                              "ed50_3": 0.5, "noise_a": 0.3, "noise_ed": 0.25},
            "tau_on": [4.0, 10.0],
            "tau_off": [6.0, 14.0],
            "fasting_shift": -0.045,
        }

    # compound truth from curated descriptors (+ idiosyncratic noise for pool compounds)
    def compound_truth(self, desc: dict[str, float], rng: np.random.Generator | None = None) -> dict[str, float]:
        L = self.p["compound_link"]
        na = rng.normal(0, L["noise_a"]) if rng is not None else 0.0
        ne = rng.normal(0, L["noise_ed"]) if rng is not None else 0.0
        lin = L["a0"] + L["a1"] * desc["D1"] + L["a2"] * desc["D2"] + na
        A = L["amax"] / (1.0 + np.exp(-lin))
        ed50 = float(np.exp(L["ed50_0"] + L["ed50_3"] * desc["D3"] + ne))
        if rng is not None:
            tau_on = float(rng.uniform(*self.p["tau_on"]))
            tau_off = float(rng.uniform(*self.p["tau_off"]))
        else:
            tau_on, tau_off = float(np.mean(self.p["tau_on"])), float(np.mean(self.p["tau_off"]))
        return {"A": float(A), "ED50": ed50, "tau_on": tau_on, "tau_off": tau_off}

    def growth(self, t: np.ndarray, species: str, sex: str, strain: str, z0: float) -> np.ndarray:
        mu0 = self.p["baseline"][species][sex][0]
        g = self.p["gmax"][species][sex] * self.p["strain_factor"].get(strain, 1.0)
        g = g * max(0.2, 1.0 - self.p["gmax_baseline_slope"] * (z0 - mu0))
        return g * (1.0 - np.exp(-np.asarray(t, float) / self.p["tau_g"]))

    def effect(self, t: np.ndarray, reg: Regimen, ct: dict[str, float] | None) -> np.ndarray:
        if reg.is_control or ct is None or reg.dose <= 0:
            return np.zeros_like(np.asarray(t, float))
        d_eff = reg.dose * self.p["route_factor"].get(reg.route, 1.0) / np.sqrt(reg.interval_days)
        emax = ct["A"] * d_eff / (ct["ED50"] + d_eff)
        return -emax * exposure_kernel(t, reg.start_day, reg.end_day, ct["tau_on"], ct["tau_off"])

    def mean_change(self, t, species, sex, strain, z0, reg, ct):
        return self.growth(t, species, sex, strain, z0) + self.effect(t, reg, ct)

    def variance_multiplier(self, reg: Regimen) -> float:
        return 1.0

    def study_term(self, rng: np.random.Generator, times: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        u = float(rng.normal(0, self.p["sd_study_slope"]))
        return u * times / T_REF, {"U_STUDY_SLOPE": u}

    def animal_deviations(self, rng: np.random.Generator, times: np.ndarray, n: int,
                          scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Animal-level latent deviations (no study term) and measurement errors, shape (n, len(times))."""
        tau = times / T_REF
        b = rng.normal(0, self.p["sd_animal_slope"], size=n)
        K = ou_cov(times, self.p["sigma_p"], self.p["ell"]) + 1e-10 * np.eye(len(times))
        ou = rng.standard_normal((n, len(times))) @ np.linalg.cholesky(K).T
        latent = b[:, None] * tau[None, :] + ou
        meas = rng.normal(0, self.p["sigma_m"], size=(n, len(times)))
        return latent, meas

    def sample_study(self, context: pd.DataFrame, regimens: pd.DataFrame, times: np.ndarray, n_draws: int,
                     seed: int, compound_reference: pd.DataFrame | None = None,
                     compound_truth: dict[str, dict] | None = None) -> dict[str, np.ndarray]:
        """Design-mode simulation conditional on observed baselines (self-consistent DGM).

        One new study effect per draw; animal effects per animal. Returns z (log weight) at
        `times` (> 0) and the conditional mean given baseline, both (n_draws, n_animals, n_times).
        """
        return _design_sample(self, context, regimens, times, n_draws, seed, compound_reference,
                              compound_truth)


# --------------------------------------------------------------------------- generator B

class GeneratorB(GeneratorA):
    name = "B"

    @staticmethod
    def default_params() -> dict[str, Any]:
        p = GeneratorA.default_params()
        p.update({
            "gompertz_b": 1.6, "gompertz_c": 0.045,
            "gmax": {"RAT": {"M": 0.85, "F": 0.48}, "DOG": {"M": 0.07, "F": 0.05}},
            "sd_study_slope": 0.045, "sd_study_quad": 0.03, "sd_animal_slope": 0.055,
            "sigma_p": 0.018, "ell": 3.5, "sigma_rw": 0.006, "sigma_m": 0.012, "t_df": 4.0,
            "hill": 1.8, "tolerance_frac": 0.4, "tolerance_tau": 12.0, "lag": [0.0, 4.0],
            "var_dose_slope": 0.35,
        })
        return p

    def compound_truth(self, desc, rng=None):
        ct = super().compound_truth(desc, rng)
        ct["lag"] = float(rng.uniform(*self.p["lag"])) if rng is not None else float(np.mean(self.p["lag"]))
        return ct

    def growth(self, t, species, sex, strain, z0):
        mu0 = self.p["baseline"][species][sex][0]
        g = self.p["gmax"][species][sex] * self.p["strain_factor"].get(strain, 1.0)
        g = g * max(0.2, 1.0 - 0.8 * (z0 - mu0))
        b, c = self.p["gompertz_b"], self.p["gompertz_c"]
        t = np.asarray(t, float)
        f = np.exp(-b * np.exp(-c * t)) - np.exp(-b)
        return g * f / (1.0 - np.exp(-b))

    def effect(self, t, reg, ct):
        if reg.is_control or ct is None or reg.dose <= 0:
            return np.zeros_like(np.asarray(t, float))
        h = self.p["hill"]
        d_eff = reg.dose * self.p["route_factor"].get(reg.route, 1.0) / np.sqrt(reg.interval_days)
        emax = ct["A"] * d_eff**h / (ct["ED50"] ** h + d_eff**h)
        k = exposure_kernel(t, reg.start_day, reg.end_day, ct["tau_on"], ct["tau_off"], lag=ct.get("lag", 0.0))
        tol = 1.0 - self.p["tolerance_frac"] * (1 - np.exp(-np.clip(np.asarray(t, float) - reg.start_day, 0, None)
                                                          / self.p["tolerance_tau"]))
        return -emax * k * tol

    def variance_multiplier(self, reg: Regimen) -> float:
        if reg.is_control or reg.dose <= 0:
            return 1.0
        return 1.0 + self.p["var_dose_slope"] * np.log1p(reg.dose / 30.0)

    def study_term(self, rng, times):
        tau = times / T_REF
        u1, u2 = float(rng.normal(0, self.p["sd_study_slope"])), float(rng.normal(0, self.p["sd_study_quad"]))
        return u1 * tau + u2 * tau**2, {"U_STUDY_SLOPE": u1, "U_STUDY_QUAD": u2}

    def animal_deviations(self, rng, times, n, scale=None):
        tau = times / T_REF
        b = rng.normal(0, self.p["sd_animal_slope"], size=n)
        K = ou_cov(times, self.p["sigma_p"], self.p["ell"]) + 1e-10 * np.eye(len(times))
        ou = rng.standard_normal((n, len(times))) @ np.linalg.cholesky(K).T
        steps = np.diff(np.concatenate([[times.min()], times]))
        rw = np.cumsum(rng.normal(0, 1, size=(n, len(times))) * np.sqrt(steps)[None, :], axis=1) * self.p["sigma_rw"]
        latent = b[:, None] * tau[None, :] + ou + rw
        df = self.p["t_df"]
        meas = rng.standard_t(df, size=(n, len(times))) * self.p["sigma_m"] / np.sqrt(df / (df - 2))
        if scale is not None:
            latent = latent * scale[:, None]
            meas = meas * scale[:, None]
        return latent, meas


GENERATORS = {"A": GeneratorA, "B": GeneratorB}


def _design_sample(gen: GeneratorA, context: pd.DataFrame, regimens: pd.DataFrame, times: np.ndarray,
                   n_draws: int, seed: int, compound_reference: pd.DataFrame | None,
                   compound_truth: dict[str, dict] | None = None) -> dict[str, np.ndarray]:
    times = np.asarray(times, float)
    grid = np.concatenate([[0.0], times])
    rng = np.random.default_rng(seed)
    regs = {r["ARM_KEY"]: Regimen.from_row(r) for r in regimens.to_dict("records")}
    cref = compound_reference.set_index("COMPOUND_ID") if compound_reference is not None else None
    cts: dict[str, dict | None] = {}
    for k, reg in regs.items():
        if reg.is_control or reg.compound_id is None:
            cts[k] = None
        elif compound_truth and reg.compound_id in compound_truth:
            cts[k] = compound_truth[reg.compound_id]
        else:
            row = cref.loc[reg.compound_id]
            cts[k] = gen.compound_truth({"D1": float(row["D1"]), "D2": float(row["D2"]), "D3": float(row["D3"])})
    n = len(context)
    species = str(context["SPECIES"].iloc[0]) if "SPECIES" in context else "RAT"
    means = np.zeros((n, len(times)))
    scale = np.ones(n)
    for i, r in enumerate(context.itertuples(index=False)):
        reg = regs[r.ARM_KEY]
        strain = getattr(r, "STRAIN", "SPRAGUE-DAWLEY")
        means[i] = gen.mean_change(times, species, r.SEX, strain, r.BASELINE_LOGW, reg, cts[r.ARM_KEY])
        scale[i] = gen.variance_multiplier(reg)
    zb = context["BASELINE_LOGW"].to_numpy(float)
    z = np.empty((n_draws, n, len(times)))
    for d in range(n_draws):
        st, _ = gen.study_term(rng, grid)
        lat, meas = gen.animal_deviations(rng, grid, n, scale=scale)
        lat = lat + st[None, :]
        dev = (lat[:, 1:] - lat[:, :1]) + (meas[:, 1:] - meas[:, :1])
        z[d] = zb[:, None] + means + dev
    return {"z": z, "mean": np.broadcast_to(zb[:, None] + means, z.shape).copy()}


# --------------------------------------------------------------------------- dataset writer

def _iso(d: dt.date) -> str:
    return d.isoformat()


def _dy(elapsed: float) -> int:
    e = int(round(elapsed))
    return e + 1 if e >= 0 else e


ARMCD_STYLES = [
    lambda k, n: [str(i + 1) for i in range(n)],
    lambda k, n: ["C"] + [f"D{i}" for i in range(1, n)],
    lambda k, n: [str(n - i) for i in range(n)],  # control has the highest number in this style
]


def generate_dataset(cfg: dict[str, Any], out_dir: str | Path, seed: int) -> dict[str, Any]:
    """Write SEND-like domains (per-study folders), curated metadata and hidden truth.

    Official defaults (no ``longitudinal`` block) stay on the original 28-day
    random weekly/twice-weekly path. A ``longitudinal`` block selects the
    versioned rich-history generator and does not rewrite official banks.
    """
    if cfg.get("longitudinal"):
        from send_bw.data.longitudinal import generate_longitudinal_dataset

        return generate_longitudinal_dataset(cfg, out_dir, seed)
    out = Path(out_dir)
    mark_generated(out)
    send_dir, meta_dir, truth_dir = out / "send", out / "metadata", out / "truth"
    for d in (send_dir, meta_dir, truth_dir):
        d.mkdir(parents=True, exist_ok=True)
    gen = GENERATORS[cfg.get("generator", "A")]()
    ss = SeedStream(seed)
    n_studies = int(cfg.get("n_studies", 16))
    n_compounds = int(cfg.get("n_compounds", 8))
    durations = cfg.get("durations", [28])
    n_per_sex = cfg.get("n_per_sex", [5, 8])
    p_missing = float(cfg.get("p_missing", 0.02))
    p_death = float(cfg.get("p_death", 0.015))
    p_dose_deviation = float(cfg.get("p_dose_deviation", 0.04))
    n_other_species = int(cfg.get("n_other_species_studies", 1))
    donor_scenario = str(cfg.get("donor_scenario", "matched"))
    donor_dev = resolve_donor_scenario(donor_scenario)
    kg_studies = int(cfg.get("n_kg_unit_studies", 1))
    lowercase_studies = int(cfg.get("n_lowercase_column_studies", 1))
    parquet_fraction = float(cfg.get("parquet_fraction", 0.25))
    n_partial_dates = int(cfg.get("n_partial_dates", 3))

    crng = ss.rng("compounds")
    compounds = []
    comp_truth = {}
    for c in range(n_compounds):
        cid = f"CMP{c + 1:02d}"
        desc = {"D1": float(crng.normal()), "D2": float(crng.normal()), "D3": float(crng.normal())}
        compounds.append({"COMPOUND_ID": cid, **{k: round(v, 4) for k, v in desc.items()},
                          "CLASS": f"CLASS_{'ABC'[c % 3]}"})
        comp_truth[cid] = gen.compound_truth(desc, crng)
    compound_ref = pd.DataFrame(compounds)

    study_rows, arm_rows, truth_animals, latent_rows, study_truth = [], [], [], [], []
    for idx in range(n_studies + n_other_species):
        is_donor = idx >= n_studies
        if is_donor:
            d = idx - n_studies
            rng = ss.rng("donor", d)
            species = "DOG"
            studyid = f"SYN{gen.name}-D{d + 1:03d}"
            style_index = d
            kg_flag = False
            lower_flag = False
            partial_n = 0
        else:
            rng = ss.rng("study", idx)
            species = "RAT"
            studyid = f"SYN{gen.name}-{idx + 1:03d}"
            style_index = idx
            kg_flag = idx < kg_studies
            lower_flag = kg_studies <= idx < kg_studies + lowercase_studies
            partial_n = n_partial_dates if idx == n_studies - 1 else 0
        strain = str(rng.choice(STRAINS[species]))
        T = float(rng.choice(durations))
        n_active = int(rng.integers(2, 5))
        n_arms = n_active + 1
        cid = str(rng.choice(compound_ref["COMPOUND_ID"]))
        ladder = np.array([3, 10, 30, 100, 300, 1000], float)
        lo = int(rng.integers(0, len(ladder) - n_active + 1))
        doses = ladder[lo:lo + n_active]
        interval = float(rng.choice([1.0, 1.0, 1.0, 7.0]))
        route = str(rng.choice(ROUTES, p=[0.8, 0.2]))
        twice_weekly = bool(rng.random() < 0.35)
        style = ARMCD_STYLES[style_index % len(ARMCD_STYLES)]
        codes = style(style_index, n_arms)
        has_recovery = bool(rng.random() < 0.5)
        start_date = dt.date(2014, 1, 6) + dt.timedelta(days=int(rng.integers(0, 3650)))
        fasted_terminal = bool(rng.random() < 0.5)
        n_ctrl_per_sex = int(rng.integers(n_per_sex[0], n_per_sex[1] + 1))
        study_rows.append({"STUDYID": studyid, "SPECIES": species, "STRAIN": strain,
                           "STUDY_CLASS": "REPEAT_DOSE_TOX", "STUDY_START": _iso(start_date),
                           "SEND_VERSION": "3.1", "DURATION_DAYS": T})
        sched = [-7.0, 0.0] + (sorted(set(list(np.arange(3.0, T + 1, 7.0)) + list(np.arange(7.0, T + 1, 7.0))))
                               if twice_weekly else list(np.arange(7.0, T + 1, 7.0)))
        if sched[-1] != T:
            sched.append(T)
        rec_sched = [T + 7.0, T + 14.0]
        arms = []
        for a in range(n_arms):
            is_ctrl = a == 0
            for cohort in (["MAIN", "RECOVERY"] if has_recovery and a in (0, n_arms - 1) else ["MAIN"]):
                code = codes[a] + ("R" if cohort == "RECOVERY" else "")
                dose = 0.0 if is_ctrl else float(doses[a - 1])
                arm_label = ("Vehicle control" if is_ctrl else f"{cid} {dose:g} mg/kg") + (
                    " recovery" if cohort == "RECOVERY" else "")
                arm = {"STUDYID": studyid, "ARMCD": code, "ARM": arm_label,
                       "COMPOUND_ID": None if is_ctrl else cid, "IS_CONTROL": is_ctrl,
                       "PLANNED_DOSE": dose, "DOSE_UNIT": "mg/kg", "ROUTE": route, "INTERVAL_DAYS": interval,
                       "PLANNED_START_DY": 1, "PLANNED_END_DY": int(T), "COHORT": cohort}
                arms.append(arm)
                arm_rows.append(arm)
        u_s_base = rng.normal(0, gen.p["study_baseline_sd"])
        dm, bw, ex, ds, supp = [], [], [], [], []
        animal_counter = 0
        grid = np.array(sorted(set(sched + (rec_sched if has_recovery else []) + [T + 1.0, T + 15.0])))
        study_dev, study_vals = gen.study_term(rng, grid)  # shared by every arm of this study
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
                    if is_donor:
                        mu0 = float(mu0) + float(donor_dev["baseline_shift"])
                    z0 = float(rng.normal(mu0 + u_s_base, sd0))
                    reg = Regimen(arm["IS_CONTROL"], arm["COMPOUND_ID"], arm["PLANNED_DOSE"], interval, 0.0,
                                  T - 1.0, route)
                    act = Regimen(**reg.__dict__)
                    deviation = (not reg.is_control) and rng.random() < p_dose_deviation
                    if deviation:
                        if rng.random() < 0.5:
                            act.end_day = float(rng.integers(5, int(T) - 3))
                        else:
                            act.dose = reg.dose * 0.5
                    ct = comp_truth.get(arm["COMPOUND_ID"]) if arm["COMPOUND_ID"] else None
                    last = T + (15.0 if arm["COHORT"] == "RECOVERY" else 1.0)
                    death_day = None
                    if rng.random() < p_death:
                        death_day = float(rng.integers(2, int(T)))
                    lat, meas = gen.animal_deviations(rng, grid, 1,
                                                      scale=np.array([gen.variance_multiplier(act)]))
                    growth = gen.growth(grid, species, sex, strain, z0)
                    effect = gen.effect(grid, act, ct)
                    if is_donor:
                        growth = growth * float(donor_dev["gmax_scale"])
                        effect = effect * float(donor_dev["effect_scale"])
                    mean = growth + effect
                    z_lat = z0 + mean + lat[0] + study_dev
                    z_obs = z_lat + meas[0]
                    truth_animals.append({"STUDYID": studyid, "USUBJID": usubjid, "Z0_TRUE": z0,
                                          "DOSE_DEVIATION": bool(deviation), "ACTUAL_END_DAY": act.end_day,
                                          "ACTUAL_DOSE": act.dose, "DEATH_DAY": death_day})
                    dm.append({"STUDYID": studyid, "DOMAIN": "DM", "USUBJID": usubjid, "SUBJID": subjid,
                               "SEX": sex, "SPECIES": species, "STRAIN": strain, "ARMCD": arm["ARMCD"],
                               "ARM": arm["ARM"], "RFSTDTC": _iso(start_date),
                               "RFENDTC": _iso(start_date + dt.timedelta(days=int(death_day or last)))})
                    ex.append({"STUDYID": studyid, "DOMAIN": "EX", "USUBJID": usubjid, "EXSEQ": 1,
                               "EXTRT": arm["COMPOUND_ID"] or "VEHICLE", "EXDOSE": act.dose,
                               "EXDOSU": "mg/kg", "EXROUTE": route,
                               "EXSTDTC": _iso(start_date),
                               "EXENDTC": _iso(start_date + dt.timedelta(
                                   days=int(min(act.end_day, (death_day or 1e9) - 1))))})
                    seq = 0
                    obs_days = sched + (rec_sched if arm["COHORT"] == "RECOVERY" else [])
                    for day in obs_days:
                        if death_day is not None and day >= death_day:
                            break
                        if day > 0 and rng.random() < p_missing:
                            continue
                        gi = int(np.where(grid == day)[0][0])
                        seq += 1
                        w = float(np.exp(z_obs[gi]))
                        latent_rows.append({"USUBJID": usubjid, "DAY": day, "Z_LATENT": float(z_lat[gi])})
                        bw.append(_bw_record(studyid, usubjid, seq, start_date, day, w))
                    term_day = None if death_day is not None else last
                    if term_day is not None and fasted_terminal:
                        gi = int(np.where(grid == term_day)[0][0])
                        seq += 1
                        w = float(np.exp(z_obs[gi] + gen.p["fasting_shift"]))
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
        study_truth.append({"STUDYID": studyid, **study_vals, "U_STUDY_BASELINE": u_s_base,
                            "COMPOUND_ID": cid})
        ts = [{"STUDYID": studyid, "DOMAIN": "TS", "TSSEQ": i + 1, "TSPARMCD": k, "TSVAL": v}
              for i, (k, v) in enumerate([("SPECIES", species), ("STRAIN", strain), ("SSTDTC", _iso(start_date)),
                                          ("SNDIGVER", "SEND Implementation Guide Version 3.1"),
                                          ("DOSDUR", f"P{int(T)}D"), ("ROUTE", route)])]
        tx = [{"STUDYID": studyid, "DOMAIN": "TX", "SETCD": a["ARMCD"], "SET": a["ARM"], "TXSEQ": j + 1,
               "TXPARMCD": "TRTDOS", "TXVAL": str(a["PLANNED_DOSE"])} for j, a in enumerate(arms)]
        frames = {"dm": dm, "bw": bw, "ex": ex, "ds": ds, "ts": ts, "tx": tx, "suppbw": supp}
        _write_study(send_dir / studyid, frames, style_index, kg=kg_flag, lower=lower_flag,
                     parquet=rng.random() < parquet_fraction, partial=partial_n, rng=rng)

    pd.DataFrame(study_rows).to_csv(meta_dir / "study_metadata.csv", index=False)
    pd.DataFrame(arm_rows).to_csv(meta_dir / "arm_regimen.csv", index=False)
    compound_ref.to_csv(meta_dir / "compound_reference.csv", index=False)
    # hidden truth (never read by fitting/feature code)
    write_json(truth_dir / "params.json", {"generator": gen.name, "params": gen.p, "compounds": comp_truth,
                                           "seed": seed, "donor_scenario": donor_scenario,
                                           "donor_deviations": donor_dev,
                                           "warning": "TRUTH-ONLY. Not for model fitting."})
    write_parquet(truth_dir / "animals.parquet", pd.DataFrame(truth_animals))
    write_parquet(truth_dir / "studies.parquet", pd.DataFrame(study_truth))
    write_parquet(truth_dir / "latent_paths.parquet", pd.DataFrame(latent_rows))
    return {"send_dir": str(send_dir), "metadata_dir": str(meta_dir), "truth_dir": str(truth_dir),
            "n_studies": n_studies + n_other_species, "generator": gen.name}


def _bw_record(studyid: str, usubjid: str, seq: int, start: dt.date, day: float, w: float) -> dict:
    return {"STUDYID": studyid, "DOMAIN": "BW", "USUBJID": usubjid, "BWSEQ": seq, "BWTESTCD": "BW",
            "BWTEST": "Body Weight", "BWORRES": f"{w:.1f}", "BWORRESU": "g", "BWSTRESC": f"{w:.1f}",
            "BWSTRESN": round(w, 1), "BWSTRESU": "g",
            "BWDTC": _iso(start + dt.timedelta(days=int(day))), "BWDY": _dy(day)}


def _write_study(folder: Path, frames: dict[str, list[dict]], s: int, kg: bool, lower: bool, parquet: bool,
                 partial: int, rng: np.random.Generator) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    empty_cols = {"suppbw": ["STUDYID", "RDOMAIN", "USUBJID", "IDVAR", "IDVARVAL", "QNAM", "QLABEL", "QVAL"]}
    for name, rows in frames.items():
        df = pd.DataFrame(rows, columns=empty_cols.get(name) if not rows else None)
        if name == "bw" and len(df):
            if kg:  # original results reported in kg, no standardized value supplied
                df["BWORRES"] = (df["BWSTRESN"].astype(float) / 1000.0).map(lambda v: f"{v:.4f}")
                df["BWORRESU"] = "kg"
                df["BWSTRESC"] = None
                df["BWSTRESN"] = None
                df["BWSTRESU"] = None
            if partial:
                idx = rng.choice(np.where(df["BWDY"].astype(int) > 1)[0], size=partial, replace=False)
                df.loc[idx, "BWDTC"] = df.loc[idx, "BWDTC"].str.slice(0, 7)
        if lower:
            df.columns = [c.lower() for c in df.columns]
        if parquet and name in {"bw", "dm"}:
            df.to_parquet(folder / f"{name}.parquet", index=False)
        else:
            df.to_csv(folder / f"{name}.csv", index=False)

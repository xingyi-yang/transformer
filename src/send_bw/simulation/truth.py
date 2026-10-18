"""TRUTH-ONLY access (synthetic oracle and stress generators).

This is the only module allowed to read the hidden generator directory. It is used exclusively
for evaluation (synthetic oracle reference and stress scenarios), never for fitting, feature
construction, tuning or model selection. Oracle results are labelled as synthetic references.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from send_bw.data.generate import GeneratorA, GeneratorB
from send_bw.simulation.designs import Request, reference_context
from send_bw.simulation.paths import Simulator
from send_bw.utils.artifacts import read_json


def load_truth(truth_dir: str | Path) -> dict[str, Any]:
    p = Path(truth_dir) / "params.json"
    if not p.exists():
        raise FileNotFoundError("no synthetic truth available (real data or truth withheld)")
    return read_json(p)


def oracle_simulator(truth_dir: str | Path, compound_ref: pd.DataFrame) -> Simulator:
    t = load_truth(truth_dir)
    gen = (GeneratorA if t["generator"] == "A" else GeneratorB)(t["params"])
    return Simulator("oracle_generator_" + t["generator"], gen, "generator", compound_reference=compound_ref,
                     compound_truth=t["compounds"], label="synthetic oracle (true data-generating parameters)")


def stress_simulators(truth_dir: str | Path | None, compound_ref: pd.DataFrame) -> list[Simulator]:
    """Independent generator B plus stress variants. Generator B never uses fitted parameters."""
    sims = [Simulator("generator_B", GeneratorB(), "generator", compound_reference=compound_ref,
                      label="independent generator B (different growth/effect/covariance families)")]
    if truth_dir is None:
        base_p, comp = GeneratorA.default_params(), None
    else:
        t = load_truth(truth_dir)
        base_p, comp = t["params"], t["compounds"]
    cs = copy.deepcopy(base_p)
    cs["sigma_p"] *= 1.6
    cs["ell"] *= 2.0
    sims.append(Simulator("stress_covariance_shift", GeneratorA(cs), "generator", compound_reference=compound_ref,
                          compound_truth=comp, label="generator A with sigma_p x1.6 and ell x2"))
    hh = copy.deepcopy(base_p)
    hh["sd_animal_slope"] *= 1.6
    hh["sd_study_slope"] *= 1.6
    sims.append(Simulator("stress_heterogeneity", GeneratorA(hh), "generator", compound_reference=compound_ref,
                          compound_truth=comp, label="generator A with animal/study slope SD x1.6"))
    for shape in ("delayed", "transient"):
        sims.append(Simulator(f"stress_{shape}_effect", GeneratorA(copy.deepcopy(base_p)), "generator",
                              shape_mode=shape, compound_reference=compound_ref, compound_truth=comp,
                              label=f"generator A with a {shape} effect shape"))
    ht = GeneratorB.default_params()
    ht["t_df"] = 3.0
    ht["sigma_m"] *= 1.5
    sims.append(Simulator("stress_heavy_tails", GeneratorB(ht), "generator", compound_reference=compound_ref,
                          label="generator B with t(3) measurement noise x1.5"))
    return sims


def monte_carlo_true_contrast(sim: Simulator, req: Request, n_animals_per_sex: int = 2000, seed: int = 0) -> np.ndarray:
    """Independent large-sample truth of the model-predicted (planned-regimen) contrast at T.

    Standardized over the request baseline distribution: the same animals are assigned to every
    group (paired), so the result is the baseline-standardized mean log-weight difference.
    """
    rng = np.random.default_rng(seed)
    base = reference_context(req, n_animals_per_sex, rng)
    base = base[base["ARM_KEY"] == req.control.name].reset_index(drop=True)
    T = np.array([req.horizon_day])
    regs = req.regimens()
    out = []
    mu_c = sim.sample(base.assign(ARM_KEY=req.control.name), regs, T, 1, seed)["mean"][0, :, 0]
    for g in req.active:
        mu_g = sim.sample(base.assign(ARM_KEY=g.name), regs, T, 1, seed)["mean"][0, :, 0]
        out.append(float(np.mean(mu_g - mu_c)))
    return np.array(out)

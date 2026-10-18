"""Joint trajectory simulation for design evaluation.

Complete latent paths are generated on the union grid of all candidate schedules BEFORE any
schedule, sample size or missingness is applied; candidates then use nested animal subsets and
visit subsets of the same paths (common random numbers).

Specified-effect scenarios: every animal is first simulated under the CONTROL regimen (so all
biological, study and parameter variation comes from the simulator), then a specified effect
delta_g * s_g(t) is added to active animals, with s_g(T) = 1 exactly so the estimand equals delta_g.
The shape s_g comes from the simulator's own predicted active-control difference (or a fixed
stress shape). 'model_predicted' simulates the actual planned regimens instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from send_bw.models.base import merge_context
from send_bw.simulation.designs import Request, reference_context


@dataclass
class Simulator:
    name: str
    obj: Any  # ResponseModel or generator with sample_study
    kind: str  # "model" | "generator"
    shape_mode: str = "own"  # own | delayed | transient | linear
    compound_reference: pd.DataFrame | None = None
    compound_truth: dict | None = None
    label: str = ""
    extra: dict = field(default_factory=dict)

    def sample(self, ctx: pd.DataFrame, regs: pd.DataFrame, times: np.ndarray, n: int, seed: int) -> dict:
        if self.kind == "model":
            return self.obj.sample_study(ctx, regs, times, n, seed)
        return self.obj.sample_study(ctx, regs, times, n, seed, compound_reference=self.compound_reference,
                                     compound_truth=self.compound_truth)


def fixed_shape(kind: str, grid: np.ndarray, T: float) -> np.ndarray:
    u = grid / T
    if kind == "linear":
        return u
    if kind == "delayed":
        return np.clip((u - 0.5) / 0.5, 0.0, None)
    if kind == "transient":
        s = 2.0 * (1 - np.exp(-grid / 5.0)) - u
        return s / s[np.argmin(np.abs(grid - T))]
    raise ValueError(kind)


def effect_shapes(sim: Simulator, req: Request, grid: np.ndarray, seed: int) -> tuple[np.ndarray, list[str]]:
    """(G_active, n_grid) shapes with s(T) = 1, from paired predictions on a reference population."""
    T = req.horizon_day
    iT = int(np.argmin(np.abs(grid - T)))
    notes = []
    if sim.shape_mode != "own":
        s = fixed_shape(sim.shape_mode, grid, T)
        return np.stack([s] * len(req.active)), [f"fixed {sim.shape_mode} effect shape"]
    rng = np.random.default_rng(seed)
    base = reference_context(req, 40, rng)
    base = base[base["ARM_KEY"] == req.control.name].reset_index(drop=True)
    regs = req.regimens()
    out = []
    for g in req.active:
        ctx_c = base.assign(ARM_KEY=req.control.name)
        ctx_g = base.assign(ARM_KEY=g.name)
        both = pd.concat([ctx_c, ctx_g], ignore_index=True)
        both["USUBJID"] = [f"R{i}" for i in range(len(both))]
        mu = _mean(sim, both, regs, grid, seed)
        n = len(base)
        diff = (mu[n:] - mu[:n]).mean(0)
        if abs(diff[iT]) < 0.002:
            out.append(grid / T)
            notes.append(f"{g.name}: predicted effect at T ~ 0; linear shape used")
        else:
            s = diff / diff[iT]
            out.append(np.clip(s, -1.0, 3.0))
    return np.stack(out), notes


def _mean(sim: Simulator, ctx: pd.DataFrame, regs: pd.DataFrame, grid: np.ndarray, seed: int) -> np.ndarray:
    if sim.kind == "model":
        return sim.obj.predict_mean(merge_context(ctx, regs), grid)
    return sim.sample(ctx, regs, grid, 1, seed)["mean"][0]


@dataclass
class SimBatch:
    grid: np.ndarray
    sex: np.ndarray  # (N,) 1 = male
    group: np.ndarray  # (N,)
    idx: np.ndarray  # (N,) within (group, sex)
    zb: np.ndarray  # (B, N)
    z_ctrl: np.ndarray  # (B, N, Tg)
    z_pred: np.ndarray  # (B, N, Tg)
    obs_mask: np.ndarray  # (B, N, Tg)
    shapes: np.ndarray  # (G, Tg)
    truth_pred: np.ndarray  # (B, G) model-implied contrast at T under planned regimens
    notes: list[str]


def simulate_batch(sim: Simulator, req: Request, n_max: tuple[int, ...], grid: np.ndarray, B: int, seed: int,
                   n_chunks: int = 8) -> SimBatch:
    rng = np.random.default_rng(seed)
    T = req.horizon_day
    iT = int(np.argmin(np.abs(grid - T)))
    shapes, notes = effect_shapes(sim, req, grid, seed + 1)
    sizes = np.full(n_chunks, B // n_chunks)
    sizes[: B % n_chunks] += 1
    zbs, zc, zp, tp = [], [], [], []
    regs_c, regs_a = req.regimens(all_control=True), req.regimens()
    ctx0 = None
    for nb in sizes:
        if nb == 0:
            continue
        ctx = reference_context(req, n_max, rng)
        ctx0 = ctx if ctx0 is None else ctx0
        s = int(rng.integers(0, 2**31 - 1))
        a = sim.sample(ctx, regs_c, grid, int(nb), s)
        b = sim.sample(ctx, regs_a, grid, int(nb), s)
        zbs.append(np.broadcast_to(ctx["BASELINE_LOGW"].to_numpy(float), (nb, len(ctx))))
        zc.append(a["z"])
        zp.append(b["z"])
        eff = b["mean"][:, :, iT] - a["mean"][:, :, iT]
        grp = ctx["GROUP"].to_numpy()
        tp.append(np.stack([eff[:, grp == g].mean(1) for g in range(1, len(req.groups))], axis=1))
    z_ctrl = np.concatenate(zc)
    N, Tg = z_ctrl.shape[1], z_ctrl.shape[2]
    md = req.missing_data
    mask = rng.random((B, N, Tg)) >= md.get("p_missed_visit", 0.0)
    drop = rng.random((B, N)) < md.get("p_dropout", 0.0)
    dday = rng.uniform(0, T, size=(B, N))
    mask &= ~(drop[:, :, None] & (grid[None, None, :] > dday[:, :, None]))
    return SimBatch(grid, (ctx0["SEX"] == "M").to_numpy().astype(int), ctx0["GROUP"].to_numpy(),
                    ctx0["IDX"].to_numpy(), np.concatenate(zbs), z_ctrl, np.concatenate(zp), mask, shapes,
                    np.concatenate(tp), notes)


def scenarios(req: Request) -> dict[str, dict[str, Any]]:
    G = len(req.active)
    d = req.deltas
    out = {"alternative": {"deltas": d, "source": "ctrl"}, "global_null": {"deltas": np.zeros(G), "source": "ctrl"}}
    if G > 1:
        for g in range(G):
            dd = d.copy()
            dd[g] = 0.0
            out[f"partial_null_{req.active[g].name}"] = {"deltas": dd, "source": "ctrl"}
    out["model_predicted"] = {"deltas": None, "source": "pred"}
    return out

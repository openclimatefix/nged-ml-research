"""Injected worlds of experiment 675, cached as hourly residuals in noise units plus truth tables.

    uv run --no-sync python e00675_worlds.py test    # 40 scored worlds (seeds 900-939) and E1-J's events
    uv run --no-sync python e00675_worlds.py train   # 200 probe-training worlds (seeds from 100000)

Experiment 660's generator is used unchanged, in its default configuration. Training worlds start
events before 2023-07-01 and hide every later reading (as experiment 672 does); scored worlds are
experiment 670's, scored from 2024-01-01. Output: `worlds/<kind>_<seed>.npz` (hourly residual of
15 Primaries in noise units) and `worlds/<kind>_<seed>_truth.parquet`; for scored worlds also
`worlds/test_<seed>_e1j.parquet`, E1-J's decoded events.
"""

# ruff: noqa: E501  # research code run once; behaviour fixed by the results

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path
from typing import Final

import e00660_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00653_common import GRID_LENGTH, SCORED_FROM
from e00653_core import events_frame
from e00660_ev34_detector import Rig, truths  # ty: ignore[unresolved-import]
from e00670_run import CONFIGS, decode_variants, params653  # ty: ignore[unresolved-import]
from e00672_data import TRAIN_END, inject_world  # ty: ignore[unresolved-import]
from e00675_pretrain import ROOT, hourly_of

WORLDS: Final[Path] = ROOT / "worlds"
N_WORKERS: Final[int] = 4


def build(job: tuple[str, int]) -> str:
    """Build one world."""
    kind, seed = job
    path = WORLDS / f"{kind}_{seed}.npz"
    if path.exists():
        return f"{kind} {seed} cached"
    rig = Rig()
    sites = rig.p653.sites
    sigma = np.array(
        [rig.frozen.sigma[s] if np.isfinite(rig.frozen.sigma[s]) else 1.0 for s in sites]
    )
    if kind == "test":
        world = gen.inject(
            panel=rig.panel, config=CONFIGS["default"], seed=seed, start_times=rig.starts
        )
        lo, hi = SCORED_FROM, GRID_LENGTH
    else:
        world = inject_world(config=CONFIGS["default"], seed=seed, upto=TRAIN_END)
        lo, hi = 0, TRAIN_END
    residual = rig.residual_of(world=world)
    hourly = np.stack([hourly_of(x=residual[s] / sigma[i]) for i, s in enumerate(sites)]).astype(
        np.float32
    )
    rows = [
        {
            "site": t.site, "start": t.start, "end": -1 if t.end is None else t.end,
            "amp_mw": t.amp_mw, "units": t.units, "kind": t.kind, "group": t.group,
            "partners": ",".join(t.partners),
        }
        for t in truths(events=world.events, lo=lo, hi=hi)
    ]  # fmt: skip
    schema = {"site": pl.String, "start": pl.Int64, "end": pl.Int64, "amp_mw": pl.Float64,
              "units": pl.Float64, "kind": pl.String, "group": pl.Int64, "partners": pl.String}  # fmt: skip
    pl.DataFrame(rows, schema=schema).write_parquet(WORLDS / f"{kind}_{seed}_truth.parquet")
    if kind == "test":
        ev = decode_variants(
            rig=rig, residual=residual, frozen=rig.frozen, params=params653(),
            rng=np.random.default_rng(seed),
        )["E1-J"]  # fmt: skip
        events_frame(events=ev).write_parquet(WORLDS / f"test_{seed}_e1j.parquet")
    np.savez_compressed(path, resid=hourly, sigma=sigma)
    return f"{kind} {seed} built"


if __name__ == "__main__":
    WORLDS.mkdir(parents=True, exist_ok=True)
    kind = sys.argv[1]
    seeds = range(900, 940) if kind == "test" else range(100000, 100200)
    with mp.get_context("spawn").Pool(N_WORKERS) as pool:
        for line in pool.imap_unordered(build, [(kind, s) for s in seeds]):
            print(line, flush=True)

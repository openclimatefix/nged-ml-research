"""Validation worlds and the real-record E1 mask of experiment 676.

    uv run --no-sync python e00676_worlds.py val     # 20 validation worlds, seeds from 105000
    uv run --no-sync python e00676_worlds.py mask    # E1 events on the real record before 2023-07-01

A validation world keeps the readings to 2024-01-01; only events starting from 2023-07-01 are
scored. Output goes to `<data>/worlds/`: `val_<seed>.npz` (hourly residual in noise units),
`val_<seed>_truth.parquet`, and `real_e1_mask.npy` (hours where E1 decodes a real event).
"""

# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00653_core import decode_world
from e00660_ev34_detector import Rig, truths  # ty: ignore[unresolved-import]
from e00670_run import CONFIGS, params653  # ty: ignore[unresolved-import]
from e00672_data import TRAIN_END, inject_world  # ty: ignore[unresolved-import]
from e00675_pretrain import hourly_of
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00676-pretrained-graph-model-for-load-transfers-part-2-edge-state-network"
)
WORLDS: Final[Path] = ROOT / "worlds"
VAL_SEEDS: Final = range(105000, 105020)
SCHEMA: Final = {
    "site": pl.String, "start": pl.Int64, "end": pl.Int64, "amp_mw": pl.Float64,
    "units": pl.Float64, "kind": pl.String, "group": pl.Int64, "partners": pl.String,
}  # fmt: skip


def build_val(seed: int, rig: Rig) -> None:
    """Build one validation world."""
    sites = rig.p653.sites
    sigma = np.array(
        [rig.frozen.sigma[s] if np.isfinite(rig.frozen.sigma[s]) else 1.0 for s in sites]
    )
    world = inject_world(config=CONFIGS["default"], seed=seed, upto=SCORED_FROM)
    residual = rig.residual_of(world=world)
    hourly = np.stack([hourly_of(x=residual[s] / sigma[i]) for i, s in enumerate(sites)])
    rows = [
        {
            "site": t.site, "start": t.start, "end": -1 if t.end is None else t.end,
            "amp_mw": t.amp_mw, "units": t.units, "kind": t.kind, "group": t.group,
            "partners": ",".join(t.partners),
        }
        for t in truths(events=world.events, lo=TRAIN_END, hi=SCORED_FROM)
    ]  # fmt: skip
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(WORLDS / f"val_{seed}_truth.parquet")
    np.savez_compressed(WORLDS / f"val_{seed}.npz", resid=hourly.astype(np.float32), sigma=sigma)


def build_mask(rig: Rig) -> None:
    """Hours (per Primary) inside an event E1 decodes on the real record before 2023-07-01."""
    sites = rig.p653.sites
    real = {
        s: np.where(np.arange(GRID_LENGTH) < TRAIN_END, v, np.nan)
        for s, v in rig.real_residual.items()
    }
    events, _ = decode_world(
        panel=rig.p653, residual=real, frozen=rig.frozen, arm="masked", params=params653()
    )
    mask = np.zeros((len(sites), GRID_LENGTH // 2), bool)
    for e in events:
        if e.primary not in sites or e.start >= TRAIN_END:
            continue
        end = min(TRAIN_END, e.start + 7 * DAY if e.end is None else e.end + 12)
        mask[sites.index(e.primary), max(0, e.start // 2 - 6) : end // 2 + 1] = True
    np.save(WORLDS / "real_e1_mask.npy", mask)
    print("masked hours share", mask[:, : TRAIN_END // 2].mean(), len(events), "events")


if __name__ == "__main__":
    WORLDS.mkdir(parents=True, exist_ok=True)
    rig = Rig()
    if sys.argv[1] == "val":
        for s in VAL_SEEDS:
            if not (WORLDS / f"val_{s}.npz").exists():
                build_val(s, rig)
                print("built", s, flush=True)
    else:
        build_mask(rig)

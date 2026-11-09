"""Worlds, features, and targets of experiment 672: experiment 660's injected worlds, for a network.

A *world* is the real power panel with synthetic events injected by experiment 660's generator. The
residual of a world (injected power minus the lag-free forecast, common mode removed) is turned
into the per-block features of experiment 658 and into per-block targets: the class of each
Primary, the sign and size of the injected change, the *donor* flag (the Primary is the donor of a
transfer that has a metered recipient), and the state of every link (off, first Primary donates,
second Primary donates).

Run with the folders of experiments 653, 657, 658, 660, and 670 on `PYTHONPATH`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import cache
from typing import Final

import e00660_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
from e00653_common import BLOCK, GRID_START, N_BLOCKS, SCORED_FROM, Panel
from e00653_common import block_means as block_means_653
from e00653_slow import fit_slow
from e00658_data import (
    CLASS_NONE,
    CLASS_RAMP,
    CLASS_ROUTINE,
    CLASS_SHIFT,
    N_CHANNELS,
    UNKNOWN_WEIGHT,
    World,
    features,
)
from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]

TRAIN_END: Final[int] = int((datetime(2023, 7, 1, tzinfo=UTC) - GRID_START).total_seconds() // 1800)
"""Half-hour index of 2023-07-01: training worlds end here, validation worlds start here."""
TRAIN_BLOCKS: Final[int] = TRAIN_END // BLOCK
VAL_BLOCKS: Final[int] = SCORED_FROM // BLOCK
"""Validation worlds cover the blocks from `TRAIN_BLOCKS` to `VAL_BLOCKS`."""
_CLASS_OF: Final[dict[str, int]] = {
    "edge_transfer": CLASS_SHIFT,
    "edge_transfer_unseen_donor": CLASS_SHIFT,
    "edge_transfer_unseen_recipient": CLASS_SHIFT,
    "temporary_shift": CLASS_SHIFT,
    "permanent_shift": CLASS_SHIFT,
    "placebo_pair": CLASS_SHIFT,
    "ramp": CLASS_RAMP,
    "routine_step": CLASS_ROUTINE,
    "pulse": CLASS_NONE,
}
"""Class of each generator event class. A placebo pair is two unconnected steps, so each end is a
shift with no link; the common-mode perturbation is not an event of one Primary and is skipped."""
assert N_CHANNELS == 7


@dataclass
class DWorld(World):
    """A world of experiment 658's layout, plus the donor target."""

    donor: np.ndarray | None = None
    """1 on blocks inside the window of a donor that has a metered recipient, shape (P, blocks)."""
    events: object | None = None
    """The generator's event table (None for the real record)."""


@cache
def rig() -> Rig:
    """The panels and forecasts of experiment 660 (loaded once per process)."""
    return Rig()


def hidden(*, panel: gen.PowerPanel, upto: int) -> gen.PowerPanel:
    """A copy of the panel with every reading from half-hour `upto` on set to NaN."""
    idx = np.arange(panel.length)
    return replace(
        panel, power={s: np.where(idx < upto, v, np.nan) for s, v in panel.power.items()}
    )


def inject_world(*, config: gen.GeneratorConfig, seed: int, upto: int | None) -> gen.InjectedWorld:
    """Inject into the real panel, or (with `upto`) into the part of it before `upto`.

    Events then start before `upto` and the readings from `upto` on are hidden, so a training
    world never shows the network a later half-hour.
    """
    r = rig()
    if upto is None:
        return gen.inject(panel=r.panel, config=config, seed=seed, start_times=r.starts)
    cutoff = datetime.fromtimestamp(GRID_START.timestamp() + 1800 * upto, tz=UTC)
    return gen.inject(
        panel=hidden(panel=r.panel, upto=upto),
        config=replace(config, cutoff=cutoff),
        seed=seed,
        start_times=r.starts,
    )


def residual_of(*, world: gen.InjectedWorld, solar: dict[str, np.ndarray] | None) -> dict:
    """The decoder's input residual, with the explicit solar term added when `solar` is given."""
    res = rig().residual_of(world=world)
    if solar is None:
        return res
    return {s: res[s] + solar.get(s, 0.0) for s in res}


def _span(*, row: dict) -> tuple[int, int]:
    b0 = row["start"] // BLOCK
    end = row["end"]
    b1 = N_BLOCKS if end is None else -(-int(end) // BLOCK)
    return b0, min(max(b1, b0 + 1), N_BLOCKS)


def build_world(
    *,
    world: gen.InjectedWorld | None,
    solar: dict[str, np.ndarray] | None,
    upto: int | None = None,
) -> DWorld:
    """Features and targets of an injected world, or of the real record when `world` is None.

    Args:
        world: The injected world, or None for the real, un-injected record.
        solar: Half-hourly explicit solar term per Primary added to the residual, or None.
        upto: Hide every residual reading from this half-hour on (real record only).

    Returns:
        The world on the 6-hour block grid.
    """
    r = rig()
    p = r.p653
    sigma = {s: (v if np.isfinite(v) else 1.0) for s, v in r.frozen.sigma.items()}
    if world is None:
        raw = r.real_residual
        if upto is not None:
            raw = {s: np.where(np.arange(len(v)) < upto, v, np.nan) for s, v in raw.items()}
        resid = {s: raw[s] + (0.0 if solar is None else solar.get(s, 0.0)) for s in raw}
    else:
        resid = residual_of(world=world, solar=solar)
    n_p = len(p.sites)
    feats = np.zeros((n_p, N_BLOCKS, N_CHANNELS), np.float32)
    slow = np.zeros((n_p, N_BLOCKS))
    for i, s in enumerate(p.sites):
        y = block_means_653(x=resid[s])
        sl = fit_slow(y=y, sigma=sigma[s], arm="masked") if np.isfinite(y).sum() > 1000 else y * 0
        slow[i] = sl
        feats[i] = features(x=y - sl, sigma=sigma[s])
    out = DWorld(feats=feats, slow=slow, residual=resid, truth=[])
    if world is not None:
        _targets(out=out, world=world, p=p, sigma=sigma)
    return out


def _targets(*, out: DWorld, world: gen.InjectedWorld, p: Panel, sigma: dict[str, float]) -> None:
    """Fill the class, weight, amplitude, donor, and link-state targets from the event table."""
    index = {s: i for i, s in enumerate(p.sites)}
    n_p = len(p.sites)
    cls = np.zeros((n_p, N_BLOCKS), np.int8)
    covered = np.zeros((n_p, N_BLOCKS), bool)
    donor = np.zeros((n_p, N_BLOCKS), np.float32)
    e_index = {e: k for k, e in enumerate(p.edges)}
    state = np.zeros((len(p.edges), N_BLOCKS), np.int8)
    rows = sorted(world.events.to_dicts(), key=lambda x: abs(x["delta"]))
    for row in rows:
        c = _CLASS_OF.get(row["event_class"])
        if c is None or row["site"] not in index:
            continue
        b0, b1 = _span(row=row)
        i = index[row["site"]]
        if row["event_class"] == "pulse":
            fresh = ~covered[i, b0:b1]
            cls[i, b0:b1][fresh] = c
            covered[i, b0:b1] = True
            continue
        cls[i, b0:b1] = c
        covered[i, b0:b1] = True
    for row in rows:
        if row["role"] != "donor" or not row["partners"] or row["site"] not in index:
            continue
        b0, b1 = _span(row=row)
        donor[index[row["site"]], b0:b1] = 1.0
        for rec in row["partners"]:
            key = tuple(sorted((row["site"], rec)))
            if key in e_index:
                state[e_index[key], b0:b1] = 1 if key[0] == row["site"] else 2
    base = rig().p653.power
    amp = np.zeros((n_p, N_BLOCKS), np.float32)
    for i, s in enumerate(p.sites):
        diff = block_means_653(x=world.power[s] - base[s])
        amp[i] = np.nan_to_num(diff / sigma[s], nan=0.0)
    present = out.feats[:, :, 1] > 0
    weight = np.where(covered, 1.0, UNKNOWN_WEIGHT) * present
    ea = np.array([index[a] for a, _ in p.edges])
    eb = np.array([index[b] for _, b in p.edges])
    known = covered[ea] | covered[eb]
    edge_w = np.where(known, 1.0, UNKNOWN_WEIGHT) * present[ea] * present[eb]
    out.cls, out.weight, out.amp = cls, weight.astype(np.float32), amp
    out.edge_state, out.edge_weight = state, edge_w.astype(np.float32)
    out.donor, out.events = donor, world.events

"""Inputs and targets of experiment 658: injected worlds turned into network features and labels.

A *world* is the real residual panel of experiment 653 with synthetic events added. This module
turns a world into per-block features (the network's input) and per-block targets (class, sign,
amplitude, and edge state), all on the 6-hour block grid of experiment 653.

Run with the folder of experiment 653 on `PYTHONPATH`, because the panel, the injector, and the slow
level are imported from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np
from e00653_common import BLOCK, DAY, GRID_LENGTH, GRID_START, N_BLOCKS, SCORED_FROM, Panel
from e00653_common import block_means as block_means_653
from e00653_inject import GenParams, Truth, inject
from e00653_slow import fit_slow

N_CHANNELS: Final[int] = 7
CLIP_UNITS: Final[float] = 30.0
TRAIN_END: Final[int] = int((datetime(2023, 7, 1, tzinfo=UTC) - GRID_START).total_seconds() // 1800)
"""Half-hour index of 2023-07-01: training worlds end here, validation worlds start here."""
TRAIN_BLOCKS: Final[int] = TRAIN_END // BLOCK
VAL_BLOCKS: Final[int] = SCORED_FROM // BLOCK
"""Validation worlds cover the blocks from `TRAIN_BLOCKS` to `VAL_BLOCKS`."""
UNKNOWN_WEIGHT: Final[float] = 0.1
"""Weight of the 'no event' target on time with no injected event (unlabelled, never negative)."""
CLASS_NONE, CLASS_SHIFT, CLASS_RAMP, CLASS_ROUTINE = 0, 1, 2, 3
_CLASS_OF: Final[dict[str, int]] = {
    "edge": CLASS_SHIFT,
    "solo": CLASS_SHIFT,
    "permanent_shift": CLASS_SHIFT,
    "permanent_transfer": CLASS_SHIFT,
    "placebo": CLASS_SHIFT,
    "ramp": CLASS_RAMP,
    "routine": CLASS_ROUTINE,
    "pulse": CLASS_NONE,
}


@dataclass
class World:
    """Features, slow levels, and (for injected worlds) targets of one residual-panel copy."""

    feats: np.ndarray
    """Shape (primaries, blocks, N_CHANNELS), float32."""
    slow: np.ndarray
    """Slow level in megawatts, shape (primaries, blocks)."""
    residual: dict[str, np.ndarray]
    """The world's half-hourly residual (common mode removed), per Primary."""
    truth: list[Truth]
    cls: np.ndarray | None = None
    """Class per block, shape (primaries, blocks), int8."""
    weight: np.ndarray | None = None
    """Weight of the class target, shape (primaries, blocks)."""
    amp: np.ndarray | None = None
    """Injected amplitude in noise units per block."""
    edge_state: np.ndarray | None = None
    """Edge state per block (0 off, 1 first Primary donates, 2 second), shape (edges, blocks)."""
    edge_weight: np.ndarray | None = None


def _window_mean(*, z: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of finite values of z[t+lo : t+hi) for every block t."""
    n = len(z)
    ok = np.isfinite(z)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, z, 0.0))])
    k = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi, 0, n)
    cnt = k[b] - k[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (c[b] - c[a]) / cnt
    mean[cnt == 0] = np.nan
    return mean, cnt


def _step(*, z: np.ndarray, k: int) -> np.ndarray:
    """Mean of the next k blocks minus mean of the previous k blocks, zero when under-observed."""
    after, na = _window_mean(z=z, lo=0, hi=k)
    before, nb = _window_mean(z=z, lo=-k, hi=0)
    out = after - before
    out[(na < k // 2) | (nb < k // 2) | ~np.isfinite(out)] = 0.0
    return out


def features(*, x: np.ndarray, sigma: float) -> np.ndarray:
    """Network input of one Primary: shape (blocks, N_CHANNELS).

    Args:
        x: Block means of the residual with the slow level removed, in megawatts (NaN if missing).
        sigma: The Primary's noise unit in megawatts.

    Returns:
        Channels: clipped residual in noise units, present flag, 1-day step, 3-day step, share of
        blocks in a centred 60-day window whose 1-day step exceeds 4 noise units, and the sine and
        cosine of the block's position in the day.
    """
    z = x / sigma
    present = np.isfinite(z)
    zc = np.clip(np.nan_to_num(z, nan=0.0), -CLIP_UNITS, CLIP_UNITS)
    s1 = np.clip(_step(z=z, k=4), -CLIP_UNITS, CLIP_UNITS)
    s3 = np.clip(_step(z=z, k=12), -CLIP_UNITS, CLIP_UNITS)
    big = (np.abs(s1) > 4).astype(float)
    rate, _ = _window_mean(z=np.where(present, big, np.nan), lo=-120, hi=120)
    phase = 2 * np.pi * (np.arange(len(z)) % 4) / 4
    return np.stack(
        [
            zc,
            present.astype(float),
            s1,
            s3,
            np.nan_to_num(rate, nan=0.0),
            np.sin(phase),
            np.cos(phase),
        ],
        axis=1,
    ).astype(np.float32)


def build_world(
    *,
    panel: Panel,
    residual: dict[str, np.ndarray],
    sigma: dict[str, float],
    rng: np.random.Generator | None,
    lo: int,
    hi: int,
    gen: GenParams | None,
    upto: int | None,
) -> World:
    """Inject events between half-hours `lo` and `hi` (if `rng` is given) and build the features.

    Args:
        panel: The residual panel of experiment 653.
        residual: The real residual with the common mode removed.
        sigma: Noise unit per Primary in megawatts.
        rng: Random generator for the injector, or None for the uninjected residual.
        lo: First half-hour that may carry an injected event.
        hi: Half-hour after the last that may carry one.
        gen: Injector settings (ignored when `rng` is None).
        upto: Everything from this half-hour on is hidden (NaN), or None to keep the whole grid.

    Returns:
        The world with features and, when injected, targets.
    """
    base = {
        s: (v if upto is None else np.where(np.arange(len(v)) < upto, v, np.nan))
        for s, v in residual.items()
    }
    truth: list[Truth] = []
    world = base
    if rng is not None:
        world, truth = inject(
            panel=panel,
            residual=base,
            sigma=sigma,
            rng=rng,
            lo=lo,
            hi=hi,
            gen=gen or GenParams(),
            kinds=("transfer", "negatives"),
        )
    n_p = len(panel.sites)
    feats = np.zeros((n_p, N_BLOCKS, N_CHANNELS), np.float32)
    slow = np.zeros((n_p, N_BLOCKS))
    for i, s in enumerate(panel.sites):
        y = block_means_653(x=world[s])
        sl = fit_slow(y=y, sigma=sigma[s], arm="masked") if np.isfinite(y).sum() > 1000 else y * 0
        slow[i] = sl
        feats[i] = features(x=y - sl, sigma=sigma[s])
    out = World(feats=feats, slow=slow, residual=world, truth=truth)
    if rng is not None:
        _targets(world=out, panel=panel, base=base, sigma=sigma)
    return out


def _block_span(*, start: int, end: int | None) -> tuple[int, int]:
    b0 = start // BLOCK
    b1 = N_BLOCKS if end is None else -(-end // BLOCK)
    return b0, max(b1, b0 + 1)


def _targets(
    *, world: World, panel: Panel, base: dict[str, np.ndarray], sigma: dict[str, float]
) -> None:
    """Fill the class, weight, amplitude, and edge targets of an injected world."""
    sites = panel.sites
    index = {s: i for i, s in enumerate(sites)}
    n_p = len(sites)
    cls = np.zeros((n_p, N_BLOCKS), np.int8)
    covered = np.zeros((n_p, N_BLOCKS), bool)
    for t in sorted(world.truth, key=lambda x: abs(x.amp_mw)):
        c = _CLASS_OF.get(t.kind)
        if c is None:
            continue
        b0, b1 = _block_span(start=t.start, end=t.end)
        i = index[t.site]
        if t.kind == "pulse":
            fresh = ~covered[i, b0:b1]
            cls[i, b0:b1][fresh] = c
            covered[i, b0:b1] |= True
            continue
        cls[i, b0:b1] = c
        covered[i, b0:b1] = True
    amp = np.zeros((n_p, N_BLOCKS), np.float32)
    for i, s in enumerate(sites):
        diff = block_means_653(
            x=world.residual[s] - np.where(np.isfinite(base[s]), base[s], np.nan)
        )
        amp[i] = np.nan_to_num(diff / sigma[s], nan=0.0)
    present = world.feats[:, :, 1] > 0
    weight = np.where(covered, 1.0, UNKNOWN_WEIGHT) * present
    edges = panel.edges
    e_index = {e: k for k, e in enumerate(edges)}
    state = np.zeros((len(edges), N_BLOCKS), np.int8)
    for t in world.truth:
        if t.kind not in ("edge", "permanent_transfer") or t.amp_mw >= 0 or not t.partners:
            continue
        b0, b1 = _block_span(start=t.start, end=t.end)
        for r in t.partner_amp:
            key = tuple(sorted((t.site, r)))
            if key in e_index:
                state[e_index[key], b0:b1] = 1 if key[0] == t.site else 2
    ea = np.array([index[a] for a, _ in edges])
    eb = np.array([index[b] for _, b in edges])
    known = covered[ea] | covered[eb]
    edge_w = np.where(known, 1.0, UNKNOWN_WEIGHT) * present[ea] * present[eb]
    world.cls, world.weight, world.amp = cls, weight.astype(np.float32), amp
    world.edge_state, world.edge_weight = state, edge_w.astype(np.float32)


def graph_arrays(
    *, panel: Panel, neighbours: dict[str, list[str]] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised neighbour matrix and the edge index pairs of a graph.

    Args:
        panel: The panel (its sites fix the row order).
        neighbours: Neighbour lists to use instead of the real graph (for the rewired placebo).

    Returns:
        The matrix (primaries, primaries) and an integer array (edges, 2) of site indices.
    """
    nb = neighbours or panel.neighbours
    index = {s: i for i, s in enumerate(panel.sites)}
    a = np.zeros((len(panel.sites), len(panel.sites)), np.float32)
    pairs = set()
    for s, ns in nb.items():
        for q in ns:
            if s in index and q in index:
                a[index[s], index[q]] = 1.0
                pairs.add(tuple(sorted((s, q))))
    deg = a.sum(axis=1, keepdims=True)
    a = np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)
    e = np.array([(index[x], index[y]) for x, y in sorted(pairs)], dtype=np.int64)
    return a, e


assert N_BLOCKS * BLOCK == GRID_LENGTH
assert DAY == 48

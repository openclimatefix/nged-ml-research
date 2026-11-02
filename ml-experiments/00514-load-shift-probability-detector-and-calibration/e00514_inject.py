# ruff: noqa: E501
"""Experiment 514: the series-level injector and the world builder.

A world is a 90-day window of all 20 scaled series (16 Primaries, 4 supply points) taken from real
residual history, with load-shift pulses added to a donor Primary (and, for half the pulses, the
opposite change to one or two of its neighbours, as in experiments 456 and 482), a random cutoff,
and the truth rows. It does not use experiment 507's window-level generator.
"""

from dataclasses import dataclass, field
from typing import Final

import e00514_data as d514  # ty: ignore[unresolved-import]
import e00514_detect as det  # ty: ignore[unresolved-import]
import numpy as np

WORLD_LENGTH: Final[int] = 90 * 48
EVAL_ROWS: Final[int] = 30 * 48
"""Rows before the cut that carry labels (the last 30 days of the world)."""
MIN_CUTOFF: Final[int] = 40 * 48
FIRST_START: Final[int] = 14 * 48
HELD_OUT_PRIMARIES: Final[tuple[int, ...]] = (3, 8, 13)
SIZE_RANGE: Final[tuple[float, float]] = (0.4, 3.5)
DURATION_RANGE: Final[tuple[int, int]] = (2, 672)
SHAPES_TRAIN: Final[tuple[str, ...]] = ("box", "own")
SHAPE_HELD_OUT: Final[str] = "circuit"
MIRROR_SHARE: Final[float] = 0.5
LEAK_RANGE: Final[tuple[float, float]] = (0.3, 1.2)
MAX_RAMP: Final[int] = 4


@dataclass
class Pulse:
    """One injected pulse."""

    series: int
    start: int
    duration: int
    size: float
    sign: int
    shape: str
    mirrored: bool
    genuine: bool = False


@dataclass
class World:
    """A world: the series as seen at the cutoff, and the truth."""

    matrix: np.ndarray
    """(20, WORLD_LENGTH) series with NaN from `end` on."""
    truth: np.ndarray
    """(20, WORLD_LENGTH) True inside an injected load-shift pulse (not a genuine level change)."""
    genuine: np.ndarray
    """(20, WORLD_LENGTH) True from the start of an injected genuine level change."""
    end: int
    shape: np.ndarray
    """(16, WORLD_LENGTH) causal demand-shape profile of the Primaries."""
    scales: det.Scales
    evaluated: list[int]
    pulses: list[Pulse] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _profile(*, kind: str, start: int, duration: int, rng: np.random.Generator, own: np.ndarray) -> np.ndarray:
    """Return the within-pulse profile, normalised to mean 1 over the pulse."""
    t = np.arange(start, start + duration)
    if kind == "box":
        prof = np.ones(duration)
    elif kind == "own":
        seg = np.nan_to_num(own[start : start + duration], nan=np.nanmean(own) if np.isfinite(own).any() else 1.0)
        prof = np.maximum(seg, 1e-3)
    elif kind == SHAPE_HELD_OUT:
        tod = (t % 48) / 2.0
        if rng.random() < 0.5:
            prof = np.where((tod >= 0.5) & (tod < 7.5), 1.0, 0.2)
        else:
            b1, b2 = rng.uniform(0.2, 1.5, size=2)
            prof = 0.3 + b1 * np.exp(-((tod - 8) ** 2) / 8.0) + b2 * np.exp(-((tod - 18) ** 2) / 8.0)
    else:
        raise ValueError(kind)
    prof = prof / prof.mean()
    ramp = int(rng.integers(0, MAX_RAMP + 1))
    if ramp and duration > 2 * ramp:
        edge = np.linspace(1 / (ramp + 1), 1 - 1 / (ramp + 1), ramp)
        prof[:ramp] *= edge
        prof[-ramp:] *= edge[::-1]
    return prof


def add_pulse(
    *, matrix: np.ndarray, truth: np.ndarray, pulse: Pulse, scales: det.Scales, own: np.ndarray,
    neighbours: list[int], rng: np.random.Generator,
) -> None:  # fmt: skip
    """Add a pulse (and its mirror) to `matrix` and mark the truth rows."""
    a, dur = pulse.start, pulse.duration
    prof = _profile(kind=pulse.shape, start=a, duration=dur, rng=rng, own=own)
    amp = pulse.sign * pulse.size * scales.spread[pulse.series]
    delta = amp * prof
    matrix[pulse.series, a : a + dur] += delta
    truth[pulse.series, a : a + dur] = True
    if pulse.mirrored and neighbours:
        n_rec = int(rng.integers(1, min(2, len(neighbours)) + 1))
        recipients = rng.choice(neighbours, size=n_rec, replace=False)
        leak = float(rng.uniform(*LEAK_RANGE))
        mw = delta * scales.mw_per_z[pulse.series]
        for q in recipients:
            matrix[q, a : a + dur] += -leak * mw / n_rec / scales.mw_per_z[q]
            truth[q, a : a + dur] = True


def make_world(
    *, fold: d514.Fold, month_start: int, rng: np.random.Generator, kind: str,
    shapes: tuple[str, ...], donors: tuple[int, ...] | None, shape_all: np.ndarray,
    scales: det.Scales, event_rows: np.ndarray, forced_start: int | None = None,
    force: dict | None = None,
) -> World | None:  # fmt: skip
    """Build one world.

    Args:
        fold: The cached fold whose residual history supplies the real series.
        month_start: Grid index of the month whose history fixed `scales`.
        rng: The world's random generator.
        kind: `pulses`, `pulses_genuine`, `null`, or `dip` (a real D8 window, no injection).
        shapes: Shapes a pulse may take.
        donors: Donor Primaries allowed (`None` for any Primary outside the held-out set).
        shape_all: (16, n) demand-shape profiles.
        scales: Noise scales fitted on rows below `month_start`.
        event_rows: (16, n) True inside a labelled event padded by 2 days.
        forced_start: Window start for `dip` worlds.
        force: Fixed `size`, `duration`, `shape`, `sign`, and `mirrored` of a single pulse (recall grid).

    Returns:
        The world, or `None` if no donor window had enough data.
    """
    w0 = forced_start if forced_start is not None else month_start + int(rng.integers(0, 20 * 48))
    w1 = w0 + WORLD_LENGTH
    if w1 >= fold.refit:
        return None
    full = fold.partners()
    matrix = full[:, w0:w1].copy()
    truth = np.zeros_like(matrix, dtype=bool)
    genuine = np.zeros_like(matrix, dtype=bool)
    end = int(rng.integers(MIN_CUTOFF, WORLD_LENGTH)) - d514.CUT_STEPS
    graph = d514.neighbour_graph()
    allowed = [p for p in (donors if donors is not None else range(16)) if donors is not None or p not in HELD_OUT_PRIMARIES]
    def usable(p: int) -> bool:
        seg = matrix[p, :end]
        return bool(np.isfinite(seg).mean() > 0.8 and event_rows[p, w0 : w0 + end].mean() < 0.02)
    candidates = [p for p in allowed if usable(p)]
    pulses: list[Pulse] = []
    if kind in ("pulses", "pulses_genuine"):
        if not candidates:
            return None
        donor = int(rng.choice(candidates))
        occupied = np.zeros(WORLD_LENGTH, dtype=bool)
        for _ in range(1 if force else int(rng.integers(1, 3))):
            dur = int(force["duration"]) if force else int(np.exp(rng.uniform(np.log(DURATION_RANGE[0]), np.log(DURATION_RANGE[1]))))
            lo, hi = FIRST_START, end - 1
            if force:
                hi = max(lo + 1, end - dur - 2 * det.FLANK_MAX if end - dur - 2 * det.FLANK_MAX > lo else end - 1)
            if hi <= lo:
                continue
            a = int(rng.integers(lo, hi))
            if occupied[max(0, a - 96) : min(a + dur + 96, WORLD_LENGTH)].any():
                continue
            occupied[a : a + dur] = True
            pulse = Pulse(
                series=donor, start=a, duration=min(dur, WORLD_LENGTH - a),
                size=float(force["size"]) if force else float(np.exp(rng.uniform(np.log(SIZE_RANGE[0]), np.log(SIZE_RANGE[1])))),
                sign=int(force["sign"]) if force else int(rng.choice([-1, 1])),
                shape=str(force["shape"]) if force else str(rng.choice(shapes)),
                mirrored=bool(force["mirrored"]) if force else bool(rng.random() < MIRROR_SHARE),
            )  # fmt: skip
            neighbours = [q for q in graph.get(donor, []) if q not in HELD_OUT_PRIMARIES]
            add_pulse(
                matrix=matrix, truth=truth, pulse=pulse, scales=scales,
                own=shape_all[donor, w0:w1].astype(np.float64), neighbours=neighbours, rng=rng,
            )  # fmt: skip
            pulses.append(pulse)
    if kind == "pulses_genuine":
        others = [p for p in candidates if p not in {x.series for x in pulses}]
        if others:
            g = int(rng.choice(others))
            a = int(rng.integers(FIRST_START, max(FIRST_START + 1, end - 4)))
            size = float(np.exp(rng.uniform(np.log(SIZE_RANGE[0]), np.log(SIZE_RANGE[1]))))
            sign = int(rng.choice([-1, 1]))
            matrix[g, a:] += sign * size * scales.spread[g]
            genuine[g, a:] = True
            pulses.append(Pulse(series=g, start=a, duration=WORLD_LENGTH - a, size=size, sign=sign, shape="step", mirrored=False, genuine=True))
    if kind in ("null", "dip") and not candidates:
        return None
    matrix[:, end:] = np.nan
    injected = sorted({x.series for x in pulses} | {int(q) for q in np.flatnonzero(truth.any(axis=1))})
    pool = [p for p in candidates if p not in injected]
    extra = [int(rng.choice(pool))] if pool else []
    evaluated = [*injected, *extra] if kind != "dip" else [8] if 8 in candidates else extra
    if not evaluated:
        return None
    return World(
        matrix=matrix, truth=truth, genuine=genuine, end=end,
        shape=shape_all[:, w0:w1].astype(np.float64), scales=scales, evaluated=evaluated,
        pulses=pulses, meta={"w0": w0, "kind": kind},
    )  # fmt: skip


def world_candidates(*, world: World) -> list[dict]:
    """Return, per evaluated series, its candidates, features, and the row index range evaluated."""
    edges = det.edge_matrix(matrix=world.matrix, scales=world.scales)
    out = []
    lo_rows = max(det.FLANK_MIN, world.end - EVAL_ROWS)
    for i in world.evaluated:
        cands = det.scan_series(x=world.matrix[i], sigma=world.scales.sigma[i], end=world.end)
        x = det.candidate_features(
            cands=cands, series=i, matrix=world.matrix, edges=edges, scales=world.scales,
            shape=world.shape[i] if i < 16 else None,
        )  # fmt: skip
        out.append({"series": i, "cands": cands, "x": x, "rows": (lo_rows, world.end)})
    return out


def candidate_labels(*, world: World, entry: dict) -> np.ndarray:
    """Return 1 for a candidate with at least half its rows inside an injected pulse, else 0."""
    cands, i = entry["cands"], entry["series"]
    y = np.zeros(len(cands), dtype=int)
    for j in range(len(cands)):
        a = int(cands.start[j])
        b = int(min(a + cands.duration[j], world.end))
        if b > a and world.truth[i, a:b].mean() >= 0.5:
            y[j] = 1
    return y

# ruff: noqa: E501
"""Experiment 706, stage 2: the training and scoring tables of the learned look-back arms (CPU only).

Stage 2 trains two small neural networks that decide how much of the look-back offset of stage 1 to
apply. This module builds everything the networks read, once, so that each of the 240 fits only
loads arrays. Nothing here fits a network.

What is built (all under `<experiment folder>/s2/cache/`, written through temporary files):

- `real_dense.npz`: the stored normal-running quantiles, the cleaned power at every target row, the
  weather of the target rows, and the capacities, on the dense (series, origin, row) layout of
  `e00706_common`.
- `real_entries.npz`: one entry per (series, origin) of the real world. An entry holds every
  look-back feature of the origin: see "Entry tables" below.
- `bank_<month>_rows.npz` and `bank_<month>_entries.npz`: the rows and entries of the injected-shift
  bank of experiment 667 for the months 2025-11 to 2026-06 (a bank row at or after the holdout is
  never read).
- `ramp_train_<month>_entries.npz` and `ramp_test_<month>_entries.npz`: own-only ramp worlds for the
  fold of that month.
- `graphs.npz`: the real neighbour graph and the rewired graph; `noise.npz`: the noise unit.

Entry tables. Every table holds, per entry (a series and an origin in some world), arrays over
`WINDOW_DAYS` = 1, 3, 7, and 14 days of exponentially weighted look-back windows that end 6 hours
before the origin:

- `gate` (E, 4, 6, 3): per window and lead band, the ridge intercept and the two slopes on the
  irradiance driver and the cold driver, as in arm D of stage 1 (an exponentially weighted window
  in place of a box window).
- `nb_gate` (2, E, 4, 6): per graph (0 real, 1 rewired), the mean over the series' neighbours of
  the neighbours' exponentially weighted window mean residual, whose newest residual is 6 hours
  older than the origin.
- `box` (E, 6): arm G-lb's offset per lead band, 0.75 times the 14-day box mean of the band residual.
- `slow` (E, 6): the part of `box` that the change-point split of arm N-flag applies in full; the
  step part is `box - slow`.
- `flag_own` (E, 6) and `flag_nb` (2, E, 5): the 11 inputs of arm N-flag, from the series' own
  28-day daily residuals and from the neighbours' steps at the same change point.
- `w_rule` (E,): arm N-rule's weight, 0 when the step in the 14-day window exceeds 3 noise units and
  at least half of it arrives within one day, else 1.
- `covered` (E,): whether the 14-day window holds at least half the half-hours of residual that a
  complete window would hold; an uncovered entry has `box` of zero, as in stage 1.
- `flag_y` (E,): the generator's event flag, 1 when the series' own injected event switched on or
  off inside the 28-day window, 0 when none did or the world is an own-only ramp, and -1 in the
  real world, where it is unknown. The flag is 1 for every kind of event, own-only events included,
  so arm N-flag's probability is trained to detect any injected step and not a transfer alone.

Look-ahead rules, unchanged from stage 1. The residuals of a window end 6 hours before the origin,
the weather of the regressors in a window is the 00:00 UTC run of the origin day of each forecast
that produced a residual, and the weather of a target row is the 00:00 UTC run of the origin's own
day. The neighbours' residuals obey the same 6-hour lag.

Approximations (each is printed by the builder where it can be measured):

- The smallest-lead residual of a world is the residual of the lead-band-0 forecasts (leads up to
  24 hours), where stage 1's `smallest` takes the forecast of the newest origin only.
- A bank world holds the re-run Candidate forecasts of the Primary that carries the event, on even
  half-hours from 7 days before the onset to 14 days after the end. Before that window the series
  is filled with the residuals of the Candidate forecasts in the bank's own `real_<month>.npz`
  files (clean power minus the Candidate median, joined on Primary, origin, and valid time; even
  half-hours only), so that one forecaster supplies the whole series. The stored-NRA residuals of
  stage 1 are never mixed into a bank series. Every residual of a bank world is on
  even half-hours only, and its coverage thresholds are halved.
- The neighbours' residuals in a bank world are the stored-NRA residuals of stage 1 plus the shifts that the generator
  of experiment 650 added to those Primaries in that pass (their own event and the mirrored step of
  a transfer), with the forecast held fixed (no re-run). The bank stores no re-run forecast for a
  Primary outside its own event window.
- Own-only ramps are additive worlds on the real rows: the truth is the real power plus a ramp of
  10 points of capacity over 60 days (held afterwards), the forecast is the stored normal-running
  forecast, the neighbours' residuals are the real ones. This is a no-re-run world, so it
  understates how far a re-run forecast would have followed the ramp.
"""

import csv
import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import e00706_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.signal import lfilter
from studies.eval_protocol.metrics import MEDIAN_INDEX
from studies.experiment_store import read_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

S2_DIR: Final[Path] = c.OUT_DIR / "s2"
CACHE_DIR: Final[Path] = S2_DIR / "cache"
HEAD_DATA: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00667-*")) / "prevailing" / "head_data"
NGED_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15"
BANK_MONTHS: Final[tuple[str, ...]] = (
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
"""The bank months that are ever opened; the months 2026-07 and later are never read."""
FOLD_MONTHS: Final[tuple[str, ...]] = (
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
EMBARGO_DAYS: Final[int] = 7
WINDOW_DAYS: Final[tuple[float, ...]] = (1.0, 3.0, 7.0, 14.0)
N_WINDOWS: Final[int] = len(WINDOW_DAYS)
N_GATE_INPUTS: Final[int] = 4 * N_WINDOWS
"""Per window: the global offset, the irradiance slope term, the cold slope term, the neighbours' offset."""
N_FLAG_OWN: Final[int] = 6
N_FLAG_NB: Final[int] = 5
N_FLAG: Final[int] = N_FLAG_OWN + N_FLAG_NB
N_GRAPHS: Final[int] = 2
GRAPH_NAMES: Final[tuple[str, ...]] = ("real", "rewired")
N_CP_DAYS: Final[int] = 28
"""Days of daily mean residuals in the change-point window."""
MIN_SEGMENT_DAYS: Final[int] = 3
CP_IMPROVEMENT: Final[float] = 0.5
"""A change point is accepted when it cuts the sum of squares to this share of the unsplit value."""
RULE_NOISE_UNITS: Final[float] = 3.0
RULE_SHARE: Final[float] = 0.5
RULE_WINDOW_DAYS: Final[int] = 14
NOISE_HISTORY_ORIGINS: Final[int] = 180
NOISE_MIN_VALUES: Final[int] = 10
NOISE_FALLBACK_POINTS: Final[float] = 1.5
NOISE_FLOOR_POINTS: Final[float] = 0.25
RAMP_POINTS: Final[float] = 10.0
RAMP_HALF_HOURS: Final[int] = 60 * c.PER_DAY
N_RAMP_TRAIN_WORLDS: Final[int] = 48
RAMP_TEST_LEAD_DAYS: Final[int] = 30
"""A test ramp starts this many days before its fold month, so the fold month sees the ramp at 5 to 10 points."""
BANK_ORIGIN_LAG_HALF_HOURS: Final[int] = 6
"""A bank row's `oi` is the 09:00 UTC network origin; the forecast origin of stage 1 is 3 hours later."""
EVENT_PAD_BEFORE: Final[int] = 7 * c.PER_DAY
GRAPH_SEED: Final[int] = 706_002
GRAPH_KEYS: Final[frozenset[str]] = frozenset({"nb_gate", "flag_nb"})
"""Table arrays whose first axis is the graph, not the entry."""
FLAG_TRANSFER_DAYS: Final[int] = 28
RULE_CHECK_TOLERANCE_POINTS: Final[float] = 1e-3
"""The largest difference between arm N-rule at weight 1 and stage 1's stored `g` correction."""

type Table = dict[str, np.ndarray]


# --------------------------------------------------------------------------- small helpers


def month_start_stamp(*, month: str) -> int:
    """Return the stamp of 00:00 UTC on the first day of `month` on the stage 1 grid."""
    year, number = (int(part) for part in month.split("-"))
    start = dt.datetime(year, number, 1, tzinfo=dt.UTC)
    return int((start - c.GRID_START).total_seconds() // 1800)


def embargo_stamp(*, month: str) -> int:
    """Return the stamp before which a fold's training rows must have their valid time."""
    return month_start_stamp(month=month) - EMBARGO_DAYS * c.PER_DAY


def save_arrays(*, path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write `arrays` as an uncompressed `.npz` through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)  # ty: ignore[invalid-argument-type]
    temporary.replace(path)


def load_arrays(*, path: Path) -> dict[str, np.ndarray]:
    """Read every array of an `.npz` into memory."""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def concat_tables(*, tables: list[Table]) -> Table:
    """Concatenate entry tables along the entry axis (axis 1 for the per-graph arrays)."""
    return {
        name: np.concatenate([table[name] for table in tables], axis=1 if name in GRAPH_KEYS else 0)
        for name in tables[0]
    }


# --------------------------------------------------------------------------- graphs


def neighbour_lists(*, inputs: c.Inputs) -> list[list[int]]:
    """Return each Primary's metered neighbours from NGED's interconnection list, in the order of experiment 400.

    The logic of `nged_neighbours` in `e00400_blend.py`, with the neighbours of a Primary sorted by
    the number in the label, which is the order that the generator of experiment 650 draws receivers
    from.
    """
    label_of = dict(zip(inputs.ids.tolist(), inputs.labels, strict=True))
    with (NGED_DIR / "Monitor Direction.csv").open() as handle:
        monitors = list(csv.DictReader(handle))
    sites_of_number: dict[str, set[str]] = {}
    for row in monitors:
        if int(row["TimeSeriesID"]) in label_of:
            sites_of_number.setdefault(row["SubstationNumber"], set()).add(
                label_of[int(row["TimeSeriesID"])]
            )
    with (NGED_DIR / "Primary Substation Interconnections.csv").open() as handle:
        pairs = [(r["PrimarySubstation1"], r["PrimarySubstation2"]) for r in csv.DictReader(handle)]
    metered: dict[str, set[str]] = {label: set() for label in inputs.labels}
    for first, second in pairs:
        for here, there in ((first, second), (second, first)):
            for site in sites_of_number.get(here, ()):
                if there in sites_of_number:
                    metered[site] |= sites_of_number[there]
    position = {label: index for index, label in enumerate(inputs.labels)}
    return [
        [
            position[other]
            for other in sorted(metered[label] - {label}, key=lambda name: int(name[1:]))
            if other in position
        ]
        for label in inputs.labels
    ]


def adjacency_of(*, lists: list[list[int]]) -> np.ndarray:
    """Return the symmetric adjacency matrix (K, K) of the neighbour lists."""
    size = len(lists)
    matrix = np.zeros((size, size), dtype=bool)
    for here, there in enumerate(lists):
        matrix[here, there] = True
    return matrix | matrix.T


def rewire(*, adjacency: np.ndarray, seed: int) -> np.ndarray:
    """Return a degree-preserving rewiring of an undirected graph, by random double edge swaps.

    Each swap replaces the edges (a, b) and (c, d) by (a, d) and (c, b) when neither new edge exists
    already and neither is a self-loop. The swaps are fixed by `seed`.

    Raises:
        ValueError: If the rewired graph has another degree sequence than the input, is equal to
            the input, or shares more than half of its edges with the input.
    """
    size = adjacency.shape[0]
    edges = [(i, j) for i in range(size) for j in range(i + 1, size) if adjacency[i, j]]
    present = set(edges)
    rng = np.random.default_rng(seed)
    for _ in range(50 * len(edges)):
        first, second = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[int(first)]
        x, y = edges[int(second)]
        if rng.random() < 0.5:
            x, y = y, x
        if a == y or x == b:
            continue
        new_first = (min(a, y), max(a, y))
        new_second = (min(x, b), max(x, b))
        if new_first in present or new_second in present or new_first == new_second:
            continue
        present -= {edges[int(first)], edges[int(second)]}
        present |= {new_first, new_second}
        edges[int(first)], edges[int(second)] = new_first, new_second
    out = np.zeros_like(adjacency)
    for i, j in edges:
        out[i, j] = out[j, i] = True
    if not np.array_equal(out.sum(axis=0), adjacency.sum(axis=0)):
        msg = "the rewired graph changed a degree"
        raise ValueError(msg)
    if np.array_equal(out, adjacency):
        msg = "the rewired graph equals the real graph"
        raise ValueError(msg)
    shared = int((out & adjacency).sum()) // 2
    if shared > len(edges) / 2:
        msg = f"the rewired graph shares {shared} of its {len(edges)} edges with the real graph"
        raise ValueError(msg)
    return out


# --------------------------------------------------------------------------- window features


def _filter(*, values: np.ndarray, decay: float) -> np.ndarray:
    """Return the exponentially decaying running sum along the last axis."""
    return lfilter([1.0], [1.0, -decay], values, axis=-1)


def _origin_ends(*, origins: np.ndarray) -> np.ndarray:
    """Return the newest stamp of each origin's look-back window (6 hours before the origin)."""
    return c.ORIGIN_STAMP + c.PER_DAY * origins - c.LAG_6H


def ew_fit(
    *,
    residual: np.ndarray,
    design: np.ndarray,
    penalty: np.ndarray,
    tau_days: float,
    origins: np.ndarray,
    density: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the ridge regression of arm D on an exponentially weighted window ending 6 hours before each origin.

    The weight of a half-hour is `exp(-age / (tau_days x 48))`. The intercept is not penalised, and
    the penalty of a slope is the stage 1 penalty. A slope is zero when the weighted count of
    half-hours with its driver above `c.ACTIVE_THRESHOLD` is under `c.MIN_ACTIVE` scaled by
    `tau_days / 14` (the share of a 14-day window that the weighting keeps). A window is covered when
    its weighted count of residuals reaches `c.MIN_COVERAGE` of the weighted count of a complete
    window of the same density; an uncovered window gives zero coefficients.

    Args:
        residual: One series' residuals per lead band, shape (B, N_STAMPS), NaN where absent.
        design: The two drivers on the stamp grid, shape (N_STAMPS, 2); a NaN row is not used.
        penalty: The ridge penalty of each slope, shape (2,).
        tau_days: The weighting's time constant in days.
        origins: Origin indices, shape (n,).
        density: The share of stamps that a complete world holds (1 for the real world, 0.5 for the
            even-half-hour bank worlds).

    Returns:
        The coefficients (intercept, irradiance slope, cold slope), shape (B, n, 3), and whether each
        window is covered, shape (B, n).
    """
    n_bands = residual.shape[0]
    decay = float(np.exp(-1.0 / (tau_days * c.PER_DAY)))
    usable = np.isfinite(design).all(axis=1)
    driver = np.where(usable[:, None], design, 0.0)
    weight = (np.isfinite(residual) & usable[None, :]).astype(np.float64)
    value = np.where(weight > 0, residual, 0.0)
    x0, x1 = driver[:, 0][None, :], driver[:, 1][None, :]
    stack = np.stack(
        [
            weight,
            weight * x0,
            weight * x1,
            weight * x0 * x0,
            weight * x0 * x1,
            weight * x1 * x1,
            weight * value,
            weight * x0 * value,
            weight * x1 * value,
            weight * (x0 > c.ACTIVE_THRESHOLD),
            weight * (x1 > c.ACTIVE_THRESHOLD),
        ],
        axis=1,
    )
    smoothed = _filter(values=stack, decay=decay)
    last = _origin_ends(origins=origins)
    seen = last >= 0
    clipped = np.maximum(last, 0)
    at = smoothed[..., clipped]
    mass = (1.0 - decay ** (clipped + 1)) / (1.0 - decay)
    covered = seen[None, :] & (at[:, 0] >= c.MIN_COVERAGE * density * mass[None, :])
    normal = np.zeros((n_bands, len(origins), 3, 3))
    normal[..., 0, 0] = at[:, 0]
    normal[..., 0, 1] = normal[..., 1, 0] = at[:, 1]
    normal[..., 0, 2] = normal[..., 2, 0] = at[:, 2]
    normal[..., 1, 1] = at[:, 3]
    normal[..., 1, 2] = normal[..., 2, 1] = at[:, 4]
    normal[..., 2, 2] = at[:, 5]
    rhs = np.stack([at[:, 6], at[:, 7], at[:, 8]], axis=-1)
    min_active = c.MIN_ACTIVE * (tau_days / 14.0) * density
    keep = np.ones((n_bands, len(origins), 3), dtype=bool)
    keep[..., 1] = at[:, 9] >= min_active
    keep[..., 2] = at[:, 10] >= min_active
    keep &= covered[..., None]
    ridge = np.array([0.0, penalty[0], penalty[1]])
    outer = keep[..., :, None] & keep[..., None, :]
    eye = np.eye(3)
    system = (
        np.where(outer, normal, 0.0)
        + np.einsum("bnp,pq->bnpq", keep * ridge, eye)
        + np.einsum("bnp,pq->bnpq", (~keep).astype(np.float64), eye)
    )
    coef = np.linalg.solve(system, (rhs * keep)[..., None])[..., 0]
    return coef, covered


def ew_mean(
    *, residual: np.ndarray, tau_days: float, origins: np.ndarray, density: float
) -> np.ndarray:
    """Return the exponentially weighted mean of `residual` (…, N_STAMPS) at each origin, shape (…, n).

    The window ends 6 hours before the origin, and the coverage rule is the one of `ew_fit`; an
    uncovered window gives 0.
    """
    decay = float(np.exp(-1.0 / (tau_days * c.PER_DAY)))
    weight = np.isfinite(residual).astype(np.float64)
    value = np.where(weight > 0, residual, 0.0)
    smoothed = _filter(values=np.stack([weight, value], axis=0), decay=decay)
    last = _origin_ends(origins=origins)
    clipped = np.maximum(last, 0)
    total_weight, total = smoothed[0][..., clipped], smoothed[1][..., clipped]
    mass = (1.0 - decay ** (clipped + 1)) / (1.0 - decay)
    covered = (last >= 0) & (total_weight >= c.MIN_COVERAGE * density * mass)
    return np.where(covered, total / np.maximum(total_weight, 1e-12), 0.0)


def box_mean(
    *, residual: np.ndarray, origins: np.ndarray, window_half_hours: int, density: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the box-window mean of `residual` (…, N_STAMPS) and whether each window is covered, shape (…, n).

    The window of an origin ends 6 hours before it. It is covered when at least `c.MIN_COVERAGE` x
    `density` of its stamps hold a residual and it spans at least one day, as in `c.fit_window`.
    """
    finite = np.isfinite(residual)
    count = np.zeros((*residual.shape[:-1], residual.shape[-1] + 1))
    total = np.zeros_like(count)
    np.cumsum(finite, axis=-1, out=count[..., 1:])
    np.cumsum(np.where(finite, residual, 0.0), axis=-1, out=total[..., 1:])
    last = _origin_ends(origins=origins)
    first = np.maximum(last - window_half_hours + 1, 0)
    stop = np.maximum(last + 1, 0)
    span = stop - first
    n_window = count[..., stop] - count[..., first]
    covered = (last >= 0) & (n_window >= c.MIN_COVERAGE * density * span) & (span >= c.PER_DAY)
    mean = (total[..., stop] - total[..., first]) / np.maximum(n_window, 1.0)
    return np.where(covered, mean, 0.0), covered


def daily_means(*, residual: np.ndarray, origins: np.ndarray, density: float) -> np.ndarray:
    """Return the 28 daily mean residuals of each origin's window, shape (…, n, 28); NaN for a thin day.

    The window of an origin ends 6 hours before it, and day `d` covers its stamps `48 d` to
    `48 d + 47` from the window start. A day needs half the half-hours of a complete day (scaled by
    `density`).
    """
    length = N_CP_DAYS * c.PER_DAY
    size = residual.shape[-1]
    first = _origin_ends(origins=origins) - length + 1
    index = first[:, None] + np.arange(length)[None, :]
    block = np.where(index >= 0, residual[..., np.clip(index, 0, size - 1)], np.nan)
    block = block.reshape(*block.shape[:-1], N_CP_DAYS, c.PER_DAY)
    finite = np.isfinite(block)
    seen = finite.sum(axis=-1)
    total = np.where(finite, block, 0.0).sum(axis=-1)
    return np.where(seen >= 0.5 * c.PER_DAY * density, total / np.maximum(seen, 1), np.nan)


class Split(NamedTuple):
    """The best single change point of each row of daily mean residuals."""

    accepted: np.ndarray
    """Whether the split cuts the sum of squares to `CP_IMPROVEMENT` of the unsplit value, (E,)."""
    split: np.ndarray
    """Index of the first day after the change, 0 to 27; -1 when not accepted, (E,)."""
    before: np.ndarray
    after: np.ndarray
    """The means of the days before and after the change (NaN when not accepted), (E,)."""
    slope: np.ndarray
    """The least-squares ramp slope over the window, in points per day, (E,)."""
    fit_ratio: np.ndarray
    """Sum of squares of the best step fit over that of the ramp fit, clipped to 0 to 3, (E,)."""
    share: np.ndarray
    """The share of the step that the first day after the change shows, clipped to 0 to 1, (E,)."""


def split_levels(*, days: np.ndarray) -> Split:
    """Return the best single change point of each row of `days` (E, 28), which may hold NaN."""
    finite = np.isfinite(days)
    value = np.where(finite, days, 0.0)
    size = days.shape[0]
    count = np.zeros((size, N_CP_DAYS + 1))
    total = np.zeros_like(count)
    squares = np.zeros_like(count)
    np.cumsum(finite, axis=1, out=count[:, 1:])
    np.cumsum(value, axis=1, out=total[:, 1:])
    np.cumsum(value**2, axis=1, out=squares[:, 1:])
    n_all, sum_all, square_all = count[:, -1], total[:, -1], squares[:, -1]
    single = square_all - sum_all**2 / np.maximum(n_all, 1)
    candidates = np.arange(MIN_SEGMENT_DAYS, N_CP_DAYS - MIN_SEGMENT_DAYS + 1)
    n_before, sum_before, square_before = (
        count[:, candidates],
        total[:, candidates],
        squares[:, candidates],
    )
    n_after = n_all[:, None] - n_before
    sum_after = sum_all[:, None] - sum_before
    square_after = square_all[:, None] - square_before
    sse = (square_before - sum_before**2 / np.maximum(n_before, 1)) + (
        square_after - sum_after**2 / np.maximum(n_after, 1)
    )
    sse = np.where((n_before >= MIN_SEGMENT_DAYS) & (n_after >= MIN_SEGMENT_DAYS), sse, np.inf)
    rows = np.arange(size)
    best = sse.argmin(axis=1)
    best_sse = sse[rows, best]
    accepted = (
        (n_all >= N_CP_DAYS // 2) & np.isfinite(best_sse) & (best_sse <= CP_IMPROVEMENT * single)
    )
    accepted &= single > 0
    before = sum_before[rows, best] / np.maximum(n_before[rows, best], 1)
    after = sum_after[rows, best] / np.maximum(n_after[rows, best], 1)
    day = np.arange(N_CP_DAYS, dtype=np.float64)[None, :]
    sum_x = (finite * day).sum(axis=1)
    sum_xx = (finite * day**2).sum(axis=1)
    sum_xy = (value * day).sum(axis=1)
    denominator = n_all * sum_xx - sum_x**2
    slope = np.where(
        denominator > 1e-9, (n_all * sum_xy - sum_x * sum_all) / np.maximum(denominator, 1e-9), 0.0
    )
    sse_ramp = single - slope * (sum_xy - sum_x * sum_all / np.maximum(n_all, 1))
    ratio = np.clip(np.where(np.isfinite(best_sse), best_sse, single) / (sse_ramp + 1e-6), 0.0, 3.0)
    first_day = candidates[best]
    first_value = days[rows, first_day]
    second_value = days[rows, np.minimum(first_day + 1, N_CP_DAYS - 1)]
    first_after = np.where(np.isfinite(first_value), first_value, second_value)
    step = after - before
    share = np.where(
        np.abs(step) > 1e-6,
        np.clip((first_after - before) / np.where(np.abs(step) > 1e-6, step, 1.0), 0.0, 1.0),
        0.0,
    )
    share = np.where(np.isfinite(share), share, 0.5)
    return Split(
        accepted=accepted,
        split=np.where(accepted, first_day, -1),
        before=np.where(accepted, before, np.nan),
        after=np.where(accepted, after, np.nan),
        slope=slope,
        fit_ratio=ratio,
        share=np.where(accepted, share, 0.0),
    )


def neighbour_steps(*, daily: np.ndarray, split: Split) -> np.ndarray:
    """Return every series' step at an origin's change point, shape (K, n); 0 where not accepted.

    The step of a series is its mean daily residual over the 7 days from the change point minus its
    mean over the 7 days before it, all days ending 6 hours before the origin.

    Args:
        daily: Every series' daily mean residuals, shape (K, n, 28).
        split: The change point of the entry's own series, over the same n origins.
    """
    day = np.arange(N_CP_DAYS)[None, :]
    at = np.maximum(split.split, 0)[:, None]
    finite = np.isfinite(daily)
    value = np.where(finite, daily, 0.0)

    def window_mean(*, mask: np.ndarray) -> np.ndarray:
        chosen = mask[None, :, :] & finite
        count = chosen.sum(axis=-1)
        return np.where(count > 0, (value * chosen).sum(axis=-1) / np.maximum(count, 1), np.nan)

    step = window_mean(mask=(day >= at) & (day < at + 7)) - window_mean(
        mask=(day >= at - 7) & (day < at)
    )
    return np.where(np.isfinite(step) & split.accepted[None, :], step, 0.0)


# --------------------------------------------------------------------------- worlds


@dataclass(frozen=True, slots=True, eq=False)
class RealWorld:
    """The real residuals and drivers that every world builds on."""

    inputs: c.Inputs
    banded: np.ndarray
    """Band residuals in points, shape (K, 6, N_STAMPS), NaN where absent."""
    drivers: np.ndarray
    """The look-back drivers of arm D, shape (K, N_STAMPS, 2)."""
    penalty: np.ndarray
    """Ridge penalties of the two slopes, shape (K, 2)."""
    noise: np.ndarray
    """The noise unit in points, shape (K, N_ORIGINS)."""
    capacity: np.ndarray
    """Capacity in MW or MVA, shape (K,)."""


def real_noise(*, banded0: np.ndarray) -> np.ndarray:
    """Return the noise unit of each series at each origin, in points, shape (K, N_ORIGINS).

    The noise at origin `o` is the standard deviation of the series' 14-day box means of the
    smallest-lead residual at the origins up to `o - 1` (at most the previous 180), from the real
    world only. With fewer than 10 such means the pooled median over series is used, and with none
    `NOISE_FALLBACK_POINTS`. The unit never falls under `NOISE_FLOOR_POINTS`.
    """
    origins = np.arange(c.N_ORIGINS)
    mean, covered = box_mean(
        residual=banded0, origins=origins, window_half_hours=14 * c.PER_DAY, density=1.0
    )
    values = np.where(covered, mean, np.nan)
    n_series = banded0.shape[0]
    noise = np.full((n_series, c.N_ORIGINS), np.nan)
    for origin in range(c.N_ORIGINS):
        segment = values[:, max(0, origin - NOISE_HISTORY_ORIGINS) : origin]
        count = np.isfinite(segment).sum(axis=1)
        for series in np.flatnonzero(count >= NOISE_MIN_VALUES):
            noise[series, origin] = float(np.nanstd(segment[series]))
    pooled = np.array(
        [
            np.nanmedian(noise[:, origin]) if np.isfinite(noise[:, origin]).any() else np.nan
            for origin in range(c.N_ORIGINS)
        ]
    )
    noise = np.where(np.isfinite(noise), noise, pooled[None, :])
    noise = np.where(np.isfinite(noise), noise, NOISE_FALLBACK_POINTS)
    return np.maximum(noise, NOISE_FLOOR_POINTS)


def build_real_world(*, inputs: c.Inputs) -> RealWorld:
    """Return the real world's band residuals, drivers, penalties, and noise unit."""
    residuals = c.build_residuals(inputs=inputs)
    spec = c.ArmSpec(name="d", drivers=(c.PV_DRIVER, c.COLD_DRIVER))
    drivers = np.stack(
        [
            c.lookback_design(inputs=inputs, series=series, spec=spec)
            for series in range(len(inputs.ids))
        ]
    )
    penalty = np.stack(
        [c.penalties(inputs=inputs, series=series, spec=spec) for series in range(len(inputs.ids))]
    )
    return RealWorld(
        inputs=inputs,
        banded=residuals.banded,
        drivers=drivers,
        penalty=penalty,
        noise=real_noise(banded0=residuals.banded[:, 0, :]),
        capacity=inputs.capacity,
    )


class Neighbourhood(NamedTuple):
    """Every series' window features at the origins of a world, for the neighbour inputs."""

    origins: np.ndarray
    """The origins, shape (n,)."""
    means: np.ndarray
    """Exponentially weighted window mean per series, window, and band, shape (K, 4, 6, n)."""
    daily: np.ndarray
    """Daily mean smallest-lead residual per series, shape (K, n, 28)."""


def neighbourhood(*, banded: np.ndarray, origins: np.ndarray, density: float) -> Neighbourhood:
    """Return every series' window means and daily means at `origins` for the band residuals `banded` (K, 6, N_STAMPS)."""
    means = np.stack(
        [
            ew_mean(residual=banded, tau_days=tau, origins=origins, density=density)
            for tau in WINDOW_DAYS
        ],
        axis=1,
    )
    daily = daily_means(residual=banded[:, 0, :], origins=origins, density=density)
    return Neighbourhood(origins=origins, means=means, daily=daily)


def _mean_where(*, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the mean over the finite entries of each row of `values` (n, 28) inside `mask`; NaN if none."""
    chosen = mask & np.isfinite(values)
    count = chosen.sum(axis=1)
    return np.where(
        count > 0, np.where(chosen, values, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan
    )


def block_tables(
    *,
    world: RealWorld,
    series: int,
    residual: np.ndarray,
    origins: np.ndarray,
    around: Neighbourhood,
    adjacencies: np.ndarray,
    density: float,
) -> Table:
    """Return the entry table of one series at `origins` in one world.

    Args:
        world: The real world, for the drivers, penalties, noise unit, and capacities.
        series: The series' position.
        residual: The series' band residuals in this world, shape (6, N_STAMPS).
        origins: The entries' origins, a subset of `around.origins`.
        around: Every series' neighbour features in this world.
        adjacencies: The graphs, shape (2, K, K), real first.
        density: The share of stamps the world holds.

    Returns:
        The table, with `flag_y` set to -1 (the caller sets it where the world knows the flag).
    """
    n = len(origins)
    position = np.searchsorted(around.origins, origins)
    if not np.array_equal(around.origins[position], origins):
        msg = "an entry origin is missing from the neighbourhood"
        raise ValueError(msg)
    gate = np.zeros((n, N_WINDOWS, c.N_BANDS, 3))
    for index, tau in enumerate(WINDOW_DAYS):
        coef, _ = ew_fit(
            residual=residual,
            design=world.drivers[series],
            penalty=world.penalty[series],
            tau_days=tau,
            origins=origins,
            density=density,
        )
        gate[:, index] = np.swapaxes(coef, 0, 1)
    mean14, covered_band = box_mean(
        residual=residual, origins=origins, window_half_hours=14 * c.PER_DAY, density=density
    )
    box = c.FACTOR_LOOK_BACK * mean14.T
    daily = daily_means(residual=residual[0], origins=origins, density=density)
    daily_band = daily_means(residual=residual, origins=origins, density=density)
    split = split_levels(days=daily)
    day = np.arange(N_CP_DAYS)[None, :]
    before_mask = day < np.maximum(split.split, 0)[:, None]
    slow_level = np.stack(
        [_mean_where(values=daily_band[band], mask=before_mask) for band in range(c.N_BANDS)],
        axis=1,
    )
    slow = np.where(
        split.accepted[:, None] & np.isfinite(slow_level), c.FACTOR_LOOK_BACK * slow_level, box
    )
    noise = world.noise[series, origins]
    step = np.where(split.accepted, split.after - split.before, 0.0)
    step_over_noise = np.abs(step) / noise
    own = np.stack(
        [
            np.clip(step_over_noise, 0.0, 10.0) / 10.0,
            np.clip(mean14[0] / noise, -10.0, 10.0) / 10.0,
            np.where(split.accepted, (N_CP_DAYS - 1 - split.split) / N_CP_DAYS, 0.0),
            split.share,
            np.clip(split.slope * 14.0 / noise, -10.0, 10.0) / 10.0,
            split.fit_ratio / 3.0,
        ],
        axis=1,
    )
    rule_zero = (
        split.accepted
        & (split.split >= N_CP_DAYS - RULE_WINDOW_DAYS)
        & (step_over_noise > RULE_NOISE_UNITS)
        & (split.share >= RULE_SHARE)
    )
    steps = neighbour_steps(daily=around.daily[:, position], split=split)
    noise_all = world.noise[:, origins]
    nb_gate = np.zeros((N_GRAPHS, n, N_WINDOWS, c.N_BANDS))
    flag_nb = np.zeros((N_GRAPHS, n, N_FLAG_NB))
    for graph in range(N_GRAPHS):
        mine = np.flatnonzero(adjacencies[graph, series])
        if len(mine) == 0:
            flag_nb[graph, :, 4] = 1.0
            continue
        nb_gate[graph] = np.moveaxis(around.means[mine][..., position].mean(axis=0), -1, 0)
        ratio = world.capacity[mine] / world.capacity[series]
        their = steps[mine]
        scaled = their * ratio[:, None]
        sign = np.sign(step)
        denominator = np.abs(step) + 1e-6
        balance = np.clip((step + scaled.sum(axis=0)) / denominator, -3.0, 3.0)
        opposite = np.clip(-sign[None, :] * scaled / denominator[None, :], 0.0, 3.0).max(axis=0)
        same = ((np.sign(their) == sign[None, :]) & (np.abs(their) > noise_all[mine])).mean(axis=0)
        live = split.accepted
        flag_nb[graph, :, 0] = np.where(live, balance / 3.0, 0.0)
        flag_nb[graph, :, 1] = np.where(live, opposite / 3.0, 0.0)
        flag_nb[graph, :, 2] = np.where(live, same, 0.0)
        flag_nb[graph, :, 3] = len(mine) / 4.0
    return {
        "series": np.full(n, series, dtype=np.int8),
        "origin": origins.astype(np.int16),
        "gate": gate.astype(np.float32),
        "nb_gate": nb_gate.astype(np.float32),
        "box": box.astype(np.float32),
        "slow": slow.astype(np.float32),
        "flag_own": own.astype(np.float32),
        "flag_nb": flag_nb.astype(np.float32),
        "w_rule": np.where(rule_zero, 0.0, 1.0).astype(np.float32),
        "covered": covered_band[0],
        "accepted": split.accepted,
        "step": step.astype(np.float32),
        "flag_y": np.full(n, -1, dtype=np.int8),
    }


# --------------------------------------------------------------------------- the real world and ramps


def build_real_table(*, world: RealWorld, adjacencies: np.ndarray, around: Neighbourhood) -> Table:
    """Return the entries of every (series, origin) of the real world, in series-major order."""
    origins = np.arange(c.N_ORIGINS)
    return concat_tables(
        tables=[
            block_tables(
                world=world,
                series=series,
                residual=world.banded[series],
                origins=origins,
                around=around,
                adjacencies=adjacencies,
                density=1.0,
            )
            for series in range(len(world.inputs.ids))
        ]
    )


def ramp_shift(*, stamp: np.ndarray, start: np.ndarray | int, sign: np.ndarray | int) -> np.ndarray:
    """Return an own-only ramp in points at `stamp`: a linear rise over 60 days from `start`, then held."""
    rise = np.clip((stamp - start) / RAMP_HALF_HOURS, 0.0, 1.0)
    return sign * RAMP_POINTS * rise


def build_ramp_tables(
    *,
    world: RealWorld,
    adjacencies: np.ndarray,
    around: Neighbourhood,
    month: str,
) -> tuple[Table, Table]:
    """Return the training and the test ramp tables of the fold `month`.

    Training worlds draw a series, a sign, and a start stamp that leaves the ramp at least 20 days
    inside the training period; their entries run from 7 days before the start to the embargo. Test
    worlds start 30 days before the fold month, one per series and sign, and their entries are the
    origins of the fold month. Both tables carry `ramp_k`, `ramp_s0`, and `ramp_sign` per entry.
    """
    fold_index = FOLD_MONTHS.index(month)
    rng = np.random.default_rng([c.EXPERIMENT_ID, 3, fold_index])
    limit = embargo_stamp(month=month)
    n_series = len(world.inputs.ids)
    first_stamp = c.FIRST_ORIGIN_DEC * c.PER_DAY
    worlds = [
        (
            int(rng.integers(n_series)),
            int(rng.choice([-1, 1])),
            int(rng.integers(first_stamp, limit - 20 * c.PER_DAY)),
        )
        for _ in range(N_RAMP_TRAIN_WORLDS)
    ]
    train_tables: list[Table] = []
    for series, sign, start in worlds:
        first = max(c.FIRST_ORIGIN_DEC, (start - c.ORIGIN_STAMP) // c.PER_DAY - 7)
        last = (limit - c.ORIGIN_STAMP) // c.PER_DAY
        origins = np.arange(first, last)
        train_tables.append(
            _ramp_block(
                world=world,
                around=around,
                adjacencies=adjacencies,
                series=series,
                sign=sign,
                start=start,
                origins=origins,
            )
        )
    test_start = month_start_stamp(month=month) - RAMP_TEST_LEAD_DAYS * c.PER_DAY
    origins = c.origin_of_month(month=month)
    test_tables = [
        _ramp_block(
            world=world,
            around=around,
            adjacencies=adjacencies,
            series=series,
            sign=sign,
            start=test_start,
            origins=origins,
        )
        for series in range(n_series)
        for sign in (-1, 1)
    ]
    return concat_tables(tables=train_tables), concat_tables(tables=test_tables)


def _ramp_block(
    *,
    world: RealWorld,
    around: Neighbourhood,
    adjacencies: np.ndarray,
    series: int,
    sign: int,
    start: int,
    origins: np.ndarray,
) -> Table:
    """Return the entries of one own-only ramp world: the real residuals plus a ramp in every band."""
    stamps = np.arange(c.N_STAMPS)
    shift = ramp_shift(stamp=stamps, start=start, sign=sign)
    table = block_tables(
        world=world,
        series=series,
        residual=world.banded[series] + shift[None, :],
        origins=origins,
        around=around,
        adjacencies=adjacencies,
        density=1.0,
    )
    n = len(origins)
    table["flag_y"] = np.zeros(n, dtype=np.int8)
    table["ramp_s0"] = np.full(n, start, dtype=np.int32)
    table["ramp_sign"] = np.full(n, sign, dtype=np.int8)
    return table


# --------------------------------------------------------------------------- the bank


class RealBank(NamedTuple):
    """The Candidate's forecasts of the real world, from the bank's own `real_<month>.npz` files."""

    key: np.ndarray
    """Sorted join keys of (Primary, origin index, valid-time index), shape (n,)."""
    median: np.ndarray
    """The Candidate's median in fractions of capacity, aligned with `key`, shape (n,)."""
    residual: np.ndarray
    """Clean power minus the Candidate median in points, per Primary and lead band, (K, 6, N_STAMPS); NaN where absent."""


_KEY_LIMIT: Final[int] = 2**20


def _row_key(*, prim: np.ndarray, oi: np.ndarray, vi: np.ndarray) -> np.ndarray:
    """Return the join key of (Primary, origin index, valid-time index) rows."""
    return (prim.astype(np.int64) << 40) | (oi.astype(np.int64) << 20) | vi.astype(np.int64)


def load_real_bank(*, month: str, n_series: int) -> RealBank:
    """Return the Candidate's real-world rows of the bank months up to `month`, before the holdout.

    Raises:
        ValueError: If an index exceeds the key width or a join key occurs twice.
    """
    keys: list[np.ndarray] = []
    medians: list[np.ndarray] = []
    total = np.zeros(n_series * c.N_BANDS * c.N_STAMPS)
    count = np.zeros_like(total)
    for real_month in (m for m in BANK_MONTHS if m <= month):
        with np.load(HEAD_DATA / f"real_{real_month}.npz") as real:
            keep = real["vi"] < c.LIB_HOLDOUT_INDEX
            prim = real["prim"][keep].astype(np.int64)
            oi = real["oi"][keep].astype(np.int64)
            vi = real["vi"][keep].astype(np.int64)
            median = real["q"][keep][:, MEDIAN_INDEX].astype(np.float64)
            y = real["y"][keep].astype(np.float64)
        if (oi >= _KEY_LIMIT).any() or (vi >= _KEY_LIMIT).any():
            msg = f"real_{real_month}: an index does not fit the join key"
            raise ValueError(msg)
        keys.append(_row_key(prim=prim, oi=oi, vi=vi))
        medians.append(median)
        row_h = vi - oi - BANK_ORIGIN_LAG_HALF_HOURS
        stamp = vi - c.LIB_OFFSET
        ok = (
            (row_h >= 0)
            & (row_h < c.N_H)
            & (stamp >= 0)
            & (stamp < c.N_STAMPS)
            & np.isfinite(y)
            & np.isfinite(median)
        )
        flat = (prim[ok] * c.N_BANDS + c.BAND_OF_H[row_h[ok]]) * c.N_STAMPS + stamp[ok]
        total += np.bincount(flat, weights=(y[ok] - median[ok]) * 100.0, minlength=len(total))
        count += np.bincount(flat, minlength=len(total))
    key = np.concatenate(keys)
    order = np.argsort(key)
    key = key[order]
    if (key[1:] == key[:-1]).any():
        msg = f"{month}: a (Primary, origin, valid time) key occurs twice in the real bank rows"
        raise ValueError(msg)
    residual = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return RealBank(
        key=key,
        median=np.concatenate(medians)[order],
        residual=residual.reshape(n_series, c.N_BANDS, c.N_STAMPS),
    )


def real_median_of(
    *, real: RealBank, prim: np.ndarray, oi: np.ndarray, vi: np.ndarray
) -> np.ndarray:
    """Return the Candidate's real-world median (fraction of capacity) of each (Primary, origin, valid time); NaN if absent."""
    key = _row_key(prim=prim, oi=oi, vi=vi)
    position = np.minimum(np.searchsorted(real.key, key), len(real.key) - 1)
    return np.where(real.key[position] == key, real.median[position], np.nan)


def generator_world(
    *, world: RealWorld, month: str, events: pl.DataFrame, lists: list[list[int]]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return the generator's (own, mirror) shifts, as shares of capacity on the library grid, for each pass of `month`.

    The generator is experiment 650's `build`, fed the cleaned power of stage 1 (the real power
    before the month after `month`, as the bank was built) and the neighbour lists in the order of
    experiment 400. The events it draws are compared with the stored events table.

    Raises:
        ValueError: If a drawn event differs from the stored events table, or a constant event's own
            shift differs from the bank's by more than 1e-4 of capacity.
    """
    import e00650_bank as generator  # ty: ignore[unresolved-import]

    inputs = world.inputs
    month_index = int(
        (
            dt.datetime(int(month[:4]), int(month[5:]), 1, tzinfo=dt.UTC) - c.LIB_GRID_START
        ).total_seconds()
        // 1800
    )
    year_end = int(
        (dt.datetime(2026, 11, 1, tzinfo=dt.UTC) - c.LIB_GRID_START).total_seconds() // 1800
    )
    clean = np.full((len(inputs.ids), year_end), np.nan)
    clean[:, c.LIB_OFFSET : c.LIB_OFFSET + c.N_STAMPS] = inputs.truth / 100.0
    year, number = int(month[:4]), int(month[5:])
    next_start = int(
        (
            dt.datetime(year + (number == 12), number % 12 + 1, 1, tzinfo=dt.UTC) - c.LIB_GRID_START
        ).total_seconds()
        // 1800
    )
    clean[:, next_start:] = np.nan
    neighbours = {series: lists[series] for series in range(len(lists))}
    is_mw = np.array([str(group).startswith("signed-MW") for group in inputs.groups])
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pass_id in sorted(events["pass"].unique().to_list()):
        own, mirror, drawn = generator.build(
            clean=clean,
            month_idx=month_index,
            month_start_idx=month_index,
            r=int(pass_id),
            nb_of=neighbours,
            capacity=inputs.capacity,
            is_mw=is_mw,
            window_end_idx=year_end,
        )
        stored = {
            row["primary"]: row
            for row in events.filter(pl.col("pass") == pass_id).iter_rows(named=True)
        }
        for event in drawn:
            row = stored.get(event.primary)
            if row is None:
                if event.end_idx > event.onset_idx:
                    msg = f"{month} pass {pass_id}: drawn event of primary {event.primary} is not stored"
                    raise ValueError(msg)
                continue
            expected = (
                row["kind"],
                row["onset_idx"],
                row["end_idx"],
                row["ramp_half_hours"],
                row["n_receivers"],
            )
            got = (
                event.kind,
                event.onset_idx,
                event.end_idx,
                event.ramp_half_hours,
                len(event.receivers),
            )
            if expected != got:
                msg = f"{month} pass {pass_id} primary {event.primary}: stored event {expected} != drawn {got}"
                raise ValueError(msg)
        out[int(pass_id)] = (own, mirror)
    return out


def _check_own_shift(
    *,
    world: RealWorld,
    series: int,
    kind: str,
    generated: np.ndarray,
    stored: np.ndarray,
    label: str,
) -> None:
    """Raise if the regenerated own shift of a constant signed-MW event differs from the bank's.

    Proportional and MVA-only events depend on the cleaned power that the generator saw, which
    stage 1 reads from another table, so those differences are printed when above 1 point.

    Raises:
        ValueError: If a constant signed-MW event differs by more than 1e-4 of capacity.
    """
    if len(stored) == 0:
        return
    worst = float(np.abs(generated - stored).max())
    signed = str(world.inputs.groups[series]).startswith("signed-MW")
    if kind == "constant" and signed and worst > 1e-4:
        msg = f"{label}: regenerated own shift differs from the bank's by {worst:.5f} of capacity"
        raise ValueError(msg)
    if worst > 0.01:
        print(
            f"  CHECK {label}: {kind} own shift differs from the bank's by up to {100 * worst:.2f} points"
        )


def _switched_in_window(*, edge: int, last: np.ndarray) -> np.ndarray:
    """Return whether the stamp `edge` lies in the 28 days that end at each stamp of `last`."""
    return (edge <= last) & (edge > last - FLAG_TRANSFER_DAYS * c.PER_DAY)


def build_bank_month(  # noqa: PLR0915
    *,
    world: RealWorld,
    adjacencies: np.ndarray,
    lists: list[list[int]],
    month: str,
) -> tuple[Table, Table]:
    """Return the rows and the entry table of one bank month.

    A row is one (series, origin, row, pass) of the bank: its entry, its series, origin, and row of
    stage 1, its base forecast and clean power and its shift, all in points of capacity, and
    `q_real`, the Candidate's real-world median of the same row (NaN where the real file lacks it).
    See the module docstring for how a bank world's residuals are built.

    Raises:
        ValueError: If the bank has an origin that is not 09:00 UTC or a row outside 3 hours to 14
            days after it.
    """
    events = pl.read_parquet(HEAD_DATA / f"events_{month}.parquet")
    real_bank = load_real_bank(month=month, n_series=len(world.inputs.ids))
    with np.load(HEAD_DATA / f"bank_{month}.npz") as bank:
        keep = bank["vi"] < c.LIB_HOLDOUT_INDEX
        prim = bank["prim"][keep].astype(np.int64)
        oi = bank["oi"][keep].astype(np.int64)
        vi = bank["vi"][keep].astype(np.int64)
        pass_ = bank["pass_"][keep].astype(np.int64)
        y = bank["y"][keep].astype(np.float64)
        shift = bank["shift"][keep].astype(np.float64)
        q = bank["q"][keep].astype(np.float32)
    origin_numerator = oi + BANK_ORIGIN_LAG_HALF_HOURS - c.LIB_OFFSET - c.ORIGIN_STAMP
    origin, remainder = np.divmod(origin_numerator, c.PER_DAY)
    if (remainder != 0).any():
        msg = f"{month}: a bank origin is not at 09:00 UTC"
        raise ValueError(msg)
    row_h = vi - oi - BANK_ORIGIN_LAG_HALF_HOURS
    if (row_h < 0).any() or (row_h >= c.N_H).any():
        msg = f"{month}: a bank row lies outside 3 hours to 14 days after its origin"
        raise ValueError(msg)
    stamp = vi - c.LIB_OFFSET
    in_grid = (stamp >= 0) & (stamp < c.N_STAMPS)
    usable = in_grid & (origin >= 0) & (origin < c.N_ORIGINS)
    residual_points = (y + shift - q[:, MEDIAN_INDEX]) * 100.0
    real_median = real_median_of(real=real_bank, prim=prim, oi=oi, vi=vi)
    gap_sum = np.zeros(2)
    gap_count = np.zeros(2)
    band_of_row = c.BAND_OF_H[np.clip(row_h, 0, c.N_H - 1)]
    drawn = generator_world(world=world, month=month, events=events, lists=lists)
    even = (np.arange(c.N_STAMPS) % 2) == 0
    tables: list[Table] = []
    rows: list[Table] = []
    n_entries = 0
    for pass_id in sorted(drawn):
        own_world, mirror_world = drawn[pass_id]
        shift_world = (
            100.0 * (own_world + mirror_world)[:, c.LIB_OFFSET : c.LIB_OFFSET + c.N_STAMPS]
        )
        neighbour_banded = np.where(
            even[None, None, :], world.banded + shift_world[:, None, :], np.nan
        )
        in_pass = pass_ == pass_id
        origins_world = np.unique(origin[in_pass & usable])
        if len(origins_world) == 0:
            continue
        around = neighbourhood(banded=neighbour_banded, origins=origins_world, density=0.5)
        for series in sorted(set(prim[in_pass].tolist())):
            mine = in_pass & (prim == series)
            event = events.filter((pl.col("pass") == pass_id) & (pl.col("primary") == series))
            if event.height != 1:
                msg = f"{month} pass {pass_id} primary {series}: {event.height} events"
                raise ValueError(msg)
            onset = int(event["onset_idx"][0]) - c.LIB_OFFSET
            end = int(event["end_idx"][0]) - c.LIB_OFFSET
            low = onset - EVENT_PAD_BEFORE
            _check_own_shift(
                world=world,
                series=series,
                kind=str(event["kind"][0]),
                generated=own_world[series, vi[mine & in_grid]],
                stored=shift[mine & in_grid],
                label=f"{month} pass {pass_id} primary {series}",
            )
            flat = band_of_row[mine & in_grid] * c.N_STAMPS + stamp[mine & in_grid]
            weights = residual_points[mine & in_grid]
            total = np.bincount(flat, weights=weights, minlength=c.N_BANDS * c.N_STAMPS)
            count = np.bincount(flat, minlength=c.N_BANDS * c.N_STAMPS)
            own = np.where(count > 0, total / np.maximum(count, 1), np.nan).reshape(
                c.N_BANDS, c.N_STAMPS
            )
            splice = (np.arange(c.N_STAMPS) < low) & even
            own = np.where(splice[None, :] & ~np.isfinite(own), real_bank.residual[series], own)
            before = mine & usable & (stamp < onset)
            bank_median = q[before, MEDIAN_INDEX].astype(np.float64) * 100.0
            nra_median = world.inputs.q_nra[
                prim[before], origin[before], row_h[before], MEDIAN_INDEX
            ].astype(np.float64)
            gap_real = np.abs(bank_median - real_median[before] * 100.0)
            gap_nra = bank_median - nra_median
            gap_sum += (float(np.nansum(gap_real)), float(np.nansum(gap_nra)))
            gap_count += (int(np.isfinite(gap_real).sum()), int(np.isfinite(gap_nra).sum()))
            chosen = mine & usable
            origins = np.unique(origin[chosen])
            if len(origins) == 0:
                continue
            table = block_tables(
                world=world,
                series=series,
                residual=own,
                origins=origins,
                around=around,
                adjacencies=adjacencies,
                density=0.5,
            )
            last = c.ORIGIN_STAMP + c.PER_DAY * origins - c.LAG_6H
            table["flag_y"] = (
                _switched_in_window(edge=onset, last=last)
                | _switched_in_window(edge=end, last=last)
            ).astype(np.int8)
            tables.append(table)
            index = np.flatnonzero(chosen)
            rows.append(
                {
                    "entry": (np.searchsorted(origins, origin[index]) + n_entries).astype(np.int32),
                    "k": np.full(len(index), series, dtype=np.int8),
                    "o": origin[index].astype(np.int16),
                    "h": row_h[index].astype(np.int16),
                    "q": (q[index] * 100.0).astype(np.float32),
                    "q_real": (real_median[index] * 100.0).astype(np.float32),
                    "y": (y[index] * 100.0).astype(np.float32),
                    "shift": (shift[index] * 100.0).astype(np.float32),
                    "stamp": stamp[index].astype(np.int32),
                    "pass_": np.full(len(index), pass_id, dtype=np.int16),
                }
            )
            n_entries += len(origins)
        print(f"  {month} pass {pass_id}: {n_entries:,} entries so far", flush=True)
    print(
        f"  CHECK {month}: on bank rows before the onset, mean |bank q - real q| (should be about 0) "
        f"{gap_sum[0] / max(gap_count[0], 1):.5f} points over {int(gap_count[0]):,} rows; "
        f"mean (Candidate median - NRA median) {gap_sum[1] / max(gap_count[1], 1):.4f} points "
        f"over {int(gap_count[1]):,} rows"
    )
    return concat_tables(tables=rows), concat_tables(tables=tables)


# --------------------------------------------------------------------------- preparing everything


def real_dense_arrays(*, inputs: c.Inputs) -> dict[str, np.ndarray]:
    """Return the dense arrays that every fit reads: base quantiles, target power, target weather."""
    return {
        "q_nra": inputs.q_nra.astype(np.float32),
        "truth": c.target_truth(inputs=inputs).astype(np.float32),
        "weather": inputs.weather.astype(np.float32),
        "capacity": inputs.capacity.astype(np.float64),
        "is_mva": inputs.is_mva,
    }


def check_rule_matches_stage_one(*, inputs: c.Inputs, table: Table) -> None:
    """Raise unless `box` equals the correction of stage 1's stored arm G-lb wherever N-rule's weight is 1.

    The stored `g` predictions are read through `read_predictions` and compared on the median, on
    the rows where the stored forecast exists, the correction is finite, and an apparent-power
    series was not clipped at zero.

    Raises:
        ValueError: If no row can be compared, or the largest difference exceeds
            `RULE_CHECK_TOLERANCE_POINTS`.
    """
    frame = read_predictions(experiment_id=c.EXPERIMENT_ID, name="g", row_set_id=c.ROW_SET_ID)
    stored = c.dense_from_frame(frame=frame, ids=inputs.ids, capacity=inputs.capacity)[
        ..., MEDIAN_INDEX
    ]
    correction = stored - inputs.q_nra[..., MEDIAN_INDEX]
    n_series = len(inputs.ids)
    box = table["box"].reshape(n_series, c.N_ORIGINS, c.N_BANDS)[:, :, c.BAND_OF_H]
    weight = table["w_rule"].reshape(n_series, c.N_ORIGINS)
    comparable = (
        (weight == 1.0)[..., None]
        & np.isfinite(correction)
        & (~inputs.is_mva[:, None, None] | (stored > 0.0))
    )
    if not comparable.any():
        msg = "no row can be used to compare N-rule with stage 1's stored g correction"
        raise ValueError(msg)
    worst = float(np.abs(correction[comparable] - box[comparable]).max())
    print(
        f"CHECK N-rule at weight 1 vs the stored g correction: largest difference {worst:.6f} points "
        f"over {int(comparable.sum()):,} rows"
    )
    if worst > RULE_CHECK_TOLERANCE_POINTS:
        msg = f"N-rule at weight 1 differs from stage 1's stored g by up to {worst:.5f} points"
        raise ValueError(msg)


def prepare_all(*, inputs: c.Inputs, only: str | None = None) -> None:
    """Build and cache every table of stage 2 (a table that exists is kept).

    Args:
        inputs: The loaded inputs of stage 1.
        only: Build only this table name, one of `graphs`, `real`, `ramp`, or a bank month; or None.
    """
    started = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def wanted(name: str) -> bool:
        return only is None or only == name

    graphs_path = CACHE_DIR / "graphs.npz"
    lists = neighbour_lists(inputs=inputs)
    if not graphs_path.exists():
        real = adjacency_of(lists=lists)
        rewired = rewire(adjacency=real, seed=GRAPH_SEED)
        save_arrays(path=graphs_path, arrays={"adjacency": np.stack([real, rewired])})
        print(
            f"GRAPHS real {int(real.sum()) // 2} edges, degrees {real.sum(axis=0).tolist()}; "
            f"rewired {int(rewired.sum()) // 2} edges, {int((real & rewired).sum()) // 2} shared with real"
        )
    adjacencies = load_arrays(path=graphs_path)["adjacency"]
    world = build_real_world(inputs=inputs)
    save_noise = CACHE_DIR / "noise.npz"
    if not save_noise.exists():
        save_arrays(path=save_noise, arrays={"noise": world.noise})
    print(f"REAL WORLD built in {time.time() - started:.0f} s, peak {c.peak_memory_gb():.2f} GB")
    real_path = CACHE_DIR / "real_entries.npz"
    around_real = neighbourhood(banded=world.banded, origins=np.arange(c.N_ORIGINS), density=1.0)
    if wanted("real") and not (CACHE_DIR / "real_dense.npz").exists():
        save_arrays(path=CACHE_DIR / "real_dense.npz", arrays=real_dense_arrays(inputs=inputs))
    if wanted("real") and not real_path.exists():
        table = build_real_table(world=world, adjacencies=adjacencies, around=around_real)
        save_arrays(path=real_path, arrays=table)
        print(
            f"REAL entries {len(table['origin']):,}: change point accepted in "
            f"{float(table['accepted'].mean()):.3f}, N-rule weight 0 in "
            f"{float((table['w_rule'] == 0).mean()):.3f}, covered {float(table['covered'].mean()):.3f}"
        )
    if wanted("real"):
        check_rule_matches_stage_one(inputs=inputs, table=load_arrays(path=real_path))
    if wanted("ramp"):
        for month in FOLD_MONTHS:
            train_path = CACHE_DIR / f"ramp_train_{month}_entries.npz"
            test_path = CACHE_DIR / f"ramp_test_{month}_entries.npz"
            if train_path.exists() and test_path.exists():
                continue
            train, test = build_ramp_tables(
                world=world, adjacencies=adjacencies, around=around_real, month=month
            )
            save_arrays(path=train_path, arrays=train)
            save_arrays(path=test_path, arrays=test)
            print(
                f"RAMP {month}: {len(train['origin']):,} train, {len(test['origin']):,} test entries"
            )
    for month in BANK_MONTHS:
        if not wanted(month):
            continue
        rows_path = CACHE_DIR / f"bank_{month}_rows.npz"
        entries_path = CACHE_DIR / f"bank_{month}_entries.npz"
        if rows_path.exists() and entries_path.exists():
            continue
        rows, entries = build_bank_month(
            world=world, adjacencies=adjacencies, lists=lists, month=month
        )
        save_arrays(path=entries_path, arrays=entries)
        save_arrays(path=rows_path, arrays=rows)
        print(
            f"BANK {month}: {len(rows['k']):,} rows, {len(entries['origin']):,} entries, covered "
            f"{float(entries['covered'].mean()):.3f}, transfer flag {float((entries['flag_y'] == 1).mean()):.3f}, "
            f"peak {c.peak_memory_gb():.2f} GB",
            flush=True,
        )
    print(f"PREPARED in {time.time() - started:.0f} s, peak {c.peak_memory_gb():.2f} GB")

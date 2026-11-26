"""Experiment 732: the neighbour input table, built from stored power files.

D8's three electrical neighbours are D3, D5, and D14. Their cleaned half-hourly power is read with
experiment 722's `load_power` and placed on D8's base calendar, one row per base day and half-hour.
D3 is metered in MVA (a magnitude). D5 and D14 are metered in signed megawatts and are folded with a
reactive floor drawn per segment, so no neighbour channel carries a sign.

This module holds numpy-only functions, so tests use invented arrays. `build_arrays` and
`write_table` are the only functions that read or write files.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00732_paths  # noqa: F401
import numpy as np
import polars as pl

NEIGHBOURS: Final[tuple[str, ...]] = ("D3", "D5", "D14")
"""The neighbours, in the order of their channel groups."""
SIGNED_NEIGHBOURS: Final[frozenset[str]] = frozenset({"D5", "D14"})
"""Neighbours metered in signed megawatts; D3 is a magnitude in MVA."""
UNITS: Final[tuple[str, ...]] = ("D8", *NEIGHBOURS)
"""The four noise units: D8 first, then the neighbours."""
CHANNELS_PER_NEIGHBOUR: Final = 4
N_NEIGHBOUR_CHANNELS: Final = CHANNELS_PER_NEIGHBOUR * len(NEIGHBOURS)
MIN_COVERAGE: Final = 0.90
"""The share of base-day half-hours each neighbour must cover."""
LAGS: Final[tuple[int, ...]] = (-1, 0, 1)
REACTIVE_FLOOR_MAX_MVA: Final = 0.3
STEP_WINDOW: Final = 4
"""Half-hours in the 2 hours before and after a half-hour, for the step statistic."""
NOISE_SCALE: Final = 1.4826
HALF_HOURS: Final = 48
TABLE_NAME: Final = "neighbours/table.parquet"


class NeighbourError(AssertionError):
    """Raised when the neighbour table fails a gate."""


@dataclass(frozen=True)
class NoiseUnits:
    """The four noise units, in points, computed from training base days alone.

    Attributes:
        values: Noise unit by unit name (`UNITS`).
    """

    values: Mapping[str, float]

    def of(self, *, unit: str) -> float:
        """Return the noise unit of one series."""
        return float(self.values[unit])


def place_on_calendar(
    *, grid: np.ndarray, dates: np.ndarray, first_day: np.datetime64
) -> np.ndarray:
    """Place a master-grid series on `(n_days, 48)` rows of the given base days.

    Args:
        grid: Values on the master grid, one per half-hour from midnight of `first_day`, NaN where
            absent.
        dates: The base days, `datetime64[D]`, `(n,)`.
        first_day: The day at which the master grid starts.

    Returns:
        `(n, 48)` values, NaN where the grid does not reach.
    """
    offset = (dates.astype("datetime64[D]") - first_day.astype("datetime64[D]")).astype(np.int64)
    idx = offset[:, None] * HALF_HOURS + np.arange(HALF_HOURS)[None, :]
    ok = (idx >= 0) & (idx < len(grid))
    out = np.full(idx.shape, np.nan)
    out[ok] = grid[idx[ok]]
    return out


def coverage(*, table: Mapping[str, np.ndarray], reading: np.ndarray) -> dict[str, float]:
    """Return each neighbour's share of D8's present half-hours that it also covers."""
    present = np.isfinite(reading)
    if not present.any():
        raise NeighbourError("D8 has no present half-hour, so coverage is undefined")
    return {name: float(np.isfinite(table[name])[present].mean()) for name in NEIGHBOURS}


def coverage_problems(*, shares: Mapping[str, float]) -> list[str]:
    """Return one line per neighbour whose coverage is below `MIN_COVERAGE`."""
    return [
        f"coverage of {name} is {share:.3f}, below {MIN_COVERAGE}"
        for name, share in shares.items()
        if not share >= MIN_COVERAGE
    ]


def _changes(*, values: np.ndarray) -> np.ndarray:
    """Return half-hour changes within each day (NaN at 00:00 and where either end is missing)."""
    out = np.full(values.shape, np.nan)
    out[:, 1:] = values[:, 1:] - values[:, :-1]
    return out


def lag_correlations(*, reading: np.ndarray, neighbour: np.ndarray) -> dict[int, float]:
    """Return the correlation of D8's and a neighbour's half-hour changes at each lag.

    A lag `k` pairs D8's change at half-hour `t` with the neighbour's change at `t + k`, inside one
    day. NaN where either change is missing is dropped.

    Args:
        reading: D8's reading, `(n, 48)`.
        neighbour: The neighbour's series on the same calendar, `(n, 48)`.

    Returns:
        The Pearson correlation by lag in `LAGS`.
    """
    a, b = _changes(values=reading), _changes(values=neighbour)
    out: dict[int, float] = {}
    for lag in LAGS:
        lo, hi = max(0, -lag), HALF_HOURS - max(0, lag)
        x = a[:, lo:hi].ravel()
        y = b[:, lo + lag : hi + lag].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            raise NeighbourError(f"too few or constant changes to correlate at lag {lag}")
        out[lag] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return out


def alignment_problems(*, correlations: Mapping[str, Mapping[int, float]]) -> list[str]:
    """Return one line per neighbour whose correlation at lag 0 is not the largest of the three.

    D8 and a neighbour mirror a transfer with opposite signs, so the strongest relation is the
    largest in absolute value.
    """
    problems: list[str] = []
    for name, by_lag in correlations.items():
        strength = {lag: abs(value) for lag, value in by_lag.items()}
        if strength[0] < max(strength[lag] for lag in LAGS if lag != 0):
            problems.append(f"{name}: correlation at lag 0 is not the largest of lags {LAGS}")
    return problems


def fold(*, signed: np.ndarray, floor: float | np.ndarray) -> np.ndarray:
    """Fold a signed series to a magnitude with a reactive floor: sqrt(x^2 + q^2).

    Args:
        signed: Signed power in points, any shape.
        floor: The reactive term q in points, broadcast against `signed`.

    Returns:
        The magnitude, NaN where `signed` is NaN.
    """
    return np.sqrt(np.square(signed) + np.square(floor))


def draw_floor(*, rng: np.random.Generator, points_per_mva: float) -> float:
    """Draw one segment's reactive floor in points, uniform from 0 to 0.3 MVA."""
    return float(rng.uniform(0.0, REACTIVE_FLOOR_MAX_MVA) * points_per_mva)


def training_median(*, folded: np.ndarray, train_rows: np.ndarray) -> float:
    """Return the median of a folded neighbour over the training base days (the channel scale)."""
    values = folded[train_rows]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise NeighbourError("a neighbour has no finite training reading")
    median = float(np.median(values))
    if not median > 0:
        raise NeighbourError("a neighbour's training median is not positive")
    return median


def daily_shape(*, values: np.ndarray, weekend: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    """Return the mean daily profile per half-hour, weekdays and weekends apart, `(2, 48)`."""
    out = np.zeros((2, HALF_HOURS))
    for k in (0, 1):
        rows = train_rows & (weekend == bool(k))
        sub = values[rows]
        if sub.shape[0] == 0 or not np.isfinite(sub).any(axis=0).all():
            raise NeighbourError("a daily shape has a half-hour with no training value")
        out[k] = np.nanmean(sub, axis=0)
    return out


def step_statistic(
    *, values: np.ndarray, dates: np.ndarray, weekend: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    """Return the mean of the next 2 hours minus the mean of the previous 2 hours, shape removed.

    The daily shape is subtracted first, so the statistic does not respond to the usual morning or
    evening ramp. The windows cross midnight when the next or previous calendar day is a row; across
    a gap in the days, the half-hours near the gap are NaN.

    Args:
        values: A series on the base calendar, `(n, 48)`.
        dates: The base days, sorted, `datetime64[D]`.
        weekend: Whether each day is a weekend or bank holiday, `(n,)`.
        shape: `daily_shape` output.

    Returns:
        `(n, 48)` statistic, NaN where either window lacks data.
    """
    n = values.shape[0]
    flat = (values - shape[weekend.astype(int)]).reshape(-1)
    total = len(flat)
    ok = np.isfinite(flat)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, flat, 0.0))])
    cnt = np.concatenate([[0], np.cumsum(ok)])
    out = np.full(total, np.nan)
    t = np.arange(STEP_WINDOW, total - STEP_WINDOW + 1)
    after = (csum[t + STEP_WINDOW] - csum[t]) / STEP_WINDOW
    before = (csum[t] - csum[t - STEP_WINDOW]) / STEP_WINDOW
    full = (cnt[t + STEP_WINDOW] - cnt[t] == STEP_WINDOW) & (
        cnt[t] - cnt[t - STEP_WINDOW] == STEP_WINDOW
    )
    day_id = np.repeat(
        np.cumsum(
            np.concatenate([[0], np.diff(dates.astype("datetime64[D]").astype(np.int64)) != 1])
        ),
        HALF_HOURS,
    )
    same = day_id[t - STEP_WINDOW] == day_id[t + STEP_WINDOW - 1]
    out[t] = np.where(full & same, after - before, np.nan)
    return out.reshape(n, HALF_HOURS)


def noise_unit(*, statistic: np.ndarray, train_rows: np.ndarray) -> float:
    """Return 1.4826 times the median absolute step statistic over the training base days."""
    values = np.abs(statistic[train_rows])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise NeighbourError("no finite step statistic on the training days")
    unit = NOISE_SCALE * float(np.median(values))
    if not unit > 0:
        raise NeighbourError("a noise unit is not positive")
    return unit


def half_hour_change(*, scaled: np.ndarray) -> np.ndarray:
    """Return `x[t] - x[t - 1]` along the last axis, 0 at the first half-hour, NaN where missing."""
    out = np.zeros(scaled.shape)
    out[..., 1:] = scaled[..., 1:] - scaled[..., :-1]
    return out


def channels(
    *, folded: np.ndarray, median: float, d8_change: np.ndarray, hidden: np.ndarray | None = None
) -> np.ndarray:
    """Return one neighbour's four channels along a new leading axis.

    The channels are the folded reading over the training median, its half-hour change, the change
    multiplied by the sign of D8's half-hour change of reading, and a presence mask. A missing or
    hidden half-hour has value 0 in every channel and 0 in the mask.

    Args:
        folded: The folded reading, `(..., T)`, NaN where missing.
        median: The training median of the neighbour.
        d8_change: D8's half-hour change of reading, `(..., T)`, 0 where D8 is missing.
        hidden: True where D8's reading is hidden, the same shape; the neighbour is hidden there.

    Returns:
        `(4, ..., T)`.
    """
    scaled = folded / median
    present = np.isfinite(scaled)
    if hidden is not None:
        present = present & ~hidden
    value = np.where(present, scaled, 0.0)
    change = np.where(present & np.roll(present, 1, axis=-1), half_hour_change(scaled=value), 0.0)
    change[..., 0] = 0.0
    aligned = change * np.sign(d8_change)
    return np.stack([value, change, aligned, present.astype(float)])


def build_arrays(
    *, dates: np.ndarray, log: object, name: str = "neighbours"
) -> dict[str, np.ndarray]:
    """Read D3, D5, and D14 from the stored power files and place them on D8's base calendar.

    The values are in points, the units of D8's reading.

    Args:
        dates: D8's base days, `datetime64[D]`.
        log: Experiment 722's `ReadLog`.
        name: The read-log name.

    Returns:
        `(n, 48)` arrays by neighbour name, NaN where absent. D5 and D14 stay signed.
    """
    import e00722_common as common722
    import e00722_core as core722
    import e00722_data as data722
    import e00729_core as core729

    topology = common722.load_topology()
    ids = topology.ids_of(labels=NEIGHBOURS)
    if len(ids) != len(NEIGHBOURS):
        raise NeighbourError("a neighbour label has no time series id")
    grids = data722.load_power(ids=[ids[n] for n in NEIGHBOURS], log=log, name=name)  # ty: ignore[invalid-argument-type]
    first = np.datetime64(core722.GRID_FIRST_END.date(), "D")
    scale = core729.POINTS / core729.CAPACITY_MVA
    return {
        n: place_on_calendar(grid=grids[ids[n]], dates=dates, first_day=first) * scale
        for n in NEIGHBOURS
    }


def write_table(*, root: Path, dates: np.ndarray, arrays: Mapping[str, np.ndarray]) -> Path:
    """Write the neighbour table once, one row per base day and half-hour.

    Raises:
        FileExistsError: When the table exists already.
    """
    path = root / TABLE_NAME
    if path.exists():
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(dates)
    frame = pl.DataFrame(
        {
            "day": np.repeat(dates.astype("datetime64[D]").astype(np.int64), HALF_HOURS),
            "half_hour": np.tile(np.arange(HALF_HOURS), n),
            **{name: arrays[name].reshape(-1) for name in NEIGHBOURS},
        }
    )
    frame.write_parquet(path)
    return path


def read_table(*, root: Path, dates: np.ndarray) -> dict[str, np.ndarray]:
    """Read the stored table and check that its days are exactly `dates`."""
    frame = pl.read_parquet(root / TABLE_NAME)
    stored = frame["day"].to_numpy().reshape(-1, HALF_HOURS)[:, 0].astype("datetime64[D]")
    if len(stored) != len(dates) or not np.array_equal(stored, dates.astype("datetime64[D]")):
        raise NeighbourError("the stored neighbour table covers different days than D8's calendar")
    return {
        name: frame[name].fill_null(float("nan")).to_numpy().reshape(-1, HALF_HOURS)
        for name in NEIGHBOURS
    }


def gate_problems(
    *, arrays: Mapping[str, np.ndarray], reading: np.ndarray, names: Sequence[str] = NEIGHBOURS
) -> list[str]:
    """Return every coverage and alignment failure of a neighbour table."""
    problems = coverage_problems(shares=coverage(table=arrays, reading=reading))
    corr = {n: lag_correlations(reading=reading, neighbour=arrays[n]) for n in names}
    return problems + alignment_problems(correlations=corr)

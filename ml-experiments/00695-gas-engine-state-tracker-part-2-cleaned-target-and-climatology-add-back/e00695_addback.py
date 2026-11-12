# ruff: noqa: E501
"""Add the engine back to a forecast of the engine-free series, as a mixture over the engine's output.

The network forecasts the cleaned series, the demand level without the engine, as 11 quantiles. The
metered reading is the level minus the engine's output o, folded into apparent power:
T_o(c) = sqrt((c - o)^2 + q^2) for o > 0 and T_0(c) = c. For a mixture of outputs o_j with weights
w_j, the reading's distribution function is

    G(x) = w_0 F(x) + sum over j >= 1 of w_j [F(o_j + r) - F(o_j - r)],  r = sqrt(max(x^2 - q^2, 0)),

where F is the forecast's distribution function, piecewise linear between the 11 quantiles and
extrapolated linearly in the quantile function to the levels 0.001 and 0.999, with no mass beyond.
The output quantiles are G inverted at the 11 levels. Where the unit count is the tracked state, the
outputs are j u for j = 0 to 3; for the host's true engine, the outputs are bins of 0.25 MW.

The weights of a forecast are a climatology: the weighted mean of the engine-state distributions of
the half-hours in the same cell (local half-hour of the day, weekend flag) over the 84 days that end
24 hours before the native origin, each weighted 0.5^(age / 28 days), with the cell smoothed over
its two neighbouring half-hours. Reading no state younger than 24 hours before the native origin
keeps every state inside the readings that had arrived by the origin.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00695_common import PIPELINE_GRID_START, Q_FLOOR_MVA, QUANTILE_COLUMNS

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
TAIL_LEVELS: Final[tuple[float, float]] = (0.001, 0.999)
CDF_LEVELS: Final[np.ndarray] = np.concatenate([[TAIL_LEVELS[0]], LEVELS, [TAIL_LEVELS[1]]])
X_STEP: Final[float] = 0.01
"""Step (MVA) of the grid on which the mixture distribution function is evaluated."""
WINDOW_DAYS: Final[int] = 84
HALF_LIFE_DAYS: Final[float] = 28.0
MIN_AGE: Final[int] = 48
"""Half-hours: a state younger than 24 hours before the native origin is never read."""
NATIVE_ORIGIN_OFFSET: Final[np.timedelta64] = np.timedelta64(3, "h")
"""The contract origin is 12:00 UTC and the network's native origin is 09:00 UTC."""
NEIGHBOUR_WEIGHTS: Final[tuple[float, float, float]] = (0.25, 0.5, 0.25)
IDENTITY_WEIGHT: Final[float] = 1.0 - 1e-9
"""A row whose weight on "no engine" is this high keeps its forecast untouched."""


def quantile_function_points(*, q: np.ndarray) -> np.ndarray:
    """Return each row's 13 strictly increasing quantile-function points.

    The 11 forecast quantiles are sorted within the row. The two tail points extend the end slopes
    linearly to the levels 0.001 and 0.999. A tiny increasing offset makes the points strictly
    increasing so the inverse of the quantile function exists.
    """
    q = np.sort(np.asarray(q, dtype=np.float64), axis=1)
    low_slope = (q[:, 1] - q[:, 0]) / (LEVELS[1] - LEVELS[0])
    high_slope = (q[:, -1] - q[:, -2]) / (LEVELS[-1] - LEVELS[-2])
    low = q[:, 0] - low_slope * (LEVELS[0] - TAIL_LEVELS[0])
    high = q[:, -1] + high_slope * (TAIL_LEVELS[1] - LEVELS[-1])
    points = np.concatenate([low[:, None], q, high[:, None]], axis=1)
    return points + 1e-9 * np.arange(points.shape[1])[None, :]


def mixture_quantiles(
    *,
    clean_q: np.ndarray,
    outputs: np.ndarray,
    weights: np.ndarray,
    q_floor: float = Q_FLOOR_MVA,
    signed: bool = False,
) -> np.ndarray:
    """Return the 11 quantiles of the reading from the cleaned forecast and the engine's outputs.

    Args:
        clean_q: The forecast of the engine-free series, shape (rows, 11), sorted within each row.
        outputs: The engine's possible outputs (MVA), shape (m,), with `outputs[0] == 0`.
        weights: Each row's weight on each output, shape (rows, m), rows summing to 1.
        q_floor: The reactive-power floor of the fold.
        signed: Use the signed rule max(c - o, 0) with no fold (arm A1n).

    Returns:
        The reading's quantiles at `LEVELS`, shape (rows, 11).
    """
    assert outputs[0] == 0.0, "the first output is no engine"
    out = np.asarray(clean_q, dtype=np.float64).copy()
    points = quantile_function_points(q=clean_q)
    for i in np.nonzero(weights[:, 0] < IDENTITY_WEIGHT)[0]:
        xs, w = points[i], weights[i]
        live = np.nonzero(w > 0.0)[0]
        high = (
            max(xs[-1], float(outputs[live].max()) - xs[0], xs[-1] - float(outputs[live].min()))
            + 1.0
        )
        grid = np.arange(min(xs[0] - 0.5, 0.0), high, X_STEP)

        def cdf(x: np.ndarray, xs: np.ndarray = xs) -> np.ndarray:
            return np.interp(x, xs, CDF_LEVELS, left=0.0, right=1.0)

        g = np.zeros_like(grid)
        for j in live:
            o = float(outputs[j])
            if signed:
                g += w[j] * cdf(grid + o)
            elif o == 0.0:
                g += w[j] * cdf(grid)
            else:
                r = np.sqrt(np.maximum(grid**2 - q_floor**2, 0.0))
                g += w[j] * (cdf(o + r) - cdf(o - r))
        g = g + 1e-12 * np.arange(len(g))
        out[i] = np.interp(LEVELS, g, grid)
    return out


@dataclass(frozen=True)
class Calendar:
    """Cell of each half-hour of the pipeline grid: local half-hour of day and weekend flag."""

    cell: np.ndarray
    n_grid: int


def build_calendar(*, n_grid: int) -> Calendar:
    """Return the cell index (0 to 95: weekday cells 0 to 47, weekend cells 48 to 95) of each half-hour."""
    start = np.datetime64(PIPELINE_GRID_START.replace(tzinfo=None), "us")
    stamps = start + np.arange(n_grid) * np.timedelta64(1800, "s")
    local = pl.Series(stamps).dt.replace_time_zone("UTC").dt.convert_time_zone("Europe/London")
    half_hour = (local.dt.hour() * 2 + local.dt.minute() // 30).to_numpy().astype(int)
    weekend = (local.dt.weekday() >= 6).to_numpy().astype(int)
    return Calendar(cell=half_hour + 48 * weekend, n_grid=n_grid)


def climatology_table(*, mass: np.ndarray, calendar: Calendar, native_origin: int) -> np.ndarray:
    """Return the weights of each cell for one native origin, shape (96, m).

    Args:
        mass: The engine-state distribution of every half-hour of the grid, shape (T, m).
        calendar: The cell of every half-hour.
        native_origin: The grid index of the network's native origin (09:00 UTC).

    Returns:
        The smoothed, recency-weighted mean distribution of each cell over the 84 days that end 24
        hours before the origin.
    """
    last = native_origin - MIN_AGE
    first = max(last - WINDOW_DAYS * 48, 0)
    t = np.arange(first, last + 1)
    age_days = (native_origin - t) / 48.0
    recency = 0.5 ** (age_days / HALF_LIFE_DAYS)
    cells = calendar.cell[t]
    m = mass.shape[1]
    numerator = np.zeros((96, m))
    denominator = np.zeros(96)
    np.add.at(numerator, cells, mass[t] * recency[:, None])
    np.add.at(denominator, cells, recency)
    smoothed_n = np.zeros_like(numerator)
    smoothed_d = np.zeros_like(denominator)
    for block in (0, 48):
        n_block, d_block = numerator[block : block + 48], denominator[block : block + 48]
        for shift, weight in zip((-1, 0, 1), NEIGHBOUR_WEIGHTS, strict=True):
            smoothed_n[block : block + 48] += weight * np.roll(n_block, shift, axis=0)
            smoothed_d[block : block + 48] += weight * np.roll(d_block, shift, axis=0)
    return smoothed_n / np.maximum(smoothed_d, 1e-12)[:, None]


def row_weights(
    *,
    forecast: pl.DataFrame,
    mass: np.ndarray,
    calendar: Calendar,
    variant: str,
) -> np.ndarray:
    """Return each forecast row's weights on the engine's outputs.

    Args:
        forecast: Rows with `power_fcst_init_time` (the contract origin) and `valid_time`.
        mass: The engine-state distribution of every half-hour of the grid, shape (T, m).
        calendar: The cell of every half-hour.
        variant: `climatology` (the trailing climatology of the row's cell), `state` (the state at
            the valid time, a bound that reads the scored reading), or `truth` (the same for a
            known injected engine, a bound).

    Returns:
        The weights, shape (rows, m).
    """
    grid_start = np.datetime64(PIPELINE_GRID_START.replace(tzinfo=None), "us")
    valid = (
        (forecast["valid_time"].dt.replace_time_zone(None).to_numpy() - grid_start)
        // np.timedelta64(30, "m")
    ).astype(int)
    if variant in {"state", "truth"}:
        return mass[valid].astype(np.float64)
    assert variant == "climatology", variant
    origin = forecast["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    native = ((origin - NATIVE_ORIGIN_OFFSET - grid_start) // np.timedelta64(30, "m")).astype(int)
    out = np.empty((len(forecast), mass.shape[1]))
    for o in np.unique(native):
        table = climatology_table(mass=mass, calendar=calendar, native_origin=int(o))
        rows = native == o
        out[rows] = table[calendar.cell[valid[rows]]]
    return out


def add_back(
    *,
    forecast: pl.DataFrame,
    mass: np.ndarray,
    outputs: np.ndarray,
    calendar: Calendar,
    variant: str,
    signed: bool = False,
) -> pl.DataFrame:
    """Return the forecast with its 11 quantile columns replaced by the reading's quantiles.

    Args:
        forecast: Rows with `power_fcst_init_time`, `valid_time`, and the 11 quantile columns (MVA).
        mass: The engine-state distribution of every half-hour of the grid, shape (T, m).
        outputs: The output of each of the m states (MVA), `outputs[0] == 0`.
        calendar: The cell of every half-hour.
        variant: As in `row_weights`.
        signed: As in `mixture_quantiles`.

    Returns:
        The same rows with the quantile columns replaced.
    """
    weights = row_weights(forecast=forecast, mass=mass, calendar=calendar, variant=variant)
    clean_q = forecast.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    reading_q = mixture_quantiles(clean_q=clean_q, outputs=outputs, weights=weights, signed=signed)
    return forecast.with_columns(
        [
            pl.Series(name, reading_q[:, i].astype(np.float32))
            for i, name in enumerate(QUANTILE_COLUMNS)
        ]
    )

"""Level states for experiment 395: the stack's filter and a one-state filter on stored residuals.

Every function works in points of effective capacity (1 point is 1% of the Primary's capacity) and
on the dense half-hourly grid of `e00087_common`. Nothing here fits an XGBoost model.

- `residual_stream` builds a Primary's residual on the grid from a stored forecast: the actual
  minus the median of the shortest-lead forecast that targets each half-hour.
- `stack_filter_correction` runs experiment 377's two-state filter (K1) on that stream.
- `one_state_level` is the steady-state-gain Kalman filter of a random-walk level (K2, K3): its
  update is `level += gain * (reading - level)`, applied to the Primary's readings in order.
  `level_correction` multiplies the newest level by `phi ** gap`, where `gap` is the number of
  half-hours between that reading and the target.
"""

from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_filters as filters  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.signal import lfilter
from studies.dp_calendar import calendar_columns

from studies import dp_assimilation as assimilation

KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
QUANTILES: Final[list[str]] = [
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
]  # fmt: skip
NEWEST_AGE_HOURS: Final[float] = assimilation.DEFAULT_DELAY_HOURS
GAINS: Final[tuple[float, ...]] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
"""Steady-state gains of the grid search."""
DECAYS: Final[tuple[float, ...]] = (1.0, 0.9998, 0.9995, 0.999, 0.998, 0.995, 0.99)
"""Per-half-hour decays of the grid search (0.999 is a half-life of about 14 days)."""
PRIOR: Final[tuple[float, float]] = (0.05, 0.998)
"""The gain and decay used where a Primary has no residuals before the month."""
MIN_TRAIN_CASES: Final[int] = 2_000
FIT_LEAD_HOURS: Final[float] = 240.0
FIT_STRIDE: Final[int] = 7


def grid_ends() -> np.ndarray:
    """Return the grid's valid times as `datetime64[us]`."""
    return common.grid_times().dt.replace_time_zone(None).to_numpy()


def band_of_grid() -> np.ndarray:
    """Return the hour band of every grid row."""
    return np.asarray(calendar_columns(end_times=common.grid_times()).band)


def _grid_size() -> int:
    return common.grid_times().len()


def residual_stream(*, forecast: pl.DataFrame, actual_points: pl.DataFrame) -> np.ndarray:
    """Return one Primary's residual (actual minus median, in points) on the grid.

    Args:
        forecast: The Primary's forecast in points, with the key columns and `q50`.
        actual_points: Its actuals in points: `valid_time`, `actual_points`.

    Returns:
        A grid array, NaN where the half-hour has no forecast or no reading. A half-hour's residual
        comes from the forecast with the shortest lead (the newest origin) that targets it.
    """
    nearest = (
        forecast.sort("valid_time", "power_fcst_init_time", descending=[False, True])
        .group_by("valid_time", maintain_order=True)
        .agg(pl.col("q50").first())
        .join(actual_points, on="valid_time")
        .with_columns(residual=pl.col("actual_points") - pl.col("q50"))
        .filter(pl.col("residual").is_finite())
    )
    out = np.full(_grid_size(), np.nan)
    out[np.asarray(common.grid_index(times=nearest["valid_time"]))] = nearest["residual"].to_numpy()
    return out


def case_positions(*, cases: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return each case's newest usable grid position and its target's grid position."""
    cutoff = assimilation.usable_cutoff_positions(
        end_times=grid_ends(),
        initialisation=cases["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy(),
        delay_hours=NEWEST_AGE_HOURS,
    )
    return cutoff, np.asarray(common.grid_index(times=cases["valid_time"]))


def stack_filter_correction(
    *, residual: np.ndarray, cases: pl.DataFrame, settings: filters.FilterSettings
) -> np.ndarray:
    """Return K1's correction in points: the stack's filter state added to every quantile."""
    run = filters.filter_stream(residual=residual, band=band_of_grid(), settings=settings)
    cutoff, target = case_positions(cases=cases)
    return assimilation.correction_at(
        trace=run.trace, cutoff=cutoff, target=target, band=band_of_grid()[target]
    ).total


def one_state_level(*, residual: np.ndarray, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the grid positions of the readings and the level after each, in points."""
    positions = np.flatnonzero(np.isfinite(residual))
    level = lfilter([gain], [1.0, -(1.0 - gain)], residual[positions])
    return positions, level


def level_correction(
    *,
    positions: np.ndarray,
    level: np.ndarray,
    cutoff: np.ndarray,
    target: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Return the correction in points: the newest level decayed over the gap to the target."""
    found = np.searchsorted(positions, cutoff, side="right") - 1
    known = found >= 0
    out = np.zeros(len(target))
    gap = target[known] - positions[found[known]]
    out[known] = level[found[known]] * decay**gap
    return out


def fit_gain_and_decay(
    *, residuals: list[np.ndarray], training: list[pl.DataFrame]
) -> tuple[float, float, int]:
    """Return the gain and decay that minimise the corrected median's absolute error.

    Args:
        residuals: Per Primary, the residual stream on the grid (only readings before the month
            may be non-NaN).
        training: Per Primary, the cases to score: `power_fcst_init_time`, `valid_time`, `q50`
            and `actual_points`, all with valid times before the month.

    Returns:
        The gain, the decay, and the number of training cases (the prior when under
        `MIN_TRAIN_CASES`).
    """
    prepared = []
    for residual, cases in zip(residuals, training, strict=True):
        cases = cases.filter(pl.col("lead_hours") <= FIT_LEAD_HOURS).gather_every(FIT_STRIDE)  # noqa: PLW2901
        if cases.height:
            cutoff, target = case_positions(cases=cases)
            prepared.append((residual, cutoff, target, cases))
    n_cases = sum(c[3].height for c in prepared)
    if n_cases < MIN_TRAIN_CASES:
        return PRIOR[0], PRIOR[1], n_cases
    best = (np.inf, PRIOR[0], PRIOR[1])
    for gain in GAINS:
        states = [one_state_level(residual=r, gain=gain) for r, *_ in prepared]
        for decay in DECAYS:
            total = 0.0
            for (positions, level), (_, cutoff, target, cases) in zip(
                states, prepared, strict=True
            ):
                correction = level_correction(
                    positions=positions, level=level, cutoff=cutoff, target=target, decay=decay
                )
                error = cases["actual_points"].to_numpy() - cases["q50"].to_numpy() - correction
                total += float(np.abs(error).sum())
            if total < best[0]:
                best = (total, gain, decay)
    return best[1], best[2], n_cases


def widen(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Return the frame with every quantile moved away from the median by `factor`."""
    return frame.with_columns(
        *[
            (pl.col("q50") + (pl.col(q) - pl.col("q50")) * factor).alias(q)
            for q in QUANTILES
            if q != "q50"
        ]
    )

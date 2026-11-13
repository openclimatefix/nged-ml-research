"""Gas A's record by local day, and the two kinds of injected truth for experiment 697's controls.

Gas A is the 44-day record of a gas engine's output (experiment 194's stored `g2_window.parquet`).
Its output and dates are never printed or drawn. The record is cut into whole **local** days after
its end stamps are moved to half-hour starts, and a day is drawn at random, with replacement, for
every local day of the control's period.

This module copies `drive_series` of `e00695_engine.py` and changes it to cut and place local days.
Experiment 695 cut whole UTC days of the end stamps, which puts the same half-hour of the day in
two different local days across a clock change. Here a target half-hour takes the record's output
at the same clock reading, so the injected engine keeps one local-time profile all year. A local
day with 46 or 50 half-hours reads the record's 48 clock readings and skips or repeats the hour
that the clock change skips or repeats.

**Two kinds of truth.** The `null` truth draws days ignoring the weekday, so the injected engine
has no weekday pattern, no season, no link between days, and no link to weather or price except
through the half-hour of the day. The `weekday` truth draws a weekday from the weekdays and a
weekend day from the weekend days, as experiment 695 did.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00694_common import load_record  # ty: ignore[unresolved-import]
from e00697_common import (  # ty: ignore[unresolved-import]
    DAY,
    START_SHIFT,
    UNIT_MW,
    TruthKindType,
    add_local_fields,
    add_segments,
    half_hour_starts,
)

DRAW_SEED: Final[int] = 0
"""The default seed of the day draw."""


@dataclass(frozen=True)
class RecordDays:
    """Gas A's complete local days: output in MW by local half-hour, and weekend flags."""

    output: np.ndarray
    weekend: np.ndarray


def record_frame() -> pl.DataFrame:
    """Return Gas A's record at half-hour starts, with local fields.

    The record's `time` is an end stamp, as the data-processing page states for the cleaned power
    table that experiment 194 read it from. The returned `start` is `time` minus 30 minutes.

    Returns:
        One row per half-hour, sorted by `start`, with `start`, `power` (MW), `units` (experiment
        194's derived count), `file_price` (the stored file's price column, joined by experiment
        194 at the end stamp), the local fields of `add_local_fields`, and `segment`.
    """
    raw = load_record().sort("time")
    frame = raw.select(
        start=pl.col("time") - pl.lit(START_SHIFT),
        power=pl.col("power").cast(pl.Float64),
        units=pl.col("units"),
        file_price=pl.col("price"),
    )
    return add_segments(frame=add_local_fields(frame=frame))


def record_days() -> RecordDays:
    """Return Gas A's complete local days.

    A local day is complete when the record holds all 48 clock readings of the date. The record's
    one gap, and the first and last days, remove the days they touch. Output is clipped at zero.
    """
    frame = record_frame()
    complete = (
        frame.group_by("local_date")
        .agg(n=pl.len(), distinct=pl.col("local_half_hour").n_unique())
        .filter((pl.col("n") == DAY) & (pl.col("distinct") == DAY))["local_date"]
    )
    days = frame.filter(pl.col("local_date").is_in(complete)).sort("local_date", "local_half_hour")
    parts = days.partition_by("local_date", maintain_order=True)
    output = np.stack([np.clip(p["power"].to_numpy(), 0.0, None) for p in parts])
    weekend = np.array([bool(p["weekend"][0]) for p in parts])
    return RecordDays(output=output, weekend=weekend)


def grid_calendar(*, n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the local date number, the local half-hour, and the weekend flag of each grid index.

    Args:
        n_grid: The length of the panel grid.

    Returns:
        Three arrays of length `n_grid`, for the half-hour that ends at each index's end stamp.
    """
    frame = add_local_fields(
        frame=pl.DataFrame({"start": half_hour_starts(index=np.arange(n_grid))})
    )
    return (
        frame["local_date"].dt.epoch("d").to_numpy(),
        frame["local_half_hour"].to_numpy().astype(int),
        frame["weekend"].to_numpy(),
    )


def drive_series(
    *,
    calendar: tuple[np.ndarray, np.ndarray, np.ndarray],
    first: int,
    last: int,
    kind: TruthKindType,
    seed: int,
    days: RecordDays,
) -> np.ndarray:
    """Return the injected output (MW) on the panel grid, drawn by whole local days.

    Args:
        calendar: The output of `grid_calendar`.
        first: The first grid index that carries the engine.
        last: One past the last grid index that carries the engine.
        kind: `null` draws a record day for each local date ignoring the weekday, and `weekday`
            draws within the weekday class.
        seed: The seed of the draw.
        days: Gas A's complete local days.

    Returns:
        An array as long as the grid with the engine's output on `[first, last)` and zero elsewhere.
        The draw takes dates in chronological order, so a seed gives one truth.
    """
    date_number, half_hour, weekend = calendar
    dates, inverse = np.unique(date_number[first:last], return_inverse=True)
    date_is_weekend = np.zeros(len(dates), dtype=bool)
    date_is_weekend[inverse] = weekend[first:last]
    rng = np.random.default_rng(seed)
    if kind == "null":
        choice = rng.integers(len(days.output), size=len(dates))
    else:
        pools = {
            True: np.flatnonzero(days.weekend),
            False: np.flatnonzero(~days.weekend),
        }
        choice = np.empty(len(dates), dtype=int)
        for i, is_weekend in enumerate(date_is_weekend):
            pool = pools[bool(is_weekend)]
            choice[i] = pool[int(rng.integers(len(pool)))]
    g = np.zeros(len(date_number))
    g[first:last] = days.output[choice[inverse], half_hour[first:last]]
    return g


def units_of(*, g: np.ndarray) -> np.ndarray:
    """Return whole units running, from the output in MW, rounded to the nearest 1.97 MW."""
    return np.rint(np.asarray(g) / UNIT_MW).astype(int)

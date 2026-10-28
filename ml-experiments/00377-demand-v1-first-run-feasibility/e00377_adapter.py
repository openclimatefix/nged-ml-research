"""Adapter from stored Primary-demand forecasts to the frames of `studies.eval_protocol`.

The stored forecasts of experiments 87, 374, and 376 carry the quantile columns `q01` to `q99`, the
keys `time_series_id`, `power_fcst_init_time`, and `valid_time`, and extra columns (`arm`, `seed`,
`actual`, `site`, and others). `studies.eval_protocol` wants one row per key. This module builds
the four inputs the scorer needs: the forecast frame, the actuals, the series table, and the
context table of the slices.
"""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths
from studies.eval_protocol.metrics import QUANTILE_COLUMNS
from studies.eval_protocol.slices import KEY_COLUMNS, SERIES_GROUPS

PRIMARY_LABELS: Final[tuple[str, ...]] = tuple(sorted(SERIES_GROUPS, key=lambda s: int(s[1:])))
"""The 16 Primaries."""


def series_table(*, labels: Sequence[str] = PRIMARY_LABELS) -> pl.DataFrame:
    """Return `time_series_id`, `label`, and `capacity_mw` of the named Primaries."""
    table = pl.read_parquet(default_input_paths().series_table)
    return (
        table.filter(pl.col("site").is_in(list(labels)))
        .select(
            "time_series_id",
            label=pl.col("site"),
            capacity_mw=pl.col("effective_capacity_mw").cast(pl.Float64),
        )
        .sort("label")
    )


def forecast_frame(
    *,
    stored: pl.DataFrame,
    series: pl.DataFrame,
    unit: Literal["mw", "points", "fraction"],
    arm: str | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Return one row per key, with `KEY_COLUMNS` and the 11 quantile columns in MW.

    Args:
        stored: A stored forecast.
        series: The series table, which supplies the dtype of `time_series_id` and the capacity.
        unit: The unit of the stored quantiles: `"points"` (1% of effective capacity, as experiments
            374, 376, and 377 store), `"fraction"` (of effective capacity, as experiment 87 stores
            BG2, climatology, and the manual heuristic), or `"mw"` (already in MW).
        arm: The value of the `arm` column to keep, for a file that holds several arms.
        seed: The value of the `seed` column to keep.

    Returns:
        The frame, sorted by key.

    Raises:
        ValueError: If a key is held twice after the filters.
    """
    frame = stored
    if arm is not None:
        frame = frame.filter(pl.col("arm") == arm)
    if "seed" in frame.columns:
        frame = frame.filter(pl.col("seed") == seed)
    frame = frame.select(*KEY_COLUMNS, *QUANTILE_COLUMNS).with_columns(
        pl.col("time_series_id").cast(series.schema["time_series_id"])
    )
    if unit != "mw":
        mw_per_unit = {"points": 0.01, "fraction": 1.0}[unit]
        frame = (
            frame.join(series.select("time_series_id", "capacity_mw"), on="time_series_id")
            .with_columns(pl.col(*QUANTILE_COLUMNS) * pl.col("capacity_mw") * mw_per_unit)
            .drop("capacity_mw")
        )
    if frame.select(*KEY_COLUMNS).is_duplicated().any():
        msg = "the stored forecast holds a key twice after the arm and seed filters"
        raise ValueError(msg)
    return frame.sort(*KEY_COLUMNS)


def actuals_and_context(
    *, cache: FeatureCache, series: pl.DataFrame, first: datetime, last: datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the actuals (MW) and the slice context of every half-hour in `[first, last)`.

    The actual is the reading in points times the series' capacity over 100. The context carries
    the ERA5 temperature, the sun's elevation, and the unit PV shape that the slices read.
    """
    times = cache.times_series()
    window = ((times >= first) & (times < last)).to_numpy()
    pieces = []
    for series_id, label, capacity in series.iter_rows():
        site = cache.sites[label]
        pieces.append(
            pl.DataFrame(
                {
                    "time_series_id": pl.Series(
                        [series_id] * int(window.sum()), dtype=series.schema["time_series_id"]
                    ),
                    "valid_time": times.filter(window),
                    "actual": site.y[window] * capacity / 100.0,
                    "temperature_c": site.temperature_c[window],
                    "elevation_deg": site.elevation_deg[window],
                    "pv_unit_shape": site.pv_unit_shape[window],
                }
            )
        )
    frame = pl.concat(pieces)
    actuals = frame.select("time_series_id", "valid_time", "actual")
    context = frame.select(
        "time_series_id", "valid_time", "temperature_c", "elevation_deg", "pv_unit_shape"
    )
    return actuals, context


def events_from_masks(*, cache: FeatureCache, series: pl.DataFrame) -> pl.DataFrame:
    """Return the padded hand-filtered events as `time_series_id`, `start`, and `end` rows.

    The cache holds each series' hand-filtered mask as `switching_mask`; each run of true values is
    one event, padded as the cache pads it.
    """
    times = cache.times_series()
    rows = []
    for series_id, label, _ in series.iter_rows():
        mask = cache.sites[label].switching_mask.astype(np.int8)
        edges = np.diff(np.concatenate([[0], mask, [0]]))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            rows.append((series_id, times[int(start)], times[int(end) - 1]))
    return pl.DataFrame(
        rows,
        schema={
            "time_series_id": series.schema["time_series_id"],
            "start": pl.Datetime("us", "UTC"),
            "end": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )


def stored_path(*, root: Path, pattern: str) -> Path:
    """Return the single path matching a glob pattern under `root`."""
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        msg = f"{pattern!r} matched {len(matches)} paths under {root}"
        raise ValueError(msg)
    return matches[0]

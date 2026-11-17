# ruff: noqa: E501
"""Experiment 701: the manual heuristic and the climatology for the runs 2026-07-01 to 2026-10-31.

Experiment 87's `build_references` only reads the stored forecasts of the two baselines from the
`power_forecasts` Delta table, and that table ends with the run of 2026-06-30. So this script builds
the two arms for the new runs itself, with the rules of the production package
`baseline_forecasters` (read from its code), and writes them in the layout of experiment 87's arms
(`ref_manual_heuristic.parquet`, `ref_climatology.parquet`; fractions of capacity at 11 levels, the
native 09:00 UTC origin) through experiment 87's own `finalise` and `write_arm`.

- **Manual heuristic.** Each of the 13 power lags (6 weekly lags of 168 to 1008 hours, 7 annual
  lags of 49 to 55 weeks) is one member, in the order of the lag. A lag is dropped from a case when
  it is no longer than the lead time, measured from the native origin (`valid_time` minus
  `power_fcst_init_time`, `lead >= lag` drops the member), or when the observed power at
  `valid_time - lag` is missing. The 11 quantiles of the case's members are taken as experiment 87's
  `stored_quantiles` takes them, then divided by the series' effective capacity. The observed power
  is the cleaned power of `e00087_common.OBSERVED_PATH` (the experiment 607 copy). The production
  forecasts hold the observed power rounded to a coarser grid than the cleaned power (a difference of
  up to 2.4e-4 MW), so the quantiles of the new runs differ from a production run by up to 1e-4 of
  capacity, and no quantile is bit-equal.
- **Climatology.** The production climatology is a lookup of 51 quantiles per series, Europe/London
  calendar month, half-hour of day, and weekday-or-weekend flag, fitted once on power before
  2025-07-01. (A lookup fitted on power from 2024-04-01 to 2025-07-01 comes closest to the stored
  forecasts, but not exactly, so the first day of the fit is not known.) Its stored forecasts for
the 365 stored runs are that lookup read off. The script
  recovers the lookup from those stored forecasts (it asserts that every cell holds one value
  throughout) and reads the new runs' forecasts from it, so the new months use the same fitted
  climatology as experiment 700's, with no refit. Every valid time of the stored runs lies on the
  calendar of one cell, so every cell that the production lookup holds is recovered.

Subcommands, all writing under `<this experiment's private folder>/refs/`:

- `check`: builds 10 stored runs (2026-06-15 to 2026-06-24) into `refs/check` (the climatology
  lookup recovered only from stored runs before 2026-06-01, so the check is not circular), and
  compares the two arms with experiment 87's stored arms.
- `build`: builds the 123 new runs into `refs/arms`.

    export PYTHONPATH=EXPERIMENT_FOLDERS POLARS_MAX_THREADS=4
    uv run --no-sync python ml-experiments/00701-*/e00701_refs.py check|build
"""

import argparse
import dataclasses
import itertools
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_assemble as assemble  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import e00701_assemble as asm
import e00701_common as mine
import numpy as np
import polars as pl
from baseline_forecasters.climatology import (  # ty: ignore[unresolved-import]
    CLIMATOLOGY_QUANTILE_COLUMNS,
    _local_calendar_cell_keys,
)
from ml_core.features.feature_engineer import DEFAULT_LOCAL_TIMEZONE

REFS_DIR: Final[Path] = mine.ROOT / "refs"
CELL_KEYS: Final[list[str]] = [
    "time_series_id",
    "local_month",
    "local_half_hour_of_day",
    "local_is_weekend",
]
LAGS_HOURS: Final[list[int]] = sorted(
    [*range(168, 1009, 168), *(weeks * 168 for weeks in range(49, 56))]
)
"""The 13 power lags of `conf/model/manual_heuristic.yaml`: member `i` is the `i`th shortest."""
RUN_HOURS: Final[int] = 360
"""A run's forecast covers valid times up to its 00 UTC initialisation plus 15 days."""
CHUNK_RUNS: Final[int] = 8
CHECK_RUNS: Final[list[datetime]] = [
    datetime(2026, 6, 15, tzinfo=UTC) + timedelta(days=d) for d in range(10)
]
STORED_LOOKUP_STRIDE_DAYS: Final[int] = 7
"""Every 7th stored run is read to recover the lookup; runs last 15 days, so each valid time is read twice."""
FIRST_STORED_ORIGIN: Final[datetime] = datetime(2025, 6, 16, 9, tzinfo=UTC)
LAST_STORED_ORIGIN: Final[datetime] = datetime(2026, 6, 30, 9, tzinfo=UTC)
STORED_VALID_END: Final[datetime] = datetime(2026, 6, 30, 18, 30, tzinfo=UTC)
"""Experiment 87's stored arms end with the valid time 2026-06-30 18:00, the end of the stored rows."""


def origins_of(*, runs: list[datetime]) -> list[datetime]:
    """Return the native `power_fcst_init_time` of runs (initialisation plus the publication delay)."""
    return [common.origin_of(run=run) for run in runs]


def case_grid(*, ids: list[int], origins: list[datetime]) -> pl.DataFrame:
    """Return one row per series, origin, and valid time of the runs, up to 15 days after the run."""
    steps = pl.DataFrame({"k": pl.Series(np.arange(1, 2 * (RUN_HOURS - 9) + 1), dtype=pl.Int64)})
    grid = (
        pl.DataFrame({"power_fcst_init_time": pl.Series(origins, dtype=pl.Datetime("us", "UTC"))})
        .join(steps, how="cross")
        .with_columns(
            valid_time=pl.col("power_fcst_init_time") + pl.duration(minutes=pl.col("k") * 30)
        )
        .drop("k")
    )
    return pl.DataFrame({"time_series_id": pl.Series(ids, dtype=pl.Int32)}).join(grid, how="cross")


def quantile_table(*, members: pl.DataFrame, capacity: pl.DataFrame) -> pl.DataFrame:
    """Return the 11 quantiles of each case's members, as experiment 87's `stored_quantiles` takes them.

    Args:
        members: `time_series_id`, `power_fcst_init_time`, `valid_time`, `ensemble_member`, and
            `power_fcst` (MW) sorted by the case keys and the member.
        capacity: `time_series_id` and `capacity` (MW).

    Returns:
        The quantiles over capacity, with the `seed` column, as `build_references` hands them to
        `emit`.
    """
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    frame = members.with_columns(pl.col("power_fcst").cast(pl.Float64)).sort(
        *keys, "ensemble_member"
    )
    count = int(frame["ensemble_member"].n_unique())
    first = frame.gather_every(count).select(keys)
    last = frame.slice(count - 1).gather_every(count).select(keys)
    if frame.height % count == 0 and first.equals(last):
        cube = frame["power_fcst"].to_numpy().reshape(-1, count)
        quantiles = np.quantile(cube, common.LEVELS, axis=1).T
        table = first.hstack(pl.DataFrame(quantiles, schema=list(common.LEVEL_COLUMNS)))
    else:
        table = frame.group_by(keys).agg(
            *[
                pl.col("power_fcst").quantile(float(level), "linear").alias(name)
                for level, name in zip(common.LEVELS, common.LEVEL_COLUMNS, strict=True)
            ]
        )
    return (
        table.cast({"time_series_id": pl.Int32})
        .join(capacity, on="time_series_id", how="left", validate="m:1")
        .with_columns(*[pl.col(name) / pl.col("capacity") for name in common.LEVEL_COLUMNS])
        .drop("capacity")
        .with_columns(seed=pl.lit(common.SEED).cast(pl.Int8))
    )


def heuristic_members(*, cases: pl.DataFrame, power: pl.DataFrame) -> pl.DataFrame:
    """Return the manual heuristic's member rows of the cases: one per lag whose analogue is observed.

    A lag no longer than the lead time from the native origin is dropped (`lead >= lag`), as
    `ml_core.features._lags._nullify_leaky_lags` drops it, and so is a lag whose analogue hour has
    no observed power.
    """
    pieces = []
    for member, hours in enumerate(LAGS_HOURS):
        shifted = power.select(
            "time_series_id",
            valid_time=pl.col("valid_time") + pl.duration(hours=hours),
            power_fcst=pl.col("power"),
        )
        pieces.append(
            cases.filter(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
                < hours
            )
            .join(shifted, on=["time_series_id", "valid_time"], how="inner")
            .with_columns(ensemble_member=pl.lit(member, dtype=pl.Int8))
        )
    return pl.concat(pieces).drop_nulls(subset=["power_fcst"])


def stored_climatology_cells(*, origins: list[datetime], ids: list[int]) -> pl.DataFrame:
    """Return the production climatology's lookup, read off its stored forecasts.

    Args:
        origins: The forecast origins whose stored climatology rows are read.
        ids: The time series ids to read.

    Returns:
        One row per series, local calendar cell, and ensemble member (51 quantiles per cell).

    Raises:
        ValueError: If one cell holds two different values for one member in the stored rows.
    """
    parts = []
    for chunk in itertools.batched(origins, CHUNK_RUNS, strict=False):
        stored = (
            pl.scan_delta(str(common.POWER_FORECASTS_DIR))
            .filter(
                (pl.col("experiment_name") == "climatology")
                & (pl.col("fold_id") == assemble.FOLD_ID)
                & pl.col("power_fcst_init_time").is_in(list(chunk))
                & pl.col("time_series_id").is_in(ids)
            )
            .select("time_series_id", "valid_time", "ensemble_member", "power_fcst")
            .with_columns(
                **_local_calendar_cell_keys(pl.col("valid_time"), time_zone=DEFAULT_LOCAL_TIMEZONE)
            )
            .group_by(*CELL_KEYS, "ensemble_member")
            .agg(
                value=pl.col("power_fcst").first(),
                distinct=pl.col("power_fcst").n_unique(),
            )
            .collect(engine="streaming")
        )
        if (stored["distinct"] > 1).any():
            msg = "a stored climatology cell holds two values for one member"
            raise ValueError(msg)
        parts.append(stored.drop("distinct"))
    long = pl.concat(parts).unique(subset=[*CELL_KEYS, "ensemble_member"], keep="first")
    merged = (
        pl.concat(parts)
        .group_by(*CELL_KEYS, "ensemble_member")
        .agg(low=pl.col("value").min(), high=pl.col("value").max())
    )
    if (merged["low"] != merged["high"]).any():
        msg = "a climatology cell has different values in different stored runs"
        raise ValueError(msg)
    return long.pivot(on="ensemble_member", index=CELL_KEYS, values="value").rename(
        {str(m): CLIMATOLOGY_QUANTILE_COLUMNS[m] for m in range(len(CLIMATOLOGY_QUANTILE_COLUMNS))}
    )


def climatology_members(*, cases: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    """Return the climatology's 51 member rows of every case whose calendar cell the lookup holds."""
    keyed = cases.with_columns(
        **_local_calendar_cell_keys(pl.col("valid_time"), time_zone=DEFAULT_LOCAL_TIMEZONE)
    )
    joined = keyed.join(lookup, on=CELL_KEYS, how="inner")
    return (
        joined.unpivot(
            on=list(CLIMATOLOGY_QUANTILE_COLUMNS),
            index=["time_series_id", "power_fcst_init_time", "valid_time"],
            variable_name="quantile_column",
            value_name="power_fcst",
        )
        .with_columns(ensemble_member=pl.col("quantile_column").str.slice(-2).cast(pl.Int8))
        .drop("quantile_column")
    )


def build(*, runs: list[datetime], out: Path, lookup_origin_end: datetime) -> None:
    """Build both arms for the runs into `out` (a folder under `refs/`)."""
    ctx = dataclasses.replace(
        asm.make_context(stored_only=False), origins=origins_of(runs=runs), arms_dir=out
    )
    primaries = ctx.series.filter(pl.col("substation_type") == "Primary")
    ids = [int(i) for i in primaries["time_series_id"].to_list()]
    capacity = pl.DataFrame(
        {
            "time_series_id": pl.Series(ids, dtype=pl.Int32),
            "capacity": [ctx.capacity[i] for i in ids],
        }
    )
    power = pl.read_parquet(
        common.OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"]
    ).filter(pl.col("time_series_id").is_in(ids))
    stride = timedelta(days=STORED_LOOKUP_STRIDE_DAYS)
    lookup_origins = []
    origin = FIRST_STORED_ORIGIN
    while origin <= min(LAST_STORED_ORIGIN, lookup_origin_end):
        lookup_origins.append(origin)
        origin += stride
    started = time.monotonic()
    lookup = stored_climatology_cells(origins=lookup_origins, ids=ids)
    cells = lookup.group_by("time_series_id").len().sort("time_series_id")
    print(
        f"climatology lookup recovered from {len(lookup_origins)} stored runs "
        f"(origins {lookup_origins[0]:%Y-%m-%d} to {lookup_origins[-1]:%Y-%m-%d}): cells per series "
        f"{dict(zip(cells['time_series_id'].to_list(), cells['len'].to_list(), strict=True))} "
        f"of {12 * 48 * 2}, {time.monotonic() - started:.0f} s",
        flush=True,
    )
    for name, make in (
        ("ref_manual_heuristic", lambda cases: heuristic_members(cases=cases, power=power)),
        ("ref_climatology", lambda cases: climatology_members(cases=cases, lookup=lookup)),
    ):
        started = time.monotonic()
        pieces = []
        for chunk in itertools.batched(ctx.origins, CHUNK_RUNS, strict=False):
            members = make(case_grid(ids=ids, origins=list(chunk)))
            pieces.append(quantile_table(members=members, capacity=capacity))
        table = pl.concat(pieces)
        print(
            f"{name}: {table.height:,} cases, {table['power_fcst_init_time'].n_unique()} runs",
            flush=True,
        )
        assemble.emit(ctx=ctx, name=name, table=table, started=started)


def compare(*, out: Path) -> None:
    """Compare the check arms with experiment 87's stored arms: keys, then the quantiles."""
    for name in ("ref_manual_heuristic", "ref_climatology"):
        new = pl.read_parquet(out / f"{name}.parquet").filter(
            pl.col("valid_time") < STORED_VALID_END
        )
        old = (
            pl.scan_parquet(mine.E87_WORK / "arms" / f"{name}.parquet")
            .filter(
                (pl.col("group") == "Primary")
                & pl.col("power_fcst_init_time").is_in(
                    new["power_fcst_init_time"].unique().to_list()
                )
            )
            .collect()
        )
        keys = list(assemble.KEY_COLUMNS)
        joined = new.join(old, on=keys, suffix="_old", how="full", coalesce=True)
        only_new = joined.filter(pl.col("q50_old").is_null()).height
        only_old = joined.filter(pl.col("q50").is_null()).height
        both = joined.drop_nulls(subset=["q50", "q50_old"])
        gap = np.abs(
            both.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
            - both.select([f"{c}_old" for c in common.LEVEL_COLUMNS]).to_numpy().astype(np.float64)
        )
        exact = int((gap == 0).all(axis=1).sum())
        actual = np.nanmax(np.abs(both["actual"].to_numpy() - both["actual_old"].to_numpy()))
        capacity_gap = float(
            np.nanmax(np.abs(both["capacity_mw"].to_numpy() - both["capacity_mw_old"].to_numpy()))
        )
        print(
            f"{name}: new {new.height:,} rows, stored (Primary, same runs) {old.height:,}; rows only in "
            f"new {only_new}, only in stored {only_old}; rows with all 11 quantiles bit-equal "
            f"{exact:,} of {both.height:,}; largest quantile difference {gap.max():.3g}, mean "
            f"{gap.mean():.3g} (fractions of capacity); largest difference of the observed value "
            f"{actual:.3g}; capacity {capacity_gap:.3g}"
        )


def main() -> None:
    """Run one subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "build"))
    args = parser.parse_args()
    if args.command == "check":
        out = REFS_DIR / "check"
        build(runs=CHECK_RUNS, out=out, lookup_origin_end=datetime(2026, 6, 1, 9, tzinfo=UTC))
        compare(out=out)
    else:
        runs = [
            mine.NEW_FIRST_RUN + timedelta(days=d)
            for d in range((mine.NEW_LAST_RUN - mine.NEW_FIRST_RUN).days + 1)
        ]
        build(runs=runs, out=REFS_DIR / "arms", lookup_origin_end=LAST_STORED_ORIGIN)


if __name__ == "__main__":
    main()

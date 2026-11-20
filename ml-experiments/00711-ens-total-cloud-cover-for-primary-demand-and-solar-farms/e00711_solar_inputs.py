"""Build the solar network's inputs, the 51-member cloud summaries, and the observed output.

This is a copy of experiment 666's `e00666_solar_inputs.py` with four changes (plan section 9):

- **The runs start at 2020-10-15 and end at 2026-06-30**, so the training window can reach the
  first back-filled run. The 14 earliest runs are donors only, because the time-shifted placebo of
  a run needs the run issued 14 days earlier; the farm table keeps
  them as donors, and `e00711_solar.daylight_rows` starts the rows at `SOLAR_ROW_FIRST_RUN`
  (2020-10-29).
- **The 51-member `total_cloud_cover_atmosphere` is read with the other members** and gets the
  linear interpolation that temperature gets (cloud is a continuous variable). The table holds
  `cloud_mean` and `cloud_spread` (percent divided by 100), `cloud_clear_share` and
  `cloud_overcast_share` (the shares of members at or below 20% and at or above 80%) for **every
  half-hour with a stamp, daylight or not**, because a placebo donor run is read at the same lead
  and the donor's half-hour need not be a daylight half-hour of the target run.
- **`cloud_missing`** is the run-date rule of `e00711_common.is_missing_run`: it is true for every
  run in the 19-month gap (2024-04-01 to 2025-11-20) and for run 2024-01-03. The cloud columns are
  null on exactly those runs, and the builder asserts so.
- **The row-level origin assertion** `e00711_origin.assert_solar_rows` runs on every farm's rows:
  `init_time + 9 h <= origin`, `init_time` is 00:00 UTC of the origin's date, the origin is 12:00
  UTC, and the valid time is after the origin. The half-hour labelled 12:00 UTC on the run's day is
  left out, because its period ends at the origin, so each run holds 671 half-hours, not 672.

The table reads the NWP Delta table at the Delta version that `e00711_cloud.py pin` recorded. A farm
cell is the nearest H3 cell of the store; no cell index or coordinate is printed or written.

```bash
# environment as in e00711_common.py; PYTHONPATH holds the folders of experiments 707 and 666
OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 systemd-run --user --scope -p MemoryMax=12G \
    -p MemorySwapMax=0 uv run --no-sync python ml-experiments/00711-*/e00711_solar_inputs.py
```
"""

import argparse
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00666_common as c666  # ty: ignore[unresolved-import]
import e00711_common as common
import e00711_origin as origin_rules
import h3
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)
import polars as pl
import pvlib
from studies.power import scan_power
from studies.pv_dataset import pv_sites

N_MEMBERS: Final[int] = common.N_MEMBERS
RUN_FIRST: Final[datetime] = common.CACHE_FIRST_RUN
SOLAR_ROW_FIRST_RUN: Final[datetime] = RUN_FIRST + timedelta(days=14)
"""The first run whose rows are kept: its 14-days-earlier donor is the first back-filled run."""
RUN_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Exclusive end of the 00 UTC runs (the last run is 2026-06-30)."""
GRID_END: Final[datetime] = datetime(2026, 7, 17, tzinfo=UTC)
"""The sun-position grid runs to here, beyond the last valid time of the last run."""
SOLAR_DIR: Final[Path] = common.WORK_DIR / "solar_inputs"
CLOUD_COLUMNS: Final[tuple[str, ...]] = (
    "cloud_mean",
    "cloud_spread",
    "cloud_clear_share",
    "cloud_overcast_share",
)
INPUTS: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_sd",
    "temp_mean",
    "temp_sd",
    "wind_mean",
    "wind_sd",
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
    "pl_mean",
    "pl_sd",
    "cell_mean",
    "cell_sd",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
    "lead_hours",
)
"""The 18 inputs of experiment 659's `PLP` arm."""
EXTRA_VARIABLES: Final[dict[str, str]] = {
    "dew_point_temperature_2m": "dew_point",
    "downward_long_wave_radiation_flux_surface": "longwave",
    "precipitation_surface": "precipitation",
    "pressure_surface": "pressure_surface",
    "temperature_2m": "temp_2m",
}
SOLAR_FIRST_LEAD_HOURS: Final[int] = 12
"""The origin is this many hours after the run."""


def cloud_summary(*, cloud_percent: np.ndarray) -> dict[str, np.ndarray]:
    """Return the four cloud columns of half-hours from their members' cloud cover (0 to 100).

    A half-hour with a NaN member has NaN in all four columns, so a missing run is never read as
    clear (a comparison with NaN is false and would give a share of 0).

    Args:
        cloud_percent: Shape (half-hours, 51).

    Returns:
        The columns of `CLOUD_COLUMNS`, each of shape (half-hours,): the mean and the population
        standard deviation divided by 100, and the member shares at or below 20% and at or above
        80%.
    """
    bad = np.isnan(cloud_percent).any(axis=1)
    with np.errstate(invalid="ignore"):
        out = {
            "cloud_mean": cloud_percent.mean(axis=1) / 100.0,
            "cloud_spread": cloud_percent.std(axis=1) / 100.0,
            "cloud_clear_share": (cloud_percent <= common.CLEAR_MAX_PERCENT).mean(axis=1),
            "cloud_overcast_share": (cloud_percent >= common.OVERCAST_MIN_PERCENT).mean(axis=1),
        }
    return {k: np.where(bad, np.nan, v) for k, v in out.items()}


def nearest_cell(*, latitude: float, longitude: float, cells: list[int]) -> int:
    """Return the store's H3 cell whose centre is nearest the farm."""
    centres = np.array([h3.cell_to_latlng(h3.int_to_str(c)) for c in cells])
    distance = (centres[:, 0] - latitude) ** 2 + (
        (centres[:, 1] - longitude) * np.cos(np.radians(latitude))
    ) ** 2
    return int(cells[int(np.argmin(distance))])


def _farm_scan(*, cell: int, version: int) -> pl.LazyFrame:
    """Return the table at the pinned version, filtered to the farm's cell and the runs."""
    return common.scan_nwp(version=version).filter(
        (pl.col("h3_index") == cell)
        & (pl.col("init_time") >= RUN_FIRST)
        & (pl.col("init_time") < RUN_END)
    )


def read_members(
    *, cell: int, version: int
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the (init, valid) keys and the (stamps, 51) shortwave, temperature, wind, cloud."""
    raw = (
        _farm_scan(cell=cell, version=version)
        .select(
            "init_time",
            "valid_time",
            "ensemble_member",
            ghi="downward_short_wave_radiation_flux_surface",
            temperature_c="temperature_2m",
            speed=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            cloud=common.CLOUD_COLUMN,
        )
        .collect()
        .sort("init_time", "valid_time", "ensemble_member")
    )
    counts = raw.group_by("init_time", "valid_time").len()
    complete = counts.filter(pl.col("len") == N_MEMBERS).select("init_time", "valid_time")
    raw = raw.join(complete, on=["init_time", "valid_time"], how="semi").sort(
        "init_time", "valid_time", "ensemble_member"
    )
    keys = raw.select("init_time", "valid_time").unique(maintain_order=True)
    n = keys.height

    def grid(name: str) -> np.ndarray:
        return raw[name].to_numpy().reshape(n, N_MEMBERS).astype(np.float64)

    return (
        keys.with_row_index("idx"),
        grid("ghi"),
        grid("temperature_c"),
        grid("speed"),
        grid("cloud"),
    )


def read_extras(*, cell: int, version: int) -> pl.DataFrame:
    """Return the ensemble means of relative humidity, longwave, precipitation, and pressure."""
    return (
        _farm_scan(cell=cell, version=version)
        .select(
            "init_time",
            "valid_time",
            "ensemble_member",
            *[pl.col(k).alias(v) for k, v in EXTRA_VARIABLES.items()],
        )
        .group_by("init_time", "valid_time")
        .agg(pl.len().alias("n_members"), pl.exclude("ensemble_member").mean())
        .filter(pl.col("n_members") == N_MEMBERS)
        .collect()
        .with_columns(
            relative_humidity=(
                100.0
                * (
                    (17.625 * pl.col("dew_point") / (243.04 + pl.col("dew_point"))).exp()
                    / (17.625 * pl.col("temp_2m") / (243.04 + pl.col("temp_2m"))).exp()
                )
            ).clip(0.0, 100.0)
        )
        .drop("n_members", "temp_2m", "dew_point")
        .rename({"valid_time": "period_end"})
    )


def sun_grid(*, latitude: float, longitude: float) -> pl.DataFrame:
    """Return zenith, azimuth, and the mean cosine of each half-hour's 3-hour and 6-hour periods."""
    start = RUN_FIRST - timedelta(hours=6)
    stop = GRID_END + timedelta(hours=6)
    times = pl.datetime_range(
        start, stop, interval="30m", time_zone="UTC", eager=True, closed="left"
    ).alias("time")
    index = pd.DatetimeIndex((times - timedelta(minutes=15)).to_numpy(), tz="UTC")
    position = pvlib.location.Location(latitude, longitude).get_solarposition(index)
    zenith = position["apparent_zenith"].to_numpy()
    grid = pl.DataFrame(
        {
            "time": times,
            "zenith_deg": zenith,
            "sun_azimuth_deg": position["azimuth"].to_numpy(),
            "cosine": np.clip(np.cos(np.radians(zenith)), 0.0, None),
        }
    )
    for hours in (3, 6):
        grid = grid.with_columns(
            **{
                f"end{hours}": (pl.col("time") - timedelta(microseconds=1)).dt.truncate(f"{hours}h")
                + timedelta(hours=hours)
            }
        ).with_columns(**{f"mean_cosine{hours}": pl.col("cosine").mean().over(f"end{hours}")})
    return grid


def physics_lite(
    *, ghi: np.ndarray, temp: np.ndarray, wind: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-member physics-lite power (unit capacity) and the Faiman cell temperature."""
    cell = temp + ghi / (c666.U0 + c666.U1 * wind)
    power = np.clip(ghi / 1000.0 * (1.0 + c666.GAMMA * (cell - 25.0)), 0.0, 1.0)
    return power, cell


def day_rows(*, grid: pl.DataFrame, stamp_keys: pl.DataFrame, day: int) -> pl.DataFrame:
    """Return day `day`'s half-hours with their stamp indices (null where a stamp is missing)."""
    low = SOLAR_FIRST_LEAD_HOURS + 24 * (day - 1)
    return (
        grid.with_columns(init_time=(pl.col("time") - timedelta(hours=low)).dt.truncate("1d"))
        .filter((pl.col("init_time") >= RUN_FIRST) & (pl.col("init_time") < RUN_END))
        .with_columns(lead3=(pl.col("end3") - pl.col("init_time")).dt.total_hours())
        .with_columns(fine=pl.col("lead3") <= c666.THREE_HOURLY_UNTIL_HOURS)
        .with_columns(
            period_end=pl.when(pl.col("fine")).then(pl.col("end3")).otherwise(pl.col("end6")),
            mean_cos=pl.when(pl.col("fine"))
            .then(pl.col("mean_cosine3"))
            .otherwise(pl.col("mean_cosine6")),
            step_hours=pl.when(pl.col("fine")).then(3).otherwise(6),
        )
        .with_columns(
            previous_end=pl.col("period_end") - pl.duration(hours=1) * pl.col("step_hours"),
            daylight=pl.col("mean_cos") > 0,
            day=pl.lit(day),
        )
        .join(stamp_keys, on=["init_time", "period_end"], how="left")
        .join(
            stamp_keys.select("init_time", previous_end="period_end", previous_idx="idx"),
            on=["init_time", "previous_end"],
            how="left",
        )
        .sort("init_time", "time")
    )


def interpolation_weights(*, rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the stamp index, previous stamp index, and weight of the previous stamp of each row.

    A temperature, wind, or cloud value is interpolated linearly between the stamp and the previous
    stamp, falling back to the stamp when the previous stamp is missing.
    """
    idx = rows["idx"].to_numpy()
    prev = rows["previous_idx"].fill_null(rows["idx"]).to_numpy()
    step_minutes = rows["step_hours"].to_numpy() * 60.0
    weight = (
        (rows["period_end"] - rows["time"]).dt.total_minutes().to_numpy() + 15.0
    ) / step_minutes
    return idx, prev, weight


def cloud_rows(*, rows: pl.DataFrame, cloud_p: np.ndarray) -> pl.DataFrame:
    """Return the four cloud columns of every half-hour that has a stamp, daylight or not.

    Cloud cover is interpolated linearly between stamps (production's rule for a continuous
    variable) and clipped to 0 to 100 before the summaries are taken.
    """
    stamped = rows.filter(pl.col("idx").is_not_null())
    idx, prev, weight = interpolation_weights(rows=stamped)
    cloud = np.clip(cloud_p[idx] * (1 - weight[:, None]) + cloud_p[prev] * weight[:, None], 0, 100)
    return stamped.select("init_time", "time").with_columns(
        **{k: pl.Series(v) for k, v in cloud_summary(cloud_percent=cloud).items()}
    )


def build_day(
    *,
    rows: pl.DataFrame,
    ghi_p: np.ndarray,
    temp_p: np.ndarray,
    speed_p: np.ndarray,
    cloud_p: np.ndarray,
    extras: pl.DataFrame,
) -> pl.DataFrame:
    """Return the rows with the 18 inputs (stamped daylight half-hours) and the cloud columns."""
    ready = rows.filter(pl.col("daylight") & pl.col("idx").is_not_null())
    idx, prev, weight = interpolation_weights(rows=ready)
    scale = ready["cosine"].to_numpy() / ready["mean_cos"].to_numpy()
    ghi = ghi_p[idx] * scale[:, None]
    temp = temp_p[idx] * (1 - weight[:, None]) + temp_p[prev] * weight[:, None]
    wind = speed_p[idx] * (1 - weight[:, None]) + speed_p[prev] * weight[:, None]
    power, cell = physics_lite(ghi=ghi, temp=temp, wind=wind)
    azimuth = np.radians(ready["sun_azimuth_deg"].to_numpy())
    computed = ready.select("init_time", "time").with_columns(
        ghi_mean=pl.Series(ghi.mean(axis=1)),
        ghi_sd=pl.Series(ghi.std(axis=1)),
        temp_mean=pl.Series(temp.mean(axis=1)),
        temp_sd=pl.Series(temp.std(axis=1)),
        wind_mean=pl.Series(wind.mean(axis=1)),
        wind_sd=pl.Series(wind.std(axis=1)),
        pl_mean=pl.Series(power.mean(axis=1)),
        pl_sd=pl.Series(power.std(axis=1)),
        cell_mean=pl.Series(cell.mean(axis=1)),
        cell_sd=pl.Series(cell.std(axis=1)),
        elevation_deg=90.0 - ready["zenith_deg"],
        azimuth_sin=pl.Series(np.sin(azimuth)),
        azimuth_cos=pl.Series(np.cos(azimuth)),
    )
    computed = computed.join(
        ready.select("init_time", "time", "period_end").join(
            extras, on=["init_time", "period_end"], how="left"
        ),
        on=["init_time", "time"],
        how="left",
    ).drop("period_end")
    return (
        rows.join(computed, on=["init_time", "time"], how="left")
        .join(cloud_rows(rows=rows, cloud_p=cloud_p), on=["init_time", "time"], how="left")
        .with_columns(lead_hours=(pl.col("time") - pl.col("init_time")).dt.total_minutes() / 60.0)
    )


def missing_runs() -> list[datetime]:
    """Return the runs that carry no cloud: the 19-month gap and run 2024-01-03."""
    runs = common.runs_between(first=RUN_FIRST, last=RUN_END - timedelta(days=1))
    return [r for r in runs if common.is_missing_run(run=r)]


def archive_gaps(*, site: str, stamps: pl.DataFrame) -> list[str]:
    """Return the runs from 2020-10-15 to 2026-06-30 that lack a complete stamp up to lead 348 h.

    Run 2024-01-03 is absent upstream and is not reported.
    """
    expected_leads = [
        *range(9, c666.THREE_HOURLY_UNTIL_HOURS + 1, 3),
        *range(150, 349, 6),
    ]
    present = stamps.with_columns(
        lead_h=((pl.col("valid_time") - pl.col("init_time")).dt.total_hours()).cast(pl.Int64)
    )
    have_by_run = {init: set(g["lead_h"].to_list()) for (init,), g in present.group_by("init_time")}
    gaps = []
    for init in common.runs_between(first=RUN_FIRST, last=RUN_END - timedelta(days=1)):
        if init == common.UPSTREAM_GAP_RUN:
            continue
        have = have_by_run.get(init, set())
        missing = [h for h in expected_leads if h not in have]
        if missing:
            gaps.append(
                f"{site} {init:%Y-%m-%d}: {len(missing)} of {len(expected_leads)} stamps without "
                f"51 members (first lead {missing[0]} h, last lead {missing[-1]} h)"
            )
    return gaps


def assert_cloud_missing_pattern(*, table: pl.DataFrame) -> None:
    """Assert that a stamped half-hour's cloud is null exactly on the runs that carry none.

    Args:
        table: Rows with `init_time`, `idx` as `has_stamp`, `cloud_mean`, and `cloud_missing`.

    Raises:
        ValueError: If a run with cloud holds a null cloud summary, or a missing run holds a value.
    """
    stamped = table.filter(pl.col("has_stamp"))
    wrong = stamped.filter(pl.col("cloud_mean").is_null() != pl.col("cloud_missing"))
    if wrong.height:
        runs = wrong["init_time"].unique().sort().head(3).to_list()
        msg = f"{wrong.height} stamped half-hours break the run-date rule, e.g. runs {runs}"
        raise ValueError(msg)


def build_farm(*, site: str, version: int) -> None:
    """Write one farm's file and print the interpolation statistics; raise on archive gaps."""
    common.assert_cloud_in_schema()
    row = pv_sites().filter(pl.col("site") == site).row(0, named=True)
    store_cells = (
        common.scan_nwp(version=version, run=common.CACHE_FIRST_RUN)
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )
    cell = nearest_cell(latitude=row["latitude"], longitude=row["longitude"], cells=store_cells)
    keys, ghi_p, temp_p, speed_p, cloud_p = read_members(cell=cell, version=version)
    gaps = archive_gaps(site=site, stamps=keys)
    if gaps:
        msg = "the archive lacks stamps with 51 members:\n" + "\n".join(gaps[:50])
        raise ValueError(msg)
    extras = read_extras(cell=cell, version=version)
    grid = sun_grid(latitude=row["latitude"], longitude=row["longitude"])
    actual = (
        scan_power()
        .filter(pl.col("time_series_id") == row["time_series_id"])
        .select("time", actual=pl.col("power") / row["effective_capacity_mw"])
        .collect()
    )
    stamp_keys = keys.rename({"valid_time": "period_end"})
    pieces = []
    for day in c666.SOLAR_DAYS:
        rows = day_rows(grid=grid, stamp_keys=stamp_keys, day=day)
        daylight = rows.filter(pl.col("daylight"))
        no_stamp = int(daylight["idx"].is_null().sum())
        with_stamp = daylight.height - no_stamp
        print(
            f"{site} day {day}: {daylight.height:,} daylight half-hours, {no_stamp:,} without a "
            f"stamp ({with_stamp:,} with one)",
            flush=True,
        )
        pieces.append(
            build_day(
                rows=rows,
                ghi_p=ghi_p,
                temp_p=temp_p,
                speed_p=speed_p,
                cloud_p=cloud_p,
                extras=extras,
            )
        )
    table = (
        pl.concat(pieces)
        .join(actual, on="time", how="left")
        .filter(pl.col("time") > pl.col("init_time") + pl.duration(hours=SOLAR_FIRST_LEAD_HOURS))
        .with_columns(
            has_stamp=pl.col("idx").is_not_null(),
            cloud_missing=pl.col("init_time").dt.date().is_in([r.date() for r in missing_runs()]),
            **{c: pl.col(c).fill_nan(None) for c in CLOUD_COLUMNS},
        )
    )
    assert_cloud_missing_pattern(table=table)
    origin = table["init_time"] + timedelta(hours=SOLAR_FIRST_LEAD_HOURS)
    origin_rules.assert_solar_rows(
        init_time=table["init_time"], origin=origin, valid_time=table["time"]
    )
    out_table = table.select(
        series=pl.lit(site),
        time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32),
        init_time="init_time",
        time="time",
        day="day",
        lead_hours="lead_hours",
        daylight="daylight",
        has_stamp="has_stamp",
        **{name: pl.col(name) for name in INPUTS if name != "lead_hours"},
        **{c: pl.col(c) for c in CLOUD_COLUMNS},
        cloud_missing="cloud_missing",
        actual="actual",
    ).sort("init_time", "time")
    SOLAR_DIR.mkdir(parents=True, exist_ok=True)
    target = SOLAR_DIR / f"{site}.parquet"
    out_table.write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)
    print(
        f"{site}: {out_table.height:,} rows "
        f"(runs before {SOLAR_ROW_FIRST_RUN:%Y-%m-%d} are donors); "
        f"origin rule asserted on every row",
        flush=True,
    )


def main() -> None:
    """Build every farm, one after another, at the pinned Delta version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=None)
    args = parser.parse_args()
    version = common.read_pinned_version()
    failing = common.checks_pass_at(version=version)
    if failing:
        msg = f"the cloud checks have not passed at Delta version {version}: {failing}"
        raise SystemExit(msg)
    for site in args.sites or pv_sites()["site"].to_list():
        build_farm(site=site, version=version)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

"""Build the solar network's inputs and observed output for 00 UTC runs 2024-04-01 to 2026-10-31.

Copied from experiments 310 (`e00310_members.py`, `e00310_features.py`), 430 (`e00430_members.py`)
and 659 (`e00659_solar_wind.py`, `e00659_solar_fit.py`'s `farm_band_frame`) and merged into one
pass, with the day bands extended from 1 to 10 days to 1 to 14 days. Day `d` holds the half-hours
whose lead from the 00 UTC run is in `[12 + 24(d-1), 36 + 24(d-1))` hours, so each half-hour of a
run appears once, and a run's 14 days cover 672 half-hours (12:00 on the run's day to 347.5 h).

For each farm the script reads the 51 members' shortwave, temperature, and 10 m wind speed at the
farm's nearest H3 cell, and the ensemble means of relative humidity, longwave, precipitation and
surface pressure at each stamp. The stamps are 3-hourly to lead 144 h and 6-hourly after. Shortwave
is a mean over the period ending at the stamp and is spread over the period's half-hours by the
cosine of the zenith; temperature and wind are interpolated linearly between the stamp and the
previous stamp, falling back to the stamp when the previous stamp is missing. A half-hour is a
daylight half-hour when the mean cosine of the zenith over its 3 h or 6 h period is positive.

Each farm's file holds every half-hour of every run, with `daylight` and the 18 inputs null where
the half-hour is not daylight or a stamp is missing. `actual` is the farm's cleaned power over the
farm's effective capacity from `studies.power.scan_power` (null where unobserved), which equals
experiment 218's `output` to float precision.

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00666-*/e00666_solar_inputs.py
"""

import os
from datetime import timedelta
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00666_common as common
import h3
import numpy as np
import pandas as pd  # noqa: TID251  (pvlib's API takes a pandas index)
import polars as pl
import pvlib
from studies.power import scan_power
from studies.pv_dataset import pv_sites
from studies.sources import REPO_DATA_DIR

NWP_URI: Final[str] = str(REPO_DATA_DIR / "NWP")
N_MEMBERS: Final[int] = 51
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
"""The 18 inputs of experiment 659's `PLP` arm (ten raw, four physics-lite, four sun columns)."""
EXTRA_VARIABLES: Final[dict[str, str]] = {
    "dew_point_temperature_2m": "dew_point",
    "downward_long_wave_radiation_flux_surface": "longwave",
    "precipitation_surface": "precipitation",
    "pressure_surface": "pressure_surface",
    "temperature_2m": "temp_2m",
}


def nearest_cell(*, latitude: float, longitude: float, cells: list[int]) -> int:
    """Return the store's H3 cell whose centre is nearest the farm."""
    centres = np.array([h3.cell_to_latlng(h3.int_to_str(c)) for c in cells])
    distance = (centres[:, 0] - latitude) ** 2 + (
        (centres[:, 1] - longitude) * np.cos(np.radians(latitude))
    ) ** 2
    return int(cells[int(np.argmin(distance))])


def read_members(*, cell: int) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return the (init, valid) keys and the (stamps, 51) shortwave, temperature, and wind speed."""
    raw = (
        pl.scan_delta(NWP_URI)
        .filter(
            (pl.col("h3_index") == cell)
            & (pl.col("init_time") >= common.ENS_START)
            & (pl.col("init_time") < common.SOLAR_INIT_END)
        )
        .select(
            "init_time",
            "valid_time",
            "ensemble_member",
            ghi="downward_short_wave_radiation_flux_surface",
            temperature_c="temperature_2m",
            speed=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
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

    return keys.with_row_index("idx"), grid("ghi"), grid("temperature_c"), grid("speed")


def read_extras(*, cell: int) -> pl.DataFrame:
    """Return the ensemble means of relative humidity, longwave, precipitation, and pressure."""
    return (
        pl.scan_delta(NWP_URI)
        .filter(
            (pl.col("h3_index") == cell)
            & (pl.col("init_time") >= common.ENS_START)
            & (pl.col("init_time") < common.SOLAR_INIT_END)
        )
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
    start = common.ENS_START - timedelta(hours=6)
    stop = common.SOLAR_GRID_END + timedelta(hours=6)
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
    cell = temp + ghi / (common.U0 + common.U1 * wind)
    power = np.clip(ghi / 1000.0 * (1.0 + common.GAMMA * (cell - 25.0)), 0.0, 1.0)
    return power, cell


def day_rows(*, grid: pl.DataFrame, stamp_keys: pl.DataFrame, day: int) -> pl.DataFrame:
    """Return day `day`'s half-hours with their stamp indices (null where a stamp is missing)."""
    low = 12 + 24 * (day - 1)
    return (
        grid.with_columns(init_time=(pl.col("time") - timedelta(hours=low)).dt.truncate("1d"))
        .filter(
            (pl.col("init_time") >= common.ENS_START)
            & (pl.col("init_time") < common.SOLAR_INIT_END)
        )
        .with_columns(lead3=(pl.col("end3") - pl.col("init_time")).dt.total_hours())
        .with_columns(fine=pl.col("lead3") <= common.THREE_HOURLY_UNTIL_HOURS)
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


def build_day(
    *,
    rows: pl.DataFrame,
    ghi_p: np.ndarray,
    temp_p: np.ndarray,
    speed_p: np.ndarray,
    extras: pl.DataFrame,
) -> pl.DataFrame:
    """Return the rows with the 18 inputs filled for daylight half-hours that have a stamp."""
    ready = rows.filter(pl.col("daylight") & pl.col("idx").is_not_null())
    idx = ready["idx"].to_numpy()
    prev = ready["previous_idx"].fill_null(ready["idx"]).to_numpy()
    step_minutes = ready["step_hours"].to_numpy() * 60.0
    weight = (
        (ready["period_end"] - ready["time"]).dt.total_minutes().to_numpy() + 15.0
    ) / step_minutes
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
    return rows.join(computed, on=["init_time", "time"], how="left").with_columns(
        lead_hours=(pl.col("time") - pl.col("init_time")).dt.total_minutes() / 60.0
    )


def archive_gaps(*, site: str, stamps: pl.DataFrame) -> list[str]:
    """Return the runs 2025-11-01 to 2026-10-31 that lack a complete stamp up to lead 348 h."""
    expected_leads = [
        *range(9, common.THREE_HOURLY_UNTIL_HOURS + 1, 3),
        *range(150, 349, 6),
    ]
    expected = pl.DataFrame({"lead_h": expected_leads})
    inits = pl.datetime_range(
        common.FIRST_ORIGIN, common.LAST_RUN, interval="1d", time_zone="UTC", eager=True
    )
    present = stamps.with_columns(
        lead_h=((pl.col("valid_time") - pl.col("init_time")).dt.total_hours()).cast(pl.Int64)
    )
    gaps = []
    for init in inits:
        have = set(present.filter(pl.col("init_time") == init)["lead_h"].to_list())
        missing = [h for h in expected["lead_h"].to_list() if h not in have]
        if missing:
            gaps.append(
                f"{site} {init:%Y-%m-%d}: {len(missing)} of {len(expected_leads)} stamps without "
                f"51 members (first lead {missing[0]} h, last lead {missing[-1]} h)"
            )
    return gaps


def build_farm(site: str) -> None:
    """Write one farm's file and print the interpolation statistics; raise on archive gaps."""
    row = pv_sites().filter(pl.col("site") == site).row(0, named=True)
    store_cells = (
        pl.scan_delta(NWP_URI)
        .filter(pl.col("init_time") == common.ENS_START)
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )
    cell = nearest_cell(latitude=row["latitude"], longitude=row["longitude"], cells=store_cells)
    keys, ghi_p, temp_p, speed_p = read_members(cell=cell)
    gaps = archive_gaps(site=site, stamps=keys)
    if gaps:
        msg = "the archive lacks stamps with 51 members:\n" + "\n".join(gaps)
        raise ValueError(msg)
    extras = read_extras(cell=cell)
    grid = sun_grid(latitude=row["latitude"], longitude=row["longitude"])
    actual = (
        scan_power()
        .filter(pl.col("time_series_id") == row["time_series_id"])
        .select("time", actual=pl.col("power") / row["effective_capacity_mw"])
        .collect()
    )
    stamp_keys = keys.rename({"valid_time": "period_end"})
    pieces = []
    for day in common.SOLAR_DAYS:
        rows = day_rows(grid=grid, stamp_keys=stamp_keys, day=day)
        daylight = rows.filter(pl.col("daylight"))
        no_stamp = daylight["idx"].is_null().sum()
        no_previous = daylight.filter(pl.col("idx").is_not_null())["previous_idx"].is_null().sum()
        with_stamp = daylight.height - no_stamp
        print(
            f"{site} day {day}: {daylight.height:,} daylight half-hours, {no_stamp:,} without a "
            f"stamp, previous stamp missing in {no_previous / with_stamp:.4%} of the "
            f"{with_stamp:,} with one",
            flush=True,
        )
        pieces.append(
            build_day(rows=rows, ghi_p=ghi_p, temp_p=temp_p, speed_p=speed_p, extras=extras)
        )
    table = (
        pl.concat(pieces)
        .join(actual, on="time", how="left")
        .select(
            series=pl.lit(site),
            time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32),
            init_time="init_time",
            time="time",
            day="day",
            lead_hours="lead_hours",
            daylight="daylight",
            **{name: pl.col(name) for name in INPUTS if name != "lead_hours"},
            actual="actual",
        )
        .sort("init_time", "time")
    )
    common.SOLAR_DIR.mkdir(parents=True, exist_ok=True)
    out = common.SOLAR_DIR / f"{site}.parquet"
    table.write_parquet(out.with_suffix(".tmp"))
    out.with_suffix(".tmp").rename(out)
    print(f"{site}: {table.height:,} rows, cell {cell}", flush=True)


def main() -> None:
    """Build every farm, one after another."""
    for site in pv_sites()["site"].to_list():
        build_farm(site)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

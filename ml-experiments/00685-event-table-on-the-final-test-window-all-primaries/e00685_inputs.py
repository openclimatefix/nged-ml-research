"""Extend experiment 103's row table and the weather files to 2026-10-31, for experiment 685.

Writes to this experiment's data folder (`inputs/`), never over an existing file:

- `rows.parquet`: experiment 103's rows plus the half-hours ending 2026-07-01 00:30 to
  2026-10-31 23:30 for every Primary already in the table, with experiment 103's flags (see
  `window_rows`).
- `era5_features.parquet`: columns `time_series_id`, `valid_time`, `temperature_2m`, the only ERA5
  column the feature cache reads. Stored rows plus the extension.
- `cams.parquet`: the stored CAMS demand-point rows plus the 2026-09-11 to 2026-11-01 file.

The script checks that its half-hourly temperature rebuild reproduces the stored temperature where
the two overlap, so the extension follows the stored recipe.

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    uv run --no-sync python ml-experiments/00685-*/e00685_inputs.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.dp_features import CAMS_PATH, ERA5_PATH, PART_1_DIR, ROWS_NAME, _cams_on_half_hours
from studies.power import scan_power
from studies.pv_dataset import nearest_era5_cell
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00685-event-table-on-the-final-test-window-all-primaries"
)
OUT: Final[Path] = EXPERIMENT_DIR / "inputs"
EXTENSION_ERA5: Final[Path] = EXPERIMENT_DIR / "era5_box_extension" / "t2m.parquet"
STORED_ERA5_BOX: Final[Path] = (
    REPO_DATA_DIR / "studies/downloads/reanalysis/ERA5/stage1_box/t2m.parquet"
)
SERIES_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)
EXTENSION_CAMS: Final[Path] = CAMS_PATH.with_name(
    "cams_demand_points_2026-09-11_2026-10-31.parquet"
)
WINDOW_FIRST: Final[datetime] = datetime(2026, 7, 1, 0, 30, tzinfo=UTC)
WINDOW_LAST: Final[datetime] = datetime(2026, 10, 31, 23, 30, tzinfo=UTC)
"""First and last half-hour ending time added: the stored rows end at 2026-07-01 00:00."""
DARK_BELOW_W_M2: Final[float] = 1.0
"""Experiment 103's definition: a half-hour is dark where CAMS irradiance is under 1 W/m2."""


def extended_cams() -> pl.DataFrame:
    """Stored CAMS demand-point rows plus the later file."""
    old = pl.read_parquet(CAMS_PATH, columns=["site", "time", "ghi_w_m2"])
    new = pl.read_parquet(EXTENSION_CAMS, columns=["site", "time", "ghi_w_m2"]).filter(
        pl.col("time") > old["time"].max()
    )
    out = pl.concat([old, new]).sort("site", "time")
    if out.group_by("site", "time").len()["len"].max() != 1:
        msg = "CAMS has a duplicated (site, time) key"
        raise ValueError(msg)
    if out["time"].max() < WINDOW_LAST:
        msg = f"CAMS ends at {out['time'].max()}, before {WINDOW_LAST}"
        raise ValueError(msg)
    return out


def hourly_temperature(*, series: pl.DataFrame, box_path: Path) -> pl.DataFrame:
    """Hourly 2 m temperature in degrees Celsius at each series' nearest stage-1 box cell."""
    box = pl.read_parquet(box_path)
    assigned = nearest_era5_cell(sites=series, era5=box.head(121 * 2)).select(
        "time_series_id",
        pl.col("cell_latitude").alias("latitude"),
        pl.col("cell_longitude").alias("longitude"),
    )
    return (
        assigned.join(box, on=["latitude", "longitude"], how="inner")
        .select("time_series_id", "time", temperature_2m=pl.col("value").cast(pl.Float64) - 273.15)
        .sort("time_series_id", "time")
    )


def half_hourly_temperature(*, hourly: pl.DataFrame) -> pl.DataFrame:
    """Hourly values on :00 and their mean on :30, as the stored ERA5 features hold them."""
    on_hour = hourly.rename({"time": "valid_time"})
    mid = (
        hourly.sort("time_series_id", "time")
        .with_columns(
            next_value=pl.col("temperature_2m").shift(-1).over("time_series_id"),
            next_time=pl.col("time").shift(-1).over("time_series_id"),
        )
        .filter(pl.col("next_time") == pl.col("time") + pl.duration(hours=1))
        .select(
            "time_series_id",
            (pl.col("time") + pl.duration(minutes=30)).alias("valid_time"),
            temperature_2m=(pl.col("temperature_2m") + pl.col("next_value")) / 2,
        )
    )
    return pl.concat([on_hour, mid]).sort("time_series_id", "valid_time")


def extended_era5() -> pl.DataFrame:
    """Stored temperature plus the rebuilt extension, after checking the rebuild on the overlap."""
    series = pl.read_parquet(SERIES_PATH, columns=["time_series_id", "latitude", "longitude"])
    stored = pl.read_parquet(ERA5_PATH, columns=["time_series_id", "valid_time", "temperature_2m"])
    ext = hourly_temperature(series=series, box_path=EXTENSION_ERA5)
    old_box = hourly_temperature(series=series, box_path=STORED_ERA5_BOX)
    rebuilt = half_hourly_temperature(hourly=pl.concat([old_box, ext]))
    check = stored.join(rebuilt, on=["time_series_id", "valid_time"], suffix="_new", how="inner")
    worst = float(
        (check["temperature_2m"].cast(pl.Float64) - check["temperature_2m_new"]).abs().max()  # ty: ignore[invalid-argument-type]
    )
    print(f"rebuild vs stored temperature: {check.height} rows, max abs difference {worst:.2e} C")
    if worst > 1e-3 or check.height < 0.99 * stored.height:
        msg = "the temperature rebuild does not reproduce the stored file"
        raise ValueError(msg)
    new = rebuilt.filter(pl.col("valid_time") > stored["valid_time"].max())
    return pl.concat([stored.cast({"temperature_2m": pl.Float64}), new]).sort(
        "time_series_id", "valid_time"
    )


def window_rows(*, stored: pl.DataFrame, cams: pl.DataFrame) -> pl.DataFrame:
    """Rows of the window for each Primary: cleaned power and experiment 103's flags.

    A half-hour is `present` when `scan_power` (the cleaned table) holds a finite reading. A
    Primary is `in_pool` there when its role is in the pool and the half-hour is present, as in
    experiment 103. There is no early cut, no lockdown, and no hand-filtered switching mask in the
    window, so `default_rows` equals `in_pool`. `dark` is CAMS irradiance under 1 W/m2.
    """
    primaries = (
        stored.filter(pl.col("time_series_id").is_not_null())
        .group_by("time_series_id")
        .agg(pl.col("site").first(), pl.col("role").first())
    )
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(primaries["time_series_id"].to_list()),
            pl.col("time") >= WINDOW_FIRST,
            pl.col("time") <= WINDOW_LAST,
        )
        .select("time_series_id", "time", "power")
        .collect()
    )
    grid = pl.datetime_range(
        WINDOW_FIRST, WINDOW_LAST, interval="30m", time_zone="UTC", eager=True
    ).alias("time")
    pool_roles = ["mw_long", "mw_short", "mva_dark"]
    frame = (
        primaries.join(pl.DataFrame({"time": grid}), how="cross")
        .join(power, on=["time_series_id", "time"], how="left")
        .join(
            _cams_on_half_hours(cams=cams).select("site", "time", "ghi_w_m2"),
            on=["site", "time"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            power=pl.col("power").cast(pl.Float32),
            present=pl.col("power").is_finite().fill_null(value=False),
        )
        .with_columns(
            in_pool=pl.col("role").is_in(pool_roles) & pl.col("present"),
            early_cut=pl.lit(value=False),
            lockdown=pl.lit(value=False),
            switching_mask=pl.lit(value=False),
            dark=(pl.col("ghi_w_m2") < DARK_BELOW_W_M2).fill_null(value=False),
            scarce_kind=pl.lit(None, dtype=pl.String),
            season_year=pl.lit(None, dtype=pl.Int32),
            day_type=pl.lit(None, dtype=pl.String),
        )
        .with_columns(default_rows=pl.col("in_pool"))
    )
    return frame.select(stored.columns).cast(dict(stored.schema))


def main() -> None:
    """Write the three extended inputs, refusing to overwrite any file."""
    OUT.mkdir(parents=True, exist_ok=True)
    cams = extended_cams()
    era5 = extended_era5()
    stored = pl.read_parquet(PART_1_DIR / ROWS_NAME)
    window = window_rows(stored=stored, cams=cams)
    rows = pl.concat([stored, window]).sort("site", "time")
    print(
        "window rows:",
        window.height,
        "present share by Primary:",
        window.group_by("site").agg(pl.col("present").mean()).sort("site").rows(),
    )
    for name, frame in (("rows", rows), ("era5_features", era5), ("cams", cams)):
        target = OUT / f"{name}.parquet"
        if target.exists():
            msg = f"{target} exists; delete it by hand to rebuild"
            raise FileExistsError(msg)
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(name, frame.height, "rows")


if __name__ == "__main__":
    main()

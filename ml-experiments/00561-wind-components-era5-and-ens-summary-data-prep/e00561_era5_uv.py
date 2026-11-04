"""Build the ERA5 wind components and total cloud cover for the demand pretraining table.

The existing table `era5_features.parquet` holds the wind speeds at 10 m and 100 m but not the
components. This script reads the four components that the stage 1 ERA5 box already holds on disk
(`u10`, `v10`, `u100`, `v100`; no download), puts them on the table's half-hourly grid with the
table's own code, and writes `era5_wind_components.parquet` beside it. The existing file is not
touched. Total cloud cover (`tcc`, an instant in the range 0 to 1, where the ensemble's is a
percentage) is stored raw beside the components.

Run with `PYTHONPATH` set to `studies/stage1_pilot` and `DATA_PATH_INTERNAL` set:

    uv run --no-sync python e00561_era5_uv.py
"""

import json
import time
from datetime import UTC, datetime
from typing import Final

import demand_inputs as demand  # ty: ignore[unresolved-import]
import polars as pl
import stage1_inputs as inputs  # ty: ignore[unresolved-import]

COMPONENTS: Final[tuple[str, ...]] = ("u10", "v10", "u100", "v100", "tcc")
OUTPUT_NAMES: Final[dict[str, str]] = {
    "u10": "wind_u_10m",
    "v10": "wind_v_10m",
    "u100": "wind_u_100m",
    "v100": "wind_v_100m",
    "tcc": "total_cloud_cover",
}
EXISTING: Final = inputs.INPUTS_DIR.parent / "demand" / "inputs" / "era5_features.parquet"
OUTPUT: Final = EXISTING.with_name("era5_wind_components.parquet")


def build() -> pl.DataFrame:
    """Return the components for every series at every half-hour of the existing table."""
    series = demand.series_table()
    hourly = demand.era5_hourly(series=series)
    cells = hourly.select("latitude", "longitude").unique()
    cloud = demand._read_box_variable(name="tcc", cells=cells)
    hourly = hourly.join(cloud, on=["latitude", "longitude", "time"], how="inner", validate="m:1")
    pieces = []
    for row in series.iter_rows(named=True):
        one = hourly.filter(pl.col("time_series_id") == row["time_series_id"]).select(
            "time", *COMPONENTS
        )
        grid = inputs.half_hourly(
            hourly=one.sort("time"),
            instants=list(COMPONENTS),
            period=[],
            irradiance=[],
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        pieces.append(
            grid.rename(OUTPUT_NAMES).with_columns(
                time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32)
            )
        )
    return (
        pl.concat(pieces)
        .select("time_series_id", "valid_time", *OUTPUT_NAMES.values())
        .with_columns(pl.col(*OUTPUT_NAMES.values()).cast(pl.Float32))
        .sort("time_series_id", "valid_time")
    )


def validate(*, frame: pl.DataFrame) -> dict[str, object]:
    """Check the new table against the existing one and against physical limits."""
    old = pl.read_parquet(EXISTING)
    keys = ["time_series_id", "valid_time"]
    same_keys = frame.select(keys).equals(old.select(keys).sort(keys))
    joined = frame.join(old, on=keys, how="inner", validate="1:1")
    speeds = {}
    for height in ("10", "100"):
        derived = (pl.col(f"wind_u_{height}m") ** 2 + pl.col(f"wind_v_{height}m") ** 2).sqrt()
        diff = joined.select((derived - pl.col(f"wind_speed_{height}m")).abs().max()).item()
        speeds[f"max_abs_speed_difference_{height}m_m_s"] = float(diff)
    duplicates = frame.height - frame.select(keys).unique().height
    gaps = (
        frame.group_by("time_series_id")
        .agg(
            n=pl.len(),
            span=(
                (pl.col("valid_time").max() - pl.col("valid_time").min()).dt.total_minutes() // 30
            )
            + 1,
        )
        .filter(pl.col("n") != pl.col("span"))
        .height
    )
    stats = {
        c: {
            "min": float(frame[c].min()),  # ty: ignore[invalid-argument-type]
            "max": float(frame[c].max()),  # ty: ignore[invalid-argument-type]
            "mean": float(frame[c].mean()),  # ty: ignore[invalid-argument-type]
            "nulls": int(frame[c].null_count()),
            "nans": int(frame[c].is_nan().sum()),
        }
        for c in OUTPUT_NAMES.values()
    }
    hour_means = (
        frame.with_columns(hour=pl.col("valid_time").dt.hour())
        .group_by("hour")
        .agg(pl.col("wind_u_100m").mean())
        .sort("hour")["wind_u_100m"]
    )
    report = {
        "rows": frame.height,
        "existing_rows": old.height,
        "same_keys_as_existing_table": same_keys,
        "duplicate_keys": duplicates,
        "series_with_gaps": gaps,
        "first_valid_time": str(frame["valid_time"].min()),
        "last_valid_time": str(frame["valid_time"].max()),
        "columns": stats,
        **speeds,
        "hour_of_day_mean_u100_spread_m_s": float(hour_means.max() - hour_means.min()),  # ty: ignore[unsupported-operator],
    }
    if not same_keys or duplicates or gaps:
        msg = f"validation failed: {report}"
        raise ValueError(msg)
    return report


def main() -> None:
    """Build, validate, and write the table."""
    started = time.monotonic()
    frame = build()
    report = validate(frame=frame)
    report["built_at_utc"] = datetime.now(UTC).isoformat()
    report["seconds"] = round(time.monotonic() - started)
    temporary = OUTPUT.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.rename(OUTPUT)
    report["bytes"] = OUTPUT.stat().st_size
    OUTPUT.with_name("era5_wind_components_validation.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Write `wind.parquet` and `solar.parquet` of the one-year backtest and check the contract.

Each file holds, per series, daily origins from 2025-11-01 to 2026-10-31 (365) and the 672
half-hours from 12:00 on the origin day, so `series * 365 * 672` rows. Columns, in order:
`time_series_id` (Int8, the original id), `valid_time`, `nwp_init_time` (the 00 UTC run),
`power_fcst_init_time` (12:00 UTC on the run's day), and the 11 quantiles `q01` ... `q99` as
`Float32`. The quantiles are in MW: the fit's fraction of capacity times the refit's new capacity,
averaged over the 3 seeds quantile by quantile, then sorted within the row. Solar half-hours with
no daylight carry zeros in every quantile. The files hold no observed power.

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    uv run --no-sync python ml-experiments/00666-*/e00666_export.py
"""

import os
from datetime import timedelta

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00666_common as common
import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites, wind_sites

KEY_COLUMNS = ["time_series_id", "valid_time", "nwp_init_time", "power_fcst_init_time"]
N_HALF_HOURS = common.HORIZON_HALF_HOURS


def average_seeds(*, files: list, keys: list[str], n_refits: int) -> pl.DataFrame:
    """Return the seed-averaged, row-sorted quantiles (MW) of a network's fit files."""
    frame = pl.concat([pl.read_parquet(p) for p in sorted(files)])
    seeds = frame.group_by("refit").agg(n=pl.col("seed").n_unique())
    if frame["refit"].n_unique() != n_refits or (seeds["n"] != len(common.SEEDS)).any():
        msg = "a refit lacks a seed or a refit is missing"
        raise ValueError(msg)
    mean = frame.group_by(keys).agg(
        pl.col(common.QUANTILES).mean(), seeds=pl.len(), refits=pl.col("refit").n_unique()
    )
    if (mean["seeds"] != len(common.SEEDS)).any() or (mean["refits"] != 1).any():
        msg = "a case is not predicted by exactly 3 seeds of one refit"
        raise ValueError(msg)
    values = np.sort(mean.select(common.QUANTILES).to_numpy(), axis=1)
    return mean.select(keys).with_columns(
        **{q: pl.Series(values[:, i]) for i, q in enumerate(common.QUANTILES)}
    )


def grid(*, ids: list[int]) -> pl.DataFrame:
    """Return every (series, origin, valid time) key of the contract."""
    runs = pl.datetime_range(
        common.FIRST_ORIGIN, common.LAST_RUN, interval="1d", time_zone="UTC", eager=True
    )
    offsets = pl.Series(
        [timedelta(hours=common.RUN_TO_ORIGIN_HOURS, minutes=30 * k) for k in range(N_HALF_HOURS)]
    )
    return (
        pl.DataFrame({"time_series_id": pl.Series(ids, dtype=pl.Int32)})
        .join(pl.DataFrame({"nwp_init_time": runs}), how="cross")
        .join(pl.DataFrame({"offset": offsets}), how="cross")
        .with_columns(valid_time=pl.col("nwp_init_time") + pl.col("offset"))
        .drop("offset")
    )


def finish(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the contract's columns in order, with the origin at 12:00 UTC."""
    return frame.select(
        pl.col("time_series_id").cast(pl.Int8),
        "valid_time",
        "nwp_init_time",
        power_fcst_init_time=pl.col("nwp_init_time") + timedelta(hours=common.RUN_TO_ORIGIN_HOURS),
        **{q: pl.col(q).cast(pl.Float32) for q in common.QUANTILES},
    ).sort("time_series_id", "nwp_init_time", "valid_time")


def wind_frame() -> pl.DataFrame:
    """Return the wind file's rows."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    mean = average_seeds(
        files=list((common.FIT_DIR / "wind").glob("refit=*_seed=*.parquet")), keys=keys, n_refits=12
    )
    wide = mean.with_columns(
        nwp_init_time=pl.col("power_fcst_init_time")
        - timedelta(hours=common.RUN_TO_PUBLICATION_HOURS)
    ).drop("power_fcst_init_time")
    ids = wind_sites()["time_series_id"].to_list()
    return finish(
        frame=grid(ids=ids).join(
            wide, on=["time_series_id", "nwp_init_time", "valid_time"], how="left"
        )
    )


def solar_frame() -> pl.DataFrame:
    """Return the solar file's rows, with zeros at half-hours without daylight."""
    keys = ["time_series_id", "init_time", "time"]
    mean = average_seeds(
        files=list((common.FIT_DIR / "solar").glob("refit=*_seed=*.parquet")), keys=keys, n_refits=4
    ).rename({"init_time": "nwp_init_time", "time": "valid_time"})
    flags = pl.concat(
        [
            pl.read_parquet(common.SOLAR_DIR / f"{s}.parquet").select(
                "time_series_id",
                nwp_init_time="init_time",
                valid_time="time",
                daylight="daylight",
            )
            for s in pv_sites()["site"].to_list()
        ]
    )
    ids = pv_sites()["time_series_id"].to_list()
    joined = (
        grid(ids=ids)
        .join(flags, on=["time_series_id", "nwp_init_time", "valid_time"], how="left")
        .join(mean, on=["time_series_id", "nwp_init_time", "valid_time"], how="left")
    )
    if joined["daylight"].null_count():
        msg = "a half-hour of the contract has no row in the solar inputs"
        raise ValueError(msg)
    lost = joined.filter(pl.col("daylight") & pl.col("q50").is_null()).height
    if lost:
        msg = f"{lost} daylight half-hours have no forecast"
        raise ValueError(msg)
    return finish(
        frame=joined.with_columns(
            pl.when(pl.col("daylight")).then(pl.col(q)).otherwise(0.0).alias(q)
            for q in common.QUANTILES
        ).drop("daylight")
    )


def check(*, frame: pl.DataFrame, n_series: int, name: str) -> None:
    """Assert the contract's checks and print what was checked."""
    expected = n_series * common.N_ORIGINS * N_HALF_HOURS
    quantiles = frame.select(list(common.QUANTILES)).to_numpy()
    assert frame.columns == [*KEY_COLUMNS, *common.QUANTILES], "columns or their order"
    assert frame.schema["time_series_id"] == pl.Int8, "time_series_id is not Int8"
    origins = frame.group_by("time_series_id").agg(n=pl.col("nwp_init_time").n_unique())
    assert (origins["n"] == common.N_ORIGINS).all(), "a series lacks exactly 365 origins"
    assert frame.height == expected, f"{name}: {frame.height} rows, expected {expected}"
    assert frame.select(KEY_COLUMNS).n_unique() == frame.height, "duplicate keys"
    assert np.isfinite(quantiles).all(), "a null or non-finite quantile"
    assert (np.diff(quantiles, axis=1) >= 0).all(), "quantiles cross"
    assert (quantiles >= 0).all(), "a negative quantile"
    for column in ("valid_time", "nwp_init_time", "power_fcst_init_time"):
        assert frame.schema[column] == pl.Datetime("us", "UTC"), f"{column} is not UTC"
    # No feature newer than the 00 UTC run: the run is at 00:00 and the origin is 12:00 after it.
    assert (frame["nwp_init_time"].dt.hour() == 0).all(), "a run not at 00 UTC"
    assert (frame["power_fcst_init_time"] - frame["nwp_init_time"] == timedelta(hours=12)).all(), (
        "an origin not 12 h after its run"
    )
    lead = (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_minutes()
    assert lead.min() == 0, "valid times start before the origin"
    assert lead.max() == (N_HALF_HOURS - 1) * 30, "valid times end off the grid"
    print(
        f"{name}: {frame.height:,} rows = {n_series} series x {common.N_ORIGINS} origins x "
        f"{N_HALF_HOURS}; no duplicate keys, nulls, crossings, or negatives; UTC",
        flush=True,
    )


def write(*, frame: pl.DataFrame, name: str) -> None:
    """Write `frame` atomically as `<name>.parquet` in the shared folder."""
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = common.OUTPUT_DIR / f"{name}.parquet"
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(path)
    print(f"wrote {path}", flush=True)


def main() -> None:
    """Build, check, and write both files."""
    wind = wind_frame()
    check(frame=wind, n_series=3, name="wind")
    write(frame=wind, name="wind")
    solar = solar_frame()
    check(frame=solar, n_series=6, name="solar")
    write(frame=solar, name="solar")


if __name__ == "__main__":
    main()

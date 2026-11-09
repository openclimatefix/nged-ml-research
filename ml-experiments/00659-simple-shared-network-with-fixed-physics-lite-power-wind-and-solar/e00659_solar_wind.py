# ruff: noqa: E501
"""Rebuild the solar member arrays of experiment 430 with each member's 10 m wind speed.

Experiment 430 stored each member's shortwave and temperature at every row of experiment 310's
table. The Faiman cell-temperature model also needs the wind speed, so this script repeats
`e00430_members.build_farm` with the 10 m wind speed (the root of the squared eastward and
northward components), interpolated in time exactly as the temperature is, and writes the
(rows, 51) float32 speed array beside a check that the rebuilt temperature equals experiment
430's stored temperature, row for row. That check proves the rows line up.

    PYTHONPATH=<310>:<430>:<659 folders> systemd-run --user --scope -p MemoryMax=8G \
        -p OOMPolicy=continue uv run --no-sync python ml-experiments/00659-*/e00659_solar_wind.py
"""

import os
from datetime import timedelta

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_members as m310  # ty: ignore[unresolved-import]
import e00430_common as common430  # ty: ignore[unresolved-import]
import e00659_common as common
import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites


def read_wind(*, cell: int) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the (init, valid) keys and the (stamps, 51) 10 m wind speed, as `read_members` orders."""
    raw = (
        pl.scan_delta(m310.NWP_URI)
        .filter(
            (pl.col("h3_index") == cell)
            & (pl.col("init_time") >= c310.ENS_START)
            & (pl.col("init_time") < pl.lit(c310.GRID_END).str.to_datetime(time_zone="UTC"))
        )
        .select(
            "init_time",
            "valid_time",
            "ensemble_member",
            speed=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
        )
        .collect()
        .sort("init_time", "valid_time", "ensemble_member")
    )
    counts = raw.group_by("init_time", "valid_time").len()
    complete = counts.filter(pl.col("len") == c310.N_MEMBERS).select("init_time", "valid_time")
    raw = raw.join(complete, on=["init_time", "valid_time"], how="semi").sort(
        "init_time", "valid_time", "ensemble_member"
    )
    keys = raw.select("init_time", "valid_time").unique(maintain_order=True)
    n = keys.height
    return keys.with_row_index("idx"), raw["speed"].to_numpy().reshape(n, c310.N_MEMBERS)


def build_farm(site: str) -> None:
    """Write every day's wind-speed array for one farm, checked against experiment 430's arrays."""
    row = pv_sites().filter(pl.col("site") == site).row(0, named=True)
    store_cells = (
        pl.scan_delta(m310.NWP_URI)
        .filter(pl.col("init_time") == c310.ENS_START)
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )
    cell = m310.nearest_cell(
        latitude=row["latitude"], longitude=row["longitude"], cells=store_cells
    )
    keys, speed_p = read_wind(cell=cell)
    _, _, temp_p = m310.read_members(cell=cell)
    grid = m310.sun_grid(row=row)
    output = pl.read_parquet(c310.FARM_ROWS).filter(pl.col("site") == site).select("time", "output")
    stamp_keys = keys.rename({"valid_time": "period_end"})
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = common.DATA_DIR / "solar_wind"
    out_dir.mkdir(exist_ok=True)
    for day in c310.DAYS:
        low = 12 + 24 * (day - 1)
        rows = (
            output.join(grid, on="time", how="inner")
            .with_columns(init_time=(pl.col("time") - timedelta(hours=low)).dt.truncate("1d"))
            .with_columns(lead3=(pl.col("end3") - pl.col("init_time")).dt.total_hours())
            .with_columns(fine=pl.col("lead3") <= c310.THREE_HOURLY_UNTIL_HOURS)
            .with_columns(
                period_end=pl.when(pl.col("fine")).then(pl.col("end3")).otherwise(pl.col("end6")),
                mean_cos=pl.when(pl.col("fine"))
                .then(pl.col("mean_cosine3"))
                .otherwise(pl.col("mean_cosine6")),
                step_hours=pl.when(pl.col("fine")).then(3).otherwise(6),
            )
            .with_columns(
                previous_end=pl.col("period_end") - pl.duration(hours=1) * pl.col("step_hours")
            )
            .join(stamp_keys, on=["init_time", "period_end"], how="inner")
            .join(
                stamp_keys.select("init_time", previous_end="period_end", previous_idx="idx"),
                on=["init_time", "previous_end"],
                how="left",
            )
            .filter(pl.col("mean_cos") > 0)
            .sort("time")
        )
        idx = rows["idx"].to_numpy()
        prev = rows["previous_idx"].fill_null(rows["idx"]).to_numpy()
        step_minutes = rows["step_hours"].to_numpy() * 60.0
        weight = (
            (rows["period_end"] - rows["time"]).dt.total_minutes().to_numpy() + 15.0
        ) / step_minutes
        speed = speed_p[idx] * (1 - weight[:, None]) + speed_p[prev] * weight[:, None]
        reference = np.load(common430.MEMBER_DIR / f"{site}_d{day}.npy")
        if reference.shape[0] != speed.shape[0]:
            msg = f"farm {site} day {day}: {speed.shape[0]} rows, experiment 430 has {reference.shape[0]}"
            raise ValueError(msg)
        # Re-derive the temperature with the same join to prove the rows line up.
        temperature = temp_p[idx] * (1 - weight[:, None]) + temp_p[prev] * weight[:, None]
        if not np.allclose(reference[:, :, 1], temperature.astype(np.float32), atol=1e-4):
            msg = f"farm {site} day {day}: rebuilt temperature differs from experiment 430's"
            raise ValueError(msg)
        np.save(out_dir / f"{site}_d{day}.npy", speed.astype(np.float32))
        print("saved", site, day, speed.shape, flush=True)


def main() -> None:
    """Build every farm, one after another."""
    for site in c310.FARMS:
        build_farm(site)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

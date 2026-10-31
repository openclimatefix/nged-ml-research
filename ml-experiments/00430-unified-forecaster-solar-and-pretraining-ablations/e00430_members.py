"""Save the 51 members' shortwave and temperature for every farm, day, and row of experiment 310.

Experiment 310 stored only the member mean and spread of shortwave and temperature, plus the
fleet-shape plant's output per member. The member route needs each member's inputs, so this script
repeats experiment 310's row construction (`e00310_members.build_farm`) and writes the (rows, 51, 2)
float32 array of [shortwave W/m2, temperature degC] beside a check that the rows match experiment
310's table row for row.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00430-*/e00430_members.py
"""

import os
from datetime import timedelta

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_members as m310  # ty: ignore[unresolved-import]
import e00430_common as common
import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites


def build_farm(site: str) -> None:
    """Write every day's member array for one farm and check it against experiment 310's table."""
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
    keys, ghi_p, temp_p = m310.read_members(cell=cell)
    grid = m310.sun_grid(row=row)
    output = pl.read_parquet(c310.FARM_ROWS).filter(pl.col("site") == site).select("time", "output")
    stamp_keys = keys.rename({"valid_time": "period_end"})
    common.MEMBER_DIR.mkdir(parents=True, exist_ok=True)
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
        scale = rows["cosine"].to_numpy() / rows["mean_cos"].to_numpy()
        ghi = ghi_p[idx] * scale[:, None]
        temperature = temp_p[idx] * (1 - weight[:, None]) + temp_p[prev] * weight[:, None]
        reference = pl.read_parquet(c310.table_path(site=site, day=day))
        same = (
            reference.height == rows.height
            and reference["time"].equals(rows["time"])
            and reference["init_time"].equals(rows["init_time"])
            and np.allclose(
                reference["ghi_mean"].to_numpy(), ghi.mean(axis=1), rtol=1e-6, atol=1e-6
            )
        )
        if not same:
            msg = f"farm {site} day {day}: rebuilt rows differ from experiment 310's table"
            raise ValueError(msg)
        array = np.stack([ghi, temperature], axis=2).astype(np.float32)
        np.save(common.MEMBER_DIR / f"{site}_d{day}.npy", array)
        print("saved", site, day, array.shape, flush=True)


def main() -> None:
    """Build every farm, one after another (the reads dominate and one core is the budget)."""
    for site in c310.FARMS:
        build_farm(site)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

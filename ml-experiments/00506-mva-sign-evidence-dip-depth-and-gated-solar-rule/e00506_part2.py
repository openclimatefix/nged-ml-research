"""Experiment 506, Part 2: upstream meters and the metered generator behind D8, as a short check.

Reads NGED's hierarchy files (`Substations.csv`, `Monitor Direction.csv`) from the private data
folder. Prints units, the share of negative cleaned readings, the children of each grid and bulk
supply point, the share of those children that are metered, NGED's own "Reverse Power Flow?" label
for each of the eight, and the metered generator behind D8 against D8's demand level.

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-* \
        uv run --no-sync python ml-experiments/00506-*/e00506_part2.py
"""

import e00506_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.experiment_store import HOLDOUT_START
from studies.power import scan_power

HIER = data.ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15"
NODES = ("D1", "D6", "D13", "D17")
GENERATOR_ID_BEHIND_D8 = 33


def main() -> None:
    """Print the Part 2 numbers."""
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    series = data.series_table()
    subs = pl.read_csv(HIER / "Substations.csv")
    mon = pl.read_csv(HIER / "Monitor Direction.csv")
    mon = mon.join(
        series.select("site", time_series_id=pl.col("time_series_id").cast(pl.Int64)),
        left_on="TimeSeriesID",
        right_on="time_series_id",
        how="left",
    )
    power = (
        scan_power()
        .filter(pl.col("time") < HOLDOUT_START)
        .select("time_series_id", "time", "power")
        .collect()
    )
    neg = power.group_by("time_series_id").agg(
        n=pl.len(), negative_share=(pl.col("power") < 0).mean()
    )
    node_rows = mon.filter(pl.col("site").is_in(NODES)).join(
        neg, left_on="TimeSeriesID", right_on="time_series_id", how="left"
    )
    print("== nodes: units, type, negative share of cleaned readings")
    print(
        node_rows.select("site", "TimeSeriesName", "Units", "SubstationType", "n", "negative_share")
    )
    print("== NGED's reverse-flow label for the eight and the six")
    print(
        mon.filter(pl.col("site").is_in([*data.MVA_ONLY, *data.SIGNED_SIX]))
        .select("site", "Units", "Directional Monitoring?", "Reverse Power Flow?")
        .sort("site")
    )
    print("== children of each node and share metered")
    for site in NODES:
        row = node_rows.filter(pl.col("site") == site).row(0, named=True)
        num = row["SubstationNumber"]
        kids = subs.filter(
            ((pl.col("bsp_substation_number") == num) | (pl.col("gsp_substation_number") == num))
            & (pl.col("substation_number") != num)
        )
        kid_mon = mon.filter(pl.col("SubstationNumber").is_in(kids["substation_number"].to_list()))
        metered_sites = kid_mon.filter(pl.col("site").is_not_null())["site"].to_list()
        in_store = [s for s in metered_sites if s.startswith("D")]
        eight = [s for s in in_store if s in data.MVA_ONLY]
        print(
            site,
            "children in hierarchy file:",
            kids.height,
            "| with a series in the study set:",
            len(in_store),
            sorted(in_store),
            "| of the eight:",
            eight,
            "| share metered:",
            round(len(in_store) / max(kids.height, 1), 3),
        )
    # Generator behind D8 against D8's demand level on sunny middays.
    g = power.filter(pl.col("time_series_id") == GENERATOR_ID_BEHIND_D8)
    stamps = data.grid()
    gen = (
        pl.DataFrame({"time": stamps})
        .join(
            g.select("time", "power").with_columns(pl.col("time").dt.cast_time_unit("us")),
            on="time",
            how="left",
        )["power"]
        .to_numpy()
        .astype(float)
    )
    d8_cap = float(series.filter(pl.col("site") == "D8")["effective_capacity_mw"][0])
    d8 = data.power_points(site="D8") / 100 * d8_cap
    u, k = data.cams(site="D8")
    hour = data.local_hour_start(stamps=stamps)
    local = stamps.dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    key = local.dt.month().to_numpy() * 2 + (local.dt.weekday().to_numpy() >= 6)
    dark = (np.nan_to_num(u, nan=1.0) < 0.01) & np.isfinite(d8)
    level = {
        int(c): float(np.median(d8[dark & (key == c)]))
        for c in np.unique(key)
        if (dark & (key == c)).sum() > 50
    }
    demand = np.array([level.get(int(c), np.nan) for c in key])
    sunny = (
        (hour >= 10)
        & (hour < 15)
        & (np.nan_to_num(k) >= 0.7)
        & np.isfinite(gen)
        & np.isfinite(demand)
    )
    print("== generator behind D8: sunny-midday half-hours with generator output", int(sunny.sum()))
    print(
        "share where generator MW exceeds D8's dark-hour demand level (MVA):",
        round(float((gen[sunny] > demand[sunny]).mean()), 4),
    )
    print(
        "generator MW p50, p90, max:",
        [round(float(np.percentile(gen[sunny], q)), 2) for q in (50, 90, 100)],
        "| D8 demand level p50:",
        round(float(np.median(demand[sunny])), 2),
    )
    ok = sunny & np.isfinite(d8)
    print(
        "share where D8 reading is below the generator's output:",
        round(float((d8[ok] < gen[ok]).mean()), 4),
        "| correlation of generator and D8 reading:",
        round(float(np.corrcoef(gen[ok], d8[ok])[0, 1]), 3),
    )


if __name__ == "__main__":
    main()

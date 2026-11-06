"""Read-only data check for the BSP/GSP pilot design.

Reads NGED's cleaned power Delta table and prints history, flags, and monthly coverage.
Fits and scores nothing.
"""

import sys
from pathlib import Path
from typing import Final

import polars as pl

NGED: Final[Path] = Path(sys.argv[1])
SERIES: Final[dict[int, str]] = {1: "D17 BSP", 2: "D13 BSP", 27: "D6 GSP", 28: "D1 GSP"}
DEV_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")


def main() -> None:
    """Print the history, flagged share, signed share, and monthly coverage of the four series."""
    raw = (
        pl.scan_delta(str(NGED / "power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(list(SERIES)))
        .collect()
    )
    clean = (
        pl.scan_delta(str(NGED / "cleaned_power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(list(SERIES)))
        .collect()
    )
    print(raw.schema, clean.schema, sep="\n")
    print(
        raw.group_by("time_series_id")
        .agg(
            pl.col("time").min().alias("first"),
            pl.col("time").max().alias("last"),
            pl.len().alias("rows"),
            pl.col("power").is_null().sum().alias("null"),
            (pl.col("power") == 0).sum().alias("zero"),
            (pl.col("power") < 0).sum().alias("negative"),
            pl.col("power").min().alias("min"),
            pl.col("power").quantile(0.5).alias("median"),
            pl.col("power").max().alias("max"),
        )
        .sort("time_series_id")
    )
    print(
        clean.group_by("time_series_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("drop_reason").is_not_null().sum().alias("flagged"),
            pl.col("drop_reason").unique().alias("reasons"),
        )
        .sort("time_series_id")
    )
    pl.Config.set_tbl_rows(400).set_tbl_cols(20).set_fmt_str_lengths(60)
    month = clean.with_columns(pl.col("time").dt.strftime("%Y-%m").alias("month"))
    cov = (
        month.group_by("time_series_id", "month")
        .agg(
            pl.len().alias("rows"),
            pl.col("drop_reason").is_null().sum().alias("clean_rows"),
        )
        .with_columns(
            pl.col("month").map_elements(_days, return_dtype=pl.Int64).alias("days"),
        )
        .with_columns(
            (pl.col("rows") / (48 * pl.col("days"))).alias("present"),
            (pl.col("clean_rows") / (48 * pl.col("days"))).alias("clean_share"),
        )
        .sort("time_series_id", "month")
    )
    print("Months with clean share below 0.99, up to 2026-10, by series:")
    print(
        cov.filter((pl.col("clean_share") < 0.99) & (pl.col("month") < "2026-11")).select(
            "time_series_id", "month", "present", "clean_share"
        )
    )
    print("Development months:")
    print(
        cov.filter(pl.col("month").is_in(DEV_MONTHS))
        .select("time_series_id", "month", "rows", "clean_rows", "clean_share")
        .sort("month", "time_series_id")
    )
    # Hierarchy residuals on rows where both are clean.
    wide = (
        clean.filter(pl.col("drop_reason").is_null())
        .pivot(on="time_series_id", index="time", values="power")
        .sort("time")
    )
    for gsp, bsp in ((27, "1"), (28, "2")):
        both = wide.select(
            "time", pl.col(str(gsp)).alias("gsp"), pl.col(bsp).alias("bsp")
        ).drop_nulls()
        both = both.with_columns(
            (pl.col("gsp") - pl.col("bsp")).alias("resid"), pl.col("time").dt.year().alias("year")
        )
        print(
            f"GSP {gsp} minus BSP {bsp}: correlation "
            f"{both.select(pl.corr('gsp', 'bsp')).item():.3f}"
        )
        print(
            both.group_by("year")
            .agg(
                pl.len().alias("n"),
                pl.col("gsp").mean().alias("gsp_mean"),
                pl.col("bsp").mean().alias("bsp_mean"),
                pl.col("resid").mean().alias("resid_mean"),
                pl.col("resid").std().alias("resid_sd"),
                pl.col("gsp").std().alias("gsp_sd"),
            )
            .sort("year")
        )
    # Yearly history, to give the data available before each fold's cutoff.
    for sid in SERIES:
        s = clean.filter((pl.col("time_series_id") == sid) & pl.col("drop_reason").is_null())
        for m in DEV_MONTHS:
            start = pl.datetime(int(m[:4]), int(m[5:]), 1, time_zone="UTC")
            n = s.filter(
                (pl.col("time") < start) & (pl.col("time") >= start - pl.duration(days=730))
            ).height
            print(sid, m, "clean half-hours in the 24 months before:", n, "of", 730 * 48)


def _days(month: str) -> int:
    year, mon = (int(x) for x in month.split("-"))

    return __import__("calendar").monthrange(year, mon)[1]


if __name__ == "__main__":
    main()

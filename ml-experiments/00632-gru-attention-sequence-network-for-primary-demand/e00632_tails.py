# ruff: noqa: E501, D103
"""Experiment 632: tail calibration (q01, q05, q95, q99), cold winter nights, and median-minus-mean checks.

Usage: python e00632_tails.py
"""

import polars as pl
from e00632_report import ARMS, present, scored
from e00632_score import ROOT


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(14)
    out = ROOT / "tables"
    out.mkdir(exist_ok=True)
    arms = [a for a in ARMS if present(a) and a != "seq_h3w_ly"]
    frames = []
    for a in arms:
        df = (
            scored(a)
            .filter(pl.col("day") >= 2)
            .with_columns(
                below01=(pl.col("observed_power") < pl.col("q01")).cast(pl.Float64),
                below05=(pl.col("observed_power") < pl.col("q05")).cast(pl.Float64),
                above95=(pl.col("observed_power") > pl.col("q95")).cast(pl.Float64),
                above99=(pl.col("observed_power") > pl.col("q99")).cast(pl.Float64),
                night=(
                    pl.col("month").is_in(["2025-12", "2026-01"])
                    & (pl.col("valid_time").dt.hour() < 7)
                ),
                hour_ok=pl.lit(True),
            )
        )
        frames.append(df)
    df = pl.concat(frames, how="diagonal_relaxed")
    tails = (
        df.group_by("arm", "group")
        .agg(
            pl.col("below01").mean().round(3),
            pl.col("below05").mean().round(3),
            pl.col("above95").mean().round(3),
            pl.col("above99").mean().round(3),
        )
        .sort("group", "arm")
    )
    print(
        "share of readings below q01 (nominal 0.01), below q05 (0.05), above q95 (0.05), above q99 (0.01); days 2-7"
    )
    print(tails)
    tails.write_csv(out / "tail_exceedances.csv")
    night = (
        df.filter(pl.col("night"))
        .group_by("arm", "group")
        .agg(pl.col("crps").mean().round(3), pl.col("above90").mean().round(3), pl.len().alias("n"))
        .sort("group", "arm")
    )
    print("winter nights (valid 00:00-06:30 UTC in 2025-12 and 2026-01), days 2-7")
    print(night)
    night.write_csv(out / "winter_nights.csv")
    # signed bias of the median, in points of capacity
    bias = (
        df.with_columns(
            bias=((pl.col("q50") - pl.col("observed_power")) / pl.col("capacity") * 100)
        )
        .group_by("arm", "group")
        .agg(pl.col("bias").mean().round(3))
        .sort("group", "arm")
    )
    print("mean signed error of the median (forecast minus observed, points of capacity)")
    print(bias)


if __name__ == "__main__":
    main()

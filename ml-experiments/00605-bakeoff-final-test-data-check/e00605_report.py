# ruff: noqa: D103
"""Print every number the experiment 605 write-up quotes, from the stored check tables."""

import json

import polars as pl
from e00605_power import DEVELOPMENT_MONTHS, OUT_DIR

TEST_MONTHS = ("2026-07", "2026-08", "2026-09", "2026-10")


def main() -> None:
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_fmt_str_lengths(40)
    t = pl.read_parquet(OUT_DIR / "power_by_series_month.parquet")
    gaps = pl.read_parquet(OUT_DIR / "long_gaps_test_months.parquet")
    print("== per group and month: missing share (%) and flagged half-hours")
    g = (
        t.group_by("group", "month")
        .agg(
            missing_pct=(100 * pl.col("missing").sum() / pl.col("expected").sum()).round(2),
            flagged=pl.col("flagged").sum(),
            series=pl.col("label").n_unique(),
        )
        .sort("group", "month")
    )
    print(g.pivot(on="month", index="group", values="missing_pct"))
    print(g.group_by("month").agg(pl.col("flagged").sum()).sort("month"))
    print("== per series: missing share over the development months and over the test window")
    per = (
        t.filter(pl.col("group") != "Solar")
        .with_columns(
            window=pl.when(pl.col("month").is_in(list(DEVELOPMENT_MONTHS)))
            .then(pl.lit("dev"))
            .when(pl.col("month").is_in(list(TEST_MONTHS)))
            .then(pl.lit("test"))
            .otherwise(pl.lit("other"))
        )
        .filter(pl.col("window") != "other")
        .group_by("group", "label", "window")
        .agg(
            missing_pct=(100 * pl.col("missing").sum() / pl.col("expected").sum()).round(2),
            flagged=pl.col("flagged").sum(),
        )
        .pivot(on="window", index=["group", "label"], values=["missing_pct", "flagged"])
        .sort("group", "label")
    )
    print(per)
    print("== test-window gaps of 3 hours or more (non-solar), count by series and the longest")
    print(
        gaps.group_by("group", "label")
        .agg(n=pl.len(), longest_hours=pl.col("gap_hours").max())
        .sort("group", "label")
    )
    print("== drop reasons by month")
    print(pl.read_parquet(OUT_DIR / "drop_reasons.parquet").drop("time_series_id"))
    print("== solar midday (10:00-14:00 UTC) missing share (%) by month")
    print(
        pl.read_parquet(OUT_DIR / "solar_midday_missing.parquet")
        .filter(pl.col("month") >= "2026-04")
        .filter(pl.col("month") <= "2026-10")
        .pivot(on="month", index="label", values="missing_pct")
    )
    print("== generator zero readings")
    print(pl.read_parquet(OUT_DIR / "generator_last_zero.parquet"))
    print(pl.read_parquet(OUT_DIR / "generator_zero_share.parquet"))
    print("== ENS")
    runs = pl.read_parquet(OUT_DIR / "ens_runs.parquet")
    print(
        runs.select(
            runs=pl.len(),
            first=pl.col("init_time").min(),
            last=pl.col("init_time").max(),
            members=pl.col("members").unique(),
            valid_times=pl.col("valid_times").unique(),
            max_lead_h=pl.col("max_lead_h").unique(),
        )
    )
    print("missing runs:", pl.read_parquet(OUT_DIR / "ens_missing_runs.parquet").height)
    print(pl.read_parquet(OUT_DIR / "inputs_end.parquet"))
    print("== assertions")
    print(json.dumps(json.loads((OUT_DIR / "assertions.json").read_text()), indent=1))
    print(
        pl.read_parquet(OUT_DIR / "lag_presence.parquet")
        .group_by("month", "lag")
        .agg(
            mean_present=pl.col("share_present").mean().round(4),
            worst_origin=pl.col("share_present").min().round(4),
        )
        .sort("month", "lag")
    )


if __name__ == "__main__":
    main()

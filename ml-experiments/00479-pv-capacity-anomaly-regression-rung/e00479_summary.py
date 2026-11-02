"""Apply experiment 479's pre-registered gates and write the summary tables.

Reads `work/known_<site>.parquet` and `work/real_<site>.parquet`, writes `work/gates.parquet`,
`work/known_summary.parquet`, `work/placebo_b_summary.parquet`, and
`work/capacity_table.parquet`. Prints the gate table.
"""

import e00479_common as common
import numpy as np
import polars as pl

WIDEN_SD: float = 1.28
"""Widening of the coverage interval, in standard deviations of recovery over the farms."""


def known_summary() -> pl.DataFrame:
    """Return per (site, cloud, world, mw) median recovery, coverage, and placebo percentile."""
    frames = [pl.read_parquet(common.WORK_DIR / f"known_{s}.parquet") for s in common.SIGNED_SITES]
    known = pl.concat(frames, how="diagonal").filter(pl.col("kind") != "missing")
    groups = ["site", "cloud", "kind", "mw"]
    cell = known.group_by(groups).agg(
        pl.col("point").median().alias("recovery_median"),
        pl.col("point").min().alias("recovery_min"),
        pl.col("point").max().alias("recovery_max"),
        pl.col("point").std().alias("sd_over_farms"),
        pl.col("point").quantile(0.95).alias("point_p95"),
        pl.len().alias("n"),
    )
    covered = (
        known.join(cell.select(*groups, "sd_over_farms"), on=groups)
        .with_columns(
            lo=pl.col("p10") - WIDEN_SD * pl.col("sd_over_farms"),
            hi=pl.col("p90") + WIDEN_SD * pl.col("sd_over_farms"),
        )
        .with_columns(
            covered_alone=(pl.col("p10") <= 1.0) & (pl.col("p90") >= 1.0),
            covered_widened=(pl.col("lo") <= 1.0) & (pl.col("hi") >= 1.0),
        )
        .group_by(groups)
        .agg(
            pl.col("covered_alone").mean().alias("coverage_alone"),
            pl.col("covered_widened").mean().alias("coverage_widened"),
        )
    )
    return cell.join(covered, on=groups)


def gates(*, summary: pl.DataFrame) -> pl.DataFrame:
    """Return per (site, cloud, world) gate results: placebo, recovery, coverage, pass."""
    placebo = summary.filter(pl.col("kind") == "placebo").select(
        "site", "cloud", pl.col("point_p95").alias("placebo_p95")
    )
    out = []
    for world in ("easy", "cloud_linear", "cloud_squared"):
        rows = summary.filter(pl.col("kind") == world)
        agg = rows.group_by("site", "cloud").agg(
            pl.col("recovery_median").median().alias("recovery"),
            pl.col("coverage_widened").mean().alias("coverage"),
            pl.col("coverage_alone").mean().alias("coverage_alone"),
        )
        out.append(agg.with_columns(world=pl.lit(world)))
    table = pl.concat(out).join(placebo, on=["site", "cloud"])
    return table.with_columns(
        passes=(pl.col("placebo_p95") < common.GATE_PLACEBO)
        & (pl.col("recovery") >= common.GATE_RECOVERY)
        & (pl.col("coverage") >= common.GATE_COVERAGE),
        gated=pl.col("world") != "cloud_squared",
    ).sort("world", "cloud", "site")


def placebo_b() -> pl.DataFrame:
    """Return the irradiance-shuffle placebo's 95th percentile share of the real capacity."""
    frames = [pl.read_parquet(common.WORK_DIR / f"real_{s}.parquet") for s in common.SIGNED_SITES]
    real = pl.concat(frames, how="diagonal").filter(pl.col("kind") == "irradiance_shuffle")
    return (
        real.filter(pl.col("season_year") == 2024)
        .with_columns(share=pl.col("point") / pl.col("real_point"))
        .group_by("site", "cloud")
        .agg(
            pl.col("share").quantile(0.95).alias("share_p95"),
            pl.col("share").median().alias("share_median"),
        )
        .sort("cloud", "site")
    )


def capacity_table() -> pl.DataFrame:
    """Return AC and DC capacity percentiles at the latest season-year for every Primary."""
    frames = [pl.read_parquet(common.WORK_DIR / f"real_{s}.parquet") for s in common.ALL_SITES]
    real = pl.concat(frames, how="diagonal").filter(pl.col("kind") == "real")
    latest = real.filter(pl.col("season_year") == common.LATEST_SEASON_YEAR)
    columns = [
        (pl.col(q) * ratio).alias(f"dc_{ratio}_{q}")
        for ratio in common.DC_AC_RATIOS
        for q in ("p10", "p50", "p90")
    ]
    return latest.select("site", "cloud", "prior", "p05", "p10", "p50", "p90", "p95", *columns)


def main() -> None:
    """Write the tables and print the gate table."""
    summary = known_summary()
    table = gates(summary=summary)
    summary.write_parquet(common.WORK_DIR / "known_summary.parquet")
    table.write_parquet(common.WORK_DIR / "gates.parquet")
    pb = placebo_b()
    pb.write_parquet(common.WORK_DIR / "placebo_b_summary.parquet")
    capacity_table().write_parquet(common.WORK_DIR / "capacity_table.parquet")
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=40):
        print(table)
        print(pb)
        print(
            summary.filter(
                pl.col("kind").is_in(["load_only_linear", "load_only_squared", "profile_only"])
            )
            .select("site", "cloud", "kind", "recovery_median")
            .sort("kind", "cloud", "site")
        )
        print(
            table.group_by("world", "cloud").agg(
                pl.col("passes").sum().alias("n_pass"), pl.len().alias("n_sites")
            )
        )
    _ = np


if __name__ == "__main__":
    main()

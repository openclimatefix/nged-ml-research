# ruff: noqa: D103
"""Where the missing half-hours sit: long gaps, midday solar gaps, and the end of zero readings.

Reads only observed power. Needs `e00605_power.py` for the shared series labels.
"""

from typing import Final

import polars as pl
from e00605_power import OUT_DIR, series_labels
from studies.sources import REPO_DATA_DIR

TEST_START: Final = pl.datetime(2026, 6, 30, 23, 30, time_zone="UTC")
TEST_END: Final = pl.datetime(2026, 11, 1, 0, 0, time_zone="UTC")
LONG_GAP: Final = pl.duration(hours=3)


def main() -> None:
    labels = series_labels()
    power = (
        pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(labels["time_series_id"].to_list()))
        .filter(pl.col("time") > pl.datetime(2025, 9, 1, time_zone="UTC"))
        .collect()
        .join(labels, on="time_series_id")
        .sort("time_series_id", "time")
    )
    # 1. Gaps longer than three hours inside the test months.
    gaps = (
        power.with_columns(gap=pl.col("time").diff().over("time_series_id"))
        .filter(pl.col("time") > TEST_START, pl.col("time") <= TEST_END, pl.col("gap") > LONG_GAP)
        .filter(pl.col("group") != "Solar")
        .select(
            "group", "label", end=pl.col("time"), gap_hours=pl.col("gap").dt.total_minutes() / 60
        )
        .sort("group", "label", "end")
    )
    gaps.write_parquet(OUT_DIR / "long_gaps_test_months.parquet")
    # 2. Solar: share of midday (10:00-14:00 UTC) half-hours missing, by month.
    midday = (
        power.filter(pl.col("group") == "Solar")
        .filter(pl.col("time").dt.hour().is_between(10, 13))
        .with_columns(month=(pl.col("time") - pl.duration(minutes=30)).dt.strftime("%Y-%m"))
        .group_by("label", "month")
        .agg(present=pl.len())
        .with_columns(
            expected=pl.col("month").map_elements(
                lambda m: (
                    pl.date_range(
                        pl.date(int(m[:4]), int(m[5:]), 1),
                        pl.date(int(m[:4]), int(m[5:]), 1).dt.month_end(),
                        "1d",
                        eager=True,
                    ).len()
                    * 8
                ),
                return_dtype=pl.Int64,
            )
        )
        .with_columns(missing_pct=(100 * (1 - pl.col("present") / pl.col("expected"))).round(1))
        .sort("label", "month")
    )
    midday.write_parquet(OUT_DIR / "solar_midday_missing.parquet")
    # 3. Last exact-zero reading of each generator series, and zero count per month.
    zeros = (
        power.filter(pl.col("group").is_in(["Solar", "Wind"]), pl.col("power") == 0)
        .group_by("group", "label")
        .agg(last_zero=pl.col("time").max(), n_zero=pl.len())
        .sort("group", "label")
    )
    zeros.write_parquet(OUT_DIR / "generator_last_zero.parquet")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    print(gaps)
    print(midday.pivot(on="month", index="label", values="missing_pct"))
    print(zeros)


if __name__ == "__main__":
    main()

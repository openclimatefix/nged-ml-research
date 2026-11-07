"""Experiment 618, step 1: count the very sunny half-hours per month and apply the stop rule.

```bash
export DATA_PATH_INTERNAL=...
uv run --no-sync python ml-experiments/00618-*/e00618_step1.py
```
"""

import e00618_common as c
import polars as pl


def main() -> None:
    """Write `step1_counts.csv` and print the stop-rule verdict."""
    rows = c.load_rows()
    high = rows.filter(pl.col("clear_sky") >= c.CLEAR_SKY_MIN_W_M2)
    months = (
        high.group_by("site", "month")
        .agg(sunny=pl.col("sunny").sum())
        .with_columns(calendar_month=pl.col("month").dt.month())
    )
    months.sort("site", "month").write_csv(c.DATA_DIR / "step1_counts.csv")
    per_farm = (
        months.group_by("site")
        .agg(
            months_with_data=pl.len(),
            median_sunny=pl.col("sunny").median(),
            share_months_ge_20=(pl.col("sunny") >= c.MIN_SUNNY_PER_MONTH).mean(),
            total_sunny=pl.col("sunny").sum(),
        )
        .sort("site")
    )
    print(per_farm)
    print(
        months.filter(pl.col("site").is_in(list(c.TESTABLE)))
        .group_by("site", "calendar_month")
        .agg(pl.col("sunny").median())
        .sort("site", "calendar_month")
        .pivot(on="calendar_month", index="site", values="sunny")
    )
    testable = per_farm.filter(pl.col("site").is_in(list(c.TESTABLE)))
    n_under = int((testable["median_sunny"] < c.MIN_SUNNY_PER_MONTH).sum())
    print(f"farms under {c.MIN_SUNNY_PER_MONTH} per month: {n_under} of {len(testable)}")
    print("STOP" if n_under >= 3 else "STEP 1 PASSES")
    per_farm.write_csv(c.DATA_DIR / "step1_summary.csv")


if __name__ == "__main__":
    main()

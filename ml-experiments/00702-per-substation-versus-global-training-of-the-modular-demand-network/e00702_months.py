"""Experiment 702: the evidence table that selects the four test months.

Usage: uv run --no-sync python ml-experiments/00702-*/e00702_months.py

The script reads stored tables only and fits nothing. For each calendar month from 2025-11 to
2026-10 it prints:

- the mean ERA5 2 m temperature, shortwave irradiance, and 10 m wind speed over the grid cells of
  the 16 Primaries, and each mean's anomaly against the same calendar month of the earlier years
  (2019-09 to 2025-10), in standard deviations of that calendar month's yearly means;
- the mean observed power of the 16 Primaries as a share of capacity;
- the scores of the stored control forecast (experiment 677, step 5 with injected events, monthly
  refits, the mean of seeds 0 to 2) by origin month at leads under 240 hours: the CRPS and the
  absolute error of the median, in points of capacity, the CRPS without D15, the CRPS of D15 and
  of D18 alone, and the CRPS gain over climatology;
- the largest fall in the control's pooled CRPS from leaving one Primary out, and that Primary.

The table is written to `months_evidence.csv` in the experiment's private data folder and printed.
The plan (`result.md`) states the selection rule that the table is read with.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00702-per-substation-versus-global-training-of-the-modular-demand-network"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
E607: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00607-bakeoff-extend-cached-inputs-to-2026-10-31" / "work"
)
"""Experiment 607's cached ERA5 features and observed power, both half-hourly."""
SITE_MAP: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free"
    / "site_map.csv"
)
"""One row per Primary: `time_series_id`, the D-label, and `capacity_mw`."""
CONTROL_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
    / "scores"
    / "step5w_with_injected_sp1_s012"
    / "rows.parquet"
)
"""Per-row losses of the control forecast, the manual heuristic, and climatology."""
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
"""Time of half-hour index 0 of `origin_idx` in the control's loss table."""
FIRST_MONTH: Final[datetime] = datetime(2025, 11, 1, tzinfo=UTC)
LAST_MONTH_STOP: Final[datetime] = datetime(2026, 11, 1, tzinfo=UTC)
HEADLINE_LEAD_HOURS: Final[float] = 240.0
"""The control's scores cover the half-hours from 0 to 239.5 hours after the 12:00 UTC origin.

The plan calls this range "0 to 240 hours" and scores every arm on it.
"""
WEATHER_COLUMNS: Final[tuple[str, ...]] = ("temperature_2m", "shortwave", "wind_speed_10m")


def weather_by_month(*, series_ids: list[int]) -> pl.LazyFrame:
    """Return each fold month's mean weather and its anomaly against earlier years.

    Args:
        series_ids: The `time_series_id` values of the 16 Primaries.

    Returns:
        One row per month from 2025-11 to 2026-10, with the columns `month` (Date), the three
        `WEATHER_COLUMNS` (Float64, the month's mean over Primaries and half-hours), and one
        `<column>_z` (Float64) per weather column: the month's mean minus the mean of the same
        calendar month in the earlier years, over the standard deviation of those earlier yearly
        means.
    """
    monthly = (
        pl.scan_parquet(E607 / "era5_features.parquet")
        .filter(pl.col("time_series_id").is_in(series_ids))
        .group_by(month=pl.col("valid_time").dt.truncate("1mo").dt.date())
        .agg(pl.col(*WEATHER_COLUMNS).mean())
        .with_columns(calendar_month=pl.col("month").dt.month())
    )
    history = (
        monthly.filter(pl.col("month") < FIRST_MONTH.date())
        .group_by("calendar_month")
        .agg(
            *[pl.col(name).mean().alias(f"{name}_mean") for name in WEATHER_COLUMNS],
            *[pl.col(name).std().alias(f"{name}_sd") for name in WEATHER_COLUMNS],
        )
    )
    return (
        monthly.filter(pl.col("month") >= FIRST_MONTH.date())
        .join(history, on="calendar_month", how="left")
        .with_columns(
            **{
                f"{name}_z": (pl.col(name) - pl.col(f"{name}_mean")) / pl.col(f"{name}_sd")
                for name in WEATHER_COLUMNS
            }
        )
        .select("month", *WEATHER_COLUMNS, *[f"{name}_z" for name in WEATHER_COLUMNS])
    )


def demand_by_month(*, sites: pl.DataFrame) -> pl.LazyFrame:
    """Return each fold month's mean observed power as a share of capacity.

    Args:
        sites: One row per Primary with `time_series_id` (Int64) and `capacity_mw` (Float64).

    Returns:
        One row per month with `month` (Date) and `demand_share` (Float64), the mean over
        Primaries and half-hours of power over capacity.
    """
    return (
        pl.scan_parquet(E607 / "observed.parquet")
        .filter(pl.col("valid_time") >= FIRST_MONTH, pl.col("valid_time") < LAST_MONTH_STOP)
        .join(sites.lazy().cast({"time_series_id": pl.Int32}), on="time_series_id", how="inner")
        .group_by(month=pl.col("valid_time").dt.truncate("1mo").dt.date())
        .agg(demand_share=(pl.col("power") / pl.col("capacity_mw")).mean())
    )


def control_scores_by_month(*, sites: pl.DataFrame) -> pl.LazyFrame:
    """Return the control forecast's scores by origin month, at leads under 240 hours.

    Args:
        sites: One row per Primary with `time_series_id` (Int64) and `label` (String).

    Returns:
        One row per month with `month` (Date) and five Float64 columns in points of capacity:
        `control_crps` and `control_mae` over the 16 Primaries, `crps_without_d15` over the other
        15 Primaries, and `d15_crps` and `d18_crps` for those two Primaries alone. The last column,
        `gain_over_climatology_pct` (Float64), is the CRPS gain in percent over the 16 Primaries.
    """
    origin = pl.lit(GRID_START) + pl.duration(minutes=30) * pl.col("origin_idx")
    crps = pl.col("Model_crps")
    return (
        pl.scan_parquet(CONTROL_ROWS)
        .filter(pl.col("ok"), pl.col("lead_h") < HEADLINE_LEAD_HOURS)
        .join(sites.lazy().cast({"time_series_id": pl.Int8}), on="time_series_id", how="inner")
        .group_by(month=origin.dt.truncate("1mo").dt.date())
        .agg(
            control_crps=crps.mean(),
            control_mae=pl.col("Model_mae").mean(),
            crps_without_d15=crps.filter(pl.col("label") != "D15").mean(),
            d15_crps=crps.filter(pl.col("label") == "D15").mean(),
            d18_crps=crps.filter(pl.col("label") == "D18").mean(),
            climatology_crps=pl.col("Climatology_crps").mean(),
        )
        .with_columns(
            gain_over_climatology_pct=100.0
            * (1.0 - pl.col("control_crps") / pl.col("climatology_crps"))
        )
        .drop("climatology_crps")
    )


def leave_one_out_by_month(*, sites: pl.DataFrame) -> pl.LazyFrame:
    """Return, per origin month, the largest fall in the control's CRPS with one Primary left out.

    Args:
        sites: One row per Primary with `time_series_id` (Int64) and `label` (String).

    Returns:
        One row per month with `month` (Date), `largest_leave_one_out_fall` (Float64, points of
        capacity: the pooled CRPS of the 16 Primaries minus the pooled CRPS of the other 15, for
        the Primary where that difference is largest), and `largest_fall_label` (String, that
        Primary's label). The leads are those of `control_scores_by_month`.
    """
    origin = pl.lit(GRID_START) + pl.duration(minutes=30) * pl.col("origin_idx")
    per_primary = (
        pl.scan_parquet(CONTROL_ROWS)
        .filter(pl.col("ok"), pl.col("lead_h") < HEADLINE_LEAD_HOURS)
        .join(sites.lazy().cast({"time_series_id": pl.Int8}), on="time_series_id", how="inner")
        .group_by("label", month=origin.dt.truncate("1mo").dt.date())
        .agg(loss=pl.col("Model_crps").sum(), rows=pl.col("Model_crps").len())
    )
    total_loss = pl.col("loss").sum().over("month")
    total_rows = pl.col("rows").sum().over("month")
    return (
        per_primary.with_columns(
            fall=total_loss / total_rows
            - (total_loss - pl.col("loss")) / (total_rows - pl.col("rows"))
        )
        .sort("fall", descending=True)
        .group_by("month", maintain_order=True)
        .agg(
            largest_leave_one_out_fall=pl.col("fall").first(),
            largest_fall_label=pl.col("label").first(),
        )
    )


def main() -> None:
    """Print the evidence table and write it to the experiment's data folder."""
    sites = pl.read_csv(SITE_MAP).select("time_series_id", "label", "capacity_mw")
    assert sites.height == 16, sites.height
    table = (
        weather_by_month(series_ids=sites["time_series_id"].to_list())
        .join(demand_by_month(sites=sites), on="month", how="left")
        .join(control_scores_by_month(sites=sites), on="month", how="left")
        .join(leave_one_out_by_month(sites=sites), on="month", how="left")
        .sort("month")
        .collect()
    )
    assert table.height == 12, table.height
    OUT.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT / "months_evidence.csv")
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=2, tbl_width_chars=200):
        print(table)


if __name__ == "__main__":
    main()

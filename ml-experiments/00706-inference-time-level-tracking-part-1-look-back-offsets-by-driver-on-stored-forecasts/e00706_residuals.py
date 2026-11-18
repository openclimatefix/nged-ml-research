# ruff: noqa: E501
"""Experiment 706, stage 0: the residual table, the look-ahead assertions, and the printed example.

Run it from the repository root with the environment of `CLAUDE.md` set:

    uv run --no-sync python ml-experiments/00706-*/e00706_residuals.py --n-jobs 4

Reads the stored one-year backtest (`primaries.parquet`, `ground_truth_primaries.parquet`,
`series_capacity.parquet`) and the ensemble summaries of experiment 607 (as extended by
`ml-experiments/00667-*/e00667_extend.py`). Every row with a valid time on or after 2026-07-01 is
dropped before any table is built. Writes `residuals.parquet` (one row per Primary and half-hour)
under the experiment's data folder and prints:

- the count of half-hours with no cleaned power, by drop reason (some rows have no recorded reason),
  which no look-back or score reads;
- the weather look-ahead assertions, row by row;
- the printed example of Primary 17, origin 2026-03-10 12:00 UTC, valid 2026-03-15 13:00 UTC, which
  the page quotes;
- the peak memory of the process.

`--month YYYY-MM` writes only that month's valid times and a table with the month in its name. It is
a
smoke test of the writing code: it still builds every residual, so it does not measure the memory
of one month.
"""

import argparse
import datetime as dt
from typing import Final

import e00706_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol.metrics import MEDIAN_INDEX
from studies.experiment_store import HOLDOUT_START

EXAMPLE_SERIES: Final[int] = 17
EXAMPLE_ORIGIN: Final[dt.datetime] = dt.datetime(2026, 3, 10, 12, tzinfo=dt.UTC)
EXAMPLE_VALID: Final[dt.datetime] = dt.datetime(2026, 3, 15, 13, tzinfo=dt.UTC)
EXAMPLE_FACTS: Final[dict[str, str]] = {
    "window_first": "2026-02-24 06:30",
    "window_last": "2026-03-10 06:00",
    "newest_forecast_origin": "2026-03-09 12:00",
    "newest_forecast_run": "2026-03-09 00:00",
    "newest_power": "2026-03-10 06:00",
}


def assert_backtest_look_ahead() -> None:
    """Assert `nwp_init_time + 9 h <= power_fcst_init_time` on every stored Primary row."""
    summary = (
        pl.scan_parquet(c.BACKTEST_DIR / "primaries.parquet")
        .filter(pl.col("valid_time") < HOLDOUT_START)
        .select(
            n_rows=pl.len(),
            ok=(
                pl.col("nwp_init_time") + pl.duration(hours=9) <= pl.col("power_fcst_init_time")
            ).all(),
            run_is_midnight=(
                pl.col("nwp_init_time") == pl.col("nwp_init_time").dt.truncate("1d")
            ).all(),
            origin_is_noon=(
                pl.col("power_fcst_init_time")
                == pl.col("power_fcst_init_time").dt.truncate("1d") + pl.duration(hours=12)
            ).all(),
            valid_not_before_origin=(pl.col("valid_time") >= pl.col("power_fcst_init_time")).all(),
        )
        .collect()
    )
    row = summary.row(0, named=True)
    assert row["ok"], "a stored row has nwp_init_time + 9 h after power_fcst_init_time"
    assert row["run_is_midnight"], "a stored row's nwp_init_time is not 00:00 UTC"
    assert row["origin_is_noon"], "a stored row's power_fcst_init_time is not 12:00 UTC"
    assert row["valid_not_before_origin"], "a stored row's valid time is before its origin"
    print(
        f"ASSERT stored rows {row['n_rows']:,}: nwp_init_time + 9 h <= power_fcst_init_time on every "
        "row (run 00:00 UTC, origin 12:00 UTC, valid time not before the origin)"
    )


def assert_look_back_look_ahead() -> None:
    """Assert, for every origin and lead band, that every look-back input was known at the origin.

    For origin `t` and a half-hour `s` of its window, the forecast of lead band `j` is issued at
    origin day `(s - 24) // 48 - j`, the same arithmetic as `build_residuals` and `lookback_design`.
    The assertions are that the run day is before the origin's day, that the issue time is at or
    before `s`, that `s` is at or before `t` minus 6 hours, and that the run of 00:00 UTC plus 9
    hours is not after the issue time.
    """
    checked = 0
    for origin in range(c.N_ORIGINS):
        origin_stamp = c.ORIGIN_STAMP + c.PER_DAY * origin
        last = origin_stamp - c.LAG_6H
        first = max(last - 14 * c.PER_DAY + 1, 0)
        stamps = np.arange(first, last + 1)
        assert (stamps <= origin_stamp - c.LAG_6H).all(), "a window half-hour is after t - 6 h"
        for lead_day in range(c.LOOK_BACK_LEADS):
            run_day = (stamps - c.ORIGIN_STAMP) // c.PER_DAY - lead_day
            usable = (run_day >= 0) & (run_day < c.N_ORIGINS)
            run_day, used = run_day[usable], stamps[usable]
            issued_stamp = c.ORIGIN_STAMP + c.PER_DAY * run_day
            run_stamp = issued_stamp - 12 * 2
            assert (run_day < origin).all(), (
                f"origin {origin}: a run day is not before the origin day"
            )
            assert (issued_stamp <= used).all(), (
                f"origin {origin}, lead day {lead_day}: a forecast was issued after its half-hour"
            )
            assert (used <= origin_stamp - c.LAG_6H).all()
            assert (run_stamp + 9 * 2 <= issued_stamp).all(), "a run is later than 09:00 UTC"
            checked += len(used)
    print(
        f"ASSERT look-back: {checked:,} (origin, half-hour, lead band) triples over {c.N_ORIGINS} "
        "origins: run day before the origin day; issue time <= half-hour <= origin - 6 h; "
        "weather run + 9 h <= issue time"
    )


def assert_run_files_issue_at_nine() -> None:
    """Assert that every ensemble-summary file of an origin day issues its forecast at 09:00 UTC."""
    for origin in range(c.N_ORIGINS):
        day = c.GRID_START.date() + dt.timedelta(days=origin)
        times = (
            pl.scan_parquet(c.E607_WORK / "ens_summary" / f"init={day:%Y-%m-%d}.parquet")
            .filter(pl.col("valid_time") < HOLDOUT_START)
            .select(pl.col("power_fcst_init_time").unique())
            .collect()["power_fcst_init_time"]
            .to_list()
        )
        expected = dt.datetime(day.year, day.month, day.day, 9, tzinfo=dt.UTC)
        assert times == [expected], f"run {day}: power_fcst_init_time is {times}, not {expected}"
    print(f"ASSERT {c.N_ORIGINS} ensemble-summary files each issue at 09:00 UTC on their own day")


def print_null_counts(*, inputs: c.Inputs) -> None:
    """Print the half-hours with no cleaned power by drop reason and month; no look-back reads
    them."""
    table = (
        inputs.null_counts.with_columns(
            pl.col("drop_reason").cast(pl.String).fill_null("(no reason recorded)")
        )
        .group_by("month", "drop_reason")
        .agg(half_hours=pl.col("half_hours").sum())
        .sort("month", "drop_reason")
    )
    by_reason = (
        table.group_by("drop_reason").agg(half_hours=pl.col("half_hours").sum()).sort("drop_reason")
    )
    with pl.Config(tbl_rows=60):
        print("Half-hours with no cleaned power, 16 Primaries, by drop reason:")
        print(by_reason)
        print("Half-hours with no cleaned power, 16 Primaries, by month and drop reason:")
        print(table)
    total = int(inputs.null_counts["half_hours"].sum())
    print(f"TOTAL half-hours with null power: {total:,} of {len(inputs.ids) * c.N_STAMPS:,}")


def print_weather_coverage(*, inputs: c.Inputs) -> None:
    """Print how many target rows lack ensemble weather, and the ranges of the two drivers."""
    finite_forecast = np.isfinite(inputs.q_nra[..., MEDIAN_INDEX])
    missing = int((finite_forecast & ~np.isfinite(inputs.weather[..., 0])).sum())
    print(
        f"WEATHER rows with a stored forecast but no x_pv: {missing:,} of {int(finite_forecast.sum()):,}; "
        f"x_pv range {np.nanmin(inputs.weather[..., 0]):.3f} to {np.nanmax(inputs.weather[..., 0]):.3f}; "
        f"x_cold range {np.nanmin(inputs.weather[..., 1]):.3f} to {np.nanmax(inputs.weather[..., 1]):.3f}"
    )
    print("DRIVER variance over 2024-11 to 2025-10 (x_pv, x_cold) per series:")
    for label, variance in zip(inputs.labels, inputs.driver_var, strict=True):
        print(f"  {label}: {variance[0]:.4f} {variance[1]:.4f}")


def print_example(*, inputs: c.Inputs, residuals: c.Residuals) -> None:
    """Print the worked example of the page, and assert it matches the design's facts."""
    series = int(np.flatnonzero(inputs.ids == EXAMPLE_SERIES)[0])
    assert inputs.labels[series] == "D5", "series 17 is not D5"
    origin = (EXAMPLE_ORIGIN - c.origin_to_utc(origin=0)).days
    row = int((EXAMPLE_VALID - EXAMPLE_ORIGIN) / dt.timedelta(minutes=30))
    last = c.ORIGIN_STAMP + c.PER_DAY * origin - c.LAG_6H
    first = last - 14 * c.PER_DAY + 1
    finite_power = np.flatnonzero(np.isfinite(inputs.truth[series, : last + 1]))
    newest_power = int(finite_power[-1])
    newest_origin = (last - c.ORIGIN_STAMP) // c.PER_DAY
    facts = {
        "window_first": c.stamp_to_utc(stamp=first).strftime("%Y-%m-%d %H:%M"),
        "window_last": c.stamp_to_utc(stamp=last).strftime("%Y-%m-%d %H:%M"),
        "newest_forecast_origin": c.origin_to_utc(origin=newest_origin).strftime("%Y-%m-%d %H:%M"),
        "newest_forecast_run": (
            c.origin_to_utc(origin=newest_origin) - dt.timedelta(hours=12)
        ).strftime("%Y-%m-%d %H:%M"),
        "newest_power": c.stamp_to_utc(stamp=newest_power).strftime("%Y-%m-%d %H:%M"),
    }
    run = c.origin_to_utc(origin=origin) - dt.timedelta(hours=12)
    lead_from_run = (EXAMPLE_VALID - run).total_seconds() / 3600
    band = int(c.BAND_OF_H[row])
    print(
        f"EXAMPLE Primary {EXAMPLE_SERIES} ({inputs.labels[series]}), origin {EXAMPLE_ORIGIN:%Y-%m-%d %H:%M} UTC,"
    )
    print(
        f"  valid time {EXAMPLE_VALID:%Y-%m-%d %H:%M} UTC, row {row}, lead from the origin {row / 2:.1f} h,"
    )
    print(
        f"  lead from the 00:00 UTC run of {run:%Y-%m-%d} {lead_from_run:.0f} h, lead band {c.BAND_NAMES[band]}"
    )
    for name, value in facts.items():
        print(f"  {name}: {value}")
        assert value == EXAMPLE_FACTS[name], (
            f"example {name} is {value}, the design says {EXAMPLE_FACTS[name]}"
        )
    assert round(lead_from_run) == 133, "the example's lead from the run is not 133 h"
    reread = c.read_run(day=run.date(), ids=inputs.ids)
    one = reread.filter(
        pl.col("time_series_id") == EXAMPLE_SERIES, pl.col("valid_time") == EXAMPLE_VALID
    )
    x_pv_file, x_cold_file = c.drivers_from(
        shortwave=one["mean_shortwave"].to_numpy(),
        temperature=one["mean_temperature_2m"].to_numpy(),
    )
    x_stored = inputs.weather[series, origin, row]
    assert np.allclose([x_pv_file[0], x_cold_file[0]], x_stored, atol=1e-5), (
        "the dense weather differs from the ensemble summary file"
    )
    print(
        f"  target drivers from the run's file: x_pv {x_pv_file[0]:.3f}, x_cold {x_cold_file[0]:.3f}"
    )
    median = float(inputs.q_nra[series, origin, row, MEDIAN_INDEX])
    print(
        f"  stored normal-running median {median:.2f} points ({median * inputs.capacity[series] / 100:.2f} MW)"
    )
    for spec in (
        c.ArmSpec(name="g"),
        c.ArmSpec(name="d", drivers=(c.PV_DRIVER, c.COLD_DRIVER)),
    ):
        correction, fit = c.series_corrections(
            inputs=inputs, residuals=residuals, series=series, spec=spec
        )
        coefficients = ", ".join(f"{value:+.3f}" for value in fit.coef[origin, band])
        print(
            f"  arm {spec.name}: coefficients (intercept, slopes) {coefficients}; correction "
            f"{correction[origin, row]:+.3f} points = {correction[origin, row] * inputs.capacity[series] / 100:+.3f} MW; "
            f"covered {bool(fit.covered[origin, band])}"
        )
    in_window = np.isfinite(inputs.truth[series, first : last + 1]).sum()
    print(f"  look-back half-hours with cleaned power: {in_window} of {14 * c.PER_DAY}")


def residual_table(*, inputs: c.Inputs, residuals: c.Residuals, month: str | None) -> pl.DataFrame:
    """Return one row per Primary and half-hour with the truth, the residuals, and the drivers."""
    stamps = np.arange(c.N_STAMPS)
    valid = c.utc_series(micros=c.GRID_START_US + c.STEP_US * stamps)
    origin = (stamps - c.ORIGIN_STAMP) // c.PER_DAY
    row = (stamps - c.ORIGIN_STAMP) % c.PER_DAY
    usable = (origin >= 0) & (origin < c.N_ORIGINS)
    frames = []
    for index, series in enumerate(inputs.ids):
        x_pv = np.full(c.N_STAMPS, np.nan)
        x_cold = np.full(c.N_STAMPS, np.nan)
        x_pv[usable] = inputs.weather[index, origin[usable], row[usable], 0]
        x_cold[usable] = inputs.weather[index, origin[usable], row[usable], 1]
        frames.append(
            pl.DataFrame(
                {
                    "time_series_id": pl.Series([int(series)] * c.N_STAMPS, dtype=pl.Int8),
                    "valid_time": valid,
                    "power_points": inputs.truth[index],
                    "r_smallest": residuals.smallest[index],
                    **{
                        f"r_band_{band}": residuals.banded[index, band] for band in range(c.N_BANDS)
                    },
                    "x_pv": x_pv,
                    "x_cold": x_cold,
                }
            )
        )
    table = pl.concat(frames)
    if month is not None:
        table = table.filter(pl.col("valid_time").dt.strftime("%Y-%m") == month)
    return table


def main() -> None:
    """Build the residual table, run the assertions, and print the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=4, help="threads that read ensemble runs")
    parser.add_argument(
        "--month", default=None, help="smoke test: write only this valid-time month, YYYY-MM"
    )
    arguments = parser.parse_args()
    assert_backtest_look_ahead()
    assert_look_back_look_ahead()
    assert_run_files_issue_at_nine()
    inputs = c.load_inputs(load_prevailing=False, n_jobs=arguments.n_jobs)
    print_null_counts(inputs=inputs)
    print_weather_coverage(inputs=inputs)
    residuals = c.build_residuals(inputs=inputs)
    print_example(inputs=inputs, residuals=residuals)
    table = residual_table(inputs=inputs, residuals=residuals, month=arguments.month)
    name = (
        "residuals.parquet" if arguments.month is None else f"residuals_{arguments.month}.parquet"
    )
    c.atomic_write_parquet(frame=table, path=c.OUT_DIR / name)
    print(f"WROTE {c.OUT_DIR / name}: {table.height:,} rows")
    print(f"PEAK MEMORY {c.peak_memory_gb():.2f} GB")


if __name__ == "__main__":
    main()

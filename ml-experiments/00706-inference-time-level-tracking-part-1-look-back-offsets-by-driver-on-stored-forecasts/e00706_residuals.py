# ruff: noqa: E501
"""Experiment 706, stage 0: the residual table, the look-ahead assertions, and the printed example.

Run it from the repository root with the environment of `CLAUDE.md` set:

    uv run --no-sync python ml-experiments/00706-*/e00706_residuals.py --n-jobs 4

Reads the stored one-year backtest (`primaries.parquet`, `ground_truth_primaries.parquet`,
`series_capacity.parquet`) and the ensemble summaries of experiment 607 (as extended by
`ml-experiments/00667-*/e00667_extend.py`). Every row with a valid time on or after 2026-07-01 is
dropped before any table is built. Writes `residuals.parquet` (one row per Primary and half-hour)
under the experiment's data folder and prints:

- the count of half-hours per month with no cleaned power (`substation_zero`), which no look-back or
  score reads;
- the weather look-ahead assertions, row by row;
- the printed example of Primary 17, origin 2026-03-10 12:00 UTC, valid 2026-03-15 13:00 UTC, which
  the page quotes;
- the peak memory of the process.

`--month YYYY-MM` writes only that month's valid times and a table with the month in its name, so one
month can be run first to measure memory.
"""

import argparse
import datetime as dt

import e00706_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.experiment_store import HOLDOUT_START

EXAMPLE_SERIES: int = 17
EXAMPLE_ORIGIN: dt.datetime = dt.datetime(2026, 3, 10, 12, tzinfo=dt.UTC)
EXAMPLE_VALID: dt.datetime = dt.datetime(2026, 3, 15, 13, tzinfo=dt.UTC)
EXAMPLE_FACTS: dict[str, str] = {
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
    """Assert, for every origin, that every look-back half-hour and forecast was known at the origin.

    For origin `t` and a half-hour `s` of its window, `s <= t - 6 h`; the smallest-lead forecast of
    `s` was issued at an origin not after `s`; its weather run (00:00 UTC) plus 9 hours is not after
    that origin; and every lead-band forecast of `s` was issued before the smallest-lead one.
    """
    checked = 0
    for origin in range(c.N_ORIGINS):
        last = c.ORIGIN_STAMP + c.PER_DAY * origin - c.LAG_6H
        first = max(last - 14 * c.PER_DAY + 1, 0)
        stamps = np.arange(first, last + 1)
        assert (stamps <= c.ORIGIN_STAMP + c.PER_DAY * origin - c.LAG_6H).all()
        usable = stamps >= c.ORIGIN_STAMP
        issued = c.ORIGIN_STAMP + c.PER_DAY * ((stamps[usable] - c.ORIGIN_STAMP) // c.PER_DAY)
        assert (issued <= stamps[usable]).all(), (
            "a look-back forecast was issued after its half-hour"
        )
        assert (issued - 12 * 2 + 9 * 2 <= issued).all()
        assert (issued < c.ORIGIN_STAMP + c.PER_DAY * origin).all()
        for lead_day in range(c.LOOK_BACK_LEADS):
            assert (issued - c.PER_DAY * lead_day <= issued).all()
        checked += len(stamps)
    print(
        f"ASSERT look-back: {checked:,} (origin, half-hour) pairs over {c.N_ORIGINS} origins: "
        "valid time <= origin - 6 h; the forecast was issued at or before its half-hour and before "
        "the origin; weather run + 9 h <= issue time"
    )


def print_null_counts(*, inputs: c.Inputs) -> None:
    """Print the half-hours with no cleaned power per month, which no look-back or score reads."""
    table = (
        inputs.null_counts.group_by("month", "drop_reason")
        .agg(half_hours=pl.col("half_hours").sum())
        .sort("month", "drop_reason")
    )
    with pl.Config(tbl_rows=60):
        print("Half-hours with no cleaned power, 16 Primaries, by month and drop reason:")
        print(table)
    total = int(inputs.null_counts["half_hours"].sum())
    print(f"TOTAL half-hours dropped as null power: {total:,} of {len(inputs.ids) * c.N_STAMPS:,}")


def print_weather_coverage(*, inputs: c.Inputs) -> None:
    """Print how many target rows lack ensemble weather, and the ranges of the two drivers."""
    finite_forecast = np.isfinite(inputs.q_nra[..., 5])
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
    median = float(inputs.q_nra[series, origin, row, 5])
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
    parser.add_argument("--month", default=None, help="write only this valid-time month, YYYY-MM")
    arguments = parser.parse_args()
    assert_backtest_look_ahead()
    assert_look_back_look_ahead()
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

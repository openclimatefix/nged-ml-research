# ruff: noqa: ANN202
"""Tests of the origin-time assertions and example rows of experiment 711 on synthetic rows."""

from datetime import UTC, datetime, timedelta

import e00711_origin as origin
import numpy as np
import polars as pl
import pytest

US_PER_HOUR = 3_600 * 1_000_000
RUN = datetime(2026, 1, 14, tzinfo=UTC)


def _us(t: datetime) -> int:
    return int(t.timestamp() * 1_000_000)


def _primary_frame(*, init: datetime = RUN, origin_time: datetime | None = None) -> pl.DataFrame:
    origin_time = origin_time or init + timedelta(hours=9)
    target = origin.PRIMARY_EXAMPLE_TARGET
    return pl.DataFrame(
        {
            "time_series_id": [1, 1],
            "nwp_init_time": [init, init],
            "power_fcst_init_time": [origin_time, origin_time],
            "valid_time": [target, target + timedelta(minutes=30)],
            "cloud_mean": [0.8, 0.7],
            "cloud_spread": [0.2, 0.3],
        }
    ).cast(
        {
            c: pl.Datetime("us", "UTC")
            for c in ("nwp_init_time", "power_fcst_init_time", "valid_time")
        }
    )


def test_origin_rule_accepts_a_run_exactly_nine_hours_old():
    init = np.array([_us(RUN)])
    origin.assert_origin_after_run(nwp_init=init, origin=init + 9 * US_PER_HOUR)


def test_origin_rule_rejects_a_run_less_than_nine_hours_old():
    init = np.array([_us(RUN), _us(RUN)])
    with pytest.raises(ValueError, match="1 rows"):
        origin.assert_origin_after_run(
            nwp_init=init, origin=np.array([_us(RUN) + 9 * US_PER_HOUR, _us(RUN) + 8 * US_PER_HOUR])
        )


def test_primary_rows_pass_for_the_example_frame():
    origin.assert_primary_rows(frame=_primary_frame())


def test_primary_rows_reject_a_late_origin_and_a_non_midnight_run():
    with pytest.raises(ValueError, match="exactly its run plus 9 hours"):
        origin.assert_primary_rows(frame=_primary_frame(origin_time=RUN + timedelta(hours=10)))
    with pytest.raises(ValueError, match="less than 9 hours"):
        origin.assert_primary_rows(frame=_primary_frame(origin_time=RUN + timedelta(hours=8)))
    shifted = RUN + timedelta(hours=6)
    with pytest.raises(ValueError, match="not a 00:00 UTC run"):
        origin.assert_primary_rows(frame=_primary_frame(init=shifted))


def test_primary_example_line_names_the_run_and_never_the_target_day_run():
    line = origin.primary_example(frame=_primary_frame())
    assert "run 2026-01-14 00:00 UTC" in line
    assert "origin 2026-01-14 09:00 UTC" in line
    assert "target 2026-01-19 17:30 UTC (lead 128.5 hours)" in line
    assert "never from the run of 2026-01-19" in line


def test_primary_example_rejects_a_row_from_the_run_of_the_target_day():
    wrong_run = datetime(2026, 1, 19, tzinfo=UTC)
    with pytest.raises(ValueError, match="does not read the run of 2026-01-14"):
        origin.primary_example(frame=_primary_frame(init=wrong_run))


def _solar_series(*, init: datetime, hour: int = 12, valid_hours: float = 71.5):
    start = datetime(init.year, init.month, init.day, hour, tzinfo=UTC)
    dtype = pl.Datetime("us", "UTC")
    return (
        pl.Series([init], dtype=dtype),
        pl.Series([start], dtype=dtype),
        pl.Series([start + timedelta(hours=valid_hours)], dtype=dtype),
    )


def test_solar_rows_pass_for_the_example():
    init, org, valid = _solar_series(init=origin.SOLAR_EXAMPLE_RUN)
    origin.assert_solar_rows(init_time=init, origin=org, valid_time=valid)
    assert valid[0] == origin.SOLAR_EXAMPLE_TARGET


def test_solar_rows_reject_a_run_that_is_not_midnight_of_the_origin_date():
    init = pl.Series([datetime(2026, 4, 10, 3, tzinfo=UTC)], dtype=pl.Datetime("us", "UTC"))
    _, org, valid = _solar_series(init=origin.SOLAR_EXAMPLE_RUN)
    with pytest.raises(ValueError, match="00:00 UTC of its origin's date"):
        origin.assert_solar_rows(init_time=init, origin=org, valid_time=valid)


def test_solar_rows_reject_an_origin_before_the_nine_hour_delay():
    init, org, valid = _solar_series(init=origin.SOLAR_EXAMPLE_RUN, hour=8)
    with pytest.raises(ValueError, match="less than 9 hours"):
        origin.assert_solar_rows(init_time=init, origin=org, valid_time=valid)


def test_solar_rows_reject_an_origin_that_is_not_noon():
    init, org, valid = _solar_series(init=origin.SOLAR_EXAMPLE_RUN, hour=13)
    with pytest.raises(ValueError, match="not 12:00 UTC"):
        origin.assert_solar_rows(init_time=init, origin=org, valid_time=valid)


def test_solar_rows_reject_a_valid_time_not_after_the_origin():
    init, org, valid = _solar_series(init=origin.SOLAR_EXAMPLE_RUN, valid_hours=0)
    with pytest.raises(ValueError, match="not after its origin"):
        origin.assert_solar_rows(init_time=init, origin=org, valid_time=valid)


def test_solar_example_line_reads_the_run_of_the_origin_day():
    frame = pl.DataFrame(
        {
            "init_time": [origin.SOLAR_EXAMPLE_RUN],
            "time": [origin.SOLAR_EXAMPLE_TARGET],
            "cloud_mean": [0.5],
            "cloud_spread": [0.1],
        }
    ).cast({"init_time": pl.Datetime("us", "UTC"), "time": pl.Datetime("us", "UTC")})
    line = origin.solar_example(frame=frame)
    assert "run 2026-04-10 00:00 UTC, origin 2026-04-10 12:00 UTC" in line
    assert "target 2026-04-13 11:30 UTC" in line

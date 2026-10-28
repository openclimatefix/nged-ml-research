"""Known-answer tests of the evaluation protocol's slice masks and fold rule.

Each test builds a calendar by hand, so the expected count of every slice is written down rather
than computed by the code under test. `valid_time` marks the end of a half-hour, so a half-hour
that ends at 00:00 belongs to the day before.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from studies.eval_protocol.slices import (
    SERIES_GROUPS,
    SLICE_NAMES,
    add_slice_columns,
    cold_start_flag,
    fold_plan,
)


def utc(*args: int) -> datetime:
    """Return a UTC datetime."""
    return datetime(*args, tzinfo=UTC)


def make_rows(
    *,
    valid_times: list[datetime],
    series: int | list[int] = 1,
    lead_hours: float | list[float] = 12.0,
    temperature_c: float | list[float] = 10.0,
    elevation_deg: float | list[float] = 30.0,
    pv_unit_shape: float | list[float] = 0.5,
    clear_sky_index: float | list[float] | None = None,
) -> pl.DataFrame:
    """Return rows with the slice inputs, one value per row or one value for all of them."""
    n = len(valid_times)

    def column(value: float | list[float]) -> list[float]:
        return value if isinstance(value, list) else [value] * n

    leads = column(lead_hours)
    frame = pl.DataFrame(
        {
            "time_series_id": series if isinstance(series, list) else [series] * n,
            "valid_time": valid_times,
            "power_fcst_init_time": [
                v - timedelta(hours=h) for v, h in zip(valid_times, leads, strict=True)
            ],
            "temperature_c": column(temperature_c),
            "elevation_deg": column(elevation_deg),
            "pv_unit_shape": column(pv_unit_shape),
        }
    )
    if clear_sky_index is not None:
        frame = frame.with_columns(clear_sky_index=pl.Series(column(clear_sky_index)))
    return frame


def flags(*, rows: pl.DataFrame, name: str, **kwargs: object) -> list[bool]:
    """Return one slice's flag for each row."""
    return add_slice_columns(rows=rows, **kwargs)[name].to_list()  # ty: ignore[invalid-argument-type]


def test_lead_bands_hold_whole_days_of_half_hours() -> None:
    """A band is open at the bottom and closed at the top, so a day holds 48 half-hours."""
    leads = [0.5, 24.0, 24.5, 48.0, 48.5, 96.0, 96.5, 168.0, 168.5, 240.0, 240.5, 336.0, 336.5]
    rows = make_rows(valid_times=[utc(2025, 6, 10, 12)] * len(leads), lead_hours=leads)
    out = add_slice_columns(rows=rows)
    counts = {name: int(out[name].sum()) for name in out.columns if name.startswith("lead_")}
    assert counts == {
        "lead_day_1": 2,
        "lead_day_2": 2,
        "lead_days_3_4": 2,
        "lead_days_5_7": 2,
        "lead_days_8_10": 2,
        "lead_days_11_14": 2,
        "lead_days_1_10": 10,
    }
    assert out["all"].all()


def test_a_lead_band_holds_exactly_48_half_hours_per_day() -> None:
    """Half-hours at leads 0.5, 1.0, ... 24.0 are one day; 24.5 starts the next."""
    leads = [0.5 * k for k in range(1, 97)]
    out = add_slice_columns(
        rows=make_rows(valid_times=[utc(2025, 6, 10, 12)] * 96, lead_hours=leads)
    )
    assert (out["lead_day_1"].sum(), out["lead_day_2"].sum()) == (48, 48)


def test_time_of_day_bands_are_read_in_local_time_from_the_half_hour_start() -> None:
    """Bands are night 00-06, morning 06-11, midday 11-16, evening 16-24 by local start hour."""
    winter = [
        utc(2025, 1, 15, 0, 30),  # starts 00:00: night
        utc(2025, 1, 15, 6, 0),  # starts 05:30: night
        utc(2025, 1, 15, 6, 30),  # starts 06:00: morning
        utc(2025, 1, 15, 11, 0),  # starts 10:30: morning
        utc(2025, 1, 15, 11, 30),  # starts 11:00: midday
        utc(2025, 1, 15, 16, 0),  # starts 15:30: midday
        utc(2025, 1, 15, 16, 30),  # starts 16:00: evening
        utc(2025, 1, 16, 0, 0),  # starts 23:30: evening
    ]
    out = add_slice_columns(rows=make_rows(valid_times=winter))
    got = [
        next(n for n in ("night", "morning", "midday", "evening") if out[f"time_{n}"][i])
        for i in range(len(winter))
    ]
    assert got == ["night", "night", "morning", "morning", "midday", "midday", "evening", "evening"]
    summer = add_slice_columns(rows=make_rows(valid_times=[utc(2025, 7, 1, 5, 30)]))
    assert summer["time_morning"].to_list() == [True]  # starts 05:00 UTC, 06:00 British Summer Time


def test_winter_is_december_to_february_in_local_time() -> None:
    """The month is the local month of the half-hour's start."""
    times = [
        utc(2025, 12, 1, 0, 0),  # starts 30 November 23:30
        utc(2025, 12, 1, 0, 30),
        utc(2026, 2, 28, 12),
        utc(2026, 3, 1, 0, 0),  # starts 28 February 23:30
        utc(2026, 3, 1, 0, 30),
        utc(2025, 11, 15, 12),
        utc(2025, 7, 15, 12),
    ]
    rows = make_rows(valid_times=times)
    assert flags(rows=rows, name="winter") == [False, True, True, True, False, False, False]


def test_winter_nights_need_the_sun_more_than_5_degrees_below_the_horizon() -> None:
    """The elevation boundary is strict: exactly minus 5 degrees is not night."""
    rows = make_rows(
        valid_times=[utc(2026, 1, 10, 12)] * 4 + [utc(2025, 7, 10, 12)],
        elevation_deg=[-10.0, -5.1, -5.0, 0.0, -30.0],
    )
    assert flags(rows=rows, name="winter_nights") == [True, True, False, False, False]


def test_cold_nights_are_3_degrees_or_below_with_the_sun_below_minus_5() -> None:
    """The temperature boundary is inclusive; a missing temperature or elevation is not cold."""
    rows = make_rows(
        valid_times=[utc(2026, 1, 10, 12)] * 7,
        temperature_c=[3.0, 3.01, 2.99, -5.0, float("nan"), 0.0, 0.0],
        elevation_deg=[-20.0, -20.0, -20.0, -20.0, -20.0, -5.0, float("nan")],
    )
    assert flags(rows=rows, name="cold_nights") == [True, False, True, True, False, False, False]


def test_sunny_unit_shape_is_each_series_own_top_third_of_daylight() -> None:
    """The edge is the 2/3 quantile of the series' daylight unit shapes, found series by series."""
    unit_a = [float(k) for k in range(1, 21)] + [0.0, 0.005]  # 20 daylight rows, 2 night rows
    unit_b = [100.0, 200.0, 300.0, 0.0]  # a series with far larger values
    times = [utc(2025, 6, 10, 12)] * (len(unit_a) + len(unit_b))
    rows = make_rows(
        valid_times=times,
        series=[1] * len(unit_a) + [2] * len(unit_b),
        pv_unit_shape=[*unit_a, *unit_b],
    )
    out = add_slice_columns(rows=rows)
    # Series 1: the 2/3 quantile of 1..20 is 13.667, so 14..20 are sunny.
    sunny_a = out.filter(pl.col("time_series_id") == 1)["sunny_unit_shape"].to_list()
    assert sunny_a == [False] * 13 + [True] * 7 + [False, False]
    # Series 2: the 2/3 quantile of (100, 200, 300) is 233.3, so only 300 is sunny.
    sunny_b = out.filter(pl.col("time_series_id") == 2)["sunny_unit_shape"].to_list()
    assert sunny_b == [False, False, True, False]


def test_sunny_april_to_june_needs_sunny_in_a_local_month_of_april_to_june() -> None:
    """Sunny rows outside April to June, and April to June rows that are not sunny, are left out."""
    months = [3, 4, 5, 6, 7]
    times = [utc(2025, m, 15, 12) for m in months]
    rows = make_rows(
        valid_times=[*times, utc(2025, 5, 16, 12)],
        pv_unit_shape=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    )
    # Every daylight row has unit shape 1.0, so the edge is 1.0 and all five are sunny.
    out = add_slice_columns(rows=rows)
    assert out["sunny_unit_shape"].to_list() == [True] * 5 + [False]
    assert out["sunny_april_june"].to_list() == [False, True, True, True, False, False]


def test_sunny_clear_sky_index_needs_daylight_and_an_index_of_0_9() -> None:
    """The slice exists only when a clear-sky index column is passed."""
    rows = make_rows(
        valid_times=[utc(2025, 6, 10, 12)] * 5,
        pv_unit_shape=[0.5, 0.5, 0.0, 0.5, 0.5],
        clear_sky_index=[0.9, 0.89, 1.0, float("nan"), 1.2],
    )
    assert flags(rows=rows, name="sunny_clear_sky_index") == [True, False, False, False, True]
    assert (
        "sunny_clear_sky_index" not in add_slice_columns(rows=rows.drop("clear_sky_index")).columns
    )


def test_economy_7_step_is_local_half_hours_1_to_6() -> None:
    """The step covers the half-hours starting 00:30 to 03:00 local time, in summer as in winter."""
    times = [
        utc(2025, 1, 15, 0, 30),  # starts 00:00: half-hour 0
        utc(2025, 1, 15, 1, 0),  # starts 00:30: half-hour 1
        utc(2025, 1, 15, 3, 30),  # starts 03:00: half-hour 6
        utc(2025, 1, 15, 4, 0),  # starts 03:30: half-hour 7
        utc(2025, 7, 1, 0, 0),  # starts 23:30 UTC, 00:30 British Summer Time: half-hour 1
        utc(2025, 7, 1, 1, 0),  # starts 00:30 UTC, 01:30 British Summer Time: half-hour 3
    ]
    assert flags(rows=make_rows(valid_times=times), name="economy_7_step") == [
        False, True, True, False, True, True,
    ]  # fmt: skip


def test_christmas_runs_from_24_december_to_1_january() -> None:
    """The window is by local date, and a half-hour ending at midnight belongs to the day before."""
    times = [
        utc(2025, 12, 24, 0, 0),  # starts 23 December 23:30
        utc(2025, 12, 24, 0, 30),
        utc(2025, 12, 31, 12),
        utc(2026, 1, 1, 12),
        utc(2026, 1, 2, 0, 0),  # starts 1 January 23:30
        utc(2026, 1, 2, 0, 30),
        utc(2025, 12, 10, 12),
    ]
    assert flags(rows=make_rows(valid_times=times), name="christmas") == [
        False, True, True, True, True, False, False,
    ]  # fmt: skip


def test_bank_holidays_are_the_named_holiday_periods() -> None:
    """The slice holds bank holidays, the Easter weekend, bridge days, and the Christmas run."""
    times = [
        utc(2025, 4, 18, 12),  # Good Friday
        utc(2025, 4, 19, 12),  # Easter Saturday
        utc(2025, 4, 21, 12),  # Easter Monday
        utc(2025, 5, 5, 12),  # May Day
        utc(2025, 8, 25, 12),  # Late Summer Bank Holiday
        utc(2025, 12, 25, 12),  # Christmas Day
        utc(2025, 12, 29, 12),  # the Christmas run
        utc(2025, 6, 10, 12),  # an ordinary Tuesday
        utc(2025, 12, 16, 12),  # an ordinary Tuesday
    ]
    assert flags(rows=make_rows(valid_times=times), name="bank_holidays") == [
        True, True, True, True, True, True, True, False, False,
    ]  # fmt: skip


def test_event_slices_cover_the_event_and_the_seven_days_after_it() -> None:
    """An event covers [start, end) and the following days cover [end, end + 7 days)."""
    times = [
        utc(2026, 1, 9, 23, 30),  # before
        utc(2026, 1, 10, 0, 0),  # start: inside
        utc(2026, 1, 11, 23, 30),  # last half-hour inside
        utc(2026, 1, 12, 0, 0),  # end: following day 1
        utc(2026, 1, 18, 23, 30),  # last following half-hour
        utc(2026, 1, 19, 0, 0),  # end + 7 days: clear
    ]
    events = pl.DataFrame(
        {"time_series_id": [1], "start": [utc(2026, 1, 10)], "end": [utc(2026, 1, 12)]}
    )
    rows = make_rows(valid_times=times)
    inside = flags(rows=rows, name="event_rows", events=events)
    after = flags(rows=rows, name="event_following_7_days", events=events)
    assert inside == [False, True, True, False, False, False]
    assert after == [False, False, False, True, True, False]


def test_an_event_without_a_series_covers_every_series_and_one_with_a_series_covers_it_alone() -> (
    None
):
    """A null `time_series_id` is an event at every series."""
    times = [utc(2026, 1, 10, 12)] * 2
    rows = make_rows(valid_times=times, series=[1, 2])
    own = pl.DataFrame(
        {"time_series_id": [2], "start": [utc(2026, 1, 10)], "end": [utc(2026, 1, 11)]}
    )
    everywhere = own.with_columns(time_series_id=pl.lit(None, dtype=pl.Int64))
    assert flags(rows=rows, name="event_rows", events=own) == [False, True]
    assert flags(rows=rows, name="event_rows", events=everywhere) == [True, True]


def test_event_and_cold_start_slices_are_absent_without_their_inputs() -> None:
    """Four slices need an optional input and the other slices never depend on it."""
    out = add_slice_columns(rows=make_rows(valid_times=[utc(2026, 1, 10, 12)]))
    assert {"event_rows", "event_following_7_days", "cold_start_months"}.isdisjoint(out.columns)
    assert {"sunny_clear_sky_index"}.isdisjoint(out.columns)
    assert [n for n in SLICE_NAMES if n in out.columns] == [
        c for c in out.columns if c in SLICE_NAMES
    ]


def test_missing_input_column_is_an_error() -> None:
    """A table without `temperature_c` cannot define a cold night, so it is refused."""
    rows = make_rows(valid_times=[utc(2026, 1, 10, 12)]).drop("temperature_c")
    with pytest.raises(ValueError, match="temperature_c"):
        add_slice_columns(rows=rows)


def test_fold_plan_needs_12_months_before_a_training_cut_that_precedes_the_origins() -> None:
    """With a first reading on 2024-08-15, the folds for July and August 2025 are cold starts."""
    months = [utc(2025 + (m > 12), (m - 1) % 12 + 1, 1) for m in range(7, 19)]
    plan = fold_plan(first_reading=utc(2024, 8, 15), scored_months=months)
    assert plan["scored_month"].to_list() == months
    assert plan["enough_history"].to_list() == [False, False] + [True] * 10
    # The cut leaves the publication lag before the first origin of the scored month.
    assert plan["training_cut"].to_list() == [m - timedelta(hours=9) for m in months]
    assert (plan["training_cut"] < plan["scored_month"]).all()


def test_fold_plan_boundary_is_exactly_12_months_at_the_training_cut() -> None:
    """A first reading exactly 12 months before the cut is enough, and one second later is not."""
    month = [utc(2025, 9, 1)]
    cut = utc(2025, 8, 31, 15)
    assert fold_plan(first_reading=utc(2024, 8, 31, 15), scored_months=month)["enough_history"][0]
    late = utc(2024, 8, 31, 15) + timedelta(seconds=1)
    assert not fold_plan(first_reading=late, scored_months=month)["enough_history"][0]
    assert cut == fold_plan(first_reading=late, scored_months=month)["training_cut"][0]


def test_fold_plan_refuses_months_that_are_not_contiguous_or_not_month_starts() -> None:
    """Folds are contiguous calendar months, each starting at 00:00 UTC on day 1."""
    with pytest.raises(ValueError, match="consecutive"):
        fold_plan(first_reading=utc(2024, 1, 1), scored_months=[utc(2025, 7, 1), utc(2025, 9, 1)])
    with pytest.raises(ValueError, match="day 1"):
        fold_plan(first_reading=utc(2024, 1, 1), scored_months=[utc(2025, 7, 2)])


def test_cold_start_slice_flags_the_rows_of_folds_with_short_history() -> None:
    """The fold of a row is its origin's month; a series with no first reading is never flagged."""
    origin_months = [utc(2025, 8, 3), utc(2025, 9, 3), utc(2025, 9, 3)]
    rows = make_rows(valid_times=[o + timedelta(hours=12) for o in origin_months], series=[1, 1, 2])
    first_readings = {1: utc(2024, 8, 15)}
    assert cold_start_flag(rows=rows, first_readings=first_readings).tolist() == [
        True, False, False,
    ]  # fmt: skip
    assert flags(rows=rows, name="cold_start_months", first_readings=first_readings) == [
        True, False, False,
    ]  # fmt: skip


def test_series_groups_match_the_protocol() -> None:
    """Six signed-MW, eight MVA-only, and two short-history Primaries."""
    groups: dict[str, list[str]] = {}
    for label, group in SERIES_GROUPS.items():
        groups.setdefault(group, []).append(label)
    assert {g: sorted(v) for g, v in groups.items()} == {
        "signed-MW": sorted(["D5", "D9", "D10", "D11", "D14", "D16"]),
        "MVA-only": sorted(["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]),
        "short-history": ["D15", "D18"],
    }

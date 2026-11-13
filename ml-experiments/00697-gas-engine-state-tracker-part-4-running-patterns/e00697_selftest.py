r"""Known-answer tests for experiment 697's helpers, on small synthetic inputs.

    E=ml-experiments
    export PYTHONPATH=$(ls -d $E/00697-* $E/00694-* $E/00695-* | paste -sd:)
    uv run --no-sync python $E/00697-*/e00697_selftest.py

The tests read no stored data and write nothing. They cover the stamp shift at local midnight,
local time across both clock changes, the gas-day join, the day-grouping helper, spell and hazard
extraction, the week-block interval's coverage on a planted difference, a null draw whose Q1 is
zero, the questions on a planted weekday effect, the placement of a record day across a clock
change, and the folded injection's span.
"""

from datetime import UTC, date, datetime

import numpy as np
import polars as pl
from e00694_common import GRID_START  # ty: ignore[unresolved-import]
from e00697_analysis import block_shares, q1, q2, q3, q6, q7, q8  # ty: ignore[unresolved-import]
from e00697_common import (  # ty: ignore[unresolved-import]
    BLOCK_NAMES,
    CLOCK_CHANGE_DATES,
    EVENING_FIRST,
    EVENING_LAST,
    add_local_fields,
    add_segments,
    day_block,
    difference_estimate,
    extract_spells,
    gas_day_expression,
    group_within,
    half_hour_starts,
    hazard_columns,
    inject_span,
    pooled_hazard,
    ratio,
    resample_week_sums,
    share_columns,
    share_difference_statistic,
    week_estimate,
)
from e00697_engine import RecordDays, drive_series, grid_calendar  # ty: ignore[unresolved-import]


def index_of_end_stamp(*, end: datetime) -> int:
    """Return the panel-grid index whose end stamp is `end`."""
    return int((end - GRID_START).total_seconds() // 1800)


def test_stamp_shift_at_local_midnight() -> None:
    """The half-hour that ends at 23:30 UTC in British Summer Time starts at local midnight."""
    index = np.array(
        [
            index_of_end_stamp(end=datetime(2023, 6, 1, 23, 0, tzinfo=UTC)),
            index_of_end_stamp(end=datetime(2023, 6, 1, 23, 30, tzinfo=UTC)),
        ]
    )
    frame = add_local_fields(frame=pl.DataFrame({"start": half_hour_starts(index=index)}))
    assert frame["local_date"].to_list() == [date(2023, 6, 1), date(2023, 6, 2)]
    assert frame["local_half_hour"].to_list() == [47, 0]
    # A half-hour ending at local midnight belongs to the previous local day.
    winter = np.array([index_of_end_stamp(end=datetime(2023, 1, 2, 0, 0, tzinfo=UTC))])
    got = add_local_fields(frame=pl.DataFrame({"start": half_hour_starts(index=winter)}))
    assert got["local_date"].to_list() == [date(2023, 1, 1)]
    assert got["local_half_hour"].to_list() == [47]


def test_local_time_across_both_clock_changes() -> None:
    """Spring forward drops the local hour 01:00 to 02:00, and autumn repeats it."""
    first = index_of_end_stamp(end=datetime(2024, 3, 31, 0, 30, tzinfo=UTC))
    spring = add_local_fields(
        frame=pl.DataFrame({"start": half_hour_starts(index=np.arange(first, first + 4))})
    )
    # Starts 00:00, 00:30, 01:00, 01:30 UTC are 00:00, 00:30, 02:00, 02:30 local.
    assert spring["local_half_hour"].to_list() == [0, 1, 4, 5]
    first = index_of_end_stamp(end=datetime(2024, 10, 26, 23, 30, tzinfo=UTC))
    autumn = add_local_fields(
        frame=pl.DataFrame({"start": half_hour_starts(index=np.arange(first, first + 6))})
    )
    # Starts 23:00 to 01:30 UTC: 00:00, 00:30, 01:00, 01:30 BST, then 01:00 and 01:30 GMT again.
    assert autumn["local_half_hour"].to_list() == [0, 1, 2, 3, 2, 3]
    assert len(CLOCK_CHANGE_DATES) == 14
    full = add_local_fields(
        frame=pl.DataFrame(
            {
                "start": half_hour_starts(
                    index=np.arange(
                        index_of_end_stamp(end=datetime(2024, 3, 1, tzinfo=UTC)),
                        index_of_end_stamp(end=datetime(2024, 12, 1, tzinfo=UTC)),
                    )
                )
            }
        )
    )
    counts = dict(full.filter(pl.col("clock_change_day")).group_by("local_date").len().iter_rows())
    assert counts[date(2024, 3, 31)] == 46
    assert counts[date(2024, 10, 27)] == 50
    assert date(2024, 4, 1) not in counts


def test_gas_day_join() -> None:
    """A half-hour starting at 04:30 local is on the previous gas day, and 05:00 starts the day."""
    starts = pl.Series(
        "start",
        [datetime(2023, 1, 10, 4, 30, tzinfo=UTC), datetime(2023, 1, 10, 5, 0, tzinfo=UTC)],
    ).dt.cast_time_unit("us")
    frame = pl.DataFrame({"start": starts}).with_columns(gas_day=gas_day_expression())
    assert frame["gas_day"].to_list() == [date(2023, 1, 9), date(2023, 1, 10)]
    summer = pl.DataFrame(
        {"start": pl.Series([datetime(2023, 7, 10, 3, 30, tzinfo=UTC)]).dt.cast_time_unit("us")}
    ).with_columns(gas_day=gas_day_expression())
    assert summer["gas_day"].to_list() == [date(2023, 7, 9)]  # 04:30 local on 10 July


def test_group_within_and_blocks() -> None:
    """Thirds within a cell are equal-sized rank intervals, and null stays null."""
    frame = pl.DataFrame(
        {
            "cell": [1] * 6 + [2] * 3,
            "x": [5.0, 1.0, 3.0, 2.0, 6.0, 4.0, 9.0, 7.0, 8.0],
        }
    ).with_columns(group=group_within(value="x", n_groups=3, within=("cell",)))
    assert frame["group"].to_list() == [2, 0, 1, 0, 2, 1, 2, 0, 1]
    assert day_block(half_hour=np.array([0, 9, 10, 17, 18, 31, 32, 37, 38, 47])).tolist() == [
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4
    ]  # fmt: skip
    assert (EVENING_FIRST, EVENING_LAST) == (32, 37)


def test_spells_and_hazards() -> None:
    """Complete spells drop the cut-off ends of a segment and any run over an invalid half-hour."""
    state = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    segment = np.array([0] * 10 + [1] * 4)
    valid = np.ones(14, dtype=bool)
    spells = extract_spells(state=state, valid=valid, segment=segment)
    # Segment 0 runs: 11 (cut), 000, 1, 0, 111 (cut). Segment 1: 00 (cut), 11 (cut).
    assert spells.length.tolist() == [3, 1, 1]
    assert spells.value.tolist() == [False, True, False]
    valid[3] = False
    spells = extract_spells(state=state, valid=valid, segment=segment)
    assert spells.length.tolist() == [1, 1]
    columns = hazard_columns(lengths=np.array([1, 1, 2, 3, 3, 3]), max_length=4)
    sums = columns.sum(axis=0)
    # At length 1: 2 spells end of 6 at risk. At length 2: 1 of 4. At length 3: 3 of 3.
    assert pooled_hazard(sums=sums, first=1, last=1, max_length=4) == 2 / 6
    assert pooled_hazard(sums=sums, first=2, last=2, max_length=4) == 1 / 4
    assert pooled_hazard(sums=sums, first=1, last=3, max_length=4) == 6 / 13


def test_segments_split_at_gaps_and_group_changes() -> None:
    """A gap in the half-hours or a change of the group column starts a new segment."""
    starts = [datetime(2023, 1, 1, 0, 0, tzinfo=UTC), datetime(2023, 1, 1, 0, 30, tzinfo=UTC)]
    starts += [datetime(2023, 1, 1, 2, 0, tzinfo=UTC), datetime(2023, 1, 1, 2, 30, tzinfo=UTC)]
    frame = pl.DataFrame({"start": pl.Series(starts).dt.cast_time_unit("us"), "g": [1, 1, 1, 2]})
    out = add_segments(frame=frame, group_columns=("g",))
    assert out["segment"].to_list() == [1, 1, 2, 3]


def test_interval_coverage_on_a_planted_difference() -> None:
    """The week-block interval holds a planted difference in most replications and excludes zero."""
    rng = np.random.default_rng(1)
    hits = 0
    excluded = 0
    n_rep = 60
    for rep in range(n_rep):
        n_weeks = 40
        week = np.repeat(np.arange(n_weeks), 12)
        weekday = np.tile(np.array([True] * 6 + [False] * 6), n_weeks)
        week_effect = np.repeat(rng.normal(0, 0.08, n_weeks), 12)
        p = np.clip(0.2 + 0.3 * weekday + week_effect, 0, 1)
        state = rng.random(len(week)) < p
        columns = share_columns(state=state, mask_a=weekday, mask_b=~weekday)
        est = week_estimate(
            week=week,
            columns=columns,
            statistic=share_difference_statistic,
            seed=rep,
            n_resamples=300,
        )
        assert est is not None
        hits += int(est.lo <= 0.3 <= est.hi)
        excluded += int(est.excludes_zero)
    assert hits >= 0.85 * n_rep, hits
    assert excluded == n_rep


def test_difference_of_independent_estimates() -> None:
    """The difference of two independent estimates has the difference of the points."""
    week = np.repeat(np.arange(20), 5)
    cols = share_columns(
        state=np.tile([1, 0, 1, 1, 0], 20),
        mask_a=np.tile([1, 1, 0, 0, 0], 20),
        mask_b=np.tile([0, 0, 1, 1, 1], 20),
    )
    a = week_estimate(
        week=week, columns=cols, statistic=share_difference_statistic, seed=1, n_resamples=100
    )
    b = week_estimate(
        week=week, columns=cols, statistic=share_difference_statistic, seed=2, n_resamples=100
    )
    assert a is not None
    assert b is not None
    d = difference_estimate(first=a, second=b)
    assert d.point == 0.0
    point, draws = resample_week_sums(week=week, columns=cols, seed=3, n_resamples=50)
    assert draws.shape == (50, 4)
    assert np.allclose(draws.sum(axis=0) / 50, point, rtol=0.2)
    assert np.isnan(ratio(numerator=1.0, denominator=0.0))


def synthetic_days() -> RecordDays:
    """Return 20 weekday days that run 16:00 to 19:00 and 10 weekend days that do not run."""
    output = np.zeros((30, 48))
    output[:20, 32:38] = 4.0
    weekend = np.array([False] * 20 + [True] * 10)
    return RecordDays(output=output, weekend=weekend)


def evening_difference(
    *, g: np.ndarray, calendar: tuple[np.ndarray, np.ndarray, np.ndarray], lo: int, hi: int
) -> float:
    """Return the weekday-minus-weekend running share of 16:00 to 19:00 on `[lo, hi)`."""
    _, half_hour, weekend = calendar
    evening = (half_hour[lo:hi] >= EVENING_FIRST) & (half_hour[lo:hi] <= EVENING_LAST)
    on = g[lo:hi] > 0
    wk = ~weekend[lo:hi]
    return float(on[evening & wk].mean() - on[evening & ~wk].mean())


def test_null_draw_has_no_weekday_pattern() -> None:
    """A null draw's Q1 is near zero, and a weekday-class draw carries the full pattern."""
    calendar = grid_calendar(n_grid=48 * 800)
    days = synthetic_days()
    null = drive_series(calendar=calendar, first=0, last=48 * 800, kind="null", seed=0, days=days)
    within = drive_series(
        calendar=calendar, first=0, last=48 * 800, kind="weekday", seed=0, days=days
    )
    # Weekday running days are 2 in 3 of the null draw's days, on weekdays and weekends alike.
    assert abs(evening_difference(g=null, calendar=calendar, lo=0, hi=48 * 800)) < 0.12
    assert evening_difference(g=within, calendar=calendar, lo=0, hi=48 * 800) > 0.95
    again = drive_series(calendar=calendar, first=0, last=48 * 800, kind="null", seed=0, days=days)
    assert np.array_equal(null, again)


def test_record_day_keeps_one_local_profile_across_a_clock_change() -> None:
    """The record's clock readings are placed by local half-hour, on 46-, 48-, and 50-slot days."""
    first = index_of_end_stamp(end=datetime(2024, 10, 26, 12, 0, tzinfo=UTC))
    n_grid = first + 48 * 4
    calendar = grid_calendar(n_grid=n_grid)
    profile = np.arange(48, dtype=float)
    days = RecordDays(output=np.tile(profile, (3, 1)), weekend=np.array([False, False, True]))
    g = drive_series(calendar=calendar, first=first, last=n_grid, kind="null", seed=0, days=days)
    local = add_local_fields(
        frame=pl.DataFrame({"start": half_hour_starts(index=np.arange(n_grid))})
    )
    window = local.with_columns(g=pl.Series(g)).slice(first, 48 * 4)
    assert np.array_equal(
        window["g"].to_numpy(), window["local_half_hour"].to_numpy().astype(float)
    )
    day = window.filter(pl.col("local_date") == date(2024, 10, 27))
    assert day.height == 50
    # The repeated clock reading 01:00 (half-hour 2) gets the same output twice.
    assert day.filter(pl.col("local_half_hour") == 2)["g"].to_list() == [2.0, 2.0]
    assert np.all(g[:first] == 0.0)


def test_injection_changes_only_the_span() -> None:
    """The folded injection leaves the reading outside the span untouched."""
    y = np.full(100, 5.0)
    y[7] = np.nan
    g = np.zeros(100)
    g[30:60] = 2.0
    out = inject_span(y_all=y, g=g, first=20, last=70)
    assert np.array_equal(out[:20], y[:20], equal_nan=True)
    assert np.array_equal(out[70:], y[70:])
    assert np.isclose(out[40], np.sqrt((np.sqrt(25 - 0.14**2) - 2.0) ** 2 + 0.14**2))
    assert np.isclose(out[25], 5.0)


def synthetic_view(*, effect: float) -> pl.DataFrame:
    """Return a 2-year synthetic view whose weekday evenings run more often by `effect`."""
    rng = np.random.default_rng(5)
    n = 2 * 365 * 48
    frame = add_segments(
        frame=add_local_fields(frame=pl.DataFrame({"start": half_hour_starts(index=np.arange(n))})),
        group_columns=(),
    )
    evening = frame["local_half_hour"].is_between(32, 37).to_numpy() & ~frame["weekend"].to_numpy()
    p = 0.06 + effect * evening
    return frame.with_columns(
        observed=pl.lit(value=True),
        era=pl.lit("net"),
        state=pl.Series(rng.random(n) < p).cast(pl.Float64),
        system_price=pl.Series(rng.normal(size=n)),
        reading=pl.Series(rng.normal(size=n)),
    )


def test_questions_on_a_planted_weekday_effect() -> None:
    """Q1 recovers a planted effect and finds none in a null view; Q2 to Q8 run."""
    planted = q1(view=synthetic_view(effect=0.3), seed=1)
    null = q1(view=synthetic_view(effect=0.0), seed=1)
    assert planted is not None
    assert null is not None
    assert planted.lo > 0.2
    assert planted.excludes_zero
    assert not null.excludes_zero
    view = synthetic_view(effect=0.0)
    assert q2(view=view, seed=2) is not None
    contrast, single = q3(view=view, seed=3)
    assert contrast is not None
    assert single is not None
    assert q6(view=view, seed=4) is not None
    assert q7(view=view, seed=5) is not None
    assert q8(view=view, seed=6) is not None
    assert block_shares(view=view).height == 2 * len(BLOCK_NAMES)


def main() -> None:
    """Run every test and print one line each."""
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()

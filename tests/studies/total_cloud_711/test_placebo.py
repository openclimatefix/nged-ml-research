# ruff: noqa: ANN202, C420, PERF401
"""Tests of the placebo donor maps of experiment 711 on synthetic run calendars."""

from datetime import UTC, datetime, timedelta

import e00711_common as common
import e00711_placebo as placebo
import numpy as np
import polars as pl
import pytest

CUT = datetime(2026, 1, 1, tzinfo=UTC)


def _targets(last: datetime = datetime(2026, 3, 31, tzinfo=UTC)) -> list[datetime]:
    return placebo.target_runs(last=last)


def _available() -> list[datetime]:
    return common.cache_runs(last=datetime(2026, 3, 31, tzinfo=UTC))


def test_shuffled_donors_obey_the_three_rules_and_the_missing_pattern():
    targets, available = _targets(), _available()
    donors = placebo.shuffled_donor_map(
        targets=targets, available=available, cut=CUT, seed=0, series_key=5
    )
    placebo.assert_shuffled_rules(targets=targets, donors=donors, cut=CUT)
    missing = donors == placebo.UNMAPPED
    assert missing.sum() == sum(common.cloud_missing_flags(runs=targets))
    assert missing[targets.index(datetime(2024, 1, 3, tzinfo=UTC))]
    assert missing[targets.index(datetime(2025, 11, 20, tzinfo=UTC))]
    assert not missing[targets.index(datetime(2025, 11, 21, tzinfo=UTC))]


def test_shuffled_donors_are_published_before_the_cut_even_for_a_later_run():
    targets, available = _targets(), _available()
    donors = placebo.shuffled_donor_map(
        targets=targets, available=available, cut=CUT, seed=1, series_key=2
    )
    mapped = donors[donors != placebo.UNMAPPED]
    last_day = placebo.day_number(run=CUT - common.PUBLICATION_DELAY)
    assert mapped.max() <= last_day
    # A scored run of 2026-02 reads a donor from an earlier year, issued before the cut.
    scored = targets.index(datetime(2026, 2, 10, tzinfo=UTC))
    assert placebo.run_of(day=int(donors[scored])).year < 2026


def test_shuffled_map_is_fixed_by_series_cut_and_seed_and_differs_between_them():
    targets, available = _targets(), _available()

    def draw(seed: int, series_key: int):
        return placebo.shuffled_donor_map(
            targets=targets, available=available, cut=CUT, seed=seed, series_key=series_key
        )

    assert (draw(0, 5) == draw(0, 5)).all()
    assert (draw(0, 5) != draw(1, 5)).any()
    assert (draw(0, 5) != draw(0, 6)).any()


def test_shuffled_map_raises_when_a_run_has_no_donor_in_another_year():
    targets = [datetime(2026, 7, 1, tzinfo=UTC)]
    with pytest.raises(ValueError, match="no donor"):
        placebo.shuffled_donor_map(
            targets=targets,
            available=[datetime(2026, 6, 25, tzinfo=UTC)],
            cut=datetime(2026, 7, 2, tzinfo=UTC),
            seed=0,
            series_key=1,
        )


def test_assertion_catches_a_donor_after_the_cut_in_the_same_year_or_too_far_away():
    targets = [datetime(2025, 12, 1, tzinfo=UTC)]
    late = np.array([placebo.day_number(run=datetime(2025, 12, 31, 0, tzinfo=UTC))])
    with pytest.raises(ValueError, match="published after"):
        placebo.assert_shuffled_rules(
            targets=targets, donors=late, cut=datetime(2025, 12, 20, tzinfo=UTC)
        )
    same_year = np.array([placebo.day_number(run=datetime(2025, 12, 5, tzinfo=UTC))])
    with pytest.raises(ValueError, match="calendar year"):
        placebo.assert_shuffled_rules(targets=targets, donors=same_year, cut=CUT)
    far = np.array([placebo.day_number(run=datetime(2023, 6, 1, tzinfo=UTC))])
    with pytest.raises(ValueError, match="15 days"):
        placebo.assert_shuffled_rules(targets=targets, donors=far, cut=CUT)


def test_missing_pattern_assertion_catches_a_donor_for_a_gap_run_and_a_missing_donor():
    gap_run = datetime(2024, 6, 1, tzinfo=UTC)
    ok_run = datetime(2023, 6, 1, tzinfo=UTC)
    donor = placebo.day_number(run=datetime(2022, 6, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="differ"):
        placebo.assert_same_missing_pattern(targets=[gap_run], donors=np.array([donor]))
    with pytest.raises(ValueError, match="differ"):
        placebo.assert_same_missing_pattern(targets=[ok_run], donors=np.array([placebo.UNMAPPED]))
    placebo.assert_same_missing_pattern(
        targets=[gap_run, ok_run], donors=np.array([placebo.UNMAPPED, donor])
    )


def test_donor_in_the_gap_is_rejected():
    in_gap = np.array([placebo.day_number(run=datetime(2024, 6, 1, tzinfo=UTC))])
    with pytest.raises(ValueError, match="carries no cloud"):
        placebo.assert_donors_carry_cloud(donors=in_gap)


def test_shifted_donor_is_14_days_earlier_and_falls_back_in_order():
    targets, available = _targets(), _available()
    donors = placebo.shifted_donor_map(targets=targets, available=available)
    placebo.assert_shifted_rules(targets=targets, donors=donors)

    def donor_of(run: datetime) -> datetime:
        return placebo.run_of(day=int(donors[targets.index(run)]))

    assert donor_of(datetime(2023, 6, 20, tzinfo=UTC)) == datetime(2023, 6, 6, tzinfo=UTC)
    # 14 days before 2024-01-17 is the upstream gap run, so the run 21 days earlier is used.
    assert donor_of(datetime(2024, 1, 17, tzinfo=UTC)) == datetime(2023, 12, 27, tzinfo=UTC)
    # The first runs of the recent segment have no donor 7, 14, or 21 days earlier, so they take the
    # run issued 2 years and 14 days earlier.
    first_recent = datetime(2025, 11, 21, tzinfo=UTC)
    assert donor_of(first_recent) == first_recent - timedelta(days=744)
    assert donor_of(first_recent).year == 2023
    assert (
        donors[donors != placebo.UNMAPPED]
        < [
            placebo.day_number(run=t)
            for t, d in zip(targets, donors, strict=True)
            if d != placebo.UNMAPPED
        ]
    ).all()


def test_shifted_map_leaves_gap_runs_missing():
    targets, available = _targets(), _available()
    donors = placebo.shifted_donor_map(targets=targets, available=available)
    assert (donors == placebo.UNMAPPED).sum() == sum(common.cloud_missing_flags(runs=targets))


def test_shifted_rule_catches_a_donor_that_is_not_earlier():
    run = datetime(2023, 6, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="not issued before"):
        placebo.assert_shifted_rules(targets=[run], donors=np.array([placebo.day_number(run=run)]))


def _rows(runs: list[datetime]) -> pl.DataFrame:
    """Cloud rows with a value that identifies run and lead: day number * 1000 + lead index."""
    dtype = pl.Datetime("us", "UTC")
    records = []
    for run in runs:
        origin = common.origin_of(run=run)
        for i in range(1, 4):
            records.append(
                {
                    "time_series_id": 1,
                    "nwp_init_time": run,
                    "power_fcst_init_time": origin,
                    "valid_time": origin + timedelta(minutes=30 * i),
                    "cloud_mean": placebo.day_number(run=run) * 1000.0 + i,
                    "cloud_spread": float(i),
                }
            )
    return pl.DataFrame(records).cast(
        {c: dtype for c in ("nwp_init_time", "power_fcst_init_time", "valid_time")}
    )


def test_apply_donor_map_takes_the_donor_runs_values_at_the_same_leads():
    runs = [datetime(2023, 6, d, tzinfo=UTC) for d in (1, 2, 3)]
    donors = np.array(
        [
            placebo.day_number(run=runs[2]),
            placebo.day_number(run=runs[0]),
            placebo.UNMAPPED,
        ]
    )
    out = placebo.apply_donor_map(
        rows=_rows(runs), targets=runs, donors=donors, columns=("cloud_mean", "cloud_spread")
    ).sort("nwp_init_time", "valid_time")
    assert out["nwp_init_time"].unique().len() == 2
    first = out.filter(pl.col("nwp_init_time") == runs[0])
    assert first["cloud_mean"].to_list() == [
        placebo.day_number(run=runs[2]) * 1000.0 + i for i in (1, 2, 3)
    ]
    second = out.filter(pl.col("nwp_init_time") == runs[1])
    assert second["cloud_mean"].to_list() == [
        placebo.day_number(run=runs[0]) * 1000.0 + i for i in (1, 2, 3)
    ]


def test_apply_donor_map_raises_when_the_donor_run_has_no_rows():
    runs = [datetime(2023, 6, 1, tzinfo=UTC)]
    with pytest.raises(ValueError, match="absent"):
        placebo.apply_donor_map(
            rows=_rows(runs),
            targets=runs,
            donors=np.array([placebo.day_number(run=datetime(2022, 6, 1, tzinfo=UTC))]),
            columns=("cloud_mean",),
        )


def test_apply_donor_map_rejects_the_rows_of_two_series():
    runs = [datetime(2023, 6, 1, tzinfo=UTC), datetime(2023, 6, 2, tzinfo=UTC)]
    rows = pl.concat(
        [_rows(runs), _rows(runs).with_columns(time_series_id=pl.lit(2, dtype=pl.Int64))]
    )
    with pytest.raises(ValueError, match="one series"):
        placebo.apply_donor_map(
            rows=rows,
            targets=runs,
            donors=np.array([placebo.day_number(run=runs[1]), placebo.day_number(run=runs[0])]),
            columns=("cloud_mean",),
        )


def test_a_stored_map_for_another_target_list_is_rejected():
    stored = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="different list"):
        placebo._assert_stored_calendar(stored_target=stored, targets=_targets())
    current = np.array([placebo.day_number(run=r) for r in _targets()])
    placebo._assert_stored_calendar(stored_target=current, targets=_targets())

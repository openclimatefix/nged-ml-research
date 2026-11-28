"""Known-answer tests of the match rule, the sign test, the intervals, and the verdict mapping."""
# ruff: noqa: D103, E402

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
sys.path.insert(0, str(Path(__file__).parent))
import e00736_match as m
import e00736_score as score
from e00736_match import PairedCounts


def arr(*v: float) -> np.ndarray:
    return np.asarray(v)


def test_onset_mid_day_goes_through_the_plus_one_index() -> None:
    # 13:20 is 5-minute tick 160 of the day, which lies in the half-hour 26 (13:00 to 13:30).
    hh_1320 = m.half_hour_of_tick(tick=13 * 12 + 4)
    assert hh_1320 == 26
    assert m.panel_index(grid_day=10, half_hour=hh_1320) == 10 * 48 + 27
    # 00:10 is tick 2, in half-hour 0, so the panel index is the day's first index plus 1.
    hh_0010 = m.half_hour_of_tick(tick=2)
    assert hh_0010 == 0
    assert m.panel_index(grid_day=10, half_hour=hh_0010) == 10 * 48 + 1


@pytest.mark.parametrize(("tick", "dense"), [(13 * 12 + 4, 26), (2, 0)])
def test_offset_call_at_the_plus_one_index_matches_the_transfer(tick: int, dense: int) -> None:
    # A segment whose first dense day is panel day 100; the offset model calls the start at the
    # panel index of the injected half-hour plus 1, which is also the half-hour of the call.
    seg = SimpleNamespace(
        dates=np.arange("2022-04-10", "2022-04-14", dtype="datetime64[D]"), x=np.zeros((4, 48))
    )
    day0 = int((seg.dates[0] - np.datetime64("2022-01-01")).astype(int))
    start_hh = 48 + m.half_hour_of_tick(tick=tick)
    assert start_hh == 48 + dense
    layout = pl.DataFrame({"seg": [0], "month": ["m"], "month_idx": [0], "draw": [0]})
    events = pl.DataFrame(
        {
            "primary": ["D8", "D3"],
            "start": [day0 * 48 + start_hh + 1, day0 * 48 + start_hh + 1],
            "end": [-1, -1],
            "amp_mw": [1.0, -1.0],
        }
    )
    calls = score.offset_calls(events=events, draw=0, segments=[seg], layout=layout)
    assert calls["time"].to_list() == [start_hh]
    truth = pl.DataFrame(
        {
            "seg": [0],
            "sign": [1],
            "size_points": [20.0],
            "start_hh": [start_hh],
            "end_hh": [start_hh + 60],
        }
    )
    found, both = score.offset_found(truth=truth, calls=calls, tolerance=12)
    assert found.tolist() == [True]
    assert both.tolist() == [False]
    # Writing the call at the bank half-hour (without the +1) would still be inside the tolerance
    # but the mapping must give exactly the injected half-hour.
    assert calls["time"][0] == truth["start_hh"][0]


def test_merge_calls_chains_same_sign_within_two_hours() -> None:
    t, s = m.merge_calls(times=arr(10, 12, 15, 30, 31), signs=arr(1, 1, 1, 1, -1))
    assert t.tolist() == [10, 30, 31]
    assert s.tolist() == [1, 1, -1]
    # gap of exactly 4 half-hours (2 hours) does not merge
    t, _ = m.merge_calls(times=arr(0, 4), signs=arr(1, 1))
    assert t.tolist() == [0, 4]
    # opposite signs never merge
    t, s = m.merge_calls(times=arr(0, 1), signs=arr(1, -1))
    assert t.tolist() == [0, 1]


def test_match_larger_transfer_first_and_one_call_one_transfer() -> None:
    # two transfers of the same sign share one call; the larger one gets it.
    got = m.match_edges(
        truth_times=arr(100, 104),
        truth_signs=arr(1, 1),
        truth_sizes=arr(5.0, 9.0),
        call_times=arr(102),
        call_signs=arr(1),
    )
    assert got.tolist() == [-1, 0]
    # sign must agree, and the tolerance is inclusive
    got = m.match_edges(
        truth_times=arr(100, 100),
        truth_signs=arr(1, -1),
        truth_sizes=arr(5.0, 4.0),
        call_times=arr(112, 80),
        call_signs=arr(1, 1),
    )
    assert got.tolist() == [0, -1]
    got = m.match_edges(
        truth_times=arr(100),
        truth_signs=arr(1),
        truth_sizes=arr(5.0),
        call_times=arr(113),
        call_signs=arr(1),
    )
    assert got.tolist() == [-1]


def test_match_takes_the_nearest_call() -> None:
    got = m.match_edges(
        truth_times=arr(100),
        truth_signs=arr(1),
        truth_sizes=arr(5.0),
        call_times=arr(95, 101, 108),
        call_signs=arr(1, 1, 1),
    )
    assert got.tolist() == [1]


def test_sign_test_known_values() -> None:
    assert m.sign_test_p(b=0, c=0) == 1.0
    assert m.sign_test_p(b=0, c=6) == pytest.approx(2 / 64)
    assert m.sign_test_p(b=0, c=5) == pytest.approx(2 / 32)
    assert m.sign_test_p(b=1, c=5) == pytest.approx(2 * 7 / 64)
    assert m.sign_test_p(b=3, c=3) == 1.0
    assert m.sign_test_p(b=0, c=6) < m.SIGN_ALPHA < m.sign_test_p(b=0, c=5)


def test_paired_counts_and_c1() -> None:
    pc = m.paired_counts(
        found_head=np.array([1, 1, 0, 0, 0, 0], bool),
        found_offset=np.array([1, 0, 1, 1, 0, 0], bool),
    )
    assert pc == PairedCounts(both=1, only_head=1, only_offset=2, neither=2)
    assert pc.n == 6
    assert pc.c1 == pytest.approx(1 / 6)


def test_group_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    head = rng.random(60) < 0.5
    offset = rng.random(60) < 0.5
    groups = np.repeat(np.arange(6), 10)
    lo, hi = m.group_interval(found_head=head, found_offset=offset, groups=groups, n_resamples=2000)
    point = (offset.sum() - head.sum()) / 60
    assert lo <= point <= hi
    assert (
        m.months_without_discordant(
            found_head=np.array([1, 0, 1], bool),
            found_offset=np.array([1, 1, 1], bool),
            months=np.array([0, 1, 2]),
            all_months=np.array([0, 1, 2, 3]),
        )
        == 3
    )


def counts(*, b: int, c: int, n: int = 21) -> PairedCounts:
    return PairedCounts(both=0, only_head=b, only_offset=c, neither=n - b - c)


NARROW = (-0.05, 0.10)
WIDE = (-0.30, 0.40)


@pytest.mark.parametrize(
    ("kwargs", "want"),
    [
        ({"counts": counts(b=0, c=6)}, "supported"),
        ({"counts": counts(b=0, c=6), "v1_passes": False}, "partly supported"),
        ({"counts": counts(b=6, c=0)}, "refuted"),
        ({"counts": counts(b=0, c=6, n=100)}, "small gain"),
        (
            {"counts": counts(b=1, c=4), "interval_world": NARROW, "interval_month": NARROW},
            "not supported",
        ),
        ({"counts": counts(b=1, c=4), "interval_world": WIDE}, "inconclusive"),
        ({"counts": counts(b=0, c=6, n=19)}, "inconclusive"),
        ({"counts": counts(b=0, c=6), "empty_months": 3}, "inconclusive"),
        ({"counts": counts(b=0, c=6), "gates_passed": False}, "invalid"),
    ],
)
def test_every_verdict_row_is_reachable(kwargs: dict, want: str) -> None:
    base: dict = {
        "gates_passed": True,
        "interval_world": NARROW,
        "interval_month": NARROW,
        "empty_months": 0,
        "v1_passes": True,
    }
    assert m.verdict(**(base | kwargs)) == want  # ty: ignore[invalid-argument-type]


def test_the_verdict_rows_are_all_covered_by_the_mapping() -> None:
    assert set(m.VERDICTS) == {
        "supported",
        "partly supported",
        "refuted",
        "small gain",
        "not supported",
        "inconclusive",
        "invalid",
    }


def test_a_gain_of_three_does_not_count_on_21_transfers() -> None:
    assert m.sign_test_p(b=0, c=3) > m.SIGN_ALPHA
    assert counts(b=0, c=4).c1 == pytest.approx(4 / 21)
    assert counts(b=0, c=3).c1 < m.PORTING_THRESHOLD <= counts(b=0, c=4).c1


def test_wider_interval_decides_between_not_supported_and_inconclusive() -> None:
    # the world interval is narrow and below +0.15; the month interval is wider and above it.
    out = m.verdict(
        gates_passed=True,
        counts=counts(b=2, c=3),
        interval_world=(-0.1, 0.1),
        interval_month=(-0.3, 0.2),
        empty_months=0,
        v1_passes=True,
    )
    assert out == "inconclusive"


def test_v1_and_false_calls() -> None:
    assert m.v1_passes(offset_false_per_100=1.1, head_false_per_100=1.0)
    assert not m.v1_passes(offset_false_per_100=1.11, head_false_per_100=1.0)
    flags = m.false_calls(
        call_times=arr(100, 300, 500, 700),
        call_signs=arr(1, 1, 1, -1),
        start_times=arr(105),
        start_signs=arr(1),
        end_times=arr(498, 700),
        end_signs=arr(-1, -1),
    )
    # 100 near the start (same sign); 300 false; 500 has the wrong sign for the end; 700 matches
    assert flags.tolist() == [False, True, True, False]

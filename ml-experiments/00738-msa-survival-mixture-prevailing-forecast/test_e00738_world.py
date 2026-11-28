"""Tests of the injected worlds: seeds, renewal draws, shifts, and strata."""
# ruff: noqa: D103, E402, PT018

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
# the tests never write to the experiment's data folder
os.environ.setdefault("E00738_OUT", str(Path(tempfile.gettempdir()) / "scratch" / "e738_tests"))
os.environ.setdefault("E00738_SCRATCH", "1")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))

import e00738_common as common
import e00738_world as world

IS_MW = np.array([True] * 8 + [False] * 8)
START = 40_000


def test_seed_spaces_are_disjoint_and_named() -> None:
    firsts = {
        s: world.seed_key(space=s, a=1)[0] for s in ("train", "valid", "resampled", "scoring")
    }
    assert len(set(firsts.values())) == 4
    assert firsts["scoring"] == 650
    with pytest.raises(ValueError, match="unknown seed space"):
        world.seed_key(space="other", a=1)


def test_pass_families_and_the_scoring_pass_list() -> None:
    passes = world.scoring_passes()
    assert len(passes) == 16
    assert [world.family_of(r=r) for r in passes] == ["train"] * 12 + ["held_out"] * 4
    with pytest.raises(ValueError, match="unknown pass"):
        world.family_of(r=50)


def test_scoring_draw_is_deterministic_and_keyed_by_month_pass_and_primary() -> None:
    kw: dict[str, Any] = {
        "month_idx": 3, "r": 2, "primary": 5, "month_start_idx": START,
        "window_end_idx": START + 10**6,
    }  # fmt: skip
    a = world.draw_scoring_shift(is_mw=IS_MW, **kw)
    b = world.draw_scoring_shift(is_mw=IS_MW, **kw)
    assert a == b
    c = world.draw_scoring_shift(is_mw=IS_MW, **{**kw, "primary": 6})
    assert a != c
    assert a.family == "train"
    assert a.depth > 0
    assert a.sign in {-1, 1}


def test_held_out_shifts_use_the_held_out_grid() -> None:
    s = world.draw_scoring_shift(
        month_idx=0, r=common.HELD_FIRST, primary=0, month_start_idx=START,
        window_end_idx=START + 10**6, is_mw=IS_MW,
    )  # fmt: skip
    assert s.family == "held_out"
    assert s.kind == "constant"
    assert s.drawn_days in world.HELD_DAYS
    assert s.depth in world.HELD_DEPTHS


def test_shift_clipped_flag() -> None:
    s = world.draw_scoring_shift(
        month_idx=0, r=0, primary=0, month_start_idx=START, window_end_idx=START + 5, is_mw=IS_MW
    )
    assert s.end_idx <= START + 5
    assert s.clipped == (s.end_idx - s.onset_idx < round(s.drawn_days * world.PER_DAY))


def test_resampled_world_takes_durations_from_the_real_list() -> None:
    real = np.array([0.2, 1.5, 6.0, 40.0])
    s = world.draw_resampled_shift(
        month_idx=0, r=0, primary=1, month_start_idx=START, window_end_idx=START + 10**6,
        is_mw=IS_MW, real_days=real,
    )  # fmt: skip
    assert s.family == "resampled"
    assert s.drawn_days in real


def test_renewal_shifts_in_shift_share_is_about_a_quarter() -> None:
    n_in = n_all = 0
    start, end = 10_000, 10_000 + 400 * world.PER_DAY
    days = np.arange(start, end, world.PER_DAY)
    for m in range(6):
        for p in range(16):
            shifts = world.renewal_shifts(
                space="train", fit_seed=0, month_idx=m, pass_no=0, primary=p,
                window_start_idx=start, window_end_idx=end, is_mw=IS_MW,
            )  # fmt: skip
            inside = np.zeros(len(days), bool)
            for s in shifts:
                assert s.family == "train"
                assert s.end_idx <= end
                inside |= (days >= s.onset_idx) & (days < s.end_idx)
            n_in += int(inside.sum())
            n_all += len(days)
    assert n_in / n_all == pytest.approx(0.25, abs=0.05)


def test_renewal_draws_differ_by_space_and_repeat_within_one() -> None:
    kw: dict[str, Any] = {
        "fit_seed": 0, "month_idx": 1, "pass_no": 0, "primary": 2, "window_start_idx": 10_000,
        "window_end_idx": 10_000 + 300 * world.PER_DAY, "is_mw": IS_MW,
    }  # fmt: skip
    a = world.renewal_shifts(space="train", **kw)
    b = world.renewal_shifts(space="train", **kw)
    v = world.renewal_shifts(space="valid", **kw)
    assert a == b
    assert a != v


def test_envelope_is_zero_outside_and_ramps_up_inside() -> None:
    s = world.Shift(primary=0, kind="constant", onset_idx=10, ramp_half_hours=4, end_idx=30,
                    drawn_days=1.0, depth=0.1, sign=1, family="train")  # fmt: skip
    e = world.envelope(n=40, shift=s)
    assert np.all(e[:10] == 0) and np.all(e[30:] == 0)
    np.testing.assert_allclose(e[10:14], [0.25, 0.5, 0.75, 1.0])
    assert np.all(e[14:30] == 1.0)


def test_build_own_adds_signed_depth_and_keeps_mva_power_non_negative() -> None:
    clean = np.full((16, 100), 0.05, np.float32)
    clean[3, 50] = np.nan
    mw = world.Shift(primary=0, kind="constant", onset_idx=10, ramp_half_hours=1, end_idx=40,
                     drawn_days=1.0, depth=0.1, sign=-1, family="train")  # fmt: skip
    mva = world.Shift(primary=12, kind="constant", onset_idx=10, ramp_half_hours=1, end_idx=40,
                      drawn_days=1.0, depth=0.1, sign=-1, family="train")  # fmt: skip
    own = world.build_own(clean=clean, shifts=[mw, mva], is_mw=IS_MW)
    assert own[0, 20] == pytest.approx(-0.1)
    assert own[0, 50] == 0.0
    assert own[12, 20] == pytest.approx(-0.05)  # cut at zero power
    injected = world.inject(clean=clean, own=own)
    assert injected[12, 20] == pytest.approx(0.0)
    assert np.isnan(injected[3, 50])


def test_proportional_shift_has_the_drawn_mean_depth() -> None:
    clean = (0.2 + 0.1 * np.sin(np.arange(480) * 2 * np.pi / 48)).astype(np.float32)
    s = world.Shift(primary=0, kind="proportional", onset_idx=48, ramp_half_hours=1, end_idx=432,
                    drawn_days=8.0, depth=0.05, sign=1, family="train")  # fmt: skip
    d = world.shift_delta(clean_row=clean, shift=s)
    assert d[48:432].mean() == pytest.approx(0.05, rel=0.02)


def _strata(*, origin: int, valid: int, onset: int, end: int, clipped: bool = False) -> str:
    return str(
        world.strata(
            origin_idx=np.array([origin]), valid_idx=np.array([valid]), onset_idx=np.array([onset]),
            end_idx=np.array([end]), clipped=np.array([clipped]),
        )[0]
    )  # fmt: skip


def test_strata_boundaries_are_half_open() -> None:
    d = world.PER_DAY
    onset, end = 0, 30 * d
    assert _strata(origin=0, valid=d, onset=onset, end=end) == "far"  # R = 30 days
    assert _strata(origin=end - 10 * d, valid=end, onset=onset, end=end) == "middle"  # R = 10
    assert _strata(origin=end - 10 * d - 1, valid=end, onset=onset, end=end) == "far"
    assert _strata(origin=end - d, valid=end, onset=onset, end=end) == "near"  # R = 1
    assert _strata(origin=end - d - 1, valid=end, onset=onset, end=end) == "middle"
    assert _strata(origin=end, valid=end + d, onset=onset, end=end) == "ended_24h"  # S = 0
    assert _strata(origin=end + d, valid=end + 2 * d, onset=onset, end=end) == "ended_24h"
    assert _strata(origin=end + d + 1, valid=end + 2 * d, onset=onset, end=end) == "ended_1_7"
    assert _strata(origin=end + 7 * d, valid=end + 8 * d, onset=onset, end=end) == "ended_1_7"
    assert _strata(origin=end + 7 * d + 1, valid=end + 9 * d, onset=onset, end=end) == "event_free"


def test_strata_onset_ahead_and_clipped() -> None:
    d = world.PER_DAY
    assert _strata(origin=0, valid=5 * d, onset=2 * d, end=9 * d) == "onset_ahead"
    assert _strata(origin=0, valid=d, onset=2 * d, end=9 * d) == "event_free"
    assert _strata(origin=3 * d, valid=4 * d, onset=2 * d, end=9 * d, clipped=True) == (
        "in_force_clipped"
    )
    # a clipped shift never reaches an ended stratum
    assert _strata(origin=10 * d, valid=11 * d, onset=2 * d, end=9 * d, clipped=True) == (
        "event_free"
    )


def test_every_row_has_exactly_one_stratum() -> None:
    rng = np.random.default_rng(0)
    o = rng.integers(0, 2000, 5000)
    names = world.strata(
        origin_idx=o, valid_idx=o + rng.integers(0, 480, 5000),
        onset_idx=rng.integers(0, 2000, 5000), end_idx=rng.integers(0, 3000, 5000),
        clipped=rng.random(5000) < 0.2,
    )  # fmt: skip
    assert set(names) <= {*world.STRATA, "in_force_clipped"}
    assert len(names) == 5000


def test_true_delta_follows_the_ramp() -> None:
    s = world.Shift(primary=0, kind="constant", onset_idx=10, ramp_half_hours=4, end_idx=30,
                    drawn_days=1.0, depth=0.1, sign=-1, family="train")  # fmt: skip
    d = world.true_delta_at(shift=s, valid_idx=np.array([9, 10, 13, 20, 30]))
    np.testing.assert_allclose(d, [0.0, -0.025, -0.1, -0.1, 0.0])
    assert world.true_age_days(origin_idx=np.array([58]), onset_idx=np.array([10]))[0] == 1.0


def test_stratum_shares_of_the_scoring_worlds_match_the_plan() -> None:
    shares = world.simulate_stratum_shares(lead_days=(10.0,))
    assert sum(shares[k] for k in (*world.STRATA, "in_force_clipped")) == pytest.approx(1.0)
    for name, want in common.EXPECTED_SHARES.items():
        tol = (
            common.STRATUM_TOLERANCE_END
            if name in {"near", "ended_24h"}
            else common.STRATUM_TOLERANCE
        )
        assert shares[name] == pytest.approx(want, abs=tol), name

"""Tests of the event table, the grades, PELT, and the called rates of experiment 734."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from e00653_inject import Truth  # ty: ignore[unresolved-import]
from e00653_segment import Event  # ty: ignore[unresolved-import]

sys.path.insert(0, str(Path(__file__).parent))
import e00734_events as ev
import e00734_hmm as hmm
import e00734_pelt as pelt

BLOCK: Final[int] = 12
PRIORS: Final[hmm.Priors] = hmm.Priors(rho_on=3e-4, rho_off=1 / 240, rho_cont=0.2)


def decode(*, x: np.ndarray, xh: np.ndarray | None = None, scale: float = 1.0) -> tuple:
    """Forward-backward, Viterbi, and the extracted events of a block series with sigma 1."""
    sc = np.full(4, scale)
    post, o_path, l_path = hmm.decode(x=x, f=np.zeros(len(x)), sigma=1.0, scale=sc, p=PRIORS)
    half = np.repeat(x, BLOCK) if xh is None else xh
    events = ev.extract(
        site="A",
        o_path=o_path,
        l_path=l_path,
        post=post,
        xh=half,
        sigma=1.0,
        record_end=len(x) * BLOCK,
    )
    return post, o_path, events


def event(
    *,
    site: str = "A",
    start: int,
    end: int | None,
    amp: float = 1.0,
    conf: float = 1.0,
    cls: str = "shift_no_partner",
) -> Event:
    """An event with the fields the scoring reads."""
    return Event(site, start, end, amp, abs(amp), conf, 0.0, cls=cls)


def test_a_mid_day_step_gives_one_event_with_the_right_sign_and_start() -> None:
    """Test a +5 unit step from 12:00 UTC, in Student-t noise, is one called o-event."""
    n = 365 * hmm.BLOCKS_PER_DAY
    start = 100 * hmm.BLOCKS_PER_DAY + 2
    x = np.random.default_rng(1).standard_t(4, n)
    x[start : start + 30 * hmm.BLOCKS_PER_DAY] += 5.0
    _, _, events = decode(x=x)
    o_events = [e for e in events if e.cls != ev.LEVEL_CLS and e.confidence >= 0.5]
    assert len(o_events) == 1
    assert o_events[0].amp_mw == pytest.approx(5.0, abs=0.7)
    assert abs(o_events[0].start - start * BLOCK) <= BLOCK
    assert o_events[0].end is not None
    assert abs(o_events[0].end - (start + 120) * BLOCK) <= BLOCK


def test_a_cascade_of_three_steps_in_consecutive_blocks_is_one_event() -> None:
    """Test steps of +2, +2, +2 in 3 consecutive blocks, back after 20 days, are one o-event."""
    n = 400
    x = 0.3 * np.random.default_rng(2).standard_t(4, n)
    x[100] += 2.0
    x[101] += 4.0
    x[102:182] += 6.0
    _, o_path, events = decode(x=x, scale=0.3)
    assert o_path[100] == 2
    assert o_path[102] == 6
    o_events = [e for e in events if e.cls != ev.LEVEL_CLS]
    assert len(o_events) == 1
    assert o_events[0].amp_mw > 4.0


def test_a_step_that_starts_mid_block_is_one_event_with_a_start_within_two_half_hours() -> None:
    """Test a +4 step at 09:00 UTC, so the first block holds half of it, is one o-event."""
    n_days = 120
    half = 0.3 * np.random.default_rng(3).standard_t(4, n_days * 48)
    h0 = 40 * 48 + 18
    half[h0 : h0 + 20 * 48] += 4.0
    x = half.reshape(-1, BLOCK).mean(axis=1)
    _, o_path, events = decode(x=x, xh=half, scale=0.15)
    assert o_path[h0 // BLOCK] in (2, 3, 4)
    o_events = [e for e in events if e.cls != ev.LEVEL_CLS]
    assert len(o_events) == 1
    assert abs(o_events[0].start - h0) <= 2
    assert o_events[0].end is not None
    assert abs(o_events[0].end - (h0 + 20 * 48)) <= 2


def test_a_permanent_step_is_a_level_event_graded_neither_with_a_two_peaked_posterior() -> None:
    """Test a permanent +6 step is an l-event graded "neither", with a two-peaked o just after."""
    n = 330
    x = np.random.default_rng(4).standard_t(4, n)
    x[300:] += 6.0
    post, _, events = decode(x=x)
    level = [e for e in events if e.cls == ev.LEVEL_CLS]
    assert len(level) == 1
    assert level[0].end is None
    assert level[0].amp_mw == pytest.approx(6.0)
    graded = ev.grade(events=ev.called(events=level, tau=0.5), neighbours={})
    assert graded[0].kind == "neither"
    just_after = post.o_marg[312:330]
    assert just_after[:, 20].mean() > 0.2
    assert just_after[:, 25:28].sum(axis=1).mean() > 0.2


def test_the_ramp_diagnostic_runs_and_reports_both_settings() -> None:
    """Test the ramp diagnostic returns a count of called events per setting and ramp length."""
    flat = hmm.Priors(rho_on=3e-4, rho_off=1 / 240, rho_cont=0.2, ramp_rule=False)
    out = ev.ramp_diagnostic(priors={"H-nbr": PRIORS, "H-nbr-noramp": flat}, tau=0.5)
    assert set(out) == {"H-nbr", "H-nbr-noramp"}
    for per_ramp in out.values():
        assert set(per_ramp) == set(ev.RAMP_BLOCKS)
        assert all(v["called"] >= 0.0 for v in per_ramp.values())


def test_block_quantiles_read_the_first_level_that_reaches_each_quantile() -> None:
    """Test quantiles are the first level whose cumulative probability reaches them."""
    o_marg = np.zeros((2, 41), dtype=np.float32)
    o_marg[0, 23] = 1.0
    o_marg[1, 20] = 0.5
    o_marg[1, 26] = 0.5
    post = hmm.Posterior(
        o_marg=o_marg, l_jump=np.zeros(2), up_jump=np.zeros(2), down_jump=np.zeros(2), loglik=0.0
    )
    q = ev.block_quantiles(post=post, sigma=0.5)
    np.testing.assert_allclose(q[0], 1.5)
    np.testing.assert_allclose(q[1], [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0])


def test_grade_marks_mirrored_events_within_three_blocks_and_not_beyond() -> None:
    """Test an opposite-sign neighbour within 3 blocks mirrors an event, and 4 blocks does not."""
    nb: dict[str, tuple[str, ...]] = {"A": ("B",), "B": ("A",), "C": ("A",)}
    a = event(site="A", start=1000, end=2000, amp=-2.0)
    b = event(site="B", start=1000 + 3 * BLOCK, end=2000, amp=2.0)
    c = event(site="C", start=1000 + 4 * BLOCK, end=None, amp=2.0)
    solo = event(site="A", start=5000, end=None, amp=-3.0)
    level = event(site="B", start=1000, end=None, amp=-2.0, cls=ev.LEVEL_CLS)
    ev.grade(events=[a, b, c, solo, level], neighbours=nb)
    assert a.kind == "mirrored and returned"
    assert a.cls == "edge_transfer"
    assert a.partners == ["B"]
    assert c.kind == "neither"
    assert solo.kind == "neither"
    assert level.kind == "neither"
    returned_alone = event(site="A", start=9000, end=9500, amp=1.0)
    open_pair = event(site="B", start=9000, end=None, amp=-1.0)
    ev.grade(events=[returned_alone, open_pair], neighbours=nb)
    assert returned_alone.kind in {"mirrored, not yet returned", "mirrored and returned"}
    assert open_pair.kind == "mirrored, not yet returned"


def test_matching_needs_an_intersection_over_union_of_one_half_among_called_events() -> None:
    """Test a truth is found only by a called event with IoU of 0.5 or more and the same sign."""
    truth = [Truth("A", 1000, 1480, 2.0, 5.0, "solo", 0)]
    end = 100_000

    def found(e: Event, tau: float = 0.5) -> bool:
        return bool(ev.match_called(truth=truth, events=[e], tau=tau, world_end=end)[0]["found"])

    assert found(event(start=1000, end=1960, amp=2.0))
    assert not found(event(start=1000, end=1990, amp=2.0))
    assert not found(event(start=1000, end=1480, amp=2.0, conf=0.4))
    assert found(event(start=1000, end=1480, amp=2.0, conf=0.4), tau=0.4)
    assert not found(event(start=1000, end=1480, amp=-2.0))
    assert not found(event(site="B", start=1000, end=1480, amp=2.0))
    assert not found(event(start=1030, end=1510, amp=2.0))


def test_called_rates_count_all_calls_and_two_unit_calls_with_one_megawatt_threshold() -> None:
    """Test the two rates, the megawatt threshold from experiment 653's unit, and the interval."""
    sigma653 = {"A": 1.0, "B": 2.0}
    events = [
        event(site="A", start=100, end=None, amp=1.9),
        event(site="A", start=200, end=None, amp=2.0),
        event(site="B", start=300, end=None, amp=3.9),
        event(site="B", start=400, end=None, amp=-4.0),
        event(site="B", start=5000, end=None, amp=-9.0),
    ]
    for e in events:
        e.units = 99.0
    all_rate, big_rate = ev.called_rates(
        events=events, sigma653=sigma653, primary_years=2.0, lo=0, hi=1000
    )
    assert all_rate == pytest.approx(2.0)
    assert big_rate == pytest.approx(1.0)
    all_rate, big_rate = ev.called_rates(
        events=events, sigma653=sigma653, primary_years=1.0, lo=0, hi=10_000
    )
    assert (all_rate, big_rate) == (5.0, 3.0)


def test_pelt_recovers_a_known_step_set_on_noise_free_data() -> None:
    """Test PELT returns exactly the change indices of a piecewise-constant series."""
    y = np.concatenate([np.zeros(50), np.full(70, 3.0), np.zeros(80), np.full(100, -2.0)])
    assert pelt.pelt(y=y, penalty=2.0) == [50, 120, 200]
    y[60:65] = np.nan
    assert pelt.pelt(y=y, penalty=2.0) == [50, 120, 200]
    assert pelt.pelt(y=np.zeros(100), penalty=2.0) == []
    noisy = y + 0.1 * np.random.default_rng(0).standard_normal(len(y))
    assert len(pelt.pelt(y=noisy, penalty=20.0)) <= len(pelt.pelt(y=noisy, penalty=2.0))


def test_pelt_events_give_one_event_for_a_step_and_return() -> None:
    """Test a step of 3 units with a return becomes one event with the right sign and timing."""
    y = np.concatenate([np.zeros(50), np.full(70, 3.0), np.zeros(80)])
    x = np.repeat(y, BLOCK)
    events = pelt.pelt_events(
        site="A", y=y, slow=np.zeros(len(y)), sigma=1.0, xh=x, penalty_units=5.0, record_end=len(x)
    )
    assert len(events) == 1
    assert events[0].start == 50 * BLOCK
    assert events[0].end == 120 * BLOCK
    assert events[0].amp_mw == pytest.approx(3.0)
    assert events[0].confidence == 1.0

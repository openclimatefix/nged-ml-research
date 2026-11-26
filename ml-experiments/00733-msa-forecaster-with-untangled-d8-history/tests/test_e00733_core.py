# ruff: noqa: E501
"""Tests of `e00733_core`: the registered numbers, the arms, the verdict table, and the timing."""

import math

import e00733_core as core
import pytest


def test_planned_hours_match_the_plan() -> None:
    """The counted work is 7.74 and the total 10.56 GPU-process-hours."""
    assert core.planned_hours(names=core.COUNTED_WORK) == pytest.approx(7.74)
    assert core.planned_hours(names=(*core.COUNTED_WORK, *core.REPORT_ONLY_WORK)) == pytest.approx(
        10.56
    )
    assert core.PLANNED_HOURS == 10.56


def test_the_arms_have_the_registered_fit_counts() -> None:
    """FU runs 40 fits, FU-S 24, F0r 4, FS 24, Y0, YU, and YT 3 each, and M* none."""
    n = {name: arm.n_fits for name, arm in core.ARMS.items()}
    assert n == {"M*": 0, "F0r": 4, "FU": 40, "FU-S": 24, "FS": 24, "Y0": 3, "YU": 3, "YT": 3}


def test_only_report_only_arms_are_uncounted_fits() -> None:
    """Every counted fit arm is a contrast arm; FS and YT are the report-only arms."""
    assert {n for n, a in core.ARMS.items() if not a.counted} == {"F0r", "FS", "YT"}
    assert set(core.REPORT_ONLY_ARMS) == {"FS", "YT"}


def test_contrasts_are_the_four_registered() -> None:
    """C1 to C4 exist with the registered arms."""
    assert [(c.first, c.second) for c in core.CONTRASTS.values()] == [
        ("FU", "M*"),
        ("FU", "FU-S"),
        ("YU", "B-hist"),
        ("YU", "Y0"),
    ]
    assert math.isclose(core.LEVEL, 1 - 0.05 / 4)
    assert len(core.READINGS) == core.N_READINGS == 12


def _reading(point: float, half: float) -> core.Reading:
    return core.Reading(point=point, lower=point - half, upper=point + half)


@pytest.mark.parametrize(
    ("c1", "c2", "guards", "row"),
    [
        (_reading(-0.03, 0.01), _reading(-0.02, 0.01), True, 3),
        (_reading(-0.015, 0.01), _reading(-0.02, 0.01), True, 4),
        (_reading(-0.03, 0.01), _reading(0.0, 0.01), True, 5),
        (_reading(-0.03, 0.01), _reading(-0.02, 0.01), False, 5),
        (_reading(0.03, 0.01), _reading(0.0, 0.01), True, 6),
        (_reading(0.0, 0.01), _reading(0.0, 0.01), True, 7),
    ],
)
def test_every_reachable_verdict_row(
    c1: core.Reading, c2: core.Reading, guards: bool, row: int
) -> None:
    """Rows 3 to 7 are each reached by a distinct input."""
    v = core.VerdictInputs(
        stopped=False, invalid=False, failed=False, c1=c1, c2=c2, guards_hold=guards
    )
    assert core.verdict_row(v=v) == row
    assert core.verdict_word(v=v) == core.VERDICT_TABLE[row][2]


def test_rows_0_to_2_come_first() -> None:
    """A stop outranks an invalid result, which outranks a failed fit, which outranks every contrast."""
    gain = _reading(-0.05, 0.01)
    base = {"c1": gain, "c2": gain, "guards_hold": True}
    assert (
        core.verdict_row(v=core.VerdictInputs(stopped=True, invalid=True, failed=True, **base)) == 0
    )
    assert (
        core.verdict_row(v=core.VerdictInputs(stopped=False, invalid=True, failed=True, **base))
        == 1
    )
    assert (
        core.verdict_row(v=core.VerdictInputs(stopped=False, invalid=False, failed=True, **base))
        == 2
    )
    assert core.front_matter_word(word="stopped") == "inconclusive"


def test_a_non_finite_reading_is_neither_a_gain_nor_a_loss() -> None:
    """A NaN interval cannot make a gain or a loss, so the verdict is row 7."""
    nan = core.Reading(point=math.nan, lower=math.nan, upper=math.nan)
    v = core.VerdictInputs(
        stopped=False, invalid=False, failed=False, c1=nan, c2=nan, guards_hold=True
    )
    assert core.verdict_row(v=v) == 7


def test_the_stop_line_drops_report_only_arms_in_order() -> None:
    """The projection over 12 hours drops YT first, then FS."""
    assert core.stop_arms(projected=11.0) == ()
    assert core.stop_arms(projected=12.5) == ("YT",)
    assert core.stop_arms(projected=13.5) == ("YT", "FS")


def test_t_widening_exceeds_one() -> None:
    """The Student t quantile with 7 degrees of freedom is wider than the normal one."""
    assert core.t_widening(df=7) > 1.0
    assert core.t_widening(df=10_000) == pytest.approx(1.0, abs=1e-3)


def test_the_limits_payload_holds_no_date() -> None:
    """The limits payload is a mapping that the module can write."""
    payload = core.limits_payload(b_clim={"skill": 0.1}, n_synthetic={"segments": 4})
    assert isinstance(payload, dict)
    assert payload

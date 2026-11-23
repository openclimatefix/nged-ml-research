# ruff: noqa: E501, PLR0911, PT018
"""Experiment 718, WP7b: the thresholds, the group readings, the verdict, and the stop and closure rules (design section 7).

Every function here is pure, so that `self-test` checks the rules on synthetic contrasts. A contrast is
the dict that experiment 713's `contrast` returns (point, lower, upper, t_guard, seed_guard, and so on),
except that the noise-floor guard is recomputed here from the seed count of the part, because experiment
713's constant is the 5-seed floor. Nothing here changes a threshold or a rule of the design: the numbers
below are those of section 7.1 to 7.4.
Usage: e00718_decision.py [self-test | thresholds]
"""

import sys
from typing import Final

import numpy as np
from scipy import stats

TAU: Final = 0.02
"""The smallest gain worth porting, in points of CRPS (section 7.2)."""
N_COUNTED: Final = 4
LEVEL: Final = 100.0 * (1.0 - 0.05 / N_COUNTED)
"""98.75%: four counted readings per part."""
SENSITIVITY_LEVEL: Final = 100.0 * (1.0 - 0.05 / (2 * N_COUNTED))
"""99.375%: eight readings across both parts, a labelled sensitivity reading."""
REPORT_LEVEL: Final = 95.0
DELTA: Final = {5: {"six": 0.06, "eight": 0.16}, 2: {"six": 0.094, "eight": 0.25}}
"""The detectable difference by seed count (section 7.1)."""
NOISE_FLOOR: Final = {5: {"six": 0.034, "eight": 0.090}, 2: {"six": 0.054, "eight": 0.142}}
"""The noise floor by seed count (section 7.1)."""
GROUPS: Final = ("six", "eight")
STOP_BIG: Final = 0.15
STOP_MODERATE: Final = 0.10
TAIL_LIMIT: Final = 0.05
COLD_NIGHT_LIMIT: Final = 0.05
PREDICTIONS: Final = {
    "A1": {"six": (-0.06, 0.16), "eight": (-0.16, 0.25)},
    "B1": {"six": (-0.16, 0.06), "eight": (-0.25, 0.16)},
}
"""The ranges of section 7.2, written before any fit."""
MONTHLY_SD: Final = {"six": 0.10, "eight": 0.33}
N_MONTHS: Final = 8


def floor_guard(*, c: dict, group: str, n_seeds: int) -> bool:
    """Return whether the point estimate reaches the noise floor of the part's seed count."""
    return abs(c["point"]) >= NOISE_FLOOR[n_seeds][group]


def group_reading(*, c: dict, group: str, n_seeds: int) -> str:
    """Return the group's reading by the first row of the table of section 7.3 that holds."""
    d, lo, hi = c["point"], c["lower"], c["upper"]
    guards = bool(
        c["t_guard"] and c["seed_guard"] and floor_guard(c=c, group=group, n_seeds=n_seeds)
    )
    if lo > TAU and guards:
        return "clear loss"
    if lo > 0 and guards:
        return "loss"
    if hi < 0 and d <= -TAU and guards:
        return "gain"
    if hi < 0 and c["seed_guard_4_of_5"]:
        return "small gain"
    delta = DELTA[n_seeds][group]
    if lo > -delta and hi < delta:
        return "no detectable difference"
    return "cannot tell"


def placebo_is_harmful(*, placebo_vs_star: dict[str, str]) -> bool:
    """Return whether the placebo arm minus `M*` reads "loss" or "clear loss" in either group."""
    return any(r in {"loss", "clear loss"} for r in placebo_vs_star.values())


def verdict(
    *, readings: dict[str, str], placebo_upper: dict[str, float], placebo_point: dict[str, float],
    placebo_vs_star: dict[str, str], guards_hold: bool,
) -> str:  # fmt: skip
    """Return the verdict by the first row of the verdict table of section 7.3."""
    r = list(readings.values())
    if "clear loss" in r:
        return "refuted"
    if "loss" in r or all(x == "no detectable difference" for x in r):
        return "not supported"
    harmful = placebo_is_harmful(placebo_vs_star=placebo_vs_star)
    if all(x == "gain" for x in r):
        if all(placebo_upper[g] < 0 for g in GROUPS) and not harmful and guards_hold:
            return "supported"
        return "partly supported"
    if "gain" in r and "loss" not in r and "clear loss" not in r:
        return "partly supported"
    if "small gain" in r and "gain" not in r and "loss" not in r:
        small = [g for g in GROUPS if readings[g] == "small gain"]
        if all(placebo_point[g] < 0 for g in small):
            return "small gain"
    return "inconclusive"


def supported_guards(
    *, sign_holds_leave_primary: bool, sign_holds_leave_month: bool, tails_ok: bool,
    mae_moves_with_crps: bool, cold_night_difference: float, swap_keeps_sign: bool | None,
) -> dict:  # fmt: skip
    """Return the guards of `supported` for one group (section 7.3); an unread swap fails the guard."""
    out = {
        "leave_one_primary_out": sign_holds_leave_primary,
        "leave_one_month_out": sign_holds_leave_month,
        "tails_within_0.05": tails_ok,
        "mae_moves_with_crps": mae_moves_with_crps,
        "cold_nights_within_0.05": cold_night_difference <= COLD_NIGHT_LIMIT,
        "selection_swap_keeps_sign": swap_keeps_sign is True,
    }
    out["all_hold"] = all(out.values())
    return out


def stop_rule(*, first_contrast: dict[str, float], monthly: dict[str, np.ndarray]) -> bool:
    """Return whether a part stops: the first contrast on the screen folds is above +0.15 in both groups, or above +0.10 in both and positive in 3 of 4 folds in both."""
    big = all(first_contrast[g] > STOP_BIG for g in GROUPS)
    moderate = all(
        first_contrast[g] > STOP_MODERATE and int((monthly[g] > 0).sum()) >= 3 for g in GROUPS
    )
    return bool(big or moderate)


Z_LIMIT: Final = {"six": 0.076, "eight": 0.20}
"""The rule on arm `Z` (section 4): twice the seed-noise standard error for 2 seeds and 4 months."""


def z_rule_fires(*, z_minus_star: dict[str, float]) -> bool:
    """Return whether `Z` minus `M*` exceeds the limit in absolute value in either group, which fires audit trigger 1."""
    return any(abs(z_minus_star[g]) > Z_LIMIT[g] for g in GROUPS)


def nested_beats_container(*, nested_minus_container: dict[str, float]) -> bool:
    """Return whether a nested arm beats its container (`M*` beating `U27`, a placebo beating its treatment) in both groups: audit trigger 5."""
    return all(nested_minus_container[g] < 0 for g in GROUPS)


def audit_trigger_1(*, placebo_point: dict[str, float]) -> bool:
    """Return whether A2 or B2 is positive in both groups on the screen folds (audit trigger 1)."""
    return all(placebo_point[g] > 0 for g in GROUPS)


def u27_advances(*, u27_minus_u27a: dict[str, float], u27_minus_star: dict[str, float]) -> bool:
    """Return whether `U27` advances to the other four folds on its own rule (the manager must also agree to the hours)."""
    return all(u27_minus_u27a[g] < 0 and u27_minus_star[g] <= 0 for g in GROUPS)


def line_closure(
    *, verdicts: dict[str, str], first_points: dict[str, dict[str, float]], u27_exception: bool | None
) -> dict:  # fmt: skip
    """Return whether the line closes, and why (section 7.4).

    `verdicts` maps "part1" and "part2" to a verdict word; `first_points` maps them to the point estimates
    of A1 and B1 by group; `u27_exception` is whether `U27` minus `U27-A` is below zero with its 95%
    interval, or None where `U27-A` was not fitted and the exception could not be read.
    """
    both_null = all(v in {"not supported", "refuted"} for v in verdicts.values())
    both_inconclusive = all(v == "inconclusive" for v in verdicts.values())
    nonnegative = all(first_points[p][g] >= 0 for p in verdicts for g in GROUPS)
    if both_null:
        return {
            "closed": u27_exception is not True,
            "by": "verdict",
            "u27_exception": u27_exception,
            "u27_exception_unreadable": u27_exception is None,
        }
    if both_inconclusive and nonnegative:
        return {
            "closed": True,
            "by": "closure rule, not a verdict",
            "u27_exception": u27_exception,
            "u27_exception_unreadable": u27_exception is None,
        }
    return {
        "closed": False,
        "by": "none",
        "u27_exception": u27_exception,
        "u27_exception_unreadable": u27_exception is None,
    }


def reachable_thresholds(*, n_seeds: int, level: float = LEVEL) -> list[str]:
    """Return the printed thresholds of section 7.2 from the monthly spread of `M1` minus `P` (0.10 at the six, 0.33 at the eight)."""
    t = float(stats.t.ppf(1.0 - (1.0 - level / 100.0) / 2.0, N_MONTHS - 1))
    lines = [
        f"Level {level:.2f}%, {N_MONTHS} month blocks, t = {t:.2f} with {N_MONTHS - 1} degrees of freedom, {n_seeds} seeds per arm"
    ]
    for g in GROUPS:
        half = t * MONTHLY_SD[g] / np.sqrt(N_MONTHS)
        delta, floor = DELTA[n_seeds][g], NOISE_FLOOR[n_seeds][g]
        lines.append(
            f"{g}: the t half-width is about {half:.2f}; a 'gain' needs D below about {-max(half, floor):.2f}; "
            + (
                f"'no detectable difference' needs the interval inside +-{delta} and so |D| below about {max(delta - half, 0):.2f}"
                if half < delta
                else f"'no detectable difference' is out of reach, because the interval ({half:.2f}) is wider than delta ({delta})"
            )
        )
    return lines


def self_test() -> int:
    """Check the rules on synthetic contrasts, including each verdict row and the closure rule."""

    def c(
        point: float, half: float = 0.05, *, t: bool = True, seeds: bool = True, four: bool = True
    ) -> dict:
        return {
            "point": point,
            "lower": point - half,
            "upper": point + half,
            "t_guard": t,
            "seed_guard": seeds,
            "seed_guard_4_of_5": four,
        }

    g5 = lambda x, grp="six": group_reading(c=x, group=grp, n_seeds=5)  # noqa: E731
    assert g5(c(0.30, 0.05)) == "clear loss"
    assert g5(c(0.04, 0.03)) == "loss"
    assert g5(c(0.04, 0.03, t=False)) != "loss"
    assert g5(c(0.03, 0.02)) != "loss"  # below the noise floor of 0.034
    assert g5(c(-0.20, 0.05)) == "gain"
    assert g5(c(-0.20, 0.05, seeds=False)) == "small gain"
    assert g5(c(-0.20, 0.05, seeds=False, four=False)) == "cannot tell"
    assert g5(c(0.0, 0.03)) == "no detectable difference"
    assert g5(c(0.0, 0.5)) == "cannot tell"
    assert group_reading(c=c(0.0, 0.08), group="six", n_seeds=2) == "no detectable difference"
    assert group_reading(c=c(0.0, 0.08), group="six", n_seeds=5) == "cannot tell"
    base = {
        "placebo_upper": {"six": -0.1, "eight": -0.1},
        "placebo_point": {"six": -0.2, "eight": -0.2},
        "placebo_vs_star": {"six": "gain", "eight": "gain"},
        "guards_hold": True,
    }

    def v(readings: tuple[str, str], **kw) -> str:  # noqa: ANN003
        return verdict(readings={"six": readings[0], "eight": readings[1]}, **{**base, **kw})  # ty: ignore[invalid-argument-type]

    assert v(("clear loss", "gain")) == "refuted"
    assert v(("loss", "gain")) == "not supported"
    assert v(("no detectable difference",) * 2) == "not supported"  # ty: ignore[invalid-argument-type]
    assert v(("gain", "gain")) == "supported"
    assert v(("gain", "gain"), placebo_upper={"six": 0.1, "eight": -0.1}) == "partly supported"
    assert (
        v(("gain", "gain"), placebo_vs_star={"six": "loss", "eight": "gain"}) == "partly supported"
    )
    assert v(("gain", "gain"), guards_hold=False) == "partly supported"
    assert v(("gain", "cannot tell")) == "partly supported"
    assert v(("small gain", "cannot tell")) == "small gain"
    assert (
        v(("small gain", "cannot tell"), placebo_point={"six": 0.1, "eight": -0.2})
        == "inconclusive"
    )
    assert v(("cannot tell", "cannot tell")) == "inconclusive"
    assert v(("small gain", "gain")) == "partly supported"
    assert stop_rule(
        first_contrast={"six": 0.2, "eight": 0.2}, monthly={g: np.zeros(4) for g in GROUPS}
    )
    assert stop_rule(
        first_contrast={"six": 0.12, "eight": 0.12},
        monthly={g: np.array([1.0, 1, 1, -1]) for g in GROUPS},
    )
    assert not stop_rule(
        first_contrast={"six": 0.12, "eight": 0.12},
        monthly={g: np.array([1.0, 1, -1, -1]) for g in GROUPS},
    )
    assert not stop_rule(
        first_contrast={"six": 0.2, "eight": 0.1}, monthly={g: np.ones(4) for g in GROUPS}
    )
    assert audit_trigger_1(placebo_point={"six": 0.1, "eight": 0.1}) and not audit_trigger_1(
        placebo_point={"six": 0.1, "eight": -0.1}
    )
    assert u27_advances(
        u27_minus_u27a={"six": -0.1, "eight": -0.1}, u27_minus_star={"six": 0.0, "eight": -0.1}
    )
    assert not u27_advances(
        u27_minus_u27a={"six": -0.1, "eight": 0.1}, u27_minus_star={"six": 0.0, "eight": -0.1}
    )
    pts = {"part1": {"six": 0.01, "eight": 0.0}, "part2": {"six": 0.0, "eight": 0.02}}
    assert line_closure(
        verdicts={"part1": "inconclusive", "part2": "inconclusive"},
        first_points=pts,
        u27_exception=None,
    )["by"].startswith("closure rule")
    neg = {**pts, "part2": {"six": -0.01, "eight": 0.02}}
    assert not line_closure(
        verdicts={"part1": "inconclusive", "part2": "inconclusive"},
        first_points=neg,
        u27_exception=None,
    )["closed"]
    assert line_closure(
        verdicts={"part1": "not supported", "part2": "refuted"},
        first_points=pts,
        u27_exception=False,
    )["closed"]
    assert not line_closure(
        verdicts={"part1": "not supported", "part2": "refuted"},
        first_points=pts,
        u27_exception=True,
    )["closed"]
    assert line_closure(
        verdicts={"part1": "not supported", "part2": "refuted"},
        first_points=pts,
        u27_exception=None,
    )["u27_exception_unreadable"]
    guards = supported_guards(
        sign_holds_leave_primary=True,
        sign_holds_leave_month=True,
        tails_ok=True,
        mae_moves_with_crps=True,
        cold_night_difference=0.06,
        swap_keeps_sign=True,
    )
    assert not guards["all_hold"] and not guards["cold_nights_within_0.05"]
    assert not supported_guards(
        sign_holds_leave_primary=True,
        sign_holds_leave_month=True,
        tails_ok=True,
        mae_moves_with_crps=True,
        cold_night_difference=0.0,
        swap_keeps_sign=None,
    )["all_hold"]
    lines = reachable_thresholds(n_seeds=2)
    assert ("out of reach" in lines[2] and "0.01" in lines[1]) or "0.0" in lines[1]
    print("\n".join(lines))
    print(
        "e00718_decision: all checks passed (group readings, the six verdict rows, the stop, audit, advance, and closure rules)"
    )
    return 0


def main() -> int:
    """Run the self-test, or print the thresholds."""
    command = sys.argv[1] if len(sys.argv) > 1 else "self-test"
    if command == "thresholds":
        for n in (5, 2):
            print("\n".join(reachable_thresholds(n_seeds=n)))
        return 0
    return self_test()


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: E501, PLR0911
"""Experiment 696: the pre-registered verdict table, as a pure function.

The table is the decision rule of `result.md`, first branch that applies:

1. `invalid` if a leakage test fails.
2. `failed` if the reproduction gate G0, the identity control, the liveness check, or an alignment test
   cannot be met.
3. `not supported`, with audit trigger 5 fired, if C1 passes and C2 does not.
4. `inconclusive` if C1 and C2 pass and M1 fails.
5. `supported` if C1 passes with an estimate of -0.16 or beyond, C2 passes, M1 passes, the C3 intervals
   do not all lie above zero, the normalised MAE of the median at D8 does not rise beyond its 95%
   interval, and no other Primary moves for the worse.
6. `partly supported` if C1 passes at -0.16 or beyond, C2 and M1 pass, and another condition of branch
   5 fails.
7. `small gain` if C1 passes with an estimate short of -0.16 and C2 and M1 pass.
8. `refuted` if all three C1 intervals lie above zero.
9. `not supported` in every other case.

A contrast "passes" when all three of its decision intervals lie below zero.
"""

from dataclasses import dataclass
from typing import Final

BRANCH_NAMES: Final[dict[int, str]] = {
    1: "invalid",
    2: "failed",
    3: "not supported",
    4: "inconclusive",
    5: "supported",
    6: "partly supported",
    7: "small gain",
    8: "refuted",
    9: "not supported",
}


@dataclass(frozen=True)
class VerdictInputs:
    """Everything the verdict table reads."""

    leakage_failed: bool
    gate_failed: bool
    """G0, the identity control, the liveness check, or an alignment test failed."""
    c1_passes: bool
    c1_estimate: float
    """Points of capacity, negative is a gain."""
    c1_all_above_zero: bool
    c2_passes: bool
    m1_passes: bool
    c3_all_above_zero: bool
    mae_rises: bool
    other_primary_worse: bool
    threshold: float
    """The smallest gain worth porting, a positive number of points of capacity."""


def verdict(*, v: VerdictInputs) -> tuple[int, str, bool]:
    """Return the branch, the verdict word, and whether audit trigger 5 fires, for the first branch that applies."""
    if v.leakage_failed:
        return 1, BRANCH_NAMES[1], False
    if v.gate_failed:
        return 2, BRANCH_NAMES[2], False
    if v.c1_passes and not v.c2_passes:
        return 3, BRANCH_NAMES[3], True
    if v.c1_passes and v.c2_passes and not v.m1_passes:
        return 4, BRANCH_NAMES[4], False
    if v.c1_passes and v.c2_passes and v.m1_passes:
        if v.c1_estimate > -v.threshold:
            return 7, BRANCH_NAMES[7], False
        clean = not v.c3_all_above_zero and not v.mae_rises and not v.other_primary_worse
        return (5, BRANCH_NAMES[5], False) if clean else (6, BRANCH_NAMES[6], False)
    if v.c1_all_above_zero:
        return 8, BRANCH_NAMES[8], False
    return 9, BRANCH_NAMES[9], False

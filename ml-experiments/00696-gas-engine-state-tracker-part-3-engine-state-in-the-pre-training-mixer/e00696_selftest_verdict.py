"""Every branch of experiment 696's verdict table, on synthetic inputs."""

from dataclasses import replace

import pytest
from e00696_verdict import VerdictInputs, verdict

BASE = VerdictInputs(
    leakage_failed=False,
    gate_failed=False,
    c1_passes=True,
    c1_estimate=-0.2,
    c1_all_above_zero=False,
    c2_passes=True,
    m1_passes=True,
    c3_all_above_zero=False,
    mae_rises=False,
    other_primary_worse=False,
    threshold=0.16,
)

CASES = {
    "branch 1 beats every later branch": (
        replace(BASE, leakage_failed=True, gate_failed=True),
        (1, "invalid", False),
    ),
    "branch 2: a gate failed": (replace(BASE, gate_failed=True), (2, "failed", False)),
    "branch 3: C1 passes, the placebo does too": (
        replace(BASE, c2_passes=False),
        (3, "not supported", True),
    ),
    "branch 3 comes before branch 4": (
        replace(BASE, c2_passes=False, m1_passes=False),
        (3, "not supported", True),
    ),
    "branch 4: no measured mechanism": (replace(BASE, m1_passes=False), (4, "inconclusive", False)),
    "branch 5: everything holds": (BASE, (5, "supported", False)),
    "branch 5 at the threshold": (replace(BASE, c1_estimate=-0.16), (5, "supported", False)),
    "branch 6: the host arm with the true state is worse": (
        replace(BASE, c3_all_above_zero=True),
        (6, "partly supported", False),
    ),
    "branch 6: the MAE rises": (replace(BASE, mae_rises=True), (6, "partly supported", False)),
    "branch 6: another Primary moves for the worse": (
        replace(BASE, other_primary_worse=True),
        (6, "partly supported", False),
    ),
    "branch 7: a gain short of the threshold": (
        replace(BASE, c1_estimate=-0.1),
        (7, "small gain", False),
    ),
    "branch 8: all three C1 intervals above zero": (
        replace(BASE, c1_passes=False, c1_estimate=0.3, c1_all_above_zero=True),
        (8, "refuted", False),
    ),
    "branch 9: an interval that includes zero": (
        replace(BASE, c1_passes=False, c1_estimate=-0.05),
        (9, "not supported", False),
    ),
    "branch 9: C1 fails and C2 and M1 pass": (
        replace(BASE, c1_passes=False, c1_estimate=-0.3),
        (9, "not supported", False),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_verdict_branch(name: str) -> None:
    """Each branch of the table is taken by the inputs that define it."""
    inputs, expected = CASES[name]
    assert verdict(v=inputs) == expected


def test_every_branch_is_covered() -> None:
    """All nine branches appear among the cases."""
    assert {expected[0] for _, expected in CASES.values()} == set(range(1, 10))

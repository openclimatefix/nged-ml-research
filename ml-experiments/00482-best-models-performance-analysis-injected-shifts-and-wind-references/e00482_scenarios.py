# ruff: noqa: E501
"""Experiment 482: the eight injected scenarios, the lag spread, and the shift arrays.

A scenario shifts one **donor** Primary's power by `sign * k * spread` (capacity share) over `days`
days that start 72 hours into a test month. A load shift moves only the **lags** (the target stays
the clean actual). A genuine level change moves the lags **and** the target. A mirrored scenario adds
the load-conserving offset `-step * capacity_donor / n / capacity_recipient` to up to two of the
donor's neighbours over the same half-hours, exactly as experiment 462's `add_mirrors` does. The
**lag spread** of a Primary is the standard deviation of its 24-hour change in power (48
half-hours) over the clean rows before 2025-10-01, in fractions of effective capacity.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np

START_OFFSET_STEPS: Final[int] = 72 * 2
"""The event starts 72 hours into the test month."""
SPREAD_BEFORE: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
MAX_RECIPIENTS: Final[int] = 2
SCENARIO_SEED: Final[int] = 482


@dataclass(frozen=True)
class Scenario:
    """One injected scenario."""

    name: str
    k: float
    days: int
    sign: int
    mirror: bool
    genuine: bool


SCENARIOS: Final[tuple[Scenario, ...]] = (
    Scenario("S1: +1x, 1 day, no mirror", 1.0, 1, +1, False, False),
    Scenario("S2: -3x, 1 day, no mirror", 3.0, 1, -1, False, False),
    Scenario("S3: +1x, 14 days, mirror", 1.0, 14, +1, True, False),
    Scenario("S4: -3x, 14 days, no mirror", 3.0, 14, -1, False, False),
    Scenario("S5: +3x, 14 days, mirror", 3.0, 14, +1, True, False),
    Scenario("S6: -1x, 14 days, no mirror", 1.0, 14, -1, False, False),
    Scenario("S7: +3x, 1 day, mirror", 3.0, 1, +1, True, False),
    Scenario("S8: +2x, 3 days, genuine level change", 2.0, 3, +1, False, True),
)


def lag_spread(*, power: np.ndarray) -> np.ndarray:
    """Return each Primary's standard deviation of the 24-hour change before `SPREAD_BEFORE`."""
    stop = int(common.grid_index(times=SPREAD_BEFORE))
    change = power[:, 48:stop] - power[:, : stop - 48]
    return np.nanstd(change, axis=1)


def event_window(*, month: str) -> tuple[int, int]:
    """Return the first grid index of a month and of the day the event starts (placeholder)."""
    start = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
    return int(common.grid_index(times=start)), START_OFFSET_STEPS


def shift_array(
    *, scenario: Scenario, month: str, donor: int, spread: np.ndarray, capacity: np.ndarray,
    neighbours: list[list[int]], shape: tuple[int, int],
) -> tuple[np.ndarray, int, int, list[int]]:  # fmt: skip
    """Return `(shift, a, b, recipients)`: the (primaries, grid) capacity-share shift and its window."""
    first, offset = event_window(month=month)
    a = first + offset
    b = a + scenario.days * 48
    shift = np.zeros(shape, dtype=np.float64)
    step = scenario.sign * scenario.k * float(spread[donor])
    shift[donor, a:b] = step
    recipients: list[int] = []
    if scenario.mirror and neighbours[donor]:
        rng = np.random.default_rng(SCENARIO_SEED + 1000 * donor + int(scenario.name[1]))
        n = min(MAX_RECIPIENTS, len(neighbours[donor]))
        recipients = [int(q) for q in rng.choice(neighbours[donor], size=n, replace=False)]
        for q in recipients:
            shift[q, a:b] += -step * capacity[donor] / n / capacity[q]
    return shift.astype(np.float32), a, b, recipients

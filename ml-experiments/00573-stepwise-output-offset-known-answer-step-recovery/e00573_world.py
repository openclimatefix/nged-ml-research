# ruff: noqa: E501
"""Experiment 573: the known-answer worlds Ks (steps), Kr (ramp), and K0 (clean).

A world adds a known component, a fraction of capacity, to the cleaned power of six Primaries. The
grid starts on 2019-09-01 at 00:00 UTC with 48 half-hours a day, so `grid index // 48` is the day
index used by every offset. The truth of a world is the list of its change points per Primary, as
`(day index, signed size in capacity fraction)`; a transient step has an up change point and a down
change point.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np

INJECTED: Final[tuple[int, ...]] = (0, 2, 8, 5, 7, 6)
"""Primary indices injected: D2, D4, D12 (MVA-only) and D9, D11, D10 (signed-MW), the same six as the sibling known-answer experiment."""
MVA_INJECTED: Final[frozenset[int]] = frozenset({0, 2, 8})
WORLDS: Final[tuple[str, ...]] = ("Ks", "Kr", "K0")
GRID_DAY0: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
PER_DAY: Final[int] = 48
STEP_RANGE: Final[tuple[float, float]] = (0.02, 0.06)
RAMP_SIZE: Final[float] = 0.03


def day_of(*, t: datetime) -> int:
    """Return the day index (days since 2019-09-01) of a time."""
    return (t - GRID_DAY0).days


def draw_steps(*, primary: int) -> list[tuple[int, float]]:
    """Return a Primary's change points in world Ks, sorted by day.

    Events start between 2022-01-01 and 2026-05-31 with at least 30 days between the end of one
    event and the start of the next. About half the events are transient (they revert after 5 to
    60 days, which adds a second change point of the opposite sign); the rest persist.
    """
    rng = np.random.default_rng([573, primary])
    day = day_of(t=datetime(2022, 1, 1, tzinfo=UTC)) + int(rng.integers(0, 120))
    last = day_of(t=datetime(2026, 5, 31, tzinfo=UTC))
    points: list[tuple[int, float]] = []
    while day <= last:
        size = float(rng.uniform(*STEP_RANGE) * rng.choice([-1.0, 1.0]))
        points.append((day, size))
        end = day
        if rng.random() < 0.5:
            end = day + int(rng.integers(5, 61))
            points.append((end, -size))
        day = end + 30 + int(rng.integers(0, 330))
    return points


def ramp_points(*, primary: int) -> tuple[int, int, float]:
    """Return a Primary's ramp in world Kr: the first day, the last day, and the signed end size."""
    rng = np.random.default_rng([5730, primary])
    start = day_of(t=datetime(2023, 11, 1, tzinfo=UTC)) + int(rng.integers(0, 330))
    return start, start + 365, float(RAMP_SIZE * rng.choice([-1.0, 1.0]))


def truth_steps(*, world: str) -> dict[int, list[tuple[int, float]]]:
    """Return the change points per injected Primary (empty for Kr and K0)."""
    if world != "Ks":
        return {}
    return {p: draw_steps(primary=p) for p in INJECTED}


def injection(*, world: str, grid: int) -> np.ndarray:
    """Return the world's added component (16 Primaries, `grid` half-hours), a fraction of capacity."""
    out = np.zeros((16, grid), dtype=np.float32)
    days = np.arange(grid) // PER_DAY
    if world == "Ks":
        for p, points in truth_steps(world="Ks").items():
            for day, size in points:
                out[p, days >= day] += size
    elif world == "Kr":
        for p in INJECTED:
            first, last, size = ramp_points(primary=p)
            frac = np.clip((np.arange(grid) / PER_DAY - first) / (last - first), 0.0, 1.0)
            out[p] = (size * frac).astype(np.float32)
    elif world != "K0":
        raise ValueError(world)
    return out


def offset_years(*, start_day: int, end_day: int) -> float:
    """Return the length of a span of days in years."""
    return (end_day - start_day) / 365.25


def _selftest() -> None:
    steps = truth_steps(world="Ks")
    assert set(steps) == set(INJECTED)
    for p, points in steps.items():
        days = [d for d, _ in points]
        assert days == sorted(days), p
        assert all(b - a >= 5 for a, b in zip(days, days[1:], strict=False)), p
        print(p, len(points), "change points, first", GRID_DAY0 + timedelta(days=days[0]))
    inj = injection(world="Kr", grid=2300 * PER_DAY)
    assert abs(inj[0, -1]) == RAMP_SIZE


if __name__ == "__main__":
    _selftest()

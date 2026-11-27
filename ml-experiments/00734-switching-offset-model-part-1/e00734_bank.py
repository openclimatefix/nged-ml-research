"""The world bank of experiment 734, and the guard that keeps scoring seeds from tuning."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

TUNING_SEEDS: Final[tuple[int, ...]] = tuple(range(734000, 734030))
"""Thirty tuning worlds, with the power of the scored years hidden."""
TIMED_SEED: Final[int] = 734000
"""The tuning world that the timed run decodes."""
SCORED_DEFAULT_SEEDS: Final[tuple[int, ...]] = tuple(range(900, 940))
CASCADE_SEEDS: Final[tuple[int, ...]] = tuple(range(734100, 734120))
COLD_SEEDS: Final[tuple[int, ...]] = tuple(range(734200, 734220))
SENSITIVITY_SEEDS: Final[tuple[int, ...]] = tuple(range(1000, 1010))
BIAS_SEEDS: Final[tuple[int, ...]] = tuple(1000 + 17 * k + len("seasonal_bias") for k in range(2))
"""Seeds of experiment 653's two seasonal-bias worlds."""


def solar_seeds() -> tuple[int, ...]:
    """Seeds of experiment 686's 24 solar-growth worlds."""
    from e00686_false_events import N_WORLDS, seed_of  # ty: ignore[unresolved-import]

    return tuple(seed_of(k=k) for k in range(N_WORLDS))


def off_limits(*, extra: Iterable[int] = ()) -> frozenset[int]:
    """Every seed of a scored, cascade, cold-spell, sensitivity, solar, or bias world.

    Args:
        extra: Seeds to add, such as the solar-growth seeds when experiment 686 is not importable.

    Returns:
        The seeds that the tuning code must never decode.
    """
    return frozenset(
        {
            *SCORED_DEFAULT_SEEDS,
            *CASCADE_SEEDS,
            *COLD_SEEDS,
            *SENSITIVITY_SEEDS,
            *BIAS_SEEDS,
            *extra,
        }
    )


def guard_tuning(*, seeds: Iterable[int], extra: Iterable[int] = ()) -> tuple[int, ...]:
    """Return `seeds` if every one is a tuning seed, else raise.

    Args:
        seeds: The seeds a tuning job is about to decode.
        extra: Further off-limits seeds (the solar-growth worlds).

    Returns:
        The seeds as a tuple.

    Raises:
        ValueError: If a seed is off-limits or is not one of `TUNING_SEEDS`.
    """
    seeds = tuple(seeds)
    banned = off_limits(extra=extra)
    for s in seeds:
        if s in banned or s not in TUNING_SEEDS:
            raise ValueError(f"seed {s} is not a tuning seed and must not reach the tuning code")
    return seeds

"""Synthetic inputs that the tests of experiment 722 share (no real data, no `torch`)."""

import e00722_core as core
import numpy as np


def make_weights(*, n_days: int, n_boot: int = 300, seed: int = 1) -> np.ndarray:
    """Return a master draw of whole days with fewer replicates than the real job."""
    return core.day_weights(n_days=n_days, n_boot=n_boot, rng=np.random.default_rng(seed))


def calendar(*, n_days: int) -> dict[str, np.ndarray]:
    """Return matched keys (one month, one day type), the day of each row, and an all-quiet mask."""
    n = n_days * core.HALF_HOURS_PER_DAY
    rows = np.arange(n)
    key = core.matched_key(
        local_half_hour=rows % core.HALF_HOURS_PER_DAY,
        month=np.ones(n, dtype=int),
        day_type=np.zeros(n, dtype=int),
    )
    return {"key": key, "day": rows // core.HALF_HOURS_PER_DAY, "quiet": np.ones(n, dtype=bool)}


def block_flow(*, n: int, rng: np.random.Generator, block: int = 7) -> np.ndarray:
    """Return a signed flow of constant blocks (levels of either sign) plus small noise.

    Most hourly changes are noise, so the robust noise scale is small and the block steps are
    large; steps between levels of opposite sign are the crossing hours.
    """
    levels = rng.choice([-4.0, -2.0, 2.0, 4.0], size=n // block + 1)
    return np.repeat(levels, block)[:n] + rng.normal(0, 0.02, n)

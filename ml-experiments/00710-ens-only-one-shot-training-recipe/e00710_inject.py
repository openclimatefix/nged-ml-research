"""Synthetic switching events for training: one world per (fold, network seed).

**A world is the observed power of the 16 Primaries with random lag-only events added.** The
generator is `synthetic_clean` of experiment 581 (`e00581_bank.py`), unchanged: for each Primary it
draws events until 5% of the half-hours are covered, with a length of 3, 7, 15, or 30 days, a depth
of 0.15, 0.30, or 0.50 of capacity, a random sign, and a random start. The shift goes to the power
that feeds the three lag inputs and never to the target.

**The world of fold `YYYY-MM` and seed `s` has the seed `7_100_000 + 1_000 * s + YYYYMM`.** It spans
the grid start to the refit minus 3.5 hours, so no half-hour at or after the refit changes. Every
stage and epoch of every `i` arm with that fold and seed reads the world, so the arms stay paired.
The `SEED_RANGES` of experiment 699 run from 202,511 to 9,202,606, and none overlaps the range of
this experiment, 7,302,410 to 7,306,606.

**The lag columns are rebuilt from the world's power** with `e00707_common.origin_lags`. The
held-out rows keep their unshifted lags.
"""

from dataclasses import dataclass
from functools import cache
from typing import Final

import e00581_bank as bank581  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SEED_BASE: Final[int] = 7_100_000
SEED_STRIDE: Final[int] = 1_000
LAG_NAMES: Final[tuple[str, ...]] = ("lag_origin", "lag_day", "lag_week")
OTHER_SEED_RANGES: Final[dict[str, tuple[int, int]]] = {
    "stage3": (202511, 202606),
    "bank660": (862511, 862606),
    "stage2": (2202511, 2202606),
    "stage1": (5202511, 5202606),
    "bank581": (9202511, 9202606),
}
"""Experiment 699's `SEED_RANGES`, copied so the new seeds can be checked against them."""


@dataclass
class World:
    """The shifted power of every Primary for one (fold, seed).

    Attributes:
        month: The fold, such as `2026-01`.
        seed: The network seed.
        world_seed: The seed of the event generator.
        power: Each Primary's power over capacity on the dense grid (float32), by site label.
        clean: Each Primary's observed power, by site label (the unshifted twin).
        n_events: The number of events drawn.
        last_index: The grid index at which the world stops (the refit minus 3.5 hours).
    """

    month: str
    seed: int
    world_seed: int
    power: dict[str, np.ndarray]
    clean: dict[str, np.ndarray]
    n_events: int
    last_index: int


def world_seed(*, month: str, seed: int) -> int:
    """Return `7_100_000 + 1_000 * seed + YYYYMM`.

    Args:
        month: The fold, such as `2026-01`.
        seed: The network seed.

    Returns:
        The generator's seed.
    """
    return SEED_BASE + SEED_STRIDE * seed + int(month.replace("-", ""))


def assert_seed_free(*, value: int) -> None:
    """Assert that a world seed lies outside every range that experiment 699 used.

    Args:
        value: A world seed.

    Raises:
        ValueError: If the seed lies inside one of experiment 699's ranges.
    """
    for name, (low, high) in OTHER_SEED_RANGES.items():
        if low <= value <= high:
            msg = f"world seed {value} lies inside the range of {name}"
            raise ValueError(msg)


def observed_power() -> dict[str, np.ndarray]:
    """Return each Primary's observed power over capacity on the grid, by site label.

    Returns:
        The float32 arrays of `common.series_and_arrays`, NaN where unobserved.
    """
    _series, arrays = common.series_and_arrays()
    return {
        row["site"]: arrays[row["site"]].power for row in prep.primaries().iter_rows(named=True)
    }


def build_world(*, month: str, seed: int) -> World:
    """Return the (fold, seed) world of synthetic events.

    Args:
        month: The fold, such as `2026-01`; the world ends 3.5 hours before its refit.
        seed: The network seed.

    Returns:
        The world with its shifted power.
    """
    sites = prep.primaries()["site"].to_list()
    clean = observed_power()
    stacked = np.stack([clean[s] for s in sites])
    last = int(
        common.grid_index(times=prep.refit_of(month=month) - common.REFIT_CUTOFF_AGE)  # ty: ignore[invalid-argument-type]
    )
    value = world_seed(month=month, seed=seed)
    assert_seed_free(value=value)
    shifted, n_events = bank581.synthetic_clean(clean=stacked, first=0, last=last, seed=value)
    return World(
        month=month,
        seed=seed,
        world_seed=value,
        power={s: shifted[i] for i, s in enumerate(sites)},
        clean=clean,
        n_events=n_events,
        last_index=last,
    )


def lag_column_index() -> list[int]:
    """Return the column indices of the three lag values in the raw input matrix."""
    names = prep.feature_names()[0]
    return [names.index(n) for n in LAG_NAMES]


def relag(
    *,
    x: np.ndarray,
    valid_us: np.ndarray,
    origin_us: np.ndarray,
    power: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy of the raw rows with the shift of the world added to the three lag values.

    Each lag value becomes the cached value plus (the lag of `power` minus the lag of `observed`),
    where that difference is finite and zero otherwise. A world equal to the observed power
    therefore reproduces the cached rows bit for bit, whatever built the cache, and a lag that
    reads a missing observation keeps its cached value. Only the three lag value columns change.
    The lag ages, the missing flags, and the target keep their values, because the shift is added
    to the power at the same times.

    Args:
        x: The raw (unstandardised) input matrix of one Primary's rows.
        valid_us: Valid times in microseconds since the epoch.
        origin_us: Origin times in microseconds since the epoch.
        power: The Primary's power on the grid with the world's shift added.
        observed: The Primary's observed power on the grid.

    Returns:
        The rebuilt matrix and a boolean mask of the rows whose lag values changed.
    """
    start = int(common.GRID_START.timestamp() * 1e6)
    target = ((valid_us - start) // 1_800_000_000).astype(np.int64)
    origin = ((origin_us - start) // 1_800_000_000).astype(np.int64)
    delta = common.origin_lags(power=power, target=target, origin=origin) - common.origin_lags(
        power=observed, target=target, origin=origin
    )
    delta = np.where(np.isfinite(delta), delta, 0.0).astype(x.dtype)
    columns = lag_column_index()
    out = x.copy()
    out[:, columns] = x[:, columns] + delta
    changed = np.any(delta != 0, axis=1)
    return out, changed


def shift_rows(
    *,
    x: np.ndarray,
    valid_us: np.ndarray,
    origin_us: np.ndarray,
    held: np.ndarray,
    power: np.ndarray | None,
    observed: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw rows to train on and the mask of rows that carry a shifted lag.

    With no world (`power` is `None`), the rows are returned unchanged and the mask is all false.
    With a world, the held-out rows keep their unshifted lags and every other row takes the shift
    of the world's power.

    Args:
        x: The raw input matrix of one Primary's rows.
        valid_us: Valid times in microseconds since the epoch.
        origin_us: Origin times in microseconds since the epoch.
        held: Whether each row is in a held-out week.
        power: The Primary's shifted power, or `None` for the `n` state.
        observed: The Primary's observed power; given whenever `power` is given.

    Returns:
        The rows (the input array itself if `power` is `None`) and the shifted-lag mask.
    """
    if power is None or observed is None:
        return x, np.zeros(len(x), dtype=bool)
    rebuilt, changed = relag(
        x=x, valid_us=valid_us, origin_us=origin_us, power=power, observed=observed
    )
    rebuilt[held] = x[held]
    return rebuilt, changed & ~held


def injected_share(
    *,
    shifted: np.ndarray,
    primary: np.ndarray,
    weight: np.ndarray | None,
    n_primaries: int,
) -> dict[str, float | list[float]]:
    """Return the unweighted and weighted share of rows with a shifted lag, overall and per Primary.

    Args:
        shifted: Whether each training row carries a shifted lag.
        primary: Each row's Primary index.
        weight: Each row's weight, or `None` for all ones.
        n_primaries: The number of Primaries.

    Returns:
        The two shares over all rows, and the two shares of each Primary (NaN for a Primary with no
        row).
    """
    w = np.ones(len(shifted)) if weight is None else weight.astype(np.float64)
    s = shifted.astype(np.float64)
    per_unweighted, per_weighted = [], []
    for p in range(n_primaries):
        rows = primary == p
        n, total = float(rows.sum()), float(w[rows].sum())
        per_unweighted.append(float(s[rows].sum() / n) if n else float("nan"))
        per_weighted.append(float((w[rows] * s[rows]).sum() / total) if total else float("nan"))
    return {
        "unweighted": float(s.mean()),
        "weighted": float((w * s).sum() / w.sum()),
        "unweighted_per_primary": per_unweighted,
        "weighted_per_primary": per_weighted,
    }


@cache
def real_events_table() -> list[dict]:
    """Return the verified real events of experiment 580 as per-Primary rows.

    Returns:
        One row per (transfer, Primary side) with `primary`, `onset_idx`, `end_idx`, and `depth`.
    """
    site_map = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    table = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00580-verified-load-shift-event-set-and-label-audit/work/events_verified.parquet"
    )
    return bank581.verified_events(
        capacity_mw=dict(
            zip(site_map["label"].to_list(), site_map["capacity_mw"].to_list(), strict=True)
        ),
        label_to_primary=dict(
            zip(site_map["label"].to_list(), site_map["primary"].to_list(), strict=True)
        ),
        table=table,
    )

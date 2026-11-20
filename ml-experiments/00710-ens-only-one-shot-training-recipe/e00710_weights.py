"""Recency weights for experiment 710, as pure functions of a row's age at the refit.

**A row's age is the refit time minus the row's valid time, in days** (`e00707_folds.build_fold`,
lines 305 to 306), so `exponential(half_life_years=0.5)` reproduces experiment 707's fine-tune
weight `0.5 ** (age / 182.625)` bit for bit. A weight never reads the target, the weather, or any
row after the refit.

**The loss is normalised inside each batch** as `(w * nll).sum() / w.sum()` (`weighted_mean_nll`),
so multiplying every weight by a constant changes no loss value.

The rules are: the exponential weight, the hard window (a row mask), the exponential weight with a
floor, a half-life that changes with the epoch (`SCHEDULES`), a half-life per Primary with each
Primary's share of the total weight fixed (`per_primary`), and the placebo (`permuted_age_days`),
which gives each row the age of a row from another calendar month.
"""

import math
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]

US_PER_DAY: Final[int] = 86_400 * 1_000_000
HALF_HOUR_US: Final[int] = 1_800_000_000
DAYS_PER_YEAR: Final[float] = 365.25
"""The year length that experiment 707 uses for its half-life (`0.5 * 365.25` days)."""
INFINITE_HALF_LIFE: Final[float] = math.inf
"""A half-life of infinity gives every row a weight of one."""

SCHEDULES: Final[dict[str, tuple[float, ...]]] = {
    "S": (INFINITE_HALF_LIFE, 8.0, 4.0, 2.0, 1.0, 0.5),
    "CYC": (INFINITE_HALF_LIFE,) * 4 + (2.0,) * 4 + (0.5,) * 4,
}
"""The half-life in years of each epoch (index 0 is the first epoch); the last value repeats."""

PLACEBO_SEED_BASE: Final[int] = 8_100_000
"""The placebo's permutation seed is this plus the fold as `YYYYMM`."""


def age_days(*, valid_us: np.ndarray, refit_us: int) -> np.ndarray:
    """Return each row's age at the refit in days, as float64.

    Args:
        valid_us: Valid times in microseconds since the epoch.
        refit_us: The refit time in microseconds since the epoch.

    Returns:
        The refit time minus the valid time, in days.
    """
    return (refit_us - valid_us) / US_PER_DAY


def exponential(*, age: np.ndarray, half_life_years: float) -> np.ndarray:
    """Return `0.5 ** (age / half_life)` as float32; all ones for an infinite half-life.

    Args:
        age: Row ages in days.
        half_life_years: The half-life in years.

    Returns:
        One weight per row.
    """
    if math.isinf(half_life_years):
        return np.ones(age.shape, dtype=np.float32)
    return (0.5 ** (age / (half_life_years * DAYS_PER_YEAR))).astype(np.float32)


def floor_exponential(*, age: np.ndarray, half_life_years: float, floor: float) -> np.ndarray:
    """Return `floor + (1 - floor) * 0.5 ** (age / half_life)` as float32.

    Args:
        age: Row ages in days.
        half_life_years: The half-life in years.
        floor: The weight that an infinitely old row keeps.

    Returns:
        One weight per row.
    """
    decay = exponential(age=age, half_life_years=half_life_years).astype(np.float64)
    return (floor + (1.0 - floor) * decay).astype(np.float32)


def within_window(*, age: np.ndarray, window_days: float) -> np.ndarray:
    """Return the rows of a hard window: age below `window_days` (the rows have weight one).

    Args:
        age: Row ages in days.
        window_days: The window length in days.

    Returns:
        A boolean mask; the rows outside it are dropped, not down-weighted.
    """
    return age < window_days


def schedule_half_life(*, schedule: str, epoch: int) -> float:
    """Return the half-life in years that a schedule uses in an epoch (0-based).

    Args:
        schedule: A key of `SCHEDULES`.
        epoch: The epoch index, from 0.

    Returns:
        The half-life in years; the last entry of the schedule repeats after its end.
    """
    values = SCHEDULES[schedule]
    return values[min(epoch, len(values) - 1)]


def per_primary(
    *,
    age: np.ndarray,
    primary: np.ndarray,
    half_lives_years: np.ndarray,
    counted: np.ndarray,
) -> np.ndarray:
    """Return a weight with each Primary's own half-life and its share of the total fixed.

    Each Primary's weights are rescaled so that the Primary's share of the total weight over the
    `counted` rows equals its share of the `counted` rows (its share under equal weights).

    Args:
        age: Row ages in days.
        primary: Each row's Primary index.
        half_lives_years: The half-life of each Primary, indexed by Primary; infinity is allowed.
        counted: The rows (the training rows) that define the shares.

    Returns:
        One float32 weight per row.
    """
    raw = np.ones(age.shape, dtype=np.float64)
    for p, h in enumerate(half_lives_years):
        rows = primary == p
        raw[rows] = exponential(age=age[rows], half_life_years=float(h)).astype(np.float64)
    out = raw.copy()
    for p in range(len(half_lives_years)):
        in_p = counted & (primary == p)
        total = raw[in_p].sum()
        if total > 0:
            out[primary == p] = raw[primary == p] * (in_p.sum() / total)
    return out.astype(np.float32)


def permuted_age_days(
    *, valid_us: np.ndarray, refit_us: int, floor_us: int, fold_yyyymm: int
) -> np.ndarray:
    """Return the ages after permuting them among whole calendar months (the placebo).

    The calendar months (UTC) of the rows whose month starts at or after `floor_us` are permuted
    with a generator seeded by `PLACEBO_SEED_BASE + fold_yyyymm`. A row of month `j` takes the valid
    time of month `pi(j)`'s first half-hour plus the row's offset inside month `j`, clipped to the
    last half-hour of month `pi(j)` and to the refit; its age is the refit minus that valid time.
    A row of the partial first month (before `floor_us`) keeps its own age. Every Primary shares
    the permutation, because it depends on valid times only.

    Args:
        valid_us: Valid times in microseconds since the epoch.
        refit_us: The refit time in microseconds since the epoch.
        floor_us: The window floor; a month that starts before it is not permuted.
        fold_yyyymm: The fold as an integer such as 202601.

    Returns:
        The permuted ages in days, float64, with the shape of `valid_us`.
    """
    month = valid_us.astype("datetime64[us]").astype("datetime64[M]").astype(np.int64)
    starts = month.astype("datetime64[M]").astype("datetime64[us]").astype(np.int64)
    unique = np.unique(month[starts >= floor_us])
    rng = np.random.default_rng(PLACEBO_SEED_BASE + fold_yyyymm)
    target_month = unique[rng.permutation(len(unique))]
    new_valid = valid_us.astype(np.int64).copy()
    for m, t in zip(unique.tolist(), target_month.tolist(), strict=True):
        rows = month == m
        t_start = int(np.datetime64(t, "M").astype("datetime64[us]").astype(np.int64))
        t_end = int(np.datetime64(t + 1, "M").astype("datetime64[us]").astype(np.int64))
        offset = valid_us[rows] - int(starts[rows][0])
        moved = np.minimum(t_start + offset, t_end - HALF_HOUR_US)
        new_valid[rows] = np.minimum(moved, refit_us)
    kept = starts < floor_us
    new_valid[kept] = valid_us[kept]
    return age_days(valid_us=new_valid, refit_us=refit_us)


def weighted_mean_nll(*, nll: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Return the batch loss `(w * nll).sum() / w.sum()`.

    Args:
        nll: The negative log-likelihood of each row of the batch.
        w: The weight of each row of the batch.

    Returns:
        The weighted mean, a scalar that does not change when `w` is multiplied by a constant.
    """
    return (w * nll).sum() / w.sum()


def effective_rows(*, w: torch.Tensor) -> torch.Tensor:
    """Return the effective number of rows of a weight vector, `(sum w) ** 2 / sum(w ** 2)`.

    Args:
        w: The weights of a batch or of a whole stage.

    Returns:
        A scalar tensor, computed in float64.
    """
    w64 = w.double()
    return w64.sum() ** 2 / (w64**2).sum()

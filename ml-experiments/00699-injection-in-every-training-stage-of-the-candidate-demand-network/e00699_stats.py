# ruff: noqa: E501
"""Experiment 699: the two intervals of every contrast, on sums by seed and origin month.

Every statistic of the experiment is a ratio of two sums, taken over the origin months and seeds
that a draw selects. A per-row loss has the same denominator for every seed (the number of shared
rows in the month). The copied share of a shifted lag has the shift's sum of squares as the
denominator. Both are therefore a pair of arrays of shape (months, seeds): `num` and `den`.

- `bootstrap_combination`: 50,000 draws. Each draw resamples the months with replacement (the same
  months for both arms), then inside each drawn month resamples the seeds with replacement, afresh
  for every month. With `paired_seeds=False` (family H: the arms share no weights) the two arms
  draw their seeds independently. With `paired_seeds=True` (families S2 and S3: seed s of one arm
  and seed s of the other arm share weights) one draw of seeds serves both arms.
- `t_combination`: a row-weighted t interval on the monthly differences, with months as the units.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats

N_BOOT: Final = 50_000
CHUNK: Final = 5_000


@dataclass(frozen=True)
class Interval:
    """A contrast's point estimate and its two intervals at one level (percent)."""

    point: float
    boot_lo: float
    boot_hi: float
    t_lo: float
    t_hi: float
    level: float

    def below_zero(self) -> bool:
        """Return whether both intervals lie entirely below zero."""
        return self.boot_hi < 0.0 and self.t_hi < 0.0

    def above_zero(self) -> bool:
        """Return whether both intervals lie entirely above zero."""
        return self.boot_lo > 0.0 and self.t_lo > 0.0

    def includes_zero(self) -> bool:
        """Return whether both intervals include zero."""
        return self.boot_lo <= 0.0 <= self.boot_hi and self.t_lo <= 0.0 <= self.t_hi

    def upper_below(self, *, bound: float) -> bool:
        """Return whether both upper bounds are below `bound`."""
        return self.boot_hi < bound and self.t_hi < bound

    def lower_above(self, *, bound: float) -> bool:
        """Return whether both lower bounds are above `bound`."""
        return self.boot_lo > bound and self.t_lo > bound


def pooled(*, num: np.ndarray, den: np.ndarray) -> float:
    """Return the ratio of the sums over every month and seed."""
    return float(num.sum() / den.sum())


@dataclass(frozen=True)
class Term:
    """One arm's sums in a linear combination of arms.

    `sign` is +1 or -1. Terms with the same `group` share one draw of seeds in the bootstrap (the arms
    share weights seed by seed); terms in different groups draw their seeds independently.
    """

    num: np.ndarray
    den: np.ndarray
    sign: float
    group: int


def t_combination(*, terms: list[Term], level: float) -> tuple[float, float, float]:
    """Return the centre and the t interval of the signed sum of the terms' statistics.

    The month's statistic of a term is the ratio of its sums over the seeds, and the month's weight
    is its share of the first term's denominator.

    Args:
        terms: The arms, each with its sign. All hold arrays of shape (months, seeds).
        level: Coverage in percent.

    Returns:
        The centre, the lower bound, and the upper bound.
    """
    months = terms[0].num.shape[0]
    d = sum(t.sign * t.num.sum(axis=1) / t.den.sum(axis=1) for t in terms)
    w = terms[0].den.sum(axis=1) / terms[0].den.sum()
    centre = float((w * d).sum())
    if months < 2:
        return centre, float("nan"), float("nan")
    se = float(np.sqrt(months / (months - 1) * ((w * (d - centre)) ** 2).sum()))
    crit = float(stats.t.ppf(0.5 + level / 200.0, months - 1))
    return centre, centre - crit * se, centre + crit * se


def bootstrap_combination(
    *,
    terms: list[Term],
    level: float,
    resample_seeds: bool = True,
    seed: int = 699,
    n_boot: int = N_BOOT,
) -> tuple[float, float]:
    """Return the bootstrap interval of the signed sum of the terms' pooled statistics.

    Each draw resamples the months with replacement (the same months for every term), then inside each
    drawn month resamples the seeds with replacement, afresh for every month and for every group of
    terms. With `resample_seeds=False` the seeds stay fixed and only the months are resampled.
    """
    months, seeds = terms[0].num.shape
    assert all(t.num.shape == (months, seeds) for t in terms), [t.num.shape for t in terms]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for start in range(0, n_boot, CHUNK):
        size = min(CHUNK, n_boot - start)
        month = rng.integers(0, months, size=(size, months))[:, :, None]
        picks: dict[int, np.ndarray] = {}
        total = np.zeros(size)
        for t in terms:
            if t.group not in picks:
                picks[t.group] = (
                    rng.integers(0, seeds, size=(size, months, seeds))
                    if resample_seeds
                    else np.broadcast_to(np.arange(seeds), (size, months, seeds))
                )
            pick = picks[t.group]
            total += t.sign * (
                t.num[month, pick].sum(axis=(1, 2)) / t.den[month, pick].sum(axis=(1, 2))
            )
        draws[start : start + size] = total
    tail = (100.0 - level) / 200.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return float(lo), float(hi)


def contrast(
    *,
    num_a: np.ndarray,
    den_a: np.ndarray,
    num_b: np.ndarray,
    den_b: np.ndarray,
    level: float,
    paired_seeds: bool,
    resample_seeds: bool = True,
    n_boot: int = N_BOOT,
) -> Interval:
    """Return arm b minus arm a with both intervals at `level`."""
    terms = [Term(num_a, den_a, -1.0, 0), Term(num_b, den_b, 1.0, 0 if paired_seeds else 1)]
    point = pooled(num=num_b, den=den_b) - pooled(num=num_a, den=den_a)
    boot_lo, boot_hi = bootstrap_combination(
        terms=terms, level=level, resample_seeds=resample_seeds, n_boot=n_boot
    )
    centre, t_lo, t_hi = t_combination(terms=terms, level=level)
    assert np.isclose(centre, point, rtol=1e-6, atol=1e-9), (centre, point)
    return Interval(point, boot_lo, boot_hi, t_lo, t_hi, level)

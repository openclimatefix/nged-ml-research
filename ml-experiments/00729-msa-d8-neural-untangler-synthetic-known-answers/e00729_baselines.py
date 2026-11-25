"""Experiment 729: the baselines P0, P0f, B1, B2, and the wrapper of experiment 694's M2 tracker.

P0 reads the reading as signed import. P0f gives P0's point the neural networks' lobe-width floor.
B1 and B2 use the fact that injected output is never negative: if base demand D is below the
reading R, then X = +R is impossible, so P(X < 0) = pi * P(D < R), where pi is the injected share
of the world being scored. B1 estimates P(D < R) from a month-by-half-hour climatology of the
training base days, and B2 from a linear regression of base demand on calendar and temperature.
"""

from dataclasses import dataclass
from typing import Final

import e00729_core as core
import e00729_dist as dist
import numpy as np

N_LOCAL_HH: Final = 48


@dataclass(frozen=True)
class Features:
    """The per-half-hour features that the baselines read.

    Attributes:
        r: The reading in points.
        local_hh: Local half-hour of the day, `0..47`.
        local_month: Local calendar month, `1..12`.
        weekend: Saturday or Sunday.
        weekday: Day of the week, `0..6`.
        holiday: Bank holiday.
        t2m: ERA5 2 m temperature, degrees Celsius (NaN allowed).
    """

    r: np.ndarray
    local_hh: np.ndarray
    local_month: np.ndarray
    weekend: np.ndarray
    weekday: np.ndarray
    holiday: np.ndarray
    t2m: np.ndarray


def reading_to_import(*, r: np.ndarray) -> np.ndarray:
    """Return `sqrt(max(R^2 - Qhat^2, 0))`, P0's point forecast of the signed power."""
    return np.sqrt(np.maximum(r**2 - core.QHAT_POINTS**2, 0.0))


def point_lobes(*, xhat: np.ndarray, spread: float, p: np.ndarray | None = None) -> dist.Lobes:
    """Return lobes at `+xhat` (import) and `-xhat` (reverse flow) with a given spread."""
    n = len(xhat)
    return dist.Lobes(
        p=np.zeros(n) if p is None else p,
        mu_i=xhat,
        s_i=np.full(n, spread),
        mu_r=-xhat,
        s_r=np.full(n, spread),
    )


def p0(*, f: Features) -> dist.Lobes:
    """P0: the reading read as signed import, as a point distribution."""
    return point_lobes(xhat=reading_to_import(r=f.r), spread=dist.POINT_SPREAD)


def p0f(*, f: Features) -> dist.Lobes:
    """P0f: P0's point forecast with the neural networks' 0.5-point lobe-width floor."""
    return point_lobes(xhat=reading_to_import(r=f.r), spread=core.FLOOR_POINTS)


@dataclass(frozen=True)
class Climatology:
    """Sorted base-demand samples in each (local month, local half-hour, weekend) cell.

    Attributes:
        cells: The sorted samples of each cell, indexed by `cell_id`.
    """

    cells: dict[int, np.ndarray]

    @staticmethod
    def cell_id(*, month: np.ndarray, hh: np.ndarray, weekend: np.ndarray) -> np.ndarray:
        """Return the integer id of each (month, half-hour, weekend) cell."""
        return (month.astype(int) * N_LOCAL_HH + hh.astype(int)) * 2 + weekend.astype(int)

    @classmethod
    def fit(cls, *, demand: np.ndarray, f: Features) -> "Climatology":
        """Fit the climatology on training base demand (points) with its features."""
        ids = cls.cell_id(month=f.local_month, hh=f.local_hh, weekend=f.weekend)
        ok = np.isfinite(demand)
        cells = {int(i): np.sort(demand[ok & (ids == i)]) for i in np.unique(ids[ok])}
        return cls(cells=cells)

    def prob_below(self, *, r: np.ndarray, f: Features) -> np.ndarray:
        """Return `P(D < R)` per row from the empirical CDF (with a half-count continuity term)."""
        ids = self.cell_id(month=f.local_month, hh=f.local_hh, weekend=f.weekend)
        out = np.full(len(r), 0.5)
        for i in np.unique(ids):
            sample = self.cells.get(int(i))
            if sample is None or len(sample) == 0:
                continue
            sel = ids == i
            out[sel] = (np.searchsorted(sample, r[sel], side="left") + 0.5) / (len(sample) + 1.0)
        return out


def design_matrix(*, f: Features, t_mean: float) -> np.ndarray:
    """Return the regression design: half-hour one-hot, weekday dummies, holiday, temperature."""
    n = len(f.r)
    hh = np.zeros((n, N_LOCAL_HH))
    hh[np.arange(n), f.local_hh.astype(int)] = 1.0
    dow = np.zeros((n, 6))
    for d in range(1, 7):
        dow[:, d - 1] = f.weekday == d
    t = np.nan_to_num(f.t2m, nan=t_mean) - t_mean
    return np.column_stack([hh, dow, f.holiday.astype(float), t, t**2 / 10.0])


@dataclass(frozen=True)
class Regression:
    """A linear regression of base demand with a Gaussian residual.

    Attributes:
        coef: The coefficients of `design_matrix`.
        sd: The residual standard deviation in points.
        t_mean: The temperature mean used to centre the design.
    """

    coef: np.ndarray
    sd: float
    t_mean: float

    @classmethod
    def fit(cls, *, demand: np.ndarray, f: Features) -> "Regression":
        """Fit by least squares on the rows where the demand is finite."""
        ok = np.isfinite(demand)
        t_mean = float(np.nanmean(f.t2m)) if np.isfinite(f.t2m).any() else 0.0
        x = design_matrix(f=f, t_mean=t_mean)[ok]
        coef, *_ = np.linalg.lstsq(x, demand[ok], rcond=None)
        resid = demand[ok] - x @ coef
        return cls(coef=coef, sd=float(max(resid.std(), 1e-3)), t_mean=t_mean)

    def prob_below(self, *, r: np.ndarray, f: Features) -> np.ndarray:
        """Return `P(D < R)` under the Gaussian regression."""
        mean = design_matrix(f=f, t_mean=self.t_mean) @ self.coef
        return dist.norm_cdf(z=(r - mean) / self.sd)


def sign_prior_lobes(*, f: Features, prob_below: np.ndarray, pi: float) -> dist.Lobes:
    """B1 or B2: lobes at plus and minus R' with `P(X < 0) = pi * P(D < R)` and the floor spread."""
    return point_lobes(
        xhat=reading_to_import(r=f.r),
        spread=core.FLOOR_POINTS,
        p=np.clip(pi * prob_below, 0.0, 1.0),
    )


def injected_share(*, mix: dict[str, float]) -> float:
    """Return the share of segments that carry an injection in a world mix."""
    return 1.0 - mix["none"]


def m2_p_any(*, reading_points: np.ndarray) -> np.ndarray:
    """Return experiment 694's M2 smoothed probability that any engine unit runs, per half-hour.

    Args:
        reading_points: A continuous half-hourly series of folded readings in points.

    Returns:
        P(an engine unit runs) for each half-hour, from `grid_track` with the causal spread.
    """
    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_trackers as t694  # ty: ignore[unresolved-import]

    y = reading_points / core.POINTS * core.CAPACITY_MVA
    lvl = c694.quantise_spread(spread=c694.causal_spread(y=y))
    return np.asarray(t694.grid_track(y=y, lvl=lvl).p_any_smooth)


def m2_lobes(*, f: Features, prob_below: np.ndarray, p_any: np.ndarray) -> dist.Lobes:
    """M2 as a baseline: `P(X < 0) = P(engine runs) * P(D < R)`, lobes at plus and minus R'."""
    return point_lobes(
        xhat=reading_to_import(r=f.r),
        spread=core.FLOOR_POINTS,
        p=np.clip(p_any * prob_below, 0.0, 1.0),
    )

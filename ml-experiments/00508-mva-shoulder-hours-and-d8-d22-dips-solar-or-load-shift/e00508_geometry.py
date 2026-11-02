"""Day-by-half-hour matrices with solar elevation, farm injection, and the Part A windows.

Experiment 508. The shoulder windows are defined by the sun's elevation (cloud-free), split by
whether the sun is rising or setting, and restricted to April to September. Elevation limits are
calibrated once so that the clear-sky fleet shape `u` of experiment 412 spans 0.05 to 0.3.
"""

from datetime import date
from typing import Final

import e00506_data as data  # ty: ignore[unresolved-import]
import e00508_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from pvlib import solarposition

MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
MID_SLOTS: Final[slice] = slice(20, 30)
"""UK-clock 10:00 to 15:00."""
MIN_HH: Final[dict[str, int]] = {"midday": 6, "morning": 3, "evening": 3, "day": 40}
DARK_ELEVATION: Final[float] = -6.0


class Mats:
    """One site's complete UK days (exactly 48 half-hours) as (days x 48) matrices."""

    def __init__(self, *, site: str, extra: dict[str, np.ndarray] | None = None) -> None:
        """Reshape the grid into complete days and add the sun's elevation."""
        frame = common.base_frame(site=site)
        dates = frame["date"].to_numpy()
        ids, first, counts = np.unique(dates, return_index=True, return_counts=True)
        keep = counts == 48
        rows = (first[keep][:, None] + np.arange(48)[None, :]).astype(int)
        self.site = site
        self.dates = ids[keep]
        self.ordinal = np.array(
            [
                date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100).toordinal()
                for d in self.dates
            ]
        )
        self.weekend = np.array([date.fromordinal(int(o)).weekday() >= 5 for o in self.ordinal])
        self.p = frame["p"].to_numpy()[rows]
        self.u = frame["u"].to_numpy()[rows]
        self.k = frame["k"].to_numpy()[rows]
        self.demand = frame["demand"].to_numpy()[rows]
        self.before_cut = frame["before_cut"].to_numpy()[rows]
        self.extra = {n: a[rows] for n, a in (extra or {}).items()}
        self.year = (self.dates // 10000).astype(int)
        self.month = (self.dates // 100 % 100).astype(int)
        self.block = self.year * 100 + self.month
        self.el = elevation(site=site)[rows]
        self.rising = np.gradient(self.el, axis=1) > 0

    def in_months(self) -> np.ndarray:
        """Return the days in April to September."""
        return np.isin(self.month, MONTHS)


def elevation(*, site: str) -> np.ndarray:
    """Return the sun's apparent elevation (degrees) at each half-hour midpoint on the grid."""
    path = common.WORK / f"elevation_{site}.npy"
    if path.exists():
        return np.load(path)
    row = data.series_table().filter(pl.col("site") == site).row(0, named=True)
    mid = data.grid().dt.offset_by("-15m").to_pandas()
    out = solarposition.get_solarposition(mid, float(row["latitude"]), float(row["longitude"]))
    values = out["apparent_elevation"].to_numpy().astype(np.float64)
    common.WORK.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    return values


def calibrate_shoulder(*, mats: Mats) -> tuple[float, float]:
    """Return the elevations (degrees) at which the clear-sky fleet shape is 0.05 and 0.3.

    Uses rising-sun half-hours of April to September with a clear-sky index of at least 0.9.
    """
    ok = mats.in_months()[:, None] & (mats.k >= 0.9) & mats.rising
    lo = float(np.median(mats.el[ok & (np.abs(mats.u - 0.05) < 0.01)]))
    hi = float(np.median(mats.el[ok & (np.abs(mats.u - 0.30) < 0.02)]))
    return lo, hi


def farm_extension_factor(*, site: str, mats: Mats) -> float:
    """Return the median farm-to-`u` ratio where `u` is 0.3 to 0.5 and the farm is on."""
    farm = mats.extra["farm"]
    ok = (farm > 0) & (mats.u >= 0.3) & (mats.u <= 0.5)
    return float(np.median(farm[ok] / mats.u[ok]))


def signed(*, mats: Mats, level: float) -> tuple[np.ndarray, float]:
    """Return signed power with the injected farm and the share of shoulder half-hours extended.

    signed = p - level x median|p| x farm. The recorded farm shape is zero where the sun is below
    15 degrees (zenith at least 75), which is where the shoulders sit, so there the farm shape is
    the site's own fleet shape `u` times the farm-to-`u` ratio measured at `u` of 0.3 to 0.5.
    """
    p = mats.p
    if level == 0 or mats.site not in data.FARM_OF_SITE:
        return p.copy(), 0.0
    farm = mats.extra["farm"].copy()
    factor = farm_extension_factor(site=mats.site, mats=mats)
    extend = (farm == 0.0) & (mats.u > 0.005)
    farm = np.where(extend, factor * mats.u, farm)
    sunny_mid = np.zeros_like(p, dtype=bool)
    sunny_mid[:, MID_SLOTS] = True
    before = (
        mats.before_cut
        & sunny_mid
        & (mats.k >= data.SUNNY_K)
        & np.isfinite(p)
        & np.isin(mats.month, range(4, 10))[:, None]
    )
    scale = level * float(np.median(np.abs(p[before])))
    return p - scale * farm, float(extend.mean())


def readings(*, mats: Mats, sig: np.ndarray) -> dict[str, np.ndarray]:
    """Return the floor-0 and power-factor-0.97 readings (raw `|p|` for the eight)."""
    if mats.site in data.MVA_ONLY:
        raw = np.abs(sig)
        return {"Q0": raw, "PF097": raw}
    return {
        "Q0": np.abs(sig),
        "PF097": np.sqrt(sig**2 + (common.TAN_PHI_097 * np.abs(mats.demand)) ** 2),
    }


def window_masks(*, mats: Mats, band: tuple[float, float]) -> dict[str, np.ndarray]:
    """Return each window's (days x 48) mask: midday, morning, evening, and the whole day."""
    shoulder = (mats.el >= band[0]) & (mats.el <= band[1])
    mid = np.zeros_like(shoulder)
    mid[:, MID_SLOTS] = True
    return {
        "midday": mid,
        "morning": shoulder & mats.rising,
        "evening": shoulder & ~mats.rising,
        "day": np.ones_like(shoulder),
    }

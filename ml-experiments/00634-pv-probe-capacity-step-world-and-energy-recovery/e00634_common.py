# ruff: noqa: E501
"""Shared pieces of experiment 634: paths, the injected conditions (constant and step), and the probe frame.

An injected condition changes the target of the pretraining rows of the six signed-MW Primaries by
a known shape in megawatts. Net demand falls by the shape for PV and rises for a cloud-following
load. All conditions share one row mask, so that an injected arm and its uninjected twin train on
the same rows.
"""

import re
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00634-pv-probe-capacity-step-world-and-energy-recovery"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
CUTOFFS: Final[tuple[str, ...]] = ("2025-07", "2025-10", "2026-03", "2026-06")
"""The spring cutoff and the 2026-06 cutoff (development data only; July 2026 onwards untouched)."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SIX: Final[tuple[str, ...]] = pv.SIGNED_SIX
FARMS: Final[str] = "ABCF"
"""Metered solar farms used as injected shapes: D has 14% and E 24 to 78% daylight gaps, so both are left out.

The six Primaries take farms A, B, C, F, A, B in order, shifted by the rotation."""
DAY_US: Final[int] = 86_400_000_000
CLOUD_SHARE_WEAK: Final[float] = 0.10
CLOUD_SHARE_STRONG: Final[float] = 0.30
"""Mean cloud load over daylight half-hours, as a share of the 2 MW injected capacity.

0.10 is experiment 129's constant. 0.30 is a stronger load, because the 0.10 load may be too weak
to test the 0.15 bar.
"""
STEP_STARTS: Final[dict[str, str]] = {"spring": "2025-04-01", "summer": "2025-07-01"}
"""First day of each capacity step. The pretraining rows start on 2024-04-01, so each step has a year without it before."""
CONDITIONS: Final[tuple[str, ...]] = (
    "control",
    "step_spring_mw2",
    "step_spring_mw4",
    "step_summer_mw2",
    "step_summer_mw4",
    "shufstep_spring_mw4",
    "shufstep_summer_mw4",
)
PATTERN: Final = re.compile(
    r"(?P<kind>farm|phys|profile|shuffled)_mw(?P<mw>\d+)(_r(?P<rot>\d))?"
    r"|(?P<cloud>cloudlin|cloudsq)_(?P<size>weak|strong)"
    r"|(?P<step>step|shufstep)_(?P<month>spring|summer)_mw(?P<stepmw>\d+)"
)


@dataclass(frozen=True)
class Condition:
    """A parsed condition name."""

    name: str
    kind: str  # control, farm, phys, profile, shuffled, cloud
    mw: float
    rotation: int
    squared: bool
    share: float
    start_us: int = 0


def parse(*, name: str) -> Condition:
    """Parse a condition name such as `farm_mw4_r1` or `cloudsq_strong`."""
    if name == "control":
        return Condition(name, "control", 0.0, 0, squared=False, share=0.0)
    match = PATTERN.fullmatch(name)
    assert match is not None, name
    if match.group("step"):
        start = datetime.fromisoformat(STEP_STARTS[match.group("month")]).replace(tzinfo=UTC)
        kind = "farm" if match.group("step") == "step" else "shuffled"
        return Condition(
            name, kind, float(match.group("stepmw")), 0, squared=False, share=0.0,
            start_us=int(start.timestamp() * 1e6),
        )  # fmt: skip
    if match.group("cloud"):
        share = CLOUD_SHARE_STRONG if match.group("size") == "strong" else CLOUD_SHARE_WEAK
        return Condition(
            name, "cloud", 2.0, 0, squared=match.group("cloud") == "cloudsq", share=share
        )
    return Condition(
        name, match.group("kind"), float(match.group("mw")), int(match.group("rot") or 0),
        squared=False, share=0.0,
    )  # fmt: skip


def weights_path(*, cutoff: str, condition: str, seed: int) -> Path:
    """Return where a pretrained network is stored."""
    return WORK / "pretrained" / cutoff / f"{condition}_s{seed}.pt"


def frame_path(*, cutoff: str) -> Path:
    """Return where a cutoff's probe frame is stored."""
    return WORK / "frames" / f"{cutoff}.npz"


def six_indices() -> dict[str, int]:
    """Return the row-index of each signed-MW Primary in the 16-Primary order."""
    sites = [r["site"] for r in prep.primaries().iter_rows(named=True)]
    return {s: sites.index(s) for s in SIX}


def effective_capacity_mw() -> np.ndarray:
    """Return the effective capacity (MW) of every Primary, in the 16-Primary order."""
    return np.array(
        [float(r["effective_capacity_mw"]) for r in prep.primaries().iter_rows(named=True)]
    )


def haurwitz_w_m2(*, cos_zenith: np.ndarray) -> np.ndarray:
    """Return clear-sky global horizontal irradiance by the Haurwitz model (zero below the horizon)."""
    cz = np.clip(cos_zenith, 0.0, 1.0)
    return np.where(cz > 0.02, 1098.0 * cz * np.exp(-0.059 / np.maximum(cz, 0.02)), 0.0)


def raw_weather(*, x: np.ndarray, standardiser: object) -> tuple[np.ndarray, np.ndarray]:
    """Return (global horizontal irradiance in W/m2, 2 m temperature in degrees C) from standardised x."""
    mean, std = standardiser.mean, standardiser.std  # ty: ignore[unresolved-attribute]
    ghi = x[:, 0].astype(np.float64) * std[0] + mean[0]
    temp = x[:, 2].astype(np.float64) * std[2] + mean[2]
    return np.clip(ghi, 0.0, None), temp


def time_us(*, target_idx: np.ndarray) -> np.ndarray:
    """Return the valid time in microseconds since the epoch."""
    return c.prep422.GRID_START_US + target_idx.astype(np.int64) * c.prep422.STEP_US


def unit_curve(
    *, ghi: np.ndarray, x: np.ndarray, times_us: np.ndarray, temp: np.ndarray
) -> np.ndarray:
    """Return the physics unit curve (AC output per MW of capacity at its 99th percentile)."""
    cos_zenith = x[:, 14].astype(np.float64)
    doy = pl.Series(times_us).cast(pl.Datetime("us")).dt.ordinal_day().to_numpy().astype(np.float64)
    mean, _ = pv.unit_shape(
        ghi_mean=ghi, ghi_sd=np.zeros_like(ghi), cos_zenith=cos_zenith,
        sin_azimuth=x[:, 15].astype(np.float64), cos_azimuth=x[:, 16].astype(np.float64),
        doy=doy, temperature_c=temp,
    )  # fmt: skip
    return mean / pv.u99()


class Injector:
    """Builds the injected change of the target, in points of effective capacity, for a row set."""

    def __init__(
        self, *, primary: np.ndarray, target_idx: np.ndarray, x: np.ndarray, standardiser: object
    ) -> None:
        """Hold the rows' keys and the physical quantities every condition needs."""
        self.primary, self.target_idx, self.x = primary, target_idx, x
        self.t_us = time_us(target_idx=target_idx)
        self.six = six_indices()
        self.cap = effective_capacity_mw()
        self.in_six = np.isin(primary, list(self.six.values()))
        self.ghi, self.temp = raw_weather(x=x, standardiser=standardiser)
        self.series = {f: p5.farm_norm_series(farm=f) for f in FARMS}
        self.unique, self.inverse = np.unique(self.t_us, return_inverse=True)
        self._cache: dict[tuple, np.ndarray] = {}
        self.gap = self._gap_mask()

    def _gap_mask(self) -> np.ndarray:
        """Return True on rows of the six Primaries where any farm A to F has a gap (held or shuffled)."""
        gap = np.zeros(len(self.unique), dtype=bool)
        for f in FARMS:
            for salt in (None, 0):
                gap |= np.isnan(
                    p9.series_at(times_us=self.unique, series=self.series[f], shuffle_salt=salt)
                )
        return gap[self.inverse] & self.in_six

    def farm_series(self, *, farm: str, kind: str) -> np.ndarray:
        """Return a farm's normalised series at the unique times (held, profile, or shuffled)."""
        key = (farm, kind)
        if key not in self._cache:
            unique = self.unique
            series = self.series[farm]
            profile, anomaly = p9.split_profile_anomaly(series=series)
            if kind == "farm":
                v = p9.series_at(times_us=unique, series=series, shuffle_salt=None)
            elif kind == "profile":
                v = p9.series_at(times_us=unique, series=profile, shuffle_salt=None)
            else:
                v = p9.series_at(times_us=unique, series=profile, shuffle_salt=None) + p9.series_at(
                    times_us=unique, series=anomaly, shuffle_salt=0
                )
            self._cache[key] = v
        return self._cache[key]

    def cloud_response(self, *, squared: bool) -> np.ndarray:
        """Return 129's day-long cloud response (mean over the day's daylight half-hours), per row."""
        cz = self.x[:, 14].astype(np.float64)
        clear = haurwitz_w_m2(cos_zenith=cz)
        daylight = clear > 10.0
        cloud = np.where(daylight, np.clip(1.0 - self.ghi / np.maximum(clear, 1e-9), 0.0, 1.0), 0.0)
        # The grid step is local; use the UTC calendar day of the valid time.
        day = self.t_us // DAY_US
        frame = pl.DataFrame(
            {
                "p": self.primary,
                "d": day,
                "t": self.target_idx,
                "cloud": cloud,
                "day_light": daylight,
            }
        ).unique(subset=["p", "t"], keep="first")
        daily = frame.filter(pl.col("day_light")).group_by(["p", "d"]).agg(r=pl.col("cloud").mean())
        joined = pl.DataFrame({"p": self.primary, "d": day}).join(daily, on=["p", "d"], how="left")
        response = np.where(daylight, joined["r"].fill_null(0.0).to_numpy(), 0.0)
        return response**2 if squared else response

    def change_points(self, *, condition: Condition) -> np.ndarray:
        """Return the change to add to the target, in points (zero outside the six Primaries)."""
        n = len(self.primary)
        change = np.zeros(n, dtype=np.float64)
        if condition.kind == "control":
            return change
        for j, site in enumerate(SIX):
            sel = self.primary == self.six[site]
            inc = 1.0 / self.cap[self.six[site]]
            if condition.kind in {"farm", "profile", "shuffled"}:
                farm = FARMS[(j + condition.rotation) % len(FARMS)]
                shape = self.farm_series(farm=farm, kind=condition.kind)[self.inverse[sel]]
                after = (self.t_us[sel] >= condition.start_us).astype(np.float64)
                change[sel] = -condition.mw * inc * shape * after
            elif condition.kind == "phys":
                shape = unit_curve(
                    ghi=self.ghi[sel], x=self.x[sel], times_us=self.t_us[sel], temp=self.temp[sel]
                )
                change[sel] = -condition.mw * inc * shape
            else:
                response = self.cloud_response(squared=condition.squared)[sel]
                daylight = response > 0
                scale = condition.share * condition.mw / max(float(response[daylight].mean()), 1e-9)
                change[sel] = scale * response * inc
        return change

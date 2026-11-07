# ruff: noqa: E501
"""Shared pieces of experiment 638: the step conditions and the climatological placebo.

Experiment 634's injector, rows, frames, and control networks are reused unchanged. This module adds
a second family of injected shapes (steps that start in other years, and a placebo whose shape is the
farm's multi-year day-of-year mean) and the paths of this experiment's own networks and probes.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00638-pv-probe-step-placebo-climatological-and-ordinary-year-followup"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
CUTOFFS: Final[tuple[str, ...]] = k634.CUTOFFS
SEEDS: Final[tuple[int, ...]] = k634.SEEDS
STEP_STARTS: Final[dict[str, str]] = {
    "spring23": "2023-04-01",
    "summer24": "2024-07-01",
    "spring25": "2025-04-01",
    "summer25": "2025-07-01",
}
"""First day of each capacity step. `spring25` and `summer25` are experiment 634's two steps."""
REAL_IN_634: Final[dict[str, str]] = {"spring25": "step_spring_mw4", "summer25": "step_summer_mw4"}
"""Experiment 634's name for the real-farm steps that this experiment does not retrain."""
CONDITIONS: Final[tuple[str, ...]] = (
    "step_spring23_mw4",
    "step_summer24_mw4",
    "climstep_spring23_mw4",
    "climstep_summer24_mw4",
    "climstep_spring25_mw4",
    "climstep_summer25_mw4",
)
DEV_END_US: Final[int] = int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1e6)
"""Nothing dated 2026-07-01 or later enters the climatological profile."""
SMOOTH_DAYS: Final = 7
"""Half-width, in days of the year, of the window that smooths the profile."""
PATTERN: Final = re.compile(r"(?P<kind>step|climstep)_(?P<key>[a-z0-9]+)_mw(?P<mw>\d+)")


@dataclass(frozen=True)
class Step:
    """A parsed condition name."""

    name: str
    placebo: bool
    key: str
    mw: float
    start_us: int


def parse(*, name: str) -> Step:
    """Parse a condition name such as `climstep_spring23_mw4`."""
    match = PATTERN.fullmatch(name)
    assert match is not None, name
    start = datetime.fromisoformat(STEP_STARTS[match.group("key")]).replace(tzinfo=UTC)
    return Step(
        name=name, placebo=match.group("kind") == "climstep", key=match.group("key"),
        mw=float(match.group("mw")), start_us=int(start.timestamp() * 1e6),
    )  # fmt: skip


def weights_path(*, cutoff: str, condition: str, seed: int) -> Path:
    """Return where a pretrained network of this experiment is stored."""
    return WORK / "pretrained" / cutoff / f"{condition}_s{seed}.pt"


def probes_dir(*, shift_days: int) -> Path:
    """Return where this experiment's probe results are stored."""
    return WORK / ("probes" if shift_days == 0 else f"probes_shift{shift_days}")


def climatological_profile(
    *, series: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return a farm's day-of-year mean profile at the farm's own timestamps.

    The mean is taken per day of the year and half-hour over every year before 2026-07-01, ignoring
    gaps, then smoothed over `SMOOTH_DAYS` days either side (circularly). The profile carries the
    seasonal and daily shape of the farm and almost no information about the weather of any one
    date (one date moves its own profile value by under 2%).
    """
    t, v = series
    stamp = pl.Series(t).cast(pl.Datetime("us"))
    doy = stamp.dt.ordinal_day().to_numpy().astype(np.int64) - 1
    half_hour = (stamp.dt.hour().to_numpy() * 2 + stamp.dt.minute().to_numpy() // 30).astype(
        np.int64
    )
    ok = (t < DEV_END_US) & ~np.isnan(v)
    total = np.zeros((366, 48))
    count = np.zeros((366, 48))
    np.add.at(total, (doy[ok], half_hour[ok]), v[ok])
    np.add.at(count, (doy[ok], half_hour[ok]), 1.0)
    window_total = sum(np.roll(total, s, axis=0) for s in range(-SMOOTH_DAYS, SMOOTH_DAYS + 1))
    window_count = sum(np.roll(count, s, axis=0) for s in range(-SMOOTH_DAYS, SMOOTH_DAYS + 1))
    profile = np.where(window_count > 0, window_total / np.maximum(window_count, 1.0), 0.0)
    return t, profile[doy, half_hour]


class Injector(k634.Injector):
    """Experiment 634's injector with the steps and the climatological placebo of this experiment."""

    def farm_series(self, *, farm: str, kind: str) -> np.ndarray:
        """Return a farm's normalised series at the unique times; `clim` is the day-of-year mean."""
        if kind != "clim":
            return super().farm_series(farm=farm, kind=kind)
        key = (farm, kind)
        if key not in self._cache:
            profile = climatological_profile(series=self.series[farm])
            self._cache[key] = p9.series_at(times_us=self.unique, series=profile, shuffle_salt=None)
        return self._cache[key]

    def change_points_step(self, *, condition: Step) -> np.ndarray:
        """Return the change to add to the target, in points: the real farm or its climatology from the start day."""
        change = np.zeros(len(self.primary), dtype=np.float64)
        for j, site in enumerate(k634.SIX):
            sel = self.primary == self.six[site]
            farm = k634.FARMS[j % len(k634.FARMS)]
            shape = self.farm_series(farm=farm, kind="clim" if condition.placebo else "farm")
            after = (self.t_us[sel] >= condition.start_us).astype(np.float64)
            change[sel] = (
                -condition.mw / self.cap[self.six[site]] * shape[self.inverse[sel]] * after
            )
        return change


def real_condition(*, key: str) -> str:
    """Return the name of the real-farm step with a given start key, in whichever experiment trained it."""
    return REAL_IN_634.get(key, f"step_{key}_mw4")

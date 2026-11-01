"""Constants, shapes and injected series shared by the scripts of experiment 474.

Experiment 474 repeats experiment 269's injection test on the convex fit of experiment 129 with a
placebo that keeps the day-of-year profile and breaks the day-to-day weather link, and with a ladder
of solar shapes. The plan is in
`docs/ml-experiments/00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md`.

**Privacy.** The farms (labels A to F) are metered generators. Only ratios of recovered to injected
capacity and distance bands may leave the private data folder.

Run scripts with `PYTHONPATH` set to the folders of experiments 269 and 129.
"""

from dataclasses import replace
from pathlib import Path
from typing import Final

import e00129_common as common129
import e00269_common as c269
import numpy as np
from studies.dp_features import FeatureCache, SiteFeatures
from studies.pv_dataset import pv_sites
from studies.pv_physics import PlantParameters
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from studies.dp_features import unit_pv_shape, default_input_paths
import polars as pl

EXPERIMENT_ID: Final[int] = 474
SLUG: Final[str] = "pv-capacity-placebo-gated-ladder-and-per-primary-intervals"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files; private and never committed."""

INJECTION_MW: Final[tuple[float, ...]] = (2.0, 4.0)
"""Injected AC capacity in megawatts. 1 MW is below seed noise on the network base, so is dropped."""
N_PLACEBO_SHUFFLES: Final[int] = 20
"""Within-month day shuffles per placebo at a rung."""
PROFILE_HALF_WINDOW_DAYS: Final[int] = 7
"""The day-of-year profile is smoothed over this many days either side."""
TILT_PHYSICS_DEG: Final[float] = 20.0
"""Experiments 282 and 293: the metered farms fit best near this tilt."""
DC_AC_CENTRE: Final[float] = 1.25
DC_AC_LOW: Final[float] = 1.1
DC_AC_HIGH: Final[float] = 1.5
"""C2: a fixed assumption with a range, not a fit (experiment 73: the ratio is not identified)."""
GATE_LIMIT: Final[float] = 0.20
"""A rung's placebo 95th percentile must stay under this for the rung to pass."""
PASS_RECOVERY: Final[float] = 0.85
"""A rung's real raw recovery must reach this for the rung to pass."""
INTERVAL: Final[tuple[float, float]] = (0.1, 0.9)
"""The nominal 80% interval."""
SIGNED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""E is left out: its history is too short for the read-out season-year (experiment 269)."""
RUNGS: Final[tuple[str, ...]] = ("C0_pooled_table", "C1_physics_tilt20", "C1b_physics_tilt35")

_RESPONSE_CACHE: dict[tuple[str, str], object] = {}


def points_from_mw(*, cache: FeatureCache, site: str, mw: float) -> float:
    """Return a capacity in megawatts as points of the Primary's effective capacity."""
    return mw / cache.sites[site].table.effective_capacity_mw * 100.0


def pooled_shape(*, cache: FeatureCache, site: str, farms: tuple[str, ...]) -> np.ndarray:
    """Return experiment 269's pooled empirical shape of `farms`, with fits cached by (site, farm)."""
    features = cache.sites[site]
    shapes = []
    for farm in farms:
        key = (site, farm)
        if key not in _RESPONSE_CACHE:
            _RESPONSE_CACHE[key] = c269.learned_response(cache=cache, site=site, farm=farm)
        table = _RESPONSE_CACHE[key]
        shapes.append(
            table.predict(
                irradiance_w_m2=np.nan_to_num(features.ghi_w_m2),
                elevation_deg=features.elevation_deg,
            )
        )
    shape = np.mean(shapes, axis=0)
    shape[features.elevation_deg <= c269.RESPONSE_MIN_ELEVATION_DEG] = 0.0
    shape[~np.isfinite(features.ghi_w_m2)] = np.nan
    return shape


def physics_shape(*, cache: FeatureCache, site: str, tilt_deg: float) -> np.ndarray:
    """Return the physics unit shape (plane-of-array chain, clipped at the AC rating)."""
    cams = pl.read_parquet(default_input_paths().cams)
    plant = PlantParameters(
        tilt_deg=tilt_deg, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
    )
    return unit_pv_shape(
        cams=cams,
        table=cache.sites[site].table,
        grid=pl.Series(cache.times_series()),
        plant=plant,
    )


def rung_shape(*, cache: FeatureCache, site: str, rung: str, farm: str | None) -> np.ndarray:
    """Return a rung's unit shape; the pooled table leaves out the injected `farm` when given."""
    if rung == "C0_pooled_table":
        farms = tuple(f for f in c269.ALL_FARMS if f != farm)
        return pooled_shape(cache=cache, site=site, farms=farms)
    if rung == "C1_physics_tilt20":
        return physics_shape(cache=cache, site=site, tilt_deg=TILT_PHYSICS_DEG)
    if rung == "C1b_physics_tilt35":
        return cache.sites[site].pv_unit_shape
    raise ValueError(rung)


def with_shape(*, cache: FeatureCache, site: str, shape: np.ndarray, y: np.ndarray) -> FeatureCache:
    """Return the cache with the Primary's unit shape and target replaced."""
    features: SiteFeatures = replace(cache.sites[site], pv_unit_shape=shape, y=y)
    return replace(cache, sites={**cache.sites, site: features})


# --- profile and shuffled-anomaly placebo -------------------------------------------------------


def day_slot(*, cache: FeatureCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-half-hour day number (days since epoch), day of year, and half-hour slot (0-47)."""
    start = cache.end_times - np.timedelta64(30, "m")
    day = start.astype("datetime64[D]")
    day_number = day.astype(np.int64)
    doy = pl.Series(day).dt.ordinal_day().to_numpy().astype(np.int64) - 1
    slot = ((start - day) / np.timedelta64(30, "m")).astype(np.int64)
    return day_number, np.minimum(doy, 365), slot


def day_of_year_profile(*, cache: FeatureCache, series: np.ndarray) -> np.ndarray:
    """Return the series' mean by day of year and half-hour slot, smoothed, on the cache's grid."""
    _, doy, slot = day_slot(cache=cache)
    total = np.zeros((366, 48))
    count = np.zeros((366, 48))
    ok = np.isfinite(series)
    np.add.at(total, (doy[ok], slot[ok]), series[ok])
    np.add.at(count, (doy[ok], slot[ok]), 1.0)
    kernel_total = np.zeros_like(total)
    kernel_count = np.zeros_like(count)
    for shift in range(-PROFILE_HALF_WINDOW_DAYS, PROFILE_HALF_WINDOW_DAYS + 1):
        kernel_total += np.roll(total, shift, axis=0)
        kernel_count += np.roll(count, shift, axis=0)
    table = np.divide(kernel_total, kernel_count, out=np.zeros_like(total), where=kernel_count > 0)
    return table[doy, slot]


def shuffled_placebo(
    *, cache: FeatureCache, norm: np.ndarray, profile: np.ndarray, seed: int
) -> np.ndarray:
    """Return profile plus the anomaly of another day of the same calendar month.

    Days are permuted within each calendar year-month, so the placebo keeps the season and the
    time of day, and loses only the day-to-day weather link. A half-hour where `norm` is `NaN`
    stays `NaN`, so every placebo trains on the same rows as the real injection.
    """
    day_number, _, slot = day_slot(cache=cache)
    start = (cache.end_times - np.timedelta64(30, "m")).astype("datetime64[M]").astype(np.int64)
    anomaly = norm - profile
    first_day = day_number.min()
    n_days = day_number.max() - first_day + 1
    grid_anomaly = np.full((n_days, 48), np.nan)
    grid_anomaly[day_number - first_day, slot] = anomaly
    day_month = np.full(n_days, -1)
    day_month[day_number - first_day] = start
    mapping = np.arange(n_days)
    rng = np.random.default_rng(seed)
    for month in np.unique(day_month[day_month >= 0]):
        members = np.flatnonzero(day_month == month)
        mapping[members] = rng.permutation(members)
    shuffled = grid_anomaly[mapping[day_number - first_day], slot]
    shuffled = np.where(np.isfinite(shuffled), shuffled, np.nan_to_num(anomaly, nan=0.0))
    out = np.clip(profile + shuffled, 0.0, None)
    return np.where(np.isfinite(norm), out, np.nan)


def cloud_load(*, cache: FeatureCache, site: str, points: float) -> np.ndarray:
    """Return experiment 129's cloud-driven load at the injected capacity."""
    return common129.cloud_load(cache=cache, site=site, capacity_points=points)


def farm_distance_band(*, cache: FeatureCache, site: str, farm: str) -> str:
    """Return a coarse distance band between a Primary and a farm; never the pairing itself."""
    table = cache.sites[site].table
    row = next(r for r in pv_sites().to_dicts() if r["site"] == farm)
    km = c269.distance_km(
        lat_a=table.latitude, lon_a=table.longitude, lat_b=row["latitude"], lon_b=row["longitude"]
    )
    return "under 15 km" if km < 15 else "15 to 30 km" if km < 30 else "30 km or more"

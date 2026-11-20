"""Experiment 709: the Primary panel, the climatology residual, and the six daily detector features.

Every array is indexed on the half-hourly UTC grid that starts at 2019-09-01 00:00 (48 half-hours
per UTC day) and stops on 2026-06-30, so no reading from the final-test window is ever read. Power
is a share of the series' effective capacity.

**The as-of rule.** A feature of day `d` reads only readings of day `d` and a climatology fitted on
readings that end 28 days before the start of the month of `d`. The detector state used at the
forecast origin of day `D` is therefore the state after day `D - 1`, whose last reading ends 9 hours
before the 09:00 UTC origin.

**Features per Primary and UTC day.** The mean residual in four bands of the day (night 22:00 to
06:00, morning 06:00 to 11:00, midday 11:00 to 16:00, evening 16:00 to 22:00), and the day's 5th and
95th percentile of the half-hourly residual. The residual is power minus a month-by-half-hour mean
taken over the 730 days that end 28 days before the start of the day's month. Each feature is
divided by a robust spread (1.4826 times the median absolute deviation) measured on the first 365
days that have features, and the day-by-day detectors start only after those 365 days.
"""

import json
import warnings
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.eval_protocol.slices import MVA_ONLY_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 709
SLUG: Final[str] = "era-signal-in-the-candidate-part-1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
"""Private folder of this experiment's tables and reports."""

DEMAND_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand"
SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
OBSERVED_PATH: Final[Path] = DEMAND_DIR / "inputs" / "observed.parquet"
ERA5_PATH: Final[Path] = DEMAND_DIR / "inputs" / "era5_features.parquet"
NEIGHBOURS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00383-load-shift-xgboost-oracle-first-part-1/work/neighbours_nged.json"
)
"""Electrical neighbours of each Primary, the edge list that experiments 660 and 687 use."""

GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
FIRST_DAY: Final[date] = date(2019, 9, 1)
LAST_DAY: Final[date] = date(2026, 6, 30)
"""The last UTC day that any script of this experiment reads."""
HALF_HOURS_PER_DAY: Final[int] = 48
N_DAYS: Final[int] = (LAST_DAY - FIRST_DAY).days + 1
N_HALF_HOURS: Final[int] = N_DAYS * HALF_HOURS_PER_DAY
_STEP_US: Final[int] = 1_800_000_000

PRIMARIES: Final[tuple[str, ...]] = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19",
    "D21", "D22",
)  # fmt: skip
"""The 16 Primary substations ('primary demand' series), in site-number order."""
SHORT_HISTORY_SITES: Final[tuple[str, ...]] = ("D15", "D18")
SHORT_HISTORY_CUT: Final[date] = date(2025, 3, 1)
"""Readings of D15 and D18 before this day are dropped (the noisy start of a short history)."""

CLIMATOLOGY_WINDOW_DAYS: Final[int] = 730
CLIMATOLOGY_LAG_DAYS: Final[int] = 28
MIN_CLIMATOLOGY_DAYS: Final[int] = 330
"""Valid days the climatology window needs. D15 and D18 reach it only from 2026-03."""
MIN_DAYS_IN_MONTH: Final[int] = 10
MIN_READINGS_PER_DAY: Final[int] = 24
SCALE_WINDOW_DAYS: Final[int] = 365
MIN_SCALE_DAYS: Final[int] = 200
SPREAD_FLOOR: Final[float] = 0.002
"""Smallest robust spread, 0.2% of capacity, so a flat feature cannot give a huge scaled value."""

BAND_NAMES: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening")
BAND_HALF_HOURS: Final[tuple[np.ndarray, ...]] = (
    np.concatenate([np.arange(44, 48), np.arange(0, 12)]),
    np.arange(12, 22),
    np.arange(22, 32),
    np.arange(32, 44),
)
"""Half-hours of the UTC day in each band; together they cover all 48 once."""
FEATURE_NAMES: Final[tuple[str, ...]] = (*BAND_NAMES, "p05", "p95")
N_FEATURES: Final[int] = len(FEATURE_NAMES)
BAND_WEIGHTS: Final[np.ndarray] = np.array([len(b) for b in BAND_HALF_HOURS] + [0, 0]) / 48.0
"""Share of the day in each band, zero for the two percentile features."""


@dataclass(frozen=True)
class Panel:
    """The 16 Primaries' power on the dense half-hourly grid.

    Attributes:
        sites: Site labels, in `PRIMARIES` order.
        power: Power over effective capacity, shape (sites, half-hours), NaN where unobserved.
        capacity_mw: Effective capacity of each site in MW.
        is_mva: True for the eight MVA-only Primaries, whose power is never negative.
        neighbours: Electrical neighbours of each site, as indices into `sites`, symmetric.
    """

    sites: tuple[str, ...]
    power: np.ndarray
    capacity_mw: np.ndarray
    is_mva: np.ndarray
    neighbours: dict[int, list[int]]

    @property
    def n_days(self) -> int:
        """Number of UTC days on the grid."""
        return self.power.shape[1] // HALF_HOURS_PER_DAY

    def with_power(self, *, power: np.ndarray) -> Panel:
        """Return a copy of the panel that holds different power.

        Args:
            power: The new power array, the same shape as `self.power`.

        Returns:
            The new panel.
        """
        return replace(self, power=power)

    def truncated(self, *, n_days: int) -> Panel:
        """Return the panel cut to its first `n_days` days.

        Args:
            n_days: Number of days to keep.

        Returns:
            The shorter panel.
        """
        return replace(self, power=self.power[:, : n_days * HALF_HOURS_PER_DAY])


def day_index(*, day: date) -> int:
    """Return the grid index of a UTC day.

    Args:
        day: The calendar day.

    Returns:
        Days since 2019-09-01.
    """
    return (day - FIRST_DAY).days


def day_of(*, index: int) -> date:
    """Return the calendar day of a grid day index.

    Args:
        index: Days since 2019-09-01.

    Returns:
        The calendar day.
    """
    return FIRST_DAY + timedelta(days=int(index))


def _neighbour_indices(*, sites: tuple[str, ...]) -> dict[int, list[int]]:
    raw = json.loads(NEIGHBOURS_PATH.read_text())
    edges: dict[str, set[str]] = {site: set() for site in sites}
    for a, others in raw.items():
        for b in others:
            if a in edges and b in edges:
                edges[a].add(b)
                edges[b].add(a)
    return {i: sorted(sites.index(b) for b in edges[a]) for i, a in enumerate(sites)}


def load_panel() -> Panel:
    """Read the 16 Primaries' cleaned power from the stage 1 pilot's inputs.

    Returns:
        The panel, with D15 and D18 cut before 2025-03-01 and nothing from 2026-07-01 on.

    Raises:
        ValueError: If the series table does not hold the 16 expected Primaries.
    """
    series = (
        pl.read_parquet(SERIES_PATH)
        .filter(pl.col("group") == "primary demand")
        .with_columns(number=pl.col("site").str.slice(1).cast(pl.Int32))
        .sort("number")
    )
    sites = tuple(series["site"].to_list())
    if sites != PRIMARIES:
        msg = f"expected the 16 Primaries {PRIMARIES}, found {sites}"
        raise ValueError(msg)
    ids = series["time_series_id"].to_list()
    capacity = series["effective_capacity_mw"].to_numpy().astype(np.float64)
    start_us = int(GRID_START.timestamp() * 1e6)
    observed = (
        pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
        .filter(pl.col("time_series_id").is_in(ids))
        .with_columns(index=(pl.col("valid_time").dt.epoch("us") - start_us) // _STEP_US)
        .filter(pl.col("index").is_between(0, N_HALF_HOURS - 1))
    )
    power = np.full((len(sites), N_HALF_HOURS), np.nan)
    for row, (series_id, cap) in enumerate(zip(ids, capacity, strict=True)):
        part = observed.filter(pl.col("time_series_id") == series_id)
        power[row, part["index"].to_numpy()] = part["power"].to_numpy() / cap
    cut = day_index(day=SHORT_HISTORY_CUT) * HALF_HOURS_PER_DAY
    for site in SHORT_HISTORY_SITES:
        power[sites.index(site), :cut] = np.nan
    return Panel(
        sites=sites,
        power=power,
        capacity_mw=capacity,
        is_mva=np.array([site in MVA_ONLY_LABELS for site in sites]),
        neighbours=_neighbour_indices(sites=sites),
    )


@dataclass(frozen=True)
class DailyWeather:
    """ERA5 weather per site on the half-hourly grid, and daily summaries of it.

    Attributes:
        shortwave: Surface shortwave radiation in W/m2, shape (sites, half-hours).
        shortwave_midday: Mean shortwave over 10:00 to 16:00 UTC of each day, shape (sites, days).
        temperature: Mean 2 m temperature of each day in degrees Celsius, shape (sites, days).
    """

    shortwave: np.ndarray
    shortwave_midday: np.ndarray
    temperature: np.ndarray


def load_weather(*, panel: Panel) -> DailyWeather:
    """Read ERA5 shortwave and temperature for the panel's sites.

    ERA5 is the reanalysis that the classifier reads for days at least 7 days before the origin,
    because ERA5 arrives about 5 days late.

    Args:
        panel: The panel whose sites and grid the weather follows.

    Returns:
        The half-hourly shortwave and the daily summaries.
    """
    series = pl.read_parquet(SERIES_PATH).filter(pl.col("site").is_in(panel.sites))
    ids = dict(zip(series["site"].to_list(), series["time_series_id"].to_list(), strict=True))
    start_us = int(GRID_START.timestamp() * 1e6)
    era5 = (
        pl.read_parquet(
            ERA5_PATH, columns=["time_series_id", "valid_time", "shortwave", "temperature_2m"]
        )
        .filter(pl.col("time_series_id").is_in(list(ids.values())))
        .with_columns(index=(pl.col("valid_time").dt.epoch("us") - start_us) // _STEP_US)
        .filter(pl.col("index").is_between(0, N_HALF_HOURS - 1))
    )
    shortwave = np.full((len(panel.sites), N_HALF_HOURS), np.nan)
    temperature = np.full((len(panel.sites), N_HALF_HOURS), np.nan)
    for row, site in enumerate(panel.sites):
        part = era5.filter(pl.col("time_series_id") == ids[site])
        shortwave[row, part["index"].to_numpy()] = part["shortwave"].to_numpy()
        temperature[row, part["index"].to_numpy()] = part["temperature_2m"].to_numpy()
    by_day = shortwave.reshape(len(panel.sites), -1, HALF_HOURS_PER_DAY)
    return DailyWeather(
        shortwave=shortwave,
        shortwave_midday=np.nanmean(by_day[:, :, 20:32], axis=2),
        temperature=np.nanmean(temperature.reshape(by_day.shape), axis=2),
    )


def _months(*, n_days: int) -> list[tuple[int, int, int]]:
    """Return (first day index, stop day index, month of year) of each calendar month."""
    out: list[tuple[int, int, int]] = []
    start = 0
    while start < n_days:
        first = day_of(index=start)
        following = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
        stop = min(day_index(day=following), n_days)
        out.append((start, stop, first.month))
        start = stop
    return out


def climatology_residual(*, power_site: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one Primary's residual against its month-by-half-hour climatology.

    The climatology of a calendar month is the mean, at each half-hour of the day, over the days of
    the same month of the year inside the 730 days that end 28 days before that month starts.

    Args:
        power_site: One Primary's power over capacity, length a whole number of days.

    Returns:
        The residual shaped (days, 48), NaN where no climatology exists or no reading exists, and a
        boolean per day that says whether the day has a climatology.
    """
    n_days = power_site.size // HALF_HOURS_PER_DAY
    grid = power_site.reshape(n_days, HALF_HOURS_PER_DAY)
    valid_day = np.isfinite(grid).sum(axis=1) >= MIN_READINGS_PER_DAY
    month_of_day = np.array([day_of(index=i).month for i in range(n_days)])
    residual = np.full_like(grid, np.nan)
    exists = np.zeros(n_days, dtype=bool)
    for start, stop, month in _months(n_days=n_days):
        high = start - CLIMATOLOGY_LAG_DAYS
        if high <= 0:
            continue
        low = max(high - CLIMATOLOGY_WINDOW_DAYS, 0)
        if valid_day[low:high].sum() < MIN_CLIMATOLOGY_DAYS:
            continue
        same_month = valid_day[low:high] & (month_of_day[low:high] == month)
        if same_month.sum() < MIN_DAYS_IN_MONTH:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            climatology = np.nanmean(grid[low:high][same_month], axis=0)
        residual[start:stop] = grid[start:stop] - climatology[None, :]
        exists[start:stop] = True
    return residual, exists


def daily_features(*, residual: np.ndarray) -> np.ndarray:
    """Return the six raw daily features of a residual shaped (days, 48).

    Args:
        residual: Half-hourly residual of one Primary.

    Returns:
        An array (days, 6) of the four band means and the day's 5th and 95th percentile, NaN where
        a band has fewer than half its half-hours or the day has fewer than 24 readings.
    """
    n_days = residual.shape[0]
    out = np.full((n_days, N_FEATURES), np.nan)
    finite = np.isfinite(residual)
    for k, half_hours in enumerate(BAND_HALF_HOURS):
        band = residual[:, half_hours]
        count = finite[:, half_hours].sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(band, axis=1)
        out[:, k] = np.where(count >= len(half_hours) / 2, mean, np.nan)
    enough = finite.sum(axis=1) >= MIN_READINGS_PER_DAY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out[:, 4] = np.where(enough, np.nanpercentile(residual, 5, axis=1), np.nan)
        out[:, 5] = np.where(enough, np.nanpercentile(residual, 95, axis=1), np.nan)
    return out


@dataclass(frozen=True)
class SiteFeatures:
    """One Primary's residual, daily features, and scaling.

    Attributes:
        residual: Half-hourly residual over capacity, flat, length days times 48.
        exists: Whether each day has a climatology.
        raw: Daily features in capacity share, shape (days, 6).
        scaled: The features divided by `spread`, shape (days, 6), NaN where raw is NaN.
        spread: Robust spread of each feature on the first 365 days that have features.
        daily_mean: Mean residual of each day, NaN with fewer than 36 readings.
        first_day: Index of the first day with all six features, or -1 if there is none.
        ready_day: First day on which the detectors may run, or the number of days if never.
    """

    residual: np.ndarray
    exists: np.ndarray
    raw: np.ndarray
    scaled: np.ndarray
    spread: np.ndarray
    daily_mean: np.ndarray
    first_day: int
    ready_day: int


def compute_site_features(*, power_site: np.ndarray) -> SiteFeatures:
    """Compute one Primary's residual, daily features, and robust scaling.

    Args:
        power_site: One Primary's power over capacity.

    Returns:
        The site's `SiteFeatures`. A Primary whose first 365 days of features hold fewer than 200
        valid days per feature gets `ready_day` equal to the number of days, which means `none`.
    """
    residual, exists = climatology_residual(power_site=power_site)
    n_days = residual.shape[0]
    raw = daily_features(residual=residual)
    complete = np.isfinite(raw).all(axis=1)
    spread = np.full(N_FEATURES, np.nan)
    first_day, ready_day = -1, n_days
    if complete.any():
        first_day = int(np.argmax(complete))
        window = raw[first_day : first_day + SCALE_WINDOW_DAYS]
        if np.isfinite(window).all(axis=1).sum() >= MIN_SCALE_DAYS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                centre = np.nanmedian(window, axis=0)
                mad = np.nanmedian(np.abs(window - centre), axis=0)
            spread = np.maximum(1.4826 * mad, SPREAD_FLOOR)
            ready_day = min(first_day + SCALE_WINDOW_DAYS, n_days)
    valid = np.isfinite(residual).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        daily_mean = np.where(valid >= 36, np.nanmean(residual, axis=1), np.nan)
    return SiteFeatures(
        residual=residual.reshape(-1),
        exists=exists,
        raw=raw,
        scaled=raw / spread[None, :],
        spread=spread,
        daily_mean=daily_mean,
        first_day=first_day,
        ready_day=ready_day,
    )


def compute_panel_features(*, panel: Panel) -> list[SiteFeatures]:
    """Compute the features of every site of a panel.

    Args:
        panel: The panel, real or injected.

    Returns:
        One `SiteFeatures` per site, in panel order.
    """
    return [compute_site_features(power_site=row) for row in panel.power]


def main() -> None:
    """Print how many days of features each Primary has, from the real panel."""
    panel = load_panel()
    features = compute_panel_features(panel=panel)
    with pl.Config(tbl_rows=20):
        print(
            pl.DataFrame(
                {
                    "site": list(panel.sites),
                    "first_feature_day": [
                        None if f.first_day < 0 else str(day_of(index=f.first_day))
                        for f in features
                    ],
                    "ready_day": [
                        "never" if f.ready_day >= panel.n_days else str(day_of(index=f.ready_day))
                        for f in features
                    ],
                    "spread_midday": [float(f.spread[2]) for f in features],
                }
            )
        )


if __name__ == "__main__":
    main()

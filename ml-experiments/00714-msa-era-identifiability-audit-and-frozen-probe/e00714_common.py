"""Experiment 714: shared constants, the stored bank loader, and the weather-only reference level.

Every array is indexed on the half-hourly UTC grid of experiment 709 (48 half-hours per day, from
2019-09-01 00:00). Power is a share of the series' effective capacity.

**Reference level for the size of a change.** Experiment 667's stored forecasts cannot serve. They
start on 2025-11-01, so the 14 days before the onset of the 2025-11 fold are not stored. Their
normal-running-arrangement output reads the Candidate's lags, so it is not lag-free. Their valid
times are scored in MW per forecast origin, not as a continuous series for the days before an
origin. The reference level is therefore the fallback: a linear fit of the Primary's own power on
calendar dummies and ERA5 weather, fitted on the 730 days that end 28 days before the start of the
fold month (the convention of experiment 709's climatology).

**Weather.** ERA5 stands in for the numerical weather prediction analysis that a live forecast
reads at the origin. The reanalysis arrives about 5 days late in live use, so the weather of the
last 5 days before an origin is read optimistically. The weather of the days after the origin is
never read.
"""

import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import e00709_features as f709  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 714
SLUG: Final[str] = "msa-era-identifiability-audit-and-frozen-probe"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
"""Private folder of this experiment's worlds, feature tables, and reports."""
BANK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00709-era-signal-in-the-candidate-part-1" / "worlds"
)
"""Experiment 709's stored injected passes, read only."""

DAY: Final[int] = f709.HALF_HOURS_PER_DAY
N_HH: Final[int] = f709.N_HALF_HOURS
MONTHS: Final[tuple[str, ...]] = ("2025-11", "2026-01", "2026-04", "2026-06")
"""The four fold months of experiment 709's bank."""
BANK_PASSES_PER_MONTH: Final[int] = 16
ORIGIN_DAYS: Final[tuple[int, ...]] = (7, 14, 28)
"""Days after the onset at which the forecast origin sits."""
ORIGIN_HOUR_HH: Final[int] = 18
"""The origin is at 09:00 UTC, half-hour 18 of the day."""
LATENCY_HH: Final[int] = 12
"""Power newer than the origin minus 6 hours is not available (the Candidate's cut)."""
WINDOW_DAYS: Final[int] = 14
WINDOW_HH: Final[int] = WINDOW_DAYS * DAY
OLDER_AGES: Final[tuple[int, ...]] = (21, 28, 42, 63, 91, 364)
"""Ages in days of the six older same-weekday rows."""
STORE_BEFORE_DAYS: Final[int] = 400
STORE_AFTER_DAYS: Final[int] = 45
"""Days kept before and after the onset day in a stored pass window."""
EXCLUDED_HOSTS: Final[tuple[str, ...]] = ("D8", "D15", "D18")
"""D8 keeps its real eras unmixed with injected ones; D15 and D18 have no climatology."""

TRAIN_DAYS: Final[int] = 730
TRAIN_LAG_DAYS: Final[int] = 28
MIN_TRAIN_ROWS: Final[int] = 8000
RIDGE: Final[float] = 1e-3


class _Stub:
    """Stand-in for a class of an experiment module that the unpickler does not import."""

    def __setstate__(self, state: object) -> None:
        self.state = state


class _BankUnpickler(pickle.Unpickler):
    """Unpickler that replaces classes of `e00*` experiment modules by stubs.

    The bank pickles hold one class of experiment 709 (`e00709_signal.Signal`, in the detector
    signals that this experiment does not read). Importing it needs `e00709_classify`, then
    `e00687_features`, then `e00598_signals`, and every further module of that chain on the path.
    """

    def find_class(self, module: str, name: str) -> type:
        """Return a stub for experiment modules, the real class otherwise."""
        if module.startswith("e00"):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


def month_start_day(*, month: str) -> int:
    """Return the grid day index of the first day of a `YYYY-MM` month."""
    year, mon = (int(part) for part in month.split("-"))
    return f709.day_index(day=date(year, mon, 1))


@dataclass(frozen=True)
class PassPower:
    """Injected power of one pass over a window of the grid.

    Attributes:
        start_hh: Grid index of column 0.
        power: Injected power, shape (sites, half-hours), NaN where unobserved.
    """

    start_hh: int
    power: np.ndarray

    def get(self, *, site: int, lo: int, hi: int) -> np.ndarray:
        """Return `power[site, lo:hi]` in grid coordinates, NaN outside the stored window."""
        out = np.full(hi - lo, np.nan)
        a, b = max(lo, self.start_hh), min(hi, self.start_hh + self.power.shape[1])
        if b > a:
            out[a - lo : b - lo] = self.power[site, a - self.start_hh : b - self.start_hh]
        return out


def window_bounds(*, onset_day: int) -> tuple[int, int]:
    """Return the stored window `[lo, hi)` in grid half-hours for an onset day."""
    lo = max(onset_day - STORE_BEFORE_DAYS, 0) * DAY
    hi = min(onset_day + STORE_AFTER_DAYS, f709.N_DAYS) * DAY
    return lo, hi


def load_bank_pass(*, month: str, index: int) -> dict:
    """Read one of experiment 709's bank passes and cut its power to the stored window.

    Args:
        month: Fold month.
        index: Pass index within the month, 0 to 15.

    Returns:
        A dict with `passes` (a `PassPower`), `truth` (the truth table), and `touched`.
    """
    path = BANK_DIR / f"bank_{month}_p{index}.pkl"
    with path.open("rb") as handle:
        parts = _BankUnpickler(handle).load()
    lo, hi = window_bounds(onset_day=month_start_day(month=month))
    return {
        "power": PassPower(start_hh=lo, power=np.asarray(parts["power"][:, lo:hi], np.float32)),
        "truth": parts["truth_table"],
        "touched": list(parts["touched"]),
        "n_hh": int(parts["power"].shape[1]),
    }


@dataclass(frozen=True)
class Context:
    """The real panel, the ERA5 weather, and the references of the fold months.

    Attributes:
        panel: The real panel of experiment 709.
        weather: ERA5 shortwave per half-hour and temperature per day.
        references: Reference level by fold-month start day, each a `PassPower` over the stored
            window of that month.
    """

    panel: f709.Panel
    weather: f709.DailyWeather
    references: dict[str, PassPower]


def design(*, weather: f709.DailyWeather, site: int, day_lo: int, day_hi: int) -> np.ndarray:
    """Return the design matrix of one site for whole days `[day_lo, day_hi)`.

    Columns: 96 dummies (half-hour of the day, weekday or weekend), daily mean temperature and its
    square, shortwave over 700 W/m2 and its square, and the annual cycle.

    Args:
        weather: The ERA5 weather.
        site: Site index.
        day_lo: First day.
        day_hi: Day after the last.

    Returns:
        An array of shape ((day_hi - day_lo) * 48, 102); NaN where the weather is missing.
    """
    days = np.arange(day_lo, day_hi)
    n = days.size
    hh = np.tile(np.arange(DAY), n)
    weekend = np.repeat(np.array([f709.day_of(index=int(d)).weekday() >= 5 for d in days]), DAY)
    dummies = np.zeros((n * DAY, 2 * DAY))
    dummies[np.arange(n * DAY), hh + DAY * weekend] = 1.0
    temp = np.repeat(weather.temperature[site, day_lo:day_hi], DAY) / 10.0
    sw = weather.shortwave[site, day_lo * DAY : day_hi * DAY] / 700.0
    phase = 2 * np.pi * np.repeat(days, DAY) / 365.25
    return np.column_stack([dummies, temp, temp**2, sw, sw**2, np.sin(phase), np.cos(phase)])


def fit_reference(
    *, panel: f709.Panel, weather: f709.DailyWeather, site: int, month_day: int
) -> np.ndarray | None:
    """Fit the weather-only reference of one site for the fold month that starts on `month_day`.

    The fit reads real power only on the days `[month_day - 758, month_day - 28)`.

    Args:
        panel: The real panel.
        weather: The ERA5 weather.
        site: Site index.
        month_day: Grid day of the first day of the fold month.

    Returns:
        The coefficient vector, or None if the site has too few usable rows.
    """
    hi = month_day - TRAIN_LAG_DAYS
    lo = max(hi - TRAIN_DAYS, 0)
    x = design(weather=weather, site=site, day_lo=lo, day_hi=hi)
    y = panel.power[site, lo * DAY : hi * DAY]
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if ok.sum() < MIN_TRAIN_ROWS:
        return None
    xs, ys = x[ok], y[ok]
    gram = xs.T @ xs + RIDGE * np.eye(xs.shape[1])
    return np.linalg.solve(gram, xs.T @ ys)


def month_day_of(*, day: int) -> int:
    """Return the grid day of the first day of the calendar month that contains grid day `day`."""
    when = f709.day_of(index=day)
    return f709.day_index(day=date(when.year, when.month, 1))


def build_reference(*, panel: f709.Panel, weather: f709.DailyWeather, month_day: int) -> PassPower:
    """Predict the reference level of every site over the stored window of a fold month.

    Args:
        panel: The real panel.
        weather: The ERA5 weather.
        month_day: Grid day of the first day of the fold month.

    Returns:
        The reference as a `PassPower`; NaN rows for a site without a fit.
    """
    lo, hi = window_bounds(onset_day=month_day)
    out = np.full((len(panel.sites), hi - lo), np.nan, np.float32)
    for site in range(len(panel.sites)):
        beta = fit_reference(panel=panel, weather=weather, site=site, month_day=month_day)
        if beta is None:
            continue
        x = design(weather=weather, site=site, day_lo=lo // DAY, day_hi=hi // DAY)
        out[site] = x @ beta
    return PassPower(start_hh=lo, power=out)


def load_context() -> Context:
    """Load the real panel and weather. References are built on demand."""
    panel = f709.load_panel()
    return Context(panel=panel, weather=f709.load_weather(panel=panel), references={})


def reference_for(*, ctx: Context, month_day: int) -> PassPower:
    """Return (building and caching it) the reference level of a fold month."""
    key = str(month_day)
    if key not in ctx.references:
        ctx.references[key] = build_reference(
            panel=ctx.panel, weather=ctx.weather, month_day=month_day
        )
    return ctx.references[key]

"""Experiment 636: shared code for the monthly-slope known-answer test at the metered solar farms.

Rows come from experiment 218's `farm_rows_cams.parquet` (CAMS irradiance and ERA5 temperature,
every daylight half-hour, output as a fraction of stored capacity). Two monthly estimators are
computed per farm: the slope of output on a fitted irradiance expectation over all daylight
half-hours (weighted by the clear-sky index squared), and experiment 618's sunny-row envelope. A
third, `slope_nozero`, drops half-hours with zero output where the expectation is large and is
labelled post hoc. Farms appear only as labels A to F.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00636-metered-solar-monthly-slope-capacity-known-answer-test"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
TESTABLE: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""Solar E has 20 months of data and is left out of every injection test."""
END_EXCLUSIVE: Final[str] = "2026-07-01"
CLEAR_SKY_MIN_ALL_W_M2: Final[float] = 100.0
EXPECTATION_MIN: Final[float] = 0.02
SUNNY_CSI_MIN: Final[float] = 0.9
SUNNY_CLEAR_SKY_MIN_W_M2: Final[float] = 700.0
SUNNY_RUN_MIN: Final[int] = 6
MIN_ROWS_PER_SUNNY_DAY: Final[int] = 6
AC_LIMIT_QUANTILE: Final[float] = 0.995
CLIPPED_SHARE_OF_LIMIT: Final[float] = 0.97
CLIP_HEADROOM: Final[float] = 0.15
"""Assumed excess of unclipped DC power over the limit on a clipped half-hour (a stress case)."""
ZERO_DROP_EXPECTATION: Final[float] = 0.3
HORIZONS: Final[tuple[int, ...]] = (1, 3, 6, 12)
LOSSES: Final[tuple[float, ...]] = (0.03, 0.05, 0.075, 0.10, 0.15)
ESTIMATORS: Final[tuple[str, ...]] = ("slope", "slope_nozero", "envelope")
START_MONTHS: Final[tuple[tuple[int, int], ...]] = tuple(
    (y, m) for y in range(2021, 2026) for m in (1, 4, 7, 10)
)
"""The 20 start months: January, April, July, and October of 2021 to 2025."""
EPOCH_YEAR: Final[int] = 2019
MIN_DAYLIGHT_ROWS: Final[int] = 500


def month_index(year: int, month: int) -> int:
    """Return the number of months from January 2019 to the given month."""
    return (year - EPOCH_YEAR) * 12 + (month - 1)


@dataclass(frozen=True)
class Farm:
    """One farm's rows as arrays, with the quantities each world needs precomputed."""

    site: str
    y: np.ndarray
    x: np.ndarray
    mo: np.ndarray
    day: np.ndarray
    csi: np.ndarray
    daylight: np.ndarray
    sunny: np.ndarray
    ac_limit: float
    dc_hat: np.ndarray
    clipped_share: float


def design(df: pl.DataFrame) -> np.ndarray:
    """Return the irradiance, temperature, and sun-position design matrix of the slope estimator."""
    g = df["ghi_w_m2"].to_numpy() / 1000.0
    t = df["temperature_c"].to_numpy()
    az = np.deg2rad(df["sun_azimuth_deg"].to_numpy())
    e = df["elevation_deg"].to_numpy() / 90.0
    dni = df["dni_w_m2"].to_numpy() / 1000.0
    dhi = df["dhi_w_m2"].to_numpy() / 1000.0
    return np.column_stack(
        [
            g,
            g**2,
            g * t,
            dni,
            dhi,
            dni * np.sin(az),
            dni * np.cos(az),
            dni * e,
            g * e,
            np.ones_like(g),
        ]
    )


def robust_fit(x: np.ndarray, y: np.ndarray, *, iterations: int = 15) -> np.ndarray:
    """Return Huber-weighted least-squares coefficients (iteratively reweighted)."""
    w = np.ones_like(y)
    b = np.zeros(x.shape[1])
    for _ in range(iterations):
        sw = np.sqrt(w)
        b = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]
        r = y - x @ b
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
    return b


def load_farms() -> dict[str, Farm]:
    """Return every farm's rows to 2026-06-30 as a `Farm`, keyed by site label."""
    d = (
        pl.read_parquet(FARM_ROWS)
        .filter(pl.col("time") < pl.lit(END_EXCLUSIVE).str.to_datetime(time_zone="UTC"))
        .filter(pl.col("output").is_not_null())
        .sort("site", "time")
        .with_columns(csi=(pl.col("ghi_w_m2") / pl.col("clear_sky")).clip(0, 1.5))
        .with_columns(
            ok=(pl.col("csi") >= SUNNY_CSI_MIN) & (pl.col("clear_sky") >= SUNNY_CLEAR_SKY_MIN_W_M2)
        )
        .with_columns(
            brk=(
                (pl.col("time").diff().over("site") != pl.duration(minutes=30)).fill_null(True)
                | (pl.col("ok") != pl.col("ok").shift().over("site"))
            )
        )
        .with_columns(run=pl.col("brk").cum_sum().over("site"))
        .with_columns(run_len=pl.col("ok").sum().over("site", "run"))
        .with_columns(sunny=pl.col("ok") & (pl.col("run_len") >= SUNNY_RUN_MIN))
    )
    farms: dict[str, Farm] = {}
    for site in sorted(d["site"].unique().to_list()):
        f = d.filter(pl.col("site") == site)
        y = f["output"].to_numpy().astype(float)
        x = design(f)
        daylight = f["clear_sky"].to_numpy() >= CLEAR_SKY_MIN_ALL_W_M2
        limit = float(np.quantile(y[daylight], AC_LIMIT_QUANTILE))
        clipped = (y >= CLIPPED_SHARE_OF_LIMIT * limit) & (y <= limit)
        dc_hat = np.where(clipped, y * (1.0 + CLIP_HEADROOM), y)
        t = f["time"].dt
        mo = ((t.year() - EPOCH_YEAR) * 12 + t.month() - 1).to_numpy()
        farms[site] = Farm(
            site=site,
            y=y,
            x=x,
            mo=mo,
            day=f["time"].dt.date().cast(pl.Int32).to_numpy(),
            csi=f["csi"].to_numpy(),
            daylight=daylight,
            sunny=f["sunny"].to_numpy(),
            ac_limit=limit,
            dc_hat=dc_hat,
            clipped_share=float(clipped[daylight].mean()),
        )
    return farms


def _slope_by_month(farm: Farm, y: np.ndarray, *, drop_zero: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (months, ratio) of the csi^2-weighted least-squares slope through the origin."""
    m_all = farm.daylight
    b = robust_fit(farm.x[m_all], y[m_all])
    u = farm.x @ b
    sel = m_all & (u > EXPECTATION_MIN)
    if drop_zero:
        sel &= ~((y <= 0.0) & (u > ZERO_DROP_EXPECTATION))
    w = farm.csi[sel] ** 2
    months = farm.mo[sel]
    lo = months.min()
    uy = np.bincount(months - lo, weights=w * u[sel] * y[sel])
    uu = np.bincount(months - lo, weights=w * u[sel] ** 2)
    ok = uu > 0
    return np.arange(len(uu))[ok] + lo, uy[ok] / uu[ok]


def _envelope_by_month(farm: Farm, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (months, level) of experiment 618's sunny-day median ratio, averaged by month."""
    s = farm.sunny
    if s.sum() < 50:
        return np.array([], dtype=int), np.array([])
    xs = farm.x[s]
    expectation = xs @ robust_fit(xs, y[s])
    ok = expectation > 0.05
    tab = pl.DataFrame(
        {
            "mo": farm.mo[s][ok],
            "day": farm.day[s][ok],
            "r": y[s][ok] / expectation[ok],
        }
    )
    daily = (
        tab.group_by("mo", "day")
        .agg(v=pl.col("r").median(), n=pl.len())
        .filter(pl.col("n") >= MIN_ROWS_PER_SUNNY_DAY)
    )
    monthly = daily.group_by("mo").agg(c=pl.col("v").mean()).sort("mo")
    return monthly["mo"].to_numpy(), monthly["c"].to_numpy()


def deseasonalise(months: np.ndarray, c: np.ndarray, *, before: int | None) -> np.ndarray:
    """Divide each month's value by the median of the same calendar month across years.

    With `before` set, the median uses only months earlier than `before`, so an injected step
    cannot move the reference. A calendar month with no earlier year falls back to all years.
    """
    cal = months % 12
    out = np.empty_like(c)
    for k in range(12):
        sel = cal == k
        if not sel.any():
            continue
        ref = sel if before is None else sel & (months < before)
        if not ref.any():
            ref = sel
        out[sel] = c[sel] / np.median(c[ref])
    return out


def monthly_levels(
    farm: Farm, y: np.ndarray, *, start: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return each estimator's monthly level for `y`, deseasonalised on months before `start`."""
    raw = {
        "slope": _slope_by_month(farm, y, drop_zero=False),
        "slope_nozero": _slope_by_month(farm, y, drop_zero=True),
        "envelope": _envelope_by_month(farm, y),
    }
    return {
        k: (m, deseasonalise(m, c, before=start)) if len(m) else (m, c) for k, (m, c) in raw.items()
    }


def step_statistic(months: np.ndarray, level: np.ndarray, *, start: int, horizon: int) -> float:
    """Return the mean level of `horizon` months from `start`, over the previous 12, minus 1.

    The statistic is NaN when fewer than 2 months before or no month after have data, and the
    known-answer scoring counts those windows as missed.
    """
    before = level[(months < start) & (months >= start - 12)]
    after = level[(months >= start) & (months < start + horizon)]
    if len(before) < 2 or len(after) < 1:
        return float("nan")
    return float(np.mean(after) / np.mean(before) - 1.0)


def inject_ac_loss(farm: Farm, *, start: int, loss: float) -> np.ndarray:
    """World 1: multiply all output from `start` on by `1 - loss`."""
    y = farm.y.copy()
    y[farm.mo >= start] *= 1.0 - loss
    return y


def inject_dc_loss(farm: Farm, *, start: int, loss: float) -> np.ndarray:
    """World 2: remove `loss` of the DC capacity before the inverter limit from `start` on.

    The scaled DC power is clipped at the farm's AC limit, so flat tops stay at the limit.
    """
    y = farm.y.copy()
    sel = farm.mo >= start
    scaled = (1.0 - loss) * farm.dc_hat[sel]
    y[sel] = np.minimum(scaled, np.maximum(farm.ac_limit, (1.0 - loss) * farm.y[sel]))
    return y


def inject_curtailment(farm: Farm, *, start: int, share: float, seed: int) -> np.ndarray:
    """World 4: zero a random `share` of month `start`'s daylight half-hours, nothing else."""
    y = farm.y.copy()
    idx = np.flatnonzero((farm.mo == start) & farm.daylight)
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=round(share * len(idx)), replace=False)
    y[pick] = 0.0
    return y


def cloudiest_months(farm: Farm, *, n: int) -> list[int]:
    """Return the `n` months, among the start-eligible 2021 to 2025 ones, cloudier than usual most.

    Cloudiness is the month's mean clear-sky index over daylight half-hours minus the median of
    the same calendar month across years, so a winter month is cloudy only against other winters.
    """
    lo, hi = month_index(2021, 1), month_index(2025, 12)
    months = np.arange(lo, hi + 1)
    mean_csi = np.array(
        [
            farm.csi[farm.daylight & (farm.mo == m)].mean()
            if (farm.daylight & (farm.mo == m)).sum() >= MIN_DAYLIGHT_ROWS
            else np.nan
            for m in months
        ]
    )
    anomaly = mean_csi.copy()
    for k in range(12):
        sel = (months % 12 == k) & ~np.isnan(mean_csi)
        anomaly[sel] = mean_csi[sel] - np.median(mean_csi[sel])
    order = np.argsort(np.where(np.isnan(anomaly), np.inf, anomaly))
    return [int(months[i]) for i in order[:n]]

# ruff: noqa: E501
"""Experiment 723: sky sensors, stretches, concordance, injected worlds, and closure fits.

Everything here is pure: arrays on the half-hour grid of `e00722_core` in, arrays and tables out,
so the tests run on synthetic arrays. A grid index `i` stands for the half-hour that ends at
`GRID_FIRST_END + i * 30 minutes`, and a meter reads positive for import. The stretch machinery
(`touch_level`, `touches`, `day_stretches`, `concordance`) is experiment 722's, imported and not
copied.

**R2, identities with their conditions.** The change of a folded reading obeys `dm = s * c * dX`
only within one sign `s` (`c = |X| / m`, between 0 and 1). The fold ratio `s * (1 + kappa) * g` holds
only under the linearity assumption of the amendment's section 2.1, namely that the change of the
rest of the residual is `kappa` times the child's change plus a term independent of both.

**Provisional definitions.** The functions `reading_fast_change`, `fold`, `level_reference_rows`,
and `sensor_series` hold definitions that the plan page does not state; each docstring says so.
"""

import itertools
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import e00722_core as core722  # ty: ignore[unresolved-import]
import e00722_sky as sky  # ty: ignore[unresolved-import]
import e00723_core as core
import numpy as np
import polars as pl

HOURLY_LAG: Final[int] = core722.HOURLY_LAG_HALF_HOURS
"""Hourly changes use a lag of two half-hours, on hour-ending rows."""

SENSOR_NAMES: Final[tuple[str, ...]] = ("farm", "meter", "cams", "shifted")
"""The four sensor variants that job W2 scores."""


# ---------------------------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------------------------
def n_days_of_grid() -> int:
    """Return the number of whole UTC days on the master grid."""
    return (core.N_GRID + core.HALF_HOURS_PER_DAY - 1) // core.HALF_HOURS_PER_DAY


def row_of_instant(*, instant: datetime) -> int:
    """Return the grid row of the half-hour that ends at `instant`."""
    return int((instant - core722.GRID_FIRST_END) / timedelta(minutes=30))


def hour_end_rows(*, n: int) -> np.ndarray:
    """Return True on rows whose end stamp falls on the hour (the odd rows)."""
    return np.arange(n) % 2 == 1


def later_start_day(*, first_row: int) -> int:
    """Return the first day of the later span: the day that holds the first row with both transformers."""
    return int(first_row) // core.HALF_HOURS_PER_DAY


def later_span_days(*, first_row: int) -> int:
    """Return the number of calendar days from the first both-transformer day to the holdout."""
    return n_days_of_grid() - later_start_day(first_row=first_row)


def start_row_of_span(*, first_row: int) -> int:
    """Return the first grid row of the later span, which starts at midnight of its first day."""
    return later_start_day(first_row=first_row) * core.HALF_HOURS_PER_DAY


def span(*, x: np.ndarray, start: int) -> np.ndarray:
    """Return the rows of the later span."""
    return x[start:]


def month_of_rows(*, n: int) -> np.ndarray:
    """Return the calendar month (1 to 12) of each of the first `n` grid rows."""
    n_days = (n + core.HALF_HOURS_PER_DAY - 1) // core.HALF_HOURS_PER_DAY
    return np.repeat(sky.month_of_day(n_days=n_days), core.HALF_HOURS_PER_DAY)[:n]


def utc_hour_of_rows(*, n: int) -> np.ndarray:
    """Return the UTC hour (a fraction) at the start of each grid half-hour."""
    return (np.arange(n) % core.HALF_HOURS_PER_DAY) / 2.0


# ---------------------------------------------------------------------------------------------
# Farm sensor
# ---------------------------------------------------------------------------------------------
def farm_share(*, output_mw: np.ndarray, capacity_mw: float) -> np.ndarray:
    """Return a farm's output as a share of its capacity."""
    return output_mw / capacity_mw


def sensor_series(*, shares: Sequence[np.ndarray]) -> np.ndarray:
    """Return the mean share of the farms of a sensor, NaN unless every farm has a value.

    PROVISIONAL: the plan page says "mean output"; requiring every farm at a row avoids steps that
    appear when a farm drops out of the mean.
    """
    stack = np.stack(list(shares))
    return np.where(np.isfinite(stack).all(axis=0), stack.mean(axis=0), np.nan)


def day_ratio(*, z: np.ndarray, u_clear: np.ndarray, lit: np.ndarray, n_days: int) -> np.ndarray:
    """Return `z_bar_d`, the day's ratio of summed `Z` to summed `u_clear` over lit half-hours."""
    day = sky.day_of(n=len(z))
    use = lit & np.isfinite(z) & np.isfinite(u_clear)
    zs = np.bincount(day[use], weights=z[use], minlength=n_days)
    us = np.bincount(day[use], weights=u_clear[use], minlength=n_days)
    return np.divide(zs, us, out=np.full(n_days, np.nan), where=us > 0)


def _lit_pair(*, lit: np.ndarray) -> np.ndarray:
    """Return True at rows `t` where both `t - 1` and `t` are lit."""
    out = np.zeros(lit.shape, dtype=bool)
    out[1:] = lit[1:] & lit[:-1]
    return out


def fast_sensor_change(*, z: np.ndarray, u_clear: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Return `dZ~_t = dZ_t - z_bar_d * du_clear_t` on lit pairs, NaN elsewhere.

    The second term removes the arc of the sun.
    """
    n_days = (len(z) + core.HALF_HOURS_PER_DAY - 1) // core.HALF_HOURS_PER_DAY
    zbar = day_ratio(z=z, u_clear=u_clear, lit=lit, n_days=n_days)
    day = sky.day_of(n=len(z))
    out = np.full(z.shape, np.nan)
    out[1:] = (z[1:] - z[:-1]) - zbar[day[1:]] * (u_clear[1:] - u_clear[:-1])
    return np.where(_lit_pair(lit=lit), out, np.nan)


def arc_change(*, u_clear: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Return `d u_clear` on lit pairs, NaN elsewhere."""
    out = np.full(u_clear.shape, np.nan)
    out[1:] = u_clear[1:] - u_clear[:-1]
    return np.where(_lit_pair(lit=lit), out, np.nan)


def trailing_capacity(
    *, z: np.ndarray, u_clear: np.ndarray, lit: np.ndarray, n_days: int
) -> np.ndarray:
    """Return `c_Z` per day: the 95th percentile of `Z / u_clear` over lit half-hours of the trailing 91 days.

    The grid may end part-way through its last day (the master grid does), and the missing
    half-hours are NaN. PROVISIONAL: "trailing" excludes the day itself, and a day with no earlier lit half-hour gets
    NaN. The result has one value per day.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lit & np.isfinite(z) & (u_clear > 0), z / u_clear, np.nan)
    whole = n_days * core.HALF_HOURS_PER_DAY
    if len(ratio) > whole or len(ratio) <= whole - core.HALF_HOURS_PER_DAY:
        msg = "the grid must end inside its last day"
        raise ValueError(msg)
    # The master grid stops before the holdout, so its last day can be short; pad it with NaN.
    ratio = np.concatenate([ratio, np.full(whole - len(ratio), np.nan)])
    by_day = [row[np.isfinite(row)] for row in ratio.reshape(n_days, core.HALF_HOURS_PER_DAY)]
    out = np.full(n_days, np.nan)
    for d in range(1, n_days):
        recent = by_day[max(0, d - core.TRAILING_CAPACITY_DAYS) : d]
        merged = np.concatenate(recent) if recent else np.array([])
        if merged.size:
            out[d] = float(np.percentile(merged, core.CAPACITY_P95))
    return out


def index_change(
    *, z: np.ndarray, u_clear: np.ndarray, lit: np.ndarray, c_z_day: np.ndarray
) -> np.ndarray:
    """Return `k^Z_t - k^Z_{t-1}` with `k^Z = Z / (c_Z * u_clear)`, NaN unless both half-hours are lit."""
    day = sky.day_of(n=len(z))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(lit & (u_clear > 0), z / (c_z_day[day] * u_clear), np.nan)
    out = np.full(z.shape, np.nan)
    out[1:] = k[1:] - k[:-1]
    return np.where(_lit_pair(lit=lit), out, np.nan)


@dataclass(frozen=True)
class SensorChange:
    """A sensor's fast change `dz` and index change `dk`, both on the grid (NaN where undefined).

    A sensor with `hourly` True (the CAMS sensor) holds hourly changes (`t - 2` to `t`) on
    hour-ending rows, so the reading's fast change and the transfer mask it is paired with are
    hourly too (`align_reading`).
    """

    dz: np.ndarray
    dk: np.ndarray
    hourly: bool = False

    def from_row(self, start: int) -> SensorChange:
        """Return the sensor's changes from grid row `start` on."""
        return SensorChange(dz=self.dz[start:], dk=self.dk[start:], hourly=self.hourly)


def hourly_sum(*, x: np.ndarray) -> np.ndarray:
    """Return `x[t] + x[t - 1]`, the change over two half-hours when `x` is a half-hour change (NaN in row 0 and where either is NaN)."""
    out = np.full(x.shape, np.nan)
    out[1:] = x[1:] + x[:-1]
    return out


def align_reading(
    *, dm: np.ndarray, masked: np.ndarray, sensor: SensorChange
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reading's fast change and the transfer mask on the sensor's time step.

    For a half-hourly sensor both are returned unchanged. For an hourly sensor the reading's fast
    change is the sum of its two half-hour fast changes, and a row is masked when either half-hour
    change is.
    """
    if not sensor.hourly:
        return dm, masked
    out_mask = masked.copy()
    out_mask[1:] |= masked[:-1]
    return hourly_sum(x=dm), out_mask


def farm_sensor(*, z: np.ndarray, u_clear: np.ndarray, lit: np.ndarray) -> SensorChange:
    """Return the farm sensor's changes from the mean share `z` of its farms."""
    n_days = (len(z) + core.HALF_HOURS_PER_DAY - 1) // core.HALF_HOURS_PER_DAY
    c_z = trailing_capacity(z=z, u_clear=u_clear, lit=lit, n_days=n_days)
    return SensorChange(
        dz=fast_sensor_change(z=z, u_clear=u_clear, lit=lit),
        dk=index_change(z=z, u_clear=u_clear, lit=lit, c_z_day=c_z),
    )


def cams_sensor(*, s: Any, lit: np.ndarray) -> SensorChange:
    """Return the CAMS sensor on hour-ending rows: the hourly `du~` and the hourly change of the clear-sky index.

    `s` is a `Sky` of experiment 722 (`u`, `u_clear`, `ghi`, `ghi_clear`, `zenith`). The index of
    experiment 39's frame is held inside each hour, so its half-hour change on an hour-ending row
    is 0. The sensor therefore sums the two half-hour changes that end at the row, which is the
    change from `t - 2` to `t`, and the reading's fast change is summed the same way
    (`align_reading`). RULED (review finding 5).
    """
    n_days = (len(s.u) + core.HALF_HOURS_PER_DAY - 1) // core.HALF_HOURS_PER_DAY
    k = sky.clear_sky_index(ghi=s.ghi, ghi_clear=s.ghi_clear)
    du = sky.fill_sky_change(
        du=sky.fast_irradiance_change(u=s.u, u_clear=s.u_clear, k=k, lit=lit, n_days=n_days),
        zenith=s.zenith,
    )
    dk = sky.fast_k_change(k=k, lit=lit)
    hour = hour_end_rows(n=len(k))
    return SensorChange(
        dz=np.where(hour, hourly_sum(x=du), np.nan),
        dk=np.where(hour, hourly_sum(x=dk), np.nan),
        hourly=True,
    )


def meter_sensor(*, residual: np.ndarray, lit: np.ndarray, farm_dk: np.ndarray) -> SensorChange:
    """Return the meter sensor: `-d R` of the leave-out residual on lit pairs.

    PROVISIONAL: the plan page gives the meter sensor no sky index, so the usable-pair rule reads
    the farm sensor's index change `farm_dk` for it. The meter sensor carries no verdict.
    """
    d = np.full(residual.shape, np.nan)
    d[1:] = -(residual[1:] - residual[:-1])
    return SensorChange(dz=np.where(_lit_pair(lit=lit), d, np.nan), dk=farm_dk)


def shifted_sensor(*, sensor: SensorChange, days: int = core.PLACEBO_SHIFT_DAYS) -> SensorChange:
    """Return the sensor's fast change shifted `days` days later, so that row `t` reads `dz` of `t - days`.

    RULED: only `dz` is shifted. The index change `dk` stays the real sensor's, so the usable-pair
    rule selects the same pairs as the real sensor and the placebo is read on the area's stretches
    and pairs.
    """
    rows = days * core.HALF_HOURS_PER_DAY
    return SensorChange(
        dz=core722.shift_rows(x=sensor.dz, rows=rows), dk=sensor.dk, hourly=sensor.hourly
    )


# ---------------------------------------------------------------------------------------------
# The sensor audit (job W0)
# ---------------------------------------------------------------------------------------------
def resolution_shares(*, k: np.ndarray, u_clear: np.ndarray, lit: np.ndarray) -> dict[str, float]:
    """Return the share of hour-ending lit pairs on which `k` is unchanged, and on which `u_clear` is.

    The first share is expected at 100% and the second at 0%. Report only, apart from the arc term.
    """
    rows = hour_end_rows(n=len(k)) & _lit_pair(lit=lit)
    dk = np.full(k.shape, np.nan)
    dk[1:] = k[1:] - k[:-1]
    du = arc_change(u_clear=u_clear, lit=lit)
    k_ok = rows & np.isfinite(dk)
    u_ok = rows & np.isfinite(du)
    return {
        "share_index_unchanged": float((dk[k_ok] == 0).mean()) if k_ok.any() else float("nan"),
        "share_u_clear_unchanged": float((du[u_ok] == 0).mean()) if u_ok.any() else float("nan"),
        "n_rows_index": int(k_ok.sum()),
        "n_rows_u_clear": int(u_ok.sum()),
    }


def correlation(
    *, a: np.ndarray, b: np.ndarray, where: np.ndarray | None = None
) -> tuple[float, int]:
    """Return the Pearson correlation of two arrays over rows where both are finite (and `where`)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if where is not None:
        ok &= where
    n = int(ok.sum())
    if n < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan"), n
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), n


def arc_removal_r(*, dz: np.ndarray, d_u_clear: np.ndarray) -> tuple[float, int]:
    """Return the correlation of a farm's `dZ~` with `d u_clear` over daylight half-hours."""
    return correlation(a=dz, b=d_u_clear)


def coherence_r(
    *, dz_of: Mapping[str, np.ndarray], farm: str, in_season: np.ndarray
) -> tuple[float, int]:
    """Return a farm's correlation with the mean `dZ~` of the other farms of its host's sensor.

    The rows are daylight half-hours between April and September (`in_season`).
    """
    others = [v for k, v in dz_of.items() if k != farm]
    if not others:
        return float("nan"), 0
    mean_others = np.where(
        np.isfinite(np.stack(others)).all(axis=0), np.stack(others).mean(axis=0), np.nan
    )
    return correlation(a=dz_of[farm], b=mean_others, where=in_season)


def keep_coherent(
    *, dz_of: Mapping[str, np.ndarray], in_season: np.ndarray
) -> tuple[list[str], dict[str, float]]:
    """Return the farms of a host's sensor that pass the coherence floor of 0.5, and each farm's `r`.

    A farm under 0.5 is dropped from this host's sensor only. The pass is single: the others' mean
    is formed from every farm that entered the check.
    """
    r_of = {f: coherence_r(dz_of=dz_of, farm=f, in_season=in_season)[0] for f in dz_of}
    return [f for f, r in r_of.items() if core.coherence_ok(r=r)], r_of


# ---------------------------------------------------------------------------------------------
# Readings: folding and the reading's fast change
# ---------------------------------------------------------------------------------------------
def fold(
    *,
    signed_mw: np.ndarray,
    capacity_mw: float,
    floor_points: float = core.REACTIVE_FLOOR_POINTS,
    substeps: int = core.REACTIVE_SUBSTEPS,
) -> np.ndarray:
    """Return the folded reading `sqrt(X^2 + Q^2)` averaged within each half-hour.

    PROVISIONAL: "within-half-hour averaging" is not defined on the plan page. The signed series
    is interpolated linearly between the previous row and this row at `substeps` sub-steps (the
    design's section 2.5 says "for example 6"), and the half-hour's reading is the mean of
    `sqrt(X^2 + Q^2)` over the sub-steps. `Q` is `floor_points` points (1% of capacity each). The
    first row and any row next to a missing value are NaN.
    """
    q = floor_points / 100.0 * capacity_mw
    out = np.full(signed_mw.shape, np.nan)
    prev, cur = signed_mw[:-1], signed_mw[1:]
    frac = (np.arange(1, substeps + 1) / substeps)[:, None]
    path = prev[None, :] + frac * (cur - prev)[None, :]
    out[1:] = np.sqrt(path**2 + q**2).mean(axis=0)
    return out


def farm_norm(*, power_mw: np.ndarray, zenith_deg: np.ndarray, ghi: np.ndarray) -> np.ndarray:
    """Return a farm's recorded power over its own daylight 99th percentile: the series that is injected.

    RULED: this is experiment 720's `farm_norm` (function `farm_norm` of `e00506_data.py`, in the
    folder `00506-mva-sign-evidence-dip-depth-and-gated-solar-rule`, which experiment 720's script
    `e00720_rung0b.py` calls on branch `msa-mva-to-mw-rung-0`), copied here because that function
    reads a stored frame by path. The series is 0 at night (a zenith of at least
    `FARM_MAX_ZENITH_DEG`), NaN where the power is missing or is a dropout (0 or less with an
    irradiance above `FARM_DROPOUT_GHI`), and the power clipped at 0 and divided by the
    `FARM_NORM_PERCENTILE`th percentile of the daylight power otherwise. The percentile is taken
    over the arrays given, which end before the holdout. The series is NaN everywhere when no
    daylight power exists. A grid half-hour with no frame row has a NaN zenith and power, so it is
    NaN here.
    """
    power = np.asarray(power_mw, dtype=float)
    zenith = np.asarray(zenith_deg, dtype=float)
    night = zenith >= core.FARM_MAX_ZENITH_DEG
    daylight = (zenith < core.FARM_MAX_ZENITH_DEG) & np.isfinite(power)
    if not daylight.any():
        return np.full(power.shape, np.nan)
    p99 = float(np.percentile(power[daylight], core.FARM_NORM_PERCENTILE))
    with np.errstate(invalid="ignore"):
        dropout = (power <= 0.0) & (np.asarray(ghi, dtype=float) > core.FARM_DROPOUT_GHI)
    clipped = np.clip(power, 0.0, None) / p99
    return np.where(night, 0.0, np.where(~np.isfinite(power) | dropout, np.nan, clipped))


def inject(*, signed_mw: np.ndarray, farm_norm_series: np.ndarray, scale: float) -> np.ndarray:
    """Return the host's signed series with a farm's normalised output added as generation: `X - scale * F`.

    `F` is `farm_norm`, so the host's series is NaN wherever the farm's series is NaN (a gap or a
    dropout). That is experiment 720's masking of the host where the farm is missing.
    """
    return signed_mw - scale * farm_norm_series


def level_reference_rows(
    *, k: np.ndarray, month: np.ndarray, hour: np.ndarray, before_row: int
) -> np.ndarray:
    """Return the rows on which an injection level's reference median is taken.

    RULED (experiment 720's definition, copied from `build_probe_world` in `e00720_rung0b.py` of the
    folder `00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters`, because that function needs
    experiment 720's panel and fold objects): a
    half-hour that starts from 10:00 to 14:00 UTC, from April to September, with a clear-sky index
    of at least 0.7, and a row before `before_row`. The caller also requires finite host and farm
    values (`injection_scale`).
    """
    rows = np.arange(len(k))
    return (
        np.isin(month, core.REFERENCE_MONTHS)
        & (hour >= core.MIDDAY_UTC_HOURS[0])
        & (hour < core.MIDDAY_UTC_HOURS[1])
        & np.isfinite(k)
        & (k >= core.SUNNY_MIN_K)
        & (rows < before_row)
    )


def reference_row_count(
    *, host_signed_mw: np.ndarray, farm_norm_series: np.ndarray, rows: np.ndarray
) -> int:
    """Return the reference rows on which the host's signed series and the farm's normalised output are finite."""
    return int((rows & np.isfinite(host_signed_mw) & np.isfinite(farm_norm_series)).sum())


def injection_scale(
    *,
    host_signed_mw: np.ndarray,
    farm_norm_series: np.ndarray,
    rows: np.ndarray,
    level: float,
) -> float:
    """Return the scale `a` in megawatts, so that the injected series is `a * F`, with `F` the `farm_norm` series.

    RULED (experiment 720's definition): the injected series is the farm's normalised output
    (`farm_norm`, between 0 and about 1) times `level` times the median absolute signed reading of
    the host on the reference rows (`level_reference_rows`). The farm's capacity does not enter.
    The median is NaN, and so is the scale, when fewer than 200 reference rows have both values.
    The caller logs and counts every NaN scale.
    """
    ok = rows & np.isfinite(host_signed_mw) & np.isfinite(farm_norm_series)
    if int(ok.sum()) < core.MIN_REFERENCE_ROWS:
        return float("nan")
    return level * float(np.median(np.abs(host_signed_mw[ok])))


def trailing_overcast_median(
    *,
    change: np.ndarray,
    key: np.ndarray,
    overcast_row: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Return, for each row in `rows`, the median change at the same local half-hour and day type on overcast days.

    The days are those of ages 2 to 91 before the row. NaN where no such day has a change.
    """
    half_hour = (key // 2) // 13
    day_type = key % 2
    ages = range(core.OVERCAST_LOOKBACK_DAYS[0], core.OVERCAST_LOOKBACK_DAYS[1] + 1)
    stack = np.full((len(ages), len(rows)), np.nan)
    for i, age in enumerate(ages):
        src = rows - core.HALF_HOURS_PER_DAY * age
        ok = src >= 0
        s = src[ok]
        r = rows[ok]
        match = (
            (key[s] >= 0)
            & (key[r] >= 0)
            & (half_hour[s] == half_hour[r])
            & (day_type[s] == day_type[r])
            & overcast_row[s]
            & np.isfinite(change[s])
        )
        stack[i, ok] = np.where(match, change[s], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
    return np.where(np.isfinite(stack).any(axis=0), med, np.nan)


def reading_fast_change(
    *,
    reading_mw: np.ndarray,
    capacity_mw: float,
    lit: np.ndarray,
    key: np.ndarray,
    overcast_row: np.ndarray,
    start: int,
) -> np.ndarray:
    """Return the reading's fast change `dm~_t = (m_t - m_{t-1}) - df_t` in points, for rows from `start` on.

    `df_t` is the median change of the reading at the same local half-hour and day type on
    overcast days of the trailing ages 2 to 91 days (version 2, section 6.2). PROVISIONAL: the
    overcast definition is the clear-sky-index definition of experiment 722 (a daily mean below
    0.35), which the plan page's removal of the intermittent class leaves untouched, and a row
    with no overcast day in its window is NaN. The result has the length of the later span, and is
    NaN unless both half-hours of the pair are lit.
    """
    points = reading_mw / capacity_mw * 100.0
    change = np.full(points.shape, np.nan)
    change[1:] = points[1:] - points[:-1]
    rows = np.arange(start, len(points))
    df = trailing_overcast_median(change=change, key=key, overcast_row=overcast_row, rows=rows)
    out = change[start:] - df
    return np.where(_lit_pair(lit=lit)[start:], out, np.nan)


def window_mask(*, windows: Sequence[tuple[int, int]], n: int) -> np.ndarray:
    """Return True at rows `t` whose change `t-1` to `t` has an end inside a window or within 1 hour of it.

    Each window is a pair of grid rows (first, last). A change is dropped when either of its two
    end stamps lies in `[first - 2, last + 2]` half-hours.
    """
    flag = np.zeros(n + 1, dtype=np.int64)
    margin = core.TRANSFER_MARGIN_HALF_HOURS
    for first, last in windows:
        lo, hi = max(first - margin, 0), min(last + margin, n - 1)
        if hi >= lo:
            flag[lo] += 1
            flag[hi + 1] -= 1
    inside = np.cumsum(flag)[:n] > 0
    out = inside.copy()
    out[1:] |= inside[:-1]
    return out


# ---------------------------------------------------------------------------------------------
# Stretches and concordance
# ---------------------------------------------------------------------------------------------
STRETCH_SCHEMA: Final[dict[str, Any]] = {
    "day": pl.Int64,
    "first": pl.Int64,
    "last": pl.Int64,
    "is_post_bounce": pl.Boolean,
    "n_post_bounce_in_day": pl.Int64,
    "truth": pl.String,
    **{f"c_{s}": pl.Float64 for s in SENSOR_NAMES},
    **{f"n_{s}": pl.Int64 for s in SENSOR_NAMES},
}
"""The columns of the stretch table of one world."""


def threshold_diagnostics(
    *, dm: np.ndarray, sensor: SensorChange, masked: np.ndarray
) -> dict[str, float]:
    """Return R3's diagnostics for one sensor: the two thresholds, the pass shares, the 99th percentile, and the largest value.

    The shares are over daylight pairs (rows where the reading's fast change and the sensor's
    index change both exist), and each threshold alone and both together. A masked row is excluded.
    """
    dm, masked = align_reading(dm=dm, masked=masked, sensor=sensor)
    ok = np.isfinite(dm) & np.isfinite(sensor.dk) & ~masked
    n = int(ok.sum())
    big_k = ok & (np.abs(sensor.dk) >= core.SIZE_OF_SKY_INDEX_CHANGE)
    big_m = ok & (np.abs(dm) >= core.SIZE_OF_READING_CHANGE_POINTS)

    def share(mask: np.ndarray) -> float:
        return float(mask.sum() / n) if n else float("nan")

    def p99(x: np.ndarray) -> float:
        v = np.abs(x[ok])
        return float(np.percentile(v, 99)) if v.size else float("nan")

    def largest(x: np.ndarray) -> float:
        v = np.abs(x[ok])
        return float(v.max()) if v.size else float("nan")

    return {
        "threshold_index": core.SIZE_OF_SKY_INDEX_CHANGE,
        "threshold_reading_points": core.SIZE_OF_READING_CHANGE_POINTS,
        "n_daylight_pairs": n,
        "share_index": share(big_k),
        "share_reading": share(big_m),
        "share_both": share(big_k & big_m),
        "p99_index_change": p99(sensor.dk),
        "max_index_change": largest(sensor.dk),
        "p99_reading_change": p99(dm),
        "max_reading_change": largest(dm),
    }


def usable_pairs(*, dm: np.ndarray, sensor: SensorChange, masked: np.ndarray) -> np.ndarray:
    """Return True at rows `t` where the pair `t - 1`, `t` is usable under the plan page's rule (`t - 2`, `t` for an hourly sensor)."""
    dm, masked = align_reading(dm=dm, masked=masked, sensor=sensor)
    return (
        np.isfinite(dm)
        & np.isfinite(sensor.dz)
        & np.isfinite(sensor.dk)
        & ~masked
        & (np.abs(sensor.dk) >= core.SIZE_OF_SKY_INDEX_CHANGE)
        & (np.abs(dm) >= core.SIZE_OF_READING_CHANGE_POINTS)
    )


def stretches_of(
    *, reading_mw: np.ndarray, signed_mw: np.ndarray, capacity_mw: float, lit: np.ndarray
) -> dict[int, list[tuple[int, int, bool]]]:
    """Return each day's stretches `(first, last, is_post_bounce)` from the raw reading's touches."""
    usable = np.isfinite(reading_mw) & np.isfinite(signed_mw)
    level = sky.touch_level(reading=reading_mw, lit=lit, capacity_mw=capacity_mw)
    touch = sky.touches(reading=reading_mw, lit=lit, level=level)
    return sky.day_stretches(touch=touch, lit=lit, usable=usable)


def stretch_table(
    *,
    stretches: dict[int, list[tuple[int, int, bool]]],
    signed_mw: np.ndarray,
    dm: np.ndarray,
    sensors: Mapping[str, SensorChange],
    masked: np.ndarray,
) -> pl.DataFrame:
    """Return one row per stretch with its truth and, for each sensor, its concordance and usable pairs.

    Args:
        stretches: Output of `stretches_of`, on the later span.
        signed_mw: The world's signed series, on the later span (negative is export).
        dm: The reading's fast change in points, on the later span.
        sensors: The sensors' changes by name, on the later span. A name absent from `SENSOR_NAMES`
            is ignored; a name of `SENSOR_NAMES` that is absent gives NaN and 0.
        masked: True where the transfer mask drops the pair, on the later span.

    Returns:
        A frame with the columns of `STRETCH_SCHEMA`. `C` is NaN with fewer than one usable pair.
    """
    usable = {name: usable_pairs(dm=dm, sensor=s, masked=masked) for name, s in sensors.items()}
    rows: list[dict[str, Any]] = []
    for day, spans in stretches.items():
        n_post = sum(1 for *_, post in spans if post)
        for first, last, post in spans:
            seg = signed_mw[first : last + 1]
            share_export = (
                float((seg < 0).mean()) if seg.size and np.isfinite(seg).all() else float("nan")
            )
            row: dict[str, Any] = {
                "day": day,
                "first": first,
                "last": last,
                "is_post_bounce": post,
                "n_post_bounce_in_day": n_post,
                "truth": core.truth_of(share_export_half_hours=share_export)
                if np.isfinite(share_export)
                else "mixed",
            }
            for name in SENSOR_NAMES:
                if name in sensors:
                    sl = slice(first + 1, last + 1)
                    dm_s, _ = align_reading(dm=dm, masked=masked, sensor=sensors[name])
                    c, n, _ = sky.concordance(
                        d_reading=dm_s[sl], d_sky=sensors[name].dz[sl], usable=usable[name][sl]
                    )
                    row[f"c_{name}"], row[f"n_{name}"] = c, n
                else:
                    row[f"c_{name}"], row[f"n_{name}"] = float("nan"), 0
            rows.append(row)
    return pl.DataFrame(rows, schema=STRETCH_SCHEMA)


def scorable(
    *, table: pl.DataFrame, sensor: str, post_bounce_one_stretch_days: bool = True
) -> pl.DataFrame:
    """Return the scorable stretches of one sensor: post-bounce, on one-stretch days, with enough usable pairs.

    The bars bind on days with exactly one post-bounce stretch, and a stretch is scored when it
    holds at least 8 usable pairs. Set `post_bounce_one_stretch_days` False for a population
    without the one-stretch rule (job W3's anchor stretches and the D3 control are filtered by the
    caller).
    """
    keep = pl.col("is_post_bounce") & (pl.col(f"n_{sensor}") >= core.MIN_USABLE_PAIRS)
    if post_bounce_one_stretch_days:
        keep = keep & (pl.col("n_post_bounce_in_day") == 1)
    return table.filter(keep)


def stage_counts(
    *,
    reading_mw: np.ndarray,
    signed_mw: np.ndarray,
    lit: np.ndarray,
    masked: np.ndarray,
    dm: np.ndarray,
    sensor: SensorChange,
    table: pl.DataFrame,
    n_days: int,
) -> dict[str, int]:
    """Return R1's counts at each filter stage for one world and sensor.

    The stages are rows with a finite reading and signed value, daylight rows, daylight rows
    outside a transfer window, usable pairs, scorable stretches, and the scorable stretches in each
    truth class. The count of days is the number of days with a stretch.
    """
    finite = np.isfinite(reading_mw) & np.isfinite(signed_mw)
    day_lit = finite & lit
    out_window = day_lit & ~masked
    use = usable_pairs(dm=dm, sensor=sensor, masked=masked)
    sc = scorable(table=table, sensor="farm")
    return {
        "n_rows_finite": int(finite.sum()),
        "n_rows_daylight": int(day_lit.sum()),
        "n_rows_outside_transfer_window": int(out_window.sum()),
        "n_usable_pairs": int(use.sum()),
        "n_scorable_stretches": sc.height,
        "n_scorable_export": int(sc.filter(pl.col("truth") == "export").height),
        "n_scorable_import": int(sc.filter(pl.col("truth") == "import").height),
        "n_scorable_mixed": int(sc.filter(pl.col("truth") == "mixed").height),
        "n_days_with_stretch": int(table["day"].n_unique()) if table.height else 0,
        "n_days": n_days,
    }


def peak_lag_on_anchors(
    *, dm: np.ndarray, dz: np.ndarray, stretches: dict[int, list[tuple[int, int, bool]]]
) -> tuple[int, float]:
    """Return the lag at which the reading's and the sensor's fast changes correlate most, on anchor stretches.

    Only the pairs inside an anchor stretch (the first and the last stretch of a day) are used.
    """
    rows = np.zeros(len(dm), dtype=bool)
    for spans in stretches.values():
        for first, last, post in spans:
            if not post:
                rows[first + 1 : last + 1] = True
    return sky.peak_lag(x=np.where(rows, dm, np.nan), y=np.where(rows, dz, np.nan))


# ---------------------------------------------------------------------------------------------
# Closure fits (read-outs)
# ---------------------------------------------------------------------------------------------
def hourly_demeaned(*, x: np.ndarray, key: np.ndarray, quiet: np.ndarray) -> np.ndarray:
    """Return the hourly change of `x` less its matched quiet-time mean."""
    return core722.demean_matched(x=core722.change(x=x, lag=HOURLY_LAG), key=key, quiet=quiet)


def closure_fit(
    *,
    response: np.ndarray,
    regressors: Mapping[str, np.ndarray],
    rows: np.ndarray,
    key: np.ndarray,
    quiet: np.ndarray,
    day: np.ndarray,
    weights: np.ndarray,
) -> core722.Fit:
    """Regress the response's hourly change on the regressors' hourly changes, through the origin.

    Every series is a level series on the grid. The changes are hourly, de-meaned by matched quiet
    time, and used on hour-ending `rows` where `rows` is True. The standard errors come from a
    bootstrap of whole days. The result is a read-out with no bar.
    """
    names = tuple(regressors)
    y = np.where(rows, hourly_demeaned(x=response, key=key, quiet=quiet), np.nan)
    x = np.stack(
        [
            np.where(rows, hourly_demeaned(x=regressors[n], key=key, quiet=quiet), np.nan)
            for n in names
        ],
        axis=1,
    )
    return core722.bootstrap_ols(y=y, x=x, day=day, weights=weights, names=names)


def power_row(
    *, response_change: np.ndarray, regressor_change: np.ndarray, rows: np.ndarray
) -> dict[str, float]:
    """Return a power-table row: the row count, the two standard deviations, and the predicted standard error."""
    ok = rows & np.isfinite(response_change) & np.isfinite(regressor_change)
    n = int(ok.sum())
    sd_y = float(response_change[ok].std()) if n > 1 else float("nan")
    sd_x = float(regressor_change[ok].std()) if n > 1 else float("nan")
    se = core.predicted_se(sd_response=sd_y, sd_regressor=sd_x, n_rows=n) if n > 1 else float("nan")
    return {
        "n_rows": n,
        "sd_response": sd_y,
        "sd_regressor": sd_x,
        "predicted_se": se,
        "detection_limit": core.detection_limit(se=se),
    }


def utc_window_rows(*, start: datetime, end: datetime | None) -> tuple[int, int]:
    """Return the grid rows (first, last) of a window from `start` to `end`, or of the calendar day of `start`.

    A window without an end time is the UTC calendar day of its start.
    """
    start = start if start.tzinfo is not None else start.replace(tzinfo=UTC)
    end = end if end is None or end.tzinfo is not None else end.replace(tzinfo=UTC)
    first_day = datetime(start.year, start.month, start.day, tzinfo=UTC)
    if end is None:
        a, b = first_day, first_day + timedelta(days=1)
    else:
        a, b = start.astimezone(UTC), end.astimezone(UTC)
    return row_of_instant(instant=a), row_of_instant(instant=b)


def pairs_of(*, hosts: Sequence[str], farms: Sequence[str]) -> list[tuple[str, str]]:
    """Return every host and farm pair."""
    return list(itertools.product(hosts, farms))


def hourly_mask(*, masked: np.ndarray) -> np.ndarray:
    """Return True at rows whose hourly change (lag 2) includes a masked half-hour change."""
    out = masked.copy()
    out[1:] |= masked[:-1]
    return out


def closure_rows(
    *,
    zenith: np.ndarray,
    masked: np.ndarray,
    start: int,
    kind: str,
    min_elevation_deg: float | None = None,
) -> np.ndarray:
    """Return the hour-ending rows of the later span for a closure fit.

    Args:
        zenith: The host's sun zenith in degrees on the grid.
        masked: The host's transfer mask on the grid.
        start: First grid row of the later span.
        kind: `lit` (sun at 10 degrees or more at both ends of the hour), `dark` (sun at or below
            0 degrees at both ends), or `all`.
        min_elevation_deg: Overrides the 10 degrees of `lit` (for example 30 for D10's export hours).

    Returns:
        A boolean array on the grid; masked hours are excluded.
    """
    elevation = 90.0 - zenith
    limit = core.DAYLIGHT_ELEVATION_DEG if min_elevation_deg is None else min_elevation_deg
    lit = np.isfinite(elevation) & (elevation >= limit)
    dark = np.isfinite(elevation) & (elevation <= core722.DARK_ELEVATION_DEG)
    pick = {"lit": lit, "dark": dark, "all": np.isfinite(elevation)}[kind]
    both = pick.copy()
    both[HOURLY_LAG:] &= pick[:-HOURLY_LAG]
    rows = hour_end_rows(n=len(zenith)) & both & ~hourly_mask(masked=masked)
    rows[:start] = False
    return rows

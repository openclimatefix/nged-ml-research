r"""Build the difficult-row slice masks of experiment 442 from ENS weather and the calendar.

Every mask is defined from the ensemble-mean weather of the forecast's own run (the
`diag_temperature` and `diag_csi` columns experiment 437 stored beside its inputs) and from the
calendar, never from the realised weather and never from a score. A row is one Primary, forecast
origin, and valid time. The masks are:

- `cold_winter_night`: solar elevation below 0 degrees, local month November to March, and ENS-mean
  2 m temperature at or below the Primary's 10th percentile of winter-night ENS temperatures in the
  fold's training window (the 730 days before the fold's refit, from the first stored run on
  2024-04-01; leads above 0 and up to 240 hours).
- `cold_sunny_day` and `cold_overcast_day`: daylight half-hours (elevation above 0) of a local day
  whose ENS-mean temperature, averaged over the day's 46 to 50 half-hours, is below 5 C, and whose
  daily mean ENS clear-sky index (daylight half-hours with a finite value, at least 8 of them) is at
  least 0.7 (sunny) or at most 0.3 (overcast).
- `christmas`: all half-hours of a local date from 24 December to 2 January.
- `other_bank_holiday`: all half-hours of Good Friday to Easter Monday, the early-May, spring, and
  August bank holidays (Mondays) of England and Wales.
- `heat_wave_day`: all half-hours of a local day whose ENS-mean daily maximum temperature is at or
  above the Primary's 95th percentile of June to August daily maxima in the fold's training window,
  when the day before or after in the same run is also at or above it.

A local day counts as complete when the run holds at least 46 of its half-hours, so the origin day
(28 half-hours) and the last day of a run (20 half-hours) never enter a daily statistic. The mask
table covers the lead band (0, 240] hours, the protocol's headline band, for the origins of the six
folds of experiment 442.

```bash
# PYTHONPATH as in e00442_score.py
uv run --no-sync python ml-experiments/00442-*/e00442_masks.py
```
"""

import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00437_features as feats  # ty: ignore[unresolved-import]
import e00442_fit as fit  # ty: ignore[unresolved-import]
import polars as pl

from studies import solar

EXPERIMENT_DIR_NAME: Final[str] = fit.EXPERIMENT_DIR_NAME
WORK_DIR: Final[Path] = fit.OUT_DIR
MASK_PATH: Final[Path] = WORK_DIR / "masks.parquet"
THRESHOLD_PATH: Final[Path] = WORK_DIR / "thresholds.parquet"
MASK_NAMES: Final[tuple[str, ...]] = (
    "cold_winter_night",
    "cold_sunny_day",
    "cold_overcast_day",
    "christmas",
    "other_bank_holiday",
    "heat_wave_day",
)
WINDOW_DAYS: Final[int] = base.WINDOW_DAYS
MAX_LEAD_HOURS: Final[float] = 240.0
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
SUMMER_MONTHS: Final[tuple[int, ...]] = (6, 7, 8)
WINTER_NIGHT_QUANTILE: Final[float] = 0.10
HEAT_QUANTILE: Final[float] = 0.95
COLD_DAY_C: Final[float] = 5.0
SUNNY_INDEX: Final[float] = 0.7
OVERCAST_INDEX: Final[float] = 0.3
MIN_DAY_ROWS: Final[int] = 46
MIN_DAYLIGHT_ROWS: Final[int] = 8
TIMEZONE: Final[str] = "Europe/London"


def easter_sunday(*, year: int) -> date:
    """Return the date of Easter Sunday (the anonymous Gregorian algorithm).

    Args:
        year: A calendar year.

    Returns:
        Easter Sunday of that year.
    """
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = (h + el - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def bank_holiday_dates(*, years: tuple[int, ...]) -> list[date]:
    """Return the non-Christmas bank holidays of England and Wales that the mask names.

    These are Good Friday to Easter Monday, the early-May bank holiday (first Monday in May), the
    spring bank holiday (last Monday in May), and the summer bank holiday (last Monday in August).

    Args:
        years: Calendar years to list.

    Returns:
        The dates, in no particular order.
    """
    out: list[date] = []
    for year in years:
        easter = easter_sunday(year=year)
        out.extend(easter + timedelta(days=offset) for offset in (-2, -1, 0, 1))
        first_may = date(year, 5, 1)
        out.append(first_may + timedelta(days=(7 - first_may.weekday()) % 7))
        for month in (5, 8):
            last = date(year, month, 31)
            out.append(last - timedelta(days=last.weekday()))
    return out


def site_rows(*, site: str, latitude: float, longitude: float) -> pl.DataFrame:
    """Return one Primary's ENS table with lead, elevation, and local date and month columns.

    Args:
        site: The Primary's label, such as `D2`.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        One row per forecast origin and valid time, with the lead in hours.
    """
    table = pl.read_parquet(
        feats.table_path(site=site),
        columns=["origin_us", "valid_us", "diag_temperature", "diag_csi"],
    ).select(
        power_fcst_init_time=pl.col("origin_us").cast(pl.Datetime("us", "UTC")),
        valid_time=pl.col("valid_us").cast(pl.Datetime("us", "UTC")),
        temperature=pl.col("diag_temperature").cast(pl.Float64),
        csi=pl.col("diag_csi").cast(pl.Float64),
    )
    if table["temperature"].is_nan().any() or table["temperature"].is_null().any():
        msg = f"{site}: the ENS temperature holds a NaN or null"
        raise ValueError(msg)
    stamps = table.select("valid_time").unique().sort("valid_time")
    zenith = solar.zenith(
        stamps=stamps["valid_time"].dt.offset_by("-15m"), latitude=latitude, longitude=longitude
    )
    elevation = stamps.with_columns(elevation_deg=pl.Series(90.0 - zenith))
    start_local = pl.col("valid_time").dt.offset_by("-30m").dt.convert_time_zone(TIMEZONE)
    return (
        table.join(elevation, on="valid_time", how="left", validate="m:1")
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            local_date=start_local.dt.date(),
            local_month=start_local.dt.month(),
        )
        .filter((pl.col("lead_hours") > 0) & (pl.col("lead_hours") <= MAX_LEAD_HOURS))
    )


def daily_statistics(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return per origin and local date the complete-day temperature and clear-sky statistics.

    Args:
        rows: The output of `site_rows`.

    Returns:
        Columns `power_fcst_init_time`, `local_date`, `n_rows`, `day_mean_temperature`,
        `day_max_temperature`, `day_mean_csi`, and `n_daylight_csi`.
    """
    daylight_csi = pl.when((pl.col("elevation_deg") > 0) & pl.col("csi").is_finite()).then(
        pl.col("csi")
    )
    return rows.group_by("power_fcst_init_time", "local_date").agg(
        n_rows=pl.len(),
        day_mean_temperature=pl.col("temperature").mean(),
        day_max_temperature=pl.col("temperature").max(),
        day_mean_csi=daylight_csi.mean(),
        n_daylight_csi=daylight_csi.count(),
    )


def thresholds_for(*, rows: pl.DataFrame, daily: pl.DataFrame, refit: datetime) -> dict[str, float]:
    """Return the Primary's two percentile thresholds from the runs before a refit.

    Args:
        rows: The output of `site_rows`.
        daily: The output of `daily_statistics` for the same rows.
        refit: The fold's refit time; only runs with an origin in the 730 days before it count.

    Returns:
        `winter_night_p10` and `summer_daily_max_p95` in degrees C.
    """
    first = refit - timedelta(days=WINDOW_DAYS)
    in_window = pl.col("power_fcst_init_time").is_between(first, refit, closed="left")
    nights = rows.filter(
        in_window & pl.col("local_month").is_in(WINTER_MONTHS) & (pl.col("elevation_deg") < 0)
    )
    summers = daily.join(
        rows.select("power_fcst_init_time", "local_date", "local_month").unique(
            subset=["power_fcst_init_time", "local_date"]
        ),
        on=["power_fcst_init_time", "local_date"],
        how="left",
    ).filter(
        in_window & pl.col("local_month").is_in(SUMMER_MONTHS) & (pl.col("n_rows") >= MIN_DAY_ROWS)
    )
    return {
        "winter_night_p10": float(
            cast("float", nights["temperature"].quantile(WINTER_NIGHT_QUANTILE))
        ),
        "summer_daily_max_p95": float(
            cast("float", summers["day_max_temperature"].quantile(HEAT_QUANTILE))
        ),
        "n_winter_night_rows": float(nights.height),
        "n_summer_days": float(summers.height),
    }


def masks_for_fold(
    *, rows: pl.DataFrame, daily: pl.DataFrame, month: str, thresholds: dict[str, float]
) -> pl.DataFrame:
    """Return one Primary's mask columns for the runs whose origin falls in a fold's month.

    Args:
        rows: The output of `site_rows`.
        daily: The output of `daily_statistics` for the same rows.
        month: The fold, such as `2025-12`.
        thresholds: The output of `thresholds_for` for this Primary and fold.

    Returns:
        The key columns, `fold`, `lead_hours`, `local_date`, `elevation_deg`, and the `MASK_NAMES`.
    """
    in_fold = pl.col("power_fcst_init_time").dt.strftime("%Y-%m") == month
    fold_rows = rows.filter(in_fold)
    complete = daily.filter(in_fold & (pl.col("n_rows") >= MIN_DAY_ROWS)).sort(
        "power_fcst_init_time", "local_date"
    )
    hot = pl.col("day_max_temperature") >= thresholds["summer_daily_max_p95"]
    complete = complete.with_columns(
        hot_day=hot,
        heat_wave=hot
        & (
            hot.shift(1).over("power_fcst_init_time").fill_null(False)
            & (pl.col("local_date").diff().over("power_fcst_init_time") == timedelta(days=1))
            | hot.shift(-1).over("power_fcst_init_time").fill_null(False)
            & (pl.col("local_date").diff(-1).over("power_fcst_init_time") == timedelta(days=-1))
        ),
        cold_day=pl.col("day_mean_temperature") < COLD_DAY_C,
        sunny=pl.col("n_daylight_csi") >= MIN_DAYLIGHT_ROWS,
    ).select(
        "power_fcst_init_time",
        "local_date",
        "heat_wave",
        cold_sunny=pl.col("cold_day") & pl.col("sunny") & (pl.col("day_mean_csi") >= SUNNY_INDEX),
        cold_overcast=pl.col("cold_day")
        & pl.col("sunny")
        & (pl.col("day_mean_csi") <= OVERCAST_INDEX),
    )
    years = tuple(sorted({d.year for d in fold_rows["local_date"].unique().to_list()}))
    holidays = bank_holiday_dates(years=years)
    return (
        fold_rows.join(complete, on=["power_fcst_init_time", "local_date"], how="left")
        .with_columns(
            fold=pl.lit(month),
            cold_winter_night=pl.col("local_month").is_in(WINTER_MONTHS)
            & (pl.col("elevation_deg") < 0)
            & (pl.col("temperature") <= thresholds["winter_night_p10"]),
            cold_sunny_day=pl.col("cold_sunny").fill_null(False) & (pl.col("elevation_deg") > 0),
            cold_overcast_day=pl.col("cold_overcast").fill_null(False)
            & (pl.col("elevation_deg") > 0),
            christmas=(pl.col("local_month") == 12) & (pl.col("local_date").dt.day() >= 24)
            | (pl.col("local_month") == 1) & (pl.col("local_date").dt.day() <= 2),
            other_bank_holiday=pl.col("local_date").is_in(holidays),
            heat_wave_day=pl.col("heat_wave").fill_null(False),
        )
        .select(
            "power_fcst_init_time",
            "valid_time",
            "fold",
            "lead_hours",
            "local_date",
            "elevation_deg",
            "temperature",
            *MASK_NAMES,
        )
    )


def build() -> None:
    """Write the mask table and the per-Primary, per-fold thresholds."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pieces: list[pl.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    for primary in base.primaries().iter_rows(named=True):
        site = primary["site"]
        rows = site_rows(
            site=site, latitude=float(primary["latitude"]), longitude=float(primary["longitude"])
        )
        daily = daily_statistics(rows=rows)
        for month in fit.MONTHS:
            refit = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
            thresholds = thresholds_for(rows=rows, daily=daily, refit=refit)
            threshold_rows.append({"label": site, "fold": month, **thresholds})
            pieces.append(
                masks_for_fold(
                    rows=rows, daily=daily, month=month, thresholds=thresholds
                ).with_columns(label=pl.lit(site))
            )
        print(f"{site}: built")
    table = pl.concat(pieces)
    table.write_parquet(MASK_PATH)
    pl.DataFrame(threshold_rows).write_parquet(THRESHOLD_PATH)
    print(f"{MASK_PATH}: {table.height:,} rows")
    print(pl.DataFrame(threshold_rows).sort("label", "fold"))
    print(
        table.group_by("fold")
        .agg(*[pl.col(m).sum().alias(m) for m in MASK_NAMES], rows=pl.len())
        .sort("fold")
    )


def main() -> None:
    """Build the masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()

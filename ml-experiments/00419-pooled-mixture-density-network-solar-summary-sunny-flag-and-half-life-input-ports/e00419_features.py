"""Extra input columns of experiment 419: the solar summary and the sunny flag.

Both are built from a run's ensemble-mean shortwave (`mean_shortwave` of experiment 397's summary
tables), so the training rows and the forecast rows use the same function. A column reads only its
own run: the daily means group by the run's `power_fcst_init_time`, so no value comes from a later
run or from the observed power.

- `shortwave_day_mean`: the mean over the local day of the valid time, over the steps of the run
  that fall in that day (a day cut short by the run's end or start uses the steps available);
- `shortwave_prev_day_mean`: the same for the previous local day of the same run, replaced by the
  day's own mean where the run holds no step of the previous day (the first local day of a run);
- `sunny_flag`: 1 where the ensemble-mean shortwave exceeds 0.6 times the Haurwitz clear-sky
  irradiance, as experiment 381's arm A4;
- `sunny_times_shortwave`: the flag times the ensemble-mean shortwave.
"""

from typing import Final

import numpy as np
import polars as pl

SOLAR_COLUMNS: Final[tuple[str, ...]] = ("shortwave_day_mean", "shortwave_prev_day_mean")
SUNNY_COLUMNS: Final[tuple[str, ...]] = ("sunny_flag", "sunny_times_shortwave")
GROUP_COLUMNS: Final[dict[str, tuple[str, ...]]] = {"solar": SOLAR_COLUMNS, "sunny": SUNNY_COLUMNS}
SUNNY_INDEX: Final[float] = 0.6
HAURWITZ_PEAK: Final[float] = 1098.0
HAURWITZ_DECAY: Final[float] = 0.057
MIN_COSINE: Final[float] = 0.01
MIN_CLEAR_SKY: Final[float] = 1.0


def extra_column_names(*, groups: tuple[str, ...]) -> list[str]:
    """Return the added column names of the groups, in matrix order."""
    return [c for g in groups for c in GROUP_COLUMNS[g]]


def _local_date(*, valid_us: str) -> pl.Expr:
    """Return the local date of a valid time held as microseconds (the half-hour ending at it)."""
    return (
        pl.col(valid_us)
        .cast(pl.Datetime("us", "UTC"))
        .dt.offset_by("-30m")
        .dt.convert_time_zone("Europe/London")
        .dt.date()
    )


def daily_means(
    *, origin_us: np.ndarray, valid_us: np.ndarray, shortwave: np.ndarray
) -> pl.DataFrame:
    """Return, per run and local date, the mean shortwave and the previous local day's mean.

    Args:
        origin_us: Each step's run origin (`power_fcst_init_time`), microseconds since the epoch.
        valid_us: Each step's valid time, microseconds since the epoch.
        shortwave: Each step's ensemble-mean shortwave.

    Returns:
        One row per run and local date with `origin_us`, `local_date`, `shortwave_day_mean`, and
        `shortwave_prev_day_mean` (null where the run holds no step of the previous local day).
    """
    steps = pl.DataFrame(
        {"origin_us": origin_us, "valid_us": valid_us, "shortwave": shortwave.astype(np.float64)}
    ).with_columns(local_date=_local_date(valid_us="valid_us"))
    daily = steps.group_by("origin_us", "local_date").agg(
        pl.col("shortwave").mean().alias("shortwave_day_mean")
    )
    previous = daily.with_columns(local_date=pl.col("local_date") + pl.duration(days=1)).rename(
        {"shortwave_day_mean": "shortwave_prev_day_mean"}
    )
    return daily.join(
        previous, on=["origin_us", "local_date"], how="left", validate="1:1", maintain_order="left"
    )


def solar_columns(
    *, origin_us: np.ndarray, valid_us: np.ndarray, daily: pl.DataFrame
) -> np.ndarray:
    """Return the solar-summary columns (rows, 2) of rows, looked up in `daily` by run and date.

    A previous-day mean that is missing (the first local day of a run) is replaced by the day's own
    mean.

    Raises:
        ValueError: If a row's run and local date are not in `daily`.
    """
    rows = pl.DataFrame({"origin_us": origin_us, "valid_us": valid_us}).with_columns(
        local_date=_local_date(valid_us="valid_us")
    )
    joined = rows.join(
        daily, on=["origin_us", "local_date"], how="left", validate="m:1", maintain_order="left"
    )
    if joined["shortwave_day_mean"].null_count():
        msg = "a row's run and local date are missing from the daily means"
        raise ValueError(msg)
    joined = joined.with_columns(
        pl.col("shortwave_prev_day_mean").fill_null(pl.col("shortwave_day_mean"))
    )
    return joined.select(SOLAR_COLUMNS).to_numpy().astype(np.float32)


def sunny_columns(*, shortwave: np.ndarray, cos_zenith: np.ndarray) -> np.ndarray:
    """Return the sunny flag and its product with shortwave (rows, 2), as 381 arm A4."""
    cosine = np.clip(cos_zenith.astype(np.float64), 0.0, None)
    clear = HAURWITZ_PEAK * cosine * np.exp(-HAURWITZ_DECAY / np.maximum(cosine, MIN_COSINE))
    sw = shortwave.astype(np.float64)
    index = np.where(clear > MIN_CLEAR_SKY, sw / np.maximum(clear, MIN_CLEAR_SKY), 0.0)
    flag = (index > SUNNY_INDEX).astype(np.float64)
    return np.stack([flag, flag * sw], axis=1).astype(np.float32)


def extra_matrix(
    *,
    groups: tuple[str, ...],
    origin_us: np.ndarray,
    valid_us: np.ndarray,
    shortwave: np.ndarray,
    cos_zenith: np.ndarray,
    daily: pl.DataFrame,
) -> np.ndarray:
    """Return the added columns (rows, n) of the groups, in the order of `extra_column_names`."""
    parts = []
    for group in groups:
        if group == "solar":
            parts.append(solar_columns(origin_us=origin_us, valid_us=valid_us, daily=daily))
        else:
            parts.append(sunny_columns(shortwave=shortwave, cos_zenith=cos_zenith))
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(origin_us), 0), np.float32)

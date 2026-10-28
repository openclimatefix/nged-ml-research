"""The row slices of the demand evaluation protocol: which half-hours each score is reported over.

Every slice is a boolean column on a table of scored half-hours. `add_slice_columns` adds one column
per slice, named as in `SLICE_NAMES`, so a metric table can group by slice without each experiment
rewriting the definition. The definitions are those of the substation cards (experiments 250 and
252, `e00250_score.py` and `e00252_score.py`), with the deviations listed under "Deviations from
the cards" below.

Input columns of `add_slice_columns`:

- `time_series_id`: the series. The sunny-by-unit-shape edge is computed per series.
- `valid_time`: the UTC end of the half-hour, as in NGED's cleaned power.
- `power_fcst_init_time`: the UTC forecast origin. The lead and the fold's month come from it.
- `temperature_c`: **the ERA5 2 m temperature interpolated to the half-hour**
  (`SiteFeatures.temperature_c` in `studies.dp_features`), at the series' own location. It is an
  analysis value at the valid time, never a forecast, so a cold-night row is the same row for every
  forecast under test.
- `elevation_deg`: the sun's elevation at the half-hour's midpoint (`SiteFeatures.elevation_deg`).
- `pv_unit_shape`: the photovoltaic (PV) output per MW of AC capacity from CAMS irradiance
  (`SiteFeatures.pv_unit_shape`, built by `studies.dp_features.unit_pv_shape`).
- `clear_sky_index` (optional): **a realised value, which the caller must pass**: the global
  horizontal irradiance over the Haurwitz clear-sky irradiance (`studies.baselines.haurwitz_w_m2`),
  zero where the clear-sky irradiance is under 20 W/m2 (`clear_sky_index` in `e00323_basis.py`).
  Experiments 178, 235, and 269 take it from the CAMS realised irradiance. A forecast of the index
  (experiments 325 and 329 take it from the ensemble-mean shortwave flux) is not what the protocol
  asks for, because the sunny half-hours must be the same for every forecast under test. The slice
  `sunny_clear_sky_index` exists only when this column is present.

**Which sunny definition feeds the protected-slice tolerance.** The protocol's protected-slice
tolerance reads `sunny_clear_sky_index` (sunny by clear-sky index). `sunny_unit_shape` and
`sunny_april_june` (sunny by unit shape) are card slices, reported beside it and never read by the
tolerance. A caller that does not pass the realised `clear_sky_index` column therefore has no
protected sunny slice.

**One slice table for every arm.** `sunny_unit_shape` takes a per-series quantile of the rows it is
given, so two forecasts scored on different rows would be compared on different half-hours. The
slice columns are therefore computed once, by `build_slice_table`, on the row set that every arm
shares, and `studies.eval_protocol.metrics` joins that table to every arm.

**Slice definitions** (a missing or non-finite input makes a slice's flag false, never null):

- `all`: every row.
- `lead_day_1`, `lead_day_2`, `lead_days_3_4`, `lead_days_5_7`, `lead_days_8_10`,
  `lead_days_11_14`: the lead in hours, `valid_time - power_fcst_init_time`, in `(24 * (first - 1),
  24 * last]`. The interval is open at the bottom and closed at the top so that a day holds 48
  half-hours: the first half-hour of a run ends 0.5 hours after the origin and the last half-hour of
  day 1 ends at 24.0 hours. `lead_days_1_10` is the headline band, `(0, 240]`.
- `time_night`, `time_morning`, `time_midday`, `time_evening`: the local (Europe/London) hour the
  half-hour starts in, from `studies.dp_calendar`: night 00:00 to 05:59, morning 06:00 to 10:59,
  midday 11:00 to 15:59, evening 16:00 to 23:59.
- `sunny_unit_shape`: the unit PV shape is at or above the series' own 2/3 quantile of its daylight
  rows, where daylight is a unit shape above 0.01. The edge is computed over the rows passed to
  `build_slice_table`, which must be the half-hours shared by every arm under comparison.
- `sunny_april_june`: `sunny_unit_shape` in local April, May, or June.
- `sunny_clear_sky_index`: daylight (unit shape above 0.01) with a clear-sky index of 0.9 or more.
- `winter`: local month December, January, or February.
- `winter_nights`: `winter` and the sun more than 5 degrees below the horizon.
- `cold_nights`: the sun more than 5 degrees below the horizon and the ERA5 temperature at 3 degrees
  Celsius or below.
- `economy_7_step`: local half-hours of day 1 to 6 (the half-hour starting 00:30 to 03:00), which
  the card's `e7_step` flag holds.
- `christmas`: local date from 24 December to 31 December, and 1 January.
- `bank_holidays`: the local date falls in any of the 13 named holiday periods of
  `studies.dp_calendar.HOLIDAY_PERIODS`, which are the English bank holidays plus Easter Saturday
  and Sunday, bridge days, and the Christmas run (24 December and 27 to 31 December).
- `cold_start_months`: the fold had under 12 months of power history at its training cut. Present
  only when `first_readings` is passed to `add_slice_columns`.
- `event_rows` and `event_following_7_days`: the half-hour lies inside a hand-labelled event, or in
  the 7 days after one ends. Present only when `events` is passed.

**Deviations from the cards.** The cards are not consistent with the protocol or with each other, so
four thresholds differ from the card that is closest:

- Cold nights: the cards of experiments 250, 252, 354, and 364 use a temperature below 2 degrees
  Celsius at night. The protocol says 3 degrees Celsius or below, as experiment 281's `cold3`
  column (which reads the night as the `night` and `evening` hour bands) does. This module uses the
  protocol's 3 degrees Celsius with the card's night (sun below minus 5 degrees). Experiments 176,
  281, and 311 also have a `cold2` flag that tests a day mean below 2 degrees.
- Winter: the cards and experiment 277 use November to March. The protocol says December to
  February, which this module uses.
- Christmas: the cards use Christmas Day and Boxing Day only. Experiment 281 adds the Christmas run.
  The protocol says 24 December to 1 January, which this module uses.
- Thin: the cards define no threshold, and `e00250_score.py` skips a slice of under 100 rows.
  `eval_protocol.metrics.THIN_HALF_HOURS` takes that figure.

The module imports `studies.dp_calendar` for the local calendar and the holiday periods, because
copying the holiday lookup would let the two drift apart.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Final

import numpy as np
import polars as pl

from studies.dp_calendar import BAND_NAMES, calendar_columns

KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
"""The columns that identify one scored half-hour of one forecast origin."""

HOURS_PER_DAY: Final[int] = 24
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
"""A unit PV shape above this is daylight (`SUN_UP` in `e00250_score.py`)."""

SUNNY_QUANTILE: Final[float] = 2 / 3
"""The share of a series' daylight rows below the sunny edge: the top third is sunny."""

NIGHT_ELEVATION_DEG: Final[float] = -5.0
"""The sun is below this elevation at night, in degrees."""

COLD_NIGHT_MAX_TEMPERATURE_C: Final[float] = 3.0
"""A night at this ERA5 temperature or below is a cold night (the protocol's 3 degrees)."""

CLEAR_SKY_INDEX_SUNNY: Final[float] = 0.9
"""A clear-sky index at or above this is sunny (`CSI_FORECAST_SUNNY` in `e00329_common.py`)."""

WINTER_MONTHS: Final[tuple[int, ...]] = (12, 1, 2)
SUNNY_SEASON_MONTHS: Final[tuple[int, ...]] = (4, 5, 6)
ECONOMY_7_STEP_HALF_HOURS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
"""The local half-hours of day (0 to 47) that the card's Economy 7 step covers."""

CHRISTMAS_LAST_DAY_OF_YEAR: Final[tuple[int, int]] = (12, 31)
CHRISTMAS_FIRST_DAY: Final[int] = 24
EVENT_FOLLOW_DAYS: Final[int] = 7
"""How many days after a hand-labelled event are reported on their own."""

MIN_TRAINING_MONTHS: Final[int] = 12
"""A fold needs this many months of power history at its training cut."""

PUBLICATION_LAG_HOURS: Final[int] = 9
"""The delay between a power reading and its availability in production, which the experiments
call `PUBLICATION_DELAY` (nine hours). The training cut of a fold is this far before the fold's
first forecast origin."""

LEAD_BANDS: Final[Mapping[str, tuple[float, float]]] = {
    "lead_day_1": (0.0, 24.0),
    "lead_day_2": (24.0, 48.0),
    "lead_days_3_4": (48.0, 96.0),
    "lead_days_5_7": (96.0, 168.0),
    "lead_days_8_10": (168.0, 240.0),
    "lead_days_11_14": (240.0, 336.0),
    "lead_days_1_10": (0.0, 240.0),
}
"""Each lead band as `(low, high]` in hours."""

TIME_OF_DAY_SLICES: Final[Mapping[str, str]] = {f"time_{name}": name for name in BAND_NAMES}

SLICE_NAMES: Final[tuple[str, ...]] = (
    "all",
    *LEAD_BANDS,
    *TIME_OF_DAY_SLICES,
    "sunny_clear_sky_index",
    "sunny_unit_shape",
    "sunny_april_june",
    "cold_start_months",
    "winter",
    "winter_nights",
    "cold_nights",
    "economy_7_step",
    "christmas",
    "bank_holidays",
    "event_rows",
    "event_following_7_days",
)
"""Every slice this module can add, in reporting order. Four need an optional input."""

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "valid_time",
    "power_fcst_init_time",
    "temperature_c",
    "elevation_deg",
    "pv_unit_shape",
)

# Series groups of the protocol. The labels are the anonymised Primary names.
SIGNED_MW_LABELS: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY_LABELS: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SHORT_HISTORY_LABELS: Final[tuple[str, ...]] = ("D15", "D18")
SERIES_GROUPS: Final[Mapping[str, str]] = {
    **dict.fromkeys(SIGNED_MW_LABELS, "signed-MW"),
    **dict.fromkeys(MVA_ONLY_LABELS, "MVA-only"),
    **dict.fromkeys(SHORT_HISTORY_LABELS, "short-history"),
}
"""The group of each Primary label."""

GROUP_NOTES: Final[Mapping[str, str]] = {
    "MVA-only": "reactive power assumed zero",
    "short-history": "report only",
}
"""The label the protocol attaches to every number of a group."""


def lead_hours_expression() -> pl.Expr:
    """Return the lead, in hours, of a row: `valid_time - power_fcst_init_time`."""
    return (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0


def _check_columns(*, rows: pl.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        msg = f"the rows lack the columns {missing}"
        raise ValueError(msg)


def _sunny_unit_shape(*, rows: pl.DataFrame) -> np.ndarray:
    """Return the daylight-top-third flag, with each series' edge from its own rows."""
    unit = rows["pv_unit_shape"].to_numpy().astype(np.float64)
    series = rows["time_series_id"].to_numpy()
    daylight = np.nan_to_num(unit, nan=0.0) > SUN_UP_UNIT_SHAPE
    sunny = np.zeros(len(unit), dtype=bool)
    for value in np.unique(series):
        mine = series == value
        lit = mine & daylight
        if lit.any():
            edge = float(np.quantile(unit[lit], SUNNY_QUANTILE))
            sunny |= lit & (unit >= edge)
    return sunny


def fold_plan(*, first_reading: datetime, scored_months: Sequence[datetime]) -> pl.DataFrame:
    """Return the training cut and the history rule of each fold of one series.

    Fold k trains on power up to the end of month M(k) and scores the forecast origins of the next
    month. The training cut is that month's first instant, 00:00 UTC, less the publication lag of
    the power data, so the cut is strictly before the first scored origin. A fold has enough
    history when the series' first reading is at least `MIN_TRAINING_MONTHS` before the cut.

    Args:
        first_reading: The series' first power reading, a UTC datetime.
        scored_months: The first instant of each scored month, 00:00 UTC on day 1, in order.

    Returns:
        One row per scored month: `scored_month`, `training_cut`, `history_needed_from`
        (the latest first reading that still gives enough history), and `enough_history`.

    Raises:
        ValueError: If a scored month does not start on day 1 at 00:00, or the months are not
            consecutive calendar months.
    """
    months = pl.Series("scored_month", list(scored_months), dtype=pl.Datetime("us", "UTC"))
    if (months != months.dt.truncate("1mo")).any():
        msg = "every scored month must start on day 1 at 00:00 UTC"
        raise ValueError(msg)
    if (months.dt.offset_by("1mo").head(-1) != months.tail(-1)).any():
        msg = "the scored months must be consecutive calendar months"
        raise ValueError(msg)
    return (
        pl.DataFrame({"scored_month": months})
        .with_columns(
            training_cut=pl.col("scored_month") - pl.duration(hours=PUBLICATION_LAG_HOURS)
        )
        .with_columns(
            history_needed_from=pl.col("training_cut").dt.offset_by(f"-{MIN_TRAINING_MONTHS}mo")
        )
        .with_columns(enough_history=pl.col("history_needed_from") >= pl.lit(first_reading))
    )


def cold_start_flag(*, rows: pl.DataFrame, first_readings: Mapping[int, datetime]) -> np.ndarray:
    """Return, per row, whether its fold had under 12 months of history at the training cut.

    The fold of a row is the calendar month of its forecast origin, and `fold_plan` states the rule.

    Args:
        rows: Rows with `time_series_id` and `power_fcst_init_time`.
        first_readings: Each series' first power reading, a UTC datetime.

    Returns:
        A boolean array, true where the history is too short. A series with no entry in
        `first_readings` is flagged false.
    """
    first = pl.DataFrame(
        {
            "time_series_id": list(first_readings),
            "first_reading": list(first_readings.values()),
        },
        schema_overrides={
            "time_series_id": rows.schema["time_series_id"],
            "first_reading": pl.Datetime("us", "UTC"),
        },
    )
    joined = rows.select("time_series_id", "power_fcst_init_time").join(
        first, on="time_series_id", how="left", maintain_order="left"
    )
    cut = pl.col("power_fcst_init_time").dt.truncate("1mo") - pl.duration(
        hours=PUBLICATION_LAG_HOURS
    )
    needed_from = cut.dt.offset_by(f"-{MIN_TRAINING_MONTHS}mo")
    return joined.select(
        (needed_from < pl.col("first_reading")).fill_null(value=False).alias("cold")
    )["cold"].to_numpy()


def event_flags(*, rows: pl.DataFrame, events: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the in-event flag and the 7-days-after flag of each row.

    Args:
        rows: Rows with `time_series_id` and `valid_time`.
        events: One row per hand-labelled event with `time_series_id` (null for an event that
            covers every series), and `start` and `end`, UTC datetimes. An event covers
            `[start, end)`, and its following days cover `[end, end + 7 days)`.

    Returns:
        The two boolean arrays, one entry per row.
    """
    inside = np.zeros(rows.height, dtype=bool)
    after = np.zeros(rows.height, dtype=bool)
    valid = rows["valid_time"]
    series = rows["time_series_id"]
    for start, end, series_id in events.select("start", "end", "time_series_id").iter_rows():
        mine = series.is_not_null() if series_id is None else series == series_id
        inside |= (mine & (valid >= start) & (valid < end)).fill_null(value=False).to_numpy()
        follow_end = end + timedelta(days=EVENT_FOLLOW_DAYS)
        after |= (mine & (valid >= end) & (valid < follow_end)).fill_null(value=False).to_numpy()
    return inside, after


def _as_column(*, name: str, value: np.ndarray | pl.Expr) -> pl.Series | pl.Expr:
    """Return a slice's flags as a named column."""
    return value.alias(name) if isinstance(value, pl.Expr) else pl.Series(name, value)


def add_slice_columns(
    *,
    rows: pl.DataFrame,
    events: pl.DataFrame | None = None,
    first_readings: Mapping[int, datetime] | None = None,
) -> pl.DataFrame:
    """Add one boolean column per slice, named as in `SLICE_NAMES`.

    Args:
        rows: Scored half-hours with the columns in `REQUIRED_COLUMNS`, and optionally
            `clear_sky_index`. `valid_time` and `power_fcst_init_time` are UTC-aware datetimes.
        events: Hand-labelled events, as `event_flags` reads them. Without it the two event slices
            are not added.
        first_readings: Each series' first power reading, as `cold_start_flag` reads it. Without it
            the `cold_start_months` slice is not added.

    Returns:
        `rows` with the slice columns appended.

    Raises:
        ValueError: If `rows` lacks a required column.
    """
    _check_columns(rows=rows)
    calendar = calendar_columns(end_times=rows["valid_time"])
    local_date = calendar.local_date
    month = local_date.astype("datetime64[M]").astype(np.int64) % 12 + 1
    day = (local_date - local_date.astype("datetime64[M]")).astype(np.int64) + 1
    lead = lead_hours_expression()
    elevation = np.nan_to_num(rows["elevation_deg"].to_numpy().astype(np.float64), nan=90.0)
    temperature = np.nan_to_num(rows["temperature_c"].to_numpy().astype(np.float64), nan=np.inf)
    unit = np.nan_to_num(rows["pv_unit_shape"].to_numpy().astype(np.float64), nan=0.0)
    night = elevation < NIGHT_ELEVATION_DEG
    winter = np.isin(month, WINTER_MONTHS)
    sunny = _sunny_unit_shape(rows=rows)
    flags: dict[str, np.ndarray | pl.Expr] = {"all": np.ones(rows.height, dtype=bool)}
    flags |= {name: (lead > low) & (lead <= high) for name, (low, high) in LEAD_BANDS.items()}
    flags |= {
        slice_name: calendar.band == BAND_NAMES.index(band)
        for slice_name, band in TIME_OF_DAY_SLICES.items()
    }
    if "clear_sky_index" in rows.columns:
        index = np.nan_to_num(rows["clear_sky_index"].to_numpy().astype(np.float64), nan=0.0)
        flags["sunny_clear_sky_index"] = (unit > SUN_UP_UNIT_SHAPE) & (
            index >= CLEAR_SKY_INDEX_SUNNY
        )
    flags |= {
        "sunny_unit_shape": sunny,
        "sunny_april_june": sunny & np.isin(month, SUNNY_SEASON_MONTHS),
        "winter": winter,
        "winter_nights": winter & night,
        "cold_nights": night & (temperature <= COLD_NIGHT_MAX_TEMPERATURE_C),
        "economy_7_step": np.isin(calendar.half_hour, ECONOMY_7_STEP_HALF_HOURS),
        "christmas": ((month == CHRISTMAS_LAST_DAY_OF_YEAR[0]) & (day >= CHRISTMAS_FIRST_DAY))
        | ((month == 1) & (day == 1)),
        "bank_holidays": calendar.holiday_period >= 0,
    }
    if first_readings is not None:
        flags["cold_start_months"] = cold_start_flag(rows=rows, first_readings=first_readings)
    if events is not None:
        flags["event_rows"], flags["event_following_7_days"] = event_flags(rows=rows, events=events)
    ordered = [name for name in SLICE_NAMES if name in flags]
    return rows.with_columns(*[_as_column(name=name, value=flags[name]) for name in ordered])


def build_slice_table(
    *,
    rows: pl.DataFrame,
    events: pl.DataFrame | None = None,
    first_readings: Mapping[int, datetime] | None = None,
) -> pl.DataFrame:
    """Return the slice columns of the half-hours that every arm shares, keyed by `KEY_COLUMNS`.

    Compute this once, on the row set that all arms under comparison are scored on, and pass the
    result to `studies.eval_protocol.metrics.score_forecast`. Every arm then sees the same
    half-hours in each slice.

    Args:
        rows: The shared half-hours: `KEY_COLUMNS` and the other columns of `REQUIRED_COLUMNS`, and
            optionally `clear_sky_index`. One row per key.
        events: Hand-labelled events, as in `add_slice_columns`.
        first_readings: Each series' first power reading, as in `add_slice_columns`.

    Returns:
        `KEY_COLUMNS` and one boolean column per available slice.

    Raises:
        ValueError: If `rows` lacks a required column or holds a key twice.
    """
    _check_columns(rows=rows)
    if rows.select(*KEY_COLUMNS).is_duplicated().any():
        msg = "the rows of a slice table must hold each key once"
        raise ValueError(msg)
    with_slices = add_slice_columns(rows=rows, events=events, first_readings=first_readings)
    return with_slices.select(
        *KEY_COLUMNS, *[name for name in SLICE_NAMES if name in with_slices.columns]
    )

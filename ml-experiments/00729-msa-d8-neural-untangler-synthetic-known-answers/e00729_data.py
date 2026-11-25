"""Experiment 729: loaders for the real inputs.

The loaders cover D8's reading, the BSP meter, the CAMS and ERA5 weather fields, the metered farms,
the bank holidays, the metered engine record, and the private cut.

Every loader reads one file and returns arrays; none prints a value. D8's reading and the BSP A
meter come from the cleaned power store through `studies.power.scan_power` (half-hours the
`substation_zero` rule flags are already dropped). The CAMS index, clear-sky irradiance, zenith,
and 2 m temperature come from experiment 39's frame; the 10 m wind speed and the precipitation come
from the stored ERA5 features of D8's grid cell, the cell of that temperature. The metered engine
record is experiment 194's window table. The private cut (the engine's connection date) is read
through `engine_constants.engine_datetime` and is never printed or stored in any file of the
repository.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00729_core as core
import e00729_sim as sim
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

FRAME_39: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection/work/frame.parquet"
)
ERA_FRAME: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00697-gas-engine-state-tracker-part-4-running-patterns/frames/frame_d8.parquet"
)
ERA5_FEATURES: Final = study_dir_for(study="stage1_pilot") / "demand/inputs/era5_features.parquet"
"""ERA5 weather at each series' grid cell; its temperature is the one in experiment 39's frame."""
MM_PER_HOUR_PER_KG_M2_S: Final = 3600.0
"""The file gives precipitation in kg per square metre per second; each unit is 3,600 mm an hour."""
RECORD_GLOB: Final = "00194-*/g2_window.parquet"
D8_SITE: Final = "D8"
MIN_CLEAR_W_M2: Final = 5.0
"""Below this clear-sky irradiance the clear-sky index is undefined (NaN)."""
ERA_NET: Final = "net"
ERA_SOLAR: Final = "solar"
ERA_FADE: Final = "fade"
ERA_PRE: Final = "pre"
"""The era labels: experiment 697's three, and "pre" for every earlier day."""
FARM_COLUMNS: Final = ("site", "kind", "time", "power", "ghi_u", "ghi_u_clear", "zenith")
D8_COLUMNS: Final = ("site", "kind", "time", "ghi_u", "ghi_u_clear", "zenith", "temperature_2m")


@dataclass(frozen=True)
class DayTable:
    """D8's days with their real fields, before any split.

    Attributes:
        calendar: The arrays of every D8 day with at least 40 half-hours, up to `REAL_UP_TO`.
        era: The era label of each day: "pre", "net", "solar", or "fade".
        is_base: The day is a base day (before the private cut and with at least 40 half-hours).
        dropped_half_hours: The number of missing half-hours of each day.
    """

    calendar: sim.Calendar
    era: np.ndarray
    is_base: np.ndarray
    dropped_half_hours: np.ndarray


def hourly_to_half_hour(*, values: np.ndarray) -> np.ndarray:
    """Interpolate an hourly index linearly to half-hours.

    The two half-hours of an hour share one value `v`. The first half-hour takes `0.75 v + 0.25
    v_prev` and the second `0.75 v + 0.25 v_next`; where a neighbour hour is missing, the half-hour
    keeps `v`. Positions pair up as `(2k, 2k + 1)` along the flattened array, which must start at a
    whole hour.

    Args:
        values: A 1-D array of half-hours (NaN where missing) with an even length.

    Returns:
        The interpolated array.
    """
    pairs = values.reshape(-1, 2)
    v = pairs[:, 0]
    nxt = np.append(v[1:], np.nan)
    prev = np.insert(v[:-1], 0, np.nan)
    first = np.where(np.isfinite(prev), 0.75 * v + 0.25 * prev, v)
    second = np.where(np.isfinite(nxt), 0.75 * v + 0.25 * nxt, v)
    return np.stack([first, second], axis=1).reshape(-1)


def era5_wind_and_precipitation(*, t0: datetime, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ERA5 10 m wind speed and precipitation at D8's grid cell on the half-hour grid.

    The file holds the hourly ERA5 values at the whole hours and, between them, interpolated values
    that are ignored here. Wind speed is an instantaneous value, so it is treated like the 2 m
    temperature: the two half-hours of an hour share the whole hour's value, and
    `hourly_to_half_hour` interpolates it. Precipitation is an accumulation over the hour that ends
    at its stamp, so each half-hour takes the accumulation of the hour that contains it.

    Args:
        t0: The first instant of the grid (midnight UTC).
        n_steps: The number of half-hours of the grid (even).

    Returns:
        Wind speed in metres per second and precipitation in millimetres per hour, each `(n_steps,)`
        with NaN where ERA5 has no value.
    """
    frame = (
        pl.read_parquet(
            ERA5_FEATURES,
            columns=["time_series_id", "valid_time", "wind_speed_10m", "precipitation"],
        )
        .filter(
            (pl.col("time_series_id") == core.D8_SERIES_ID)
            & (pl.col("valid_time").dt.minute() == 0)
        )
        .select("valid_time", "wind_speed_10m", "precipitation")
    )
    n_hours = n_steps // 2
    hour = ((frame["valid_time"] - t0).dt.total_seconds() // 3600).to_numpy().astype(np.int64)
    ok = (hour >= 0) & (hour <= n_hours)
    wind_h = np.full(n_hours + 1, np.nan)
    rain_h = np.full(n_hours + 1, np.nan)
    wind_h[hour[ok]] = frame["wind_speed_10m"].cast(pl.Float64).to_numpy()[ok]
    rain_h[hour[ok]] = (
        frame["precipitation"].cast(pl.Float64).to_numpy()[ok] * MM_PER_HOUR_PER_KG_M2_S
    )
    wind = hourly_to_half_hour(values=np.repeat(wind_h[:n_hours], 2))
    rain = np.repeat(rain_h[1 : n_hours + 1], 2)
    return wind, rain


def _grid(*, frame: pl.DataFrame, t0: datetime, n_steps: int, column: str) -> np.ndarray:
    """Place one column of a frame (with a `start` column) on a dense half-hour grid."""
    out = np.full(n_steps, np.nan)
    idx = ((frame["start"] - t0).dt.total_seconds() // 1800).to_numpy().astype(np.int64)
    ok = (idx >= 0) & (idx < n_steps)
    out[idx[ok]] = frame[column].cast(pl.Float64).fill_null(float("nan")).to_numpy()[ok]
    return out


def private_cut() -> datetime:
    """Return the first instant that is not a pre-engine base instant (never printed)."""
    import engine_constants as ec  # ty: ignore[unresolved-import]

    eras = pl.read_parquet(ERA_FRAME, columns=["start", "era"])
    net_start = eras.filter(pl.col("era") == ERA_NET)["start"].min()
    return min(net_start, ec.engine_datetime(key="d8_engine_connected_date")).replace(tzinfo=UTC)  # ty: ignore[unresolved-attribute, invalid-argument-type]


def era_of_days(*, days: np.ndarray) -> np.ndarray:
    """Return the era label of each `datetime64[D]` day, from experiment 697's era frame."""
    eras = (
        pl.read_parquet(ERA_FRAME, columns=["start", "era"])
        .with_columns(day=pl.col("start").dt.date())
        .group_by("day")
        .agg(era=pl.col("era").first())
    )
    mapping = dict(zip(eras["day"].to_list(), eras["era"].to_list(), strict=True))
    out = [mapping.get(d.astype("datetime64[D]").astype(date), ERA_PRE) for d in days]
    return np.array(out)


def _power_days(*, ids: tuple[int, ...], up_to: date) -> pl.DataFrame:
    """Return half-hourly power of the given series with the period-start time."""
    from studies.power import scan_power

    raw = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(list(ids)))
        .select("time_series_id", "time", "power")
        .collect()
    )
    return raw.with_columns(start=pl.col("time") - pl.duration(minutes=30)).filter(
        pl.col("start")
        < datetime(up_to.year, up_to.month, up_to.day, tzinfo=UTC) + timedelta(days=1)
    )


def _day_arrays(*, frame: pl.DataFrame, dates: np.ndarray, scale: float) -> np.ndarray:
    """Return `(n_days, 48)` points for the sorted `dates` from a frame of `start` and `power`."""
    out = np.full((len(dates), 48), np.nan)
    day = frame["start"].dt.date().to_numpy().astype("datetime64[D]")
    pos = np.clip(np.searchsorted(dates, day), 0, len(dates) - 1)
    hh = (frame["start"].dt.hour() * 2 + frame["start"].dt.minute() // 30).to_numpy()
    ok = dates[pos] == day
    out[pos[ok], hh[ok]] = frame["power"].to_numpy()[ok] * scale
    return out


def farm_set(*, dates: np.ndarray, frame: pl.DataFrame, t0: datetime, n_steps: int) -> sim.FarmSet:
    """Build the farms' normalised output and skies on `dates` from experiment 39's farm rows."""
    import pvlib
    from studies.pv_dataset import pv_sites
    from studies.pv_physics import solar_position

    sites = pv_sites().sort("site")
    names = sites["site"].to_list()
    fields: dict[str, list[np.ndarray]] = {
        k: [] for k in ("power", "ghi", "dni", "dhi", "zenith", "azimuth")
    }
    day_idx = (dates - np.datetime64(t0.date(), "D")).astype(int)
    for row in sites.iter_rows(named=True):
        f = frame.filter(pl.col("site") == row["site"]).with_columns(
            start=pl.col("time") - pl.duration(minutes=30)
        )
        power = (
            _grid(frame=f, t0=t0, n_steps=n_steps, column="power") / row["effective_capacity_mw"]
        )
        clear = _grid(frame=f, t0=t0, n_steps=n_steps, column="ghi_u_clear")
        ghi = _grid(frame=f, t0=t0, n_steps=n_steps, column="ghi_u")
        with np.errstate(invalid="ignore", divide="ignore"):
            csi = np.where(clear > MIN_CLEAR_W_M2, ghi / clear, np.nan)
        ghi_i = hourly_to_half_hour(values=csi) * clear
        mid = pl.Series(
            [t0 + timedelta(minutes=30 * i + 15) for i in range(n_steps)],
            dtype=pl.Datetime("us", "UTC"),
        )
        zen, azi = solar_position(stamps=mid, latitude=row["latitude"], longitude=row["longitude"])
        doy = mid.dt.ordinal_day().to_numpy()
        split = pvlib.irradiance.erbs(ghi=np.nan_to_num(ghi_i), zenith=zen, datetime_or_doy=doy)
        arrays = {
            "power": power, "ghi": ghi_i, "dni": np.asarray(split["dni"]),
            "dhi": np.asarray(split["dhi"]), "zenith": zen, "azimuth": azi,
        }  # fmt: skip
        for k, v in arrays.items():
            fields[k].append(v.reshape(-1, 48)[day_idx])
    stacked = {k: np.stack(v) for k, v in fields.items()}
    power_ok = np.isfinite(stacked["power"]).sum(axis=2) >= core.MIN_HH_FARM_DAY
    sky_ok = np.isfinite(stacked["ghi"]).sum(axis=2) >= core.MIN_HH_FARM_DAY
    stacked["ghi"] = np.nan_to_num(stacked["ghi"])
    stacked["dni"] = np.nan_to_num(stacked["dni"])
    stacked["dhi"] = np.nan_to_num(stacked["dhi"])
    assert len(names) == stacked["power"].shape[0]
    return sim.FarmSet(
        power=stacked["power"], power_ok=power_ok, sky_ok=sky_ok, ghi=stacked["ghi"],
        dni=stacked["dni"], dhi=stacked["dhi"], zenith=stacked["zenith"],
        azimuth=stacked["azimuth"],
    )  # fmt: skip


def bank_holiday_dates() -> np.ndarray:
    """Return the English bank holidays as `datetime64[D]`."""
    from studies.dp_calendar import bank_holiday_names

    return np.array(sorted(bank_holiday_names()), dtype="datetime64[D]")


def load_day_table(*, closure_days: frozenset[date] = frozenset()) -> DayTable:
    """Load every D8 day up to `REAL_UP_TO` with its real fields and its base-day flag.

    Args:
        closure_days: Days within 24 hours of a logged closure at D8 (dropped from the base).

    Returns:
        The table. Days with fewer than 40 half-hours are left out.
    """
    up_to = date.fromisoformat(core.REAL_UP_TO)
    power = _power_days(ids=(core.D8_SERIES_ID, core.BSP_SERIES_ID), up_to=up_to)
    d8 = power.filter(pl.col("time_series_id") == core.D8_SERIES_ID)
    counts = (
        d8.group_by(pl.col("start").dt.date().alias("day"))
        .agg(n=pl.len())
        .filter(pl.col("n") >= core.MIN_HALF_HOURS_IN_DAY)
    )
    days = np.array(sorted(counts["day"].to_list()), dtype="datetime64[D]")
    scale = core.POINTS / core.CAPACITY_MVA
    reading = _day_arrays(frame=d8, dates=days, scale=scale)
    bsp = _day_arrays(
        frame=power.filter(pl.col("time_series_id") == core.BSP_SERIES_ID), dates=days, scale=scale
    )
    frame = pl.read_parquet(FRAME_39, columns=list(set(FARM_COLUMNS) | set(D8_COLUMNS)))
    t0 = datetime(*days[0].astype(date).timetuple()[:3], tzinfo=UTC) - timedelta(days=2)
    t0 = t0.replace(hour=0, minute=0)
    n_steps = (int((days[-1] - np.datetime64(t0.date(), "D")).astype(int)) + 3) * 48
    d8f = frame.filter(pl.col("site") == D8_SITE).with_columns(
        start=pl.col("time") - pl.duration(minutes=30)
    )
    day_idx = (days - np.datetime64(t0.date(), "D")).astype(int)
    clear = _grid(frame=d8f, t0=t0, n_steps=n_steps, column="ghi_u_clear")
    ghi = _grid(frame=d8f, t0=t0, n_steps=n_steps, column="ghi_u")
    with np.errstate(invalid="ignore", divide="ignore"):
        csi = np.where(clear > MIN_CLEAR_W_M2, ghi / clear, np.nan)
    zenith = _grid(frame=d8f, t0=t0, n_steps=n_steps, column="zenith")
    t2m = hourly_to_half_hour(
        values=_grid(frame=d8f, t0=t0, n_steps=n_steps, column="temperature_2m")
    )
    wind, rain = era5_wind_and_precipitation(t0=t0, n_steps=n_steps)
    csi_i = hourly_to_half_hour(values=csi)
    cut = private_cut()
    cal = sim.Calendar(
        dates=days, reading=reading, bsp=bsp,
        csi=csi_i.reshape(-1, 48)[day_idx], clear=clear.reshape(-1, 48)[day_idx],
        cosz=np.clip(np.cos(np.radians(zenith)), 0.0, None).reshape(-1, 48)[day_idx],
        t2m=t2m.reshape(-1, 48)[day_idx],
        wind10=wind.reshape(-1, 48)[day_idx], precip=rain.reshape(-1, 48)[day_idx],
        d8_ghi=np.nan_to_num(csi_i * clear).reshape(-1, 48)[day_idx],
        holidays=bank_holiday_dates(),
        farms=farm_set(
            dates=days, frame=frame.filter(pl.col("kind") == "farm"), t0=t0, n_steps=n_steps
        ),
    )  # fmt: skip
    cut_day = np.datetime64(cut.date(), "D")
    near_closure = np.isin(days, np.array(sorted(closure_days), dtype="datetime64[D]"))
    return DayTable(
        calendar=cal, era=era_of_days(days=days),
        is_base=(days < cut_day) & ~near_closure,
        dropped_half_hours=np.isnan(reading).sum(axis=1),
    )  # fmt: skip


def select_days(*, cal: sim.Calendar, mask: np.ndarray) -> sim.Calendar:
    """Return the calendar restricted to the days where `mask` is true."""
    f = cal.farms
    farms = sim.FarmSet(
        power=f.power[:, mask], power_ok=f.power_ok[:, mask], sky_ok=f.sky_ok[:, mask],
        ghi=f.ghi[:, mask], dni=f.dni[:, mask], dhi=f.dhi[:, mask], zenith=f.zenith[:, mask],
        azimuth=f.azimuth[:, mask],
    )  # fmt: skip
    return sim.Calendar(
        dates=cal.dates[mask], reading=cal.reading[mask], bsp=cal.bsp[mask], csi=cal.csi[mask],
        clear=cal.clear[mask], cosz=cal.cosz[mask], t2m=cal.t2m[mask],
        wind10=cal.wind10[mask], precip=cal.precip[mask], d8_ghi=cal.d8_ghi[mask],
        holidays=cal.holidays, farms=farms,
    )  # fmt: skip


def month_table(*, cal: sim.Calendar) -> tuple[list[str], list[int], list[int]]:
    """Return the base months, their base-day counts, and their counts of days with BSP data."""
    keys = np.array([str(d)[:7] for d in cal.dates])
    has_bsp = np.isfinite(cal.bsp).any(axis=1)
    months = sorted(set(keys.tolist()))
    return (
        months,
        [int((keys == m).sum()) for m in months],
        [int(((keys == m) & has_bsp).sum()) for m in months],
    )


def metered_record() -> pl.DataFrame:
    """Return the metered engine record (experiment 194's window table).

    Returns:
        The table sorted by `time`, with `power` (MW) and `units` (the number of units running).
    """
    path = next(ML_EXPERIMENTS_DATA_DIR.glob(RECORD_GLOB))
    return pl.read_parquet(path).sort("time")


def record_by_day(*, record: pl.DataFrame, days: np.ndarray, column: str = "power") -> np.ndarray:
    """Return a column of the record on the given days, `(n_days, 48)`, NaN where it has no row.

    Args:
        record: The metered engine record.
        days: The days, as `datetime64[D]`.
        column: `power` (returned in points of D8's capacity) or `units` (returned as it is).

    Returns:
        The array.
    """
    frame = record.with_columns(start=pl.col("time") - pl.duration(minutes=30)).select(
        "start", column
    )
    scale = core.POINTS / core.CAPACITY_MVA if column == "power" else 1.0
    return _day_arrays(frame=frame.rename({column: "power"}), dates=days, scale=scale)


def closure_days_from_sheets(
    *, sheets: dict[str, pl.DataFrame], substation_number: int
) -> set[date]:
    """Return the days within 24 hours of a logged closure of one substation.

    A sheet is used when it has a column whose name holds "substation" (but not "name"), a column
    whose name holds "start", "from", or "begin", and optionally one whose name holds "end", "to",
    or "until". A closure runs from its start to its end (or the start alone when there is no end
    column), and one day is added on each side.

    Args:
        sheets: The workbook's sheets by name.
        substation_number: The substation's number.

    Returns:
        The days, as dates (empty when no sheet has the columns).
    """
    days: set[date] = set()
    for frame in sheets.values():
        names = {c.lower(): c for c in frame.columns}
        sub = next((v for k, v in names.items() if "substation" in k and "name" not in k), None)
        beg = next(
            (v for k, v in names.items() if any(p in k for p in ("start", "from", "begin"))), None
        )
        end = next((v for k, v in names.items() if any(p in k for p in ("end", "until"))), None)
        if sub is None or beg is None:
            continue
        mine = frame.filter(pl.col(sub).cast(pl.String, strict=False) == str(substation_number))
        for row in mine.iter_rows(named=True):
            first, last = row[beg], row[end] if end is not None else row[beg]
            if not isinstance(first, datetime | date):
                continue
            last = last if isinstance(last, datetime | date) else first
            a = first.date() if isinstance(first, datetime) else first
            b = last.date() if isinstance(last, datetime) else last
            days.update(a + timedelta(days=k) for k in range(-1, (b - a).days + 2))
    return days


def load_closure_days() -> tuple[frozenset[date], str]:
    """Load the switching logs and return D8's closure days and a status word.

    Only the expected failures are caught: a missing workbook or module (`missing`), and a workbook
    or metadata file that cannot be read as the code expects (`unreadable`). Any other error
    propagates.

    Returns:
        The days (empty unless the status is `loaded`) and the status: `loaded`, `missing`, or
        `unreadable`.
    """
    try:
        import e00722_data as d722  # ty: ignore[unresolved-import]

        sheets = pl.read_excel(d722.SWITCHING_LOG_PATH, sheet_id=0)
        number = (
            pl.read_parquet(d722.METADATA_PATH)
            .filter(pl.col("time_series_id") == core.D8_SERIES_ID)["substation_number"]
            .item()
        )
        days = closure_days_from_sheets(sheets=sheets, substation_number=int(number))
    except OSError, ImportError:
        return frozenset(), "missing"
    except KeyError, ValueError, pl.exceptions.PolarsError:
        return frozenset(), "unreadable"
    return frozenset(days), "loaded"


def atomic_write_text(*, path: Path, text: str) -> None:
    """Write a text file atomically (temporary file in the same folder, then a rename)."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)

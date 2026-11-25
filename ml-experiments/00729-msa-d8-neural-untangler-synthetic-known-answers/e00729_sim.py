# ruff: noqa: E501
"""Experiment 729: the engine and solar simulators, the fold, and the synthetic BSP meter.

A segment is one calendar month of base days that carries one continuous injection draw. Every
segment array is dense in calendar days: shape `(D + 2, 48)`, with one padding day on each side
that holds NaN, so a 96-step window is a plain slice and a lag that falls outside the month reads
NaN (masked). Values are in points of D8's capacity. The simulators read ranges from
`e00729_core.RANGES` (drawn from published descriptions, never fitted) and read no metered engine
record and no private constants file.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import e00729_core as core
import numpy as np
from studies.pv_physics import SunAndSky, ac_power_mw, plane_of_array_w_m2

UK: ZoneInfo = ZoneInfo("Europe/London")


def utc_offset_hours(*, day: date) -> float:
    """Return local time minus UTC, in hours, at noon of a calendar day (0 or 1 in the UK)."""
    from datetime import datetime

    offset = datetime(day.year, day.month, day.day, 12, tzinfo=UK).utcoffset()
    assert offset is not None
    return offset.total_seconds() / 3600.0


@dataclass(frozen=True)
class FarmSet:
    """The metered farms' normalised output and the satellite sky at each farm, on the base days.

    Attributes:
        power: Output over effective capacity, shape `(F, n, 48)`, NaN where missing.
        power_ok: At least `MIN_HH_FARM_DAY` half-hours of data on the base day, `(F, n)`.
        sky_ok: The satellite sky at the farm is present on the base day, `(F, n)`.
        ghi: CAMS global horizontal irradiance at the farm in W/m2, `(F, n, 48)`.
        dni: Direct normal irradiance, `(F, n, 48)`.
        dhi: Diffuse horizontal irradiance, `(F, n, 48)`.
        zenith: Solar zenith in degrees, `(F, n, 48)`.
        azimuth: Solar azimuth in degrees clockwise from north, `(F, n, 48)`.
    """

    power: np.ndarray
    power_ok: np.ndarray
    sky_ok: np.ndarray
    ghi: np.ndarray
    dni: np.ndarray
    dhi: np.ndarray
    zenith: np.ndarray
    azimuth: np.ndarray


@dataclass(frozen=True)
class Calendar:
    """D8's base days with their real fields, aligned on one axis of `n` base days.

    Attributes:
        dates: The base days, sorted, `datetime64[D]`, `(n,)`.
        reading: D8's cleaned reading in points, NaN at dropped half-hours, `(n, 48)`.
        bsp: The BSP A meter's signed power in points, NaN where the meter has no data, `(n, 48)`.
        csi: The CAMS clear-sky index at D8, `(n, 48)`.
        clear: The clear-sky irradiance at D8 over 1000 W/m2, `(n, 48)`.
        cosz: The cosine of the solar zenith angle at D8 (never negative), `(n, 48)`.
        t2m: ERA5 2 m temperature at D8's cell in degrees Celsius, `(n, 48)`.
        d8_ghi: CAMS global horizontal irradiance at D8 in W/m2, `(n, 48)`, for the farm check.
        holidays: Bank-holiday dates, `datetime64[D]`.
        farms: The metered farms.
    """

    dates: np.ndarray
    reading: np.ndarray
    bsp: np.ndarray
    csi: np.ndarray
    clear: np.ndarray
    cosz: np.ndarray
    t2m: np.ndarray
    d8_ghi: np.ndarray
    holidays: np.ndarray
    farms: FarmSet


@dataclass
class Segment:
    """One simulated month: base days plus an injection, dense in calendar days.

    Attributes:
        month: `YYYY-MM`. Held in memory and in private files only; never printed.
        world: One of `WORLD_TYPES`.
        source: "S1", "S2", or "none".
        dates: The calendar days with one padding day each side, `datetime64[D]`, `(D + 2,)`.
        cal_index: Index into `Calendar` of each dense day, -1 for padding and non-base days.
        base: Base demand in points, interpolated over dropped half-hours, `(D + 2, 48)`.
        valid: True where the real reading exists (scored and trained on), `(D + 2, 48)`.
        engine: Engine output in points, `(D + 2, 48)`.
        solar: Solar output in points, `(D + 2, 48)`.
        x: Signed truth in points, `(D + 2, 48)`.
        r: The folded reading in points, NaN off base days, `(D + 2, 48)`.
        bsp: The synthetic BSP meter in points, NaN where the real meter is missing.
        csi: As in `Calendar`, copied onto the dense axis.
        clear: As in `Calendar`.
        cosz: As in `Calendar`.
        t2m: As in `Calendar`.
        s1_fallback: The day took source S2 because no farm had data, `(D + 2,)`.
        has_bsp: At least `MIN_DAYS_IN_MONTH` base days in the segment have BSP data.
        params: The drawn parameters, for the range assertion (never printed).
    """

    month: str
    world: str
    source: str
    dates: np.ndarray
    cal_index: np.ndarray
    base: np.ndarray
    valid: np.ndarray
    engine: np.ndarray
    solar: np.ndarray
    x: np.ndarray
    r: np.ndarray
    bsp: np.ndarray
    csi: np.ndarray
    clear: np.ndarray
    cosz: np.ndarray
    t2m: np.ndarray
    s1_fallback: np.ndarray
    has_bsp: bool
    params: dict[str, float] = field(default_factory=dict)

    @property
    def n_dense(self) -> int:
        """The number of calendar days in the month (without padding)."""
        return int(self.dates.shape[0] - 2)

    @property
    def base_days(self) -> np.ndarray:
        """Dense indices (including the padding offset) of the base days."""
        return np.flatnonzero(self.cal_index >= 0)


# --------------------------------------------------------------------------- uniform helpers


def uniform(*, rng: np.random.Generator, name: str, record: dict[str, float]) -> float:
    """Draw one parameter uniformly from its range in `RANGES`, and record it."""
    lo, hi = core.RANGES[name]
    value = float(rng.uniform(lo, hi))
    record[name] = value
    return value


def integer(*, rng: np.random.Generator, name: str, record: dict[str, float]) -> int:
    """Draw one integer parameter uniformly from its inclusive range, and record it."""
    lo, hi = core.RANGES[name]
    value = int(rng.integers(int(lo), int(hi) + 1))
    record[name] = float(value)
    return value


def assert_in_ranges(*, params: dict[str, float]) -> None:
    """Raise `AssertionError` when a recorded parameter lies outside its range in `RANGES`."""
    for name, value in params.items():
        lo, hi = core.RANGES[name]
        assert lo - 1e-9 <= value <= hi + 1e-9, f"parameter {name} lies outside its range"


# --------------------------------------------------------------------------- the engine


def draw_engine(  # noqa: PLR0915
    *, rng: np.random.Generator, utc_offset: np.ndarray, quiet_day: np.ndarray,
    record: dict[str, float],
) -> np.ndarray:  # fmt: skip
    """Return engine output in points, shape `(D, 48)`, from the plan's parameter table.

    The simulation runs on a 5-minute grid and averages to half-hours, so a half-hour that holds a
    start or a stop carries a partial mean.

    Args:
        rng: The generator of the segment.
        utc_offset: Local time minus UTC in hours, for each calendar day, `(D,)`.
        quiet_day: True on a weekend or bank holiday, `(D,)`.
        record: Receives every drawn parameter.

    Returns:
        The engine output, `(D, 48)`, in points, never negative.
    """
    n_days = len(utc_offset)
    sub = core.SUBSTEPS_PER_HALF_HOUR
    per_day = 48 * sub
    n_units = integer(rng=rng, name="n_units", record=record)
    sizes = np.array([uniform(rng=rng, name="unit_points", record={}) for _ in range(n_units)])
    cap = core.RANGES["total_points_cap"][1]
    if sizes.sum() > cap:
        sizes = sizes * cap / sizes.sum()
    record["unit_points"] = float(sizes.mean())
    record["min_unit_points"] = float(sizes.min())
    record["total_points_cap"] = float(sizes.sum())
    no_run = uniform(rng=rng, name="no_run_share", record=record)
    p_eve = uniform(rng=rng, name="evening_prob", record=record)
    p_mor = uniform(rng=rng, name="morning_prob", record=record)
    p_mid = uniform(rng=rng, name="midday_prob", record=record)
    quiet = uniform(rng=rng, name="weekend_factor", record=record)
    spells = uniform(rng=rng, name="spells_per_30_days", record=record)
    trip_prob = uniform(rng=rng, name="trip_prob", record=record)
    trip_lo, trip_hi = (int(v) for v in core.RANGES["trip_half_hours"])
    unit_out = np.zeros((n_units, n_days * per_day))
    kinds = (
        (p_eve, "evening_start_h", "evening_hours"),
        (p_mor, "morning_start_h", "morning_hours"),
        (p_mid, "midday_start_h", "midday_hours"),
    )
    for day in range(n_days):
        if rng.uniform() < no_run:
            continue
        factor = quiet if quiet_day[day] else 1.0
        for prob, start_name, length_name in kinds:
            if rng.uniform() >= prob * factor:
                continue
            start_local = uniform(rng=rng, name=start_name, record={})
            length_h = uniform(rng=rng, name=length_name, record={})
            k = int(rng.integers(1, n_units + 1))
            share = rng.uniform(*core.RANGES["unit_output_share"], size=k)
            ramp_hh = float(rng.uniform(*core.RANGES["ramp_half_hours"]))
            start_utc = start_local - utc_offset[day]
            a = round((day * 24 + start_utc) * 2 * sub)
            b = a + round(length_h * 2 * sub)
            lo, hi = max(a, 0), min(b, n_days * per_day)
            if hi <= lo:
                continue
            ramp = round(ramp_hh * sub)
            profile = np.ones(hi - lo)
            if ramp > 0:
                steps = min(ramp, hi - lo)
                profile[:steps] = (np.arange(steps) + 1.0) / (steps + 1.0)
            for u in range(k):
                contribution = sizes[u] * share[u] * profile
                for _ in range(int(rng.poisson(trip_prob * length_h * 2))):
                    t0 = int(rng.integers(0, hi - lo))
                    hh = int(rng.integers(trip_lo, trip_hi + 1))
                    contribution[t0 : t0 + hh * sub] = 0.0
                unit_out[u, lo:hi] = np.maximum(unit_out[u, lo:hi], contribution)
    for _ in range(int(rng.poisson(spells * n_days / 30.0))):
        length = integer(rng=rng, name="spell_days", record={})
        start = int(rng.integers(0, n_days))
        level = rng.uniform(*core.SPELL_LEVEL_RANGE)
        lo, hi = start * per_day, min(n_days, start + length) * per_day
        unit_out[:, lo:hi] = np.maximum(unit_out[:, lo:hi], (sizes * level)[:, None])
    total = unit_out.sum(axis=0).reshape(n_days, 48, sub).mean(axis=2)
    return np.clip(total, 0.0, None)


# --------------------------------------------------------------------------- solar


def sky_of(*, farms: FarmSet, farm: int, days: np.ndarray) -> SunAndSky:
    """Return the sun and sky of one farm over the given calendar-axis day indices."""
    take = (farm, days)
    return SunAndSky(
        half_hour_end_time=np.zeros(farms.ghi[take].size),
        ghi_w_m2=farms.ghi[take].ravel(),
        dni_w_m2=farms.dni[take].ravel(),
        dhi_w_m2=farms.dhi[take].ravel(),
        zenith_deg=farms.zenith[take].ravel(),
        sun_azimuth_deg=farms.azimuth[take].ravel(),
    )


def ar1_noise(*, rng: np.random.Generator, size: int) -> np.ndarray:
    """Return stationary AR(1) noise with autocorrelation 0.9 and standard deviation 0.05."""
    rho = core.NOISE_AUTOCORRELATION
    sd = core.RANGES["noise_sd"][0]
    eps = rng.normal(0.0, sd * np.sqrt(1.0 - rho**2), size)
    out = np.empty(size)
    out[0] = rng.normal(0.0, sd)
    for t in range(1, size):
        out[t] = rho * out[t - 1] + eps[t]
    return out


def draw_solar(
    *, rng: np.random.Generator, farms: FarmSet, cal_days: np.ndarray, record: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Return solar output in points for the base days of a segment, with the source it used.

    Args:
        rng: The generator of the segment.
        farms: The metered farms and their skies.
        cal_days: Calendar-axis indices of the segment's base days, `(m,)`.
        record: Receives every drawn parameter.

    Returns:
        The output `(m, 48)`, the fallback flags `(m,)` (the day took source S2 because no farm had
        data), the source name ("S1" or "S2"), and the farm index drawn for the segment.
    """
    capacity = float(np.exp(rng.uniform(*np.log(core.RANGES["solar_capacity_points"]))))
    record["solar_capacity_points"] = capacity
    n_farms = farms.power.shape[0]
    primary = int(rng.integers(0, n_farms))
    use_s1 = bool(rng.uniform() < core.S1_SHARE)
    tilt = uniform(rng=rng, name="tilt_deg", record=record)
    azimuth = uniform(rng=rng, name="azimuth_deg", record=record)
    ratio = uniform(rng=rng, name="dc_ac_ratio", record=record)
    out = np.zeros((len(cal_days), 48))
    fallback = np.zeros(len(cal_days), dtype=bool)
    noise = ar1_noise(rng=rng, size=len(cal_days) * 48).reshape(-1, 48)
    for i, day in enumerate(cal_days):
        if use_s1:
            candidates = [primary, *rng.permutation(n_farms).tolist()]
            chosen = next((f for f in candidates if farms.power_ok[f, day]), None)
            if chosen is not None:
                shape = np.nan_to_num(farms.power[chosen, day], nan=0.0)
                night = farms.zenith[chosen, day] >= core.SUN_UP_ZENITH_DEG
                out[i] = capacity * np.clip(np.where(night, 0.0, shape), 0.0, 1.2)
                continue
            fallback[i] = True
        sky_farm = primary if farms.sky_ok[primary, day] else None
        if sky_farm is None:
            sky_farm = next((f for f in range(n_farms) if farms.sky_ok[f, day]), None)
        if sky_farm is None:
            fallback[i] = True
            continue
        sky = sky_of(farms=farms, farm=sky_farm, days=np.array([day]))
        poa = plane_of_array_w_m2(sky=sky, tilt_deg=tilt, azimuth_deg=azimuth, tracker=False)
        unit = ac_power_mw(poa_w_m2=poa, dc_capacity_mw=ratio, ac_capacity_mw=1.0)
        unit = np.where(farms.zenith[sky_farm, day] >= core.SUN_UP_ZENITH_DEG, 0.0, unit)
        out[i] = capacity * unit * (1.0 + noise[i])
    return np.clip(np.nan_to_num(out), 0.0, None), fallback, "S1" if use_s1 else "S2", primary


# --------------------------------------------------------------------------- fold and BSP


def fold_reading(
    *, x: np.ndarray, rng: np.random.Generator, record: dict[str, float]
) -> np.ndarray:
    """Return the meter's reading `sqrt(X^2 + Q^2)` with the segment's reactive floor Q (points).

    Args:
        x: Signed power in points.
        rng: The generator of the segment.
        record: Receives the drawn floor and the lowest half-hour factor.

    Returns:
        The reading in points, same shape as `x`.
    """
    q_mva = uniform(rng=rng, name="q_mva", record=record)
    lo, hi = core.RANGES["q_factor"]
    factor = rng.uniform(lo, hi, size=x.shape)
    record["q_factor"] = float(factor.min())
    q = q_mva / core.CAPACITY_MVA * core.POINTS * factor
    return np.sqrt(x**2 + q**2)


def synthetic_bsp(
    *, bsp_real: np.ndarray, engine: np.ndarray, solar: np.ndarray, beta: float
) -> np.ndarray:
    """Return the real BSP A meter minus beta times the injection (NaN where the meter is missing)."""
    return bsp_real - beta * (engine + solar)


# --------------------------------------------------------------------------- segments


def dense_days(*, first: date, last: date) -> np.ndarray:
    """Return the calendar days from `first - 1` to `last + 1` as `datetime64[D]`."""
    n = (last - first).days + 3
    return np.datetime64(first - timedelta(days=1), "D") + np.arange(n)


def interpolate_rows(*, values: np.ndarray) -> np.ndarray:
    """Fill NaN half-hours of each row by linear interpolation along the day (edges held)."""
    out = values.copy()
    for i in range(out.shape[0]):
        ok = np.isfinite(out[i])
        if ok.all() or not ok.any():
            continue
        out[i] = np.interp(np.arange(out.shape[1]), np.flatnonzero(ok), out[i, ok])
    return out


def to_dense(*, calendar_values: np.ndarray, cal_index: np.ndarray) -> np.ndarray:
    """Place base-day rows on the dense axis, NaN elsewhere (float32)."""
    out = np.full((len(cal_index), calendar_values.shape[1]), np.nan, dtype=np.float32)
    on = cal_index >= 0
    out[on] = calendar_values[cal_index[on]]
    return out


def simulate_segment(
    *, cal: Calendar, month: str, use: np.ndarray, world: str, rng: np.random.Generator
) -> Segment:
    """Simulate one segment: one calendar month of base days with one injection draw.

    Args:
        cal: The calendar of base days.
        month: `YYYY-MM` of the segment.
        use: Which base days of `cal` the segment may use (False for guard days that a split drops).
        world: One of `WORLD_TYPES`.
        rng: The generator of this segment alone.

    Returns:
        The segment.
    """
    first_of = np.datetime64(f"{month}-01", "D")
    nxt = (first_of.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
    n_cal = int((nxt - first_of).astype(int))
    dates = first_of - np.timedelta64(1, "D") + np.arange(n_cal + 2)
    pos = np.searchsorted(cal.dates, dates)
    pos_c = np.clip(pos, 0, len(cal.dates) - 1)
    present = (cal.dates[pos_c] == dates) & use[pos_c]
    present[0] = present[-1] = False
    cal_index = np.where(present, pos_c, -1)
    params: dict[str, float] = {}
    offsets = np.array(
        [utc_offset_hours(day=d.astype("datetime64[D]").astype(date)) for d in dates[1:-1]]
    )
    weekday = (dates[1:-1].astype("datetime64[D]").astype(int) + 3) % 7  # 1970-01-01 is a Thursday
    quiet = (weekday >= 5) | np.isin(dates[1:-1], cal.holidays)
    shape = (n_cal + 2, 48)
    base = to_dense(calendar_values=interpolate_rows(values=cal.reading), cal_index=cal_index)
    valid = to_dense(calendar_values=np.isfinite(cal.reading).astype(float), cal_index=cal_index)
    valid = np.nan_to_num(valid, nan=0.0) > 0.5
    engine = np.zeros(shape, dtype=np.float32)
    solar = np.zeros(shape, dtype=np.float32)
    s1_fallback = np.zeros(n_cal + 2, dtype=bool)
    source = "none"
    beta = uniform(rng=rng, name="beta", record=params)
    if world in ("engine", "both"):
        engine[1:-1] = draw_engine(rng=rng, utc_offset=offsets, quiet_day=quiet, record=params)
    if world in ("solar", "both"):
        base_days = np.flatnonzero(cal_index >= 0)
        out, fallback, source, _ = draw_solar(
            rng=rng, farms=cal.farms, cal_days=cal_index[base_days], record=params
        )
        solar[base_days] = out
        s1_fallback[base_days] = fallback
    x = base - engine - solar
    r = fold_reading(x=np.nan_to_num(x), rng=rng, record=params).astype(np.float32)
    r[cal_index < 0] = np.nan
    bsp_real = to_dense(calendar_values=cal.bsp, cal_index=cal_index)
    bsp = synthetic_bsp(bsp_real=bsp_real, engine=engine, solar=solar, beta=beta).astype(np.float32)
    has_bsp = int(np.isfinite(bsp_real).any(axis=1).sum()) >= core.MIN_DAYS_IN_MONTH
    assert_in_ranges(params=params)
    return Segment(
        month=month, world=world, source=source, dates=dates, cal_index=cal_index,
        base=base, valid=valid, engine=engine, solar=solar, x=x.astype(np.float32), r=r,
        bsp=bsp, csi=to_dense(calendar_values=cal.csi, cal_index=cal_index),
        clear=to_dense(calendar_values=cal.clear, cal_index=cal_index),
        cosz=to_dense(calendar_values=cal.cosz, cal_index=cal_index),
        t2m=to_dense(calendar_values=cal.t2m, cal_index=cal_index),
        s1_fallback=s1_fallback, has_bsp=has_bsp, params=params,
    )  # fmt: skip


def real_segment(
    *, cal: Calendar, month: str, use: np.ndarray, reading: np.ndarray, truth: np.ndarray
) -> Segment:
    """Return a segment of real days with no injection, for the real-data checks.

    Args:
        cal: The calendar that holds the real fields (BSP, CAMS, temperature).
        month: `YYYY-MM` of the segment.
        use: Which days of `cal` the segment may use.
        reading: The reading in points to feed the network, `(n, 48)` on the calendar axis.
        truth: The answer in points, `(n, 48)` on the calendar axis (NaN where unknown).

    Returns:
        A segment whose `r` is `reading`, whose `x` is `truth`, and whose engine and solar are zero.
    """
    first_of = np.datetime64(f"{month}-01", "D")
    nxt = (first_of.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
    n_cal = int((nxt - first_of).astype(int))
    dates = first_of - np.timedelta64(1, "D") + np.arange(n_cal + 2)
    pos = np.clip(np.searchsorted(cal.dates, dates), 0, len(cal.dates) - 1)
    present = (cal.dates[pos] == dates) & use[pos]
    present[0] = present[-1] = False
    cal_index = np.where(present, pos, -1)
    x = to_dense(calendar_values=truth, cal_index=cal_index)
    r = to_dense(calendar_values=reading, cal_index=cal_index)
    shape = (n_cal + 2, 48)
    bsp = to_dense(calendar_values=cal.bsp, cal_index=cal_index)
    return Segment(
        month=month, world="none", source="none", dates=dates, cal_index=cal_index,
        base=r, valid=np.isfinite(r) & np.isfinite(x), engine=np.zeros(shape, np.float32),
        solar=np.zeros(shape, np.float32), x=x, r=r, bsp=bsp,
        csi=to_dense(calendar_values=cal.csi, cal_index=cal_index),
        clear=to_dense(calendar_values=cal.clear, cal_index=cal_index),
        cosz=to_dense(calendar_values=cal.cosz, cal_index=cal_index),
        t2m=to_dense(calendar_values=cal.t2m, cal_index=cal_index),
        s1_fallback=np.zeros(n_cal + 2, dtype=bool),
        has_bsp=int(np.isfinite(bsp).any(axis=1).sum()) >= core.MIN_DAYS_IN_MONTH,
        params={},
    )  # fmt: skip

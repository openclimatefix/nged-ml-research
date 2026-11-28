"""Gates G1 to G5 of experiment 737, the known-answer checks, and the attribute check.

G1 and G1b recover a synthetic fleet from synthetic noise (they fit, and run in the `g1` stage). G2
checks that the injection is exact and nonzero. G3 checks that the injection and the design do not
depend on the power. G4 checks the half-hour timing convention against the real record. G5 is
computed by `e00737_score.g5_record`. The known-answer checks do not fit. A failed gate gives the
verdict `invalid`.
"""

from __future__ import annotations

import dataclasses
import importlib
import warnings
from typing import Any, Final

import numpy as np
from e00737_common import (
    FARM_IDS,
    G1_COVERAGE,
    G1_DRAWS,
    G1_SEED,
    G1_TOLERANCE,
    G1B_MIN_SHARE,
    GRID_LENGTH,
    HALF_HOURS_PER_DAY,
    INTERVAL_Z,
    N_QUARTERS,
    OUT,
    PRIMARIES,
    RIDGE_SCALE_GRID,
    RIDGE_SEED,
    RIDGE_STEP_GROWTH_SD,
    Tuned,
    atomic_json,
)
from e00737_fit import PrimaryModel, build_model, fit_arm, random_walk_prior
from e00737_forward import (
    PRIOR_FLEET,
    derated_ac_unit_output,
    faiman_cell_temperature_c,
    interpolate_hourly,
)
from e00737_inputs import (
    FEATURE_COLUMNS,
    PrimaryInputs,
    assert_cut,
    build_fit_data,
    load_farm_per_unit,
    load_primary,
    unit_term,
)
from e00737_worlds import (
    build_world,
    capacity_path,
    check_injection,
    world_specs,
)

HASATTR: Final[dict[str, tuple[str, ...]]] = {
    "studies.pv_physics": (
        "half_hour_sun_and_sky",
        "plane_of_array_w_m2",
        "daylight",
        "SunAndSky",
    ),
    "studies.pv_separation": (
        "baseline_design",
        "separate",
        "solar_basis",
        "RIDGE_PER_COLUMN",
        "FLEET_ORIENTATIONS",
        "FLEET_TILT_DEG",
    ),
    "studies.pv_dataset": ("pv_sites",),
    "studies.sources": ("ML_EXPERIMENTS_DATA_DIR",),
    "e00039_common": ("CAMS_DEMAND_POINTS_PATH",),
    "e00653_common": ("SERIES_PATH", "GRID_LENGTH", "GRID_START"),
    "e00660_generator": ("PowerPanel", "apply_events"),
    "e00040_arms": ("confound_shapes",),
    "e00040_model": ("Window",),
}
"""Every attribute taken from an imported module of this repository."""
EXPECTED_SCORED_QUARTERS: Final[dict[str, int]] = {
    "D5": 18,
    "D9": 11,
    "D10": 18,
    "D11": 18,
    "D14": 18,
    "D16": 18,
}
"""Quarters scored at 60% usable daylight half-hours, from the plan's table."""
G1_PRIMARY: Final[str] = PRIMARIES[0]
G1_SIZE_MW: Final[float] = 2.5
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
WINTER_MONTHS: Final[tuple[int, ...]] = (10, 11, 12, 1, 2, 3)
NIGHT_GHI_MAX_W_M2: Final[float] = 1.0
DARK_ZENITH_DEG: Final[float] = 100.0
FAIMAN_EXPECTED_C: Final[float] = 40.7
SHIFTS: Final[tuple[int, ...]] = (-1, 0, 1)
SCAN_SHIFTS: Final[tuple[float, ...]] = tuple(float(k) for k in np.arange(-2.0, 2.0001, 0.25))
"""Shifts of the model's unit contrast that G4 scans, in half-hours."""
G4_POOLED_DIFFERENCE: Final[float] = 0.5
"""The pooled best shift of the Primaries passes G4 when it is within this many half-hours of the
metered farms' pooled best shift."""
G4_PRIMARY_DIFFERENCE: Final[float] = 0.75
"""A Primary's best shift is within bounds when it is within this many half-hours of the mean best
shift of the metered farms' contrast against that Primary's sky."""
G4_MIN_PRIMARIES_WITHIN: Final[int] = 4
"""Primaries that must be within `G4_PRIMARY_DIFFERENCE` for G4 to pass. A Primary outside the bound
is excluded from the real-record results."""
G4_MIN_FARM_MONTHS: Final[int] = 5
"""Months of April to September that a farm's contrast needs, against one Primary's sky."""
G4_V2_POOLED_RANGE: Final[tuple[float, float]] = (-0.5, 1.5)
"""Version 2's pooled pass range, kept as a diagnostic that does not decide the gate."""
G4_V2_PRIMARY_RANGE: Final[tuple[float, float]] = (-1.0, 2.0)
"""Version 2's per-Primary pass range (strict), kept as a diagnostic only."""
G4_MIN_DAYS: Final[int] = 3
"""Days needed in each tercile, for one half-hour of one month, to count as readable."""
G4_MIN_RELIABLE_SHARE: Final[float] = 0.9
"""Share of a day's clear-sky irradiance that reliable CAMS hours must hold for the day to count."""
UNREADABLE_SHIFT: Final[int] = -99
"""The best shift reported when a contrast is unreadable (which fails the gate)."""


def missing_attributes() -> list[str]:
    """Return every `module.attribute` of `HASATTR` that an imported module lacks.

    Returns:
        The missing names (empty when all exist).
    """
    out: list[str] = []
    for module, names in HASATTR.items():
        mod = importlib.import_module(module)
        out += [f"{module}.{a}" for a in names if not hasattr(mod, a)]
    return out


def known_answers() -> dict[str, Any]:
    """Run the three forward-model known-answer tests and the cut check.

    Returns:
        A record with `passed` and each check's value.
    """
    cell = float(
        faiman_cell_temperature_c(
            poa_w_m2=np.array([800.0]), air_c=np.array([20.0]), wind_ms=np.array([2.0])
        )[0]
    )
    # Cell temperature 45 C at G = 1000 W/m2 and no wind needs an air temperature of 5 C.
    derate_first = float(
        derated_ac_unit_output(
            poa_w_m2=np.array([1000.0]),
            air_c=np.array([5.0]),
            wind_ms=np.array([0.0]),
            dc_ac_ratio=1.5,
        )[0]
    )
    hours = np.array(["2022-01-01T00:00", "2022-01-01T01:00"], dtype="datetime64[us]")
    middle = interpolate_hourly(
        hourly_times=hours,
        values=np.array([4.0, 8.0]),
        at_times=np.array(["2022-01-01T00:30"], dtype="datetime64[us]"),
    )[0]
    cut_ok = True
    try:
        assert_cut(times=np.array(["2026-07-01T00:30"], dtype="datetime64[us]"), what="probe")
        cut_ok = False
    except ValueError:
        cut_ok = True
    passed = (
        abs(cell - FAIMAN_EXPECTED_C) < 0.05
        and abs(derate_first - 1.0) < 1e-12
        and abs(middle - 6.0) < 1e-12
        and cut_ok
    )
    return {
        "passed": bool(passed),
        "faiman_cell_c": cell,
        "derate_before_clip_gives": derate_first,
        "clip_before_derate_would_give": 0.92,
        "interpolation_mid": float(middle),
        "cut_enforced": cut_ok,
    }


def quarter_counts(*, inputs: dict[str, PrimaryInputs]) -> dict[str, Any]:
    """Compare each Primary's scored-quarter count with the plan's table.

    Args:
        inputs: The Primaries' inputs by label.

    Returns:
        A record with `passed`, the counts, and the dropped half-hours per Primary.
    """
    counts = {p: int(i.scored.sum()) for p, i in inputs.items()}
    dropped = {
        p: {
            "no_power": int((~np.isfinite(i.power)).sum()),
            "unreliable_cams": int((~i.reliable).sum()),
        }
        for p, i in inputs.items()
    }
    return {
        "passed": counts == EXPECTED_SCORED_QUARTERS,
        "scored_quarters": counts,
        "dropped_half_hours": dropped,
    }


def g2(*, inputs: dict[str, PrimaryInputs]) -> dict[str, Any]:
    """Gate G2: the injection equals its formula, is nonzero by day, and is zero at night.

    It checks one 5 MW ramp world of each Primary (a mismatched fleet), and the unit output of the
    prior's own fleet.

    Args:
        inputs: The Primaries' inputs by label.

    Returns:
        A record with `passed` and each Primary's check.
    """
    records = {}
    for site, inp in inputs.items():
        spec = next(s for s in world_specs(primary=site) if s.name == "main_5_ramp_737101")
        world = build_world(inputs=inp, spec=spec)
        records[site] = check_injection(inputs=inp, world=world)
        unit = unit_term(inputs=inp, fleet=PRIOR_FLEET)
        records[site]["prior_unit_zero_at_night"] = bool(np.all(unit[~inp.sun_up] == 0.0))
    passed = all(r["passed"] and r["prior_unit_zero_at_night"] for r in records.values())
    return {"passed": bool(passed), "worlds": records}


def direct_trailing_mean(*, air_c: np.ndarray, window: int) -> np.ndarray:
    """Return the mean of the `window` values ending at each index, by a separate method.

    Args:
        air_c: The series.
        window: The window length.

    Returns:
        The means for the indices from `window - 1` on.
    """
    kernel = np.ones(window) / window
    return np.convolve(air_c, kernel, mode="valid")


def g3(*, inputs: dict[str, PrimaryInputs]) -> dict[str, Any]:
    """Gate G3: the injection and the design are the same when the power is shuffled.

    A copy of each Primary with its finite power values shuffled must give the same injected
    output and the same design matrix. The trailing temperature means are checked against a
    direct calculation from values at or before each half-hour's end, and the feature columns
    that are read must hold no forecast, lead, or rolling-mean column.

    Args:
        inputs: The Primaries' inputs by label.

    Returns:
        A record with `passed` and each Primary's checks.
    """
    rng = np.random.default_rng(3)
    records = {}
    for site, inp in inputs.items():
        finite = np.isfinite(inp.power)
        shuffled = inp.power.copy()
        shuffled[finite] = rng.permutation(inp.power[finite])
        twin = dataclasses.replace(inp, power=shuffled)
        spec = next(s for s in world_specs(primary=site) if s.name == "main_2.5_step_737102")
        a = build_world(inputs=inp, spec=spec).injected
        b = build_world(inputs=twin, spec=spec).injected
        design_a = build_fit_data(inputs=inp)
        design_b = build_fit_data(inputs=twin)
        same_design = bool(
            np.array_equal(design_a.rows, design_b.rows)
            and np.array_equal(design_a.base, design_b.base)
        )
        air = inp.weather.air_c
        columns = inp.weather.temperature_columns
        checks = {}
        for col, window in ((2, 48), (3, 144)):
            direct = direct_trailing_mean(air_c=air, window=window)
            stored = columns[window - 1 :, col]
            slope, intercept = np.polyfit(direct, stored, 1)
            checks[f"trailing_{window}"] = float(
                np.abs(stored - (slope * direct + intercept)).max()
            )
        records[site] = {
            "injection_identical": bool(np.array_equal(a, b)),
            "design_identical": same_design,
            "trailing_mean_max_gap": checks,
            "trailing_ok": all(v < 1e-8 for v in checks.values()),
        }
    forbidden = [
        c for c in FEATURE_COLUMNS if any(k in c for k in ("forecast", "lead", "init", "mean"))
    ]
    passed = not forbidden and all(
        r["injection_identical"] and r["design_identical"] and r["trailing_ok"]
        for r in records.values()
    )
    return {"passed": bool(passed), "forbidden_columns": forbidden, "primaries": records}


def seasonal_curve(*, inputs: PrimaryInputs, values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return the mean summer value minus the mean winter value, by UTC half-hour of the day.

    Args:
        inputs: The Primary's inputs.
        values: A series on the grid.
        rows: The half-hours to use.

    Returns:
        48 values.
    """
    months = (inputs.end_time - np.timedelta64(30, "m")).astype("datetime64[M]").astype(
        int
    ) % 12 + 1
    slot = np.arange(GRID_LENGTH) % 48
    out = np.zeros(48)
    for sign, season in ((1.0, SUMMER_MONTHS), (-1.0, WINTER_MONTHS)):
        use = rows & np.isin(months, season)
        sums = np.bincount(slot[use], weights=values[use], minlength=48)
        counts = np.bincount(slot[use], minlength=48)
        out += sign * sums / np.maximum(counts, 1)
    return out


def best_shift(*, real_curve: np.ndarray, model_curve: np.ndarray) -> tuple[int, dict[int, float]]:
    """Return the shift of the model curve that correlates best with the real dip.

    The real curve is a dip (summer power minus winter power is negative at midday), so it is
    compared with minus the model curve. A shift of +1 delays the model curve by one half-hour.

    Args:
        real_curve: The real summer-minus-winter power by half-hour of the day.
        model_curve: The model's summer-minus-winter unit output by half-hour of the day.

    Returns:
        The best shift and the correlation at each shift.
    """
    with np.errstate(invalid="ignore"):
        corr = {k: float(np.corrcoef(real_curve, -np.roll(model_curve, k))[0, 1]) for k in SHIFTS}
    return max(corr, key=lambda k: corr[k] if np.isfinite(corr[k]) else -np.inf), corr


def day_table(*, inputs: PrimaryInputs) -> dict[str, np.ndarray]:
    """Return each UTC day's month, weekday flag, and daily clear-sky index.

    The clear-sky index of a day is its summed CAMS irradiance over its summed clear-sky
    irradiance, over the day's reliable CAMS hours. CAMS marks many dawn and dusk hours unreliable,
    so a day is readable when its reliable hours hold at least `G4_MIN_RELIABLE_SHARE` of its
    clear-sky irradiance. A weekend day or a bank holiday is not a usable weekday.

    Args:
        inputs: The Primary's inputs.

    Returns:
        Arrays with one entry per day: `month` (1 to 12), `year_month` (months since 1970),
        `index` (the clear-sky index, NaN where it is unreadable), and `weekday` (a usable
        weekday: Monday to Friday, not a bank holiday).
    """
    start = (inputs.end_time[::HALF_HOURS_PER_DAY] - np.timedelta64(30, "m")).astype(
        "datetime64[D]"
    )
    year_month = start.astype("datetime64[M]").astype(int)
    ghi = inputs.hourly["ghi_w_m2"].to_numpy().reshape(-1, 24)
    clear = inputs.hourly["clear_sky_ghi_w_m2"].to_numpy().reshape(-1, 24)
    reliable = inputs.hourly["reliable"].to_numpy().reshape(-1, 24)
    clear_sum = clear.sum(axis=1)
    clear_reliable = np.where(reliable, clear, 0.0).sum(axis=1)
    readable = (clear_sum > 0) & (clear_reliable >= G4_MIN_RELIABLE_SHARE * clear_sum)
    index = np.divide(
        np.where(reliable, ghi, 0.0).sum(axis=1),
        clear_reliable,
        out=np.full(len(clear_sum), np.nan),
        where=readable,
    )
    holiday = inputs.weather.holiday.reshape(-1, HALF_HOURS_PER_DAY).any(axis=1)
    # 1970-01-01 was a Thursday, so (days + 3) % 7 is 0 for Monday.
    weekday = ((start.astype(int) + 3) % 7 < 5) & ~holiday
    return {
        "month": year_month % 12 + 1,
        "year_month": year_month,
        "index": index,
        "weekday": weekday,
    }


def tercile_contrast(
    *, inputs: PrimaryInputs, values: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Return the mean over months of April to September of the sunniest minus dullest tercile.

    Within each calendar month of each year, the usable weekdays are split by their daily
    clear-sky index. For each half-hour of the day, the mean of `values` on the top-tercile days
    minus its mean on the bottom-tercile days is taken over the days with a row at that half-hour
    (at least `G4_MIN_DAYS` in each tercile), and these differences are averaged over the months.
    Both terciles come from the same month, so the demand's seasonal swing cancels.

    Args:
        inputs: The Primary's inputs.
        values: A series on the grid.
        rows: The half-hours to use.

    Returns:
        48 values (NaN where no month is readable), and the number of months that were used.
    """
    days = day_table(inputs=inputs)
    by_day = np.where(rows, values, np.nan).reshape(-1, HALF_HOURS_PER_DAY)
    curves = []
    for year_month in np.unique(days["year_month"]):
        in_month = days["year_month"] == year_month
        if int(year_month) % 12 + 1 not in SUMMER_MONTHS:
            continue
        usable = np.flatnonzero(in_month & days["weekday"] & np.isfinite(days["index"]))
        if len(usable) < 3 * G4_MIN_DAYS:
            continue
        low, high = np.quantile(days["index"][usable], [1 / 3, 2 / 3])
        top = by_day[usable[days["index"][usable] >= high]]
        bottom = by_day[usable[days["index"][usable] <= low]]
        enough = (np.isfinite(top).sum(axis=0) >= G4_MIN_DAYS) & (
            np.isfinite(bottom).sum(axis=0) >= G4_MIN_DAYS
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            curve = np.nanmean(top, axis=0) - np.nanmean(bottom, axis=0)
        curves.append(np.where(enough, curve, np.nan))
    if not curves:
        return np.full(HALF_HOURS_PER_DAY, np.nan), {"months_used": 0}
    stacked = np.array(curves)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
    return mean, {"months_used": len(curves)}


def daylight_slots(*, inputs: PrimaryInputs) -> np.ndarray:
    """Return the half-hours of the day at which the sun is up on most days of April to September.

    Args:
        inputs: The Primary's inputs.

    Returns:
        48 booleans.
    """
    months = day_table(inputs=inputs)["month"]
    summer = np.isin(months, SUMMER_MONTHS)
    up = inputs.sun_up.reshape(-1, HALF_HOURS_PER_DAY)[summer]
    return up.mean(axis=0) > 0.5


def shifted_correlation(
    *, real_curve: np.ndarray, model_curve: np.ndarray, slots: np.ndarray
) -> tuple[int, dict[int, float]]:
    """Return the best shift and the correlation at each shift, over the chosen half-hours.

    The real contrast is negative at midday (more sun means less net load), so it is compared with
    minus the model's contrast. A shift of +1 delays the model curve by one half-hour.

    Args:
        real_curve: The real contrast by half-hour of the day.
        model_curve: The model's contrast by half-hour of the day.
        slots: The half-hours of the day to correlate over.

    Returns:
        The best shift (-99 when no shift is readable) and the correlation at each shift (NaN
        where unreadable).
    """
    corr: dict[int, float] = {}
    for k in SHIFTS:
        shifted = -np.roll(model_curve, k)
        use = slots & np.isfinite(real_curve) & np.isfinite(shifted)
        with np.errstate(invalid="ignore"):
            corr[k] = (
                float(np.corrcoef(real_curve[use], shifted[use])[0, 1])
                if use.sum() >= 8
                else np.nan
            )
    readable = {k: v for k, v in corr.items() if np.isfinite(v)}
    if len(readable) < len(SHIFTS):
        return UNREADABLE_SHIFT, corr
    return max(readable, key=lambda k: readable[k]), corr


def delayed_curve(*, curve: np.ndarray, shift: float) -> np.ndarray:
    """Return a 48-slot curve delayed by `shift` half-hours, interpolating linearly between slots.

    Args:
        curve: A curve by half-hour of the day.
        shift: The delay in half-hours (negative advances the curve).

    Returns:
        The curve at slot `s - shift`, wrapping around midnight.
    """
    whole = int(np.floor(shift))
    fraction = shift - whole
    if fraction == 0.0:
        return np.roll(curve, whole)
    return (1.0 - fraction) * np.roll(curve, whole) + fraction * np.roll(curve, whole + 1)


def scan_correlation(
    *, real_curve: np.ndarray, model_curve: np.ndarray, slots: np.ndarray
) -> tuple[float, dict[float, float]]:
    """Return the best shift and the correlation at each of `SCAN_SHIFTS`, over the chosen slots.

    The real contrast is negative at midday, so it is compared with minus the model's contrast. A
    shift of +1 delays the model curve by one half-hour.

    Args:
        real_curve: The real contrast by half-hour of the day.
        model_curve: The model's contrast by half-hour of the day.
        slots: The half-hours of the day to correlate over.

    Returns:
        The best shift (-99 when any shift is unreadable) and the correlation at each shift.
    """
    corr: dict[float, float] = {}
    for k in SCAN_SHIFTS:
        shifted = -delayed_curve(curve=model_curve, shift=k)
        use = slots & np.isfinite(real_curve) & np.isfinite(shifted)
        with np.errstate(invalid="ignore"):
            corr[k] = (
                float(np.corrcoef(real_curve[use], shifted[use])[0, 1])
                if use.sum() >= 8
                else np.nan
            )
    readable = {k: v for k, v in corr.items() if np.isfinite(v)}
    if len(readable) < len(SCAN_SHIFTS):
        return float(UNREADABLE_SHIFT), corr
    return max(readable, key=lambda k: readable[k]), corr


def shift_passes(*, best: float, bounds: tuple[float, float], strict: bool) -> bool:
    """Return whether a best shift lies inside the bounds (strictly, or inclusive of both ends).

    Args:
        best: The best shift in half-hours.
        bounds: The lower and upper bound.
        strict: Whether the bounds are excluded.

    Returns:
        Whether the shift passes.
    """
    low, high = bounds
    return low < best < high if strict else low <= best <= high


def farm_reference(*, inp: PrimaryInputs, farms: list[np.ndarray]) -> dict[str, Any]:
    """Return the metered farms' best shifts of the same contrast against one Primary's sky.

    Each farm's output per unit of capacity replaces the Primary's power in `tercile_contrast`
    (with its sign flipped, so that it dips at midday like net load), over the half-hours where
    the farm has a reading and the sky is reliable. The scan and the prior fleet are those of the
    Primary's own contrast. The farm file is end-labelled, so each farm is correctly indexed.

    Args:
        inp: The Primary's inputs, which supply the sky, the days, and the model's unit output.
        farms: Each farm's output per unit of capacity on the grid, NaN where missing.

    Returns:
        The best shift of each readable farm (`shifts`), and the farms' contrast and model curves
        and the daylight half-hours (`real`, `model`, `slots`, for pooling).
    """
    unit = unit_term(inputs=inp, fleet=PRIOR_FLEET)
    slots = daylight_slots(inputs=inp)
    shifts: list[float] = []
    real_curves, model_curves = [], []
    for farm in farms:
        rows = np.isfinite(farm) & inp.reliable
        real, info = tercile_contrast(inputs=inp, values=-np.nan_to_num(farm), rows=rows)
        model, _ = tercile_contrast(inputs=inp, values=unit, rows=rows)
        if info["months_used"] < G4_MIN_FARM_MONTHS:
            continue
        shift, _ = scan_correlation(real_curve=real, model_curve=model, slots=slots)
        if shift == UNREADABLE_SHIFT:
            continue
        shifts.append(shift)
        real_curves.append(real)
        model_curves.append(model)
    return {"shifts": shifts, "real": real_curves, "model": model_curves, "slots": slots}


def pooled_farm_shift(
    *, real_curves: list[np.ndarray], model_curves: list[np.ndarray], slots: np.ndarray
) -> float:
    """Return the best shift of the farms' contrast curves averaged over every Primary and farm.

    Args:
        real_curves: The farms' contrast curves, one for each readable pair of a Primary and a farm.
        model_curves: The model's contrast curves of the same pairs.
        slots: The half-hours of the day to correlate over.

    Returns:
        The best shift (-99 when there are no curves or no shift is readable).
    """
    if not real_curves:
        return float(UNREADABLE_SHIFT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        real = np.nanmean(real_curves, axis=0)
        model = np.nanmean(model_curves, axis=0)
    return scan_correlation(real_curve=real, model_curve=model, slots=slots)[0]


def g4(*, inputs: dict[str, PrimaryInputs], farms: list[np.ndarray]) -> dict[str, Any]:
    """Gate G4 (version 3): the half-hour timing convention, against the metered farms.

    For each Primary, the real power and the model's unit solar output are each contrasted between
    the sunniest and the dullest tercile of the days of the same month (April to September,
    weekdays), so that the demand's seasonal swing cancels. The model's contrast is delayed by each
    shift from -2 to +2 half-hours in steps of 0.25 (interpolating linearly), and the best shift is
    the one that correlates best with the real contrast over the daylight half-hours. The fleet mix
    is the prior mix. The metered farms, whose file is end-labelled and so correctly indexed, give
    the reference for the same method: each farm's contrast against the same Primary's sky has its
    own best shift, because the Primaries are net load and sit later than the farms by the fleet's
    lean and the demand response. The gate passes when at least `G4_MIN_PRIMARIES_WITHIN` of the
    Primaries have a best shift, minus the mean best shift of the farms against their sky, within
    `G4_PRIMARY_DIFFERENCE`, the pooled best shift of the Primaries minus the pooled best shift of
    the farms lies within `G4_POOLED_DIFFERENCE`, and the CAMS hours that are dark at both of their
    half-hours carry no irradiance. A Primary or a farm reference with no readable contrast is not
    within the bound. Each Primary outside the bound is listed in `excluded_primaries`, and the
    score stage leaves it out of the real-record results. The best
    integer shift, the registered summer-minus-winter version, and version 2's absolute ranges are
    kept as diagnostics that do not decide the gate.

    Args:
        inputs: The Primaries' inputs by label.
        farms: Each metered farm's output per unit of capacity on the grid (private).

    Returns:
        A record with `passed`, `excluded_primaries`, each Primary's best shift, farm reference and
        correlation curve, the pooled ones, and the diagnostics. It holds shifts as numbers and no
        farm identifiers.
    """
    records: dict[str, dict[str, Any]] = {}
    real_all, model_all, slots_all = [], [], []
    farm_real_all, farm_model_all = [], []
    old_real_all, old_model_all = [], []
    diagnostic: dict[str, dict[str, Any]] = {}
    integer_diagnostic: dict[str, dict[str, Any]] = {}
    for site, inp in inputs.items():
        rows = np.isfinite(inp.power) & inp.reliable
        unit = unit_term(inputs=inp, fleet=PRIOR_FLEET)
        power = np.nan_to_num(inp.power)
        real, info = tercile_contrast(inputs=inp, values=power, rows=rows)
        model, _ = tercile_contrast(inputs=inp, values=unit, rows=rows)
        slots = daylight_slots(inputs=inp)
        shift, corr = scan_correlation(real_curve=real, model_curve=model, slots=slots)
        reference = farm_reference(inp=inp, farms=farms)
        farm_real_all += reference["real"]
        farm_model_all += reference["model"]
        reference_shift = (
            float(np.mean(reference["shifts"])) if reference["shifts"] else float(UNREADABLE_SHIFT)
        )
        readable = shift != UNREADABLE_SHIFT and bool(reference["shifts"])
        difference = shift - reference_shift if readable else float("nan")
        integer_shift, integer_corr = shifted_correlation(
            real_curve=real, model_curve=model, slots=slots
        )
        integer_diagnostic[site] = {"best_shift": integer_shift, "correlation": integer_corr}
        dark = (inp.sky.zenith_deg.reshape(-1, 2) > DARK_ZENITH_DEG).all(axis=1)
        ghi = inp.hourly["ghi_w_m2"].to_numpy()
        night_max = float(ghi[dark & inp.hourly["reliable"].to_numpy()].max())
        records[site] = {
            "best_shift": shift,
            "farm_reference_shift": reference_shift,
            "farms_used": len(reference["shifts"]),
            "farm_reference_min": float(min(reference["shifts"], default=np.nan)),
            "farm_reference_max": float(max(reference["shifts"], default=np.nan)),
            "difference": difference,
            "correlation": corr,
            "readable": readable,
            "months_used": info["months_used"],
            "daylight_half_hours": int(slots.sum()),
            "dark_hour_max_ghi_w_m2": night_max,
        }
        real_all.append(real)
        model_all.append(model)
        slots_all.append(slots)
        old_real = seasonal_curve(inputs=inp, values=power, rows=rows)
        old_model = seasonal_curve(inputs=inp, values=unit, rows=rows)
        old_shift, old_corr = best_shift(real_curve=old_real, model_curve=old_model)
        diagnostic[site] = {
            "best_shift": old_shift,
            "correlation": old_corr,
            "full_day_minimum_slot": int(np.argmin(old_real)),
        }
        old_real_all.append(old_real)
        old_model_all.append(old_model)
    pooled_slots = np.any(slots_all, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pooled_real = np.nanmean(real_all, axis=0)
        pooled_model = np.nanmean(model_all, axis=0)
    pooled_shift, pooled_corr = scan_correlation(
        real_curve=pooled_real, model_curve=pooled_model, slots=pooled_slots
    )
    farm_pooled_shift = pooled_farm_shift(
        real_curves=farm_real_all, model_curves=farm_model_all, slots=pooled_slots
    )
    pooled_readable = UNREADABLE_SHIFT not in (pooled_shift, farm_pooled_shift)
    pooled_difference = pooled_shift - farm_pooled_shift if pooled_readable else float("nan")
    pooled_integer_shift, pooled_integer_corr = shifted_correlation(
        real_curve=pooled_real, model_curve=pooled_model, slots=pooled_slots
    )
    old_pooled_shift, old_pooled_corr = best_shift(
        real_curve=np.mean(old_real_all, axis=0), model_curve=np.mean(old_model_all, axis=0)
    )
    for r in records.values():
        r["within_bound"] = bool(
            r["readable"] and abs(float(r["difference"])) <= G4_PRIMARY_DIFFERENCE
        )
    excluded = [site for site, r in records.items() if not r["within_bound"]]
    passed = (
        pooled_readable
        and abs(pooled_difference) <= G4_POOLED_DIFFERENCE
        and len(records) - len(excluded) >= G4_MIN_PRIMARIES_WITHIN
        and all(float(r["dark_hour_max_ghi_w_m2"]) <= NIGHT_GHI_MAX_W_M2 for r in records.values())
    )
    v2_passed = shift_passes(best=pooled_shift, bounds=G4_V2_POOLED_RANGE, strict=False) and all(
        shift_passes(best=float(r["best_shift"]), bounds=G4_V2_PRIMARY_RANGE, strict=True)
        for r in records.values()
    )
    return {
        "passed": bool(passed),
        "version": "3-farm-reference",
        "excluded_primaries": excluded,
        "pooled_best_shift": pooled_shift,
        "pooled_farm_reference_shift": farm_pooled_shift,
        "pooled_difference": pooled_difference,
        "pooled_correlation": pooled_corr,
        "primaries": records,
        "diagnostic_v2_absolute_ranges": {
            "decides_gate": False,
            "passed_if_it_decided": bool(v2_passed),
        },
        "diagnostic_integer_shift": {
            "decides_gate": False,
            "pooled_best_shift": pooled_integer_shift,
            "pooled_correlation": pooled_integer_corr,
            "primaries": integer_diagnostic,
        },
        "diagnostic_summer_minus_winter": {
            "decides_gate": False,
            "passed_if_it_decided": bool(
                old_pooled_shift == 0 and all(d["best_shift"] == 0 for d in diagnostic.values())
            ),
            "pooled_best_shift": old_pooled_shift,
            "pooled_correlation": old_pooled_corr,
            "primaries": diagnostic,
        },
    }


def stage0_gates() -> dict[str, Any]:
    """Run every gate and check that needs no fit.

    Returns:
        The record: attributes, known answers, quarters, G2, G3, and G4.
    """
    missing = missing_attributes()
    inputs = {p: load_primary(site=p) for p in PRIMARIES}
    farms = [load_farm_per_unit(farm=k) for k in range(len(FARM_IDS))]
    return {
        "attributes": {"passed": not missing, "missing": missing},
        "known_answers": known_answers(),
        "quarters": quarter_counts(inputs=inputs),
        "G2": g2(inputs=inputs),
        "G3": g3(inputs=inputs),
        "G4": g4(inputs=inputs, farms=farms),
    }


# --- G1 and G1b ----------------------------------------------------------------------------


def g1_reference(*, model: PrimaryModel, tuned: Tuned) -> dict[str, np.ndarray]:
    """Fit arm A to the real record once, for the synthetic series' baseline and noise scales.

    Args:
        model: The first Primary's model inputs.
        tuned: The tuned growth standard deviation and ridge scale.

    Returns:
        The baseline's contribution at the fit rows, and the daylight and night scales.
    """
    fit = fit_arm(
        model=model,
        power=model.inputs.power,
        prior=random_walk_prior(growth_sd=tuned.growth_sd),
        ridge_scale=tuned.ridge_scale,
    )
    n_base = model.data.base.shape[1]
    return {
        "baseline": model.data.base @ fit.theta[:n_base],
        "sigma": np.array([fit.state.sigma_day, fit.state.sigma_night]),
    }


def synthetic_series(
    *, model: PrimaryModel, reference: dict[str, np.ndarray], seed: int, heavy: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Build one synthetic series: baseline, the prior's own fleet, and noise.

    Args:
        model: The first Primary's model inputs.
        reference: `g1_reference`'s output.
        seed: The random seed of the noise.
        heavy: Use Student-t noise with 4 degrees of freedom at the daytime scale instead of
            Gaussian noise at each class's scale.

    Returns:
        The power on the grid (NaN outside the fit rows), and the injected capacity of each
        quarter.
    """
    rng = np.random.default_rng(seed)
    data = model.data
    truth = capacity_path(shape="ramp", size_mw=G1_SIZE_MW)
    sigma_day, sigma_night = reference["sigma"]
    if heavy:
        noise = sigma_day * rng.standard_t(4.0, size=len(data.rows))
    else:
        noise = np.where(data.daylight, sigma_day, sigma_night) * rng.standard_normal(
            len(data.rows)
        )
    series = (
        reference["baseline"] - truth[data.quarter] * model.terms[PRIOR_FLEET.dc_ac_ratio] + noise
    )
    power = np.full(GRID_LENGTH, np.nan)
    power[data.rows] = series
    return power, truth


def g1_draw(
    *, model: PrimaryModel, reference: dict[str, np.ndarray], tuned: Tuned, draw: int, heavy: bool
) -> dict:
    """Fit one synthetic series of G1 (or G1b).

    Args:
        model: The first Primary's model inputs.
        reference: `g1_reference`'s output.
        tuned: The tuned growth standard deviation and ridge scale.
        draw: The draw's number.
        heavy: Use Student-t noise with 4 degrees of freedom at the daytime scale (G1b) instead of
            Gaussian noise at each class's scale (G1).

    Returns:
        The fitted capacities, their standard deviations, and the truth.
    """
    power, truth = synthetic_series(
        model=model,
        reference=reference,
        seed=G1_SEED + draw + (100_000 if heavy else 0),
        heavy=heavy,
    )
    fit = fit_arm(
        model=model,
        power=power,
        prior=random_walk_prior(growth_sd=tuned.growth_sd),
        ridge_scale=tuned.ridge_scale,
    )
    return {
        "capacity": fit.capacity.tolist(),
        "sd": fit.sd.tolist(),
        "truth": truth.tolist(),
        "converged": fit.converged,
        "n_iter": fit.n_iter,
    }


def ridge_draw(*, model: PrimaryModel, reference: dict[str, np.ndarray], draw: int) -> np.ndarray:
    """Fit one Gaussian synthetic series at each ridge scale and return the capacity errors.

    Args:
        model: The first Primary's model inputs.
        reference: `g1_reference`'s output, fitted at the unscaled ridge.
        draw: The draw's number, from 0 to `RIDGE_DRAWS` - 1.

    Returns:
        For each ridge scale of `RIDGE_SCALE_GRID`, the largest absolute gap between the fitted and
        the injected capacity over the 18 quarters, megawatts.
    """
    power, truth = synthetic_series(
        model=model, reference=reference, seed=RIDGE_SEED + draw, heavy=False
    )
    prior = random_walk_prior(growth_sd=RIDGE_STEP_GROWTH_SD)
    return np.array(
        [
            float(
                np.abs(
                    fit_arm(model=model, power=power, prior=prior, ridge_scale=scale).capacity
                    - truth
                ).max()
            )
            for scale in RIDGE_SCALE_GRID
        ]
    )


def g1_collect(*, draws: list[dict], heavy: bool) -> dict[str, Any]:
    """Summarise the draws of G1 (or G1b).

    Args:
        draws: The records of `g1_draw`.
        heavy: Whether these are the heavy-tailed draws (G1b).

    Returns:
        The record with `passed`.
    """
    cap = np.array([d["capacity"] for d in draws])
    sd = np.array([d["sd"] for d in draws])
    truth = np.array(draws[0]["truth"])
    scored = np.ones(N_QUARTERS, dtype=bool)
    share = float(cap.mean(axis=0)[scored].sum() / truth[scored].sum())
    if heavy:
        return {"passed": share >= G1B_MIN_SHARE, "recovered_share": share, "draws": len(draws)}
    gap = float(np.abs(cap.mean(axis=0) - truth).max() / truth.mean())
    coverage = float((np.abs(cap - truth) <= INTERVAL_Z * sd).mean())
    return {
        "passed": bool(
            len(draws) >= 1 and gap <= G1_TOLERANCE and G1_COVERAGE[0] <= coverage <= G1_COVERAGE[1]
        ),
        "max_gap_share_of_mean_capacity": gap,
        "coverage": coverage,
        "recovered_share": share,
        "draws": len(draws),
    }


def g1_model() -> PrimaryModel:
    """Build the model inputs of the Primary that G1 and G1b use.

    Returns:
        The model inputs of the first Primary.
    """
    return build_model(inputs=load_primary(site=G1_PRIMARY))


def write_g1_reference(*, tuned: Tuned) -> None:
    """Write the reference fit that every G1 draw reads.

    Args:
        tuned: The tuned growth standard deviation and ridge scale.
    """
    model = g1_model()
    ref = g1_reference(model=model, tuned=tuned)
    path = OUT / "g1" / "reference.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("reference.tmp.npz")
    np.savez_compressed(tmp, **ref)  # ty: ignore[invalid-argument-type]
    tmp.rename(path)


def read_g1_reference() -> dict[str, np.ndarray]:
    """Read the reference fit that `write_g1_reference` wrote.

    Returns:
        The baseline and scales.
    """
    with np.load(OUT / "g1" / "reference.npz") as f:
        return {k: f[k] for k in f.files}


def run_g1_draw(*, tuned: Tuned, draw: int, heavy: bool) -> None:
    """Fit one draw of G1 or G1b and write its record.

    Args:
        tuned: The tuned growth standard deviation and ridge scale.
        draw: The draw's number.
        heavy: Whether this is a G1b draw.
    """
    record = g1_draw(
        model=g1_model(), reference=read_g1_reference(), tuned=tuned, draw=draw, heavy=heavy
    )
    atomic_json(obj=record, path=OUT / "g1" / f"{'g1b' if heavy else 'g1'}_{draw:03d}.json")


def collect_g1() -> dict[str, dict[str, Any]]:
    """Read every draw's record and summarise G1 and G1b.

    Returns:
        The records of `G1` and `G1b`.

    Raises:
        FileNotFoundError: When a draw's file is missing.
    """
    import json

    out = {}
    for heavy, name in ((False, "G1"), (True, "G1b")):
        stem = "g1b" if heavy else "g1"
        files = [OUT / "g1" / f"{stem}_{d:03d}.json" for d in range(G1_DRAWS)]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            msg = f"{name}: {len(missing)} draw files are missing, such as {missing[0]}"
            raise FileNotFoundError(msg)
        out[name] = g1_collect(draws=[json.loads(f.read_text()) for f in files], heavy=heavy)
    return out

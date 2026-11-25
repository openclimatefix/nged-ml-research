# ruff: noqa: E501
"""Experiment 721: the synthetic world and the builders of `W2` and `W0`.

Everything in this module is NumPy on the CPU and reads no neural network.

**The synthetic world** keeps experiment 713's real grid, weather and calendar of four Primaries
    (read by
the caller) and replaces the power with `X = D - C u`, where `D` is the gross demand (a fixed
function of
calendar and temperature) and `C u` is a PV term whose yield curve `u` is experiment 716's. The
reading is
`sqrt(X² + Q²)` after experiment 720's within-half-hour interpolation. Two Primaries export on clear
middays and two never do. The size `C` of each exporting Primary is calibrated so that its share of
days
with at least one reverse-flow half-hour from March to September lands in the real 70% to 93% range.

**The injected worlds** subtract a real solar farm's recorded output (private, loaded by the caller)
    from
a host's signed power and fold the result through the meter, using experiment 720's interpolation
(`reading_from_signed`) and the injection size rule of its `build_probe_world`: the farm's size is
`level` times the host's median absolute signed reading on sunny summer-midday half-hours before
2025-07-01. No farm series, generator state, or injected size is ever written by this module.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import e00721_core as core
import e00721_score as score
import numpy as np

STEPS_PER_DAY: Final = 48
US_PER_DAY: Final = 86_400 * 1_000_000
EXPORT_SHARE_TARGET: Final = 0.80
"""Aim of the calibration of an exporting Primary's PV size: the middle of the real 70% to 93%
range."""
EXPORT_SHARE_RANGE: Final = (0.70, 0.93)
SUMMER_MONTHS: Final = (3, 4, 5, 6, 7, 8, 9)
"""March to September, the months over which the export share of a window is counted."""
NON_EXPORTER_PV_SHARE: Final = 0.05
"""A non-exporting Primary's PV size as a share of capacity: far under the gross demand's floor."""
SUNNY_K: Final = 0.7
"""Clear-sky index at or above which a half-hour is sunny, as in experiments 412 and 720."""
MIDDAY_HOURS: Final = (10, 14)
"""UTC hours [10, 14) of the sunny midday of the injection size rule."""
SUMMER_MONTHS_LEVEL: Final = (4, 5, 6, 7, 8, 9)
LEVEL_WINDOW_END_US: Final = int(np.datetime64("2025-07-01", "us").astype(np.int64))
MIN_LEVEL_ROWS: Final = 200
EXPORT_GATE_CORRELATION: Final = 0.3
EXPORT_GATE_PERCENTILE_POINTS: Final = 3.0
MIN_GATE_ROWS: Final = 200


# --------------------------------------------------------------------------- the synthetic world


@dataclass(frozen=True)
class SyntheticArrays:
    """The synthetic world's arrays, each `(n_primary, G)` in fractions of capacity unless stated.

    Attributes:
        demand: Gross demand with the day offset (what the world uses).
        demand_mean: Gross demand before the day offset (the noise-free function of calendar and
            temperature).
        pv: The PV term `C u` (zero for a non-exporter's tiny term, never negative).
        x: The signed power `demand - pv`.
        reading: The folded reading after the within-half-hour interpolation.
        u: The yield curve in 0 to 1.
        capacity_share: `C` per Primary, in fractions of capacity.
        exporter: Which Primaries export on clear middays.
        export_share: Per Primary, the share of March to September days with at least one
            reverse-flow half-hour.
    """

    demand: np.ndarray
    demand_mean: np.ndarray
    pv: np.ndarray
    x: np.ndarray
    reading: np.ndarray
    u: np.ndarray
    capacity_share: np.ndarray
    exporter: np.ndarray
    export_share: np.ndarray


def known_demand(
    *, temperature_c: np.ndarray, day_type: np.ndarray, half_hour: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Return experiment 713's known gross demand `0.5 + 0.02 max(0, 15.5 - T) + w[day type.

    Half-hour]` per Primary.

    Args:
        temperature_c: `(n_primary, G)` 2 m temperature in degrees Celsius.
        day_type: `(G,)` day type 0 to 7.
        half_hour: `(G,)` local half-hour of the day.
        table: `(8, 48)` the fixed table `w` of experiment 713.

    Returns:
        The gross demand in fractions of capacity, `(n_primary, G)`.
    """
    return 0.5 + 0.02 * np.maximum(0.0, 15.5 - temperature_c) + table[day_type, half_hour][None, :]


def _day_axis(*, n_grid: int) -> np.ndarray:
    return np.arange(n_grid) // STEPS_PER_DAY


def export_share_of(*, x: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Return, per Primary, the share of March to September days with at least one half-hour of `x.

    < 0`.
    """
    n_primary, n_grid = x.shape
    n_days = n_grid // STEPS_PER_DAY
    days = x[:, : n_days * STEPS_PER_DAY].reshape(n_primary, n_days, STEPS_PER_DAY)
    month_day = month[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)[:, 0]
    in_season = np.isin(month_day, SUMMER_MONTHS)
    exported = (days < 0.0).any(axis=2)
    return exported[:, in_season].mean(axis=1)


def calibrate_capacity_share(
    *, demand: np.ndarray, u: np.ndarray, month: np.ndarray, target: float = EXPORT_SHARE_TARGET
) -> float:
    """Return the PV size `C` (a share of capacity) at which one Primary's export share is `target`.

    The share rises with `C` (a larger PV term can only create more reverse-flow days), so a
    bisection
    on `C` finds it. `demand` and `u` are `(G,)` arrays of one Primary.
    """
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        share = float(export_share_of(x=(demand - mid * u)[None, :], month=month)[0])
        if share < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_synthetic(
    *, demand_mean: np.ndarray, u: np.ndarray, month: np.ndarray, exporter: np.ndarray, seed: int,
    day_offset_points: float = 0.0, q_points: float = core.REACTIVE_FLOOR_POINTS,
) -> SyntheticArrays:  # fmt: skip
    """Return the synthetic world on the real grid.

    Args:
        demand_mean: `(n_primary, G)` noise-free gross demand (`known_demand`).
        u: `(n_primary, G)` yield curve in 0 to 1.
        month: `(G,)` calendar month (1 to 12) of each grid index.
        exporter: `(n_primary,)` bool, whether a Primary exports on clear middays.
        seed: Seed of the day offsets.
        day_offset_points: Standard deviation of a per-day offset of the gross demand, in points of
            capacity. It is zero until the reachability check fails and then 5 points.
        q_points: The reactive floor in points.

    Returns:
        The synthetic world's arrays.
    """
    from e00720_worlds import reading_from_signed  # ty: ignore[unresolved-import]

    n_primary, n_grid = demand_mean.shape
    rng = np.random.default_rng([seed, core.RANDOM_SEED_OF_WORLD])
    day = _day_axis(n_grid=n_grid)
    n_days = int(day.max()) + 1
    offsets = rng.normal(0.0, day_offset_points / 100.0, size=(n_primary, n_days))[:, day]
    demand = demand_mean + offsets
    cap = np.empty(n_primary)
    for i in range(n_primary):
        cap[i] = (
            calibrate_capacity_share(demand=demand[i], u=u[i], month=month)
            if exporter[i]
            else NON_EXPORTER_PV_SHARE
        )
    pv = cap[:, None] * u
    x = demand - pv
    q = q_points / 100.0
    reading = np.stack([reading_from_signed(x=x[i], q=q) for i in range(n_primary)])
    return SyntheticArrays(
        demand=demand, demand_mean=demand_mean, pv=pv, x=x, reading=reading, u=u, capacity_share=cap,
        exporter=np.asarray(exporter), export_share=export_share_of(x=x, month=month),
    )  # fmt: skip


def synthetic_on_real_grid(
    *, temperature_c: np.ndarray, u: np.ndarray, stamp_us: np.ndarray, day_type: np.ndarray,
    half_hour: np.ndarray, table: np.ndarray, primaries: tuple[int, ...], exporters: tuple[int, ...],
    seed: int, day_offset_points: float = 0.0,
) -> tuple[SyntheticArrays, np.ndarray, np.ndarray]:  # fmt: skip
    """Return the synthetic world of `primaries` on the real grid, and its `(n_primary, G)` embeddings.

    The gross demand is `known_demand` of the real calendar and the real 2 m temperature, and the PV
    term is `C u` with the real shortwave likelihood `u`. Both embeddings are NaN at every Primary
    that is not in `primaries`.

    Args:
        temperature_c: `(n_primary, G)` real 2 m temperature in degrees Celsius, without NaN.
        u: `(n_primary, G)` real yield curve in 0 to 1, without NaN.
        stamp_us: `(G,)` the grid's stamps in microseconds.
        day_type: `(G,)` day type 0 to 7.
        half_hour: `(G,)` local half-hour of the day.
        table: `(8, 48)` the fixed table `w` of experiment 713.
        primaries: The `n_primary`-axis indices of the synthetic Primaries.
        exporters: The subset of `primaries` that export on clear middays.
        seed: Seed of the day offsets.
        day_offset_points: Standard deviation of the per-day demand offset, in points.

    Returns:
        The synthetic arrays of the chosen Primaries (axis 0 is `primaries` in order), the signed
        power `(n_primary, G)`, and the folded reading `(n_primary, G)`.
    """
    idx = list(primaries)
    demand_mean = known_demand(
        temperature_c=temperature_c[idx], day_type=day_type, half_hour=half_hour, table=table
    )
    _, month = hour_month(valid_us=stamp_us)
    syn = build_synthetic(
        demand_mean=demand_mean, u=u[idx], month=month, exporter=np.array([p in exporters for p in idx]),
        seed=seed, day_offset_points=day_offset_points,
    )  # fmt: skip
    signed = np.full(temperature_c.shape, np.nan)
    reading = np.full(temperature_c.shape, np.nan)
    signed[idx] = syn.x
    reading[idx] = syn.reading
    return syn, signed, reading


def shifted_sky(*, a: np.ndarray, days: int = core.SHIFT_DAYS) -> np.ndarray:
    """Return `a` from `days` days earlier on the last axis (the placebo sky), NaN at the start."""
    out = np.full(a.shape, np.nan, dtype=np.float64)
    step = days * STEPS_PER_DAY
    out[..., step:] = a[..., :-step]
    return out


# --------------------------------------------------------------------------- the margin's checks


def _auc_with_ties(*, score_: np.ndarray, positive: np.ndarray) -> float:
    return score.auc(score=score_, positive=positive)


def sky_blind_rule_scores(
    *, demand_mean: np.ndarray, reading_v: np.ndarray, pv_capacity: float
) -> np.ndarray:
    """Return the sign score of the sky-blind rule that knows the noise-free gross demand exactly.

    A reading `v` is either an import (`X = +v`, so the PV term is `D - v`) or an export (`X = -v`,
    so the
    PV term is `D + v`). The rule scores an export by which hypotheses leave a feasible PV term, one
    in
    `[0, pv_capacity]`: 1 when only the export hypothesis is feasible, 0 when only the import
    hypothesis
    is, and 0.5 when both or neither are. It reads no sky.
    """
    pv_if_import = demand_mean - reading_v
    pv_if_export = demand_mean + reading_v
    import_ok = (pv_if_import >= 0.0) & (pv_if_import <= pv_capacity)
    export_ok = (pv_if_export >= 0.0) & (pv_if_export <= pv_capacity)
    return np.where(export_ok & ~import_ok, 1.0, np.where(import_ok & ~export_ok, 0.0, 0.5))


def reachability_check(
    *, demand_mean: np.ndarray, reading: np.ndarray, x_true: np.ndarray, u: np.ndarray, day: np.ndarray,
    pv_capacity: float,
) -> dict:  # fmt: skip
    """Return the reachability check of the sky-blind rule.

    The rule's area must sit at least 0.10 plus 3 standard errors below 1.

    The token set is the margin's: the real shortwave likelihood `u` at least 0.3 and the signed
    power
    outside the 2-point dead band, on the exporting Primaries (1-D arrays of the pooled tokens). The
    standard error is the single-area form of Hanley and McNeil (1982) with the class counts set to
    the
    numbers of distinct days that hold an export token and an import token.

    Args:
        demand_mean: Noise-free gross demand of each token, in fractions of capacity.
        reading: The folded reading of each token, in fractions of capacity.
        x_true: The signed power of each token, in fractions of capacity.
        u: The real shortwave likelihood of each token.
        day: A day label of each token.
        pv_capacity: The PV size `C` that the rule's feasibility test allows, in fractions of
            capacity.

    Returns:
        The area, its standard error, the bound, the day counts, and whether the check passes.
    """
    ok = (u >= core.U_MIN) & (np.abs(x_true) * 100.0 >= core.DEAD_BAND_POINTS)
    q = core.REACTIVE_FLOOR_POINTS / 100.0
    mag = np.sqrt(np.maximum(reading[ok] ** 2 - q**2, 0.0))
    scores = sky_blind_rule_scores(
        demand_mean=demand_mean[ok], reading_v=mag, pv_capacity=pv_capacity
    )
    reverse = x_true[ok] < 0.0
    area = score.auc(score=scores, positive=reverse)
    days_ok = day[ok]
    n_pos = float(len(np.unique(days_ok[reverse])))
    n_neg = float(len(np.unique(days_ok[~reverse])))
    if not (np.isfinite(area) and n_pos > 0 and n_neg > 0):
        return {"area": area, "se": float("nan"), "pass": False, "evaluable": False}
    se = score.hanley_mcneil_se(area=area, n_pos=n_pos, n_neg=n_neg)
    bound = 1.0 - core.AREA_MARGIN - core.SE_REQUIRED * se
    return {
        "area": area, "se": se, "bound": bound, "n_days_export": n_pos, "n_days_import": n_neg,
        "pass": area <= bound, "evaluable": True,
    }  # fmt: skip


# --------------------------------------------------------------------------- the per-Primary export gate


def primary_gate_open(
    *, reading_points: np.ndarray, u: np.ndarray, k: np.ndarray, valid_us: np.ndarray
) -> dict:  # fmt: skip
    """Return experiment 417's Primary-level gate.

    (design A) for one Primary from its training-window rows.

    The gate is open when the Spearman correlation of the reading with the scaled CAMS unit shape
    `u` on
    summer-midday rows (UTC 10 to 14, April to September) with clear-sky index `k >= 0.7` exceeds
    +0.3, or
    the 1st percentile of those readings is below 3 points of capacity. With fewer than 200 such
    rows the
    gate is shut. The caller passes the rows up to the refit and nothing later.
    """
    from scipy.stats import spearmanr

    hour, month = hour_month(valid_us=valid_us)
    sunny = (
        (hour >= MIDDAY_HOURS[0]) & (hour < MIDDAY_HOURS[1]) & (month >= 4) & (month <= 9)
        & (np.nan_to_num(k) >= SUNNY_K) & np.isfinite(reading_points) & np.isfinite(u)
    )  # fmt: skip
    n = int(sunny.sum())
    if n < MIN_GATE_ROWS:
        return {"open": False, "spearman": float("nan"), "p1_points": float("nan"), "n_rows": n}
    rho = float(spearmanr(reading_points[sunny], u[sunny]).statistic)
    p1 = float(np.percentile(reading_points[sunny], 1.0))
    return {
        "open": bool(rho > EXPORT_GATE_CORRELATION or p1 < EXPORT_GATE_PERCENTILE_POINTS),
        "spearman": rho, "p1_points": p1, "n_rows": n,
    }  # fmt: skip


def hour_month(*, valid_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the UTC hour and the calendar month of valid times in microseconds."""
    t = np.asarray(valid_us, dtype=np.int64).astype("datetime64[us]")
    hour = ((t - t.astype("datetime64[D]")) // np.timedelta64(1, "h")).astype(np.int64)
    month = t.astype("datetime64[M]").astype(np.int64) % 12 + 1
    return hour, month


# --------------------------------------------------------------------------- the injected worlds


def sunny_midday_mask(*, y_pct: np.ndarray, k: np.ndarray, stamp_us: np.ndarray) -> np.ndarray:
    """Return the half-hours that set an injected farm's size.

    Sunny summer middays before 2025-07-01 with a reading.
    """
    hour, month = hour_month(valid_us=stamp_us)
    return (
        (hour >= MIDDAY_HOURS[0]) & (hour < MIDDAY_HOURS[1]) & (month >= 4) & (month <= 9)
        & (np.nan_to_num(k) >= SUNNY_K) & (stamp_us < LEVEL_WINDOW_END_US) & np.isfinite(y_pct)
    )  # fmt: skip


def injection_increment(
    *, y_pct: np.ndarray, k: np.ndarray, stamp_us: np.ndarray, level: float = core.INJECTION_LEVEL, name: str = "host"
) -> float:  # fmt: skip
    """Return the injected farm's size in points.

    `level` times the host's median absolute signed sunny summer-midday reading.

    Args:
        y_pct: The host's signed power in points of capacity.
        k: The CAMS clear-sky index at the host.
        stamp_us: Valid times in microseconds.
        level: The multiple of the median.
        name: A site label for the error message.

    Returns:
        The injected farm's size in points of the host's capacity.

    Raises:
        ValueError: With fewer than 200 such half-hours.
    """
    sunny = sunny_midday_mask(y_pct=y_pct, k=k, stamp_us=stamp_us)
    if int(sunny.sum()) < MIN_LEVEL_ROWS:
        msg = f"{name}: only {int(sunny.sum())} sunny summer-midday rows before 2025-07-01"
        raise ValueError(msg)
    return level * float(np.median(np.abs(y_pct[sunny])))


@dataclass(frozen=True)
class InjectedSeries:
    """One Primary's series in a world. The injected farm's own output is never stored.

    Attributes:
        signed_pct: The signed power in points after the injection (NaN on a dropped half-hour).
        reading_pct: The folded reading in points (within-half-hour interpolation, floor `Q`).
        increment_pct: The injected farm's size in points (0 with no injection). It is private.
    """

    signed_pct: np.ndarray
    reading_pct: np.ndarray
    increment_pct: float


def inject_and_fold(
    *, y_pct: np.ndarray, k: np.ndarray, stamp_us: np.ndarray, farm: np.ndarray | None, drop: np.ndarray,
    level: float = core.INJECTION_LEVEL, q_points: float = core.REACTIVE_FLOOR_POINTS, name: str = "host",
) -> InjectedSeries:  # fmt: skip
    """Subtract a farm from a host's signed power and fold the result through the meter.

    Args:
        y_pct: The host's signed power in points of capacity (NaN where flagged).
        k: The CAMS clear-sky index at the host.
        stamp_us: Valid times in microseconds.
        farm: The farm's normalised output on the grid (NaN in a gap), or `None` for no injection.
        drop: `(G,)` bool, the half-hours dropped because any farm has a gap. Every arm reads the
            same rows.
        level: The injection size as a multiple of the host's median absolute sunny summer-midday
            reading.
        q_points: The reactive floor.
        name: A label used in an error message (a site label, never a farm).

    Returns:
        The signed power, the folded reading, and the injected size.
    """
    from e00720_worlds import reading_from_signed  # ty: ignore[unresolved-import]

    if farm is None:
        increment = 0.0
        x = y_pct.copy()
    else:
        increment = injection_increment(y_pct=y_pct, k=k, stamp_us=stamp_us, level=level, name=name)
        x = y_pct - increment * farm
    x = np.where(drop, np.nan, x)
    reading = reading_from_signed(x=x, q=q_points)
    return InjectedSeries(signed_pct=x, reading_pct=reading, increment_pct=increment)


def farm_gap_mask(*, farms: list[np.ndarray]) -> np.ndarray:
    """Return the half-hours at which any of the farms has a gap (NaN), which every world drops."""
    gap = np.zeros(len(farms[0]), dtype=bool)
    for f in farms:
        gap |= ~np.isfinite(f)
    return gap


def dropped_share(*, drop: np.ndarray, window: slice) -> float:
    """Return the share of a fold's half-hours that the farm-gap rule drops."""
    seg = drop[window]
    return float(seg.mean()) if seg.size else float("nan")


@dataclass
class WorldArrays:
    """One world's per-Primary arrays on the grid, in fractions of capacity, shape `(16, G)`.

    The injected farm's own output is not a field. `signed` holds the signed truth of the six
    signed-MW
    Primaries (NaN for the eight MVA-only Primaries, whose sign is unknown); `fold_view` is what the
    meter reads (the folded reading at the hidden pair and at the MVA-only eight, the folded reading
    of
    the visible signed Primaries too); `signed_view` is what the network is trained on as a target
    and
    reads as history unless the sample is folded (the signed power of the visible Primaries, the
    folded
    reading everywhere else).

    Attributes:
        world: `W2` or `W0`.
        sites: The 16 site labels in Primary order.
        signed: Signed truth.
        fold_view: The folded reading of every Primary.
        signed_view: The target and the unfolded history.
        increment_pct: Per-Primary injected size in points (0 where none). Private: never written to
            a published file.
        drop: `(G,)` the dropped half-hours.
    """

    world: str
    sites: list[str]
    signed: np.ndarray
    fold_view: np.ndarray
    signed_view: np.ndarray
    increment_pct: np.ndarray
    drop: np.ndarray


def build_world_arrays(
    *, world: str, sites: list[str], power_pct: np.ndarray, k: np.ndarray, stamp_us: np.ndarray,
    farm_for_site: dict[str, np.ndarray], level: float = core.INJECTION_LEVEL,
) -> WorldArrays:  # fmt: skip
    """Return `W2` (hidden pair injected) or `W0`.

    (hidden pair folded with no injection) for all 16 Primaries.

    The four visible signed Primaries carry the same injection in both worlds. The eight MVA-only
    Primaries are as the meter gives them. Every signed Primary drops the half-hours at which any
    farm
    has a gap, in both worlds, so that every arm of both worlds reads the same rows.
    """
    assert world in {"W2", "W0"}
    n_primary, n_grid = power_pct.shape
    drop = farm_gap_mask(farms=list(farm_for_site.values()))
    signed = np.full((n_primary, n_grid), np.nan)
    fold_view = np.full((n_primary, n_grid), np.nan)
    signed_view = np.full((n_primary, n_grid), np.nan)
    increments = np.zeros(n_primary)
    for i, site in enumerate(sites):
        if site in (*core.HIDDEN_PAIR, *core.VISIBLE_SIGNED):
            inject = world == "W2" or site in core.VISIBLE_SIGNED
            s = inject_and_fold(
                y_pct=power_pct[i], k=k[i], stamp_us=stamp_us, farm=farm_for_site[site] if inject else None,
                drop=drop, level=level, name=site,
            )  # fmt: skip
            signed[i] = s.signed_pct / 100.0
            fold_view[i] = s.reading_pct / 100.0
            signed_view[i] = signed[i] if site in core.VISIBLE_SIGNED else fold_view[i]
            increments[i] = s.increment_pct
        else:
            fold_view[i] = power_pct[i] / 100.0
            signed_view[i] = fold_view[i]
    return WorldArrays(world, list(sites), signed, fold_view, signed_view, increments, drop)


def assert_no_two_worlds_on_shared_dates(
    *, series: dict[str, np.ndarray], mask: np.ndarray | None = None
) -> None:
    """Refuse a figure's data that holds two of the host's reading, the injected reading.

    And the signed truth on shared dates.

    The difference of any two is the injected farm's output.

    Args:
        series: Maps a label (`host`, `injected`, `truth`) to the array a figure would draw.
        mask: Half-hours to consider, or `None` for all.

    Raises:
        ValueError: When two or more of the labels `host`, `injected`, `truth` are present and
            finite on a
            common half-hour.
    """
    labels = [lab for lab in ("host", "injected", "truth") if lab in series]
    if len(labels) < 2:
        return
    finite = np.ones(len(series[labels[0]]), dtype=bool) if mask is None else mask.copy()
    for lab in labels:
        finite &= np.isfinite(series[lab])
    if bool(finite.any()):
        msg = f"{labels} share {int(finite.sum())} dated half-hours: their difference is the injected farm's output"
        raise core.CorrectnessFailureError(msg)


def holdout_guard(*, valid_us: np.ndarray) -> None:
    """Raise when any valid time is on or after 2026-07-01 (the farm frame runs to 2026-09-10)."""
    stop = int(np.datetime64(core.HOLDOUT_START_ISO, "us").astype(np.int64))
    if bool((np.asarray(valid_us) >= stop).any()):
        msg = f"a valid time on or after {core.HOLDOUT_START_ISO} is in the data"
        raise core.CorrectnessFailureError(msg)


def clip_to_holdout(
    *, farm_at: Callable[[np.ndarray], np.ndarray], stamp_us: np.ndarray
) -> np.ndarray:
    """Return the farm's output on the grid with every half-hour on or after 2026-07-01 set to NaN."""
    stop = int(np.datetime64(core.HOLDOUT_START_ISO, "us").astype(np.int64))
    out = np.asarray(farm_at(stamp_us), dtype=np.float64).copy()
    out[stamp_us >= stop] = np.nan
    return out

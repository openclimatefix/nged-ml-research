# ruff: noqa: E501
"""Experiment 732: the real-day readings P1 to P4 and K3, as pure functions on arrays.

H2, K1, and K2 are experiment 730's functions (`e00730_real`) called with U* in place of SD*. Every
function here returns aggregates only: a pass or fail word, a correlation, a ratio, or a share. No
function returns the level, the daily shape, or the peak of the solar output on a real solar-era day,
and P2 alone returns a mean (of the pre-engine era, which holds no solar farm) rounded to 0.1 points.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import e00729_arms as arms729
import e00729_real as real729
import e00729_sim as sim
import e00730_real as real730
import e00732_arms as arms
import e00732_core as core
import e00732_score as score
import numpy as np
import polars as pl

h2 = real730.h2
k_reading = real730.k_reading

MIDDAY: Final = slice(2 * core.P3_MIDDAY_UTC[0], 2 * core.P3_MIDDAY_UTC[1])
"""The half-hours of the midday window of P3 (10:00 to 14:00 UTC)."""
MIN_DAYS_PER_SKY: Final = 5
"""The fewest clear days, and the fewest cloudy days, that P3's ratio reads."""
K3_WINDOW_HALF_HOURS: Final = int(core.K3_WINDOW_HOURS * 2)
"""K3 counts a logged edge as hit when a flag lies within this many half-hours of its logged time."""


def p1_night(
    *, pv: np.ndarray, cosz: np.ndarray, valid: np.ndarray, in_sample: np.ndarray | None = None
) -> dict:
    """Return P1: the night bar on real half-hours, with the in-sample part (pre-engine training months) apart.

    Args:
        pv: The solar output `(N,)` of every real half-hour.
        cosz: The cosine of the zenith angle `(N,)`.
        valid: Valid half-hours `(N,)`.
        in_sample: Half-hours of the pre-engine training months (base days of the training banks).

    Returns:
        The night bar of all half-hours, of the held-out ones, and of the in-sample ones.
    """
    out = {"all": score.night_bar(pv=pv, cosz=cosz, valid=valid)}
    if in_sample is not None:
        out["held_out"] = score.night_bar(pv=pv, cosz=cosz, valid=valid & ~in_sample)
        out["in_sample"] = score.night_bar(pv=pv, cosz=cosz, valid=valid & in_sample)
    return out


def p2_no_farm(*, pv: np.ndarray, cosz: np.ndarray, valid: np.ndarray) -> dict:
    """Return P2: on pre-engine held-out sun-up half-hours, the mean at most 1 point and at most 1% above 5 points."""
    keep = valid & score.sun_up(cosz=cosz)
    if not score.finite_on(pv, keep=keep):
        return {"mean_rounded": float("nan"), "share_above": float("nan"), "n": 0, "pass": False}
    mean, share = float(pv[keep].mean()), float(np.mean(pv[keep] > core.P2_EXCEED_POINTS))
    return {
        "mean_rounded": round(mean, 1),
        "share_above": share,
        "n": int(keep.sum()),
        "pass": bool(mean <= core.P2_MEAN_POINTS and share <= core.P2_EXCEED_SHARE),
    }


def ranks(*, a: np.ndarray) -> np.ndarray:
    """Return the ranks of `a` (1 to n), with tied values sharing their mean rank."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):
        tie = a == v
        if tie.sum() > 1:
            r[tie] = r[tie].mean()
    return r


def spearman(*, a: np.ndarray, b: np.ndarray) -> float:
    """Return Spearman's rank correlation (NaN for fewer than 3 points or a constant input)."""
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(a=a), ranks(a=b))[0, 1])


def p3_follows_sky(*, pv: np.ndarray, csi: np.ndarray, cosz: np.ndarray, valid: np.ndarray) -> dict:
    """Return P3 on whole days `(D, 48)`: the Spearman correlation, and the clear-over-cloudy midday ratio.

    The correlation is between the daily sum of the solar output and the daily mean of the clear-sky
    index on sun-up half-hours. The ratio is the mean midday output of clear days over that of cloudy
    days. The function returns the correlation, the ratio, the day counts, and the pass flag, never a level.
    """
    up = valid & score.sun_up(cosz=cosz)
    days = up.sum(axis=1) >= 1
    if days.sum() < 3:
        return {
            "spearman": float("nan"),
            "ratio": float("nan"),
            "n_days": int(days.sum()),
            "pass": False,
        }
    daily_pv = np.where(up, pv, 0.0).sum(axis=1)[days]
    daily_csi = (np.where(up, csi, 0.0).sum(axis=1) / np.maximum(up.sum(axis=1), 1))[days]
    rho = spearman(a=daily_pv, b=daily_csi)
    mid_valid = valid[:, MIDDAY].all(axis=1)
    mid_pv = pv[:, MIDDAY].mean(axis=1)
    mid_csi = csi[:, MIDDAY].mean(axis=1)
    clear = mid_valid & (mid_csi >= core.P3_CLEAR_CSI)
    cloudy = mid_valid & (mid_csi < core.P3_CLOUDY_CSI)
    if (
        clear.sum() < MIN_DAYS_PER_SKY
        or cloudy.sum() < MIN_DAYS_PER_SKY
        or mid_pv[cloudy].mean() <= 0
    ):
        return {"spearman": rho, "ratio": float("nan"), "n_days": int(days.sum()), "pass": False}
    ratio = float(mid_pv[clear].mean() / mid_pv[cloudy].mean())
    return {
        "spearman": rho,
        "ratio": ratio,
        "n_days": int(days.sum()),
        "pass": bool(rho > core.P3_SPEARMAN and ratio > core.P3_RATIO),
    }


def p4_absorb(
    *,
    pv: np.ndarray,
    running: np.ndarray,
    hh: np.ndarray,
    valid: np.ndarray,
    private_log: Callable[[str], None] | None = None,
) -> dict:
    """Return P4's pass word on K1's half-hours: the solar output does not rise when the metered engine runs.

    At each UTC half-hour the mean output of the running half-hours is compared with that of the
    half-hours when the engine is off, and the differences are averaged with the running counts as
    weights. The function returns a pass flag and a void flag, never a mean, a difference, or the
    number of running half-hours, which `private_log` receives when it is given.

    Args:
        pv: The solar output of each half-hour.
        running: Whether the metered engine runs in each half-hour.
        hh: The half-hour of the day of each half-hour.
        valid: Whether each half-hour is read.
        private_log: Receives the number of running half-hours (for the private log only).

    Returns:
        `{"pass": bool, "void": bool}`.
    """
    if private_log is not None:
        private_log(f"P4: {int((valid & running).sum())} running half-hours read")
    if not score.finite_on(pv, keep=valid & running):
        return {"pass": False, "void": False}
    num, den = 0.0, 0.0
    for h in range(48):
        on = valid & running & (hh == h)
        off = valid & ~running & (hh == h)
        if on.any() and off.any():
            num += float(on.sum()) * (float(pv[on].mean()) - float(pv[off].mean()))
            den += float(on.sum())
    if den == 0.0:
        return {"pass": False, "void": True}
    return {"pass": bool(num / den <= core.P4_MAX_DIFFERENCE_POINTS), "void": False}


@dataclass(frozen=True)
class Edge:
    """A logged closure edge.

    Attributes:
        when: The logged time (midnight when the log holds a date only).
        has_time: Whether the log holds a time of day.
    """

    when: datetime
    has_time: bool


def closure_edges_from_sheets(
    *, sheets: dict[str, pl.DataFrame], substation_number: int
) -> list[Edge]:
    """Return the start and the end of every logged closure of one substation, with their times.

    The sheets are read as experiment 729's `closure_days_from_sheets` reads them (the same column rules);
    the difference is that each closure gives its two edges and not the days around them.
    """
    edges: list[Edge] = []
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
            for column in (beg, end):
                value = row[column] if column is not None else None
                if isinstance(value, datetime):
                    edges.append(Edge(when=value, has_time=True))
                elif isinstance(value, date):
                    edges.append(
                        Edge(when=datetime(value.year, value.month, value.day), has_time=False)  # noqa: DTZ001
                    )
    return edges


def k3_hit_rate(
    *, edges: Sequence[Edge], step_p: np.ndarray, day_index: dict[date, int], valid: np.ndarray
) -> dict:
    """Return K3's hit share: logged edges with a step probability above 0.5 within 2 hours (or on the logged day).

    Args:
        edges: The logged edges.
        step_p: The step output `(D, 48)` on the real days.
        day_index: The row of each real day in `step_p`.
        valid: Valid half-hours `(D, 48)`.

    Returns:
        The hit share and the number of edges on days with a reading (void below 10).
    """
    hits, n = 0, 0
    for e in edges:
        d = day_index.get(e.when.date())
        if d is None or not valid[d].any():
            continue
        n += 1
        flagged = (step_p[d] > core.K3_HIT_THRESHOLD) & valid[d]
        if e.has_time:
            h = e.when.hour * 2 + e.when.minute // 30
            lo, hi = max(h - K3_WINDOW_HALF_HOURS, 0), min(h + K3_WINDOW_HALF_HOURS + 1, 48)
            hits += int(flagged[lo:hi].any())
        else:
            hits += int(flagged.any())
    if n < core.K3_MIN_EDGES:
        return {"void": True, "n_edges": n, "hit_share": float("nan")}
    return {"void": False, "n_edges": n, "hit_share": hits / n}


def k3_rules(
    *,
    edges: Sequence[Edge],
    steps: Mapping[str, np.ndarray],
    day_index: dict[date, int],
    valid: np.ndarray,
    closure_days: np.ndarray,
) -> dict[str, dict[str, dict]]:
    """Return K3 for each rule (U*, U-N, and B-TN): the hit share and the flagged steps per day, counts and shares only.

    Args:
        edges: The logged edges of D8.
        steps: The step output of each rule `(D, 48)` on the real days.
        day_index: The row of each real day.
        valid: Valid half-hours `(D, 48)`.
        closure_days: Days within the logged closures `(D,)`.

    Returns:
        For each rule, `hit` (`k3_hit_rate`) and `false` (`k3_false_steps`).
    """
    return {
        name: {
            "hit": k3_hit_rate(edges=edges, step_p=p, day_index=day_index, valid=valid),
            "false": k3_false_steps(step_p=p, valid=valid, closure_days=closure_days),
        }
        for name, p in steps.items()
    }


def by_hour(
    *, pv: np.ndarray, hh: np.ndarray, keep: np.ndarray, threshold: float
) -> dict[int, dict[str, float]]:
    """Return, for each UTC hour of the kept half-hours, their count and the share of them above `threshold` points.

    The caller passes only night half-hours or pre-engine half-hours, so the breakdown never shows a
    level of a real solar era. An hour with no kept half-hour is absent.
    """
    out = {}
    for hour in range(24):
        sel = keep & (hh // 2 == hour)
        if sel.any():
            out[hour] = {"n": float(sel.sum()), "share_above": float(np.mean(pv[sel] > threshold))}
    return out


def night_by_hour(
    *, pv: np.ndarray, cosz: np.ndarray, valid: np.ndarray, hh: np.ndarray
) -> dict[int, dict[str, float]]:
    """Return P1's breakdown by UTC hour on night half-hours: the count and the share above 0.5 points."""
    keep = valid & (cosz <= core.NIGHT_COSZ)
    return by_hour(pv=pv, hh=hh, keep=keep, threshold=core.NIGHT_EXCEED_POINTS)


def pre_engine_by_hour(
    *, pv: np.ndarray, cosz: np.ndarray, valid: np.ndarray, hh: np.ndarray, pre_engine: np.ndarray
) -> dict[int, dict[str, float]]:
    """Return P2's breakdown by UTC hour on sun-up half-hours of the pre-engine era: the count and the share above 5 points."""
    keep = valid & pre_engine & score.sun_up(cosz=cosz)
    return by_hour(pv=pv, hh=hh, keep=keep, threshold=core.P2_EXCEED_POINTS)


def negative_power_shares(*, signed: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Return the share of half-hours with negative signed power at D5 and D14, for the private log only.

    A neighbour whose share is above zero exports at times, which bears on the sign cue of its
    channels. Missing half-hours (NaN) are not counted. The numbers stay in the private log.

    Args:
        signed: Each neighbour's signed power in points (any shape).

    Returns:
        The share for D5 and for D14 (NaN where a neighbour has no reading).
    """
    out = {}
    for name in ("D5", "D14"):
        x = np.asarray(signed[name], dtype=np.float64)
        x = x[np.isfinite(x)]
        out[name] = float(np.mean(x < 0.0)) if x.size else float("nan")
    return out


def real_day_windows(
    *,
    cal: sim.Calendar,
    use: np.ndarray,
    reading: np.ndarray,
    truth: np.ndarray,
    stats: arms729.InputStats,
    medians: Mapping[str, float],
    neighbours: Mapping[str, np.ndarray] | None,
) -> tuple[list[arms.Prepared], list[sim.Segment]]:
    """Return the prepared segments of real days, with the neighbours' real series in their channels.

    Args:
        cal: The calendar of D8's real days.
        use: The days to include `(D,)`.
        reading: D8's reading in points on the calendar axis.
        truth: The answer in points (NaN where there is none).
        stats: The standardisation constants of the temperature and wind speed.
        medians: The training median of each folded neighbour.
        neighbours: The neighbours' series on the calendar `(D, 48)`, or None for zero channels.

    Returns:
        The prepared segments and the pseudo-segments they came from, one pair per month.
    """
    segments = real729.real_segments(cal=cal, use=use, reading=reading, truth=truth)
    prepared = arms.prepare_plain(
        segments=segments,
        holidays=cal.holidays,
        stats=stats,
        medians=medians,
        neighbours=neighbours,
    )
    return prepared, segments


def k3_false_steps(*, step_p: np.ndarray, valid: np.ndarray, closure_days: np.ndarray) -> dict:
    """Return the number of flagged steps per day on held-out pre-engine days with no logged closure.

    The count is an upper bound on the false-step rate, because the logs may miss transfers.
    """
    days = ~closure_days & valid.any(axis=1)
    if not days.any():
        return {"void": True, "n_days": 0, "flagged_per_day": float("nan")}
    flagged = ((step_p > core.K3_HIT_THRESHOLD) & valid)[days].sum(axis=1)
    return {"void": False, "n_days": int(days.sum()), "flagged_per_day": float(flagged.mean())}


def load_closure_edges() -> tuple[list[Edge], str]:
    """Load the switching logs and return D8's closure edges and a status word (`loaded`, `missing`, `unreadable`).

    Only the expected failures are caught, as in experiment 729's `load_closure_days`, so a status other
    than `loaded` makes K3 void and is printed.
    """
    import e00729_core as core729

    try:
        import e00722_data as d722  # ty: ignore[unresolved-import]

        sheets = pl.read_excel(d722.SWITCHING_LOG_PATH, sheet_id=0)
        number = (
            pl.read_parquet(d722.METADATA_PATH)
            .filter(pl.col("time_series_id") == core729.D8_SERIES_ID)["substation_number"]
            .item()
        )
        edges = closure_edges_from_sheets(sheets=sheets, substation_number=int(number))
    except OSError, ImportError:
        return [], "missing"
    except KeyError, ValueError, pl.exceptions.PolarsError:
        return [], "unreadable"
    return edges, "loaded"

# ruff: noqa: E501
"""Experiment 732: the real-day readings P1 to P4 and K3, as pure functions on arrays.

H2, K1, and K2 are experiment 730's functions (`e00730_real`) called with U* in place of SD*. Every
function here returns aggregates only: a pass or fail word, a correlation, a ratio, or a share. No
function returns the level, the daily shape, or the peak of the solar output on a real solar-era day,
and P2 alone returns a mean (of the pre-engine era, which holds no solar farm) rounded to 0.1 points.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final
from datetime import date, datetime

import e00732_paths  # noqa: F401

# isort: split
import e00730_real as real730
import e00732_core as core
import e00732_score as score
import numpy as np
import polars as pl

h2 = real730.h2
k_reading = real730.k_reading

MIDDAY: Final = slice(20, 28)
"""Half-hours 20 to 27 are 10:00 to 14:00 UTC."""
CLEAR_INDEX: Final = 0.8
CLOUDY_INDEX: Final = 0.3
MIN_DAYS_PER_SKY: Final = 5
P2_MEAN_MAX: Final = 1.0
P2_EXCEED_POINTS: Final = 5.0
P2_EXCEED_SHARE: Final = 0.01
P3_MIN_CORRELATION: Final = 0.5
P3_MIN_RATIO: Final = 2.0
P4_MAX_DIFFERENCE: Final = 1.0
K3_MIN_EDGES: Final = 10
K3_WINDOW_HALF_HOURS: Final = 4


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
    if not keep.any():
        return {"mean_rounded": float("nan"), "share_above": float("nan"), "n": 0, "pass": False}
    mean, share = float(pv[keep].mean()), float(np.mean(pv[keep] > P2_EXCEED_POINTS))
    return {
        "mean_rounded": round(mean, 1),
        "share_above": share,
        "n": int(keep.sum()),
        "pass": bool(mean <= P2_MEAN_MAX and share <= P2_EXCEED_SHARE),
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
    clear = mid_valid & (mid_csi >= CLEAR_INDEX)
    cloudy = mid_valid & (mid_csi < CLOUDY_INDEX)
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
        "pass": bool(rho > P3_MIN_CORRELATION and ratio > P3_MIN_RATIO),
    }


def p4_absorb(*, pv: np.ndarray, running: np.ndarray, hh: np.ndarray, valid: np.ndarray) -> dict:
    """Return P4's pass word on K1's half-hours: the solar output does not rise when the metered engine runs.

    At each UTC half-hour the mean output of the running half-hours is compared with that of the
    half-hours when the engine is off, and the differences are averaged with the running counts as
    weights. The function returns a pass flag and counts, never a mean or a difference.
    """
    num, den = 0.0, 0.0
    for h in range(48):
        on = valid & running & (hh == h)
        off = valid & ~running & (hh == h)
        if on.any() and off.any():
            num += float(on.sum()) * (float(pv[on].mean()) - float(pv[off].mean()))
            den += float(on.sum())
    if den == 0.0:
        return {"pass": False, "n_running": int((valid & running).sum()), "void": True}
    return {
        "pass": bool(num / den <= P4_MAX_DIFFERENCE),
        "n_running": int((valid & running).sum()),
        "void": False,
    }


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
                        Edge(when=datetime(value.year, value.month, value.day), has_time=False)
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
        flagged = (step_p[d] > core.STEP_THRESHOLD) & valid[d]
        if e.has_time:
            h = e.when.hour * 2 + e.when.minute // 30
            lo, hi = max(h - K3_WINDOW_HALF_HOURS, 0), min(h + K3_WINDOW_HALF_HOURS + 1, 48)
            hits += int(flagged[lo:hi].any())
        else:
            hits += int(flagged.any())
    if n < K3_MIN_EDGES:
        return {"void": True, "n_edges": n, "hit_share": float("nan")}
    return {"void": False, "n_edges": n, "hit_share": hits / n}


def k3_false_steps(*, step_p: np.ndarray, valid: np.ndarray, closure_days: np.ndarray) -> dict:
    """Return the number of flagged steps per day on held-out pre-engine days with no logged closure.

    The count is an upper bound on the false-step rate, because the logs may miss transfers.
    """
    days = ~closure_days & valid.any(axis=1)
    if not days.any():
        return {"void": True, "n_days": 0, "flagged_per_day": float("nan")}
    flagged = ((step_p > core.STEP_THRESHOLD) & valid)[days].sum(axis=1)
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

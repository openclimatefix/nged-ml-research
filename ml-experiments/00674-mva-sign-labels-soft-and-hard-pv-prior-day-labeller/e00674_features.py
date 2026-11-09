# ruff: noqa: E501
"""Day features for experiment 674: shoulder minimum, W-shape score, export ratio, PV stratum.

PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-*:ml-experiments/00508-*

Reuses experiment 506's data layer and experiment 508's sun-elevation day matrices (`Mats`, the
shoulder windows, the farm injection). The W-shape score is copied from experiment 417's
`e00417_day_flip.py` (the demand profile there needs the fold tables of experiments 405 and 417,
which this experiment does not use), with the ERA5 temperature term set to zero and no bank
holidays. No model is fitted.
"""

from dataclasses import dataclass
from typing import Final

import e00506_data as data  # ty: ignore[unresolved-import]
import e00508_geometry as geo  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SLUG: Final[str] = "mva-sign-labels-soft-and-hard-pv-prior-day-labeller"
WORK = data.ML_EXPERIMENTS_DATA_DIR / f"00674-{SLUG}" / "work"
R4_TABLE: Final = (
    data.ML_EXPERIMENTS_DATA_DIR
    / "00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals"
    / "work"
    / "table_R4_pooled_table_era5_cams_mean_cloudterm.parquet"
)
LEVELS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.0)
CUT_DATE: Final[int] = 20260701
"""Days on or after this date are dropped (the CAMS arrays run on to the end of July 2026)."""
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
SUNNY_K: Final[float] = 0.7
CLOUDY_K: Final[float] = 0.3
GATE_SHARE: Final[float] = 0.5
"""The capacity gate: AC p90 PV capacity at least this share of the night import (in MW)."""
UNKNOWN_RATIO: Final[float] = 3.0
LOW_DEMAND_SHARE: Final[float] = 0.2
MIN_SHOULDER_HALF_HOURS: Final[int] = 3
MIN_NEGATIVE_HALF_HOURS: Final[int] = 2
"""A reverse-flow day has at least this many negative daylight half-hours (as in 417 and 508)."""

# --- W-shape score, copied from experiment 417 (`e00417_day_flip.py`) ---------------------------
U_LOW: Final[float] = 0.1
U_HIGH: Final[float] = 0.95
MIN_DAY_MAX_U: Final[float] = 0.6
OVERCAST_K: Final[float] = 0.3
MIN_OVERCAST_HOURS: Final[int] = 3
MIN_SIDE: Final[int] = 3
BLOCK_DAYS: Final[int] = 28
MAX_EARLIER_BLOCKS: Final[int] = 6
BROKEN_ROUGHNESS: Final[float] = 0.05
US_PER_HOUR: Final[int] = 3_600_000_000


@dataclass
class Hourly:
    """A Primary's hourly means (UTC hours) of the reading, `u`, and `k`."""

    day: np.ndarray
    hod: np.ndarray
    r: np.ndarray
    u: np.ndarray
    k: np.ndarray
    weekend: np.ndarray
    month: np.ndarray
    noon_hod: float


def hourly(*, mats: geo.Mats, reading: np.ndarray, lon: float) -> Hourly:
    """Average the finite half-hours of the complete days into UTC hours (both halves required)."""
    midnight = (
        pl.Series(
            [
                f"{int(d) // 10000:04d}-{int(d) // 100 % 100:02d}-{int(d) % 100:02d}"
                for d in mats.dates
            ]
        )
        .str.to_datetime("%Y-%m-%d", time_unit="us")
        .dt.replace_time_zone("Europe/London")
        .dt.epoch("us")
        .to_numpy()
    )
    start_us = (midnight[:, None] + np.arange(48)[None, :] * 1_800_000_000).ravel()
    r, u, k = reading.ravel(), mats.u.ravel(), mats.k.ravel()
    start = pl.Series(start_us).cast(pl.Datetime("us"))
    dow = start.dt.weekday().to_numpy()
    ok = np.isfinite(r)
    frame = (
        pl.DataFrame(
            {
                "h": (start_us // US_PER_HOUR)[ok],
                "r": r[ok],
                "u": u[ok],
                "k": k[ok],
                "w": (dow >= 6)[ok],
            }
        )
        .group_by("h")
        .agg(
            pl.col("r").mean(),
            pl.col("u").mean(),
            pl.col("k").mean(),
            pl.col("w").max(),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") == 2)
        .sort("h")
    )
    h = frame["h"].to_numpy()
    month = pl.Series(h * US_PER_HOUR).cast(pl.Datetime("us")).dt.month().to_numpy()
    return Hourly(
        day=h // 24,
        hod=h % 24,
        r=frame["r"].to_numpy(),
        u=frame["u"].to_numpy(),
        k=frame["k"].to_numpy(),
        weekend=frame["w"].to_numpy().astype(bool),
        month=month,
        noon_hod=12.0 - lon / 15.0,
    )


@dataclass
class Profile:
    """Overcast-day demand medians by (block, day type, hour of day)."""

    median_r: dict[tuple[int, int, int], float]


def build_profile(*, h: Hourly) -> Profile:
    """Return the overcast-day profile (experiment 417's, without the temperature slope)."""
    daylight = h.u > 0.01
    frame = pl.DataFrame({"d": h.day, "k": np.where(daylight, h.k, np.nan), "u": h.u}).with_columns(
        pl.col("k").fill_nan(None)
    )
    daily = frame.group_by("d").agg(
        pl.col("k").mean().alias("mk"), (pl.col("u") > 0.01).sum().alias("nd")
    )
    overcast = {
        int(d) for d, mk, nd in daily.iter_rows() if nd >= 6 and mk is not None and mk < OVERCAST_K
    }
    sel = np.isin(h.day, list(overcast))
    grouped = (
        pl.DataFrame(
            {
                "b": (h.day // BLOCK_DAYS)[sel],
                "w": h.weekend[sel].astype(int),
                "h": h.hod[sel],
                "r": h.r[sel],
            }
        )
        .group_by("b", "w", "h")
        .agg(pl.col("r").median().alias("mr"), pl.len().alias("n"))
    )
    return Profile(
        median_r={
            (b, w, hh): mr for b, w, hh, mr, n in grouped.iter_rows() if n >= MIN_OVERCAST_HOURS
        }
    )


def expected(*, p: Profile, b: int, w: int, hh: int) -> float:
    """Return the expected demand reading, widening day type and then block when a cell is empty."""
    earlier = [(b - k, ww) for k in range(1, MAX_EARLIER_BLOCKS + 1) for ww in (w, 1 - w)]
    for bb, ww in ((b, w), (b, 1 - w), *earlier, (b + 1, w), (b + 1, 1 - w)):
        if (bb, ww, hh) in p.median_r:
            return p.median_r[(bb, ww, hh)]
    return float("nan")


def pav_decreasing_rss(*, y: np.ndarray) -> float:
    """Return the residual sum of squares of the best non-increasing fit to `y`."""
    blocks: list[list[float]] = []
    for v in y:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, n2 = blocks.pop()
            m1, n1 = blocks.pop()
            blocks.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])
    fit = np.concatenate([[m] * int(n) for m, n in blocks])
    return float(np.sum((y - fit) ** 2))


def limb_score(*, u: np.ndarray, e: np.ndarray) -> float:
    """Return the score of one limb: how much better a V fits than a non-increasing curve."""
    order = np.argsort(u)
    u, e = u[order], e[order]
    n = len(u)
    if n < 2 * MIN_SIDE:
        return 0.0
    iso = pav_decreasing_rss(y=e)
    best_rss = np.inf
    for k in range(MIN_SIDE, n - MIN_SIDE):
        design = np.stack([np.ones(n), u, np.maximum(u - u[k], 0.0)], axis=1)
        coef, *_ = np.linalg.lstsq(design, e, rcond=None)
        if coef[1] >= 0 or coef[1] + coef[2] <= 0:
            continue
        rss = float(np.sum((e - design @ coef) ** 2))
        best_rss = min(best_rss, rss)
    if not np.isfinite(best_rss) or best_rss <= 0:
        return 0.0
    return max((iso - best_rss) / best_rss, 0.0)


def w_scores(*, h: Hourly) -> dict[int, float]:
    """Return the W-shape score of each eligible April to September day (key: UTC day number)."""
    profile = build_profile(h=h)
    out: dict[int, float] = {}
    daily_max = pl.DataFrame({"d": h.day, "u": h.u}).group_by("d").agg(pl.col("u").max())
    for d, umax in daily_max.iter_rows():
        if umax < MIN_DAY_MAX_U:
            continue
        sel = np.flatnonzero(h.day == d)
        if h.month[sel[0]] not in SUMMER_MONTHS:
            continue
        b = int(d) // BLOCK_DAYS
        e_all = np.array(
            [
                h.r[i] ** 2 - expected(p=profile, b=b, w=int(h.weekend[i]), hh=int(h.hod[i])) ** 2
                for i in sel
            ]
        )
        u_day = h.u[sel]
        use = (u_day >= U_LOW) & (u_day < U_HIGH) & np.isfinite(e_all)
        daylight = u_day >= U_LOW
        u_by_time = u_day[daylight][np.argsort(h.hod[sel][daylight])]
        rough = float(np.mean(np.abs(np.diff(u_by_time, n=2)))) if len(u_by_time) > 2 else 0.0
        if rough > BROKEN_ROUGHNESS:
            if use.sum() < 2 * MIN_SIDE + 1:
                continue
            out[int(d)] = limb_score(u=u_day[use], e=e_all[use])
        else:
            morning = use & (h.hod[sel] + 0.5 < h.noon_hod)
            afternoon = use & (h.hod[sel] + 0.5 >= h.noon_hod)
            if morning.sum() < 2 * MIN_SIDE or afternoon.sum() < 2 * MIN_SIDE:
                continue
            out[int(d)] = min(
                limb_score(u=u_day[morning], e=e_all[morning]),
                limb_score(u=u_day[afternoon], e=e_all[afternoon]),
            )
    return out


# --- day table ----------------------------------------------------------------------------------


def pv_gate_table() -> dict[str, float]:
    """Return each Primary's AC p90 PV capacity (MW) from the latest season-year of experiment 474."""
    r4 = pl.read_parquet(R4_TABLE).sort("site", "season_year")
    return {
        s: float(g["ac_p90"][-1])
        for s, g in ((k[0], v) for k, v in r4.group_by("site", maintain_order=True))
    }


def load_mats(*, site: str) -> geo.Mats:
    """Return the site's day matrices (with the farm column for the six)."""
    extra = (
        {"farm": data.farm_norm(farm=data.FARM_OF_SITE[site])}
        if site in data.FARM_OF_SITE
        else None
    )
    return geo.Mats(site=site, extra=extra)


def day_table(
    *, site: str, mats: geo.Mats, level: float, gate_mw: dict[str, float]
) -> pl.DataFrame:
    """Return one row per complete UK day for one Primary and injection level.

    Columns: site, level, date, month, block, r (day export ratio), night_import (points), s
    (shoulder minimum over the night import), w (W-shape score, null when not computable), k_mid
    (mean clear-sky index 10:00 to 15:00), gate, crossing (signed power negative and positive in
    daylight), any_negative, low_demand, n_morning, n_evening, mid_min (lowest 10:00 to 15:00
    reading in points).
    """
    cap_mw = float(data.series_table().filter(pl.col("site") == site)["effective_capacity_mw"][0])
    sig, _ = geo.signed(mats=mats, level=level)
    reading = np.abs(sig)
    finite = np.isfinite(sig)
    band = geo.calibrate_shoulder(mats=mats)
    masks = geo.window_masks(mats=mats, band=band)
    dark = finite & (mats.el < geo.DARK_ELEVATION) & np.isin(mats.month, SUMMER_MONTHS)[:, None]
    night = float(np.median(reading[dark]))
    mid = np.zeros_like(finite)
    mid[:, geo.MID_SLOTS] = True
    mid = mid & finite
    n_mid = mid.sum(axis=1)
    with np.errstate(all="ignore"):
        mid_median = np.array(
            [
                np.median(reading[i, mid[i]]) if n_mid[i] >= 6 else np.nan
                for i in range(len(mats.dates))
            ]
        )
        k_mid = np.nanmean(
            np.where(mats.k[:, geo.MID_SLOTS] >= 0, mats.k[:, geo.MID_SLOTS], np.nan), axis=1
        )
    shoulder_min = {}
    n_shoulder = {}
    for name in ("morning", "evening"):
        m = masks[name] & finite
        n_shoulder[name] = m.sum(axis=1)
        shoulder_min[name] = np.where(m, reading, np.inf).min(axis=1)
    s = np.minimum(shoulder_min["morning"], shoulder_min["evening"]) / night
    s = np.where(
        (n_shoulder["morning"] >= MIN_SHOULDER_HALF_HOURS)
        & (n_shoulder["evening"] >= MIN_SHOULDER_HALF_HOURS),
        s,
        np.nan,
    )
    lit = finite & (mats.el > 0)
    smin = np.where(lit, sig, np.inf).min(axis=1)
    smax = np.where(lit, sig, -np.inf).max(axis=1)
    n_negative = (lit & (sig < 0)).sum(axis=1)
    crossing = (n_negative >= MIN_NEGATIVE_HALF_HOURS) & (smax > 0)
    # Low-demand days: lowest 20% of the Primary's own April to September days by the median
    # 10:00 to 15:00 reading *before* injection.
    raw_mid = np.array(
        [
            np.median(np.abs(mats.p[i, mid[i]])) if n_mid[i] >= 6 else np.nan
            for i in range(len(mats.dates))
        ]
    )
    summer = mats.in_months()
    cut = float(np.nanquantile(raw_mid[summer], LOW_DEMAND_SHARE))
    # Injected farm size in MW (a lab shortcut: the capacity table would be re-estimated).
    farm_mw = 0.0
    if level > 0 and site in data.FARM_OF_SITE:
        farm_mw = float(
            level
            * np.median(
                np.abs(
                    mats.p[
                        mats.before_cut
                        & (mats.k >= data.SUNNY_K)
                        & finite
                        & np.isin(mats.month, range(4, 10))[:, None]
                        & np.isin(np.arange(48), range(20, 30))[None, :]
                    ]
                )
            )
            * cap_mw
            / 100.0
        )
    night_mw = night * cap_mw / 100.0
    gate = bool(
        np.isfinite(gate_mw.get(site, np.nan))
        and gate_mw.get(site, 0.0) + farm_mw >= GATE_SHARE * night_mw
    )
    return pl.DataFrame(
        {
            "site": site,
            "level": level,
            "date": mats.dates,
            "month": mats.month,
            "block": mats.block,
            "r": pl.Series(mid_median / night).fill_nan(None),
            "night_import": night,
            "s": pl.Series(s).fill_nan(None),
            "k_mid": pl.Series(k_mid).fill_nan(None),
            "gate": gate,
            "crossing": crossing,
            "any_negative": (smin < 0) & lit.any(axis=1),
            "low_demand": raw_mid <= cut,
            "mid_min": np.where(mid, reading, np.inf).min(axis=1),
        }
    ).filter(pl.col("date") < CUT_DATE)


def add_w(*, table: pl.DataFrame, site: str, mats: geo.Mats, level: float) -> pl.DataFrame:
    """Add the W-shape score to a day table (null on days the score cannot be computed)."""
    sig, _ = geo.signed(mats=mats, level=level)
    row = data.series_table().filter(pl.col("site") == site).row(0, named=True)
    h = hourly(mats=mats, reading=np.abs(sig), lon=float(row["longitude"]))
    scores = w_scores(h=h)
    # UTC day number -> UK date integer through the day's noon.
    days = np.array(sorted(scores))
    dates = (
        pl.Series(days * 86_400_000_000 + 43_200_000_000)
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )
    date_int = (
        dates.dt.year().cast(pl.Int64) * 10000
        + dates.dt.month().cast(pl.Int64) * 100
        + dates.dt.day().cast(pl.Int64)
    ).to_numpy()
    w_frame = pl.DataFrame({"date": date_int, "w": [scores[int(d)] for d in days]})
    return table.join(w_frame, on="date", how="left")

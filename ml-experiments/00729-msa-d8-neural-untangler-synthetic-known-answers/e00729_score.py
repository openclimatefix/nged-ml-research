"""Experiment 729: the scored rows, the per-row metrics, the two-way bootstrap, and the bars.

**The scored rows** are the 48 half-hours of each base day of a bank's segments, where the real
reading exists. Every arm and baseline is scored on the same rows, in the same order.

**The two-way bootstrap** resamples, within each test month, whole blocks of 7 consecutive base
dates and, drawn independently, whole segments (the 8 draws of the month). A resampled cell pairs
one drawn block with one drawn segment and carries all 48 half-hours of each date in the block. The
point estimate is the mean over seeds, and each resample uses the seed-averaged losses.

**The Kish widening.** The Kish number of the two-way design is the product of the Kish numbers of
its two factors. Below 20 of either factor, the interval is widened about its point by the ratio of
the Student t quantile (Kish number minus 1 degrees of freedom, at the contrast's level) to the
normal quantile, using the smaller factor. The two-stage resampler of experiment 721 draws folds and
blocks of one factor family and cannot draw segments, so it is not used; the Kish number and the
t-ratio rule are the same as 721's.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import e00729_banks as banks
import e00729_baselines as base
import e00729_core as core
import e00729_dist as dist
import e00729_sim as sim
import numpy as np

SOURCE_CODE = {"none": 0, "S1": 1, "S2": 2}
EPOCH = np.datetime64("1970-01-01", "D")


@dataclass
class Rows:
    """The scored half-hours of a bank, as parallel arrays of length `N`.

    Attributes:
        seg: The segment (draw) number, `0..n_segments-1`.
        month: The index of the segment's month among the bank's months.
        date: The base day as days since 1970-01-01.
        hh: The UTC half-hour, `0..47`.
        x: The signed truth in points.
        r: The reading in points.
        engine: The engine output in points.
        solar: The solar output in points.
        world: The world type as an index into `WORLD_TYPES`.
        source: 0 none, 1 S1, 2 S2.
        fallback: The base day took source S2 because no farm had data.
        has_bsp: The segment's month carries real BSP steps.
        features: The baselines' features.
    """

    seg: np.ndarray
    month: np.ndarray
    date: np.ndarray
    hh: np.ndarray
    x: np.ndarray
    r: np.ndarray
    engine: np.ndarray
    solar: np.ndarray
    world: np.ndarray
    source: np.ndarray
    fallback: np.ndarray
    has_bsp: np.ndarray
    features: base.Features

    @property
    def n(self) -> int:
        """The number of scored half-hours."""
        return int(self.x.shape[0])


def rows_from_segments(*, segments: Sequence[sim.Segment], holidays: np.ndarray) -> Rows:
    """Build the scored rows of a list of segments (the base days' valid half-hours)."""
    months = {m: i for i, m in enumerate(sorted({s.month for s in segments}))}
    parts: dict[str, list[np.ndarray]] = {
        k: []
        for k in [
            "seg",
            "month",
            "date",
            "hh",
            "x",
            "r",
            "engine",
            "solar",
            "world",
            "source",
            "fallback",
            "has_bsp",
            "local_hh",
            "local_month",
            "weekend",
            "weekday",
            "holiday",
            "t2m",
        ]
    }
    for k, s in enumerate(segments):
        days = np.flatnonzero(s.cal_index >= 0)
        for d in days:
            ok = s.valid[d] & np.isfinite(s.x[d])
            hh = np.flatnonzero(ok)
            n = len(hh)
            if n == 0:
                continue
            date = s.dates[d].astype("datetime64[D]")
            offset = sim.utc_offset_hours(day=date.astype(object))
            weekday = int((date - EPOCH).astype(int) + 3) % 7
            local = (hh + round(2 * offset)) % 48
            parts["seg"].append(np.full(n, k))
            parts["month"].append(np.full(n, months[s.month]))
            parts["date"].append(np.full(n, int((date - EPOCH).astype(int))))
            parts["hh"].append(hh)
            parts["x"].append(s.x[d][hh])
            parts["r"].append(s.r[d][hh])
            parts["engine"].append(s.engine[d][hh])
            parts["solar"].append(s.solar[d][hh])
            parts["world"].append(np.full(n, core.WORLD_TYPES.index(s.world)))
            parts["source"].append(np.full(n, SOURCE_CODE[s.source]))
            parts["fallback"].append(np.full(n, bool(s.s1_fallback[d])))
            parts["has_bsp"].append(np.full(n, s.has_bsp))
            parts["local_hh"].append(local)
            parts["local_month"].append(np.full(n, int(str(date)[5:7])))
            parts["weekend"].append(np.full(n, weekday >= 5))
            parts["weekday"].append(np.full(n, weekday))
            parts["holiday"].append(np.full(n, bool(np.isin(date, holidays))))
            parts["t2m"].append(s.t2m[d][hh])
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    feats = base.Features(
        r=cat["r"].astype(np.float64), local_hh=cat["local_hh"], local_month=cat["local_month"],
        weekend=cat["weekend"], weekday=cat["weekday"], holiday=cat["holiday"],
        t2m=cat["t2m"].astype(np.float64),
    )  # fmt: skip
    return Rows(
        seg=cat["seg"], month=cat["month"], date=cat["date"], hh=cat["hh"],
        x=cat["x"].astype(np.float64), r=cat["r"].astype(np.float64),
        engine=cat["engine"].astype(np.float64), solar=cat["solar"].astype(np.float64),
        world=cat["world"], source=cat["source"], fallback=cat["fallback"],
        has_bsp=cat["has_bsp"], features=feats,
    )  # fmt: skip


def subset_masks(*, rows: Rows) -> dict[str, np.ndarray]:
    """Return the row subsets every table reports: all, sign, dead band, world, source, daylight."""
    x = rows.x
    out = {
        "all": np.ones(rows.n, dtype=bool),
        "import": x >= 0,
        "reverse": x < 0,
        "dead_band": np.abs(x) < core.DEAD_BAND_POINTS,
        "reverse_outside_dead_band": x < -core.DEAD_BAND_POINTS,
    }
    for i, w in enumerate(core.WORLD_TYPES):
        out[f"world_{w}"] = rows.world == i
    out["source_S1"] = (rows.source == 1) & ~rows.fallback
    out["source_S2"] = rows.source == 2
    lo, hi = core.DAYLIGHT_HOURS_UTC
    day = (rows.hh >= 2 * lo) & (rows.hh < 2 * hi)
    out["daylight"] = day
    out["other_hours"] = ~day
    return out


# --------------------------------------------------------------------------- per-row metrics


@dataclass
class RowMetrics:
    """The per-row metrics of one forecast, seed-averaged where the forecast has seeds.

    Attributes:
        crps: The signed CRPS in points.
        pinball: Twice the mean pinball loss over the 11 levels, in points.
        mae: The absolute error of the median, in points of capacity.
        median: The median of the forecast distribution, in points.
        p_neg: P(X < 0).
        covers: Whether the lobe on the true side covers the truth (two-lobe heads only).
        width: The width of that lobe's 5% to 95% range.
        q05_cover: The truth lies above the 5% quantile.
        q95_cover: The truth lies below the 95% quantile.
    """

    crps: np.ndarray
    pinball: np.ndarray
    mae: np.ndarray
    median: np.ndarray
    p_neg: np.ndarray
    covers: np.ndarray
    width: np.ndarray
    q05_cover: np.ndarray
    q95_cover: np.ndarray


def lobe_metrics(*, lobes: dist.Lobes, x: np.ndarray) -> RowMetrics:
    """Return every per-row metric of a two-lobe forecast."""
    q = dist.mixture_quantiles(lobes=lobes)
    return RowMetrics(
        crps=dist.mixture_crps(lobes=lobes, y=x),
        pinball=dist.pinball_points(q=q, y=x),
        mae=np.abs(x - q[:, 5]),
        median=q[:, 5],
        p_neg=dist.prob_negative(lobes=lobes),
        covers=dist.lobe_covers(lobes=lobes, y=x).astype(float),
        width=dist.lobe_width(lobes=lobes, y=x),
        q05_cover=(x >= q[:, 0]).astype(float),
        q95_cover=(x <= q[:, -1]).astype(float),
    )


def quantile_metrics(*, q: np.ndarray, x: np.ndarray) -> RowMetrics:
    """Return every per-row metric of a quantile forecast `(N, 11)`."""
    taus = np.asarray(core.QUANTILE_LEVELS)
    qs = np.maximum.accumulate(q, axis=1)
    p_neg = np.array([np.interp(0.0, row, taus, left=0.0, right=1.0) for row in qs])
    nan = np.full(len(x), np.nan)
    return RowMetrics(
        crps=dist.quantile_crps(q=qs, y=x),
        pinball=dist.pinball_points(q=qs, y=x),
        mae=np.abs(x - qs[:, 5]),
        median=qs[:, 5],
        p_neg=p_neg,
        covers=nan,
        width=nan,
        q05_cover=(x >= qs[:, 0]).astype(float),
        q95_cover=(x <= qs[:, -1]).astype(float),
    )


def average_metrics(*, per_seed: Sequence[RowMetrics]) -> RowMetrics:
    """Return the mean over seeds of every per-row metric (P(X < 0) too, as an equal mixture)."""
    fields = RowMetrics.__dataclass_fields__
    return RowMetrics(**{k: np.mean([getattr(m, k) for m in per_seed], axis=0) for k in fields})


def subset_means(
    *, metrics: RowMetrics, masks: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Return mean crps, pinball, and mae on each subset (NaN when a subset is empty)."""
    out: dict[str, dict[str, float]] = {}
    for name, m in masks.items():
        n = int(m.sum())
        out[name] = {"n": float(n)} | {
            k: (float(getattr(metrics, k)[m].mean()) if n else float("nan"))
            for k in ("crps", "pinball", "mae")
        }
    return out


# --------------------------------------------------------------------------- the bootstrap


@dataclass
class CellTable:
    """Sums and counts of a metric difference in every (block, segment) cell of each month.

    Attributes:
        sums: One `(n_blocks, n_segments)` array of summed differences per month.
        counts: One array of row counts per month, the same shape.
        months: The month indices that the arrays follow.
    """

    sums: list[np.ndarray]
    counts: list[np.ndarray]
    months: list[int] = field(default_factory=list)


def cell_table(
    *, rows: Rows, diff: np.ndarray, keep: np.ndarray, months: Sequence[int] | None = None
) -> CellTable:
    """Aggregate a per-row difference into (block, segment) cells within each month.

    Blocks are numbered from all of the month's dates before `keep` is applied, so a block with no
    kept row stays a resampling unit with a zero count.

    Args:
        rows: The scored rows.
        diff: The per-row difference (a minus b).
        keep: The rows of the reading (a subset mask).
        months: The months to include (default: all).

    Returns:
        The table.
    """
    use = sorted(set(rows.month.tolist())) if months is None else list(months)
    sums, counts = [], []
    for m in use:
        sel = rows.month == m
        dates = rows.date[sel]
        block = (dates - dates.min()) // core.BLOCK_DAYS
        seg_ids = np.unique(rows.seg[sel])
        seg_pos = np.searchsorted(seg_ids, rows.seg[sel])
        n_b, n_s = int(block.max()) + 1, len(seg_ids)
        k = keep[sel]
        s = np.zeros((n_b, n_s))
        c = np.zeros((n_b, n_s))
        np.add.at(s, (block[k], seg_pos[k]), diff[sel][k])
        np.add.at(c, (block[k], seg_pos[k]), 1.0)
        sums.append(s)
        counts.append(c)
    return CellTable(sums=sums, counts=counts, months=use)


def kish_factors(*, table: CellTable) -> tuple[float, float]:
    """Return the Kish numbers of the block factor and of the segment factor (over all months)."""
    blocks = np.concatenate([c.sum(axis=1) for c in table.counts])
    segs = np.concatenate([c.sum(axis=0) for c in table.counts])
    return core.kish_number(weights=blocks), core.kish_number(weights=segs)


def two_way_draws(*, table: CellTable, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Return the statistic of `n_resamples` two-way resamples (NaN where no row is drawn)."""
    num = np.zeros(n_resamples)
    den = np.zeros(n_resamples)
    for s, c in zip(table.sums, table.counts, strict=True):
        n_b, n_s = s.shape
        a = rng.multinomial(n_b, np.full(n_b, 1.0 / n_b), size=n_resamples).astype(np.float64)
        b = rng.multinomial(n_s, np.full(n_s, 1.0 / n_s), size=n_resamples).astype(np.float64)
        num += np.einsum("rb,bs,rs->r", a, s, b)
        den += np.einsum("rb,bs,rs->r", a, c, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


@dataclass(frozen=True)
class ContrastReading:
    """One contrast: point estimate and widened interval, in points.

    Attributes:
        reading: The point and the interval.
        raw_lower: The lower end before widening.
        raw_upper: The upper end before widening.
        kish_blocks: The Kish number of the block factor.
        kish_segments: The Kish number of the segment factor.
        widening: The widening ratio applied (1 when none).
        n_rows: The rows of the reading.
        n_resamples: The resamples drawn.
    """

    reading: core.Reading
    raw_lower: float
    raw_upper: float
    kish_blocks: float
    kish_segments: float
    widening: float
    n_rows: int
    n_resamples: int


def read_contrast(
    *, table: CellTable, n_resamples: int = core.N_RESAMPLES, seed: int = core.GENERATOR_SEED,
    level: float = core.LEVEL,
) -> ContrastReading:  # fmt: skip
    """Return the point estimate and widened interval of a contrast.

    The generator is built inside the call, so a reading never depends on what was read before it.

    Args:
        table: The cell table of the difference.
        n_resamples: The number of resamples.
        seed: The generator seed.
        level: The two-sided level.

    Returns:
        The reading with its Kish numbers.
    """
    total = sum(float(c.sum()) for c in table.counts)
    kb, ks = kish_factors(table=table)
    if total <= 0:
        nan = float("nan")
        return ContrastReading(core.Reading(nan, nan, nan), nan, nan, kb, ks, 1.0, 0, n_resamples)
    point = sum(float(s.sum()) for s in table.sums) / total
    draws = two_way_draws(table=table, n_resamples=n_resamples, rng=np.random.default_rng(seed))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(draws, [alpha, 1.0 - alpha])
    ratio = 1.0
    if min(kb, ks) < core.MIN_KISH_FACTOR:
        ratio = max(1.0, core.widening_ratio(kish=min(kb, ks), level=level))
    return ContrastReading(
        reading=core.Reading(point, point - ratio * (point - lo), point + ratio * (hi - point)),
        raw_lower=float(lo), raw_upper=float(hi), kish_blocks=kb, kish_segments=ks,
        widening=ratio, n_rows=int(total), n_resamples=n_resamples,
    )  # fmt: skip


def contrast(
    *, rows: Rows, a: np.ndarray, b: np.ndarray, keep: np.ndarray,
    months: Sequence[int] | None = None, n_resamples: int = core.N_RESAMPLES,
) -> ContrastReading:  # fmt: skip
    """Return the reading of `a - b` (per-row metrics) on the rows of `keep`."""
    return read_contrast(
        table=cell_table(rows=rows, diff=a - b, keep=keep, months=months),
        n_resamples=n_resamples,
    )


def bsp_months(*, rows: Rows) -> list[int]:
    """Return the month indices that carry real BSP steps."""
    return sorted({int(m) for m in rows.month[rows.has_bsp]})


# --------------------------------------------------------------------------- bars and tables


def sign_calibration(*, rows: Rows, p_neg: np.ndarray) -> dict[str, float | bool | list[float]]:
    """Return the sign calibration bar on rows outside the dead band.

    Args:
        rows: The scored rows.
        p_neg: Arm D's seed-mean P(X < 0).

    Returns:
        The reliability term, the largest gap of a bin of at least 200 rows, the bins, and `pass`.
    """
    keep = np.abs(rows.x) >= core.DEAD_BAND_POINTS
    rel = dist.reliability_term(prob=p_neg[keep], observed=rows.x[keep] < 0)
    ok = bool(
        rel["reliability"] <= core.RELIABILITY_BAR and rel["max_gap"] <= core.BIN_GAP_BAR  # ty: ignore[unsupported-operator]
    )
    return {**rel, "pass": ok}  # ty: ignore[invalid-return-type]


def lobe_coverage_bar(*, covers: np.ndarray) -> dict[str, float | bool]:
    """Return the lobe coverage bar: at least 85% of half-hours covered."""
    share = float(np.mean(covers))
    return {"coverage": share, "pass": share >= core.LOBE_COVERAGE_BAR}


def false_sign_share(*, p_neg: np.ndarray) -> float:
    """Return the share of half-hours with P(X < 0) above 0.5."""
    return float(np.mean(p_neg > 0.5))


def harm_check(
    *, d: RowMetrics, p0f_metrics: RowMetrics, p0_metrics: RowMetrics, keep: np.ndarray,
    real: bool,
) -> dict[str, float | bool]:  # fmt: skip
    """Return H1 (synthetic import rows) or H2 (real rows): arm D against P0f and P0.

    Args:
        d: Arm D's metrics.
        p0f_metrics: P0f's metrics.
        p0_metrics: P0's metrics.
        keep: The import rows of the reading.
        real: True for H2, which adds the false-sign share limit.

    Returns:
        The two excesses, the false-sign share (H2), and `pass`.
    """
    crps_excess = float((d.crps - p0f_metrics.crps)[keep].mean())
    mae_excess = float((d.mae - p0_metrics.mae)[keep].mean())
    share = false_sign_share(p_neg=d.p_neg[keep])
    ok = core.harm_passes(
        crps_excess=crps_excess, mae_excess=mae_excess, false_sign_share=share if real else None
    )
    return {
        "crps_excess": crps_excess, "mae_excess": mae_excess, "false_sign_share": share, "pass": ok,
    }  # fmt: skip


def reachability(*, rows: Rows) -> dict[str, float | bool]:
    """Return the reverse-flow counts outside the dead band, in all rows and in BSP months."""
    rev = rows.x < -core.DEAD_BAND_POINTS
    n_all, n_bsp = int(rev.sum()), int((rev & rows.has_bsp).sum())
    return {
        "n_all": n_all, "n_bsp": n_bsp,
        "pass": core.reachability_ok(n_reverse_all=n_all, n_reverse_bsp=n_bsp),
    }  # fmt: skip


def bank_reverse_counts(*, bank: banks.Bank) -> tuple[int, int]:
    """Return the reverse-flow half-hours outside the dead band in a bank, all and BSP months."""
    return banks.reverse_counts(bank=bank)

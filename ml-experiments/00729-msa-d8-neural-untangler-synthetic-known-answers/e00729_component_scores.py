# ruff: noqa: E501
"""Experiment 729: the solar head of arm G1 and the engine head of arm G2 against the true series.

**The script reads stored predictions and the stored banks, and nothing else.** It fits nothing, it
touches no real file, and it uses synthetic days only. Each head is a Gaussian (a mean and a
standard deviation per half-hour, stored as `aux_mu` and `aux_s`); the 90% interval is the mean plus
and minus 1.645 standard deviations, with its lower end clipped at zero because the true output is
never negative.

**The table compares each head with two predictors that need no network.** The all-zero predictor
has the mean absolute error of the true output itself. The climatology predictor is the mean true
output of each (season, half-hour) cell of the first training bank, applied to the test rows. All
errors are in points of D8's capacity. The table reports the mean over seeds 0 to 2 with the
smallest and largest seed beside it.

**Published figures show synthetic engine days and binned aggregates; solar days built on the
metered farms (source S1) stay private.** A solar day of source S1 is a real farm's normalised
output scaled by a drawn capacity, so its shape is a real generator's series. Those days go to
`PRIVATE/components`. A day of source S2 (the PV model on CAMS) may be published.

**Every day is picked by a fixed rule.** Each rule keeps the complete days (48 scored half-hours)
that meet a condition and takes the day with the median head error, or the largest for the
worst-case rules. A tie goes to the lowest (segment, day). No rule draws a random number.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import altair as alt
import e00729_core as core
import e00729_event_figures as ev
import e00729_figures as figures
import e00729_real as real
import e00729_sim as sim
import numpy as np
import polars as pl

Z90: Final = 1.6448536269514722
"""The standard-normal quantile of the 95% point, so that mean +/- `Z90` sd is a 90% interval."""
HEADS: Final = {"solar": "G1", "engine": "G2"}
"""The head and the arm that carries it."""
MIN_BIN_HALF_HOURS: Final = 100
"""A binned plot keeps a bin of true output only when at least this many half-hours fall in it."""
BIN_WIDTH_POINTS: Final = 10.0
MIN_GROUP_DAYS: Final = 5
"""A (segment, half-hour) group enters the CAMS check only with this many days of CAMS data."""
SEASONS: Final = ev.SEASONS
WORLD_NAMES: Final = core.WORLD_TYPES
TRUTH_COLOUR: Final = "#000000"
HEAD_COLOUR: Final = {"solar": figures.GREEN, "engine": figures.BLUE}
PANEL_WIDTH_PX: Final = 285
PANEL_HEIGHT_PX: Final = 110
MIN_PRODUCING_HALF_HOURS: Final = 20


# --------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class ComponentRows:
    """The scored half-hours of a bank, in the order of `score.rows_from_segments`.

    Attributes:
        seg: The segment number.
        day: The dense day index inside the segment.
        date: The calendar day, `datetime64[D]`.
        hh: The UTC half-hour, 0 to 47.
        world: The world as an index into `core.WORLD_TYPES`.
        source: 0 none, 1 S1, 2 S2.
        solar: The true solar output in points.
        engine: The true engine output in points.
        csi: The CAMS clear-sky index (NaN where missing).
        night: True where the sun is down at D8.
        fallback: True on a day that left its drawn solar source because no metered farm had a
            satellite sky (`Segment.s1_fallback`); the day holds zero solar output.
    """

    seg: np.ndarray
    day: np.ndarray
    date: np.ndarray
    hh: np.ndarray
    world: np.ndarray
    source: np.ndarray
    solar: np.ndarray
    engine: np.ndarray
    csi: np.ndarray
    night: np.ndarray
    fallback: np.ndarray

    def __len__(self) -> int:
        """Return the number of half-hours."""
        return len(self.seg)

    def month(self) -> np.ndarray:
        """Return the calendar month, 1 to 12, of each half-hour."""
        return self.date.astype("datetime64[M]").astype(int) % 12 + 1


SOURCE_CODES: Final = {"none": 0, "S1": 1, "S2": 2}


def segment_rows(*, segments: Sequence[sim.Segment]) -> ComponentRows:
    """Build the rows of a list of segments: the valid half-hours of the base days, in scoring order."""
    cols: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "seg",
            "day",
            "date",
            "hh",
            "world",
            "source",
            "solar",
            "engine",
            "csi",
            "night",
            "fallback",
        )
    }
    for k, s in enumerate(segments):
        for d in np.flatnonzero(s.cal_index >= 0):
            hh = np.flatnonzero(s.valid[d] & np.isfinite(s.x[d]))
            n = len(hh)
            if n == 0:
                continue
            cols["seg"].append(np.full(n, k))
            cols["day"].append(np.full(n, d))
            cols["date"].append(np.full(n, s.dates[d].astype("datetime64[D]")))
            cols["hh"].append(hh)
            cols["world"].append(np.full(n, core.WORLD_TYPES.index(s.world)))
            cols["source"].append(np.full(n, SOURCE_CODES[s.source]))
            cols["solar"].append(s.solar[d][hh])
            cols["engine"].append(s.engine[d][hh])
            cols["csi"].append(s.csi[d][hh])
            cols["night"].append(s.cosz[d][hh] <= 0.0)
            cols["fallback"].append(np.full(n, bool(s.s1_fallback[d])))
    return ComponentRows(**{k: np.concatenate(v) for k, v in cols.items()})


# --------------------------------------------------------------------------- metrics


def interval(*, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 90% interval of a Gaussian head, its lower end clipped at zero."""
    return np.maximum(mu - Z90 * sd, 0.0), mu + Z90 * sd


def correlation(*, a: np.ndarray, b: np.ndarray) -> float:
    """Return the Pearson correlation of two arrays (NaN when either is constant)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mae(*, pred: np.ndarray, truth: np.ndarray) -> float:
    """Return the mean absolute error (NaN with no half-hours)."""
    return float(np.mean(np.abs(pred - truth))) if len(truth) else float("nan")


def climatology_predictor(*, train: ComponentRows, test: ComponentRows, column: str) -> np.ndarray:
    """Return the mean true output of each (season, half-hour) cell of `train`, taken at the `test` rows."""
    out = np.zeros(len(test))
    train_cell = (train.month() % 12 // 3) * 48 + train.hh
    test_cell = (test.month() % 12 // 3) * 48 + test.hh
    values = getattr(train, column)
    sums = np.bincount(train_cell, weights=values, minlength=4 * 48)
    counts = np.bincount(train_cell, minlength=4 * 48)
    means = np.divide(sums, counts, out=np.zeros(4 * 48), where=counts > 0)
    out[:] = means[test_cell]
    return out


def monotone_share(*, rows: ComponentRows, values: np.ndarray) -> float:
    """Return the share of (segment, half-hour) groups in which `values` rise with the CAMS index.

    A group is the daytime half-hours of one segment at one half-hour of the day, across its days
    that have CAMS data. It counts when the Spearman rank correlation of `values` with the clear-sky
    index is positive. Groups with fewer than `MIN_GROUP_DAYS` days, or with constant `values`, are
    left out. The result is descriptive; no bar applies.
    """
    key = rows.seg * 48 + rows.hh
    ok = ~rows.night & np.isfinite(rows.csi)
    positive = total = 0
    for g in np.unique(key[ok]):
        sel = ok & (key == g)
        if sel.sum() < MIN_GROUP_DAYS or np.ptp(values[sel]) == 0 or np.ptp(rows.csi[sel]) == 0:
            continue
        rho = correlation(a=ranks(x=values[sel]), b=ranks(x=rows.csi[sel]))
        total += 1
        positive += bool(rho > 0)
    return positive / total if total else float("nan")


def ranks(*, x: np.ndarray) -> np.ndarray:
    """Return the average ranks of an array (ties share the mean rank)."""
    return pl.Series(x).rank("average").to_numpy()


def head_metrics(
    *, head: str, rows: ComponentRows, mu: np.ndarray, sd: np.ndarray
) -> dict[str, float]:
    """Return the scalar metrics of one head at one seed (all in points of capacity).

    The coverage of the 90% interval is reported over half-hours with true output only, because
    the interval's lower end is clipped at zero and every zero-truth half-hour lies inside it.
    The source-S1 error leaves out the days that fell back to no solar output.

    Args:
        head: "solar" or "engine".
        rows: The test rows.
        mu: The head's mean at each row.
        sd: The head's standard deviation at each row.

    Returns:
        The metrics by name.
    """
    truth = getattr(rows, head)
    lo, hi = interval(mu=mu, sd=sd)
    nonzero = truth > 0
    out = {
        "mae_all": mae(pred=mu, truth=truth),
        "mae_nonzero": mae(pred=mu[nonzero], truth=truth[nonzero]),
        "correlation": correlation(a=mu, b=truth),
        "coverage_90_nonzero": float(np.mean(((truth >= lo) & (truth <= hi))[nonzero])) if nonzero.any() else float("nan"),
    }  # fmt: skip
    for w, name in enumerate(WORLD_NAMES):
        sel = rows.world == w
        out[f"mae_world_{name}"] = mae(pred=mu[sel], truth=truth[sel])
    if head == "engine":
        out["f1_running"] = real.f1_score(truth=nonzero, flag=mu > core.K1_RUNNING_POINTS)
    else:
        out["night_mean"] = float(np.mean(mu[rows.night]))
        out["night_max"] = float(np.max(mu[rows.night]))
        out["monotone_share"] = monotone_share(rows=rows, values=mu)
        out["monotone_share_truth"] = monotone_share(rows=rows, values=truth)
        for src, code in (("s1", 1), ("s2", 2)):
            sel = (rows.source == code) & ~rows.fallback
            out[f"mae_source_{src}"] = mae(pred=mu[sel], truth=truth[sel])
        s2 = rows.source == 2
        out["fallback_share_s2"] = float(np.mean(rows.fallback[s2])) if s2.any() else float("nan")
    return out


def baseline_metrics(*, head: str, rows: ComponentRows, train: ComponentRows) -> dict[str, float]:
    """Return the all-zero and the climatology errors of one head's true series (no network)."""
    truth = getattr(rows, head)
    clim = climatology_predictor(train=train, test=rows, column=head)
    nonzero = truth > 0
    return {
        "mae_all_zero": mae(pred=np.zeros(len(truth)), truth=truth),
        "mae_nonzero_zero": mae(pred=np.zeros(int(nonzero.sum())), truth=truth[nonzero]),
        "mae_all_climatology": mae(pred=clim, truth=truth),
        "mae_nonzero_climatology": mae(pred=clim[nonzero], truth=truth[nonzero]),
    }


def seed_summary(*, per_seed: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Return the mean, the smallest, and the largest value of each metric over the seeds."""
    keys = per_seed[0].keys()
    out = {}
    for k in keys:
        v = np.array([s[k] for s in per_seed], dtype=float)
        out[k] = {
            "mean": float(np.nanmean(v)),
            "min": float(np.nanmin(v)),
            "max": float(np.nanmax(v)),
        }
    return out


TABLE_ROWS: Final = (
    ("mae_all", "Head, all half-hours", "mae_all_zero", "mae_all_climatology"),
    ("mae_nonzero", "Head, half-hours with output", "mae_nonzero_zero", "mae_nonzero_climatology"),
)


def markdown_table(*, summaries: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str:
    """Return the component table as markdown (mean over seeds, with the smallest and largest seed)."""

    def cell(s: Mapping[str, float]) -> str:
        return f"{s['mean']:.3f} ({s['min']:.3f} to {s['max']:.3f})"

    lines = ["| Measure | Solar head (arm G1) | Engine head (arm G2) |", "|---|---|---|"]

    def row(label: str, key: str) -> None:
        cells = [
            cell(summaries[h][key]) if key in summaries[h] else "n/a" for h in ("solar", "engine")
        ]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")

    for key, label in (
        ("mae_all", "MAE, all half-hours (points)"),
        ("mae_all_zero", "MAE of the all-zero predictor, all half-hours"),
        ("mae_all_climatology", "MAE of the climatology predictor, all half-hours"),
        ("mae_nonzero", "MAE, half-hours with true output (points)"),
        ("mae_nonzero_zero", "MAE of the all-zero predictor, half-hours with true output"),
        (
            "mae_nonzero_climatology",
            "MAE of the climatology predictor, half-hours with true output",
        ),
        *((f"mae_world_{w}", f"MAE, {w} worlds (points)") for w in WORLD_NAMES),
        ("correlation", "Correlation of head mean with true output"),
        ("coverage_90_nonzero", "Coverage of the 90% interval, half-hours with true output"),
        ("f1_running", "F1 of the running flag (head mean above 10 points)"),
        ("night_mean", "Mean head output at night (points)"),
        ("night_max", "Largest head output at night (points)"),
        (
            "monotone_share",
            "Share of (segment, half-hour) groups where the head rises with the CAMS index",
        ),
        ("monotone_share_truth", "The same share for the true output"),
        ("mae_source_s1", "MAE, source S1 half-hours (points)"),
        ("mae_source_s2", "MAE, source S2 half-hours (points)"),
        (
            "fallback_share_s2",
            "Share of source S2 half-hours on days with no farm sky (zero output)",
        ),
    ):
        row(label, key)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- days


@dataclass(frozen=True)
class DayPick:
    """One picked day.

    Attributes:
        label: The rule's name.
        rows: The indices of the day's 48 rows.
        date: The calendar day (private figures only).
    """

    label: str
    rows: np.ndarray
    date: np.datetime64


def complete_days(*, rows: ComponentRows) -> list[np.ndarray]:
    """Return the row indices of every day that has all 48 half-hours, ordered by (segment, day)."""
    key = rows.seg * 1000 + rows.day
    order = np.argsort(key, kind="stable")
    out = []
    start = 0
    sorted_key = key[order]
    for end in range(1, len(order) + 1):
        if end == len(order) or sorted_key[end] != sorted_key[start]:
            idx = order[start:end]
            if len(idx) == 48:
                out.append(idx[np.argsort(rows.hh[idx])])
            start = end
    return out


def pick_by_key(
    *, days: Sequence[np.ndarray], keep: Callable[[np.ndarray], bool], key: Callable[[np.ndarray], float],
    mode: str,
) -> np.ndarray | None:  # fmt: skip
    """Return the kept day with the median ("median"), largest ("largest"), or smallest key; None if none.

    Days arrive ordered by (segment, day), so a tie keeps the earliest. For the median of an even
    number of days, the lower middle day is taken.
    """
    kept = [(key(d), i) for i, d in enumerate(days) if keep(d)]
    if not kept:
        return None
    kept.sort(key=lambda t: (t[0], t[1]))
    if mode == "median":
        return days[kept[(len(kept) - 1) // 2][1]]
    if mode == "largest":
        best = max(v for v, _ in kept)
        return days[next(i for v, i in kept if v == best)]
    return days[kept[0][1]]


def day_error(*, truth: np.ndarray, mu: np.ndarray) -> Callable[[np.ndarray], float]:
    """Return a key function: the mean absolute error of the head on a day."""
    return lambda d: float(np.mean(np.abs(mu[d] - truth[d])))


RULES: Final = {
    "solar_median": "the day with solar output (at least 20 producing half-hours) at the median head error",
    "solar_largest": "the day with solar output (at least 20 producing half-hours) at the largest head error",
    "solar_clear": "the day with solar output (at least 20 producing half-hours) at the highest mean CAMS clear-sky index",
    "solar_cloudy": "the day with solar output (at least 20 producing half-hours) at the lowest mean CAMS clear-sky index",
    "engine_evening": "an engine run centred between 17:00 and 23:00 UTC (a peak above 10 points), at the median head error",
    "engine_morning": "an engine run centred between 06:00 and 12:00 UTC (a peak above 10 points), at the median head error",
    "engine_and_solar": "a day of a world with both engine and solar output, an engine peak above 10 points, at the median head error",
    "engine_no_run": "a day of an engine or both world with no engine output, at the largest head peak",
}


def centroid_hh(*, day: np.ndarray) -> float:
    """Return the output-weighted mean half-hour of a day (NaN with no output)."""
    total = float(day.sum())
    return float((np.arange(48) * day).sum() / total) if total > 0 else float("nan")


def pick_days(
    *, head: str, rows: ComponentRows, mu: np.ndarray, source: int | None = None
) -> list[DayPick]:  # fmt: skip
    """Pick the four days of a head by the fixed rules in `RULES`.

    Args:
        head: "solar" or "engine".
        rows: The test rows.
        mu: The head's mean at each row (seed 0).
        source: For the solar head, only days of this source code (1 for S1, 2 for S2).

    Returns:
        The days that exist, in the order of the rules.
    """
    days = complete_days(rows=rows)
    truth = getattr(rows, head)
    err = day_error(truth=truth, mu=mu)
    if head == "solar":

        def producing(d: np.ndarray) -> bool:
            return (truth[d] > 0).sum() >= MIN_PRODUCING_HALF_HOURS and (
                source is None or rows.source[d[0]] == source
            )

        def csi_mean(d: np.ndarray) -> float:
            v = rows.csi[d][truth[d] > 0]
            return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

        usable = [d for d in days if producing(d) and np.isfinite(csi_mean(d))]
        rules = (
            ("solar_median", lambda d: True, err, "median"),
            ("solar_largest", lambda d: True, err, "largest"),
            ("solar_clear", lambda d: True, csi_mean, "largest"),
            ("solar_cloudy", lambda d: True, csi_mean, "smallest"),
        )
        picks = [
            (name, pick_by_key(days=usable, keep=keep, key=key, mode=mode))
            for name, keep, key, mode in rules
        ]
    else:
        engine_world = {core.WORLD_TYPES.index("engine"), core.WORLD_TYPES.index("both")}
        both = core.WORLD_TYPES.index("both")

        def runs(lo: float, hi: float) -> Callable[[np.ndarray], bool]:
            return lambda d: (
                truth[d].max() > core.K1_RUNNING_POINTS and lo <= centroid_hh(day=truth[d]) < hi
            )

        picks = [
            ("engine_evening", pick_by_key(days=days, keep=runs(34, 46), key=err, mode="median")),
            ("engine_morning", pick_by_key(days=days, keep=runs(12, 24), key=err, mode="median")),
            ("engine_and_solar", pick_by_key(days=days, keep=lambda d: rows.world[d[0]] == both and truth[d].max() > core.K1_RUNNING_POINTS, key=err, mode="median")),
            ("engine_no_run", pick_by_key(days=days, keep=lambda d: rows.world[d[0]] in engine_world and truth[d].max() == 0, key=lambda d: float(mu[d].max()), mode="largest")),
        ]  # fmt: skip
    return [DayPick(name, d, rows.date[d[0]]) for name, d in picks if d is not None]


# --------------------------------------------------------------------------- charts


def day_panel(
    *, pick: DayPick, rows: ComponentRows, head: str, mu: np.ndarray, sd: np.ndarray, show_y: bool,
    y_max: float, title: str,
) -> alt.LayerChart:  # fmt: skip
    """Return one panel: the true output, the head mean, and the 90% band over one day."""
    d = pick.rows
    lo, hi = interval(mu=mu[d], sd=sd[d])
    frame = pl.DataFrame(
        {
            "hour": rows.hh[d] / 2.0,
            "truth": getattr(rows, head)[d],
            "mean": mu[d],
            "lo": lo,
            "hi": hi,
        }
    )
    colour = HEAD_COLOUR[head]
    x = ev.hours_axis(n_days=1)
    scale = alt.Scale(domain=[0, y_max], nice=False)
    ytitle = "Points of capacity" if show_y else None
    band = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_area(color=colour, opacity=0.25, aria=False)
        .encode(x=x, y=alt.Y("lo:Q", title=ytitle, scale=scale), y2="hi:Q")
    )  # ty: ignore[unresolved-attribute]
    mean = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(color=colour, strokeWidth=1.6, aria=False)
        .encode(x=x, y=alt.Y("mean:Q", scale=scale))
    )  # ty: ignore[unresolved-attribute]
    truth = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(color=TRUTH_COLOUR, strokeWidth=1.2, strokeDash=[4, 2], aria=False)
        .encode(x=x, y=alt.Y("truth:Q", scale=scale))
    )  # ty: ignore[unresolved-attribute]
    return (band + mean + truth).properties(
        width=PANEL_WIDTH_PX,
        height=PANEL_HEIGHT_PX,
        title=alt.TitleParams(title, anchor="start", fontSize=11),
    )


def days_chart(
    *, picks: Sequence[DayPick], rows: ComponentRows, head: str, mu: np.ndarray, sd: np.ndarray,
    title: str, subtitle: Sequence[str], dated: bool,
) -> alt.VConcatChart:  # fmt: skip
    """Return up to four day panels in two columns on a common y axis, with the rules in the subtitle."""
    top = max(
        float(
            max(getattr(rows, head)[p.rows].max(), interval(mu=mu[p.rows], sd=sd[p.rows])[1].max())
        )
        for p in picks
    )
    panels = [
        day_panel(
            pick=p,
            rows=rows,
            head=head,
            mu=mu,
            sd=sd,
            show_y=i % 2 == 0,
            y_max=top * 1.05,
            title=f"{p.label.replace('_', ' ')}{' ' + str(p.date) if dated else ''}",
        )
        for i, p in enumerate(picks)
    ]
    rows_of = [alt.hconcat(*panels[i : i + 2]) for i in range(0, len(panels), 2)]
    return alt.vconcat(*rows_of).properties(
        title=alt.TitleParams(title, subtitle=list(subtitle), anchor="start")
    )


def bin_summary(*, truth: np.ndarray, mu: np.ndarray) -> pl.DataFrame:
    """Return, for each bin of true output, the mean true output and the head mean with its spread.

    The spread is the 10th to 90th percentile of the head mean.

    The first bin holds the half-hours of zero output. Bins with fewer than `MIN_BIN_HALF_HOURS`
    half-hours are left out.
    """
    idx = np.where(truth <= 0, 0, 1 + np.floor(truth / BIN_WIDTH_POINTS)).astype(int)
    rows = []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() < MIN_BIN_HALF_HOURS:
            continue
        rows.append(
            {
                "truth": float(truth[sel].mean()), "mean": float(mu[sel].mean()),
                "p10": float(np.percentile(mu[sel], 10)), "p90": float(np.percentile(mu[sel], 90)),
            }
        )  # fmt: skip
    return pl.DataFrame(
        rows, schema={"truth": pl.Float64, "mean": pl.Float64, "p10": pl.Float64, "p90": pl.Float64}
    )


def binned_chart(*, rows: ComponentRows, head: str, mu: np.ndarray, title: str) -> alt.LayerChart:
    """Return the head mean against the true output in bins, with the 10th to 90th percentile band and the diagonal."""
    frame = bin_summary(truth=getattr(rows, head), mu=mu)
    top = float(max(float(frame["truth"].max()), float(frame["p90"].max())))  # ty: ignore[invalid-argument-type]
    colour = HEAD_COLOUR[head]
    x = alt.X(
        "truth:Q",
        title="True output (points of capacity, mean of the bin)",
        scale=alt.Scale(domain=[0, top], nice=False),
    )
    scale = alt.Scale(domain=[0, top], nice=False)
    band = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_area(color=colour, opacity=0.25, aria=False)
        .encode(
            x=x, y=alt.Y("p10:Q", title="Head mean (points of capacity)", scale=scale), y2="p90:Q"
        )
    )  # ty: ignore[unresolved-attribute]
    line = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(color=colour, strokeWidth=1.8, point=True, aria=False)
        .encode(x=x, y=alt.Y("mean:Q", scale=scale))
    )  # ty: ignore[unresolved-attribute]
    diag = (
        alt.Chart(pl.DataFrame({"truth": [0.0, top], "mean": [0.0, top]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color=TRUTH_COLOUR, strokeDash=[4, 2], aria=False)
        .encode(x=x, y=alt.Y("mean:Q", scale=scale))
    )  # ty: ignore[unresolved-attribute]
    return (band + line + diag).properties(
        width=figures.WIDTH_PX - 200, height=figures.WIDTH_PX - 200,
        title=alt.TitleParams(title, subtitle=ev.wrap_text("Line: mean head output in bins of true output. Band: 10th to 90th percentile. Dashed: head equals truth. Seed 0.", width=70), anchor="start"),
    )  # fmt: skip


# --------------------------------------------------------------------------- the stage


def write_component_outputs(
    *, test_segments: Sequence[sim.Segment], train_segments: Sequence[sim.Segment],
    predictions: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]], out_dir: Path,
    private_dir: Path, scores_dir: Path, log: Callable[[str], None],
) -> list[Path]:  # fmt: skip
    """Write the component table, its data, the published figures, and the private solar-day figures.

    Args:
        test_segments: The segments of the stored test bank.
        train_segments: The segments of the first stored training bank (the climatology's fit).
        predictions: For each head ("solar", "engine"), the stored test predictions of its arm by seed.
        out_dir: The folder of the published figures.
        private_dir: The private folder for solar days of source S1.
        scores_dir: The folder of the table and its data.
        log: Receives one line per file written.

    Returns:
        The published figures (run `svgo` on each before committing).

    Raises:
        ValueError: When a stored prediction does not have one value per test row.
    """
    rows = segment_rows(segments=test_segments)
    train = segment_rows(segments=train_segments)
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for head, by_seed in predictions.items():
        per_seed = []
        for seed, pred in sorted(by_seed.items()):
            if len(pred["aux_mu"]) != len(rows):
                msg = f"the stored prediction of {HEADS[head]} at seed {seed} does not match the test rows"
                raise ValueError(msg)
            per_seed.append(
                head_metrics(head=head, rows=rows, mu=pred["aux_mu"], sd=pred["aux_s"])
                | baseline_metrics(head=head, rows=rows, train=train)
            )
        summaries[head] = seed_summary(per_seed=per_seed)
    scores_dir.mkdir(parents=True, exist_ok=True)
    (scores_dir / "components.md").write_text(markdown_table(summaries=summaries))
    pl.DataFrame(
        [{"head": h, "metric": k, **v} for h, ms in summaries.items() for k, v in ms.items()]
    ).write_csv(scores_dir / "components.csv")
    log("component table written")
    written: list[Path] = []
    private_dir.mkdir(parents=True, exist_ok=True)
    for head in predictions:
        pred0 = predictions[head][min(predictions[head])]
        mu, sd = pred0["aux_mu"], pred0["aux_s"]
        charts: dict[str, alt.TopLevelMixin] = {}
        picks = pick_days(head=head, rows=rows, mu=mu, source=None if head == "engine" else 2)
        if picks:
            charts[f"components_{head}_days.svg"] = (
                days_chart(picks=picks, rows=rows, head=head, mu=mu, sd=sd, title=f"The {head} head against the true {head} output on four days", subtitle=ev.wrap_text(" ".join(f"{p.label.replace('_', ' ')}: {RULES[p.label]}." for p in picks), width=95), dated=False)
            )  # fmt: skip
        charts[f"components_{head}_binned.svg"] = binned_chart(
            rows=rows, head=head, mu=mu, title=f"The {head} head against the true {head} output"
        )
        for name, chart in charts.items():
            figures.save_svg(chart=chart, path=out_dir / name)
            written.append(out_dir / name)
            log(f"{name}: written")
    private_picks = pick_days(
        head="solar",
        rows=rows,
        mu=predictions["solar"][min(predictions["solar"])]["aux_mu"],
        source=1,
    )
    if private_picks:
        p0 = predictions["solar"][min(predictions["solar"])]
        figures.save_svg(
            chart=days_chart(picks=private_picks, rows=rows, head="solar", mu=p0["aux_mu"], sd=p0["aux_s"], title="The solar head on four days of source S1 (private)", subtitle=ev.wrap_text(" ".join(f"{p.label.replace('_', ' ')}: {RULES[p.label]}." for p in private_picks), width=95), dated=True),
            path=private_dir / "components_solar_days_s1.svg",
        )  # fmt: skip
    log("private figures written to PRIVATE/components")
    return written

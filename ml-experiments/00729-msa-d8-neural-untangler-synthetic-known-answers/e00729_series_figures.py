# ruff: noqa: E501
"""Experiment 729: time-series figures of stored predictions on single 48-hour windows.

Every figure reads stored predictions only (`predictions/<arm>_<arch>_s0/test.npz`, and
`real_r1.npz` for the real days) and the stored test bank. Nothing is fitted or re-scored. Each
figure shows one window: the middle 24 hours are scored, and the 12 hours either side are the
context the networks read. The top row holds the true signed power (synthetic days only), the folded
reading, and each arm's median with its 5% to 95% band; the row below holds P(X<0) with the 0.5
line; the BSP figure has a third row of BSP steps.

**Every day is picked by a fixed rule, never by eye.** A rule keeps the complete days (48 scored
half-hours) that meet a condition and takes the day with the median of a stated key, or the largest
for the two worst-case figures. A tie goes to the lowest (segment, date). No rule draws random
numbers, so a rerun picks the same days. `RULES` states each rule in words for the captions.

**No figure shows a calendar date, a month, a metered engine, or an inferred engine on a real day.**
The x axis is the hour of the day in UTC. The real-day figures plot D8's reading and each arm's
output on days before the engine was connected.
"""

import itertools
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00729_baselines as base
import e00729_core as core
import e00729_dist as dist
import e00729_figures as figures
import e00729_leakage as leak
import e00729_paths as paths
import e00729_score as score
import numpy as np
import polars as pl

HOURS_BEFORE: Final = 12
"""Hours of context before the scored day, and (as `HOURS_AFTER`) after it."""
HOURS_AFTER: Final = 12
N_CONTEXT: Final = 2 * (HOURS_BEFORE + HOURS_AFTER + 24)
"""Half-hours in a window: 96."""
PLOT_WIDTH_PX: Final = figures.WIDTH_PX - 60
TOP_HEIGHT_PX: Final = 170
STRIP_HEIGHT_PX: Final = 70
WRAP_CHARS: Final = 95
TITLE_WRAP_CHARS: Final = 58
BAND_LEVELS: Final = (0.05, 0.5, 0.95)
LOW_READING_MVA: Final = core.READING_BAND_EDGES_MVA[0]
"""A reading under this level (0.3 MVA, the top of the simulated reactive power) hides the sign."""
MIN_LOW_READING_HALF_HOURS: Final = 4
MIN_OUTPUT_POINTS: Final = 2.0
"""An engine or solar output under this many points does not count as running or shining."""
EVENING_CENTROID_HH: Final = (34, 46)
MORNING_CENTROID_HH: Final = (12, 24)
MIDDAY_CENTROID_HH: Final = (20, 28)
SEED: Final = 0
"""Every arm is read at seed 0."""

TRUTH_NAME: Final = "True signed power"
READING_NAME: Final = "Folded reading"
ARM_LABELS: Final = {
    "P0f": "P0f (reading as import)", "A": "A (reading only)", "B": "B (A + sunshine)",
    "C": "C (B + BSP steps)", "D": "D (C + calendar and weather)", "Dq": "D-quantile (11 quantiles)",
    "Dsign": "D-sign (sign only)",
}  # fmt: skip
SERIES_COLOURS: Final = {
    TRUTH_NAME: "#000000", READING_NAME: "#ADADAD",
    ARM_LABELS["P0f"]: "#555555", ARM_LABELS["A"]: figures.SKY, ARM_LABELS["B"]: figures.BLUE,
    ARM_LABELS["C"]: figures.GREEN, ARM_LABELS["D"]: figures.ORANGE,
    ARM_LABELS["Dq"]: figures.PURPLE, ARM_LABELS["Dsign"]: figures.PURPLE,
}  # fmt: skip
CONTEXT_SHADE: Final = "#F2F2F2"


@dataclass(frozen=True)
class Day:
    """One complete scored day of a set of rows, with the facts that the picking rules read.

    Attributes:
        seg: The segment number.
        date: The base day as days since 1970-01-01 (used to find the day, never printed).
        index: The 48 row indices of the day, in half-hour order.
        world: The world type as an index into `core.WORLD_TYPES`.
        has_bsp: The segment carries real BSP steps.
        min_x: The lowest signed truth of the day in points.
        min_r: The lowest reading of the day in points.
        n_low: The half-hours with a reading under 0.3 MVA.
        engine_centroid: The output-weighted mean half-hour of the engine, or NaN with no output.
        engine_max: The largest engine output in points.
        solar_centroid: The output-weighted mean half-hour of the solar output, or NaN.
        solar_max: The largest solar output in points.
    """

    seg: int
    date: int
    index: np.ndarray
    world: int
    has_bsp: bool
    min_x: float
    min_r: float
    n_low: int
    engine_centroid: float
    engine_max: float
    solar_centroid: float
    solar_max: float


@dataclass(frozen=True)
class Forecast:
    """One arm's output on the 48 scored half-hours of a day.

    Attributes:
        lower: The 5% quantile in points.
        median: The median in points.
        upper: The 95% quantile in points.
        p_neg: P(X < 0).
    """

    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    p_neg: np.ndarray


@dataclass(frozen=True)
class WindowData:
    """The 96 half-hours of a window, from 12:00 UTC the day before to 12:00 UTC the day after.

    Attributes:
        truth: The signed truth in points (NaN where unknown, and always NaN for real days).
        reading: The folded reading in points (NaN off base days).
        bsp_step: The half-hour change of the BSP meter in points (NaN where missing).
    """

    truth: np.ndarray
    reading: np.ndarray
    bsp_step: np.ndarray


def centroid(*, hh: np.ndarray, output: np.ndarray) -> float:
    """Return the output-weighted mean half-hour, or NaN when there is no output."""
    total = float(np.sum(output))
    return float(np.sum(hh * output) / total) if total > 0 else float("nan")


def complete_days(*, rows: score.Rows) -> list[Day]:
    """Return the days with all 48 half-hours scored, in (segment, date) order.

    Args:
        rows: The scored rows, grouped by segment and then by day.

    Returns:
        One `Day` for each complete day.
    """
    key = rows.seg.astype(np.int64) * 1_000_000 + rows.date.astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(key)]
    low_level = LOW_READING_MVA / core.CAPACITY_MVA * core.POINTS
    out = []
    for a, b in itertools.pairwise(starts):
        if b - a != 48 or not np.array_equal(rows.hh[a:b], np.arange(48)):
            continue
        idx = np.arange(a, b)
        out.append(
            Day(
                seg=int(rows.seg[a]), date=int(rows.date[a]), index=idx, world=int(rows.world[a]),
                has_bsp=bool(rows.has_bsp[a]), min_x=float(rows.x[idx].min()),
                min_r=float(rows.r[idx].min()), n_low=int((rows.r[idx] < low_level).sum()),
                engine_centroid=centroid(hh=rows.hh[idx], output=rows.engine[idx]),
                engine_max=float(rows.engine[idx].max()),
                solar_centroid=centroid(hh=rows.hh[idx], output=rows.solar[idx]),
                solar_max=float(rows.solar[idx].max()),
            )
        )  # fmt: skip
    return out


def median_by(*, days: Sequence[Day], key: Callable[[Day], float]) -> Day | None:
    """Return the day with the median key (the lower middle for an even count), or None if empty."""
    if not days:
        return None
    order = sorted(days, key=lambda d: (key(d), d.seg, d.date))
    return order[(len(order) - 1) // 2]


def largest_by(*, days: Sequence[Day], key: Callable[[Day], float]) -> Day | None:
    """Return the day with the largest key (the lowest (segment, date) on a tie), or None if empty."""
    if not days:
        return None
    return min(days, key=lambda d: (-key(d), d.seg, d.date))


def in_range(*, value: float, bounds: tuple[int, int]) -> bool:
    """Return whether `bounds[0] <= value < bounds[1]` (False for NaN)."""
    return bounds[0] <= value < bounds[1]


def world_is(*, day: Day, names: Sequence[str]) -> bool:
    """Return whether a day's world type is one of `names`."""
    return core.WORLD_TYPES[day.world] in names


def reverses(*, day: Day) -> bool:
    """Return whether the day has reverse flow outside the dead band."""
    return day.min_x < -core.DEAD_BAND_POINTS


RULES: Final = {
    "evening_engine": "an engine-only day with reverse flow whose engine output is centred between 17:00 and 23:00 UTC; the day with the median depth of reverse flow",
    "morning_engine": "an engine-only day with reverse flow whose engine output is centred between 06:00 and 12:00 UTC; the day with the median depth of reverse flow",
    "solar_midday": "a solar-only day with reverse flow whose solar output is centred between 10:00 and 14:00 UTC; the day with the median depth of reverse flow",
    "engine_and_solar": "an engine-and-solar day with reverse flow in which both outputs reach at least 2 points; the day with the median depth of reverse flow",
    "ambiguous": "a day with at least 4 half-hours of reading under 0.3 MVA; the day with the median number of such half-hours",
    "plain_import": "an all-import day of the no-injection worlds; the day with the median of arm D's largest P(X<0) of the day",
    "largest_harm": "a day of the test bank on which every half-hour is import; the day with the largest mean of arm D's signed CRPS minus P0f's",
    "bsp": "an engine or engine-and-solar day in a BSP month with reverse flow; the day with the median depth of reverse flow",
    "quantile_head": "the day of the evening-engine figure",
    "sign_head": "the day of the ambiguous-reading figure",
    "real_median": "a real day before the engine was connected; the day with the median of arm D's largest P(X<0) of the day",
    "real_largest": "a real day before the engine was connected; the day with the largest of arm D's largest P(X<0) of the day",
}  # fmt: skip
"""Each picking rule in words, for the captions."""


def pick_scenarios(
    *, days: Sequence[Day], d_p_neg: np.ndarray, d_harm: np.ndarray
) -> dict[str, Day | None]:
    """Pick the day of each synthetic figure by its rule.

    Args:
        days: The complete days of the test bank.
        d_p_neg: Arm D's P(X<0) on every row of the bank.
        d_harm: Arm D's signed CRPS minus P0f's on every row.

    Returns:
        The picked day of each name in `RULES` that is not a real-day rule (None where no day
        meets the rule, so the figure is skipped).
    """

    def depth(d: Day) -> float:
        return d.min_x

    def run_centred(*, names: Sequence[str], bounds: tuple[int, int], solar: bool) -> Day | None:
        keep = [
            d for d in days
            if world_is(day=d, names=names) and reverses(day=d)
            and in_range(value=d.solar_centroid if solar else d.engine_centroid, bounds=bounds)
        ]  # fmt: skip
        return median_by(days=keep, key=depth)

    both = [
        d for d in days
        if world_is(day=d, names=("both",)) and reverses(day=d)
        and d.engine_max >= MIN_OUTPUT_POINTS and d.solar_max >= MIN_OUTPUT_POINTS
    ]  # fmt: skip
    bsp = [
        d for d in days
        if d.has_bsp and world_is(day=d, names=("engine", "both")) and reverses(day=d)
    ]  # fmt: skip
    low = [d for d in days if d.n_low >= MIN_LOW_READING_HALF_HOURS]
    none = [d for d in days if world_is(day=d, names=("none",)) and d.min_x >= 0]
    imports = [d for d in days if d.min_x >= 0]
    return {
        "evening_engine": run_centred(names=("engine",), bounds=EVENING_CENTROID_HH, solar=False),
        "morning_engine": run_centred(names=("engine",), bounds=MORNING_CENTROID_HH, solar=False),
        "solar_midday": run_centred(names=("solar",), bounds=MIDDAY_CENTROID_HH, solar=True),
        "engine_and_solar": median_by(days=both, key=depth),
        "ambiguous": median_by(days=low, key=lambda d: d.n_low),
        "plain_import": median_by(days=none, key=lambda d: float(d_p_neg[d.index].max())),
        "largest_harm": largest_by(days=imports, key=lambda d: float(d_harm[d.index].mean())),
        "bsp": median_by(days=bsp, key=depth),
    }


def pick_real(*, days: Sequence[Day], d_p_neg: np.ndarray) -> dict[str, Day | None]:
    """Pick the two real days: the median and the largest of arm D's largest P(X<0) of the day."""

    def key(d: Day) -> float:
        return float(d_p_neg[d.index].max())

    return {
        "real_median": median_by(days=days, key=key),
        "real_largest": largest_by(days=days, key=key),
    }


def window_data(*, seg: Any, date: int, with_truth: bool) -> WindowData:
    """Return the 96 half-hours of a window around a scored day.

    Args:
        seg: The segment, with dense `dates`, `x`, `r`, and `bsp` arrays of shape `(days, 48)`.
        date: The scored day as days since 1970-01-01.
        with_truth: Whether to return the signed truth (False for real days, where it is only
            the reading).

    Returns:
        The window's truth, reading, and BSP step. A neighbouring day that is not a base day
        gives NaN.
    """
    dates = np.asarray(seg.dates).astype("datetime64[D]")
    day = int(np.flatnonzero(dates == np.datetime64(int(date), "D"))[0])

    def span(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        return np.concatenate([v[day - 1, 24:], v[day], v[day + 1, :24]])

    bsp = span(seg.bsp)
    return WindowData(
        truth=span(seg.x) if with_truth else np.full(N_CONTEXT, np.nan),
        reading=span(seg.r),
        bsp_step=np.r_[np.nan, np.diff(bsp)],
    )


def p0f_prediction(*, r: np.ndarray) -> dict[str, np.ndarray]:
    """Return P0f as a stored-prediction dictionary (a point at the reading, 0.5-point spread)."""
    lobes = base.point_lobes(xhat=base.reading_to_import(r=r), spread=core.FLOOR_POINTS)
    return {
        "p": lobes.p,
        "mu_i": lobes.mu_i,
        "s_i": lobes.s_i,
        "mu_r": lobes.mu_r,
        "s_r": lobes.s_r,
    }


def lobes_of(*, pred: Mapping[str, np.ndarray]) -> dist.Lobes:
    """Return the two-lobe distributions of a stored prediction."""
    return dist.Lobes(pred["p"], pred["mu_i"], pred["s_i"], pred["mu_r"], pred["s_r"])


def forecast_on(*, pred: Mapping[str, np.ndarray], index: np.ndarray) -> Forecast:
    """Summarise a stored prediction on selected rows.

    A quantile prediction is made non-decreasing across its levels, as scoring does, and gives
    P(X<0) by linear interpolation of its 11 quantiles, which is 0 below the 5% quantile and 1
    above the 95% quantile. A two-lobe prediction gives P(X<0) as the mixture's mass below zero,
    which tends to 0.5 as the reading nears zero whatever the stored lobe probability `p`.

    Args:
        pred: The stored arrays (`p`, `mu_i`, ... or `q`).
        index: The row indices.

    Returns:
        The 5%, 50%, and 95% quantiles and P(X<0) of each row.
    """
    if "q" in pred:
        q = np.maximum.accumulate(np.asarray(pred["q"])[index], axis=1)
        levels = np.asarray(core.QUANTILE_LEVELS)
        p = np.array([np.interp(0.0, row, levels, left=0.0, right=1.0) for row in q])
        mid = list(core.QUANTILE_LEVELS).index(0.5)
        return Forecast(lower=q[:, 0], median=q[:, mid], upper=q[:, -1], p_neg=p)
    lobes = lobes_of(pred=pred).subset(rows=index)
    q = dist.mixture_quantiles(lobes=lobes, levels=BAND_LEVELS)
    return Forecast(
        lower=q[:, 0], median=q[:, 1], upper=q[:, 2], p_neg=dist.prob_negative(lobes=lobes)
    )


def prediction_path(*, arm: str, arch: str, name: str) -> Path:
    """Return the stored prediction file of an arm at seed 0."""
    return paths.data_root() / "predictions" / f"{arm}_{arch}_s{SEED}" / f"{name}.npz"


def load_prediction(*, arm: str, arch: str, name: str, n_rows: int) -> dict[str, np.ndarray]:
    """Read one stored prediction file as float64 arrays and check its length.

    Args:
        arm: The arm.
        arch: The architecture.
        name: The stored set ("test" or "real_r1").
        n_rows: The number of scored rows that the file must cover.

    Returns:
        The stored arrays.

    Raises:
        CorrectnessFailureError: When a stored array does not have `n_rows` rows.
    """
    with np.load(prediction_path(arm=arm, arch=arch, name=name)) as f:
        pred = {k: f[k].astype(np.float64) for k in f.files}
    if any(len(v) != n_rows for v in pred.values()):
        msg = f"the stored prediction of arm {arm} ({name}) does not match the scored rows"
        raise leak.CorrectnessFailureError(msg)
    return pred


def wrap(*, text: str, width: int = WRAP_CHARS) -> list[str]:
    """Wrap a title or subtitle to the plot width, as a list of lines."""
    return textwrap.wrap(text, width=width)


def x_encoding() -> alt.X:
    """Return the shared x encoding: hours from the scored day's midnight, labelled as hour of day."""
    return alt.X(
        "hour:Q", title="Hour of day, UTC (the scored day is the unshaded middle part)",
        scale=alt.Scale(domain=[-HOURS_BEFORE, 24 + HOURS_AFTER], nice=False),
        axis=alt.Axis(values=[-12, -6, 0, 6, 12, 18, 24, 30, 36], labelExpr="(datum.value + 24) % 24", grid=False),
    )  # fmt: skip


def context_shade() -> alt.Chart:
    """Return the grey blocks over the 12 hours of context before and after the scored day."""
    blocks = pl.DataFrame({"start": [-HOURS_BEFORE, 24], "end": [0, 24 + HOURS_AFTER]})
    return (
        alt.Chart(blocks)
        .mark_rect(color=CONTEXT_SHADE, aria=False)
        .encode(x=alt.X("start:Q"), x2="end:Q")  # ty: ignore[unresolved-attribute]
    )


def colour_encoding(*, names: Sequence[str], legend: bool) -> alt.Color:
    """Return the shared colour encoding of the named series."""
    return alt.Color(
        "series:N",
        scale=alt.Scale(domain=list(names), range=[SERIES_COLOURS[n] for n in names]),
        legend=alt.Legend(orient="top", columns=2, title=None, labelLimit=360) if legend else None,
    )


def dash_encoding(*, names: Sequence[str]) -> alt.StrokeDash:
    """Return the dash encoding: P0f is dashed, as a backup to its colour, and every other series is solid."""
    return alt.StrokeDash(
        "series:N",
        scale=alt.Scale(
            domain=list(names), range=[[6, 3] if n == ARM_LABELS["P0f"] else [1, 0] for n in names]
        ),
        legend=None,
    )


def window_chart(
    *, title: str, subtitle: str, window: WindowData, forecasts: Mapping[str, Forecast],
    bands: Sequence[str], show_bsp: bool,
) -> alt.VConcatChart:  # fmt: skip
    """Return a window chart: signed power on top, P(X<0) below, and optionally BSP steps.

    Args:
        title: What the figure shows.
        subtitle: Which day, by what rule, with the units and the band.
        window: The 96 half-hours of truth, reading, and BSP step.
        forecasts: Each arm's forecast on the 48 scored half-hours, in drawing order (keys are
            the arm names of `ARM_LABELS`).
        bands: The arms whose 5% to 95% band is drawn.
        show_bsp: Whether to add the BSP-step row.

    Returns:
        The chart.
    """
    hours = -HOURS_BEFORE + np.arange(N_CONTEXT) / 2.0
    scored = np.arange(48) / 2.0
    names = [TRUTH_NAME, READING_NAME, *(ARM_LABELS[a] for a in forecasts)]
    names = [n for n in names if n != TRUTH_NAME or np.isfinite(window.truth).any()]
    parts = [pl.DataFrame({"hour": hours, "series": READING_NAME, "value": window.reading})]
    if np.isfinite(window.truth).any():
        parts.append(pl.DataFrame({"hour": hours, "series": TRUTH_NAME, "value": window.truth}))
    parts += [
        pl.DataFrame({"hour": scored, "series": ARM_LABELS[a], "value": f.median})
        for a, f in forecasts.items()
    ]
    lines = pl.concat(parts).with_columns(pl.col("value").fill_nan(None))
    band_frame = pl.concat(
        [
            pl.DataFrame(
                {
                    "hour": scored,
                    "series": ARM_LABELS[a],
                    "lower": forecasts[a].lower,
                    "upper": forecasts[a].upper,
                }
            )
            for a in bands
        ]
    )
    prob = pl.concat(
        [
            pl.DataFrame({"hour": scored, "series": ARM_LABELS[a], "p": f.p_neg})
            for a, f in forecasts.items()
        ]
    )
    x = x_encoding()
    shared = colour_encoding(names=names, legend=True)
    quiet = shared
    top = (
        context_shade()
        + alt.Chart(band_frame)
        .mark_area(opacity=0.14, aria=False)
        .encode(x=x, y="lower:Q", y2="upper:Q", color=quiet)  # ty: ignore[unresolved-attribute]
        + alt.Chart(lines)
        .mark_line(aria=False, strokeWidth=1.6, invalid=None)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("value:Q", title="Signed power (points of capacity)"),
            color=shared,
            strokeDash=dash_encoding(names=names),
        )
    ).properties(width=PLOT_WIDTH_PX, height=TOP_HEIGHT_PX)
    half = (
        alt.Chart(pl.DataFrame({"y": [0.5]}))
        .mark_rule(strokeDash=[4, 4], color="#555555", aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    middle = (
        context_shade()
        + alt.Chart(prob)
        .mark_line(aria=False, strokeWidth=1.6, invalid=None)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "p:Q",
                title="P(X<0)",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(values=[0, 0.5, 1]),
            ),
            color=quiet,
            strokeDash=dash_encoding(names=names),
        )
        + half
    ).properties(width=PLOT_WIDTH_PX, height=STRIP_HEIGHT_PX)
    panels = [top, middle]
    if show_bsp:
        steps = pl.DataFrame({"hour": hours, "step": window.bsp_step}).filter(
            pl.col("step").is_finite()
        )
        panels.append(
            (
                context_shade()
                + alt.Chart(steps)
                .mark_bar(color="#555555", aria=False, size=3)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x, y=alt.Y("step:Q", title="BSP step (points)")
                )
            ).properties(width=PLOT_WIDTH_PX, height=STRIP_HEIGHT_PX)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            wrap(text=title, width=TITLE_WRAP_CHARS), subtitle=wrap(text=subtitle), anchor="start"
        )
    )


@dataclass(frozen=True)
class FigureSpec:
    """One figure: its file, title, rule, arms, and extra rows.

    Attributes:
        name: The key of the rule in `RULES` and of the picked day.
        file: The SVG file name.
        title: The title.
        arms: The arms drawn, in order.
        bands: The arms with a 5% to 95% band.
        show_bsp: Whether the BSP-step row is drawn.
    """

    name: str
    file: str
    title: str
    arms: tuple[str, ...]
    bands: tuple[str, ...]
    show_bsp: bool = False


SPECS: Final = (
    FigureSpec("evening_engine", "series_01_evening_engine.svg", "An evening engine run that reverses the flow", ("A", "B", "D"), ("D",)),
    FigureSpec("morning_engine", "series_02_morning_engine.svg", "A morning engine run that reverses the flow", ("A", "B", "D"), ("D",)),
    FigureSpec("solar_midday", "series_03_solar_midday.svg", "Solar output alone reverses the flow at midday", ("A", "B", "D"), ("D",)),
    FigureSpec("engine_and_solar", "series_04_engine_and_solar.svg", "Engine and solar output together", ("A", "B", "D"), ("D",)),
    FigureSpec("ambiguous", "series_05_ambiguous.svg", "A day on which the reading comes close to zero and the sign is hard to tell", ("A", "B", "D"), ("D",)),
    FigureSpec("plain_import", "series_06_plain_import.svg", "A plain import day with no injection: the probability of reverse flow that the arms invent", ("P0f", "A", "B", "D"), ("D",)),
    FigureSpec("largest_harm", "series_07_largest_harm.svg", "The import day on which arm D does worst against P0f", ("P0f", "A", "B", "D"), ("D",)),
    FigureSpec("bsp", "series_08_bsp_steps.svg", "A month with real BSP steps: arm C, which reads the steps, beside arm B, which does not", ("B", "C"), ("B", "C"), True),
    FigureSpec("quantile_head", "series_09_quantile_head.svg", "The 11-quantile head beside the two-lobe head on one window", ("D", "Dq"), ("D", "Dq")),
    FigureSpec("sign_head", "series_10_sign_head.svg", "The sign-only head beside the two-lobe head on one window", ("D", "Dsign"), ("D", "Dsign")),
    FigureSpec("real_median", "series_11_real_median.svg", "A real day before the engine was connected, with the typical false alarm of arm D", ("A", "B", "D"), ("D",)),
    FigureSpec("real_largest", "series_12_real_largest.svg", "A real day before the engine was connected, with the largest false alarm of arm D", ("A", "B", "D"), ("D",)),
)  # fmt: skip
"""The 12 figures, in the order of the page."""


def forecasts_for(
    *, spec: FigureSpec, arch: str, name: str, index: np.ndarray, r: np.ndarray
) -> dict[str, Forecast]:
    """Return the forecast of each arm of a figure on one day.

    Args:
        spec: The figure.
        arch: The chosen architecture.
        name: The stored prediction set ("test" or "real_r1").
        index: The day's 48 row indices.
        r: The reading of every row, in points (P0f reads it).

    Returns:
        The forecasts in drawing order.
    """
    out = {}
    for arm in spec.arms:
        pred = (
            p0f_prediction(r=r)
            if arm == "P0f"
            else load_prediction(arm=arm, arch=arch, name=name, n_rows=len(r))
        )
        out[arm] = forecast_on(pred=pred, index=index)
    return out


def caption_rule(*, spec: FigureSpec) -> str:
    """Return the subtitle sentence that states the day's rule."""
    return f"Rule for the day: {RULES[spec.name]}. Arms at seed {SEED}. Band: 5% to 95%."


def write_series_figures(
    *, arch: str, test_segments: Sequence[Any], rows: score.Rows,
    real_segments: Sequence[Any] | None, real_rows: score.Rows | None,
    out_dir: Path, log: Callable[[str], None],
) -> list[Path]:  # fmt: skip
    """Pick the days by rule, draw the figures, and return their paths.

    Args:
        arch: The chosen architecture.
        test_segments: The segments of the test bank, in the order of `rows`.
        rows: The scored rows of the test bank.
        real_segments: The real held-out pseudo-segments of set R1 (None to skip the real figures).
        real_rows: Their scored rows.
        out_dir: The folder to write to.
        log: Receives one line for each figure (the figure's name and the day's rule only).

    Returns:
        The written paths (run `svgo` on each before committing).
    """
    d_pred = load_prediction(arm="D", arch=arch, name="test", n_rows=rows.n)
    d_lobes = lobes_of(pred=d_pred)
    p0f = lobes_of(pred=p0f_prediction(r=rows.r))
    d_p = dist.prob_negative(lobes=d_lobes)
    harm = dist.mixture_crps(lobes=d_lobes, y=rows.x) - dist.mixture_crps(lobes=p0f, y=rows.x)
    days = complete_days(rows=rows)
    picked = pick_scenarios(days=days, d_p_neg=d_p, d_harm=harm)
    picked["quantile_head"] = picked["evening_engine"]
    picked["sign_head"] = picked["ambiguous"]
    real_picked: dict[str, Day | None] = {}
    if real_segments is not None and real_rows is not None:
        d_real = lobes_of(
            pred=load_prediction(arm="D", arch=arch, name="real_r1", n_rows=real_rows.n)
        )
        real_picked = pick_real(
            days=complete_days(rows=real_rows), d_p_neg=dist.prob_negative(lobes=d_real)
        )
    written = []
    for spec in SPECS:
        is_real = spec.name.startswith("real_")
        day = real_picked.get(spec.name) if is_real else picked.get(spec.name)
        if day is None:
            log(f"{spec.file}: no day meets the rule, figure skipped")
            continue
        src_rows, segs = (real_rows, real_segments) if is_real else (rows, test_segments)
        assert src_rows is not None
        assert segs is not None
        window = window_data(seg=segs[day.seg], date=day.date, with_truth=not is_real)
        fc = forecasts_for(
            spec=spec,
            arch=arch,
            name="real_r1" if is_real else "test",
            index=day.index,
            r=src_rows.r,
        )
        chart = window_chart(
            title=spec.title, subtitle=caption_rule(spec=spec), window=window, forecasts=fc,
            bands=spec.bands, show_bsp=spec.show_bsp,
        )  # fmt: skip
        figures.save_svg(chart=chart, path=out_dir / spec.file)
        written.append(out_dir / spec.file)
        log(f"{spec.file}: {RULES[spec.name]}")
    return written

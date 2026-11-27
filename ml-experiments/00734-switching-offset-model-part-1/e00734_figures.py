"""Figures of experiment 734: event figures, the synthetic world gallery, and the picks for both.

`event_figure` draws one event: the demand of every Primary involved (donor, recipient, and each
mirror partner the model used), the decoded offset with its quantile band, and the shaded window
of the event, from a margin before the event to the same margin after it. It takes an event table,
a list of Primaries, and the margin, so the write-up can draw any event. `pick_categories` and
`pick_real` choose the events of each category of the plan. `gallery` draws tuning worlds only,
and `scored_figures` and `real_figures` run after scoring.

Primaries are labelled by D code. No metered generator's series is drawn: the demand of a Primary
is a substation's power, and an injected change is a synthetic series.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import altair as alt
import e00734_bank as bank
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

if TYPE_CHECKING:
    import e00734_decode as dec

DAY: Final[int] = 48
"""Half-hours per day."""
MARGIN_FRAC: Final[float] = 0.25
"""Margin on each side of an event, as a share of the event's duration."""
MARGIN_MIN_DAYS: Final[float] = 2.0
"""Smallest margin on each side of an event."""
OPEN_EVENT_DAYS: Final[float] = 60.0
"""Days drawn for an event with no end, from its start."""
PER_GRADE: Final[int] = 6
"""Injected events drawn for each grade and each other category."""
PER_REAL: Final[int] = 12
"""Real events drawn for the real-record table."""
MAX_PRIMARIES: Final[int] = 4
"""Most Primaries drawn in one figure (donor, recipients, and partners; the rest are listed)."""
MAX_POINTS: Final[int] = 700
"""Most points of a line in one panel; longer windows are averaged."""
MAX_WIDTH: Final[int] = 1000
"""Widest SVG in pixels: the docs page's text column."""
MAX_SVG_BYTES: Final[int] = 450_000
"""Size under which a raw SVG is kept; the repository's hook rejects 500 KB after optimisation."""
PLOT_WIDTH: Final[int] = 600
PANEL_HEIGHT: Final[int] = 110
GRADES: Final[tuple[str, ...]] = (
    "mirrored and returned",
    "mirrored, not yet returned",
    "returned, no visible partner",
    "neither",
)
CALL_COLOURS: Final[dict[str, str]] = {"E1-J-sym": ocf.DATA_GREEN, "PELT": ocf.DATA_SKY}
GALLERY_ARMS: Final[tuple[str, ...]] = ("H-own", "H-nbr")
Shade = Literal["truth", "called"]


@dataclass
class EventSource:
    """The series that an event figure draws."""

    power: dict[str, np.ndarray]
    """Half-hourly demand (or injected change) of each Primary in megawatts."""
    record_end: int
    """One past the last half-hour of the record."""
    blocks: dict[str, pl.DataFrame] = field(default_factory=dict)
    """Per Primary, the block quantiles `block`, `q05`, `q10`, `q50`, `q90`, `q95` in megawatts."""
    calls: dict[str, pl.DataFrame] = field(default_factory=dict)
    """Per method, called events with `primary`, `start`, `end` (-1 for none), and `amp_mw`."""


def event_extent(*, event: pl.DataFrame, record_end: int) -> tuple[int, int]:
    """First half-hour and one past the last half-hour of an event table's rows.

    Args:
        event: Rows with `start` and `end`; an end of null or -1 is an open event.
        record_end: One past the last half-hour of the record.

    Returns:
        The earliest start and the latest end, an open event running `OPEN_EVENT_DAYS` days.
    """
    start = int(event["start"].min())  # ty: ignore[invalid-argument-type]
    ends = [
        min(record_end, int(r["start"]) + int(OPEN_EVENT_DAYS * DAY))
        if r["end"] is None or r["end"] < 0
        else min(record_end, int(r["end"]))
        for r in event.iter_rows(named=True)
    ]
    return start, max(ends)


def draw_window(
    *,
    start: int,
    end: int,
    record_end: int,
    margin_frac: float = MARGIN_FRAC,
    min_margin_days: float = MARGIN_MIN_DAYS,
) -> tuple[int, int]:
    """Half-hours drawn: the event plus a margin of a share of its duration, at least 2 days.

    Args:
        start: First half-hour of the event.
        end: One past the last half-hour of the event.
        record_end: One past the last half-hour of the record.
        margin_frac: The margin on each side as a share of the event's duration.
        min_margin_days: The smallest margin in days.

    Returns:
        The first and one-past-last half-hour of the window, inside `[0, record_end]`.
    """
    margin = max(math.ceil(margin_frac * (end - start)), int(min_margin_days * DAY))
    return max(0, start - margin), min(record_end, end + margin)


def thin(*, x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Average a series over equal groups of points so that at most `max_points` remain."""
    n = len(x)
    k = max(1, math.ceil(n / max_points))
    if k == 1:
        return x, y
    m = n // k * k
    xs = x[:m].reshape(-1, k).mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a group with no reading averages to NaN
        ys = np.nanmean(y[:m].reshape(-1, k), axis=1)
    return xs, ys


def involved_primaries(
    *, site: str, start: int, group_sites: Sequence[str], calls: pl.DataFrame, tol: int = 24
) -> list[str]:
    """The Primary of an event, its group, and the partners of a matching call.

    Args:
        site: The Primary of the event.
        start: Its first half-hour.
        group_sites: Every Primary of the injected group (donor and recipients), may be empty.
        calls: A call table with `primary`, `start`, and `partners` (a comma-separated string).
        tol: How far a call's start may lie from `start` to count as the same event.

    Returns:
        The Primary first, then the others in order of appearance, without repeats.
    """
    out = [site, *group_sites]
    if calls.height:
        near = calls.filter((pl.col("primary") == site) & ((pl.col("start") - start).abs() <= tol))
        for p in near["partners"].to_list():
            out += [s for s in (p or "").split(",") if s]
    return list(dict.fromkeys(out))


def clip_spans(*, spans: Sequence[tuple[int, int]], lo: int, hi: int) -> list[tuple[int, int]]:
    """The spans that overlap the window `[lo, hi)`, cut to the window.

    A mark outside the plotted days would widen the SVG far beyond the page, so every span is cut
    to the window before it reaches the chart.

    Args:
        spans: Pairs of the first half-hour and one past the last half-hour.
        lo: First half-hour of the window.
        hi: One past the last half-hour of the window.

    Returns:
        The overlapping spans, each inside the window.
    """
    return [(max(a, lo), min(b, hi)) for a, b in spans if b > lo and a < hi]


def _panel(
    *,
    primary: str,
    event: pl.DataFrame,
    source: EventSource,
    lo: int,
    hi: int,
    ref: int,
    shade: tuple[int, int],
    max_points: int,
    demand_title: str,
    show_x: bool,
) -> alt.VConcatChart:
    """The demand panel and the offset panel of one Primary."""
    x_axis = alt.Axis(title="Days from the start of the event" if show_x else None)
    domain = [(lo - ref) / DAY, (hi - ref) / DAY]
    scale = alt.Scale(domain=domain, nice=False)
    rows = event.filter(pl.col("primary") == primary)
    if rows.height:
        raw = [
            (int(a), min(hi, int(a) + int(OPEN_EVENT_DAYS * DAY)))
            if e is None or e < 0
            else (int(a), int(e))
            for a, e in zip(rows["start"].to_list(), rows["end"].to_list(), strict=True)
        ]
    else:
        raw = [shade]
    drawn_spans = clip_spans(spans=raw, lo=lo, hi=hi) or [(lo, lo)]
    rect = pl.DataFrame(
        {
            "x0": [(a - ref) / DAY for a, _ in drawn_spans],
            "x1": [(b - ref) / DAY for _, b in drawn_spans],
        }
    )
    shade_layer = (
        alt.Chart(rect)  # ty: ignore[unresolved-attribute]
        .mark_rect(opacity=0.22, color=ocf.BRAND_ORANGE)
        .encode(x=alt.X("x0:Q", scale=scale, axis=None), x2="x1:Q")
    )
    series = source.power[primary]
    idx = np.arange(lo, min(hi, len(series)))
    xs, ys = thin(x=(idx - ref) / DAY, y=series[idx], max_points=max_points)
    demand = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            pl.DataFrame({"day": xs, "mw": ys})
            .with_columns(pl.col("mw").fill_nan(None))
            .drop_nulls()
        )
        .mark_line(color=ocf.DATA_BLUE, strokeWidth=0.8)
        .encode(
            x=alt.X("day:Q", scale=scale, axis=alt.Axis(title=None, labels=False)),
            y=alt.Y("mw:Q", scale=alt.Scale(zero=False), title=demand_title),
        )
    )
    top = (shade_layer + demand).properties(
        width=PLOT_WIDTH, height=PANEL_HEIGHT, title=f"Primary {primary}"
    )
    blocks = source.blocks.get(primary)
    layers: list[alt.Chart | alt.LayerChart] = [shade_layer]
    if blocks is not None and blocks.height:
        b = blocks.filter((pl.col("block") * 12 + 6 >= lo) & (pl.col("block") * 12 + 6 < hi))
        day = ((b["block"].to_numpy() * 12 + 6) - ref) / DAY
        step = max(1, math.ceil(len(day) / max_points))
        band = pl.DataFrame(
            {
                "day": day[::step],
                "q05": b["q05"].to_numpy()[::step],
                "q95": b["q95"].to_numpy()[::step],
                "q10": b["q10"].to_numpy()[::step],
                "q90": b["q90"].to_numpy()[::step],
                "q50": b["q50"].to_numpy()[::step],
            }
        )
        enc_x = alt.X("day:Q", scale=scale, axis=x_axis)
        layers += [
            alt.Chart(band)  # ty: ignore[unresolved-attribute]
            .mark_area(color=ocf.DATA_PURPLE_LIGHT, opacity=0.6, interpolate="step-after")
            .encode(x=enc_x, y=alt.Y("q05:Q", title="Offset (MW)"), y2="q95:Q"),
            alt.Chart(band)  # ty: ignore[unresolved-attribute]
            .mark_area(color=ocf.DATA_PURPLE, opacity=0.3, interpolate="step-after")
            .encode(x=enc_x, y="q10:Q", y2="q90:Q"),
            alt.Chart(band)  # ty: ignore[unresolved-attribute]
            .mark_line(color=ocf.DATA_PURPLE, strokeWidth=1.0, interpolate="step-after")
            .encode(x=enc_x, y="q50:Q"),
        ]
    for method, table in source.calls.items():
        t = table.filter(pl.col("primary") == primary) if table.height else table
        if not t.height:
            continue
        kept = [
            (int(st), hi if e is None or e < 0 else int(e), float(amp))
            for st, e, amp in zip(
                t["start"].to_list(), t["end"].to_list(), t["amp_mw"].to_list(), strict=True
            )
            if (hi if e is None or e < 0 else int(e)) > lo and int(st) < hi
        ]
        if not kept:
            continue
        seg = pl.DataFrame(
            {
                "x0": [(max(st, lo) - ref) / DAY for st, _, _ in kept],
                "x1": [(min(e, hi) - ref) / DAY for _, e, _ in kept],
                "amp": [amp for _, _, amp in kept],
            }
        )
        layers.append(
            alt.Chart(seg)  # ty: ignore[unresolved-attribute]
            .mark_rule(color=CALL_COLOURS.get(method, ocf.DATA_AMBER), strokeWidth=1.2)
            .encode(x=alt.X("x0:Q", scale=scale, axis=x_axis), x2="x1:Q", y="amp:Q")
        )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=ocf.ENSEMBLE_LINE, strokeWidth=0.5)
    layers.append(zero.encode(y="y:Q"))  # ty: ignore[unresolved-attribute]
    bottom = alt.layer(*layers).properties(width=PLOT_WIDTH, height=PANEL_HEIGHT)
    return alt.vconcat(top, bottom, spacing=2)


def event_figure(
    *,
    event: pl.DataFrame,
    primaries: Sequence[str],
    source: EventSource,
    margin_frac: float = MARGIN_FRAC,
    min_margin_days: float = MARGIN_MIN_DAYS,
    shade: Shade = "truth",
    title: str = "",
    demand_title: str = "Demand (MW)",
    max_points: int = MAX_POINTS,
    note: str = "",
) -> alt.VConcatChart:
    """Draw one event at every Primary involved, with the offset's quantile band and the calls.

    Each Primary gets a demand panel and an offset panel on the same days. The offset panel holds
    the 90% band (q05 to q95, light), the 80% band (q10 to q90), the median of the switched
    megawatts, and thin lines for the events that other methods called at that Primary.

    Args:
        event: The event's rows with `primary`, `start`, `end` (null or -1 for no end); the
            window shaded at a Primary is its own rows, else the whole event.
        primaries: The Primaries to draw, in order; only the first `MAX_PRIMARIES` are drawn.
        source: The series to draw.
        margin_frac: The margin on each side as a share of the event's duration.
        min_margin_days: The smallest margin on each side in days.
        shade: `truth` when the shaded window is an injected event, `called` when it is a called
            event; the key in the title says which.
        title: The figure's title.
        demand_title: The y-axis title of the demand panels.
        max_points: Most points per line.
        note: Text added to the key under the title, such as the priors that made the decode.

    Returns:
        The chart, one pair of panels per Primary.
    """
    start, end = event_extent(event=event, record_end=source.record_end)
    lo, hi = draw_window(
        start=start,
        end=end,
        record_end=source.record_end,
        margin_frac=margin_frac,
        min_margin_days=min_margin_days,
    )
    drawn = [p for p in primaries if p in source.power][:MAX_PRIMARIES]
    panels = [
        _panel(
            primary=p,
            event=event,
            source=source,
            lo=lo,
            hi=hi,
            ref=start,
            shade=(start, end),
            max_points=max_points,
            demand_title=demand_title,
            show_x=k == len(drawn) - 1,
        )
        for k, p in enumerate(drawn)
    ]
    key = (
        "Orange: injected event. Purple: decoded switched megawatts (median, 80% and 90% bands)."
        if shade == "truth"
        else "Orange: called event. Purple: decoded switched megawatts (median, 80% and 90% bands)."
    )
    if note:
        key += f" {note}"
    if source.calls:
        names = ", ".join(f"{m} ({_colour_name(m)})" for m in source.calls)
        key += f" Thin lines: {names}."
    chart = alt.vconcat(*panels, spacing=8).properties(
        title=alt.TitleParams(text=title, subtitle=_wrap(text=key, width=105))
    )
    return chart.configure_view(strokeWidth=0)  # ty: ignore[invalid-return-type]


def _colour_name(method: str) -> str:
    return {"E1-J-sym": "green", "PELT": "blue"}.get(method, "amber")


def _wrap(*, text: str, width: int) -> list[str]:
    """Split text into lines of about `width` characters at spaces."""
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return [*lines, cur] if cur else lines


def svg_width(*, text: str) -> float:
    """The width in pixels that the root `svg` element declares."""
    match = re.search(r'<svg[^>]*?\swidth="([0-9.]+)"', text)
    if match is None:
        raise ValueError("no width on the svg element")
    return float(match.group(1))


def save_svg(*, chart: alt.VConcatChart, path: Path) -> int:
    """Save a chart as SVG and return its size in bytes (the file is replaced atomically)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    chart.save(tmp, format="svg")
    width = svg_width(text=tmp.read_text(encoding="utf-8")[:2000])
    if width > MAX_WIDTH:
        tmp.unlink()
        raise ValueError(f"{path.name} is {width} px wide; the limit is {MAX_WIDTH}")
    tmp.rename(path)
    return path.stat().st_size


def _make(
    n: int,
    *,
    event: pl.DataFrame,
    sites: Sequence[str],
    src: EventSource,
    title: str,
    y_title: str = "Demand (MW)",
    shade: Shade = "truth",
    note: str = "",
) -> alt.VConcatChart:
    """An event figure with `n` points per line (the argument `save_within_size` varies)."""
    return event_figure(
        event=event,
        primaries=sites,
        source=src,
        shade=shade,
        title=title,
        demand_title=y_title,
        max_points=n,
        note=note,
    )


def save_within_size(*, make: Callable[[int], alt.VConcatChart], path: Path) -> int:
    """Save `make(max_points)` and halve the points until the SVG fits `MAX_SVG_BYTES`."""
    points = MAX_POINTS
    while True:
        size = save_svg(chart=make(points), path=path)
        if size <= MAX_SVG_BYTES or points <= 50:
            return size
        points //= 2


def _sample(*, df: pl.DataFrame, n: int) -> pl.DataFrame:
    """Up to `n` rows of a table, the same rows on every run."""
    return df if df.height <= n else df.sample(n=n, seed=734)


def pick_categories(
    *, matches: pl.DataFrame, calls: pl.DataFrame, per_category: int = PER_GRADE
) -> list[dict]:
    """Choose the injected events to draw in each category of the plan.

    The categories are every grade of a called event, hits, misses and false calls, the long and
    shallow and the long and deep events, and cascades.

    Args:
        matches: H-nbr's match rows over the worlds with `world`, `site`, `start`, `end`, `found`,
            `cell` (from `add_cells`), `group`, and `cascaded`.
        calls: H-nbr's call tables with `world`, `primary`, `start`, `end`, `grade`, `partners`,
            `matched`, and `on_hard_negative`.
        per_category: Events per category.

    Returns:
        One dict per event with `category`, `world`, `kind` (`truth` or `called`), and the event's
        `primary`, `start`, `end`, and for a truth event its `group`.
    """
    out: list[dict] = []

    def add_called(*, category: str, rows: pl.DataFrame) -> None:
        out.extend(
            {
                "category": category,
                "world": int(r["world"]),
                "kind": "called",
                "primary": r["primary"],
                "start": int(r["start"]),
                "end": int(r["end"]),
                "group": None,
            }
            for r in _sample(df=rows, n=per_category).iter_rows(named=True)
        )

    def add_truth(*, category: str, rows: pl.DataFrame) -> None:
        out.extend(
            {
                "category": category,
                "world": int(r["world"]),
                "kind": "truth",
                "primary": r["site"],
                "start": int(r["start"]),
                "end": int(r["end"]),
                "group": int(r["group"]),
            }
            for r in _sample(df=rows, n=per_category).iter_rows(named=True)
        )

    for g in GRADES:
        add_called(category=f"grade: {g}", rows=calls.filter(pl.col("grade") == g))
    add_called(
        category="false call",
        rows=calls.filter(~pl.col("matched") & ~pl.col("on_hard_negative")),
    )
    if matches.height:
        for cell in ("long, shallow", "long, deep"):
            sub = matches.filter(pl.col("cell") == cell)
            add_truth(category=f"hit, {cell}", rows=sub.filter(pl.col("found")))
            add_truth(category=f"miss, {cell}", rows=sub.filter(~pl.col("found")))
        if "cascaded" in matches.columns:
            casc = matches.filter(pl.col("cascaded"))
            add_truth(category="cascade, found", rows=casc.filter(pl.col("found")))
            add_truth(category="cascade, missed", rows=casc.filter(~pl.col("found")))
    return out


def pick_real(
    *, calls: pl.DataFrame, logged: pl.DataFrame, per_category: int = PER_REAL, tol: int = 48
) -> list[dict]:
    """Choose the real events to draw: called events, and the logged shifts found and missed.

    Args:
        calls: H-nbr's called events on the real record with `primary`, `start`, `end`, `grade`.
        logged: NGED's logged shifts of more than 1 day with `primary`, `start`, and `end`.
        per_category: Events per category (half of them from mirrored grades in the called set).
        tol: How close a call's start must be to a logged start to count as finding it.

    Returns:
        One dict per event with `category`, `primary`, `start`, and `end`.
    """
    out: list[dict] = []

    def add(*, category: str, rows: pl.DataFrame) -> None:
        out.extend(
            {
                "category": category,
                "primary": r["primary"],
                "start": int(r["start"]),
                "end": int(r["end"]),
            }
            for r in _sample(df=rows, n=per_category).iter_rows(named=True)
        )

    scored = calls.filter(pl.col("start") >= 0)
    add(
        category="called, mirrored", rows=scored.filter(pl.col("grade").str.starts_with("mirrored"))
    )
    add(
        category="called, not mirrored",
        rows=scored.filter(~pl.col("grade").str.starts_with("mirrored")),
    )
    if logged.height:
        found = [
            bool(
                scored.filter(
                    (pl.col("primary") == r["primary"])
                    & ((pl.col("start") - r["start"]).abs() <= tol)
                ).height
            )
            for r in logged.iter_rows(named=True)
        ]
        flag = pl.Series("found", found, dtype=pl.Boolean)
        add(category="logged shift found", rows=logged.filter(flag))
        add(category="logged shift missed", rows=logged.filter(~flag))
    return out


def injected_rows(*, table: pl.DataFrame, group: int | None, site: str, start: int) -> pl.DataFrame:
    """The rows of an injected group (donor and recipients) as an event table."""
    if group is None:
        rows = table.filter((pl.col("site") == site) & (pl.col("start") == start))
    else:
        rows = table.filter(pl.col("group_id") == group)
    return rows.select(primary=pl.col("site"), start=pl.col("start"), end=pl.col("end"))


# runners ---------------------------------------------------------------------------------------


def _blocks_frame(*, post: object, sigma: float) -> pl.DataFrame:
    """Block quantiles of one decoded Primary as a table for `EventSource.blocks`."""
    import e00734_events as ev

    q = ev.block_quantiles(post=post, sigma=sigma)  # ty: ignore[invalid-argument-type]
    cols = {f"q{int(v * 100):02d}": q[:, j] for j, v in enumerate(ev.QUANTILES)}
    return pl.DataFrame({"block": np.arange(len(q)), **cols})


EXAMPLE_NOTE: Final[str] = (
    "Decoded with fixed example priors, not tuned (rho_on 3e-4, rho_off 1/240, rho_cont 0.2, "
    "kappa 5, tau 0.9: the middle of the plan's grid)."
)
"""Said in the key of every gallery figure that draws a decode."""
MIN_IOU: Final[float] = 0.2
"""Overlap of a called event with the truth window that counts as the model decoding the event."""
SIZE_NAMES: Final[tuple[str, ...]] = ("small", "medium", "large")
EXAMPLE_KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("edge_transfer", "edge"),
    ("temporary_shift", "solo"),
)
"""Event classes of the decode examples and the name of each."""
SHORT_DAYS: Final[float] = 3.0
LONG_DAYS: Final[float] = 14.0


@dataclass
class DecodedWorld:
    """One decoded world: the block quantiles, the called events, and P(zero) of each Primary."""

    blocks: dict[str, pl.DataFrame]
    calls: pl.DataFrame
    p_zero: dict[str, np.ndarray]


def _example_priors() -> object:
    import e00734_hmm as hmm

    return hmm.Priors(rho_on=3e-4, rho_off=1 / 240, rho_cont=0.2, kappa=5.0)


def decode_world(*, ctx: dec.Context, power: dict[str, np.ndarray], n_blocks: int) -> DecodedWorld:
    """Decode a world with arm H-nbr and the example priors, for the figures."""
    import e00734_decode as dec_
    import e00734_hmm as hmm

    d = dec_.WorldDecoder(
        ctx=ctx, power=power, cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh, n_blocks=n_blocks
    )
    ap = dec_.ArmParams(priors=_example_priors(), tau=0.9)  # ty: ignore[invalid-argument-type]
    res = d.run_arm(arm="H-nbr", params=ap)
    return DecodedWorld(
        blocks={s: _blocks_frame(post=p, sigma=res.blocks.sigma[s]) for s, p in res.posts.items()},
        calls=dec_.events_frame(events=d.final_events(result=res, tau=ap.tau)),
        p_zero={s: p.o_marg[:, hmm.FULL_GRID.zero].astype(float) for s, p in res.posts.items()},
    )


def window_stats(
    *,
    blocks: pl.DataFrame,
    p_zero: np.ndarray,
    calls: pl.DataFrame,
    primary: str,
    start: int,
    end: int,
) -> dict[str, float | bool]:
    """What the decode says inside a truth window at one Primary.

    Args:
        blocks: The Primary's block quantiles `block`, `q10`, `q50`, `q90` in megawatts.
        p_zero: The posterior probability that the offset is zero, per block.
        calls: Called events with `primary`, `start`, `end` (-1 for none).
        primary: The Primary.
        start: First half-hour of the truth window.
        end: One past the last half-hour of the truth window.

    Returns:
        The mean probability of zero, the largest absolute median, the largest 90th percentile,
        and the smallest 10th percentile of the window's blocks, and whether a called event
        overlaps the window by `MIN_IOU` or more (with the best overlap).
    """
    n = blocks.height
    lo, hi = min(start // 12, n - 1), min(max(-(-end // 12), start // 12 + 1), n)
    q50 = blocks["q50"].to_numpy()[lo:hi]
    best = 0.0
    for c in calls.filter(pl.col("primary") == primary).iter_rows(named=True):
        c_end = c["end"] if c["end"] >= 0 else start + 60 * DAY
        inter = max(0, min(end, c_end) - max(start, c["start"]))
        union = max(end, c_end) - min(start, c["start"])
        best = max(best, inter / union if union > 0 else 0.0)
    return {
        "mean_p_zero": float(np.mean(p_zero[lo:hi])),
        "max_abs_q50_mw": float(np.max(np.abs(q50))),
        "max_q90_mw": float(np.max(blocks["q90"].to_numpy()[lo:hi])),
        "min_q10_mw": float(np.min(blocks["q10"].to_numpy()[lo:hi])),
        "best_iou": best,
        "decoded": best >= MIN_IOU,
    }


def pick_decode_examples(
    *, table: pl.DataFrame, max_half_hours: int
) -> tuple[list[dict], list[dict]]:
    """Injected events for the decode figures, one slot per size, event kind, and duration.

    The slots are small, medium, and large (terciles of the largest `units` in the group) for an
    edge transfer and for a solo shift. Slots alternate between a short event (3 days or less) and
    a long one (14 days or more), taking any duration when the preferred one is missing.

    Args:
        table: One world's injected events, with `group_id`, `site`, `event_class`, `label`,
            `units`, `start`, and `end`.
        max_half_hours: Events must start and end before this half-hour.

    Returns:
        The chosen slots, then the other candidates in table order, each a dict with `group`,
        `size`, `kind`, `days`, and `units`.
    """
    pos = table.filter(
        (pl.col("label") != "hard_negative")
        & pl.col("end").is_not_null()
        & (pl.col("end") > pl.col("start"))
        & (pl.col("end") <= max_half_hours)
        & pl.col("event_class").is_in([c for c, _ in EXAMPLE_KINDS])
    )
    groups = pos.group_by("group_id", maintain_order=True).agg(
        units=pl.col("units").abs().max(),
        start=pl.col("start").min(),
        end=pl.col("end").max(),
        event_class=pl.col("event_class").first(),
    )
    out: list[dict] = []
    for event_class, kind in EXAMPLE_KINDS:
        rows = groups.filter(pl.col("event_class") == event_class).sort("units")
        n = rows.height
        for i, r in enumerate(rows.iter_rows(named=True)):
            out.append(
                {
                    "group": int(r["group_id"]),
                    "size": SIZE_NAMES[min(2, 3 * i // max(n, 1))],
                    "kind": kind,
                    "days": (r["end"] - r["start"]) / DAY,
                    "units": float(r["units"]),
                }
            )
    slots: list[dict] = []
    used: set[int] = set()
    for kind in (k for _, k in EXAMPLE_KINDS):
        for j, size in enumerate(SIZE_NAMES):
            pool = [
                c for c in out if c["kind"] == kind and c["size"] == size and c["group"] not in used
            ]
            want_short = j % 2 == 0
            pool.sort(
                key=lambda c: (
                    not (c["days"] <= SHORT_DAYS if want_short else c["days"] >= LONG_DAYS)
                )
            )
            if pool:
                slots.append(pool[0])
                used.add(pool[0]["group"])
    return slots, [c for c in out if c["group"] not in used]


def gallery(*, out: Path) -> None:
    """Draw the synthetic world gallery on tuning worlds only, in `out / "gallery"`.

    The gallery holds each injected component alone (edge transfers, solo shifts, permanent
    shifts, staged returns, overlapping events, cascades, cold spells, hard negatives), the same
    events on the real residual, event figures of the offset model on tuning worlds with untuned
    mid-grid priors, and the real-data aggregates. No scored world is read.
    """
    import e00660_generator as gen  # ty: ignore[unresolved-import]
    import e00734_decode as dec_
    import e00734_layers as layers
    import e00734_panel as pan

    ctx = dec_.load_context()
    rig = ctx.rig
    folder = out / "gallery"
    seeds = bank.guard_tuning(seeds=bank.TUNING_SEEDS[:2])
    sizes: dict[str, int] = {}
    examples: list[dict] = []
    cfg = gen.default_config()
    for seed in seeds:
        world = gen.inject(panel=rig.tune_panel, config=cfg, seed=seed, start_times=rig.starts)
        table = world.events
        resid = rig.residual_of(world=world)
        base = {s: world.power[s] - rig.tune_panel.power[s] for s in world.power}
        record_end = dec_.record_end_of(power=rig.tune_panel.power)
        decoded = decode_world(ctx=ctx, power=world.power, n_blocks=pan.TUNE_BLOCKS)
        calls = {"H-nbr called": decoded.calls}
        for label, pick in _gallery_picks(table=table).items():
            for k, rows in enumerate(pick):
                sites = rows["primary"].unique(maintain_order=True).to_list()
                for what, series in (("injected", base), ("on_real_residual", resid)):
                    src = EventSource(
                        power=series, record_end=record_end, blocks=decoded.blocks, calls=calls
                    )
                    name = f"{label}_{seed}_{k}_{what}.svg"
                    title = f"{label} in tuning world {seed} ({what.replace('_', ' ')})"
                    y_title = "Change (MW)" if what == "injected" else "Residual (MW)"
                    sizes[name] = save_within_size(
                        make=partial(
                            _make,
                            event=rows,
                            sites=sites,
                            src=src,
                            title=title,
                            y_title=y_title,
                            note=EXAMPLE_NOTE,
                        ),
                        path=folder / name,
                    )
        sizes |= _cascade_gallery(
            ctx=ctx, world=world, seed=seed, folder=folder, layers=layers, gen=gen, pan=pan
        )
        made, rows_out = _posterior_gallery(
            world=world,
            decoded=decoded,
            seed=seed,
            folder=folder,
            record_end=record_end,
            n_blocks=pan.TUNE_BLOCKS,
        )
        sizes |= made
        examples += rows_out
    sizes |= _cold_gallery(rig=rig, folder=folder, seed=seeds[0], layers=layers)
    sizes |= _solar_gallery(folder=folder)
    sizes |= _aggregates(out=out, ctx=ctx, folder=folder)
    json = __import__("json")
    (folder / "sizes.json").write_text(json.dumps(sizes, indent=1))
    (folder / "decode_examples.json").write_text(json.dumps(examples, indent=1))


def _gallery_picks(*, table: pl.DataFrame) -> dict[str, list[pl.DataFrame]]:
    """Event tables of the gallery's injected components, from one world's table."""
    pos = table.filter(pl.col("label") != "hard_negative")
    picks: dict[str, list[pl.DataFrame]] = {}

    def group_rows(*, rows: pl.DataFrame, n: int) -> list[pl.DataFrame]:
        gids = rows["group_id"].unique(maintain_order=True).to_list()[:n]
        return [
            table.filter(pl.col("group_id") == g).select(
                primary=pl.col("site"), start=pl.col("start"), end=pl.col("end")
            )
            for g in gids
        ]

    long = pl.col("end").is_null() | ((pl.col("end") - pl.col("start")) > 2 * DAY)
    picks["edge_transfer"] = group_rows(
        rows=pos.filter((pl.col("event_class") == "edge_transfer") & long), n=3
    )
    picks["solo_shift"] = group_rows(
        rows=pos.filter((pl.col("event_class") == "temporary_shift") & long), n=1
    )
    picks["permanent_shift"] = group_rows(
        rows=pos.filter(pl.col("event_class") == "permanent_shift"), n=1
    )
    picks["staged_return"] = group_rows(rows=pos.filter(pl.col("staged_hh") > 0), n=1)
    picks["overlapping"] = _overlapping(table=pos)
    for name, rows in zip(
        ("hard_routine_step", "hard_ramp", "hard_pulse", "hard_placebo_pair", "hard_common_mode"),
        (
            table.filter(pl.col("event_class") == c)
            for c in ("routine_step", "ramp", "pulse", "placebo_pair", "common_mode")
        ),
        strict=True,
    ):
        picks[name] = group_rows(rows=rows, n=1)
    return {k: v for k, v in picks.items() if v}


def dec_record_end(*, power: dict[str, np.ndarray]) -> int:
    """One past the last half-hour with a reading at any Primary."""
    import e00734_decode as dec_

    return dec_.record_end_of(power=power)


def _overlapping(*, table: pl.DataFrame) -> list[pl.DataFrame]:
    """One pair of injected events whose windows overlap at one Primary, as an event table."""
    for site in table["site"].unique(maintain_order=True).to_list():
        rows = table.filter(pl.col("site") == site).sort("start").to_dicts()
        for a, b in pairwise(rows):
            if a["end"] is None or b["start"] < a["end"]:
                return [
                    pl.DataFrame(
                        {
                            "primary": [site, site],
                            "start": [a["start"], b["start"]],
                            "end": [a["end"], b["end"]],
                        },
                        schema={"primary": pl.String, "start": pl.Int64, "end": pl.Int64},
                    )
                ]
    return []


def _cascade_gallery(
    *,
    ctx: dec.Context,
    world: object,
    seed: int,
    folder: Path,
    layers: object,
    gen: object,
    pan: object,
) -> dict[str, int]:
    """Cascaded starts and returns of one tuning world: the injected change and the decode."""
    rig = ctx.rig
    rng = np.random.default_rng(seed + 7_340_000)
    table = layers.cascade_events(events=world.events, rng=rng)  # ty: ignore[unresolved-attribute]
    power = gen.apply_events(panel=rig.panel, events=table.drop("cascade_id", "cascade_end"))  # ty: ignore[unresolved-attribute]
    change = {s: power[s] - rig.tune_panel.power[s] for s in power}
    record_end = dec_record_end(power=rig.tune_panel.power)
    decoded = decode_world(ctx=ctx, power=power, n_blocks=pan.TUNE_BLOCKS)  # ty: ignore[unresolved-attribute]
    out: dict[str, int] = {}
    casc = table.filter(pl.col("cascade_id").is_not_null())
    ids = casc["cascade_id"].unique(maintain_order=True).to_list()[:3]
    for k, cid in enumerate(ids):
        rows = casc.filter(pl.col("cascade_id") == cid)
        ev_rows = rows.select(
            primary=pl.col("site"), start=pl.col("start"), end=pl.col("cascade_end")
        )
        src = EventSource(
            power=change,
            record_end=record_end,
            blocks=decoded.blocks,
            calls={"H-nbr called": decoded.calls},
        )
        name = f"cascade_{seed}_{k}.svg"
        out[name] = save_within_size(
            make=partial(
                _make,
                event=ev_rows,
                sites=ev_rows["primary"].unique(maintain_order=True).to_list(),
                src=src,
                title=f"Cascaded event {cid} in tuning world {seed}: 2 or 3 steps within a day",
                y_title="Change (MW)",
                note=EXAMPLE_NOTE,
            ),
            path=folder / name,
        )
    return out


def _posterior_gallery(
    *,
    world: object,
    decoded: DecodedWorld,
    seed: int,
    folder: Path,
    record_end: int,
    n_blocks: int,
) -> tuple[dict[str, int], list[dict]]:
    """Decode figures of injected events of every size and kind on one tuning world.

    The six slots of `pick_decode_examples` come first. When fewer than two of them are decoded
    (a called event overlaps the truth window), decoded events from the other candidates are
    added, so the gallery shows what a decode looks like as well as what it misses.
    """
    table = world.events  # ty: ignore[unresolved-attribute]
    slots, rest = pick_decode_examples(table=table, max_half_hours=n_blocks * 12)

    def stats_of(cand: dict) -> dict:
        rows = table.filter(pl.col("group_id") == cand["group"]).sort(
            pl.col("units").abs(), descending=True
        )
        lead = rows.row(0, named=True)
        return window_stats(
            blocks=decoded.blocks[lead["site"]],
            p_zero=decoded.p_zero[lead["site"]],
            calls=decoded.calls,
            primary=lead["site"],
            start=int(lead["start"]),
            end=int(lead["end"]),
        ) | {"primary": lead["site"]}

    chosen = [(c, stats_of(c)) for c in slots]
    missing = max(0, 2 - sum(bool(st["decoded"]) for _, st in chosen))
    if missing:
        extra = [(c, stats_of(c)) for c in rest]
        chosen += [(c, st) for c, st in extra if st["decoded"]][:missing]
    out: dict[str, int] = {}
    diag: list[dict] = []
    for k, (cand, st) in enumerate(chosen):
        rows = injected_rows(table=table, group=cand["group"], site="", start=0)
        src = EventSource(
            power=dict(world.power),  # ty: ignore[unresolved-attribute]
            record_end=record_end,
            blocks=decoded.blocks,
            calls={"H-nbr called": decoded.calls},
        )
        name = f"posterior_{seed}_{k}.svg"
        verdict = "decoded" if st["decoded"] else "not decoded"
        title = (
            f"Tuning world {seed}: {cand['size']} {cand['kind']} event, "
            f"{cand['units']:.1f} noise units, {cand['days']:.1f} days ({verdict})"
        )
        out[name] = save_within_size(
            make=partial(
                _make,
                event=rows,
                sites=rows["primary"].unique(maintain_order=True).to_list(),
                src=src,
                title=title,
                note=EXAMPLE_NOTE,
            ),
            path=folder / name,
        )
        diag.append({"figure": name, "seed": seed} | cand | st)
    return out, diag


def _solar_gallery(*, folder: Path) -> dict[str, int]:
    """Two of experiment 686's solar-growth worlds: daylight blocks against the night block.

    The injected change is experiment 653's, drawn as daily means of the daylight half-hours
    (06:00 to 18:00 UTC) and of the night half-hours at the Primary with the largest growth.
    """
    from e00653_common import GRID_LENGTH, SCORED_FROM  # ty: ignore[unresolved-import]
    from e00653_inject import GenParams, inject  # ty: ignore[unresolved-import]
    from e00653_run import setup  # ty: ignore[unresolved-import]
    from e00686_false_events import seed_of  # ty: ignore[unresolved-import]

    panel, res, fz = setup()
    out: dict[str, int] = {}
    for k in range(2):
        world, _ = inject(
            panel=panel,
            residual=res,
            sigma=fz.sigma,
            rng=np.random.default_rng(seed_of(k=k)),
            lo=SCORED_FROM,
            hi=GRID_LENGTH,
            gen=GenParams(),
            kinds=("pv_growth",),
        )
        change = {s: np.nan_to_num(world[s] - res[s]) for s in world}
        site = max(change, key=lambda s: float(np.max(np.abs(change[s]))))
        days = np.arange(SCORED_FROM, GRID_LENGTH) // DAY
        hod = np.arange(SCORED_FROM, GRID_LENGTH) % DAY
        v = change[site][SCORED_FROM:GRID_LENGTH]
        rows = []
        for label, mask in (
            ("daylight (06:00 to 18:00 UTC)", (hod >= 12) & (hod < 36)),
            ("night (00:00 to 06:00 UTC)", hod < 12),
        ):
            d = np.unique(days[mask])
            m = np.array([v[mask & (days == x)].mean() for x in d])
            rows.append(pl.DataFrame({"day": (d - d[0]).astype(float), "mw": m, "block": label}))
        chart = (
            alt.Chart(pl.concat(rows))  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=0.9)
            .encode(
                x=alt.X("day:Q", title="Days from the start of the scored years"),
                y=alt.Y("mw:Q", title="Injected change (MW)"),
                color=alt.Color(
                    "block:N",
                    scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
            .properties(
                width=PLOT_WIDTH, height=140, title=f"Solar growth world {k}: Primary {site}"
            )
        )
        name = f"solar_growth_{k}.svg"
        out[name] = save_svg(chart=chart, path=folder / name)  # ty: ignore[invalid-argument-type]
    return out


def _cold_gallery(*, rig: object, folder: Path, seed: int, layers: object) -> dict[str, int]:
    """The additions of three cold spells at all 15 Primaries, the zero-loading Primaries marked."""
    add, spells = layers.cold_spells(rig=rig, rng=np.random.default_rng(seed + 7_340_000))  # ty: ignore[unresolved-attribute]
    out: dict[str, int] = {}
    lam0 = {s for s, v in add.items() if not np.any(v)}
    for k, r in enumerate(spells.head(3).iter_rows(named=True)):
        lo, hi = draw_window(
            start=r["start"], end=r["end"], record_end=len(next(iter(add.values())))
        )
        frames = []
        for s, v in add.items():
            idx = np.arange(lo, hi)
            xs, ys = thin(x=(idx - r["start"]) / DAY, y=v[idx], max_points=MAX_POINTS)
            frames.append(
                pl.DataFrame({"day": xs, "mw": ys, "primary": s, "zero_loading": s in lam0})
            )
        data = pl.concat(frames)
        chart = (
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=0.8)
            .encode(
                x=alt.X("day:Q", title="Days from the start of the cold spell"),
                y=alt.Y("mw:Q", title="Added demand (MW)"),
                detail="primary:N",
                color=alt.Color(
                    "zero_loading:N",
                    scale=alt.Scale(domain=[False, True], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                    legend=alt.Legend(title="Primary with zero loading", orient="top"),
                ),
            )
            .properties(
                width=PLOT_WIDTH, height=160, title=f"Cold spell {k}: added demand at all Primaries"
            )
        )
        name = f"cold_spell_{k}.svg"
        out[name] = save_svg(chart=chart, path=folder / name)  # ty: ignore[invalid-argument-type]
    return out


def _aggregates(*, out: Path, ctx: dec.Context, folder: Path) -> dict[str, int]:
    """Noise units, the shared term's explained variance, and the MVA unknown share per Primary."""
    import json

    stage0 = out / "stage0.json"
    if not stage0.exists():
        return {}
    data = json.loads(stage0.read_text())
    rows = [
        {
            "primary": s,
            "noise unit without (MW)": v["noise_unit_without"],
            "noise unit with the shared term (MW)": v["noise_unit_with"],
            "explained variance": v["explained_variance"],
            "shared term": "on" if v["shared_switch"] else "switched off by the per-Primary rule",
        }
        for s, v in data.items()
    ]
    df = pl.DataFrame(rows)
    off = df.filter(pl.col("shared term") != "on")["primary"].to_list()
    sizes: dict[str, int] = {}
    colour = alt.Color(
        "shared term:N",
        scale=alt.Scale(
            domain=["on", "switched off by the per-Primary rule"],
            range=[ocf.DATA_BLUE, ocf.ENSEMBLE_LINE],
        ),
        legend=alt.Legend(title=None, orient="bottom"),
    )
    for col, name, title in (
        (
            "noise unit without (MW)",
            "noise_units",
            "Noise unit of the arm C residual (MW)",
        ),
        (
            "explained variance",
            "explained_variance",
            "Share of the noise variance that the shared term removes. "
            + (
                f"Zero at {', '.join(off)}: the per-Primary rule switched the term off there "
                "(it lowers the noise unit by too little), so nothing is subtracted."
                if off
                else "The term is on at every Primary."
            ),
        ),
    ):
        chart = (
            alt.Chart(df)  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                x=alt.X("primary:N", sort=None, title="Primary"),
                y=alt.Y(f"{col}:Q", title=col),
                color=colour,
            )
            .properties(
                width=PLOT_WIDTH,
                height=140,
                title=alt.TitleParams(
                    text=name.replace("_", " ").capitalize(), subtitle=_wrap(text=title, width=105)
                ),
            )
        )
        sizes[f"{name}.svg"] = save_svg(chart=chart, path=folder / f"{name}.svg")  # ty: ignore[invalid-argument-type]
    return sizes


def scored_figures(*, out: Path, per_category: int = PER_GRADE) -> None:
    """Draw injected-world event figures for the scored worlds (after scoring only).

    Reads the match and call tables of H-nbr and its stored block quantiles, and draws each chosen
    event with the called events of E1-J-sym and PELT overlaid. The injected world is rebuilt from
    its seed, which decodes nothing. The default worlds give every category except the cascades,
    and the cascade worlds give the cascades.
    """
    import e00734_decode as dec_
    import e00734_report as rep
    import e00734_score as score
    import e00734_tasks as tasks

    ctx = dec_.load_context()
    for set_name in ("default", "cascade"):
        scored = out / "scored" / set_name
        matches = score.add_cells(rows=rep.read_files(folder=scored, arm="H-nbr"))
        calls = rep.read_files(folder=scored, arm="H-nbr", suffix="_calls")
        if not calls.height:
            continue
        picks = [
            p
            for p in pick_categories(matches=matches, calls=calls, per_category=per_category)
            if p["category"].startswith("cascade") == (set_name == "cascade")
        ]
        worlds: dict[int, dict] = {}
        for k, p in enumerate(picks):
            if p["world"] not in worlds:
                worlds[p["world"]] = tasks.world_for(set_name=set_name, seed=p["world"], ctx=ctx)
            w = worlds[p["world"]]
            if p["kind"] == "truth":
                rows = injected_rows(
                    table=w["table"], group=p["group"], site=p["primary"], start=p["start"]
                )
            else:
                rows = pl.DataFrame(
                    {"primary": [p["primary"]], "start": [p["start"]], "end": [p["end"]]}
                )
            sites = involved_primaries(
                site=p["primary"],
                start=p["start"],
                group_sites=rows["primary"].to_list(),
                calls=calls.filter(pl.col("world") == p["world"]),
            )
            overlays = {
                m: rep.read_files(folder=scored, arm=m, suffix="_calls").filter(
                    pl.col("world") == p["world"]
                )
                for m in ("E1-J-sym", "PELT")
            }
            src = EventSource(
                power=w["power"],
                record_end=dec_.record_end_of(power=w["power"]),
                blocks=_scored_blocks(scored=scored, world=p["world"]),
                calls={m: t for m, t in overlays.items() if t.height},
            )
            stem = p["category"].replace(", ", "_").replace(": ", "_").replace(" ", "_")
            save_within_size(
                make=partial(
                    _make,
                    event=rows,
                    sites=sites,
                    src=src,
                    shade=p["kind"],
                    title=f"{p['category']}: {set_name} world {p['world']}",
                ),
                path=out / "figures" / "scored" / set_name / f"{stem}_{p['world']}_{k}.svg",
            )


def _scored_blocks(*, scored: Path, world: int) -> dict[str, pl.DataFrame]:
    """H-nbr's stored block quantiles of one world, per Primary (empty if not stored)."""
    path = scored / f"blocks_H-nbr_{world}.parquet"
    if not path.exists():
        return {}
    return {k[0]: g for k, g in pl.read_parquet(path).group_by("site")}


def real_figures(*, out: Path, per_category: int = PER_REAL) -> None:
    """Draw event figures of the real record (after scoring only): called events and logged shifts.

    Primaries are named by D code, the demand is the Primary's real power, and the offset is the
    stored block table of H-nbr. E1-J-sym's stored table and PELT's table are overlaid.
    """
    import e00734_decode as dec_
    import e00734_tasks as tasks
    from e00653_common import E580, SCORED_FROM  # ty: ignore[unresolved-import]
    from e00653_inject import OFFSET_2019  # ty: ignore[unresolved-import]

    ctx = dec_.load_context(sigma_from="whole")
    rig = ctx.rig
    real = out / "real"
    calls = pl.read_parquet(real / "events_H-nbr.parquet")
    lab = pl.read_parquet(E580 / "events_labelled_status.parquet").filter(
        (pl.col("label_length_days") > 1)
        & pl.col("label").is_in(list(rig.p653.sites))
        & (pl.col("label_onset_idx") - OFFSET_2019 >= SCORED_FROM)
    )
    logged = pl.DataFrame(
        {
            "primary": lab["label"],
            "start": lab["label_onset_idx"] - OFFSET_2019,
            "end": lab["label_end_idx"] - OFFSET_2019,
        }
    )
    picks = pick_real(calls=calls, logged=logged, per_category=per_category)
    blocks_t = pl.read_parquet(real / "blocks_H-nbr.parquet")
    blocks = {k[0]: g for k, g in blocks_t.group_by("primary")}
    stored = pl.read_parquet(
        tasks.ML_EXPERIMENTS_DATA_DIR
        / "00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
        / "real_events_E1-J-sym.parquet"
    )
    overlays = {
        "E1-J-sym": stored.select("primary", "start", "end", "amp_mw"),
        "PELT": pl.read_parquet(real / "events_PELT.parquet").select(
            "primary", "start", "end", "amp_mw"
        ),
    }
    src = EventSource(
        power=dict(rig.panel.power),
        record_end=dec_.record_end_of(power=rig.panel.power),
        blocks=blocks,
        calls=overlays,
    )
    for k, p in enumerate(picks):
        rows = pl.DataFrame({"primary": [p["primary"]], "start": [p["start"]], "end": [p["end"]]})
        sites = involved_primaries(site=p["primary"], start=p["start"], group_sites=[], calls=calls)
        name = f"real_{p['category'].replace(', ', '_').replace(' ', '_')}_{k}.svg"
        save_within_size(
            make=partial(
                _make,
                event=rows,
                sites=sites,
                src=src,
                shade="called",
                title=f"Real record, {p['category']}: Primary {p['primary']}",
            ),
            path=out / "figures" / "real" / name,
        )

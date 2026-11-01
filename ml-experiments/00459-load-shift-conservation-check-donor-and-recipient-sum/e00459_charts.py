# ruff: noqa: E501
"""Draw experiment 459's figures: six events, the leak-share histograms, and the share over time."""

from pathlib import Path
from typing import Final

import altair as alt
import e00459_conservation as cons  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00459-load-shift-conservation-check-donor-and-recipient-sum"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS: Final[Path] = Path(__file__).parents[2] / "docs" / "ml-experiments" / SLUG
MARGIN_HALF_HOURS: Final[int] = 3 * 48
SMOOTH: Final[int] = 12
"""Half-hours in the rolling mean that smooths each plotted deviation (6 hours)."""
MEMBER_ORDER: Final[list[str]] = ["D5", "D10", "D16", "D21", "D22"]
CLIP: Final[float] = 3.0
"""Leak shares above this value are drawn in the last bin."""


def pick(*, frame: pl.DataFrame, column: str) -> list[int]:
    """Return the event rows at the 25th, 50th, and 75th percentile of `column`."""
    ordered = frame.sort(column)
    n = ordered.height
    return [ordered["row"][int(q * (n - 1))] for q in (0.25, 0.5, 0.75)]


def smooth(values: np.ndarray) -> np.ndarray:
    """Return a rolling mean that ignores missing half-hours."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - SMOOTH + 1) : i + 1]
        if np.isfinite(chunk).sum() >= SMOOTH // 2:
            out[i] = np.nanmean(chunk)
    return out


def event_figure(*, grid: cons.Grid, events: list[dict], rows: pl.DataFrame) -> None:
    """Draw three observed-recipient and three unobserved-recipient events at the six."""
    included = rows.filter(
        (pl.col("kind") == "event") & (pl.col("reason") == "") & (pl.col("group") == "six")
    )
    chosen = [(r, True) for r in pick(frame=included.filter(pl.col("observed")), column="g1_leak")]
    chosen += [
        (r, False) for r in pick(frame=included.filter(~pl.col("observed")), column="g2_leak")
    ]
    by_row = {e["row"]: e for e in events if e["usable"]}
    panels = []
    for number, (row, observed) in enumerate(chosen, start=1):
        e = by_row[row]
        lo, hi = e["onset"] - MARGIN_HALF_HOURS, e["end"] + MARGIN_HALF_HOURS
        members = e["g1"] if observed else e["g2"]
        dev = cons.deviation(grid=grid, members=members, onset=e["onset"], stop=hi)
        pre = cons.deviation(
            grid=grid, members=members, onset=lo, stop=e["onset"]
        )  # 3 days before, own baseline
        both = np.vstack([pre, dev])
        days = (np.arange(lo, hi) - e["onset"]) / 48.0
        series = {f"{e['donor']} (donor)": both[:, 0]}
        if observed:
            series[f"{members[1]} (recipient)"] = both[:, 1]
            series["Sum of the donor and its recipients"] = both.sum(axis=1)
        else:
            series["Sum of the donor and its graph neighbours"] = both.sum(axis=1)
        frame = pl.concat(
            [
                pl.DataFrame({"day": days, "deviation_mw": smooth(v), "series": name})
                for name, v in series.items()
            ]
        ).drop_nans("deviation_mw")
        span = pl.DataFrame({"a": [0.0], "b": [(e["end"] - e["onset"]) / 48.0]})
        shade = alt.Chart(span).mark_rect(color=ocf.GREY_3, opacity=0.5).encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
        lines = (
            alt.Chart(frame)  # ty: ignore[unresolved-attribute]
            .mark_line(clip=True)
            .encode(
                x=alt.X("day:Q", title="Days since the labelled onset"),
                y=alt.Y("deviation_mw:Q", title="Deviation from baseline (MW or MVA)"),
                color=alt.Color(
                    "series:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=list(series),
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.BLACK_1]
                        if observed
                        else [ocf.DATA_BLUE, ocf.BLACK_1],
                    ),
                ),
            )
        )
        title = alt.TitleParams(
            [
                f"Event {number}: donor {e['donor']}, {'observed' if observed else 'unobserved'}, {e['onset_utc'][:10]}"
            ],
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        )
        panels.append(
            (shade + lines).properties(width=(CONTENT_WIDTH_PX - 170) // 3, height=130, title=title)
        )
    alt.vconcat(alt.hconcat(*panels[:3]), alt.hconcat(*panels[3:])).properties(
        title=alt.TitleParams(
            [
                "Six Primaries with signed-MW readings: deviation from the 7-day baseline, smoothed over 6 hours.",
                "Top row: events at the 25th, 50th, and 75th percentile of the G1 leak share. Bottom row: the same for G2 where no recipient is among the 16.",
                "Blue: the donor. Orange: the first recipient (top row). Black: the sum (G1 on top, G2 below). Shaded: the labelled event. NGED's event labels are incomplete.",
            ],
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        )
    ).save(DOCS / "figure_events.svg")


def histogram_figure(*, rows: pl.DataFrame) -> None:
    """Draw the leak-share histogram of G1 against the placebo, at the six and at the eight."""
    parts = []
    for metering in ("six", "eight"):
        events = rows.filter(
            (pl.col("kind") == "event")
            & (pl.col("reason") == "")
            & pl.col("observed")
            & (pl.col("group") == metering)
        )
        placebo = rows.filter(
            (pl.col("kind") == "placebo") & pl.col("row").is_in(events["row"].to_list())
        )
        for kind, frame in (
            ("Labelled events", events),
            ("Placebo windows", placebo),
        ):
            parts.append(
                frame.select(
                    leak=pl.col("g1_leak").clip(upper_bound=CLIP),
                    kind=pl.lit(kind),
                    metering=pl.lit(
                        "Primaries with signed-MW readings"
                        if metering == "six"
                        else "Primaries with MVA-only readings"
                    ),
                ).drop_nans("leak")
            )
    data = pl.concat(parts)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "leak:Q",
                bin=alt.Bin(step=0.25, extent=[0, CLIP + 0.25]),
                title=f"Leak share of the donor-plus-recipients sum (last bin: {CLIP} or more)",
            ),
            y=alt.Y("count()", stack=None, title="Share of windows"),
            color=alt.Color(
                "kind:N",
                legend=alt.Legend(title=None, orient="bottom"),
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
            ),
        )
        .transform_joinaggregate(total="count()", groupby=["metering", "kind"])
        .transform_calculate(share="1 / datum.total")
        .encode(y=alt.Y("sum(share):Q", stack=None, title="Share of windows"))
        .properties(width=CONTENT_WIDTH_PX - 120, height=140)
        .facet(
            row=alt.Row(
                "metering:N",
                title=None,
                sort=["Primaries with signed-MW readings", "Primaries with MVA-only readings"],
            )
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Leak share of group G1: events with a recipient among the 16 Primaries, against 20 placebo windows per event.",
                    "A leak share of 0 means the donor's step vanishes from the sum; 1 means the sum shows the whole step.",
                    "Placebo windows lie outside labelled events, not outside all events: NGED's labels are incomplete.",
                ],
                anchor="start",
                fontSize=12,
                fontWeight="normal",
            )
        )
    )
    chart.save(DOCS / "figure_leak_histogram.svg")


def share_figure(*, grid: cons.Grid, events: list[dict]) -> None:
    """Draw the 30-day trailing median share of each member for the three groups with most events."""
    counts: dict[tuple[str, ...], int] = {}
    for e in events:
        if e["usable"] and e["observed"] and e["group"] == "six":
            key = tuple(sorted(e["g1"]))
            counts[key] = counts.get(key, 0) + 1
    top = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]]
    forbidden = cons.forbidden_mask(grid=grid, events=events)
    stamps = pl.datetime_range(  # ty: ignore[no-matching-overload]
        grid.t0,
        grid.t0 + np.timedelta64((grid.n - 1) * 30, "m"),  # ty: ignore[unsupported-operator]
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    panels = []
    for members in top:
        block = grid.x[:, [grid.column[m] for m in members]]
        total = block.sum(axis=1)
        ok = np.isfinite(total) & (total > 0.1 * sum(grid.capacity[m] for m in members))
        frames = []
        for c, m in enumerate(members):
            share = np.where(ok, block[:, c] / np.where(ok, total, 1), np.nan)
            med = cons.trailing_share(share=share, usable=ok & ~forbidden)
            day_times = stamps.gather_every(48)
            frames.append(
                pl.DataFrame({"day": day_times, "share": med[: day_times.len()], "member": m})
            )
        data = pl.concat(frames).drop_nans("share")
        panels.append(
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_line(clip=True)
            .encode(
                x=alt.X("day:T", title=None),
                y=alt.Y("share:Q", title="30-day trailing median share"),
                color=alt.Color(
                    "member:N",
                    legend=alt.Legend(title=None, orient="right"),
                    scale=alt.Scale(
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE]
                    ),
                ),
            )
            .properties(
                width=CONTENT_WIDTH_PX - 200,
                height=110,
                title=alt.TitleParams(
                    [f"Group {' + '.join(members)}"],
                    anchor="start",
                    fontSize=12,
                    fontWeight="normal",
                ),
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Each Primary's share of its group sum, outside labelled events (the six's three most frequent groups)."
            ],
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        )
    ).save(DOCS / "figure_share_over_time.svg")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    g = cons.Grid()
    graph = {k: tuple(v) for k, v in cons.neighbours.nged_neighbours()[0].items()}
    evs = cons.read_events(grid=g, graph=graph)
    table = pl.read_parquet(WORK / "event_and_placebo_rows.parquet")
    event_figure(grid=g, events=evs, rows=table)
    histogram_figure(rows=table)
    share_figure(grid=g, events=evs)

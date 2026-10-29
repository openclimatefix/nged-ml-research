"""Draw the figures of experiment 390's write-up from the tables `e00390_build.py` wrote.

The events of the fan figure are chosen by a rule fixed before any event was inspected: at
each of D5, D14, and D16, the event with the largest matched depth and the event with the
longest duration, among the events that end on or after 2025-07-01 (D11 has none, so D16,
the next signed-MW Primary with two events, replaces it; the choice used event counts
only); plus the event with the most rows that carry a lag inside it, among the six signed-
MW Primaries. An event picked twice is drawn once.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00390-xgboost-d11-d14-gap-and-load-shift-event-lags \
    uv run --no-sync python ml-experiments/00390-*/e00390_charts.py
```
"""

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00390_report as rep  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00390-xgboost-d11-d14-gap-and-load-shift-event-lags"
)
WORK: Final[Path] = rep.WORK
WIDTH: Final[int] = 680
WRAP: Final[int] = 110
SCORED_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
PICK_PRIMARIES: Final[tuple[str, ...]] = ("D5", "D14", "D16")
HOURS_AHEAD: Final[tuple[float, float]] = (24.0, 48.0)


def title_params(title: str, subtitle: str) -> alt.TitleParams:
    """Return a title wrapped to the plot width."""
    return alt.TitleParams(
        text=textwrap.wrap(title, 85), subtitle=textwrap.wrap(subtitle, WRAP), anchor="start"
    )


def figure_gap_split(table: pl.DataFrame) -> None:
    """Draw the bundle-minus-stack gap at D11 and D14 by season and by demand quartile."""
    stack = rep.load("stack", table).filter(pl.col(rep.HEADLINE))
    bundle = rep.load("bundle", table).filter(pl.col(rep.HEADLINE))
    joined = (
        bundle.filter(pl.col("label").is_in(rep.GAP_PRIMARIES))
        .join(stack.select(*rep.KEYS, stack_crps="crps"), on=rep.KEYS)
        .with_columns(
            gap=pl.col("crps") - pl.col("stack_crps"),
            month=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.month(),
        )
        .with_columns(
            season=pl.col("month").replace_strict(
                {
                    12: "Winter",
                    1: "Winter",
                    2: "Winter",
                    3: "Spring",
                    4: "Spring",
                    5: "Spring",
                    6: "Summer",
                    7: "Summer",
                    8: "Summer",
                    9: "Autumn",
                    10: "Autumn",
                    11: "Autumn",
                }
            ),
            quartile=pl.col("actual_points")
            .bin_quantiles([0.25, 0.5, 0.75], labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
            .over("time_series_id")
            .cast(pl.String),
        )
    )
    parts = []
    for col, kind in (("season", "Season"), ("quartile", "Demand quartile")):
        parts.append(
            joined.group_by("label", col)
            .agg(gap=pl.col("gap").mean())
            .rename({col: "slice"})
            .with_columns(kind=pl.lit(kind))
        )
    data = pl.concat(parts)
    panels = []
    for i, (kind, order) in enumerate(
        (
            ("Season", ["Winter", "Spring", "Summer", "Autumn"]),
            ("Demand quartile", ["Q1 lowest", "Q2", "Q3", "Q4 highest"]),
        )
    ):
        panels.append(
            alt.Chart(data.filter(pl.col("kind") == kind))
            .mark_bar()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("slice:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("label:N"),
                y=alt.Y(
                    "gap:Q",
                    title="Bundle minus stack CRPS (points)" if i == 0 else None,
                    scale=alt.Scale(domain=[-0.2, 1.2]),
                ),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(domain=["D11", "D14"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                    legend=alt.Legend(title=None, orient="top") if i == 0 else None,
                ),
            )
            .properties(width=WIDTH // 2 - 90, height=190, title=kind)
        )
    chart = alt.hconcat(*panels).properties(
        title=title_params(
            "The gap at D11 and D14 sits in winter and in the highest-demand quarter of half-hours",
            "Weekly bundle minus GAM stack, mean CRPS at days 1 to 10, shared half-hours of "
            "2025-07 to 2026-06. Positive means the bundle is worse. Demand quartiles are of "
            "each Primary's own readings. Seasons are by London calendar month.",
        )
    )
    chart.save(DOCS_DIR / "gap_split.svg")


def tail_by_phase(table: pl.DataFrame) -> None:
    """Draw q95 coverage by phase relative to the event, for three forecasts."""
    events = pl.read_parquet(WORK / "events.parquet").select("event_id", "start", "end")
    out = []
    for name, label in (("bundle", "Weekly bundle"), ("bg2_only_W", "Weekly BG2 member"),
                        ("stack", "GAM stack")):  # fmt: skip
        rows = rep.load(name, table).filter(pl.col(rep.HEADLINE))
        clean = rows.filter(~pl.col("lag_in_event"))
        out.append({"arm": label, "phase": "Clean lags", "mean": clean["coverage_q95"].mean(),
                    "lower": clean["coverage_q95"].mean(), "upper": clean["coverage_q95"].mean(),
                    "events": 0})  # fmt: skip
        lagged = (
            rows.filter(pl.col("lag_in_event"))
            .join(events.rename({"event_id": "lag_event_id"}), on="lag_event_id", how="left")
            .with_columns(
                hours=(pl.col("power_fcst_init_time") - pl.col("start")).dt.total_minutes() / 60,
                event_id=pl.col("lag_event_id"),
            )
            .with_columns(
                phase=pl.when(pl.col("power_fcst_init_time") > pl.col("end"))
                .then(pl.lit("After the end"))
                .when(pl.col("hours") < 24)
                .then(pl.lit("First 24 hours"))
                .otherwise(pl.lit("1 to 7 days in"))
            )
        )
        for phase in ("First 24 hours", "1 to 7 days in", "After the end"):
            part = lagged.filter(pl.col("phase") == phase)
            means, ints, n_events = rep.cluster_interval(part, ["coverage_q95"])
            out.append({"arm": label, "phase": phase, "mean": means["coverage_q95"],
                        "lower": ints["coverage_q95"][0], "upper": ints["coverage_q95"][1],
                        "events": n_events})  # fmt: skip
    data = pl.DataFrame(out)
    data.write_csv(WORK / "tail_by_phase.csv")
    order = ["Clean lags", "First 24 hours", "1 to 7 days in", "After the end"]
    colours = {"Weekly bundle": ocf.DATA_BLUE, "Weekly BG2 member": ocf.DATA_SKY,
               "GAM stack": ocf.BRAND_ORANGE}  # fmt: skip
    target = (
        alt.Chart(pl.DataFrame({"y": [0.95]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0.4, 1])))
    )  # ty: ignore[unresolved-attribute]
    base_chart = alt.Chart(data).encode(
        x=alt.X("phase:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=150)),
        xOffset=alt.XOffset("arm:N"),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    bars = base_chart.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "lower:Q",
            title="Share of readings at or below q95",
            scale=alt.Scale(domain=[0.4, 1], clamp=True),
        ),
        y2="upper:Q",
    )
    dots = base_chart.mark_point(filled=True, size=60).encode(y="mean:Q")  # ty: ignore[unresolved-attribute]
    chart = (bars + dots + target).properties(
        width=WIDTH - 90,
        height=230,
        title=title_params(
            "When a lag lies inside a load-shift event, the weekly bundle's q95 covers far fewer "
            "readings than the 95% it should",
            "Share of readings at or below the forecast's q95, days 1 to 10, all 16 Primaries. "
            "Dashed line: 95%, the calibrated value. Dot: estimate. Line: 95% interval from "
            "resampling whole events (no line for clean lags: over 1.9 million rows).",
        ),
    )
    chart.save(DOCS_DIR / "tail_by_phase.svg")


def pick_events() -> pl.DataFrame:
    """Return the events to draw, chosen by the fixed rule in the module docstring."""
    events = pl.read_parquet(WORK / "events.parquet").filter(
        pl.col("end") >= SCORED_START, pl.col("start") < datetime(2026, 7, 1, tzinfo=UTC)
    )
    picks = []
    for label in PICK_PRIMARIES:
        mine = events.filter(pl.col("label") == label)
        deep = mine.filter(pl.col("depth_share").is_not_null()).sort("depth_share", descending=True)
        if deep.height:
            picks.append((deep["event_id"][0], "deepest"))
        picks.append((mine.sort("duration_hours", descending=True)["event_id"][0], "longest"))
    flags = pl.read_parquet(WORK / "event_flags.parquet")
    most = (
        flags.filter(pl.col("lag_in_event"))
        .join(events.select("event_id", "label"), left_on="lag_event_id", right_on="event_id")
        .filter(pl.col("label").is_in(rep.SIGNED_SIX))
        .group_by("lag_event_id")
        .len()
        .sort("len", descending=True)
    )
    picks.append((most["lag_event_id"][0], "most rows with a lag in the event"))
    chosen: dict[int, str] = {}
    for event_id, why in picks:
        chosen[event_id] = f"{chosen[event_id]}; {why}" if event_id in chosen else why
    names = pl.DataFrame({"event_id": list(chosen), "why": list(chosen.values())})
    return events.join(names, on="event_id").sort("label", "start")


def figure_fans(table: pl.DataFrame) -> None:
    """Draw the day-ahead fan of the weekly bundle and the GAM stack through each chosen event."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    picked = pick_events()
    picked.write_csv(WORK / "fan_events.csv")
    bundle = s381.arm_frame(regime="weekly", arm="bundle_solar+sunny+hl1", series=series)
    stack = base.stack_frames(series=series)["stack"]
    pieces = []
    panel_names = []
    for row in picked.iter_rows(named=True):
        lo = row["start"] - timedelta(days=1)
        hi = row["end"] + timedelta(days=7)
        sid = row["time_series_id"]
        panel = f"{row['label']}: {row['why']} ({row['duration_hours'] / 24:.0f} days)"
        panel_names.append(panel)
        actual = ctx["actuals"].filter(
            pl.col("time_series_id") == sid, pl.col("valid_time").is_between(lo, hi)
        )
        for name, frame in (("Weekly bundle", bundle), ("GAM stack", stack)):
            fan = (
                frame.filter(
                    pl.col("time_series_id") == sid,
                    pl.col("valid_time").is_between(lo, hi),
                    (
                        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                        / 60
                    ).is_between(HOURS_AHEAD[0], HOURS_AHEAD[1], closed="left"),
                )
                .select("valid_time", "q05", "q50", "q95")
                .with_columns(arm=pl.lit(name), panel=pl.lit(panel))
            )
            pieces.append(fan)
        pieces.append(
            actual.select("valid_time", q50=pl.col("actual")).with_columns(
                arm=pl.lit("Reading"), panel=pl.lit(panel)
            )
        )
    data = pl.concat(pieces, how="diagonal")
    colours = {"Weekly bundle": ocf.DATA_BLUE, "GAM stack": ocf.BRAND_ORANGE, "Reading": "#202020"}
    scale = alt.Scale(domain=list(colours), range=list(colours.values()))
    panels = []
    for i, panel in enumerate(panel_names):
        mine = data.filter(pl.col("panel") == panel)
        bands = mine.filter(pl.col("arm") != "Reading")
        fans = (
            alt.Chart(bands)
            .mark_area(opacity=0.28)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("valid_time:T", title=None),
                y=alt.Y("q05:Q", title="MW"),
                y2="q95:Q",
                color=alt.Color(
                    "arm:N",
                    scale=scale,
                    legend=alt.Legend(title=None, orient="top") if i == 0 else None,
                ),
            )
        )
        medians = (
            alt.Chart(bands)
            .mark_line(strokeWidth=1)
            .encode(  # ty: ignore[unresolved-attribute]
                x="valid_time:T", y="q50:Q", color=alt.Color("arm:N", scale=scale, legend=None)
            )
        )
        reading = (
            alt.Chart(mine.filter(pl.col("arm") == "Reading"))
            .mark_line(strokeWidth=1.2, color="#202020")
            .encode(x="valid_time:T", y="q50:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(fans, medians, reading).properties(
                width=WIDTH - 80,
                height=105,
                title=alt.TitleParams(panel, anchor="start", fontSize=11),
            )
        )
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(x="independent", y="independent", color="shared")
        .properties(
            title=title_params(
                "Day-ahead forecasts through four hand-labelled load-shift events",
                "Black line: meter reading. Shaded band: q05 to q95 of the forecast made 24 to 48 "
                "hours earlier. Line: its median. Events were chosen by a rule fixed before any "
                "was inspected (deepest and longest at D5, D14, and D16; most rows with a lagged "
                "reading inside the event). Each panel runs from 1 day before the event to 7 days "
                "after its end.",
            )
        )
    )
    chart.save(DOCS_DIR / "event_fans.svg")


def main() -> None:
    """Write the SVG files."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    table = pl.read_parquet(WORK / "slice_table.parquet")
    figure_gap_split(table)
    tail_by_phase(table)
    figure_fans(table)


if __name__ == "__main__":
    main()

"""Experiment 401: time series plots of real load-shift events and how each forecast responds.

For each of 3 events at the signed-MW six Primaries and 3 at the MVA-only eight, the plot shows the
actual reading, the median and 90% band (q05 to q95) of the forecast with lags (experiment 381's
weekly bundle), the forecast without lags (experiment 383's raw lagless arm), and the fixed 50/50
mixture of the two (this experiment's `mix_w50`), with the event's corrected start and end shaded.
Each valid time shows the forecast issued with the shortest lead of at least 24 hours (day ahead).
Events are the corrected label table's unpadded events (Part F intervals, corrected depth), chosen
as the 3 per group with the largest labelled depth as a share of capacity among those of 1 to 7 days
that fall in the scored months. Demand Primaries only, so no generator is shown.

```bash
PYTHONPATH=<as in e00401_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00401-*/e00401_event_plots.py
```
"""
# ruff: noqa: E501

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00400_blend as blend  # ty: ignore[unresolved-import]
import e00401_arms as arms
import e00401_events as ev
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / "00401-real-and-synthetic-load-shift-median-share-and-fixed-forecast-mixture"
)  # fmt: skip
SIX: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
EIGHT: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
SCORED_FROM: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
WIDTH: Final[int] = 680
PANEL_HEIGHT: Final[int] = 120
ARMS: Final[dict[str, tuple[str, str]]] = {
    "Forecast with power lags": ("lags", ocf.DATA_BLUE),
    "Forecast without power lags": ("lagless", ocf.BRAND_ORANGE),
    "50/50 mixture of the two": ("mix_w50", ocf.DATA_GREEN),
}


def day_ahead(*, name: str, label: str, start: datetime, end: datetime) -> pl.DataFrame:
    """Return, per valid time, the forecast with the shortest lead of at least 24 hours (MW)."""
    path = (
        blend.WORK / f"rows_{name}.parquet"
        if name in ("lagless", "lags")
        else arms.WORK / f"rows_{name}.parquet"
    )
    return (
        pl.read_parquet(path, columns=["label", "valid_time", "lead_hours", "flagged", "q05_mw", "q50_mw", "q95_mw", "actual_mw"])
        .filter(pl.col("label") == label, pl.col("valid_time").is_between(start, end), pl.col("lead_hours") >= 24, ~pl.col("flagged"))
        .sort("valid_time", "lead_hours")
        .unique("valid_time", keep="first", maintain_order=True)
    )  # fmt: skip


def pick_events() -> pl.DataFrame:
    """Return the 3 events per group to draw."""
    events = ev.events_with_depths()
    grid = common.grid_times()
    chosen = []
    for group, labels in (("signed-MW six", SIX), ("MVA-only eight", EIGHT)):
        sub = events.filter(pl.col("label").is_in(labels), pl.col("length_days").is_between(1, 7)).with_columns(
            onset=pl.Series([grid[i] for i in events.filter(pl.col("label").is_in(labels), pl.col("length_days").is_between(1, 7))["onset_idx"]]),
            end=pl.Series([grid[i] for i in events.filter(pl.col("label").is_in(labels), pl.col("length_days").is_between(1, 7))["end_idx"]]),
        )  # fmt: skip
        sub = sub.filter(pl.col("onset") >= SCORED_FROM + timedelta(days=3)).sort(
            "depth_share", descending=True
        )
        # One event per Primary so the three plots show three Primaries.
        chosen.append(
            sub.unique("label", keep="first", maintain_order=True)
            .head(3)
            .with_columns(group=pl.lit(group))
        )
    return pl.concat(chosen)


def draw(*, event: dict) -> alt.VConcatChart:
    """Return the stacked panels of one event."""
    start = event["onset"] - timedelta(days=2)
    end = event["end"] + timedelta(days=2)
    span = pl.DataFrame({"s": [event["onset"]], "e": [event["end"]]})
    panels = []
    for title, (name, colour) in ARMS.items():
        frame = day_ahead(name=name, label=event["label"], start=start, end=end)
        scale = alt.Scale(domain=[start, end])
        base = alt.Chart(frame).encode(
            x=alt.X(
                "valid_time:T",
                scale=scale,
                axis=alt.Axis(title=None, format="%d %b", tickCount="day", labelOverlap=True),
            )
        )
        band = base.mark_area(opacity=0.25, color=colour).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05_mw:Q", title="MW"), y2="q95_mw:Q"
        )
        median = base.mark_line(color=colour, strokeWidth=1.5).encode(y="q50_mw:Q")  # ty: ignore[unresolved-attribute]
        actual = base.mark_line(color=ocf.BLACK_1, strokeWidth=1).encode(y="actual_mw:Q")  # ty: ignore[unresolved-attribute]
        rect = alt.Chart(span).mark_rect(color=ocf.GREY_3, opacity=0.35).encode(x="s:T", x2="e:T")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(rect, band, median, actual).properties(
                width=WIDTH - 80, height=PANEL_HEIGHT, title=title
            )
        )
    sign = "drop" if event["corrected_depth_mw"] < 0 else "rise"
    return (
        alt.vconcat(*panels)
        .properties(
            title=alt.TitleParams(
                f"Primary {event['label']} ({event['group']}): labelled load shift, a {sign} of "
                f"{abs(event['corrected_depth_mw']):.1f} MW ({event['depth_share']:.0f}% of capacity)",
                subtitle=[
                    "Black: actual reading. Line and band: median and 90% band of the day-ahead forecast (lead of at least 24 hours).",
                    "Grey shading: the event, from its labelled start to its labelled end.",
                ],
            )
        )
        .resolve_scale(y="independent")
    )


def main() -> None:
    """Write one SVG per event."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for i, event in enumerate(pick_events().iter_rows(named=True), start=1):
        chart = draw(event=event)
        out = DOCS_DIR / f"event_{i}_{event['label']}.svg"
        chart.save(out)
        print("wrote", out.name, event["label"], event["onset"], event["end"], flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

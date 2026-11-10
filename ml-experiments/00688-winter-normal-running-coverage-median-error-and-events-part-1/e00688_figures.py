"""Figures of experiment 688 (CPU only), written as SVG to the folder given as the first argument.

PYTHONPATH=ml-experiments/00668-*:ml-experiments/00683-*:ml-experiments/00688-* \
    DATA_PATH_INTERNAL=<production data folder> \
    uv run --no-sync python e00688_figures.py <output folder>

Reads the tables written by `e00688_tables.py`. Every panel is at least 280 px wide, every line is
1 px, and a light grey rectangle marks a labelled or verified load transfer. Substations are
named by their time series id.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00688_diagnose as dg  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

STAGE_COLOURS: Final[dict[str, str]] = {
    "none": "#7F7F7F",
    "events": "#17E58F",
    "shift": "#FF4901",
    "events+shift": "#306BFF",
}
STAGE_NAMES: Final[dict[str, str]] = {
    "none": "arm 2 as it is",
    "events": "step 1: drop transfers",
    "shift": "step 2: re-centre",
    "events+shift": "steps 1 and 2",
}
STAGE_SCALE = alt.Scale(domain=list(STAGE_NAMES.values()), range=list(STAGE_COLOURS.values()))
GREY: Final[str] = "#D3D3D3"
HEIGHT: Final[int] = 280


def rule(*, value: float, dash: bool) -> alt.Chart:
    """Return a horizontal reference line at `value` percent."""
    return (
        alt.Chart(pl.DataFrame({"y": [value]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=1, color="#000000", strokeDash=[4, 3] if dash else [1, 0])
        .encode(y="y:Q")
    )


def coverage_by_month(*, tables: Path, out: Path) -> None:
    """Headline: pooled arm 2 coverage per origin month for each step, with the 0.86 stop line."""
    cov = pl.read_csv(tables / "coverage_by_stage.csv").filter(
        (pl.col("set") == "scaled")
        & pl.col("stage").is_in(list(STAGE_COLOURS))
        & pl.col("period").str.starts_with("20")
        & ~pl.col("period").str.contains("to")
    )
    cov = cov.with_columns(stage_name=pl.col("stage").replace_strict(STAGE_NAMES))
    base = alt.Chart(cov.to_pandas())  # ty: ignore[unresolved-attribute]
    x = alt.X("period:O", title="Origin month", axis=alt.Axis(labelAngle=-45))
    line = base.mark_line(strokeWidth=1, point=True).encode(  # ty: ignore[unresolved-attribute]
        x=x,
        y=alt.Y("coverage:Q", title="q05-to-q95 coverage (%)", scale=alt.Scale(domain=[70, 100])),
        color=alt.Color("stage_name:N", title="Stage", scale=STAGE_SCALE),
    )
    err = base.mark_errorbar(ticks=True).encode(  # ty: ignore[unresolved-attribute]
        x=x,
        y=alt.Y("coverage_lo:Q", title="q05-to-q95 coverage (%)"),
        y2="coverage_hi:Q",
        color=alt.Color("stage_name:N", scale=STAGE_SCALE, legend=None),
    )
    chart = (line + err + rule(value=86.0, dash=True) + rule(value=90.0, dash=False)).properties(
        width=620, height=HEIGHT
    )
    chart.save(out / "coverage_by_month.svg")


def series_coverage(*, tables: Path, out: Path) -> None:
    """Coverage of each Primary over the two winter months, before and after each step."""
    ser = pl.read_csv(tables / "series_coverage.csv").filter(pl.col("period") == "winter")
    ser = ser.with_columns(
        stage_name=pl.col("stage").replace_strict(STAGE_NAMES),
        primary=pl.col("time_series_id").cast(pl.String),
    )
    order = ser.filter(pl.col("stage") == "none").sort("coverage")["primary"].to_list()
    base = alt.Chart(ser.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("primary:N", title="Primary (time series id)", sort=order),
        color=alt.Color("stage_name:N", title="Stage", scale=STAGE_SCALE),
    )
    dots = base.mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "coverage:Q",
            title="q05-to-q95 coverage, 2025-12 and 2026-01 (%)",
            scale=alt.Scale(domain=[30, 100]),
        )
    )
    bars = base.mark_rule(strokeWidth=1).encode(x="coverage_lo:Q", x2="coverage_hi:Q")  # ty: ignore[unresolved-attribute]
    vlines = (
        alt.Chart(pl.DataFrame({"x": [86.0, 90.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=1, color="#000000", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    (bars + dots + vlines).properties(width=620, height=16 * 28).save(out / "series_coverage.svg")


def week_frame(
    *, rows: dict[str, np.ndarray], evt: np.ndarray, series: int, start: datetime
) -> pl.DataFrame:
    """Return the day-ahead forecasts (lead 12 to 36 hours) of one series over seven days."""
    first, last = dg.idx_of(start), dg.idx_of(start + timedelta(days=7))
    sel = (
        (rows["series"] == series)
        & (rows["vidx"] >= first)
        & (rows["vidx"] < last)
        & (rows["lead_h"] >= 12.0)
        & (rows["lead_h"] < 36.0)
    )
    parts = []
    for stage in ("none", "events+shift"):
        q, _ = dg.stage_quantiles(rows=rows, which="scaled", stage=stage, evt=evt)
        parts.append(
            pl.DataFrame(
                {
                    "valid_time": [
                        lib.GRID_START + timedelta(minutes=30 * int(v)) for v in rows["vidx"][sel]
                    ],
                    "q05": q[sel, 1],
                    "q25": q[sel, 3],
                    "q50": q[sel, 5],
                    "q75": q[sel, 7],
                    "q95": q[sel, 9],
                    "y": rows["y"][sel],
                    "stage": STAGE_NAMES[stage],
                }
            )
        )
    return pl.concat(parts)


def fan(*, frame: pl.DataFrame, spans: pl.DataFrame, title: str, width: int) -> alt.Chart:
    """Return the 90% band, 50% band, median, and actual of `frame`, one panel per stage."""
    base = alt.Chart().encode(x=alt.X("valid_time:T", title="Valid time (UTC)"))
    band90 = base.mark_area(opacity=0.25, color="#306BFF").encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q05:Q", title="Power (% of capacity)"), y2="q95:Q"
    )
    band50 = base.mark_area(opacity=0.35, color="#306BFF").encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
    median = base.mark_line(strokeWidth=1, color="#306BFF").encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
    actual = base.mark_line(strokeWidth=1, color="#FF4901").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    layers = [band90, band50, median, actual]
    if spans.height > 0:
        rect = (
            alt.Chart(spans.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rect(color=GREY, opacity=0.6)
            .encode(x="start:T", x2="end:T")
        )
        layers = [rect, *layers]
    return alt.layer(*layers, data=frame.to_pandas()).properties(  # ty: ignore[invalid-return-type]
        width=width, height=HEIGHT, title=title
    )


def spans_in(*, spans: pl.DataFrame, series: int, lo: datetime, hi: datetime) -> pl.DataFrame:
    """Return the event spans of one series that overlap [lo, hi), clipped, as datetimes."""
    rows = []
    for r in spans.filter(pl.col("time_series_id") == series).iter_rows(named=True):
        start = lib.GRID_START + timedelta(minutes=30 * r["start"])
        end = lib.GRID_START + timedelta(minutes=30 * r["end"])
        if end > lo and start < hi:
            rows.append({"start": max(start, lo), "end": min(end, hi)})
    return pl.DataFrame(
        rows, schema={"start": pl.Datetime("us", "UTC"), "end": pl.Datetime("us", "UTC")}
    )


def example_weeks(
    *, rows: dict[str, np.ndarray], evt: np.ndarray, spans: pl.DataFrame, out: Path
) -> None:
    """Day-ahead forecasts over one week, before and after re-centring, for two Primaries."""
    cases = [(17, datetime(2026, 1, 10, tzinfo=UTC)), (13, datetime(2026, 1, 20, tzinfo=UTC))]
    charts = []
    for series, start in cases:
        frame = week_frame(rows=rows, evt=evt, series=series, start=start)
        spans_s = spans_in(spans=spans, series=series, lo=start, hi=start + timedelta(days=7))
        for stage in ("arm 2 as it is", "steps 1 and 2"):
            sub = frame.filter(pl.col("stage") == stage)
            charts.append(
                fan(frame=sub, spans=spans_s, title=f"Primary {series}: {stage}", width=300)
            )
    alt.concat(*charts, columns=2).save(out / "example_week_day_ahead.svg")


def example_fan_14_days(
    *, rows: dict[str, np.ndarray], evt: np.ndarray, spans: pl.DataFrame, out: Path
) -> None:
    """One forecast of the full 14-day horizon, before and after re-centring."""
    origin = datetime(2026, 1, 14, 12, tzinfo=UTC)
    first = dg.idx_of(origin)
    charts = []
    for series in (17, 13):
        sel = (rows["series"] == series) & (rows["day"] == first // 48)
        for stage in ("none", "events+shift"):
            q, _ = dg.stage_quantiles(rows=rows, which="scaled", stage=stage, evt=evt)
            frame = pl.DataFrame(
                {
                    "valid_time": [
                        lib.GRID_START + timedelta(minutes=30 * int(v)) for v in rows["vidx"][sel]
                    ],
                    "q05": q[sel, 1],
                    "q25": q[sel, 3],
                    "q50": q[sel, 5],
                    "q75": q[sel, 7],
                    "q95": q[sel, 9],
                    "y": rows["y"][sel],
                }
            )
            spans_s = spans_in(
                spans=spans, series=series, lo=origin, hi=origin + timedelta(days=14)
            )
            charts.append(
                fan(
                    frame=frame,
                    spans=spans_s,
                    title=f"Primary {series}: {STAGE_NAMES[stage]}",
                    width=300,
                )
            )
    alt.concat(*charts, columns=2).save(out / "example_fan_14_days.svg")


def main(out_dir: str) -> None:
    """Write every figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coverage_by_month(tables=dg.TABLES, out=out)
    series_coverage(tables=dg.TABLES, out=out)
    rows = dg.load_rows()
    spans = dg.event_spans(kind="primary")
    evt = dg.event_mask(spans=spans, series=rows["series"], vidx=rows["vidx"])
    example_weeks(rows=rows, evt=evt, spans=spans, out=out)
    example_fan_14_days(rows=rows, evt=evt, spans=spans, out=out)


if __name__ == "__main__":
    main(sys.argv[1])

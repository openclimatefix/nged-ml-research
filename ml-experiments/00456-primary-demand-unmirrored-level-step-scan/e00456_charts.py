# ruff: noqa: E501
"""Draw experiment 456's figures as SVG from the scan's tables (Primary substations only).

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    PYTHONPATH=ml-experiments/00087-...:ml-experiments/00456-... uv run --no-sync python \
        ml-experiments/00456-*/e00456_charts.py
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00456_run as run  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf  # noqa: F401  (registers the theme)
import polars as pl

WORK_DIR: Final[Path] = run.WORK_DIR
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00456-primary-demand-unmirrored-level-step-scan"
)
WIDTH: Final[int] = 480
ORIGIN: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)


def forecast_impact() -> alt.TopLevelMixin:
    """Return B-mix's lead over the lagged head by row set, with 95% intervals."""
    data = pl.read_csv(WORK_DIR / "forecast_impact.csv").filter(
        pl.col("row_set").is_in(
            ["step-free", "scan step only (unlabelled)", "labelled only", "labelled and scan step"]
        )
    )
    base = alt.Chart(data).encode(y=alt.Y("row_set:N", sort=None, title=None))  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("bmix_lead:Q", title="B-mix lead over the lagged head (points of capacity)")
    )
    bars = base.mark_rule().encode(x="lead_lo:Q", x2="lead_hi:Q")  # ty: ignore[unresolved-attribute]
    return (points + bars).properties(
        width=WIDTH, height=130, title="B-mix keeps most of its lead on rows inside a scan step"
    )


def injection_recall() -> alt.TopLevelMixin:
    """Return recall of injected steps by size and duration (scan-quiet windows)."""
    rows = pl.read_csv(WORK_DIR / "injection_quiet.csv")
    data = rows.group_by("size", "duration").agg(recall=pl.col("detected").mean(), n=pl.len())
    return (
        alt.Chart(data.with_columns(size=pl.col("size").cast(pl.String) + " x threshold"))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("duration:O", title="Injected step duration (days)"),
            y=alt.Y("recall:Q", scale=alt.Scale(domain=[0, 1]), title="Recall"),
            color=alt.Color("size:N", title=None),
        )
        .properties(
            width=WIDTH,
            height=180,
            title="Steps of 1.5 times the threshold are almost always found",
        )
    )


def chance_mirror() -> alt.TopLevelMixin:
    """Return the share of steps called mirrored at the true start and at shifted starts."""
    data = pl.read_csv(WORK_DIR / "chance_mirror.csv").with_columns(
        kind=pl.when(pl.col("start_offset_days") == 0)
        .then(pl.lit("true start"))
        .otherwise(pl.lit("start shifted by 30 to 90 days (chance)"))
    )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(
                "mirrored_share:Q",
                scale=alt.Scale(domain=[0, 0.7]),
                title="Share of steps called mirrored",
            ),
            y=alt.Y("neighbours:N", title=None),
            color=alt.Color("kind:N", title=None),
        )
        .properties(
            width=WIDTH, height=90, title="The mirror test passes as often at the wrong dates"
        )
    )


def false_alarms() -> alt.TopLevelMixin:
    """Return detected steps and surrogate false alarms per Primary-year, by Primary."""
    steps = pl.read_csv(WORK_DIR / "candidate_steps.csv")
    fa = pl.read_csv(WORK_DIR / "false_alarms.csv")
    counts = (
        steps.group_by("site")
        .len()
        .join(fa, on="site", how="right")
        .with_columns(steps_per_year=pl.col("len").fill_null(0) / pl.col("years"))
    )
    data = counts.select(
        "site",
        pl.col("steps_per_year").alias("Steps found in the real series"),
        pl.col("false_alarms_per_year").alias("Steps found in surrogates with no steps"),
    ).unpivot(index="site")
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("site:N", sort=alt.EncodingSortField(field="site"), title=None),
            y=alt.Y("value:Q", title="Steps per Primary-year"),
            color=alt.Color("variable:N", title=None),
            xOffset="variable:N",
        )
        .properties(
            width=WIDTH,
            height=180,
            title="Two Primaries find as many steps in step-free surrogates as in the data",
        )
    )


def example(
    *,
    site: str,
    start_day: int,
    others: list[str],
    name: str,
    levels: dict[str, np.ndarray],
    table: pl.DataFrame,
) -> alt.TopLevelMixin:
    """Return a chart of one step: the Primary's and its neighbours' daily residual level."""
    lo, hi = max(0, start_day - 60), start_day + 90
    rows = []
    for s in [site, *others]:
        for d in range(lo, hi):
            v = levels[s][d]
            if np.isfinite(v):
                rows.append(
                    {
                        "date": ORIGIN + timedelta(days=d),
                        "series": s,
                        "level": float(v),
                        "focus": s == site,
                    }
                )
    steps = table.filter(
        (pl.col("site") == site) & (pl.col("start_day") >= lo) & (pl.col("start_day") < hi)
    )
    marks = pl.DataFrame({"date": [ORIGIN + timedelta(days=int(d)) for d in steps["start_day"]]})
    line = (
        alt.Chart(pl.DataFrame(rows))
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("date:T", title=None),
            y=alt.Y("level:Q", title="Daily residual (points of capacity)"),
            color=alt.Color("series:N", title=None),
            opacity=alt.condition(alt.datum.focus, alt.value(1.0), alt.value(0.45)),
        )
    )
    rule = alt.Chart(marks).mark_rule(strokeDash=[4, 3]).encode(x="date:T")  # ty: ignore[unresolved-attribute]
    return (line + rule).properties(width=WIDTH, height=170, title=name)


def main() -> None:
    """Write every figure as an SVG."""
    levels, _, _ = run.load()
    graph = run.scan.neighbour_sets()
    steps = pl.read_csv(WORK_DIR / "candidate_steps.csv")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_forecast_impact.svg": forecast_impact(),
        "figure_2_injection_recall.svg": injection_recall(),
        "figure_3_chance_mirror.svg": chance_mirror(),
        "figure_4_false_alarms.svg": false_alarms(),
    }
    trusted = {"D2", "D3", "D8", "D10", "D11", "D12", "D14", "D19", "D21"}
    un = (
        steps.filter((pl.col("class") == "unmirrored") & pl.col("site").is_in(trusted))
        .with_columns(size=pl.col("points").abs())
        .sort("size", descending=True)
    )
    mi = (
        steps.filter(pl.col("class") == "mirrored")
        .with_columns(size=pl.col("points").abs())
        .sort("size", descending=True)
    )
    for i, row in enumerate(un.head(2).iter_rows(named=True)):
        others = sorted(graph.get(row["site"], set()) & set(levels))[:2]
        figures[f"figure_{5 + i}_unmirrored_example.svg"] = example(
            site=row["site"],
            start_day=row["start_day"],
            others=others,
            name=f"Unmirrored step at {row['site']} ({row['points']:+.1f} points)",
            levels=levels,
            table=steps,
        )
    row = mi.row(0, named=True)
    figures["figure_7_mirrored_example.svg"] = example(
        site=row["site"],
        start_day=row["start_day"],
        others=[row["nearest_neighbour"]],
        name=f"Mirrored step at {row['site']} ({row['points']:+.1f} points) and {row['nearest_neighbour']}",
        levels=levels,
        table=steps,
    )
    for name, chart in figures.items():
        chart.save(DOCS_DIR / name)  # ty: ignore[unresolved-attribute]


if __name__ == "__main__":
    main()

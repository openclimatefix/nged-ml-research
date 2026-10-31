# ruff: noqa: E501
"""Draw experiment 402's event figures: power, and the gate's weight on the lagged forecast."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00402-lag-gate-learned-weight-injected-load-shifts-part-1"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
WEIGHT: Final[str] = "w::gate full (group)"
EVENTS: Final[dict[str, list[tuple[str, int]]]] = {
    "six": [("D5", 146), ("D16", 55), ("D5", 149)],
    "eight": [("D8", 182), ("D8", 177), ("D8", 181)],
}
GROUP_LABEL: Final[dict[str, str]] = {
    "six": "Primaries with signed-MW readings",
    "eight": "Primaries with MVA-only readings",
}
MARGIN_DAYS: Final[int] = 4


def _series() -> pl.DataFrame:
    """Return, for each site and valid time, the forecast issued at 09:00 that day (shortest lead)."""
    scores = pl.read_parquet(
        WORK / "gate_scores_r1.parquet",
        columns=["kind", "site", "valid_time", "lead_hours", "target_real", "cluster", WEIGHT],
    ).filter(pl.col("kind") == "real")
    actual = pl.concat(
        [
            pl.read_parquet(p, columns=["site", "valid_time", "lead_hours", "actual"])
            for p in sorted((WORK / "worlds").glob("real_r1_*.parquet"))
        ]
    )
    return (
        scores.join(actual, on=["site", "valid_time", "lead_hours"])
        .sort("lead_hours")
        .group_by("site", "valid_time")
        .first()
    )


def event_figure(*, group: str, series: pl.DataFrame) -> None:
    """Draw three labelled events for one metering group, each as a power panel and a weight panel."""
    panels = []
    for number, (site, cluster) in enumerate(EVENTS[group], start=1):
        s = series.filter(pl.col("site") == site)
        ev = s.filter((pl.col("cluster") == cluster) & pl.col("target_real"))
        t0, t1 = ev["valid_time"].min(), ev["valid_time"].max()
        w = (
            s.filter(
                (pl.col("valid_time") >= t0 - pl.duration(days=MARGIN_DAYS))
                & (pl.col("valid_time") <= t1 + pl.duration(days=MARGIN_DAYS))
            )
            .rename({WEIGHT: "w_gate"})
            .with_columns(
                day=(pl.col("valid_time") - t0).dt.total_seconds() / 86400.0,
                inside=pl.col("target_real") & (pl.col("cluster") == cluster),
            )
        )
        span = pl.DataFrame({"a": [0.0], "b": [(t1 - t0).total_seconds() / 86400.0]})  # ty: ignore[unsupported-operator]
        x = alt.X("day:Q", title="Days since the event began (valid time)")
        shade = alt.Chart(span).mark_rect(color=ocf.GREY_3, opacity=0.5).encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
        power = (
            alt.Chart(w)  # ty: ignore[unresolved-attribute]
            .mark_line(color=ocf.DATA_BLUE, clip=True)
            .encode(x=x, y=alt.Y("actual:Q", title="Power (MW or MVA)"))
        )
        weight = (
            alt.Chart(w)  # ty: ignore[unresolved-attribute]
            .mark_line(color=ocf.BRAND_ORANGE, clip=True)
            .encode(
                x=x,
                y=alt.Y("w_gate:Q", title="Weight on lags", scale=alt.Scale(domain=[0, 1])),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [0.5]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=ocf.GREY_1, strokeDash=[4, 3])
            .encode(y="y:Q")
        )
        title = alt.TitleParams(
            [f"Event {number} at Primary {site}: shaded area is the labelled event"],
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        )
        panels.append(
            alt.vconcat(
                (shade + power).properties(width=CONTENT_WIDTH_PX - 90, height=90, title=title),
                (shade + weight + rule).properties(width=CONTENT_WIDTH_PX - 90, height=70),
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                f"{GROUP_LABEL[group]}: label-free gate around three labelled events.",
                "Blue: measured power. Orange: the gate's weight on the forecast with lags, from the forecast",
                "issued at 09:00 each day (forecast R1, leave-one-month-out). Dashed: weight 0.5.",
            ],
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        )
    ).save(DOCS_DIR / f"figure_events_{group}.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    data = _series()
    for g in EVENTS:
        event_figure(group=g, series=data)

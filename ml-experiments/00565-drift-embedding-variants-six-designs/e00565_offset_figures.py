# ruff: noqa: E501
"""Experiment 565: figures of the learnt D5 offset and the A10 embedding, one panel per Primary.

Reads `offset_knots.csv` (from `e00565_offsets.py`) and experiment 400's labelled events. One line is
the mean over seeds 0 to 4 of the last fold's (2026-06) pretrained weights, the band is the
minimum to maximum over those seeds, dots are the monthly knots, and the orange spans are the
labelled load-shift events (184 events).
"""

from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import e00510_common as c  # ty: ignore[unresolved-import]
import polars as pl
from e00565_design6 import W
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00565-drift-embedding-variants-six-designs"
)
WIDTH = 680
GROUPS = {"MVA-only eight": "eight MVA Primaries", "signed-MW six": "six MW Primaries"}
FIELDS = {
    "d5_capacity_fraction": (
        "D5",
        "Offset (% of capacity)",
        100.0,
        "d5_pct",
    ),
    "d5_mw": ("D5", "Offset (MW or MVA)", 1.0, "d5_mw"),
    "a10_pc1": (
        "A10",
        "Embedding PC1 (arb. units)",
        1.0,
        "a10_pc",
    ),
}


def events() -> pl.DataFrame:
    """Return each labelled event's Primary label and start and end dates."""
    grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    t = pl.read_parquet(c.worlds402.cf.EVENTS)
    f = lambda i: datetime.fromtimestamp((grid + i * step) / 1e6, UTC)  # noqa: E731
    return pl.DataFrame(
        {
            "label": t["label"],
            "start": [f(i) for i in t["onset_idx"]],
            "end": [
                max(f(j), f(i).replace(hour=0))
                for i, j in zip(t["onset_idx"], t["end_idx"], strict=True)
            ],
        }
    ).with_columns(pl.col("end") + pl.duration(days=1))


def figure(*, knots: pl.DataFrame, ev: pl.DataFrame, field: str, group: str) -> alt.Chart:
    """Return one stacked set of panels for one field and group."""
    arm, ytitle, scale, _ = FIELDS[field]
    k = knots.filter(pl.col("group") == group).with_columns(
        date=pl.col("month").str.to_date().cast(pl.Datetime), v=pl.col(field) * scale
    )
    order = sorted(k["label"].unique().to_list(), key=lambda s: int(s[1:]))
    agg = k.group_by("label", "date").agg(
        mean=pl.col("v").mean(), lo=pl.col("v").min(), hi=pl.col("v").max()
    )
    e = ev.filter(pl.col("label").is_in(order)).with_columns(
        start=pl.col("start").dt.replace_time_zone(None),
        end=pl.col("end").dt.replace_time_zone(None),
    )
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
    base = alt.Chart(agg).encode(x=x)
    band = base.mark_area(opacity=0.35, color=theme.DATA_BLUE).encode(
        y=alt.Y("lo:Q", title=ytitle), y2="hi:Q"
    )
    line = base.mark_line(color=theme.DATA_BLUE).encode(y="mean:Q")
    dots = base.mark_point(color=theme.DATA_BLUE, size=8, filled=True).encode(y="mean:Q")
    spans = (
        alt.Chart(e)
        .mark_rect(color=theme.BRAND_ORANGE, opacity=0.55)
        .encode(x="start:T", x2="end:T")
    )
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="grey", strokeDash=[3, 3])
        .encode(y="z:Q")
    )
    return alt.vconcat(
        *[
            alt.layer(
                spans.transform_filter(alt.datum.label == s),
                band.transform_filter(alt.datum.label == s),
                line.transform_filter(alt.datum.label == s),
                dots.transform_filter(alt.datum.label == s),
                zero,
            ).properties(
                width=WIDTH - 80, height=70, title=alt.TitleParams(s, anchor="start", fontSize=11)
            )
            for s in order
        ]
    ).properties(
        title=alt.TitleParams(
            [
                f"Arm {arm}: learnt slow {'offset' if arm == 'D5' else 'embedding'}, {GROUPS[group]}",
                "Line: mean of 5 seeds. Band: minimum to maximum over seeds. Dots: monthly knots.",
                "Orange spans: labelled load-shift events (incomplete). Last fold's weights.",
            ],
            anchor="start",
            fontSize=11,
        )
    )


def main() -> None:
    """Write the SVG figures next to the page."""
    knots = pl.read_csv(W / "scores/offset_knots.csv")
    ev = events()
    DOCS.mkdir(parents=True, exist_ok=True)
    for field, (_, _, _, stem) in FIELDS.items():
        for group in GROUPS:
            ch = figure(knots=knots, ev=ev, field=field, group=group)
            ch.save(str(DOCS / f"offset_{stem}_{'mva' if group.startswith('MVA') else 'mw'}.svg"))


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Draw the figures for experiment 194 from the tables the other scripts wrote."""

import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00194_common as common
import numpy as np
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_figure_guard as guard

DOCS_DIR: Final[Path] = Path("docs/ml-experiments/00194-" + common.SLUG)
WIDTH: Final[int] = CONTENT_WIDTH_PX - 80
CHANGE: Final[date] = date(2023, 8, 14)
FALL: Final[date] = date(2026, 3, 9)


def rule(*, day: date, colour: str, dash: list[int] | None = None) -> alt.Chart:
    """Return a vertical rule at a date."""
    return (
        alt.Chart(pl.DataFrame({"x": [day]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=colour, strokeDash=dash or [], strokeWidth=2)
        .encode(x="x:T")
    )


def main() -> None:  # noqa: D103
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # Figure 1: weekly left-tail share, D8 against controls.
    frames = []
    for label in ("D8", *common.CONTROL_LABELS):
        w = pl.read_parquet(common.DATA_DIR / f"weekly_{label}.parquet").filter(pl.col("n") >= 168)
        frames.append(w.select("time", "share").with_columns(site=pl.lit(label)))
    weekly = pl.concat(frames)
    # weekly files stop at 2026-07; they cover the whole window.
    lines = (
        alt.Chart(weekly)
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("time:T", title="Week"),
            y=alt.Y("share:Q", title="Share of half-hours below -2.5 MW"),
            color=alt.Color(
                "site:N",
                title=None,
                scale=alt.Scale(
                    domain=["D8", "D3", "D5", "D14"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE],
                ),
            ),
            strokeWidth=alt.condition(alt.datum.site == "D8", alt.value(2.5), alt.value(1)),
        )
    )
    fig1 = (
        lines + rule(day=CHANGE, colour="black") + rule(day=FALL, colour="grey", dash=[4, 3])
    ).properties(
        width=WIDTH,
        height=200,
        title="D8's shortfall starts 2023-08-14 (black) and fades from March 2026 (grey dashed)",
    )
    guard.save_chart(chart=fig1, path=DOCS_DIR / "figure_1.svg")

    # Figure 2: pooled before and after the break; no dates, no megawatt values, no counts per day.
    both = pl.read_parquet(common.DATA_DIR / "d8_vs_g2.parquet").filter(
        pl.col("time").is_between(
            datetime(2023, 7, 15, tzinfo=UTC), datetime(2023, 9, 15, tzinfo=UTC), closed="left"
        )
    )
    last_g2 = both.filter(pl.col("g2").is_not_null())["time"].dt.date().max()
    pooled = (
        both.with_columns(
            period=pl.when(pl.col("time").dt.date() <= last_g2)
            .then(pl.lit("Before the break"))
            .otherwise(pl.lit("After the break")),
        )
        .group_by("period")
        .agg(
            tail=pl.col("tail").mean(),
            readings=pl.col("g2").is_not_null().mean(),
        )
    )
    order = ["Before the break", "After the break"]
    top = (
        alt.Chart(pooled)
        .mark_bar(color=ocf.BRAND_ORANGE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("period:N", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("tail:Q", title="D8 shortfall share"),
        )
        .properties(
            width=WIDTH // 2,
            height=110,
            title=[
                "D8's shortfall share is higher after the break, once G2's output",
                "readings stop arriving (the weeks either side, pooled)",
            ],
        )
    )
    bot = (
        alt.Chart(pooled)
        .mark_bar(color=ocf.DATA_BLUE_LIGHT, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("period:N", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("readings:Q", title="Share of half-hours with a G2 reading"),
        )
        .properties(width=WIDTH // 2, height=90)
    )
    guard.save_chart(chart=alt.vconcat(top, bot), path=DOCS_DIR / "figure_2.svg")

    # Figure 3: G2 running by price fifth.
    g2 = pl.read_parquet(common.DATA_DIR / "g2_window.parquet")
    edges = np.quantile(g2["price"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    fifth = np.searchsorted(edges, g2["price"].to_numpy(), side="right")
    running = (g2["units"] >= 1).to_numpy()
    rows = []
    for f in range(5):
        k, n = int(running[fifth == f].sum()), int((fifth == f).sum())
        lo, hi = common.wilson(k=k, n=n)
        rows.append({"fifth": f + 1, "share": k / n, "low": lo, "high": hi})
    d = pl.DataFrame(rows)
    bars = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "fifth:O",
                title="Fifth of the system price (1 = cheapest)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "share:Q", title="Share of half-hours G2 ran", scale=alt.Scale(domain=[0, 0.35])
            ),
        )
    )
    err = alt.Chart(d).mark_rule(color="black").encode(x="fifth:O", y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    fig3 = (bars + err).properties(
        width=WIDTH // 2,
        height=170,
        title="G2 ran in 9% of the cheapest and 26% of the dearest fifth (95% intervals)",
    )
    guard.save_chart(chart=fig3, path=DOCS_DIR / "figure_3.svg")

    # Figure 4: residual histograms.
    d8 = pl.read_parquet(common.DATA_DIR / "d8_resid.parquet")
    periods = {
        "1 before the engine (2022-01 to 2023-06-12)": (
            datetime(2022, 1, 1, tzinfo=UTC),
            datetime(2023, 6, 13, tzinfo=UTC),
        ),
        "2 G2 output arrives (2023-07-03 to 08-15)": (common.G2_FIRST, common.G2_LAST),
        "3 after (2023-09 to 2025-12)": (
            datetime(2023, 9, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
        ),
        "4 spring 2026 (April to June)": (
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 7, 1, tzinfo=UTC),
        ),
    }
    bins = np.arange(-8, 4.01, 0.5)
    hist_rows = []
    for name, (a, b) in periods.items():
        arr = d8.filter(pl.col("time").is_between(a, b, closed="left"))["resid"].to_numpy()
        h, _ = np.histogram(arr, bins=bins)
        hist_rows += [
            {"period": name, "mw": float(bins[i] + 0.25), "share": float(h[i] / arr.size)}
            for i in range(h.size)
        ]
    fig4 = (
        alt.Chart(pl.DataFrame(hist_rows))
        .mark_bar(color=ocf.BRAND_ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("mw:Q", title="D8 metered minus expected power (MW)"),
            y=alt.Y("share:Q", title=None),
            row=alt.Row("period:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=WIDTH - 120,
            height=70,
            title="D8's residual gains a long negative tail after G2's output stops arriving (share of half-hours)",
        )
    )
    guard.save_chart(chart=fig4, path=DOCS_DIR / "figure_4.svg")

    # Figure 5: monthly share with the scored year of experiment 191.
    monthly = pl.read_parquet(common.DATA_DIR / "d8_monthly.parquet")
    band = (
        alt.Chart(pl.DataFrame({"a": [date(2025, 7, 1)], "b": [date(2026, 7, 1)]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=ocf.DATA_BLUE_LIGHT, opacity=0.4)
        .encode(x="a:T", x2="b:T")
    )
    line = (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=2.5, point=True)
        .encode(x=alt.X("time:T", title="Month"), y=alt.Y("share:Q", title="Share below -2.5 MW"))
    )
    fig5 = (band + line).properties(
        width=WIDTH,
        height=160,
        title="Experiment 191's scored year (blue) holds seven months of the shortfall and five of its absence",
    )
    guard.save_chart(chart=fig5, path=DOCS_DIR / "figure_5.svg")


if __name__ == "__main__":
    main()

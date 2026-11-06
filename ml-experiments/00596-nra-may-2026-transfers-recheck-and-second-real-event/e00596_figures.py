# ruff: noqa: E501
"""Experiment 596: figures for the May 2026 transfer recheck (reads the files of `e00596_check.py`)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00582_arms as a  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00596-nra-may-2026-transfers-recheck-and-second-real-event"
DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
DAY: Final[int] = 48
W: Final[int] = 680
EVENT_COLOURS: Final[dict[str, str]] = {
    "D10": theme.DATA_BLUE,
    "D11": theme.DATA_SKY,
    "D12": theme.DATA_PURPLE,
    "D16": theme.BRAND_ORANGE,
}


def day_of(idx: int) -> datetime:
    """Return the UTC start of the day holding half-hour grid index `idx`."""
    return START + timedelta(minutes=30 * idx)


def daily(*, arr: np.ndarray, lo: int, hi: int) -> pl.DataFrame:
    """Return the daily mean of `arr[lo:hi]` with the date."""
    n = (hi - lo) // DAY
    vals = np.nanmean(arr[lo : lo + n * DAY].reshape(n, DAY), axis=1)
    return pl.DataFrame(
        {"date": [day_of(lo + i * DAY) for i in range(n)], "value": vals.astype(float)}
    )


def main() -> None:
    """Draw the three figures of the write-up."""
    tab = pl.read_csv(DATA / "check_table.csv")
    gap = np.load(DATA / "gap_series.npz")["gap"]
    t = a.Table(with_weekmask=False)
    lo = int((datetime(2026, 4, 20, tzinfo=UTC) - START).total_seconds() // 1800)
    hi = int((datetime(2026, 6, 8, tzinfo=UTC) - START).total_seconds() // 1800)
    gap_rows, obs_rows, span_rows, exp_rows = [], [], [], []
    for r in tab.iter_rows(named=True):
        on = int(
            (datetime.fromisoformat(r["onset"]).replace(tzinfo=UTC) - START).total_seconds() // 1800
        )
        en = int(
            (datetime.fromisoformat(r["end"]).replace(tzinfo=UTC) - START).total_seconds() // 1800
        )
        p, lab = r["primary"], r["label"]
        g = daily(arr=gap[p], lo=lo, hi=hi).filter(pl.col("value").is_not_nan())
        gap_rows.append(g.with_columns(label=pl.lit(lab)))
        obs_rows.append(daily(arr=t.clean[p], lo=lo, hi=hi).with_columns(label=pl.lit(lab)))
        span_rows.append({"label": lab, "a": day_of(on), "b": day_of(en)})
        exp_rows.append({"label": lab, "a": day_of(on), "b": day_of(en), "depth": r["depth_share"]})
    spans = pl.DataFrame(span_rows)
    exps = pl.DataFrame(exp_rows)
    cols = alt.Scale(domain=list(EVENT_COLOURS), range=list(EVENT_COLOURS.values()))

    def panel(*, data: pl.DataFrame, y_title: str, with_depth: bool) -> alt.Chart:
        base = (
            alt.Chart(data)
            .mark_line(aria=False, strokeWidth=1.8)
            .encode(
                x=alt.X(
                    "date:T",
                    title=None,
                    axis=alt.Axis(
                        format="%d %b",
                        values=[
                            datetime(2026, 4, 20, tzinfo=UTC) + timedelta(days=7 * i)
                            for i in range(7)
                        ],
                    ),
                ),
                y=alt.Y("value:Q", title=y_title),
                color=alt.Color("label:N", scale=cols, legend=None),
            )
        )
        shade = (
            alt.Chart(spans)
            .mark_rect(aria=False, opacity=0.12, color="#6b6b6b")
            .encode(x="a:T", x2="b:T")
        )
        layers = [shade, base]
        if with_depth:
            layers.append(
                alt.Chart(exps)
                .mark_rule(aria=False, strokeDash=[5, 3], color="black")
                .encode(x="a:T", x2="b:T", y="depth:Q")
            )
        return (
            alt.layer(*layers)
            .properties(width=W - 90, height=110)
            .facet(
                row=alt.Row(
                    "label:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")
                ),
            )
            if False
            else alt.layer(*layers).properties(width=W - 90, height=110)
        )

    def stacked(*, data: pl.DataFrame, y_title: str, with_depth: bool) -> alt.VConcatChart:
        parts = []
        for lab in EVENT_COLOURS:
            d = data.filter(pl.col("label") == lab)
            sp = spans.filter(pl.col("label") == lab)
            ex = exps.filter(pl.col("label") == lab)
            line = (
                alt.Chart(d)
                .mark_line(aria=False, strokeWidth=1.8, color=EVENT_COLOURS[lab])
                .encode(
                    x=alt.X(
                        "date:T",
                        title=None,
                        axis=alt.Axis(
                            format="%d %b",
                            values=[
                                datetime(2026, 4, 20, tzinfo=UTC) + timedelta(days=7 * i)
                                for i in range(7)
                            ],
                        ),
                    ),
                    y=alt.Y("value:Q", title=f"{lab}: {y_title}", scale=alt.Scale(zero=False)),
                )
            )
            shade = (
                alt.Chart(sp)
                .mark_rect(aria=False, opacity=0.15, color="#6b6b6b")
                .encode(x="a:T", x2="b:T")
            )
            layers = [shade, line]
            if with_depth:
                layers.append(
                    alt.Chart(ex)
                    .mark_rule(aria=False, strokeDash=[5, 3], color="black", strokeWidth=1.5)
                    .encode(x="a:T", x2="b:T", y="depth:Q")
                )
            parts.append(alt.layer(*layers).properties(width=W - 110, height=95))
        return alt.vconcat(*parts, spacing=8)

    fig1 = stacked(
        data=pl.concat(gap_rows), y_title="observed minus lag-free median", with_depth=True
    ).properties(
        title=alt.TitleParams(
            "The lag-free forecast does not sit one depth away from the observed power in May 2026",
            subtitle=[
                "Daily mean of observed power minus the lag-free median, 24 to 120 h ahead, as a share of capacity.",
                "Grey band: the event span. Dashed line: the fitted depth (the gap the event should open; D12 shows the mean over its merged span).",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        )
    )
    fig1.save(DOCS / "gap_by_event.svg")
    fig2 = stacked(
        data=pl.concat(obs_rows), y_title="observed daily mean", with_depth=False
    ).properties(
        title=alt.TitleParams(
            "Observed daily power at the four Primaries, April to June 2026",
            subtitle=[
                "Daily mean of cleaned power as a share of effective capacity. Grey band: the event span.",
                "A transfer should show as a step at the start and the opposite step at the end.",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        )
    )
    fig2.save(DOCS / "observed_by_event.svg")
    bars = tab.select("label", "depth_share", "S", "placebo_p95").with_columns(
        neg_p95=-pl.col("placebo_p95")
    )
    long = pl.concat(
        [
            bars.select(
                "label",
                pl.lit("Fitted depth").alias("kind"),
                pl.col("depth_share").abs().alias("v"),
            ),
            bars.select(
                "label",
                pl.lit("Measured signal S (signed to match the depth)").alias("kind"),
                (pl.col("S") * pl.col("depth_share").sign()).alias("v"),
            ),
            bars.select(
                "label",
                pl.lit("Placebo 95th percentile of |S|").alias("kind"),
                pl.col("placebo_p95").alias("v"),
            ),
        ]
    )
    fig3 = (
        alt.Chart(long)
        .mark_bar(aria=False)
        .encode(
            x=alt.X("label:N", title=None, sort=list(EVENT_COLOURS)),
            xOffset=alt.XOffset("kind:N"),
            y=alt.Y("v:Q", title="Share of capacity"),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "Fitted depth",
                        "Measured signal S (signed to match the depth)",
                        "Placebo 95th percentile of |S|",
                    ],
                    range=[
                        theme.GREY_2 if hasattr(theme, "GREY_2") else "#6b6b6b",
                        theme.BRAND_ORANGE,
                        theme.DATA_BLUE,
                    ],
                ),
                legend=alt.Legend(orient="top", direction="vertical", labelLimit=500),
            ),
        )
        .properties(
            width=W - 90,
            height=200,
            title=alt.TitleParams(
                "The measured gap is a fraction of each fitted depth and below the placebo limit",
                subtitle=[
                    "For an event to pass, the orange bar must reach half the grey bar and exceed the blue bar."
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            ),
        )
    )
    fig3.save(DOCS / "signal_vs_depth.svg")


if __name__ == "__main__":
    main()

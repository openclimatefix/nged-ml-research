# ruff: noqa: D103, E501
"""Experiment 648 figures (CPU). Run `e00648_score.py` for every arm and `e00648_summary.py` first.

`python e00648_figures.py headline` draws the gain of every arm over its network's reference, with
95% intervals, and the guards. Charts are 680 px wide (the docs text column).
"""

import sys
from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate"
SCORE = ML_EXPERIMENTS_DATA_DIR / NAME / "score"
DOCS = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / NAME
WIDTH = 480
LABELS = {
    "lag32": "Lag encoder width 32",
    "lag128": "Lag encoder width 128",
    "lagdeep": "Lag encoder depth 2",
    "nbbn8": "Neighbour bottleneck 8 (vs none)",
    "nbbn16": "Neighbour bottleneck 16 (vs none)",
    "nbbn32": "Neighbour bottleneck 32 (vs none)",
    "comp5": "5 mixture components",
    "hid128": "Hidden width 128",
    "hid256": "Hidden width 256",
    "hid384": "Hidden width 384",
    "drop0": "Dropout 0.0",
    "drop2": "Dropout 0.2",
    "lrhalf": "Learning rate x0.5",
    "lrdouble": "Learning rate x2",
    "epoch2": "Epochs and patience x2",
}
NET = {"k1": "K1 (lag network)", "c2": "Class 2 candidate (adds neighbour summaries)"}


def headline() -> None:
    d = pl.read_csv(SCORE / "screen.csv").filter(pl.col("arm") != "613")
    d = d.with_columns(
        row=pl.col("arm").replace_strict(LABELS, default=None),
        network=pl.col("net").replace_strict(NET),
    )
    order = [LABELS[a] for a in LABELS]
    panels = []
    for net in ("k1", "c2"):
        dn = d.filter(pl.col("net") == net)
        margin = float(dn["margin"][0])
        base = alt.Chart(dn).encode(
            y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        )
        rules = (
            alt.Chart(pl.DataFrame({"x": [0.0, margin], "k": ["Zero", "Margin to pass"]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], strokeWidth=2)
            .encode(
                x="x:Q",
                color=alt.Color(
                    "k:N",
                    scale=alt.Scale(
                        domain=["Zero", "Margin to pass"],
                        range=[theme.DATA_SKY, theme.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        bars = base.mark_rule(color=theme.DATA_BLUE).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        pts = base.mark_point(filled=True, size=60, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "gain:Q",
                title="Gain over the unchanged network, 24-120 h CRPS (points of capacity; positive is better)",
                scale=alt.Scale(domain=[-0.25, 0.25], nice=False),
                axis=alt.Axis(values=[-0.2, -0.1, 0, 0.1, 0.2]),
            )
        )
        panels.append(
            alt.layer(rules, bars, pts).properties(
                width=WIDTH, height=22 * dn.height + 20, title=NET[net]
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "One setting changed at a time, three test months, one seed. Dot: estimate; line: 95%",
                "interval from resampling Primary-month cells. Primaries D15 and D18 cut before 2025-03.",
            ],
            anchor="start",
            fontSize=12,
        )
    )
    chart.save(str(DOCS / "figure_1_headline.svg"))

    g = d.select("net", "row", "d_follow", "d_safe95").unpivot(
        index=["net", "row"], variable_name="measure", value_name="v"
    )
    g = g.with_columns(
        measure=pl.col("measure").replace_strict(
            {
                "d_follow": "Change in the share of an injected shift followed (limit +0.03; lower is better)",
                "d_safe95": "Change in the safe-side share at q95 (limit -0.03; higher is better)",
            }
        ),
        network=pl.col("net").replace_strict({"k1": "K1", "c2": "Class 2 candidate"}),
    )
    gp = []
    for m, lim in zip(
        g["measure"].unique(maintain_order=True).to_list(), (0.03, -0.03), strict=True
    ):
        dm = g.filter(pl.col("measure") == m)
        base = alt.Chart(dm).encode(
            y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        )
        pts = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "v:Q",
                title=m,
                scale=alt.Scale(domain=[-0.15, 0.15], nice=False),
                axis=alt.Axis(values=[-0.1, -0.05, 0, 0.05, 0.1]),
            ),
            color=alt.Color(
                "network:N",
                scale=alt.Scale(
                    domain=["K1", "Class 2 candidate"], range=[theme.DATA_BLUE, theme.DATA_PURPLE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            yOffset="network:N",
        )
        rule = (
            alt.Chart(pl.DataFrame({"x": [lim]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color=theme.BRAND_ORANGE, strokeWidth=2)
            .encode(x="x:Q")
        )
        gp.append(alt.layer(rule, pts).properties(width=WIDTH, height=22 * len(order) + 20))
    alt.vconcat(*gp).save(str(DOCS / "figure_2_guards.svg"))


if __name__ == "__main__":
    {"headline": headline}[sys.argv[1]]()

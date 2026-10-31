"""Draw the figures of experiment 426 stage 0 from its stored tables."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

STAGE0: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00426-mdn-neighbouring-primaries-graph-and-bottom-up-design"
    / "stage0"
)
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00426-mdn-neighbouring-primaries-graph-and-bottom-up-design"
)
WIDTH: Final[int] = 560


def figure_one() -> None:
    """Draw the mean lagged correlation per graph and test month at the signed-MW six."""
    table = pl.read_csv(STAGE0 / "stage0a.csv").filter(
        pl.col("group") == "six", pl.col("band") == "d1"
    )
    long = (
        pl.concat(
            [
                table.select("month", pl.col("true").alias("value")).with_columns(
                    graph=pl.lit("Interconnected graph"), lo=pl.col("value"), hi=pl.col("value")
                ),
                table.select(
                    "month",
                    pl.col("random_mean").alias("value"),
                    (pl.col("random_mean") - 2 * pl.col("random_sd")).alias("lo"),
                    (pl.col("random_mean") + 2 * pl.col("random_sd")).alias("hi"),
                ).with_columns(graph=pl.lit("Random graphs, same degrees")),
                table.select(
                    "month",
                    pl.col("length_mean").alias("value"),
                    (pl.col("length_mean") - 2 * pl.col("length_sd")).alias("lo"),
                    (pl.col("length_mean") + 2 * pl.col("length_sd")).alias("hi"),
                ).with_columns(graph=pl.lit("Shortest-edge random graphs")),
            ],
            how="diagonal",
        )
        .select("month", "graph", "value", "lo", "hi")
        .to_pandas()
    )
    order = ["Interconnected graph", "Random graphs, same degrees", "Shortest-edge random graphs"]
    scale = alt.Scale(domain=order, range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE])
    base = alt.Chart(long).encode(
        x=alt.X("month:N", title="Test month"),
        color=alt.Color("graph:N", scale=scale, sort=order, legend=alt.Legend(orient="top")),
        xOffset="graph:N",
    )
    chart = (
        base.mark_errorbar().encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("lo:Q", title="Mean lagged residual correlation"), y2="hi:Q"
        )
        + base.mark_point(filled=True, size=70).encode(y="value:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=240)
    chart.save(DOCS_DIR / "figure_1_lagged_correlation.svg")


def figure_two() -> None:
    """Draw the squared-error reduction of the true neighbours against each control."""
    table = pl.read_csv(STAGE0 / "stage0b.csv").filter(pl.col("group") != "d15_d18")
    names = {
        "vs_own": "Own lags only",
        "vs_random": "Random neighbours",
        "vs_length": "Shortest-edge random",
    }
    data = table.with_columns(
        control=pl.col("contrast").replace_strict(names),
        panel=pl.col("group").replace_strict({"six": "Signed-MW six", "eight": "MVA-only eight"}),
        rows=pl.col("handling").replace_strict(
            {"all_rows": "All rows", "no_event_rows": "Rows outside labelled events"}
        ),
    ).to_pandas()
    order = list(names.values())
    base = alt.Chart(data).encode(
        y=alt.Y("control:N", sort=order, title="Compared with"),
        color=alt.Color(
            "rows:N",
            scale=alt.Scale(
                domain=["All rows", "Rows outside labelled events"],
                range=[theme.BRAND_ORANGE, theme.DATA_BLUE],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
        yOffset="rows:N",
    )
    chart = (
        base.mark_errorbar().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Held-out squared error lowered by true neighbours (%)"),
            x2="hi:Q",
        )
        + base.mark_point(filled=True, size=70).encode(x="reduction_pct:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=110)
    chart.facet(row=alt.Row("panel:N", title=None)).save(
        DOCS_DIR / "figure_2_neighbour_lag_ridge.svg"
    )


def figure_three() -> None:
    """Draw the node MAE of the network, the XGBoost control, and the bottom-up medians."""
    table = pl.read_csv(STAGE0 / "stage0c.csv")
    mean = table.group_by("node", "arm").agg(pl.col("mae_points").mean())
    names = {
        "N1": "Network N1 (experiment 415)",
        "xgb_control": "Monthly XGBoost control",
        "bottom_up_A": "Bottom-up from Primaries",
        "bottom_up_B_with_node_lags": "Bottom-up plus node's own lags",
    }
    data = mean.with_columns(arm_name=pl.col("arm").replace_strict(names)).to_pandas()
    order = list(names.values())
    chart = (
        alt.Chart(data)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm_name:N", sort=order, title=None),
            x=alt.X("mae_points:Q", title="Mean absolute error, median (points of node capacity)"),
            color=alt.Color(
                "arm_name:N",
                sort=order,
                scale=alt.Scale(
                    domain=order,
                    range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE, "#1A1A1A"],
                ),
                legend=None,
            ),
            column=alt.Column("node:N", sort=["D13", "D17", "D1", "D6"], title=None),
        )
        .properties(width=120, height=160)
    )
    chart.save(DOCS_DIR / "figure_3_bottom_up_node.svg")


if __name__ == "__main__":
    figure_one()
    figure_two()
    figure_three()

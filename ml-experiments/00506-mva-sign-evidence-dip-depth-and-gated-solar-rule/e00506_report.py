"""Experiment 506 figures, drawn from the stored Part 1 and Part 4 tables.

uv run --no-sync python ml-experiments/00506-*/e00506_report.py
"""

from pathlib import Path

import altair as alt
import e00506_data as data  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS = Path("docs/ml-experiments/00506-mva-sign-evidence-dip-depth-and-gated-solar-rule")
ocf.register_and_enable_theme() if hasattr(ocf, "register_and_enable_theme") else None


def main() -> None:
    """Write the three figures."""
    apply = pl.read_parquet(data.WORK / "part1_apply.parquet").filter(pl.col("arm") == "Q0")
    apply = apply.with_columns(
        group=pl.when(pl.col("site").is_in(data.MVA_ONLY))
        .then(pl.lit("MVA-only"))
        .otherwise(pl.lit("signed MW")),
    )
    order = [*data.SIGNED_SIX, *data.MVA_ONLY]
    base = alt.Chart(apply).encode(y=alt.Y("site:N", sort=order, title=None))
    chart = (
        base.mark_rule().encode(x="min_dip:Q", x2="median_dip:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "p10_dip:Q", title="Dip depth: minimum reading over the window, points of capacity"
            ),
            shape=alt.Shape("group:N"),
        )
    ).properties(
        width=420,
        height=300,
        title="Daily dip at sunny middays: 10th percentile (dots), minimum to median (bar)",
    )
    chart.save(DOCS / "figure_1_dip_depth_per_primary.svg")

    auc = pl.read_parquet(data.WORK / "part1_auc.parquet").with_columns(
        level=pl.col("level").cast(pl.Utf8)
    )
    b = alt.Chart(auc).encode(
        y=alt.Y("arm:N", sort=["Q0", "Q2", "Q4", "PF097"], title="Reactive floor"),
        color=alt.Color("level:N", title="Injection level"),
    )
    chart2 = (
        b.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("auc_lo:Q", scale=alt.Scale(domain=[0.7, 1.0]), title="Area under the curve"),
            x2="auc_hi:Q",
        )
        + b.mark_point(filled=True, size=70).encode(x="auc:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=420,
        height=200,
        title="Detecting a sunny-midday sign change with the dip depth (95% month-block interval)",
    )
    chart2.save(DOCS / "figure_2_auc_by_reactive_floor.svg")

    six = pl.read_parquet(data.WORK / "part4_six.parquet").filter(pl.col("capacity") == "p50")
    six = six.with_columns(level=pl.col("level").cast(pl.Utf8))
    c = alt.Chart(six).encode(
        y=alt.Y("site:N", sort=list(data.SIGNED_SIX), title=None), color="level:N"
    )
    chart3 = (
        c.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "false_reverse_lo:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Share of reverse-flow-free sunny half-hours called reverse",
            ),
            x2="false_reverse_hi:Q",
        )
        + c.mark_point(filled=True, size=70).encode(x="false_reverse:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=420,
        height=200,
        title="Gated rule false-reverse rate at the six (experiment 408: 3.6%)",
    )
    chart3.save(DOCS / "figure_3_gated_rule_false_reverse.svg")


if __name__ == "__main__":
    main()

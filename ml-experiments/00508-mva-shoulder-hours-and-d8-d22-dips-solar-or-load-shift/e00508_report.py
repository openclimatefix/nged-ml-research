"""Experiment 508 figures, drawn from the stored tables.

PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-*:ml-experiments/00508-* \
        uv run --no-sync python ml-experiments/00508-*/e00508_report.py
"""

from pathlib import Path

import altair as alt
import e00508_common as common  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS = Path("docs/ml-experiments/00508-mva-shoulder-hours-and-d8-d22-dips-solar-or-load-shift")
COLOURS = {
    "midday": "#FF4901",
    "morning": "#306BFF",
    "evening": "#10C5F7",
    "day": "#B701FF",
}
ocf.register_and_enable_theme() if hasattr(ocf, "register_and_enable_theme") else None


def main() -> None:
    """Write the four figures."""
    auc = (
        pl.read_parquet(common.WORK / "partA_auc.parquet")
        .filter(pl.col("estimable"), pl.col("arm") == "Q0")
        .rename({"window": "Window"})
    )
    base = alt.Chart(auc).encode(
        y=alt.Y("ratio_bin:N", title="Export-to-import ratio"),
        yOffset="Window:N",
        color=alt.Color(
            "Window:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
        ),
    )
    chart1 = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("auc_lo:Q", scale=alt.Scale(domain=[0.3, 1.0]), title="Area under the curve"),
            x2="auc_hi:Q",
        )
        + base.mark_point(filled=True, size=60).encode(x="auc:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=420,
        height=220,
        title="Detecting a sign-change day, floor 0 (95% month-block interval)",
    )
    chart1.save(DOCS / "figure_1_auc_by_window_and_ratio.svg")

    rates = pl.read_parquet(common.WORK / "partB_sky_placebo.parquet").filter(
        pl.col("year") != "pooled"
    )
    long = pl.concat(
        [
            rates.select(
                "site", "year", rate=f"{sky}_rate", lo=f"{sky}_lo", hi=f"{sky}_hi"
            ).with_columns(Sky=pl.lit(sky))
            for sky in ("sunny", "cloudy")
        ]
    )
    c2 = alt.Chart(long).encode(
        x=alt.X("year:N", title="Year (March to October)"),
        color=alt.Color(
            "Sky:N", scale=alt.Scale(domain=["sunny", "cloudy"], range=["#FF4901", "#306BFF"])
        ),
        xOffset="Sky:N",
    )
    chart2 = (
        (
            c2.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("lo:Q", title="Share of days with a midday dip"), y2="hi:Q"
            )
            + c2.mark_point(filled=True, size=50).encode(y="rate:Q")  # ty: ignore[unresolved-attribute]
        )
        .properties(width=210, height=180)
        .facet(column=alt.Column("site:N", title=None))
    )
    chart2.save(DOCS / "figure_2_dip_rate_sunny_and_cloudy.svg")

    engine = pl.read_parquet(common.WORK / "partB_engine_d8.parquet").select(
        "date", Dip="engine_share", Placebo="placebo_share", Control="control_share"
    )
    eng = engine.unpivot(index="date", variable_name="Half-hours", value_name="share")
    chart3 = (
        alt.Chart(eng)
        .mark_bar()  # ty: ignore[unresolved-attribute]
        .encode(
            x=alt.X("Half-hours:N", sort=["Dip", "Placebo", "Control"], title=None),
            y=alt.Y(
                "mean(share):Q",
                title="Mean share flagged as engine running",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Half-hours:N",
                scale=alt.Scale(
                    domain=["Dip", "Placebo", "Control"], range=["#FF4901", "#306BFF", "#10C5F7"]
                ),
                legend=None,
            ),
        )
        .properties(
            width=300,
            height=200,
            title="D8: engine states in dip half-hours, matched days, and D5, D10, D11",
        )
    )
    chart3.save(DOCS / "figure_3_engine_overlap_at_d8.svg")

    shape = (
        pl.read_parquet(common.WORK / "partB_shape.parquet")
        .with_columns(pl.col("label_solar").fill_nan(None))
        .group_by("site", "kind")
        .agg(share_solar=pl.col("label_solar").mean(), n=pl.col("label_solar").drop_nulls().len())
        .with_columns(label=pl.col("site") + " " + pl.col("kind").str.replace("_", " "))
    )
    chart4 = (
        alt.Chart(shape)
        .mark_bar(color="#306BFF")  # ty: ignore[unresolved-attribute]
        .encode(
            y=alt.Y("label:N", title=None, sort="-x"),
            x=alt.X(
                "share_solar:Q", title="Share labelled solar-shaped", scale=alt.Scale(domain=[0, 1])
            ),
        )
        .properties(width=360, height=220, title="Shape test: known answers and the dip days")
    )
    chart4.save(DOCS / "figure_4_shape_test.svg")


if __name__ == "__main__":
    main()

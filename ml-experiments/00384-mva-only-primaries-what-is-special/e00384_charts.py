"""Draw the figures of experiment 384 from the table `e00384_analysis.py` writes.

PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
    uv run --no-sync python ml-experiments/00384-*/e00384_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00384_analysis as analysis  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

NAME: Final[str] = "00384-mva-only-primaries-what-is-special"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / NAME
WIDTH: Final[int] = 600
GROUPS: Final[list[str]] = ["signed-MW", "MVA-only", "short-history"]
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
SHOWN: Final[dict[str, str]] = {
    "m4_step_rate": "Share of half-hours with a step above 5% of capacity",
    "step_p99": "99th-percentile half-hour step (points of capacity)",
    "acf_1d": "Residual autocorrelation at 1 day",
    "price_corr": "Correlation of residual with the half-hourly price",
    "capacity_mw": "Effective capacity (MW)",
    "n_neighbours": "Interconnected neighbouring Primaries",
}
LABELS: Final[dict[str, str]] = {
    **SHOWN, "step_rate_10": "Share of half-hours with a step above 10% of capacity",
    "m1_profile_share": "Share of variance in the weekly profile",
}  # fmt: skip
CORRELATED: Final[list[str]] = ["m4_step_rate", "step_rate_10", "acf_1d", "price_corr",
                                "m1_profile_share", "capacity_mw"]  # fmt: skip


def colour() -> alt.Color:
    """Return the group colour encoding."""
    return alt.Color("group:N", scale=alt.Scale(domain=GROUPS, range=COLOURS),
                     legend=alt.Legend(title=None, orient="top"))  # fmt: skip


def with_group(table: pl.DataFrame) -> pl.DataFrame:
    """Return the table with the three plotted groups."""
    return table.with_columns(
        group=pl.when(pl.col("label").is_in(analysis.SHORT_HISTORY)).then(pl.lit("short-history"))
        .when(pl.col("units") == "MVA").then(pl.lit("MVA-only")).otherwise(pl.lit("signed-MW"))
    )  # fmt: skip


def strip_figure(table: pl.DataFrame) -> alt.Chart:
    """Return one panel per measure, one dot per Primary, by group."""
    long = table.unpivot(list(SHOWN), index=["label", "group"]).with_columns(
        panel=pl.col("variable").replace(SHOWN)
    )
    base = alt.Chart(long).encode(
        y=alt.Y("group:N", sort=GROUPS, title=None), color=colour(),
        x=alt.X("value:Q", title=None),
    )  # fmt: skip
    dots = base.mark_circle(size=60)  # ty: ignore[invalid-argument-type]
    text = base.mark_text(dx=0, dy=-9, fontSize=8).encode(text="label:N")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(dots, text).properties(width=250, height=90)  # ty: ignore[invalid-argument-type]
        .facet(facet=alt.Facet("panel:N", sort=list(SHOWN.values()), title=None), columns=2)
        .resolve_scale(x="independent")
        .properties(title=alt.TitleParams(
            ["The two metering groups look alike on size, step rate, and autocorrelation",
             "Dot: one Primary, from readings before 2025-07-01."],
            anchor="start", fontSize=11, subtitleFontSize=10))
    )  # fmt: skip


def scatter(*, table: pl.DataFrame, measure: str, label: str) -> alt.LayerChart | alt.FacetChart:
    """Return a scatter of the experiment 377 gap against one measure."""
    points = alt.Chart(table).encode(
        x=alt.X(f"{measure}:Q", title=label),
        y=alt.Y("gap_377:Q", title="Stack minus BG2 (points; positive: stack worse)"),
        color=colour(),
    )
    return alt.layer(
        points.mark_circle(size=70),  # ty: ignore[invalid-argument-type]
        points.mark_text(dx=9, dy=-6, fontSize=9).encode(text="label:N"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=200)


def rho_figure(table: pl.DataFrame) -> alt.LayerChart | alt.FacetChart:
    """Return Spearman rho with its leave-one-out range, by subset and measure."""
    rng = np.random.default_rng(analysis.SEED)
    subsets = {
        "n = 16": [],
        "n = 14 (without D15, D18)": list(analysis.SHORT_HISTORY),
        "MVA-only seven (without D8)": [*analysis.SIGNED_MW, *analysis.SHORT_HISTORY, "D8"],
    }
    rows = []
    for name, drop in subsets.items():
        sub = table.filter(~pl.col("label").is_in(drop))
        for measure in CORRELATED:
            r = analysis.correlate(table=sub, measure=measure, gap="gap_377", rng=rng)
            rows.append({**r, "subset": name, "measure": LABELS.get(measure, measure)})
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y("measure:N", title=None, sort=None, axis=alt.Axis(labelLimit=420)),
        yOffset=alt.YOffset("subset:N", sort=list(subsets)),
        color=alt.Color("subset:N", sort=list(subsets),
                        scale=alt.Scale(domain=list(subsets), range=COLOURS),
                        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=300)),
    )  # fmt: skip
    rule = (alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#000000", strokeDash=[3, 3])
            .encode(x="x:Q"))  # fmt: skip  # ty: ignore[unresolved-attribute]
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("loo_min:Q", title="Spearman rho with the gap"), x2="loo_max:Q"
    )
    dots = base.mark_circle(size=60).encode(x="rho:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(rule, bars, dots).properties(
        width=WIDTH, height=300, padding={"left": 5, "right": 40, "top": 5, "bottom": 5},
        title=alt.TitleParams(
            ["Step rates track the gap only once the short-history Primaries are set aside",
             "Dot: rank correlation with the experiment 377 gap. Line: leave-one-out range."],
            anchor="start", fontSize=11, subtitleFontSize=10))  # fmt: skip


def main() -> None:
    """Write the three SVG figures."""
    table = with_group(pl.read_parquet(analysis.OWN / "measures_table.parquet"))
    DOCS.mkdir(parents=True, exist_ok=True)
    strip_figure(table).save(DOCS / "figure_1_groups_compared.svg")
    rho_figure(table).save(DOCS / "figure_2_rank_correlations.svg")
    alt.vconcat(
        scatter(table=table, measure="m4_step_rate",
                label="Share of half-hours with a step above 5% of capacity"),
        scatter(table=table, measure="price_corr",
                label="Correlation of the residual with the half-hourly price"),
    ).properties(title=alt.TitleParams(
        ["The gap to BG2 against step rate and price correlation, 16 Primaries",
         "Dot: one Primary. D8 is the largest gap."], anchor="start", fontSize=11,
        subtitleFontSize=10)).save(DOCS / "figure_3_gap_scatter.svg")  # fmt: skip


if __name__ == "__main__":
    main()

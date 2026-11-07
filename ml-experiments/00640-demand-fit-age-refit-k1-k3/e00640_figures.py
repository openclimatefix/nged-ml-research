"""Experiment 640 figures (CPU): read the score tables and write SVGs for the page.

Usage: python e00640_figures.py OUT_DIR
"""

import sys
from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00640-demand-fit-age-refit-k1-k3" / "score"
NETS = ["K1", "K3"]
COLOUR = alt.Scale(domain=NETS, range=[theme.DATA_BLUE, theme.DATA_PURPLE])
LEGEND = alt.Legend(title=None, orient="bottom")
W = 300
AGE_AXIS = alt.Axis(values=[0, 4, 8, 16, 26])
AGE_TITLE = "Fit age (weeks before the forecast month)"


def headline(*, out: Path) -> None:
    """Draw the CRPS increase against the age-0 fit, with 97.5% month-block intervals."""
    d = pl.read_csv(ROOT / "contrasts.csv").with_columns(
        net=pl.col("net").str.to_uppercase(),
        panel=pl.when(pl.col("subset") == "all Primaries")
        .then(pl.lit("All 16 Primaries"))
        .otherwise(pl.lit("14 Primaries (without D15 and D18)")),
    )
    base = alt.Chart(d).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("age:Q", title=AGE_TITLE, axis=AGE_AXIS, scale=alt.Scale(domain=[0, 27])),
        color=alt.Color("net:N", scale=COLOUR, legend=LEGEND),
    )
    chart = (
        (
            base.mark_line(point=True, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("pct_vs_age0:Q", title="CRPS above the age-0 fit (%)")
            )
            + base.mark_errorbar(ticks=True).encode(y=alt.Y("lo:Q", title=""), y2="hi:Q")  # ty: ignore[unresolved-attribute]
        )
        .properties(width=W, height=280)
        .facet(column=alt.Column("panel:N", title=None, sort=["All 16 Primaries"]))
    )
    chart.save(str(out / "crps_vs_fit_age.svg"))


def by_group(*, out: Path) -> None:
    """Draw the CRPS against fit age for D15 and D18 beside the other 14 Primaries."""
    parts = []
    for net in ("k1", "k3"):
        d = pl.read_parquet(ROOT / f"cells_{net}.parquet").filter(pl.col("free"))
        g = (
            d.with_columns(
                panel=pl.when(pl.col("group") == "D15 and D18")
                .then(pl.lit("D15 and D18"))
                .otherwise(pl.lit("14 other Primaries"))
            )
            .group_by(["panel", "age"])
            .agg((pl.col("crps").sum() / pl.col("n").sum() * 100).alias("crps"))
            .with_columns(net=pl.lit(net.upper()))
        )
        parts.append(g)
    d = pl.concat(parts)
    chart = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("age:Q", title=AGE_TITLE, axis=AGE_AXIS, scale=alt.Scale(domain=[0, 27])),
            y=alt.Y("crps:Q", title="CRPS (% of capacity)", scale=alt.Scale(zero=True)),
            color=alt.Color("net:N", scale=COLOUR, legend=LEGEND),
        )
        .properties(width=W, height=280)
        .facet(column=alt.Column("panel:N", title=None, sort=["14 other Primaries"]))
        .resolve_scale(y="independent")
    )
    chart.save(str(out / "crps_by_primary_group.svg"))


def coverage(*, out: Path) -> None:
    """Draw the event-free q05-q95 coverage against fit age."""
    d = pl.read_csv(ROOT / "by_age.csv").with_columns(net=pl.col("net").str.to_uppercase())
    line = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("age:Q", title=AGE_TITLE, axis=AGE_AXIS, scale=alt.Scale(domain=[0, 27])),
            y=alt.Y(
                "cover90:Q",
                title="Share of readings inside q05-q95",
                scale=alt.Scale(domain=[0.7, 0.95]),
            ),
            color=alt.Color("net:N", scale=COLOUR, legend=LEGEND),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.9]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color="#7a7a7a")
        .encode(y="y:Q")
    )
    (line + rule).properties(width=2 * W, height=260).save(str(out / "coverage_vs_fit_age.svg"))


def weekly(*, out: Path) -> None:
    """Draw the slope per 4 weeks by lead band for the 14 Primaries and all 16."""
    d = (
        pl.read_csv(ROOT / "slopes.csv")
        .filter(~pl.col("subset").str.contains(r"\|"))
        .filter(pl.col("subset").is_in(["all Primaries, event-free rows", "without D15 and D18"]))
        .with_columns(
            net=pl.col("net").str.to_uppercase(),
            panel=pl.when(pl.col("subset") == "without D15 and D18")
            .then(pl.lit("14 Primaries (without D15 and D18)"))
            .otherwise(pl.lit("All 16 Primaries")),
        )
    )
    order = ["0-24 h", "24-120 h", "120-240 h", "0-240 h"]
    base = alt.Chart(d).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lead:N", title="Forecast lead time", sort=order),
        color=alt.Color("net:N", scale=COLOUR, legend=LEGEND),
        yOffset="net:N",
    )
    chart = (
        (
            base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "slope_pct_per_4wk:Q",
                    title="CRPS change per 4 weeks of fit age (% of age-0 CRPS)",
                )
            )
            + base.mark_errorbar().encode(x=alt.X("lo:Q", title=""), x2="hi:Q")  # ty: ignore[unresolved-attribute]
        )
        .properties(width=W, height=180)
        .facet(column=alt.Column("panel:N", title=None, sort=["All 16 Primaries"]))
    )
    chart.save(str(out / "slope_by_lead_band.svg"))


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    headline(out=out_dir)
    by_group(out=out_dir)
    coverage(out=out_dir)
    weekly(out=out_dir)

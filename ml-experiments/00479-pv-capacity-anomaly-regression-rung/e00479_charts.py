"""Figures of experiment 479, written as SVG into the experiment's docs folder.

Reads the tables of `e00479_summary.py`. Substations may be named; no metered generator appears
(farms are Solar A to F by letter and only pooled ratios are drawn).
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00479_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 640
NO_CLOUD: Final[str] = "without cloud terms"
CLOUD: Final[str] = "with cloud terms"
VARIANT_SCALE = alt.Scale(domain=[NO_CLOUD, CLOUD], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE])
WORLDS: Final[dict[str, str]] = {
    "easy": "Easy world",
    "cloud_linear": "Cloud-following load (linear)",
    "cloud_squared": "Cloud-following load (squared, report only)",
}


def label(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the readable estimator-variant column."""
    return frame.with_columns(
        variant=pl.when(pl.col("cloud")).then(pl.lit(CLOUD)).otherwise(pl.lit(NO_CLOUD))
    )


def rule(*, data: pl.DataFrame, value: float) -> alt.Chart:
    """Return a vertical gate line that shares the facet's data."""
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#606060", strokeDash=[4, 3], aria=False)
        .encode(x=alt.datum(value))
    )


def recovery_chart() -> alt.TopLevelMixin:
    """Headline: raw recovery of an injected farm, by world and estimator variant."""
    gates = label(frame=pl.read_parquet(common.WORK_DIR / "gates.parquet")).with_columns(
        world_label=pl.col("world").replace_strict(WORLDS)
    )
    dots = (
        alt.Chart(gates)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            x=alt.X(
                "recovery:Q",
                title="Raw recovery (1 is all)",
                scale=alt.Scale(domain=[0.5, 1.1]),
            ),
            y=alt.Y(
                "site:N",
                title=None,
                sort=list(common.SIGNED_SITES),
                axis=alt.Axis(labelOverlap=False),
            ),
            color=alt.Color(
                "variant:N", scale=VARIANT_SCALE, legend=alt.Legend(orient="top", title=None)
            ),
            yOffset="variant:N",
        )
    )
    line = rule(data=gates, value=common.GATE_RECOVERY)
    return (
        alt.layer(dots, line)
        .properties(width=WIDTH / 3 - 40, height=170)
        .facet(column=alt.Column("world_label:N", title=None, sort=list(WORLDS.values())))
        .properties(
            title=alt.TitleParams(
                text="Injected-farm recovery by world (median over 2 MW injections of five farms)",
                subtitle="Dashed line: the 0.85 gate. Six Primaries with a signed MW reading.",
                anchor="start",
            )
        )
    )


def placebo_chart() -> alt.TopLevelMixin:
    """Both placebos against the 0.20 gate."""
    gates = label(frame=pl.read_parquet(common.WORK_DIR / "gates.parquet")).filter(
        pl.col("world") == "easy"
    )
    a = gates.select(
        "site", "variant", value=pl.col("placebo_p95"), test=pl.lit("Profile-aware placebo")
    )
    b = label(frame=pl.read_parquet(common.WORK_DIR / "placebo_b_summary.parquet")).select(
        "site", "variant", value=pl.col("share_p95"), test=pl.lit("Irradiance-shuffle placebo")
    )
    data = pl.concat([a, b])
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            x=alt.X(
                "value:Q",
                title="Placebo share, 95th percentile",
                scale=alt.Scale(domain=[0, 0.3]),
            ),
            y=alt.Y(
                "site:N",
                title=None,
                sort=list(common.SIGNED_SITES),
                axis=alt.Axis(labelOverlap=False),
            ),
            color=alt.Color(
                "variant:N", scale=VARIANT_SCALE, legend=alt.Legend(orient="top", title=None)
            ),
            yOffset="variant:N",
        )
    )
    return (
        alt.layer(dots, rule(data=data, value=common.GATE_PLACEBO))
        .properties(width=WIDTH / 2 - 60, height=170)
        .facet(column=alt.Column("test:N", title=None))
        .properties(
            title=alt.TitleParams(
                text="Both placebos stay near or under the 0.20 gate",
                subtitle="Dashed line: the 0.20 gate. 20 shuffles per Primary.",
                anchor="start",
            )
        )
    )


def spurious_chart() -> alt.TopLevelMixin:
    """Capacity the estimator invents from a cloud-following load with no farm injected."""
    summary = label(frame=pl.read_parquet(common.WORK_DIR / "known_summary.parquet")).filter(
        pl.col("kind").is_in(["load_only_linear", "load_only_squared"])
    )
    summary = summary.with_columns(
        load=pl.col("kind").replace_strict(
            {
                "load_only_linear": "Linear cloud-following load",
                "load_only_squared": "Squared cloud-following load",
            }
        )
    )
    dots = (
        alt.Chart(summary)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            x=alt.X("recovery_median:Q", title="Share of a 2 MW farm found"),
            y=alt.Y(
                "site:N",
                title=None,
                sort=list(common.SIGNED_SITES),
                axis=alt.Axis(labelOverlap=False),
            ),
            color=alt.Color(
                "variant:N", scale=VARIANT_SCALE, legend=alt.Legend(orient="top", title=None)
            ),
            yOffset="variant:N",
        )
    )
    return (
        alt.layer(dots, rule(data=summary, value=0.0))
        .properties(width=WIDTH / 2 - 60, height=170)
        .facet(column=alt.Column("load:N", title=None))
        .properties(
            title=alt.TitleParams(
                text="A cloud-following load looks like solar capacity without the cloud terms",
                subtitle="No farm is injected. Zero is the right answer.",
                anchor="start",
            )
        )
    )


def capacity_chart() -> alt.TopLevelMixin:
    """Capacity at the latest season-year: this rung against experiment 474's table."""
    mine = label(
        frame=pl.read_parquet(common.WORK_DIR / "capacity_table.parquet").filter(
            pl.col("prior") == "none"
        )
    ).select("site", "variant", low=pl.col("p10"), mid=pl.col("p50"), high=pl.col("p90"))
    e474 = (
        pl.read_parquet(common.E474_WORK_DIR / "nged_table.parquet")
        .filter(pl.col("season_year") == common.LATEST_SEASON_YEAR)
        .select(
            "site",
            variant=pl.lit("experiment 474"),
            low=pl.col("ac_p10"),
            mid=pl.col("ac_p50"),
            high=pl.col("ac_p90"),
        )
    )
    data = pl.concat([mine, e474]).filter(pl.col("site") != "D15")
    order = data.filter(pl.col("variant") == CLOUD).sort("mid", descending=True)["site"].to_list()
    scale = alt.Scale(
        domain=[NO_CLOUD, CLOUD, "experiment 474"],
        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#606060"],
    )
    base = alt.Chart(data).encode(
        y=alt.Y("site:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)),
        yOffset="variant:N",
        color=alt.Color("variant:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
    )
    bars = base.mark_rule(strokeWidth=3, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="AC capacity (MW), season-year July 2025 to June 2026"), x2="high:Q"
    )
    dots = base.mark_point(filled=True, size=40, aria=False).encode(x="mid:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, dots).properties(
        width=WIDTH - 90,
        height=24 * len(order) + 40,
        title=alt.TitleParams(
            text="Unmetered solar capacity: anomaly regression against experiment 474",
            subtitle=[
                "Dot: median. Line: 10th to 90th percentile of a 2,000-draw month-block bootstrap",
                "(not widened). Recovery below 1 in the known answer means the capacity is higher.",
            ],
            anchor="start",
        ),
    )


def main() -> None:
    """Write the figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    recovery_chart().save(DOCS_DIR / "figure_01_recovery.svg")
    placebo_chart().save(DOCS_DIR / "figure_02_placebos.svg")
    spurious_chart().save(DOCS_DIR / "figure_03_cloud_load_without_farm.svg")
    capacity_chart().save(DOCS_DIR / "figure_04_capacity.svg")


if __name__ == "__main__":
    main()

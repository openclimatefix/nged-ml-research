# ruff: noqa: E501
"""Experiment 722: the figure of the cloud-class audit values against their thresholds.

Reads `u0_class_audit.csv` (aggregates only: one value per site and condition) and writes one SVG
beside the write-up. Run: `uv run --no-sync python e00722_figure.py <output.svg>`.
"""

import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00722_guard as guard
import plotting.ocf_theme as ocf
import polars as pl

PANELS: Final[tuple[tuple[str, str, float, float, str], ...]] = (
    (
        "3",
        "Condition 3: share of days that are intermittent, at each of 16 Primaries",
        0.2,
        0.6,
        "Pass if 0.2 to 0.6",
    ),
    (
        "2",
        "Condition 2: share of clear days with a large change, at each of 6 farms",
        0.0,
        0.05,
        "Pass if at most 0.05",
    ),
    (
        "1",
        "Condition 1: share of days that are clear, at each of 6 farms",
        0.7,
        1.0,
        "Pass if at least 0.7",
    ),
    (
        "4",
        "Condition 4: share of days that change class when the elevation limit moves, all sites pooled",
        0.0,
        0.1,
        "Pass if at most 0.1",
    ),
)
"""Condition label in the CSV, panel title, lower and upper end of the passing range, key text."""

WIDTH_PX: Final[int] = 640


def build(*, audit: pl.DataFrame) -> alt.VConcatChart:
    """Draw one strip of dots per condition, with the passing range shaded.

    Args:
        audit: The class-audit table, one row per condition and site.

    Returns:
        The vertically stacked chart.
    """
    panels = []
    for condition, title, low, high, key in PANELS:
        rows = (
            audit.filter(pl.col("condition").cast(pl.String) == condition)
            .filter(pl.col("scope") == "all_sites" if condition == "4" else pl.lit(True))
            .with_row_index("lane")
            .with_columns(
                (pl.col("lane") % 4).alias("lane"),
                pl.when(pl.col("passed"))
                .then(pl.lit("passes"))
                .otherwise(pl.lit("fails"))
                .alias("result"),
            )
            .select("scope", "value", "lane", "result")
        )
        band = pl.DataFrame({"low": [low], "high": [high]})
        shade = (
            alt.Chart(alt.Data(values=band.to_dicts()))
            .mark_rect(color="#EDEDED")
            .encode(x=alt.X("low:Q", scale=alt.Scale(domain=[0, 1])), x2="high:Q")  # ty: ignore[unresolved-attribute]
        )
        dots = (
            alt.Chart(alt.Data(values=rows.to_dicts()))
            .mark_circle(size=70, opacity=0.9, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("value:Q", title=None, axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
                y=alt.Y("lane:Q", axis=None, scale=alt.Scale(domain=[-0.5, 3.5])),
                color=alt.Color(
                    "result:N",
                    scale=alt.Scale(
                        domain=["passes", "fails"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                    ),
                    legend=alt.Legend(title=None, orient="top") if condition == "3" else None,
                ),
            )
        )
        panels.append(
            alt.layer(shade, dots).properties(
                width=WIDTH_PX - 40,
                height=50,
                title=alt.TitleParams(
                    f"{title}. Grey: {key.lower()}.", fontSize=12, anchor="start"
                ),
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Every condition of the cloud-class audit fails at the sites tested, except the clear-day share",
            fontSize=13,
            anchor="start",
        )
    )


def main() -> None:
    """Read the audit table and write the figure to the path given on the command line."""
    audit = pl.read_csv(
        guard.OUT_DIR / "u0_class_audit.csv", schema_overrides={"condition": pl.String}
    )
    build(audit=audit).save(Path(sys.argv[1]))


if __name__ == "__main__":
    main()

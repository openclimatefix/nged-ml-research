# ruff: noqa: E501
"""Experiment 571: the recovery chart.

The chart shows the share of the oracle's CRPS gain each arm recovers (injected Primaries, 24-120 h).
"""

from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00571-drift-offset-and-embedding-hybrid-known-answer-worlds"
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
LABEL = {"D5": "Offset (D5)", "A10": "Embedding (A10)", "HJ": "Joint hybrid", "HS": "Staged hybrid"}
WORLD = {"Wa-big": "Steps only", "Wb1-big": "Sensitivity only", "Wc-big": "Both"}


def main() -> None:
    """Write the SVG next to the page."""
    t = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / NAME / "summary_recovery.csv").filter(
        (pl.col("group") == "both")
        & pl.col("arm").is_in(list(LABEL))
        & pl.col("world").is_in(list(WORLD))
    )
    t = t.with_columns(
        arm_label=pl.col("arm").replace(LABEL),
        world_label=pl.col("world").replace(WORLD),
        pct=pl.col("recovery") * 100,
    )
    chart = (
        alt.Chart(alt.Data(values=t.to_dicts()))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm_label:N", title=None, sort=list(LABEL.values())),
            x=alt.X("pct:Q", title="Share of the oracle's CRPS gain recovered (%)"),
            color=alt.Color(
                "arm_label:N",
                legend=None,
                scale=alt.Scale(
                    range=[theme.BRAND_ORANGE, theme.DATA_GREEN, theme.DATA_PURPLE, theme.DATA_SKY]
                ),
            ),
            row=alt.Row("world_label:N", title=None, sort=list(WORLD.values())),
        )
        .properties(width=420, height=70)
    )
    chart.save(DOCS / "recovery_by_world.svg")


if __name__ == "__main__":
    main()

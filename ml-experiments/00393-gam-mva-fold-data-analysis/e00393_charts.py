"""Draw the figures of experiment 393 from the tables `e00393_analysis` stored.

DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00393-*/e00393_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00393-gam-mva-fold-data-analysis" / "work"
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00393-gam-mva-fold-data-analysis"
)
COLOURS: Final[dict[str, str]] = {"signed-MW": ocf.DATA_BLUE, "MVA-only": ocf.BRAND_ORANGE}
SCALE = alt.Scale(domain=list(COLOURS), range=list(COLOURS.values()))
LEGEND = alt.Legend(title=None, orient="top")


def main() -> None:
    """Write the three SVG figures."""
    c2 = pl.read_parquet(WORK / "contrast2_table.parquet").filter(
        pl.col("group") != "short-history"
    )
    points = alt.Chart(c2).encode(
        x=alt.X("share_nz:Q", title="Share of half-hours below 10% of the Primary's peak"),
        y=alt.Y("gap_bg2:Q", title="Stack minus BG2, CRPS, days 1 to 10 (points of capacity)"),
        color=alt.Color("group:N", scale=SCALE, legend=LEGEND),
    )
    chart = (
        points.mark_point(filled=True, size=80)  # ty: ignore[unresolved-attribute]
        + points.mark_text(dx=12, dy=-6).encode(text="label:N")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=440,
        height=260,
        title=alt.TitleParams(
            "The stack trails BG2 most at the Primaries with near-zero half-hours (D12, D22, D8)",
            anchor="start",
        ),
    )
    chart.save(DOCS / "gap_against_near_zero_share.svg")

    c1 = (
        pl.read_parquet(WORK / "contrast1_table.parquet")
        .unpivot(
            index="label", on=["signed_gap", "folded_gap"], variable_name="target", value_name="gap"
        )
        .with_columns(
            pl.col("target").replace({"signed_gap": "signed target", "folded_gap": "folded target"})
        )
    )
    folded = (
        alt.Chart(c1)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80)
        .encode(
            y=alt.Y("label:N", title=None),
            x=alt.X("gap:Q", title="Stack minus BG2, CRPS, days 1 to 10 (points of capacity)"),
            color=alt.Color(
                "target:N",
                scale=alt.Scale(
                    domain=["signed target", "folded target"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=LEGEND,
            ),
        )
        .properties(
            width=440,
            height=200,
            title=alt.TitleParams(
                "Folding the signed target barely moves the gap at the six signed-MW Primaries",
                anchor="start",
            ),
        )
    )
    folded.save(DOCS / "signed_against_folded_gap.svg")

    c3 = pl.read_parquet(WORK / "contrast3_level_table.parquet")
    order = ["0-10%", "10-20%", "20-40%", "40-60%", "60-80%", "80%+"]
    levels = (
        alt.Chart(c3)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "level_bin:N", sort=order, title="Observation as a share of the Primary's peak"
            ),
            y=alt.Y("gap:Q", title="Stack minus BG2, CRPS (points of capacity)"),
            xOffset="group:N",
            color=alt.Color("group:N", scale=SCALE, legend=LEGEND),
        )
        .properties(
            width=440,
            height=240,
            title=alt.TitleParams(
                "The stack's gap to BG2 is largest when demand is lowest", anchor="start"
            ),
        )
    )
    levels.save(DOCS / "gap_by_flow_level.svg")


if __name__ == "__main__":
    main()

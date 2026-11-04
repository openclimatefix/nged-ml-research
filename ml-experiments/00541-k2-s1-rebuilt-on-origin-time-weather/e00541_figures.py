# ruff: noqa: E501
"""Draw experiment 541's two figures from the stored score tables (SVG; optimise with `npx.

svgo@4 --multipass --precision=1 --final-newline`).

1. `figure_1_crps_by_lead.svg`: CRPS by lead band of the old K2-S1, the rebuilt K2-S1, L-1d, and
A-pool. 2. `figure_2_rebuilt_minus_old.svg`: rebuilt minus old K2-S1 by lead band, Bonferroni
month-block intervals.
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from e00541_common import SCORES
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00541-k2-s1-rebuilt-on-origin-time-weather"
)
BANDS: Final[list[str]] = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h"]
SETS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}
LABELS: Final[dict[str, str]] = {
    "K2-S1": "K2-S1, old weather",
    "K2-S1-origin": "K2-S1, heads trained on look-ahead weather, scored on origin-time weather",
    "K2-S1-origin-trained": "K2-S1, heads trained and scored on origin-time weather",
    "L-1d": "L-1d (daily)",
    "A-pool": "A-pool",
}
COLOURS: Final[dict[str, str]] = {
    "K2-S1": theme.BLACK_2,
    "K2-S1-origin": theme.BRAND_ORANGE_LIGHT,
    "K2-S1-origin-trained": theme.BRAND_ORANGE,
    "L-1d": theme.DATA_BLUE,
    "A-pool": theme.DATA_SKY,
}

CONTRASTS: Final[dict[str, str]] = {
    "K2-S1-origin - K2-S1": theme.BRAND_ORANGE_LIGHT,
    "K2-S1-origin-trained - K2-S1": theme.BRAND_ORANGE,
}


def figure_crps() -> None:
    """Draw CRPS by lead band."""
    t = pl.read_csv(SCORES / "absolute.csv").filter(
        pl.col("arm").is_in(list(LABELS))
        & pl.col("set").is_in(list(SETS))
        & pl.col("band").is_in(BANDS)
    )
    t = t.with_columns(arm_label=pl.col("arm").replace(LABELS))
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        panels.append(
            alt.Chart(part)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("band:N", sort=BANDS, title="Lead time of the forecast"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm_label:N",
                    title=None,
                    scale=alt.Scale(
                        domain=list(LABELS.values()), range=[COLOURS[a] for a in LABELS]
                    ),
                ),
            )
            .properties(width=520, height=200, title=alt.TitleParams(label, anchor="start"))
        )  # ty: ignore[unresolved-attribute]
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Without the look-ahead, K2-S1 scores worse than before beyond 24 h, with either training weather.",
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_1_crps_by_lead.svg")


def figure_difference() -> None:
    """Draw the rebuilt-minus-old difference with its Bonferroni month-block interval."""
    t = pl.read_csv(SCORES / "contrasts.csv").filter(
        pl.col("contrast").is_in(list(CONTRASTS))
        & pl.col("set").is_in(list(SETS))
        & pl.col("band").is_in(BANDS)
    )
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        base = alt.Chart(part).encode(
            x=alt.X("band:N", sort=BANDS, title="Lead time of the forecast"), xOffset="contrast:N"
        )
        colour = alt.Color(
            "contrast:N",
            title=None,
            scale=alt.Scale(domain=list(CONTRASTS), range=list(CONTRASTS.values())),
            legend=None,
        )
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low_bonferroni_month:Q", title="Rebuilt minus old CRPS (points)"),
            y2="high_bonferroni_month:Q",
            color=colour,
        )
        dot = base.mark_point(filled=True, size=70).encode(y="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color=theme.BLACK_2)
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=520, height=190, title=alt.TitleParams(label, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Removing the look-ahead raises K2-S1's CRPS from 24 h onwards.",
                "Left of each pair: heads trained on look-ahead weather.",
                "Right: heads trained on origin-time weather.",
                "Dot: difference. Bar: Bonferroni interval, 5 month blocks (near min to max).",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_2_rebuilt_minus_old.svg")


if __name__ == "__main__":
    figure_crps()
    figure_difference()

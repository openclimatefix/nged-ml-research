# ruff: noqa: E501
"""Draw experiment 548's figures from the stored score tables (SVG; optimise with svgo as CLAUDE.md says).

1. `figure_1_crps_by_lead.svg`: CRPS by lead band of E, L-1d, R-DW, A-pool, and the rebuilt K2-S1.
2. `figure_2_e_minus_comparators.svg`: E minus A-pool, L-1d, and R-DW by lead band, Bonferroni
   month-block intervals.
3. `figure_3_injected_shift_leak.svg`: injected-shift leak of E, L-1d, and R-DW.
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows"
    / "scores"
)
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows"
)
BANDS: Final[list[str]] = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h"]
SETS: Final[dict[str, str]] = {
    "eight": "MVA-only eight Primaries",
    "six": "Signed-MW six Primaries",
}
LABELS: Final[dict[str, str]] = {
    "E": "E: L-1d plus trailing temperature windows",
    "L-1d": "L-1d (daily-refit lag-free network)",
    "R-DW": "R-DW (L-1d plus day-ago and week-ago lags)",
    "A-pool": "A-pool (lag-enabled network)",
    "K2-S1-origin-trained": "K2-S1 rebuilt on origin-time weather",
}
COLOURS: Final[dict[str, str]] = {
    "E": theme.BRAND_ORANGE,
    "L-1d": theme.DATA_BLUE,
    "R-DW": theme.DATA_GREEN,
    "A-pool": theme.DATA_PURPLE,
    "K2-S1-origin-trained": theme.DATA_SKY,
}
CONTRASTS: Final[dict[str, str]] = {
    "E - A-pool": theme.DATA_PURPLE,
    "E - L-1d": theme.DATA_BLUE,
    "E - R-DW": theme.DATA_GREEN,
}
WIDTH: Final[int] = 560


def figure_crps() -> None:
    """Draw CRPS by lead band."""
    t = pl.read_csv(SCORES / "absolute.csv").filter(
        (pl.col("slice") == "all rows")
        & (pl.col("metric") == "crps")
        & pl.col("arm").is_in(list(LABELS))
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
                y=alt.Y("value:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm_label:N",
                    title=None,
                    legend=alt.Legend(orient="top", columns=1, labelLimit=600),
                    scale=alt.Scale(
                        domain=list(LABELS.values()), range=[COLOURS[a] for a in LABELS]
                    ),
                ),
            )
            .properties(width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start"))
        )  # ty: ignore[unresolved-attribute]
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Trailing temperature windows lower the daily lag-free network's CRPS in every band;",
                "it still trails the lag-enabled A-pool at 0 to 120 h at the eight.",
                "CRPS in points of capacity; smaller is better. Five folds, two seeds, no labelled event.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_1_crps_by_lead.svg")


def figure_difference() -> None:
    """Draw E minus each comparator with its Bonferroni month-block interval."""
    t = pl.read_csv(SCORES / "contrasts.csv").filter(
        (pl.col("slice") == "all rows")
        & (pl.col("metric") == "crps")
        & pl.col("contrast").is_in(list(CONTRASTS))
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
            legend=alt.Legend(orient="top", labelLimit=600),
            scale=alt.Scale(domain=list(CONTRASTS), range=list(CONTRASTS.values())),
        )
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low_bonferroni_month:Q", title="E minus comparator (points)"),
            y2="high_bonferroni_month:Q",
            color=colour,
        )
        dot = base.mark_point(filled=True, size=60).encode(y="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color=theme.BLACK_2)
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "E is better than L-1d in every band by point estimate, and worse than A-pool",
                "at 0 to 120 h at the eight. Dot: difference in CRPS (negative favours E).",
                "Bar: Bonferroni interval, 5 month blocks (near min to max). Line: same as comparator.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_2_e_minus_comparators.svg")


def figure_leak() -> None:
    """Draw the injected-shift leak by lead group."""
    t = pl.read_csv(SCORES / "stress_leak.csv").filter(pl.col("band") != "all")
    t = t.with_columns(arm_label=pl.col("arm").replace(LABELS))
    order = ["day1", "days2_3", "days4_10"]
    names = {"day1": "Day 1", "days2_3": "Days 2 to 3", "days4_10": "Days 4 to 10"}
    t = t.with_columns(group=pl.col("band").replace(names))
    chart = (
        alt.Chart(t.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("group:N", sort=[names[o] for o in order], title="Lead time of the forecast"),
            xOffset="arm_label:N",
            y=alt.Y("leak:Q", title="Leak (points of capacity)"),
            color=alt.Color(
                "arm_label:N",
                title=None,
                legend=alt.Legend(orient="top", columns=1, labelLimit=600),
                scale=alt.Scale(
                    domain=[LABELS[a] for a in ("E", "L-1d", "R-DW")],
                    range=[COLOURS[a] for a in ("E", "L-1d", "R-DW")],
                ),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=alt.TitleParams(
                [
                    "The trailing windows add little to the forecast's response to an injected load shift;",
                    "the day-ago and week-ago lags add more. Leak: change in the median forecast,",
                    "points of capacity; smaller is better.",
                    "Seed 0, five folds, 511's stress world.",
                ],
                anchor="start",
            ),
        )
    )
    chart.save(DOCS / "figure_3_injected_shift_leak.svg")


if __name__ == "__main__":
    figure_crps()
    figure_difference()
    figure_leak()

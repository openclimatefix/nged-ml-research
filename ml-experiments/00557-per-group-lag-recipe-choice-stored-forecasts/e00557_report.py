# ruff: noqa: E501
"""Figures of experiment 557 from the tables its scorer stored.

Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

import os
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

DD: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
SCORES: Final[Path] = DD / "00557-per-group-lag-recipe-choice-stored-forecasts" / "scores"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00557-per-group-lag-recipe-choice-stored-forecasts"
)
BANDS: Final[list[str]] = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h"]
LABELS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}


def main() -> None:
    """Draw figure 1 (lag-light minus lag-free, with intervals) and figure 2 (absolute CRPS)."""
    contrasts = (
        pl.read_csv(SCORES / "contrasts.csv")
        .filter(
            pl.col("band").is_in(BANDS),
            pl.col("slice") == "all rows",
            pl.col("metric") == "crps",
            pl.col("contrast") == "RDWT-D15 - E",
            pl.col("set").is_in(list(LABELS)),
        )
        .with_columns(group=pl.col("set").replace(LABELS))
        .to_pandas()
    )
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(domain=list(LABELS.values()), range=[theme.DATA_BLUE, theme.BRAND_ORANGE]),
        title="Primaries",
    )
    x = alt.X("band:N", sort=BANDS, title="Lead time")
    base = alt.Chart(contrasts)
    rule = base.mark_rule().encode(
        x=x,
        y=alt.Y("low_month:Q", title="Lag-light minus lag-free CRPS (points of capacity)"),
        y2="high_month:Q",
        color=colour,
        xOffset="group:N",
    )
    pts = base.mark_point(filled=True, size=60).encode(x=x, y="difference:Q", color=colour, xOffset="group:N")  # fmt: skip
    zero = alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(color="black", strokeDash=[3, 3]).encode(y="y:Q")  # fmt: skip
    (rule + pts + zero).properties(
        width=480,
        height=260,
        title=alt.TitleParams(
            "Lag-light is better in every band (negative favours lag-light). Bars: month-block interval, 5 blocks.",
            anchor="start",
        ),
    ).save(DOCS_DIR / "figure_1_lag_light_minus_lag_free.svg")
    absolute = (
        pl.read_csv(SCORES / "absolute.csv")
        .filter(
            pl.col("set") == "fourteen",
            pl.col("band").is_in(BANDS),
            pl.col("slice") == "all rows",
            pl.col("metric") == "crps",
            pl.col("arm").is_in(["RDWT-D15", "E", "c_per_group", "A-pool"]),
        )
        .with_columns(
            option=pl.col("arm").replace(
                {
                    "RDWT-D15": "(a) lag-light everywhere",
                    "E": "(b) lag-free everywhere",
                    "c_per_group": "(c) per-group choice",
                    "A-pool": "A-pool (reference)",
                }
            )
        )
        .to_pandas()
    )
    order = [
        "(a) lag-light everywhere",
        "(b) lag-free everywhere",
        "(c) per-group choice",
        "A-pool (reference)",
    ]
    alt.Chart(absolute).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("band:N", sort=BANDS, title="Lead time"),
        y=alt.Y("value:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "option:N",
            sort=order,
            scale=alt.Scale(
                domain=order, range=[theme.DATA_BLUE, "#b07aa1", theme.BRAND_ORANGE, "black"]
            ),
            title="Option",
        ),
    ).properties(
        width=480,
        height=260,
        title=alt.TitleParams(
            "CRPS of the three options at all 14 Primaries (the six: clean rows)", anchor="start"
        ),
    ).save(DOCS_DIR / "figure_2_options_crps.svg")


if __name__ == "__main__":
    main()

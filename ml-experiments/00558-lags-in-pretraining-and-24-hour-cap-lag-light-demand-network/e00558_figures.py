# ruff: noqa: D103, E501
"""Draw experiment 558's two figures from the stored score tables."""

import os
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

NAME: Final[str] = "00558-lags-in-pretraining-and-24-hour-cap-lag-light-demand-network"
OUT: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / NAME / "work/scores"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
BANDS: Final[list[str]] = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h"]
SETS: Final[dict[str, str]] = {
    "eight": "MVA-only eight Primaries",
    "six": "Signed-MW six Primaries",
}
ARMS: Final[dict[str, str]] = {
    "L-1d": theme.BLACK_2,
    "R-DW": theme.DATA_GREEN,
    "R-DW-24": theme.BRAND_ORANGE,
    "R-DW-24-PT": theme.DATA_BLUE,
    "A-pool-24": theme.DATA_PURPLE,
}


def figure_crps() -> None:
    t = pl.read_csv(OUT / "absolute.csv").filter(
        pl.col("arm").is_in(list(ARMS)) & pl.col("band").is_in(BANDS)
    )
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        chart = (
            alt.Chart(part)
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("band:N", sort=BANDS, title="Lead time"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(ARMS), range=list(ARMS.values())),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )  # ty: ignore[unresolved-attribute]
            .properties(width=560, height=230, title=alt.TitleParams(label, anchor="start"))
        )
        panels.append(chart)
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "The 24 h cap costs about 0.1 points at 24-120 h; lags in pretraining win it back.",
                "Interim, unaudited. Lower is better.",
            ],
            anchor="start",
            fontSize=12,
        )
    ).save(DOCS / "figure_1_crps_by_lead_band.svg")


def figure_contrasts() -> None:
    names = {
        "R-DW-24 - R-DW": "Cap: R-DW-24 - R-DW",
        "R-DW-24-PT - R-DW-24": "Pretraining lags: PT - R-DW-24",
        "A-pool-24 - R-DW-24-PT": "A-pool-24 - R-DW-24-PT",
    }
    t = (
        pl.read_csv(OUT / "contrasts.csv")
        .filter(
            (pl.col("set") == "eight")
            & pl.col("contrast").is_in(list(names))
            & pl.col("band").is_in(BANDS)
        )
        .with_columns(what=pl.col("contrast").replace_strict(names, return_dtype=pl.String))
        .to_pandas()
    )
    base = alt.Chart(t).encode(
        y=alt.Y("band:N", sort=BANDS, title="Lead time"),
        color=alt.Color(
            "what:N",
            legend=alt.Legend(orient="top", title=None, columns=1),
            scale=alt.Scale(
                domain=list(names.values()),
                range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE],
            ),
        ),
        yOffset="what:N",
    )  # ty: ignore[unresolved-attribute]
    bar = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low_month:Q",
            title="CRPS difference (points of capacity; negative favours the first arm)",
        ),
        x2="high_month:Q",
    )  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
        .mark_rule(color=theme.BLACK_2)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    alt.layer(zero, bar, dot).properties(
        width=560,
        height=330,
        title=alt.TitleParams(
            [
                "At the eight, the cap's cost excludes zero from 6 h to 120 h; the pretraining gain from 12 h to 72 h.",
                "Dot: mean. Bar: 95% interval over 5 month blocks (near min to max). Interim, unaudited.",
            ],
            anchor="start",
            fontSize=12,
        ),
    ).save(DOCS / "figure_2_contrasts_at_the_eight.svg")


figure_crps()
figure_contrasts()

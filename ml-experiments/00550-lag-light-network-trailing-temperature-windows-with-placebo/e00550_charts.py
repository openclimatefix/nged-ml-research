# ruff: noqa: E501
"""Draw experiment 550's figures.

The figures are the window contrast on cold nights by lead band, and the injected-shift leak.
"""

from pathlib import Path

import altair as alt
import e00521_stats as s  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ID = "00550-lag-light-network-trailing-temperature-windows-with-placebo"
OUT = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / ID
BANDS = list(s.BANDS)
contrasts = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / ID / "scores" / "contrasts.csv").filter(
    (pl.col("metric") == "crps")
    & (pl.col("slice") == "cold winter nights")
    & pl.col("set").is_in(["eight", "six"])
    & pl.col("band").is_in(BANDS)
    & pl.col("contrast").is_in(["RDW-T - RDW-Tnull", "RDW-T - R-DW"])
)
base = alt.Chart(contrasts.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
    x=alt.X("band:N", sort=BANDS, title="Lead time", axis=alt.Axis(labelAngle=0)),
    color=alt.Color(
        "contrast:N",
        title=None,
        scale=alt.Scale(
            domain=["RDW-T - RDW-Tnull", "RDW-T - R-DW"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
    ),
    xOffset="contrast:N",
)
fig1 = (
    (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "low_month:Q",
                title="CRPS difference (points of capacity; below zero favours the windows)",
            ),
            y2="high_month:Q",
        )
        + base.mark_point(filled=True, size=50).encode(y="difference:Q")  # ty: ignore[unresolved-attribute]
    )
    .properties(width=300, height=240)
    .facet(
        column=alt.Column("set:N", title=None, sort=["eight", "six"]),
        title="Cold winter nights, folds 2025-12, 2026-01, 2026-03. Bars: month-block interval, 3 blocks.",
    )
)
fig1.save(OUT / "figure_1_cold_night_contrast.svg")

leak = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / ID / "scores" / "stress_leak.csv").filter(
    pl.col("band") != "all"
)
arms = ["L-1d", "R-DW", "RDW-T", "RDW-Tnull"]
fig2 = (
    alt.Chart(leak.to_pandas())  # ty: ignore[unresolved-attribute]
    .mark_bar()
    .encode(
        x=alt.X(
            "band:N",
            sort=["day1", "days2_3", "days4_10"],
            title="Lead group",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("leak:Q", title="Forecast change per 100 points of injected shift"),
        color=alt.Color(
            "arm:N",
            title=None,
            scale=alt.Scale(
                domain=arms, range=[ocf.DATA_GREEN, ocf.BRAND_ORANGE, ocf.DATA_BLUE, "#8a8a8a"]
            ),
        ),
        xOffset="arm:N",
    )
    .properties(
        width=520,
        height=240,
        title="Injected-shift leak (seed 0, five folds, mean over folds); lower is safer",
    )
)
fig2.save(OUT / "figure_2_injected_shift_leak.svg")

# ruff: noqa: D103
"""Draw experiment 547's figures from the private score tables and the stored forecasts."""

from pathlib import Path

import altair as alt
import e00454_score as score454  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00435_tailcore import widen  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ID = "00547-lag-light-and-lag-free-blend-by-lead-band-stored-forecasts"
SCORES = ML_EXPERIMENTS_DATA_DIR / ID / "scores"
E539 = ML_EXPERIMENTS_DATA_DIR / "00539-l1d-plus-day-and-week-lags" / "work" / "predictions"
OUT = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / ID
BANDS = list(s.BANDS)
LEVELS = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
COLOURS = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]


def figure_gap() -> None:
    absolute = pl.read_csv(SCORES / "absolute.csv").filter(
        (pl.col("set") == "eight") & pl.col("band").is_in(BANDS) & pl.col("crps").is_not_null()
    )
    pool = absolute.filter(pl.col("arm") == "A-pool").select("band", pool="crps")
    data = (
        absolute.filter(pl.col("arm").is_in(["L-1d", "R-DW", "blend"]))
        .join(pool, on="band")
        .with_columns(gap=pl.col("crps") - pl.col("pool"))
    )
    chart = (
        alt.Chart(data.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("band:N", sort=BANDS, title="Lead time", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("gap:Q", title="CRPS minus A-pool (points of capacity; lower is better)"),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=["L-1d", "R-DW", "blend"], range=COLOURS),
            ),
        )
        .properties(
            width=620,
            height=260,
            title=[
                "Eight MVA-only Primaries, folds 2025-12 to 2026-06.",
                "Zero is A-pool; the blend and R-DW lie on top of each other at 24 h and beyond.",
            ],
        )
    )
    chart.save(OUT / "figure_1_gap_to_a_pool.svg")


def figure_weights() -> None:
    w = pl.read_csv(SCORES / "weights.csv").filter(pl.col("set") == "eight")
    chart = (
        alt.Chart(w.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("band:N", sort=BANDS, title="Lead time", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "weight:Q", title="Weight on R-DW (1 = R-DW alone)", scale=alt.Scale(domain=[0, 1])
            ),
            color=alt.Color(
                "fold:N",
                title="Scored fold",
                scale=alt.Scale(
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE]
                ),
            ),
        )
        .properties(
            width=620,
            height=240,
            title=[
                "Fitted weight on the lag-light network R-DW, eight MVA-only Primaries.",
                "Each line is one scored fold, fitted on the folds before it.",
            ],
        )
    )
    chart.save(OUT / "figure_2_weights.svg")


def figure_example() -> None:
    month, primary = "2025-12", 0
    base = score454.month_base(month=month).filter(pl.col("primary") == primary)
    origin = int(base["origin"].unique().sort()[20])
    base = base.filter(pl.col("origin") == origin).sort("target").with_columns(band=s.band_expr())
    w = pl.read_csv(SCORES / "weights.csv").filter(
        (pl.col("set") == "eight") & (pl.col("fold") == month)
    )
    wb = dict(zip(w["band"].to_list(), w["weight"].to_list(), strict=True))
    keys = ["primary", "origin", "target"]
    q = {
        a: base.select(keys).join(
            pl.read_parquet(E539 / a / f"{month}_c0.parquet"),
            on=keys,
            how="left",
            maintain_order="left",
        )
        for a in ("L-1d", "R-DW")
    }
    weight = np.array([wb[b] for b in base["band"].to_list()])[:, None]
    mix = (
        weight * q["R-DW"].select(LEVELS).to_numpy()
        + (1 - weight) * q["L-1d"].select(LEVELS).to_numpy()
    )
    blend = widen(frame=pl.DataFrame(mix, schema=LEVELS, orient="row"), factor=1.15)
    hours = base["lead_hours"].to_numpy()
    long = pl.DataFrame(
        {
            "hours": hours,
            "actual": base["actual"].to_numpy() * 100,
            "L-1d": q["L-1d"]["q50"].to_numpy() * 100,
            "R-DW": q["R-DW"]["q50"].to_numpy() * 100,
            "blend": blend["q50"].to_numpy() * 100,
            "low": blend["q05"].to_numpy() * 100,
            "high": blend["q95"].to_numpy() * 100,
        }
    )

    def panel(frame: pl.DataFrame, title: str, tick: list[int]) -> alt.LayerChart:
        band = (
            alt.Chart(frame.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.2, color=ocf.DATA_GREEN)
            .encode(
                x=alt.X(
                    "hours:Q", title="Hours after the forecast origin", axis=alt.Axis(values=tick)
                ),
                y=alt.Y("low:Q", title="Demand (% of capacity)", scale=alt.Scale(zero=False)),
                y2="high:Q",
            )
        )
        lines = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                frame.unpivot(
                    on=["actual", "L-1d", "R-DW", "blend"],
                    index="hours",
                    variable_name="series",
                    value_name="v",
                ).to_pandas()
            )
            .mark_line()
            .encode(
                x="hours:Q",
                y=alt.Y("v:Q"),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["actual", "L-1d", "R-DW", "blend"],
                        range=["#000000", ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                    ),
                ),
            )
        )
        return (band + lines).properties(width=620, height=200, title=title)

    chart = alt.vconcat(
        panel(
            long.filter(pl.col("hours") <= 36),
            "Day ahead (first 36 h): medians, and the blend's 90% band (shaded)",
            [0, 6, 12, 18, 24, 30, 36],
        ),
        panel(long, "Full horizon (10 days), same forecast", [0, 48, 96, 144, 192, 240]),
    )
    chart.save(OUT / "figure_3_example_forecast.svg")


figure_gap()
figure_weights()
figure_example()

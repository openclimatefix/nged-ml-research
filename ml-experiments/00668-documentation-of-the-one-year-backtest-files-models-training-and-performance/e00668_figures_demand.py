# ruff: noqa: E501
"""Figures of the demand backtest scores: score against lead time, by season, and by season and lead.

uv run --no-sync python ml-experiments/00668-*/e00668_figures_demand.py bsps OUT_DIR
Reads `scores_668/<name>/scores.csv` (private) and writes SVGs for the public page. A BSP, a GSP, or
a Primary is a substation, so the figures carry no generator series.
"""

import sys
from pathlib import Path

import altair as alt
import e00668_score_lib as lib
import polars as pl

COLOURS = {"Model": "#306BFF", "Manual heuristic": "#FF4901", "Climatology": "#7F7F7F"}
SEASON_COLOURS = {
    "autumn (Nov 2025)": "#FC9700",
    "winter": "#306BFF",
    "spring": "#17E58F",
    "summer": "#FF4901",
    "autumn (Sep to Oct 2026)": "#B701FF",
}
WIDTH, HEIGHT = 620, 300
UNIT_NOTE = "points of capacity"


def horizon_chart(*, scores: pl.DataFrame, label: str, model: str) -> alt.Chart:
    """Return the MAE of the median against forecast horizon, one line per system."""
    frame = (
        scores.filter((pl.col("slice") == "lead day") & (pl.col("metric") == "mae"))
        .with_columns(day=pl.col("group").cast(pl.Int32) + 0.5)
        .with_columns(system=pl.col("system").replace({"Model": model}))
        .to_pandas()
    )
    domain = [model, "Manual heuristic", "Climatology"]
    return (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "day:Q",
                title="Forecast horizon (days after the forecast origin)",
                axis=alt.Axis(values=[0, 2, 4, 6, 8, 10, 12, 14]),
                scale=alt.Scale(domain=[0, 14]),
            ),
            y=alt.Y(
                "mean:Q",
                title=f"Median-forecast MAE ({UNIT_NOTE}; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "system:N",
                title=None,
                scale=alt.Scale(
                    domain=domain,
                    range=[COLOURS["Model"], COLOURS["Manual heuristic"], COLOURS["Climatology"]],
                ),
                legend=alt.Legend(orient="top", columns=3),
            ),
        )
        .properties(
            width=WIDTH,
            height=HEIGHT,
            title=alt.TitleParams(
                [
                    f"Normalised MAE of the median forecast by horizon at the {label}",
                    "One-year backtest, origins 2025-11-01 to 2026-10-31 at 12:00 UTC; daily bins.",
                ],
                anchor="start",
                fontSize=13,
            ),
        )
    )


def season_chart(*, scores: pl.DataFrame, label: str, model: str) -> alt.Chart:
    """Return the MAE of the median by season for the three systems, with 95% intervals."""
    order = list(SEASON_COLOURS)
    frame = (
        scores.filter((pl.col("slice") == "season") & (pl.col("metric") == "mae"))
        .with_columns(system=pl.col("system").replace({"Model": model}))
        .to_pandas()
    )
    base = alt.Chart(frame).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("group:N", title=None, sort=order, axis=alt.Axis(labelAngle=0, labelLimit=140)),
        color=alt.Color(
            "system:N",
            title=None,
            scale=alt.Scale(
                domain=[model, "Manual heuristic", "Climatology"],
                range=[COLOURS["Model"], COLOURS["Manual heuristic"], COLOURS["Climatology"]],
            ),
            legend=alt.Legend(orient="top", columns=3),
        ),
        xOffset=alt.XOffset("system:N"),
    )
    points = base.mark_point(filled=True, size=70).encode(
        y=alt.Y(
            "mean:Q",
            title=f"Median-forecast MAE ({UNIT_NOTE}; smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule(strokeWidth=2).encode(y="lo:Q", y2="hi:Q")
    return (points + bars).properties(
        width=WIDTH,
        height=HEIGHT,
        title=alt.TitleParams(
            [
                f"Normalised MAE of the median forecast by meteorological season at the {label}",
                "Season of the forecast origin. Dot: mean. Line: 95% interval from resampling origin days.",
            ],
            anchor="start",
            fontSize=13,
        ),
    )


def season_lead_chart(*, scores: pl.DataFrame, label: str) -> alt.VConcatChart:
    """Return two panels by lead band: the model's MAE and its CRPS gain over the heuristic."""
    bands = list(lib.LEAD_BANDS)
    frame = scores.filter(pl.col("slice") == "season and lead band").with_columns(
        season=pl.col("group").str.split(" | ").list.get(0),
        band=pl.col("group").str.split(" | ").list.get(1),
    )
    season_scale = alt.Scale(domain=list(SEASON_COLOURS), range=list(SEASON_COLOURS.values()))
    colour = alt.Color(
        "season:N", title=None, scale=season_scale, legend=alt.Legend(orient="top", columns=3)
    )
    x = alt.X("band:N", title="Forecast horizon", sort=bands, axis=alt.Axis(labelAngle=0))
    mae = (
        alt.Chart(
            frame.filter((pl.col("system") == "Model") & (pl.col("metric") == "mae")).to_pandas()  # ty: ignore[unresolved-attribute]
        )
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("mean:Q", title=f"Model MAE ({UNIT_NOTE})", scale=alt.Scale(zero=False)),
            color=colour,
        )
        .properties(width=WIDTH, height=HEIGHT)
    )
    gain = (
        alt.Chart(
            frame.filter(
                (pl.col("system") == "Manual heuristic") & (pl.col("metric") == "crps")
            ).to_pandas()  # ty: ignore[unresolved-attribute]
        )
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y(
                "gain_pct:Q",
                title="CRPS gain over the manual heuristic (%; larger is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("season:N", legend=None, scale=season_scale),
        )
        .properties(width=WIDTH, height=HEIGHT)
    )
    return alt.vconcat(mae, gain).properties(
        title=alt.TitleParams(
            [
                f"Model error and gain over the manual heuristic by season and horizon, {label}",
                "Season of the forecast origin. Gain: how far the model's CRPS is below the heuristic's, in percent.",
            ],
            anchor="start",
            fontSize=13,
        )
    )


def main(name: str, out: Path, label: str, model: str) -> None:
    """Draw the three figures for one file."""
    scores = pl.read_csv(lib.FOLDER / "scores_668" / name / "scores.csv")
    out.mkdir(parents=True, exist_ok=True)
    horizon_chart(scores=scores, label=label, model=model).save(out / f"{name}_mae_by_horizon.svg")
    season_chart(scores=scores, label=label, model=model).save(out / f"{name}_mae_by_season.svg")
    season_lead_chart(scores=scores, label=label).save(out / f"{name}_season_by_horizon.svg")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4])

# ruff: noqa: E501
"""Draw experiment 611's figures from the stored scores and the Parquet export (no model is fitted).

Usage: python e00611_figures.py
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, cast

import altair as alt
import e00611_pool as pool
import polars as pl
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00611-bsp-gsp-pilot-backtests-for-the-bake-off"
)
SCORES: Final[Path] = pool.WORK / "scores"
EXPORT: Final[Path] = pool.WORK / "export"
WIDTH: Final = 620
NAME: Final[dict[str, str]] = {
    "design1": "Design 1 (lag-free)",
    "design2": "Design 2 (lag network)",
    "bg2": "BG2 (weekly XGBoost)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
ORDER: Final[list[str]] = list(NAME.values())
COLOUR: Final[list[str]] = [
    theme.DATA_BLUE,
    theme.DATA_SKY,
    theme.BRAND_ORANGE,
    theme.DATA_PURPLE,
    theme.DATA_GREEN,
]
SERIES: Final[dict[str, str]] = {
    "D13": "D13 (bulk supply point, Skegness)",
    "D17": "D17 (bulk supply point, Boston)",
    "D6": "D6 (grid supply point, Walpole)",
    "D1": "D1 (grid supply point, Bicker Fen; mostly exports power)",
}
BANDS: Final[list[str]] = ["0-24h", "24-120h", "120-240h"]


def colour_scale() -> alt.Scale:
    """Return the arm colours, in the order of `ORDER`."""
    return alt.Scale(domain=ORDER, range=COLOUR)


def metric_figure(
    *,
    rows: pl.DataFrame,
    value: str,
    title: str,
    y_title: str,
    file: str,
    rule: float | None = None,
) -> None:
    """Draw a per-arm dot for each lead band with the minimum-to-maximum range over the 5 months; one panel per group."""
    monthly = (
        rows.group_by("arm", "group", "band", "month")
        .agg(pl.col(value).mean().alias("v"))
        .group_by("arm", "group", "band")
        .agg(
            pl.col("v").mean().alias("mean"),
            pl.col("v").min().alias("lo"),
            pl.col("v").max().alias("hi"),
            pl.col("month").n_unique().alias("months"),
        )
        .with_columns(pl.col("arm").replace(NAME).alias("Arm"))
    )
    panels = []
    for group, label in (
        ("BSP", "Bulk supply points (2 series)"),
        ("GSP", "Grid supply points (2 series)"),
    ):
        sub = monthly.filter(pl.col("group") == group)
        base = alt.Chart(sub).encode(
            y=alt.Y("band:N", sort=BANDS, title="Lead band"),
            yOffset=alt.YOffset("Arm:N", sort=ORDER),
            color=alt.Color(
                "Arm:N",
                scale=colour_scale(),
                sort=ORDER,
                legend=alt.Legend(title="Forecast", orient="top", columns=2),
            ),
        )
        layers = [
            base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lo:Q", title=y_title, scale=alt.Scale(zero=False)), x2="hi:Q"
            ),
            base.mark_point(filled=True, size=60, opacity=1).encode(x="mean:Q"),  # ty: ignore[unresolved-attribute]
        ]
        if rule is not None:
            layers.append(
                alt.Chart(pl.DataFrame({"x": [rule]}))  # ty: ignore[unresolved-attribute]
                .mark_rule(strokeDash=[4, 4], color="#000000", opacity=0.6)
                .encode(x="x:Q")
            )
        panels.append(alt.layer(*layers).properties(width=WIDTH, height=280, title=label))
    alt.vconcat(*panels).resolve_scale(color="shared").properties(
        title=alt.TitleParams(
            title,
            subtitle=[
                "Dot: mean of the 5 monthly values. Line: minimum to maximum monthly value (5 development months)."
            ],
        )
    ).save(DOCS / file)


def window_frame(*, design: str, series: str, start: datetime, days: int) -> pl.DataFrame:
    """Return the day-ahead forecast (issued 09:00 UTC the day before) and the observed power for a window."""
    id_ = {"D1": 28, "D6": 27, "D13": 2, "D17": 1}[series]
    parts = [pl.read_parquet(f) for f in sorted(EXPORT.glob(f"{design}_*.parquet"))]
    frame = (
        pl.concat(parts)
        .filter(
            pl.col("time_series_id") == id_, pl.col("lead_hours") >= 15, pl.col("lead_hours") < 39
        )
        .filter(
            (pl.col("valid_time") >= start) & (pl.col("valid_time") < start + timedelta(days=days))
        )
        .sort("valid_time")
    )
    return frame.with_columns(pl.col("valid_time").dt.replace_time_zone(None))


def pick_windows(*, series: str) -> dict[str, object]:
    """Return the first day of a quiet 14-day window (smallest half-hour-to-half-hour change in June 2026) and of a cold-night window.

    The cold-night window is the 14 days centred on the night (18:00 to 06:00 UTC) of the highest observed
    power in 2025-12 and 2026-01.
    """
    id_ = {"D1": 28, "D6": 27, "D13": 2, "D17": 1}[series]
    obs = (
        pl.concat([pl.read_parquet(f) for f in sorted(EXPORT.glob("design1_*.parquet"))])
        .filter(pl.col("time_series_id") == id_)
        .select("valid_time", "observed_power")
        .unique()
        .sort("valid_time")
        .drop_nulls()
    )
    june = obs.filter(pl.col("valid_time") >= pl.datetime(2026, 6, 1, time_zone="UTC")).filter(
        pl.col("valid_time") < pl.datetime(2026, 6, 28, time_zone="UTC")
    )
    best, best_start = float("inf"), None
    for d in range(13):
        s = cast(datetime, june["valid_time"].min()) + timedelta(days=d)
        w = june.filter(
            (pl.col("valid_time") >= s) & (pl.col("valid_time") < s + timedelta(days=14))
        )
        spread = float(cast(float, w["observed_power"].diff().abs().mean()))
        if spread < best:
            best, best_start = spread, s
    win = obs.filter(pl.col("valid_time").dt.month().is_in([12, 1])).filter(
        (pl.col("valid_time").dt.hour() >= 18) | (pl.col("valid_time").dt.hour() < 6)
    )
    peak = win.sort("observed_power", descending=True)["valid_time"][0]
    cold = (peak - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    return {"quiet": best_start, "cold": cold, "peak": peak}


def series_figure(*, design: str, series: str, file: str) -> dict[str, object]:
    """Draw two panels (a quiet fortnight, a fortnight around the coldest-night peak) of one design's day-ahead forecast."""
    wins = pick_windows(series=series)
    panels = []
    for key, label in (
        ("quiet", "A smooth fortnight (the least jagged 14 days of June 2026)"),
        ("cold", "A fortnight around the night of highest power"),
    ):
        start = cast(datetime, wins[key])
        fr = window_frame(design=design, series=series, start=start, days=14)
        long = pl.concat(
            [
                fr.select("valid_time", pl.col("observed_power").alias("v")).with_columns(
                    kind=pl.lit("Observed power")
                ),
                fr.select("valid_time", pl.col("q50").alias("v")).with_columns(
                    kind=pl.lit("Median forecast")
                ),
            ]
        )
        kinds = ["Observed power", "Median forecast", "q05 to q95 band"]
        colour = alt.Scale(domain=kinds, range=["#000000", theme.DATA_BLUE, theme.DATA_BLUE])
        legend = alt.Legend(title=None, orient="top", symbolType="square", columns=3)
        band = (
            alt.Chart(fr)  # ty: ignore[unresolved-attribute]
            .transform_calculate(kind="'q05 to q95 band'")
            .mark_area(opacity=0.3)
            .encode(
                x=alt.X("valid_time:T", title="Time (UTC)", axis=alt.Axis(format="%-d %b")),
                y=alt.Y("q05:Q", title="Power (MW)"),
                y2="q95:Q",
                color=alt.Color("kind:N", scale=colour, legend=legend),
            )
        )
        lines = (
            alt.Chart(long)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.3)
            .encode(
                x="valid_time:T", y="v:Q", color=alt.Color("kind:N", scale=colour, legend=legend)
            )
        )
        panels.append(
            alt.layer(band, lines).properties(
                width=WIDTH, height=290, title=f"{label}, from {start:%-d %b %Y}"
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            SERIES[series],
            subtitle=[
                f"{NAME[design]}: forecast issued 09:00 UTC the day before (15 to 39 h ahead)."
            ],
        )
    )
    chart.save(DOCS / file)
    return {k: str(v) for k, v in wins.items()}


def main() -> None:
    """Draw every figure."""
    rows = pl.read_parquet(SCORES / "rows.parquet").with_columns(
        covered=pl.col("covered").cast(pl.Float64)
    )
    metric_figure(
        rows=rows,
        value="crps_points",
        title="Continuous ranked probability score at the bulk and grid supply points",
        y_title="CRPS (points of effective capacity; smaller is better)",
        file="figure_1_crps_by_lead_band.svg",
    )
    metric_figure(
        rows=rows,
        value="covered",
        title="Share of rows with the observed power inside the q05 to q95 band",
        y_title="Coverage of the q05 to q95 band (target 0.90)",
        file="figure_2_coverage_by_lead_band.svg",
        rule=0.9,
    )
    metric_figure(
        rows=rows,
        value="width_points",
        title="Width of the q05 to q95 band",
        y_title="Mean band width (points of effective capacity; narrower is sharper)",
        file="figure_3_width_by_lead_band.svg",
    )
    chosen = {}
    for i, s in enumerate(("D13", "D17", "D6", "D1"), start=4):
        chosen[s] = series_figure(design="design1", series=s, file=f"figure_{i}_day_ahead_{s}.svg")
    (SCORES / "windows.json").write_text(__import__("json").dumps(chosen, indent=1))
    print(chosen)


if __name__ == "__main__":
    main()

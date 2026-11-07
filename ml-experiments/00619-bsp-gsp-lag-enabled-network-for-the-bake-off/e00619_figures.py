# ruff: noqa: E501
"""Draw experiment 619's figures from the stored scores and the Parquet export (no model is fitted).

Usage: python e00619_figures.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00611_figures as f611  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00619-bsp-gsp-lag-enabled-network-for-the-bake-off"
W611: Final[Path] = f611.EXPORT.parent
SCORES: Final[Path] = ROOT / "scores"
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00619-bsp-gsp-lag-enabled-network-for-the-bake-off"
)
WIDTH: Final = 620
NAME: Final[dict[str, str]] = {
    "lag_enabled": "Lag-enabled network",
    "design1": "Lag-free network (611 Design 1)",
    "bg2": "BG2 (weekly XGBoost)",
    "climatology": "Climatology",
}
ORDER: Final[list[str]] = list(NAME.values())
COLOUR: Final[list[str]] = [theme.DATA_BLUE, theme.DATA_SKY, theme.BRAND_ORANGE, theme.DATA_GREEN]
GROUPS: Final[dict[str, str]] = {
    "BSP": "Bulk supply points (2 series)",
    "GSP": "Grid supply points (2 series)",
}
IDS: Final[dict[str, int]] = {"D1": 28, "D6": 27, "D13": 2, "D17": 1}
SERIES: Final[dict[str, str]] = f611.SERIES


def day_figure(
    *,
    table: pl.DataFrame,
    value: str,
    title: str,
    y_title: str,
    file: str,
    rule: float | None = None,
) -> None:
    """Draw one line per forecast against the day of lead time, one panel per group."""
    panels = []
    for group, label in GROUPS.items():
        sub = table.filter(pl.col("group") == group).with_columns(
            Forecast=pl.col("arm").replace(NAME)
        )
        base = alt.Chart(sub).encode(
            x=alt.X("day:O", title="Day of lead time (day 1 is 0 to 24 h ahead)"),
            y=alt.Y(f"{value}:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Forecast:N", scale=alt.Scale(domain=ORDER, range=COLOUR), sort=ORDER,
                legend=alt.Legend(title="Forecast", orient="top", columns=2),
            ),
        )  # fmt: skip
        layers = [
            base.mark_line(point=True, strokeWidth=2),  # ty: ignore[unresolved-attribute]
        ]
        if rule is not None:
            layers.append(
                alt.Chart(pl.DataFrame({"y": [rule]}))
                .mark_rule(strokeDash=[4, 4], color="#000000")
                .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
            )
        panels.append(alt.layer(*layers).properties(width=WIDTH, height=280, title=label))
    alt.vconcat(*panels).resolve_scale(color="shared").properties(
        title=alt.TitleParams(
            title,
            subtitle=[
                "Each point pools every scored row of that day over the 5 development months."
            ],
        )
    ).save(DOCS / file)


def difference_figure(
    *, daily: pl.DataFrame, metric: str, title: str, y_title: str, file: str
) -> None:
    """Draw lag-enabled minus lag-free by day with the unadjusted 95% month-block interval."""
    panels = []
    for group, label in GROUPS.items():
        sub = daily.filter((pl.col("group") == group) & (pl.col("metric") == metric))
        base = alt.Chart(sub).encode(x=alt.X("day:O", title="Day of lead time"))
        layers = [
            base.mark_rule(strokeWidth=2, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("lower:Q", title=y_title), y2="upper:Q"
            ),
            base.mark_point(filled=True, size=70, color=theme.DATA_BLUE, opacity=1).encode(  # ty: ignore[unresolved-attribute]
                y="difference:Q"
            ),
            alt.Chart(pl.DataFrame({"y": [0.0]}))
            .mark_rule(strokeDash=[4, 4], color="#000000")
            .encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
        ]
        panels.append(alt.layer(*layers).properties(width=WIDTH, height=280, title=label))
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            title,
            subtitle=[
                "Dot: mean over the 5 months. Line: 95% interval from resampling the 5 months (not adjusted for the number of days)."
            ],
        )
    ).save(DOCS / file)


def window_frame(
    *,
    arm: str,
    series: str,
    start: datetime,
    days: int,
    lo: float,
    hi: float,
    origin: datetime | None = None,
) -> pl.DataFrame:
    """Return an arm's forecast rows with a lead in [lo, hi) hours and the observed power for a window."""
    folder = ROOT / "export" if arm == "lag_enabled" else W611 / "export"
    prefix = "lag_enabled" if arm == "lag_enabled" else "design1"
    parts = [pl.read_parquet(p) for p in sorted(folder.glob(f"{prefix}_*.parquet"))]
    frame = (
        pl.concat(parts)
        .filter(
            pl.col("time_series_id") == IDS[series],
            pl.col("lead_hours") >= lo,
            pl.col("lead_hours") < hi,
        )
        .filter(
            (pl.col("valid_time") >= start) & (pl.col("valid_time") < start + timedelta(days=days))
        )
        .sort("valid_time")
    )
    if origin is not None:
        frame = frame.filter(
            pl.col("power_fcst_init_time") == origin.replace(tzinfo=__import__("datetime").UTC)
        )
    return frame.with_columns(pl.col("valid_time").dt.replace_time_zone(None))


def two_panel(
    *,
    series: str,
    start: datetime,
    days: int,
    lo: float,
    hi: float,
    origin: datetime | None = None,
    title: str,
    subtitle: str,
    file: str,
) -> None:
    """Draw the lag-free forecast and the lag-enabled forecast in two panels over one window."""
    kinds = ["Observed power", "Median forecast", "q05 to q95 band"]
    colour = alt.Scale(domain=kinds, range=["#000000", theme.DATA_BLUE, theme.DATA_SKY])
    legend = alt.Legend(title=None, orient="top", symbolType="square", columns=3)
    panels = []
    for arm, label in (("design1", "Lag-free network"), ("lag_enabled", "Lag-enabled network")):
        fr = window_frame(
            arm=arm, series=series, start=start, days=days, lo=lo, hi=hi, origin=origin
        )
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
        band = (
            alt.Chart(fr)  # ty: ignore[unresolved-attribute]
            .transform_calculate(kind="'q05 to q95 band'")
            .mark_area(opacity=0.45)
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
        panels.append(alt.layer(band, lines).properties(width=WIDTH, height=290, title=label))
    alt.vconcat(*panels).properties(title=alt.TitleParams(title, subtitle=[subtitle])).save(
        DOCS / file
    )


def main() -> None:
    """Draw every figure."""
    by_day = pl.read_csv(SCORES / "by_day.csv").with_columns(pl.col("day").cast(pl.Int32))
    daily = pl.read_csv(SCORES / "contrasts_by_day.csv")
    difference_figure(
        daily=daily,
        metric="crps_points",
        title="CRPS of the lag-enabled network minus the lag-free network",
        y_title="CRPS difference (points; below 0 favours lags)",
        file="figure_1_crps_difference_by_day.svg",
    )
    day_figure(
        table=by_day,
        value="crps_points",
        title="CRPS by day of lead time",
        y_title="CRPS (points of effective capacity; smaller is better)",
        file="figure_2_crps_by_day.svg",
    )
    day_figure(
        table=by_day,
        value="mae_points",
        title="Mean absolute error of the median by day of lead time",
        y_title="MAE of q50 (points of effective capacity; smaller is better)",
        file="figure_3_mae_by_day.svg",
    )
    day_figure(
        table=by_day,
        value="coverage_q05_q95",
        title="Share of rows inside the q05 to q95 band",
        y_title="Coverage (target 0.90)",
        file="figure_4_coverage_by_day.svg",
        rule=0.9,
    )
    wins = json.loads((W611 / "scores" / "windows.json").read_text())
    chosen = {}
    for i, s in enumerate(("D13", "D17", "D6", "D1"), start=5):
        start = datetime.fromisoformat(wins[s]["cold"])
        two_panel(
            series=s,
            start=start,
            days=14,
            lo=15,
            hi=39,
            title=SERIES[s],
            subtitle="Day-ahead forecast issued 09:00 UTC the day before (15 to 39 h ahead), fortnight around the highest-power night of December 2025 or January 2026.",
            file=f"figure_{i}_day_ahead_{s}.svg",
        )
        chosen[s] = str(start)
    for i, s in enumerate(("D17", "D6"), start=9):
        origin = datetime.fromisoformat(wins[s]["cold"]) + timedelta(days=5, hours=9)
        two_panel(
            series=s,
            start=origin,
            days=10,
            lo=0,
            hi=240,
            origin=origin,
            title=SERIES[s],
            subtitle=f"One forecast run, issued {origin:%-d %b %Y} 09:00 UTC, over its full 10 days.",
            file=f"figure_{i}_one_run_{s}.svg",
        )
    (SCORES / "windows.json").write_text(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()

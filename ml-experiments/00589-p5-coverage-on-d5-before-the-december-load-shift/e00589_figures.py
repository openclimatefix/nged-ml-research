# ruff: noqa: E501, D103, ANN201, DTZ001
"""Experiment 589: figures from the CSV tables written by `e00589_tables.py` (run that first).

Usage: DATA_PATH_INTERNAL=... python e00589_figures.py
"""

from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00589-p5-coverage-on-d5-before-the-december-load-shift"
T = ML_EXPERIMENTS_DATA_DIR / NAME / "tables"
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
COLOUR = {"C0": "#6b6b6b", "P5": theme.BRAND_ORANGE, "C1": theme.DATA_BLUE}
ONSET_DAY = 2311  # 2025-12-29, the verified onset at D5
GRID_DAY0 = "2019-09-01"


def day_date(col: str = "day") -> pl.Expr:
    return (pl.lit(GRID_DAY0).str.to_date() + pl.duration(days=pl.col(col))).cast(pl.Datetime)


def d5_daily() -> None:
    d = pl.read_csv(T / "d5_daily.csv").filter(
        pl.col("day").is_between(ONSET_DAY - 40, ONSET_DAY + 20)
    )
    d = d.with_columns(date=day_date()).with_columns(
        (pl.col(c) * 100) for c in d.columns if c != "day"
    )
    base = alt.Chart(d.to_pandas() if False else d).encode(
        x=alt.X("date:T", title="Valid day (2025-26)")
    )
    layers = []
    for arm in ("C0", "P5"):
        layers.append(
            base.mark_area(opacity=0.18, color=COLOUR[arm]).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y(
                    f"{arm}_q05:Q",
                    title="Power, % of capacity (daily mean, forecasts issued 24-48 h ahead)",
                ),
                y2=f"{arm}_q95:Q",
            )
        )
        layers.append(base.mark_line(color=COLOUR[arm], strokeWidth=1.5).encode(y=f"{arm}_q50:Q"))  # ty: ignore[unresolved-attribute]
    layers.append(base.mark_line(color="black", strokeWidth=2).encode(y="obs:Q"))  # ty: ignore[unresolved-attribute]
    rule = (
        alt.Chart(pl.DataFrame({"date": [day_date_py(ONSET_DAY)]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="black")
        .encode(x="date:T")
    )
    (
        alt.layer(*layers, rule).properties(
            width=620,
            height=260,
            title=alt.TitleParams(
                [
                    "D5: observed power (black), median and 90% band of C0 (grey) and P5 (orange)",
                    "Dashed line: verified load-shift onset, 2025-12-29",
                ],
                fontSize=11,
                anchor="start",
            ),
        )
    ).save(str(DOCS / "d5_daily_forecast.svg"))


def day_date_py(day: int):
    import datetime as dt

    return dt.datetime(2019, 9, 1) + dt.timedelta(days=day)


def by_primary() -> None:
    d = pl.read_csv(T / "event_free_by_primary.csv")
    chart = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=70)
        .encode(
            y=alt.Y("label:N", sort=alt.EncodingSortField("cover90", op="min"), title="Primary"),
            x=alt.X(
                "cover90:Q",
                title="Share of observed power inside q05-q95, event-free rows, 24-120 h",
                scale=alt.Scale(domain=[0.78, 0.96]),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(COLOUR), range=list(COLOUR.values())),
                legend=alt.Legend(title="Arm", orient="top"),
            ),
        )
    )
    nominal = alt.Chart(pl.DataFrame({"x": [0.9]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    (chart + nominal).properties(
        width=480,
        height=300,
        title=alt.TitleParams(
            ["Coverage on rows with no load-shift nearby, by Primary", "Dashed line: nominal 0.90"],
            fontSize=11,
            anchor="start",
        ),
    ).save(str(DOCS / "event_free_coverage_by_primary.svg"))


def december_bias() -> None:
    d = pl.read_csv(T / "december_before_d5_onset_by_primary.csv")
    chart = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=70)
        .encode(
            y=alt.Y(
                "label:N",
                sort=alt.EncodingSortField("median_minus_y_pct", op="min"),
                title="Primary",
            ),
            x=alt.X(
                "median_minus_y_pct:Q",
                title="Median forecast minus observed power, % of capacity (negative = forecast too low)",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(COLOUR), range=list(COLOUR.values())),
                legend=alt.Legend(title="Arm", orient="top"),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule().encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    (chart + zero).properties(
        width=480,
        height=300,
        title=alt.TitleParams(
            [
                "December 2025 fold, valid times before 2025-12-29, 24-120 h",
                "D5 sits lowest in all three arms",
            ],
            fontSize=11,
            anchor="start",
        ),
    ).save(str(DOCS / "december_median_bias_by_primary.svg"))


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    d5_daily()
    by_primary()
    december_bias()

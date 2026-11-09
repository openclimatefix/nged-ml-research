# ruff: noqa: D103, E501
# ruff: noqa: E501
"""Figures of experiment 659, drawn from the assembled tables and score CSVs only (no refits).

uv run --no-sync python ml-experiments/00659-*/e00659_figures.py OUTDIR
"""

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import altair as alt
import e00659_common as common
import polars as pl

SCORES = common.DATA_DIR / "scores"
WIDTH = 640
COLOUR = {
    "S": "#306BFF",
    "PLP": "#FF4901",
    "PL": "#10C5F7",
    "S_cal": "#B701FF",
    "PLP_cal": "#17E58F",
    "default": "#000000",
    "climatology": "#9A9A9A",
    "GAM": "#8FB0FF",
}
LABEL = {
    "wind_S": "S (raw weather)",
    "wind_PLP": "PLP (raw weather + physics-lite)",
    "wind_PL": "PL (physics-lite only)",
    "wind_S_cal": "S + calendar",
    "wind_PLP_cal": "PLP + calendar",
    "wind_default2_old_capacity": "wind default 2",
    "wind_climatology": "climatology",
    "solar_S": "S (raw weather)",
    "solar_PLP": "PLP (raw weather + physics-lite)",
    "solar_PL": "PL (physics-lite only)",
    "solar_a2_member_quarterly": "a2_member",
    "solar_gam": "GAM",
    "solar_climatology": "climatology",
}
COL_OF = {
    "wind_S": "S",
    "wind_PLP": "PLP",
    "wind_PL": "PL",
    "wind_S_cal": "S_cal",
    "wind_PLP_cal": "PLP_cal",
    "wind_default2_old_capacity": "default",
    "wind_climatology": "climatology",
    "solar_S": "S",
    "solar_PLP": "PLP",
    "solar_PL": "PL",
    "solar_a2_member_quarterly": "default",
    "solar_gam": "GAM",
    "solar_climatology": "climatology",
}


def save(chart: alt.TopLevelMixin, out: Path, name: str) -> None:
    chart.save(out / f"{name}.svg")


def score_chart(
    *, csv: str, names: list[str], title: list[str], subtitle: list[str], first_band: str = "0-24 h"
) -> alt.TopLevelMixin:
    frame = pl.read_csv(SCORES / csv).filter(
        pl.col("candidate").is_in(names) & (pl.col("band") != "all (0-240 h)")
    )
    frame = frame.with_columns(
        arm=pl.col("candidate").replace_strict(LABEL),
        colour=pl.col("candidate").replace_strict(COL_OF),
    )
    order = [LABEL[n] for n in names]
    colours = [COLOUR[COL_OF[n]] for n in names]
    base = alt.Chart(frame.to_pandas()).encode(
        y=alt.Y(
            "band:N",
            title=None,
            sort=["0-24 h", "24-120 h", "120-240 h"],
            axis=alt.Axis(labelExpr=f"datum.label == '0-24 h' ? '{first_band}' : datum.label"),
        ),
        color=alt.Color(
            "arm:N",
            sort=order,
            scale=alt.Scale(domain=order, range=colours),
            legend=alt.Legend(title=None, orient="top", columns=2),
        ),
        yOffset=alt.YOffset("arm:N", sort=order),
    )
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("score:Q", title="Quantile score (points of capacity; smaller is better)")
    )
    lines = base.mark_rule(color="black").encode(x="score_lo:Q", x2="score_hi:Q")  # ty: ignore[unresolved-attribute]
    return (bars + lines).properties(
        width=WIDTH, height=300, title=alt.TitleParams(title, subtitle=subtitle, anchor="start")
    )


def contrast_chart() -> alt.TopLevelMixin:
    wind = pl.read_csv(SCORES / "wind_cleaned_zeros_vs_default2.csv").with_columns(
        tech=pl.lit("Wind: arm minus wind default 2")
    )
    solar = pl.read_csv(SCORES / "solar_vs_a2_member.csv").with_columns(
        tech=pl.lit("Solar: arm minus a2_member")
    )
    frame = (
        pl.concat([wind, solar])
        .filter(
            (pl.col("band") == "all (0-240 h)")
            & (pl.col("metric") == "score")
            & pl.col("arm").is_in(list(LABEL))
            & ~pl.col("arm").is_in(["wind_climatology", "solar_climatology"])
        )
        .with_columns(
            arm_label=pl.col("arm").replace_strict(LABEL),
            colour=pl.col("arm").replace_strict(COL_OF),
        )
    )
    panels = []
    for tech in frame["tech"].unique(maintain_order=True):
        part = frame.filter(pl.col("tech") == tech).to_pandas()
        base = alt.Chart(part).encode(y=alt.Y("arm_label:N", title=None, sort=None))
        rule = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="black")
            .encode(
                x=alt.X(
                    "x:Q",
                    scale=alt.Scale(domain=[-0.25, 0.25]),
                    axis=alt.Axis(values=[-0.2, -0.1, 0, 0.1, 0.2]),
                )
            )
        )
        dots = base.mark_point(filled=True, size=70, color="#FF4901").encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Quantile score difference (points of capacity; negative: the arm scores better)",
                scale=alt.Scale(domain=[-0.25, 0.25]),
                axis=alt.Axis(values=[-0.2, -0.1, 0, 0.1, 0.2]),
            )
        )
        lines = base.mark_rule(color="#FF4901").encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo99:Q", scale=alt.Scale(domain=[-0.25, 0.25])), x2="hi99:Q"
        )
        panels.append((rule + lines + dots).properties(width=WIDTH, height=140, title=tech))
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "No arm scores detectably worse than its current default",
            subtitle=[
                "Dot: estimate over 0 to 240 h. Line: 99% interval from resampling the 5 forecast-origin months.",
                "Dashed: same as the default.",
            ],
            anchor="start",
        )
    )


def stretch(*, name: str, series: str, month: str, low: float, days: int = 14) -> pl.DataFrame:
    frame = (
        pl.read_parquet(common.TABLE_DIR / f"{name}.parquet")
        if (common.TABLE_DIR / f"{name}.parquet").exists()
        else pl.read_parquet(
            common.TABLE_DIR.parent.parent
            / "00615-bakeoff-wind-and-solar-development-month-backtests"
            / "tables"
            / f"{name}.parquet"
        )
    )
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    part = frame.filter(
        (pl.col("series") == series)
        & (pl.col("lead_hours") >= low)
        & (pl.col("lead_hours") < low + 24)
        & (pl.col("valid_time") >= start)
        & (pl.col("valid_time") < start + timedelta(days=days))
    ).sort("valid_time")
    return part.with_columns(day=(pl.col("valid_time") - start).dt.total_seconds() / 86400 + 1.0)


def example(
    *,
    arms: list[tuple[str, str]],
    series: str,
    month: str,
    low: float,
    title: str,
    subtitle: list[str],
) -> alt.TopLevelMixin:
    panels = []
    for name, label in arms:
        part = stretch(name=name, series=series, month=month, low=low)
        pct = (
            part.with_columns(
                *[(pl.col(c) * 100) for c in ("q05", "q25", "q50", "q75", "q95", "actual")]
            )
            .select("day", "q05", "q25", "q50", "q75", "q95", "actual")
            .to_pandas()
        )
        x = alt.X(
            "day:Q",
            title="Day of the 14-day stretch",
            scale=alt.Scale(domain=[1, 15]),
            axis=alt.Axis(values=[1, 3, 5, 7, 9, 11, 13, 15]),
        )
        b90 = (
            alt.Chart(pct)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.25, color="#FF4901", clip=True)
            .encode(x=x, y=alt.Y("q05:Q", title="% of capacity"), y2="q95:Q")
        )
        b50 = (
            alt.Chart(pct)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.35, color="#FF4901", clip=True)
            .encode(x=x, y="q25:Q", y2="q75:Q")
        )
        med = (
            alt.Chart(pct).mark_line(color="#FF4901", clip=True, aria=False).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
        )
        obs = (
            alt.Chart(pct).mark_line(color="black", clip=True, aria=False).encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append((b90 + b50 + med + obs).properties(width=WIDTH, height=190, title=label))
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(title, subtitle=subtitle, anchor="start")
    )


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    wind = ["wind_S", "wind_PLP", "wind_PL", "wind_default2_old_capacity", "wind_climatology"]
    solar = [
        "solar_S",
        "solar_PLP",
        "solar_PL",
        "solar_a2_member_quarterly",
        "solar_gam",
        "solar_climatology",
    ]
    save(
        score_chart(
            csv="wind_cleaned_zeros_overall.csv",
            names=wind,
            title=["Wind: the simple arms score as well as wind default 2 at every lead"],
            subtitle=[
                "Bar: quantile score with idle readings set to zero, 3 wind farms, 5 development months.",
                "Line: 99% interval from resampling the 5 months.",
            ],
        ),
        out,
        "wind_score_by_lead_band",
    )
    save(
        score_chart(
            csv="solar_overall.csv",
            names=solar,
            title=["Solar: the simple arms score as well as a2_member at every lead"],
            subtitle=[
                "Bar: quantile score, 6 solar farms, daylight half-hours, 5 development months.",
                "Line: 99% interval from resampling the 5 months. Leads run from 12 h.",
            ],
            first_band="12-24 h",
        ),
        out,
        "solar_score_by_lead_band",
    )
    save(contrast_chart(), out, "contrasts")
    sub = [
        "Black: observed power. Orange line: median forecast. Bands: 50% and 90% intervals.",
        "Forecasts issued 24 to 48 h ahead.",
    ]
    save(
        example(
            arms=[
                ("wind_PLP", "Wind farm W1, PLP arm"),
                ("wind_default2_old_capacity", "Wind farm W1, wind default 2"),
            ],
            series="W1",
            month="2025-12",
            low=24.0,
            title="Wind farm W1, 14 days of December 2025",
            subtitle=sub,
        ),
        out,
        "wind_W1_day_ahead",
    )
    save(
        example(
            arms=[
                ("solar_PLP", "Solar A, PLP arm"),
                ("solar_a2_member_quarterly", "Solar A, a2_member"),
            ],
            series="Solar A",
            month="2026-06",
            low=12.0,
            title="Solar farm A, 14 days of June 2026",
            subtitle=[
                "Black: observed power. Orange line: median forecast. Bands: 50% and 90% intervals.",
                "Day-ahead forecasts (12 to 36 h ahead), daylight half-hours only; lines break at night.",
            ],
        ),
        out,
        "solar_A_day_ahead",
    )


if __name__ == "__main__":
    main()

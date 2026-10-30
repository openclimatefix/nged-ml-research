# ruff: noqa: E501, PERF401
"""Charts for experiment 413: scores per fold, contrasts, learned curves, and example forecasts.

Reads `work/score_table.parquet` (written by `e00413_score.py`), the saved interpretation tables of
D2 and the stored forecasts, and writes SVG charts to the experiment's docs folder. Run
`npx svgo@4 --multipass --precision=1 --final-newline` over each SVG afterwards.

```bash
PYTHONPATH=<folders of `e00413_score.py`> uv run --no-sync python ml-experiments/00413-*/e00413_charts.py
```
"""

import argparse
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00413_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00413-differentiable-physics-demand-network-part-1"
)
WIDTH: Final[int] = 680
ARM_COLOURS: Final[dict[str, str]] = {
    "D0": theme.DATA_BLUE,
    "D0Y": theme.DATA_SKY,
    "D1": theme.DATA_PURPLE,
    "D2": theme.BRAND_ORANGE,
    "GAM stack": theme.DATA_GREEN,
}
GROUPS: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight")
EXAMPLE_SITE_LABEL: Final[str] = "D5"
EXAMPLE_ORIGIN: Final[str] = "2026-01-12"


def save(*, chart: alt.Chart | alt.VConcatChart | alt.LayerChart, name: str) -> None:
    """Write a chart as SVG into the docs folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")


def scores_chart(*, table: pl.DataFrame) -> alt.VConcatChart:
    """Return CRPS by fold for each arm and the GAM stack, one panel per group."""
    rename = {"stack": "GAM stack"}
    data = (
        table.filter(
            pl.col("metric") == "crps", pl.col("arm").is_in(["D0", "D0Y", "D1", "D2", "stack"])
        )
        .with_columns(pl.col("arm").replace(rename))
        .rename({"value": "CRPS"})
    )
    panels = []
    for group in GROUPS:
        panels.append(
            alt.Chart(data.filter(pl.col("group") == group))  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("fold:N", title="Test month"),
                y=alt.Y("CRPS:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                    legend=alt.Legend(orient="top", title=None),
                ),
                strokeDash=alt.StrokeDash("arm:N", legend=None),
            )
            .properties(width=WIDTH, height=190, title=f"{group}, days 1 to 10")
        )
    return alt.vconcat(*panels)


def contrast_chart(*, table: pl.DataFrame) -> alt.VConcatChart:
    """Return D2 minus D0 and D2 minus D0Y per fold, one panel per group."""
    crps = table.filter(pl.col("metric") == "crps").pivot(
        on="arm", index=["fold", "group"], values="value"
    )
    data = pl.concat(
        [
            crps.select(
                "fold",
                "group",
                pl.lit(label).alias("contrast"),
                (pl.col("D2") - pl.col(ref)).alias("gap"),
            )
            for label, ref in (("D2 minus D0", "D0"), ("D2 minus D0Y", "D0Y"))
        ]
    )
    panels = []
    for group in GROUPS:
        base = alt.Chart(data.filter(pl.col("group") == group))
        panels.append(
            alt.layer(
                base.mark_point(filled=True, size=90).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("fold:N", title="Test month"),
                    y=alt.Y("gap:Q", title="Difference in CRPS (points)"),
                    color=alt.Color(
                        "contrast:N",
                        scale=alt.Scale(
                            domain=["D2 minus D0", "D2 minus D0Y"],
                            range=[theme.BRAND_ORANGE, theme.DATA_SKY],
                        ),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                ),
                alt.Chart(pl.DataFrame({"zero": [0.0]})).mark_rule(color="#888").encode(y="zero:Q"),  # ty: ignore[unresolved-attribute]
            ).properties(width=WIDTH, height=170, title=f"{group} (above zero: D2 is worse)")
        )
    return alt.vconcat(*panels)


def curves_charts(*, month: str, seed: int = 0) -> tuple[alt.Chart, alt.Chart]:
    """Return D2's learned shapes (weekday and Sunday) and its temperature responses."""
    z = np.load(fit.OUT_DIR / "models" / "D2" / month / f"s{seed}" / "interpretation.npz")
    shapes = z["shapes"]
    frames = []
    for k in range(shapes.shape[1]):
        for day, name in ((2, "Tuesday"), (7, "Sunday")):
            sel = z["day_of_week"] == day
            frames.append(
                pl.DataFrame(
                    {
                        "hour": z["hour"][sel],
                        "value": shapes[sel, k],
                        "shape": f"shape {k + 1}",
                        "day": name,
                    }
                )
            )
    shape_chart = (
        alt.Chart(pl.concat(frames))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("hour:Q", title="Hour of day (UTC)", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
            y=alt.Y("value:Q", title="Shape (standardised)"),
            color=alt.Color("shape:N", legend=alt.Legend(orient="top", columns=8, title=None)),
        )
        .properties(width=WIDTH, height=150)
        .facet(row=alt.Row("day:N", title=None))
    )
    names = ["heating 1", "heating 2", "heating 3", "cooling", "shortwave", "solar zenith"]
    resp = z["responses_of_temperature"]
    data = pl.concat(
        [
            pl.DataFrame(
                {"temperature": z["temperature_c"], "value": resp[:, j], "curve": names[j]}
            )
            for j in range(4)
        ]
    )
    response_chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("temperature:Q", title="Ensemble-mean temperature (degrees Celsius)"),
            y=alt.Y("value:Q", title="Response (unit root-mean-square)"),
            color=alt.Color("curve:N", legend=alt.Legend(orient="top", title=None)),
        )
        .properties(width=WIDTH, height=180)
    )
    return shape_chart, response_chart


def forecast_panel(
    *, forecast: pl.DataFrame, actual: pl.DataFrame, title: str
) -> alt.TopLevelMixin:
    """Return the median, 50% and 90% bands, and the actuals of one run over a window."""
    base = alt.Chart(forecast).encode(x=alt.X("valid_time:T", title="Valid time (UTC)"))
    return alt.layer(
        base.mark_area(opacity=0.2, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Power (MW)"), y2="q95:Q"
        ),
        base.mark_area(opacity=0.35, color=theme.BRAND_ORANGE).encode(y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_line(color=theme.BRAND_ORANGE).encode(y="q50:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(actual)  # ty: ignore[unresolved-attribute]
        .mark_line(color="#222", strokeDash=[3, 2])
        .encode(x="valid_time:T", y="actual:Q"),
    ).properties(width=WIDTH, height=150, title=title)


def example_charts(*, month: str) -> tuple[alt.VConcatChart, alt.VConcatChart]:
    """Return a day-ahead and a 14-day forecast of D0 and D2 against the actuals at one Primary."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    series_id = series.filter(pl.col("label") == EXAMPLE_SITE_LABEL)["time_series_id"][0]
    actuals = ctx["actuals"].filter(pl.col("time_series_id") == series_id)
    day_ahead, fourteen = [], []
    for arm in ("D0", "D2"):
        stored = pl.read_parquet(fit.OUT_DIR / "seed_preds" / arm / f"{month}_s0.parquet")
        frame = adapter.forecast_frame(stored=stored, series=series, unit="fraction").filter(
            pl.col("time_series_id") == series_id
        )
        origin = frame.filter(
            pl.col("power_fcst_init_time")
            >= pl.lit(EXAMPLE_ORIGIN).str.to_datetime(time_zone="UTC")
        )["power_fcst_init_time"].min()
        run = frame.filter(pl.col("power_fcst_init_time") == origin)
        lead = (pl.col("valid_time") - origin).dt.total_hours()
        for window, store, lo, hi in (
            ("day ahead (hours 12 to 36)", day_ahead, 12, 36),
            ("14 days", fourteen, 0, 336),
        ):
            piece = run.filter(lead >= lo, lead < hi)
            actual = actuals.filter(
                pl.col("valid_time").is_between(
                    piece["valid_time"].min(), piece["valid_time"].max()
                )
            )
            store.append(
                forecast_panel(
                    forecast=piece,
                    actual=actual,
                    title=f"{arm}, {EXAMPLE_SITE_LABEL}, run {origin:%Y-%m-%d %H:%M}, {window}",
                )
            )
    return alt.vconcat(*day_ahead), alt.vconcat(*fourteen)


def main() -> None:
    """Draw the charts for the folds in the score table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2026-01", help="the fold the curves are read from")
    args = parser.parse_args()
    table = pl.read_parquet(fit.OUT_DIR / "score_table.parquet")
    save(chart=scores_chart(table=table), name="scores_by_fold")
    save(chart=contrast_chart(table=table), name="contrasts_by_fold")
    shapes, responses = curves_charts(month=args.month)
    save(chart=shapes, name="learned_shapes")
    save(chart=responses, name="learned_temperature_responses")
    day_ahead, fourteen = example_charts(month=args.month)
    save(chart=day_ahead, name="example_day_ahead")
    save(chart=fourteen, name="example_14_day")


if __name__ == "__main__":
    main()

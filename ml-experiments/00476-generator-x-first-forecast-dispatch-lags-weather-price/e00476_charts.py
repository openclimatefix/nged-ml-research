# ruff: noqa: E501
"""Draw the anonymised figures for experiment 476 from the stored forecasts and score tables.

Every chart shows fractions of the series' maximum absolute output (points are 1% of it) and hours
after the forecast origin. No calendar date, fuel type, plant type, or size appears.
"""

import os
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00476_fit as fit  # ty: ignore[unresolved-import]
import e00476_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

SLUG: Final[str] = "00476-generator-x-first-forecast-dispatch-lags-weather-price"
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
PRIVATE: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / SLUG
HEIGHT: Final[int] = 200
ARM_LABELS: Final[dict[str, str]] = {
    "A0": "A0 calendar and lags",
    "A1": "A1 plus weather",
    "A2": "A2 plus price lags",
    "A3": "A3 calendar and weather",
    "clim": "Same-hour climatology",
    "persist": "Persistence",
}
ARM_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN, "#6B7280", "#9CA3AF",
]  # fmt: skip
MDE: Final[float] = 0.0190


def _save(chart: Any, name: str, *, private: bool = False) -> None:
    sized = chart.properties(width=CONTENT_WIDTH_PX - 160)
    target = PRIVATE / "figures" if private else DOCS_DIR
    target.mkdir(parents=True, exist_ok=True)
    sized.save(target / name)  # ty: ignore[unresolved-attribute]
    if os.environ.get("E476_PNG"):
        sized.save(Path(os.environ["E476_PNG"]) / name.replace(".svg", ".png"), scale_factor=1.5)  # ty: ignore[unresolved-attribute]


def contrast_figure() -> None:
    """Figure 1: each pre-registered contrast with its 99% interval, days 1 to 10."""
    c = pl.read_csv(PRIVATE / "contrasts.csv").filter(
        (pl.col("band") == "days 1-10") & (pl.col("contrast") != "C2 A0 - persistence")
    )
    base = alt.Chart(c).encode(y=alt.Y("contrast:N", title=None, sort=None))  # ty: ignore[unresolved-attribute]
    rule = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#111827").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    band = (
        alt.Chart(pl.DataFrame({"a": [-MDE], "b": [MDE]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color="#E5E7EB", opacity=0.7)
        .encode(x="a:Q", x2="b:Q")
    )
    bars = base.mark_rule(strokeWidth=3, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q",
            title="Quantile score difference (points of maximum output; negative is better for the first arm)",
        ),
        x2="high:Q",
    )
    dots = base.mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="difference:Q"
    )
    _save(
        alt.layer(band, rule, bars, dots).properties(
            height=HEIGHT,
            title=alt.TitleParams(
                "No arm beats same-hour climatology by the minimum detectable effect",
                subtitle=[
                    "Dot: pooled difference. Line: 99% interval from resampling origin days (interim, 4 folds).",
                    "Grey band: one minimum detectable effect either side of zero. Persistence (C2) is off the chart at -4.2.",
                ],
            ),
        ),
        "figure_1.svg",
    )


def fold_figure() -> None:
    """Figure 2: the two headline contrasts by fold."""
    c = pl.read_csv(PRIVATE / "contrasts.csv").filter(
        (pl.col("band") == "days 1-10")
        & pl.col("contrast").is_in(["C1 A0 - climatology", "C3 A1 - A0", "C4 A2 - A0"])
    )
    rows = []
    for r in c.iter_rows(named=True):
        import json

        rows += [
            {"contrast": r["contrast"], "fold": k, "difference": v}
            for k, v in json.loads(r["fold_differences"]).items()
        ]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_circle(size=110, aria=False)
        .encode(
            x=alt.X(
                "difference:Q",
                title="Quantile score difference (points of maximum output; negative is better)",
            ),
            y=alt.Y("contrast:N", title=None, sort=None),
            color=alt.Color(
                "fold:N",
                title="Test month",
                scale=alt.Scale(range=ARM_COLOURS[:4]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            height=HEIGHT - 40,
            title=alt.TitleParams(
                "A0 loses to climatology in 3 of 4 test months; price lags help in all 4",
                subtitle="Each dot is one test month, days 1 to 10. Zero means the same score as the second arm.",
            ),
        )
    )
    _save(chart, "figure_2.svg")


def lead_figure(table: pl.DataFrame) -> None:
    """Figure 3: mean quantile score by lead day."""
    t = (
        table.with_columns(day=((pl.col("lead") - 1e-9) // 24 + 1).cast(pl.Int64))
        .filter(~pl.col("arm").is_in(["persist", "clim_cell"]))
        .group_by("arm", "day")
        .agg(pl.col("score").mean() * 100)
        .with_columns(label=pl.col("arm").replace(ARM_LABELS))
    )
    chart = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "day:Q",
                title="Lead (days after the forecast origin)",
                axis=alt.Axis(values=list(range(1, 11))),
            ),
            y=alt.Y(
                "score:Q",
                title="Quantile score (points of maximum output; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                title=None,
                scale=alt.Scale(
                    domain=[ARM_LABELS[a] for a in ("A0", "A1", "A2", "A3", "clim")],
                    range=[*ARM_COLOURS[:4], ARM_COLOURS[4]],
                ),
                legend=alt.Legend(orient="top", columns=2),
            ),
        )
        .properties(
            height=HEIGHT,
            title=alt.TitleParams(
                "The arms and climatology score within a few tenths of a point at every lead",
                subtitle="Persistence (about 8 points at every lead day) is off the chart.",
            ),
        )
    )
    _save(chart, "figure_3.svg")


def example_figure(*, month: str, name: str, hours: float, start: float, title: str) -> None:
    """Draw A2's median and bands against the actual output for one origin, into the private store."""
    pred = pl.read_parquet(fit.PRED_DIR / "A2" / f"{month}_s0.parquet").filter(
        pl.col("actual").is_not_null()
    )
    origins = pred["origin_us"].unique().sort().to_list()
    origin = origins[len(origins) // 2]
    one = (
        pred.filter(pl.col("origin_us") == origin)
        .filter((pl.col("lead") > start) & (pl.col("lead") <= start + hours))
        .sort("valid_us")
    )
    one = one.filter(pl.col("actual").is_not_nan())
    d = (
        one.select("lead", "actual", "q05", "q25", "q50", "q75", "q95").to_pandas()
        if False
        else one
    )
    base = alt.Chart(d).encode(x=alt.X("lead:Q", title="Hours after the forecast origin"))  # ty: ignore[unresolved-attribute]
    wide = base.mark_area(opacity=0.2, color=ocf.DATA_BLUE).encode(y="q05:Q", y2="q95:Q")  # ty: ignore[unresolved-attribute]
    narrow = base.mark_area(opacity=0.35, color=ocf.DATA_BLUE).encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
    med = base.mark_line(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q50:Q", title="Output (fraction of maximum absolute output)")
    )
    act = base.mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.2).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    _save(
        alt.layer(wide, narrow, med, act).properties(
            height=HEIGHT,
            title=alt.TitleParams(
                title,
                subtitle=[
                    "Orange: what happened. Blue line: A2 median; dark band 50% and light band 90%.",
                    f"One forecast run in {month[:4]}-{month[5:]} from seed 0; no calendar date shown.",
                ],
            ),
        ),
        name,
        private=True,
    )


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    table = score.collect(seeds=fit.SEEDS, factor=1.0)
    contrast_figure()
    fold_figure()
    lead_figure(table)
    example_figure(
        month="2026-01",
        name="figure_4.svg",
        hours=24,
        start=12,
        title="Day-ahead forecast, winter: the band is wide and the median is flat",
    )
    example_figure(
        month="2026-01",
        name="figure_5.svg",
        hours=240,
        start=0,
        title="Ten-day forecast, winter: the band stays about as wide at every lead",
    )
    example_figure(
        month="2026-06", name="figure_6.svg", hours=240, start=0, title="Ten-day forecast, summer"
    )
    np.random.seed(0)


if __name__ == "__main__":
    main()

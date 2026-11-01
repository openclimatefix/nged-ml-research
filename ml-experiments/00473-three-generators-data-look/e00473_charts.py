"""Draw the anonymised aggregate figures for experiment 473 from the tables `e00473_look.py` wrote.

Every chart shows capacity fractions of the maximum absolute output, with no fuel type, plant type,
or size, and no raw half-hourly trace.
"""

import json
import os
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

SLUG: Final[str] = "00473-three-generators-data-look"
DATA: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"])
PRIVATE: Final[Path] = DATA / "ml-experiments" / SLUG
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
LABEL_TO_TYPE: Final[dict[str, str]] = json.loads((PRIVATE / "label_map_PRIVATE.json").read_text())
HEIGHT: Final[int] = 170
LABELS: Final[list[str]] = ["Generator X", "Generator Y", "Generator Z"]
COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE]
LETTERS: Final[dict[str, str]] = {"X": "Generator X", "Y": "Generator Y", "Z": "Generator Z"}
ACTIVE: Final[float] = 0.05


def _save(chart: alt.Chart | alt.VConcatChart, name: str) -> None:
    chart.properties(width=CONTENT_WIDTH_PX - 80).save(DOCS_DIR / name)  # ty: ignore[unresolved-attribute]


def _colour() -> alt.Color:
    return alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(domain=LABELS, range=COLOURS),
        legend=alt.Legend(orient="top"),
    )


def main() -> None:  # noqa: D103
    monthly = pl.concat(
        [
            pl.read_csv(PRIVATE / f"monthly_{k}.csv").with_columns(series=pl.lit(v))
            for k, v in LETTERS.items()
        ]
    ).with_columns(
        month=pl.col("month").str.slice(0, 10).str.to_date(), active=pl.col("share_active") * 100
    )
    fig1 = (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_line(point=False)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("active:Q", title="Active half-hours (%)"),
            color=_colour(),
        )
        .properties(height=HEIGHT)
    )
    _save(fig1, "figure_1.svg")
    cover = pl.read_csv(PRIVATE / "coverage.csv").with_columns(
        month=pl.col("month").str.slice(0, 10).str.to_date(),
        series=pl.col("series").replace(LETTERS),
    )
    cover = cover.with_columns(coverage=(pl.col("n") / 1440 * 100).clip(upper_bound=100))
    fig2 = (
        alt.Chart(cover)  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("coverage:Q", title="Half-hours present (% of a 30-day month)"),
            color=_colour(),
        )
        .properties(height=HEIGHT)
    )
    _save(fig2, "figure_2.svg")
    hours = pl.concat(
        [
            pl.read_csv(PRIVATE / f"hour_local_{k}.csv").with_columns(series=pl.lit(v))
            for k, v in LETTERS.items()
        ]
    )
    fig3 = (
        alt.Chart(hours)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "h:Q", title="Hour of day (UK local time)", axis=alt.Axis(values=[0, 6, 12, 18, 24])
            ),
            y=alt.Y("mean:Q", title="Mean output (fraction of maximum absolute)"),
            color=_colour(),
        )
        .properties(height=HEIGHT)
    )
    _save(fig3, "figure_3.svg")
    deciles = pl.concat(
        [
            pl.read_csv(PRIVATE / f"price_decile_{k}.csv").with_columns(series=pl.lit(v))
            for k, v in LETTERS.items()
            if (PRIVATE / f"price_decile_{k}.csv").exists()
        ]
    ).with_columns(active=pl.col("share_active") * 100)
    fig4 = (
        alt.Chart(deciles)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "price_median:Q", title="Median market index price in the decile (GBP per MWh)"
            ),
            y=alt.Y("active:Q", title="Active half-hours (%)"),
            color=_colour(),
        )
        .properties(height=HEIGHT)
    )
    _save(fig4, "figure_4.svg")
    meta = pl.read_parquet(DATA / "NGED" / "metadata.parquet")
    power = pl.scan_delta(str(DATA / "NGED" / "cleaned_power_time_series.delta"))
    rows = []
    for k, kind in LABEL_TO_TYPE.items():
        tsid = meta.filter(pl.col("time_series_type") == kind)["time_series_id"][0]
        v = power.filter(pl.col("time_series_id") == tsid).select("power").collect()["power"]
        f = v / v.abs().max()
        total = f.len()
        parts = {
            "active export": (f >= ACTIVE).sum(),
            "active import": (f <= -ACTIVE).sum(),
            "idle, small import": ((f < 0) & (f > -ACTIVE)).sum(),
            "idle, exact zero": (f == 0).sum(),
            "idle, small export": ((f > 0) & (f < ACTIVE)).sum(),
        }
        rows += [
            {"series": LETTERS[k], "reading": a, "share": b / total * 100} for a, b in parts.items()
        ]
    order = [
        "active export",
        "active import",
        "idle, small import",
        "idle, exact zero",
        "idle, small export",
    ]
    fig5 = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("series:N", title=None, sort=LABELS),
            x=alt.X("share:Q", title="Share of half-hours (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "reading:N",
                title=None,
                sort=order,
                scale=alt.Scale(
                    domain=order,
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_SKY,
                        ocf.DATA_BLUE_LIGHT,
                        ocf.DATA_PURPLE,
                    ],
                ),
                legend=alt.Legend(orient="top", columns=3),
            ),
        )
        .properties(height=110)
    )
    _save(fig5, "figure_5.svg")
    pl.DataFrame(rows).write_csv(PRIVATE / "composition.csv")


if __name__ == "__main__":
    main()

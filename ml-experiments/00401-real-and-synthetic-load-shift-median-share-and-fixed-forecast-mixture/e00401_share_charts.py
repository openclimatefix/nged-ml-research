"""Experiment 401, Job 1: the two share figures.

The numbers are keyed in from `e00401_share.py`'s printed report (tables S1, R1, and B5; medians
over events with 95% intervals from resampling events).

```bash
uv run --no-sync python ml-experiments/00401-*/e00401_share_charts.py
```
"""
# ruff: noqa: E501

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / "00401-real-and-synthetic-load-shift-median-share-and-fixed-forecast-mixture"
)  # fmt: skip
STATUS: Final[list[str]] = ["0 lags", "1 or 2 lags", "3 lags"]
# (series, lag status, median, lower, upper)
OBSERVED: Final[tuple[tuple[str, str, float, float, float], ...]] = (
    ("Real events: monthly forecast with lags", "0 lags", 0.604, 0.195, 0.786),
    ("Real events: monthly forecast with lags", "1 or 2 lags", 0.833, 0.584, 0.990),
    ("Real events: monthly forecast with lags", "3 lags", 0.937, 0.626, 1.011),
    ("Real events: forecast with no power lags at all", "0 lags", 0.592, 0.216, 0.766),
    ("Real events: forecast with no power lags at all", "1 or 2 lags", 0.754, 0.453, 0.934),
    ("Real events: forecast with no power lags at all", "3 lags", 0.948, 0.426, 1.018),
    ("Synthetic steps: monthly forecast with lags", "0 lags", -0.006, -0.028, 0.047),
    ("Synthetic steps: monthly forecast with lags", "1 or 2 lags", 0.016, 0.012, 0.072),
    ("Synthetic steps: monthly forecast with lags", "3 lags", 0.283, 0.111, 0.436),
)  # fmt: skip
RESPONSE: Final[tuple[tuple[str, str, float, float, float], ...]] = (
    ("Real events: forecast with lags minus forecast without lags", "0 lags", -0.007, -0.032, 0.026),
    ("Real events: forecast with lags minus forecast without lags", "1 or 2 lags", 0.019, -0.011, 0.058),
    ("Real events: forecast with lags minus forecast without lags", "3 lags", 0.032, -0.001, 0.053),
    ("Synthetic steps: change in the median when the lags are shifted", "0 lags", 0.000, 0.000, 0.000),
    ("Synthetic steps: change in the median when the lags are shifted", "1 or 2 lags", 0.014, 0.008, 0.022),
    ("Synthetic steps: change in the median when the lags are shifted", "3 lags", 0.323, 0.135, 0.385),
)  # fmt: skip
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]


def draw(*, rows: tuple, title: str, subtitle: list[str], x_title: str, name: str) -> None:
    """Write one dot-and-interval figure."""
    frame = pl.DataFrame(
        rows, schema=["series", "status", "median", "lower", "upper"], orient="row"
    )
    order = list(dict.fromkeys(frame["series"].to_list()))
    base = alt.Chart(frame).encode(
        y=alt.Y("status:N", sort=STATUS, title=None),
        yOffset=alt.YOffset("series:N", sort=order),
        color=alt.Color("series:N", sort=order, scale=alt.Scale(domain=order, range=COLOURS[: len(order)]),
                        legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=620)),
    )  # fmt: skip
    dots = base.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("median:Q", title=x_title)
    )
    bars = base.mark_rule(strokeWidth=2, aria=False).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=ocf.GREY_2, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = alt.layer(zero, bars, dots).properties(
        width=600, height=210, title=alt.TitleParams(title, subtitle=subtitle)
    )
    chart.save(DOCS_DIR / name)


def main() -> None:
    """Write both figures."""
    draw(
        rows=OBSERVED,
        title="A forecast with no power lags shows the same share of the step as the forecast with lags",
        subtitle=[
            "Share of the labelled step the median forecast sits at: 1 minus (actual minus median) over the step. Dot: median over events.",
            "Line: 95% interval from resampling events. Rows are grouped by how many of the 3 lagged readings fall inside the event.",
        ],
        x_title="Share of the labelled step (0: median at normal running; 1: at the event reading)",
        name="share_by_lag_status.svg",
    )
    draw(
        rows=RESPONSE,
        title="In real events the lags move the median by 3% of the step or less",
        subtitle=[
            "Change in the median forecast as a share of the labelled step. Dot: median over events.",
            "Line: 95% interval from resampling events. Real events: weekly forecast with lags minus",
            "weekly forecast without lags, same rows. Synthetic: same Primaries and depths, lags shifted.",
        ],
        x_title="Change in the median, as a share of the labelled step (more is a bigger response)",
        name="lag_response_real_and_synthetic.svg",
    )


if __name__ == "__main__":
    main()

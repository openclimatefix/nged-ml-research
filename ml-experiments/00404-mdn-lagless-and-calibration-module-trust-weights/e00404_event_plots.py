# ruff: noqa: E501
"""Draw day-ahead forecasts of experiment 404 around real load-shift events (no fit).

Each figure shows one real event on one Primary: the actual readings, the B-mix forecast with its
90% band, and the median of B1 (learned lag weights) and B3 (attention over lags), all taken from
forecasts issued 12 to 36 hours ahead. The horizontal axis counts days from the event's onset.
Primaries are substations, so their series carry no anonymisation requirement.

```bash
PYTHONPATH=<as e00404_fit.py shows> \
    uv run --no-sync python ml-experiments/00404-*/e00404_event_plots.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00404-mdn-lagless-and-calibration-module-trust-weights"
)
EVENTS: Final[tuple[tuple[str, str, int, int], ...]] = (
    ("D5", "2026-01", 112321, 112550),
    ("D5", "2026-01", 111746, 111922),
    ("D5", "2025-10", 107175, 107377),
    ("D14", "2025-10", 106721, 106748),
    ("D16", "2026-01", 112194, 112242),
    ("D14", "2026-03", 114748, 114772),
)
"""Primary, scored month, and onset and end grid index of each plotted real event."""
MARGIN: Final[int] = 3 * 48
LEAD_RANGE: Final[tuple[float, float]] = (12.0, 36.0)
WIDTH: Final[int] = 680
LINES: Final[dict[str, str]] = {
    "Actual": theme.BLACK_1,
    "B-mix": theme.BRAND_ORANGE,
    "B1": theme.DATA_PURPLE,
    "B3": theme.DATA_BLUE,
}


def day_ahead(*, arm: str, month: str, primary: int) -> pl.DataFrame:
    """Return an arm's forecasts issued 12 to 36 hours ahead for one Primary on the clean world."""
    frame = pl.read_parquet(score.prediction_path(arm=arm, month=month, seed=0, world=-1))
    return frame.filter(
        (pl.col("primary") == primary) & pl.col("lead_hours").is_between(*LEAD_RANGE)
    ).unique(subset="target", keep="first")


def figure(*, site: str, month: str, onset: int, end: int, number: int) -> None:
    """Draw one event's figure."""
    sites = fit.prep.primaries()["site"].to_list()
    primary = sites.index(site)
    lo, hi = onset - MARGIN, end + MARGIN
    parts = []
    for arm in ("B-mix", "B1", "B3"):
        part = day_ahead(arm=arm, month=month, primary=primary).filter(
            pl.col("target").is_between(lo, hi)
        )
        parts.append(
            part.select(
                "target",
                (pl.col("q50") * 100).alias("median"),
                (pl.col("q05") * 100).alias("low"),
                (pl.col("q95") * 100).alias("high"),
            ).with_columns(series=pl.lit(arm))
        )
    mix = parts[0]
    actual = (
        day_ahead(arm="B-mix", month=month, primary=primary)
        .filter(pl.col("target").is_between(lo, hi))
        .select("target", (pl.col("actual") * 100).alias("median"))
        .with_columns(series=pl.lit("Actual"))
    )
    lines = pl.concat([actual, *[p.select(actual.columns) for p in parts]]).with_columns(
        day=(pl.col("target") - onset) / 48.0
    )
    band = mix.with_columns(day=(pl.col("target") - onset) / 48.0)
    day_range = [float(lines["day"].min()), float(lines["day"].max())]  # ty: ignore[invalid-argument-type]
    x = alt.X(
        "day:Q",
        title="Days since the event began (shaded: the event)",
        scale=alt.Scale(domain=day_range),
    )
    shade = (
        alt.Chart(pl.DataFrame({"a": [0.0], "b": [(end - onset) / 48.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rect(color=theme.GREY_3, opacity=0.35)
        .encode(x="a:Q", x2="b:Q")
    )
    area = (
        alt.Chart(band.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_area(color=theme.BRAND_ORANGE, opacity=0.18)
        .encode(x=x, y=alt.Y("low:Q", title="Power (% of capacity)"), y2="high:Q")
    )
    line = (
        alt.Chart(lines.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False)
        .encode(
            x=x,
            y="median:Q",
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(LINES), range=list(LINES.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeWidth=alt.condition(alt.datum.series == "Actual", alt.value(1.8), alt.value(1.2)),
        )
    )
    chart = (shade + area + line).properties(
        width=WIDTH - 110,
        height=190,
        title=alt.TitleParams(
            [
                f"Primary {site}, a real load-shift event in {month}: day-ahead forecasts (12 to 36 hours ahead)",
                "Orange band: B-mix 90% interval. Lines: median of each forecast. Real readings, one fit per month.",
            ],
            anchor="start",
        ),
    )
    chart.save(FIGURE_DIR / f"figure_{number}_event_{site}_{month}_{onset}.svg")


def main() -> None:
    """Draw the event figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for i, (site, month, onset, end) in enumerate(EVENTS, start=5):
        figure(site=site, month=month, onset=onset, end=end, number=i)


if __name__ == "__main__":
    main()

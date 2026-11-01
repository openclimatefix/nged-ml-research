# ruff: noqa: E501
r"""Draw the figures of experiment 467 from the stored scores and forecasts (no fit).

- Figure 1: construct minus B-mix on three row sets at the signed-MW six, with month dots.
- Figures 2 and 3: a day-ahead and a 14-day forecast of one Primary (C1e-2 beside B-mix).
- Figure 4: six real events with B-mix, C1e-2, and the anchor LL.

```bash
<PYTHONPATH as e00467_arm.py shows, plus experiment 445's folder> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00467-*/e00467_figures.py
```
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00445_figures as f445  # ty: ignore[unresolved-import]
import e00454_build as build  # ty: ignore[unresolved-import]
import e00467_arm as arm_mod  # ty: ignore[unresolved-import]
import e00467_score as score467  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00467-anchor-plus-penalised-lag-correction-network-for-the-lagless-forecast"
)
WIDTH: Final[int] = 680
POINTS: Final[float] = 100.0
ORANGE, BLUE, GREEN, PURPLE = (
    theme.BRAND_ORANGE,
    theme.DATA_BLUE,
    theme.DATA_GREEN,
    theme.DATA_PURPLE,
)
CLASS_TITLES: Final[dict[str, str]] = {
    "clean": "Rows with no labelled event",
    "harm": "Harm rows (lags touch an event)",
    "event": "Rows inside real events",
}
LINES: Final[dict[str, str]] = {"B-mix": GREEN, "C1e-2": ORANGE, "LL": PURPLE}
ORDER: Final[tuple[str, ...]] = (
    "LL",
    "Z1e-2",
    "C1e-3",
    "C1e-2",
    "C1e-1",
    "C1",
    "B-mix",
    "S1-mix",
    "S1",
    "A0-inj",
)
PALETTE: Final[list[str]] = [
    PURPLE,
    "#999",
    ORANGE,
    ORANGE,
    ORANGE,
    ORANGE,
    GREEN,
    GREEN,
    BLUE,
    BLUE,
]
SCORE_DIR: Final[Path] = arm_mod.WORK_DIR / "scores"


def figure_scores() -> None:
    """Figure 1: absolute CRPS of each arm on three row sets, per metering group."""
    absolute = pl.read_csv(SCORE_DIR / "absolute_all.csv").filter(pl.col("group") != "report_only")
    absolute = absolute.filter(pl.col("arm").is_in(list(ORDER)))
    titles = {"signed_six": "Signed-MW six", "mva_eight": "MVA-only eight"}
    panels = []
    for group, title in titles.items():
        data = absolute.filter(pl.col("group") == group).with_columns(
            row_set=pl.col("cls").replace(CLASS_TITLES)
        )
        panels.append(
            alt.Chart(data.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                x=alt.X("arm:N", sort=list(ORDER), title=None),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)"),
                color=alt.Color(
                    "arm:N", sort=list(ORDER), legend=None, scale=alt.Scale(range=PALETTE)
                ),
                column=alt.Column("row_set:N", title=title, sort=list(CLASS_TITLES.values())),
            )
            .properties(width=150, height=150)
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Absolute CRPS of the correction arms and references (interim, 5 months, 2 fit seeds)",
                "Z1e-2 is the same network with the lag inputs set to zero.",
            ],
            anchor="start",
        )
    ).save(FIGURE_DIR / "figure_1_scores.svg")


def read_arm(*, arm: str, month: str) -> pl.DataFrame:
    """Return a construct's first cell with quantiles and actuals in points of capacity."""
    d = pl.read_parquet(build.prediction_path(arm=arm, month=month, cell=0))
    cols = [c for c in d.columns if c.startswith("q")]
    d = d.select("primary", "origin", "target", "lead_hours", "actual", *cols)
    return d.with_columns(*[pl.col(c) * POINTS for c in [*cols, "actual"]])


def figure_examples(*, site: str) -> None:
    """Figures 2 and 3: a day-ahead and a 14-day forecast, a winter and a summer origin."""
    times = common.grid_times()
    primary = fit.prep.primaries()["site"].to_list().index(site)
    for number, name, hours in ((2, "day_ahead", 36.0), (3, "14_day", 400.0)):
        panels = []
        for month in ("2026-01", "2026-06"):
            frames = []
            for arm in ("B-mix", "C1e-2"):
                d = read_arm(arm=arm, month=month).filter(pl.col("primary") == primary)
                origins = np.sort(d["origin"].unique().to_numpy())
                origin = int(origins[len(origins) // 2])
                frames.append(
                    d.filter(
                        pl.col("origin") == origin, pl.col("lead_hours") <= hours
                    ).with_columns(arm=pl.lit(arm))
                )
            d = pl.concat(frames)
            d = d.with_columns(time=pl.Series([times[t] for t in d["target"].to_list()]))
            x = alt.X("time:T", title="Valid time (UTC)")
            colour = alt.Color(
                "arm:N",
                scale=alt.Scale(domain=["B-mix", "C1e-2"], range=[GREEN, ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            )
            data = d.to_pandas()
            outer = (
                alt.Chart(data)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.12)
                .encode(
                    x=x,
                    y=alt.Y(
                        "q05:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)
                    ),
                    y2="q95:Q",
                    color=colour,
                )
            )
            inner = (
                alt.Chart(data)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.25)
                .encode(x=x, y="q25:Q", y2="q75:Q", color=colour)
            )
            median = alt.Chart(data).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
            truth = (
                alt.Chart(data).mark_line(color="#333", strokeDash=[2, 2]).encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
            )
            origin_time = times[int(frames[0]["origin"][0])]
            panels.append(
                (outer + inner + median + truth).properties(
                    width=WIDTH - 90,
                    height=170,
                    title=f"Primary {site}, forecast issued {origin_time:%Y-%m-%d %H:%M}",
                )
            )
        alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                [
                    f"{name.replace('_', '-')} forecast: B-mix and the correction network C1e-2 (penalty 1e-2), interim",
                    "Line: median. Dark band: 25% to 75%. Light band: 5% to 95%. Dashed: the cleaned reading.",
                ],
                anchor="start",
            )
        ).save(FIGURE_DIR / f"figure_{number}_example_{name}.svg")


def figure_events() -> None:
    """Figure 4: six real events (three per metering group), forecasts issued the day before."""
    times = common.grid_times()
    real = score.real_event_mask()
    panels = []
    for i, e in enumerate(f445.pick_events(real=real), start=1):
        frames = []
        for arm in LINES:
            d = read_arm(arm=arm, month=e["month"]).filter(pl.col("primary") == e["p"])
            origins = np.sort(d["origin"].unique().to_numpy())
            origin = int(origins[np.argmin(np.abs(origins - (e["start"] - 48)))])
            frames.append(
                d.filter(pl.col("origin") == origin, pl.col("lead_hours") <= 120).with_columns(
                    arm=pl.lit(arm)
                )
            )
        d = pl.concat(frames)
        d = d.with_columns(time=pl.Series([times[t] for t in d["target"].to_list()]))
        lo, hi = times[e["start"]], times[e["end"]]
        x = alt.X(
            "time:T",
            title=None,
            scale=alt.Scale(domain=[lo - timedelta(days=1), hi + timedelta(days=2)]),
        )
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(LINES), range=list(LINES.values())),
            legend=alt.Legend(title=None, orient="top") if i == 1 else None,
        )
        shade = (
            alt.Chart(pl.DataFrame({"a": [lo], "b": [hi]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.12, color="#888")
            .encode(x="a:T", x2="b:T")
        )
        ref = d.filter(pl.col("arm") == "B-mix").to_pandas()
        band = (
            alt.Chart(ref)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.12, color=GREEN)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
            )
        )
        median = alt.Chart(d.to_pandas()).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        actual = alt.Chart(ref).mark_line(color="#333").encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        group = "signed-MW six" if e["group"] == "signed_six" else "MVA-only eight"
        panels.append(
            (shade + band + median + actual).properties(
                width=WIDTH - 90, height=130, title=f"Event {i}: Primary {e['site']} ({group})"
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Labelled load-shift events: B-mix, C1e-2, and anchor (LL) medians (interim)",
                "Forecasts issued the day before the padded event starts (shaded). Black: the cleaned reading.",
                "Band: B-mix 5% to 95%. Forecasts reach 5 days. Some real events are not logged.",
            ],
            anchor="start",
        )
    ).save(FIGURE_DIR / "figure_4_real_events.svg")


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    build.prediction_path = score467.prediction_path
    figure_scores()
    figure_examples(site="D5")
    figure_events()


if __name__ == "__main__":
    main()

r"""Draw the figures of experiment 470 from the stored scores and forecasts (no fit).

- Figure 1: each candidate minus A-pool on the gating slices, with Bonferroni 95% intervals.
- Figure 2: CRPS by lead day at the eight and the six for the candidates and the references.
- Figures 3 and 4: a day-ahead and a 14-day forecast of one Primary (A-pool, K2-S1, K2-425A).
- Figure 5: six real events (three per metering group) with the same three forecasts.

```bash
source .e470env.sh
uv run --no-sync python ml-experiments/00470-*/e00470_figures.py
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
import e00470_build as build  # ty: ignore[unresolved-import]
import e00470_score as scoring  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import widen  # ty: ignore[unresolved-import]
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00470-first-release-demand-forecaster-assembly-and-comparison"
)
WIDTH: Final[int] = 680
POINTS: Final[float] = 100.0
ORANGE, BLUE, GREEN, PURPLE = (
    theme.BRAND_ORANGE,
    theme.DATA_BLUE,
    theme.DATA_GREEN,
    theme.DATA_PURPLE,
)
LINES: Final[dict[str, str]] = {"A-pool": BLUE, "K2-S1": PURPLE, "K2-425A": ORANGE}
SLICE_LABELS: Final[dict[str, str]] = {
    "eight": "Eight MVA-only Primaries, all rows",
    "six_clean": "Signed-MW six, rows with no labelled event",
    "six_nra": "Signed-MW six, real events (normal-running actuals)",
    "cold_eight": "Eight, cold winter nights",
    "cold_six": "Six, cold winter nights",
    "band_eight": "Eight, 12 to 36 hours ahead",
    "band_six": "Six, 12 to 36 hours ahead",
}
CONTRAST_ORDER: Final[list[str]] = [
    "K2-S1 - A-pool", "K2-425A - A-pool", "K2-425A+S1 - A-pool", "A-gam-in - A-pool",
    "K2-425A - K2-S1", "K2-425A+S1 - K2-S1",
]  # fmt: skip
SERIES: Final[dict[str, str]] = {
    "A-pool": BLUE, "K2-S1": PURPLE, "K2-425A": ORANGE, "A-gam-in": theme.DATA_GREEN,
    "BG2": theme.BLACK_1, "manual_heuristic": theme.GREY_3, "climatology": "#bbbbbb",
}  # fmt: skip


def figure_contrasts() -> None:
    """Figure 1: contrasts with Bonferroni intervals on the seven gating slices."""
    frame = (
        pl.read_csv(scoring.SCORE_DIR / "contrasts.csv")
        .filter((pl.col("metric") == "crps") & pl.col("slice").is_in(list(SLICE_LABELS)))
        .with_columns(label=pl.col("slice").replace(SLICE_LABELS))
        .to_pandas()
    )
    colour = alt.Color(
        "contrast:N", sort=CONTRAST_ORDER, legend=alt.Legend(title=None, orient="top", columns=2)
    )
    y = alt.Y(
        "label:N", sort=list(SLICE_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330)
    )
    bars = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(
            y=y, x="low_bonferroni:Q", x2="high_bonferroni:Q", color=colour, yOffset="contrast:N"
        )
    )  # ty: ignore[unresolved-attribute]
    points = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=45)
        .encode(
            y=y,
            x=alt.X(
                "difference:Q",
                title="CRPS difference (points of capacity; negative favours the first arm)",
            ),
            color=colour,
            yOffset="contrast:N",
        )
    )  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[3, 3], color=theme.BLACK_1)
        .encode(x="z:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + points).properties(
        width=WIDTH - 330, height=380,
        title=alt.TitleParams(
            ["Each candidate against A-pool or K2-S1 (interim, selection-biased)",
             "Dot: mean over 5 folds. Line: Bonferroni 95% interval (54 tests),",
             "resampling blocks of one forecast day."],
            anchor="start"),
    )  # fmt: skip
    chart.save(FIGURE_DIR / "figure_1_contrasts.svg")


def figure_by_day() -> None:
    """Figure 2: CRPS by lead day on the rows every candidate and reference holds."""
    table = scoring.load_rows().filter(pl.col("ref14"))
    panels = []
    for group, title in (
        ("mva_eight", "Eight MVA-only Primaries"),
        ("signed_six", "Signed-MW six, rows with no labelled event"),
    ):
        part = table.filter(pl.col("group") == group)
        if group == "signed_six":
            part = part.filter(pl.col("cls") == "clean")
        part = part.with_columns(
            day=(pl.col("lead_hours") / 24.0).ceil().cast(pl.Int32).clip(1, 10)
        )
        rows = []
        for arm in SERIES:
            by = (
                part.group_by("day")
                .agg(pl.col(f"{arm}__crps").mean().alias("crps"))
                .with_columns(arm=pl.lit(arm))
            )
            rows.append(by)
        data = pl.concat(rows).drop_nulls().to_pandas()
        panels.append(
            alt.Chart(data)
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("day:O", title="Lead day"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)"),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
                    legend=alt.Legend(title=None, orient="top", columns=4),
                ),
            )
            .properties(width=WIDTH - 90, height=190, title=title)
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "CRPS by lead day (interim; BG2 stops at day 7)",
                "Neural arms widened by 1.15; rows every arm holds",
            ],
            anchor="start",
        )
    ).save(FIGURE_DIR / "figure_2_crps_by_day.svg")


def read_arm(*, arm: str, month: str) -> pl.DataFrame:
    """Return an arm's cell 0 widened 1.15, in points of capacity, with the 454 row flags."""
    base = pl.read_parquet(build.prediction_path(arm="B-mix", month=month, cell=0)).select(
        *build.KEYS
    )
    stored = pl.read_parquet(build.prediction_path(arm=arm, month=month, cell=0))
    actual = scoring.score454.month_base(month=month).select(*build.KEYS, "actual", "lead_hours")
    d = widen(
        frame=base.join(stored, on=build.KEYS, how="left", maintain_order="left").join(
            actual, on=build.KEYS
        ),
        factor=scoring.HEADLINE,
    )
    return d.with_columns(
        *[pl.col(c).cast(pl.Float64) * POINTS for c in [*score.LEVEL_NAMES, "actual"]]
    )


def figure_examples(*, site: str) -> None:
    """Figures 3 and 4: a day-ahead and a 14-day forecast, a winter and a summer origin."""
    times = common.grid_times()
    primary = fit.prep.primaries()["site"].to_list().index(site)
    for number, name, hours in ((3, "day_ahead", 36.0), (4, "14_day", 400.0)):
        panels = []
        for month in ("2026-01", "2026-06"):
            frames = []
            for arm in LINES:
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
                scale=alt.Scale(domain=list(LINES), range=list(LINES.values())),
                legend=alt.Legend(title=None, orient="top"),
            )
            data = d.to_pandas()
            outer = (
                alt.Chart(data)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.10)
                .encode(
                    x=x,
                    y=alt.Y(
                        "q05:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)
                    ),
                    y2="q95:Q",
                    color=colour,
                )
            )  # ty: ignore[unresolved-attribute]
            inner = (
                alt.Chart(data)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.22)
                .encode(x=x, y="q25:Q", y2="q75:Q", color=colour)
            )  # ty: ignore[unresolved-attribute]
            median = alt.Chart(data).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
            truth = (
                alt.Chart(data).mark_line(color="#333", strokeDash=[2, 2]).encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
            )  # ty: ignore[unresolved-attribute]
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
                    f"{name.replace('_', '-')} forecast: A-pool, K2-S1, and K2-425A (interim)",
                    "Line: median. Dark band: 25% to 75%. Light band: 5% to 95%.",
                    "Dashed: the cleaned reading.",
                ],
                anchor="start",
            )
        ).save(FIGURE_DIR / f"figure_{number}_example_{name}.svg")


def figure_events() -> None:
    """Figure 5: six real events (three per metering group), forecasts issued the day before."""
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
        )  # ty: ignore[unresolved-attribute]
        ref = d.filter(pl.col("arm") == "A-pool").to_pandas()
        band = (
            alt.Chart(ref)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.12, color=BLUE)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
            )
        )  # ty: ignore[unresolved-attribute]
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
                "Labelled load-shift events: A-pool, K2-S1, and K2-425A medians (interim)",
                "Forecast issued the day before the padded event starts (shaded).",
                "Black: the cleaned reading. Band: A-pool 5% to 95%.",
                "Forecasts reach 5 days. Some real events are not logged.",
            ],
            anchor="start",
        )
    ).save(FIGURE_DIR / "figure_5_real_events.svg")


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_by_day()
    figure_examples(site="D5")
    figure_events()


if __name__ == "__main__":
    main()

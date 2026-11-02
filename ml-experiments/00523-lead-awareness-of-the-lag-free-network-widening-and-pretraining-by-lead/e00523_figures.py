# ruff: noqa: E501
"""Draw the figures of experiment 523 from the tables of `e00523_scores.py`,
`e00523_lead_swap.py`, and the panels.

Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

1. `figure_1_pretrained_minus_unpretrained.svg`: L0PT minus L0 CRPS by lead band, Bonferroni
   intervals;
2. `figure_2_width_growth.svg`: 90% band width growth from 12-24 h to 120-240 h, pooled and within
   one half-hour of the day;
3. `figure_3_lead_swap.svg`: change in the 90% width and in the median when L-1d's lead input is
   replaced;
4. `figure_4_per_band_widening.svg`: per-band widening minus flat 1.15, CRPS by lead band;
5. `figure_5_pit.svg`: PIT histograms at 120 to 240 h;
6. `figure_6_example_*.svg`: day-ahead and 10-day forecasts of D4.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00521_refs as refs  # ty: ignore[unresolved-import]
import e00521_stats as st  # ty: ignore[unresolved-import]
import e00523_panel as panel  # ty: ignore[unresolved-import]
import e00523_scores as sc  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00523-lead-awareness-of-the-lag-free-network-widening-and-pretraining-by-lead"
)
SCORES: Final[Path] = panel.DATA / "scores"
WIDTH: Final[int] = 560
BAND_ORDER: Final[list[str]] = list(st.BANDS)
SETS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}
ARM_COLOURS: Final[dict[str, str]] = {
    "L-1d": theme.DATA_BLUE,
    "K2-S1": theme.BLACK_2,
    "LL": theme.BRAND_ORANGE,
    "L0PT": theme.DATA_AMBER,
}


def zero_rule() -> alt.Chart:
    """Return a horizontal rule at zero."""
    return (
        alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color=theme.BLACK_2)
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]


def interval_panels(
    *, table: pl.DataFrame, title: str, ytitle: str, caption: list[str], path: str
) -> None:
    """Draw a contrast by band, one panel per set, with day-block and whole-month Bonferroni bars."""
    panels = []
    for key, label in SETS.items():
        part = table.filter(pl.col("set") == key).to_pandas()
        base = alt.Chart(part).encode(
            x=alt.X("band:N", sort=BAND_ORDER, title="Lead time of the forecast")
        )  # ty: ignore[unresolved-attribute]
        month_bar = base.mark_rule(strokeWidth=1.5, color=theme.DATA_SKY).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low_bonferroni_month:Q", title=ytitle), y2="high_bonferroni_month:Q"
        )
        day_bar = base.mark_rule(strokeWidth=4, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            y="low_bonferroni_day:Q", y2="high_bonferroni_day:Q"
        )
        dot = base.mark_point(filled=True, size=70, color=theme.BLACK_2).encode(y="difference:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero_rule(), month_bar, day_bar, dot).properties(
                width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(title=alt.TitleParams([title, *caption], anchor="start")).save(
        DOCS_DIR / path
    )


def figure_pretraining() -> None:
    """Draw L0PT minus LL by lead band."""
    t = pl.read_csv(SCORES / "edit8_contrasts.csv").filter(
        (pl.col("metric") == "crps")
        & (pl.col("band") != "all")
        & (pl.col("contrast") == "L0PT - L0")
    )
    interval_panels(
        table=t,
        title="ERA5 pretraining lowers the lag-free network's CRPS in every lead band (below zero = pretrained is better).",
        ytitle="L0PT minus L0 CRPS (points)",
        caption=[
            "Dot: difference. Thick bar: Bonferroni interval, origin-day blocks. Thin bar: same level, whole-month blocks (4)."
        ],
        path="figure_1_pretrained_minus_unpretrained.svg",
    )


def figure_per_band() -> None:
    """Draw per-band widening minus flat 1.15."""
    t = pl.read_csv(SCORES / "part2_contrasts.csv").filter(
        (pl.col("contrast") == "per-band - flat1.15")
        & (pl.col("metric") == "crps")
        & (pl.col("band") != "all")
    )
    interval_panels(
        table=t,
        title="Per-band widening fitted on earlier held-out forecasts changes CRPS by under 0.03 points in every band.",
        ytitle="Per-band minus flat 1.15 CRPS (points)",
        caption=[
            "Dot: difference. Thick bar: Bonferroni interval, origin-day blocks. Thin bar: whole-month blocks (4)."
        ],
        path="figure_4_per_band_widening.svg",
    )


def figure_width_growth() -> None:
    """Draw width growth 120-240 h over 12-24 h, pooled and within one half-hour of the day."""
    t = pl.read_csv(SCORES / "width_ratios.csv").filter(
        (pl.col("ratio") == "120-240h over 12-24h") & (pl.col("arm") != "LL (L0PT folds)")
    )
    long = pl.concat(
        [
            t.select("set", "arm", kind=pl.lit("pooled over all hours"), ratio=pl.col("pooled")),
            t.select(
                "set",
                "arm",
                kind=pl.lit("same half-hour of the day"),
                ratio=pl.col("hour_stratified"),
            ),
        ]
    ).with_columns(growth=(pl.col("ratio") - 1.0) * 100.0)
    panels = []
    for key, label in SETS.items():
        part = long.filter(pl.col("set") == key).to_pandas()
        panels.append(
            alt.Chart(part)
            .mark_bar()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("arm:N", sort=list(ARM_COLOURS), title=None),
                xOffset="kind:N",
                y=alt.Y("growth:Q", title="Width growth (%)"),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(range=[theme.DATA_SKY, theme.DATA_BLUE]),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
            .properties(width=WIDTH, height=170, title=alt.TitleParams(label, anchor="start"))
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Most of the apparent widening with lead is time of day.",
                "Growth of the 90% band from 12-24 h to 120-240 h.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_2_width_growth.svg")


def figure_lead_swap() -> None:
    """Draw the lead-swap changes by true lead band."""
    t = (
        pl.read_csv(SCORES / "lead_swap.csv")
        .filter((pl.col("condition") != "true") & (pl.col("band") != "all"))
        .with_columns(width_change=(pl.col("w90_ratio") - 1.0) * 100.0)
    )
    panels = []
    for column, title in (
        ("width_change", "Change in mean 90% width (%)"),
        ("median_shift_mean_abs", "Mean absolute change in the median (points of capacity)"),
    ):
        panels.append(
            alt.Chart(t.to_pandas())
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("band:N", sort=BAND_ORDER, title="True lead time of the row"),
                y=alt.Y(f"{column}:Q", title=title),
                color=alt.Color("condition:N", legend=alt.Legend(orient="top", title=None)),
            )
            .properties(width=WIDTH, height=190)
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "L-1d (fold 2026-01, seed 0) barely changes its band when its lead input is replaced.",
                "Day 1, day 10, and a shuffled lead change the clock; the two same-hour conditions change only the lead.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_3_lead_swap.svg")


def figure_pit() -> None:
    """Draw PIT histograms at 120 to 240 h."""
    t = pl.read_csv(SCORES / "pit.csv").filter(pl.col("band") == "120-240h").to_pandas()
    panels = []
    for key, label in SETS.items():
        part = t[t["set"] == key]
        bars = (
            alt.Chart(part)
            .mark_bar()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("bin:O", title="PIT bin (tenths)"),
                y=alt.Y("share:Q", title="Share of rows"),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                    legend=alt.Legend(orient="top", title=None),
                ),
                xOffset="arm:N",
            )
        )
        ideal = (
            alt.Chart(pl.DataFrame({"y": [0.1]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color=theme.BLACK_2, strokeDash=[3, 2])
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(bars, ideal).properties(
                width=WIDTH, height=170, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "PIT histogram at 120 to 240 h, raw quantiles (dashed: perfectly calibrated).",
                "The end bins hold the tails; the 11 stored quantiles are interpolated linearly.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_5_pit.svg")


EXAMPLE_DAYS: Final[dict[str, tuple[str, str]]] = {
    "winter": ("2026-01-12", "2026-01"),
    "summer": ("2026-06-16", "2026-06"),
}
EXAMPLE_PRIMARY: Final[int] = 2
EXAMPLE_ARMS: Final[dict[str, str]] = {"L-1d": "L1d", "K2-S1": "K2c0", "L0PT": "PTs0"}


def figure_examples() -> None:
    """Draw a day-ahead and a 10-day forecast of D4 for L-1d, K2-S1, and L0PT."""
    for season, (day_text, month) in EXAMPLE_DAYS.items():
        table = pl.read_parquet(panel.panel_path(month=month))
        day = datetime.fromisoformat(day_text).replace(hour=9, tzinfo=UTC)
        origin = int((day.timestamp() * 1e6 - refs.GRID_START_US) // refs.STEP_US)
        part = table.filter((pl.col("primary") == EXAMPLE_PRIMARY) & (pl.col("origin") == origin))
        frames = []
        for arm, cell in EXAMPLE_ARMS.items():
            q = sc.widen(q=sc.quantiles(table=part, cell=cell), factor=sc.FLAT)
            frames.append(
                pl.DataFrame(
                    {"hours": part["lead_hours"], "actual": part["actual"] * 100.0, "arm": arm,
                     "q05": q[:, 1] * 100.0, "q50": q[:, 5] * 100.0, "q95": q[:, 9] * 100.0}
                )
            )  # fmt: skip
        t = pl.concat(frames)
        for horizon, limit, tag in (("day-ahead", 36.0, "day_ahead"), ("10-day", 240.0, "10_days")):
            data = t.filter(pl.col("hours") <= limit).to_pandas()
            scale = alt.Scale(
                domain=["L-1d", "K2-S1", "L0PT"],
                range=[ARM_COLOURS[a] for a in ("L-1d", "K2-S1", "L0PT")],
            )
            base = alt.Chart(data).encode(
                x=alt.X("hours:Q", title="Hours after the forecast origin")
            )  # ty: ignore[unresolved-attribute]
            band = base.mark_area(opacity=0.2).encode(  # ty: ignore[unresolved-attribute]
                y="q05:Q",
                y2="q95:Q",
                color=alt.Color("arm:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
            )
            med = base.mark_line().encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("q50:Q", title="Power (% of capacity)"),
                color=alt.Color("arm:N", scale=scale),
            )
            actual = base.mark_line(color=theme.BRAND_ORANGE, strokeDash=[3, 2]).encode(  # ty: ignore[unresolved-attribute]
                y="actual:Q"
            )
            alt.layer(band, med, actual).properties(
                width=WIDTH,
                height=220,
                title=alt.TitleParams(
                    [
                        f"{horizon.capitalize()} forecast, D4 ({season}, origin {day_text} 09:00 UTC).",
                        "Lines: medians; shading: 90% band (1.15 widening); dashed: actual.",
                    ],
                    anchor="start",
                ),
            ).save(DOCS_DIR / f"figure_6_example_{season}_{tag}.svg")


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_pretraining()
    figure_width_growth()
    figure_lead_swap()
    figure_per_band()
    figure_pit()
    figure_examples()


if __name__ == "__main__":
    main()

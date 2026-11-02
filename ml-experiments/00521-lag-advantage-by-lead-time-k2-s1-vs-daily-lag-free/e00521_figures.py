# ruff: noqa: E501
"""Draw the figures of experiment 521 from the tables `e00521_stats.py` stored.

Figures (SVG in the experiment's docs folder; optimise each with `npx svgo@4 --multipass
--precision=1 --final-newline`):

1. `figure_1_gap_by_lead.svg`: L-1d minus K2-S1 CRPS by lead band, Bonferroni intervals;
2. `figure_2_crps_by_lead.svg`: CRPS by lead band of K2-S1, L-1d, L-Ww, and the monthly lag-free arms;
3. `figure_3_references.svg`: each lag-free arm's CRPS minus K2-S1's, by lead band;
4. `figure_4_calibration.svg`: q95 and q99 coverage by lead band for K2-S1 and L-1d;
5. `figure_5_per_primary.svg`: the gap per Primary, at 120 to 240 h and pooled;
6. `figure_6_half_hour.svg`: the gap by UTC half-hour of the valid time;
7. `figure_7_slices.svg`: the pooled gap on slices.
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from e00521_refs import DATA, GRID_START_US, STEP_US, WIDENING
from e00521_stats import BANDS
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free"
)
SCORES: Final[Path] = DATA / "scores"
WIDTH: Final[int] = 560
D4_CAPACITY: Final[float] = 12.138
"""Capacity of D4 in MW (the site map of experiment 470)."""
BAND_ORDER: Final[list[str]] = list(BANDS)
SETS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}
COLOURS: Final[dict[str, str]] = {
    "K2-S1": theme.BLACK_2,
    "K2-S1-3cell": "#808080",
    "L-1d": theme.DATA_BLUE,
    "L-Ww": theme.DATA_SKY,
    "LL": theme.BRAND_ORANGE_LIGHT,
    "LL-ens3": theme.DATA_AMBER,
    "LL-ens5": theme.BRAND_ORANGE,
}
LABELS: Final[dict[str, str]] = {
    "K2-S1": "K2-S1 (cell c0)",
    "K2-S1-3cell": "K2-S1 (mean of 3 cells)",
    "L-1d": "L-1d (daily)",
    "L-Ww": "L-Ww (weekly)",
    "LL": "LL (monthly, 1 head)",
    "LL-ens3": "LL-ens3 (monthly, 3 heads)",
    "LL-ens5": "LL-ens5 (monthly, 5 heads)",
}


def _scale(*, arms: list[str]) -> alt.Scale:
    return alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOURS[a] for a in arms])


def _zero() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color=theme.BLACK_2)
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]


def figure_gap() -> None:
    """Draw the gap by lead band with the origin-day Bonferroni bar and the month-block bar."""
    t = pl.read_csv(SCORES / "contrasts.csv").filter(
        (pl.col("contrast") == "L-1d - K2-S1")
        & (pl.col("metric") == "crps")
        & pl.col("set").is_in(list(SETS))
        & (pl.col("band") != "all")
    )
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        base = alt.Chart(part).encode(
            x=alt.X("band:N", sort=BAND_ORDER, title="Lead time of the forecast")
        )  # ty: ignore[unresolved-attribute]
        month_bar = base.mark_rule(strokeWidth=1.5, color=theme.DATA_SKY).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low_bonferroni_month:Q", title="L-1d minus K2-S1 CRPS (points)"),
            y2="high_bonferroni_month:Q",
        )  # ty: ignore[unresolved-attribute]
        day_bar = base.mark_rule(strokeWidth=4, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            y="low_bonferroni_day:Q", y2="high_bonferroni_day:Q"
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=70, color=theme.BLACK_2).encode(y="difference:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(_zero(), month_bar, day_bar, dot).properties(
                width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Where K2-S1 beats the daily-refit lag-free network L-1d (above zero) and where it does not.",
                "Dot: gap. Thick bar: Bonferroni interval, origin-day blocks. Thin bar: same level, whole-month blocks (5).",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_1_gap_by_lead.svg")


def figure_crps() -> None:
    """Draw each arm's CRPS by lead band."""
    t = pl.read_csv(SCORES / "absolute.csv").filter(
        pl.col("set").is_in(list(SETS)) & (pl.col("band") != "all")
    )
    arms = ["K2-S1", "K2-S1-3cell", "L-1d", "L-Ww", "LL", "LL-ens3", "LL-ens5"]
    t = t.filter(pl.col("arm").is_in(arms)).with_columns(label=pl.col("arm").replace(LABELS))
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        panels.append(
            alt.Chart(part).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("band:N", sort=BAND_ORDER, title="Lead time of the forecast"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color("label:N", scale=_scale(arms=arms), legend=alt.Legend(orient="top", title=None, columns=3)),
            ).properties(width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start"))
        )  # fmt: skip
    alt.vconcat(*panels).properties(
        title=alt.TitleParams("CRPS by lead time, same rows for every arm.", anchor="start")
    ).save(DOCS_DIR / "figure_2_crps_by_lead.svg")


def figure_references() -> None:
    """Draw each lag-free arm's CRPS minus K2-S1's by lead band."""
    t = pl.read_csv(SCORES / "contrasts.csv").filter(
        pl.col("set").is_in(list(SETS))
        & (pl.col("metric") == "crps")
        & (pl.col("band") != "all")
        & pl.col("contrast").is_in(
            [
                "L-1d - K2-S1",
                "L-Ww - K2-S1",
                "L-1d - K2-S1-3cell",
                "L-1d - LL-ens5",
                "K2-S1 - LL-ens5",
            ]
        )
    )
    # Monthly arms minus K2-S1: LL-ens5 - K2-S1 is the negative of "K2-S1 - LL-ens5"'s sign flipped.
    flipped = t.filter(pl.col("contrast") == "K2-S1 - LL-ens5").with_columns(
        difference=-pl.col("difference"), contrast=pl.lit("LL-ens5 - K2-S1")
    )
    t = pl.concat(
        [t.filter(pl.col("contrast") != "K2-S1 - LL-ens5"), flipped], how="diagonal_relaxed"
    )
    t = t.filter(pl.col("contrast").is_in(["L-1d - K2-S1", "L-Ww - K2-S1", "LL-ens5 - K2-S1"]))
    names = {
        "L-1d - K2-S1": "L-1d (daily)",
        "L-Ww - K2-S1": "L-Ww (weekly)",
        "LL-ens5 - K2-S1": "LL-ens5 (monthly, 5 heads)",
    }
    colours = [theme.DATA_BLUE, theme.DATA_SKY, theme.BRAND_ORANGE]
    t = t.with_columns(arm=pl.col("contrast").replace(names))
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        panels.append(
            alt.layer(
                _zero(),
                alt.Chart(part).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("band:N", sort=BAND_ORDER, title="Lead time of the forecast"),
                    y=alt.Y("difference:Q", title="CRPS minus K2-S1 (points)"),
                    color=alt.Color("arm:N", scale=alt.Scale(domain=list(names.values()), range=colours), legend=alt.Legend(orient="top", title=None, columns=3)),
                ),
            ).properties(width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start"))
        )  # fmt: skip
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Each lag-free arm's CRPS minus K2-S1's (above zero: K2-S1 better).",
                "Intervals are in the tables; this figure shows the pattern.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_3_references.svg")


def figure_calibration() -> None:
    """Draw q95 and q99 coverage by lead band for K2-S1 and L-1d."""
    t = pl.read_csv(SCORES / "absolute.csv").filter(
        pl.col("set").is_in(list(SETS))
        & (pl.col("band") != "all")
        & pl.col("arm").is_in(["K2-S1", "L-1d"])
    )
    long = t.unpivot(
        index=["set", "band", "arm"], on=["cov_q01", "cov_q05", "cov_q95", "cov_q99"]
    ).with_columns(
        level=pl.col("variable").str.replace("cov_", ""), arm=pl.col("arm").replace(LABELS)
    )
    nominal = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
    panels = []
    arms = ["K2-S1", "L-1d"]
    for key, label in SETS.items():
        for level in ("q05", "q95", "q99"):
            part = long.filter((pl.col("set") == key) & (pl.col("level") == level)).to_pandas()
            line = alt.Chart(part).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("band:N", sort=BAND_ORDER, title=None),
                y=alt.Y("value:Q", title=f"{level} coverage", scale=alt.Scale(zero=False)),
                color=alt.Color("arm:N", scale=_scale(arms=arms), legend=alt.Legend(orient="top", title=None, columns=2)),
            )  # fmt: skip
            rule = (
                alt.Chart(pl.DataFrame({"y": [nominal[level]]}).to_pandas())  # ty: ignore[unresolved-attribute]
                .mark_rule(strokeDash=[4, 3], color=theme.BLACK_1)
                .encode(y="y:Q")
            )  # ty: ignore[unresolved-attribute]
            panels.append(
                alt.layer(rule, line).properties(
                    width=WIDTH // 2,
                    height=130,
                    title=alt.TitleParams(f"{label}, {level}", anchor="start"),
                )
            )
    rows = [alt.hconcat(*panels[i : i + 3]) for i in range(0, len(panels), 3)]
    alt.vconcat(*rows).properties(
        title=alt.TitleParams(
            [
                "Share of actuals at or below each quantile, by lead time; dashed line: nominal level."
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_4_calibration.svg")


def figure_per_primary() -> None:
    """Draw the gap per Primary at 120 to 240 h and pooled."""
    t = pl.read_csv(SCORES / "per_primary.csv").filter(
        pl.col("set").is_in(list(SETS)) & pl.col("band").is_in(["120-240h", "all"])
    )
    panels = []
    for key, label in SETS.items():
        part = (
            t.filter(pl.col("set") == key)
            .with_columns(
                band=pl.col("band").replace({"all": "All leads", "120-240h": "120 to 240 h"})
            )
            .to_pandas()
        )
        panels.append(
            alt.layer(
                alt.Chart(part).mark_bar().encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("primary:N", title=None, sort=alt.EncodingSortField(field="gap", op="mean", order="descending")),
                    y=alt.Y("gap:Q", title="L-1d minus K2-S1 CRPS (points)"),
                    xOffset="band:N",
                    color=alt.Color("band:N", scale=alt.Scale(domain=["All leads", "120 to 240 h"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE]), legend=alt.Legend(orient="top", title=None, columns=2)),
                ),
                _zero(),
            ).properties(width=WIDTH, height=170, title=alt.TitleParams(label, anchor="start"))
        )  # fmt: skip
    alt.vconcat(*panels).properties(
        title=alt.TitleParams("The gap per Primary (above zero: K2-S1 better).", anchor="start")
    ).save(DOCS_DIR / "figure_5_per_primary.svg")


def figure_half_hour() -> None:
    """Draw the gap by UTC half-hour of the valid time, for three lead bands."""
    t = pl.read_csv(SCORES / "half_hour.csv")
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        panels.append(
            alt.layer(
                alt.Chart(part).mark_line().encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("half_hour:Q", title="Half-hour of the valid time (UTC; 0 = 00:00)"),
                    y=alt.Y("d:Q", title="L-1d minus K2-S1 CRPS (points)"),
                    color=alt.Color("coarse:N", sort=["0-24h", "24-120h", "120-240h"], scale=alt.Scale(domain=["0-24h", "24-120h", "120-240h"], range=[theme.DATA_SKY, theme.DATA_BLUE, theme.BRAND_ORANGE]), legend=alt.Legend(orient="top", title="Lead time", columns=3)),
                ),
                _zero(),
            ).properties(width=WIDTH, height=170, title=alt.TitleParams(label, anchor="start"))
        )  # fmt: skip
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "The gap by half-hour of day (above zero: K2-S1 better).", anchor="start"
        )
    ).save(DOCS_DIR / "figure_6_half_hour.svg")


def figure_slices() -> None:
    """Draw the pooled gap on each slice, with its origin-day 95% interval (no correction)."""
    t = (
        pl.read_csv(SCORES / "slices.csv")
        .filter(
            (pl.col("band") == "all")
            & (pl.col("contrast") == "L-1d - K2-S1")
            & pl.col("set").is_in(["eight", "six_all"])
        )
        .with_columns(
            set=pl.col("set").replace(
                {"eight": SETS["eight"], "six_all": "Signed-MW six (all rows)"}
            )
        )
    )
    panels = []
    order = list(dict.fromkeys(t["slice"].to_list()))
    for label in ("MVA-only eight", "Signed-MW six (all rows)"):
        part = t.filter(pl.col("set") == label).to_pandas()
        base = alt.Chart(part).encode(y=alt.Y("slice:N", sort=order, title=None))  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(
                base.mark_rule(strokeWidth=3, color=theme.DATA_BLUE).encode(x=alt.X("low_day:Q", title="L-1d minus K2-S1 CRPS (points)"), x2="high_day:Q"),  # ty: ignore[unresolved-attribute]
                base.mark_point(filled=True, size=60, color=theme.BLACK_2).encode(x="difference:Q"),  # ty: ignore[unresolved-attribute]
                alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color=theme.BLACK_2).encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
            ).properties(width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start"))
        )  # fmt: skip
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            ["The pooled gap on slices; 95% origin-day interval, no multiplicity correction."],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_7_slices.svg")


EXAMPLES: Final[dict[str, tuple[int, str]]] = {"winter": (2, "2026-01"), "summer": (2, "2026-06")}
"""Example name -> (Primary index of D4, month of the fold)."""
EXAMPLE_DAYS: Final[dict[str, str]] = {"winter": "2026-01-12", "summer": "2026-06-16"}


def figure_examples() -> None:
    """Draw a day-ahead and a 10-day forecast of K2-S1 (cell c0) and L-1d against the actuals, D4."""
    from datetime import UTC, datetime

    e454 = (
        ML_EXPERIMENTS_DATA_DIR
        / "00454-lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads/work/predictions"
    )
    e511 = (
        ML_EXPERIMENTS_DATA_DIR
        / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/work/predictions"
    )
    for season, (primary, month) in EXAMPLES.items():
        day = datetime.fromisoformat(EXAMPLE_DAYS[season]).replace(hour=9, tzinfo=UTC)
        origin = int((day.timestamp() * 1e6 - GRID_START_US) // STEP_US)
        keys = ["primary", "origin", "target"]
        pick = (pl.col("primary") == primary) & (pl.col("origin") == origin)
        k2 = pl.read_parquet(e511 / "K2-S1" / f"{month}_c0.parquet").filter(pick)
        ld = pl.read_parquet(e511 / "L-1d" / f"{month}_c0.parquet").filter(pick)
        act = (
            pl.read_parquet(e454 / "K2-S1" / f"{month}_s0_w-1.parquet")
            .filter(pick)
            .select(*keys, "actual", "lead_hours")
        )
        frames = []
        for name, frame in (("K2-S1", k2), ("L-1d", ld)):
            widened = frame.with_columns(
                *[
                    (pl.col("q50") + WIDENING * (pl.col(q) - pl.col("q50"))).alias(q)
                    for q in ("q05", "q25", "q75", "q95")
                ]
            )
            frames.append(widened.join(act, on=keys).with_columns(arm=pl.lit(name)))
        t = pl.concat(frames, how="vertical_relaxed").with_columns(
            hours=pl.col("lead_hours"), actual=pl.col("actual") / D4_CAPACITY * 100.0
        )
        for col in ("q05", "q25", "q50", "q75", "q95"):
            t = t.with_columns((pl.col(col) / D4_CAPACITY * 100.0).alias(col))
        for horizon, limit, tag in (("day-ahead", 36.0, "day_ahead"), ("10-day", 240.0, "10_days")):
            part = t.filter(pl.col("hours") <= limit).to_pandas()
            base = alt.Chart(part).encode(
                x=alt.X("hours:Q", title="Hours after the forecast origin")
            )  # ty: ignore[unresolved-attribute]
            band = base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
                y="q05:Q",
                y2="q95:Q",
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(
                        domain=["K2-S1", "L-1d"], range=[theme.BLACK_2, theme.DATA_BLUE]
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )  # ty: ignore[unresolved-attribute]
            med = base.mark_line().encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("q50:Q", title="Power (% of capacity)"), color="arm:N"
            )  # ty: ignore[unresolved-attribute]
            actual = base.mark_line(color=theme.BRAND_ORANGE, strokeDash=[3, 2]).encode(  # ty: ignore[unresolved-attribute]
                y="actual:Q"
            )  # ty: ignore[unresolved-attribute]
            alt.layer(band, med, actual).properties(
                width=WIDTH,
                height=220,
                title=alt.TitleParams(
                    [
                        f"{horizon.capitalize()} forecast, D4 ({season}, origin {EXAMPLE_DAYS[season]} 09:00 UTC).",
                        "Lines: medians; shading: 90% band; dashed orange: actual.",
                    ],
                    anchor="start",
                ),
            ).save(DOCS_DIR / f"figure_8_example_{season}_{tag}.svg")


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_gap()
    figure_crps()
    figure_references()
    figure_calibration()
    figure_per_primary()
    figure_half_hour()
    figure_slices()
    figure_examples()


if __name__ == "__main__":
    main()

# ruff: noqa: E501
r"""Draw the figures of experiment 481 from the tables `e00481_score.py` stores.

Run `e00481_score.py demand`, `wind`, and `s1` first: they write `work/figure_tables/` in the
experiment's data folder. Figures (680 pixels wide, SVG, in this folder; optimise each with
`npx svgo@4 --multipass --precision=1 --final-newline` before committing):

1. `figure_1_score_by_k.svg`: score against the number of seeds averaged (K), one panel per family,
   with the 5th to 95th percentile band across the random subsets of seeds;
2. `figure_2_marginal_gain.svg`: the gain from the next K with its Bonferroni interval, against the
   minimum detectable effect;
3. `figure_3_widening.svg`: score change from widening the tails, against K;
4. `figure_4_score_by_networks.svg`: S1 score against networks served, beside experiment 454's
   stored constructs;
5. `figure_5_example_*.svg`: a day-ahead and a 14-day forecast of one Primary, K = 1 against K = 10.

```bash
DATA_PATH_INTERNAL=... uv run --no-sync python docs/ml-experiments/00481-*/e00481_figures.py
```
"""

import json
import os
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

DATA_DIR: Final[Path] = (
    Path(os.environ["DATA_PATH_INTERNAL"])
    / "ml-experiments"
    / "00481-seed-averaging-diminishing-returns-for-demand-and-wind"
)
TABLE_DIR: Final[Path] = DATA_DIR / "work" / "figure_tables"
PART_DIR: Final[Path] = DATA_DIR / "work" / "report_parts"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parent
WIDTH: Final[int] = 600
FAMILIES: Final[dict[str, str]] = {
    "demand": "Primary demand (arm A, signed-MW six)",
    "wind": "Wind farms (member_plain)",
    "s1": "Lagged demand (S1, signed-MW six)",
}
PRIMARY: Final[dict[str, tuple[str, str, float]]] = {
    "demand": ("signed-MW six", "days 1-10", 1.15),
    "wind": ("all farms", "true days 1-10", 1.0),
    "s1": ("signed_six", "clean", 1.0),
}
WIDENINGS: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
WIDENING_COLOURS: Final[list[str]] = [
    "#1A1A1A",
    theme.DATA_BLUE,
    theme.DATA_GREEN,
    theme.BRAND_ORANGE,
]
FAMILY_COLOURS: Final[list[str]] = [theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_GREEN]


def _title(*lines: str) -> alt.TitleParams:
    """Return a left-anchored title of the given lines."""
    return alt.TitleParams(list(lines), anchor="start")


def _primary_curve(*, family: str) -> pl.DataFrame:
    """Return the family's CRPS curve at its primary row set and widening."""
    group, mask, widening = PRIMARY[family]
    return pl.read_parquet(TABLE_DIR / f"{family}_curve.parquet").filter(
        (pl.col("group") == group)
        & (pl.col("mask") == mask)
        & (pl.col("widening") == widening)
        & (pl.col("metric") == "crps")
    )


def figure_score_by_k() -> None:
    """Draw score against K with the band across random subsets, one panel per family."""
    panels = []
    for family, name in FAMILIES.items():
        curve = _primary_curve(family=family).to_pandas()  # ty: ignore[unresolved-attribute]
        base = alt.Chart(curve).encode(
            x=alt.X(
                "k:Q",
                title="Seeds averaged (K)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(values=sorted(curve["k"].unique().tolist())),
            )
        )  # ty: ignore[unresolved-attribute]
        band = base.mark_area(opacity=0.25, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "p05:Q", title="Score (points; smaller is better)", scale=alt.Scale(zero=False)
            ),
            y2="p95:Q",
        )  # ty: ignore[unresolved-attribute]
        line = base.mark_line(color=theme.BRAND_ORANGE, strokeWidth=2, point=True).encode(  # ty: ignore[unresolved-attribute]
            y="mean:Q"
        )  # ty: ignore[unresolved-attribute]
        panels.append((band + line).properties(width=WIDTH, height=150, title=_title(name)))
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(y="independent")
        .properties(
            title=_title(
                "Score against the number of seeds averaged. Line: mean over random subsets of K seeds;",
                "band: 5th to 95th percentile across subsets (all subsets when fewer than 100 exist).",
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_score_by_k.svg")


def figure_marginal_gain() -> None:
    """Draw the gain from the next K with its Bonferroni interval and the minimum detectable effect."""
    panels = []
    for family, name in FAMILIES.items():
        gains = pl.read_parquet(TABLE_DIR / f"{family}_gain.parquet").with_columns(
            step=pl.format("K {} to {}", pl.col("k"), pl.col("k_next"))
        )
        mde = json.loads((PART_DIR / f"{family}.json").read_text())["mde"]
        order = gains["step"].to_list()
        frame = gains.to_pandas()  # ty: ignore[unresolved-attribute]
        base = alt.Chart(frame).encode(y=alt.Y("step:N", sort=order, title=None))  # ty: ignore[unresolved-attribute]
        bars = base.mark_rule(strokeWidth=3, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Gain from the next K (points)"), x2="high:Q"
        )  # ty: ignore[unresolved-attribute]
        points = base.mark_point(color=theme.BRAND_ORANGE, filled=True, size=70).encode(x="gain:Q")  # ty: ignore[unresolved-attribute]
        limits = (
            alt.Chart(pl.DataFrame({"mde": [mde, -mde, 0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#1A1A1A")
            .encode(x="mde:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            (limits + bars + points).properties(
                width=WIDTH,
                height=110,
                title=_title(f"{name}; dashed: 0 and plus or minus the MDE ({mde:.3f})"),
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            "Marginal gain from averaging more seeds, with the month-block bootstrap interval at the",
            "Bonferroni level (99.55%, 11 contrasts). Positive means the larger K scores better.",
        )
    )
    chart.save(DOCS_DIR / "figure_2_marginal_gain.svg")


def figure_widening() -> None:
    """Draw the score change from widening the tails, per K and family."""
    pieces = []
    for family, name in FAMILIES.items():
        group, mask, _ = PRIMARY[family]
        curve = pl.read_parquet(TABLE_DIR / f"{family}_curve.parquet").filter(
            (pl.col("group") == group) & (pl.col("mask") == mask) & (pl.col("metric") == "crps")
        )
        base = curve.filter(pl.col("widening") == 1.0).select("k", unwidened=pl.col("mean"))
        pieces.append(
            curve.join(base, on="k").with_columns(
                change=pl.col("mean") - pl.col("unwidened"),
                family=pl.lit(name),
                widening_label=pl.col("widening").map_elements(
                    lambda w: f"{w:.2f}", return_dtype=pl.String
                ),
            )
        )
    frame = pl.concat(pieces).to_pandas()  # ty: ignore[unresolved-attribute]
    labels = [f"{w:.2f}" for w in WIDENINGS]
    chart = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "k:Q",
                title="Seeds averaged (K)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(values=[1, 2, 3, 5, 6, 10]),
            ),
            y=alt.Y("change:Q", title="Score change (points)"),
            color=alt.Color(
                "widening_label:N",
                scale=alt.Scale(domain=labels, range=WIDENING_COLOURS),
                legend=alt.Legend(orient="top", title="Tail widening factor"),
            ),
            row=alt.Row("family:N", title=None, sort=list(FAMILIES.values())),
        )
        .resolve_scale(y="independent")
        .properties(
            width=WIDTH,
            height=130,
            title=_title(
                "Effect of widening the four tail quantiles about the median, against K. Below zero is better;",
                "the lowest line at each K is the best widening.",
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_3_widening.svg")


def figure_score_by_networks() -> None:
    """Draw S1's score against networks served, beside experiment 454's stored constructs."""
    table = pl.read_parquet(TABLE_DIR / "s1_comparison.parquet").filter(pl.col("variant") == "raw")
    stored = table.filter(pl.col("source") == "experiment 454 stored").to_pandas()  # ty: ignore[unresolved-attribute]
    ours = table.filter(pl.col("source") != "experiment 454 stored").to_pandas()  # ty: ignore[unresolved-attribute]
    x = alt.X(
        "networks:Q",
        title="Networks served per forecast (a lagless table counts as one)",
        scale=alt.Scale(domain=[0, 11]),
    )
    y = alt.Y("crps:Q", title="Score (points; smaller is better)", scale=alt.Scale(zero=False))
    lines = (
        alt.Chart(ours)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "source:N",
                scale=alt.Scale(range=[theme.BRAND_ORANGE, theme.DATA_BLUE]),
                legend=alt.Legend(orient="top", title=None, labelLimit=400, columns=1),
            ),
        )  # ty: ignore[unresolved-attribute]
    )
    points = alt.Chart(stored).mark_point(filled=True, size=70, color="#1A1A1A").encode(x=x, y=y)  # ty: ignore[unresolved-attribute]
    labels = (
        alt.Chart(stored)  # ty: ignore[unresolved-attribute]
        .mark_text(dx=8, dy=-8, align="left", color="#1A1A1A")
        .encode(x=x, y=y, text="arm:N")
    )  # ty: ignore[unresolved-attribute]
    chart = (lines + points + labels).properties(
        width=WIDTH,
        height=260,
        title=_title(
            "Lagged demand score against networks served. Black: constructs stored by experiment 454",
            "(signed-MW six, rows with no labelled event). Coloured: S1 seeds averaged here.",
        ),
    )
    chart.save(DOCS_DIR / "figure_4_score_by_networks.svg")


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of one Primary, K = 1 against K = 10, per season."""
    frame = pl.read_parquet(TABLE_DIR / "demand_examples.parquet").with_columns(
        day=pl.col("lead_hours") / 24.0
    )
    seasons = {"2026-01-12": "winter", "2026-06-15": "summer"}
    for origin, season in seasons.items():
        for horizon, window in (("day_ahead", (1.0, 2.0)), ("14_days", (0.0, 14.0))):
            sub = frame.filter(
                (pl.col("origin") == origin) & pl.col("day").is_between(*window)
            ).to_pandas()  # ty: ignore[unresolved-attribute]
            panels = []
            for k, panel_title in (
                ("K=1", "D10, one seed (K=1)"),
                ("K=10", "D10, ten seeds averaged (K=10)"),
            ):
                part = sub[sub["k"] == k]
                base = alt.Chart(part).encode(
                    x=alt.X(
                        "day:Q",
                        title="Days after the forecast was made",
                        scale=alt.Scale(domain=list(window), nice=False),
                    )
                )  # ty: ignore[unresolved-attribute]
                wide = base.mark_area(opacity=0.2, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                )  # ty: ignore[unresolved-attribute]
                narrow = base.mark_area(opacity=0.35, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
                    y="q25:Q", y2="q75:Q"
                )  # ty: ignore[unresolved-attribute]
                median = base.mark_line(color=theme.BRAND_ORANGE, strokeWidth=2).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
                actual = base.mark_line(color="black", strokeWidth=1.5).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
                panels.append(
                    (wide + narrow + median + actual).properties(
                        width=WIDTH, height=170, title=_title(panel_title)
                    )
                )
            chart = alt.vconcat(*panels).properties(
                title=_title(
                    f"Forecast of D10 made in {season} ({'day ahead' if horizon == 'day_ahead' else '14 days'}), widened by 1.15.",
                    "Orange line: median. Bands: 50% and 90%. Black line: what happened.",
                )
            )
            chart.save(DOCS_DIR / f"figure_5_example_D10_{season}_{horizon}.svg")


def main() -> None:
    """Draw every figure whose tables exist."""
    steps = {
        "score_by_k": (figure_score_by_k, ("demand", "wind", "s1")),
        "marginal_gain": (figure_marginal_gain, ("demand", "wind", "s1")),
        "widening": (figure_widening, ("demand", "wind", "s1")),
        "score_by_networks": (figure_score_by_networks, ("s1_comparison",)),
        "examples": (figure_examples, ("demand_examples",)),
    }
    for name, (step, needs) in steps.items():
        missing = [n for n in needs if not any(TABLE_DIR.glob(f"{n}*.parquet"))]
        if missing:
            print(f"{name} skipped: no tables for {missing}")
            continue
        step()
        print(f"{name} drawn")


if __name__ == "__main__":
    main()

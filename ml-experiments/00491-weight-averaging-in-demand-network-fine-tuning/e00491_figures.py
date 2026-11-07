# ruff: noqa: E501
r"""Draw the figures of experiment 491 from the tables `e00491_score.py` stores.

Run `e00491_score.py build`, `rows`, `stats`, and `widening` first. Figures (SVG, in this folder;
optimise each with `npx svgo@4 --multipass --precision=1 --final-newline` before committing):

1. `figure_1_score_by_arm.svg`: CRPS of each arm at the eight and the six;
2. `figure_2_closure_share.svg`: share of the gap from S0f to Pool5 that each arm closes;
3. `figure_3_contrasts.svg`: counted contrasts against S0f with Bonferroni intervals;
4. `figure_4_example_*.svg`: a day-ahead and a 14-day forecast of one Primary.

```bash
source /tmp/e491env.sh
uv run --no-sync python docs/ml-experiments/00491-*/e00491_figures.py
```
"""

import os
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

DATA_DIR: Final[Path] = (
    Path(os.environ["DATA_PATH_INTERNAL"])
    / "ml-experiments"
    / "00491-weight-averaging-in-demand-network-fine-tuning"
    / "work"
)
SCORES: Final[Path] = DATA_DIR / "scores"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parent
WIDTH: Final[int] = 600
SLICE_NAMES: Final[dict[str, str]] = {
    "eight": "Eight MVA-only Primaries",
    "six_clean": "Six, no labelled event",
    "six_nra": "Six, real events",
    "cold_eight": "Eight, cold winter nights",
    "cold_six": "Six, cold winter nights",
    "band_eight": "Eight, 12 to 36 hours",
    "band_six": "Six, 12 to 36 hours",
}
ARM_ORDER: Final[list[str]] = [
    "S0", "S0f", "EMA99", "EMA", "SWA", "Pool2", "Pool3", "PoolEMA3", "PoolSWA3", "Pool5",
]  # fmt: skip
EXAMPLE_PRIMARY: Final[int] = 6
"""Primary index of D10 in experiment 470's grid."""
EXAMPLE_ORIGIN: Final[str] = "2026-01-12"


def _title(*lines: str) -> alt.TitleParams:
    """Return a left-anchored title of the given lines."""
    return alt.TitleParams(list(lines), anchor="start")


def figure_score_by_arm() -> None:
    """Draw the CRPS of each arm, one panel for the eight and one for the six."""
    absolute = pl.read_csv(SCORES / "absolute.csv").filter(
        pl.col("slice").is_in(["eight", "six_clean"]) & pl.col("arm").is_in(ARM_ORDER)
    )
    panels = []
    for name in ("eight", "six_clean"):
        part = absolute.filter(pl.col("slice") == name).to_pandas()  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.Chart(part)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=70, color=theme.BRAND_ORANGE)
            .encode(
                x=alt.X("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                y=alt.Y("arm:N", sort=ARM_ORDER, title=None),
            )
            .properties(width=WIDTH, height=190, title=_title(SLICE_NAMES[name]))
        )
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                "Score of each arm, 1.15 widening, mean of three seeds (smaller is better).",
                "Pool2, Pool3, and Pool5 average 2, 3, and 5 networks; the other arms serve one.",
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_score_by_arm.svg")


def figure_closure() -> None:
    """Draw the share of the S0f-to-Pool5 gap closed, with the block-bootstrap interval."""
    closure = pl.read_csv(SCORES / "closure.csv").filter(
        (pl.col("start") == "S0f")
        & pl.col("slice").is_in(["eight", "six_clean"])
        & pl.col("arm").is_in(["EMA99", "EMA", "SWA", "Pool2", "PoolEMA3", "PoolSWA3"])
    )
    panels = []
    for name in ("eight", "six_clean"):
        part = closure.filter(pl.col("slice") == name).to_pandas()  # ty: ignore[unresolved-attribute]
        base = alt.Chart(part).encode(y=alt.Y("arm:N", sort=ARM_ORDER, title=None))
        bars = base.mark_rule(color=theme.DATA_BLUE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Share of the gap closed", scale=alt.Scale(domain=[-0.2, 1.3])),
            x2="high:Q",
        )
        dots = base.mark_point(filled=True, size=70, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
            x="closure_share:Q"
        )
        rule = (
            alt.Chart(pl.DataFrame({"x": [0.4]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 4], color="#1A1A1A")
            .encode(x="x:Q")
        )
        panels.append(
            (bars + dots + rule).properties(
                width=WIDTH, height=170, title=_title(SLICE_NAMES[name])
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            "Share of the gap from S0f to Pool5 that each arm closes. Dot: estimate; bar: 95%",
            "month-block bootstrap interval; dashed line: the 40% threshold of the prediction.",
        )
    )
    chart.save(DOCS_DIR / "figure_2_closure_share.svg")


def figure_contrasts() -> None:
    """Draw the counted CRPS contrasts of EMA and SWA against S0f with Bonferroni intervals."""
    contrasts = (
        pl.read_csv(SCORES / "contrasts.csv")
        .filter(
            pl.col("contrast").is_in(["EMA - S0f", "SWA - S0f", "EMA - Pool5", "SWA - Pool5"])
            & (pl.col("metric") == "crps")
            & pl.col("counted")
        )
        .with_columns(pl.col("slice").replace(SLICE_NAMES).alias("slice_name"))
        .to_pandas()  # ty: ignore[unresolved-attribute]
    )
    panels = []
    for reference in ("S0f", "Pool5"):
        part = contrasts[contrasts["contrast"].str.endswith(reference)]
        base = alt.Chart(part).encode(
            y=alt.Y("slice_name:N", sort=list(SLICE_NAMES.values()), title=None),
            color=alt.Color(
                "contrast:N",
                scale=alt.Scale(range=[theme.BRAND_ORANGE, theme.DATA_BLUE]),
                legend=alt.Legend(orient="top", title=None),
            ),
            yOffset="contrast:N",
        )
        bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low_bonferroni:Q", title="Change in CRPS (points; negative is better)"),
            x2="high_bonferroni:Q",
        )
        dots = base.mark_point(filled=True, size=60).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#1A1A1A")
            .encode(x="x:Q")
        )
        panels.append(
            (bars + dots + zero).properties(
                width=WIDTH, height=230, title=_title(f"Arms minus {reference}")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            "Counted CRPS contrasts with Bonferroni intervals (54 tests). Minimum detectable",
            "effect for a three-seed mean: 0.083 at the eight, 0.112 at the six.",
        )
    )
    chart.save(DOCS_DIR / "figure_3_contrasts.svg")


def example_table() -> pl.DataFrame:
    """Return the example Primary's forecasts and actuals for one origin (read from the stored cells)."""
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00425_prep as prep425  # ty: ignore[unresolved-import]
    import e00470_build as build470  # ty: ignore[unresolved-import]
    import e00470_score as score470  # ty: ignore[unresolved-import]
    import e00491_score as score  # ty: ignore[unresolved-import]
    from e00435_tailcore import widen  # ty: ignore[unresolved-import]

    month = EXAMPLE_ORIGIN[:7]
    ctx = prep425.cleaned_scoring_context(series=adapter.series_table())
    base = score470.month_context(month=month, sites=build470.site_map(), ctx=ctx).filter(
        (pl.col("primary") == EXAMPLE_PRIMARY)
        & (pl.col("power_fcst_init_time").dt.strftime("%Y-%m-%d") == EXAMPLE_ORIGIN)
    )
    capacity = float(
        build470.site_map().filter(pl.col("primary") == EXAMPLE_PRIMARY)["capacity_mw"][0]
    )
    pieces = []
    for arm, cell in (("S0", 0), ("EMA", 0), ("Pool5", 0)):
        wide = widen(
            frame=base.select(*score.KEYS).join(
                pl.read_parquet(score.pred_path(arm=arm, month=month, cell=cell)),
                on=score.KEYS,
                how="left",
            ),
            factor=1.15,
        )
        pieces.append(
            base.select(*score.KEYS, "valid_time", "lead_hours", "actual")
            .join(wide, on=score.KEYS)
            .with_columns(
                arm=pl.lit(arm),
                **{q: pl.col(q) * capacity for q in ("q05", "q50", "q95", "actual")},
            )
        )
    return pl.concat(pieces)


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of one Primary: S0, EMA, and Pool5."""
    table = example_table()
    for tag, hours, width in (("day_ahead", (12.0, 36.0), WIDTH), ("14_days", (0.0, 340.0), WIDTH)):
        part = table.filter(pl.col("lead_hours").is_between(*hours)).to_pandas()  # ty: ignore[unresolved-attribute]
        x = alt.X("valid_time:T", title=None)
        band = (
            alt.Chart(part[part["arm"] == "EMA"])  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.2, color=theme.DATA_BLUE)
            .encode(x=x, y=alt.Y("q05:Q", title="Power (MW)"), y2="q95:Q")
        )
        lines = (
            alt.Chart(part)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.5)
            .encode(
                x=x,
                y="q50:Q",
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(
                        domain=["S0", "EMA", "Pool5"],
                        range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_GREEN],
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )
        actual = (
            alt.Chart(part[part["arm"] == "S0"])  # ty: ignore[unresolved-attribute]
            .mark_line(color="#1A1A1A", strokeWidth=2)
            .encode(x=x, y="actual:Q")
        )
        chart = (band + lines + actual).properties(
            width=width,
            height=200,
            title=_title(
                f"D10, forecast made on {EXAMPLE_ORIGIN}: medians of S0, EMA, and Pool5 (colours),",
                "the actual power (black), and the EMA 5% to 95% band (1.15 widening).",
            ),
        )
        chart.save(DOCS_DIR / f"figure_4_example_D10_{tag}.svg")


def main() -> None:
    """Draw every figure."""
    figure_score_by_arm()
    figure_closure()
    figure_contrasts()
    figure_examples()


if __name__ == "__main__":
    main()

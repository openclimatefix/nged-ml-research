"""Experiment 492: draw the two-constant-minus-table-P contrasts against their equivalence limits.

Reads `contrasts.csv` written by `e00492_two_constant.py`. Writes two SVG figures beside the
write-up. Earlier-folds-only fit, clean rows, A-pool and A-gam-in.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00492_two_constant as two  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00492-two-constant-tail-widening-in-place-of-table-p-for-demand"
)
WIDTH: Final[int] = 640


def draw(*, kind: str, limit: float, title: str, name: str) -> None:
    """Draw the contrasts of `kind` ("cov" or "pin") with a +/- `limit` band."""
    c = pl.read_csv(two.WORK / "contrasts.csv").filter(
        (pl.col("slice") == "clean")
        & (pl.col("scheme") == "earlier")
        & (pl.col("arm") != "K2-425A")
        & (pl.col("group") != "report_only")
        & pl.col("metric").str.starts_with(kind)
    )
    if kind == "pin":
        c = c.with_columns(
            diff=pl.col("diff") / pl.col("table_p"),
            low=pl.col("low") / pl.col("table_p"),
            high=pl.col("high") / pl.col("table_p"),
        )
    c = c.with_columns(
        cell=pl.col("arm")
        + ", "
        + pl.col("group").replace({"mva_eight": "eight", "signed_six": "six"}),
        tail=pl.col("metric").str.slice(-3),
        lo=pl.lit(-limit),
        hi=pl.lit(limit),
    )
    band = (
        alt.Chart(c).mark_rect(opacity=0.15, color=theme.DATA_GREEN).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    )
    base = alt.Chart(c).encode(y=alt.Y("cell:N", title=None))
    bars = base.mark_rule(color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title=title),  # ty: ignore[unresolved-attribute]
        x2="high:Q",
    )
    pts = base.mark_point(filled=True, size=50, color=theme.BRAND_ORANGE).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
    (
        alt.layer(band, bars, pts)
        .properties(width=WIDTH // 4, height=90)
        .facet(column=alt.Column("tail:N", title=None, sort=["q01", "q05", "q95", "q99"]))
        .save(FIGURE_DIR / name)
    )


def main() -> None:
    """Draw both figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    draw(
        kind="cov",
        limit=two.LIMIT_COVERAGE,
        title="Coverage: two constants minus table P",
        name="figure_1_coverage_difference.svg",
    )
    draw(
        kind="pin",
        limit=two.LIMIT_PINBALL,
        title="Pinball loss: relative difference",
        name="figure_2_pinball_difference.svg",
    )


if __name__ == "__main__":
    main()

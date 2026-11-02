"""Figure 1 of experiment 490: each arm's clean-row score minus K2-S1, with Bonferroni intervals."""

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = (
    ML_EXPERIMENTS_DATA_DIR
    / "00490-one-lag-robust-network-distilled-from-the-lagged-plus-lagless-mixture"
    / "work"
    / "scores"
)
ORDER = ["K2-S1", "B-mix", "S1", "D-K2", "D-Bmix", "D-K2-corr", "A0-inj"]
TITLES = {"signed_six": "Signed-MW six Primaries", "mva_eight": "MVA-only eight Primaries"}


def main() -> alt.VConcatChart:
    """Return the figure."""
    c = pl.read_csv(WORK / "contrasts.csv").filter(
        (pl.col("reference") == "K2-S1") & (pl.col("metric") == "crps")
        & (pl.col("variant") == "raw") & (pl.col("basis") == "metered")
        & (pl.col("cls") == "clean") & (pl.col("arm") != "K2-S1")
    )  # fmt: skip
    panels = []
    for group, title in TITLES.items():
        p = c.filter(pl.col("group") == group).to_pandas()
        y = alt.Y("arm:N", sort=ORDER, title=None)
        base = alt.Chart(p)
        rule = base.mark_rule(color=theme.BLACK_1, strokeDash=[3, 3]).encode(x=alt.datum(0.0))  # ty: ignore[unresolved-attribute]
        bars = alt.Chart(p).mark_rule(strokeWidth=2).encode(y=y, x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        dots = (
            alt.Chart(p)
            .mark_point(filled=True, size=70)
            .encode(  # ty: ignore[unresolved-attribute]
                y=y,
                x=alt.X(
                    "diff:Q", title="Score minus K2-S1 (points of capacity; negative is better)"
                ),
            )
        )
        panels.append((rule + bars + dots).properties(width=420, height=190, title=title))
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "One distilled network against K2-S1, rows with no labelled event (interim)",
                "Dot: mean over 5 months. Line: Bonferroni 95% interval resampling whole months.",
            ],
            anchor="start",
        )
    )


if __name__ == "__main__":
    import sys

    main().save(sys.argv[1])  # ty: ignore[unresolved-attribute]

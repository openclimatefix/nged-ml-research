# ruff: noqa: E501
"""Experiment 572: the headline chart, each arm's CRPS minus M0's in three lead bands.

Dot: estimate (2 seeds). Line: interval from resampling whole months (5 blocks); the E2 contrast is
Bonferroni-corrected for 24 tests, the others are uncorrected 95% intervals.
"""

from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00572-two-offset-part-1-lags-versus-the-slow-output-offset"
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
ARMS = [
    ("E2", "E2: offset learnt in pre-training, no lags", theme.BRAND_ORANGE),
    ("E2ft", "E2-ft: offset learnt in fine-tuning only", theme.BRAND_ORANGE_LIGHT),
    ("V3", "v3: lags at least 24 h old", theme.DATA_GREEN),
    ("V3D5", "v3+D5: lags and offset", theme.DATA_PURPLE),
]
BANDS = ["0-24h", "24-120h", "120-240h"]


def zero_rule() -> alt.Chart:
    """Return the dashed line at zero."""
    return (
        alt.Chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="z:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]


def main() -> None:
    """Write the SVG next to the page."""
    t = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / NAME / "work/scores/part1_contrasts.csv").filter(
        (pl.col("slice") == "all rows")
        & (pl.col("b") == "M0")
        & pl.col("set").is_in(["eight", "six"])
    )
    rows = [
        {
            "arm": label,
            "band": r["band"],
            "group": "The eight MVA Primaries" if r["set"] == "eight" else "The six MW Primaries",
            "d": r["difference"],
            "lo": r["low_bonferroni_month"],
            "hi": r["high_bonferroni_month"],
        }
        for arm, label, _ in ARMS
        for r in t.filter(pl.col("a") == arm).iter_rows(named=True)
        if r["band"] in BANDS
    ]
    df = pl.DataFrame(rows)
    order = [a[1] for a in ARMS]
    colours = [a[2] for a in ARMS]
    base = alt.Chart().encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("arm:N", scale=alt.Scale(domain=order, range=colours), legend=None),
    )
    rule = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title=None),
        x2="hi:Q",
    )
    dot = base.mark_point(filled=True, size=50).encode(x="d:Q")  # ty: ignore[unresolved-attribute]
    layers = alt.layer(zero_rule(), rule, dot).properties(width=170, height=130)
    chart = layers.facet(
        data=df,
        row=alt.Row("group:N", title=None),
        column=alt.Column("band:O", sort=BANDS, title="Lead band"),
    ).properties(
        title=alt.TitleParams(
            [
                "Lags and the output offset each lower the error, and lags plus the offset lower it most",
                "Horizontal axis: CRPS minus M0's, in points of capacity (more negative is better). Dot: estimate (2 seeds). Line: 95% interval from resampling whole months (E2: Bonferroni, 24 tests; others uncorrected).",
                "Dashed line: same as M0 (no lags, no offset). Experiment 565's coarse M0 and D5 are the M0 and E2 arms.",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "contrasts_vs_m0.svg"))


if __name__ == "__main__":
    main()

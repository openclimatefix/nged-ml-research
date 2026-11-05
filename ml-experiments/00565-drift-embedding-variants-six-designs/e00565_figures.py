# ruff: noqa: E501
"""Experiment 565: the headline chart, every design's CRPS minus the coarse M0's at 24-120 h.

Reads the contrast tables the screening scorer wrote (`screen`, `screen_d2`, `screen_h`, `d5x`,
`d5conf`). Dot: estimate. Line: Bonferroni 95% interval from resampling whole months.
"""

from pathlib import Path

import altair as alt
import polars as pl
from e00565_design6 import W
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00565-drift-embedding-variants-six-designs"
)
ROWS = [
    ("screen", "C-A10", "1. Monthly embedding (A10)", theme.DATA_GREEN),
    ("screen_d2", "D2", "2. Slow-trend input", theme.DATA_PURPLE),
    ("screen_d2", "D2-null", "2. Slow-trend input, placebo", theme.DATA_PURPLE_LIGHT),
    ("screen", "D3m", "3. Random-walk embedding, monthly", theme.DATA_SKY),
    ("screen", "D3q", "3. Random-walk embedding, quarterly", theme.DATA_SKY),
    ("screen", "D4", "4. Shared global-time embedding", theme.BRAND_ORANGE_LIGHT),
    ("screen", "D4-null", "4. Shared global-time embedding, placebo", theme.BRAND_ORANGE_LIGHT),
    ("screen_h", "H", "1 + 5. Embedding and offset together", theme.DATA_GREEN_LIGHT),
    ("d5x", "D5", "5. Output offset, seeds 0-4 (post hoc)", theme.BRAND_ORANGE),
    ("d5conf", "D5", "5. Output offset, fresh seeds 5-9", theme.BRAND_ORANGE),
]


def main() -> None:
    """Write the SVG next to the page."""
    rows = []
    for prefix, arm, label, colour in ROWS:
        t = pl.read_csv(W / f"scores/{prefix}_contrasts.csv").filter(
            (pl.col("band") == "24-120h")
            & (pl.col("arm") == arm)
            & (pl.col("ref") == "C-M0")
            & pl.col("set").is_in(["eight", "six"])
        )
        rows += [
            {
                "design": label,
                "group": "The eight MVA Primaries"
                if r["set"] == "eight"
                else "The six MW Primaries",
                "d": r["difference"],
                "lo": r["low_bonferroni_month"],
                "hi": r["high_bonferroni_month"],
                "colour": colour,
            }
            for r in t.iter_rows(named=True)
        ]
    df = pl.DataFrame(rows)
    order = [r[2] for r in ROWS]
    colours = {r[2]: r[3] for r in ROWS}

    def panel(group: str, *, last: bool) -> alt.LayerChart:
        """Return one group's layered chart."""
        d = df.filter(pl.col("group") == group)
        base = alt.Chart(d).encode(
            y=alt.Y("design:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.Color(
                "design:N",
                scale=alt.Scale(domain=order, range=[colours[k] for k in order]),
                legend=None,
            ),
        )
        xt = (
            "CRPS minus coarse M0's (points of capacity; more negative is better)" if last else None
        )
        rule = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title=xt, scale=alt.Scale(domain=[-0.7, 1.0])), x2="hi:Q"
        )
        dot = base.mark_point(filled=True, size=50).encode(x="d:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"z": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="grey", strokeDash=[4, 3])
            .encode(x="z:Q")
        )
        return alt.layer(zero, rule, dot).properties(  # ty: ignore[invalid-return-type]
            width=360, height=190, title=alt.TitleParams(group, anchor="start", fontSize=12)
        )

    chart = alt.vconcat(
        panel("The eight MVA Primaries", last=False), panel("The six MW Primaries", last=True)
    ).properties(
        title=alt.TitleParams(
            [
                "Only the output offset beats the coarse M0 at the eight MVA Primaries; no design beats it at the six MW Primaries",
                "Dot: estimate (2 seeds; D5: 5 seeds twice). Line: 95% interval, Bonferroni, from resampling whole months.",
                "Dashed line: same as coarse M0. 24-120 h leads. Placebos permute months (4) or use old summaries (2).",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "contrasts_vs_coarse_m0.svg"))


if __name__ == "__main__":
    main()

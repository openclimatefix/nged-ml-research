"""Draw the known-answer recovery of experiment 405: real-sky and shuffled-sky recovery per Primary.

Reads `kat_gates.json` (written by `e00405_kat_report.py`) and writes the figure beside `result.md`.
Only ratios of recovered to injected capacity are drawn.
"""

import json
from pathlib import Path

import altair as alt
import e00405_pv as pv  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

OUT = Path(__file__).resolve().parents[2] / (
    "docs/ml-experiments/00405-mdn-explicit-pv-term-learned-capacity-known-answer"
)


def main() -> None:
    """Write the recovery figure."""
    gates = json.loads((pv.WORK_DIR / "kat_gates.json").read_text())
    rows = []
    for arm, g in gates.items():
        for site, v in g["sites"].items():
            rows.append({"arm": arm, "site": site, "kind": "Real sky", "value": v["divided"]})
            placebo = v["placebo_over_held"] * v["divided"]
            rows.append({"arm": arm, "site": site, "kind": "Shuffled sky (placebo)", "value": placebo})
    data = pl.DataFrame(rows).to_pandas()
    colour = alt.Color(
        "kind:N",
        scale=alt.Scale(
            domain=["Real sky", "Shuffled sky (placebo)"],
            range=[theme.DATA_BLUE, theme.BRAND_ORANGE],
        ),
        legend=alt.Legend(title=None, orient="top", direction="horizontal"),
    )
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", title=None, sort=list(pv.SIGNED_SIX)),
            y=alt.Y("value:Q", title="Recovered share of the injected capacity (after the divisor)"),
            color=colour,
            xOffset="kind:N",
        )
    )
    chart = bars.properties(width=260, height=220).facet(column=alt.Column("arm:N", title=None))
    chart.save(OUT / "figure_1_known_answer_recovery.svg")


if __name__ == "__main__":
    main()

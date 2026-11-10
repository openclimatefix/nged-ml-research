"""Figure for experiment 675: pretrained minus comparison arm, donor megawatts found, per cell."""

# ruff: noqa: E501  # research code run once; behaviour fixed by the results

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00675_pretrain import ROOT

CELLS = ["long, deep", "medium, deep", "long, shallow", "medium, shallow"]
COLOURS = {"random-init encoder (R)": ocf.BRAND_ORANGE if hasattr(ocf, "BRAND_ORANGE") else "#FF4901",
           "no-encoder logistic (F)": "#306BFF",
           "causal pretraining (P-C)": "#B701FF"}  # fmt: skip


def main(out: Path) -> None:
    """Write the figure as SVG and PNG into `out`."""
    c = json.loads((ROOT / "probe" / "contrasts.json").read_text())
    names = {
        "R": "random-init encoder (R)",
        "F": "no-encoder logistic (F)",
        "P-C": "causal pretraining (P-C)",
    }
    rows = [
        {"cell": cell, "arm": names[b], "diff": v["mean"], "lo": v["ci"][0], "hi": v["ci"][1]}
        for cell in CELLS for b in names for v in [c[cell][f"P-M minus {b} donor MW found"]]
    ]  # fmt: skip
    df = pl.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y("cell:N", sort=CELLS, title=None), yOffset=alt.YOffset("arm:N", sort=list(names.values())),
                                color=alt.Color("arm:N", sort=list(names.values()), scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                                                legend=alt.Legend(title="P-M minus", orient="top", direction="vertical")))  # fmt: skip
    pts = base.mark_point(filled=True, size=60, aria=False).encode(x=alt.X("diff:Q", title="Donor megawatts found: masked pretraining (P-M) minus the comparison arm (share of megawatts; more positive is better for P-M)"))  # fmt: skip  # ty: ignore[unresolved-attribute]
    bars = base.mark_rule(aria=False).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (bars + pts + zero).properties(width=680 - 130, height=300, title=alt.Title(
        ["Masked pretraining finds fewer donor megawatts than a random-init encoder",
         "in the deep cells. Dot: mean of three seeds. Line: 95% interval from resampling the",
         "40 scored worlds (not fit variance). Dashed line: no difference."]))  # fmt: skip
    chart.save(out / "figure_contrasts.svg")
    chart.save(out / "figure_contrasts.png", scale_factor=2)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

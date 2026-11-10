"""Figure for experiment 676: paired share in the two deep cells by arm, against the bars."""

# ruff: noqa: E501  # research chart script

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import plotting.ocf_theme  # noqa: F401  (registers the OCF theme)
import polars as pl
from e00676_worlds import ROOT

CELLS = ["long, deep", "medium, deep"]
ARMS = {"E1-J": "#555555", "F-hyb": "#FF4901", "R-hyb": "#306BFF", "G": "#17E58F", "G-R": "#B701FF", "G-0": "#FFC000"}  # fmt: skip
NAMES = {"E1-J": "E1-J (event table plus search)", "F-hyb": "F-hyb (logistic donor alarms)", "R-hyb": "R-hyb (random-init encoder alarms)", "G": "G (edge-state network)", "G-R": "G-R (rewired graph)", "G-0": "G-0 (no links)"}  # fmt: skip


def main(out: Path) -> None:
    """Write the figure as SVG into `out`."""
    s = json.loads((ROOT / "summary.json").read_text())["cells"]
    rows = [
        {"cell": c, "arm": NAMES[a], "paired": s[c][a]["paired"], "lo": s[c][a]["paired_ci"][0], "hi": s[c][a]["paired_ci"][1]}
        for c in CELLS for a in ARMS if a in s[c]
    ]  # fmt: skip
    names = [NAMES[a] for a in ARMS if a in s[CELLS[0]]]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("cell:N", sort=CELLS, title=None), yOffset=alt.YOffset("arm:N", sort=names),
        color=alt.Color("arm:N", sort=names, scale=alt.Scale(domain=names, range=[ARMS[a] for a in ARMS if a in s[CELLS[0]]]),
                        legend=alt.Legend(title="Arm", orient="top", direction="vertical")),
    )  # fmt: skip
    pts = base.mark_point(filled=True, size=60, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "paired:Q",
            title="Share of injected transfer megawatts paired correctly (more is better)",
        )
    )
    bars = base.mark_rule(aria=False).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    chart = (bars + pts).properties(
        width=680 - 130, height=340,
        title=alt.Title(
            "The edge-state network (G) pairs no more than the hybrids in the deep cells",
            subtitle=["Dot: paired share (one fit for G, G-R, G-0 and F-hyb; mean of 3 seeds for R-hyb).",
                      "Line: 95% interval from resampling the 40 worlds, not fit variance. Bars to pass: 0.395 and 0.347."],
        ),
    )  # fmt: skip
    chart.save(out / "figure_paired_by_arm.svg")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

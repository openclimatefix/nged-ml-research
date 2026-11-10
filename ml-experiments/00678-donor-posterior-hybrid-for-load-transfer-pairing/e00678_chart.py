"""Figure for experiment 678: paired share of each hybrid minus E1-J, per cell."""

# ruff: noqa: E501  # research chart script

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import plotting.ocf_theme  # noqa: F401  (registers the OCF theme)
import polars as pl
from e00678_run import OUT

CELLS = ["long, deep", "medium, deep", "long, shallow", "medium, shallow"]
ARMS = {"F-hyb": ("logistic regression (F)", "#FF4901"), "R-hyb": ("random-init encoder (R)", "#306BFF"),
        "P-M-hyb": ("masked pretraining (P-M)", "#B701FF"), "P-C-hyb": ("causal pretraining (P-C)", "#17E58F")}  # fmt: skip


def main(out: Path) -> None:
    """Write the figure as SVG into `out`."""
    s = json.loads((OUT / "summary.json").read_text())["cells"]
    rows = [
        {"cell": c, "arm": ARMS[a][0], "diff": s[c][a]["paired_minus_e1j"], "lo": s[c][a]["paired_minus_e1j_ci"][0], "hi": s[c][a]["paired_minus_e1j_ci"][1]}
        for c in CELLS for a in ARMS
    ]  # fmt: skip
    names = [v[0] for v in ARMS.values()]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("cell:N", sort=CELLS, title=None), yOffset=alt.YOffset("arm:N", sort=names),
        color=alt.Color("arm:N", sort=names, scale=alt.Scale(domain=names, range=[v[1] for v in ARMS.values()]),
                        legend=alt.Legend(title="Hybrid with donor alarms from", orient="top", direction="vertical")),
    )  # fmt: skip
    pts = base.mark_point(filled=True, size=60, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "diff:Q",
            title="Paired share of transfer megawatts: hybrid minus E1-J, in share (more positive is better)",
        )
    )
    bars = base.mark_rule(aria=False).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    bar = (
        alt.Chart(pl.DataFrame({"x": [0.10]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000", strokeDash=[1, 3])
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = (bars + pts + zero + bar).properties(
        width=680 - 130, height=300,
        title=alt.Title(
            "Donor alarms raise pairing by 0.03 to 0.08 in the deep cells, below the bar",
            subtitle=["Dot: mean of seeds (F is one fit). Line: 95% interval from resampling the 40 worlds,",
                      "not fit variance. Dashed line: same as E1-J. Dotted line: the pre-registered +0.10 bar."],
        ),
    )  # fmt: skip
    chart.save(out / "figure_paired_minus_e1j.svg")
    if (png := __import__("os").environ.get("PNG_DIR")) is not None:
        chart.save(Path(png) / "figure_paired_minus_e1j.png", scale_factor=2)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

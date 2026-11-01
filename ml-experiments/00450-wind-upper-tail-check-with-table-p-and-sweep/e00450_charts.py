"""Draw experiment 00450's figures as SVG from `work/score_*.json` and the farm table.

    source ml-experiments/00450-*/env.sh
    uv run --no-sync python ml-experiments/00450-*/e00450_charts.py

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`.
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00450-wind-upper-tail-check-with-table-p-and-sweep"
)
WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00450-wind-upper-tail-check-with-table-p-and-sweep" / "work"
)
LABELS: Final[dict[str, str]] = {
    "w32": "hidden width 32", "w128": "hidden width 128", "d1": "depth 1", "d3": "depth 3",
    "e8": "encoder width 8", "e32": "encoder width 32", "k2": "2 mixture components",
    "k5": "5 mixture components", "a6": "ERA5 pretraining",
}  # fmt: skip
FARM_SHARE: Final[dict[str, list[float]]] = json.loads((WORK / "farm_share.json").read_text())


def main() -> None:
    """Write both figures."""
    scores = json.loads((WORK / "score_w32_w128_d1_d3_e8_e32_k2_k5_a6.json").read_text())
    rows = [
        {"setting": LABELS[k], "d": v["kept rows"]["d_qs"][0], "lo": v["kept rows"]["d_qs"][1],
         "hi": v["kept rows"]["d_qs"][2]}
        for k, v in scores.items() if k in LABELS
    ]  # fmt: skip
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(y=alt.Y("setting:N", sort=None, title=None))
    chart = (
        base.mark_rule(color=ocf.DATA_BLUE).encode(
            x=alt.X("lo:Q", title="Quantile score minus the base network's (points of capacity)"),
            x2="hi:Q",
        )
        + base.mark_point(color=ocf.DATA_BLUE, filled=True, size=60).encode(x="d:Q")
        + alt.Chart(pl.DataFrame({"x": [-0.10, 0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    ).properties(
        width=440,
        title="No setting lowers the plain wind network's score by the 0.10-point margin",
    )
    chart.save(DOCS / "figure_1_contrasts.svg")
    farms = pl.DataFrame(
        [
            {"setting": "base" if k == "base" else LABELS[k], "farm": f, "share": v[i]}
            for k, v in FARM_SHARE.items() for i, f in enumerate(("Farm A", "Farm B", "Farm C"))
        ]
    )  # fmt: skip
    bars = (
        alt.Chart(farms)
        .mark_circle(size=70)
        .encode(
            x=alt.X("share:Q", title="Share of observations above q99 after table P (%)"),
            y=alt.Y("setting:N", sort=None, title=None),
            color=alt.Color("farm:N", title=None),
        )
    )
    limit = alt.Chart(pl.DataFrame({"x": [1.5]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    (bars + limit).properties(
        width=440, title="Farm A stays above the 1.5% limit in every setting"
    ).save(DOCS / "figure_2_upper_tail_by_farm.svg")


if __name__ == "__main__":
    main()

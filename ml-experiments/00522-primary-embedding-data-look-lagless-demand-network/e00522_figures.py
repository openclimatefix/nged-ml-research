# ruff: noqa: E501
"""Draw the figures of experiment 522 from `work/part1.json`, `work/part2.json`, and `work/part3.json`.

Figures (SVG, in the experiment's docs folder; optimise each with `npx svgo@4 --multipass
--precision=1 --final-newline`):

1. `figure_1_part3_crps_change.svg`: the CRPS change when the embedding is replaced, per network and arm;
2. `figure_2_distance_matrix.svg`: the 16 x 16 distance matrix of the movement (fold 2026-01, 5 seeds);
3. `figure_3_leading_dimensions.svg`: the two leading dimensions of the movement, with Primary names;
4. `figure_4_displacement.svg`: each Primary's displacement from the start after pretraining.

```bash
uv run --no-sync python ml-experiments/00522-*/e00522_figures.py
```
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00522_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00522-primary-embedding-data-look-lagless-demand-network"
)
WIDTH: Final[int] = 600
NETWORKS: Final[dict[str, str]] = {
    "L0PT_pretrained": "Lag-free, pretrained",
    "L0PT_finetuned": "Lag-free, fine-tuned",
    "A": "Lagged (experiment 425 arm A)",
}
ARMS: Final[dict[str, str]] = {
    "derangement": "Vectors swapped between Primaries (mean of 20)",
    "mean_vector": "Every Primary gets the mean vector",
    "start_vector": "Each Primary gets its random start vector",
    "fresh_random": "Fresh N(0, 1) vectors (mean of 20)",
}
COLOURS: Final[list[str]] = [
    theme.BRAND_ORANGE,
    theme.DATA_BLUE,
    theme.DATA_PURPLE,
    theme.DATA_GREEN,
]


def figure_part3(*, report: dict) -> None:
    """Draw the CRPS change of every arm for each network, with the day-block interval."""
    rows = []
    for key, label in NETWORKS.items():
        for arm, arm_label in ARMS.items():
            entry = report[key].get(arm)
            if entry is None:
                continue
            lo, hi = entry["day_block_interval_95"]
            rows.append(
                {
                    "network": label,
                    "arm": arm_label,
                    "change": entry["change_mean_scored_pct"],
                    "lo": lo,
                    "hi": hi,
                }
            )
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y("network:N", title=None, sort=list(NETWORKS.values())),
        color=alt.Color(
            "arm:N",
            title=None,
            scale=alt.Scale(domain=list(ARMS.values()), range=COLOURS),
            legend=alt.Legend(orient="top", columns=1, labelLimit=500),
        ),
        yOffset="arm:N",
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "change:Q",
            title="CRPS change (percentage points of effective capacity; 0 = stored vectors; higher is worse)",
        )
    )
    bars = base.mark_rule().encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000000", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (zero + bars + points).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            "Replacing the Primary embedding vectors raises CRPS",
            subtitle=[
                "Fold 2026-01, seed 0, 14 Primaries that are not report-only. Dot: mean change. Line: 95% interval from",
                "resampling the 31 origin days. Stored vectors are the zero line.",
            ],
        ),
    )
    chart.save(DOCS_DIR / "figure_1_part3_crps_change.svg")


def figure_matrix(*, report: dict) -> None:
    """Draw the movement's distance matrix for fold 2026-01."""
    d = np.array(report["2026-01"]["distance_matrix_movement"])
    sites = report["sites"]
    frame = pl.DataFrame(
        [
            {"a": sites[i], "b": sites[j], "distance": float(d[i, j])}
            for i in range(16)
            for j in range(16)
        ]
    )
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("a:N", sort=sites, title=None),
            y=alt.Y("b:N", sort=sites, title=None),
            color=alt.Color(
                "distance:Q",
                title="Distance (vector units)",
                scale=alt.Scale(range=["#FFFFFF", theme.DATA_BLUE]),
            ),
        )
        .properties(
            width=WIDTH,
            height=WIDTH,
            title=alt.TitleParams(
                "How far apart the Primaries' learned movements are",
                subtitle=[
                    "Euclidean distance between the stored vector minus its random start, averaged over 5 seeds (fold 2026-01)."
                ],
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_2_distance_matrix.svg")


def figure_dimensions(*, report: dict) -> None:
    """Draw the two leading dimensions of the movement (classical scaling), labelled by Primary."""
    xy = np.array(report["2026-01"]["mds_2d"])
    sites = report["sites"]
    signed = set(report["signed_mw_sites"])
    frame = pl.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "site": sites,
            "metering": ["Signed MW" if s in signed else "Apparent power (MVA)" for s in sites],
        }
    )
    base = alt.Chart(frame).encode(
        x=alt.X("x:Q", title="Leading dimension 1"),
        y=alt.Y("y:Q", title="Leading dimension 2"),
        color=alt.Color(
            "metering:N",
            title=None,
            scale=alt.Scale(
                domain=["Signed MW", "Apparent power (MVA)"],
                range=[theme.BRAND_ORANGE, theme.DATA_BLUE],
            ),
            legend=alt.Legend(orient="top"),
        ),
    )
    chart = (
        base.mark_point(filled=True, size=90) + base.mark_text(dy=-10).encode(text="site:N")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=360,
        title=alt.TitleParams(
            "The Primaries' learned movements in two dimensions",
            subtitle=[
                "Classical scaling of the seed-averaged distance matrix of the movement (fold 2026-01); axes have no units."
            ],
        ),
    )
    chart.save(DOCS_DIR / "figure_3_leading_dimensions.svg")


def figure_displacement(*, part1: dict) -> None:
    """Draw each Primary's displacement from the start after pretraining, with the range over the 20 cells."""
    sites = json.loads((common.WORK_DIR / "part2.json").read_text())["sites"]
    d = part1["displacement"]["L0PT"]
    frame = pl.DataFrame(
        {
            "site": sites,
            "mean": d["per_primary_mean_share"],
            "lo": d["per_primary_min_cell_share"],
            "hi": d["per_primary_max_cell_share"],
        }
    )
    base = alt.Chart(frame).encode(y=alt.Y("site:N", sort=sites, title=None))
    chart = (
        base.mark_rule(color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Displacement from the random start (share of the start vector's length)",
            ),
            x2="hi:Q",
        )
        + base.mark_point(filled=True, color=theme.BRAND_ORANGE, size=60).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=360,
        title=alt.TitleParams(
            "How far each Primary's vector moved in pretraining",
            subtitle=[
                "Dot: mean over 4 folds and 5 seeds. Line: smallest to largest of the 20 cells."
            ],
        ),
    )
    chart.save(DOCS_DIR / "figure_4_displacement.svg")


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    part1 = json.loads((common.WORK_DIR / "part1.json").read_text())
    part2 = json.loads((common.WORK_DIR / "part2.json").read_text())
    part3 = json.loads((common.WORK_DIR / "part3.json").read_text())
    figure_part3(report=part3)
    figure_matrix(report=part2)
    figure_dimensions(report=part2)
    figure_displacement(part1=part1)


if __name__ == "__main__":
    main()

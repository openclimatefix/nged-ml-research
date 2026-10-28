"""Draw the headline figure of experiment 377's write-up.

The numbers are those printed by `e00377_score.py` (per-Primary 95% intervals with 2,000
resamples, pooled 99% intervals with 20,000 resamples), typed in so the chart needs no re-score.

    uv run --no-sync python ml-experiments/00377-*/e00377_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / ("00377-demand-v1-first-run-feasibility")
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GROUP_COLOURS: Final[dict[str, str]] = {
    "signed-MW six": ocf.DATA_BLUE,
    "MVA-only eight": ocf.BRAND_ORANGE,
    "short history": ocf.DATA_PURPLE,
}
ROWS: Final[list[tuple[str, str, float, float, float]]] = [
    ("Pooled, signed-MW (99%)", "signed-MW six", -0.068, -0.280, 0.138),
    ("Pooled, MVA-only (99%)", "MVA-only eight", 0.344, 0.093, 0.595),
    ("Pooled, no D8 (99%)", "MVA-only eight", 0.178, -0.015, 0.391),
    ("D5", "signed-MW six", 0.138, -0.056, 0.333),
    ("D9", "signed-MW six", -0.001, -0.248, 0.268),
    ("D10", "signed-MW six", -0.179, -0.650, 0.161),
    ("D11", "signed-MW six", -0.143, -0.333, 0.018),
    ("D14", "signed-MW six", -0.177, -0.391, -0.007),
    ("D16", "signed-MW six", -0.043, -0.180, 0.106),
    ("D2", "MVA-only eight", 0.176, -0.029, 0.395),
    ("D3", "MVA-only eight", -0.077, -0.242, 0.063),
    ("D4", "MVA-only eight", -0.064, -0.356, 0.157),
    ("D8", "MVA-only eight", 1.612, 0.516, 2.846),
    ("D12", "MVA-only eight", 0.763, 0.324, 1.226),
    ("D19", "MVA-only eight", 0.114, -0.003, 0.265),
    ("D21", "MVA-only eight", -0.021, -0.152, 0.101),
    ("D22", "MVA-only eight", 0.358, 0.199, 0.520),
    ("D15", "short history", -0.506, -2.305, 0.835),
    ("D18", "short history", -0.148, -0.346, 0.047),
]


def main() -> None:
    """Write the SVG."""
    frame = pl.DataFrame(
        ROWS, schema=["row", "group", "difference", "lower", "upper"], orient="row"
    )
    order = [r[0] for r in ROWS]
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=order, title=None),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q", title="CRPS of default 1 minus BG2 (points of capacity; negative is better)"
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + dots).properties(
        width=WIDTH - 200,
        height=22 * len(ROWS),
        title=alt.TitleParams(
            text=textwrap.wrap(
                "Default 1 trails BG2 at the MVA-only Primaries and ties it at the signed-MW six",
                80,
            ),
            subtitle=textwrap.wrap(
                "Days 1 to 10, shared half-hours of 2025-07 to 2026-06. Short history is D15 and "
                "D18, report only. Dot: estimate. Line: 95% interval per Primary and 99% interval "
                "for pooled rows, from resampling whole months. Dashed line: same as BG2. "
                "Default 1 here is the stack without S below 168 h.",
                WRAP,
            ),
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / "headline.svg")


if __name__ == "__main__":
    main()

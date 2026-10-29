"""Draw the two figures of experiment 378's write-up.

The numbers are those printed by `e00378_score.py` (per-Primary 95% intervals with 2,000
resamples, pooled 99% intervals with 20,000 resamples), typed in so the charts need no re-score.

    uv run --no-sync python ml-experiments/00378-*/e00378_charts.py
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
    / "00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GROUP_COLOURS: Final[dict[str, str]] = {
    "signed-MW six": ocf.DATA_BLUE,
    "MVA-only eight": ocf.BRAND_ORANGE,
    "short history": ocf.DATA_PURPLE,
}
MINUS_BG2: Final[list[tuple[str, str, float, float, float]]] = [
    ("Pooled, signed-MW (99%)", "signed-MW six", -0.121, -0.314, 0.059),
    ("Pooled, MVA-only (99%)", "MVA-only eight", 0.634, 0.347, 0.943),
    ("Pooled, no D8 (99%)", "MVA-only eight", 0.158, -0.024, 0.352),
    ("D5", "signed-MW six", 0.045, -0.129, 0.211),
    ("D9", "signed-MW six", -0.060, -0.279, 0.181),
    ("D10", "signed-MW six", -0.238, -0.684, 0.088),
    ("D11", "signed-MW six", -0.161, -0.341, -0.008),
    ("D14", "signed-MW six", -0.209, -0.409, -0.051),
    ("D16", "signed-MW six", -0.099, -0.216, 0.032),
    ("D2", "MVA-only eight", 0.158, -0.036, 0.358),
    ("D3", "MVA-only eight", -0.085, -0.247, 0.048),
    ("D4", "MVA-only eight", -0.081, -0.339, 0.118),
    ("D8", "MVA-only eight", 3.966, 2.294, 5.807),
    ("D12", "MVA-only eight", 0.717, 0.286, 1.166),
    ("D19", "MVA-only eight", 0.097, -0.011, 0.234),
    ("D21", "MVA-only eight", -0.030, -0.155, 0.086),
    ("D22", "MVA-only eight", 0.331, 0.189, 0.475),
    ("D15", "short history", 0.311, -1.328, 1.526),
    ("D18", "short history", -0.208, -0.399, -0.020),
]
MINUS_DEFAULT_1: Final[list[tuple[str, str, float, float, float]]] = [
    ("Pooled, signed-MW (99%)", "signed-MW six", -0.034, -0.057, -0.016),
    ("Pooled, MVA-only (99%)", "MVA-only eight", -0.023, -0.047, -0.004),
    ("Pooled, no D8 (99%)", "MVA-only eight", -0.021, -0.041, -0.007),
    ("D5", "signed-MW six", -0.051, -0.083, -0.028),
    ("D9", "signed-MW six", -0.046, -0.078, -0.018),
    ("D10", "signed-MW six", -0.038, -0.055, -0.022),
    ("D11", "signed-MW six", -0.013, -0.021, -0.004),
    ("D14", "signed-MW six", -0.023, -0.039, -0.008),
    ("D16", "signed-MW six", -0.036, -0.059, -0.014),
    ("D2", "MVA-only eight", -0.018, -0.036, -0.002),
    ("D3", "MVA-only eight", -0.008, -0.017, 0.002),
    ("D4", "MVA-only eight", -0.019, -0.058, 0.018),
    ("D8", "MVA-only eight", -0.038, -0.148, 0.061),
    ("D12", "MVA-only eight", -0.048, -0.068, -0.029),
    ("D19", "MVA-only eight", -0.018, -0.036, -0.005),
    ("D21", "MVA-only eight", -0.007, -0.016, 0.001),
    ("D22", "MVA-only eight", -0.027, -0.053, 0.0),
    ("D15", "short history", 0.615, -0.203, 1.323),
    ("D18", "short history", -0.031, -0.053, -0.010),
]


def draw(
    *,
    rows: list[tuple[str, str, float, float, float]],
    axis: str,
    title: str,
    subtitle: str,
    name: str,
) -> None:
    """Write one forest chart."""
    frame = pl.DataFrame(
        rows, schema=["row", "group", "difference", "lower", "upper"], orient="row"
    )
    order = [r[0] for r in rows]
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
        x=alt.X("lower:Q", title=axis), x2="upper:Q"
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + dots).properties(
        width=WIDTH - 200,
        height=22 * len(rows),
        title=alt.TitleParams(
            text=textwrap.wrap(title, 80),
            subtitle=textwrap.wrap(subtitle, WRAP),
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / name)


def main() -> None:
    """Write both SVGs."""
    note = (
        "Days 1 to 10, shared half-hours of 2025-07 to 2026-06. Short history is D15 and D18, "
        "report only. Dot: estimate. Line: 95% interval per Primary and 99% interval for pooled "
        "rows, from resampling whole months. Dashed line: no difference."
    )
    draw(
        rows=MINUS_BG2,
        axis="CRPS of default 3 minus BG2 (points of capacity; negative is better)",
        title="Default 3 ties BG2 at the signed-MW six and trails it at Marsh Lane (D8)",
        subtitle=note,
        name="headline.svg",
    )
    draw(
        rows=MINUS_DEFAULT_1,
        axis="CRPS of default 3 minus default 1 (points of capacity; negative is better)",
        title="Default 3 scores better than default 1 at all 14 countable Primaries",
        subtitle=note + " Default 1 here is the solar summary S below 168 h and the stack above.",
        name="minus_default_1.svg",
    )


if __name__ == "__main__":
    main()

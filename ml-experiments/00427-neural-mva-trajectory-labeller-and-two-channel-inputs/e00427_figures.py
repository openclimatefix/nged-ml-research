# ruff: noqa: E501
"""Draw the figure of experiment 427's step 0: the labeller's false reverse flow with nothing injected."""

from pathlib import Path
from typing import Final

import altair as alt
import e00427_labeller as lab
import polars as pl

DOCS: Final[Path] = Path(
    "docs/ml-experiments/00427-neural-mva-trajectory-labeller-and-two-channel-inputs"
)
WIDTH: Final[int] = 680
BLUE, ORANGE = "#306BFF", "#FF4901"
HIDDEN_ROTATION: Final[dict[str, str]] = {
    "D5": "r1",
    "D10": "r1",
    "D9": "r2",
    "D14": "r2",
    "D11": "r3",
    "D16": "r3",
}


def false_reverse_chart() -> alt.Chart:
    """Draw, per Primary, the share of unique sunny-midday half-hours with `pi > 0.5` and no injection."""
    rows = []
    for site, rotation in HIDDEN_ROTATION.items():
        frames = [
            pl.read_parquet(f).filter(
                (pl.col("level") == 0.0) & (pl.col("window") == "test") & (pl.col("site") == site)
            )
            for f in sorted((lab.WORK_DIR / "full_gru").glob(f"{rotation}_*_none_gru_s0.parquet"))
        ]
        f = pl.concat(frames).filter(pl.col("truth_points") >= 0.0)
        valid = f["valid"].to_numpy()
        sel = (
            lab.inputs.is_midday(valid_us=valid)
            & (f["u_gate"].to_numpy() >= lab.GATE_U)
            & (f["k"].to_numpy() >= lab.SUNNY_K)
        )
        rows.append({"site": site, "share": float((f["pi"].to_numpy()[sel] > 0.5).mean()) * 100.0})
    base = alt.Chart(pl.DataFrame(rows))
    bars = base.mark_bar(color=BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "site:N",
            title=None,
            axis=alt.Axis(labelAngle=0),
            sort=["D5", "D10", "D9", "D14", "D11", "D16"],
        ),
        y=alt.Y("share:Q", title="Share of half-hours (%)"),
    )
    limit = (
        alt.Chart(pl.DataFrame({"y": [2.0]}))
        .mark_rule(color=ORANGE, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    print(rows)
    return (bars + limit).properties(
        width=WIDTH - 90,
        height=220,
        title=alt.TitleParams(
            "With nothing injected, the labeller calls reverse flow at D5 and D10",
            subtitle=[
                "Share of sunny-midday half-hours (April to September, 10:00 to 14:00 UTC, clear-sky index",
                "at least 0.7) with probability of reverse flow above 0.5; test months, no injection.",
                "Dashed line: the 2% limit. D10 is not part of the gate.",
            ],
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        ),
    )


def main() -> None:
    """Write the SVG."""
    false_reverse_chart().save(DOCS / "false_reverse.svg")


if __name__ == "__main__":
    main()

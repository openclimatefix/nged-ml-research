# ruff: noqa: E501
"""Draw the figures of experiment 191 from the score file and the stored fits."""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00191_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments/00191-dp-price-sensitive-assets-in-the-history")
WIDTH: Final[int] = CONTENT_WIDTH_PX - 80
ARM_LABELS: Final[dict[str, str]] = {
    "h": "H: true price at forecast time (cannot be forecast)",
    "m": "M: expected price given hour, day type, quarter",
    "p": "P: last settled day's price",
    "t": "T: price-driven half-hours removed from training",
}


def chart_name(site: str) -> str:
    """Return the site's label for a chart, naming the register's Primary generically."""
    return "register's Primary" if site == common.REGISTER_PRIMARY else site


def forest() -> None:
    """Figure 1: each arm's mean absolute error minus the reference arm's, at D8 and the register's Primary."""
    report = json.loads((common.WORK_DIR / "score_gated.json").read_text())
    rows = []
    for site in common.ASSET_SITES:
        row = report[site]["all"]
        base = row["a0"]["ae"]
        for arm, label in ARM_LABELS.items():
            c = row[f"{arm}_minus_a0_ae"]
            rows.append(
                {
                    "site": f"{chart_name(site)} (reference error {base:.1f} points)",
                    "label": label,
                    "diff": c["pct"],
                    "lo": c["low"] / base * 100,
                    "hi": c["high"] / base * 100,
                }
            )
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330)
        )
    )
    layer = (
        base.mark_rule(color=ocf.DATA_BLUE, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Mean absolute error minus the reference's (% of the reference; below 0 is better)",
            ),
            x2="hi:Q",
        )
        + base.mark_point(color=ocf.DATA_BLUE, filled=True, size=90).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0], "t": ["same as reference"]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        layer.properties(width=WIDTH - 330, height=150)
        .facet(
            row=alt.Row(
                "site:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=11)
            )
        )
        .properties(
            title=alt.TitleParams(
                "At D8 and the register's Primary, no price-aware arm lowers the error clearly",
                subtitle=[
                    "Dot: estimate. Line: 95% interval from resampling the 12 months of the scored year.",
                    "Orange dashed line: same error as the reference, which has no price input.",
                ],
                fontSize=12,
                subtitleFontSize=10,
                anchor="start",
                offset=14,
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_error_against_reference.svg")


def main() -> None:  # noqa: D103
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    forest()


if __name__ == "__main__":
    main()

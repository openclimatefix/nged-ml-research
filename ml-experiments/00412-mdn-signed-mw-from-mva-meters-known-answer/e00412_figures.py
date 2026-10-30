"""Draw the first-test and variant scores of experiment 412 (aggregates only, no time series).

Reads `first_scores.json` (written by `e00412_score.py`) and writes the figure beside `result.md`.
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00412_inputs as inputs
import polars as pl
from plotting import ocf_theme as theme

OUT: Final[Path] = Path(__file__).resolve().parents[2] / (
    "docs/ml-experiments/00412-mdn-signed-mw-from-mva-meters-known-answer"
)
ARMS: Final[tuple[str, ...]] = (
    "S1c",
    "S1c-shuffle",
    "S1c-u0",
    "S1c-cfix",
    "S1c-cfix-shuffle",
    "S1c-cknown",
    "S1c-cknown-shuffle",
    "S1c-sd03",
    "S1c-sd03-shuffle",
)
MONTHS: Final[tuple[str, ...]] = ("2026-03", "2026-06")
WIDTH: Final[int] = 600


def collect() -> pl.DataFrame:
    """Return one row per arm, month, and measure."""
    scores = json.loads((inputs.WORK_DIR / "first_scores.json").read_text())
    rows = []
    for key, value in scores.items():
        arm, tag, month, _seed = key.split(" ")
        if arm not in ARMS:
            continue
        by_site = {r["site"]: r for r in value if "model" in r}
        if tag == "L2_Q2":
            pooled = by_site["pooled"]["model"]
            rows.append(
                (
                    arm,
                    month,
                    "Balanced accuracy, sign hidden (pooled D5, D10)",
                    pooled["balanced_accuracy"],
                )
            )
            rows.append(
                (arm, month, "False-reverse rate, sign hidden (pooled)", pooled["false_reverse"])
            )
        if tag == "L0_Q2" and arm in ("S1c", "S1c-cfix", "S1c-cknown", "S1c-sd03"):
            rows.append(
                (
                    arm,
                    month,
                    "False-reverse rate, no injection (D5)",
                    by_site["D5"]["model"]["false_reverse"],
                )
            )
    return pl.DataFrame(rows, schema=["arm", "month", "measure", "value"], orient="row")


def main() -> None:
    """Write the figure."""
    data = collect().to_pandas()
    colour = alt.Color(
        "month:N",
        scale=alt.Scale(domain=list(MONTHS), range=[theme.DATA_BLUE, theme.BRAND_ORANGE]),
        legend=alt.Legend(title="Test month", orient="top", direction="horizontal"),
    )
    rules = {
        "Balanced accuracy, sign hidden (pooled D5, D10)": 0.85,
        "False-reverse rate, sign hidden (pooled)": 0.05,
        "False-reverse rate, no injection (D5)": 0.01,
    }
    titles = {
        "Balanced accuracy, sign hidden (pooled D5, D10)": [
            "Balanced accuracy of the sign (higher is better; dashed: 0.85 bar)",
            "Dot: one month, one seed. Sign hidden at D5 and D10; farm level 2.0, Q = 2.",
        ],
        "False-reverse rate, sign hidden (pooled)": [
            "False-reverse rate, sign hidden (lower is better; dashed: 0.05 bar)",
            "Share of reverse-free sunny half-hours called reverse. Shuffle: solar shape permuted.",
        ],
        "False-reverse rate, no injection (D5)": [
            "Placebo at D5, no farm injected (lower is better; dashed: 0.01 bar)",
            "No true reverse flow at D5. cfix: capacity fixed; cknown: set by known Primaries.",
        ],
    }
    panels = []
    for measure, bar in rules.items():
        sub = data[data["measure"] == measure]
        points = (
            alt.Chart(sub)
            .mark_point(filled=True, size=70, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("arm:N", sort=list(ARMS), title=None),
                x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
                color=colour,
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"bar": [bar]}).to_pandas())
            .mark_rule(strokeDash=[4, 3], color=theme.GREY_1)
            .encode(x="bar:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            (points + rule).properties(
                width=WIDTH,
                height=22 * len(sub["arm"].unique()) + 20,
                title=alt.TitleParams(
                    text=titles[measure][0],
                    subtitle=titles[measure][1],
                    anchor="start",
                    fontSize=13,
                    subtitleFontSize=11,
                ),
            )
        )
    chart = alt.vconcat(*panels).resolve_scale(color="shared")
    chart.save(OUT / "figure_1_first_test_and_variants.svg")


if __name__ == "__main__":
    main()

"""Draw the figures of experiment 388 from the printed score and sweep logs.

The logs are `work/score_monthly_final.log` (arm minus baseline contrasts) and `work/sweep_run.log`
(lag-encroachment sweep). Writes SVGs to the experiment's docs folder.
"""

import re
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf  # noqa: F401  (registers and enables the theme)
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00388-xgboost-lagged-power-column-sets-weekly-demand"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 680
ARM_LABELS: Final[dict[str, str]] = {
    "a": "A: six weekly analogues",
    "b": "B: analogue mean and median",
    "c": "C: two anomaly columns",
    "e": "E: without the newest reading",
    "f": "F: strict 24-hour lag, without the newest reading",
}
ARM_COLOURS: Final[list[str]] = ["#FF4901", "#306BFF", "#10C5F7", "#B701FF", "#17E58F"]
GROUPS: Final[tuple[str, ...]] = ("signed-MW", "MVA-only with D8", "MVA-only without D8")
NUMBER = r"([+-][\d.]+)"
CONTRAST = re.compile(
    r"^(days 1-10|[\d-]+ h)\s+(signed-MW|MVA-only with D8|MVA-only without D8)\s+"
    + NUMBER
    + r" \("
    + NUMBER
    + " to "
    + NUMBER
    + r"\)"
)
X_TITLE: Final[str] = "CRPS minus the baseline's (points of capacity; below 0 is better)"


def contrasts() -> pl.DataFrame:
    """Return every `arm minus reference` contrast line of the final scoring log."""
    rows, arm = [], None
    for line in (WORK / "score_monthly_final.log").read_text().splitlines():
        head = re.match(r"^===== (\w) minus (\w+) ", line)
        if head:
            arm, reference = head.groups()
            continue
        match = CONTRAST.match(line)
        if match and arm and reference == "baseline":
            band, group, diff, low, high = match.groups()
            rows.append(
                {"arm": ARM_LABELS[arm], "band": band, "group": group, "difference": float(diff),
                 "low": float(low), "high": float(high)}
            )  # fmt: skip
    return pl.DataFrame(rows)


def contrast_chart(*, data: pl.DataFrame, band: str, title: list[str]) -> alt.Chart:
    """Return dots with interval lines, one panel per group, for one lead band."""
    base = alt.Chart(data.filter(pl.col("band") == band)).encode(
        y=alt.Y("arm:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
            legend=None,
        ),
    )
    x = alt.X(
        "difference:Q", title="CRPS minus the baseline's (points of capacity; below 0 is better)"
    )
    rule = alt.Chart(pl.DataFrame({"zero": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    layer = (
        base.mark_rule(strokeWidth=3).encode(x=alt.X("low:Q", title=X_TITLE), x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=70).encode(x=x)  # ty: ignore[unresolved-attribute]
        + rule
    )
    return (
        layer.properties(width=WIDTH - 340, height=150)
        .facet(row=alt.Row("group:N", sort=list(GROUPS), title=None))
        .properties(title=title)
    )


def sweep_chart() -> alt.Chart:
    """Return the share of a synthetic step that the median follows, by arm and step size."""
    text = (WORK / "sweep_run.log").read_text()
    rows = []
    for block in text.split("===== arm ")[1:]:
        arm = block.split(" ")[0]
        for line in block.splitlines():
            cells = [c.strip() for c in line.strip("│ ").split("┆")]
            if len(cells) == 11 and cells[1] in ("-5.0", "5.0"):
                rows.append(
                    {"arm": arm, "k": cells[0], "depth": float(cells[1]), "n": int(cells[3]),
                     "shift": float(cells[4])}
                )  # fmt: skip
    frame = (
        pl.DataFrame(rows)
        .group_by("arm", "k")
        .agg(((pl.col("shift") * pl.col("n")).sum() / pl.col("n").sum()).alias("shift"))
        .with_columns(
            arm=pl.col("arm").replace({"base": "Baseline", "a": "A", "b": "B", "c": "C"}),
            k=pl.col("k").replace({"1": "newest reading", "2": "two newest", "all": "every lag"}),
        )
    )
    return (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "k:N",
                sort=["newest reading", "two newest", "every lag"],
                title="Lags covered by the step",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("arm:N", sort=["Baseline", "A", "B", "C"]),
            y=alt.Y("shift:Q", title="Median's shift as a share of the step (1 follows the step)"),
            color=alt.Color(
                "arm:N",
                sort=["Baseline", "A", "B", "C"],
                scale=alt.Scale(
                    domain=["Baseline", "A", "B", "C"],
                    range=["#306BFF", "#FF4901", "#10C5F7", "#B701FF"],
                ),
                legend=alt.Legend(orient="top", title="Arm"),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=220,
            title=[
                "A step of plus or minus 5% of capacity is added to the lags it covers; four",
                "Primaries, days 1 to 10; 0 means the median ignores the step",
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    data = contrasts()
    contrast_chart(
        data=data,
        band="days 1-10",
        title=[
            "No extra lagged-power column lowers the CRPS at days 1 to 10; dot: estimate,",
            "line: 99.83% interval from resampling whole months; monthly refits, 16 Primaries",
        ],
    ).save(DOCS_DIR / "contrast_days_1_10.svg")
    contrast_chart(
        data=data,
        band="0-24 h",
        title=["The same contrasts at leads of 0 to 24 hours"],
    ).save(DOCS_DIR / "contrast_0_24h.svg")
    contrast_chart(
        data=data,
        band="72-168 h",
        title=["The same contrasts at leads of 72 to 168 hours"],
    ).save(DOCS_DIR / "contrast_72_168h.svg")
    sweep_chart().save(DOCS_DIR / "sweep_shift_share.svg")


if __name__ == "__main__":
    main()

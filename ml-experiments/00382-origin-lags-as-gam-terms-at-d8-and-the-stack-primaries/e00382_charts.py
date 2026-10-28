"""Draw the figures of experiment 382 from the committed `printed_report.txt`."""

import ast
import re
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf  # noqa: F401  (registers and enables the theme)
import polars as pl

HERE: Final[Path] = Path(__file__).resolve().parent
DOCS_DIR: Final[Path] = (
    HERE.parents[1]
    / "docs"
    / "ml-experiments"
    / "00382-origin-lags-as-gam-terms-at-d8-and-the-stack-primaries"
)
REPORT: Final[str] = (HERE / "printed_report.txt").read_text()
BLUE: Final[str] = "#306BFF"
ORANGE: Final[str] = "#FF4901"


def per_primary() -> pl.DataFrame:
    """Return the days-1-to-10 per-Primary contrasts (spline minus default, 95%)."""
    block = REPORT.split("Per-Primary contrasts")[1].split("days 1 to 2")[0]
    rows = re.findall(r"│\s*\d+\s*┆\s*(D\d+)\s*┆\s*(\S+)\s*┆\s*(\S+)\s*┆\s*(\S+)\s*┆", block)
    return pl.DataFrame(
        [
            {"label": a, "difference": float(b), "low": float(c), "high": float(d)}
            for a, b, c, d in rows
        ]
    )


def d8_months() -> pl.DataFrame:
    """Return the D8 month-by-month CRPS difference."""
    line = next(x for x in REPORT.splitlines() if x.startswith("D8 [("))
    values = ast.literal_eval(line[3:])
    return pl.DataFrame([{"month": m, "difference": d} for m, _a, _b, d in values])


def main() -> None:
    """Write the two figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    table = per_primary().with_columns(counted=~pl.col("label").is_in(["D15", "D18"]))
    order = sorted(table["label"], key=lambda s: int(s[1:]))
    points = (
        alt.Chart(table)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order, title=None),
            x=alt.X(
                "difference:Q",
                title="CRPS, with origin lags minus without (points; below 0 is better)",
            ),
            color=alt.condition("datum.counted", alt.value(BLUE), alt.value("#999999")),
        )
    )
    bars = (
        alt.Chart(table)
        .mark_rule()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order),
            x="low:Q",
            x2="high:Q",
            color=alt.condition("datum.counted", alt.value(BLUE), alt.value("#999999")),
        )
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    (bars + points + zero).properties(width=420, height=300).save(DOCS_DIR / "per_primary.svg")
    months = d8_months()
    line = (
        alt.Chart(months)
        .mark_line(point=True, color=ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:O", title="Month of forecast origin"),
            y=alt.Y("difference:Q", title="D8 CRPS, with origin lags minus without (points)"),
        )
    )
    (line + zero.encode(y=alt.Y("x:Q"))).properties(width=420, height=260).save(
        DOCS_DIR / "d8_by_month.svg"
    )


if __name__ == "__main__":
    main()

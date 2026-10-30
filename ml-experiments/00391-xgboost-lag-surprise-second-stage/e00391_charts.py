"""Draw the headline figure of experiment 391: each contrast against the seed noise floor.

The numbers are the 99.29% intervals printed by `e00391_score.py` (see `printed_report.txt` in
`work/scores_core/` and `work/scores_subset/`), keyed in by hand because the scorer prints them.

```bash
uv run --no-sync python ml-experiments/00391-xgboost-lag-surprise-second-stage/e00391_charts.py
```
"""

# ruff: noqa: E501
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00391-xgboost-lag-surprise-second-stage"
)
WIDTH: Final[int] = 680
# (contrast, Primaries, slice, estimate, lower, upper), CRPS in points of effective capacity.
ROWS: Final[tuple[tuple[str, str, str, float, float, float], ...]] = (
    ("Seed 1 minus seed 0 of the control (noise floor)", "16", "Event-lag rows", 0.024, -0.017, 0.097),
    ("Own and neighbour surprise (N1) minus control, seed 0", "16", "Event-lag rows", 0.110, -0.249, 0.432),
    ("Own and neighbour surprise (N1) minus control, seed 1", "16", "Event-lag rows", 0.086, -0.251, 0.404),
    ("Hand-label flags (O) minus control, seed 1", "16", "Event-lag rows", 0.194, -0.052, 0.542),
    ("Seed 1 minus seed 0 of the control (noise floor)", "16", "Rows outside events", 0.001, -0.002, 0.006),
    ("Own and neighbour surprise (N1) minus control, seed 0", "16", "Rows outside events", 0.072, 0.006, 0.145),
    ("Own and neighbour surprise (N1) minus control, seed 1", "16", "Rows outside events", 0.071, 0.006, 0.143),
    ("Hand-label flags (O) minus control, seed 1", "16", "Rows outside events", 0.012, 0.000, 0.032),
)  # fmt: skip


def main() -> None:
    """Write `headline.svg`."""
    table = pl.DataFrame(
        ROWS, schema=["contrast", "primaries", "slice", "delta", "low", "high"], orient="row"
    )
    order = list(dict.fromkeys(table["contrast"]))
    y = alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    rules = alt.Chart().mark_rule(color=ocf.DATA_BLUE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X("low:Q", title="Change in CRPS (points of effective capacity; above 0 is worse)"),
        x2="high:Q",
    )  # fmt: skip
    points = alt.Chart().mark_point(color=ocf.BRAND_ORANGE, filled=True, size=90).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="delta:Q"
    )  # fmt: skip
    zero = alt.Chart().mark_rule(color="#202020").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(rules, points, zero)
        .properties(width=WIDTH - 340, height=150)
        .facet(
            data=table,
            row=alt.Row("slice:N", sort=["Event-lag rows", "Rows outside events"], title=None),
        )
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                [
                    "Surprise inputs did not help on event-lag rows",
                    "and made the forecast worse on rows outside events",
                ],
                subtitle="Dot: mean change. Bar: 99.29% interval. The seed-noise bar is narrowest.",
                anchor="start",
            )
        )
    )
    chart.save(DOCS_DIR / "headline.svg")


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Slices and figures of experiment 513: the oracle's CRPS gain.

The slices are by Primary, half-hour of day, horizon band, and asset group.

Reads the row tables of `e00513_score.py rows`, keeps rows with no labelled event (`cls == clean`) of
the priced days, and writes `slices.csv` under the work folder plus two figures in the docs folder.
All differences are arm minus P0 in points of effective capacity (negative favours the arm).

```bash
. <env script>; uv run --no-sync python ml-experiments/00513-*/e00513_slices.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00510_common as c  # ty: ignore[unresolved-import]
import e00513_score as score
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00513-lagless-demand-network-whole-day-price-shape-at-training-time"
)
SCORES: Final[Path] = score.WORK_DIR / "scores"
ASSET_PRIMARIES: Final[tuple[str, ...]] = ("D2", "D5", "D8", "D9", "D11", "D14")
ARMS: Final[tuple[str, ...]] = ("PO", "PP", "PN")
GROUPS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six_clean": "Signed-MW six"}


def load() -> pl.DataFrame:
    """Return the scored rows with the Primary name, local half-hour of day, horizon band, and gains."""
    names = c.sites()
    table = pl.concat([pl.read_parquet(SCORES / f"rows_{m}.parquet") for m in score.FIVE])
    table = table.filter(pl.col("main16") & (pl.col("cls") == "clean"))
    time_us = (
        pl.col("target").cast(pl.Int64) * score.shapes.shape.STEP_US
        + score.shapes.prep422.GRID_START_US
    )
    local = (
        pl.from_epoch(time_us, time_unit="us")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )
    table = table.with_columns(
        name=pl.col("primary").replace_strict(dict(enumerate(names)), return_dtype=pl.String),
        half_hour=(local.dt.hour().cast(pl.Int32) * 2 + (local.dt.minute() // 30).cast(pl.Int32)),
        horizon=pl.when(pl.col("lead_hours") <= 24)
        .then(pl.lit("day 1"))
        .when(pl.col("lead_hours") <= 72)
        .then(pl.lit("days 2 to 3"))
        .otherwise(pl.lit("days 4 to 10")),
        asset=pl.col("primary").replace_strict(
            {i: n in ASSET_PRIMARIES for i, n in enumerate(names)}, return_dtype=pl.Boolean
        ),
    )
    table = table.with_columns(
        evening_peak=pl.col("half_hour").is_between(32, 37),
        overnight=pl.col("half_hour") < 10,
    )
    return table.with_columns(
        *[(pl.col(f"{a}__crps") - pl.col("P0__crps")).alias(f"gain_{a}") for a in ARMS]
    )


def slices(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the mean gain of each arm by slice and group."""
    parts = []
    for label, key in {
        "primary": "name", "half_hour": "half_hour", "horizon": "horizon",
        "asset_primaries": "asset", "evening_peak": "evening_peak", "overnight": "overnight",
    }.items():  # fmt: skip
        for group_name, group in (("eight", "mva_eight"), ("six_clean", "signed_six")):
            part = table.filter(pl.col("group") == group)
            grouped = part.group_by(key).agg(
                pl.len().alias("n"), *[pl.col(f"gain_{a}").mean() for a in ARMS]
            )
            grouped = grouped.rename({key: "value"}).with_columns(pl.col("value").cast(pl.String))
            parts.append(grouped.with_columns(slice=pl.lit(label), group=pl.lit(group_name)))
    return pl.concat(parts).select("slice", "group", "value", "n", *[f"gain_{a}" for a in ARMS])


def figures(*, out: pl.DataFrame) -> None:
    """Draw the oracle gain by Primary and by half-hour of day."""
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(domain=list(GROUPS), range=[theme.DATA_BLUE, theme.BRAND_ORANGE]),
        legend=alt.Legend(
            title=None, labelExpr="datum.value == 'eight' ? 'MVA-only eight' : 'Signed-MW six'"
        ),
    )
    prim = out.filter(pl.col("slice") == "primary").to_pandas()
    chart1 = (
        alt.Chart(prim)
        .mark_circle(size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("gain_PO:Q", title="Oracle minus control, CRPS points (negative is better)"),
            y=alt.Y("value:N", sort=list(c.sites()), title=None),
            color=colour,
        )
        .properties(width=520, height=320)
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="#808080").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    (chart1 + zero).save(DOCS_DIR / "figure_1_oracle_gain_by_primary.svg")
    hh = (
        out.filter(pl.col("slice") == "half_hour")
        .with_columns(pl.col("value").cast(pl.Int32))
        .to_pandas()
    )
    chart2 = (
        alt.Chart(hh)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="Local half-hour of day (0 is midnight)"),
            y=alt.Y("gain_PO:Q", title="Oracle minus control, CRPS points"),
            color=colour,
        )
        .properties(width=520, height=240)
    )
    chart2.save(DOCS_DIR / "figure_2_oracle_gain_by_half_hour.svg")


def main() -> None:
    """Write the slice table and the figures."""
    score.configure()
    out = slices(table=load())
    out.write_csv(SCORES / "slices.csv")
    pl.Config.set_tbl_rows(40)
    print(
        out.filter(
            pl.col("slice").is_in(["horizon", "asset_primaries", "evening_peak", "overnight"])
        ).sort("slice", "group", "value")
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures(out=out)


if __name__ == "__main__":
    main()

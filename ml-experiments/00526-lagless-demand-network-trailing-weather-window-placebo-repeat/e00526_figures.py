# ruff: noqa: E501
"""Draw the figures of experiment 526 from the per-row metrics that `e00526_score.py rows` stored.

1. `figure_1_headline.svg`: W-T against each comparator on cold nights and on all rows;
2. `figure_2_per_fold.svg`: W-T minus W-Tnull on cold nights in each winter fold, 11 seeds;
3. `figure_3_summer.svg`: the summer-fold slices (all rows and warm sunny days);
4. `figure_4_examples.svg`: a cold night, a cold sunny day, and a warm sunny day at one Primary.

Intervals in the figures are uncorrected 95% month-block bootstrap intervals (the Bonferroni-corrected
secondary-family values are in `contrasts.csv`). The example days are the first complete day of the
slice in the fold month at the first MVA-only Primary, chosen before any error was looked at.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00526_fit as fit
import e00526_score as score
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00526-lagless-demand-network-trailing-weather-window-placebo-repeat"
)
GROUPS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}
WIDTH: Final[int] = 560
COMPARATORS: Final[list[tuple[str, str, str]]] = [
    ("W-T", "W-Tnull", "W-T minus W-Tnull (5 seeds)"),
    ("W-T@11", "W-Tnull@11", "W-T minus W-Tnull (11 seeds)"),
    ("W-T", "W-noise3", "W-T minus W-noise3 (5 seeds)"),
    ("W-T", "W-null", "W-T minus W-null (5 seeds)"),
    ("W-T", "W0", "W-T minus W0 (5 seeds)"),
]  # fmt: skip
SLICES: Final[dict[str, str]] = {"cold_night": "cold nights", "all": "all rows"}
LINE_COLOURS: Final[dict[str, str]] = {"W0": theme.DATA_BLUE, "W-T": theme.BRAND_ORANGE, "W-Tnull": theme.DATA_GREEN}  # fmt: skip
EXAMPLES: Final[list[tuple[str, str, str]]] = [
    ("2026-01", "cold_night", "Cold night (2026-01)"),
    ("2026-02", "cold_sunny", "Cold sunny day (2026-02)"),
    ("2025-07", "warm_sunny", "Warm sunny day (2025-07)"),
]  # fmt: skip


def _zero(*, axis: str) -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"z": [0.0]}).to_pandas())
        .mark_rule(color="black")
        .encode(**{axis: "z:Q"})
    )  # ty: ignore[unresolved-attribute]


def _forest(
    *,
    frame: pl.DataFrame,
    rows: list[tuple[str, str, str]],
    slices: dict[str, str],
    file: str,
    title: list[str],
) -> None:
    records = []
    for a, b, label in rows:
        for g in GROUPS:
            for s, s_label in slices.items():
                r = score.contrast(frame=frame, a=a, b=b, group=g, slice_name=s)
                records.append(
                    {**r, "pair": label, "group_label": GROUPS[g], "slice_label": s_label}
                )
    table = pl.DataFrame(records, infer_schema_length=None).filter(pl.col("folds") > 0)
    panels = []
    order = [r[2] for r in rows]
    for label in GROUPS.values():
        part = table.filter(pl.col("group_label") == label).to_pandas()
        base = alt.Chart(part).encode(
            y=alt.Y("pair:N", sort=order, title=None), yOffset="slice_label:N"
        )  # ty: ignore[unresolved-attribute]
        colour = alt.Color("slice_label:N", scale=alt.Scale(domain=list(slices.values()), range=[theme.BRAND_ORANGE, theme.DATA_BLUE][: len(slices)]), legend=alt.Legend(title=None, orient="top"))  # fmt: skip
        bar = base.mark_rule(strokeWidth=3).encode(
            x=alt.X(
                "lo:Q", title="CRPS difference (points of capacity; negative favours the first arm)"
            ),
            x2="hi:Q",
            color=colour,
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=60).encode(x="contrast:Q", color=colour)  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(_zero(axis="x"), bar, dot).properties(width=WIDTH, height=170, title=alt.TitleParams(label, anchor="start")))  # fmt: skip
    alt.vconcat(*panels).properties(title=alt.TitleParams(title, anchor="start")).save(
        DOCS_DIR / file
    )


def _per_fold(*, frame: pl.DataFrame) -> None:
    records = []
    for g in ("eight", "six"):
        mask = score.mask_of(frame=frame, group=g, slice_name="cold_night")
        a = score.fold_table(frame=frame, name="W-T@11", metric="crps", mask=mask)
        b = score.fold_table(frame=frame, name="W-Tnull@11", metric="crps", mask=mask)
        records.extend(
            {"group": GROUPS[g], "fold": m, "contrast": (a[m][0] - b[m][0]) / a[m][1]}
            for m in sorted(a)
            if a[m][1]
        )
    table = pl.DataFrame(records)
    panels = []
    for label in GROUPS.values():
        part = table.filter(pl.col("group") == label).to_pandas()
        dots = (
            alt.Chart(part)
            .mark_point(filled=True, size=70, color=theme.BRAND_ORANGE)
            .encode(
                x=alt.X("fold:N", title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("contrast:Q", title="W-T minus W-Tnull, cold nights (points)"),
            )
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(_zero(axis="y"), dots).properties(width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start")))  # fmt: skip
    alt.vconcat(*panels).properties(title=alt.TitleParams(["W-T minus W-Tnull on cold nights in each winter fold, 11 seeds.", "Negative favours W-T."], anchor="start")).save(DOCS_DIR / "figure_2_per_fold.svg")  # fmt: skip


def _examples() -> pl.DataFrame:
    """Return the example forecasts: quantiles of W0, W-T, and W-Tnull (seed means) and the actual."""
    eight = prep.primaries().filter(
        ~pl.col("site").is_in([*score.score404.SIX_SIGNED, "D15", "D18"])
    )
    site = eight.head(1)
    clean = score.c.clean_stack()
    p_idx = int(site["primary"][0])
    ts_id, name = int(site["time_series_id"][0]), site["site"][0]
    ids = ["time_series_id", "power_fcst_init_time", "valid_time"]
    parts = []
    for month, slice_name, label in EXAMPLES:
        rows = pl.read_parquet(score.SCORE_DIR / f"rows_{month}.parquet").filter(
            (pl.col("time_series_id") == ts_id) & pl.col(slice_name)
        )
        local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
        days = rows.with_columns(day=local.dt.date()).group_by("power_fcst_init_time", "day").agg(pl.len().alias("n")).filter(pl.col("n") >= 16).sort("power_fcst_init_time", "day")  # fmt: skip
        init, day = days["power_fcst_init_time"][0], days["day"][0]
        window = rows.filter((pl.col("power_fcst_init_time") == init)).select(*ids, "actual")
        start = window["valid_time"].min()
        full = pl.read_parquet(fit.stored_forecast(arm="W0", month=month, seed=0)).filter((pl.col("time_series_id") == ts_id) & (pl.col("power_fcst_init_time") == init)).select(*ids)  # fmt: skip
        full = full.filter(pl.col("valid_time") >= start).head(80)
        valid_idx = (
            (full["valid_time"].dt.epoch("us").to_numpy() - score.c.prep422.GRID_START_US)
            // score.c.prep422.STEP_US
        ).astype(np.int64)
        actual = full.with_columns(actual=pl.Series(clean[p_idx, valid_idx].astype(np.float64)))
        for arm in ("W0", "W-T", "W-Tnull"):
            q = []
            for s in score.CORE_SEEDS:
                f = pl.read_parquet(fit.stored_forecast(arm=arm, month=month, seed=s)).select(
                    *ids, "q05", "q50", "q95"
                )
                q.append(full.join(f, on=ids, how="left"))
            avg = (
                q[0]
                .select(*ids)
                .hstack(
                    pl.DataFrame(
                        {
                            c: np.mean([x[c].to_numpy() for x in q], axis=0)
                            for c in ("q05", "q50", "q95")
                        }
                    )
                )
            )
            avg = avg.join(actual, on=ids, how="left").with_columns(
                arm=pl.lit(arm), panel=pl.lit(label), site=pl.lit(name), day=pl.lit(str(day))
            )
            parts.append(avg)
    return pl.concat(parts)


def _example_figure() -> None:
    table = _examples().with_columns(
        (pl.col(c) * 100).alias(c) for c in ("q05", "q50", "q95", "actual")
    )
    table.write_csv(score.SCORE_DIR / "example_forecasts.csv")
    panels = []
    order = [e[2] for e in EXAMPLES]
    for label in order:
        part = table.filter(pl.col("panel") == label).to_pandas()
        base = alt.Chart(part).encode(x=alt.X("valid_time:T", title="Valid time (UTC)"))  # ty: ignore[unresolved-attribute]
        colour = alt.Color("arm:N", scale=alt.Scale(domain=list(LINE_COLOURS), range=list(LINE_COLOURS.values())), legend=alt.Legend(title=None, orient="top"))  # fmt: skip
        band = (
            base.transform_filter(alt.datum.arm == "W-T")
            .mark_area(opacity=0.18, color=theme.BRAND_ORANGE)
            .encode(y="q05:Q", y2="q95:Q")
        )  # ty: ignore[unresolved-attribute]
        lines = base.mark_line(strokeWidth=2).encode(
            y=alt.Y("q50:Q", title="Demand (% of capacity)", scale=alt.Scale(zero=False)),
            color=colour,
        )  # ty: ignore[unresolved-attribute]
        actual = (
            base.transform_filter(alt.datum.arm == "W0")
            .mark_line(color="black", strokeDash=[4, 3])
            .encode(y="actual:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(band, lines, actual).properties(width=WIDTH, height=150, title=alt.TitleParams(f"{label}, {part['site'].iloc[0]}, first complete day {part['day'].iloc[0]}", anchor="start")))  # fmt: skip
    alt.vconcat(*panels).properties(title=alt.TitleParams(["Median of three arms (5-seed mean), W-T 90% band shaded, actual dashed.", "One forecast run, its first 80 steps; the first complete day of each slice, chosen before any error was read."], anchor="start")).save(DOCS_DIR / "figure_4_examples.svg")  # fmt: skip


def main() -> None:
    """Draw the figures."""
    frame = score.load_rows()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    _forest(frame=frame, rows=COMPARATORS, slices=SLICES, file="figure_1_headline.svg", title=["W-T against each comparator over 10 winter folds, rows with no labelled event.", "Dot: pooled difference; bar: uncorrected 95% month-block bootstrap interval (near min-to-max)."])  # fmt: skip
    _per_fold(frame=frame)
    summer = {"warm_sunny": "warm sunny days", "all_summer": "all summer rows"}
    _forest(frame=frame, rows=[("W-T", "W-Tnull", "W-T minus W-Tnull"), ("W-T", "W-noise3", "W-T minus W-noise3"), ("W-T", "W0", "W-T minus W0")], slices=summer, file="figure_3_summer.svg", title=["W-T on the 4 summer folds (5 seeds); 4 month blocks, so intervals are near min-to-max.", "Negative favours W-T."])  # fmt: skip
    _example_figure()


if __name__ == "__main__":
    main()

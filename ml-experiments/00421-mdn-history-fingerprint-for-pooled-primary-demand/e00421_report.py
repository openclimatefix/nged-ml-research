"""Print the tail calibration, the median-minus-mean checks, and draw the headline figure of 421.

Run with the `PYTHONPATH` of `e00421_score.py`. The mean forecast is approximated by the mean of
the 11 stored quantiles, because the networks store quantiles and not a mean.
"""

from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00421_score as sc  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics

from studies import tail_calibration

ARMS: Final[tuple[str, ...]] = ("E406", "E3g", "N3g", "NH", "NHs", "EH", "NH0")
WIDEN: Final[float] = 1.15


def widened(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame with q01, q05, q95, and q99 widened about the median by 1.15."""
    return frame.with_columns(
        *[
            (pl.col("q50") + WIDEN * (pl.col(c) - pl.col("q50"))).alias(c)
            for c in ("q01", "q05", "q95", "q99")
        ]
    )


def main() -> None:
    """Print the tables and draw the figure."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    label_of = dict(zip(series["time_series_id"].to_list(), series["label"].to_list(), strict=True))
    actuals = (
        ctx["actuals"].rename({"power": "actual"})
        if "power" in ctx["actuals"].columns
        else ctx["actuals"]
    )
    frames = {
        n: adapter.forecast_frame(stored=sc.seed_average(name=n), series=series, unit="fraction")
        for n in ARMS
    }
    for group, labels in sc.SUBSETS.items():
        ids = [i for i, label in label_of.items() if label in labels]
        print(f"\n== {group}: tails (coverage and pinball, all leads), raw / widened ==")
        for n, f in frames.items():
            sub = f.filter(pl.col("time_series_id").is_in(ids)).join(
                actuals.select("time_series_id", "valid_time", "actual"),
                on=["time_series_id", "valid_time"],
                how="inner",
            )
            sub = sub.filter(pl.col("actual").is_not_nan() & pl.col("actual").is_not_null())
            for tag, frame in (("raw", sub), ("x1.15", widened(frame=sub))):
                t = tail_calibration.tail_calibration_table(frame=frame)
                cov = {r["level"]: round(r["coverage"], 3) for r in t.iter_rows(named=True)}
                pin = {r["level"]: round(r["pinball"], 4) for r in t.iter_rows(named=True)}
                print(f"{n:6s} {tag:6s} coverage {cov} pinball {pin}")
    print("\n== median minus mean of the 11 quantiles, per Primary (MW), mean over the arm ==")
    for n in ("E3g", "NH", "EH"):
        f = frames[n].with_columns(gap=pl.col("q50") - pl.mean_horizontal(*common.LEVEL_COLUMNS))
        by = f.group_by("time_series_id").agg(pl.col("gap").mean()).sort("time_series_id")
        print(n, {label_of[i]: round(g, 3) for i, g in by.iter_rows()})
    print("\n== mean absolute error of the median, points of capacity, by group ==")
    for n, f in frames.items():
        out = {}
        for group, labels in sc.SUBSETS.items():
            ids = [i for i, label in label_of.items() if label in labels]
            sub = f.filter(pl.col("time_series_id").is_in(ids)).join(
                actuals.select("time_series_id", "valid_time", "actual"),
                on=["time_series_id", "valid_time"],
                how="inner",
            )
            cap = series.filter(pl.col("time_series_id").is_in(ids))["capacity_mw"].mean()
            sub = sub.filter(pl.col("actual").is_not_nan() & pl.col("actual").is_not_null())
            out[group] = round(float((sub["q50"] - sub["actual"]).abs().mean() / cap * 100), 3)
        print(f"{n:6s}", out)
    draw_headline()


def draw_headline() -> None:
    """Draw the contrasts of the planned plan with their Bonferroni intervals."""
    import altair as alt
    from plotting import ocf_theme as theme

    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(sc.FLAGGED_KEYS).select("time_series_id", "valid_time")
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    frames = {
        n: adapter.forecast_frame(stored=sc.seed_average(name=n), series=series, unit="fraction")
        for n in ARMS
    }
    table, _scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    data = []
    for name, pair in sc.PLAN.items():
        if pair[1] not in rows or pair[0] not in rows or name.startswith("info"):
            continue
        for group, labels in list(sc.SUBSETS.items())[:2]:
            r = metrics.month_bootstrap_contrast(
                treatment=rows[pair[0]],
                reference=rows[pair[1]],
                metric="crps",
                where=pl.col("label").is_in(list(labels)) & pl.col(sc.HEADLINE),
                level_percent=sc.BONFERRONI_LEVEL,
                n_resamples=20_000,
                slice_table=table,
            )
            data.append(
                {
                    "contrast": name,
                    "group": group,
                    **{k: r[k] for k in ("difference", "lower", "upper")},
                }
            )
    frame = pl.DataFrame(data).to_pandas()
    y = alt.Y("contrast:N", sort=list(sc.PLAN), title=None)
    off = alt.YOffset("group:N")
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(
            domain=list(list(sc.SUBSETS)[:2]), range=[theme.DATA_BLUE, theme.BRAND_ORANGE]
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    bars = (
        alt.Chart(frame)
        .mark_rule(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=off,
            x=alt.X(
                "lower:Q",
                title="Treatment minus reference, CRPS days 1 to 10 (points of capacity; below zero is better)",
            ),
            x2="upper:Q",
            color=colour,
        )
    )
    dots = (
        alt.Chart(frame)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y, yOffset=off, x="difference:Q", color=colour
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = (bars + dots + zero).properties(
        width=480,
        height=300,
        title=alt.TitleParams(
            [
                "Planned contrasts of the history-fingerprint arms",
                "Dot: estimate. Line: 99.7% interval from resampling whole months.",
            ],
            anchor="start",
        ),
    )
    chart.save(sc.FIGURE_DIR / "figure_1_contrasts.svg")


if __name__ == "__main__":
    main()

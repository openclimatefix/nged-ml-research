# ruff: noqa: E501
"""Experiment 482: real-event charts with the NRA-corrected actual, the lags, and the surprise inputs.

Extends experiment 468's real-event charts (same events, same stored experiment 454 forecasts, seed
0). Each chart has three panels sharing the x-axis (days since the event began):

1. the actual, the actual minus the normal-running-arrangement correction of experiment 435 (labelled
   events only, where the correction is not zero), the day-ahead medians of the lagless network,
   S1, B-mix, and K2-S1, and K2-S1's 90% band;
2. the three lags the networks read for each day-ahead row (newest, same time latest day, same time
   latest week), in points of capacity;
3. the surprise of each lag: the lag minus experiment 404's lagless median at the lag's own
   half-hour (the quantity S1 reads), in points of capacity.

```bash
source <paths of ml-experiments/*/ on PYTHONPATH>; export DATA_PATH_INTERNAL=<private data folder>
uv run --no-sync python ml-experiments/00482-*/e00482_real_events.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00454_score as score454  # ty: ignore[unresolved-import]
import e00468_part_b_real as real468  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00482-best-models-performance-analysis-injected-shifts-and-wind-references"
    / "real_events"
)
WIDTH: Final[int] = 680
LAG_NAMES: Final[tuple[str, ...]] = (
    "newest lag",
    "same time yesterday-or-earlier",
    "same time last week",
)
LAG_COLOURS: Final[dict[str, str]] = {
    LAG_NAMES[0]: ocf.BRAND_ORANGE, LAG_NAMES[1]: ocf.DATA_BLUE, LAG_NAMES[2]: ocf.DATA_GREEN
}  # fmt: skip


def lagless_median(*, month: str, shape: tuple[int, int]) -> np.ndarray:
    """Return experiment 404's seed-0 lagless median on the grid (NaN where unknown)."""
    table = pl.read_parquet(lagless.table_path(month=month, seed=0))
    median = np.full(shape, np.nan, dtype=np.float32)
    median[
        table["primary"].to_numpy().astype(np.int64), table["grid"].to_numpy().astype(np.int64)
    ] = table["q50"].to_numpy()
    return median


def figure(
    *, number: int, ev: dict, sites: list[str], bounds: dict, power: np.ndarray, nra: np.ndarray
) -> dict:
    """Draw one event's three-panel chart and return its summary numbers."""
    site, onset, end = ev["site"], ev["onset"], ev["end"]
    primary = sites.index(site)
    month = real468.month_for(onset=onset, end=end, bounds=bounds)
    lo, hi = onset - real468.MARGIN, end + real468.MARGIN
    frames = {
        a: real468.day_ahead(arm=a, month=month, primary=primary, lo=lo, hi=hi)
        for a in real468.ARMS
    }
    base = frames["K2-S1"]
    for a, f in frames.items():
        assert np.array_equal(f["origin"].to_numpy(), base["origin"].to_numpy()), (
            f"{a}: origins differ"
        )
    target = base["target"].to_numpy().astype(np.int64)
    origin = base["origin"].to_numpy().astype(np.int64)
    lag_grid = rows_mod.lag_grids(target=target, origin=origin)
    median = lagless_median(month=month, shape=power.shape)
    lag_value = np.where(
        lag_grid >= 0, power[primary, np.clip(lag_grid, 0, power.shape[1] - 1)], np.nan
    )
    surprise = lag_value - np.where(
        lag_grid >= 0, median[primary, np.clip(lag_grid, 0, power.shape[1] - 1)], np.nan
    )
    day = (target - onset) / 48.0
    x_domain = [float(day.min()), float(day.max())]
    x = alt.X("day:Q", title=None, scale=alt.Scale(domain=x_domain))
    correction = nra[primary, target]
    has_nra = bool(np.abs(correction).max() > 0)
    lines = [base.select("target", median=pl.col("actual") * 100, series=pl.lit("Actual"))]
    if has_nra:
        lines.append(
            base.select(
                "target",
                median=(pl.col("actual") - pl.Series(correction)) * 100,
                series=pl.lit("Actual with the normal running arrangement restored"),
            )
        )
    for a, label in real468.ARMS.items():
        lines.append(frames[a].select("target", median=pl.col("q50") * 100, series=pl.lit(label)))
    lines_df = pl.concat(lines).with_columns(day=(pl.col("target") - onset) / 48.0)
    colours = {**real468.COLOURS, "Actual with the normal running arrangement restored": ocf.GREY_1}
    band = base.select("target", low=pl.col("q05") * 100, high=pl.col("q95") * 100).with_columns(
        day=(pl.col("target") - onset) / 48.0
    )
    shade_df = pl.DataFrame({"a": [0.0], "b": [min(end - onset, hi - onset) / 48.0]})

    def shade() -> alt.Chart:
        return (
            alt.Chart(shade_df)  # ty: ignore[unresolved-attribute]
            .mark_rect(color=ocf.GREY_3, opacity=0.35, clip=True)
            .encode(x="a:Q", x2="b:Q")
        )  # ty: ignore[unresolved-attribute]

    area = (
        alt.Chart(band)  # ty: ignore[unresolved-attribute]
        .mark_area(color=ocf.BRAND_ORANGE, opacity=0.18)
        .encode(x=x, y=alt.Y("low:Q", title="Power (% of capacity)"), y2="high:Q")
    )  # ty: ignore[unresolved-attribute]
    line = (
        alt.Chart(lines_df)  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False)
        .encode(
            x=x, y="median:Q",
            color=alt.Color("series:N", scale=alt.Scale(domain=list(colours), range=list(colours.values())), legend=alt.Legend(title=None, orient="top", columns=3)),
            strokeWidth=alt.condition(alt.datum.series == "Actual", alt.value(2.0), alt.value(1.2)),
        )
    )  # fmt: skip
    top = (shade() + area + line).properties(width=WIDTH - 110, height=170)

    def long(values: np.ndarray) -> pl.DataFrame:
        return pl.concat(
            [
                pl.DataFrame({"day": day, "v": values[:, i] * 100, "lag": LAG_NAMES[i]})
                for i in range(3)
            ]
        )

    lag_chart = (
        alt.Chart(long(lag_value))  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False)
        .encode(
            x=x,
            y=alt.Y("v:Q", title="Lag (% of capacity)"),
            color=alt.value(ocf.GREY_1),
            strokeDash=alt.StrokeDash(
                "lag:N",
                scale=alt.Scale(domain=list(LAG_NAMES), range=[[1, 0], [6, 3], [2, 2]]),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
        )
        .properties(width=WIDTH - 110, height=110)
    )
    sur_chart = (
        alt.Chart(long(surprise))  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False)
        .encode(
            x=alt.X(
                "day:Q",
                title="Days since the event (or step) began; shaded: the event",
                scale=alt.Scale(domain=x_domain),
            ),
            y=alt.Y("v:Q", title="Surprise (points of capacity)"),
            color=alt.value(ocf.GREY_1),
            strokeDash=alt.StrokeDash(
                "lag:N",
                scale=alt.Scale(domain=list(LAG_NAMES), range=[[1, 0], [6, 3], [2, 2]]),
                legend=None,
            ),
        )
        .properties(width=WIDTH - 110, height=110)
    )
    err = {}
    for a in real468.ARMS:
        f = frames[a].filter((pl.col("target") >= onset) & (pl.col("target") <= end))
        err[a] = float((f["q50"] - f["actual"]).abs().mean() * 100) if f.height else float("nan")  # ty: ignore[unsupported-operator]
    scan = ev["source"].startswith("scan")
    title = alt.TitleParams(
        [
            f"Primary {site}, {ev['source']}: day-ahead forecasts (12 to 36 hours ahead), {month} fit.",
            f"Mean absolute error of the median during the event, points of capacity: lagless {err['LL']:.1f}, S1 {err['S1']:.1f}, B-mix {err['B-mix']:.1f}, K2-S1 {err['K2-S1']:.1f}.",
            "Middle: the three lags the networks read. Bottom: each lag minus the lagless median at its half-hour (S1's surprise).",
            *(["The event may be a load shift, a level change, or neither."] if scan else []),
        ],
        anchor="start",
    )
    chart = alt.vconcat(top, lag_chart, sur_chart, spacing=4).properties(title=title)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "png").mkdir(exist_ok=True)
    name = f"event_{number:02d}_{site}_{onset}"
    chart.save(OUT / f"{name}.svg")
    chart.save(OUT / "png" / f"{name}.png", scale_factor=1.5)
    inside = (lag_grid >= onset) & (lag_grid <= end)
    return dict(
        number=number, site=site, onset=onset, end=end, month=month, source=ev["source"], klass=ev["klass"],
        n_halfhours=len(day), has_nra=has_nra,
        max_abs_surprise_in_event=float(np.nanmax(np.abs(surprise[inside.any(axis=1)])) * 100) if inside.any() else float("nan"),
        **{f"mae_during_{a}": v for a, v in err.items()},
    )  # fmt: skip


def main() -> None:
    """Draw every real-event chart."""
    sites = common.series_table().filter(pl.col("substation_type") == "Primary")["site"].to_list()
    bounds = real468.month_bounds()
    events = real468.pick_events(sites=sites, bounds=bounds)
    power, _ = fit.world_power(seed=None)
    nra = np.load(score454.NRA_FILE)["f1.0"]
    out = [
        figure(number=i, ev=ev, sites=sites, bounds=bounds, power=power, nra=nra)
        for i, ev in enumerate(events.iter_rows(named=True), start=1)
    ]
    table = pl.DataFrame(out)
    table.write_csv(OUT / "events_summary.csv")
    print(table)


if __name__ == "__main__":
    main()

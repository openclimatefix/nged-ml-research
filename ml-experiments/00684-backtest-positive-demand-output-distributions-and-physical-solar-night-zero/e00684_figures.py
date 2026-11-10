# ruff: noqa: E501, D103
"""Experiment 684 figures and bootstrap intervals (CPU).

Reads the arms' forecasts (experiment 667's stored rows and this experiment's fits) and writes the
SVGs of the page to `OUT_DIR`, and prints the paired bootstrap of the CRPS difference to the
control per series and arm (resampling origin days, 2000 draws, pooled over the two folds).

    PYTHONPATH=ml-experiments/00666-*:ml-experiments/00667-*:ml-experiments/00684-* \
        uv run --no-sync python ml-experiments/00684-*/e00684_figures.py OUT_DIR
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import altair as alt
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00667_export as exp  # ty: ignore[unresolved-import]
import e00667_extend as ext  # ty: ignore[unresolved-import]
import e00667_score as sc  # ty: ignore[unresolved-import]
import e00684_demand_score as ds
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

ARMS = ("control_667", "tgauss", "gamma")
LABEL = {
    "control_667": "Stored (Gaussian)",
    "tgauss": "Truncated Gaussian",
    "gamma": "Gamma mixture",
}
COLOUR = {"control_667": theme.DATA_BLUE, "tgauss": theme.BRAND_ORANGE, "gamma": "#B701FF"}
IDS = (14, 15, 26, 17)
W = 640


def bootstrap() -> pl.DataFrame:
    """Return the CRPS difference to the control (points of capacity) with a 95% interval, per series and arm."""
    ext.redirect()
    g0, step = c510.prep422.GRID_START_US, c510.prep422.STEP_US
    clean = sc.observed_stack(grid_start_us=g0, step_us=step)
    per: dict[str, pl.DataFrame] = {}
    for arm in ARMS:
        parts = []
        for month in ds.MONTHS:
            frame = ds.arm_frame(arm=arm, month=month)
            s = sc.score_frame(frame=frame, clean=clean, grid_start_us=g0, step_us=step)
            parts.append(
                s.with_columns(valid=frame["valid_time"], day=frame["power_fcst_init_time"]).filter(
                    pl.col("observed").is_finite()
                )
            )
        per[arm] = pl.concat(parts)
    rng = np.random.default_rng(0)
    rows = []
    for ts in IDS:
        base = per["control_667"].filter(pl.col("time_series_id") == ts)
        for arm in ("tgauss", "gamma"):
            other = per[arm].filter(pl.col("time_series_id") == ts)
            joined = base.join(other, on=["time_series_id", "day", "valid"], suffix="_arm")
            diff = (
                joined.group_by("day")
                .agg(
                    (pl.col("crps_pts_arm") - pl.col("crps_pts")).sum().alias("s"),
                    pl.len().alias("n"),
                )
                .sort("day")
            )
            sums, counts = diff["s"].to_numpy(), diff["n"].to_numpy()
            idx = rng.integers(0, len(sums), size=(2000, len(sums)))
            boot = sums[idx].sum(1) / counts[idx].sum(1)
            rows.append(
                {
                    "series": ts,
                    "arm": arm,
                    "diff": sums.sum() / counts.sum(),
                    "lo": float(np.quantile(boot, 0.025)),
                    "hi": float(np.quantile(boot, 0.975)),
                    "origin_days": len(sums),
                }
            )
    table = pl.DataFrame(rows)
    print(table)
    return table


def headline(table: pl.DataFrame, out: Path) -> None:
    d = table.with_columns(
        series=pl.col("series").cast(pl.String), arm=pl.col("arm").replace(LABEL)
    )
    base = alt.Chart(
        d,
        width=W,
        height=200,
        title=alt.TitleParams(
            [
                "Change in CRPS against the stored forecasts",
                "Points of capacity, 95% interval over origin days; below zero is better",
            ],
            anchor="start",
            fontSize=12,
            subtitleFontSize=11,
        ),
    )  # ty: ignore[unresolved-attribute]
    pts = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("diff:Q", title="CRPS difference"),
        y=alt.Y("series:N", title="Primary"),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=[LABEL["tgauss"], LABEL["gamma"]], range=[COLOUR["tgauss"], COLOUR["gamma"]]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        yOffset="arm:N",
    )
    bars = base.mark_rule(strokeWidth=1).encode(  # ty: ignore[unresolved-attribute]
        x="lo:Q",
        x2="hi:Q",
        y="series:N",
        color=alt.Color(
            "arm:N",
            legend=None,
            scale=alt.Scale(
                domain=[LABEL["tgauss"], LABEL["gamma"]], range=[COLOUR["tgauss"], COLOUR["gamma"]]
            ),
        ),
        yOffset="arm:N",
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888", strokeWidth=1).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    (bars + pts + zero).save(out / "crps_change.svg")


def fans(out: Path) -> None:
    ext.redirect()
    g0, step = c510.prep422.GRID_START_US, c510.prep422.STEP_US
    clean = sc.observed_stack(grid_start_us=g0, step_us=step)
    sm = __import__("e00578_common").sites()
    for ts, origin, hours, name in (
        (14, datetime(2026, 2, 10, 12, tzinfo=UTC), 48, "day_ahead_14"),
        (14, datetime(2026, 2, 10, 12, tzinfo=UTC), 336, "fourteen_days_14"),
        (26, datetime(2026, 2, 10, 12, tzinfo=UTC), 336, "fourteen_days_26"),
    ):
        rec = sm.filter(pl.col("time_series_id") == ts).row(0, named=True)
        frames = []
        for arm in ARMS:
            f = ds.arm_frame(arm=arm, month="2026-02").filter(
                (pl.col("time_series_id") == ts) & (pl.col("power_fcst_init_time") == origin)
            )
            f = f.with_columns(
                hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                / 60.0
            ).filter(pl.col("hours") <= hours)
            idx = ((f["valid_time"].dt.epoch("us").to_numpy() - g0) // step).astype(int)
            actual = clean[rec["primary"], idx] * rec["capacity_mw"]
            frames.append(f.with_columns(arm=pl.lit(LABEL[arm]), actual=pl.Series(actual)))
        d = pl.concat(frames).with_columns(zero=pl.lit(0.0))
        base = alt.Chart(d, width=W, height=170)  # ty: ignore[unresolved-attribute]
        x = alt.X("hours:Q", title="Hours since the forecast origin")
        y = alt.Y("q50:Q", title="MVA" if rec["group"] else "MW")
        band90 = base.mark_area(opacity=0.25, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=x, y="q05:Q", y2="q95:Q"
        )
        band50 = base.mark_area(opacity=0.35, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=x, y="q25:Q", y2="q75:Q"
        )
        med = base.mark_line(color=theme.DATA_BLUE, strokeWidth=1).encode(x=x, y=y)  # ty: ignore[unresolved-attribute]
        act = base.mark_line(color="#222", strokeWidth=1).encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        zero = base.mark_rule(color="#888", strokeDash=[3, 3], strokeWidth=1).encode(y="zero:Q")  # ty: ignore[unresolved-attribute]
        (band90 + band50 + med + act + zero).facet(
            row=alt.Row("arm:N", title=None, sort=[LABEL[a] for a in ARMS])
        ).save(out / f"{name}.svg")


def ladders(out: Path) -> None:
    """Return nothing; write the 11 quantiles of each arm at two example half-hours of Primary 26 (the fitted shape)."""
    rows = []
    for tag, hour in (("night trough", 3), ("evening peak", 18)):
        for arm in ARMS:
            f = ds.arm_frame(arm=arm, month="2026-02").filter(
                (pl.col("time_series_id") == 26)
                & (pl.col("power_fcst_init_time") == datetime(2026, 2, 10, 12, tzinfo=UTC))
                & (pl.col("valid_time").dt.hour() == hour)
                & (pl.col("valid_time").dt.minute() == 0)
                & (pl.col("valid_time").dt.day() == 11)
            )
            q = f.select(exp.QCOLS).row(0)
            rows += [
                {"example": tag, "arm": LABEL[arm], "level": lv, "value": v}
                for lv, v in zip(sc.TAUS, q, strict=True)
            ]
    d = pl.DataFrame(rows)
    chart = (
        alt.Chart(d, width=300, height=220)
        .mark_line(point=True, strokeWidth=1)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("level:Q", title="Quantile level"),
            y=alt.Y("value:Q", title="MVA"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=[LABEL[a] for a in ARMS], range=[COLOUR[a] for a in ARMS]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .facet(column=alt.Column("example:N", title=None))
    )
    chart.save(out / "quantile_ladders_26.svg")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    table = bootstrap()
    table.write_csv(ds.ROOT_684 / "score" / "crps_bootstrap.csv")
    headline(table, out)
    fans(out)
    ladders(out)


if __name__ == "__main__":
    main()

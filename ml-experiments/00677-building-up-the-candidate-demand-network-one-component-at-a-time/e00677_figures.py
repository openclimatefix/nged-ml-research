# ruff: noqa: E501, D103
"""Experiment 677: figures of the ladder (SVG, public) from the private score tables.

Usage: python e00677_figures.py SPAN OUT_DIR
Reads `scores/step<N>_<version>_sp<SPAN>_s012/scores.csv`, `scores/contrasts_sp<SPAN>.csv`, and the exported
forecast files; writes `ladder_horizon.svg`, `ladder_contrasts.svg`, and the example time series.
Only Primaries appear, so the figures carry no generator series.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import e00677_export as exp
import e00677_score as sc
import numpy as np
import polars as pl

STEPS = ("1", "2", "3a", "3b", "4", "5w", "6w", "7w", "8w")
LABEL = {
    "1": "1 date-time", "2": "2 weather", "3a": "3a both (ENS only)", "3b": "3b both (pre-trained)",
    "4": "4 + 1-day lag", "5w": "5 + 1-week lag", "6w": "6 + gap", "7w": "7 + averages", "8w": "8 + neighbours (Candidate)",
}  # fmt: skip
RAMP = ["#C8D9FF", "#A9C3FF", "#8AADFF", "#6B96FF", "#4C80FF", "#306BFF", "#2252C9", "#173A94", "#0B215F"]
OTHER = {"Manual heuristic": "#FF4901", "Climatology": "#7F7F7F"}
WIDTH, HEIGHT = 620, 320


def scores_of(*, span: int, step: str) -> pl.DataFrame:
    tag = "s01234" if step[0] in "5678" else "s012"
    folder = f"step{step}_{exp.VERSION[step]}_sp{span}_{tag}"
    return pl.read_csv(sc.ROOT / "scores" / folder / "scores.csv").with_columns(step_label=pl.lit(LABEL.get(step, step)))


def horizon(*, span: int) -> alt.Chart:
    frames = []
    for s in STEPS:
        f = scores_of(span=span, step=s).filter((pl.col("slice") == "lead day") & (pl.col("metric") == "mae"))
        frames.append(f.filter(pl.col("system") == "Model").with_columns(system=pl.lit(LABEL[s])))
        if s == STEPS[-1]:
            frames.append(f.filter(pl.col("system") != "Model"))
    frame = pl.concat(frames, how="diagonal").with_columns(day=pl.col("group").cast(pl.Int32) + 0.5).to_pandas()
    domain = [LABEL[s] for s in STEPS] + list(OTHER)
    return (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("day:Q", title="Forecast horizon (days after the forecast origin)", axis=alt.Axis(values=[0, 2, 4, 6, 8, 10, 12, 14]), scale=alt.Scale(domain=[0, 14])),
            y=alt.Y("mean:Q", title="Median-forecast MAE (points of capacity; smaller is better)", scale=alt.Scale(zero=False)),
            color=alt.Color("system:N", title=None, scale=alt.Scale(domain=domain, range=[*RAMP, *OTHER.values()]), legend=alt.Legend(orient="right", columns=1)),
        )
        .properties(width=WIDTH, height=HEIGHT, title=alt.TitleParams(["Normalised MAE of the median forecast by horizon, each ladder step", "Backtest origins 12:00 UTC, refits every 3 months, 16 Primaries; steps with injected events; mean of seeds 0 to 2 (steps 1 to 4) and 0 to 4 (steps 5 to 8)."], anchor="start", fontSize=13))
    )


def contrasts(*, span: int) -> alt.Chart:
    c = pl.read_csv(sc.ROOT / "scores" / f"contrasts_sp{span}_s01234.csv").filter((pl.col("set") == "all rows") & (pl.col("part") == "all"))
    c = c.with_columns(contrast=pl.col("from").str.replace("step", "") + " to " + pl.col("to").str.replace("step", "")).to_pandas()
    base = alt.Chart(c).encode(y=alt.Y("contrast:N", title=None, sort=None), color=alt.Color("metric:N", title=None, scale=alt.Scale(domain=["mae", "crps"], range=["#306BFF", "#FF4901"])), yOffset="metric:N")
    return (base.mark_point(filled=True, size=70).encode(x=alt.X("diff:Q", title="Change in score when the component is added (points of capacity; below 0 is better)")) + base.mark_rule(strokeWidth=2).encode(x="lo:Q", x2="hi:Q")).properties(width=WIDTH, height=300, title=alt.TitleParams(["Change in score from each added component", "Dot: mean change. Line: 95% interval from resampling 7-day blocks of origin days. MAE: median error; CRPS: whole distribution."], anchor="start", fontSize=13))


def example(*, span: int, label: str, start: datetime, stop: datetime, series: int, steps: tuple[str, ...], lead_day: int) -> alt.Chart:
    truth = pl.read_parquet(sc.lib.FOLDER / "ground_truth_primaries.parquet").filter((pl.col("time_series_id") == series) & pl.col("valid_time").is_between(start, stop, closed="left")).select("valid_time", pl.col("power").alias("value")).with_columns(source=pl.lit("Observed"))
    parts = [truth]
    for s in steps:
        f = pl.read_parquet(sc.ROOT / "backtests" / f"step{s}_{exp.VERSION[s]}_sp{span}.parquet").filter((pl.col("time_series_id") == series) & pl.col("valid_time").is_between(start, stop, closed="left"))
        f = f.group_by("valid_time", "power_fcst_init_time").agg(pl.col("q50").mean()).with_columns(lead=((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() / 24).floor().cast(pl.Int32)).filter(pl.col("lead") == lead_day)
        parts.append(f.select("valid_time", pl.col("q50").alias("value")).with_columns(source=pl.lit(LABEL[s])))
    frame = pl.concat(parts).to_pandas()
    return alt.Chart(frame).mark_line(strokeWidth=1.5).encode(x=alt.X("valid_time:T", title=None), y=alt.Y("value:Q", title="Power (MW or MVA)", scale=alt.Scale(zero=False)), color=alt.Color("source:N", title=None, legend=alt.Legend(orient="top", columns=2))).properties(width=WIDTH, height=280, title=alt.TitleParams(f"{label}: median forecast {lead_day} to {lead_day + 1} days ahead", anchor="start", fontSize=13))


def main() -> None:
    span = int(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    horizon(span=span).save(out / "ladder_horizon.svg")
    contrasts(span=span).save(out / "ladder_contrasts.svg")
    steps = ("3b", "5w", "8w")
    for series, name in ((17, "D5"), (18, "D16")):
        example(span=span, label=f"{name}, the verified transfer", start=datetime(2025, 12, 28, tzinfo=UTC), stop=datetime(2026, 1, 15, tzinfo=UTC), series=series, steps=steps, lead_day=1).save(out / f"ladder_transfer_{name}.svg")
    print("figures written to", out)


if __name__ == "__main__":
    main()

# ruff: noqa: E501, D103
"""Experiment 677: figures of the ladder (SVG, public) from the private score tables.

Usage: python e00677_figures.py SPAN OUT_DIR
Reads `scores/step<N>_<version>_sp<SPAN>_s012/scores.csv`, `scores/contrasts_sp<SPAN>.csv`, and the exported
forecast files; writes `ladder_horizon.svg`, `ladder_contrasts.svg`, and the example time series.
Only Primaries appear, so the figures carry no generator series.
"""

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import e00677_export as exp
import e00677_score as sc
import numpy as np
import polars as pl

LABEL = {
    "1": "1: date-time only",
    "2": "2: weather only",
    "3a": "3a: date-time + weather (ENS only)",
    "3b": "3b: date-time + weather (pre-trained)",
    "4": "4: + 24 h lag",
    "5w": "5: + 24 h and 1-week lags",
    "6w": "6: + gap columns",
    "7w": "7: + weighted averages of the gap",
    "8w": "8: + neighbour columns (Candidate)",
    "5w@1": "9a: step 5 with monthly refits",
    "8w@1": "9b: step 8 with monthly refits",
}  # fmt: skip
# Two figures, each with its own colour set; a colour means one step within a figure. The six colours are
# Data Purple, Data Amber, Data Green, Data Deep Teal, Data Burnt Orange, and a dark navy: the worst pair
# stays 25 delta-E apart under normal vision and under deuteranopia, protanopia, and tritanopia.
COLOURS = ["#B701FF", "#FC9700", "#17E58F", "#009C75", "#BF4F04", "#0B215F"]
FIGURE_STEPS = {
    "early": ("1", "2", "3a", "3b", "4", "5w"),
    "late": ("5w", "6w", "7w", "8w", "5w@1", "8w@1"),
}
FIGURE_COLOURS = {"early": COLOURS, "late": ["#0B215F", "#B701FF", "#FC9700", "#009C75", "#BF4F04", "#17E58F"]}
REFS = {"Manual heuristic": ("#555555", [6, 3]), "Climatology": ("#9A9A9A", [2, 2]), "Experiment 667 (monthly refits)": ("#000000", [1, 0])}
WIDTH, HEIGHT = 620, 300
INTERIM = bool(os.environ.get("E677_INTERIM"))


def scores_of(*, span: int, step: str) -> pl.DataFrame:
    base, _, at = step.partition("@")
    tag = "s012" if at else ("s01234" if base[0] in "5678" else "s012")
    folder = f"step{base}_{exp.VERSION[base]}_sp{at or span}_{tag}"
    return pl.read_csv(sc.ROOT / "scores" / folder / "scores.csv").with_columns(step_label=pl.lit(LABEL.get(step, step)))


def horizon(*, span: int, which: str) -> alt.Chart:
    steps = [s for s in FIGURE_STEPS[which] if (sc.ROOT / "scores" / f"step{s.partition('@')[0]}_{exp.VERSION[s.partition('@')[0]]}_sp{s.partition('@')[2] or span}_{'s012' if s.partition('@')[2] or s[0] in '1234' else 's01234'}").exists()]
    frames = []
    for s in steps:
        f = scores_of(span=span, step=s).filter((pl.col("slice") == "lead day") & (pl.col("metric") == "mae"))
        frames.append(f.filter(pl.col("system") == "Model").with_columns(system=pl.lit(LABEL[s])))
    f = scores_of(span=span, step=steps[-1]).filter((pl.col("slice") == "lead day") & (pl.col("metric") == "mae") & (pl.col("system") != "Model"))
    frames.append(f.filter(pl.col("system") == "Manual heuristic"))
    ref = pl.read_csv(sc.lib.FOLDER / "scores_668" / "primaries_NRA" / "scores.csv").filter((pl.col("slice") == "lead day") & (pl.col("metric") == "mae") & (pl.col("system") == "Model"))
    frames.append(ref.with_columns(system=pl.lit("Experiment 667 (monthly refits)"), step_label=pl.lit("")).select(*frames[0].columns))
    if which == "early":
        frames.append(f.filter(pl.col("system") == "Climatology"))
    frame = pl.concat(frames, how="diagonal").with_columns(day=pl.col("group").cast(pl.Int32) + 0.5).to_pandas()
    refs = [r for r in REFS if r in set(frame["system"])]
    domain = [LABEL[s] for s in steps] + refs
    colours = FIGURE_COLOURS[which][: len(steps)] + [REFS[r][0] for r in refs]
    dashes = [[1, 0]] * len(steps) + [REFS[r][1] for r in refs]
    late = which == "late"
    lo_hi = alt.Scale(domain=[6.6, 9.1], zero=False) if late else alt.Scale(zero=False)
    prefix = "INTERIM, quarterly refits (steps 1 to 8). " if INTERIM else ""
    sub = "Steps 5 to 9 zoomed in. Steps with injected events. " if late else "Steps 1 to 5. "
    return (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=2, clip=True)
        .encode(
            x=alt.X("day:Q", title="Forecast horizon (days after the forecast origin)", axis=alt.Axis(values=[0, 2, 4, 6, 8, 10, 12, 14]), scale=alt.Scale(domain=[0, 14])),
            y=alt.Y("mean:Q", title="Median-forecast MAE (points of capacity; smaller is better)", scale=lo_hi),
            color=alt.Color("system:N", title=None, scale=alt.Scale(domain=domain, range=colours), legend=alt.Legend(orient="top", columns=1, symbolType="stroke", symbolStrokeWidth=3, labelLimit=500)),
            strokeDash=alt.StrokeDash("system:N", legend=None, scale=alt.Scale(domain=domain, range=dashes)),
        )
        .properties(width=WIDTH, height=HEIGHT, title=alt.TitleParams([prefix + "Normalised MAE of the median forecast by horizon.", sub + "Backtest origins 12:00 UTC, 16 Primaries, refits every 3 months unless stated; mean of seeds."], anchor="start", fontSize=12))
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


def two_leads(*, span: int, label: str, start: datetime, stop: datetime, series: int, steps: tuple[str, ...]) -> alt.VConcatChart:
    panels = [
        example(span=span, label=label, start=start, stop=stop, series=series, steps=steps, lead_day=d).properties(width=WIDTH, height=280, title=("INTERIM, quarterly refits. " if INTERIM else "") + f"{label}: median forecast {d} to {d + 1} days ahead")
        for d in (1, 5)
    ]
    return alt.vconcat(*panels).resolve_scale(color="shared")


def pick_weeks(*, series: int) -> dict[str, tuple[datetime, datetime]]:
    """Return the highest-demand winter week and the lowest-demand summer week of one series in the backtest year."""
    truth = pl.read_parquet(sc.lib.FOLDER / "ground_truth_primaries.parquet").filter(pl.col("time_series_id") == series).select("valid_time", "power")
    weekly = truth.group_by_dynamic("valid_time", every="1w", start_by="monday").agg(pl.col("power").mean(), pl.len().alias("n")).filter(pl.col("n") > 300)
    cold = weekly.filter(pl.col("valid_time").dt.month().is_in([12, 1, 2])).sort("power", descending=True)["valid_time"][0]
    warm = weekly.filter(pl.col("valid_time").dt.month().is_in([6, 7, 8])).sort("power")["valid_time"][0]
    return {"cold week": (cold, cold + timedelta(days=7)), "warm week": (warm, warm + timedelta(days=7))}


ARM_COLOUR = {"3b": "#009C75", "5w": "#0B215F", "8w": "#B701FF"}


def band_panel(*, span: int, step: str, start: datetime, stop: datetime, series: int, lead_day: int, title: str) -> alt.LayerChart:
    """Return one panel: one arm's median with its q05 to q95 and q25 to q75 bands, and the observed power on top."""
    truth = pl.read_parquet(sc.lib.FOLDER / "ground_truth_primaries.parquet").filter((pl.col("time_series_id") == series) & pl.col("valid_time").is_between(start, stop, closed="left")).select("valid_time", pl.col("power").alias("observed")).to_pandas()
    q = ["q05", "q25", "q50", "q75", "q95"]
    f = (
        pl.scan_parquet(sc.ROOT / "backtests" / f"step{step}_{exp.VERSION[step]}_sp{span}.parquet")
        .filter((pl.col("time_series_id") == series) & pl.col("valid_time").is_between(start, stop, closed="left"))
        .group_by("valid_time", "power_fcst_init_time")
        .agg(pl.col(q).mean())
        .collect()
        .with_columns(lead=((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() / 24).floor().cast(pl.Int32))
        .filter(pl.col("lead") == lead_day)
    )
    srt = np.sort(f.select(q).to_numpy(), axis=1)
    f = f.with_columns([pl.Series(c, srt[:, i]) for i, c in enumerate(q)]).sort("valid_time").to_pandas()
    colour = ARM_COLOUR[step]
    x = alt.X("valid_time:T", title=None)
    wide = alt.Chart(f).mark_area(opacity=0.2, color=colour).encode(x=x, y=alt.Y("q05:Q", title="Power (MW or MVA)", scale=alt.Scale(zero=False)), y2="q95:Q")
    narrow = alt.Chart(f).mark_area(opacity=0.35, color=colour).encode(x=x, y="q25:Q", y2="q75:Q")
    med = alt.Chart(f).mark_line(strokeWidth=1.5, color=colour).encode(x=x, y="q50:Q")
    obs = alt.Chart(truth).mark_line(strokeWidth=1.2, color="#FF4901").encode(x=x, y="observed:Q")
    return alt.layer(wide, narrow, med, obs).properties(width=WIDTH, height=280, title=title)


def band_figure(*, span: int, label: str, start: datetime, stop: datetime, series: int, lead_day: int) -> alt.VConcatChart:
    note = f"forecast made {lead_day} to {lead_day + 1} days ahead"
    prefix = "INTERIM, quarterly refits. " if INTERIM else ""
    panels = [
        band_panel(span=span, step=s, start=start, stop=stop, series=series, lead_day=lead_day, title=prefix + f"{label}, {LABEL[s]}: {note}")
        for s in ("5w", "8w")
    ]
    return alt.vconcat(*panels).properties(title=alt.TitleParams("Median (line), 50% band (dark shade), 90% band (light shade) of the forecast; observed power in orange.", anchor="start", fontSize=11))


def main() -> None:
    span = int(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for which in FIGURE_STEPS:
        horizon(span=span, which=which).save(out / f"ladder_horizon_{which}.svg")
    contrasts(span=span).save(out / "ladder_contrasts.svg")
    steps = ("3b", "5w", "8w")
    for series, name in ((17, "D5"), (18, "D16")):
        two_leads(span=span, label=f"{name}, the verified transfer", start=datetime(2025, 12, 28, tzinfo=UTC), stop=datetime(2026, 1, 15, tzinfo=UTC), series=series, steps=steps).save(out / f"ladder_transfer_{name}.svg")
    for series, name in ((17, "D5"), (18, "D16")):
        for kind, (start, stop) in pick_weeks(series=series).items():
            two_leads(span=span, label=f"{name}, {kind} ({start:%d %b %Y})", start=start, stop=stop, series=series, steps=steps).save(out / f"ladder_{kind.replace(' ', '_')}_{name}.svg")
    for series, name in ((17, "D5"), (18, "D16")):
        scen = {"verified transfer": (datetime(2025, 12, 28, tzinfo=UTC), datetime(2026, 1, 15, tzinfo=UTC)), **{k: v for k, v in pick_weeks(series=series).items()}}
        for kind, (start, stop) in scen.items():
            for d in (1, 5):
                band_figure(span=span, label=f"{name}, {kind}", start=start, stop=stop, series=series, lead_day=d).save(out / f"bands_{kind.replace(' ', '_')}_{name}_lead{d}.svg")
    print("figures written to", out)


if __name__ == "__main__":
    main()

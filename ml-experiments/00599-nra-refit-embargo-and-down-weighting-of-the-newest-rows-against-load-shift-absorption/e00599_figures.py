# ruff: noqa: E501
"""Experiment 599 figures: shares by phase, and time series of forecasts around load-shifts.

Share figure: experiment 587's code (`e00587_figures.share_figure`) with its module-level names
replaced, as experiment 595 did. Time-series figures follow the maintainer's plot rule: at most two
panels per figure, each panel at least 280 px high, one arm per panel (day-ahead median, q05-q95
band), the observed power and the normal running arrangement (NRA, the observed power minus the
event), and a legend outside the data area that names every line, band, and shade.

Usage: python e00599_figures.py [--real-arms E0 E3]   (run `e00599_score.py` first)
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import altair as alt
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00587_common as c587  # ty: ignore[unresolved-import]
import e00587_figures as f  # ty: ignore[unresolved-import]
import e00595_summary as summ  # ty: ignore[unresolved-import]
import e00599_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

f.common = SimpleNamespace(**{k: getattr(c587, k) for k in dir(c587) if not k.startswith("__")})
f.common.world_dir = common.world_dir
f.common.ARMS = dict.fromkeys(common.ALL_ARMS, ("", common.TAG))
f.summ = summ
COLOUR = {
    "E0": theme.DATA_PURPLE,
    "E1": theme.DATA_BLUE,
    "E2w": theme.DATA_GREEN,
    "E3": theme.BRAND_ORANGE,
}
f.COLOUR = COLOUR
f.SCALE = alt.Scale(domain=list(COLOUR), range=list(COLOUR.values()))
f.ARM_COLOUR = alt.Color("arm:N", scale=f.SCALE, legend=alt.Legend(orient="top", title=None))
LABEL = {
    "E0": "E0 (3-day embargo, equal weights)",
    "E1": "E1 (7-day embargo)",
    "E2w": "E2w (3-day embargo, newest 7 days halved)",
    "E3": "E3 (7-day embargo, next 7 days halved)",
}
PANEL_W, PANEL_H = 560, 290
SWITCH_START = datetime(2025, 12, 30, 1, 0, tzinfo=UTC)
SWITCH_END = datetime(2026, 1, 14, 11, 30, tzinfo=UTC)
VERIFIED_START = datetime(2025, 12, 29, tzinfo=UTC)
VERIFIED_END = datetime(2026, 1, 13, tzinfo=UTC)


def docs_dir() -> Path:
    """Return the experiment's docs folder."""
    return next((Path(__file__).resolve().parents[2] / "docs/ml-experiments").glob("00599-*"))


f.docs_dir = docs_dir


def day_ahead(*, arm: str, world: str, months: list[str], tsid: int) -> pl.DataFrame:
    """Return the day-ahead (24 to 48 h lead) q05, q50, and q95 of one series, by grid index."""
    parts = []
    for month in months:
        for day in (0, 14):
            for seed in c587.SEEDS:
                p = (
                    common.world_dir(arm=arm, world=world)
                    / "work/fits"
                    / common.TAG
                    / f"{month}_d{day}_s{seed}.parquet"
                )
                if p.exists():
                    parts.append(
                        pl.read_parquet(p)
                        .filter(pl.col("time_series_id") == tsid)
                        .with_columns(refit_day=pl.lit(day), month=pl.lit(month), seed=pl.lit(seed))
                    )
    x = pl.concat(parts).with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
        idx=(pl.col("valid_time").dt.epoch("us") - int(c587.GRID_START.timestamp() * 1e6))
        // 1_800_000_000,
    )
    x = x.filter((pl.col("lead") >= 24) & (pl.col("lead") < 48))
    # where windows overlap in valid time, keep the newer refit (later month, later day)
    x = x.sort("month", "refit_day").group_by("idx", "seed", maintain_order=True).last()
    return (
        x.group_by("idx")
        .agg(pl.col("q05").mean(), pl.col("q50").mean(), pl.col("q95").mean())
        .sort("idx")
    )


def mk(chart: Any, mark: str, mark_kw: dict, **enc: Any) -> Any:
    """Return `chart` with the named mark and encodings (typed loosely: ty loses the chart type after `mark_*`)."""
    return getattr(chart, mark)(**mark_kw).encode(**enc)


def panel(
    *,
    arm: str,
    series: pl.DataFrame,
    obs: np.ndarray,
    nra: np.ndarray,
    primary: int,
    label: str,
    lo: int,
    hi: int,
    shade: tuple[float, float],
    marks: list[tuple[float, str]],
    x0: int,
    last: bool,
) -> Any:
    """Return one panel: one arm's day-ahead median and q05-q95 band, observed power, and the NRA."""
    s = series.filter((pl.col("idx") >= lo) & (pl.col("idx") < hi))
    idx = np.arange(lo, hi)
    ref = (
        pl.DataFrame(
            {
                "idx": idx,
                "Observed power": obs[primary, idx],
                "Normal running arrangement (NRA)": nra[primary, idx],
            }
        )
        .unpivot(index="idx", variable_name="line", value_name="y")
        .drop_nulls()
        .filter(pl.col("y").is_finite())
    )
    day = (pl.col("idx") - x0) / 48
    band_name, med_name = "q05 to q95 band", f"{arm} median"
    band = s.with_columns(day.alias("day")).with_columns(pl.lit(band_name).alias("line"))
    med = s.select("idx", pl.col("q50").alias("y")).with_columns(pl.lit(med_name).alias("line"))
    ref = ref.with_columns(day.alias("day"))
    med = med.with_columns(day.alias("day"))
    names = [
        band_name,
        med_name,
        "Observed power",
        "Normal running arrangement (NRA)",
        "Event window",
        *[m for _, m in marks],
    ]
    cols = [
        "#BFD3FF",
        COLOUR[arm],
        "#222222",
        theme.DATA_GREEN,
        "#bbbbbb",
        *["#888888"] * len(marks),
    ]
    sc = alt.Scale(domain=names, range=cols)
    leg = alt.Legend(
        orient="bottom",
        title=None,
        columns=2,
        labelFontSize=11,
        labelLimit=420,
        symbolType="stroke",
    )
    colour = alt.Color("line:N", scale=sc, legend=leg)
    area = mk(
        alt.Chart(band),
        "mark_area",
        {"opacity": 0.55},
        x=alt.X("day:Q", title="Days from the event's onset" if last else None),
        y=alt.Y("q05:Q", title=f"{label} (share of capacity)", scale=alt.Scale(zero=False)),
        y2="q95:Q",
        color=colour,
    )
    lines = mk(
        alt.Chart(pl.concat([med.select("day", "y", "line"), ref.select("day", "y", "line")])),
        "mark_line",
        {"clip": True, "strokeWidth": 1.5},
        x="day:Q",
        y="y:Q",
        color=colour,
    )
    rect = mk(
        alt.Chart(pl.DataFrame({"a": [shade[0]], "b": [shade[1]], "line": ["Event window"]})),
        "mark_rect",
        {"opacity": 0.25},
        x="a:Q",
        x2="b:Q",
        color=colour,
    )
    rules = [
        mk(
            alt.Chart(pl.DataFrame({"x": [d], "line": [m]})),
            "mark_rule",
            {"strokeDash": [5, 3], "strokeWidth": 1.3},
            x="x:Q",
            color=colour,
        )
        for d, m in marks
    ]
    return alt.layer(rect, area, lines, *rules).properties(
        width=PANEL_W,
        height=PANEL_H,
        title=alt.TitleParams(LABEL[arm], anchor="start", fontSize=12),
    )


def event_figure(
    *,
    name: str,
    title: str,
    subtitle: list[str],
    arm_panels: list[tuple[str, str, c587.Event]],
    obs: np.ndarray,
    before: int = 14,
    after: int = 14,
    marks_dates: list[tuple[datetime, str]] | None = None,
    shade_dates: tuple[datetime, datetime] | None = None,
) -> None:
    """Stack two panels (one arm each, or one Primary each) and save the SVG."""
    assert len(arm_panels) <= 2
    charts = []
    for i, (arm, case, event) in enumerate(arm_panels):
        ev, no = c587.case_worlds(case=case)
        months = c587.case_months(case=case)
        nra = c587.world_clean(world=no, obs=obs)
        seen = c587.world_clean(world=ev, obs=obs)
        series = day_ahead(arm=arm, world=ev, months=months, tsid=event.time_series_id)
        x0 = event.onset_idx
        end = event.end_idx
        shade = (0.0, (end - x0) / 48)
        marks = []
        if marks_dates:
            g0 = int(c587.GRID_START.timestamp())
            marks = [((int(d.timestamp()) - g0) / 1800 / 48 - x0 / 48, m) for d, m in marks_dates]
            shade = (
                (
                    (int(shade_dates[0].timestamp()) - g0) / 1800 / 48 - x0 / 48,
                    (int(shade_dates[1].timestamp()) - g0) / 1800 / 48 - x0 / 48,
                )
                if shade_dates
                else shade
            )
        charts.append(
            panel(
                arm=arm,
                series=series,
                obs=seen,
                nra=nra,
                primary=event.primary,
                label=f"Primary {event.label}",
                lo=x0 - before * 48,
                hi=end + after * 48,
                shade=shade,
                marks=marks,
                x0=x0,
                last=i == len(arm_panels) - 1,
            )
        )
    chart = (
        alt.vconcat(*charts)
        .properties(
            title=alt.TitleParams(
                title, subtitle=subtitle, anchor="start", fontSize=13, subtitleFontSize=11
            )
        )
        .resolve_scale(color="independent")
    )
    f.save(chart, name)  # ty: ignore[invalid-argument-type]


def verified_window_check(*, obs: np.ndarray) -> str:
    """Return a text check of which window matches the step in the readings of D5 and D16."""
    sm = c587.site_map()
    lines = []
    g0 = int(c587.GRID_START.timestamp())
    for lab in ("D5", "D16"):
        p = int(sm.filter(pl.col("label") == lab)["primary"][0])
        cap = float(sm.filter(pl.col("label") == lab)["capacity_mw"][0])
        for when, what in (
            (VERIFIED_START, "580 start 2025-12-29"),
            (SWITCH_START, "maintainer start 2025-12-30 01:00"),
            (VERIFIED_END, "580 end 2026-01-13"),
            (SWITCH_END, "maintainer end 2026-01-14 11:30"),
        ):
            i = (int(when.timestamp()) - g0) // 1800
            before = np.nanmean(obs[p, i - 96 : i])
            after = np.nanmean(obs[p, i : i + 96])
            lines.append(
                f"{lab} {what}: mean 2 days after minus 2 days before = {(after - before) * cap:+.2f} MW"
            )
        # half-hourly step: largest day-to-day jump of the 2-hour mean near each candidate start
        for when, what in ((VERIFIED_START, "580 start"), (SWITCH_START, "maintainer start")):
            i = (int(when.timestamp()) - g0) // 1800
            seg = obs[p, i - 96 : i + 96]
            d = np.convolve(np.nan_to_num(seg), np.ones(4) / 4, mode="valid")
            lines.append(
                f"{lab} {what}: largest 2-hour-mean change inside +-2 days = {np.max(np.abs(np.diff(d))) * cap:.2f} MW at {datetime.fromtimestamp(g0 + (i - 96 + int(np.argmax(np.abs(np.diff(d)))) + 2) * 1800, UTC):%Y-%m-%d %H:%M}"
            )
    return "\n".join(lines)


def main() -> None:
    """Draw the share figure, the injected examples, and the verified pair."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-arms", nargs="+", default=["E0", "E3"])
    args = ap.parse_args()
    shares = pl.read_parquet(common.OUT / "scores" / "event_shares.parquet")
    f.share_figure(shares)
    obs = c510.clean_stack()
    chk = verified_window_check(obs=obs)
    (common.OUT / "scores" / "verified_window_check.txt").write_text(chk)
    print(chk)
    designs = c587.designs()
    picks = designs.filter(pl.col("start_day") == 3).filter(
        ((pl.col("depth_share_of_smaller_capacity") == 0.5) & (pl.col("length_days") == 15))
        | ((pl.col("depth_share_of_smaller_capacity") == 0.15) & (pl.col("length_days") == 30))
    )
    for r in picks.iter_rows(named=True):
        case = f"inj{r['spec_id']}"
        event = c587.case_events(case=case)[0]
        for k, pair in enumerate((("E0", "E1"), ("E2w", "E3"))):
            event_figure(
                name=f"figure_{'ab'[k]}_{case}.svg",
                title=f"Injected load-shift {case} at Primary {event.label}: median and 90% band of {pair[0]} and {pair[1]}",
                subtitle=[
                    "Half-hourly day-ahead (24 to 48 h lead) forecast, mean over 2 seeds.",
                    "The arm is fitted on the observed power, which carries the event; NRA is the observed power minus the event.",
                ],
                arm_panels=[(a, case, event) for a in pair],
                obs=obs,
            )
    ev = c587.case_events(case="verified")
    for arm in args.real_arms:
        event_figure(
            name=f"figure_real_pair_{arm}.svg",
            title=f"Verified December 2025 D5/D16 transfer, arm {arm}: median and 90% band at both Primaries",
            subtitle=[
                "Half-hourly day-ahead (24 to 48 h lead) forecast, mean over 2 seeds.",
                "Dashed lines and shade: switched window 2025-12-30 01:00 to 2026-01-14 11:30.",
            ],
            arm_panels=[(arm, "verified", e) for e in ev],
            obs=obs,
            before=14,
            after=14,
            marks_dates=[
                (SWITCH_START, "Switched 30 Dec 01:00"),
                (SWITCH_END, "Switched back 14 Jan 11:30"),
            ],
            shade_dates=(SWITCH_START, SWITCH_END),
        )


if __name__ == "__main__":
    main()

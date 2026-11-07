# ruff: noqa: E501, D103, ANN001, ANN201, PLW3301, F841
"""Figures of experiment 615, drawn from the assembled candidate tables only (no refits).

Time-series figures: two panels (one forecast candidate each, same colour in both), day-ahead
forecasts for a 14-day stretch, observed power in black, axis in day numbers (never calendar
dates), a thin strip of half-hours with no reading, and a key below. Forecast values are shown as
percent of capacity. Summary figures: scores with month-bootstrap intervals.

    source <env>; uv run --no-sync python ml-experiments/00615-*/e00615_figures.py OUTDIR
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import altair as alt
import e00588_clean_panels as cp  # ty: ignore[unresolved-import]
import e00615_common as common
import e00615_score as sc
import numpy as np
import polars as pl

IQ = {0.05: 1, 0.5: 5, 0.95: 9}
PANELS = {
    "wind": ("wind_default2", "wind_climatology"),
    "solar": ("solar_a2_member_annual", "solar_a2_member_quarterly"),
}
LEAD_LOW = {"wind": 24.0, "solar": 12.0}
"""The day-ahead window starts at this lead and spans 24 hours."""
COLOURS = {
    "solar_a2_member_quarterly": "#FF4901",
    "solar_a2_member_annual": "#FF9B73",
    "solar_gam": "#306BFF",
    "solar_xgboost": "#10C5F7",
    "solar_climatology": "#B701FF",
    "solar_manual_heuristic": "#17E58F",
    "wind_default2": "#FF4901",
    "wind_climatology": "#B701FF",
    "wind_default2_stored_tuesdays": "#FF9B73",
    "wind_default1_stored_tuesdays": "#306BFF",
}
NAMES = {
    "wind_default2": "wind default 2 (physics plant and mixture-density head)",
    "wind_climatology": "climatology",
    "solar_a2_member_annual": "a2_member, refitted once a year",
    "solar_a2_member_quarterly": "a2_member, refitted every quarter",
}
GAP = cp.entry("No reading, or dropped as idle (wind)", "band", cp.SKY, 0.95)
CHECKS: list[str] = []


def day_ahead(*, tech: str, candidate: str, series: str) -> pl.DataFrame:
    low = LEAD_LOW[tech]
    return (
        pl.read_parquet(common.TABLE_DIR / f"{candidate}.parquet")
        .filter(
            (pl.col("series") == series)
            & (pl.col("lead_hours") >= low)
            & (pl.col("lead_hours") < low + 24)
        )
        .sort("valid_time")
    )


def window_starts(*, ref: pl.DataFrame) -> pl.DataFrame:
    """Return every start day whose 14 days lie inside one development month, with the window's mean actual."""
    days = (
        ref.group_by(pl.col("valid_time").dt.date().alias("day"))
        .agg(m=pl.col("actual").mean(), n=pl.len())
        .sort("day")
    )
    have = {d: (m, n) for d, m, n in days.iter_rows()}
    rows = []
    for d in have:
        end = d + timedelta(days=13)
        if f"{d:%Y-%m}" != f"{end:%Y-%m}" or f"{d:%Y-%m}" not in common.DEV_MONTHS:
            continue
        window = [have.get(d + timedelta(days=k)) for k in range(14)]
        if any(w is None for w in window):
            continue
        rows.append({"day": d, "r": float(np.mean([w[0] for w in window]))})
    return pl.DataFrame(rows)


def pick_stretches(*, tech: str, series: str, ref: pl.DataFrame) -> dict[str, date]:
    """Return the start of each 14-day stretch: two contrasting weather types and one around a gap."""
    starts = window_starts(ref=ref).sort("r")
    names = ("high-wind", "calm") if tech == "wind" else ("very sunny", "cloudy")
    return {names[0]: starts["day"][-1], names[1]: starts["day"][0], "gap": gap_start(ref=ref)}


def stretch_missing(*, ref: pl.DataFrame, start, days: int = 14):
    """Return the missing half-hours (valid times) of a stretch, relative to the stretch's own usual slots."""
    from datetime import UTC, datetime, timedelta

    lo = datetime(start.year, start.month, start.day, tzinfo=UTC)
    hi = lo + timedelta(days=days)
    sub = ref.filter((pl.col("valid_time") >= lo) & (pl.col("valid_time") < hi))
    slots = sub.with_columns(
        slot=pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30
    )["slot"]
    usual = sorted(set(slots.to_list()))
    lo_slot, hi_slot = (usual[0], usual[-1]) if usual else (0, 47)
    have = set(sub["valid_time"].to_list())
    grid = []
    t = lo
    while t < hi:
        slot = t.hour * 2 + t.minute // 30
        if lo_slot <= slot <= hi_slot:
            grid.append(t)
        t += timedelta(minutes=30)
    return lo, hi, [t for t in grid if t not in have], len(grid)


def figure(*, tech: str, series: str, label: str, start, out: Path) -> None:
    left, right = PANELS[tech]
    ref = day_ahead(tech=tech, candidate=right, series=series)
    lo, hi, missing, grid = stretch_missing(ref=ref, start=start)
    CHECKS.append(
        f"{out.stem}: {len(missing)} of {grid} usual half-hours have no reading (strip), file {right}.parquet, series {series}"
    )
    units, ydom = [], (0.0, 1.0)
    cols = {}
    for cand in (left, right):
        d = day_ahead(tech=tech, candidate=cand, series=series).filter(
            (pl.col("valid_time") >= lo) & (pl.col("valid_time") < hi)
        )
        q = d.select(sc.Q).to_numpy()
        cols[cand] = (d, q)
    hi_y = max(float(np.nanmax(cols[c][1][:, IQ[0.95]])) for c in cols)
    hi_y = max(hi_y, max(float(cols[c][0]["actual"].max()) for c in cols))
    ydom = (-0.02, hi_y + 0.03)
    for i, cand in enumerate((left, right)):
        d, q = cols[cand]
        x = ((d["valid_time"] - lo).dt.total_minutes() / 1440.0 + 1.0).to_numpy()
        obs = d["actual"].to_numpy()
        strip = [
            {
                "x0": (t - lo).total_seconds() / 86400 + 1.0,
                "x1": (t - lo).total_seconds() / 86400 + 1.0 + 1 / 48,
            }
            for t in missing
        ]
        units.append(
            cp.arm_panel(
                x=x, q=q, iq=IQ, obs=obs, xdom=(1.0, 15.0), title=f"{label}: {NAMES.get(cand, cand)}",
                ydom=(ydom[0] * 100, ydom[1] * 100), xtype="Q", xtitle="Day of the stretch (1 to 14)" if i == 1 else None,
                strip=strip,
            )
        )  # fmt: skip
    chart = cp.figure(
        units=units,
        title=f"{label}: day-ahead forecasts (lead {LEAD_LOW[tech]:.0f} to {LEAD_LOW[tech] + 24:.0f} h) over 14 days, % of capacity",
        key=[*cp.FORECAST_ENTRIES, cp.OBS, GAP],
    )
    chart.save(str(out))
    chart.save(str(out.with_suffix(".png")), scale_factor=2)


def stretch_info(*, tech: str, series: str) -> dict:
    _, right = PANELS[tech]
    ref = day_ahead(tech=tech, candidate=right, series=series)
    return pick_stretches(tech=tech, series=series, ref=ref)


def time_series_figures(*, outdir: Path) -> list[dict]:
    notes = []
    cp.PANEL_HEIGHT = 300
    for tech, series_list in (
        ("wind", ["W1", "W2", "W3"]),
        ("solar", [f"Solar {c}" for c in common.SOLAR_SERIES]),
    ):
        for series in series_list:
            starts = stretch_info(tech=tech, series=series)
            ref = day_ahead(tech=tech, candidate=PANELS[tech][1], series=series)
            for kind, start in starts.items():
                name = (
                    f"{tech}_{series.replace(' ', '_')}_{kind.replace('-', '_').replace(' ', '_')}"
                )
                figure(
                    tech=tech, series=series, label=series, start=start, out=outdir / f"{name}.svg"
                )
                notes.append(
                    {"name": name, "series": series, "kind": kind, "month": f"{start:%Y-%m}"}
                )
    return notes


def gap_start(*, ref: pl.DataFrame) -> date:
    """Return the start day of the 14-day window inside one month with the most missing half-hours."""
    best, best_n = None, -1
    for d in window_starts(ref=ref)["day"].to_list():
        _, _, miss, _ = stretch_missing(ref=ref, start=d)
        if len(miss) > best_n:
            best, best_n = d, len(miss)
    return best


def summary_figures(*, outdir: Path) -> None:
    for tech in ("wind", "solar"):
        frame = sc.load(tech=tech).filter(pl.col("actual") > 0.0)
        cands = [
            c for c in COLOURS if c.startswith(tech) and c in frame["candidate"].unique().to_list()
        ]
        # Score by lead band with month-bootstrap 95% intervals on each candidate's own mean.
        rows = []
        for cand in cands:
            for band in [*sc.LEAD_BANDS, "all leads"]:
                sub = frame.filter(pl.col("candidate") == cand)
                if band != "all leads":
                    sub = sub.filter(pl.col("lead_band") == band)
                per = (
                    sub.group_by("series", "month")
                    .agg(pl.col("qs").mean())
                    .group_by("month")
                    .agg(pl.col("qs").mean())["qs"]
                    .to_numpy()
                )
                rng = np.random.default_rng(615)
                draws = per[rng.integers(0, per.size, size=(2000, per.size))].mean(axis=1)
                lo, hi = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    {
                        "candidate": cand,
                        "band": band,
                        "score": float(per.mean()),
                        "lo": float(lo),
                        "hi": float(hi),
                    }
                )
        df = pl.DataFrame(rows)
        order = [*sc.LEAD_BANDS, "all leads"]
        colour = alt.Color(
            "candidate:N",
            scale=alt.Scale(domain=cands, range=[COLOURS[c] for c in cands]),
            legend=alt.Legend(orient="bottom", columns=2, title="Candidate"),
        )
        base = alt.Chart(df).encode(
            x=alt.X("band:N", sort=order, title="Lead band"),
            xOffset=alt.XOffset("candidate:N", sort=cands),
        )
        chart = (
            (
                base.mark_bar(aria=False).encode(
                    color=colour,
                    y=alt.Y(
                        "score:Q", title="Quantile score (points of capacity; smaller is better)"
                    ),
                )
                + base.mark_errorbar(color="black", ticks=True).encode(
                    y=alt.Y("lo:Q", title=""), y2="hi:Q"
                )
            )
            .properties(
                width=560,
                height=300,
                title=alt.TitleParams(
                    [
                        f"{tech.capitalize()}: score by lead band, development months",
                        "Bar: mean over the 5 development months; line: 95% interval from resampling whole months",
                    ],
                    fontSize=12,
                    anchor="start",
                ),
            )
            .configure_view(stroke=None)
        )
        chart.save(str(outdir / f"{tech}_score_by_lead_band.svg"))
        chart.save(str(outdir / f"{tech}_score_by_lead_band.png"), scale_factor=2)
        # Score by month.
        m = sc.table(frame=frame, by=["month"]).select("candidate", "month", "qs")
        base = alt.Chart(m).encode(
            x=alt.X("month:N", title="Forecast-origin month"),
            xOffset=alt.XOffset("candidate:N", sort=cands),
            color=colour,
        )
        chart = (
            base.mark_bar(aria=False)
            .encode(y=alt.Y("qs:Q", title="Quantile score (points of capacity; smaller is better)"))
            .properties(
                width=560,
                height=300,
                title=alt.TitleParams(
                    [f"{tech.capitalize()}: score by month of forecast origin"],
                    fontSize=12,
                    anchor="start",
                ),
            )
            .configure_view(stroke=None)
        )
        chart.save(str(outdir / f"{tech}_score_by_month.svg"))
        chart.save(str(outdir / f"{tech}_score_by_month.png"), scale_factor=2)
        # Score by hour of day.
        h = sc.table(frame=frame, by=["hour"]).select("candidate", "hour", "qs")
        chart = (
            alt.Chart(h)
            .mark_line(aria=False, point=True)
            .encode(
                x=alt.X(
                    "hour:Q",
                    title="Hour of day (UTC)",
                    axis=alt.Axis(values=[0, 6, 12, 18, 24])
                    if tech == "wind"
                    else alt.Axis(values=[4, 8, 12, 16, 20]),
                    scale=alt.Scale(domain=[0, 24] if tech == "wind" else [4, 20]),
                ),
                y=alt.Y("qs:Q", title="Quantile score (points of capacity; smaller is better)"),
                color=colour,
            )
            .properties(
                width=560,
                height=300,
                title=alt.TitleParams(
                    [f"{tech.capitalize()}: score by hour of day (valid time)"],
                    fontSize=12,
                    anchor="start",
                ),
            )
            .configure_view(stroke=None)
        )
        chart.save(str(outdir / f"{tech}_score_by_hour.svg"))
        chart.save(str(outdir / f"{tech}_score_by_hour.png"), scale_factor=2)
        # Coverage and tail shares.
        o = sc.table(frame=frame.with_columns(all=pl.lit("all")), by=["all"])
        long = pl.concat(
            [
                o.select(
                    "candidate",
                    pl.lit("q05-q95 coverage (target 0.90)").alias("measure"),
                    pl.col("covered").alias("value"),
                ),
                o.select(
                    "candidate",
                    pl.lit("share above q99 (target 0.01)").alias("measure"),
                    pl.col("above99").alias("value"),
                ),
                o.select(
                    "candidate",
                    pl.lit("share below q01 (target 0.01)").alias("measure"),
                    pl.col("below01").alias("value"),
                ),
            ]
        )
        targets = pl.DataFrame(
            {
                "measure": [
                    "q05-q95 coverage (target 0.90)",
                    "share above q99 (target 0.01)",
                    "share below q01 (target 0.01)",
                ],
                "t": [0.9, 0.01, 0.01],
            }
        )
        pts = (
            alt.Chart(long)
            .mark_point(size=90, filled=True, aria=False)
            .encode(
                x=alt.X("value:Q", title="Share of rows"),
                y=alt.Y("candidate:N", sort=cands, title=None),
                color=colour,
            )
        )
        rule = (
            alt.Chart(targets)
            .mark_rule(color="#444444", strokeDash=[4, 3], aria=False)
            .encode(x="t:Q")
        )
        chart = (
            alt.layer(pts, rule, data=long)
            .facet(row=alt.Row("measure:N", title=None, sort=list(targets["measure"])))
            .resolve_scale(x="independent")
            .properties(
                title=alt.TitleParams(
                    [
                        f"{tech.capitalize()}: coverage and tail shares, development months",
                        "Dashed line: target",
                    ],
                    fontSize=12,
                    anchor="start",
                )
            )
            .configure_view(stroke=None)
        )
        chart.save(str(outdir / f"{tech}_coverage_tails.svg"))
        chart.save(str(outdir / f"{tech}_coverage_tails.png"), scale_factor=2)


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    notes = time_series_figures(outdir=outdir)
    (outdir / "stretches.json").write_text(json.dumps(notes, indent=1))
    (outdir / "strip_checks.txt").write_text("\n".join(CHECKS) + "\n")
    summary_figures(outdir=outdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

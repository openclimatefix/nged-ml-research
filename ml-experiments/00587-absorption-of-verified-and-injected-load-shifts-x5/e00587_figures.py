# ruff: noqa: E501
"""Experiment 587 figures: absorbed share by phase, the median forecasts around the verified pair and four injected events, and the newest-knot movement.

Run `e00587_score.py` and `e00587_knots.py` first. Every plotted series is a Primary substation (demand only),
so substations are named by their anonymised D-label and the time axis counts days from the event's onset.
"""

from pathlib import Path

import altair as alt
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00587_common as common  # ty: ignore[unresolved-import]
import e00587_summary as summ  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

W = 620
COLOUR = {"M0": theme.DATA_BLUE, "D5": theme.BRAND_ORANGE, "P5": theme.DATA_PURPLE}
LABEL = {
    "M0": "M0 (no offset, no lags)",
    "D5": "D5 (offset, no lags)",
    "P5": "P5 (offset, lags, embargo)",
}
PHASE_LABEL = {
    "before": "before",
    "first_4_days": "first 4 days",
    "day_4_to_end": "day 4 to end",
    "after_7_days": "7 days after end",
}
SCALE = alt.Scale(domain=list(COLOUR), range=list(COLOUR.values()))
ARM_COLOUR = alt.Color("arm:N", scale=SCALE, legend=alt.Legend(orient="top", title=None))


def docs_dir() -> Path:
    """Return the experiment's docs folder."""
    return next((Path(__file__).resolve().parents[2] / "docs/ml-experiments").glob("00587-*"))


def save(chart: alt.Chart, name: str) -> None:
    """Write a chart as SVG into the experiment's docs folder."""
    chart.save(docs_dir() / name)


def share_figure(shares: pl.DataFrame) -> None:
    """Draw the refit and total share by phase and arm at the day-14 refit, with 95% design-bootstrap intervals."""
    rows = []
    for value, kind in (
        ("refit_q50", "Refit share: the fit alone, forecast with no-event inputs"),
        ("total_q50", "Total share: the fit and the lag inputs"),
    ):
        d = summ.per_design(shares=shares, value=value).filter(pl.col("refit_day") == 14)
        for arm in common.ARMS:
            for phase in summ.TESTED:
                v = d.filter((pl.col("arm") == arm) & (pl.col("phase") == phase))[value].to_numpy()
                if len(v) < 2:
                    continue
                m, lo, hi = summ.interval(values=v)
                rows.append(
                    {
                        "kind": kind,
                        "arm": arm,
                        "phase": PHASE_LABEL[phase],
                        "m": m,
                        "lo": lo,
                        "hi": hi,
                        "n": len(v),
                    }
                )
    df = pl.DataFrame(rows)
    order = [PHASE_LABEL[p] for p in summ.TESTED]
    base = alt.Chart().encode(
        x=alt.X("phase:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="arm:N",
        color=ARM_COLOUR,
    )
    pts = base.mark_point(filled=True, size=70).encode(y=alt.Y("m:Q", title="Share of depth"))  # ty: ignore[unresolved-attribute]
    bars = base.mark_rule().encode(y="lo:Q", y2="hi:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#888")
        .encode(y="y:Q")
    )
    chart = (
        alt.layer(zero, bars, pts)
        .properties(width=W - 90, height=170)
        .facet(
            data=df,
            row=alt.Row(
                "kind:N",
                title=None,
                header=alt.Header(
                    labelAngle=0, labelAlign="left", labelFontSize=11, labelOrient="top"
                ),
            ),
        )
        .properties(
            title=alt.TitleParams(
                "Injected load-shifts: how much of the depth the median forecast moves toward the event",
                subtitle=[
                    "Day-14 refit, 12 designs (2 for the first 4 days); dot: mean over designs, line: 95% interval from resampling designs.",
                    "0 means the forecast did not move; 1 means it moved by the whole depth. Dashed rule: no movement.",
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            )
        )
    )
    save(chart, "figure_1_share_by_phase.svg")  # ty: ignore[invalid-argument-type]


def knots_figure() -> None:
    """Draw the newest-knot movement (share of capacity toward the event) by design for D5 and P5, with the 0.03 threshold."""
    k = pl.read_parquet(common.OUT / "scores" / "knot_movement.parquet").filter(
        (pl.col("case") != "verified") & (pl.col("refit_day") == 14)
    )
    d = k.group_by("arm", "case", "label").agg(pl.col("knot_movement").mean())
    d = d.with_columns((pl.col("case") + " " + pl.col("label")).alias("event"))
    thr = (
        alt.Chart(pl.DataFrame({"y": [-0.03, 0.03]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#888")
        .encode(y="y:Q")
    )
    pts = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=45, opacity=0.8)
        .encode(
            x=alt.X("arm:N", sort=["D5", "P5"], title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("event:N"),
            y=alt.Y(
                "knot_movement:Q", title="Newest-knot movement toward the event (share of capacity)"
            ),
            color=alt.Color("arm:N", scale=SCALE, legend=None),
        )
    )
    chart = alt.layer(thr, pts).properties(
        width=W - 90,
        height=200,
        title=alt.TitleParams(
            "Movement of the newest offset knot toward the event, by arm",
            subtitle=[
                "Day-14 refit, one dot per Primary-event of the 12 injected designs, mean over 2 seeds.",
                "Dashed rules: the plan's 0.03 of capacity. Positive means toward the event.",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        ),
    )
    save(chart, "figure_3_newest_knot_movement.svg")  # ty: ignore[invalid-argument-type]


def read_arm(*, arm: str, world: str, month: str, day: int) -> pl.DataFrame:
    """Return the day-ahead (24 to 48 h lead) median of an arm's fit, mean over seeds."""
    d = common.world_dir(arm=arm, world=world) / "work" / "fits" / common.ARMS[arm][1]
    fr = []
    for seed in common.SEEDS:
        f = d / f"{month}_d{day}_s{seed}.parquet"
        if f.exists():
            fr.append(
                pl.read_parquet(f).select(
                    "time_series_id", "power_fcst_init_time", "valid_time", "q50"
                )
            )
    x = pl.concat(fr).with_columns(
        ((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours()).alias("lead")
    )
    return (
        x.filter((pl.col("lead") >= 24) & (pl.col("lead") < 48))
        .group_by("time_series_id", "valid_time")
        .agg(pl.col("q50").mean())
    )


def example_series(*, case: str, event: common.Event, obs: np.ndarray) -> pl.DataFrame:
    """Return the observed, no-event, and per-arm median series of one Primary-event, in days from onset."""
    ev, no = common.case_worlds(case=case)
    clean_e, clean_n = (
        common.world_clean(world=ev, obs=obs)[event.primary],
        common.world_clean(world=no, obs=obs)[event.primary],
    )
    months = common.case_months(case=case)
    parts = []
    for arm in common.ARMS:
        for month in months:
            for day in (0, 14):
                r = read_arm(arm=arm, world=ev, month=month, day=day).filter(
                    pl.col("time_series_id") == event.time_series_id
                )
                parts.append(
                    r.with_columns(pl.lit(arm).alias("series"), pl.lit(day).alias("refit_day"))
                )
    f = pl.concat(parts).with_columns(
        (
            (pl.col("valid_time").dt.epoch("us") - int(common.GRID_START.timestamp() * 1e6))
            // 1_800_000_000
        ).alias("idx")
    )
    # where the day-0 and day-14 windows overlap in valid time, keep the newer refit's forecast
    f = f.sort("refit_day").group_by("series", "idx").agg(pl.col("q50").last())
    lo, hi = f["idx"].min(), f["idx"].max()
    idx = np.arange(lo, hi + 1)  # ty: ignore[no-matching-overload]  # ty: ignore[unsupported-operator]
    ref = pl.DataFrame(
        {"idx": idx, "Observed": clean_e[idx], "Normal running arrangement": clean_n[idx]}
    ).unpivot(index="idx", variable_name="series", value_name="q50")
    out = (
        pl.concat([f.select("idx", "series", "q50"), ref])
        .drop_nulls()
        .filter(pl.col("q50").is_finite())
    )
    out = out.with_columns(((pl.col("idx") - event.onset_idx) // common.PER_DAY).alias("day"))
    out = out.group_by("series", "day").agg(pl.col("q50").mean(), pl.len().alias("n"))
    out = out.filter((pl.col("n") >= 40) & (pl.col("day") >= -7) & (pl.col("day") <= 25))
    return out.with_columns(pl.lit(f"{event.label}, {case}").alias("panel"))


def examples_figure(
    *, name: str, title: str, panels: list[tuple[str, common.Event]], obs: np.ndarray
) -> None:
    """Stack one panel per Primary-event: observed power, the no-event power, and each arm's day-ahead median."""
    frames = [example_series(case=c, event=e, obs=obs) for c, e in panels]
    colours = {**COLOUR, "Observed": "#222222", "Normal running arrangement": theme.DATA_GREEN}
    sc = alt.Scale(domain=list(colours), range=list(colours.values()))
    charts = []
    for (case, event), fr in zip(panels, frames, strict=True):
        onset, end = 0.0, (event.end_idx - event.onset_idx) / common.PER_DAY
        shade = (
            alt.Chart(pl.DataFrame({"a": [onset], "b": [end]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(color="#bbbbbb", opacity=0.3)
            .encode(x="a:Q", x2="b:Q")
        )
        line = (
            alt.Chart(fr)  # ty: ignore[unresolved-attribute]
            .mark_line(clip=True, strokeWidth=1.6)
            .encode(
                x=alt.X(
                    "day:Q",
                    title="Days from the event's onset" if (case, event) == panels[-1] else None,
                ),
                y=alt.Y(
                    "q50:Q",
                    title=f"Primary {event.label} (share of capacity)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "series:N",
                    scale=sc,
                    legend=alt.Legend(orient="top", title=None)
                    if (case, event) == panels[0]
                    else None,
                ),
            )
        )
        charts.append(alt.layer(shade, line).properties(width=W - 90, height=120))
    chart = alt.vconcat(*charts).properties(
        title=alt.TitleParams(
            title,
            subtitle=[
                "Daily mean of the day-ahead (24 to 48 h lead) median of each arm, mean over 2 seeds; shaded: the event.",
                "The normal running arrangement is the observed power minus the event; the arms are fitted on the observed power. The page's shares use half-hourly values, not these daily means.",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        )
    )
    save(chart, name)  # ty: ignore[invalid-argument-type]


def cells_figure(shares: pl.DataFrame) -> None:
    """Draw each injected design's refit share (days 4 to the end and 7 days after pooled) by design cell and arm."""
    dz = common.designs().select(
        case=pl.format("inj{}", pl.col("spec_id")),
        cell=pl.format("{} days from day {}", pl.col("length_days"), pl.col("start_day")),
    )
    d = (
        shares.filter(
            (pl.col("case") != "verified")
            & (pl.col("refit_day") == 14)
            & pl.col("phase").is_in(["day_4_to_end", "after_7_days"])
        )
        .group_by("case", "arm")
        .agg(pl.col("refit_q50").mean())
        .join(dz, on="case")
    )
    order = ["7 days from day 3", "15 days from day 3", "30 days from day 3", "15 days from day 12"]
    zero = alt.Chart().mark_rule(strokeDash=[4, 3], color="#888").encode(y=alt.datum(0))  # ty: ignore[unresolved-attribute]
    pts = (
        alt.Chart(d)
        .mark_point(filled=True, size=60, opacity=0.85)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "cell:N",
                sort=order,
                title="Length of the load-shift, and the day of the month it starts",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset="arm:N",
            y=alt.Y("refit_q50:Q", title="Refit share of depth"),
            color=ARM_COLOUR,
        )
    )
    chart = alt.layer(pts, zero, data=d).properties(  # ty: ignore[invalid-argument-type]
        width=W - 90,
        height=200,
        title=alt.TitleParams(
            "Refit share of each injected load-shift by design cell",
            subtitle=[
                "Day-14 refit, one dot per design (3 per cell), days 4 to the end and the 7 days after the end pooled.",
                "0 means the forecast did not move; 1 means it moved by the whole depth.",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        ),
    )
    save(chart, "figure_5_share_by_design_cell.svg")  # ty: ignore[invalid-argument-type]


def main() -> None:
    """Draw every figure."""
    shares = pl.read_parquet(common.OUT / "scores" / "event_shares.parquet")
    share_figure(shares)
    knots_figure()
    cells_figure(shares)
    obs = c510.clean_stack()
    verified = [("verified", e) for e in common.case_events(case="verified")]
    examples_figure(
        name="figure_2_verified_pair.svg",
        title="Verified D5 and D16 transfer: daily mean of each arm's median forecast",
        panels=verified,
        obs=obs,
    )
    designs = common.designs()
    picks = designs.filter(pl.col("start_day") == 3).filter(
        ((pl.col("depth_share_of_smaller_capacity") == 0.5) & (pl.col("length_days") == 15))
        | ((pl.col("depth_share_of_smaller_capacity") == 0.3) & (pl.col("length_days") == 7))
        | ((pl.col("depth_share_of_smaller_capacity") == 0.15) & (pl.col("length_days") == 30))
        | ((pl.col("depth_share_of_smaller_capacity") == 0.3) & (pl.col("length_days") == 15))
    )
    inj = [
        (f"inj{r['spec_id']}", common.case_events(case=f"inj{r['spec_id']}")[0])
        for r in picks.iter_rows(named=True)
    ]
    examples_figure(
        name="figure_4_injected_examples.svg",
        title="Four injected load-shifts: the arms' median forecasts against the observed and the no-event power",
        panels=inj,
        obs=obs,
    )


if __name__ == "__main__":
    main()

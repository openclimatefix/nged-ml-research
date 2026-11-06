# ruff: noqa: E501, D103, ANN001, PERF401
"""Experiment 592 figures. Run `e00592_score.py` first.

Demand series only: every plotted series is a Primary substation, so Primaries carry their
D-numbers and dates. Gates and colours are fixed by `GATE_COLOUR`.
"""

from pathlib import Path

import altair as alt
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_figures as f3  # ty: ignore[unresolved-import]
import e00583_score as s3  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import e00592_score as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
W = 620
GATE_COLOUR = {
    "G0": "#6b6b6b",
    "G1": theme.DATA_SKY,
    "G2": theme.DATA_BLUE,
    "G3": theme.BRAND_ORANGE,
    "G4": theme.DATA_PURPLE,
    "G5": theme.DATA_GREEN,
}
GATE_SCALE = alt.Scale(domain=list(GATE_COLOUR), range=list(GATE_COLOUR.values()))
GATE_LEGEND = alt.Legend(title=None, orient="top", columns=3, labelLimit=400)
LEN = {
    "15 and 30 days": "Injected 15 and 30 day events",
    "all lengths": "All injected events (3, 7, 15, 30 days)",
}
LABEL_EXPR = (
    "{'G0':'G0: 3-day sustain, day-old lags','G1':'G1: 2-day sustain','G2':'G2: 1-day sustain',"
    "'G3':'G3: fresh power','G4':'G4: fresh power + mirrored neighbour','G5':'G5: fresh power, released at snap-back'}[datum.label]"
)


def read(name: str) -> pl.DataFrame:
    return pl.read_csv(
        g.WORK / name, schema_overrides={"on_trace": pl.String}, infer_schema_length=None
    )


def save(chart: alt.TopLevelMixin, name: str) -> None:
    chart.save(str(DOCS / name))  # ty: ignore[unresolved-attribute]


def vstack(*, df: pl.DataFrame, col: str, build, title: list[str]) -> alt.VConcatChart:
    """Stack one layered panel per value of `col` (faceting cannot take layers with their own data)."""
    panels = [
        build(df.filter(pl.col(col) == v)).properties(
            title=alt.TitleParams(str(v), fontSize=11, anchor="start")
        )
        for v in df[col].unique(maintain_order=True).to_list()
    ]
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(title, fontSize=11, anchor="start")
    )


def gate_colour(*, legend: bool = True) -> alt.Color:
    return alt.Color(
        "gate:N",
        scale=GATE_SCALE,
        legend=alt.Legend(title=None, orient="top", columns=2, labelExpr=LABEL_EXPR, labelLimit=500)
        if legend
        else None,
    )


# ---------------------------------------------------------------- headline and summaries


def headline() -> None:
    bank = read("bank_summary.csv").filter(
        (pl.col("group") == "both groups")
        & (pl.col("band") == "24-120 h")
        & (pl.col("phase") == "days4-end")
        & pl.col("length_class").is_in(list(LEN))
    )
    ft = read("false_triggers.csv").filter(pl.col("group") == "all 16")
    df = bank.join(
        ft.select("gate", "false_triggers_per_primary_year"), left_on="arm", right_on="gate"
    ).rename({"arm": "gate"})
    df = df.with_columns(cls=pl.col("length_class").replace(LEN))

    def build(d: pl.DataFrame) -> alt.LayerChart:
        base = alt.Chart(d).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "false_triggers_per_primary_year:Q",
                title="False triggers per Primary-year, event-free days (smaller is better)",
                scale=alt.Scale(domain=[0, 3.6]),
            ),
            y=alt.Y(
                "cover90:Q",
                title="Share of rows with the NRA inside q05-q95",
                scale=alt.Scale(domain=[0.86, 0.93]),
            ),
            color=gate_colour(),
        )
        return alt.layer(
            alt.Chart(pl.DataFrame({"y": [0.9]}))
            .mark_rule(strokeDash=[4, 3], color="#444")
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0.86, 0.93]))),  # ty: ignore[unresolved-attribute]
            alt.Chart(pl.DataFrame({"x": [1.0]}))
            .mark_rule(strokeDash=[4, 3], color="#444")
            .encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
            base.mark_rule().encode(
                y=alt.Y("cover90_lo:Q", scale=alt.Scale(domain=[0.86, 0.93])), y2="cover90_hi:Q"
            ),
            base.mark_point(filled=True, size=70),
        ).properties(width=W, height=170)

    save(
        vstack(
            df=df,
            col="cls",
            build=build,
            title=[
                "Coverage from day 4 of a load-shift to its end, against false triggers",
                "Dot: mean over events. Line: 95% interval from resampling events.",
                "Dashed lines: the 0.90 target and 1 false trigger per Primary-year. Rows 24 to 120 h ahead, both groups. Upper left is better.",
            ],
        ),
        "headline_coverage_against_false_triggers.svg",
    )


def coverage_by_phase() -> None:
    b = (
        read("bank_summary.csv")
        .filter(
            (pl.col("group") == "both groups")
            & (pl.col("band") == "24-120 h")
            & (pl.col("length_class") == "15 and 30 days")
        )
        .rename({"arm": "gate"})
    )
    order = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]
    base = alt.Chart(b).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "phase:N",
            sort=order,
            title="Phase of the load-shift (by valid time of the row)",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            "cover90:Q",
            title="Share of rows with the NRA inside q05-q95",
            scale=alt.Scale(domain=[0.75, 1.0]),
        ),
        color=gate_colour(),
        xOffset="gate:N",
    )
    chart = alt.layer(
        base.mark_point(filled=True, size=45),
        base.mark_rule().encode(
            y=alt.Y("cover90_lo:Q", scale=alt.Scale(domain=[0.75, 1.0])), y2="cover90_hi:Q"
        ),
        alt.Chart(pl.DataFrame({"y": [0.9]}))
        .mark_rule(strokeDash=[4, 3], color="#444")
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0.75, 1.0]))),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=W,
        height=230,
        title=alt.TitleParams(
            [
                "Coverage by phase and gate, injected 15 and 30 day events (672 Primary-events)",
                "Dot: mean over events. Line: 95% interval from resampling events. Dashed: the 0.90 target. Rows 24 to 120 h ahead.",
            ],
            fontSize=11,
            anchor="start",
        ),
    )
    save(chart, "coverage_by_phase.svg")


def detection_delay() -> None:
    t = read("trigger_times.csv").filter(
        pl.col("source").str.starts_with("bank") & (pl.col("length_days") >= 7)
    )
    rows = []
    for gate in g.GATES:
        d = t.filter(pl.col("gate") == gate)
        n = d.height
        for k in range(15):
            rows.append(
                {
                    "gate": gate,
                    "day": k,
                    "share": float((d["days_onset_to_trigger"].fill_null(99) <= k).sum() / n),
                }
            )
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "day:Q",
                title="Days after the load-shift begins (origin days)",
                axis=alt.Axis(values=list(range(0, 15, 2))),
            ),
            y=alt.Y(
                "share:Q",
                title="Share of injected events with the gate on by then",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=gate_colour(),
        )
        .properties(
            width=W,
            height=230,
            title=alt.TitleParams(
                [
                    "How quickly each gate turns on, injected events of 7, 15, and 30 days (1,260 Primary-events)",
                    "Higher and further left is faster. The curve stops below 1 when the gate never fires within the event.",
                ],
                fontSize=11,
                anchor="start",
            ),
        )
    )
    save(chart, "gate_on_by_day_after_onset.svg")


def end_delay() -> None:
    t = read("trigger_times.csv").filter(
        pl.col("source").str.starts_with("bank") & (pl.col("length_days") >= 7)
    )
    d = t.filter(pl.col("snapped_back")).group_by("gate", "days_end_to_snap_back").len()
    tot = d.group_by("gate").agg(pl.col("len").sum().alias("tot"))
    d = (
        d.join(tot, on="gate")
        .with_columns(share=pl.col("len") / pl.col("tot"))
        .filter(pl.col("days_end_to_snap_back") <= 8)
    )
    chart = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "days_end_to_snap_back:O",
                title="Days from the end of the load-shift to the gate releasing",
            ),
            y=alt.Y("share:Q", title="Share of events whose gate released"),
            color=gate_colour(),
            xOffset="gate:N",
        )
        .properties(
            width=W,
            height=200,
            title=alt.TitleParams(
                [
                    "Delay in detecting the end of a load-shift (injected events of 7, 15, and 30 days)",
                    "A bar at 0 or 1 means the gate released on or the day after the end.",
                ],
                fontSize=11,
                anchor="start",
            ),
        )
    )
    save(chart, "gate_release_delay.svg")


def false_alarm_cost() -> None:
    e = read("event_free.csv").filter(pl.col("group") == "both groups")
    long = pl.concat(
        [
            e.select(
                "gate",
                pl.lit(q).alias("level"),
                pl.col(f"{k}_minus_G0").alias("d"),
                pl.col(f"{k}_minus_G0_lo").alias("lo"),
                pl.col(f"{k}_minus_G0_hi").alias("hi"),
            )
            for q, k in (("Pinball loss at q05", "pin05"), ("Pinball loss at q95", "pin95"))
        ]
    ).filter(pl.col("gate") != "G0")

    def build(d: pl.DataFrame) -> alt.LayerChart:
        base = alt.Chart(d).encode(y=alt.Y("gate:N", title=None), color=gate_colour(legend=False))  # ty: ignore[unresolved-attribute]
        return alt.layer(
            alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#444").encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
            base.mark_rule().encode(
                x=alt.X("lo:Q", title="Minus G0's loss, points of capacity (smaller is better)"),
                x2="hi:Q",
            ),
            base.mark_point(filled=True, size=60).encode(x="d:Q"),
        ).properties(width=W, height=110)

    save(
        vstack(
            df=long,
            col="level",
            build=build,
            title=[
                "False-alarm cost: event-free rows, 24 to 120 h, both metering groups",
                "Dot: mean over five fold months. Line: month-block interval at the Bonferroni level",
                "(10 contrasts). Zero: same as G0.",
            ],
        ),
        "false_alarm_cost.svg",
    )


def real_pair() -> None:
    r = read("real_event_phase.csv").filter(
        (pl.col("band") == "24-120 h") & pl.col("phase").is_in(["days4-end", "after0-7"])
    )

    def build(d: pl.DataFrame) -> alt.LayerChart:
        return alt.layer(
            alt.Chart(d)
            .mark_bar()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("phase:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "cover90:Q",
                    title="Share of rows inside q05-q95",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=gate_colour(),
                xOffset="gate:N",
            ),
            alt.Chart(pl.DataFrame({"y": [0.9]}))
            .mark_rule(strokeDash=[4, 3], color="#444")
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1]))),  # ty: ignore[unresolved-attribute]
        ).properties(width=W, height=130)

    save(
        vstack(
            df=r,
            col="event",
            build=build,
            title=[
                "The two verified December 2025 events (2 Primary-events, descriptive)",
                "Dashed: the 0.90 target. Rows 24 to 120 h ahead.",
            ],
        ),
        "real_pair_coverage.svg",
    )


def timelines() -> None:
    t = read("trigger_times.csv").filter(
        pl.col("source").str.starts_with("bank") & pl.col("length_days").is_in([15.0, 30.0])
    )
    keys = (
        t.filter(pl.col("gate") == "G0")
        .unique(subset=["month", "pass", "primary"])
        .sort("month", "pass", "primary")
    )
    rng = np.random.default_rng(592)
    pick = keys[rng.choice(keys.height, size=8, replace=False).tolist()]
    out = []
    for i, k in enumerate(pick.iter_rows(named=True)):
        sub = t.filter(
            (pl.col("month") == k["month"])
            & (pl.col("pass") == k["pass"])
            & (pl.col("primary") == k["primary"])
        )
        for r in sub.iter_rows(named=True):
            for j, ch in enumerate(r["on_trace"]):
                out.append(
                    {
                        "event": f"Event {i + 1}: D{k['primary']}, {k['length_days']:.0f} days",
                        "gate": r["gate"],
                        "day": r["trace_start_rel"] + j,
                        "day2": r["trace_start_rel"] + j + 1,
                        "on": int(ch),
                    }
                )
    df = pl.DataFrame(out)

    def build(d: pl.DataFrame) -> alt.LayerChart:
        heat = (
            alt.Chart(d)
            .mark_rect()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("day:Q", title="Days since the load-shift began"),
                x2="day2:Q",
                y=alt.Y("gate:N", title=None),
                color=alt.Color(
                    "on:O",
                    scale=alt.Scale(domain=[0, 1], range=["#eeeeee", theme.BRAND_ORANGE]),
                    legend=alt.Legend(title="Gate on", orient="top"),
                ),
            )
        )
        return alt.layer(
            heat, alt.Chart(pl.DataFrame({"x": [0]})).mark_rule(color="#000").encode(x="x:Q")
        ).properties(width=W, height=80)  # ty: ignore[unresolved-attribute]

    save(
        vstack(
            df=df,
            col="event",
            build=build,
            title=[
                "When each gate is on, eight random injected events",
                "Orange: gate on that forecast origin day. Black line: the load-shift begins.",
            ],
        ),
        "gate_timelines.svg",
    )


def _mode_from_on(*, on: np.ndarray) -> np.ndarray:
    """Return event mode (on or in the 7 days after a release) from an on-state vector."""
    snap = np.zeros((1, len(on)), bool)
    snap[0, 1:] = on[:-1] & ~on[1:]
    return g.arm_masks(states={"x": (on[None, :], snap)})["x"][0][0]


def event_plot(
    *,
    t: c.X4Table,
    rows,
    q_v,
    q_m,
    obs,
    nra,
    onset,
    end,
    p,
    modes: dict[str, np.ndarray],
    title,
    out,
) -> None:
    """Stack the C1 band and the C2 band under G0, G3, and G4 around one event (day-ahead rows)."""
    x = (t.idx[rows] - onset) / 48.0
    length = (end - onset) / 48.0
    union = np.array(c.a.union_band(lag=q_v, masked=q_m))
    panels = []
    ymin = float(np.nanmin(np.minimum(obs, nra))) * 100 - 3
    ymax = float(np.nanmax(np.maximum(obs, nra))) * 100 + 3
    sc_y = alt.Scale(domain=[ymin, ymax])
    names = [("C1", None), ("G0", modes["G0"]), ("G3", modes["G3"]), ("G4", modes["G4"])]
    for name, mode in names:
        wide = np.zeros(len(rows), bool) if mode is None else mode[t.prim[rows], t.origin_day[rows]]
        q = np.where(wide[:, None], union, q_v) * 100
        df = pl.DataFrame(
            {
                "x": x,
                "q01": q[:, c.IQ[0.01]],
                "q05": q[:, c.IQ[0.05]],
                "q50": q[:, c.IQ[0.5]],
                "q95": q[:, c.IQ[0.95]],
                "q99": q[:, c.IQ[0.99]],
                "obs": obs * 100,
                "nra": nra * 100,
            }
        ).filter(pl.col("q50").is_finite())
        xax = alt.X(
            "x:Q",
            title="Days since the load-shift began" if name == "G4" else None,
            scale=alt.Scale(domain=[-3, length + 7]),
            axis=alt.Axis(values=list(range(-3, int(length) + 8, 3 if length < 20 else 6))),
        )
        span = (
            alt.Chart(pl.DataFrame({"a": [0.0], "b": [length]}))
            .mark_rect(color="#999999", opacity=0.10)
            .encode(x="a:Q", x2="b:Q")
        )  # ty: ignore[unresolved-attribute]
        layers = [
            span,
            *f3._band_layers(df=df, col=theme.DATA_PURPLE, xax=xax, sc_y=sc_y),
        ]
        label = (
            "C1 alone (no wider band)" if name == "C1" else f"C2 band under {g.GATE_LABEL[name]}"
        )
        panels.append(
            alt.layer(*layers).properties(
                width=W, height=90, title=alt.TitleParams(label, fontSize=11, anchor="start")
            )
        )
    save(
        alt.vconcat(*panels)
        .properties(title=alt.TitleParams(title, fontSize=11, anchor="start"))
        .resolve_scale(x="shared"),
        out,
    )


def event_figures() -> None:
    inp = s.Inputs()
    t = inp.t
    modes = {k: v[0] for k, v in inp.masks.items()}
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    for e in ev.iter_rows(named=True):
        rows = np.flatnonzero(
            (t.prim == e["primary"])
            & (t.lead >= 24)
            & (t.lead < 48)
            & (t.idx >= e["onset_idx"] - 3 * 48)
            & (t.idx < e["end_idx"] + 7 * 48)
        )
        rows = rows[np.argsort(t.idx[rows])]
        if len(rows) < 200:
            continue
        event_plot(t=t, rows=rows, q_v=t.arms["V"][rows], q_m=t.arms["M"][rows], obs=t.obs[rows], nra=t.nra[rows], onset=e["onset_idx"], end=e["end_idx"], p=e["primary"], modes=modes,
                   title=[f"Verified load-shift at {e['label']}, December 2025: day-ahead forecasts (issued 09:00 the day before)", "Purple: median and bands, q05-q95 (dark) and q01-q99 (light). Orange: observed power.", "Black dashed: the normal running arrangement. Grey shade: the load-shift."],
                   out=f"event_verified_{e['label']}.svg")  # fmt: skip
    # four injected events: 15 and 30 days, up and down, one per metering group where possible
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    bank_ev = pl.read_parquet(s3.BANK_PARQUET)
    kk_all = s3.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    chosen = []
    for grp in s.GROUPS_SCORED:
        for ln, sign in ((15, 1), (30, -1)):
            cand = bank_ev.filter(
                (pl.col("length_days") == ln)
                & (pl.col("sign") == sign)
                & (pl.col("month") == "2026-01")
            ).to_dicts()
            cand = [d for d in cand if str(t.group[d["primary"]]) == grp]
            if cand:
                chosen.append(cand[0])
    for k, e in enumerate(chosen):
        m, r, p = e["month"], e["pass"], e["primary"]
        in_m = t.month == m
        days_m = np.unique(t.idx[in_m] // 48)
        origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
        rows = np.flatnonzero(
            (t.prim == p)
            & in_m
            & (t.lead >= 24)
            & (t.lead < 48)
            & (t.idx >= e["onset_idx"] - 3 * 48)
            & (t.idx < e["end_idx"] + 7 * 48)
        )
        rows = rows[np.argsort(t.idx[rows])]
        vin = s3.load_injected(t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys)
        ok = np.isfinite(vin).all(axis=1)
        rows, vin = rows[ok], vin[ok]
        if len(rows) < 200:
            continue
        old = np.full_like(inp.gi["gap"], np.nan)
        sl = slice(days_m.min(), days_m.max() + 1)
        old[:, sl] = inp.gi["gap"][:, sl]
        shift = e["sign"] * e["depth"]
        old[p, e["onset_idx"] // 48 : e["end_idx"] // 48] += shift
        shifts = [(p, e["onset_idx"], e["end_idx"], float(shift))]
        nbs = inp.nb.get(p, [])
        if nbs:
            n = nbs[(r + p) % len(nbs)]
            shifts.append(
                (n, e["onset_idx"], e["end_idx"], float(-shift * t.capacity[p] / t.capacity[n]))
            )
        fresh = inp.fresh.gap(shifts=shifts)
        fresh[:, : origin_days_m[0] + 1] = np.nan
        fresh[:, origin_days_m[-1] + 1 :] = np.nan
        states = g.all_gates(
            old_gap=old,
            old_thr=inp.gi["thr"],
            fresh_gap=fresh,
            fresh_thr=inp.fresh_thr,
            origin_days=origin_days_m,
            neighbours=inp.nb,
        )
        mk = {gt: v[0] for gt, v in g.arm_masks(states=states).items()}
        word = "up" if e["sign"] > 0 else "down"
        event_plot(t=t, rows=rows, q_v=vin, q_m=t.arms["M"][rows], obs=t.obs[rows], nra=t.obs[rows], onset=e["onset_idx"], end=e["end_idx"], p=p, modes=mk,
                   title=[f"Injected load-shift at D{p} ({t.group[p]}): {e['length_days']:.0f} days, {word}, {100 * e['depth']:.0f}% of capacity", "Injected into the lag inputs only, so the orange line is the true power (the NRA).", "Purple: median and bands, q05-q95 (dark) and q01-q99 (light). Grey shade: the load-shift."],
                   out=f"event_injected_{k + 1}.svg")  # fmt: skip
    del bank


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    headline()
    coverage_by_phase()
    detection_delay()
    end_delay()
    false_alarm_cost()
    real_pair()
    timelines()
    event_figures()


if __name__ == "__main__":
    main()

# ruff: noqa: E501, D103, ANN001, D205
"""Experiment 598 figures. Run `e00598_score.py` first.

`python e00598_figures.py summary [--suffix S]` draws the figures read from the score tables:
coverage by phase, band width by arm, the follow-ratio histogram, and S against S-p when that arm exists.
`python e00598_figures.py example [--suffix S]` draws the December 2025 example event (Primaries D5
and D16, 14 days before the maintainer's onset to 14 days after the end), one figure per arm, and
prints which of the two candidate windows matches the observed readings. It rebuilds the systems'
quantiles from the stored forecasts, so it is slow.

Plot rules (plan, 2026-11-06): at most two panels per figure, each at least 280 px high, one arm per
panel set; a thin gate strip under each panel that has a gate, drawn from the stored gate state and
checked against it; the legend sits above the data area; no NRA line (the NRA is an estimate, and
every caption says coverage is scored against it); observed power is black and no other mark is
black or grey; the load-shift shade is a pale yellow that no arm uses.
"""

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import e00581_bank as bank  # noqa: F401  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_figures as f3  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00598_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
W = 620
PANEL_H = 300
STRIP_H = 14
IQ = c.IQ
SYSTEM_ORDER = ["A1", "R", "W", "S", "S-g", "S-p", "S-rs"]
LABEL = {
    "A1": "A1 alone", "R": "R: A1 + G4", "W": "W: A1, band widened", "S": "S: signals as inputs",
    "S-g": "S-g: S + G4 bit", "S-p": "S-p: noise for signals", "S-rs": "S, band width-matched",
    "S-g-rs": "S-g, width-matched", "S-p-rs": "S-p, width-matched", "G4T": "G4 on T",
}  # fmt: skip
GATE_STRIP = "#334155"
"""Dark slate for the gate strip, so that green means S-p only."""
GATE_NOTE = [
    "The gate strip is plotted at valid time (origin day plus one) and stays on for about a week after the gate releases (held).",
    "Compare the band width inside the shaded span with the same arm's width outside it.",
]


def lighten(colour: str, share: float = 0.5) -> str:
    """Return `colour` (hex) mixed with white by `share`, for a width-matched control."""
    rgb = [int(colour.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(v + (255 - v) * share):02x}" for v in rgb)


COLOUR = {
    "A1": theme.DATA_SKY, "R": theme.BRAND_ORANGE, "W": theme.DATA_PURPLE, "S": theme.DATA_BLUE,
    "S-g": theme.DATA_MAGENTA, "S-p": theme.DATA_GREEN, "S-rs": theme.DATA_BLUE_LIGHT,
    "S-g-rs": lighten(theme.DATA_MAGENTA), "S-p-rs": theme.DATA_GREEN_LIGHT, "G4T": theme.DATA_BURNT_ORANGE,
}  # fmt: skip
SHADE, SHADE_LINE = "#FFF1A8", "#B8860B"
"""The pale load-shift shade and its dashed boundary lines; no arm uses either."""
OBSERVED = "#000000"
PHASE_LABEL = {
    "before": "Before (7 days)", "days0-3": "Days 0-3", "days4-end": "Day 4 to end",
    "after0-7": "0-7 days after", "after7-14": "7-14 days after",
}  # fmt: skip
PHASE_ORDER = list(PHASE_LABEL.values())
EVENT_PRIMARIES = ("D5", "D16")
MAINTAINER_ONSET = datetime(2025, 12, 30, 1, 0, tzinfo=UTC)
MAINTAINER_END = datetime(2026, 1, 14, 11, 30, tzinfo=UTC)
CONTEXT_DAYS = 14
BAND = "24-120 h"


def grid_idx(*, when: datetime) -> int:
    return int((when.timestamp() * 1e6 - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US)


def when_of(*, idx: np.ndarray) -> list:
    base = c.k578.prep.GRID_START_US
    return [
        datetime.fromtimestamp((base + int(i) * c.k578.prep.STEP_US) / 1e6, UTC).replace(
            tzinfo=None
        )
        for i in idx
    ]


def save(chart, name: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.save(str(DOCS / name))


def to_none(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in a]


# ---------------------------------------------------------------- figures read from the score tables


def phase_chart(
    *,
    suffix: str,
    metric: str,
    title: list[str],
    domain: list[float],
    target: float | None,
    name: str,
    systems: list[str],
) -> None:
    b = pl.read_csv(score.ROOT / f"score{suffix}" / "bank_summary.csv", infer_schema_length=None)
    b = b.filter(
        (pl.col("group") == "both groups") & (pl.col("length_class") == "15 and 30 days")
        & (pl.col("band") == BAND) & (pl.col("subset") == "all months") & pl.col("system").is_in(systems)
    ).with_columns(phase_label=pl.col("phase").replace(PHASE_LABEL), name=pl.col("system").replace(LABEL))  # fmt: skip
    order = [LABEL[s] for s in systems]
    target_label = f"Target {target:.2f} (dashed)" if target is not None else None
    scale = alt.Scale(
        domain=[*order, *([target_label] if target_label else [])],
        range=[*(COLOUR[s] for s in systems), *([SHADE_LINE] if target_label else [])],
    )
    ys = alt.Scale(domain=domain)
    base = alt.Chart(b).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("phase_label:N", sort=PHASE_ORDER, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
        xOffset=alt.XOffset("name:N", sort=order),
        color=alt.Color("name:N", scale=scale, sort=order, legend=alt.Legend(title=None, orient="top", columns=3)),
    )  # fmt: skip
    layers = [
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(f"{metric}_lo:Q", scale=ys, title=title[0]), y2=f"{metric}_hi:Q"
        ),
        base.mark_point(filled=True, size=55).encode(y=alt.Y(f"{metric}:Q", scale=ys)),  # ty: ignore[unresolved-attribute]
    ]
    if target is not None:
        layers.insert(
            0,
            alt.Chart(pl.DataFrame({"y": [target], "name": [target_label]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3])
            .encode(
                y=alt.Y("y:Q", scale=ys),
                color=alt.Color(
                    "name:N",
                    scale=scale,
                    sort=scale.domain,  # ty: ignore[invalid-argument-type]
                    legend=alt.Legend(title=None, orient="top", columns=3),
                ),
            ),
        )  # ty: ignore[unresolved-attribute]
    save(
        alt.layer(*layers).properties(
            width=W, height=PANEL_H, title=alt.TitleParams(title[1:], fontSize=11, anchor="start")
        ),
        name,
    )


def width_chart(*, suffix: str) -> None:
    t = pl.read_csv(score.ROOT / f"score{suffix}" / "bank_summary.csv", infer_schema_length=None)
    systems = [
        s
        for s in ("A1", "R", "W", "S", "S-rs", "S-g", "S-g-rs", "S-p", "S-p-rs")
        if s in t["system"].unique().to_list()
    ]
    t = t.filter(
        (pl.col("group") == "both groups") & (pl.col("length_class") == "15 and 30 days")
        & (pl.col("band") == BAND) & (pl.col("subset") == "all months")
        & pl.col("phase").is_in(["days4-end", "after0-7", "after7-14"]) & pl.col("system").is_in(systems)
    ).with_columns(phase_label=pl.col("phase").replace(PHASE_LABEL), name=pl.col("system").replace(LABEL))  # fmt: skip
    order = [LABEL[s] for s in systems]
    ch = (
        alt.Chart(t).mark_bar()  # ty: ignore[unresolved-attribute]
        .encode(
            x=alt.X("phase_label:N", sort=PHASE_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("name:N", sort=order),
            y=alt.Y("width90:Q", title="Mean q95 - q05 (share of capacity; smaller is narrower)"),
            color=alt.Color("name:N", scale=alt.Scale(domain=order, range=[COLOUR[s] for s in systems]), sort=order, legend=alt.Legend(title=None, orient="top", columns=3)),
        )
        .properties(width=W, height=PANEL_H, title=alt.TitleParams(["Band width by arm on injected 15 and 30 day load-shifts, 24-120 h ahead, both metering groups", "Bar: mean over Primary-events. The width-matched controls (-rs) are rescaled to R's width on R's wide origin days in the bank, 24-120 h."], fontSize=11, anchor="start"))
    )  # fmt: skip
    save(ch, "width_by_arm.svg")


def follow_hist(*, suffix: str) -> None:
    t = pl.read_csv(
        score.ROOT / f"score{suffix}" / "bank_event_phase.csv", infer_schema_length=None
    )
    present = t["system"].unique().to_list()
    t = t.filter(
        (pl.col("phase") == "days4-end") & (pl.col("band") == BAND) & (pl.col("length_days") >= 15)
    )
    a1 = t.filter(pl.col("system") == "A1")["follow_q50"].to_numpy()
    r = t.filter(pl.col("system") == "R")["follow_q50"].to_numpy()
    merged = np.allclose(np.sort(a1), np.sort(r)) if len(a1) == len(r) else False
    series = {"A1 and R: same median" if merged else LABEL["A1"]: ("A1", a1)}
    if not merged:
        series[LABEL["R"]] = ("R", r)
    for n in ("S", "S-g", "S-p"):
        if n in present:
            series[LABEL[n]] = (n, t.filter(pl.col("system") == n)["follow_q50"].to_numpy())
    lo = float(np.floor(min(v.min() for _, v in series.values()) * 10) / 10)
    hi = float(np.ceil(max(v.max() for _, v in series.values()) * 10) / 10) + 0.1
    edges = np.round(np.arange(lo, hi + 1e-9, 0.1), 6)
    rows = []
    for name, (key, v) in series.items():
        counts, _ = np.histogram(v, bins=edges)
        for x0, x1, cnt in zip(edges[:-1], edges[1:], counts, strict=True):
            rows.append({"name": name, "x0": float(x0), "x1": float(x1), "count": int(cnt), "key": key})  # fmt: skip
    d = pl.DataFrame(rows)
    order = list(series)
    colours = [COLOUR["R" if k == "A1" and merged else k] for k, _ in series.values()]
    first = order[0]
    base = alt.Chart(d)  # ty: ignore[unresolved-attribute]
    wide = (
        base.transform_filter(alt.datum.name == first)
        .mark_line(interpolate="step-after", strokeWidth=7, opacity=0.45)
        .encode(  # ty: ignore[unresolved-attribute]
            x="x0:Q",
            y="count:Q",
            color=alt.Color(
                "name:N", scale=alt.Scale(domain=order, range=colours), sort=order, legend=None
            ),
        )
    )
    ch = (
        base.transform_filter(alt.datum.name != first).mark_line(interpolate="step-after", strokeWidth=2)  # ty: ignore[unresolved-attribute]
        .encode(
            x=alt.X("x0:Q", title="Median forecast moved towards the shift (share of the shift depth; 0 is no following)"),
            y=alt.Y("count:Q", title="Primary-events in the 0.1-wide bin"),
            color=alt.Color("name:N", scale=alt.Scale(domain=order, range=colours), sort=order, legend=alt.Legend(title=None, orient="top", columns=2)),
        )
        .properties(width=W, height=PANEL_H, title=alt.TitleParams(["Follow ratio on injected 15 and 30 day load-shifts, day 4 to the end, 24-120 h ahead", "Step outline: Primary-events per 0.1-wide bin of the follow ratio. Lower is better: the median ignores the lagged shift.", "A1 and R have identical follow ratios, because the gate widens the band and leaves the median alone."], fontSize=11, anchor="start"))
    )  # fmt: skip
    save(
        alt.layer(wide, ch).properties(width=W, height=PANEL_H, title=ch.title),
        "follow_histogram.svg",
    )


def placebo_scatter(*, suffix: str) -> None:
    t = pl.read_csv(
        score.ROOT / f"score{suffix}" / "bank_event_phase.csv", infer_schema_length=None
    )
    f = t.filter(
        (pl.col("phase") == "days4-end") & (pl.col("band") == BAND) & (pl.col("length_days") >= 15)
    )
    if "S-p" not in f["system"].unique().to_list():
        return
    a = f.filter(pl.col("system") == "S").select(
        "month", "pass", "primary", pl.col("cover90").alias("S")
    )
    b = f.filter(pl.col("system") == "S-p").select(
        "month", "pass", "primary", pl.col("cover90").alias("S-p")
    )
    j = a.join(b, on=["month", "pass", "primary"])
    dash, dot = "Equal coverage (dashed)", "One Primary-event"
    diag = pl.DataFrame({"v": [0.0, 1.0], "what": [dash, dash]})
    j = j.with_columns(what=pl.lit(dot))
    scale = alt.Scale(domain=[dot, dash], range=[theme.DATA_BLUE, SHADE_LINE])
    ax = alt.Scale(domain=[0, 1.01])
    colour = alt.Color(
        "what:N",
        scale=scale,
        sort=[dot, dash],
        legend=alt.Legend(title=None, orient="top", columns=2),
    )  # ty: ignore[invalid-argument-type]
    ch = (
        alt.layer(
            alt.Chart(diag).mark_line(strokeDash=[4, 3]).encode(x=alt.X("v:Q", title="Coverage of S (share of rows with the NRA in q05-q95)", scale=ax), y=alt.Y("v:Q", title="Coverage of S-p", scale=ax), color=colour),  # ty: ignore[unresolved-attribute]
            alt.Chart(j).mark_circle(opacity=0.35, size=40).encode(x=alt.X("S:Q", scale=ax), y=alt.Y("S-p:Q", scale=ax), color=colour),  # ty: ignore[unresolved-attribute]
        )
        .properties(width=W, height=PANEL_H, title=alt.TitleParams(["S against S-p on injected 15 and 30 day load-shifts, day 4 to the end, 24-120 h ahead", "Dots below the dashed line: the signals helped S."], fontSize=11, anchor="start"))
    )  # fmt: skip
    save(ch, "s_versus_s_p.svg")


def summary(*, suffix: str) -> None:
    present = (
        pl.read_csv(score.ROOT / f"score{suffix}" / "bank_summary.csv", infer_schema_length=None)[
            "system"
        ]
        .unique()
        .to_list()
    )
    systems = [s for s in ("A1", "R", "W", "S", "S-g", "S-p", "S-rs") if s in present]
    phase_chart(
        suffix=suffix, metric="cover90", systems=systems, name="coverage_by_phase.svg", domain=[0.7, 1.0], target=0.9,
        title=["Share of rows with the NRA inside q05-q95", "Coverage by phase of an injected load-shift: 15 and 30 day events, both metering groups, 24-120 h ahead", "Dot: mean over Primary-events. Line: 95% interval from resampling events. Dashed: the 0.90 target; closer to or above 0.90 is better. Scored against the NRA, an estimate."],
    )  # fmt: skip
    phase_chart(
        suffix=suffix, metric="nra_side95", systems=systems, name="safe_side_by_phase.svg", domain=[0.7, 1.0], target=0.95,
        title=["Share of rows with the NRA on the safe side of q05 or q95", "Safe-side share by phase of an injected load-shift: 15 and 30 day events, both metering groups, 24-120 h ahead", "Safe side: below q95 if the shift lowered the load, above q05 if it raised it. Dashed: the 0.95 target. Scored against the NRA, an estimate."],
    )  # fmt: skip
    width_chart(suffix=suffix)
    follow_hist(suffix=suffix)
    placebo_scatter(suffix=suffix)


# ---------------------------------------------------------------- the December 2025 example event


def window_match(*, t: c.X4Table, prims: dict[str, int]) -> dict:
    """Return, for the maintainer's dates and for experiment 580's verified window, the mean absolute
    step in observed power across each boundary (6 hours either side, share of capacity), summed over
    D5 and D16, and which window the readings match.
    """
    ev = sc.merged_events()
    out: dict = {"maintainer": 0.0, "580": 0.0}
    detail = []
    for lab, p in prims.items():
        row = ev.filter(
            (pl.col("primary") == p)
            & (pl.col("end_idx") >= grid_idx(when=datetime(2025, 12, 1, tzinfo=UTC)))
            & (pl.col("onset_idx") <= grid_idx(when=datetime(2026, 2, 1, tzinfo=UTC)))
        )
        assert row.height >= 1, (
            f"experiment 580 has no verified event for {lab} in December 2025 to January 2026"
        )
        o580, e580 = int(row["onset_idx"].min()), int(row["end_idx"].max())
        cands = {
            "maintainer": (grid_idx(when=MAINTAINER_ONSET), grid_idx(when=MAINTAINER_END)),
            "580": (o580, e580),
        }
        for name, (o, e) in cands.items():
            step = 0.0
            for b in (o, e):
                before = np.nanmean(t.clean[p, b - 12 : b])
                after = np.nanmean(t.clean[p, b : b + 12])
                step += abs(float(after - before))
            out[name] += step
            detail.append((lab, name, when_of(idx=np.array([o, e])), round(step, 4)))
    out["match"] = "maintainer" if out["maintainer"] >= out["580"] else "580"
    out["detail"] = detail
    return out


def strip_rects(*, x: list, on: np.ndarray) -> pl.DataFrame:
    """Return the start and end of each run of True in `on`, at the plotted times `x`."""
    runs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((x[start], x[i - 1]))
            start = None
    if start is not None:
        runs.append((x[start], x[-1]))
    return pl.DataFrame(
        {"x0": [r[0] for r in runs], "x1": [r[1] for r in runs]},
        schema={"x0": pl.Datetime, "x1": pl.Datetime},
    )


def arm_figure(
    *,
    name: str,
    sy: score.Systems,
    prims: dict[str, int],
    lo: int,
    hi: int,
    gate: np.ndarray | None,
    gate_label: str,
    caption: list[str],
    check=None,
) -> None:
    t = sy.t
    cols = {
        "obs": "Observed power",
        "median": f"{LABEL[name]}: median",
        "band": f"{LABEL[name]}: q05-q95",
        "shade": "Load-shift (maintainer's dates)",
        "dash": "Maintainer's onset and end (dashed)",
        "gate": gate_label,
    }
    domain = [
        cols["obs"],
        cols["median"],
        cols["band"],
        cols["shade"],
        cols["dash"],
        *([cols["gate"]] if gate is not None else []),
    ]
    rng = [OBSERVED, COLOUR[name], COLOUR[name], SHADE, SHADE_LINE, GATE_STRIP][: len(domain)]
    colour = alt.Color(
        "what:N",
        scale=alt.Scale(domain=domain, range=rng),
        sort=domain,
        legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=320),
    )
    parts = []
    on_x = [
        when_of(idx=np.array([grid_idx(when=MAINTAINER_ONSET)]))[0],
        when_of(idx=np.array([grid_idx(when=MAINTAINER_END)]))[0],
    ]
    for i, (lab, p) in enumerate(prims.items()):
        rows = f3.day_ahead_rows(t=t, prim=p, lo=lo, hi=hi)
        x = when_of(idx=t.idx[rows])
        q = sy.q[name][rows]
        d = pl.DataFrame({"x": x, "lo": to_none(q[:, IQ[0.05]] * 100), "hi": to_none(q[:, IQ[0.95]] * 100), "med": to_none(q[:, IQ[0.5]] * 100), "obs": to_none(t.obs[rows] * 100)}, schema={"x": pl.Datetime, "lo": pl.Float64, "hi": pl.Float64, "med": pl.Float64, "obs": pl.Float64})  # fmt: skip
        ymin = float(np.nanmin([d["lo"].min(), d["obs"].min()])) - 2
        ymax = float(np.nanmax([d["hi"].max(), d["obs"].max()])) + 2
        ys = alt.Scale(domain=[ymin, ymax])
        xs = alt.Scale(domain=[x[0], x[-1]])
        xax = alt.X(
            "x:T",
            scale=xs,
            title=None,
            axis=alt.Axis(format="%d %b") if i == 1 else alt.Axis(labels=False),
        )
        shade = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"a": [on_x[0]], "b": [on_x[1]], "what": [cols["shade"]]},
                    schema={"a": pl.Datetime, "b": pl.Datetime, "what": pl.String},
                )
            )
            .mark_rect(opacity=0.9, clip=True)
            .encode(x=alt.X("a:T", scale=xs), x2="b:T", color=colour)
        )  # ty: ignore[unresolved-attribute]
        lines = (
            alt.Chart(pl.DataFrame({"x": on_x, "what": [cols["dash"]] * 2}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[5, 4], strokeWidth=1.2)
            .encode(x=alt.X("x:T", scale=xs), color=colour)
        )  # ty: ignore[unresolved-attribute]
        band = (
            alt.Chart(d.with_columns(what=pl.lit(cols["band"])))  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.3, aria=False, clip=True)
            .encode(
                x=xax,
                y=alt.Y("lo:Q", scale=ys, title=f"{lab}: % of capacity"),
                y2="hi:Q",
                color=colour,
            )
        )  # ty: ignore[unresolved-attribute]
        med = (
            alt.Chart(d.with_columns(what=pl.lit(cols["median"])))  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.2, aria=False, clip=True)
            .encode(x=xax, y=alt.Y("med:Q", scale=ys), color=colour)
        )  # ty: ignore[unresolved-attribute]
        obs = (
            alt.Chart(d.with_columns(what=pl.lit(cols["obs"])))  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.0, aria=False, clip=True)
            .encode(x=xax, y=alt.Y("obs:Q", scale=ys), color=colour)
        )  # ty: ignore[unresolved-attribute]
        panel = alt.layer(shade, band, med, obs, lines).properties(
            width=W,
            height=PANEL_H,
            title=alt.TitleParams(lab if i else [*caption, lab], fontSize=11, anchor="start"),
        )
        parts.append(panel)
        if gate is not None:
            od = t.origin_day[rows]
            state = gate[p, od]
            if check is not None:
                check(p=p, origin_idx=t.origin_idx[rows], state=state)
            rects = strip_rects(x=x, on=state)
            rects = rects.with_columns(what=pl.lit(cols["gate"]))
            strip = (
                alt.Chart(rects)  # ty: ignore[unresolved-attribute]
                .mark_rect(opacity=0.9)
                .encode(
                    x=alt.X("x0:T", scale=xs, axis=alt.Axis(labels=False, ticks=False, title=None)),
                    x2="x1:T",
                    color=colour,
                )
                .properties(width=W, height=STRIP_H)
            )  # ty: ignore[unresolved-attribute]
            parts.append(strip)
    save(
        alt.vconcat(*parts, spacing=4).resolve_scale(color="shared"),
        f"example_event_{name.replace('-', '_')}.svg",
    )


def saved_g4_bits(*, suffix: str, months: list[str], seeds: list[int], n_days: int) -> dict:
    """Load the G4 bits that arm S-g's forecast windows used (saved by `e00598_arm.py`).

    Returns `state` (16, days), True where the seed-mean bit is at least 0.5, `seen` (16, days), True
    where any window used a bit for that Primary and origin day, and `split` (16, days), True where
    the seeds' bits differ.
    """
    work = score.ROOT / f"{score.ARM_DIR['S-g']}{suffix}" / "work" / "bits" / "P5"
    per_seed = {sd: np.full((16, n_days), np.nan, np.float32) for sd in seeds}
    for sd in seeds:
        for m in months:
            for f in sorted(work.glob(f"{m}_d*_s{sd}.npz")):
                z = np.load(f)
                day = z["origin_idx"] // score.sig.PER_DAY
                key = z["prim"].astype(np.int64) * n_days + day
                cur = per_seed[sd].reshape(-1)
                for kk, b in zip(key, z["bit"], strict=True):
                    assert np.isnan(cur[kk]) or cur[kk] == b, "one origin day has two bit values"
                    cur[kk] = b
    stack = np.stack(list(per_seed.values()))
    assert np.isfinite(stack).any(), f"no saved G4 bits under {work}"
    rows = [np.load(f) for sd in seeds for m in months for f in sorted(work.glob(f"{m}_d*_s{sd}.npz"))]  # fmt: skip
    key = np.concatenate([z["prim"].astype(np.int64) * (1 << 32) + z["origin_idx"] for z in rows])
    bit = np.concatenate([z["bit"] for z in rows])
    uniq, inv = np.unique(key, return_inverse=True)
    row_bit = np.bincount(inv, weights=bit) / np.bincount(inv)
    seen = np.isfinite(stack).any(axis=0)
    mean = np.nanmean(np.where(np.isfinite(stack), stack, np.nan), axis=0)
    split = np.nanmax(stack, axis=0) != np.nanmin(stack, axis=0)
    return {
        "state": np.nan_to_num(mean) >= 0.5, "seen": seen, "split": split & seen,
        "row_key": uniq, "row_bit": row_bit,
    }  # fmt: skip


def bits_check(*, bits: dict) -> Callable[..., None]:
    """Return a function that checks a drawn strip against the saved row-level bits.

    The strip is drawn from the day-level state; the check looks each plotted row up by (Primary,
    origin index) in the saved per-row bits and asserts the strip agrees wherever a bit was saved.
    """

    def check(*, p: int, origin_idx: np.ndarray, state: np.ndarray) -> None:
        key = p * (1 << 32) + origin_idx.astype(np.int64)
        pos = np.clip(np.searchsorted(bits["row_key"], key), 0, len(bits["row_key"]) - 1)
        hit = bits["row_key"][pos] == key
        assert hit.any(), "no saved G4 bit under the plotted window"
        assert np.array_equal(state[hit], bits["row_bit"][pos[hit]] >= 0.5), (
            "the S-g gate strip differs from the bit the forecasts used"
        )

    return check


def report_gate_difference(
    *, bits: dict, wide_t: np.ndarray, prims: list[int], lo: int, hi: int
) -> None:
    """Print how far the saved bit differs from the pooled-threshold table gate (`wide_t.npy`)."""
    d0, d1 = lo // score.sig.PER_DAY, hi // score.sig.PER_DAY
    for p in prims:
        seen = bits["seen"][p, d0:d1]
        diff = int((bits["state"][p, d0:d1][seen] != wide_t[p, d0:d1][seen]).sum())
        print(
            f"  S-g gate, Primary {p}: saved bit differs from the table gate on {diff} of {int(seen.sum())} origin days;"
            f" the seeds disagree on {int(bits['split'][p, d0:d1].sum())}",
            flush=True,
        )


def example(*, suffix: str) -> None:
    inp = k597.Inputs(arm="a1")
    t = inp.t
    arms = {}
    kk = c.k578  # noqa: F841
    keys = score.s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(keys)
    for n in score.S_ARMS:
        arm = score.Arm(name=n, suffix=suffix, months=["2025-12", "2026-01"], seeds=[0, 1])
        if arm.available():
            arm.load(t=t, order=order, sorted_keys=keys[order])
            arms[n] = arm
    factors = pl.read_csv(score.ROOT / f"score{suffix}" / "factors.csv", infer_schema_length=None)
    # a fold month the scorer did not fit takes the mean of the fitted months' factors, per system and seed
    months_ex = ["2025-12", "2026-01"]
    miss = [m for m in months_ex if m not in factors["month"].unique().to_list()]
    if miss:
        mean = factors.group_by("system", "seed").agg(pl.col("factor").mean())
        factors = pl.concat(
            [
                factors.select("system", "month", "seed", "factor"),
                *(
                    mean.with_columns(month=pl.lit(m)).select("system", "month", "seed", "factor")
                    for m in miss
                ),
            ]
        )
    sy = score.Systems(
        inp=inp, arms=arms, months=["2025-12", "2026-01"], seeds=[0, 1], factors=factors
    )
    sites = c.k578.sites()["label"].to_list()
    prims = {lab: sites.index(lab) for lab in EVENT_PRIMARIES}
    lo = grid_idx(when=MAINTAINER_ONSET) - CONTEXT_DAYS * 48
    hi = grid_idx(when=MAINTAINER_END) + CONTEXT_DAYS * 48
    wm = window_match(t=t, prims=prims)
    print(
        "window match (sum of |step| across the onset and end, share of capacity):",
        {k: round(v, 4) for k, v in wm.items() if k in {"maintainer", "580"}},
        "->",
        wm["match"],
    )
    for row in wm["detail"]:
        print("  ", row)
    extra: list[str] = []
    cap = [
        "December 2025 pair, day-ahead forecasts. Coverage is scored against the NRA, an estimate that is not drawn.",
        f"Dashed lines: the maintainer's dates ({MAINTAINER_ONSET:%Y-%m-%d %H:%M} and {MAINTAINER_END:%Y-%m-%d %H:%M}).",
        f"The observed readings match {'these dates' if wm['match'] == 'maintainer' else 'the verified window of experiment 580 (about a day earlier)'}.",
    ]
    stored = np.load(k597.WORK / "a1" / "mode_g4.npy")
    assert np.array_equal(stored, inp.mode), (
        "the rebuilt G4 state differs from experiment 597's stored state"
    )
    fac = factors.filter(pl.col("system") == "W")
    w_note = "W is A1 with its band widened about its median by a constant factor per fold month (a month without a fitted factor takes the mean of the fitted months): " + ", ".join(f"{r['month']} x{r['factor']:.3f}" for r in fac.iter_rows(named=True) if r["month"] in ("2025-12", "2026-01")) + "."  # fmt: skip
    wide_t = np.load(score.ROOT / f"score{suffix}" / "wide_t.npy")
    for n in sy.names:
        if n.endswith("-rs") or n == "G4T":
            continue
        gate, check = None, None
        if n == "R":
            gate = stored
        elif n == "S-g":
            bits = saved_g4_bits(suffix=suffix, months=["2025-12", "2026-01"], seeds=[0, 1], n_days=stored.shape[1])  # fmt: skip
            gate = bits["state"]
            report_gate_difference(bits=bits, wide_t=wide_t, prims=list(prims.values()), lo=lo, hi=hi)  # fmt: skip
            check = bits_check(bits=bits)
        glabel = "G4 on or held (stored state)" if n == "R" else "G4 bit S-g used (on or held)"
        arm_figure(
            name=n, sy=sy, prims=prims, lo=lo, hi=hi, gate=gate, gate_label=glabel,
            caption=[*cap, *extra, *([w_note] if n == "W" else []), *(GATE_NOTE if gate is not None else [])],
            check=check,
        )  # fmt: skip


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=("summary", "example"))
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    (summary if a.which == "summary" else example)(suffix=a.suffix)

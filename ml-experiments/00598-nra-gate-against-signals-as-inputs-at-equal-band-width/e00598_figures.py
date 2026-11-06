# ruff: noqa: E501, D103, ANN001, D205
"""Experiment 598 figures. Run `e00598_score.py` first.

`python e00598_figures.py summary [--suffix S]` draws the figures read from the score tables:
coverage by phase, band width by arm, the follow-ratio histogram, and S against S-p.
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
COLOUR = {
    "A1": theme.DATA_SKY, "R": theme.BRAND_ORANGE, "W": theme.DATA_PURPLE, "S": theme.DATA_BLUE,
    "S-g": theme.DATA_MAGENTA, "S-p": theme.DATA_GREEN, "S-rs": theme.DATA_BLUE_LIGHT,
    "S-g-rs": theme.DATA_MAGENTA, "S-p-rs": theme.DATA_GREEN_LIGHT, "G4T": theme.DATA_BURNT_ORANGE,
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
    scale = alt.Scale(domain=order, range=[COLOUR[s] for s in systems])
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
            alt.Chart(pl.DataFrame({"y": [target]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color=SHADE_LINE)
            .encode(y=alt.Y("y:Q", scale=ys)),
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
        s for s in ("A1", "R", "W", "S", "S-g", "S-p") if s in t["system"].unique().to_list()
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
        .properties(width=W, height=PANEL_H, title=alt.TitleParams(["Band width by arm on injected 15 and 30 day load-shifts, 24-120 h ahead, both metering groups", "Bar: mean over Primary-events. The width-matched controls equal R's width on R's wide origin days."], fontSize=11, anchor="start"))
    )  # fmt: skip
    save(ch, "width_by_arm.svg")


def follow_hist(*, suffix: str) -> None:
    t = pl.read_csv(
        score.ROOT / f"score{suffix}" / "bank_event_phase.csv", infer_schema_length=None
    )
    systems = [s for s in ("A1", "R", "S", "S-p") if s in t["system"].unique().to_list()]
    t = t.filter(
        (pl.col("phase") == "days4-end") & (pl.col("band") == BAND) & (pl.col("length_days") >= 15)
        & pl.col("system").is_in(systems)
    ).with_columns(name=pl.col("system").replace(LABEL))  # fmt: skip
    order = [LABEL[s] for s in systems]
    ch = (
        alt.Chart(t).mark_bar(opacity=0.55, binSpacing=0)  # ty: ignore[unresolved-attribute]
        .encode(
            x=alt.X("follow_q50:Q", bin=alt.Bin(step=0.1), title="Median forecast moved towards the shift (share of the shift depth; 0 is no following)"),
            y=alt.Y("count():Q", stack=None, title="Primary-events"),
            color=alt.Color("name:N", scale=alt.Scale(domain=order, range=[COLOUR[s] for s in systems]), sort=order, legend=alt.Legend(title=None, orient="top", columns=2)),
        )
        .properties(width=W, height=PANEL_H, title=alt.TitleParams(["Follow ratio on injected 15 and 30 day load-shifts, day 4 to the end, 24-120 h ahead", "Bars: Primary-events, overlaid. Lower is better: the median ignores the lagged shift."], fontSize=11, anchor="start"))
    )  # fmt: skip
    save(ch, "follow_histogram.svg")


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
    diag = pl.DataFrame({"v": [0.0, 1.0]})
    ch = (
        alt.layer(
            alt.Chart(diag).mark_line(strokeDash=[4, 3], color=SHADE_LINE).encode(x=alt.X("v:Q", title="Coverage of S (share of rows with the NRA in q05-q95)", scale=alt.Scale(domain=[0, 1])), y=alt.Y("v:Q", title="Coverage of S-p")),  # ty: ignore[unresolved-attribute]
            alt.Chart(j).mark_circle(opacity=0.35, size=40, color=theme.DATA_BLUE).encode(x="S:Q", y="S-p:Q"),  # ty: ignore[unresolved-attribute]
        )
        .properties(width=W, height=PANEL_H, title=alt.TitleParams(["S against S-p on injected 15 and 30 day load-shifts, day 4 to the end, 24-120 h ahead", "Dot: one Primary-event. Dashed: equal coverage. Dots below the dashed line: the signals helped S."], fontSize=11, anchor="start"))
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
        title=["Share of rows with the NRA inside q05-q95", "Coverage by phase of an injected load-shift: 15 and 30 day events, both metering groups, 24-120 h ahead", "Dot: mean over Primary-events. Line: 95% interval from resampling events. Dashed: the 0.90 target. Higher is better. Scored against the NRA, an estimate."],
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
) -> None:
    t = sy.t
    cols = {
        "obs": "Observed power",
        "median": f"{LABEL[name]}: median",
        "band": f"{LABEL[name]}: q05-q95",
        "shade": "Load-shift (maintainer's dates)",
        "gate": gate_label,
    }
    domain = [
        cols["obs"],
        cols["median"],
        cols["band"],
        cols["shade"],
        *([cols["gate"]] if gate is not None else []),
    ]
    rng = [OBSERVED, COLOUR[name], COLOUR[name], SHADE, theme.DATA_DEEP_TEAL][: len(domain)]
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
            alt.Chart(pl.DataFrame({"x": on_x}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[5, 4], color=SHADE_LINE, strokeWidth=1.2)
            .encode(x=alt.X("x:T", scale=xs))
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
            assert np.array_equal(state, gate[p][od]), (
                "gate strip does not match the stored gate state"
            )
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
    sy = score.Systems(inp=inp, arms=arms, months=["2025-12", "2026-01"], seeds=[0, 1])
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
    cap = [
        "December 2025 pair, day-ahead forecasts. Coverage is scored against the NRA, an estimate that is not drawn.",
        f"Dashed lines: the maintainer's dates ({MAINTAINER_ONSET:%Y-%m-%d %H:%M} and {MAINTAINER_END:%Y-%m-%d %H:%M}).",
        f"The observed readings match {'these dates' if wm['match'] == 'maintainer' else 'the verified window of experiment 580 (about a day earlier)'}.",
    ]
    stored = np.load(k597.WORK / "a1" / "mode_g4.npy")
    assert np.array_equal(stored, inp.mode), (
        "the rebuilt G4 state differs from experiment 597's stored state"
    )
    wide_t = np.load(score.ROOT / f"score{suffix}" / "wide_t.npy")
    for n in sy.names:
        if n.endswith("-rs") or n == "G4T":
            continue
        gate = stored if n == "R" else wide_t if n == "S-g" else None
        glabel = "G4 on or held (stored state)" if n == "R" else "G4-on-T bit on or held"
        arm_figure(
            name=n, sy=sy, prims=prims, lo=lo, hi=hi, gate=gate, gate_label=glabel, caption=cap
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=("summary", "example"))
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    (summary if a.which == "summary" else example)(suffix=a.suffix)

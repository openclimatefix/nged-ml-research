# ruff: noqa: E501, D103, C416, FURB192, PT018, PERF401, PLC0206
"""Experiment 590 figures: coverage, following, width, tails, event-free cost, and example forecasts.

Run `e00590_score.py` for t1, t2, t3, and p5 and `e00590_follow.py` first. Demand series only: every
plotted series is a Primary substation. Arms: P5 = experiment 574's plain P5 (clean lags, reference),
T0 = experiment 584's C1 (lag-only synthetic shifts), T1 (shifts in lags and targets), T2 (mixture),
T3 (placebo noise).
"""

from pathlib import Path

import altair as alt
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00590_report as rep
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = rep.ROOT
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
ARMS = ["P5", "T0", "T1", "T2", "T3"]
COLOUR = {
    "P5": "#6b6b6b",
    "T0": theme.DATA_PURPLE,
    "T1": theme.BRAND_ORANGE,
    "T2": theme.DATA_BLUE,
    "T3": theme.DATA_GREEN,
}
LABEL = {
    "P5": "P5: no synthetic shifts (reference)",
    "T0": "T0: shifts in lags only",
    "T1": "T1: shifts in lags and targets",
    "T2": "T2: half lag-only, half lag and target",
    "T3": "T3: placebo, random lag noise",
}
LOWER = {"P5": "p5", "T0": "t0", "T1": "t1", "T2": "t2", "T3": "t3"}
SCALE = alt.Scale(domain=[LABEL[a] for a in ARMS], range=[COLOUR[a] for a in ARMS])
W = 680
PHASES = ["before", "d0-3", "d4-end", "after0-7", "after7-14"]
PHASE_TITLE = {
    "before": "7 days before",
    "d0-3": "days 0-3",
    "d4-end": "day 4 to end",
    "after0-7": "0-7 days after",
    "after7-14": "7-14 days after",
}


def save(chart: alt.Chart | alt.VConcatChart | alt.HConcatChart, name: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.save(str(DOCS / name))


def arm_colour(title: str = "Arm") -> alt.Color:
    return alt.Color(
        "arm_label:N",
        scale=SCALE,
        sort=[LABEL[a] for a in ARMS],
        legend=alt.Legend(title=title, orient="top", columns=2, labelLimit=400),
    )


def lab(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(arm_label=pl.col("arm").replace_strict(LABEL))


def labelled(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(arm_label=pl.col("arm").replace_strict(LABEL))


def coverage_headline() -> None:
    b = (
        rep.bank()
        .filter(pl.col("phase") == "after0-7")
        .drop("arm")
        .rename({"name": "arm"})
        .pipe(lab)
    )
    nom = (
        alt.Chart(pl.DataFrame({"y": [0.9]}))
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0.7, 1.0])))
    )
    panels = []
    for g in rep.GROUPS:
        d = b.filter(pl.col("group") == g)
        base = alt.Chart(d).encode(
            x=alt.X("arm:N", sort=ARMS, title=None),
            color=arm_colour() if g == "both groups" else arm_colour().legend(None),
        )
        pts = base.mark_point(filled=True, size=90, aria=False).encode(
            y=alt.Y(
                "cover90:Q",
                scale=alt.Scale(domain=[0.7, 1.0]),
                title="Share of the NRA inside q05-q95",
            )
        )
        err = base.mark_rule(aria=False).encode(
            y=alt.Y("cover90_lo:Q", scale=alt.Scale(domain=[0.7, 1.0])), y2="cover90_hi:Q"
        )
        panels.append(alt.layer(nom, err, pts).properties(width=190, height=230, title=g))
    save(
        alt.hconcat(*panels).properties(
            title=alt.TitleParams(
                ["Coverage of the NRA after an injected load-shift ends (0-7 days after, 24-120 h, 15 and 30 day events)",
                 "Higher is better; dashed line: the 0.90 target. Dot: estimate. Line: 95% interval from resampling events. NRA: normal running arrangement. Arms: see the legend."],
                fontSize=11, anchor="start")
        ),
        "coverage_after_the_end.svg",
    )  # fmt: skip


def by_phase(
    *, value: str, title: str, sub: str, name: str, domain: list[float], nominal: float | None
) -> None:
    fe = rep.follow_events()
    d = (
        fe.group_by("arm", "phase")
        .agg(pl.col(value).mean().alias("v"), pl.col(value).std().alias("s"), pl.len().alias("n"))
        .with_columns(
            arm=pl.col("arm").str.to_uppercase(),
            lo=pl.col("v") - 1.96 * pl.col("s") / pl.col("n").sqrt(),
            hi=pl.col("v") + 1.96 * pl.col("s") / pl.col("n").sqrt(),
        )
    )
    d = lab(d.filter(pl.col("phase").is_in(PHASES)))
    base = alt.Chart(d).encode(
        x=alt.X(
            "phase:N",
            sort=PHASES,
            title=None,
            axis=alt.Axis(
                labelExpr="{'before':'7 days before','d0-3':'days 0-3','d4-end':'day 4 to end','after0-7':'0-7 days after','after7-14':'7-14 days after'}[datum.value]",
                labelAngle=0,
                labelLimit=110,
            ),
        ),
        color=arm_colour(),
    )
    line = base.mark_line(aria=False).encode(
        y=alt.Y("v:Q", title=title, scale=alt.Scale(domain=domain))
    )
    pts = base.mark_point(filled=True, size=50, aria=False).encode(y="v:Q")
    err = base.mark_rule(aria=False, opacity=0.5).encode(y="lo:Q", y2="hi:Q")
    layers = [err, line, pts]
    if nominal is not None:
        layers.insert(
            0,
            alt.Chart(pl.DataFrame({"y": [nominal]}))
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(y="y:Q"),
        )
    save(
        alt.layer(*layers).properties(
            width=W - 90,
            height=250,
            title=alt.TitleParams(
                [title + ", by phase of the injected load-shift", sub], fontSize=11, anchor="start"
            ),
        ),
        name,
    )


def width_figure() -> None:
    b = (
        rep.bank()
        .filter(pl.col("phase") == "after0-7")
        .drop("arm")
        .rename({"name": "arm"})
        .pipe(lab)
        .with_columns(width=pl.col("width90") * 100)
    )
    panels = []
    for g in ("MVA-only eight", "signed-MW six"):
        d = b.filter(pl.col("group") == g)
        base = alt.Chart(d).encode(
            color=arm_colour() if g == "MVA-only eight" else arm_colour().legend(None)
        )
        pts = base.mark_point(filled=True, size=80, aria=False).encode(
            x=alt.X(
                "width:Q", title="Mean q05-q95 width (% of capacity)", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(
                "cover90:Q",
                title="Share of the NRA inside q05-q95",
                scale=alt.Scale(domain=[0.7, 1.0]),
            ),
        )
        err = base.mark_rule(aria=False).encode(
            x="width:Q",
            y=alt.Y("cover90_lo:Q", scale=alt.Scale(domain=[0.7, 1.0])),
            y2="cover90_hi:Q",
        )
        txt = base.mark_text(dy=-12, fontSize=10, aria=False).encode(
            x="width:Q", y=alt.Y("cover90:Q", scale=alt.Scale(domain=[0.7, 1.0])), text="arm:N"
        )
        nom = (
            alt.Chart(pl.DataFrame({"y": [0.9]}))
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0.7, 1.0])))
        )
        panels.append(alt.layer(nom, err, pts, txt).properties(width=250, height=240, title=g))
    save(alt.hconcat(*panels).properties(title=alt.TitleParams(["Coverage after the end against band width (0-7 days after, 24-120 h, 15 and 30 day events)", "A wider band raises coverage; the question is whether T1 and T2 cover more than their width explains. Line: 95% interval over events."], fontSize=11, anchor="start")), "coverage_against_width.svg")  # fmt: skip


def tails_figure() -> None:
    b = (
        rep.bank()
        .filter((pl.col("phase") == "after0-7") & (pl.col("group") == "both groups"))
        .drop("arm")
        .rename({"name": "arm"})
        .pipe(lab)
    )
    long_ = (
        b.select("arm", "arm_label", *(pl.col(f"pin{q}") * 100 for q in ("01", "05", "95", "99")))
        .unpivot(index=["arm", "arm_label"], variable_name="q", value_name="pin")
        .with_columns(
            q=pl.col("q").replace({"pin01": "q01", "pin05": "q05", "pin95": "q95", "pin99": "q99"})
        )
    )
    ch = (
        alt.Chart(long_)
        .mark_bar(aria=False)
        .encode(
            x=alt.X("arm:N", sort=ARMS, title=None, axis=None),
            y=alt.Y("pin:Q", title="Pinball loss (% of capacity)"),
            color=arm_colour(),
            column=alt.Column("q:N", sort=["q01", "q05", "q95", "q99"], title=None),
        )
        .properties(
            width=110,
            height=200,
            title=alt.TitleParams(
                [
                    "Tail pinball loss, 0-7 days after an injected load-shift ends, both groups, 24-120 h",
                    "Lower is better. The tails matter most for curtailment.",
                ],
                fontSize=11,
                anchor="start",
            ),
        )
    )
    save(ch, "tail_pinball.svg")


def event_free_figure() -> None:
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        ef = rep.event_free_contrasts()
    ef = lab(ef).filter(
        pl.col("metric").is_in(["CRPS", "coverage q05-q95", "pinball q01+q99 mean"])
        & pl.col("arm").is_in(["T1", "T2", "T3"])
    )
    panels = []
    for metric, ttl in (
        ("CRPS", "CRPS minus T0 (points of capacity; negative is better)"),
        ("coverage q05-q95", "q05-q95 coverage minus T0 (0.90 is the target)"),
        ("pinball q01+q99 mean", "q01 and q99 pinball minus T0 (negative is better)"),
    ):
        d = ef.filter(pl.col("metric") == metric)
        base = alt.Chart(d).encode(
            y=alt.Y("arm:N", sort=["T1", "T2", "T3"], title=None),
            yOffset=alt.YOffset("group:N", sort=["MVA-only eight", "signed-MW six"]),
            color=arm_colour() if metric == "CRPS" else arm_colour().legend(None),
        )
        pts = base.mark_point(filled=True, size=70, aria=False).encode(
            x=alt.X("diff:Q", title=ttl),
            shape=alt.Shape(
                "group:N",
                sort=["MVA-only eight", "signed-MW six"],
                legend=alt.Legend(title="Group", orient="top")
                if metric == "coverage q05-q95"
                else None,
            ),
        )
        err = base.mark_rule(aria=False).encode(x="lo:Q", x2="hi:Q")
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        panels.append(
            alt.layer(zero, err, pts)
            .properties(width=W - 120, height=90)
            .facet(row=alt.Row("group:N", title=None), data=d)
            if False
            else alt.layer(zero, err, pts).properties(width=190, height=150, title=metric)
        )
    save(alt.vconcat(alt.hconcat(*[p for p in panels]), ).properties(title=alt.TitleParams(["Event-free rows, 24-120 h: each arm minus T0, per metering group (circle: eight MVA Primaries; square: six MW Primaries)", "Dot: mean of five fold-month differences. Line: interval at the Bonferroni level for 12 tests, from resampling the five months. Dashed: no difference."], fontSize=11, anchor="start")), "event_free_contrasts.svg")  # fmt: skip


QCOLS = ["q01", "q05", "q50", "q95", "q99"]
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]


def arm_series(
    *, view: Path, month: str, tsid: int, r: int | None, lo_us: int, hi_us: int
) -> pl.DataFrame:
    """Return the seed-and-refit mean day-ahead quantiles (forecasts issued 09:00 the day before) of one Primary."""
    folder = view / "inject" if r is not None else view / "fits/V3D5"
    months = month.split("+")
    pats = [
        f"{m}_r{r:02d}_d*_s*.parquet" if r is not None else f"{m}_d*_s*.parquet" for m in months
    ]
    frames = [
        pl.read_parquet(p, columns=[*KEYS, *QCOLS]).filter(pl.col("time_series_id") == tsid)
        for pat in pats
        for p in sorted(folder.glob(pat))
    ]
    f = pl.concat(frames).group_by(KEYS).agg([pl.col(q).mean() for q in QCOLS])
    return (
        f.filter(
            (
                pl.col("valid_time").dt.date() - pl.col("power_fcst_init_time").dt.date()
            ).dt.total_days()
            == 1
        )
        .filter(pl.col("power_fcst_init_time").dt.hour() == 9)
        .filter(
            (pl.col("valid_time").dt.epoch("us") >= lo_us)
            & (pl.col("valid_time").dt.epoch("us") <= hi_us)
        )
    )


def example_figure(
    *,
    title: list[str],
    name: str,
    month: str,
    p: int,
    tsid: int,
    r: int | None,
    onset: int,
    end: int,
    nra: np.ndarray,
    before_days: int = 3,
    after_days: int = 7,
) -> None:
    lo, hi = onset - before_days * 48, end + after_days * 48
    step, g0 = k578.prep.STEP_US, k578.prep.GRID_START_US
    parts = []
    for arm in ("P5", "T0", "T1", "T2", "T3"):
        view = {"P5": rep.ROOT / "view_p5", "T0": rep.ROOT584 / "view_c1"}.get(
            arm, rep.ROOT / f"view_{arm.lower()}"
        )
        f = arm_series(
            view=view, month=month, tsid=tsid, r=r, lo_us=g0 + lo * step, hi_us=g0 + hi * step
        )
        parts.append(f.with_columns(arm=pl.lit(arm)))
    d = pl.concat(parts).with_columns(
        idx=((pl.col("valid_time").dt.epoch("us") - g0) // step),
    )
    d = d.with_columns(day=(pl.col("idx") - onset) / 48.0, **{q: pl.col(q) * 100 for q in QCOLS})
    nr = (
        pl.DataFrame({"idx": np.arange(lo, hi + 1)})
        .with_columns(day=(pl.col("idx") - onset) / 48.0, nra=pl.Series(nra[lo : hi + 1] * 100))
        .filter(pl.col("nra").is_finite())
    )
    shade = pl.DataFrame({"x0": [0.0], "x1": [(end - onset) / 48.0]})
    panels = []
    for arm in ("P5", "T0", "T1", "T2", "T3"):
        da = d.filter(pl.col("arm") == arm)
        base = alt.Chart(da).encode(
            x=alt.X(
                "day:Q",
                title="Days since the load-shift began" if arm == "T3" else None,
                axis=alt.Axis(labels=arm == "T3"),
            )
        )
        wide = base.mark_area(opacity=0.18, color=COLOUR[arm], clip=True, aria=False).encode(
            y=alt.Y("q01:Q", title="Power (% of capacity)"), y2="q99:Q"
        )
        mid = base.mark_area(opacity=0.40, color=COLOUR[arm], clip=True, aria=False).encode(
            y="q05:Q", y2="q95:Q"
        )
        med = base.mark_line(color=COLOUR[arm], aria=False).encode(y="q50:Q")
        act = (
            alt.Chart(nr)
            .mark_line(color="black", strokeDash=[3, 2], aria=False)
            .encode(x="day:Q", y=alt.Y("nra:Q"))
        )
        sh = (
            alt.Chart(shade)
            .mark_rect(color="#999999", opacity=0.12, aria=False)
            .encode(x="x0:Q", x2="x1:Q")
        )
        panels.append(
            alt.layer(sh, wide, mid, med, act).properties(
                width=W - 90,
                height=95,
                title=alt.TitleParams(LABEL[arm], fontSize=10, anchor="start"),
            )
        )
    save(
        alt.vconcat(*panels).properties(title=alt.TitleParams(title, fontSize=11, anchor="start")),
        name,
    )


def examples() -> None:
    sm = k578.sites()
    clean = k578.clean()
    labels = dict(zip(sm["primary"].to_list(), sm["label"].to_list(), strict=True))
    groups = dict(
        zip(
            sm["primary"].to_list(),
            pl.read_csv(
                ML_EXPERIMENTS_DATA_DIR
                / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
            )["group"].to_list(),
            strict=True,
        )
    )
    tsid = dict(zip(sm["primary"].to_list(), sm["time_series_id"].to_list(), strict=True))
    month = "2025-10"
    ms = (
        k578.c.us(t=k578.c.month_start(month=month)) - k578.prep.GRID_START_US
    ) // k578.prep.STEP_US
    for sign in (1, -1):
        for gname in ("MVA-only eight", "signed-MW six"):
            cands = [
                (r, p)
                for r in range(bank.N_PASSES)
                for p in range(16)
                if groups[p] == gname and bank.event(primary=p, r=r) == (0.5 * sign, 15, sign)
            ]
            r, p = sorted(cands)[0]
            end = ms + bank.END_DAY * 48
            onset = end - 15 * 48
            example_figure(
                title=[f"{labels[p]}, {gname}: injected lag-only load-shift, {sign * 50:+d}% of capacity for 15 days (October 2025)",
                       "Only the lag inputs carry the shift, so the black dashed line (the observed power) is the NRA. Dark band: q05-q95. Light band: q01-q99. Grey shade: the shift in the lags.",
                       "Day-ahead forecasts issued at 09:00 the day before; mean of two seeds and two refits. Median: line."],
                name=f"example_injected_{gname.split()[0].lower()}_{'up' if sign > 0 else 'down'}.svg", month=month, p=p, tsid=tsid[p], r=r, onset=onset, end=end, nra=clean[p],
            )  # fmt: skip
    ver = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00580-verified-load-shift-event-set-and-label-audit/work/events_verified.parquet"
    )
    capacity = dict(
        zip(
            sm["label"].to_list(),
            pl.read_csv(
                ML_EXPERIMENTS_DATA_DIR
                / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
            )["capacity_mw"].to_list(),
            strict=True,
        )
    )
    ev = bank.verified_events(
        capacity_mw=capacity, label_to_primary={v: k for k, v in labels.items()}, table=ver
    )
    pair = ver.filter((pl.col("primary_a") == "D5") & (pl.col("primary_b") == "D16"))
    assert pair.height == 1, pair.height  # the verified December pair of experiments 580 and 587
    ev_pair = bank.verified_events(
        capacity_mw=capacity, label_to_primary={v: k for k, v in labels.items()}, table=pair
    )
    nra = bank.corrected_clean(clean=clean, events=ev)
    ev = ev_pair
    for q in ev:
        if q["label"] in ("D5", "D16") and ms and k578.c.month_start(month="2025-12") is not None:
            example_figure(
                title=[f"{q['label']}: verified real load-shift (December 2025), depth {q['depth'] * 100:+.0f}% of capacity",
                       "Forecast made with the observed (shifted) lags. Black dashed line: the NRA, the observed power with the labelled event removed. Dark band: q05-q95. Light band: q01-q99. Grey shade: the event.",
                       "Day-ahead forecasts issued at 09:00 the day before; mean of two seeds and two refits."],
                name=f"example_verified_{q['label']}.svg", month="2025-12+2026-01", p=q["primary"], tsid=tsid[q["primary"]], r=None, onset=q["onset_idx"], end=q["end_idx"], nra=nra[q["primary"]],
            )  # fmt: skip


def main() -> None:
    coverage_headline()
    by_phase(
        value="cov90",
        title="Share of the NRA inside q05-q95",
        sub="Injected 15 and 30 day events, both groups, 24-120 h. Higher is better; dashed line: 0.90. Line: 95% interval over events.",
        name="coverage_by_phase.svg",
        domain=[0.4, 1.0],
        nominal=0.9,
    )
    by_phase(
        value="follow_q50",
        title="Follow ratio of the median (0: ignores the shifted lags)",
        sub="Injected 15 and 30 day events, both groups, 24-120 h. Lower is better: the median should not follow a shift that is only in the lags. Line: 95% interval over events.",
        name="follow_by_phase.svg",
        domain=[-0.05, 0.4],
        nominal=0.0,
    )
    width_figure()
    tails_figure()
    event_free_figure()
    examples()


if __name__ == "__main__":
    main()

# ruff: noqa: E501, D103, D205, ANN001, ANN003
"""Experiment 591 figures: day-ahead bands of arms A0 to A4 around the verified December pair and
injected events, coverage by phase, the false-alarm cost, tail pinball loss, and the auxiliary heads'
recovery of the injected depth.

Run `e00591_score.py` for each arm and `e00591_probe_report.py` first. The event panels reuse
experiment 583's drawing code (`event_panels`, `verified_figures`, `coverage_figures`) with the arms
renamed: A0 is experiment 584's C1, A1 to A4 are this experiment's arms, "+band" is the gated wider
band of experiment 583. Demand series only: every plotted series is a Primary substation.
"""

from pathlib import Path

import altair as alt
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_figures as f3  # ty: ignore[unresolved-import]
import e00583_score as s3  # ty: ignore[unresolved-import]
import e00584_score as sc4  # ty: ignore[unresolved-import]
import e00591_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

ROOT = sc.ROOT
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
ARMS = ("A0", "A1", "A2", "A3", "A4")
COLOUR = {"A0": "#6b6b6b", "A1": theme.DATA_BLUE, "A2": theme.DATA_PURPLE, "A3": theme.BRAND_ORANGE,
          "A4": theme.DATA_GREEN, "A0+band": "#b0b0b0", "A2+band": "#7a2300"}  # fmt: skip
LABEL = {
    "A0": "A0: experiment 584's C1 (separate encoders, no auxiliary loss)",
    "A1": "A1: shared encoder with lag ID, no auxiliary loss",
    "A2": "A2: A1 with the auxiliary depth loss (λ = 9.09)",
    "A3": "A3: A1 with three times that λ (27.3)",
    "A4": "A4: separate encoders, auxiliary depth loss (λ = 9.09)",
}
f3.COLOUR, f3.LABEL, f3.PANELS, f3.DOCS = COLOUR, LABEL, ARMS, DOCS
f3.PHASE_ORDER = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]
STATE: dict = {}


def merged(*, name: str) -> pl.DataFrame:
    parts = []
    spec = [(a, a.lower(), "V") for a in ARMS] + [("A0+band", "a0", "A2"), ("A2+band", "a2", "A2")]
    for new, run, arm in spec:
        d = pl.read_csv(ROOT / f"score_{run}/{name}").filter(pl.col("arm") == arm)
        parts.append(d.with_columns(arm=pl.lit(new)))
    return pl.concat(parts, how="diagonal")


def arm_injected(*, arm: str, t: c.X4Table, month: str, r: int, rows: np.ndarray) -> np.ndarray:
    kk = s3.key(t.prim, t.origin_idx, t.idx)
    order, sorted_keys = np.argsort(kk), np.sort(kk)
    saved = c.INJECT
    c.INJECT = ROOT / f"view_{arm.lower()}/inject"
    try:
        return s3.load_injected(
            t=t, month=month, r=r, rows=rows, order=order, sorted_keys=sorted_keys
        )
    finally:
        c.INJECT = saved


_orig_panels = f3.event_panels


def event_panels(*, arms, rows, t, **kw) -> None:
    if STATE.get("pass") is None:
        new = {a: STATE["V"][a][rows] for a in ARMS}
    else:
        new = {
            a: arm_injected(arm=a, t=t, month="2025-10", r=STATE["pass"], rows=rows) for a in ARMS
        }
    _orig_panels(t=t, rows=rows, arms=new, **kw)


f3.event_panels = event_panels


def bank_examples(*, t: c.X4Table, gi: dict) -> None:
    """Injected events chosen by a fixed rule: in 2025-10, per metering group, the lowest-numbered
    Primary with a 15-day event of depth 0.5, up and down, taking the lowest pass; plus a 7-day pair.
    """
    ev_all = pl.read_parquet(s3.BANK_PARQUET).filter(pl.col("month") == "2025-10")
    in_m = t.month == "2025-10"
    origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
    days_m = np.unique(t.idx[in_m] // 48)
    gap_m = np.full_like(gi["gap"], np.nan)
    gap_m[:, days_m.min() : days_m.max() + 1] = gi["gap"][:, days_m.min() : days_m.max() + 1]
    for gname in s3.GROUPS_SCORED:
        for length, depth, sign in ((15, 0.5, 1), (15, 0.5, -1), (7, 0.3, 1)):
            cand = ev_all.filter(
                (pl.col("length_days") == length) & (pl.col("depth") == depth) & (pl.col("sign") == sign)
                & pl.col("primary").is_in([p for p in range(16) if t.group[p] == gname])
            ).sort("pass", "primary")  # fmt: skip
            if cand.height == 0:
                continue
            e = cand.row(0, named=True)
            r, p = e["pass"], e["primary"]
            evr = ev_all.filter(pl.col("pass") == r)
            lo, hi = e["onset_idx"] - 3 * 48, e["end_idx"] + 7 * 48
            rows = f3.day_ahead_rows(t=t, prim=p, lo=lo, hi=hi, mask=in_m)
            ok = np.isfinite(arm_injected(arm="A2", t=t, month="2025-10", r=r, rows=rows)).all(
                axis=1
            )
            rows = rows[ok]
            gap_i = gap_m.copy()
            for ee in evr.iter_rows(named=True):
                gap_i[ee["primary"], ee["onset_idx"] // 48 : ee["end_idx"] // 48] += (
                    ee["sign"] * ee["depth"]
                )
            on, snap = c.gate_state(
                gap=gap_i, thr=gi["thr"], origin_days=origin_days_m, release_days=1
            )
            hold = c.hold_from(snap=snap)
            STATE["pass"] = r
            event_panels(
                t=t, rows=rows, arms=None, obs=t.obs[rows], nra=t.obs[rows], on=on, hold=hold,
                onset=e["onset_idx"], end=e["end_idx"], prim=p,
                title=[
                    f"{t.label[p]}, {gname}: injected lag-only load-shift, {sign * depth * 100:+.0f}% of capacity for {length} days (pass {r})",
                    "Only the lag inputs carry the shift; the observed power (orange) is the normal running arrangement (black dashed, same line).",
                    "Dark band: q05-q95. Light band: q01-q99. Grey shade: the injected shift in the lags. Coloured strips: origin days in event mode (A2's gate).",
                ],
                out=f"injected_{gname.split()[0].lower()}_{length}d_{'up' if sign > 0 else 'down'}_{int(depth * 100)}pct.svg",
            )  # fmt: skip
    STATE["pass"] = None


def tails_figure() -> None:
    b = merged(name="bank_summary.csv").filter(
        (pl.col("band") == "24-120 h") & (pl.col("length_class") == "15 and 30 days")
        & (pl.col("phase") == "after0-7") & (pl.col("group") == "both groups")
    )  # fmt: skip
    long_ = pl.concat(
        [
            b.select(
                "arm", pl.lit(q).alias("quantile"), (pl.col(f"pin{q[1:]}") * 100).alias("pinball")
            )
            for q in ("q01", "q05", "q95", "q99")
        ]
    )
    arms = [a for a in COLOUR if a in set(b["arm"].to_list())]
    alt.Chart(long_).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("arm:N", sort=arms, title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("pinball:Q", title="Mean pinball loss (% of capacity; lower is better)"),
        color=alt.Color("arm:N", scale=alt.Scale(domain=arms, range=[COLOUR[a] for a in arms]), legend=None),
        column=alt.Column("quantile:N", sort=["q01", "q05", "q95", "q99"], title=None),
    ).properties(width=110, height=200, title=alt.TitleParams(["Tail pinball loss, injected events of 15 and 30 days, 0-7 days after the end, both groups, 24-120 h"], fontSize=11, anchor="start")).save(str(DOCS / "tail_pinball.svg"))  # fmt: skip


def false_alarm_figure() -> None:
    fr = pl.concat(
        [pl.read_csv(ROOT / f"score_{a.lower()}/event_free.csv").filter(pl.col("arm") == "V").with_columns(arm=pl.lit(a)) for a in ARMS],
        how="diagonal",
    ).filter((pl.col("band") == "24-120 h") & pl.col("group").is_in(list(s3.GROUPS_SCORED)))  # fmt: skip
    rows = [{"arm": r["arm"], "group": r["group"], "band": lab, "loss": r[k] * 100} for r in fr.iter_rows(named=True) for k, lab in (("loss90", "q05-q95 band"), ("loss98", "q01-q99 band")) if k in r]  # fmt: skip
    if not rows:
        return
    df = pl.DataFrame(rows)
    cols = []
    for g in s3.GROUPS_SCORED:
        base = alt.Chart(df.filter(pl.col("group") == g)).encode(
            y=alt.Y("arm:N", sort=list(ARMS), title=None),
            x=alt.X("loss:Q", title="Mean pinball loss (% of capacity; lower is better)", scale=alt.Scale(zero=False)),
            color=alt.Color("band:N", scale=alt.Scale(range=[theme.DATA_BLUE, theme.BRAND_ORANGE]), legend=alt.Legend(title="Pinball loss of", orient="top")),
            yOffset="band:N",
        )  # fmt: skip
        cols.append(
            base.mark_point(filled=True, size=50, aria=False).properties(  # ty: ignore[unresolved-attribute]
                width=240, height=160, title=g
            )
        )  # ty: ignore[unresolved-attribute]
    alt.hconcat(*cols).properties(title=alt.TitleParams(["Pinball loss on rows with no load-shift within 7 days, 24-120 h"], fontSize=11, anchor="start")).save(str(DOCS / "event_free_pinball.svg"))  # fmt: skip


def r2_figure() -> None:
    d = pl.concat(
        [
            pl.read_csv(ROOT / f"probe_tables/r2_{a.lower()}.csv").with_columns(arm=pl.lit(a))
            for a in ("A1", "A2", "A3", "A4")
        ]
    )
    d = d.group_by("arm", "depth").agg(pl.col("sse").sum(), pl.col("sst").sum()).with_columns(r2=1 - pl.col("sse") / pl.col("sst")).sort("arm", "depth")  # fmt: skip
    arms = ["A1", "A2", "A3", "A4"]
    alt.Chart(d).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("arm:N", sort=arms, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("r2:Q", title="R² of the head's output against the injected depth"),
        color=alt.Color("arm:N", scale=alt.Scale(domain=arms, range=[COLOUR[a] for a in arms]), legend=None),
        column=alt.Column("depth:O", title="Injected depth (share of capacity)"),
    ).properties(width=100, height=180, title=alt.TitleParams(["How much of the injected depth the auxiliary heads recover (held-out injected events; A1's head is untrained and acts as a floor)"], fontSize=11, anchor="start")).save(str(DOCS / "head_r2_by_depth.svg"))  # fmt: skip


def detector_figure() -> None:
    det = pl.concat(
        [
            pl.read_csv(ROOT / f"probe_tables/detector_{a.lower()}.csv").with_columns(arm=pl.lit(a))
            for a in ("A2", "A3", "A4")
        ]
    )
    fa = pl.concat(
        [
            pl.read_csv(ROOT / f"probe_tables/false_triggers_{a.lower()}.csv").with_columns(
                arm=pl.lit(a)
            )
            for a in ("A2", "A3", "A4")
        ]
    )
    d = det.group_by("arm", "thr").agg(detected=(pl.col("detected") * pl.col("n_events")).sum() / pl.col("n_events").sum()).join(
        fa.group_by("arm", "thr").agg(pl.col("n_days").sum(), pl.col("n_fire").sum()).with_columns(false_per_primary_year=pl.col("n_fire") / (pl.col("n_days") / 365.0)), on=["arm", "thr"])  # fmt: skip
    arms = ["A2", "A3", "A4"]
    alt.Chart(d).mark_line(point=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("false_per_primary_year:Q", title="False triggers per Primary-year on event-free days"),
        y=alt.Y("detected:Q", title="Share of injected events detected", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("arm:N", scale=alt.Scale(domain=arms, range=[COLOUR[a] for a in arms]), legend=alt.Legend(title="Arm", orient="top")),
        order="thr:Q",
    ).properties(width=420, height=240, title=alt.TitleParams(["Head as a detector: points are thresholds 0.05, 0.075, and 0.10 of capacity"], fontSize=11, anchor="start")).save(str(DOCS / "head_detector.svg"))  # fmt: skip


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    pack = ROOT / "score_pack"
    pack.mkdir(exist_ok=True)
    for name in ("bank_summary.csv", "real_event_phase.csv"):
        merged(name=name).write_csv(pack / name)
    sc4.setup("a2")
    t = c.X4Table()
    keys = s3.key(t.prim, t.origin_idx, t.idx)
    for a in ARMS:
        z = np.load(ROOT / f"score_{a.lower()}/row_arrays.npz")
        ka = s3.key(z["prim"], z["origin_idx"], z["idx"])
        oa, ko = np.argsort(ka), np.argsort(keys)
        assert np.array_equal(ka[oa], keys[ko]), a
        aligned = np.empty_like(z["V"])
        aligned[ko] = z["V"][oa]  # arms store the same rows in different orders
        STATE.setdefault("V", {})[a] = aligned
    gi = c.gate_inputs(t=t)
    z = np.load(ROOT / "score_a2/gate_state.npz")
    st = {"on": z["on"], "hold": z["hold"]}
    STATE["pass"] = None
    f3.verified_figures(t=t, st=st)
    bank_examples(t=t, gi=gi)
    c.WORK = pack
    f3.coverage_figures()
    tails_figure()
    false_alarm_figure()
    r2_figure()
    detector_figure()


if __name__ == "__main__":
    main()

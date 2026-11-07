# ruff: noqa: E501, D103, ANN001, DTZ001
"""Experiment 612 figures (CPU). Run the scoring, `e00612_summary.py` and `e00612_tails.py` first.

`python e00612_figures.py summary` draws the leave-one-out chart and the tail-constant chart.
`python e00612_figures.py real` draws the December 2025 example event for the two Primaries D5 and D16:
K1 over K2 (day-ahead forecasts, one arm per panel), and K3 alone. Demand series only (Primaries).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00584_score as k584  # ty: ignore[unresolved-import]
import e00588_clean_panels as cp  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00597_figures as f597  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SWITCH_TIMES = [datetime(2025, 12, 30, 1, 0), datetime(2026, 1, 14, 11, 30)]
IQ = c.IQ


def save(chart, name: str) -> None:
    chart.save(str(DOCS / name))


def summary() -> None:
    t = pl.read_csv(ROOT / "score" / "leave_one_out.csv")
    long = pl.concat(
        [
            t.select(
                arm="arm",
                measure=pl.lit("Coverage, day 4 to end"),
                d="cover_days4",
                lo="cover_days4_lo",
                hi="cover_days4_hi",
            ),
            t.select(
                arm="arm",
                measure=pl.lit("Coverage, 0-7 days after"),
                d="cover_after",
                lo="cover_after_lo",
                hi="cover_after_hi",
            ),
            t.select(
                arm="arm",
                measure=pl.lit("Median follow of the shift"),
                d="follow",
                lo="follow_lo",
                hi="follow_hi",
            ),
        ]
    )
    order = t["arm"].to_list()
    panels = []
    for j, measure in enumerate(
        ["Coverage, day 4 to end", "Coverage, 0-7 days after", "Median follow of the shift"]
    ):
        d = long.filter(pl.col("measure") == measure)
        base = alt.Chart(d).encode(
            y=alt.Y(
                "arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=220) if j == 0 else None
            )
        )
        pts = base.mark_point(filled=True, size=60, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("d:Q", title=f"{measure}: arm minus K1 (dashed = K1)")
        )
        bars = base.mark_rule(color=theme.DATA_BLUE).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color=theme.BRAND_ORANGE)
            .encode(x="x:Q")
        )
        panels.append(alt.layer(zero, bars, pts).properties(width=260, height=280))
    chart = alt.hconcat(*panels).resolve_scale(x="independent")
    save(chart, "leave_one_out.svg")

    tails = pl.read_csv(ROOT / "score" / "tails_k1_d5" / "tails_event_free.csv")
    w = (
        tails.group_by("group", "band", "arm")
        .agg((pl.col("cover90") * pl.col("n_rows")).sum() / pl.col("n_rows").sum())
        .with_columns(
            band=pl.when(pl.col("band") == 0).then(pl.lit("0-24 h")).otherwise(pl.lit("24-120 h"))
        )
    )
    arms = ["K1 (G4)", "K1 + tail constants", "K3 (lag-free)"]
    w = w.filter(pl.col("arm").is_in(arms) & (pl.col("band") == "24-120 h")).with_columns(
        group=pl.col("group").replace(
            {"signed-MW six": "Six MW Primaries", "MVA-only eight": "Eight MVA Primaries"}
        )
    )
    target = (
        alt.Chart(pl.DataFrame({"y": [0.9]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#6b6b6b")
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0.8, 1.0])))
    )
    panels = []
    for k, group in enumerate(sorted(w["group"].unique().to_list())):
        dots = (
            alt.Chart(w.filter(pl.col("group") == group))  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=120)
            .encode(
                x=alt.X("arm:N", title=group, axis=None, sort=arms),
                y=alt.Y(
                    "cover90:Q",
                    title="Share of rows inside q05-q95" if k == 0 else None,
                    scale=alt.Scale(domain=[0.8, 1.0]),
                ),
                color=alt.Color(
                    "arm:N",
                    legend=alt.Legend(title="Network (dashed line: 0.90 target)")
                    if k == 0
                    else None,
                    scale=alt.Scale(
                        domain=arms, range=[theme.DATA_BLUE, theme.BRAND_ORANGE, theme.DATA_SKY]
                    ),
                ),
            )
        )
        panels.append(alt.layer(target, dots).properties(width=220, height=280, title=group))
    chart = alt.hconcat(*panels)
    save(chart, "event_free_coverage.svg")


def aligned(*, root_arm: str, tag: str, t) -> np.ndarray:
    """Return the quantiles of the fits `arms/<root_arm>/work/fits/<tag>` aligned to the table rows (NaN where absent)."""
    e = g.a._load(root=ROOT / "arms" / root_arm / "work", tag=tag, months=MONTHS)
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    e = e.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=((pl.col("valid_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US).cast(pl.Int64),
        oidx=((pl.col("power_fcst_init_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US).cast(pl.Int64),
    )  # fmt: skip
    kk = g.s.key(e["prim"].to_numpy(), e["oidx"].to_numpy(), e["idx"].to_numpy())
    order = np.argsort(kk)
    mine = g.s.key(t.prim, t.origin_idx, t.idx)
    pos = np.clip(np.searchsorted(kk[order], mine), 0, len(kk) - 1)
    ok = kk[order][pos] == mine
    q = np.sort(e.select(list(a.QCOLS)).to_numpy().astype(np.float32)[order[pos]], axis=1)
    q[~ok] = np.nan
    return q


def day_rows(*, t, p: int, lo: int, hi: int) -> np.ndarray:
    """Return the table rows of the day-ahead forecast (issued 09:00 the day before) of Primary `p`, valid in [lo, hi)."""
    _, rows = f597.day_ahead_keys(t=t, prim=p, lo=lo, hi=hi)
    return rows


def situations() -> None:
    """Draw K1 and K2, then K3 and the class 2 candidate, for five situations: the two December Primaries, a quiet week, a cold night, a sunny day."""
    c.MONTHS = MONTHS
    work = ROOT / "arms" / "k1_d5" / "work"
    k584.ARMS["k1_d5"] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}
    twin_root = ROOT / "arms" / "k3_d5" / "work"
    orig = g.a._load
    g.a._load = lambda *, root, tag, months: orig(root=twin_root if tag == "D5" else root, tag=tag, months=months)
    k597.WORK = ROOT / "score" / "figures_tmp"
    inp = k597.Inputs(arm="k1_d5")
    t = inp.t
    g.a._load = orig
    q3_all = aligned(root_arm="k3_d5", tag="D5", t=t)
    q4_all = aligned(root_arm="c2_e3", tag="P5", t=t)
    when = lambda i: (sc.k.k.GRID_START + timedelta(minutes=30 * float(i))).replace(tzinfo=None)  # noqa: E731
    eight = [p for p in range(16) if t.group[p] == "MVA-only eight"]
    six = [p for p in range(16) if t.group[p] == "signed-MW six"]

    def draw(*, name: str, p: int, lo: int, hi: int, title: str, event=None, rules=()) -> None:
        rows = day_rows(t=t, p=p, lo=lo, hi=hi)
        v, m, o = t.arms["V"][rows], t.arms["M"][rows], t.obs[rows]
        q1 = k597.assemble(v=v, m=m, wide=inp.mode[t.prim[rows], t.origin_day[rows]])
        q3, q4 = q3_all[rows], q4_all[rows]
        x = [when(i) for i in t.idx[rows]]
        label = t.label[p]
        ydom = cp.common_ydom([(q1, o), (v, o), (np.nan_to_num(q3), o), (np.nan_to_num(q4), o)], iq=IQ)
        common = {"x": x, "iq": IQ, "obs": o, "ydom": ydom, "xtype": "T", "xdom": (when(lo), when(hi)),
                  "span": None if event is None else (when(event[0]), when(event[1])), "rules": list(rules)}  # fmt: skip
        strip = None
        if event is not None:
            strip = [{"x0": when(rr["x0"] * 48 + event[0]), "x1": when(rr["x1"] * 48 + event[0])}
                     for rr in f597.strips_for(t=t, rows=rows, wide=inp.mode, prim=p, onset=event[0])]  # fmt: skip
        else:
            on = inp.mode[t.prim[rows], t.origin_day[rows]]
            strip = [{"x0": x[i], "x1": x[min(i + 1, len(x) - 1)]} for i in np.flatnonzero(on)] if on.any() else None
        key_a = [*cp.FORECAST_ENTRIES, cp.OBS, *([cp.SHADE] if event else []), *([cp.GATE] if strip else []), *([cp.RULES] if rules else [])]
        key_b = [*cp.FORECAST_ENTRIES, cp.OBS, *([cp.SHADE] if event else []), *([cp.RULES] if rules else [])]
        save(cp.figure(units=[cp.arm_panel(q=q1, title=f"{label}: K1 (lag network with the G4 gated band)", strip=strip, **common),
                              cp.arm_panel(q=v, title=f"{label}: K2 (same fits, no gate)", xtitle="Date (UTC)", strip=None, **common)],
                       title=f"{label}: {title}", key=key_a), f"{name}_{label}_k1_k2.svg")  # fmt: skip
        save(cp.figure(units=[cp.arm_panel(q=q3, title=f"{label}: K3 (lag-free)", strip=None, **common),
                              cp.arm_panel(q=q4, title=f"{label}: class 2 candidate (neighbour summaries, signal dropout)", xtitle="Date (UTC)", strip=None, **common)],
                       title=f"{label}: {title}", key=key_b), f"{name}_{label}_k3_c2.svg")  # fmt: skip

    ev = sc.attach_depth(ev=sc.merged_events(), t=t).filter(pl.col("name").str.contains("2025-12-29"))
    for r in ev.sort("name").iter_rows(named=True):
        draw(name="event", p=r["primary"], lo=r["onset_idx"] - 14 * 48, hi=r["end_idx"] + 14 * 48,
             title="day-ahead forecasts through the December 2025 load-shift", event=(r["onset_idx"], r["end_idx"]), rules=SWITCH_TIMES)  # fmt: skip

    def pick(*, p: int, month: str, days: int, score) -> int:
        """Return the first grid index of the best `days`-day window of Primary `p` in `month` by `score` (lower is better)."""
        ok = (t.prim == p) & (t.month == month) & t.free & (t.band == 1)
        days_present = np.unique(t.idx[ok] // 48)
        best, best_s = None, np.inf
        for d0 in days_present[:-days]:
            w = ok & (t.idx // 48 >= d0) & (t.idx // 48 < d0 + days)
            if w.sum() < days * 40:
                continue
            sc_ = score(w)
            if sc_ < best_s:
                best, best_s = int(d0), sc_
        assert best is not None
        return best * 48

    p = eight[0]
    lo = pick(p=p, month="2026-03", days=7, score=lambda w: float(np.nanmean(np.abs(t.arms["V"][w][:, IQ[0.5]] - t.obs[w]))))
    draw(name="quiet_week", p=p, lo=lo, hi=lo + 7 * 48, title="a quiet week (event-free, 7 days with the smallest mean median-forecast error in March 2026)")
    hour = lambda w: ((t.idx[w] % 48) / 2.0)  # noqa: E731
    lo = pick(p=p, month="2026-01", days=3, score=lambda w: -float(np.nanmean(t.obs[w][hour(w) < 7])))
    draw(name="cold_night", p=p, lo=lo, hi=lo + 3 * 48, title="three days around the highest overnight demand in January 2026 (a proxy for a cold winter night)")
    p6 = six[0]
    lo = pick(p=p6, month="2026-06", days=3, score=lambda w: float(np.nanmean(t.obs[w][(hour(w) >= 11) & (hour(w) < 14)]) - np.nanmean(t.obs[w][hour(w) < 5])))
    draw(name="sunny_day", p=p6, lo=lo, hi=lo + 3 * 48, title="three days around the deepest midday dip in June 2026 (a proxy for a very sunny day)")


if __name__ == "__main__":
    {"summary": summary, "situations": situations}[sys.argv[1]]()

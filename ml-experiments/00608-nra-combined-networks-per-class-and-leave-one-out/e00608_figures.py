# ruff: noqa: E501, D103, ANN001, DTZ001
"""Experiment 608 figures (CPU). Run the scoring, `e00608_summary.py` and `e00608_tails.py` first.

`python e00608_figures.py summary` draws the leave-one-out chart and the tail-constant chart.
`python e00608_figures.py real` draws the December 2025 example event for the two Primaries D5 and D16:
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

NAME = "00608-nra-combined-networks-per-class-and-leave-one-out"
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


def real() -> None:
    c.MONTHS = MONTHS
    work = ROOT / "arms" / "k1_d5" / "work"
    k584.ARMS["k1_d5"] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}
    twin_root = ROOT / "arms" / "k3_d5" / "work"
    orig = g.a._load
    g.a._load = lambda *, root, tag, months: orig(
        root=twin_root if tag == "D5" else root, tag=tag, months=months
    )
    k597.WORK = ROOT / "score" / "figures_tmp"
    inp = k597.Inputs(arm="k1_d5")
    t = inp.t
    e = orig(root=twin_root, tag="D5", months=MONTHS)
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    e = e.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=(
            (pl.col("valid_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US
        ).cast(pl.Int64),
        oidx=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - c.k578.prep.GRID_START_US)
            // c.k578.prep.STEP_US
        ).cast(pl.Int64),
    )
    kk = g.s.key(e["prim"].to_numpy(), e["oidx"].to_numpy(), e["idx"].to_numpy())
    order = np.argsort(kk)
    ev = sc.attach_depth(ev=sc.merged_events(), t=t).filter(
        pl.col("name").str.contains("2025-12-29")
    )
    when = lambda i: (sc.k.k.GRID_START + timedelta(minutes=30 * float(i))).replace(tzinfo=None)  # noqa: E731
    for r in ev.sort("name").iter_rows(named=True):
        p, onset, end = r["primary"], r["onset_idx"], r["end_idx"]
        keys, rows = f597.day_ahead_keys(t=t, prim=p, lo=onset - 14 * 48, hi=end + 15 * 48)
        pos = np.clip(np.searchsorted(kk[order], keys), 0, len(kk) - 1)
        ok = kk[order][pos] == keys
        rows, pos = rows[ok], pos[ok]
        q3 = np.sort(e.select(list(a.QCOLS)).to_numpy().astype(np.float32)[order[pos]], axis=1)
        v, m = t.arms["V"][rows], t.arms["M"][rows]
        wide = inp.mode[t.prim[rows], t.origin_day[rows]]
        q1 = k597.assemble(v=v, m=m, wide=wide)
        x = [when(i) for i in t.idx[rows]]
        label = t.label[p]
        # which window matches the step: mean observed power in the 2 days before and after each candidate start
        obs_all = t.obs[rows]
        for cand in (datetime(2025, 12, 29), datetime(2025, 12, 30, 1, 0)):
            xs = np.array([(xx - cand).total_seconds() / 86400 for xx in x])
            before = obs_all[(xs > -2) & (xs < 0)]
            after = obs_all[(xs > 0) & (xs < 2)]
            print(
                label,
                cand,
                "mean power before/after (share of capacity):",
                np.nanmean(before),
                np.nanmean(after),
            )
        ydom = cp.common_ydom([(q1, t.obs[rows]), (v, t.obs[rows]), (q3, t.obs[rows])], iq=IQ)
        strip = [
            {"x0": when(rr["x0"] * 48 + onset), "x1": when(rr["x1"] * 48 + onset)}
            for rr in f597.strips_for(t=t, rows=rows, wide=inp.mode, prim=p, onset=onset)
        ]
        common = {
            "x": x, "iq": IQ, "obs": t.obs[rows], "ydom": ydom, "xtype": "T",
            "xdom": (when(onset - 14 * 48), when(end + 14 * 48)),
            "span": (when(onset), when(end)), "rules": SWITCH_TIMES,
        }  # fmt: skip
        key = [*cp.FORECAST_ENTRIES, cp.OBS, cp.SHADE, cp.GATE, cp.RULES]
        fig = cp.figure(
            units=[
                cp.arm_panel(
                    q=q1, title=f"{label}: K1 (with the G4 gated band)", strip=strip, **common
                ),
                cp.arm_panel(
                    q=v,
                    title=f"{label}: K2 (no explicit gate)",
                    xtitle="Date (UTC)",
                    strip=None,
                    **common,
                ),
            ],
            title=f"{label}: day-ahead forecasts through the December 2025 load-shift (issued 09:00 the day before)",
            key=key,
        )
        save(fig, f"real_{label}_k1_k2.svg")
        fig3 = cp.figure(
            units=[
                cp.arm_panel(
                    q=q3, title=f"{label}: K3 (lag-free)", xtitle="Date (UTC)", strip=None, **common
                )
            ],
            title=f"{label}: day-ahead forecast of the lag-free network K3 through the same load-shift",
            key=[*cp.FORECAST_ENTRIES, cp.OBS, cp.SHADE, cp.RULES],
        )
        save(fig3, f"real_{label}_k3.svg")


if __name__ == "__main__":
    {"summary": summary, "real": real}[sys.argv[1]]()

# ruff: noqa: E501, D103, ANN001, ANN202, D102,D107,PERF401
"""Experiment 651 figures and tables (CPU; run the scoring first).

Usage (through `cpu.sh`): `cpu.sh e00651_figures.py headline|tails|events|examples|all`

* `headline` draws `headline.svg` (follow on banks M, S, X; six-MW event-free CRPS) and prints its numbers.
* `tails` writes `score/tail_table.csv` and prints the tail pinball losses and coverage at 24-120 h.
* `events` draws the load-shift event figures (`event_*.svg`): the December 2025 events of D5 and
  D16 on real data, and six injected events of bank M, three on a six-MW Primary and three on an
  MVA-only eight Primary. One arm per panel, two arms per figure.
* `examples` draws `example_day_ahead.svg` and `example_5day.svg`.

Demand series (Primaries) only; no generator is plotted. Development months 2025-10 and 2025-12 only.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import e00588_clean_panels as cp  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00613_nbscore as nbs  # ty: ignore[unresolved-import]
import e00651_score as sc651  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

ROOT = sc651.ROOT
SCORE = ROOT / "score"
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
MONTHS = ("2025-10", "2025-12")
SEEDS = (0, 1)
IQ = c.IQ
BAND = "24-120 h"
ARMS_ORDER = ["A1", "C", "G0", "GCN-E", "GAT-E", "GAT-R", "GAT-F", "FLOW-E"]
ARM_NAMES = {
    "A1": "A1 (no neighbours)",
    "C": "C (stored Candidate)",
    "G0": "G0 (own gap signals)",
    "GCN-E": "GCN-E (convolution)",
    "GAT-E": "GAT-E (attention)",
    "GAT-R": "GAT-R (rewired placebo)",
    "GAT-F": "GAT-F (all Primaries)",
    "FLOW-E": "FLOW-E (edge flows)",
}
GROUP_SIX, GROUP_EIGHT = "signed-MW six", "MVA-only eight"
BANK_COLOURS = {"M": theme.DATA_BLUE, "S": theme.BRAND_ORANGE, "X": theme.DATA_GREEN}
BANK_NAMES = {
    "M": "Bank M (shift mirrored on one neighbour)",
    "S": "Bank S (shift split over two neighbours)",
    "X": "Bank X (no mirror)",
}
CONTENT_PX = 680


def save(chart, name: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.save(str(DOCS / name))
    print("wrote", DOCS / name, flush=True)


# ------------------------------------------------------------------ headline and tails


def headline() -> None:
    rows = []
    for bank in ("M", "S", "X"):
        d = pl.read_csv(SCORE / bank / "arm_summary.csv", infer_schema_length=None).filter(
            pl.col("system").is_in(ARMS_ORDER)
        )
        for r in d.iter_rows(named=True):
            rows.append(
                {"arm": ARM_NAMES[r["system"]], "bank": BANK_NAMES[bank], "bank_id": bank,
                 "est": r["follow_q50_days4-end"], "lo": r["follow_q50_days4-end_lo"],
                 "hi": r["follow_q50_days4-end_hi"], "n_events": r["n_events"]}
            )  # fmt: skip
    fol = pl.DataFrame(rows)
    print("\nFollow (share of the injected shift followed, day 4 to end; smaller is better)")
    print(
        fol.select("arm", "bank_id", "est", "lo", "hi", "n_events")
        .with_columns(pl.col("est", "lo", "hi").round(3))
        .sort(
            "bank_id",
            pl.col("arm").replace_strict({v: i for i, v in enumerate(ARM_NAMES.values())}),
        )
    )
    order = [ARM_NAMES[a] for a in ARMS_ORDER]
    bank_scale = alt.Scale(domain=list(BANK_NAMES.values()), range=list(BANK_COLOURS.values()))
    base = alt.Chart(fol).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("bank:N", sort=list(BANK_NAMES.values())),
        color=alt.Color(
            "bank:N", scale=bank_scale,
            legend=alt.Legend(title=None, orient="top", direction="vertical", columns=1, labelLimit=330),
        ),
    )  # fmt: skip
    bars = base.mark_rule(strokeWidth=2.5).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="Share of the injected shift followed, day 4 to end (smaller is better)", scale=alt.Scale(domain=[-0.005, 0.22], nice=False), axis=alt.Axis(tickCount=6)),
        x2="hi:Q",
    )  # fmt: skip
    dots = base.mark_point(filled=True, size=45).encode(x=alt.X("est:Q"))  # ty: ignore[unresolved-attribute]
    panel_a = alt.layer(bars, dots).properties(
        width=470, height=300,
        title=alt.TitleParams(["(a) Follow of a false load-shift, 252 injected events per bank. Dot: estimate. Line: 99.75% bootstrap", "interval over event clusters. A forecast that ignored the false shift entirely would score 0."], fontSize=11, anchor="start"),
    )  # fmt: skip

    ef = (
        pl.read_csv(SCORE / "M" / "event_free.csv", infer_schema_length=None)
        .filter((pl.col("band") == BAND) & (pl.col("group") == GROUP_SIX) & pl.col("system").is_in(ARMS_ORDER))
        .filter(pl.col("system") != "A1")
        .select("system", "crps", "crps_minus_A1", "crps_minus_A1_lo", "crps_minus_A1_hi")
        .with_columns(arm=pl.col("system").replace_strict(ARM_NAMES))
    )  # fmt: skip
    print(
        "\nSix-MW event-free CRPS, 24-120 h, points of capacity (minus A1; no paired C contrast is stored)"
    )
    print(
        ef.with_columns(
            pl.col("crps", "crps_minus_A1", "crps_minus_A1_lo", "crps_minus_A1_hi").round(4)
        )
    )
    base_b = alt.Chart(ef).encode(
        y=alt.Y(
            "arm:N",
            sort=[ARM_NAMES[a] for a in ARMS_ORDER if a != "A1"],
            title=None,
            axis=alt.Axis(labelLimit=200),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=theme.BRAND_ORANGE)
        .encode(x="x:Q")
    )
    bars_b = base_b.mark_rule(color=theme.DATA_BLUE, strokeWidth=2.5).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("crps_minus_A1_lo:Q", title="Event-free CRPS minus A1's (points of capacity; smaller is better)", scale=alt.Scale(domain=[-0.01, 0.25], nice=False), axis=alt.Axis(tickCount=6)),
        x2="crps_minus_A1_hi:Q",
    )  # fmt: skip
    dots_b = base_b.mark_point(filled=True, size=45, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x="crps_minus_A1:Q"
    )
    panel_b = alt.layer(zero, bars_b, dots_b).properties(
        width=470, height=190,
        title=alt.TitleParams(["(b) Event-free CRPS, six MW Primaries, 24-120 h ahead, 2025-10 and 2025-12. Dot: estimate. Line: 99.75%", "bootstrap interval over the two months. Dashed line: A1, the pretrained lag encoder without neighbours."], fontSize=11, anchor="start"),
    )  # fmt: skip
    chart = (
        alt.vconcat(panel_a, panel_b, spacing=24)
        .resolve_scale(x="independent", color="independent")
        .configure_view(stroke=None)
    )  # fmt: skip
    save(chart, "headline.svg")


def tails() -> None:
    ef = pl.read_csv(SCORE / "M" / "event_free.csv", infer_schema_length=None)
    d = (
        ef.filter((pl.col("band") == BAND) & pl.col("group").is_in([GROUP_SIX, GROUP_EIGHT]) & pl.col("system").is_in(ARMS_ORDER))
        .select("group", "system", "n_rows", "crps", "pin01", "pin05", "pin95", "pin99", "cover90")
        .with_columns(
            order=pl.col("system").replace_strict({a: i for i, a in enumerate(ARMS_ORDER)}),
            group_order=pl.col("group").replace_strict({GROUP_SIX: 0, GROUP_EIGHT: 1}),
        )
        .sort("group_order", "order")
        .drop("order", "group_order")
    )  # fmt: skip
    d.write_csv(SCORE / "tail_table.csv")
    print(
        f"\nTail pinball loss (points of capacity) and coverage of the 90% band, {BAND}, event-free rows; written to {SCORE / 'tail_table.csv'}"
    )
    with pl.Config(tbl_rows=40, tbl_cols=12, fmt_str_lengths=20, tbl_width_chars=160):
        print(
            d.with_columns(
                pl.col("crps", "pin01", "pin05", "pin95", "pin99").round(4),
                pl.col("cover90").round(3),
            )
        )
    print("\nMarkdown table")
    print("| Group | Arm | CRPS | q01 | q05 | q95 | q99 | Coverage of 90% band |")
    print("|---|---|---|---|---|---|---|---|")
    for r in d.iter_rows(named=True):
        print(
            f"| {r['group']} | {r['system']} | {r['crps']:.3f} | {r['pin01']:.3f} | {r['pin05']:.3f} | {r['pin95']:.3f} | {r['pin99']:.3f} | {r['cover90']:.3f} |"
        )


# ------------------------------------------------------------------ forecasts on the table's rows


class World:
    """The row table and a loader of each arm's stored forecasts on it."""

    def __init__(self) -> None:
        self.inp = k597.Inputs(arm="a1")
        self.t = self.inp.t
        kk = s.key(self.t.prim, self.t.origin_idx, self.t.idx)
        self.order = np.argsort(kk)
        self.sorted_keys = kk[self.order]
        self.cache: dict[tuple, np.ndarray] = {}

    def real(self, arm: str) -> np.ndarray:
        """Return the seed-mean quantiles of `arm` on the real rows of both fold months."""
        key = ("real", arm)
        if key not in self.cache:
            work = sc651.ARMS[arm][0] / "work"
            per_seed = []
            for seed in SEEDS:
                files = [
                    work / "fits/P5" / f"{m}_d{d}_s{seed}.parquet" for m in MONTHS for d in (0, 14)
                ]
                per_seed.append(
                    nbs.load_rows(
                        files=files, t=self.t, order=self.order, sorted_keys=self.sorted_keys
                    )
                )
            self.cache[key] = np.sort(np.mean(per_seed, axis=0), axis=1)
        return self.cache[key]

    def injected(self, arm: str, *, month: str, r: int) -> np.ndarray:
        """Return the seed-mean quantiles of `arm` on the rows of injected pass `r` of bank M."""
        key = ("inj", arm, month, r)
        if key not in self.cache:
            work = sc651.ARMS[arm][0] / "work"
            per_seed = []
            for seed in SEEDS:
                files = [
                    work / "inject" / f"{month}_r{r:02d}_d{d}_s{seed}.parquet" for d in (0, 14)
                ]
                per_seed.append(
                    nbs.load_rows(
                        files=files, t=self.t, order=self.order, sorted_keys=self.sorted_keys
                    )
                )
            self.cache[key] = np.sort(np.mean(per_seed, axis=0), axis=1)
        return self.cache[key]

    def day_ahead_rows(self, *, p: int, lo: int, hi: int) -> np.ndarray:
        t = self.t
        sel = (t.prim == p) & (t.idx >= lo) & (t.idx < hi) & (t.idx // 48 == t.origin_idx // 48 + 1)
        rows = np.flatnonzero(sel)
        return rows[np.argsort(t.idx[rows])]


def when(i) -> datetime:
    return (sc.k.k.GRID_START + timedelta(minutes=30 * float(i))).replace(tzinfo=None)


def wrap(text: str, width: int = 100) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return [*lines, cur]


_ORIG_X_PARTS = cp._x_parts


def set_axis(*, kind: str) -> None:
    """Choose the x axis format: `day` for windows of weeks, `hours` for a single day, `days` for a few days."""

    def x_parts(*, xdom, xtype: str, labels: bool):
        scale, axis = _ORIG_X_PARTS(xdom=xdom, xtype=xtype, labels=labels)
        if kind == "hours":
            axis = alt.Axis(format="%H:%M", labels=labels, tickCount=8)
        elif kind == "days":
            axis = alt.Axis(format="%d %b", labels=labels, tickCount={"interval": "day", "step": 1})
        return scale, axis

    cp._x_parts = x_parts


def two_arm_figure(
    *, w: World, arms: tuple[str, str], q: dict[str, np.ndarray], rows: np.ndarray, p: int,
    lo: int, hi: int, title: str, name: str, span=None, lag=None, xtitle="Date (UTC)", span_label="Load-shift, start to end",
) -> None:  # fmt: skip
    t = w.t
    x = [when(i) for i in t.idx[rows]]
    o = t.obs[rows]
    ydom = cp.common_ydom([(np.nan_to_num(q[a][rows], nan=np.nanmean(o)), o) for a in arms], iq=IQ)
    if lag is not None:
        ydom = (
            min(ydom[0], float(np.nanmin(lag)) * 100 - 2),
            max(ydom[1], float(np.nanmax(lag)) * 100 + 2),
        )
    common = {"x": x, "iq": IQ, "obs": o, "ydom": ydom, "xtype": "T", "xdom": (when(lo), when(hi)),
              "span": None if span is None else (when(span[0]), when(span[1])), "label_nra": lag}  # fmt: skip
    label = t.label[p]
    units = [
        cp.arm_panel(q=q[a][rows], title=f"Primary {label}: {ARM_NAMES.get(a, a)}", xtitle=xtitle if i == 1 else None, strip=None, **common)
        for i, a in enumerate(arms)
    ]  # fmt: skip
    key = [*cp.FORECAST_ENTRIES, cp.OBS]
    if span is not None:
        key.append(cp.entry(span_label, "band", cp.SHADE_COLOUR, 0.7))
    if lag is not None:
        key.append(cp.entry("Power the lag inputs saw (shifted span only)", "dashed", "#306BFF"))
    save(cp.figure(units=units, title=wrap(title), key=key), name)


# ------------------------------------------------------------------ the event figures


def events() -> None:
    w = World()
    t = w.t
    cp.PANEL_HEIGHT = 280
    set_axis(kind="days")
    labels = list(t.label)
    # real events: the December 2025 events of D5 and D16
    ev = (
        sc.attach_depth(ev=sc.merged_events(), t=t)
        .filter(pl.col("name").str.contains("2025-12-29"))
        .sort("name")
    )
    need = {"C", "GAT-E", "G0", "FLOW-E"}
    q_real = {a: w.real(a) for a in need}
    for r in ev.iter_rows(named=True):
        p = r["primary"]
        lo, hi = r["onset_idx"] - 7 * 48, r["end_idx"] + 7 * 48
        rows = w.day_ahead_rows(p=p, lo=lo, hi=hi)
        fin = np.isfinite(q_real["C"][rows, 0]) & np.isfinite(q_real["GAT-E"][rows, 0])
        rows = rows[fin]
        lab = labels[p]
        print(
            f"real event {lab}: onset {when(r['onset_idx'])}, end {when(r['end_idx'])}, forecasts valid {when(t.idx[rows].min())} to {when(t.idx[rows].max())}, depth {r['depth_share']:.3f} of capacity ({r['group']})",
            flush=True,
        )
        title = f"Primary {lab}, real load-shift of December 2025 ({r['group']} group): day-ahead forecasts of C and GAT-E through the event. The stored forecasts end on {when(t.idx[rows].max()).strftime('%d %B %Y')}, before the event ends on {when(r['end_idx']).strftime('%d %B %Y')}."
        two_arm_figure(w=w, arms=("C", "GAT-E"), q=q_real, rows=rows, p=p, lo=int(t.idx[rows].min()), hi=int(t.idx[rows].max()),
                       title=title, name=f"event_real_{lab}_C_GAT-E.svg", span=(r["onset_idx"], r["end_idx"]))  # fmt: skip
        if lab == "D5":
            title = f"Primary {lab}, real load-shift of December 2025: day-ahead forecasts of G0 (own gap signals) and FLOW-E (edge flows)."
            two_arm_figure(w=w, arms=("G0", "FLOW-E"), q=q_real, rows=rows, p=p, lo=int(t.idx[rows].min()), hi=int(t.idx[rows].max()),
                           title=title, name=f"event_real_{lab}_G0_FLOW-E.svg", span=(r["onset_idx"], r["end_idx"]))  # fmt: skip
    # injected events of bank M
    bank = pl.read_parquet(s.BANK_PARQUET).filter(pl.col("month").is_in(list(MONTHS)))
    plan = [
        (GROUP_SIX, +1, "2025-10", 0.30, 15), (GROUP_SIX, -1, "2025-12", 0.30, 15), (GROUP_SIX, +1, "2025-12", 0.15, 15),
        (GROUP_EIGHT, -1, "2025-10", 0.30, 15), (GROUP_EIGHT, +1, "2025-12", 0.30, 15), (GROUP_EIGHT, -1, "2025-12", 0.15, 15),
    ]  # fmt: skip
    q_inj_cache: dict[tuple, dict[str, np.ndarray]] = {}
    for k, (grp, sign, month, depth, length) in enumerate(plan):
        prims = [p for p in range(16) if t.group[p] == grp]
        cand = bank.filter(
            pl.col("primary").is_in(prims) & (pl.col("month") == month) & (pl.col("sign") == sign)
            & ((pl.col("depth") - depth).abs() < 1e-9) & (pl.col("length_days") == length)
        ).sort("primary")  # fmt: skip
        assert cand.height, (grp, sign, month, depth, length)
        e = cand.row(
            k % 3 % cand.height, named=True
        )  # a different Primary for each of the three events of a group
        p, r = e["primary"], e["pass"]
        lo, hi = e["onset_idx"] - 7 * 48, e["end_idx"] + 14 * 48
        rows = w.day_ahead_rows(p=p, lo=lo, hi=hi)
        q = {a: w.injected(a, month=month, r=r) for a in ("C", "GAT-E")}
        q_inj_cache[(month, r)] = q
        fin = np.isfinite(q["C"][rows, 0]) & np.isfinite(q["GAT-E"][rows, 0])
        rows = rows[fin]
        lab = labels[p]
        shift = sign * depth
        obs = t.obs[rows]
        lag = np.where(
            (t.idx[rows] >= e["onset_idx"]) & (t.idx[rows] < e["end_idx"]), obs + shift, np.nan
        )  # drawn inside the shifted span only
        print(
            f"injected event {lab} {month} pass {r}: shift {shift:+.2f} of capacity for {length} days, {len(rows)} hourly day-ahead rows from {when(t.idx[rows].min())} to {when(t.idx[rows].max())}",
            flush=True,
        )
        sign_word = "up" if sign > 0 else "down"
        title = (f"Primary {lab} ({grp} group), {month} fold: a false load-shift of {abs(shift) * 100:.0f}% of capacity {sign_word} for {length} days is added to the power the lag inputs see (dashed blue). "
                 f"The target power (black) is unchanged, so a median that moves towards the dashed line is wrong. The shift runs from {when(e['onset_idx']).strftime('%d %b')} to {when(e['end_idx']).strftime('%d %b')}; the plot starts at the first stored forecast.")  # fmt: skip
        two_arm_figure(w=w, arms=("C", "GAT-E"), q=q, rows=rows, p=p, lo=int(t.idx[rows].min()), hi=int(t.idx[rows].max()),
                       title=title, name=f"event_injected_{lab}_{month}_{sign_word}{abs(shift) * 100:.0f}.svg", span=(e["onset_idx"], e["end_idx"]), lag=lag, span_label="Injected false shift")  # fmt: skip
        # numbers the page can quote: the injected forecast's median against the same row's uninjected forecast
        inside = (t.idx[rows] >= e["onset_idx"] + 3 * 48) & (t.idx[rows] < e["end_idx"])
        for a in ("C", "GAT-E"):
            twin = w.real(a)[rows, IQ[0.5]]
            moved = (q[a][rows, IQ[0.5]] - twin) * np.sign(shift)
            f = float(np.nanmean(moved[inside]) / abs(shift)) if inside.any() else float("nan")
            print(
                f"   {a}: median moved by the false shift, day 4 to end, as a share of the shift (follow, this event) = {f:+.3f}",
                flush=True,
            )


# ------------------------------------------------------------------ the example forecasts


def examples() -> None:
    w = World()
    t = w.t
    cp.PANEL_HEIGHT = 280
    eight = [p for p in range(16) if t.group[p] == GROUP_EIGHT]
    p = eight[0]
    lab = t.label[p]
    q = {a: w.real(a) for a in ("C", "GAT-E")}
    origins = {}
    for m in MONTHS:
        sel = (t.prim == p) & (t.month == m)
        for o in np.unique(t.origin_idx[sel]):
            rows = np.flatnonzero((t.prim == p) & (t.origin_idx == o))
            rows = rows[np.argsort(t.idx[rows])]
            if len(rows) < 239:
                continue
            fin = (
                np.isfinite(q["C"][rows, 0]).all()
                and np.isfinite(q["GAT-E"][rows, 0]).all()
                and np.isfinite(t.obs[rows]).all()
                and bool(t.free[rows].all())
            )
            if fin:
                origins[(m, int(o))] = rows
    # winter evening: December origin whose next-day evening (16:00 to 20:00 UTC) observed demand is the highest
    best_w, best_v = None, -np.inf
    for (m, o), rows in origins.items():
        if m != "2025-12":
            continue
        nxt = rows[(t.idx[rows] // 48 == o // 48 + 1)]
        hour = (t.idx[nxt] % 48) / 2.0
        v = float(np.mean(t.obs[nxt][(hour >= 16) & (hour < 20)]))
        if v > best_v:
            best_w, best_v = (m, o), v
    # quiet week: October origin whose 5-day observed series repeats itself most closely day to day
    best_q, best_s = None, np.inf
    for (m, o), rows in origins.items():
        if m != "2025-10":
            continue
        ob = t.obs[rows]
        sc_ = float(np.mean(np.abs(ob[48:] - ob[:-48])))
        if sc_ < best_s:
            best_q, best_s = (m, o), sc_
    assert best_w is not None and best_q is not None  # noqa: PT018
    print(
        f"Primary {lab}; winter evening origin {when(best_w[1])} UTC (mean demand 16-20 h next day {best_v * 100:.1f}% of capacity); quiet week origin {when(best_q[1])} UTC (mean day-to-day change {best_s * 100:.2f}% of capacity)",
        flush=True,
    )
    # day-ahead (winter evening)
    (m, o) = best_w
    rows = origins[best_w]
    nxt = rows[(t.idx[rows] // 48 == o // 48 + 1)]
    set_axis(kind="hours")
    title = f"Primary {lab}, day-ahead forecast issued at 09:00 UTC on {when(o).strftime('%d %B %Y')} (the winter evening with the highest demand at 16:00 to 20:00 UTC of the next day among the December 2025 origins). Valid times {when(t.idx[nxt].min()).strftime('%d %B')}."
    two_arm_figure(
        w=w,
        arms=("C", "GAT-E"),
        q=q,
        rows=nxt,
        p=p,
        lo=int(t.idx[nxt].min()),
        hi=int(t.idx[nxt].max()),
        title=title,
        name="example_day_ahead.svg",
        xtitle="Time of day (UTC)",
    )
    # the full horizon (5 days) of a quiet week
    (m, o) = best_q
    rows = origins[best_q]
    set_axis(kind="days")
    title = f"Primary {lab}, forecast issued at 09:00 UTC on {when(o).strftime('%d %B %Y')} for the full horizon of 5 days (the October 2025 origin whose observed power repeats most closely from day to day)."
    two_arm_figure(
        w=w,
        arms=("C", "GAT-E"),
        q=q,
        rows=rows,
        p=p,
        lo=int(t.idx[rows].min()),
        hi=int(t.idx[rows].max()),
        title=title,
        name="example_5day.svg",
    )
    # numbers for the page
    for tag, key in (("winter evening", best_w), ("quiet week", best_q)):
        rr = origins[key]
        ob = t.obs[rr]
        for a in ("C", "GAT-E"):
            qa = q[a][rr]
            mae = float(np.mean(np.abs(qa[:, IQ[0.5]] - ob)) * 100)
            cov = float(np.mean((ob >= qa[:, IQ[0.05]]) & (ob <= qa[:, IQ[0.95]])))
            print(
                f"   {tag} {a}: median MAE {mae:.2f}% of capacity, share of observations inside the 90% band {cov:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    what = sys.argv[1]
    for step in ("headline", "tails", "events", "examples") if what == "all" else (what,):
        {"headline": headline, "tails": tails, "events": events, "examples": examples}[step]()

# ruff: noqa: E501, D103, ANN001, ANN202, PLR0915
"""Experiment 632 figures. Run the fits, `e00632_report.py` and `e00632_inject.py` first.

`python e00632_figures.py headline` draws CRPS by forecast day with intervals;
`ride` draws the injected-bank follow ratio and safe-side share; `examples` draws a week-ahead and a
day-ahead forecast of Seq in winter and summer, and Seq against an injected shift (one arm per panel).
"""

import glob
import sys
from pathlib import Path

import altair as alt
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00632_data as d
import e00632_fold as fold_mod
import e00632_inject as inj
import e00632_report as rep
import numpy as np
import polars as pl
from e00632_score import MONTHS, ROOT
from plotting import ocf_theme as theme
from studies.bootstrap import bootstrap_month_totals

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
GROUPS = {"eight MVA Primaries": "Eight MVA Primaries", "six MW Primaries": "Six MW Primaries"}
COLOURS = {
    "Seq": theme.BRAND_ORANGE, "P2-pre": theme.DATA_BLUE, "K3-pre": theme.DATA_SKY,
    "P2-hist6": theme.DATA_PURPLE, "Climatology (month, half-hour)": theme.ENSEMBLE_LINE,
    "Seq trained on synthetic shifts": theme.DATA_GREEN,
}  # fmt: skip
ARM_LABEL = {"seq": "Seq", "P2-pre": "P2-pre", "K3-pre": "K3-pre", "P2-hist6": "P2-hist6",
             "climatology": "Climatology (month, half-hour)", "seq_ls": "Seq trained on synthetic shifts"}  # fmt: skip
W = 640


def day_intervals(*, arm: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    rows = []
    frames = [rep.scored(arm, (s,)) for s in seeds]
    for g in GROUPS:
        for day in range(1, 8):
            sums, counts = [], None
            for f in frames:
                gg = (
                    f.filter((pl.col("group") == g) & (pl.col("day") == day))
                    .group_by("month").agg(pl.col("crps").sum().alias("s"), pl.len().alias("n")).sort("month")
                )
                sums.append(gg["s"].to_numpy())
                counts = gg["n"].to_numpy()
            lo, hi = bootstrap_month_totals(month_sums=np.stack(sums), month_counts=counts, level=95.0)
            rows.append({"arm": ARM_LABEL[arm], "group": GROUPS[g], "day": day,
                         "crps": float(np.stack(sums).sum(axis=1).mean() / counts.sum()), "lo": lo, "hi": hi})  # fmt: skip
    return pl.DataFrame(rows)


def headline() -> None:
    arms = [("seq", (0, 1, 2)), ("P2-pre", (0, 1, 2)), ("K3-pre", (0, 1, 2)), ("P2-hist6", (0, 1, 2)), ("climatology", (0,))]
    df = pl.concat([day_intervals(arm=a, seeds=s) for a, s in arms if rep.present(a)])
    order = [ARM_LABEL[a] for a, _ in arms if rep.present(a)]
    panels = []
    for g in GROUPS.values():
        sub = df.filter(pl.col("group") == g)
        base = alt.Chart(sub).encode(
            x=alt.X("day:O", title="Forecast day (24 h blocks of lead time)", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("arm:N", sort=order),
            color=alt.Color("arm:N", sort=order, scale=alt.Scale(domain=order, range=[COLOURS[a] for a in order]), legend=alt.Legend(title=None, orient="top", columns=3, labelLimit=400)),
        )
        bars = base.mark_rule(strokeWidth=2).encode(y=alt.Y("lo:Q", title="CRPS (points of capacity; smaller is better)", scale=alt.Scale(zero=False)), y2="hi:Q")
        pts = base.mark_point(filled=True, size=40).encode(y="crps:Q")
        panels.append(alt.layer(bars, pts).properties(width=W - 90, height=250, title=g))
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Seq ties P2-pre and both beat K3-pre; climatology is far behind",
            subtitle=["Dot: mean CRPS of event-free rows, five fold months, ERA5 and CAMS weather. Line: 95% interval", "from resampling whole months. Seeds averaged (climatology has none)."],
        )
    ).resolve_scale(y="independent")
    chart.save(str(DOCS / "headline_crps_by_day.svg"))


def ride() -> None:
    sub = pl.read_csv(ROOT.parent / "backtest_export_roster_2026-11-07" / "substations.csv")
    prims = d.prep.primaries().sort("primary").select("primary", "time_series_id").join(sub.select("time_series_id", "group"), on="time_series_id").select("primary", "group")
    rows = []
    for tag, label in (("seq", "Seq"), ("seq_s0", "Seq, seed 0"), ("seq_ls", "Seq trained on synthetic shifts"), ("P2-pre", "P2-pre"), ("K3-pre", "K3-pre")):
        files = sorted(glob.glob(str(ROOT / "inject" / f"{tag}_*.parquet")))
        if not files:
            continue
        df = pl.concat([pl.read_parquet(f) for f in files]).join(prims, on="primary").filter(pl.col("phase").is_in(["d4-end", "after0-7"]) & pl.col("group").is_in(list(GROUPS)))
        g = df.group_by("group", "phase").agg(pl.col("follow_q50").mean(), pl.col("safe").mean())
        rows += [{**r, "arm": label} for r in g.iter_rows(named=True)]
    t = pl.DataFrame(rows).with_columns(group=pl.col("group").replace(GROUPS), phase=pl.col("phase").replace({"d4-end": "Day 4 of the event to its end", "after0-7": "First 7 days after the event"}))
    order = ["K3-pre", "P2-pre", "Seq, seed 0", "Seq trained on synthetic shifts"]
    t = t.filter(pl.col("arm").is_in(order))
    cols = {"K3-pre": theme.DATA_SKY, "P2-pre": theme.DATA_BLUE, "Seq, seed 0": theme.BRAND_ORANGE, "Seq trained on synthetic shifts": theme.DATA_GREEN}
    panels = []
    for col, title in (("follow_q50", "Share of the shift followed by the median (smaller is better)"), ("safe", "Share of true readings on the safe side of the band (nominal 0.95; larger is better)")):
        panels.append(
            alt.Chart(t).mark_bar().encode(
                x=alt.X("arm:N", sort=order, title=None, axis=None),
                y=alt.Y(f"{col}:Q", title=None),
                color=alt.Color("arm:N", sort=order, scale=alt.Scale(domain=order, range=[cols[a] for a in order]), legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=400)),
                column=alt.Column("group:N", title=None), row=alt.Row("phase:N", title=None),
            ).properties(width=W / 2 - 80, height=120, title=title)
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Training Seq on synthetic load-shifts cuts how far it follows injected shifts",
            subtitle=["Experiment 581's injected bank (shifts of 15%, 30%, and 50% of capacity lasting 3, 7, or 15 days, added to the", "history power of every Primary at once), forecasts 24 to 120 h ahead, five fold months. K3-pre reads no power, so its bars are zero."],
        )
    ).save(str(DOCS / "ride_through.svg"))


def examples() -> None:
    import torch  # ty: ignore[unresolved-import]

    torch.cuda.set_per_process_memory_fraction(8 / 48)
    clean = d.clean_power()
    b1g, b1c = d.dense_b1()
    months = ("2026-01", "2026-06")
    ctx = {"raw": d.dense_raw(), "clean": clean, "b1_grid": b1g, "b1_cams": b1c, "tau": d.tau_grid(), "of": d.neighbour_lists(),
           "cap": fold_mod.capacity_of_primary(), "s": {mo: fold_mod.fold_stats(month=mo, clean=clean) for mo in months}}  # fmt: skip
    ids = d.prep.primaries().sort("primary")["time_series_id"].to_numpy()
    sub = pl.read_csv(ROOT.parent / "backtest_export_roster_2026-11-07" / "substations.csv")
    mva = [i for i, t in enumerate(ids) if sub.filter(pl.col("time_series_id") == t)["group"][0] == "eight MVA Primaries"]
    p = mva[0]
    cap = ctx["cap"][p]

    def one(frame: pl.DataFrame, origin_idx: int) -> pl.DataFrame:
        o = d.prep422.GRID_START_US + origin_idx * d.prep422.STEP_US
        return frame.filter((pl.col("time_series_id") == ids[p]) & (pl.col("power_fcst_init_time").dt.epoch("us") == o)).sort("valid_time")

    def obs(frame: pl.DataFrame, origin_idx: int) -> np.ndarray:
        idx = origin_idx + 1 + np.arange(len(frame))
        return clean[p, idx] * cap

    def lines(fr: pl.DataFrame, origin_idx: int, hours: float, title: str, twin: pl.DataFrame | None = None) -> alt.Chart:
        n = int(hours * 2)
        h = (np.arange(1, len(fr) + 1) * 0.5)[:n]
        t = pl.DataFrame({"hour": h, "q05": fr["q05"].to_numpy()[:n], "q50": fr["q50"].to_numpy()[:n], "q95": fr["q95"].to_numpy()[:n], "obs": obs(fr, origin_idx)[:n]})
        band = alt.Chart(t).mark_area(opacity=0.35, color=theme.BRAND_ORANGE).encode(x=alt.X("hour:Q", title="Hours after the forecast starts"), y=alt.Y("q05:Q", title="Power (MW or MVA)", scale=alt.Scale(zero=False)), y2="q95:Q")
        med = alt.Chart(t).mark_line(color=theme.BRAND_ORANGE).encode(x="hour:Q", y="q50:Q")
        ob = alt.Chart(t).mark_line(color=theme.BLACK_1, strokeWidth=1.2).encode(x="hour:Q", y="obs:Q")
        layers = [band, med, ob]
        if twin is not None:
            tw = pl.DataFrame({"hour": h, "q50": twin["q50"].to_numpy()[:n]})
            layers.append(alt.Chart(tw).mark_line(color=theme.DATA_BLUE, strokeDash=[5, 3]).encode(x="hour:Q", y="q50:Q"))
        return alt.layer(*layers).properties(width=W - 90, height=230, title=title)

    notes = ["Orange line: median forecast; orange band: 5% to 95% band. Black: observed power."]
    for hours, name in ((168, "example_week_ahead.svg"), (24, "example_day_ahead.svg")):
        panels = []
        for month, season in (("2026-01", "Winter"), ("2026-06", "Summer")):
            origin = fold_mod.test_origins(month=month)[11]
            fr = inj.seq_forecast(ctx=ctx, tag="seq", month=month, seeds=(0, 1, 2), clean_inj=None)
            panels.append(lines(one(fr, origin), origin, hours, f"{season}: forecast made at 09:00 UTC, Seq"))
        alt.vconcat(*panels).properties(title=alt.TitleParams(f"Seq forecasts {'a week' if hours == 168 else 'a day'} ahead for one MVA Primary", subtitle=[*notes, "Seeds averaged. Winter is January 2026 and summer is June 2026."])).save(str(DOCS / name))
    # Seq against an injected shift: event of 30% of capacity lasting 15 days, origin on day 12
    month = "2026-01"
    start = fold_mod.idx_of(t=fold_mod.month_start(month=month))
    r = (6 - p) % 12
    shifted = bank.contaminate(clean=clean, month_start_idx=start, r=r)[0]
    assert bank.event(primary=p, r=r)[:2] == (0.3, 15)
    origin = fold_mod.test_origins(month=month)[11]
    panels = []
    twin = one(inj.seq_forecast(ctx=ctx, tag="seq", month=month, seeds=(0,), clean_inj=None), origin)
    for tag, title in (("seq", "Seq"), ("seq_ls", "Seq trained on synthetic shifts")):
        fr = one(inj.seq_forecast(ctx=ctx, tag=tag, month=month, seeds=(0,), clean_inj=shifted), origin)
        tw = twin if tag == "seq" else one(inj.seq_forecast(ctx=ctx, tag=tag, month=month, seeds=(0,), clean_inj=None), origin)
        panels.append(lines(fr, origin, 120, f"{title}: history shifted by +30% of capacity", twin=tw))
    alt.vconcat(*panels).properties(title=alt.TitleParams("Seq against an injected load-shift", subtitle=["Black: true (unshifted) power. Orange: forecast made from the shifted history, with its 5% to 95% band.", "Blue dashed: median forecast of the same weights from the unshifted history. Origin: day 12, shift lasting 15 days."])).save(str(DOCS / "example_injected_shift.svg"))


if __name__ == "__main__":
    {"headline": headline, "ride": ride, "examples": examples}[sys.argv[1]]()

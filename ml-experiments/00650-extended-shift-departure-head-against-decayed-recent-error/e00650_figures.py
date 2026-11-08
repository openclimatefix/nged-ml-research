# ruff: noqa: E501, ANN001, D103, PLC0206
"""Experiment 650, step 7: the figures (CPU). SVG and PNG go to `figures/` in the data folder.

1. Scaled CRPS inside out-of-support held-out events, with month-block 95% intervals.
2. After the shift ends and on event-free rows (the two vetoes).
3. Inside-phase scaled CRPS by event depth.
4. Three examples of held-out events (two panels each: a six MW Primary and an eight MVA Primary).
5. Day-ahead forecasts across a month and one 10-day forecast from one origin on real rows.

Usage: python e00650_figures.py [score|score5]
"""

import sys

import altair as alt
import e00650_common as c650
import e00650_fit as f650
import numpy as np
import plotting.ocf_theme as ocf  # ty: ignore[unresolved-import]
import polars as pl

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "score5"
FITS = "fits" if SCOPE == "score" else "fits5"
SC = c650.WORK / SCOPE
OUT = c650.WORK / "figures"
OUT.mkdir(exist_ok=True)
W = 600
RNG = np.random.default_rng(650)
LABEL = {"K1": "K1", "R0-168": "R0 (168 h)", "Rold": "R-old", "Rnew": "R-new", "Placebo": "Placebo"}
COLOUR = {
    "K1": ocf.DATA_BLUE,
    "R0 (168 h)": ocf.DATA_SKY,
    "R-old": ocf.DATA_PURPLE,
    "R-new": ocf.BRAND_ORANGE,
    "Placebo": ocf.BRAND_ORANGE_LIGHT,
}


def save(chart, name: str) -> None:
    chart.save(OUT / f"{name}.svg")
    chart.save(OUT / f"{name}.png", scale_factor=2.0)
    print("saved", name, flush=True)


def levels(*, d: pl.DataFrame, col: str) -> pl.DataFrame:
    mm = (
        d.group_by("month", "arm", "seed")
        .agg(pl.col(col).mean())
        .group_by("month", "arm")
        .agg(pl.col(col).mean())
    )
    out = []
    for arm in LABEL:
        v = mm.filter(pl.col("arm") == arm)[col].to_numpy()
        if len(v) == 0:
            continue
        boot = np.array([RNG.choice(v, size=len(v)).mean() for _ in range(5000)])
        out.append(
            {
                "arm": LABEL[arm],
                "point": float(v.mean()),
                "lo": float(np.percentile(boot, 2.5)),
                "hi": float(np.percentile(boot, 97.5)),
            }
        )
    return pl.DataFrame(out)


def bars(
    panels: list[tuple[pl.DataFrame, str]], *, name: str, title: list[str], subtitle: list[str]
) -> None:
    order = list(COLOUR)
    charts = []
    for d, ptitle in panels:
        base = alt.Chart(d).encode(
            y=alt.Y("arm:N", sort=order, title=None),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=order, range=[COLOUR[k] for k in order]),
                legend=None,
            ),
        )
        mark = base.mark_bar(size=14, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "point:Q",
                title="Scaled CRPS, points of capacity (smaller is better)",
                scale=alt.Scale(zero=True),
            )
        )  # ty: ignore[unresolved-attribute]
        err = base.mark_rule(color=ocf.BLACK_1, strokeWidth=2, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x="lo:Q", x2="hi:Q"
        )  # ty: ignore[unresolved-attribute]
        charts.append(
            (mark + err).properties(
                width=W - 150,
                height=170,
                title=alt.TitleParams(ptitle, anchor="start", fontSize=12),
            )
        )
    save(
        alt.vconcat(*charts).properties(
            title=alt.TitleParams(title, subtitle=subtitle, anchor="start")
        ),
        name,
    )


def main() -> None:
    pr = pl.read_parquet(SC / "per_row_summary.parquet")
    held = pr.filter(pl.col("set") == "held").with_columns(
        oos=pl.col("cell").str.split("|").list.first() != ""
    )
    n_m = pr["month"].n_unique()
    sub = [
        f"Bar: mean over the {n_m} scored development months; black line: 95% interval from resampling the months.",
        "Every arm's band is scaled about its median to cover 0.90 of the rows before scoring (equal width scaling).",
    ]
    oos_in = held.filter((pl.col("phase") == "inside") & pl.col("oos"))
    all_in = held.filter(pl.col("phase") == "inside")
    bars(
        [
            (
                levels(d=oos_in, col="crps_scaled"),
                "Inside a held-out shift outside the training range",
            ),
            (levels(d=all_in, col="crps_scaled"), "Inside any held-out shift"),
        ],
        name="fig1_inside_scaled",
        title=[
            "Inside a held-out shift R-new scores below the no-fit shift R0 (168 h)",
            "after the band-widening share is removed.",
        ],
        subtitle=sub,
    )
    bars(
        [
            (
                levels(d=held.filter(pl.col("phase") == "after"), col="crps_scaled"),
                "After a held-out shift has ended (veto: at most 0.5 above K1)",
            ),
            (
                levels(d=pr.filter(pl.col("set") == "real"), col="crps_scaled"),
                "Event-free real rows (veto: at most 0.03 above K1)",
            ),
        ],
        name="fig2_vetoes",
        title=["After a shift ends and on event-free rows R-new scores worse than K1."],
        subtitle=sub,
    )
    # by depth
    d = (
        oos_in.group_by("depth", "arm", "month", "seed")
        .agg(pl.col("crps_scaled").mean())
        .group_by("depth", "arm")
        .agg(pl.col("crps_scaled").mean())
        .with_columns(label=pl.col("arm").replace(LABEL))
    )
    order = list(COLOUR)
    ch = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("depth:O", title="Shift depth, share of capacity"),
            y=alt.Y("crps_scaled:Q", title="Scaled CRPS, points of capacity"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=order, range=[COLOUR[k] for k in order]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=W - 100,
            height=240,
            title=alt.TitleParams(
                "Scaled CRPS inside a held-out shift by depth (events outside the training range)",
                anchor="start",
            ),
        )
    )
    save(ch, "fig3_by_depth")
    examples()
    real_examples()


def series_frame(
    *, z: dict, fit, name: str, month: str, prim: int, pass_: int, ev: dict
) -> pl.DataFrame:
    idx = fit[f"{name}_idx"]
    q = fit[f"{name}_q"]
    sel = (
        (z["prim"][idx] == prim)
        & (z["pass_"][idx] == pass_)
        & (z["lead"][idx] >= 24)
        & (z["lead"][idx] < 48)
    )
    vi = z["vi"][idx][sel]
    o = np.argsort(vi)
    k1 = z["q"][idx][sel][o]
    tgt = (z["y"][idx] + z["shift"][idx])[sel][o]
    res = z["res"][idx][sel][o]
    lead = z["lead"][idx][sel][o]
    r0 = np.sort(k1 + (res * np.exp(-lead / 168.0))[:, None], axis=1)
    return pl.DataFrame({"vi": vi[o], "target": tgt, "clean": z["y"][idx][sel][o], "k1_lo": k1[:, 1], "k1_med": k1[:, 5], "k1_hi": k1[:, 9], "r0_med": r0[:, 5], "rn_lo": q[sel][o][:, 1], "rn_med": q[sel][o][:, 5], "rn_hi": q[sel][o][:, 9]})  # fmt: skip


def examples() -> None:
    month = "2025-12"
    z = f650.sources(month=month, bank="new")["bank"]
    fit = np.load(c650.WORK / FITS / f"Rnew_{month}_s0.npz")
    events = pl.concat(
        [
            pl.read_parquet(c650.ROWS / f"events_{month}_r{r:02d}.parquet")
            for r in c650.HELD_OUT_PASSES
        ]
    )
    sm = c650.world()[2]
    for fig_i, (depth, days) in enumerate(((0.40, 21), (0.20, 45), (0.65, 10)), start=1):
        panels = []
        for grp_name, want_mw in (("six MW Primary", True), ("eight MVA Primary", False)):
            cand = events.filter(
                (pl.col("depth") == depth) & (pl.col("length_days").round(0) == days)
            ).filter(
                pl.col("primary").map_elements(
                    lambda p, want=want_mw: bool(sm[p]) == want, return_dtype=pl.Boolean
                )
            )
            if cand.height == 0:
                continue
            e = cand.row(0, named=True)
            df = series_frame(
                z=z, fit=fit, name="held", month=month, prim=e["primary"], pass_=e["pass"], ev=e
            )
            if df.height == 0:
                continue
            t0 = c650.month_start_idx(month=month)
            df = df.with_columns(day=(pl.col("vi") - t0) / 48.0)
            lo, hi = float(e["onset_idx"] - t0) / 48 - 7, float(e["end_idx"] - t0) / 48 + 14
            df = df.filter((pl.col("day") >= lo) & (pl.col("day") <= hi))
            long = pl.concat(
                [
                    df.select("day", pl.lit(n).alias("series"), pl.col(c).alias("v"))
                    for n, c in (
                        ("Observed plus shift (target)", "target"),
                        ("K1 median", "k1_med"),
                        ("R0 (168 h) median", "r0_med"),
                        ("R-new median", "rn_med"),
                    )
                ]
            )
            band = (
                alt.Chart(df)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.25, color=ocf.BRAND_ORANGE)
                .encode(x="day:Q", y="rn_lo:Q", y2="rn_hi:Q")
            )  # ty: ignore[unresolved-attribute]
            lines = alt.Chart(long).mark_line().encode(x=alt.X("day:Q", title="Day of the month (UTC)"), y=alt.Y("v:Q", title="Power, share of capacity", scale=alt.Scale(zero=False)), color=alt.Color("series:N", scale=alt.Scale(domain=["Observed plus shift (target)", "K1 median", "R0 (168 h) median", "R-new median"], range=[ocf.BLACK_1, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE]), legend=alt.Legend(orient="top", title=None, columns=2)))  # fmt: skip  # ty: ignore[unresolved-attribute]
            panels.append(
                (band + lines).properties(
                    width=W - 100,
                    height=200,
                    title=alt.TitleParams(
                        f"{grp_name}: depth {depth:.2f}, {days} days, {'up' if e['sign'] > 0 else 'down'}",
                        anchor="start",
                        fontSize=12,
                    ),
                )
            )
        if panels:
            save(
                alt.vconcat(*panels).properties(
                    title=alt.TitleParams(
                        [
                            "Day-ahead forecasts through a held-out shift (December 2025);",
                            "orange band is R-new's 90% interval",
                        ],
                        anchor="start",
                    )
                ),
                f"fig4_example_{fig_i}",
            )


def real_examples() -> None:
    """Figure 5: day-ahead forecasts over two weeks, and one 10-day forecast from one origin, on real rows."""
    month, prim = "2025-12", 5
    z = f650.sources(month=month, bank="new")["real"]
    fit = np.load(c650.WORK / FITS / f"Rnew_{month}_s0.npz")
    idx = fit["real_idx"]
    q = fit["real_q"]
    ms = c650.month_start_idx(month=month)
    ok = (z["prim"][idx] == prim) & np.isfinite(z["y"][idx])
    sel_day = (
        ok
        & (z["lead"][idx] >= 24)
        & (z["lead"][idx] < 48)
        & (z["vi"][idx] >= ms + 14 * 48)
        & (z["vi"][idx] < ms + 28 * 48)
    )
    o = np.argsort(z["vi"][idx][sel_day])
    d1 = pl.DataFrame(
        {
            "day": (z["vi"][idx][sel_day][o] - ms) / 48.0,
            "y": z["y"][idx][sel_day][o],
            "k1": z["q"][idx][sel_day][o][:, 5],
            "lo": q[sel_day][o][:, 1],
            "med": q[sel_day][o][:, 5],
            "hi": q[sel_day][o][:, 9],
        }
    )
    origin = np.unique(z["oi"][idx][ok & (z["oi"][idx] >= ms + 14 * 48)])[0]
    sel_h = ok & (z["oi"][idx] == origin)
    o = np.argsort(z["vi"][idx][sel_h])
    d2 = pl.DataFrame(
        {
            "hours": z["lead"][idx][sel_h][o],
            "y": z["y"][idx][sel_h][o],
            "k1": z["q"][idx][sel_h][o][:, 5],
            "lo": q[sel_h][o][:, 1],
            "med": q[sel_h][o][:, 5],
            "hi": q[sel_h][o][:, 9],
        }
    )
    panels = []
    for d, x, xt, title in (
        (
            d1,
            "day",
            "Day of December 2025 (UTC)",
            "Day-ahead forecasts (lead 24 to 48 h), a six MW Primary, 14 to 28 December",
        ),
        (d2, "hours", "Lead time, hours", "One forecast from one origin, 24 to 240 h ahead"),
    ):
        long = pl.concat(
            [
                d.select(x, pl.lit(n).alias("series"), pl.col(c).alias("v"))
                for n, c in (("Observed", "y"), ("K1 median", "k1"), ("R-new median", "med"))
            ]
        )
        band = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.25, color=ocf.BRAND_ORANGE)
            .encode(x=f"{x}:Q", y="lo:Q", y2="hi:Q")
        )  # ty: ignore[unresolved-attribute]
        lines = alt.Chart(long).mark_line().encode(x=alt.X(f"{x}:Q", title=xt), y=alt.Y("v:Q", title="Power, share of capacity", scale=alt.Scale(zero=False)), color=alt.Color("series:N", scale=alt.Scale(domain=["Observed", "K1 median", "R-new median"], range=[ocf.BLACK_1, ocf.DATA_BLUE, ocf.BRAND_ORANGE]), legend=alt.Legend(orient="top", title=None)))  # fmt: skip  # ty: ignore[unresolved-attribute]
        panels.append(
            (band + lines).properties(
                width=W - 100, height=200, title=alt.TitleParams(title, anchor="start", fontSize=12)
            )
        )
    save(
        alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                "Example forecasts on real event-free rows; orange band is R-new's 90% interval",
                anchor="start",
            )
        ),
        "fig5_real_examples",
    )


if __name__ == "__main__":
    main()

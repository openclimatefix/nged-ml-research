# ruff: noqa: E731
# ruff: noqa: D103, E501
"""Experiment 647, stage 4: the four figures (CPU).

1. P2 and P4 CRPS proxy of K1, R0, R0*, H1, H1-p, and E2 with 95% intervals (five months, five seeds).
2. Time series of K1 and H1 against the shifted target: injected-bank passes 2, 5, and 9 of 2025-12
   (one figure per pass, one panel per metering group), then the verified D5 and D16 pair.
3. The 10-day horizon from one origin of an injected pass.
4. Detector recall and days to first alarm by depth and length: H1, E2, and experiment 637.

Writes SVG to `figures/` in the data folder (optimised afterwards with svgo).

Usage: python e00647_figures.py
"""

import altair as alt
import e00647_common as c647  # ty: ignore[unresolved-import]
import e00647_data as dat  # ty: ignore[unresolved-import]
import e00647_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf  # ty: ignore[unresolved-import]
import polars as pl
from e00581_bank import event  # ty: ignore[unresolved-import]

SC = c647.WORK / "score5"
OUT = c647.WORK / "figures"
OUT.mkdir(exist_ok=True)
W = 600
ARM_LABEL = {
    "K1": "K1",
    "R0_72": "R0",
    "R0s_72": "R0*",
    "H1": "H1",
    "H1p": "H1-p (noise placebo)",
    "E2": "E2 (own encoder)",
}
ARM_COLOUR = {
    "K1": ocf.DATA_BLUE,
    "R0": ocf.DATA_SKY,
    "R0*": ocf.DATA_PURPLE,
    "H1": ocf.BRAND_ORANGE,
    "H1-p (noise placebo)": ocf.BRAND_ORANGE_LIGHT,
    "E2 (own encoder)": ocf.DATA_GREEN,
}
PER = 48


def save(chart: alt.TopLevelMixin, name: str) -> None:
    chart.save(OUT / f"{name}.svg")
    chart.save(OUT / f"{name}.png", scale_factor=2.0)
    print("saved", name, flush=True)


def fig1() -> None:
    lv = pl.read_csv(SC / "stage4_levels.csv").filter(
        (pl.col("cut") == "all") & pl.col("arm").is_in(list(ARM_LABEL))
    )
    lv = lv.with_columns(pl.col("arm").replace(ARM_LABEL).alias("label"))
    order = list(ARM_COLOUR)
    panels = []
    for sname, title in (
        (
            "P2 inside shift",
            "P2: rows inside an injected shift (scored against the shifted target)",
        ),
        (
            "P4 flagged or verified",
            "P4: days flagged by experiment 637, and the verified D5 and D16 pair",
        ),
    ):
        d = lv.filter(pl.col("set") == sname)
        base = alt.Chart(d.to_pandas() if False else d).encode(
            y=alt.Y("label:N", sort=order, title=None),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=order, range=[ARM_COLOUR[k] for k in order]),
                legend=None,
            ),
        )
        bars = base.mark_bar(size=16, aria=False).encode(
            x=alt.X(
                "point:Q",
                title="CRPS proxy, points of capacity (smaller is better)",
                scale=alt.Scale(zero=True),
            )
        )
        err = base.mark_rule(color=ocf.BLACK_1, strokeWidth=1.5, aria=False).encode(
            x="lo:Q", x2="hi:Q"
        )
        panels.append(
            (bars + err).properties(
                width=W - 150, height=150, title=alt.TitleParams(title, anchor="start", fontSize=12)
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "H1 and E2 beat K1 and both baselines where a shift is in the lags;",
                "the noise placebo H1-p does not.",
            ],
            subtitle=[
                "Bar: estimate pooled over the five development months (2025-10 to 2026-06), leave-one-month-out.",
                "Black line: 95% interval from resampling the five months and the five seeds.",
            ],
            anchor="start",
        )
    )
    save(chart, "fig1_p2_p4_bars")


def prevailing(kind: str, month: str, arm: str = "H1") -> dict[str, np.ndarray]:
    """Seed-mean q05, q50, q95 of the arm for the month's usable rows."""
    files = sorted((c647.WORK / "fits").glob(f"{arm}_{month}_lomo_s?.npz"))
    zs = [np.load(f) for f in files]
    return {k: np.mean([z[f"{kind}_{k}"] for z in zs], axis=0) for k in ("q05", "q50", "q95")} | {
        "idx": zs[0][f"{kind}_idx"]
    }


def bank_frame(month: str, r: int, prim: int) -> pl.DataFrame:
    d = fit.load(month=month)["bank"]
    pv = prevailing("bank", month)
    idx = pv["idx"]
    sel = (d["pass_"][idx] == r) & (d["prim"][idx] == prim)
    ii = idx[sel]
    ms = dat.month_start_idx(month=month)
    return pl.DataFrame(
        {
            "oi": d["oi"][ii],
            "vi": d["vi"][ii],
            "lead": d["lead"][ii],
            "k1_lo": d["q"][ii, 1],
            "k1_med": d["q"][ii, 5],
            "k1_hi": d["q"][ii, 9],
            "h1_lo": pv["q05"][sel],
            "h1_med": pv["q50"][sel],
            "h1_hi": pv["q95"][sel],
            "target": d["y"][ii] + d["shift"][ii],
            "shift": d["shift"][ii],
            "day": (d["vi"][ii] - ms) / PER,
        }
    )


def ts_panel(
    df: pl.DataFrame, *, title: str, xtitle: str = "Day of the month (valid time)", x: str = "day"
) -> alt.Chart:
    k1 = df.select(x, "k1_lo", "k1_hi", "k1_med").with_columns(arm=pl.lit("K1"))
    h1 = df.select(
        x,
        pl.col("h1_lo").alias("k1_lo"),
        pl.col("h1_hi").alias("k1_hi"),
        pl.col("h1_med").alias("k1_med"),
    ).with_columns(arm=pl.lit("H1 (prevailing)"))
    lines = pl.concat([k1, h1]).sort(x)
    scale = alt.Scale(domain=["K1", "H1 (prevailing)"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE])
    base = alt.Chart(lines).encode(
        x=alt.X(f"{x}:Q", title=xtitle),
        color=alt.Color("arm:N", scale=scale, legend=alt.Legend(title=None, orient="top")),
    )
    band = base.mark_area(opacity=0.25, aria=False).encode(
        y=alt.Y("k1_lo:Q", title="Power, share of capacity"), y2="k1_hi:Q"
    )
    med = base.mark_line(strokeWidth=1.3, aria=False).encode(y="k1_med:Q")
    tgt = (
        alt.Chart(df.sort(x).select(x, "target"))
        .mark_line(color=ocf.BLACK_1, strokeWidth=1.3, aria=False)
        .encode(x=f"{x}:Q", y="target:Q")
    )
    return (band + med + tgt).properties(
        width=W, height=190, title=alt.TitleParams(title, anchor="start", fontSize=12)
    )


def group_prims() -> dict[str, int]:
    import e00578_common as k578  # ty: ignore[unresolved-import]
    from e00647_score import GROUPS

    sm = k578.sites()
    groups = [GROUPS[g] for g in sm["group"].to_list()]
    return {
        "six MW Primary": groups.index("six MW"),
        "eight MVA Primary": groups.index("eight MVA"),
    }


def fig2_3() -> None:
    month = "2025-12"
    d = fit.load(month=month)["bank"]
    prims = group_prims()
    for r in (2, 5, 9):
        panels = []
        for gname, p in prims.items():
            if not ((d["pass_"] == r) & (d["prim"] == p)).any():
                continue
            df = bank_frame(month, r, p).filter((pl.col("lead") >= 24) & (pl.col("lead") < 48))
            depth, length, _ = event(primary=p, r=r)
            panels.append(
                ts_panel(
                    df,
                    title=f"{gname} {p}, pass {r}: shift {depth:+.2f} of capacity for {length} days, ending day 14",
                )
            )
        chart = alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                [
                    "K1 stays near its normal-arrangement forecast while the lags carry the shift;",
                    "H1 moves towards the shifted target.",
                ],
                subtitle=[
                    "Black: the shifted target (observed power plus the injected shift). Bands: q05 to q95. Lines: medians.",
                    f"Forecasts with lead 24 to 48 h, {month}, leave-one-month-out, mean over five seeds.",
                ],
                anchor="start",
            )
        )
        save(chart, f"fig2_pass{r}")
    # verified pair
    zr = fit.load(month=month)["real"]
    pv = prevailing("real", month)
    idx = pv["idx"]
    ms = dat.month_start_idx(month=month)
    panels = []
    for name, p in (("D5", 3), ("D16", 11)):
        sel = (
            (zr["prim"][idx] == p)
            & (zr["lead"][idx] >= 24)
            & (zr["lead"][idx] < 48)
            & zr["verified"][idx]
        )
        # include the weeks around the pair for context
        ctx = (
            (zr["prim"][idx] == p)
            & (zr["lead"][idx] >= 24)
            & (zr["lead"][idx] < 48)
            & (zr["vi"][idx] >= ms + 20 * PER)
            & (zr["vi"][idx] < ms + 31 * PER + 12 * PER)
        )
        ii = idx[ctx]
        df = pl.DataFrame(
            {
                "day": (zr["vi"][ii] - ms) / PER,
                "k1_lo": zr["q"][ii, 1],
                "k1_med": zr["q"][ii, 5],
                "k1_hi": zr["q"][ii, 9],
                "h1_lo": pv["q05"][ctx],
                "h1_med": pv["q50"][ctx],
                "h1_hi": pv["q95"][ctx],
                "target": zr["y"][ii],
                "lead": zr["lead"][ii],
            }
        )
        _ = sel
        panels.append(
            ts_panel(
                df,
                title=f"{name}: Primary {p}, observed power in black",
                xtitle="Day counted from 1 December 2025 (valid time)",
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            ["The verified D5 and D16 load shifts, December 2025 to January 2026."],
            subtitle=[
                "Black: observed power (it is not a shifted target here). Bands: q05 to q95; lead 24 to 48 h."
            ],
            anchor="start",
        )
    )
    save(chart, "fig2_verified_pair")
    # figure 3: one origin, 10-day horizon
    p = prims["six MW Primary"]
    r = 5
    df = bank_frame(month, r, p)
    origin = ms + 8 * PER + 18
    df = df.filter(pl.col("oi") == origin).with_columns(
        ((pl.col("vi") - origin) / PER).alias("days_ahead")
    )
    depth, length, _ = event(primary=p, r=r)
    end_ahead = (ms + 14 * PER - origin) / PER
    panel = ts_panel(
        df,
        title=f"Six MW Primary {p}, pass {r}, one forecast issued on day 8 at 09:00",
        xtitle="Days ahead of the forecast origin",
        x="days_ahead",
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [end_ahead], "t": ["injected shift ends"]}))
        .mark_rule(strokeDash=[4, 3], color=ocf.BLACK_1, aria=False)
        .encode(x="x:Q")
    )
    lab = (
        alt.Chart(pl.DataFrame({"x": [end_ahead], "t": ["injected shift ends"]}))
        .mark_text(align="left", dx=4, dy=-80, fontSize=10, color=ocf.BLACK_1, aria=False)
        .encode(x="x:Q", text="t:N")
    )
    chart = (panel + rule + lab).properties(
        title=alt.TitleParams(
            [
                "Over the 10-day horizon H1 follows the shift but keeps following it after the shift ends."
            ],
            subtitle=[
                f"Shift {depth:+.2f} of capacity for {length} days. Black: shifted target. Bands: q05 to q95. {month}, mean over five seeds."
            ],
            anchor="start",
        )
    )
    save(chart, "fig3_horizon")


def fig4() -> None:
    dt = pl.read_csv(SC / "detector.csv")
    rec = dt.filter(
        pl.col("what").str.starts_with("bank recall")
        & pl.col("variant").is_in(["lomo", "n/a"])
        & (pl.col("signal").is_in(["depth", "experiment 637"]))
        & (pl.col("arm").is_in(["H1", "E2a"]) | pl.col("arm").is_null())
    )
    rec = rec.with_columns(
        pl.col("what")
        .str.extract(r"depth ([0-9.]+) length (\d+)", 1)
        .cast(pl.Float64)
        .alias("depth"),
        pl.col("what")
        .str.extract(r"depth ([0-9.]+) length (\d+)", 2)
        .cast(pl.Int64)
        .alias("length"),
        pl.when(pl.col("signal") == "experiment 637")
        .then(pl.lit("Experiment 637 detector"))
        .otherwise(pl.col("arm"))
        .alias("system"),
    )
    rec = rec.group_by("system", "depth", "length").agg(
        (pl.col("value") * pl.col("n")).sum() / pl.col("n").sum(), pl.col("n").sum().alias("events")
    )
    days = dt.filter(
        pl.col("what").str.starts_with("bank days")
        & pl.col("variant").is_in(["lomo", "n/a"])
        & pl.col("signal").is_in(["depth", "experiment 637"])
        & (pl.col("arm").is_in(["H1", "E2a"]) | pl.col("arm").is_null())
    )
    days = days.with_columns(
        pl.col("what")
        .str.extract(r"depth ([0-9.]+) length (\d+)", 1)
        .cast(pl.Float64)
        .alias("depth"),
        pl.col("what")
        .str.extract(r"depth ([0-9.]+) length (\d+)", 2)
        .cast(pl.Int64)
        .alias("length"),
        pl.when(pl.col("signal") == "experiment 637")
        .then(pl.lit("Experiment 637 detector"))
        .otherwise(pl.col("arm"))
        .alias("system"),
    )
    days = (
        days.filter(pl.col("value").is_finite())
        .group_by("system", "depth", "length")
        .agg((pl.col("value") * pl.col("n")).sum() / pl.col("n").sum())
    )
    names = {
        "H1": "H1 (|d| >= 0.05)",
        "E2a": "E2 (|d| >= 0.05)",
        "Experiment 637 detector": "Experiment 637 detector",
    }
    cat = lambda d: d.with_columns(
        pl.col("system").replace(names),
        (pl.col("depth").cast(pl.String) + " | " + pl.col("length").cast(pl.String) + " d").alias(
            "cell"
        ),
    )
    order = [f"{dp} | {ln} d" for dp in (0.15, 0.3, 0.5) for ln in (3, 7, 15, 30)]
    dom = list(names.values())
    colour = alt.Color(
        "system:N",
        scale=alt.Scale(domain=dom, range=[ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    xa = alt.X(
        "cell:N",
        sort=order,
        title="Injected depth (share of capacity) | length",
        axis=alt.Axis(labelAngle=-60),
    )
    p1 = (
        alt.Chart(cat(rec))
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            x=xa,
            y=alt.Y(
                "value:Q", title="Share of injected events detected", scale=alt.Scale(domain=[0, 1])
            ),
            color=colour,
        )
        .properties(width=W, height=170)
    )
    p2 = (
        alt.Chart(cat(days))
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            x=xa,
            y=alt.Y("value:Q", title="Days from onset to first alarm (smaller is better)"),
            color=colour,
        )
        .properties(width=W, height=170)
    )
    chart = alt.vconcat(p1, p2).properties(
        title=alt.TitleParams(
            [
                "Alarm rate and delay: H1, E2, and the experiment 637 detector,",
                "by depth and length of the injected shift.",
            ],
            subtitle=[
                "Alarm: |d| >= 0.05 of capacity at leads 24 to 26 h. Detected events only in the lower panel.",
                "Injected bank, five development months, leave-one-month-out, mean over five seeds.",
            ],
            anchor="start",
        )
    )
    save(chart, "fig4_detector")


if __name__ == "__main__":
    fig1()
    fig2_3()
    fig4()

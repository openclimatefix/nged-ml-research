# ruff: noqa: D103, E501, PLR0915
"""Experiment 663: figures (CPU). Writes SVG files to `figures/` in the experiment's data folder."""

import json
from datetime import timedelta

import altair as alt
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00663_common as c663
import numpy as np
import plotting.ocf_theme as ocf  # ty: ignore[unresolved-import]
import polars as pl

OUT = c663.WORK / "figures"
OUT.mkdir(exist_ok=True)
TABLES = json.loads((c663.WORK / "score" / "tables.json").read_text())
WINDOW = json.loads((c663.WORK / "final" / "score" / "window.json").read_text())
RC = c663.WORK / "release_candidate"
W, H = 680, 280


def save(chart: alt.Chart | alt.VConcatChart, name: str) -> None:
    chart.save(OUT / f"{name}.svg")
    print("wrote", name, flush=True)


def fig_window() -> None:
    arms = {"K1": ocf.BLACK_1, "Manual heuristic": ocf.BRAND_ORANGE_LIGHT, "J2": ocf.DATA_PURPLE, "J2d": ocf.DATA_SKY, "E0": ocf.DATA_BLUE, "E0d": ocf.BRAND_ORANGE}
    rows = [
        {"scenario": sc, "arm": a, "crps": WINDOW[sc][a]["crps"][0], "lo": WINDOW[sc][a]["crps"][1], "hi": WINDOW[sc][a]["crps"][2]}
        for sc in ("all", "outside", "inside") for a in arms if a in WINDOW[sc]
    ]  # fmt: skip
    nice = {"all": "All rows", "outside": "No shift in force", "inside": "Shift in force"}
    df = pl.DataFrame(rows).with_columns(scenario=pl.col("scenario").replace(nice))
    base = alt.Chart(df).encode(
        x=alt.X("scenario:N", title=None, sort=list(nice.values()), axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("arm:N", sort=list(arms)),
        color=alt.Color("arm:N", scale=alt.Scale(domain=list(arms), range=list(arms.values())), legend=alt.Legend(title=None, orient="top", columns=6)),
    )  # fmt: skip
    pts = base.mark_point(filled=True, size=70).encode(y=alt.Y("crps:Q", title="Mean CRPS (points of capacity; smaller is better)", scale=alt.Scale(zero=True)))
    bars = base.mark_rule().encode(y="lo:Q", y2="hi:Q")
    save((bars + pts).properties(width=W, height=H + 40, title=["Final-test window, 1 July to 31 October 2026, lead 24 to 240 hours.", "Dot: estimate. Line: 95% interval from resampling the four origin months."]), "window_crps")


def contrast(key: str) -> dict:
    return TABLES["contrasts"][key]


def fig_dev() -> None:
    strata = [("ended24", "Shift ended in the last 24 h"), ("just_ended", "Shift recently ended"), ("near", "Shift in force, 24 h or less left"), ("far", "Shift in force, over 10 days left"), ("inforce", "Any shift in force"), ("free", "No shift in the injected week"), ("all", "All rows")]
    panels = []
    for fam, base_arm in (("E0d", "E0"), ("J2d", "J2")):
        rows = []
        for st, lab in strata:
            for kind, other in (("detector minus the same network without it", base_arm), ("real graph minus rewired graph", f"{fam}_rw0")):
                v = contrast(f"all|{st}|{fam} minus {other}|crps_s")
                rows.append({"stratum": lab, "kind": kind, "mean": v["mean"], "lo": v["lo"], "hi": v["hi"]})
        df = pl.DataFrame(rows)
        enc = {"y": alt.Y("stratum:N", title=None, sort=[s[1] for s in strata], axis=alt.Axis(labelLimit=260)), "yOffset": alt.YOffset("kind:N")}
        col = alt.Color("kind:N", scale=alt.Scale(domain=["detector minus the same network without it", "real graph minus rewired graph"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]), legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=600))
        ch = alt.Chart(df).encode(**enc, color=col)
        zero = alt.Chart(pl.DataFrame({"z": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(x="z:Q")
        pts = ch.mark_point(filled=True, size=60).encode(x=alt.X("mean:Q", title="Change in scaled CRPS (points of capacity; more negative is better)"))
        bars = ch.mark_rule().encode(x="lo:Q", x2="hi:Q")
        panels.append((zero + bars + pts).properties(width=W - 230, height=H, title=f"{'E0 family' if fam == 'E0d' else 'J2 family'} ({fam} against {base_arm})"))
    save(alt.vconcat(*panels).resolve_scale(color="shared"), "dev_contrasts")


def fig_false() -> None:
    a = TABLES["absolute"]
    arms = ["E0", "E0d", "J2", "J2d"]
    rows = []
    for arm in arms:
        if f"real|real_free|{arm}|dep" in a:
            v = a[f"real|real_free|{arm}|dep"]
            rows.append({"arm": arm, "period": "Development months", "share": 100 * v["mean"], "lo": 100 * v["lo"], "hi": 100 * v["hi"]})
        key = {"E0": "E0", "E0d": "E0d", "J2": "J2", "J2d": "J2d"}.get(arm)
        if key and "departure_share" in WINDOW["outside"][key]:
            v = WINDOW["outside"][key]["departure_share"]
            rows.append({"arm": arm, "period": "Final-test window", "share": 100 * v[0], "lo": 100 * v[1], "hi": 100 * v[2]})
    df = pl.DataFrame(rows)
    ch = alt.Chart(df).encode(x=alt.X("arm:N", sort=arms, title=None, axis=alt.Axis(labelAngle=0)), xOffset="period:N", color=alt.Color("period:N", scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]), legend=alt.Legend(title=None, orient="top")))
    top = (ch.mark_rule().encode(y="lo:Q", y2="hi:Q") + ch.mark_point(filled=True, size=70).encode(y=alt.Y("share:Q", title="Share of rows (%)"))).properties(width=W, height=H, title=["Rows with no shift in force where the prevailing forecast departs from the normal-running-arrangement forecast", "by 3 points of capacity or more (the false-alarm share; dot: estimate, line: 95% interval)"])
    rows2 = []
    for arm in arms:
        if f"real|real_free|{arm}|dep_crps" in a:
            d1, d0 = a[f"real|real_free|{arm}|dep_crps"]["mean"], a[f"real|real_free|{arm}|dep_crps_nra"]["mean"]
            rows2.append({"arm": arm, "extra": d1 - d0})
    ch2 = alt.Chart(pl.DataFrame(rows2)).mark_bar(color=ocf.BRAND_ORANGE).encode(x=alt.X("arm:N", sort=arms, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("extra:Q", title="Extra CRPS per row (points of capacity)")).properties(width=W, height=H, title=["CRPS added by the departures on development-month rows with no shift in force,", "against the normal-running-arrangement forecast on the same rows (positive is worse)"])
    save(alt.vconcat(top, ch2), "false_alarms")


def fig_delay() -> None:
    """Mean `q_end` against days from the end of the injected shift, five development months, seeds pooled."""
    import e00650_common as c650  # ty: ignore[unresolved-import]
    import e00650_fit as f650  # ty: ignore[unresolved-import]
    import e00652_fit as f652  # ty: ignore[unresolved-import]

    arms = {"E0d": "Real graph", "E0d_rw0": "Rewired graph", "E0d_noneigh": "No neighbours"}
    acc: dict[str, dict[int, list[float]]] = {a: {} for a in arms}
    for m in c650.MONTHS:
        src = f650.sources(month=m, bank="new")
        f652.with_state(z=src["bank"], month=m)
        z = src["bank"]
        for arm in arms:
            for sd in (0, 1):
                f = np.load(c663.WORK / "fits_e0" / f"{arm}_{m}_s{sd}.npz")
                idx = f["fam_idx"]
                st = z["stratum"][idx]
                keep = np.isin(st, ["just_ended", "far", "middle", "near"])
                rel = np.floor((z["oi"][idx].astype(np.int64) - z["end"][idx].astype(np.int64)) / 48.0).astype(int)
                for d, v in zip(rel[keep], f["fam_q_end"][keep], strict=True):
                    if -7 <= d <= 7:
                        acc[arm].setdefault(int(d), []).append(float(v))
    rows = [{"day": d, "arm": lab, "q_end": float(np.mean(v))} for arm, lab in arms.items() for d, v in acc[arm].items()]
    df = pl.DataFrame(rows).sort("arm", "day")
    df.write_csv(OUT / "detector_delay.csv")
    ch = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("day:Q", title="Days from the end of the injected shift (negative: still in force)", axis=alt.Axis(values=list(range(-7, 8)))),
        y=alt.Y("q_end:Q", title="Mean detector output (probability the shift has ended)"),
        color=alt.Color("arm:N", scale=alt.Scale(domain=list(arms.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]), legend=alt.Legend(title=None, orient="top")),
    )  # fmt: skip
    rule = alt.Chart(pl.DataFrame({"t": [0.0]})).mark_rule(strokeDash=[4, 3], color=ocf.BLACK_1).encode(x="t:Q")
    save((rule + ch).properties(width=W, height=H, title=["How fast the detector notices an ending, on experiment 650's injected shifts in the five development months.", "Dashed line: the shift ends. Origins are daily, so a point is one day of origins."]), "detector_delay")


def longest_window_event() -> tuple[str, object, object]:
    """The Primary and dates of the longest run of consecutive origin days with the G4 gate on."""
    z = np.load(c663.WORK / "window" / "real_window.npz")
    best = (0, 0, 0, 0)
    for p in range(16):
        sel = np.flatnonzero((z["org_prim"] == p) & z["gate"])
        oi = np.sort(z["org_oi"][sel])
        if len(oi) == 0:
            continue
        start = 0
        for i in range(1, len(oi) + 1):
            if i == len(oi) or oi[i] - oi[i - 1] != 48:
                if i - start > best[0]:
                    best = (i - start, p, int(oi[start]), int(oi[i - 1]))
                start = i
    n, p, a, b = best
    label = k578.sites().filter(pl.col("primary") == p)["label"][0]
    return label, k578.GRID_START + timedelta(minutes=30 * a), k578.GRID_START + timedelta(minutes=30 * b)


def series(*, who: str, start, stop, arms: list[str]) -> pl.DataFrame:
    s = k578.sites()
    tid = s.filter(pl.col("label") == who)["time_series_id"][0]
    cap = 1.0
    out = None
    for arm in ["E0", "E0d", "J2d"]:
        d = pl.concat([pl.read_parquet(RC / f"{arm}.parquet"), pl.read_parquet(RC / f"{arm}_window.parquet")]) if arm else None
        d = d.filter((pl.col("time_series_id") == tid) & (pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 48) & (pl.col("valid_time") >= start) & (pl.col("valid_time") < stop)).sort("valid_time")
        cols = d.select("valid_time", pl.col("observed_power").alias("Observed"), pl.col("nra_q50").alias("NRA forecast (K1)"), pl.col("prevailing_q50").alias(f"{arm} prevailing"))
        out = cols if out is None else out.join(cols.select("valid_time", f"{arm} prevailing"), on="valid_time", how="left")
    assert out is not None
    return out.with_columns(pl.col("valid_time").dt.truncate("1h")).group_by("valid_time").mean().sort("valid_time") if cap else out


def fig_event(*, who: str, start, stop, name: str, arms: list[str], title: list[str], mark_start=None, mark_end=None) -> None:
    df = series(who=who, start=start, stop=stop, arms=arms)
    unit = k578.sites().filter(pl.col("label") == who)["group"][0]
    unit_s = "MVA" if "MVA" in unit else "MW"
    cols = {"Observed": ocf.BLACK_1, "NRA forecast (K1)": ocf.DATA_BLUE}
    panels = []
    for arm, colour in zip(arms, (ocf.BRAND_ORANGE, ocf.DATA_PURPLE), strict=False):
        long = df.select("valid_time", "Observed", "NRA forecast (K1)", pl.col(f"{arm} prevailing").alias("Prevailing forecast")).unpivot(index="valid_time", variable_name="series", value_name="power")
        domain = ["Observed", "NRA forecast (K1)", "Prevailing forecast"]
        rng = [cols["Observed"], cols["NRA forecast (K1)"], colour]
        ln = alt.Chart(long).mark_line(clip=True).encode(
            x=alt.X("valid_time:T", title="Date (UTC)", axis=alt.Axis(format="%d %b", tickCount=8, labelOverlap=True)),
            y=alt.Y("power:Q", title=f"Power ({unit_s})", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(title=None, orient="top")),
            strokeWidth=alt.StrokeWidth("series:N", scale=alt.Scale(domain=domain, range=[1.2, 1.2, 2.2]), legend=None),
        )  # fmt: skip
        layers = [ln]
        if mark_start is not None:
            layers.append(alt.Chart(pl.DataFrame({"t": [mark_start]})).mark_rule(strokeDash=[4, 3], color=ocf.BLACK_1).encode(x="t:T"))
        if mark_end is not None:
            layers.append(alt.Chart(pl.DataFrame({"t": [mark_end]})).mark_rule(strokeDash=[4, 3], color=ocf.BLACK_1).encode(x="t:T"))
        panels.append(alt.layer(*layers).properties(width=W, height=H, title=f"{arm}: day-ahead median (lead 24 to 48 h)"))
    save(alt.vconcat(*panels).properties(title=title), name)


def main() -> None:
    from datetime import datetime, timezone

    fig_window()
    fig_dev()
    fig_false()
    fig_delay()
    u = timezone.utc
    ev = (datetime(2025, 12, 29, 1, tzinfo=u), datetime(2026, 1, 13, 11, tzinfo=u))
    who, a, b = longest_window_event()
    print("longest window gate run", who, a, b, flush=True)
    fig_event(who=who, start=a - timedelta(days=14), stop=b + timedelta(days=8), name="event_window", arms=["E0", "E0d"],
              title=[f"Primary {who}: the longest run of days with the event gate on in the final-test window (from {a:%d %B %Y}, dashed line; the run lasts to the end of the window).", "Black: observed power. Blue: forecast with no shift. Colour: forecast that carries the shift forward."],
              mark_start=a)  # fmt: skip
    for who in ("D5", "D16"):
        fig_event(who=who, start=datetime(2025, 12, 15, tzinfo=u), stop=datetime(2026, 1, 21, tzinfo=u), name=f"event_{who}", arms=["E0", "E0d"],
                  title=[f"Primary {who}: a labelled load shift, 29 December 2025 to 13 January 2026 (dashed lines: start and end).", "Black: observed power. Blue: forecast with no shift. Colour: forecast that carries the shift forward."],
                  mark_start=ev[0], mark_end=ev[1])  # fmt: skip


if __name__ == "__main__":
    main()

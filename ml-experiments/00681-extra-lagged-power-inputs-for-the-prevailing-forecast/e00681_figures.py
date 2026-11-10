# ruff: noqa: D103, E501, PLR0915
"""Experiment 681, step 5: figures (CPU).

uv run --no-sync python ml-experiments/00681-*/e00681_figures.py OUT_DIR

Writes SVG figures: reaction time against the number of lags, the cumulative share of injected
events that have reacted, the December 2025 to January 2026 D5 and D16 transfer (day-ahead
stitched, NRA panel above the prevailing panel with L0 and L3), and 14-day forecasts from one
winter and one summer origin at D5. Substations may be named and dated; no generator appears.
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

sys.path.insert(0, str(next(Path(__file__).resolve().parents[1].glob("00667-*/prevailing"))))

import altair as alt
import e00578_common as k578  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
import p667_data as dat  # ty: ignore[unresolved-import]
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

DATA: Final = dat.ROOT.parent / "00681-extra-lagged-power-inputs-for-the-prevailing-forecast"
ARMS: Final = {"L0": 0, "L1": 8, "L2": 11, "L3": 14}
COL: Final = {"Observed": "#000000", "NRA": "#306BFF", "L0": "#FF4901", "L3": "#B701FF", "P": "#17E58F", "P7": "#10C5F7", "L1": "#FC9700", "L2": "#306BFF"}
DAY: Final = 48
START_IDX: Final = int((datetime(2025, 12, 29, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800)
END_IDX: Final = int((datetime(2026, 1, 14, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800)
H: Final = 290


def to_time(idx: np.ndarray) -> np.ndarray:
    base = np.datetime64(k578.GRID_START.replace(tzinfo=None), "us")
    return base + np.asarray(idx) * np.timedelta64(1800 * 10**6, "us")


def boot_ci(x: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def headline(*, out: Path) -> None:
    r = pl.read_parquet(DATA / "score" / "reaction.parquet").filter(pl.col("kind") == "injected")
    rows = []
    for arm, n in {**ARMS, "P": 14, "P7": 14}.items():
        for col, name in (("start_h", "after the start"), ("end_h", "after the end")):
            x = r.filter((pl.col("arm") == arm) & pl.col(col).is_not_nan())[col].to_numpy()
            m, lo, hi = boot_ci(x)
            rows.append({"arm": arm, "lags": n, "hours": m, "lo": lo, "hi": hi, "when": name})
    df = pl.DataFrame(rows).to_pandas()
    base = alt.Chart(df).encode(x=alt.X("lags:Q", title="Extra lagged inputs (0 = experiment 667's head)"), color=alt.Color("arm:N", scale=alt.Scale(domain=list(COL)[2:] + ["L1", "L2"], range=[COL[a] for a in list(COL)[2:] + ["L1", "L2"]]), title="Arm"))  # ty: ignore[unresolved-attribute]
    chart = (base.mark_errorbar().encode(y=alt.Y("lo:Q", title="Mean hours until the prevailing output reacts"), y2="hi:Q") + base.mark_point(size=60, filled=True).encode(y="hours:Q")).properties(width=CONTENT_WIDTH_PX - 80, height=H).facet(row=alt.Row("when:N", title=None))  # ty: ignore[unresolved-attribute]
    chart.save(out / "reaction_by_lags.svg")


def cumulative(*, out: Path) -> None:
    r = pl.read_parquet(DATA / "score" / "reaction.parquet").filter(pl.col("kind") == "injected")
    rows = []
    for arm in ("L0", "L1", "L2", "L3", "P", "P7"):
        a = r.filter(pl.col("arm") == arm)
        n = a.height
        x = np.sort(a["start_h"].drop_nans().to_numpy())
        for h in range(0, 337, 24):
            rows.append({"arm": arm, "hours": h, "share": float((x <= h).sum() / n)})
    df = pl.DataFrame(rows).to_pandas()
    alt.Chart(df).mark_line(strokeWidth=1.5, point=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hours:Q", title="Hours after the event's start"),
        y=alt.Y("share:Q", title="Share of injected events reacted", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("arm:N", scale=alt.Scale(domain=list(COL), range=list(COL.values())), title="Arm"),
    ).properties(width=CONTENT_WIDTH_PX - 80, height=H).save(out / "reacted_share.svg")


def arm_q(*, arm: str, month: str) -> np.ndarray:
    if month < "2026-01":
        return np.load(dat.ROOT / "prevailing" / "fits" / f"Bhaz_{month}.npz")["real_q"].astype(np.float32)
    parts = [np.load(DATA / "fits" / arm / f"Rhaz_{month}_s{s}.npz")["real_q"] for s in (0, 1)]
    return np.sort(np.mean(parts, axis=0), axis=1)


def rows_of(*, prim: int, months: tuple[str, ...], arms: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    sm = k578.sites()
    cap = float(dict(zip(sm["primary"].to_list(), sm["capacity_mw"].to_list(), strict=True))[prim])
    out: dict[str, list[pl.DataFrame]] = {a: [] for a in ("NRA", *arms)}
    for month in months:
        z = np.load(dat.HEAD_DATA / f"real_{month}.npz")
        sel = z["prim"] == prim
        for name in out:
            q = np.sort(z["q"], axis=1) if name == "NRA" else arm_q(arm=name, month=month)
            out[name].append(pl.DataFrame({"oi": z["oi"][sel], "vi": z["vi"][sel], "lead": z["lead"][sel], "y": z["y"][sel] * cap, **{f"q{k}": q[sel, j] * cap for k, j in (("05", 1), ("25", 3), ("50", 5), ("75", 7), ("95", 9))}}))
    return {k: pl.concat(v) for k, v in out.items()}


def transfer_figure(*, label: str, prim: int, out: Path) -> None:
    data = rows_of(prim=prim, months=("2025-12", "2026-01"), arms=("L0", "L3"))
    lo, hi = START_IDX - 14 * DAY, END_IDX + 14 * DAY
    panels = []
    shade = alt.Chart(pd.DataFrame({"a": [to_time(np.array([START_IDX]))[0]], "b": [to_time(np.array([END_IDX]))[0]]})).mark_rect(color="#d9d9d9", opacity=0.6).encode(x="a:T", x2="b:T")  # ty: ignore[unresolved-attribute]
    for title, names in (("Normal-running output", ("NRA",)), ("Prevailing output: experiment 667's head (L0) and with 14 extra lags (L3)", ("L0", "L3"))):
        layers = [shade]
        for name in names:
            d = data[name].filter((pl.col("lead") >= 3.0) & (pl.col("lead") < 27.0) & (pl.col("vi") >= lo) & (pl.col("vi") < hi)).sort("vi")
            d = d.with_columns(time=pl.Series(to_time(d["vi"].to_numpy()))).to_pandas()
            colour = COL[name]
            layers += [
                alt.Chart(d).mark_area(opacity=0.18, color=colour).encode(x=alt.X("time:T", title="Date (UTC)"), y=alt.Y("q05:Q", title=f"{label} power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"),  # ty: ignore[unresolved-attribute]
                alt.Chart(d).mark_line(color=colour, strokeWidth=1).encode(x="time:T", y="q50:Q"),  # ty: ignore[unresolved-attribute]
            ]
        obs = data["NRA"].filter((pl.col("lead") >= 3.0) & (pl.col("lead") < 27.0) & (pl.col("vi") >= lo) & (pl.col("vi") < hi)).sort("vi")
        obs = obs.with_columns(time=pl.Series(to_time(obs["vi"].to_numpy()))).to_pandas()
        layers.append(alt.Chart(obs).mark_line(color="#000000", strokeWidth=1).encode(x="time:T", y="y:Q"))  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(*layers).properties(width=CONTENT_WIDTH_PX - 80, height=H, title=title))
    alt.vconcat(*panels).save(out / f"transfer_{label}.svg")


def fan_figure(*, label: str, prim: int, origin: datetime, months: tuple[str, ...], tag: str, out: Path) -> None:
    oi = int((origin - k578.GRID_START).total_seconds() // 1800)
    data = rows_of(prim=prim, months=months, arms=("L0", "L3"))
    panels = []
    for title, names in (("Normal-running output", ("NRA",)), ("Prevailing output: L0 and L3", ("L0", "L3"))):
        layers = []
        for name in names:
            d = data[name].filter(pl.col("oi") == oi).sort("vi")
            d = d.with_columns(time=pl.Series(to_time(d["vi"].to_numpy()))).to_pandas()
            colour = COL[name]
            layers += [
                alt.Chart(d).mark_area(opacity=0.12, color=colour).encode(x=alt.X("time:T", title="Date (UTC)"), y=alt.Y("q05:Q", title=f"{label} power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"),  # ty: ignore[unresolved-attribute]
                alt.Chart(d).mark_area(opacity=0.2, color=colour).encode(x="time:T", y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
                alt.Chart(d).mark_line(color=colour, strokeWidth=1).encode(x="time:T", y="q50:Q"),  # ty: ignore[unresolved-attribute]
            ]
        obs = data["NRA"].filter(pl.col("oi") == oi).sort("vi")
        obs = obs.with_columns(time=pl.Series(to_time(obs["vi"].to_numpy()))).to_pandas()
        layers.append(alt.Chart(obs).mark_line(color="#000000", strokeWidth=1).encode(x="time:T", y="y:Q"))  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(*layers).properties(width=CONTENT_WIDTH_PX - 80, height=H, title=title))
    alt.vconcat(*panels).save(out / f"fan_{label}_{tag}.svg")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    sm = k578.sites()
    prim_of = dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True))
    headline(out=out)
    cumulative(out=out)
    for label in ("D5", "D16"):
        transfer_figure(label=label, prim=prim_of[label], out=out)
    fan_figure(label="D5", prim=prim_of["D5"], origin=datetime(2026, 1, 8, 9, tzinfo=UTC), months=("2026-01",), tag="winter", out=out)
    fan_figure(label="D5", prim=prim_of["D5"], origin=datetime(2026, 6, 10, 9, tzinfo=UTC), months=("2026-06",), tag="summer", out=out)
    print("figures written to", out, os.environ.get("DATA_PATH_INTERNAL", ""))


if __name__ == "__main__":
    main()

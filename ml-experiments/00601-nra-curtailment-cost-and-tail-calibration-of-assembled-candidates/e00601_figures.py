# ruff: noqa: E501, D103, PLR0915
"""Experiment 601 figures, read from the score tables (run `e00601_real.py` and `e00601_bank.py c1|a1` first).

Usage: python e00601_figures.py
Demand series only; every plotted series is a Primary substation.
"""

from datetime import UTC, datetime
import os
from pathlib import Path

import altair as alt
import e00601_common as k
import polars as pl
from plotting import ocf_theme as theme

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / Path(__file__).resolve().parent.name
W = 620
PNG_DIR = os.environ.get("PNG_DIR")
COL = {"C0": theme.DATA_PURPLE, "E2": theme.DATA_GREEN, "C1": theme.DATA_SKY, "C1 + G4": theme.DATA_BLUE, "A1": theme.BRAND_ORANGE_LIGHT, "A1 + G4": theme.BRAND_ORANGE}
ORDER = list(COL)
SYS_SCALE = alt.Scale(domain=ORDER, range=[COL[s] for s in ORDER])
BANK_COL = {"plain": theme.DATA_BLUE, "G4": theme.BRAND_ORANGE, "twin": theme.DATA_GREEN}
BANK_NAME = {"plain": "Network alone", "G4": "Network + G4 band", "twin": "Lags-masked twin"}
BANK_SCALE = alt.Scale(domain=list(BANK_NAME.values()), range=[BANK_COL[s] for s in BANK_COL])
PHASE = {"before": "Before (7 days)", "days0-3": "Days 0-3", "days4-end": "Day 4 to end", "after0-7": "0-7 days after", "after7-14": "7-14 days after"}


def save(chart: alt.TopLevelMixin, name: str) -> None:
    chart.save(str(DOCS / f"{name}.svg"))
    if PNG_DIR:
        chart.save(str(Path(PNG_DIR) / f"{name}.png"), scale_factor=1.0)


def dots(*, df: pl.DataFrame, x: str, y: str, lo: str, hi: str, colour: str, scale: alt.Scale, title: str, ytitle: str = "", height: int = 150, rule: float | None = None, rule_label: str = "", log: bool = False, offset: str | None = None, ydomain: list | None = None, order: list | None = None):
    base = alt.Chart(df.to_pandas())
    xs = alt.Scale(type="symlog") if log else alt.Scale(zero=False)
    enc_y = alt.Y(f"{y}:N", title=ytitle, sort=order) if order else alt.Y(f"{y}:N", title=ytitle)
    layers = [
        base.mark_rule(strokeWidth=2).encode(x=alt.X(f"{lo}:Q", scale=xs, title=title), x2=f"{hi}:Q", y=enc_y, color=alt.Color(f"{colour}:N", scale=scale, legend=alt.Legend(orient="top", title=None)), **({"yOffset": f"{offset}:N"} if offset else {})),
        base.mark_point(filled=True, size=60).encode(x=alt.X(f"{x}:Q", scale=xs), y=enc_y, color=alt.Color(f"{colour}:N", scale=scale), **({"yOffset": f"{offset}:N"} if offset else {})),
    ]
    if rule is not None:
        layers.append(alt.Chart(pl.DataFrame({"v": [rule], "t": [rule_label]}).to_pandas()).mark_rule(strokeDash=[4, 3], color=theme.BLACK_1).encode(x="v:Q"))
    return alt.layer(*layers).properties(width=W, height=height)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    cost = pl.read_csv(k.WORK / "cost_real.csv")
    tail = pl.read_csv(k.WORK / "tail_real.csv")
    # F1: event-free cost, f = 0.9
    c = cost.filter(pl.col("row_set") == "event-free", pl.col("fraction") == 0.9)
    save(dots(df=c, x="gbp_per_primary_day", y="system", lo="lo", hi="hi", colour="system", scale=SYS_SCALE, title="Avoidable curtailment cost, pounds per Primary-day (smaller is better)", order=ORDER, height=190, ytitle=""), "fig_cost_event_free")
    # F2: sweep of the limit fraction
    c2 = cost.filter(pl.col("row_set") == "event-free").with_columns(f=pl.col("fraction").map_elements(lambda v: f"limit = {v:.2f} x past maximum", return_dtype=pl.String))
    save(dots(df=c2, x="gbp_per_primary_day", y="f", lo="lo", hi="hi", colour="system", scale=SYS_SCALE, title="Avoidable curtailment cost, pounds per Primary-day (smaller is better; log-like axis)", offset="system", log=True, height=260, ytitle="", order=["limit = 0.90 x past maximum", "limit = 0.75 x past maximum", "limit = 0.50 x past maximum"]), "fig_cost_fraction_sweep")
    # F3: real-data event-free tail exceedance
    for m, nominal, name in (("ex95", 0.05, "q95"), ("ex99", 0.01, "q99")):
        t = tail.filter(pl.col("row_set") == "event-free", pl.col("band") == "24-120 h", pl.col("group").is_in([k.SIX, k.EIGHT]))
        save(dots(df=t, x=m, y="group", lo=f"{m}_lo", hi=f"{m}_hi", colour="system", scale=SYS_SCALE, title=f"Share of rows with the observed power above {name} (dashed line: nominal {nominal})", offset="system", rule=nominal, height=200), f"fig_exceed_{name}_event_free")
    # F4: bank cost by phase, per network
    for arm in ("c1", "a1"):
        s = pl.read_csv(k.WORK / arm / "bank_summary.csv").filter(pl.col("group") == k.SIX, pl.col("length_class") == "15 and 30 days", pl.col("band") == "cost 12-36 h")
        s = s.with_columns(phase_label=pl.col("phase").replace(PHASE), name=pl.col("system").replace(BANK_NAME)).rename({"cost_0.9": "cost_09", "cost_0.9_lo": "cost_09_lo", "cost_0.9_hi": "cost_09_hi"})
        save(dots(df=s, x="cost_09", y="phase_label", lo="cost_09_lo", hi="cost_09_hi", colour="name", scale=BANK_SCALE, title=f"{arm.upper()}: avoidable curtailment cost, pounds per Primary-day, limit = 0.9 x past maximum (smaller is better)", offset="name", order=list(PHASE.values()), height=260), f"fig_bank_cost_{arm}")
        for m, nominal, qn in (("ex95", 0.05, "q95"), ("ex99", 0.01, "q99")):
            s2 = pl.read_csv(k.WORK / arm / "bank_summary.csv").filter(pl.col("length_class") == "15 and 30 days", pl.col("band") == "24-120 h", pl.col("group").is_in([k.SIX, k.EIGHT]))
            s2 = s2.with_columns(phase_label=pl.col("phase").replace(PHASE), name=pl.col("system").replace(BANK_NAME)).group_by("phase_label", "name").agg(pl.col(m).mean(), pl.col(f"{m}_lo").mean(), pl.col(f"{m}_hi").mean())
            save(dots(df=s2, x=m, y="phase_label", lo=f"{m}_lo", hi=f"{m}_hi", colour="name", scale=BANK_SCALE, title=f"{arm.upper()}: share of rows with the observed power above {qn} (dashed: nominal {nominal})", offset="name", rule=nominal, order=list(PHASE.values()), height=260), f"fig_bank_{qn}_{arm}")


if __name__ == "__main__":
    main()

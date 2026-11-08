# ruff: noqa: D103, E501
"""Experiment 646 figures and the numbers they show (CPU): read the score tables, write SVGs and CSVs.

Usage: python e00646_figures.py SCORE_B_DIR SCORE_C_DIR OUT_DIR
Writes `contrasts.csv` (every arm minus its comparator, event-free, 14 Primaries, by lead band and
width variant, with the month-block 95% interval) and the two SVG figures of the page.
"""

import sys
from pathlib import Path

import altair as alt
import e00645_score as s645  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

BANDS = {
    "day 1": (0, 24),
    "day 2": (24, 48),
    "days 3-5": (48, 120),
    "days 6-10": (120, 240),
    "days 1-10": (0, 240),
}
ARMS = {
    "B, run spread": "B_{n}_run",
    "B, zero spread": "B_{n}_zero",
    "C, run spread": "C_{n}_run",
    "C, zero spread": "C_{n}_zero",
}
NETS = {"K1 without the gate": ("K2", "A_K2"), "K3": ("K3", "A_K3")}
WIDTH = {"unscaled": "As fitted", "scaled": "Equal width scaling"}


def contrasts(*, score_b: Path, score_c: Path) -> pl.DataFrame:
    rb, rc = np.load(score_b / "rows.npz"), np.load(score_c / "rows.npz")
    rows = []
    for net, (short, comp) in NETS.items():
        for arm, pat in ARMS.items():
            name = pat.format(n=short)
            r = rb if name.startswith("B_") else rc
            if f"crps_{name}_unscaled" not in r.files:
                r = rb if f"crps_{name}_unscaled" in rb.files else rc
            ref = r
            base = ~r["flagged"] & ~r["short"]
            for variant in WIDTH:
                diff = r[f"crps_{name}_{variant}"] - ref[f"crps_{comp}_{variant}"]
                for band, (lo, hi) in BANDS.items():
                    m_ = base & (r["lead"] >= lo) & (r["lead"] < hi)
                    mean, mlo, mhi = s645.month_boot(block=r["month"][m_], vals=diff[m_], seed=646)
                    rows.append(
                        {
                            "network": net,
                            "arm": arm,
                            "variant": variant,
                            "band": band,
                            "diff": mean,
                            "lo": mlo,
                            "hi": mhi,
                        }
                    )
    return pl.DataFrame(rows)


def title(*, head: str, sub: list[str]) -> alt.TitleParams:
    return alt.TitleParams([head], subtitle=sub, anchor="start", fontSize=12, subtitleFontSize=11)


def headline(df: pl.DataFrame, out: Path) -> None:
    d = df.filter(pl.col("band") == "days 1-10").with_columns(
        pl.col("variant").replace(WIDTH).alias("width")
    )
    colours = {"As fitted": theme.DATA_BLUE, "Equal width scaling": theme.BRAND_ORANGE}
    panels = []
    for net in NETS:
        base = alt.Chart(d.filter(pl.col("network") == net)).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=list(ARMS), title=None),
            yOffset=alt.YOffset("width:N", sort=list(colours)),
            color=alt.Color(
                "width:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", symbolType="circle"),
            ),
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="#555", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        bar = (
            alt.Chart(pl.DataFrame({"x": [0.06]}))
            .mark_rule(color="#888", strokeDash=[2, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            (
                base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "lo:Q",
                        axis=alt.Axis(format=".2f", tickCount=6),
                        title="Arm's CRPS minus the stored forecast's CRPS (points of capacity; below 0 is better)",
                    ),
                    x2="hi:Q",
                )  # ty: ignore[unresolved-attribute]
                + base.mark_point(filled=True, size=55).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
                + zero
                + bar
            ).properties(
                width=520, height=150, title=alt.TitleParams(net, anchor="start", fontSize=11)
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=title(
            head="No arm lowers the score by the 0.06-point bar, and training on single members does not lower it",
            sub=[
                "Dot: estimate. Line: 95% interval from resampling the 5 origin months. Dashed rule: 0, the same score as",
                "the stored forecast. Dotted rule: +0.06; a gain would lie at -0.06 or lower. Event-free rows, 14 Primaries",
                "(without D15 and D18), days 1 to 10, seed 0, development months. B: stored weights run on each member.",
                "C: weights refitted on single members. Blue: bands as fitted. Orange: bands rescaled to equal width.",
            ],
        )
    )
    chart.save(str(out / "crps_difference_by_arm.svg"))


def by_band(df: pl.DataFrame, out: Path) -> None:
    d = df.filter(
        (pl.col("band") != "days 1-10")
        & (pl.col("variant") == "unscaled")
        & pl.col("arm").is_in(["B, run spread", "C, run spread"])
    )
    colours = {"B, run spread": theme.DATA_GREEN, "C, run spread": theme.DATA_PURPLE}
    panels = []
    for net in NETS:
        base = alt.Chart(d.filter(pl.col("network") == net)).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("band:N", sort=list(BANDS), title=None),
            yOffset=alt.YOffset("arm:N", sort=list(colours)),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", symbolType="circle"),
            ),
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="#555", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            (
                base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "lo:Q",
                        axis=alt.Axis(format=".2f", tickCount=6),
                        title="Arm's CRPS minus the stored forecast's CRPS (points of capacity; below 0 is better)",
                    ),
                    x2="hi:Q",
                )  # ty: ignore[unresolved-attribute]
                + base.mark_point(filled=True, size=55).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
                + zero
            ).properties(
                width=520, height=170, title=alt.TitleParams(net, anchor="start", fontSize=11)
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=title(
            head="Pooling the stored networks helps a little at long leads; refitting them does not",
            sub=[
                "Dot: estimate. Line: 95% interval from resampling the 5 origin months. Dashed rule: 0, the same score as",
                "the stored forecast. Bands as fitted, no width scaling. Event-free rows, 14 Primaries (without D15 and",
                "D18), seed 0, development months. B: stored weights run on each member. C: refitted on single members.",
            ],
        )
    )
    chart.save(str(out / "crps_difference_by_lead_band.svg"))


if __name__ == "__main__":
    sb, sc, o = (Path(a) for a in sys.argv[1:4])
    o.mkdir(parents=True, exist_ok=True)
    table = contrasts(score_b=sb, score_c=sc)
    table.write_csv(o / "contrasts.csv")
    headline(table, o)
    by_band(table, o)
    print(table.filter(pl.col("band") == "days 1-10"))

# ruff: noqa: E501, D103, PERF401
"""Experiment 606 figures. Run `e00606_score.py` first.

`python e00606_figures.py summary` draws the figures read from the score tables: follow by arm,
coverage by phase, and event-free CRPS cost by arm. `python e00606_figures.py example` draws the
December 2025 example event (Primaries D5 and D16, 14 days before the maintainer's onset to 14 days
after the end), one figure per arm, with experiment 598's drawing code pointed at this experiment's
tables and arms. The plot rules are experiment 598's (at most two panels, each at least 280 px high,
a
legend above the data, no NRA line, observed power black and nothing else black or grey).
"""

import argparse
from pathlib import Path

import altair as alt
import e00598_figures as f598  # ty: ignore[unresolved-import]
import e00606_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
f598.score = score
f598.DOCS = DOCS
score.S_ARMS = ("P5", "P7")  # the arms the example event shows  # ty: ignore[unresolved-attribute]
W, PANEL_H = f598.W, f598.PANEL_H

LABEL = {
    "A1": "A1 alone", "R": "R: A1 + G4", "W": "W: A1, band widened", "G4T": "G4 on T",
    "P3": "P3: gap", "P4": "P4: gap + sustain", "P5": "P5: gap + averages",
    "P6": "P6: P5 + sustain", "P7": "P7: P5 + neighbours",
    "P7b": "P7b: no own-plus-sum", "P7c": "P7c: P5 + best neighbour",
    "P5-p": "P5-p: noise for the inputs", "P7-p": "P7-p: noise for the inputs",
    "P5-q": "P5-q: P5 + paired shift", "P7-q": "P7-q: P7 + paired shift",
    "P5-rs": "P5, width-matched", "P7-rs": "P7, width-matched",
    "P5-q-rs": "P5-q, width-matched", "P7-q-rs": "P7-q, width-matched",
    "P5-p-rs": "P5-p, width-matched", "P7-p-rs": "P7-p, width-matched",
}  # fmt: skip
COLOUR = {
    "A1": theme.DATA_SKY, "R": theme.BRAND_ORANGE, "W": theme.DATA_PURPLE,
    "G4T": theme.DATA_BURNT_ORANGE, "P3": theme.DATA_BLUE_LIGHT, "P4": theme.DATA_GREEN_LIGHT,
    "P5": theme.DATA_BLUE, "P6": theme.DATA_MAGENTA, "P7": theme.DATA_GREEN,
    "P7b": f598.lighten(theme.DATA_GREEN), "P7c": theme.DATA_PURPLE,
    "P5-p": f598.lighten(theme.DATA_BLUE, 0.6), "P7-p": f598.lighten(theme.DATA_GREEN, 0.6),
    "P5-q": theme.DATA_BURNT_ORANGE, "P7-q": theme.DATA_MAGENTA,
    "P5-rs": f598.lighten(theme.DATA_BLUE, 0.4), "P7-rs": f598.lighten(theme.DATA_GREEN, 0.4),
    "P5-q-rs": f598.lighten(theme.DATA_BURNT_ORANGE, 0.4),
    "P7-q-rs": f598.lighten(theme.DATA_MAGENTA, 0.4),
    "P5-p-rs": f598.lighten(theme.DATA_BLUE, 0.7), "P7-p-rs": f598.lighten(theme.DATA_GREEN, 0.7),
}  # fmt: skip
f598.GATE_STRIP = (
    theme.DATA_PURPLE
)  # a hue, so the strip is not mistaken for the black observed line
f598.LABEL.update(LABEL)
f598.COLOUR.update(COLOUR)


def summary_table() -> pl.DataFrame:
    return pl.read_csv(score.ROOT / "score" / "arm_summary.csv", infer_schema_length=None)


def follow_chart() -> None:
    a = summary_table()
    order = ["A1", "P3", "P4", "P5", "P6", "P7b", "P7c", "P7", "P5-q", "P7-q", "P5-p", "P7-p"]
    a = a.filter(pl.col("system").is_in(order)).with_columns(name=pl.col("system").replace(LABEL))
    names = [LABEL[s] for s in order]
    base = alt.Chart(a).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("name:N", sort=names, title=None, axis=alt.Axis(labelLimit=260)),
    )
    chart = alt.layer(
        base.mark_rule(color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("follow_q50_days4-end_lo:Q", title="Follow: share of the injected depth that the median follows (lower is better)", scale=alt.Scale(domain=[-0.02, 0.18]), axis=alt.Axis(values=[0, 0.04, 0.08, 0.12, 0.16])),
            x2="follow_q50_days4-end_hi:Q",
        ),
        base.mark_point(filled=True, size=70, color=theme.DATA_BLUE).encode(x="follow_q50_days4-end:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=W, height=420,
        title=alt.TitleParams(
            ["Follow by arm", "Injected 15 and 30 day load-shifts, day 4 to the end, 24-120 h ahead, both metering groups, 2025-10 and 2025-12.", "Dot: mean over Primary-events. Line: 99.6% interval (Bonferroni across 12 counted contrasts) from resampling (month, Primary) clusters.", "R is not drawn: its follow equals A1's. -p: noise in place of the inputs."],
            fontSize=11, anchor="start",
        ),
    )  # fmt: skip
    f598.save(chart, "follow_by_arm.svg")


def cost_chart() -> None:
    a = summary_table()
    order = ["P3", "P4", "P5", "P6", "P7b", "P7c", "P7", "P5-q", "P7-q"]
    rows = []
    for s in order:
        r = a.filter(pl.col("system") == s)
        for grp in ("MVA-only eight", "signed-MW six"):
            rows.append({"system": s, "name": LABEL[s], "group": grp.replace("MVA-only eight", "Eight MVA Primaries").replace("signed-MW six", "Six MW Primaries"), "cost": float(r[f"ef_crps_minus_A1_{grp}"][0]), "mde": float(r[f"ef_crps_minus_A1_{grp}_mde"][0])})  # fmt: skip
    d = pl.DataFrame(rows).with_columns(
        lo=pl.col("cost") - pl.col("mde"), hi=pl.col("cost") + pl.col("mde")
    )
    names = [LABEL[s] for s in order]
    base = alt.Chart(d).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("name:N", sort=names, title=None, axis=alt.Axis(labelLimit=260)),
    )
    layers = [
        base.mark_rule(color=theme.DATA_BLUE).encode(x=alt.X("lo:Q", title="CRPS minus A1's, points of capacity"), x2="hi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70, color=theme.DATA_BLUE).encode(x="cost:Q"),  # ty: ignore[unresolved-attribute]
    ]  # fmt: skip
    chart = alt.layer(*layers).properties(width=W // 2 - 40, height=360).facet(column=alt.Column("group:N", title=None)).properties(
        title=alt.TitleParams(["Event-free CRPS cost of each arm", "Rows with no load-shift, 24-120 h ahead, 2025-10 and 2025-12. Dot: difference from A1. Line: plus and minus the minimum detectable effect."], fontSize=11, anchor="start")
    )  # fmt: skip
    f598.save(chart, "event_free_cost.svg")


def summary() -> None:
    present = (
        pl.read_csv(score.ROOT / "score" / "bank_summary.csv", infer_schema_length=None)["system"]
        .unique()
        .to_list()
    )
    systems = [s for s in ("A1", "R", "P5", "P7", "P5-rs", "P7-rs", "P7-q-rs") if s in present]
    f598.phase_chart(
        suffix="", metric="cover90", systems=systems, name="coverage_by_phase.svg", domain=[0.7, 1.0], target=0.9,
        title=["Share of rows with the NRA inside q05-q95", "Coverage by phase of an injected load-shift: 15 and 30 day events, both metering groups, 24-120 h ahead", "Dot: mean over Primary-events. Line: 95% interval from resampling events (not Bonferroni-adjusted). Dashed: the 0.90 target. Scored against the NRA, an estimate. Width-matched: band rescaled to R's mean width."],
    )  # fmt: skip
    follow_chart()
    cost_chart()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=("summary", "example"))
    a = p.parse_args()
    summary() if a.which == "summary" else f598.example(suffix="")

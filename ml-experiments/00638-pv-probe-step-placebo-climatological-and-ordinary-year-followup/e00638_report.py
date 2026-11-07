# ruff: noqa: E501
"""Experiment 638: the figures of the write-up, from `work/step_reads_shift{0,364}.csv`.

Usage: python e00638_report.py
"""

from pathlib import Path

import altair as alt
import e00638_analyse as a
import e00638_common as k
import polars as pl

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00638-pv-probe-step-placebo-climatological-and-ordinary-year-followup"
)
W = 640
LABEL = {"2025-07": "Jul 2025", "2025-10": "Oct 2025", "2026-03": "Mar 2026", "2026-06": "Jun 2026"}
ORDER = list(LABEL.values())
STEP_TITLE = {
    "spring23": "Step from April 2023 (ordinary spring)",
    "summer24": "Step from July 2024 (post-step period sunnier than before)",
    "spring25": "Step from April 2025 (unusually sunny spring)",
    "summer25": "Step from July 2025",
}
COLOURS = {"real": "#FF4901", "climatological": "#306BFF", "shuffled": "#10C5F7"}
NAMES = {
    "real": "Real farm",
    "climatological": "Climatological placebo",
    "shuffled": "Shuffled-shape placebo (634)",
}


def kinds() -> pl.DataFrame:
    """Return the per-Primary median calibrated read for every step, kind, and cutoff."""
    rec = pl.read_csv(k.WORK / "step_reads_shift0.csv")
    kind = {}
    for key in k.STEP_STARTS:
        kind[k.real_condition(key=key)] = (key, "real")
        kind[f"climstep_{key}_mw4"] = (key, "climatological")
    kind["shufstep_spring_mw4"] = ("spring25", "shuffled")
    kind["shufstep_summer_mw4"] = ("summer25", "shuffled")
    sub = rec.filter(pl.col("condition").is_in(list(kind))).with_columns(
        key=pl.col("condition").replace_strict({c: v[0] for c, v in kind.items()}),
        kind=pl.col("condition").replace_strict({c: v[1] for c, v in kind.items()}),
    )
    per = sub.group_by("key", "kind", "cutoff", "site").agg(v=pl.col("recovery_cal").median())
    return (
        per.group_by("key", "kind", "cutoff")
        .agg(med=pl.col("v").median(), lo=pl.col("v").min(), hi=pl.col("v").max())
        .with_columns(
            cutoff_label=pl.col("cutoff").replace_strict(LABEL),
            kind_name=pl.col("kind").replace_strict(NAMES),
        )
    )


def steps_figure() -> None:
    """One panel per step: calibrated read against cutoff, with the post-step donor share."""
    d = kinds()
    share = a.donor_context().with_columns(cutoff_label=pl.col("cutoff").replace_strict(LABEL))
    panels = []
    for key, title in STEP_TITLE.items():
        sel = d.filter(pl.col("key") == key)
        base = alt.Chart(sel)
        color = alt.Color(
            "kind_name:N", scale=alt.Scale(domain=list(NAMES.values()), range=list(COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", direction="vertical") if key == "spring23" else None,
        )  # fmt: skip
        x = alt.X(
            "cutoff_label:N", sort=ORDER, title="Training cutoff" if key == "summer25" else None
        )
        y = alt.Y("med:Q", scale=alt.Scale(domain=[-0.2, 1.1]), title="Calibrated read")
        band = base.mark_rule(opacity=0.5, aria=False).encode(x=x, y="lo:Q", y2="hi:Q", color=color)  # ty: ignore[unresolved-attribute]
        line = base.mark_line(point=True, aria=False).encode(x=x, y=y, color=color)  # ty: ignore[unresolved-attribute]
        sh = (
            alt.Chart(share.filter(pl.col("key") == key))  # ty: ignore[unresolved-attribute]
            .mark_point(shape="diamond", size=90, color="#555555", filled=True, aria=False)
            .encode(x=alt.X("cutoff_label:N", sort=ORDER), y="share:Q")
        )
        panels.append((band + line + sh).properties(title=title, width=W, height=170))
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            ["Calibrated read of a 4 MW capacity step at six MW Primaries (median over 3 seeds; line:", "range over the 6 Primaries). Grey diamond: share of the probe's donor rows after the step."],
            anchor="start", fontSize=12,
        )
    )  # fmt: skip
    chart.save(str(DOCS / "step_reads_by_cutoff.svg"))


def shift_figure() -> None:
    """Real-step reads with the 2025 probe dates against the 2024 probe dates."""
    a0 = pl.read_csv(k.WORK / "step_reads_shift0.csv")
    a1 = pl.read_csv(k.WORK / "step_reads_shift364.csv")
    names = [k.real_condition(key=key) for key in k.STEP_STARTS] + [
        f"climstep_{key}_mw4" for key in k.STEP_STARTS
    ]
    cols = ["cutoff", "condition", "site", "seed"]
    j = (
        a0.filter(pl.col("condition").is_in(names)).select(*cols, y0="recovery_cal")
        .join(a1.select(*cols, y1="recovery_cal"), on=cols)
        .group_by("cutoff", "condition", "site").agg(pl.col("y0", "y1").median())
        .with_columns(kind=pl.when(pl.col("condition").str.starts_with("climstep")).then(pl.lit("Climatological placebo")).otherwise(pl.lit("Real farm")))
    )  # fmt: skip
    pts = alt.Chart(j).mark_point(opacity=0.7, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("y0:Q", title="Calibrated read, probe dates in 2025", scale=alt.Scale(domain=[-0.3, 1.2])),
        y=alt.Y("y1:Q", title="Calibrated read, probe dates in 2024", scale=alt.Scale(domain=[-0.3, 1.2])),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Real farm", "Climatological placebo"], range=[COLOURS["real"], COLOURS["climatological"]]), legend=alt.Legend(title=None, orient="top")),
    )  # fmt: skip
    line = (
        alt.Chart(pl.DataFrame({"x": [-0.3, 1.2], "y": [-0.3, 1.2]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#999999", strokeDash=[4, 3], aria=False)
        .encode(x="x:Q", y="y:Q")
    )
    (line + pts).properties(
        width=W, height=300,
        title=alt.TitleParams(["Each point: one Primary, one step, one cutoff (median over 3 seeds).", "On the dashed line, moving the probe dates one year earlier changes nothing."], anchor="start", fontSize=12),
    ).save(str(DOCS / "previous_year_base.svg"))  # fmt: skip


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    steps_figure()
    shift_figure()

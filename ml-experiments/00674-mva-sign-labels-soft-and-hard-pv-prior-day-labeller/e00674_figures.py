"""Experiment 674 figures, drawn from the stored tables.

Run with `PYTHONPATH` set to the folders of experiments 87, 506, 508, and 674.
"""

from pathlib import Path

import altair as alt
import e00674_features as feat  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS = Path("docs/ml-experiments/00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller")
ARM_COLOURS = {"C": "#FF4901", "B": "#306BFF", "A": "#10C5F7"}
CLASS_COLOURS = {
    "reverse": "#FF4901",
    "unknown": "#B701FF",
    "not_scorable": "#CCCCCC",
    "no_reverse": "#306BFF",
}
ocf.register_and_enable_theme() if hasattr(ocf, "register_and_enable_theme") else None


def fig_known() -> None:
    """Figure 1: recall and false-reverse rate by arm, with intervals."""
    m = pl.read_parquet(feat.WORK / "known_answer_metrics.parquet").filter(
        pl.col("arm").is_in(list(ARM_COLOURS)),
        pl.col("subset") == "all",
        pl.col("level") == "all",
        pl.col("rbin").is_in(["1.5-3", "pooled_below_3"]),
    )
    rows = []
    for r in m.iter_rows(named=True):
        for metric, label in (
            ("recall", "Recall of reverse-flow days"),
            ("false_reverse", "False-reverse rate"),
        ):
            rows.append(
                {
                    "Arm": r["arm"],
                    "Days": "export 1.5 to 3 times night import"
                    if r["rbin"] == "1.5-3"
                    else "all days below 3 times",
                    "Metric": label,
                    "est": r[f"{metric}_est"],
                    "lo": r[f"{metric}_lo"],
                    "hi": r[f"{metric}_hi"],
                }
            )
    df = pl.DataFrame(rows)
    base = alt.Chart(df).encode(
        y=alt.Y("Arm:N", title=None),
        color=alt.Color(
            "Arm:N",
            scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
            legend=None,
        ),
    )
    chart = (
        (
            base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lo:Q", title="Share of days (95% month-block interval)"), x2="hi:Q"
            )
            + base.mark_point(filled=True, size=60).encode(x="est:Q")  # ty: ignore[unresolved-attribute]
        )
        .properties(width=300, height=110)
        .facet(row=alt.Row("Metric:N", title=None), column=alt.Column("Days:N", title=None))
    )
    chart.save(DOCS / "figure_1_recall_and_false_reverse_by_arm.svg")


def fig_eight() -> None:
    """Figure 2: label shares on the eight, April to September."""
    d = pl.read_parquet(feat.WORK / "eight_days_gate.parquet")
    rows = []
    for arm in ARM_COLOURS:
        g = d.group_by("site", f"cls_{arm}").len().rename({f"cls_{arm}": "Label"})
        tot = g.group_by("site").agg(pl.col("len").sum().alias("tot"))
        rows.append(
            g.join(tot, on="site").with_columns(
                share=pl.col("len") / pl.col("tot"), Arm=pl.lit(arm)
            )
        )
    df = pl.concat(rows)
    chart = (
        alt.Chart(df)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("share:Q", title="Share of April to September days", stack="normalize"),
            y=alt.Y("Arm:N", title=None),
            color=alt.Color(
                "Label:N",
                scale=alt.Scale(domain=list(CLASS_COLOURS), range=list(CLASS_COLOURS.values())),
            ),
        )
        .properties(width=260, height=70)
        .facet(
            row=alt.Row(
                "site:N", title=None, sort=["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
            ),
            columns=2,
        )
    )
    chart.save(DOCS / "figure_2_labels_on_the_eight_apparent_power_primaries.svg")


def fig_examples() -> None:
    """Figure 3: example days with the reading and the labels of arms C, B, and A."""
    picks = []
    six = pl.read_parquet(feat.WORK / "known_answer_days.parquet")
    d10 = (
        six.filter(pl.col("site") == "D10", pl.col("level") == 0.0, pl.col("crossing"))
        .sort("mid_min")
        .head(2)
    )
    eight = pl.read_parquet(feat.WORK / "eight_days_gate.parquet")
    picks += [(r["site"], r["date"], 0.0, r) for r in d10.iter_rows(named=True)]
    for site, n in (("D8", 2), ("D4", 1), ("D12", 1)):
        s = (
            eight.filter(
                pl.col("site") == site, pl.col("cls_C") == "reverse", pl.col("k_mid") >= 0.7
            )
            .sort("date")
            .tail(n * 7)
            .head(n)
        )
        picks += [(r["site"], r["date"], 0.0, r) for r in s.iter_rows(named=True)]
    frames = []
    for site, date, _level, row in picks:
        mats = feat.load_mats(site=site)
        i = int(np.flatnonzero(mats.dates == date)[0])
        sig = mats.p[i]
        label = f"{site} {date}: C {row['cls_C']}, B {row['cls_B']}, A {row['cls_A']}"
        frames.append(
            pl.DataFrame(
                {
                    "title": label,
                    "half_hour": np.arange(48) / 2,
                    "reading": np.abs(sig),
                    "signed": sig,
                }
            )
        )
    df = pl.concat(frames)
    chart = (
        alt.Chart(df)  # ty: ignore[unresolved-attribute]
        .mark_line(color="#306BFF")
        .encode(
            x=alt.X("half_hour:Q", title="UK clock hour"),
            y=alt.Y("reading:Q", title="Reading (points of capacity)"),
        )
        .properties(width=240, height=140)
        .facet(facet=alt.Facet("title:N", title=None), columns=3)
    )
    chart.save(DOCS / "figure_3_example_days_and_labels.svg")


if __name__ == "__main__":
    fig_known()
    fig_eight()
    fig_examples()

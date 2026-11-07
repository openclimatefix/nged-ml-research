"""Experiment 617, part A report: summarise the known-answer test and draw its figures.

Reads `known_answer_events.parquet` and `placebos.parquet` from `DATA_DIR`, prints every number the
write-up quotes, and writes `selected_step_up.json` and the SVG figures in the docs folder.
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00617_hmm as h
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00617-wind-turbine-step-capacity-known-answer-and-fit"
)
FARM_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
FARMS: Final[list[str]] = ["W1", "W2", "W3"]
RECALL_BAR: Final[float] = 0.8
FALSE_BAR: Final[float] = 1.0
STOP_RECALL: Final[float] = 0.5
STOP_FALSE: Final[float] = 3.0


def recall_table(events: pl.DataFrame) -> pl.DataFrame:
    """Return recall by step-up probability, farm, turbines out, and duration."""
    return events.group_by("step_up", "site", "m", "duration").agg(
        recall=pl.col("detected").mean(),
        n=pl.len(),
        recall_seen=pl.col("detected").filter(pl.col("observed_days") >= 1).mean(),
        seen_share=(pl.col("observed_days") >= 1).mean(),
    )


def main() -> None:
    """Print every quoted number and draw figures 1, 3 and 4."""
    events = pl.read_parquet(h.DATA_DIR / "known_answer_events.parquet")
    placebos = pl.read_parquet(h.DATA_DIR / "placebos.parquet")
    perm = (
        placebos.filter(pl.col("kind") == "permuted")
        .group_by("step_up", "site")
        .agg(
            false_per_year=pl.col("false_per_year").mean(),
            p95=pl.col("false_per_year").quantile(0.95),
        )
    )
    print("false steps per year on permuted history\n", perm.sort("step_up", "site"))
    drift = placebos.filter(pl.col("kind") == "drift").select(
        "step_up", "site", "false_per_year", "end_capacity_error"
    )
    print("drift placebo\n", drift.sort("step_up", "site"))
    target = events.filter((pl.col("m") >= 2) & (pl.col("duration") >= 14))
    key = target.group_by("step_up", "site").agg(
        recall=pl.col("detected").mean(),
        recall_seen=pl.col("detected").filter(pl.col("observed_days") >= 1).mean(),
        seen=(pl.col("observed_days") >= 1).mean(),
        n=pl.len(),
    )
    print("recall, m>=2 and duration>=14\n", key.sort("step_up", "site"))
    ok = (
        perm.group_by("step_up")
        .agg(worst=pl.col("false_per_year").max())
        .filter(pl.col("worst") <= FALSE_BAR)["step_up"]
        .to_list()
    )
    rec = (
        key.filter(pl.col("site").is_in(["W1", "W2"]))
        .group_by("step_up")
        .agg(recall=pl.col("recall").mean())
    )
    pool = (
        rec.filter(pl.col("step_up").is_in(ok))
        if ok
        else perm.group_by("step_up").agg(recall=-pl.col("false_per_year").max())
    )
    chosen = float(pool.sort("recall", descending=True)["step_up"][0])
    print("selected step-up probability a =", chosen, "(qualifying:", ok, ")")
    (h.DATA_DIR / "selected_step_up.json").write_text(json.dumps({"step_up": chosen}))
    ev = events.filter(pl.col("step_up") == chosen)
    det = ev.filter(pl.col("detected"))
    size_ok = det.group_by("site").agg(
        within_half=((pl.col("step_in_turbines") - pl.col("m")).abs() <= 0.5).mean(),
        n=pl.len(),
        mean_bias_turbines=(pl.col("step_in_turbines") - pl.col("m")).mean(),
    )
    print("step size within half a turbine, detected events\n", size_ok.sort("site"))
    runs = ev.unique(["site", "n_true", "m", "duration", "seed"])
    s_ok = runs.group_by("site", "n_true").agg(
        within15=((pl.col("n_map") / pl.col("n_true") - 1).abs() <= 0.15).mean(),
        n_map_median=pl.col("n_map").median(),
        n=pl.len(),
    )
    print("turbine count recovered within 15%\n", s_ok.sort("site", "n_true"))
    print(
        "by site",
        runs.group_by("site")
        .agg(
            within15=((pl.col("n_map") / pl.col("n_true") - 1).abs() <= 0.15).mean(),
            bias_sign=(pl.col("n_map") / pl.col("n_true") - 1).median(),
        )
        .sort("site"),
    )
    cover = runs.group_by("site").agg(
        on_outage_days=pl.col("run_covered_share").mean(), all_days=pl.col("run_covered_all").mean()
    )
    print("90% interval coverage\n", cover.sort("site"))
    rtab = recall_table(ev)
    DOCS.mkdir(parents=True, exist_ok=True)
    palette = alt.Scale(domain=FARMS, range=FARM_COLOURS)
    recall_data = rtab.filter(pl.col("m") >= 2).with_columns(
        turbines=pl.col("m").cast(pl.Utf8) + " turbines out",
        bar=pl.lit(RECALL_BAR),
        label=pl.lit("80% bar"),
    )
    line = (
        alt.Chart()
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("duration:O", title="Outage length (days)"),
            y=alt.Y("recall:Q", title="Share of outages detected", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("site:N", title="Wind farm", scale=palette),
        )
    )
    bar = alt.Chart().mark_rule(strokeDash=[4, 3], color=ocf.BLACK_1).encode(y="bar:Q")  # ty: ignore[unresolved-attribute]
    bar_label = alt.Chart().mark_text(align="left", dy=-6, x=4).encode(y="bar:Q", text="label:N")  # ty: ignore[unresolved-attribute]
    alt.layer(line, bar, bar_label, data=recall_data).properties(width=240, height=280).facet(
        column=alt.Column("turbines:N", title=None)
    ).save(DOCS / "figure_1_recall_by_duration.svg")
    step_data = det.filter(pl.col("m") >= 2).with_columns(
        turbines=pl.col("m").cast(pl.Utf8) + " turbines injected", injected=pl.col("m")
    )
    boxes = (
        alt.Chart()
        .mark_boxplot(extent="min-max")
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("duration:O", title="Outage length (days)"),
            y=alt.Y("step_in_turbines:Q", title="Recovered step (turbines)"),
            color=alt.Color("site:N", title="Wind farm", scale=palette),
            xOffset="site:N",
        )
    )
    truth_rule = (
        alt.Chart()
        .mark_rule(strokeDash=[4, 3], color=ocf.BLACK_1)  # ty: ignore[unresolved-attribute]
        .encode(y="injected:Q")
    )
    alt.layer(boxes, truth_rule, data=step_data).properties(width=240, height=280).facet(
        column=alt.Column("turbines:N", title=None)
    ).save(DOCS / "figure_3_recovered_vs_injected.svg")
    false_chart = (
        alt.Chart(perm.with_columns(a=pl.col("step_up").cast(pl.Utf8)))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("a:N", title="Daily failure probability"),
            y=alt.Y("false_per_year:Q", title="False steps a year on permuted history"),
            color=alt.Color("site:N", title="Wind farm", scale=palette),
            xOffset="site:N",
        )
        .properties(width=400, height=280)
    )
    rules = (
        alt.Chart(pl.DataFrame({"y": [FALSE_BAR, STOP_FALSE]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3])  # ty: ignore[unresolved-attribute]
        .encode(y="y:Q")
    )
    rule_labels = (
        alt.Chart(
            pl.DataFrame(
                {"y": [FALSE_BAR, STOP_FALSE], "t": ["pass bar (1 a year)", "stop rule (3 a year)"]}
            )
        )
        .mark_text(align="left", dy=-6, x=4)  # ty: ignore[unresolved-attribute]
        .encode(y="y:Q", text="t:N")
    )
    (false_chart + rules + rule_labels).save(DOCS / "figure_4_false_steps.svg")


def example_figure(step_up: float) -> None:
    """Draw one known-answer run: the injected deficit, the recovered one and its 90% band."""
    from dataclasses import replace

    import e00617_known_answer as k

    base = h.remove_fault_days(h.daily_series("W1"))
    rng = np.random.default_rng(3)
    starts = k._starts(rng, base.n_days, 60, 6)
    truth = np.zeros(base.n_days)
    factor = np.ones(base.n_days)
    for s0 in starts:
        factor[s0 : s0 + 60] = 1 - 3 / 14
        truth[s0 : s0 + 60] = 3 * base.ceiling / 14
    value = base.value * factor[base.day]
    shaped = replace(base, value=value, ceiling=float(np.quantile(value, 0.99)))
    fits = h.fit_grid(shaped, step_up=step_up)
    summary = h.deficit_summary(fits, h.log_n_weights(fits))
    years = np.arange(base.n_days) / 365.25
    frame = pl.DataFrame(
        {
            "years": years,
            "Injected deficit": truth,
            "Recovered (posterior mean)": summary["mean"],
            "q05": summary["q05"],
            "q95": summary["q95"],
            "band": "90% interval",
        }
    )
    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.3)
        .encode(  # ty: ignore[unresolved-attribute]
            color=alt.Color(
                "band:N",
                title=None,
                scale=alt.Scale(domain=["90% interval"], range=[ocf.DATA_BLUE_LIGHT]),
            ),
            x=alt.X("years:Q", title="Years since the first reading"),
            y=alt.Y("q05:Q", title="Capacity deficit (MW)"),
            y2="q95:Q",
        )
    )
    lines = (
        alt.Chart(frame.unpivot(["Injected deficit", "Recovered (posterior mean)"], index="years"))
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x="years:Q",
            y="value:Q",
            color=alt.Color(
                "variable:N",
                title=None,
                scale=alt.Scale(
                    domain=["Injected deficit", "Recovered (posterior mean)"],
                    range=[ocf.BLACK_1, ocf.DATA_BLUE],
                ),
            ),
        )
    )
    for s0 in starts:
        core = slice(s0 + 15, s0 + 45)
        pre = slice(max(s0 - 30, 0), s0)
        step = np.median(summary["mean"][core]) - np.median(summary["mean"][pre])
        print("example run, step in turbines:", round(step / (base.ceiling / 14), 2))
    (band + lines).resolve_scale(color="independent").properties(
        width=640,
        height=300,
        title="W1: six injected 60-day outages of 3 turbines",
    ).save(DOCS / "figure_5_example_run.svg")


if __name__ == "__main__":
    main()
    example_figure(json.loads((h.DATA_DIR / "selected_step_up.json").read_text())["step_up"])

# ruff: noqa: E501
"""Experiment 618: summary tables and figures from the CSVs the other scripts wrote.

Farms appear only as Solar A to Solar F. Time axes are years since the farm's first reading.

```bash
PYTHONPATH=ml-experiments/00618-* uv run --no-sync python ml-experiments/00618-*/e00618_report.py
```
"""

from pathlib import Path

import altair as alt
import e00618_common as c
import e00618_known_answer as ka
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS = Path("docs/ml-experiments/00618-solar-effective-capacity-sunny-rows-and-known-answer-test")
WIDTH = 680
HEIGHT = 300
FARM_COLOURS = {
    "A": ocf.BRAND_ORANGE,
    "B": ocf.DATA_BLUE,
    "C": ocf.DATA_SKY,
    "D": ocf.DATA_PURPLE,
    "F": ocf.DATA_GREEN,
}
Y_SCALE = alt.Scale(domain=[0.5, 1.3], clamp=True)
X_TITLE = "Years since the farm's first reading"


def tables() -> dict:
    """Print the numbers the write-up quotes and return the chosen settings."""
    penalty = float((c.DATA_DIR / "chosen_penalty.txt").read_text())
    inflate = float((c.DATA_DIR / "chosen_inflation.txt").read_text())
    inj = pl.read_csv(c.DATA_DIR / "injected_runs.csv").filter(pl.col("set") == "eval")
    at = inj.filter(pl.col("penalty") == penalty).with_columns(size=pl.col("delta").abs())
    print("penalty", penalty, "inflation", inflate)
    print(
        at.group_by("farm", "size")
        .agg(recall=pl.col("detected").mean(), n=pl.len())
        .sort("farm", "size")
        .pivot(on="size", index="farm", values="recall")
    )
    det = at.filter(pl.col("detected") > 0.5)
    det = det.with_columns(
        ok=((pl.col("est") - pl.col("delta")).abs() <= ka.SIZE_TOLERANCE * pl.col("size"))
    )
    print(
        det.group_by("farm", "size")
        .agg(size_ok=pl.col("ok").mean(), bias=(pl.col("est").abs() / pl.col("size") - 1).mean())
        .sort("farm", "size")
    )
    print(
        det.filter(pl.col("size") >= 0.08)
        .group_by("farm")
        .agg(size_ok=pl.col("ok").mean(), bias=(pl.col("est").abs() / pl.col("size") - 1).mean())
        .sort("farm")
    )
    print(
        "extra steps per injected run:",
        at.group_by("farm")
        .agg((pl.col("n_steps") - (pl.col("detected") > 0.5).cast(pl.Int8)).mean().alias("extra"))
        .sort("farm"),
    )
    print(
        "coverage at chosen inflation:",
        at.group_by("farm").agg(pl.col(f"cover_{inflate}").mean()).sort("farm"),
    )
    print("all-farm coverage:", at[f"cover_{inflate}"].mean())
    pl_ = pl.read_csv(c.DATA_DIR / "placebo_steps.csv").filter(
        (pl.col("set") == "eval") & (pl.col("penalty") == penalty)
    )
    print(
        pl_.group_by("farm", "kind")
        .agg(pl.col("per_year").mean())
        .sort("farm", "kind")
        .pivot(on="kind", index="farm", values="per_year")
    )
    print(
        pl.read_csv(c.DATA_DIR / "drift_runs.csv")
        .group_by("farm")
        .agg(
            pl.col("steps_per_year").mean(),
            pl.col("final_error").mean(),
            pl.col("final_error").abs().max().alias("worst"),
        )
        .sort("farm")
    )
    print(pl.read_csv(c.DATA_DIR / "real_steps.csv"))
    print(pl.read_csv(c.DATA_DIR / "real_comparison.csv"))
    return {"penalty": penalty, "inflate": inflate}


def fig_sunny_counts() -> None:
    """Median sunny half-hours by calendar month and farm, with the stop-rule line."""
    d = pl.read_csv(c.DATA_DIR / "step1_counts.csv").filter(pl.col("site").is_in(list(c.TESTABLE)))
    t = (
        d.group_by("site", "calendar_month")
        .agg(pl.col("sunny").median().alias("median"))
        .filter(pl.col("calendar_month").is_between(4, 8))
    )
    t = t.with_columns(
        farm="Solar " + pl.col("site"),
        month=pl.col("calendar_month").replace_strict(
            {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug"}
        ),
    )
    bars = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "month:N",
                sort=["Apr", "May", "Jun", "Jul", "Aug"],
                title="Calendar month",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset="farm:N",
            y=alt.Y("median:Q", title="Median very sunny half-hours in the month"),
            color=alt.Color(
                "farm:N",
                title=None,
                scale=alt.Scale(
                    domain=[f"Solar {f}" for f in c.TESTABLE],
                    range=[FARM_COLOURS[f] for f in c.TESTABLE],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
    )
    rule = alt.Chart(pl.DataFrame({"y": [20.0], "rule": ["Stop-rule line (20)"]})).mark_rule(strokeDash=[5, 4], size=2).encode(y="y:Q", color=alt.Color("rule:N", title=None, scale=alt.Scale(domain=["Stop-rule line (20)"], range=["#000000"]), legend=alt.Legend(orient="top")))  # fmt: skip  # ty: ignore[unresolved-attribute]
    (bars + rule).resolve_scale(color="independent").properties(
        width=WIDTH - 90, height=HEIGHT, title="Very sunny half-hours per month, Solar A to D and F"
    ).save(DOCS / "figure_1_sunny_counts.svg")


def fig_recall(*, penalty: float) -> None:
    """Recall against injected step size, one line per farm."""
    inj = pl.read_csv(c.DATA_DIR / "injected_runs.csv").filter(
        (pl.col("set") == "eval") & (pl.col("penalty") == penalty)
    )
    t = (
        inj.with_columns(size=pl.col("delta").abs() * 100, farm="Solar " + pl.col("farm"))
        .group_by("farm", "size")
        .agg(recall=pl.col("detected").mean())
    )
    domain = [f"Solar {f}" for f in c.TESTABLE]
    base = alt.Chart(t).encode(
        x=alt.X(
            "size:Q",
            title="Injected capacity step (% of capacity, up or down)",
            scale=alt.Scale(domain=[0, 21]),
        ),
        y=alt.Y("recall:Q", title="Share of injected steps found", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "farm:N",
            title=None,
            scale=alt.Scale(domain=domain, range=[FARM_COLOURS[f] for f in c.TESTABLE]),
            legend=alt.Legend(orient="top"),
        ),
    )
    bar = alt.Chart(pl.DataFrame({"y": [0.8], "rule": ["Pass bar (80%)"]})).mark_rule(strokeDash=[5, 4], size=2).encode(y="y:Q", color=alt.Color("rule:N", title=None, scale=alt.Scale(domain=["Pass bar (80%)"], range=["#000000"]), legend=alt.Legend(orient="top")))  # fmt: skip  # ty: ignore[unresolved-attribute]
    (base.mark_line(point=True, strokeWidth=2) + bar).resolve_scale(color="independent").properties(
        width=WIDTH - 90,
        height=HEIGHT,
        title="Steps of 8% are found at four of five farms; Solar D needs 12%",
    ).save(DOCS / "figure_2_recall.svg")


def fig_false_steps(*, penalty: float) -> None:
    """False steps a year against the penalty, worst farm, by placebo kind."""
    t = pl.read_csv(c.DATA_DIR / "placebo_steps.csv").filter(pl.col("set") == "tune")
    t = (
        t.group_by("farm", "kind", "penalty")
        .agg(pl.col("per_year").mean())
        .group_by("kind", "penalty")
        .agg(worst=pl.col("per_year").max())
    )
    t = t.with_columns(
        kind=pl.col("kind").replace_strict(
            {
                "real_days_shuffled": "Real days, shuffled singly",
                "real_blocks_shuffled": "Real days, shuffled in 7-day blocks",
                "base_days_shuffled": "Step-free days, shuffled singly",
                "base_blocks_shuffled": "Step-free days, shuffled in 7-day blocks",
            }
        )
    )
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_GREEN]
    line = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "penalty:Q",
                title="Penalty per step (in units of the daily noise variance)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y("worst:Q", title="False steps a year (worst farm)"),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "Real days, shuffled singly",
                        "Real days, shuffled in 7-day blocks",
                        "Step-free days, shuffled singly",
                        "Step-free days, shuffled in 7-day blocks",
                    ],
                    range=colours,
                ),
                legend=alt.Legend(orient="top", columns=2),
            ),
        )
    )
    bar = alt.Chart(pl.DataFrame({"y": [1.0], "rule": ["Pass bar (1 a year)"]})).mark_rule(strokeDash=[5, 4], size=2).encode(y="y:Q", color=alt.Color("rule:N", title=None, scale=alt.Scale(domain=["Pass bar (1 a year)"], range=["#000000"]), legend=alt.Legend(orient="top")))  # fmt: skip  # ty: ignore[unresolved-attribute]
    (line + bar).resolve_scale(color="independent").properties(
        width=WIDTH - 90,
        height=HEIGHT,
        title=f"A penalty of {penalty:g} keeps false steps within 1 a year on every placebo",
    ).save(DOCS / "figure_3_false_steps.svg")


def real_panel(*, farm: str, levels: pl.DataFrame, steps: pl.DataFrame) -> alt.Chart:
    """One farm's daily index, fitted level, and 90% band."""
    d = levels.filter(pl.col("farm") == farm).with_columns(
        points=pl.lit("Daily index"),
        band=pl.lit("90% interval of the level"),
        fit=pl.lit("Fitted level"),
    )
    domain = ["Daily index", "Fitted level", "90% interval of the level"]
    scale = alt.Scale(domain=domain, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_SKY])
    x = alt.X("years_since_first:Q", title=X_TITLE, scale=alt.Scale(domain=[0, 7.3]))
    pts = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=22, opacity=0.7)
        .encode(
            x=x,
            y=alt.Y("value:Q", title="Output over expectation", scale=Y_SCALE),
            color=alt.Color(
                "points:N", title=None, scale=scale, legend=alt.Legend(orient="top", columns=3)
            ),
        )
    )
    band = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.35)
        .encode(
            x=x,
            y=alt.Y("lo:Q", scale=Y_SCALE, title="Output over expectation"),
            y2="hi:Q",
            color=alt.Color("band:N", title=None, scale=scale),
        )
    )
    fit = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=2.5, interpolate="step-after")
        .encode(
            x=x,
            y=alt.Y("level:Q", scale=Y_SCALE, title="Output over expectation"),
            color=alt.Color("fit:N", title=None, scale=scale),
        )
    )
    return (band + pts + fit).properties(width=WIDTH - 90, height=HEIGHT, title=f"Solar {farm}")


def fig_real(*, pair: tuple[str, str], n: int) -> None:
    """Two real-fit panels in one figure."""
    levels = pl.read_csv(c.DATA_DIR / "real_levels.csv")
    steps = pl.read_csv(c.DATA_DIR / "real_steps.csv")
    panels = [real_panel(farm=f, levels=levels, steps=steps) for f in pair]
    alt.vconcat(*panels).resolve_scale(color="shared").save(
        DOCS / f"figure_{n}_real_{pair[0].lower()}_{pair[1].lower()}.svg"
    )


def fig_example() -> None:
    """Known-answer examples: a 12% fall injected into the step-free series of two farms."""
    farms = ka.load_farms()
    penalty = float((c.DATA_DIR / "chosen_penalty.txt").read_text())
    panels = []
    for farm in ("A", "D"):
        f = farms[farm]
        n = len(f["base"])
        s = n // 2
        truth = np.where(np.arange(n) >= s, 0.88, 1.0)
        y = f["base"] * truth
        _, level, _ = c.fit_levels(values=y, penalty=penalty)
        d = pl.DataFrame(
            {"years": f["years"], "obs": y, "truth": truth, "fit": level}
        ).with_columns(
            o=pl.lit("Daily index + 12% fall"), t=pl.lit("True level"), f=pl.lit("Fitted level")
        )
        domain = ["Daily index + 12% fall", "True level", "Fitted level"]
        scale = alt.Scale(domain=domain, range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE])
        x = alt.X("years:Q", title=X_TITLE, scale=alt.Scale(domain=[0, 7.3]))
        pts = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_circle(size=22, opacity=0.7)
            .encode(
                x=x,
                y=alt.Y("obs:Q", title="Output over expectation", scale=Y_SCALE),
                color=alt.Color(
                    "o:N", title=None, scale=scale, legend=alt.Legend(orient="top", columns=3)
                ),
            )
        )
        tr = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=2.5, strokeDash=[6, 4], interpolate="step-after")
            .encode(
                x=x,
                y=alt.Y("truth:Q", scale=Y_SCALE, title="Output over expectation"),
                color=alt.Color("t:N", title=None, scale=scale),
            )
        )
        ft = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=2.5, interpolate="step-after")
            .encode(
                x=x,
                y=alt.Y("fit:Q", scale=Y_SCALE, title="Output over expectation"),
                color=alt.Color("f:N", title=None, scale=scale),
            )
        )
        panels.append(
            (pts + tr + ft).properties(width=WIDTH - 90, height=HEIGHT, title=f"Solar {farm}")
        )
    alt.vconcat(*panels).resolve_scale(color="shared").save(
        DOCS / "figure_4_known_answer_example.svg"
    )


def main() -> None:
    """Write the tables and every figure."""
    DOCS.mkdir(parents=True, exist_ok=True)
    s = tables()
    fig_sunny_counts()
    fig_recall(penalty=s["penalty"])
    fig_false_steps(penalty=s["penalty"])
    fig_example()
    for n, pair in ((5, ("A", "B")), (6, ("C", "D")), (7, ("E", "F"))):
        fig_real(pair=pair, n=n)


if __name__ == "__main__":
    main()

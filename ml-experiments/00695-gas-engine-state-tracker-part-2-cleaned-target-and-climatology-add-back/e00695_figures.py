# ruff: noqa: E501
"""Experiment 695: draw the write-up's four figures from the scorer's stored tables.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00695-*/e00695_figures.py OUTPUT_DIR

Reads `scores/report.json`, `scores/d8_slices.csv`, and `scores/host_slices.csv` under the
experiment's data folder. Nothing is refitted. No figure carries a date, a series name, or the
injected engine's record: the host appears only as "the host Primary", in aggregate scores.
"""

import json
import sys
from pathlib import Path

import altair as alt
import polars as pl
from e00695_common import DATA_DIR
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, GREY_2

WIDTH = 680
FORECAST = DATA_BLUE
BOUND = BRAND_ORANGE
BOUNDS = {"AH", "A1+state", "AH+state", "WT+truth"}
KIND = {True: "bound, not a forecast", False: "forecast"}
KIND_SCALE = alt.Scale(domain=[KIND[False], KIND[True]], range=[FORECAST, BOUND])


def _kind(*, name: str) -> str:
    """Return the legend label of a system."""
    return KIND[name in BOUNDS]


def figure_1(*, report: dict, out: Path) -> None:
    """CRPS of the D8 forecast, by system, with the bounds in their own colour."""
    order = ["A0", "A1", "A1n", "AH", "A1+state", "AH+state"]
    frame = pl.DataFrame(
        {
            "system": order,
            "crps": [report["mean_crps"][n] for n in order],
            "kind": [_kind(name=n) for n in order],
        }
    )
    base = alt.Chart(frame).encode(y=alt.Y("system:N", sort=order, title=None))
    bars = base.mark_bar(height=18).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "crps:Q",
            title="CRPS (points of capacity; smaller is better)",
            scale=alt.Scale(domain=[0, 8.5]),
            axis=alt.Axis(values=[0, 2, 4, 6, 8]),
        ),
        color=alt.Color("kind:N", scale=KIND_SCALE, title=None, legend=alt.Legend(orient="top")),
    )
    labels = base.mark_text(align="left", dx=4).encode(  # ty: ignore[unresolved-attribute]
        x="crps:Q", text=alt.Text("crps:Q", format=".2f")
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [report["mean_crps"]["A0"]]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=GREY_2)
        .encode(x="x:Q")
    )
    chart = (bars + labels + rule).properties(
        width=WIDTH - 120,
        height=160,
        title=alt.TitleParams(
            "D8's forecast: the cleaned arms are no better than the raw arm",
            subtitle=[
                "Mean CRPS of the forecast of D8's metered apparent power, 2025-11 to 2026-06,",
                "leads 24 to 240 hours, mean of 3 seeds. Dashed line: the raw arm A0.",
                "Orange bars read the scored reading or a hindsight fit, so they are not forecasts.",
            ],
        ),
    )
    chart.save(out / "figure_1_d8_crps_by_system.svg")


def _by_state(*, slices: pl.DataFrame, rows: dict[str, str], order: list[str]) -> pl.DataFrame:
    """Return CRPS by system for the named slices, as long rows with a readable panel label."""
    keep = slices.filter(pl.col("slice").is_in(list(rows)) & pl.col("system").is_in(order))
    return keep.with_columns(
        panel=pl.col("slice").replace(rows),
        kind=pl.col("system").map_elements(lambda n: _kind(name=n), return_dtype=pl.String),
    )


def _state_figure(
    *, frame: pl.DataFrame, order: list[str], title: str, subtitle: list[str]
) -> alt.Chart:
    """Return a two-panel dot chart of CRPS by system in each engine state."""
    base = alt.Chart(frame).encode(y=alt.Y("system:N", sort=order, title=None))
    dots = base.mark_circle(size=120).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("crps:Q", title="CRPS (points of capacity; smaller is better)"),
        color=alt.Color("kind:N", scale=KIND_SCALE, title=None, legend=alt.Legend(orient="top")),
    )
    text = base.mark_text(align="left", dx=9).encode(  # ty: ignore[unresolved-attribute]
        x="crps:Q", text=alt.Text("crps:Q", format=".1f")
    )
    return (
        (dots + text)
        .properties(width=WIDTH - 120, height=110)
        .facet(
            row=alt.Row(
                "panel:N",
                title=None,
                sort=None,
                header=alt.Header(
                    labelOrient="top", labelAngle=0, labelAlign="left", labelPadding=4
                ),
            )
        )
        .properties(title=alt.TitleParams(title, subtitle=subtitle))
    )


def figure_2(*, slices: pl.DataFrame, out: Path) -> None:
    """D8 CRPS on rows where the engine is tracked on and off."""
    order = ["A0", "A1", "A1n", "AH", "A1+state"]
    rows = {
        "tracked on (hindsight M2 flag)": "Engine tracked running at the valid time",
        "tracked off (hindsight M2 flag)": "Engine tracked not running at the valid time",
    }
    frame = _by_state(slices=slices, rows=rows, order=order)
    chart = _state_figure(
        frame=frame,
        order=order,
        title="Adding the calendar average back does not help when the engine runs",
        subtitle=[
            "CRPS at D8 by whether the hindsight tracker flags the engine as running at the valid time.",
            "Only the bound A1+state knows that state when forecasting, which is why it scores best.",
        ],
    )
    chart.save(out / "figure_2_d8_crps_by_engine_state.svg")


def figure_3(*, report: dict, out: Path) -> None:
    """The planned contrasts, with the pre-registered and the t intervals."""
    labels = {
        "c1": "C1 D8: A1 minus A0",
        "c2": "C2 D8, engine months",
        "c3": "C3 D8, fade months",
        "c4": "C4 host: W1 minus W0",
        "c5": "C5 host: WT minus W0",
    }
    rows = []
    for key, label in labels.items():
        r = report[key]
        rows.append(
            {
                "contrast": f"{label} ({r['level']:.1f}%)",
                "estimate": r["estimate"],
                "lo": r["lower"],
                "hi": r["upper"],
                "tlo": r["t_lower"],
                "thi": r["t_upper"],
            }
        )
    frame = pl.DataFrame(rows)
    order = frame["contrast"].to_list()
    base = alt.Chart(frame).encode(y=alt.Y("contrast:N", sort=order, title=None))
    boot = base.mark_rule(strokeWidth=2.5, color=DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="CRPS difference (points of capacity; below zero is a gain)"),
        x2="hi:Q",
    )
    tint = base.mark_rule(strokeWidth=2, color=GREY_2, strokeDash=[5, 3], yOffset=7).encode(  # ty: ignore[unresolved-attribute]
        x="tlo:Q", x2="thi:Q"
    )
    dot = base.mark_circle(size=90, color=DATA_BLUE).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + boot + tint + dot).properties(
        width=WIDTH - 200,
        height=170,
        title=alt.TitleParams(
            "No planned contrast shows a gain from cleaning",
            subtitle=[
                "Dot: estimate. Blue line: interval from resampling whole origin months, one seed per draw.",
                "Grey line: t interval over the monthly differences. Rule at zero: no difference.",
                "C1, C4, and C5 are read at 98.3%; C2 and C3 are descriptive, at 95%. Grey lines are dashed.",
            ],
        ),
    )
    chart.save(out / "figure_3_contrasts.svg")


def figure_4(*, host: pl.DataFrame, out: Path) -> None:
    """Host CRPS by whether the injected engine is truly running."""
    order = ["W0", "W1", "WT", "WT+truth"]
    rows = {
        "engine truly on": "Injected engine running at the valid time",
        "engine truly off": "Injected engine not running at the valid time",
    }
    frame = _by_state(slices=host, rows=rows, order=order)
    chart = _state_figure(
        frame=frame,
        order=order,
        title="At the host, perfect cleaning with a calendar average also gains little",
        subtitle=[
            "CRPS at the host Primary, which has an engine record injected, by whether that engine is running.",
            "Only the bound WT+truth is told the true output at the valid time.",
        ],
    )
    chart.save(out / "figure_4_host_crps_by_engine_state.svg")


def main() -> None:
    """Draw the four figures into the folder named on the command line."""
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    scores = DATA_DIR / "scores"
    report = json.loads((scores / "report.json").read_text())
    figure_1(report=report, out=out)
    figure_2(slices=pl.read_csv(scores / "d8_slices.csv"), out=out)
    figure_3(report=report, out=out)
    figure_4(host=pl.read_csv(scores / "host_slices.csv"), out=out)


if __name__ == "__main__":
    main()

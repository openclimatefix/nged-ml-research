# ruff: noqa: E501
"""Experiment 661: print every number the write-up quotes and draw its figures.

Reads the JSON files written by `e00661_audit.py`, `e00661_step6.py`, `e00661_analyse5.py`, and
`e00661_analyse3.py`. Usage: python e00661_report.py
"""

import json
from pathlib import Path

import altair as alt
import e00661_adapter as ad
import e00661_degrade as g
import polars as pl

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / g.NAME
W = 560
BLUE, ORANGE, GREEN, PURPLE, GREY = "#306BFF", "#FF4901", "#17E58F", "#B701FF", "#808080"


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write an SVG beside the write-up."""
    chart.save(DOCS / f"{name}.svg")


def main() -> None:
    """Print the numbers and draw the figures."""
    s5 = json.loads((g.WORK / "analysis" / "step5_2025-07_2026-03_2026-06.json").read_text())
    s6 = json.loads((g.WORK / "audit" / "step6.json").read_text())
    p3 = json.loads((ad.WORK / "analysis" / "part3.json").read_text())
    deg = pl.read_csv(g.WORK / "audit" / "score_degraded_cams.csv")
    ens = pl.read_csv(g.WORK / "audit" / "score_ens_vs_cams.csv")
    lag = json.loads((g.WORK / "audit" / "lag.json").read_text())
    print(deg.select("window_hours", "method", "rmse_w_m2", "rmse_pct_peak", "peak_bias_pct"))
    print(ens.select("band", "method", "rmse_w_m2", "rmse_pct_peak", "peak_bias_pct", "bias_w_m2"))
    for key, v in lag["ens_vs_cams"].items():
        print(key, v["best_lag_half_hours"], round(v["corr_at_best"], 3), round(v["corr_at_0"], 3))
    # Headline: explicit minus free against input time resolution (positive favours the explicit network).
    rows = []
    for name, v in (
        ("Hourly (native)", "native"),
        ("3-hour windows", "w3"),
        ("6-hour windows", "w6"),
    ):
        r = s5["variants"][v]["sunny_middays"]
        rows.append(
            {
                "input": name,
                "gain": r["free_minus_explicit"],
                "lo": r["lo"],
                "hi": r["hi"],
                "kind": "Known weather (CAMS, ERA5)",
            }
        )
        print(
            v,
            {k: round(x, 3) for k, x in r.items()},
            s5["variants"][v].get("gain_minus_native_gain_sunny"),
        )
    e = s6["by_step_width"]["all leads"]
    rows.append(
        {
            "input": "Ensemble forecast (3-hourly then 6-hourly)",
            "gain": e["free_minus_explicit"],
            "lo": e["lo"],
            "hi": e["hi"],
            "kind": "Ensemble forecast",
        }
    )
    df = pl.DataFrame(rows)
    order = [r["input"] for r in rows]
    base = alt.Chart(df).encode(
        y=alt.Y("input:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    )
    chart = (
        base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Explicit network's CRPS gain over the free network (points of capacity; positive favours the explicit network)",
            ),
            x2="hi:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Known weather (CAMS, ERA5)", "Ensemble forecast"], range=[BLUE, ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        + base.mark_point(filled=True, size=90).encode(  # ty: ignore[unresolved-attribute]
            x="gain:Q",
            color=alt.Color(
                "kind:N",
                legend=None,
                scale=alt.Scale(
                    domain=["Known weather (CAMS, ERA5)", "Ensemble forecast"], range=[BLUE, ORANGE]
                ),
            ),
        )
        + alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    ).properties(width=W, height=160)
    save(chart, "gain_by_input_resolution")
    # Step 6: contrast by lead day.
    d6 = pl.DataFrame(
        [
            {"day": int(k), "gain": v["free_minus_explicit"], "lo": v["lo"], "hi": v["hi"]}
            for k, v in s6["by_lead_day"].items()
        ]
    )
    base = alt.Chart(d6).encode(x=alt.X("day:O", title="Lead day", axis=alt.Axis(labelAngle=0)))
    c6 = (
        base.mark_rule(color=ORANGE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "lo:Q",
                title="Explicit network's CRPS gain over the free network (points; negative means the explicit network is worse)",
            ),
            y2="hi:Q",
        )
        + base.mark_point(color=ORANGE, filled=True, size=70).encode(y="gain:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    ).properties(width=W, height=220)
    save(c6, "ensemble_gain_by_lead_day")
    for k, v in s6["by_step_width"].items():
        print("step6", k, {a: round(b, 3) for a, b in v.items()})
    # Interpolation error by method.
    d2 = deg.select("window_hours", "method", "rmse_pct_peak").with_columns(
        method=pl.col("method").replace(
            {"production": "window mean (production)", "step": "step-hold"}
        ),
        window=pl.col("window_hours").cast(pl.String) + "-hour windows",
    )
    c2 = (
        alt.Chart(d2)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "method:N",
                sort=["window mean (production)", "step-hold", "linear"],
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("rmse_pct_peak:Q", title="Root-mean-square error (% of CAMS midday peak)"),
            color=alt.Color(
                "method:N",
                legend=None,
                scale=alt.Scale(
                    domain=["window mean (production)", "step-hold", "linear"],
                    range=[BLUE, PURPLE, ORANGE],
                ),
            ),
            column=alt.Column("window:N", title=None),
        )
        .properties(width=W // 2 - 30, height=200)
    )
    save(c2, "interpolation_error")
    # Part 3.
    out = []
    names = {
        "b_minus_adapter": "Explicit network (experiment 644)",
        "a_minus_adapter": "Free network (experiment 644)",
        "nested_minus_adapter": "Mixture without the adapter",
    }
    for band in ("sunny_d1-2", "sunny_d3-5", "sunny_d6-10"):
        for k, label in names.items():
            c = p3["contrasts"][band][k]
            out.append(
                {
                    "band": band.replace("sunny_", "day "),
                    "control": label,
                    "diff": c["diff"],
                    "lo": c["lo"],
                    "hi": c["hi"],
                }
            )
            print("part3", band, label, round(c["diff"], 3), round(c["lo"], 3), round(c["hi"], 3))
    d3 = pl.DataFrame(out)
    base = alt.Chart(d3).encode(
        y=alt.Y("control:N", title=None, sort=list(names.values()), axis=alt.Axis(labelLimit=250)),
        color=alt.Color(
            "control:N",
            legend=None,
            scale=alt.Scale(domain=list(names.values()), range=[BLUE, ORANGE, PURPLE]),
        ),
    )
    c3 = (
        (
            base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "lo:Q",
                    title="Control's CRPS minus the adapter arm's (points; positive favours the adapter)",
                ),
                x2="hi:Q",
            )
            + base.mark_point(filled=True, size=70).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
            + alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=GREY, strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        .properties(width=W - 120, height=120)
        .facet(row=alt.Row("band:N", title=None))
    )
    save(c3, "adapter_contrasts")
    print("scores d1-2", p3["scores"]["sunny_d1-2"], "bars", p3["bars"])
    print("tails", json.dumps(p3["tails"], indent=1))
    print("all half hours a-adapter", p3["contrasts"]["all_half_hours"])


if __name__ == "__main__":
    main()

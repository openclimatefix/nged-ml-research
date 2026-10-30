"""Draw the two figures of experiment 417: balanced accuracy by arm, and the diagnostics' AUCs."""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00417_diagnostics as diag
import polars as pl

DOCS: Final[Path] = Path(
    "docs/ml-experiments/00417-hidden-sign-mva-export-gate-zero-crossing-labels"
)
WIDTH: Final[int] = 600
BLUE, ORANGE, GREY = "#306BFF", "#FF4901", "#7F7F7F"


def balanced_accuracy_chart() -> alt.Chart:
    """Draw pooled balanced accuracy of the four arms, real sky against shuffled sky."""
    scores = json.loads((diag.OUT_DIR.parent / "scores.json").read_text())
    rows = []
    for arm, label in (
        ("ref", "S1c reference"),
        ("A", "A: export gate"),
        ("AH", "A + H: hard gate"),
        ("C", "C: zero-crossing labels"),
    ):
        for sky, key in (
            ("Real sky", f"{arm} L2_Q2_s0"),
            ("Shuffled sky", f"{arm}-shuffle L2_Q2_s0"),
        ):
            pooled = next(s for s in scores[key]["scores"] if s["site"] == "pooled")
            rows.append({"arm": label, "sky": sky, "ba": pooled["model"]["balanced_accuracy"]})
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", title=None, sort=None),
        x=alt.X(
            "ba:Q",
            title="Balanced accuracy, hidden Primaries D5 and D10 (larger is better)",
            scale=alt.Scale(domain=[0.4, 1.0]),
        ),
        color=alt.Color(
            "sky:N",
            scale=alt.Scale(domain=["Real sky", "Shuffled sky"], range=[BLUE, GREY]),
            legend=alt.Legend(orient="top", title=None),
        ),
        yOffset="sky:N",
    )
    bar = (
        alt.Chart(pl.DataFrame({"x": [0.85]}))
        .mark_rule(color=ORANGE, strokeDash=[4, 3])
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0.4, 1.0])))
    )
    return (base.mark_circle(size=90, aria=False) + bar).properties(
        width=WIDTH,
        height=200,
        title=alt.TitleParams(
            "No arm reaches the 0.85 bar",
            subtitle=[
                "Dot: one fit on rotation r1; farm injected at 2.0 times the median sunny-midday reading.",  # noqa: E501
                "Test months 2026-03 and 2026-06. Orange rule: the registered bar.",
            ],
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        ),
    )


def auc_chart() -> alt.Chart:
    """Draw the zero-fit diagnostics' areas under the curve."""
    d = json.loads((diag.OUT_DIR / "diagnostics.json").read_text())["L2"]
    flip = json.loads((diag.OUT_DIR / "day_flip.json").read_text())["L2_Q2"]
    rows = [
        {
            "name": "C: W-shape labeller",
            "auc": d["C"]["pooled"]["auc"],
            "lo": d["C"]["pooled"]["lo"],
            "hi": d["C"]["pooled"]["hi"],
        },
        {
            "name": "D: Sunday contrast",
            "auc": d["D"]["pooled"]["auc"],
            "lo": d["D"]["pooled"]["lo"],
            "hi": d["D"]["pooled"]["hi"],
        },
        {
            "name": "E: cloud slope",
            "auc": d["E"]["pooled"]["auc"],
            "lo": d["E"]["pooled"]["lo"],
            "hi": d["E"]["pooled"]["hi"],
        },
        {
            "name": "I: day-level V in u",
            "auc": flip["auc_positive_vs_negative_days"],
            "lo": None,
            "hi": None,
        },
    ]
    frame = pl.DataFrame(
        rows, schema={"name": pl.String, "auc": pl.Float64, "lo": pl.Float64, "hi": pl.Float64}
    )
    base = alt.Chart(frame).encode(y=alt.Y("name:N", title=None, sort=None))
    dots = base.mark_circle(size=90, color=BLUE, aria=False).encode(
        x=alt.X(
            "auc:Q",
            title="Area under the curve, injected against placebo days (larger is better)",
            scale=alt.Scale(domain=[0.4, 1.0]),
        )
    )
    ci = base.mark_rule(color=BLUE).encode(
        x=alt.X("lo:Q", scale=alt.Scale(domain=[0.4, 1.0])), x2="hi:Q"
    )
    bars = (
        alt.Chart(
            pl.DataFrame({"x": [0.8, 0.9], "bar": ["0.8: stop-rule floor", "0.9: day-flip bar"]})
        )
        .mark_rule(strokeDash=[4, 3])
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0.4, 1.0])),
            color=alt.Color(
                "bar:N",
                scale=alt.Scale(range=[ORANGE, "#B701FF"]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    return (ci + dots + bars).properties(
        width=WIDTH,
        height=160,
        title=alt.TitleParams(
            "Only the zero-crossing labeller separates export days from placebo days",
            subtitle=[
                "Level 2.0, six signed Primaries, power factor 1. Dot: estimate. Line: 95% interval from resampling days (no interval for I)."  # noqa: E501
            ],
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        ),
    )


def main() -> None:
    """Write both charts as SVG."""
    for name, chart in (
        ("balanced_accuracy", balanced_accuracy_chart()),
        ("diagnostic_auc", auc_chart()),
    ):
        chart.save(DOCS / f"{name}.svg")


if __name__ == "__main__":
    main()

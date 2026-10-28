# ruff: noqa: E501
"""Draw the figures of experiment 373 from `work/scores.json`."""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf  # noqa: F401  (registers and enables the theme)
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00373-d8-meta-model-weights-worker-forecasts"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
LABELS: Final[dict[str, str]] = {
    "meta": "Meta-model", "meta_nolag": "Meta-model, no lagged power", "meta_nocal": "Meta-model, no calendar",
    "single": "Best single family (out of sample)", "b50": "50/50 blend of GAM stack and BG2",
    "gam": "GAM stack", "bg2": "BG2", "W": "Weekly XGBoost", "M": "Monthly XGBoost", "AN": "Analogue",
    "PT": "Pooled by type", "ST": "XGBoost stack", "climatology": "Climatology", "manual": "Manual heuristic",
}  # fmt: skip
CONTRAST_LABELS: Final[dict[str, str]] = {
    "c_meta__single": "Meta-model minus best single family (C1)",
    "c_meta__b50": "Meta-model minus 50/50 blend (C2)",
    "c_meta__bg2": "Meta-model minus BG2 (C3)",
    "c_meta__meta_nolag": "Meta-model minus no lagged power (C4)",
    "c_meta__meta_nocal": "Meta-model minus no calendar (C5)",
}


def main() -> None:
    """Write the three figures."""
    s = json.loads((WORK / "scores.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "contrast": CONTRAST_LABELS[k],
            "slice": name,
            "point": v[sl]["ci95"][0],
            "low": v[sl]["ci95"][1],
            "high": v[sl]["ci95"][2],
        }
        for k, v in s["contrast"].items()
        if k in CONTRAST_LABELS
        for sl, name in (
            ("all", "All scored rows"),
            ("event_day", "Bank holidays and 24 Dec to 2 Jan"),
            ("other_day", "All other days"),
        )
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("contrast:N", sort=list(CONTRAST_LABELS.values()), title=None)
    )
    chart = (
        (
            base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "low:Q",
                    title="Quantile score difference (points; below zero favours the meta-model)",
                ),
                x2="high:Q",
            )
            + base.mark_point(filled=True, size=60).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
            + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        .properties(width=300, height=130)
        .facet(
            row=alt.Row(
                "slice:N",
                title=None,
                sort=["All scored rows", "Bank holidays and 24 Dec to 2 Jan", "All other days"],
            )
        )
        .properties(
            title=alt.TitleParams(
                "The meta-model does not beat the 50/50 blend by 0.1 points at D8",
                subtitle="95% intervals, whole valid months resampled, 73,331 rows, October 2025 to June 2026",
                anchor="start",
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_contrasts.svg")

    monthly = pl.DataFrame(
        [
            {"month": m, "arm": LABELS[a], "score": v}
            for m, per in s["monthly"].items()
            for a, v in per.items()
            if a in ("meta", "single", "b50", "bg2")
        ]
    )
    alt.Chart(monthly).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("month:N", title="Valid month"),
        y=alt.Y("score:Q", title="Quantile score (points)", scale=alt.Scale(zero=False)),
        color=alt.Color("arm:N", title=None),
    ).properties(
        width=420,
        height=240,
        title=alt.TitleParams(
            "The meta-model tracks the 50/50 blend month by month", anchor="start"
        ),
    ).save(DOCS_DIR / "figure_2_monthly.svg")

    absolute = pl.DataFrame(
        [{"arm": LABELS[a], "score": v} for a, v in s["absolute"]["all"].items()]
    )
    alt.Chart(absolute).mark_bar().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("arm:N", sort="x", title=None),
        x=alt.X(
            "score:Q", title="Quantile score (points; lower is better)", scale=alt.Scale(zero=False)
        ),
    ).properties(
        width=420,
        height=300,
        title=alt.TitleParams(
            "Absolute score of every arm at D8",
            subtitle="73,331 rows, October 2025 to June 2026",
            anchor="start",
        ),
    ).save(DOCS_DIR / "figure_3_absolute.svg")
    print("wrote 3 figures")


if __name__ == "__main__":
    main()

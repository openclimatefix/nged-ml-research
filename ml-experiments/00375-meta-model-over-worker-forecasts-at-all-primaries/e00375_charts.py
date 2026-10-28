# ruff: noqa: E501
"""Draw the figures of experiment 375 from `work/scores.json`."""

import json
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf  # noqa: F401  (registers and enables the theme)
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00375-meta-model-over-worker-forecasts-at-all-primaries"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
STACK: Final = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D14",
    "D16",
    "D19",
    "D21",
    "D22",
)
SLICE_LABELS: Final[dict[str, str]] = {
    "all": "All scored rows",
    "event_day": "Bank holidays and 24 Dec to 2 Jan",
    "other_day": "All other days",
    "cold_winter_night": "Cold winter nights",
    "sunny_apr_jun": "Sunny April to June half-hours",
}
LABELS: Final[dict[str, str]] = {
    "meta": "Meta-model", "blend": "50/50 blend of GAM stack and BG2", "single": "Best single worker (out of sample)",
    "bg2": "BG2", "gam": "GAM stack", "W": "Weekly XGBoost", "M": "Monthly XGBoost", "AN": "Analogue",
    "PT": "Pooled by type", "ST": "XGBoost stack", "climatology": "Climatology", "manual": "Manual heuristic",
}  # fmt: skip


def interval_chart(
    *, rows: list[dict], y_sort: list[str], title: str, subtitle: str, height: int
) -> alt.Chart:
    """Return a dot-and-interval chart of contrasts against a dashed zero line."""
    base = alt.Chart(pl.DataFrame(rows)).encode(y=alt.Y("label:N", sort=y_sort, title=None))
    return (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q",
                title="Quantile score difference (points; below zero favours the meta-model)",
            ),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=60).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=380,
        height=height,
        title=alt.TitleParams(title, subtitle=subtitle, anchor="start"),
    )


def main() -> None:
    """Write the three figures."""
    s = json.loads((WORK / "scores.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    c1 = s["contrast"]["C1_meta_minus_blend"]["all"]["stack_sites"]
    rows = [
        {
            "label": "Pooled over 14 Primaries",
            "point": c1["pooled"][0],
            "low": c1["pooled"][1],
            "high": c1["pooled"][2],
        }
    ] + [
        {"label": site, "point": p, "low": lo, "high": hi}
        for site, (p, lo, hi) in zip(c1["sites"], c1["site"], strict=True)
    ]
    interval_chart(
        rows=rows,
        y_sort=[r["label"] for r in rows],
        title="The meta-model scores worse than the 50/50 blend, pooled and at 10 of 14 Primaries",
        subtitle="Meta-model minus 50/50 blend (C1), 95% intervals, whole valid months resampled, October 2025 to June 2026",
        height=300,
    ).save(DOCS_DIR / "figure_1_by_primary.svg")

    rows = []
    for contrast, name in (
        ("C1_meta_minus_blend", "Minus 50/50 blend (C1)"),
        ("C2_meta_minus_best_single", "Minus best single worker (C2)"),
        ("C3_meta_minus_bg2", "Minus BG2 (C3)"),
    ):
        for flag, flag_name in SLICE_LABELS.items():
            p, lo, hi = s["contrast"][contrast][flag]["stack_sites"]["pooled"]
            rows.append({"label": f"{name}, {flag_name}", "point": p, "low": lo, "high": hi})
    interval_chart(
        rows=rows,
        y_sort=[r["label"] for r in rows],
        title="The meta-model beats the blend, with an interval below zero, only in sunny April to June half-hours",
        subtitle="Pooled over the 14 Primaries with a stored GAM stack, 95% intervals",
        height=360,
    ).save(DOCS_DIR / "figure_2_by_slice.svg")

    absolute = pl.DataFrame(
        [
            {"arm": LABELS[a], "score": float(np.mean([s["absolute"][site][a] for site in STACK]))}
            for a in LABELS
        ]
    )
    alt.Chart(absolute).mark_bar().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("arm:N", sort="x", title=None),
        x=alt.X(
            "score:Q", title="Quantile score (points; lower is better)", scale=alt.Scale(zero=False)
        ),
    ).properties(
        width=380,
        height=300,
        title=alt.TitleParams(
            "The 50/50 blend is the best arm on average over the 14 Primaries",
            subtitle="Mean of the 14 Primaries' scores, October 2025 to June 2026",
            anchor="start",
        ),
    ).save(DOCS_DIR / "figure_3_absolute.svg")
    print("wrote 3 figures")


if __name__ == "__main__":
    main()

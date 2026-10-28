# ruff: noqa: E501
"""Draw the figures of experiment 368 from `work/scores.json`.

uv run --no-sync python ml-experiments/00368-*/e00368_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00368-d8-conditional-band-head-on-gam-stack-residuals"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 480
LABELS: Final[dict[str, str]] = {
    "stored": "Stack, donated band",
    "arm3": "Control: own-residual band by hour and season (354)",
    "x": "XGBoost quantile head",
    "m": "Mixture density network head",
    "xshuf": "Placebo: XGBoost head, shuffled features",
    "mshuf": "Placebo: mixture head, shuffled features",
    "bg2": "BG2",
    "manual": "Manual heuristic",
    "climatology": "Climatology",
}


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def _title(text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def figure_1(*, scores: dict) -> None:
    """D8's quantile score by arm."""
    absolute = scores["sites"]["D8"]["absolute"]["all"]
    frame = pl.DataFrame(
        [{"arm": LABELS[k], "score": absolute[k]} for k in LABELS if k in absolute]
    )
    order = frame["arm"].to_list()
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=400)),
            x=alt.X("score:Q", title="Quantile score at D8 (points of capacity, lower is better)"),
        )
        + alt.Chart(frame)
        .mark_text(align="left", dx=3)
        .encode(y=alt.Y("arm:N", sort=order), x="score:Q", text=alt.Text("score:Q", format=".2f"))  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=260,
        title=_title(
            "The XGBoost head closes 22% of the control's gap to BG2 at D8",
            "Scored year 2025-07-02 to 2026-06-30, 96,986 rows, leads 0.5 to 168 hours.",
        ),
    )
    _save(chart, name="figure_1_d8_scores")


def figure_2(*, scores: dict) -> None:
    """Change in score against the stored forecast at the other Primaries, with 99% intervals."""
    rows = []
    for head, group in (("x", "XGBoost head"), ("m", "Mixture head")):
        contrasts = scores["contrasts"][head]
        for key, label in (
            ("D2_vs_stored", "D2"),
            ("D4_vs_stored", "D4"),
            ("others_vs_stored", "Seven other Primaries, pooled"),
        ):
            point, low, high = contrasts[key]["all"]["ci99"]
            rows.append(
                {"label": label, "group": group, "estimate": point, "low": low, "high": high}
            )
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=["D2", "D4", "Seven other Primaries, pooled"], title=None),
        color=alt.Color(
            "group:N",
            legend=alt.Legend(title=None, orient="top"),
            scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        ),
        yOffset="group:N",
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q", title="Score minus the stored forecast's (points, negative is better)"
            ),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=45).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=160,
        title=_title(
            "The mixture head leaves other Primaries unchanged or slightly better; the XGBoost head does not",
            "Paired, month-resampled 99% intervals.",
        ),
    )
    _save(chart, name="figure_2_other_primaries")


def figure_3(*, scores: dict) -> None:
    """Tail shares at D8."""
    tails = scores["sites"]["D8"]["tails"]["all"]
    rows = [
        {"arm": LABELS[a], "tail": t, "share": 100 * tails[a][k]}
        for a in ("arm3", "x", "m", "bg2")
        for t, k in (
            ("below q01 (target 1%)", "below_q01"),
            ("below q05 (5%)", "below_q05"),
            ("above q95 (5%)", "above_q95"),
            ("above q99 (1%)", "above_q99"),
        )
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("tail:N", title=None, sort=None),
            y=alt.Y("share:Q", title="Share of readings (%)"),
            xOffset="arm:N",
            color=alt.Color(
                "arm:N",
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, "#8C8C8C"]),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                "The heads' lower score comes with worse tails at D8",
                "BG2's tails are the benchmark; the shares are of readings beyond each stored quantile.",
            ),
        )
    )
    _save(chart, name="figure_3_d8_tails")


def main() -> None:
    """Draw every figure."""
    scores = json.loads((WORK / "scores.json").read_text())
    figure_1(scores=scores)
    figure_2(scores=scores)
    figure_3(scores=scores)


if __name__ == "__main__":
    main()

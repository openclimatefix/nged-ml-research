"""Print every number quoted in experiment 374's write-up and draw its two figures.

Reads `work/scores.json` of this experiment and of experiment 372 (the control).

    uv run --no-sync python ml-experiments/00374-*/e00374_report.py
"""

import json
from pathlib import Path
from statistics import mean
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-stage-two-at-eight-primaries"
SLUG_372: Final[str] = "mva-fold-rule-chosen-fit-at-all-primaries"
SCORES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00374-{SLUG}" / "work" / "scores.json"
SCORES_372: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00372-{SLUG_372}" / "work" / "scores.json"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00374-{SLUG}"
)
SITES: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
CONTROL_TOLERANCE: Final[float] = 0.05
SERIES: Final[dict[str, str]] = {
    "chosen2": "Chosen fit + stage 2",
    "a1": "A1 (experiment 252)",
    "bg2": "BG2",
    "manual": "Manual heuristic",
    "climatology": "Climatology",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _fmt(triple: list[float]) -> str:
    """Format a point estimate and its interval."""
    point, low, high = triple
    return f"{point:+.2f} ({low:+.2f} to {high:+.2f})"


def print_report(*, scores: dict, control: dict) -> None:
    """Print the control gate, then the scores, contrasts, tails, and checks."""
    print("== control gate: re-run chosen_raw vs experiment 372 chosen; references vs 372")
    worst = 0.0
    for site in SITES:
        raw = scores["absolute"][site]["all"]["chosen_raw"]
        old = control["absolute"][site]["all"]["chosen"]
        diffs = {
            name: scores["absolute"][site]["all"][name] - control["absolute"][site]["all"][name]
            for name in ("bg2", "climatology", "manual")
        }
        worst = max(worst, abs(raw - old), *(abs(v) for v in diffs.values()))
        print(f"{site}: chosen_raw {raw:.3f} vs 372 {old:.3f} ({raw - old:+.4f}); refs {diffs}")
    print(f"worst absolute difference {worst:.4f}; gate passes: {worst <= CONTROL_TOLERANCE}")
    print_scores(scores=scores)


def print_scores(*, scores: dict) -> None:  # noqa: PLR0912
    """Print absolute scores, contrasts, tails, and checks."""
    print("== absolute scores, all half-hours")
    for site in SITES:
        row = scores["absolute"][site]["all"]
        print(site, {k: round(v, 2) for k, v in row.items()})
    print("== sunny / winter_night / cold_night absolute, pooled mean of the eight")
    for flag in ("sunny", "winter_night", "cold_night"):
        print(
            flag,
            {
                name: round(mean(scores["absolute"][s][flag][name] for s in SITES), 2)
                for name in ("a1", "chosen2", "chosen_raw", "bg2", "climatology", "manual")
            },
        )
    print("== contrasts, pooled over the eight")
    for column in ("c1", "c2", "c3", "c4", "a1_bg2"):
        for flag, result in scores["contrast"][column].items():
            print(column, flag, _fmt(result["pooled"]))
    print("== contrasts per Primary, all half-hours")
    for column in ("c1", "c3", "c4", "a1_bg2"):
        sites = scores["contrast"][column]["all"]["sites"]
        for site, triple in zip(sites, scores["contrast"][column]["all"]["site"], strict=True):
            print(column, site, _fmt(triple))
    print("== D12 only")
    for column, by_flag in scores["contrast_d12"].items():
        for flag, result in by_flag.items():
            print(column, flag, _fmt(result["pooled"]))
    print("== leave-one-out of c1 and c3, all half-hours, pooled over the other seven")
    for column, by_site in scores["contrast_leave_one_out"].items():
        for left_out, triple in by_site.items():
            print(column, "without", left_out, _fmt(triple))
    print("== tails: share of readings at or below the quantile, mean of the eight")
    for flag in ("all", "sunny", "cold_night"):
        for arm in ("a1", "chosen2", "bg2"):
            print(
                flag,
                arm,
                {
                    q: round(mean(scores["tails"][s][flag][arm][q] for s in SITES), 3)
                    for q in ("q01", "q05", "q95", "q99")
                },
            )
    print("== median minus mean forecast, per Primary")
    for site in SITES:
        print(
            site, {k: round(v, 3) for k, v in scores["checks"][site]["median_minus_mean"].items()}
        )
    print("== mean-forecast MAE, per Primary (MW)")
    for site in SITES:
        print(
            site, {k: round(v, 2) for k, v in scores["checks"][site]["mae_mean_forecast"].items()}
        )
    print("== rows", scores["rows"])


def draw_figures(*, scores: dict) -> None:
    """Draw the absolute-score figure and the contrast figure."""
    rows = [
        {"site": s, "series": label, "score": scores["absolute"][s]["all"][name]}
        for s in SITES
        for name, label in SERIES.items()
    ]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (
        _chart(pl.DataFrame(rows).to_pandas())
        .mark_circle(size=80)
        .encode(
            x=alt.X("site:N", sort=SITES, title="MVA-only Primary", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "score:Q",
                title="Quantile score, all half-hours (points)",
                scale=alt.Scale(type="symlog", constant=2),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=list(SERIES.values()),
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_BLUE_LIGHT,
                        ocf.DATA_AMBER,
                        ocf.DATA_PURPLE,
                    ],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
        .properties(width=560, height=260)
        .save(DOCS_DIR / "figure_1_score_by_primary.svg")
    )
    contrasts = []
    for column, label in (
        ("c1", "Chosen + stage 2 minus A1"),
        ("c3", "Chosen + stage 2 minus BG2"),
    ):
        sites = scores["contrast"][column]["all"]["sites"]
        for site, (point, low, high) in zip(
            sites, scores["contrast"][column]["all"]["site"], strict=True
        ):
            contrasts.append(
                {"site": site, "contrast": label, "point": point, "low": low, "high": high}
            )
    base = _chart(pl.DataFrame(contrasts).to_pandas()).encode(
        x=alt.X("site:N", sort=SITES, title="MVA-only Primary", axis=alt.Axis(labelAngle=0)),
        xOffset="contrast:N",
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(
                domain=["Chosen + stage 2 minus A1", "Chosen + stage 2 minus BG2"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title=None, orient="top", direction="vertical"),
        ),
    )
    interval = base.mark_rule().encode(
        y=alt.Y("low:Q", title="Difference in quantile score, all half-hours (points)"), y2="high:Q"
    )
    point = base.mark_circle(size=80).encode(y="point:Q")
    zero = (
        _chart(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    )
    (interval + point + zero).properties(width=560, height=260).save(
        DOCS_DIR / "figure_2_contrasts_by_primary.svg"
    )


def main() -> None:
    """Print the report, then draw the figures."""
    scores = json.loads(SCORES.read_text())
    print_report(scores=scores, control=json.loads(SCORES_372.read_text()))
    draw_figures(scores=scores)


if __name__ == "__main__":
    main()

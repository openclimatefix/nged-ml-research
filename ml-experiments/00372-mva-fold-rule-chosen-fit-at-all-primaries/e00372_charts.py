"""Draw the two figures of experiment 372 from `work/scores.json`.

uv run --no-sync python ml-experiments/00372-*/e00372_charts.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-rule-chosen-fit-at-all-primaries"
SCORES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00372-{SLUG}" / "work" / "scores.json"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00372-{SLUG}"
)
MVA_SITES: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
SITES: Final[list[str]] = [
    "D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22",
    "D5", "D9", "D10", "D11", "D14", "D16", "D15", "D18",
]  # fmt: skip
SERIES: Final[dict[str, str]] = {
    "chosen": "Chosen fit (MVA-only Primaries)",
    "bg2": "BG2",
    "manual": "Manual heuristic",
    "climatology": "Climatology",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def main() -> None:
    """Draw the absolute-score figure and the chosen-minus-BG2 figure."""
    scores = json.loads(SCORES.read_text())
    rows = [
        {"site": site, "series": label, "score": scores["absolute"][site]["all"][name]}
        for site in SITES
        for name, label in SERIES.items()
        if name in scores["absolute"][site]["all"]
    ]
    absolute = (
        _chart(pl.DataFrame(rows).to_pandas())
        .mark_circle(size=80)
        .encode(
            x=alt.X("site:N", sort=SITES, title="Primary substation", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "score:Q",
                title="Quantile score, all half-hours (points)",
                scale=alt.Scale(type="symlog", constant=2),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=list(SERIES.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_AMBER, ocf.DATA_BLUE_LIGHT],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
        .properties(width=560, height=260)
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    absolute.save(DOCS_DIR / "figure_1_score_by_primary.svg")
    sites = scores["contrast"]["c3"]["all"]["site"]
    gaps = pl.DataFrame(
        [
            {"site": s, "point": p, "low": lo, "high": hi}
            for s, (p, lo, hi) in zip(MVA_SITES, sites, strict=True)
        ]
    ).to_pandas()
    base = _chart(gaps).encode(x=alt.X("site:N", sort=MVA_SITES, title="MVA-only Primary"))
    interval = base.mark_rule().encode(
        y=alt.Y("low:Q", title="Chosen fit minus BG2, all half-hours (points)"), y2="high:Q"
    )
    point = base.mark_circle(size=80, color=ocf.BRAND_ORANGE).encode(y="point:Q")
    zero = (
        _chart(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    )
    (interval + point + zero).properties(width=560, height=260).save(
        DOCS_DIR / "figure_2_chosen_minus_bg2.svg"
    )


if __name__ == "__main__":
    main()

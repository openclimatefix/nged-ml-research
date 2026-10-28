"""Draw the figures of experiment 367 from `work/scores.json`."""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00367-d8-kalman-transient-off-and-adaptive-short-lead"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
ARM_LABELS: Final[dict[str, str]] = {
    "K0-control": "K0: transient off",
    "KA-control": "KA: transient scaled by own recent benefit",
}


def contrast_figure(*, scores: dict, band: str, name: str, title: str) -> None:
    """Write a dot-and-interval figure of each arm minus the control, per substation."""
    rows = []
    for site in ("D8", "D2", "D4"):
        for key, label in ARM_LABELS.items():
            v = scores[site]["slices"][band]["contrasts"][key]["score"]
            rows.append({"site": site, "arm": label, "value": v[0], "low": v[1], "high": v[2]})
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y("site:N", sort=["D8", "D2", "D4"], title=None),
        yOffset="arm:N",
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=list(ARM_LABELS.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(orient="top", title=None, direction="vertical"),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    layers = [
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q", title="Quantile score minus the stored stack's (points; below 0 is better)"
            ),
            x2="high:Q",
        ),
        base.mark_point(filled=True, size=70).encode(x="value:Q"),  # ty: ignore[unresolved-attribute]
        zero,
    ]
    chart = alt.layer(*layers).properties(width=CONTENT_WIDTH_PX - 200, height=120, title=title)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")


def main() -> None:
    """Draw both figures."""
    scores = json.loads((WORK / "scores.json").read_text())
    contrast_figure(scores=scores, band="0-24h", name="figure_1_short_lead_contrasts",
                    title="Leads of 0 to 24 hours")  # fmt: skip
    contrast_figure(scores=scores, band="all", name="figure_2_all_lead_contrasts",
                    title="All leads, 0 to 168 hours")  # fmt: skip


if __name__ == "__main__":
    main()

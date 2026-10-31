# ruff: noqa: E501
"""Draw the three figures of experiment 399's write-up from the tables `e00399_score.py` stores.

uv run --no-sync python ml-experiments/00399-*/e00399_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FOLDER: Final[str] = "00399-xgboost-short-history-start-cut-d15-d18"
SCORES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / FOLDER / "work" / "scores"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / FOLDER
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
CUTS: Final[list[str]] = ["N3", "N6", "NS", "NC", "NCZ", "NCR"]
ARM_NAMES: Final[dict[str, str]] = {
    "N3": "N3: first 3 months cut",
    "N6": "N6: first 6 months cut",
    "NS": "NS: settling rule",
    "NC": "NC: cut to 2025-03-01",
    "NCZ": "NCZ: NC, near-zero readings removed",
    "NCR": "NCR: NC, runs of near-zero removed",
}
PRIMARY_COLOURS: Final[dict[str, str]] = {"D15": ocf.DATA_BLUE, "D18": ocf.BRAND_ORANGE}


def titled(*, title: str, subtitle: str) -> alt.TitleParams:
    """Return a title with the title and subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=textwrap.wrap(title, 80), subtitle=textwrap.wrap(subtitle, WRAP), anchor="start"
    )


def change_chart() -> None:
    """Write `change.svg`: each arm's days 1 to 10 CRPS relative to the refitted N0."""
    head = pl.read_parquet(SCORES / "headline_all.parquet").filter(pl.col("metric") == "crps")
    wide = head.pivot(on="arm", index="label", values="value")
    rows = [
        {"label": r["label"], "arm": ARM_NAMES[a], "change": 100 * (r[a] / r["N0"] - 1)}
        for r in wide.iter_rows(named=True)
        for a in CUTS
    ]
    frame = pl.DataFrame(rows)
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    dots = (
        alt.Chart(frame)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("change:Q", title="CRPS change against N0 (%; negative is better)"),
            y=alt.Y("arm:N", sort=list(ARM_NAMES.values()), title=None),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=list(PRIMARY_COLOURS), range=list(PRIMARY_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    (zero + dots).properties(
        width=WIDTH - 260,
        height=26 * len(ARM_NAMES),
        title=titled(
            title="Cutting 6 months helps D15 and hurts D18; the cut to 2025-03-01 helps a little at both",
            subtitle="CRPS at days 1 to 10 of each cut, relative to the same recipe trained on every row "
            "(N0), as a percentage of N0. D15: 4 scored months. D18: 6 scored months. Each dot is a "
            "point estimate with no interval, because so few months are scored.",
        ),
    ).save(DOCS_DIR / "change.svg")


def monthly_chart() -> None:
    """Write `monthly.svg`: CRPS minus N0's by valid-time month for three cuts."""
    month = pl.read_parquet(SCORES / "monthly_all.parquet").filter(
        ~pl.col("arm").str.ends_with("_w115") & (pl.col("arm") != "N0_stored")
    )
    wide = month.pivot(on="arm", index=["label", "month"], values="crps")
    arms = ["N3", "N6", "NC"]
    frame = pl.concat(
        [
            wide.select(
                "label", "month", pl.lit(ARM_NAMES[a]).alias("arm"), diff=pl.col(a) - pl.col("N0")
            )
            for a in arms
        ]
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y="x:Q")  # ty: ignore[unresolved-attribute]
    )
    colours = {
        ARM_NAMES["N3"]: ocf.DATA_SKY,
        ARM_NAMES["N6"]: ocf.DATA_PURPLE,
        ARM_NAMES["NC"]: ocf.BRAND_ORANGE,
    }
    panels = []
    for label in ("D15", "D18"):
        part = frame.filter(pl.col("label") == label)
        lines = (
            alt.Chart(part)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=50))
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "month:T",
                    title="Valid-time month" if label == "D18" else None,
                    axis=alt.Axis(format="%b %Y", tickCount="month", labelAngle=0),
                ),
                y=alt.Y("diff:Q", title=f"{label}: CRPS minus N0 (points)"),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                    legend=alt.Legend(title=None, orient="top", columns=1),
                ),
            )
        )
        panels.append(alt.layer(zero, lines).properties(width=WIDTH - 90, height=120))
    alt.vconcat(*panels).resolve_scale(color="shared").properties(
        title=titled(
            title="At D18 the gain from cutting comes mainly from January 2026; at D15 it is largest in March",
            subtitle="CRPS at days 1 to 10 of three cuts minus the same recipe trained on every row "
            "(N0), by month of the valid time. Negative is better. Each month has one fit per cut.",
        )
    ).save(DOCS_DIR / "monthly.svg")


def bias_chart() -> None:
    """Write `bias.svg`: D15's signed median bias for the cuts and the references."""
    bias = pl.read_parquet(SCORES / "bias_all.parquet").filter(
        (pl.col("label") == "D15")
        & ~pl.col("arm").str.ends_with("_w115")
        & (pl.col("arm") != "N0_stored")
    )
    names = {
        "N0": "N0: all rows",
        "N3": "N3",
        "N6": "N6",
        "NC": "NC",
        "stack": "GAM stack",
        "bg2": "BG2",
        "manual_heuristic": "Manual heuristic",
        "climatology": "Climatology",
    }
    frame = bias.filter(pl.col("arm").is_in(list(names))).with_columns(
        arm=pl.col("arm").replace_strict(names)
    )
    xgboost = ["N0: all rows", "N3", "N6", "NC"]
    frame = frame.with_columns(
        kind=pl.when(pl.col("arm").is_in(xgboost))
        .then(pl.lit("XGBoost cut"))
        .otherwise(pl.lit("Reference"))
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("bias:Q", title="Signed bias of the median forecast (points of capacity)"),
            y=alt.Y("arm:N", sort=list(names.values()), title=None),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["XGBoost cut", "Reference"], range=[ocf.DATA_BLUE, ocf.DATA_SKY]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    (zero + bars).properties(
        width=WIDTH - 160,
        height=26 * len(names),
        title=titled(
            title="At D15 every XGBoost forecast sits about 16 to 21 points of capacity too low",
            subtitle="Forecast median minus actual, averaged over days 1 to 10 of the 4 scored months. "
            "Zero is no bias; negative means the forecast is too low. A start cut moves the "
            "bias by at most 5 points; the GAM stack's bias is -3 points.",
        ),
    ).save(DOCS_DIR / "bias.svg")


def main() -> None:
    """Write the three SVG files."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    change_chart()
    monthly_chart()
    bias_chart()


if __name__ == "__main__":
    main()

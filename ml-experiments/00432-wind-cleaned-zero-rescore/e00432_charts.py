# ruff: noqa: E501, ISC004
"""Draw experiment 00432's figures as SVG from the tables of `e00432_rescore.py`.

Shows normalised, aggregated statistics only: no farm time series is drawn. Afterwards optimise
with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00432-*/e00432_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00432-wind-cleaned-zero-rescore" / "work"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00432-wind-cleaned-zero-rescore"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "3 wind farms, one scored year, true days 1 to 10 (24 to 240 hours)."
ARM_LABELS: Final[dict[str, str]] = {
    "x1_336": "Zero below cut-in (experiment 336, x1)",
    "head_x1": "Mixture head (339, x1)",
    "head_x2": "Mixture head plus idle rule (339, x2)",
    "head_x3": "Mixture head, all half-hours (339, x3)",
    "xgb_ens_mean_M": "XGBoost on ensemble mean (343, M)",
    "plant_head_y2": "Plant plus mixture head (347, y2)",
}
ROW_SET_LABELS: Final[dict[str, str]] = {
    "raw": "Raw",
    "masked": "Masked",
    "cleaned_zero": "Cleaned zero",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def headline() -> alt.TopLevelMixin:
    """Each forecast's score minus the default's, on three row sets, with 95% intervals."""
    summary = pl.read_parquet(WORK_DIR / "summary.parquet").filter(
        (pl.col("farm") == "pooled")
        & (pl.col("band") == "days_1_to_10")
        & (pl.col("metric") == "qs")
        & pl.col("row_set").is_in(list(ROW_SET_LABELS))
        & (pl.col("arm") != "default")
    )
    data = summary.with_columns(
        arm_label=pl.col("arm").replace_strict(ARM_LABELS),
        row_set_label=pl.col("row_set").replace_strict(ROW_SET_LABELS),
    ).to_pandas()
    base = alt.Chart(data).encode(
        y=alt.Y(
            "arm_label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=320)
        ),
        color=alt.Color(
            "row_set_label:N",
            sort=list(ROW_SET_LABELS.values()),
            scale=alt.Scale(domain=list(ROW_SET_LABELS.values()), range=COLOURS),
            legend=alt.Legend(title="Row set", orient="top"),
        ),
        yOffset=alt.YOffset("row_set_label:N", sort=list(ROW_SET_LABELS.values())),
    )
    rule = alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (
        (
            rule
            + base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "diff_lo:Q",
                    title="Quantile score minus the default's (points of capacity)",
                    scale=alt.Scale(domain=[-0.6, 0.45]),
                ),
                x2="diff_hi:Q",
            )
            + base.mark_point(filled=True, size=40).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
        )
        .properties(
            width=WIDTH,
            height=260,
            title=_title(
                text="Scoring idle half-hours as zero barely changes any forecast against the default",
                lines=[
                    "Dot: pooled difference. Line: 95% interval from resampling the 12 test months.",
                    "Below zero is better than the default (Wind farms research default 1). "
                    + SCOPE,
                ],
            ),
        )
        .configure_view(stroke=None)
    )


def dropped_share() -> alt.TopLevelMixin:
    """G2: the share of scored half-hours that the experiment 210 mask drops."""
    panels = []
    for name, title, values in (
        ("wind_bin", "ERA5 100 m wind speed (m/s, 1 m/s bins)", None),
        ("hour", "Hour of day (UTC)", [0, 6, 12, 18, 24]),
        ("weekday", "Day of week (1 = Monday)", None),
    ):
        table = pl.read_parquet(WORK_DIR / f"dropped_share_{name}.parquet").filter(
            pl.col("n") >= 200
        )
        axis = alt.Axis(values=values) if values else alt.Axis()
        panels.append(
            alt.Chart(table.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_bar(color=ocf.DATA_BLUE)
            .encode(
                x=alt.X(f"{name}:O" if name != "hour" else f"{name}:Q", title=title, axis=axis),
                y=alt.Y("share_dropped:Q", title="Share dropped", scale=alt.Scale(domain=[0, 1])),
            )
            .properties(width=WIDTH, height=110)
        )
    month = pl.read_parquet(WORK_DIR / "dropped_share_month.parquet")
    panels.append(
        alt.Chart(month.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("month:O", title="Month of the scored year"),
            y=alt.Y("share_dropped:Q", title="Share dropped", scale=alt.Scale(domain=[0, 1])),
        )
        .properties(width=WIDTH, height=110)
    )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The mask drops nearly every half-hour below 3 m/s and few above it",
            lines=[
                "Share of scored half-hours that the experiment 210 mask drops, 3 wind farms pooled, "
                "one scored year. Wind-speed bins with under 200 half-hours are omitted."
            ],
        )
    )


def placebo() -> alt.TopLevelMixin:
    """G3: the mask's effect on the default's score against two sets of random drops."""
    frames = []
    for name, label in (
        ("farm_month_hour", "Placebo matching farm, month, and hour"),
        ("farm_month_hour_wind", "Placebo also matching wind-speed bin"),
    ):
        table = (
            pl.read_parquet(WORK_DIR / f"placebo_{name}.parquet")
            .filter(pl.col("arm") == "default")
            .with_columns(effect=pl.col("kept") - pl.col("all"), placebo=pl.lit(label))
        )
        frames.append(table)
    both = pl.concat(frames)
    draws = both.filter(pl.col("draw") >= 0).to_pandas()
    mask_value = float(both.filter(pl.col("draw") == -1)["effect"][0])
    marks = alt.Chart(pl.DataFrame({"effect": [mask_value]}).to_pandas()).mark_rule(
        color=ocf.BRAND_ORANGE, size=2
    )
    return alt.vconcat(
        *[
            alt.layer(
                alt.Chart(draws[draws["placebo"] == label])  # ty: ignore[unresolved-attribute]
                .mark_bar(color=GREY)
                .encode(
                    x=alt.X(
                        "effect:Q",
                        bin=alt.Bin(step=0.01),
                        title="Change in the default's score (points; positive = removed rows were easier)",
                        scale=alt.Scale(domain=[0, 0.6]),
                    ),
                    y=alt.Y("count():Q", title="Random drops"),
                ),
                marks.encode(x=alt.X("effect:Q", scale=alt.Scale(domain=[0, 0.6]))),  # ty: ignore[unresolved-attribute]
            ).properties(width=WIDTH, height=110, title=label)
            for label in sorted(draws["placebo"].unique())
        ]
    ).properties(
        title=_title(
            text="The mask removes easy half-hours, and the wind speed explains all of the effect",
            lines=[
                "Grey: 200 random drops with the mask's count per stratum. Orange line: the "
                "experiment 210 mask. " + SCOPE
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline().save(DOCS_DIR / "headline.svg")
    dropped_share().save(DOCS_DIR / "dropped_share.svg")
    placebo().save(DOCS_DIR / "placebo.svg")


if __name__ == "__main__":
    main()

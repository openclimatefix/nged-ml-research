# ruff: noqa: E501
"""Figures of experiment 524 (run `npx svgo@4 --multipass --precision=1 --final-newline` on each SVG).

Figure 1: coarse-band CRPS contrasts against L-Ww for the three counted arms. Figure 2: absolute CRPS
by arm. Figures 3 and 4: a day-ahead and a 10-day forecast of L-Ww, S2-F-B-nll, and S2-J-0.4x.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00524_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator"
)
SCORES: Final[Path] = score.WORK_DIR / "scores"
WIDTH: Final[int] = 560
EXAMPLE_ORIGIN: Final[str] = "2026-01-12"
EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}
COLOURS: Final[dict[str, str]] = {
    "L-Ww": theme.DATA_GREEN, "S2-F-B-nll": theme.BRAND_ORANGE, "S2-J-0.4x": theme.DATA_BLUE,
}  # fmt: skip
LABEL: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}


def figures() -> None:
    """Draw figures 1 and 2."""
    table = pl.read_csv(SCORES / "band_contrasts.csv").filter(
        pl.col("counted") & (pl.col("second") == "L-Ww")
    )
    table = table.with_columns(group=pl.col("group").replace_strict(LABEL)).to_pandas()
    order = ["under 48 h", "48-120 h", "120-240 h"]
    panels = []
    for group in LABEL.values():
        part = table[table["group"] == group]
        base = alt.Chart(part).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("band:N", sort=order, title="Lead band"),
            xOffset="first:N",
            color=alt.Color(
                "first:N",
                title="Arm",
                scale=alt.Scale(range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_GREEN]),
            ),
        )  # ty: ignore[unresolved-attribute]
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low_bonferroni18:Q", title="CRPS minus L-Ww"), y2="high_bonferroni18:Q"
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=60).encode(y="difference:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=150, title=alt.TitleParams(group, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "CRPS of each two-stage arm minus L-Ww (points of capacity; lower is better), five folds, two seeds,",
                "Bonferroni intervals for 18 tests (origin-day blocks). Interim and unaudited.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_1_headline.svg")
    absolute = (
        pl.read_csv(SCORES / "absolute.csv")
        .filter(pl.col("slice").is_in(["eight", "six_clean"]) & (pl.col("rowset") == "main16"))
        .with_columns(
            group=pl.col("slice").replace_strict(
                {"eight": LABEL["eight"], "six_clean": LABEL["six"]}
            )
        )
        .to_pandas()
    )
    alt.Chart(absolute).mark_bar(color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("arm:N", sort="x", title=None),
        x=alt.X("crps:Q", title="CRPS (points of capacity)"),
        row=alt.Row("group:N", title=None),
    ).properties(
        width=WIDTH,
        height=220,
        title=alt.TitleParams(
            "Absolute CRPS of every arm, five folds, two seeds, widened 1.15", anchor="start"
        ),
    ).save(DOCS_DIR / "figure_2_absolute.svg")


def examples() -> None:
    """Draw a day-ahead and a 10-day forecast of L-Ww, S2-F-B-nll, and S2-J-0.4x."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    month = EXAMPLE_ORIGIN[:7]
    frames = {}
    for arm in COLOURS:
        pooled = build470.pool_stored(
            folder=score.WORK_DIR / "seed_preds" / arm, month=month, sites=sites, seeds=(0, 1)
        )
        frames[arm] = adapter.forecast_frame(
            stored=s407.widen_global(frame=pooled.drop("primary", "origin", "target")),
            series=series,
            unit="fraction",
        )
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    origin = datetime.fromisoformat(EXAMPLE_ORIGIN).replace(hour=9, tzinfo=UTC)
    for label, kind in EXAMPLES.items():
        one = {
            a: f.filter(pl.col("time_series_id") == ids[label], pl.col("power_fcst_init_time") == origin)
            .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0).sort("valid_time")
            for a, f in frames.items()
        }  # fmt: skip
        actual = (
            ctx["actuals"]
            .filter(pl.col("time_series_id") == ids[label])
            .join(one["L-Ww"].select("valid_time", "day"), on="valid_time")
        )
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("10 days", (0.0, 10.0))):
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            main = one["S2-J-0.4x"].filter(pl.col("day").is_between(*window)).to_pandas()
            band = alt.Chart(main).encode(x=x)  # ty: ignore[unresolved-attribute]
            layers = [
                band.mark_area(opacity=0.2, color=COLOURS["S2-J-0.4x"]).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                ),  # ty: ignore[unresolved-attribute]
                band.mark_area(opacity=0.35, color=COLOURS["S2-J-0.4x"]).encode(  # ty: ignore[unresolved-attribute]
                    y="q25:Q", y2="q75:Q"
                ),  # ty: ignore[unresolved-attribute]
            ]
            for arm, f in one.items():
                sub = f.filter(pl.col("day").is_between(*window)).to_pandas()
                layers.append(
                    alt.Chart(sub)  # ty: ignore[unresolved-attribute]
                    .mark_line(color=COLOURS[arm], strokeWidth=2)
                    .encode(x=x, y="q50:Q")
                )  # ty: ignore[unresolved-attribute]
            act = actual.filter(pl.col("day").is_between(*window)).to_pandas()
            layers.append(
                alt.Chart(act)  # ty: ignore[unresolved-attribute]
                .mark_line(color="black", strokeWidth=1.5, strokeDash=[4, 3])
                .encode(x=x, y="actual:Q")
            )  # ty: ignore[unresolved-attribute]
            alt.layer(*layers).properties(
                width=WIDTH, height=200,
                title=alt.TitleParams([f"{label} ({kind}), {horizon} forecast made on {EXAMPLE_ORIGIN}, widened 1.15.", "Bands: S2-J-0.4x 50% and 90%. Medians: L-Ww green, S2-F-B-nll orange, S2-J-0.4x blue. Dashed: actual."], anchor="start"),
            ).save(DOCS_DIR / f"figure_3_example_{label}_{horizon.replace('-', '_').replace(' ', '_')}.svg")  # fmt: skip


def main() -> None:
    """Draw every figure."""
    score.setup()
    figures()
    examples()


if __name__ == "__main__":
    main()

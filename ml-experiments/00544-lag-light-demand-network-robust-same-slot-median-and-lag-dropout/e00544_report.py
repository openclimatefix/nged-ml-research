# ruff: noqa: E501
"""Figures of experiment 544 from the tables its scorer stored.

Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DD: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
MINE: Final[Path] = (
    DD / "00544-lag-light-demand-network-robust-same-slot-median-and-lag-dropout/work"
)
SCORES: Final[Path] = MINE.parent / "scores"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00544-lag-light-demand-network-robust-same-slot-median-and-lag-dropout"
)
ARMS: Final[dict[str, str]] = {
    "R-MED": theme.BRAND_ORANGE, "R-DROP": theme.DATA_BLUE, "R-DROP-absent": theme.DATA_GREEN, "R-DW": "#7f7f7f", "R9": "#b07aa1", "A-pool": "black",
}  # fmt: skip
BANDS: Final[list[str]] = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h"]
EXAMPLE_ORIGIN: Final[str] = "2026-01-12"
FOLDERS: Final[dict[str, Path]] = {
    "R-DROP": MINE / "seed_preds/R-DROP", "R-MED": MINE / "seed_preds/R-MED",
    "L-1d": DD / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/work/seed_preds/L-1d",
}  # fmt: skip
EX_COLOURS: Final[dict[str, str]] = {
    "L-1d": theme.DATA_GREEN,
    "R-DROP": theme.DATA_BLUE,
    "R-MED": theme.BRAND_ORANGE,
}


def figures() -> None:
    """Draw figures 1 and 2, then the examples."""
    absolute = pl.read_csv(SCORES / "absolute.csv").filter(pl.col("band").is_in(BANDS))
    base = absolute.filter(pl.col("arm") == "L-1d").select("set", "band", base="crps")
    diff = (
        absolute.join(base, on=["set", "band"])
        .with_columns(d=pl.col("crps") - pl.col("base"))
        .filter(pl.col("arm").is_in(list(ARMS)))
    )
    panels = []
    for group, label in (("eight", "MVA-only eight"), ("six", "Signed-MW six")):
        part = diff.filter(pl.col("set") == group).to_pandas()
        panels.append(
            alt.Chart(part)
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("band:N", sort=BANDS, title="Lead time"),
                y=alt.Y("d:Q", title="CRPS minus L-1d (points of capacity)"),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(ARMS), range=list(ARMS.values())),
                    title="Arm",
                ),
            )
            .properties(width=420, height=200, title=alt.TitleParams(label, anchor="start"))
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "CRPS relative to L-1d (negative is better), five folds, two seeds. Intervals are in the tables.",
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_1_crps_vs_l1d.svg")
    leak = pl.read_csv(SCORES / "stress_leak_by_age.csv").to_pandas()
    alt.Chart(leak).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("block_age_days:O", title="Age of the injected shift (days before the refit)"),
        y=alt.Y("leak:Q", title="Leak (points of capacity)"),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=["L-1d", "R1", "R9", "R-DW", "R-MED", "R-DROP"],
                range=[
                    "black",
                    "#59a14f",
                    "#b07aa1",
                    "#7f7f7f",
                    theme.BRAND_ORANGE,
                    theme.DATA_BLUE,
                ],
            ),
            title="Arm",
        ),
    ).properties(
        width=420,
        height=220,
        title=alt.TitleParams(
            "Injected-shift leak by block age (seed 0, five folds)", anchor="start"
        ),
    ).save(DOCS_DIR / "figure_2_leak_by_age.svg")
    examples()


def examples() -> None:
    """Draw a day-ahead and a 10-day forecast at D8."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    month = EXAMPLE_ORIGIN[:7]
    frames = {}
    for arm, folder in FOLDERS.items():
        pooled = build470.pool_stored(folder=folder, month=month, sites=sites, seeds=(0, 1))
        frames[arm] = adapter.forecast_frame(
            stored=s407.widen_global(frame=pooled.drop("primary", "origin", "target")),
            series=series,
            unit="fraction",
        )
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    origin = datetime.fromisoformat(EXAMPLE_ORIGIN).replace(hour=9, tzinfo=UTC)
    label = "D8"
    one = {a: f.filter(pl.col("time_series_id") == ids[label], pl.col("power_fcst_init_time") == origin).with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0).sort("valid_time") for a, f in frames.items()}  # fmt: skip
    actual = (
        ctx["actuals"]
        .filter(pl.col("time_series_id") == ids[label])
        .join(one["R-DROP"].select("valid_time", "day"), on="valid_time")
    )
    for horizon, window in (("day-ahead", (1.0, 2.0)), ("10 days", (0.0, 10.0))):
        x = alt.X(
            "day:Q",
            title="Days after the forecast was made",
            scale=alt.Scale(domain=list(window), nice=False),
        )
        main = one["R-DROP"].filter(pl.col("day").is_between(*window)).to_pandas()
        band = alt.Chart(main).encode(x=x)  # ty: ignore[unresolved-attribute]
        layers = [
            band.mark_area(opacity=0.2, color=EX_COLOURS["R-DROP"]).encode(
                y=alt.Y("q05:Q", title="Power (MVA)", scale=alt.Scale(zero=False)), y2="q95:Q"
            ),  # ty: ignore[unresolved-attribute]
            band.mark_area(opacity=0.35, color=EX_COLOURS["R-DROP"]).encode(y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
        ]
        for arm, f in one.items():
            layers.append(
                alt.Chart(f.filter(pl.col("day").is_between(*window)).to_pandas())
                .mark_line(color=EX_COLOURS[arm], strokeWidth=2)
                .encode(x=x, y="q50:Q")
            )  # ty: ignore[unresolved-attribute]
        layers.append(
            alt.Chart(actual.filter(pl.col("day").is_between(*window)).to_pandas())
            .mark_line(color="black", strokeWidth=1.5, strokeDash=[4, 3])
            .encode(x=x, y="actual:Q")
        )  # ty: ignore[unresolved-attribute]
        alt.layer(*layers).properties(width=420, height=200, title=alt.TitleParams([f"{label} (MVA-only Primary), {horizon} forecast made on {EXAMPLE_ORIGIN}, widened 1.15.", "Bands: R-DROP 50% and 90%. Medians: R-DROP blue, R-MED orange, L-1d green. Dashed: actual."], anchor="start")).save(DOCS_DIR / f"figure_3_example_{label}_{horizon.replace('-', '_').replace(' ', '_')}.svg")  # fmt: skip


if __name__ == "__main__":
    figures()

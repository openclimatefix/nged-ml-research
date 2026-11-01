"""Draw the figures of experiment 463 from the tables of `e00463_score.py` (no fit).

```bash
DATA_PATH_INTERNAL=<private data folder> \
    uv run --no-sync python ml-experiments/00463-*/e00463_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00463-lag-trust-cross-fitted-trust-fit-seed-zero"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DATA_404: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00404-mdn-lagless-and-calibration-module-trust-weights"
    / "work"
    / "predictions"
)
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
SITES: Final[list[str]] = [
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19",
    "D21", "D22",
]  # fmt: skip
SIGNED_SIX_INDEX: Final[tuple[int, ...]] = (3, 5, 6, 7, 9, 11)
"""Indices in `SITES` of D5, D9, D10, D11, D14, and D16."""
WIDTH: Final[int] = 430
PAIR_LABELS: Final[dict[str, str]] = {
    "B-mix": "G1-trust minus 50/50 mixture (B-mix)",
    "G1-insample": "G1-trust minus G1-insample (the M11 test)",
    "B1": "G1-trust minus B1 (stored joint in-sample trust)",
}
SET_LABELS: Final[dict[tuple[str, str], str]] = {
    ("nolabel", "metered"): "rows with no labelled event",
    ("harm", "metered"): "harm rows (lags inside an event, target outside)",
    ("event", "1.0"): "inside real events (NRA-corrected actuals)",
}
GROUP_LABELS: Final[dict[str, str]] = {"signed_six": "Signed-MW six", "mva_eight": "MVA-only eight"}


def contrast_figure() -> None:
    """Draw G1-trust minus each reference, by group and row set, with month-resampled intervals."""
    table = pl.read_csv(OUT / "pooled.csv", schema_overrides={"actual": pl.String})
    rows = []
    for (setname, actual), set_label in SET_LABELS.items():
        part = table.filter(
            (pl.col("a") == "G1-trust")
            & pl.col("b").is_in(list(PAIR_LABELS))
            & (pl.col("set") == setname)
            & (pl.col("actual") == actual)
        )
        rows.append(part.with_columns(set_label=pl.lit(set_label)))
    data = pl.concat(rows).with_columns(
        label=pl.col("b").replace(PAIR_LABELS),
        panel=pl.col("group").replace(GROUP_LABELS) + pl.lit(", ") + pl.col("set_label"),
    )
    y = alt.Y("label:N", sort=list(PAIR_LABELS.values()), title=None)
    base = alt.Chart()
    thin = base.mark_rule(strokeWidth=1, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "low_bonf:Q",
            title="G1-trust minus reference (points of capacity)",
            scale=alt.Scale(domain=[-1.0, 1.0], clamp=True),
        ),
        x2="high_bonf:Q",
    )
    thick = base.mark_rule(strokeWidth=4, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="low95:Q", x2="high95:Q"
    )
    dots = base.mark_point(filled=True, size=60, color=theme.BRAND_ORANGE).encode(y=y, x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(zero, thin, thick, dots, data=data.to_pandas())
        .properties(width=WIDTH, height=110)
        .facet(
            row=alt.Row("panel:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Cross-fitted trust (G1-trust) minus each reference",
                    "Left of the dashed line is better than the reference. Dot: mean of months.",
                    "Thick line: 95% interval. Thin line: Bonferroni interval for 20 contrasts.",
                    "Both resample the 5 scored months.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=10,
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_1_contrasts.svg")


def weights_figure() -> None:
    """Draw the learned lagged weight by lead band and month for the three learned-trust arms."""
    w = pl.read_csv(OUT / "weights.csv").with_columns(
        group_name=pl.col("group").replace(GROUP_LABELS)
    )
    bands = ["0-12h", "12-36h", "36-72h", "days4-5", "days6-10"]
    line = (
        alt.Chart(w.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("band:N", sort=bands, title="Lead band"),
            y=alt.Y(
                "weight:Q",
                title="Mean learned weight on the lagged head",
                scale=alt.Scale(domain=[0.0, 1.0]),
            ),
            color=alt.Color("month:N", title="Scored month"),
            strokeDash=alt.StrokeDash("arm:N", title=None),
        )
    )
    chart = (
        line.properties(width=WIDTH, height=170)
        .facet(row=alt.Row("group_name:N", title=None))
        .properties(
            title=alt.TitleParams(
                [
                    "Learned weight on the lagged head by lead band, month, and arm",
                    "0.5 is the fixed 50/50 mixture.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=10,
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_2_learned_weights.svg")


def series_figure(
    *, name: str, primary: int, origin: int, month: str, horizon_steps: int, title: str
) -> None:
    """Draw one forecast origin: actual, median, and 50% and 90% bands of B-mix and G1-trust."""
    frames = []
    for label, folder in (
        ("B-mix (50/50)", DATA_404 / "B-mix"),
        ("G1-trust", OUT / "predictions" / "G1-trust"),
    ):
        frames.append(
            pl.read_parquet(folder / f"{month}_s0_w-1.parquet")
            .filter(
                (pl.col("primary") == primary)
                & (pl.col("origin") == origin)
                & ((pl.col("target") - pl.col("origin")) <= horizon_steps)
            )
            .with_columns(arm=pl.lit(label), hour=(pl.col("target") - pl.col("origin")) / 2.0)
        )
    data = pl.concat(frames, how="diagonal_relaxed")
    actual = data.filter(pl.col("arm") == "B-mix (50/50)").select(
        "hour", "actual", "tgt_real", "tgt_post"
    )
    colours = {"B-mix (50/50)": theme.DATA_BLUE, "G1-trust": theme.BRAND_ORANGE}
    scale = alt.Scale(domain=list(colours), range=list(colours.values()))
    base = alt.Chart(data.to_pandas())
    band90 = base.mark_area(opacity=0.12).encode(  # ty: ignore[unresolved-attribute]
        x="hour:Q",
        y="q05:Q",
        y2="q95:Q",
        color=alt.Color("arm:N", scale=scale, legend=alt.Legend(title=None, orient="top")),
    )
    band50 = base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
        x="hour:Q", y="q25:Q", y2="q75:Q", color=alt.Color("arm:N", scale=scale, legend=None)
    )
    med = base.mark_line().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hour:Q", title="Hours after the forecast origin"),
        y=alt.Y(
            "q50:Q", title="Demand (fraction of effective capacity)", scale=alt.Scale(zero=False)
        ),
        color=alt.Color("arm:N", scale=scale, legend=None),
    )
    act = (
        alt.Chart(actual.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(color="black", strokeWidth=1.5)
        .encode(x="hour:Q", y="actual:Q")
    )
    layers = [band90, band50, med, act]
    event = actual.filter(pl.col("tgt_real") | pl.col("tgt_post"))
    if event.height:
        span = (
            alt.Chart(
                pl.DataFrame({"a": [event["hour"].min()], "b": [event["hour"].max()]}).to_pandas()
            )
            .mark_rect(opacity=0.08, color="grey")
            .encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
        )
        layers.insert(0, span)
    alt.layer(*layers).properties(
        width=WIDTH, height=220, title=alt.TitleParams(title, anchor="start", fontSize=12)
    ).save(FIGURE_DIR / f"{name}.svg")


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    contrast_figure()
    weights_figure()
    best = None
    for month in ("2026-06", "2026-03", "2026-01", "2025-12"):
        frame = pl.read_parquet(DATA_404 / "B-mix" / f"{month}_s0_w-1.parquet").filter(
            pl.col("primary").is_in(
                [SITES.index(s) for s in ("D5", "D9", "D10", "D11", "D14", "D16")]
            )
            & ((pl.col("target") - pl.col("origin")) <= 48)
            & (pl.col("tgt_real") | pl.col("tgt_post"))
        )
        if frame.is_empty():
            continue
        top = frame.group_by("primary", "origin").len().sort("len", descending=True)[0]
        if best is None or top["len"][0] > best[2]:
            best = (month, top, int(top["len"][0]))
    if best is not None:
        month, top, _ = best
        primary, origin = int(top["primary"][0]), int(top["origin"][0])
        series_figure(
            name="figure_3_day_ahead_event", primary=primary, origin=origin, month=month,
            horizon_steps=48,
            title=f"Day ahead, {SITES[primary]}, {month}: real labelled event rows shaded grey",
        )  # fmt: skip
        series_figure(
            name="figure_4_ten_day_event", primary=primary, origin=origin, month=month,
            horizon_steps=480, title=f"Ten days, {SITES[primary]}, {month}, same origin",
        )  # fmt: skip
    d5 = SITES.index("D5")
    q = pl.read_parquet(DATA_404 / "B-mix" / "2026-06_s0_w-1.parquet").filter(
        pl.col("primary") == d5
    )
    ok = [
        o
        for o in np.sort(q["origin"].unique().to_numpy())
        if q.filter(
            (pl.col("origin") == o)
            & (pl.col("target") - pl.col("origin") <= 480)
            & (pl.col("tgt_real") | pl.col("tgt_post") | pl.col("win_real") | pl.col("lag_real"))
        ).is_empty()
    ]
    clean = int(ok[len(ok) // 2]) if ok else int(q["origin"].min())  # ty: ignore[invalid-argument-type]
    series_figure(
        name="figure_5_day_ahead_no_label", primary=d5, origin=clean, month="2026-06",
        horizon_steps=48, title="Day ahead, D5, 2026-06: no labelled event nearby",
    )  # fmt: skip
    series_figure(
        name="figure_6_ten_day_no_label", primary=d5, origin=clean, month="2026-06",
        horizon_steps=480, title="Ten days, D5, 2026-06, same origin",
    )  # fmt: skip


if __name__ == "__main__":
    main()

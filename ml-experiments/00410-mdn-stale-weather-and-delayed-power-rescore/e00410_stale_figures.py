# ruff: noqa: E501
"""Draw the figures of experiment 410 from the stored contrasts and forecasts (no fit).

- Figure 1: CRPS loss against the fresh case for every stale arm, with the month-block 95% interval,
  for the pooled network and the XGBoost S1M model at the signed-MW six and the MVA-only eight.
- Figure 2: the same loss by lead band for the pooled network.
- Figures 3 to 6: a day-ahead and a 14-day forecast of one Primary in winter and in summer under the
  fresh case and the 24-hour-stale arms, with 50% and 90% bands and the actual readings.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00410-mdn-stale-weather-and-delayed-power-rescore \
uv run --no-sync python ml-experiments/00410-*/e00410_stale_figures.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00410_stale_predict import OUT_DIR  # ty: ignore[unresolved-import]
from e00410_stale_score import GROUPS, arm_forecast, widen  # ty: ignore[unresolved-import]
from plotting import ocf_theme as theme
from studies.eval_protocol import metrics

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00410-mdn-stale-weather-and-delayed-power-rescore"
)
ARM_ORDER: Final[list[str]] = ["P6", "P12", "P24", "W24", "W48", "W24_P12", "W24_P24", "DROP"]
LOSS_ARMS: Final[list[str]] = [a for a in ARM_ORDER if a != "DROP"]
"""Figure 1 leaves out DROP, whose loss of 4 to 10 points would flatten every other arm."""
SIGNED_MVA: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight")
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "day 1": (0.0, 24.0),
    "days 2-3": (24.0, 72.0),
    "days 4-7": (72.0, 168.0),
    "days 8-10": (168.0, 240.0),
}
EXAMPLE_SITE: Final[str] = "D5"
EXAMPLES: Final[dict[str, str]] = {"winter": "2026-01", "summer": "2026-06"}
EXAMPLE_ARMS: Final[tuple[str, ...]] = ("FRESH", "W24", "P24")
ARM_COLOURS: Final[dict[str, str]] = {
    "FRESH": theme.DATA_BLUE,
    "W24": theme.BRAND_ORANGE,
    "P24": theme.DATA_PURPLE,
    "P12": theme.DATA_GREEN,
}


def figure_losses() -> None:
    """Figure 1: loss against the fresh case with 95% intervals, per network and group."""
    frames = []
    for network, label in (("pooled", "Pooled network"), ("xgb_s1m", "XGBoost S1M")):
        path = OUT_DIR / "scores" / f"{network}_contrasts.parquet"
        if path.exists():
            frames.append(
                pl.read_parquet(path)
                .filter(pl.col("group").is_in(SIGNED_MVA), pl.col("arm").is_in(LOSS_ARMS))
                .with_columns(model=pl.lit(label))
            )
    data = pl.concat(frames).select("arm", "group", "model", "diff", "lower", "upper")
    base = alt.Chart(data.to_pandas())
    y = alt.Y("arm:N", sort=LOSS_ARMS, title=None)
    colour = alt.Color(
        "model:N",
        scale=alt.Scale(
            domain=["Pooled network", "XGBoost S1M"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE]
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X("lower:Q", title="CRPS loss against the fresh case (points of capacity)"),
        x2="upper:Q",
        color=colour,
        yOffset="model:N",
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="diff:Q", color=colour, yOffset="model:N"
    )
    chart = (
        (rules + dots)
        .properties(width=330, height=230)
        .facet(column=alt.Column("group:N", title=None, sort=list(SIGNED_MVA)))
        .properties(
            title=alt.TitleParams(
                [
                    "CRPS loss at days 1 to 10 when the inputs are stale (interim, five months)",
                    "Dot: estimate. Line: 95% interval from resampling whole months. DROP (4 to 10 points) is left out.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_1_loss_by_arm.svg")
    chart.save(OUT_DIR / "figure_1_loss_by_arm.png", scale_factor=2)


def actual_points(*, arrays: dict, site: str) -> pl.DataFrame:
    """Return a Primary's readings in points of capacity on the half-hourly grid."""
    return (
        pl.DataFrame({"valid_time": common.grid_times(), "actual": arrays[site].power * 100.0})
        .drop_nans()
        .drop_nulls()
    )


def figure_by_lead(*, network: str = "pooled") -> None:
    """Figure 2: the CRPS loss by lead band, from the stored quantiles and the readings."""
    _series, arrays = fd._series_and_arrays()
    primaries = prep.primaries().select("time_series_id", "site")
    fresh = widen(frame=arm_forecast(network=network, arm="FRESH", months=list(prep.SCREEN_MONTHS)))
    rows = []
    for arm in ARM_ORDER:
        if arm == "DROP":
            continue
        stale = widen(frame=arm_forecast(network=network, arm=arm, months=list(prep.SCREEN_MONTHS)))
        both = fresh.join(stale, on=list(common.KEYS), suffix="_s")
        both = both.with_columns(
            lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(
                fractional=True
            )
        )
        for site, labels in (
            ("signed-MW six", GROUPS["signed-MW six"]),
            ("MVA-only eight", GROUPS["MVA-only eight"]),
        ):
            for band, (lo, hi) in LEAD_BANDS.items():
                diffs = []
                for label in labels:
                    sid = primaries.filter(pl.col("site") == label)["time_series_id"][0]
                    part = both.filter(
                        pl.col("time_series_id") == sid,
                        pl.col("lead").is_between(lo, hi, closed="right"),
                    ).join(actual_points(arrays=arrays, site=label), on="valid_time")
                    actual = part["actual"].to_numpy()
                    q_f = part.select(common.LEVEL_COLUMNS).to_numpy() * 100.0
                    q_s = part.select([f"{c}_s" for c in common.LEVEL_COLUMNS]).to_numpy() * 100.0
                    diffs.append(
                        float(
                            metrics.quantile_score(actual=actual, quantiles=q_s).mean()
                            - metrics.quantile_score(actual=actual, quantiles=q_f).mean()
                        )
                    )
                rows.append(
                    {"arm": arm, "group": site, "band": band, "loss": float(np.mean(diffs))}
                )
    data = pl.DataFrame(rows)
    data.write_parquet(OUT_DIR / "scores" / f"{network}_loss_by_lead.parquet")
    chart = (
        alt.Chart(data.to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("band:N", sort=list(LEAD_BANDS), title=None),
            y=alt.Y("loss:Q", title="CRPS loss (points of capacity)"),
            color=alt.Color(
                "arm:N",
                sort=ARM_ORDER,
                scale=alt.Scale(range=list(theme.DATA_COLOURS)[:7]),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(width=300, height=220)
        .facet(column=alt.Column("group:N", title=None, sort=list(SIGNED_MVA)))
        .properties(
            title=alt.TitleParams(
                "Where the loss falls: CRPS loss against the fresh case by lead (pooled network, interim)",
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_2_loss_by_lead.svg")
    chart.save(OUT_DIR / "figure_2_loss_by_lead.png", scale_factor=2)


def figure_examples(*, network: str = "pooled") -> None:
    """Figures 3 to 6: day-ahead and 14-day forecasts of one Primary under fresh and stale arms."""
    _series, arrays = fd._series_and_arrays()
    sid = prep.primaries().filter(pl.col("site") == EXAMPLE_SITE)["time_series_id"][0]
    actual = actual_points(arrays=arrays, site=EXAMPLE_SITE)
    number = 3
    for season, month in EXAMPLES.items():
        origin = datetime.fromisoformat(f"{month}-15").replace(hour=9, tzinfo=UTC)
        frames = {}
        for arm in EXAMPLE_ARMS:
            f = widen(frame=arm_forecast(network=network, arm=arm, months=[month]))
            frames[arm] = f.filter(
                pl.col("time_series_id") == sid, pl.col("power_fcst_init_time") == origin
            )
        for span, hours in (("day-ahead", 48.0), ("14-day", 336.0)):
            pieces = []
            for arm, f in frames.items():
                part = f.with_columns(
                    lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(
                        fractional=True
                    ),
                    arm=pl.lit(arm),
                ).filter(pl.col("lead") <= hours)
                pieces.append(part)
            data = (
                pl.concat(pieces)
                .select("arm", "valid_time", "q05", "q25", "q50", "q75", "q95")
                .with_columns(pl.col("q05", "q25", "q50", "q75", "q95") * 100.0)
            )
            lo, hi = data["valid_time"].min(), data["valid_time"].max()
            truth = actual.filter(pl.col("valid_time").is_between(lo, hi))
            colour = alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(EXAMPLE_ARMS), range=[ARM_COLOURS[a] for a in EXAMPLE_ARMS]
                ),
                legend=alt.Legend(title=None, orient="top"),
            )
            x = alt.X("valid_time:T", title="Valid time (UTC)")
            pdata = data.to_pandas()
            band90 = (
                alt.Chart(pdata)
                .mark_area(opacity=0.12)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x,
                    y=alt.Y("q05:Q", title="Power (points of capacity)"),
                    y2="q95:Q",
                    color=colour,
                )
            )
            band50 = (
                alt.Chart(pdata)
                .mark_area(opacity=0.25)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x, y="q25:Q", y2="q75:Q", color=colour
                )
            )
            line = alt.Chart(pdata).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
            actual_line = (
                alt.Chart(truth.to_pandas())
                .mark_line(color="black", strokeDash=[3, 2])
                .encode(x="valid_time:T", y="actual:Q")  # ty: ignore[unresolved-attribute]
            )
            chart = (band90 + band50 + line + actual_line).properties(
                width=620,
                height=240,
                title=alt.TitleParams(
                    [
                        f"{EXAMPLE_SITE}, {season} example, {span} forecast (origin {origin:%Y-%m-%d %H:%M} UTC)",
                        "Line: median. Dark band: 50%. Light band: 90%. Dashed black: actual readings.",
                    ],
                    anchor="start",
                ),
            )
            name = f"figure_{number}_example_{season}_{span.replace('-', '_')}"
            chart.save(FIGURE_DIR / f"{name}.svg")
            chart.save(OUT_DIR / f"{name}.png", scale_factor=2)
            number += 1


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scores").mkdir(parents=True, exist_ok=True)
    figure_losses()
    figure_by_lead()
    figure_examples()


if __name__ == "__main__":
    main()

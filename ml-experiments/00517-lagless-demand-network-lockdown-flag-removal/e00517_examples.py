# ruff: noqa: E501
"""Draw example forecasts of experiment 517's arms C0 and C2 against the actuals.

Figures (SVG, in the experiment's docs folder; optimise each with `npx svgo@4 --multipass
--precision=1 --final-newline`): `figure_2_example_<Primary>_day_ahead.svg` and
`figure_2_example_<Primary>_10_days.svg` for Primaries D10 and D4. Each figure stacks four panels:
a winter and a summer forecast origin, each for arm C0 (flag kept) and arm C2 (flag removed,
lockdown rows dropped). Forecasts are the quantile average of seeds 0, 1, and 2.

```bash
source <paths>; uv run --no-sync python ml-experiments/00517-*/e00517_examples.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00517_score as score
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00517-lagless-demand-network-lockdown-flag-removal"
)
WIDTH: Final[int] = 600
PANEL_HEIGHT: Final[int] = 110
ARM_COLOURS: Final[dict[str, str]] = {"C0": theme.DATA_BLUE, "C2": theme.BRAND_ORANGE}
ARM_NAMES: Final[dict[str, str]] = {
    "C0": "C0 (flag kept)",
    "C2": "C2 (flag removed, lockdown rows dropped)",
}
EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}
ORIGINS: Final[dict[str, str]] = {"winter": "2026-01-12", "summer": "2026-06-15"}
HORIZONS: Final[dict[str, tuple[float, float]]] = {
    "day_ahead": (1.0, 2.0),
    "10_days": (0.0, 10.0),
}


def _panel(
    *,
    forecast: pl.DataFrame,
    actual: pl.DataFrame,
    arm: str,
    title: str,
    window: tuple[float, float],
) -> alt.LayerChart:
    """Return one panel: the arm's 50% and 90% bands and median, with the actuals dashed."""
    x = alt.X(
        "day:Q",
        title="Days after the forecast was made",
        scale=alt.Scale(domain=list(window), nice=False),
    )
    data = forecast.filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
    base = alt.Chart(data).encode(x=x)  # ty: ignore[unresolved-attribute]
    colour = ARM_COLOURS[arm]
    act = actual.filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
    return alt.layer(
        base.mark_area(opacity=0.2, color=colour, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
        ),
        base.mark_area(opacity=0.4, color=colour, aria=False).encode(y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_line(color=colour, strokeWidth=2, aria=False).encode(y="q50:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(act)  # ty: ignore[unresolved-attribute]
        .mark_line(color="black", strokeWidth=1.2, strokeDash=[4, 3], aria=False)
        .encode(x=x, y="actual:Q"),
    ).properties(
        width=WIDTH, height=PANEL_HEIGHT, title=alt.TitleParams(title, anchor="start", fontSize=11)
    )


def figure_examples() -> None:
    """Draw the day-ahead and the 10-day forecasts of C0 and C2 for each example Primary."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    frames: dict[tuple[str, str], pl.DataFrame] = {}
    for season, day in ORIGINS.items():
        for arm in score.ARMS:
            if arm == "C1":
                continue
            pooled = build470.pool_stored(
                folder=score.SEED_PREDS / arm, month=day[:7], sites=sites, seeds=score.SEEDS
            )
            frames[season, arm] = adapter.forecast_frame(
                stored=s407.widen_global(frame=pooled.drop("primary", "origin", "target")),
                series=series,
                unit="fraction",
            )
    for label, kind in EXAMPLES.items():
        for horizon, window in HORIZONS.items():
            panels = []
            for season, day in ORIGINS.items():
                origin = datetime.fromisoformat(day).replace(hour=9, tzinfo=UTC)
                for arm in ("C0", "C2"):
                    one = (
                        frames[season, arm]
                        .filter(pl.col("time_series_id") == ids[label], pl.col("power_fcst_init_time") == origin)
                        .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0)
                        .sort("valid_time")
                    )  # fmt: skip
                    actual = (
                        ctx["actuals"]
                        .filter(pl.col("time_series_id") == ids[label])
                        .join(one.select("valid_time", "day"), on="valid_time")
                    )
                    panels.append(
                        _panel(
                            forecast=one,
                            actual=actual,
                            arm=arm,
                            title=f"{ARM_NAMES[arm]}, {season} (forecast made {day[:7]})",
                            window=window,
                        )
                    )
            horizon_text = "day-ahead" if horizon == "day_ahead" else "10-day"
            chart = alt.vconcat(*panels).properties(
                title=alt.TitleParams(
                    [
                        f"{label} ({kind}): {horizon_text} forecasts made at 09:00 UTC on one winter",
                        "and one summer day. Line: median. Shades: 50% and 90% bands (widened 1.15).",
                        "Dashed: actual. Blue: C0. Orange: C2.",
                    ],
                    anchor="start",
                )
            )
            chart.save(DOCS_DIR / f"figure_2_example_{label}_{horizon}.svg")


if __name__ == "__main__":
    figure_examples()

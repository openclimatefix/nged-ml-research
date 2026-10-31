"""Experiment 435: coverage and pinball loss at the four tail quantiles, per group and lead band.

`tail_table` takes a forecast frame (`time_series_id`, `power_fcst_init_time`, `valid_time`, and the
quantile columns, all in the same unit as `actual`) and returns, per metering group, lead band, and
Primary-or-pooled key, the coverage and the pinball loss at q01, q05, q95, and q99. "Coverage" at a
quantile is the share of actuals at or below it. The widened variant scales the distance of every
quantile from the median by a factor, as experiment 407's `widen_global` does.

The `main` below scores the experiment 425 arms `R` and `A` (seeds 0 to 4 averaged, 12 folds), raw
and widened by 1.15, and writes `work/tails_425.parquet`.

```bash
# PYTHONPATH as in e00425_score.py (experiments 87, 48, 377, 381, 397, 407, 410, 418, 422, and 425)
uv run --no-sync python ml-experiments/00435-*/e00435_tails.py
```
"""

from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import e00425_score as score425  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
)
TAIL_LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
GROUPS: Final[dict[str, tuple[str, ...]]] = score425.SUBSETS
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-12h": (0.0, 12.0),
    "12-36h": (12.0, 36.0),
    "36-72h": (36.0, 72.0),
    "days4-5": (72.0, 120.0),
    "days6-10": (120.0, 240.0),
}
"""Lead bands in hours, lower bound inclusive, upper bound exclusive (the last includes 240)."""
WIDENING: Final[float] = 1.15
ARMS: Final[tuple[str, ...]] = ("R", "A")


def with_group_and_band(*, frame: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Add `label`, `group`, `lead_hours`, and `lead_band` columns."""
    group_of = {label: g for g, members in GROUPS.items() for label in members}
    band = pl.lit(None, dtype=pl.String)
    lead = pl.col("lead_hours")
    for i, (name, (low, high)) in enumerate(LEAD_BANDS.items()):
        last = i == len(LEAD_BANDS) - 1
        inside = (lead >= low) & ((lead <= high) if last else (lead < high))
        band = pl.when(inside).then(pl.lit(name)).otherwise(band)
    return (
        frame.join(series.select("time_series_id", "label"), on="time_series_id")
        .with_columns(
            group=pl.col("label").replace_strict(group_of, default=None),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
        )
        .with_columns(lead_band=band)
    )


def widen(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Scale each quantile's distance from the median by `factor`."""
    return frame.with_columns(
        *[(pl.col("q50") + factor * (pl.col(q) - pl.col("q50"))).alias(q) for q in TAIL_LEVELS]
    )


def tail_table(*, frame: pl.DataFrame, by: tuple[str, ...]) -> pl.DataFrame:
    """Return coverage and pinball loss at each tail quantile, with the row count, per `by` key.

    `frame` must carry `actual` and the quantile columns. Pinball loss is in the frame's unit.
    """
    aggs: list[pl.Expr] = [pl.len().alias("n")]
    for name, tau in TAIL_LEVELS.items():
        error = pl.col("actual") - pl.col(name)
        aggs.append((pl.col("actual") <= pl.col(name)).mean().alias(f"cov_{name}"))
        aggs.append(
            pl.when(error >= 0)
            .then(tau * error)
            .otherwise((tau - 1.0) * error)
            .mean()
            .alias(f"pin_{name}")
        )
    return frame.group_by(*by).agg(*aggs).sort(*by)


def main() -> None:
    """Score experiment 425's arms R and A raw and widened by 1.15."""
    months = tuple(
        f"{y}-{m:02d}"
        for y, m in [(2025, m) for m in range(7, 13)] + [(2026, m) for m in range(1, 7)]
    )
    series = adapter.series_table()
    actuals = prep.cleaned_scoring_context(series=series)["actuals"].filter(
        pl.col("actual").is_not_nan() & pl.col("actual").is_not_null()
    )
    frames = score425.load_frames(series=series, months=months, with_references=False)
    out: list[pl.DataFrame] = []
    for arm in ARMS:
        raw = frames[f"{arm}_raw"]
        for variant, frame in (("raw", raw), ("w1.15", widen(frame=raw, factor=WIDENING))):
            scored = with_group_and_band(
                frame=frame.join(actuals, on=["time_series_id", "valid_time"]), series=series
            ).filter(pl.col("group").is_not_null() & pl.col("lead_band").is_not_null())
            for by in (("group", "lead_band"), ("label", "lead_band")):
                table = tail_table(frame=scored, by=by).with_columns(
                    arm=pl.lit(arm), variant=pl.lit(variant), by=pl.lit("+".join(by))
                )
                out.append(table.rename({by[0]: "key"}).drop("label", strict=False))
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    result = pl.concat(out, how="diagonal")
    result.write_parquet(WORK_DIR / "tails_425.parquet")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    print(
        result.filter(pl.col("by") == "group+lead_band").sort("arm", "variant", "key", "lead_band")
    )


if __name__ == "__main__":
    main()

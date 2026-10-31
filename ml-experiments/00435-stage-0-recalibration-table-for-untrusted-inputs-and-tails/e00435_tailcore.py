"""Experiment 435: tail coverage and pinball helpers for the demand, wind, and solar scorers.

`tail_table` takes a forecast frame with `actual` and the quantile columns (same unit) and returns,
per key, the coverage and the pinball loss at q01, q05, q95, and q99. "Coverage" at a quantile is
the share of actuals at or below it. `widen` scales each quantile's distance from the median by a
factor, as experiment 407's `widen_global` does.
"""

from typing import Final

import polars as pl

TAIL_LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-12h": (0.0, 12.0),
    "12-36h": (12.0, 36.0),
    "36-72h": (36.0, 72.0),
    "days4-5": (72.0, 120.0),
    "days6-10": (120.0, 240.0),
}
"""Lead bands in hours, lower bound inclusive, upper bound exclusive (the last includes 240)."""
WIDENING: Final[float] = 1.15


def lead_band_expr(*, column: str = "lead_hours") -> pl.Expr:
    """Return the lead-band name of `column`, or null outside 0 to 240 hours."""
    band = pl.lit(None, dtype=pl.String)
    lead = pl.col(column)
    for i, (name, (low, high)) in enumerate(LEAD_BANDS.items()):
        last = i == len(LEAD_BANDS) - 1
        inside = (lead >= low) & ((lead <= high) if last else (lead < high))
        band = pl.when(inside).then(pl.lit(name)).otherwise(band)
    return band


def widen(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Scale each quantile's distance from the median by `factor`."""
    return frame.with_columns(
        *[(pl.col("q50") + factor * (pl.col(q) - pl.col("q50"))).alias(q) for q in TAIL_LEVELS]
    )


def tail_table(*, frame: pl.DataFrame, by: tuple[str, ...]) -> pl.DataFrame:
    """Return coverage and pinball loss at each tail quantile, with the row count, per `by` key.

    `frame` must carry `actual` and the quantile columns. Pinball loss is in the frame's unit.
    """
    complete = pl.all_horizontal(
        pl.col("actual").is_finite(),
        *[pl.col(q).is_finite() for q in TAIL_LEVELS],
        pl.col("q50").is_finite(),
    )
    frame = frame.filter(complete)
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

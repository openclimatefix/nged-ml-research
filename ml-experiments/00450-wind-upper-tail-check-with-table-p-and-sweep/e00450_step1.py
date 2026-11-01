"""Experiment 450, step 1: the wind upper-tail check with table P on experiment 429's forecasts.

For each stored experiment 429 arm (seeds averaged in quantile space, levels sorted), table P is
fitted as in experiment 435 (commit 552c3f87, `e00435_table_p_provisional.fit_factors`): per (farm,
lead band) and tail quantile, the factor that scales the quantile's distance from the median so
that the coverage on the fitting rows equals the nominal level. Each scored month is fitted on the
earlier months only, so the first month (2025-07) is never scored. Rows are scored at true days 1
to 10 (lead 24 to 240 hours), with and without experiment 210's idle-band mask (`masked`). The
fit uses the same row set as the scoring (all rows, or only unmasked rows). Variants: `raw`,
`w1.15` (experiment 435's fixed widening), and `P`. Intervals are 95% month-block bootstrap
intervals (11 scored months, 2,000 resamples), because the row set spans one year.

Quantiles after widening or table P are clipped to [0, 1], as the raw quantiles are, and the
fit uses only the rows scored (lead 24 hours or more).

Output: `work/step1_cells.parquet` (hits and rows per arm, row set, variant, farm, month, tail).

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00429-*/env.sh
uv run --no-sync python ml-experiments/00450-*/e00450_step1.py
```
"""

from pathlib import Path
from typing import Final

import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00429_run as run429  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00450-wind-upper-tail-check-with-table-p-and-sweep" / "work"
)
LEVELS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
TAILS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
UPPER: Final[tuple[str, ...]] = ("q95", "q99")
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-12h": (0.0, 12.0), "12-36h": (12.0, 36.0), "36-72h": (36.0, 72.0),
    "days4-5": (72.0, 120.0), "days6-10": (120.0, 240.0),
}  # fmt: skip
WIDENING: Final[float] = 1.15
MIN_FIT_ROWS: Final[int] = 200
FARM_LABEL: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}
ARMS: Final[tuple[str, ...]] = tuple(
    n
    for n in run429.ARMS
    if all((run429.PARTS_DIR / f"{n}_h{s}.npy").exists() for s in run429.SEEDS)
)


def lead_band_expr() -> pl.Expr:
    """Return the lead band of `lead_hours`, or null outside 0 to 240 hours."""
    band = pl.lit(None, dtype=pl.String)
    for i, (name, (low, high)) in enumerate(LEAD_BANDS.items()):
        last = i == len(LEAD_BANDS) - 1
        inside = (pl.col("lead_hours") >= low) & (
            (pl.col("lead_hours") <= high) if last else (pl.col("lead_hours") < high)
        )
        band = pl.when(inside).then(pl.lit(name)).otherwise(band)
    return band


def arm_frame(*, name: str, base: pl.DataFrame) -> pl.DataFrame:
    """Return the base rows with the seed-mean quantiles of one arm."""
    stack = [np.load(run429.PARTS_DIR / f"{name}_h{s}.npy") for s in run429.SEEDS]
    q = np.sort(np.mean(np.stack(stack), axis=0), axis=1)
    return base.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(LEVELS)])


def fit_factors(*, fit: pl.DataFrame) -> dict[tuple[str, str, str], float]:
    """Return one factor per (farm, lead band, tail quantile) from the fitting rows."""
    out: dict[tuple[str, str, str], float] = {}
    for (site, band), part in fit.group_by("site", "lead_band"):
        for name, tau in TAILS.items():
            spread = (pl.col(name) - pl.col("q50")).abs()
            ratio = (
                (pl.col("actual") - pl.col("q50")) / spread
                if name in UPPER
                else (pl.col("q50") - pl.col("actual")) / spread
            )
            level = tau if name in UPPER else 1.0 - tau
            use = part.filter(spread > 1e-6)
            if use.height >= MIN_FIT_ROWS:
                out[site, band, name] = float(use.select(ratio.quantile(level)).item())
    return out


def apply_factors(
    *, frame: pl.DataFrame, factors: dict[tuple[str, str, str], float]
) -> pl.DataFrame:
    """Scale each tail quantile's distance from the median by its fitted factor."""
    cols = []
    for name in TAILS:
        lookup = pl.DataFrame(
            [(s, b, f) for (s, b, n), f in factors.items() if n == name],
            schema={"site": pl.String, "lead_band": pl.String, "factor": pl.Float64},
            orient="row",
        )
        cols.append((name, lookup))
    out = frame
    for name, lookup in cols:
        out = (
            out.join(lookup, on=["site", "lead_band"], how="left")
            .with_columns(
                (pl.col("q50") + pl.col("factor").fill_null(1.0) * (pl.col(name) - pl.col("q50")))
                .clip(0.0, 1.0)
                .alias(name)
            )
            .drop("factor")
        )
    return out


def widen(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Scale each tail quantile's distance from the median by the fixed factor."""
    return frame.with_columns(
        *[
            (pl.col("q50") + WIDENING * (pl.col(q) - pl.col("q50"))).clip(0.0, 1.0).alias(q)
            for q in TAILS
        ]
    )


def hits(*, frame: pl.DataFrame, arm: str, rowset: str, variant: str) -> pl.DataFrame:
    """Return rows and hits (actual <= quantile) per farm, month, and tail quantile."""
    return (
        frame.filter(pl.col("lead_hours") >= 24.0)
        .group_by("site", "month")
        .agg(
            pl.len().alias("n"),
            *[(pl.col("actual") <= pl.col(q)).sum().alias(f"hit_{q}") for q in TAILS],
        )
        .with_columns(arm=pl.lit(arm), rowset=pl.lit(rowset), variant=pl.lit(variant))
    )


def main() -> None:
    """Score every arm raw, widened, and after table P, with and without the mask."""
    cases = data347.scored_frame()
    base = (
        cases.select("site", "month", "masked", "lead_hours", actual=pl.col("actual_norm"))
        .filter(pl.col("actual").is_finite())
        .with_columns(lead_band=lead_band_expr())
        .filter(pl.col("lead_band").is_not_null())
    )
    months = sorted(base["month"].unique())
    out: list[pl.DataFrame] = []
    for name in ARMS:
        frame = arm_frame(
            name=name,
            base=cases.select(
                "site", "month", "masked", "lead_hours", actual=pl.col("actual_norm")
            ),
        )
        frame = (
            frame.filter(pl.col("actual").is_finite())
            .with_columns(lead_band=lead_band_expr())
            .filter(pl.col("lead_band").is_not_null())
        )
        for rowset, keep in (("all rows", pl.lit(True)), ("kept rows", ~pl.col("masked"))):
            sub = frame.filter(keep)
            for i, month in enumerate(months[1:], start=1):
                earlier = sub.filter(
                    pl.col("month").is_in(months[:i]) & (pl.col("lead_hours") >= 24.0)
                )
                held = sub.filter(pl.col("month") == month)
                factors = fit_factors(fit=earlier)
                for variant, f in (
                    ("raw", held),
                    ("w1.15", widen(frame=held)),
                    ("P", apply_factors(frame=held, factors=factors)),
                ):
                    out.append(hits(frame=f, arm=name, rowset=rowset, variant=variant))
        print(name, flush=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(out).write_parquet(WORK_DIR / "step1_cells.parquet")


if __name__ == "__main__":
    main()

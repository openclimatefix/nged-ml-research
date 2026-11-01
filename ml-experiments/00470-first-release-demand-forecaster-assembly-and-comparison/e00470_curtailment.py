"""Avoidable-curtailment cost of experiment 470's candidates beside BG2 and manual_heuristic.

Follows experiment 454's `e00454_curtailment.py` (experiment 448's `curtailment`, experiment 449's
per-series costing and cell bootstrap) with this experiment's stored cells. Forecasts are the mean
over an arm's cells in quantile space, widened by 1.15 on the four tail quantiles as in the scorer,
converted to MW, on the rows with no labelled event that BG2 and manual_heuristic also hold, at the
fractions f = 0.9, 0.75, and 0.5 of the scaled limit, for the signed-MW six.

```bash
source .e470env.sh
uv run --no-sync python ml-experiments/00470-*/e00470_curtailment.py
```
"""

from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import widen  # ty: ignore[unresolved-import]

MONTHS: Final[tuple[str, ...]] = build470.MONTHS
RESAMPLES: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = ("A-pool", "K2-S1", "K2-425A", "K2-425A+S1", "A-gam-in", "B-mix")
REFERENCES: Final[tuple[str, ...]] = ("BG2", "manual_heuristic")
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
CANDIDATES: Final[tuple[str, ...]] = ("A-pool", "K2-S1", "K2-425A", "K2-425A+S1", "A-gam-in")
COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    *[(a, b) for i, a in enumerate(CANDIDATES) for b in CANDIDATES[:i]],
    *[(a, ref) for a in CANDIDATES for ref in REFERENCES],
)  # fmt: skip

_BASES: dict[str, pl.DataFrame] = {}
_CONTEXT: dict = {}


def month_base(*, month: str) -> pl.DataFrame:
    """Return a month's `gam14` rows with no labelled event, keyed by series, init, and valid."""
    if month not in _BASES:
        if not _CONTEXT:
            _CONTEXT.update(prep425.cleaned_scoring_context(series=adapter.series_table()))
        full = score470.month_context(month=month, sites=build470.site_map(), ctx=_CONTEXT)
        wanted = score470.load_rows().filter((pl.col("month") == month) & pl.col("gam14"))
        rows = full.join(wanted.select(build470.KEYS), on=build470.KEYS, how="semi")
        _BASES[month] = rows.filter(pl.col("cls") == "clean")
    return _BASES[month]


def read_construct_rows(
    *, arm: str, month: str, seeds: tuple[int, ...], factor: float, with_quantiles: bool
) -> pl.DataFrame | None:
    """Return an arm's quantiles (fractions, cells averaged, tails widened) on the month's rows."""
    del seeds, with_quantiles
    rows = month_base(month=month)
    frames = [
        pl.read_parquet(build470.prediction_path(arm=arm, month=month, cell=c))
        for c in range(score470.arm_cells(arm=arm))
    ]
    q = np.mean([f.select(list(LEVELS)).to_numpy().astype(np.float64) for f in frames], axis=0)
    mean = (
        frames[0].select(build470.KEYS).hstack(pl.DataFrame(q, schema=list(LEVELS), orient="row"))
    )
    out = rows.select(*build470.KEYS, "time_series_id", "power_fcst_init_time", "valid_time").join(
        widen(frame=mean, factor=factor), on=build470.KEYS, how="inner"
    )
    capacity = base.sites_table().select(
        pl.col("primary").cast(pl.Int16), "capacity_mw", "label", "group"
    )
    return out.join(capacity, on="primary").filter(pl.col("q50").is_finite())


def shared_keys(*, month: str, refs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the half-hours that every arm and every reference holds in the month."""
    keys = month_base(month=month).select(*base.KEYS)
    for ref in refs.values():
        finite = ref.select(*base.KEYS, "q50").filter(pl.col("q50").is_finite())
        keys = keys.join(finite.select(*base.KEYS), on=base.KEYS, how="semi")
    return keys


def build() -> pl.DataFrame:
    """Return the cost per month, arm, series, and fraction."""
    base.read_arm_rows = read_construct_rows  # ty: ignore[invalid-assignment]
    base.SEEDS = (0,)
    base.metrics.avoidable_curtailment_cost = c449.per_series_cost  # ty: ignore[invalid-assignment]
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    all_refs = base.reference_frames(series=series)
    refs = {name: all_refs[name] for name in REFERENCES}
    parts = []
    for month in MONTHS:
        cost = base.curtailment(
            months=(month,), refs=refs, keys=shared_keys(month=month, refs=refs), ctx=ctx,
            series=series, arms=ARMS,
        )  # fmt: skip
        parts.append(cost.with_columns(month=pl.lit(month)))
    return pl.concat(parts)


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    c449.ARMS = ARMS
    c449.REFERENCES = REFERENCES
    c449.COMPARISONS = COMPARISONS
    c449.RESAMPLES = RESAMPLES
    cost = build()
    out = score470.SCORE_DIR
    cost.write_parquet(out / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(out / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()

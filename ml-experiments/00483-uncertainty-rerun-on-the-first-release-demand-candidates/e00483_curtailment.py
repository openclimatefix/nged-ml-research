"""Experiment 483: avoidable-curtailment cost of the candidates, raw, widened, and after table P.

Follows experiment 470's `e00470_curtailment.py` (signed-MW six, rows with no labelled event, rows
that BG2 and `manual_heuristic` also hold, fractions f = 0.9, 0.75, and 0.5 of the scaled limit),
but costs three variants of each candidate: `raw`, `w1.15`, and `P` (table P of this experiment,
fitted leave-one-fold-out). Variants are named `<arm>|<variant>` and read from the cache.

```bash
source /tmp/e483env.sh
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00483-*/e00483_curtailment.py
```
"""

from typing import Final

import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as c470  # ty: ignore[unresolved-import]
import e00483_tails as tails  # ty: ignore[unresolved-import]
import polars as pl
from e00435_tailcore import widen  # ty: ignore[unresolved-import]

VARIANTS: Final[tuple[str, ...]] = ("raw", "w1.15", "P")
ARMS: Final[tuple[str, ...]] = tuple(
    f"{arm}|{v}" for arm in ("A-pool", "A-gam-in") for v in VARIANTS
)
_ADJUSTED: dict[str, pl.DataFrame] = {}


def adjusted(*, arm: str, variant: str) -> pl.DataFrame:
    """Return all months of an arm with the variant applied (table P is leave-one-fold-out)."""
    key = f"{arm}|{variant}"
    if key not in _ADJUSTED:
        frame = tails.load(arm=arm).filter(pl.col("cls") == "clean")
        if variant == "w1.15":
            out = widen(frame=frame, factor=1.15)
        elif variant == "P":
            tagged = tails.load(arm=arm).with_columns(group_orig=pl.col("group"))
            parts = []
            for month in tails.MONTHS:
                fit = tagged.filter((pl.col("month") != month) & (pl.col("cls") == "clean"))
                factors = tails.fit_with_fallback(fit=fit, cold_class=False)
                held = tagged.filter((pl.col("month") == month) & (pl.col("cls") == "clean"))
                parts.append(tails.prov.apply_factors(frame=held, factors=factors))
            out = pl.concat(parts)
        else:
            out = frame
        _ADJUSTED[key] = out
    return _ADJUSTED[key]


def read_variant_rows(
    *, arm: str, month: str, seeds: tuple[int, ...], factor: float, with_quantiles: bool
) -> pl.DataFrame | None:
    """Return a variant's quantiles (fractions) on the month's rows, as experiment 470's reader."""
    del seeds, with_quantiles, factor
    name, variant = arm.split("|")
    rows = c470.month_base(month=month)
    frame = adjusted(arm=name, variant=variant).filter(pl.col("month") == month)
    out = rows.select(*build470.KEYS, "time_series_id", "power_fcst_init_time", "valid_time").join(
        frame.select(*build470.KEYS, *tails.LEVELS), on=build470.KEYS, how="inner"
    )
    capacity = base.sites_table().select(
        pl.col("primary").cast(pl.Int16), "capacity_mw", "label", "group"
    )
    return out.join(capacity, on="primary").filter(pl.col("q50").is_finite())


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    c470.ARMS = ARMS
    c470.read_construct_rows = read_variant_rows
    comparisons = tuple(
        (f"{a}|{v}", f"{a}|raw") for a in ("A-pool", "A-gam-in") for v in ("w1.15", "P")
    ) + tuple(
        (f"{a}|{v}", r) for a in ("A-pool", "A-gam-in") for v in VARIANTS for r in c470.REFERENCES
    )
    c449.ARMS, c449.REFERENCES, c449.COMPARISONS = ARMS, c470.REFERENCES, comparisons
    c449.RESAMPLES = c470.RESAMPLES
    cost = c470.build()
    tails.WORK.mkdir(parents=True, exist_ok=True)
    cost.write_parquet(tails.WORK / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(tails.WORK / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=160):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()

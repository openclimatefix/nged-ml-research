"""Experiment 500: avoidable-curtailment cost of A-pool's raw forecast and every recalibration.

Follows experiment 483's `e00483_curtailment.py` (signed-MW six, rows with no labelled event that
BG2 and `manual_heuristic` also hold, fractions f = 0.9, 0.75, and 0.5 of the scaled limit), with
the variants of this experiment: `raw`, `w1.15`, `P` (table P, leave-one-fold-out), `two-constant`
(leave-one-fold-out), `RC7`, `RC14`, `RC28`, and `RC14+w1.05`. Variants are named
`A-pool|<variant>` and read from the arm frames that `e00500_conformal.py arms` wrote.

```bash
source /tmp/rc_env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00500-*/e00500_curtailment.py
```
"""

from typing import Final

import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as c470  # ty: ignore[unresolved-import]
import e00500_conformal as conformal  # ty: ignore[unresolved-import]
import polars as pl

VARIANTS: Final[tuple[str, ...]] = (
    "raw", "w1.15", "P", "two-constant", "RC7", "RC14", "RC28", "RC14+w1.05", "RC28-bias",
)  # fmt: skip
ARMS: Final[tuple[str, ...]] = tuple(f"A-pool|{v}" for v in VARIANTS)
_FRAMES: dict[str, pl.DataFrame] = {}


def variant_frame(*, variant: str) -> pl.DataFrame:
    """Return all months of a variant on clean rows."""
    if variant not in _FRAMES:
        _FRAMES[variant] = conformal.load_arm(arm=variant).filter(pl.col("cls") == "clean")
    return _FRAMES[variant]


def read_variant_rows(
    *, arm: str, month: str, seeds: tuple[int, ...], factor: float, with_quantiles: bool
) -> pl.DataFrame | None:
    """Return a variant's quantiles (fractions) on the month's rows, as experiment 470's reader."""
    del seeds, with_quantiles, factor
    _, variant = arm.split("|")
    rows = c470.month_base(month=month)
    frame = variant_frame(variant=variant).filter(pl.col("month") == month)
    out = rows.select(*build470.KEYS, "time_series_id", "power_fcst_init_time", "valid_time").join(
        frame.select(*build470.KEYS, *conformal.LEVELS), on=build470.KEYS, how="inner"
    )
    capacity = base.sites_table().select(
        pl.col("primary").cast(pl.Int16), "capacity_mw", "label", "group"
    )
    return out.join(capacity, on="primary").filter(pl.col("q50").is_finite())


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    c470.ARMS = ARMS
    c470.read_construct_rows = read_variant_rows  # ty: ignore[invalid-assignment]
    comparisons = tuple((f"A-pool|{v}", "A-pool|raw") for v in VARIANTS if v != "raw") + tuple(
        (f"A-pool|{v}", r) for v in VARIANTS for r in c470.REFERENCES
    )
    c449.ARMS, c449.REFERENCES, c449.COMPARISONS = ARMS, c470.REFERENCES, comparisons  # ty: ignore[invalid-assignment]
    c449.RESAMPLES = c470.RESAMPLES  # ty: ignore[invalid-assignment]
    cost = c470.build()
    conformal.WORK.mkdir(parents=True, exist_ok=True)
    cost.write_parquet(conformal.WORK / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(conformal.WORK / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=160):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()

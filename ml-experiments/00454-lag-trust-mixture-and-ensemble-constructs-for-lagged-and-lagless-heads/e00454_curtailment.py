"""Avoidable-curtailment cost of the experiment 454 constructs beside BG2 and manual_heuristic.

Reuses experiment 448's `curtailment` and experiment 449's per-series costing and cell bootstrap.
The constructs' forecasts are the mean over the construct's stored cells (quantile space), widened
by 1.15 about the median, converted to MW, and restricted to the rows with no labelled event that
BG2 and manual_heuristic also hold. Experiment 404's grid indices are converted to UTC times with
`e00087_common.grid_times`.

```bash
source /tmp/lt454_env.sh
uv run --no-sync python ml-experiments/00454-*/e00454_curtailment.py
```
"""

from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00454_build as build_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTHS: Final[tuple[str, ...]] = build_mod.MONTHS
ARMS: Final[tuple[str, ...]] = ("A0-inj", "S1", "B-mix", "S1-mix", "K2-S1", "K3-S1", "K3-A")
REFERENCES: Final[tuple[str, ...]] = ("BG2", "manual_heuristic")
WIDENING: Final[float] = 1.15
RESAMPLES: Final[int] = 20_000
LEVELS: Final[tuple[str, ...]] = base.LEVEL_NAMES
MEDIAN: Final[int] = 5
COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    ("S1-mix", "B-mix"), ("K2-S1", "B-mix"), ("K3-S1", "B-mix"), ("K3-A", "B-mix"),
    ("B-mix", "BG2"), ("K2-S1", "BG2"), ("S1-mix", "BG2"), ("K2-S1", "S1"),
    ("B-mix", "manual_heuristic"), ("K2-S1", "manual_heuristic"),
)  # fmt: skip


def read_construct_rows(
    *, arm: str, month: str, seeds: tuple[int, ...], factor: float, with_quantiles: bool
) -> pl.DataFrame | None:
    """Return the rows with no labelled event of a construct, as `base.read_arm_rows` does.

    `seeds` and `with_quantiles` are accepted for the signature only: the quantiles are always
    returned, as fractions of capacity, averaged over every stored cell of the construct.
    """
    del seeds, with_quantiles
    paths = sorted((build_mod.WORK_DIR / "predictions" / arm).glob(f"{month}_s*_w-1.parquet"))
    if not paths:
        return None
    frames = [pl.read_parquet(p) for p in paths]
    first = frames[0]
    for f in frames[1:]:
        assert f["target"].equals(first["target"]), "cell files differ in row order"
    q = np.mean([f.select(list(LEVELS)).to_numpy().astype(np.float64) for f in frames], axis=0)
    median = q[:, [MEDIAN]]
    q = np.sort(median + factor * (q - median), axis=1)
    actual = first["actual"].to_numpy().astype(np.float64)
    keep = ~first["win_real"].to_numpy() & ~first["tgt_post"].to_numpy() & np.isfinite(actual)
    grid = common.grid_times()
    out = first.select("primary", "lead_hours", "tgt_real").with_columns(
        actual=pl.Series(actual),
        power_fcst_init_time=grid.gather(first["origin"].to_numpy().astype(np.int64)),
        valid_time=grid.gather(first["target"].to_numpy().astype(np.int64)),
        **{n: pl.Series(q[:, i]) for i, n in enumerate(LEVELS)},
    )
    out = out.filter(pl.Series(keep)).join(
        base.sites_table().select("primary", "time_series_id", "label", "group", "capacity_mw"),
        on="primary",
    )
    return out.with_columns(time_series_id=pl.col("time_series_id").cast(pl.Int32))


def shared_keys(*, month: str, refs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the half-hours that every construct and every reference holds in the month."""
    keys: pl.DataFrame | None = None
    for arm in ARMS:
        rows = read_construct_rows(
            arm=arm, month=month, seeds=(), factor=WIDENING, with_quantiles=True
        )
        assert rows is not None, f"{arm} {month}"
        k = rows.select(*base.KEYS)
        keys = k if keys is None else keys.join(k, on=base.KEYS, how="semi")
    assert keys is not None
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
    out = build_mod.WORK_DIR / "scores"
    cost.write_parquet(out / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(out / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()

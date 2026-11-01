# ruff: noqa: E501
"""Avoidable-curtailment cost per test month and series, with a cell bootstrap (no fit).

Runs experiment 448's `curtailment` once per test month and series (signed-MW six, lead 12 to 36 h,
limits at f = 0.9, 0.75, and 0.5 of each series' maximum observed export), then resamples the
(month, series) cells with replacement for an interval on each arm's total and on paired differences.

```bash
source /tmp/e457_env.sh
uv run --no-sync python ml-experiments/00457-*/e00457_curtailment.py
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00452_run as run  # ty: ignore[unresolved-import]
import e00457_fit as fit457  # ty: ignore[unresolved-import]
import e00457_refs as refs457  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

_ORIGINAL_COST = base.metrics.avoidable_curtailment_cost
MONTHS = refs457.MONTHS
ITEMS = ("A-avg5", "Gin-avg5", "Gbase-avg5", "BG2", "manual_heuristic")
RESAMPLES = 20_000
COMPARISONS = (
    ("Gin-avg5", "A-avg5"), ("Gbase-avg5", "A-avg5"), ("Gin-avg5", "BG2"), ("A-avg5", "BG2"),
    ("Gin-avg5", "manual_heuristic"), ("A-avg5", "manual_heuristic"),
)  # fmt: skip


def per_series_cost(**kwargs: object) -> pl.DataFrame:
    """Cost each series on its own, skipping a series with no half-hours in the month."""
    series = kwargs["series"]
    pieces = []
    for row in series.iter_rows(named=True):  # ty: ignore[unresolved-attribute]
        one = pl.DataFrame([row], schema=series.schema)  # ty: ignore[unresolved-attribute]
        try:
            pieces.append(_ORIGINAL_COST(**{**kwargs, "series": one}))  # ty: ignore[invalid-argument-type]
        except ValueError:
            continue
    return pl.concat(pieces)


def build() -> pl.DataFrame:
    """Return the cost per month, forecast, series, and fraction."""
    base.run = run  # ty: ignore[invalid-assignment]
    base.metrics.avoidable_curtailment_cost = per_series_cost  # ty: ignore[invalid-assignment]
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    refs = {k: v for k, v in refs457.reference_frames(series=series).items() if k in ITEMS}
    parts = []
    for month in MONTHS:
        _, shared_base, _ = base.build_month(
            month=month, arms=("PT1",), refs=refs, seeds=run.SEEDS5
        )
        cost = base.curtailment(
            months=(month,), refs=refs, keys=shared_base.select(*base.KEYS), ctx=ctx,
            series=series, arms=(),
        )  # fmt: skip
        parts.append(cost.with_columns(month=pl.lit(month)))
    return pl.concat(parts)


def bootstrap(*, cost: pl.DataFrame) -> pl.DataFrame:
    """Return totals and paired differences (millions of pounds), 95% intervals."""
    ok = cost.filter(pl.col("status") == "ok").with_columns(
        cell=pl.col("month") + "/" + pl.col("label")
    )
    rng = np.random.default_rng(0)
    rows = []
    for fraction in sorted(ok["fraction"].unique(), reverse=True):
        wide = (
            ok.filter(pl.col("fraction") == fraction)
            .pivot(on="arm", index="cell", values="avoidable_cost_gbp")
            .sort("cell")
        )
        picks = rng.integers(0, wide.height, size=(RESAMPLES, wide.height))
        for item in ITEMS:
            v = wide[item].to_numpy()
            tot = v[picks].sum(axis=1) / 1e6
            rows.append({"fraction": fraction, "item": item, "estimate": v.sum() / 1e6,
                         "low": np.percentile(tot, 2.5), "high": np.percentile(tot, 97.5)})  # fmt: skip
        for a, b in COMPARISONS:
            d = wide[a].to_numpy() - wide[b].to_numpy()
            tot = d[picks].sum(axis=1) / 1e6
            rows.append({"fraction": fraction, "item": f"{a} minus {b}", "estimate": d.sum() / 1e6,
                         "low": np.percentile(tot, 2.5), "high": np.percentile(tot, 97.5)})  # fmt: skip
    return pl.DataFrame(rows)


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    cost = build()
    out = fit457.WORK_DIR / "scores"
    cost.write_parquet(out / "curtailment_cells.parquet")
    table = bootstrap(cost=cost)
    table.write_csv(out / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()

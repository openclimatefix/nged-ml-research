"""Avoidable-curtailment cost of experiment 462's arms per test month and series (no fit).

Experiment 449's script (`e00449_curtailment`), pointed at this experiment's forecasts: experiment
448's `build_month` reads `run.WORK_DIR/predictions/<arm>/<month>_s<seed>_w0.parquet`, so a stand-in
for `run` carries this experiment's work folder. Signed-MW six only (the cost is null for the
MVA-only eight). Arms: S3-amb, S3-rw-amb, U1n-2, and S1; references BG2 and manual_heuristic.

Runs experiment 448's `curtailment` once per test month, so each cost is one (month, series)
cell, then resamples the cells with replacement to give an interval on each arm's total and on
paired differences. The cost is additive over half-hours, so the four monthly runs sum to the
four-month total of `e00449_refs.py`.

```bash
source /tmp/pp462.sh; PYTHONPATH=$PP uv run --no-sync python ml-experiments/00462-*/e00462_curtailment.py
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_refs as refs449  # ty: ignore[unresolved-import]
import e00462_run as run462
import numpy as np
import polars as pl

_ORIGINAL_COST = base.metrics.avoidable_curtailment_cost
MONTHS = ("2026-06", "2026-01", "2025-10", "2026-03")
ARMS = ("S1", "U1n-2", "S3-amb", "S3-rw-amb")
REFERENCES = ("BG2", "manual_heuristic")
RESAMPLES = 20_000
COMPARISONS = (
    ("S3-amb", "S3-rw-amb"), ("S3-amb", "U1n-2"), ("S3-rw-amb", "U1n-2"), ("S3-amb", "S1"),
    ("S3-amb", "BG2"), ("U1n-2", "BG2"), ("S1", "BG2"), ("S3-amb", "manual_heuristic"),
)  # fmt: skip


class _Run:
    """Stand-in for experiment 449's run module: only the work folder is read."""

    WORK_DIR = run462.WORK_DIR


run = _Run()


def per_series_cost(**kwargs: object) -> pl.DataFrame:
    """Cost each series on its own, skipping a series with no half-hours in the month."""
    original = _ORIGINAL_COST
    series = kwargs["series"]
    pieces = []
    for row in series.iter_rows(named=True):  # ty: ignore[unresolved-attribute]
        one = pl.DataFrame([row], schema=series.schema)  # ty: ignore[unresolved-attribute]
        try:
            pieces.append(original(**{**kwargs, "series": one}))  # ty: ignore[invalid-argument-type]
        except ValueError:
            continue
    return pl.concat(pieces)


def build() -> pl.DataFrame:
    """Return the cost per month, arm, series, and fraction."""
    base.run = run  # ty: ignore[invalid-assignment]
    base.SEEDS = (0, 1)
    base.metrics.avoidable_curtailment_cost = per_series_cost  # ty: ignore[invalid-assignment]
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    refs = refs449.reference_frames(series=series)
    refs = {name: refs[name] for name in REFERENCES}
    parts = []
    for month in MONTHS:
        _, shared_base, _ = base.build_month(month=month, arms=ARMS, refs=refs, seeds=(0, 1))
        cost = base.curtailment(
            months=(month,), refs=refs, keys=shared_base.select(*base.KEYS), ctx=ctx,
            series=series, arms=ARMS,
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
        for arm in (*ARMS, *REFERENCES):
            v = wide[arm].to_numpy()
            tot = v[picks].sum(axis=1) / 1e6
            rows.append(
                {"fraction": fraction, "item": arm, "estimate": v.sum() / 1e6,
                 "low": np.percentile(tot, 2.5), "high": np.percentile(tot, 97.5)}
            )  # fmt: skip
        for a, b in COMPARISONS:
            d = wide[a].to_numpy() - wide[b].to_numpy()
            tot = d[picks].sum(axis=1) / 1e6
            rows.append(
                {"fraction": fraction, "item": f"{a} minus {b}", "estimate": d.sum() / 1e6,
                 "low": np.percentile(tot, 2.5), "high": np.percentile(tot, 97.5)}
            )  # fmt: skip
    return pl.DataFrame(rows)


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    cost = build()
    out = run.WORK_DIR / "figure_tables"
    out.mkdir(parents=True, exist_ok=True)
    cost.write_parquet(out / "curtailment_cells.parquet")
    table = bootstrap(cost=cost)
    table.write_csv(out / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()

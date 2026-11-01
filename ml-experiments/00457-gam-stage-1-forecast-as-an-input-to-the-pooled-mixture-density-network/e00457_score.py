# ruff: noqa: E501, PERF401
"""Score experiment 457's two arms against the stored arm A, BG2 and the GAM alone (no fit).

Rows are experiment 452's: the clean world-0 half-hours that experiment 449's `PT1` forecasts, scored
through experiment 448's row builder, so every arm and reference is scored on identical half-hours.
`PT1` only defines the row set. Contrasts are paired differences of per-row CRPS and tail pinball
loss (points of effective capacity), resampled in month-by-origin-day blocks 20,000 times. The
Bonferroni level counts `TESTS` = 6 tests (two arms times three gating row sets: the eight MVA-only
Primaries, cold nights at the eight, Christmas week at the eight). The minimum detectable effect is
experiment 428's formula on arm A's five single seeds.

```bash
source /tmp/e457_env.sh
uv run --no-sync python ml-experiments/00457-*/e00457_score.py
```
"""

import json
import math
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00452_run as run  # ty: ignore[unresolved-import]
import e00457_fit as fit457  # ty: ignore[unresolved-import]
import e00457_refs as refs457  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTHS: Final[tuple[str, ...]] = refs457.MONTHS
NEW: Final[tuple[str, ...]] = ("Gin-avg5", "Gbase-avg5")
POOLS: Final[tuple[str, ...]] = ("A-avg5", *NEW)
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("Gin-avg5", "A-avg5"), ("Gbase-avg5", "A-avg5"), ("Gin-avg5", "Gbase-avg5"),
    ("A-avg5", "BG2"), ("Gbase-avg5", "BG2"), ("Gin-avg5", "BG2"),
)  # fmt: skip
GATING: Final[tuple[tuple[str, str], ...]] = (
    ("MVA-only eight", "days 1 to 10"),
    ("MVA-only eight", "cold nights"),
    ("MVA-only eight", "Christmas week"),
)
REPORT: Final[tuple[tuple[str, str], ...]] = (
    ("signed-MW six", "days 1 to 10"), ("signed-MW six", "cold nights"),
    ("signed-MW six", "Christmas week"), ("D15 and D18", "days 1 to 10"),
    ("all 16", "days 1 to 10"), ("all 16", "cold nights"), ("all 16", "Christmas week"),
)  # fmt: skip
TESTS: Final[int] = 6
RESAMPLES: Final[int] = 20_000
TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
FACTORS: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
CHRISTMAS: Final[tuple[int, int]] = (22, 31)
OUT: Final = fit457.WORK_DIR / "scores"


def month_tables(*, refs: dict[str, pl.DataFrame], ctx: dict) -> dict[str, pl.DataFrame]:
    """Return each arm's and reference's per-row scores over the five folds, with slice flags."""
    base.run = run  # ty: ignore[invalid-assignment]
    per: dict[str, list[pl.DataFrame]] = {}
    for month in MONTHS:
        tables, shared_base, _ = base.build_month(
            month=month, arms=("PT1",), refs=refs, seeds=run.SEEDS5
        )
        flags = base.slice_flags(rows=shared_base, ctx=ctx).select(*base.KEYS, "cold_nights")
        for name, table in tables.items():
            per.setdefault(name, []).append(
                table.join(flags, on=base.KEYS).with_columns(month=pl.lit(month))
            )
    return {name: pl.concat(parts) for name, parts in per.items()}


def select_rows(*, table: pl.DataFrame, group: str, row_set: str) -> pl.DataFrame:
    """Return a table's rows for a Primary group and a row set."""
    part = table if group == "all 16" else table.filter(pl.col("group") == group)
    if row_set == "cold nights":
        part = part.filter(pl.col("cold_nights"))
    elif row_set == "Christmas week":
        day = pl.col("valid_time").dt.day()
        part = part.filter(
            (pl.col("month") == "2025-12")
            & (pl.col("valid_time").dt.month() == 12)
            & day.is_between(*CHRISTMAS)
        )
    return part


def contrast(
    *, tables: dict[str, pl.DataFrame], a: str, b: str, group: str, row_set: str, metric: str,
    rng: np.random.Generator,
) -> dict:  # fmt: skip
    """Return `a - b` for a metric with block-bootstrap intervals (negative favours `a`)."""
    left = select_rows(table=tables[a], group=group, row_set=row_set)
    right = tables[b].select(*base.KEYS, pl.col(metric).alias("m_b"))
    joined = left.select(*base.KEYS, "month", pl.col(metric)).join(right, on=base.KEYS)
    joined = joined.with_columns(
        block=pl.col("month") + pl.col("power_fcst_init_time").dt.date().cast(pl.String)
    )
    blocks = joined.group_by("block", "month").agg(
        pl.len().alias("n"), (pl.col(metric) - pl.col("m_b")).sum().alias("diff")
    )
    n, diff = blocks["n"].to_numpy().astype(float), blocks["diff"].to_numpy()
    draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
    boot = diff[draws].sum(1) / n[draws].sum(1)
    alpha = 100.0 * 0.05 / TESTS / 2.0
    by_month = blocks.group_by("month").agg(pl.col("n").sum(), pl.col("diff").sum())
    return {
        "contrast": f"{a} - {b}", "group": group, "row_set": row_set, "metric": metric,
        "n": int(n.sum()), "difference": diff.sum() / n.sum(),
        "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5)),
        "low_bonferroni": float(np.percentile(boot, alpha)),
        "high_bonferroni": float(np.percentile(boot, 100.0 - alpha)),
        "upper_one_sided_95": float(np.percentile(boot, 95.0)),
        "by_month": json.dumps({r["month"]: r["diff"] / r["n"] for r in by_month.iter_rows(named=True)}),
    }  # fmt: skip


def minimum_detectable_effect(*, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return experiment 428's minimum detectable effect from arm A's five single seeds, per group."""
    records = []
    for group in ("MVA-only eight", "signed-MW six", "D15 and D18", "all 16"):
        values = np.array(
            [
                [
                    select_rows(table=tables[f"A-s{s}"], group=group, row_set="days 1 to 10")
                    .filter(pl.col("month") == m)["crps"]
                    .mean()
                    for s in refs457.SEEDS
                ]
                for m in MONTHS
                if select_rows(table=tables["A-s0"], group=group, row_set="days 1 to 10")
                .filter(pl.col("month") == m)
                .height
            ]
        )
        folds, n_seeds = values.shape
        resid = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((resid**2).sum() / (folds * (n_seeds - 1)))
        records.append(
            {"group": group, "seed_sd": sd, "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(n_seeds)}
        )
    return pl.DataFrame(records)


def absolute(*, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return CRPS, tail pinball loss, and tail coverage per arm, group, and row set."""
    cols = ["crps", *[f"pinball_{t}" for t in TAILS], *[f"coverage_{t}" for t in TAILS]]
    records = []
    for name in (*POOLS, "BG2", "manual_heuristic", "climatology"):
        for group, row_set in (*GATING, *REPORT):
            part = select_rows(table=tables[name], group=group, row_set=row_set)
            records.append({"arm": name, "group": group, "row_set": row_set, "n": part.height,
                            **part.select([pl.col(c).mean() for c in cols]).row(0, named=True)})  # fmt: skip
    return pl.DataFrame(records)


def gam_alone(*, refs: dict[str, pl.DataFrame], rows: pl.DataFrame) -> pl.DataFrame:
    """Return the GAM's own median error against arm A's, on rows where the GAM exists (points)."""
    gam = fit457.gam_table().select(
        "time_series_id", "power_fcst_init_time", "valid_time", "gam_fraction"
    )
    out = []
    a, gin = refs["A-avg5"], refs["Gin-avg5"]
    joined = (
        rows.select(*base.KEYS, "actual", "capacity_mw", "group")
        .join(gam, on=base.KEYS)
        .join(a.select(*base.KEYS, pl.col("q50").alias("a_q50")), on=base.KEYS)
        .join(gin.select(*base.KEYS, pl.col("q50").alias("gin_q50")), on=base.KEYS)
    )
    for group in ("MVA-only eight", "signed-MW six"):
        part = joined.filter(pl.col("group") == group)
        cap = part["capacity_mw"].to_numpy()
        actual = part["actual"].to_numpy()
        out.append(
            {"group": group, "n": part.height,
             "mae_gam_points": 100 * float(np.abs(part["gam_fraction"].to_numpy() - actual).mean()),
             "mae_A_median_points": 100 * float(np.abs(part["a_q50"].to_numpy() / cap - actual).mean()),
             "mae_Gin_median_points": 100 * float(np.abs(part["gin_q50"].to_numpy() / cap - actual).mean())}
        )  # fmt: skip
    return pl.DataFrame(out)


def sweep(*, series: pl.DataFrame, ctx: dict) -> pl.DataFrame:
    """Return CRPS and outer-tail coverage of the pools at each widening factor."""
    records = []
    for factor in FACTORS:
        refs = {
            k: v
            for k, v in refs457.reference_frames(series=series, factor=factor).items()
            if k in POOLS
        }
        tables = month_tables(refs=refs, ctx=ctx)
        for name in POOLS:
            for group in ("MVA-only eight", "signed-MW six", "all 16"):
                part = select_rows(table=tables[name], group=group, row_set="days 1 to 10")
                records.append({"factor": factor, "arm": name, "group": group,
                                **part.select(pl.col("crps").mean(), pl.col("coverage_q05").mean(),
                                              pl.col("coverage_q95").mean()).row(0, named=True)})  # fmt: skip
    return pl.DataFrame(records)


def main() -> None:
    """Write the absolute scores, contrasts, minimum detectable effect, GAM-alone table, and sweep."""
    OUT.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    refs = refs457.reference_frames(series=series)
    tables = month_tables(refs=refs, ctx=ctx)
    mde = minimum_detectable_effect(tables=tables)
    print("minimum detectable effect (printed before any contrast):")
    print(mde)
    mde.write_csv(OUT / "mde.csv")
    absolute_table = absolute(tables=tables)
    absolute_table.write_csv(OUT / "absolute.csv")
    rng = np.random.default_rng(0)
    records = []
    for a, b in CONTRASTS:
        for group, row_set in (*GATING, *REPORT):
            for metric in ("crps", *[f"pinball_{t}" for t in TAILS]):
                records.append(
                    contrast(
                        tables=tables,
                        a=a,
                        b=b,
                        group=group,
                        row_set=row_set,
                        metric=metric,
                        rng=rng,
                    )
                )
    contrasts = pl.DataFrame(records)
    contrasts.write_csv(OUT / "contrasts.csv")
    gam_alone(refs=refs, rows=tables["A-avg5"]).write_csv(OUT / "gam_alone.csv")
    with pl.Config(tbl_rows=400, tbl_cols=14, tbl_width_chars=240, fmt_str_lengths=40):
        print(absolute_table.with_columns(pl.col(pl.Float64).round(3)))
        print(
            contrasts.filter(pl.col("metric") == "crps")
            .drop("by_month")
            .with_columns(pl.col(pl.Float64).round(3))
        )
    sweep(series=series, ctx=ctx).write_csv(OUT / "widening_sweep.csv")


if __name__ == "__main__":
    main()

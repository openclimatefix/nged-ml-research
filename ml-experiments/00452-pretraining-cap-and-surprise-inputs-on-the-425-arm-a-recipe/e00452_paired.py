# ruff: noqa: E501
"""Paired contrasts against the stored references, the Christmas-week slice, and the widening sweep.

Scores the five-seed quantile averages (`AS1L-avg5`, `AL-avg5`, `PT1-avg5`) beside experiment 425's
arm A (`425-A-avg5`) and BG2 on the half-hours every arm shares (experiment 448's row builder), then
resamples month-by-origin-day blocks 20,000 times for each difference. The Bonferroni level counts
`TESTS` tests, as the audit of experiment 449 did. Contrasts against `425-A-avg5` are confounded with
the fit loop (experiment 397 against experiment 428).

```bash
source /tmp/e452_env.sh
uv run --no-sync python ml-experiments/00452-*/e00452_paired.py
```
"""

from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00452_refs as refs452  # ty: ignore[unresolved-import]
import e00452_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03", "2025-12")
NEW: Final[tuple[str, ...]] = ("AS1L-avg5", "AL-avg5", "PT1-avg5")
REFERENCES: Final[tuple[str, ...]] = ("425-A-avg5", "BG2")
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("AS1L-avg5", "PT1-avg5"), ("AL-avg5", "PT1-avg5"), ("AS1L-avg5", "AL-avg5"),
    ("AS1L-avg5", "425-A-avg5"), ("AL-avg5", "425-A-avg5"), ("PT1-avg5", "425-A-avg5"),
    ("425-A-avg5", "BG2"), ("AS1L-avg5", "BG2"),
)  # fmt: skip
GROUPS: Final[tuple[str, ...]] = ("MVA-only eight", "signed-MW six", "D15 and D18", "all 16")
TESTS: Final[int] = 26
RESAMPLES: Final[int] = 20_000
FACTORS: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
CHRISTMAS: Final[tuple[int, int]] = (22, 31)


def slice_rows(*, table: pl.DataFrame, flags: pl.DataFrame, month: str) -> dict[str, pl.DataFrame]:
    """Return the row sets of a table: all rows, cold nights, and Christmas week (2025-12 only)."""
    joined = table.join(flags.select(*base.KEYS, "cold_nights"), on=base.KEYS)
    out = {"days 1 to 10": joined, "cold nights": joined.filter(pl.col("cold_nights"))}
    if month == "2025-12":
        day = pl.col("valid_time").dt.day()
        out["Christmas week"] = joined.filter(
            (pl.col("valid_time").dt.month() == 12) & day.is_between(*CHRISTMAS)
        )
    return out


def contrast(
    *, tables: dict[str, pl.DataFrame], a: str, b: str, group: str, rng: np.random.Generator
) -> dict:
    """Return the paired difference `a - b` in CRPS points and its block-bootstrap intervals."""
    left, right = tables[a], tables[b]
    joined = left.select(*base.KEYS, "month", "group", pl.col("crps")).join(
        right.select(*base.KEYS, pl.col("crps").alias("crps_b")), on=base.KEYS
    )
    if group != "all 16":
        joined = joined.filter(pl.col("group") == group)
    joined = joined.with_columns(
        block=pl.col("month") + pl.col("power_fcst_init_time").dt.date().cast(pl.String)
    )
    blocks = joined.group_by("block", "month").agg(
        pl.len().alias("n"), (pl.col("crps") - pl.col("crps_b")).sum().alias("diff")
    )
    n, diff = blocks["n"].to_numpy().astype(float), blocks["diff"].to_numpy()
    draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
    boot = diff[draws].sum(1) / n[draws].sum(1)
    alpha = 100.0 * 0.05 / TESTS / 2.0
    by_month = blocks.group_by("month").agg(pl.col("n").sum(), pl.col("diff").sum())
    return {
        "contrast": f"{a} - {b}", "group": group, "n": int(n.sum()), "difference": diff.sum() / n.sum(),
        "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5)),
        "low_bonferroni": float(np.percentile(boot, alpha)),
        "high_bonferroni": float(np.percentile(boot, 100.0 - alpha)),
        "by_month": {r["month"]: r["diff"] / r["n"] for r in by_month.iter_rows(named=True)},
    }  # fmt: skip


def main() -> None:
    """Write the paired contrast table and the widening sweep."""
    base.run = run  # ty: ignore[invalid-assignment]
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    refs = refs452.reference_frames(series=series)
    refs = {k: v for k, v in refs.items() if k in REFERENCES}
    per_month: dict[str, dict[str, pl.DataFrame]] = {}
    shared_parts = []
    for month in MONTHS:
        tables, shared_base, _ = base.build_month(
            month=month, arms=NEW, refs=refs, seeds=run.SEEDS5
        )
        flags = base.slice_flags(rows=shared_base, ctx=ctx)
        tagged = {
            name: t.with_columns(month=pl.lit(month)).join(
                shared_base.select(*base.KEYS, "valid_time", "power_fcst_init_time").unique(subset=base.KEYS),
                on=base.KEYS,
            ) if "valid_time" not in t.columns else t.with_columns(month=pl.lit(month))
            for name, t in tables.items()
        }  # fmt: skip
        per_month[month] = {"flags": flags, **tagged}
        shared_parts.append(month)
    rng = np.random.default_rng(0)
    records = []
    for fold_set, months in (("four folds", MONTHS[:4]), ("five folds", MONTHS)):
        for slice_name in ("days 1 to 10", "cold nights", "Christmas week"):
            pieces: dict[str, list[pl.DataFrame]] = {}
            for month in months:
                if slice_name == "Christmas week" and month != "2025-12":
                    continue
                flags = per_month[month].pop("flags_cached", None) or per_month[month]["flags"]
                for name in (*NEW, *REFERENCES):
                    sliced = slice_rows(table=per_month[month][name], flags=flags, month=month)
                    if slice_name in sliced:
                        pieces.setdefault(name, []).append(sliced[slice_name])
            if not pieces:
                continue
            tables = {k: pl.concat(v) for k, v in pieces.items()}
            for a, b in CONTRASTS:
                for group in GROUPS:
                    if slice_name == "Christmas week" and fold_set == "four folds":
                        continue
                    r = contrast(tables=tables, a=a, b=b, group=group, rng=rng)
                    records.append({"folds": fold_set, "slice": slice_name, **r})
    out = run.WORK_DIR / "figure_tables"
    out.mkdir(parents=True, exist_ok=True)
    result = pl.DataFrame(records).with_columns(
        pl.col("by_month").map_elements(str, return_dtype=pl.String)
    )
    result.write_csv(out / "paired_contrasts.csv")
    with pl.Config(tbl_rows=300, tbl_cols=12, fmt_str_lengths=40, tbl_width_chars=220):
        print(result.drop("by_month", "n").with_columns(pl.col(pl.Float64).round(3)))
    sweep(months=MONTHS[:4], refs=refs, series=series, out=out)


def sweep(*, months: tuple[str, ...], refs: dict, series: pl.DataFrame, out) -> None:  # noqa: ANN001
    """Write the CRPS of each quantile-averaged arm at each widening factor, per group."""
    records = []
    for arm in NEW:
        for factor in FACTORS:
            parts = []
            for month in months:
                rows = base.read_arm_rows(
                    arm=arm, month=month, seeds=run.SEEDS5, factor=factor, with_quantiles=False
                )
                if rows is not None:
                    parts.append(rows)
            table = pl.concat(parts)
            for group in GROUPS:
                part = table if group == "all 16" else table.filter(pl.col("group") == group)
                values = part.select(
                    pl.col("crps").mean(),
                    pl.col("coverage_q05").mean(),
                    pl.col("coverage_q95").mean(),
                ).row(0, named=True)
                records.append({"arm": arm, "factor": factor, "group": group, **values})
    sweep_table = pl.DataFrame(records)
    sweep_table.write_csv(out / "widening_sweep.csv")
    print(sweep_table.with_columns(pl.col(pl.Float64).round(4)))


if __name__ == "__main__":
    main()

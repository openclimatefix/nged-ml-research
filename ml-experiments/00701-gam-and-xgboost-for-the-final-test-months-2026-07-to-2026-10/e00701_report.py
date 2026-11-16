# ruff: noqa: D103, E501
"""Experiment 701: print the scores of the final-test months and of the 12 origin months together.

    PYTHONPATH=<experiment folders> \
        uv run --no-sync python ml-experiments/00701-*/e00701_report.py [--smoke]

Reads `scores/protocol_final/{rows.parquet,extra_coverage.csv}` of this experiment (origins 2026-07-01
to 2026-10-31) and `scores/protocol_year/{rows.parquet,extra_coverage.csv}` of experiment 700 (origins
2025-11-01 to 2026-06-30), and prints two tables of everything `result.md` quotes: the 4 final-test
months alone, and the 12 months of the two row sets together. The rows are the per-half-hour losses
of the nine arms on the half-hours that all of them share, so the 12-month score of an arm is the mean
of its rows in both sets. The two sets were each scored on their own shared rows, and the
`sunny_unit_shape` slice cuts each Primary's unit shape at its own quantile within a set, so that slice
is not comparable across the two sets and is not printed.

`--smoke` reads `scores/smoke_protocol_final/` (June 2026 from experiment 700's files) in place of this
experiment's rows, and experiment 700's rows of the origins before June in place of the 4 + 8 split,
so the combined table has to reproduce experiment 700's own pooled scores (the check prints their
difference). Writes `scores/protocol_final/report_12_months_by_group.csv` (no file in `--smoke`).
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
import e00701_protocol as proto
import polars as pl

ARMS: dict[str, str] = {
    "GAM": "GAM",
    "XGBoost BG2 (weekly W member)": "XGBoost BG2",
    "XGBoost with monthly refits only": "XGBoost monthly only",
    "Step 5 (quarterly refits)": "Step 5 quarterly",
    "Step 8 (quarterly refits)": "Step 8 quarterly",
    "Step 5 monthly (9a)": "Step 5 monthly",
    "Step 8 monthly (9b)": "Step 8 monthly",
    "Manual heuristic": "Manual heuristic",
    "Climatology": "Climatology",
}
SECTIONS: dict[str, str] = {
    "all": "All half-hours",
    "lead_day_1": "Day 1",
    "lead_days_8_10": "Days 8-10",
    "cold_nights": "Cold nights",
}
RULE_COLUMNS = ["mae_median", "crps", "twcrps"]
YEAR_ROWS: Path = proto.EXPERIMENT_700 / "scores" / "protocol_year"


def check_row_sets(*, new: pl.LazyFrame, old: pl.LazyFrame) -> None:
    """Stop unless the two row sets hold the same columns, dtypes, arm labels, and Primaries."""
    assert new.collect_schema() == old.collect_schema(), (
        new.collect_schema(),
        old.collect_schema(),
    )
    for column in ("arm", "label"):
        a = new.select(column).unique().collect()[column].sort().to_list()
        b = old.select(column).unique().collect()[column].sort().to_list()
        assert a == b, (column, a, b)
    arms = new.select("arm").unique().collect()["arm"].to_list()
    assert sorted(arms) == sorted(proto.ARM_NAMES), arms
    print(
        "row sets agree: same columns and dtypes, the same 9 arm labels, and the same",
        new.select("label").unique().collect().height,
        "Primaries; all losses in points of capacity",
    )
    per_arm = pl.concat(
        [
            new.group_by("arm").len().collect().with_columns(set=pl.lit("new")),
            old.group_by("arm").len().collect().with_columns(set=pl.lit("old")),
        ]
    )
    assert per_arm.group_by("set").agg(pl.col("len").n_unique())["len"].to_list() == [1, 1], (
        "an arm holds a different number of rows from the others"
    )


def print_scores(*, rows: pl.LazyFrame, header: str, labels: list[str] | None) -> None:
    """Print the mean MAE, CRPS, and twCRPS of each arm in each slice of `SECTIONS`."""
    chosen = rows if labels is None else rows.filter(pl.col("label").is_in(labels))
    for slice_name, label in SECTIONS.items():
        table = (
            chosen.filter(pl.col(slice_name))
            .group_by("arm")
            .agg(*[pl.col(m).mean() for m in RULE_COLUMNS], half_hours=pl.len())
            .collect()
        )
        if table.is_empty():
            print(f"-- {header}, {label}: no half-hours")
            continue
        print(f"-- {header}, {label} ({table['half_hours'].min():,} half-hours per arm)")
        for arm, short in ARMS.items():
            r = table.filter(pl.col("arm") == arm).row(0, named=True)
            print(
                f"  {short:22s} MAE {r['mae_median']:.2f}  CRPS {r['crps']:.2f}  twCRPS {r['twcrps']:.3f}"
            )


def pooled(*, rows: pl.LazyFrame, title: str) -> None:
    """Print the half-hours per arm, the pooled scores over every Primary, and the two groups."""
    counts = (
        rows.group_by("arm")
        .agg(half_hours=pl.len(), origins=pl.col("power_fcst_init_time").n_unique())
        .collect()
    )
    print(f"\n#### {title}")
    print(
        "half-hours per arm:",
        sorted(set(counts["half_hours"].to_list())),
        "; origins:",
        counts["origins"].max(),
        "; Primaries:",
        rows.select("label").unique().collect().height,
    )
    print_scores(rows=rows, header="pooled over the 15 Primaries", labels=None)
    for group_name, labels in cp.GROUPS.items():
        print_scores(rows=rows, header=group_name, labels=labels)
    by_day = (
        rows.with_columns(
            day=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() // 24
            ).cast(pl.Int32)
        )
        .group_by("arm", "day")
        .agg(pl.col("mae_median").mean())
        .collect()
    )
    print(
        "-- MAE of the median by forecast day, pooled over the 15 Primaries (three lowest per day)"
    )
    for day in range(14):
        col = by_day.filter(pl.col("day") == day).sort("mae_median")
        print(
            day,
            [
                (ARMS[r["arm"]], round(r["mae_median"], 2))
                for r in col.head(3).iter_rows(named=True)
            ],
        )


def coverage(*, files: list[Path]) -> None:
    """Print the coverage of the q05 to q95 and q01 to q99 bands, weighted by rows over the files."""
    frames = [pl.read_csv(f) for f in files]
    total = (
        pl.concat(frames)
        .group_by("arm", maintain_order=True)
        .agg(
            rows=pl.col("rows").sum(),
            q05_q95=(pl.col("q05_q95") * pl.col("rows")).sum() / pl.col("rows").sum(),
            q01_q99=(pl.col("q01_q99") * pl.col("rows")).sum() / pl.col("rows").sum(),
        )
    )
    print(
        "-- coverage of the q05 to q95 and q01 to q99 bands over the 13 Primaries of the two groups (nominal 0.90 and 0.98)"
    )
    print(total.with_columns(pl.col(pl.Float64).round(3)))


def main(*, smoke: bool) -> None:
    if smoke:
        new_path = proto.SMOKE.out
        old_filter = pl.col("power_fcst_init_time") < datetime(2026, 6, 1, tzinfo=UTC)
    else:
        new_path = proto.FINAL.out
        old_filter = pl.lit(value=True)
    new = pl.scan_parquet(new_path / "rows.parquet")
    old = pl.scan_parquet(YEAR_ROWS / "rows.parquet").filter(old_filter)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    check_row_sets(new=new, old=old)
    pooled(rows=new, title="Origin months of the new rows (final-test months unless --smoke)")
    coverage(files=[new_path / "extra_coverage.csv"])
    both = pl.concat([old, new])
    pooled(
        rows=both,
        title="Experiment 700's origin months and the new rows together (12 months unless --smoke)",
    )
    coverage(files=[YEAR_ROWS / "extra_coverage.csv", new_path / "extra_coverage.csv"])
    if smoke:
        combined = (
            both.filter(pl.col("all"))
            .group_by("arm")
            .agg(*[pl.col(m).mean() for m in RULE_COLUMNS])
            .collect()
        )
        year = (
            pl.scan_parquet(YEAR_ROWS / "rows.parquet")
            .filter(pl.col("all"))
            .group_by("arm")
            .agg(*[pl.col(m).mean() for m in RULE_COLUMNS])
            .collect()
        )
        gap = combined.join(year, on="arm", suffix="_700").select(
            *[(pl.col(m) - pl.col(f"{m}_700")).abs().max().alias(m) for m in RULE_COLUMNS]
        )
        print(
            "\nlargest difference of the combined pooled scores from experiment 700's own (all slice):"
        )
        print(gap)
    else:
        by_group = (
            both.group_by(
                "arm", pl.col("label").is_in(cp.GROUPS["signed-MW six"]).alias("signed_mw_six")
            )
            .agg(*[pl.col(m).mean() for m in RULE_COLUMNS], half_hours=pl.len())
            .collect()
        )
        by_group.write_csv(proto.FINAL.out / "report_12_months_by_group.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="split experiment 700's year at June")
    main(smoke=parser.parse_args().smoke)

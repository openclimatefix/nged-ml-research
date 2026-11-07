# ruff: noqa: E501
"""Experiment 619: score the lag-enabled network at the BSPs and GSPs against experiment 611's forecasts.

Reuses experiment 611's scoring functions (cleaned actuals, atom-mass CRPS, coverage, width, the
stored references of experiment 87, the month-block contrast with its Bonferroni level and minimum
detectable effect, and the Parquet export). Adds the mean absolute error of the median (q50), and
scores by day of lead time (day d is a lead of 24(d-1) to 24d hours). Every table is on the rows
that the lag-enabled network, experiment 611's Design 1, BG2, and the climatology all have.

Writes under this experiment's data folder: `scores/rows.parquet`, `scores/by_day.csv`,
`scores/by_day_series.csv`, `scores/contrasts.csv`, `scores/by_month.csv`,
`scores/row_counts.json`, and `export/lag_enabled_<month>.parquet` with `export/README.md`.

Usage: python e00619_score.py
"""

import itertools
import json
from pathlib import Path
from typing import Final

import e00611_pool as pool  # ty: ignore[unresolved-import]
import e00611_score as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00619-bsp-gsp-lag-enabled-network-for-the-bake-off"
W611: Final[Path] = pool.WORK
OUT: Final[Path] = ROOT / "scores"
EXPORT: Final[Path] = ROOT / "export"
ARM: Final[str] = "lag_enabled"
FORECASTS: Final[dict[str, tuple[Path, str]]] = {
    ARM: (ROOT, "P4E"),
    "design1": (W611, "O2"),
    "design2": (W611, "N1"),
}
"""Each forecast's work root and arm folder under `work/<group>/seed_preds/`."""
REFERENCES: Final[tuple[str, ...]] = ("bg2", "climatology", "manual_heuristic")
COMMON: Final[tuple[str, ...]] = (ARM, "design1", "bg2", "climatology")
"""The forecasts whose shared rows every table uses."""
SETS: Final[dict[str, tuple[int, int]]] = {"days1-3": (0, 72), "days4-10": (72, 240)}
"""The two lead sets of the pre-registered prediction, in hours (half-open)."""
KEYS: Final[list[str]] = ["label", "power_fcst_init_time", "valid_time"]


def load(*, arm: str, table: pl.DataFrame) -> pl.DataFrame:
    """Return the seed-averaged forecast of an arm for all 4 series in MW, with `month` and `lead_hours`."""
    root, folder_name = FORECASTS[arm]
    parts = []
    for group in pool.GROUPS:
        folder = root / "work" / group / "seed_preds" / folder_name
        by_month: dict[str, list[Path]] = {}
        for f in sorted(folder.glob("*_s*.parquet")):
            by_month.setdefault(f.stem.split("_s")[0], []).append(f)
        assert sorted(by_month) == list(s.MONTHS), (arm, group, sorted(by_month))
        for month, files in by_month.items():
            assert len(files) == 2, (arm, group, month, files)
            parts.append(s.seed_average(files=files).with_columns(month=pl.lit(month)))
    frame = pl.concat(parts).join(table.select("time_series_id", "capacity_mw"), on="time_series_id")
    frame = frame.with_columns(*[(pl.col(c) * pl.col("capacity_mw")).alias(c) for c in s.LEVELS])
    return frame.with_columns(
        lead_hours=(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
        )
    )


def score(*, name: str, frame: pl.DataFrame, act: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Return one row per scored forecast row, in points of effective capacity, with the day of lead time."""
    j = frame.join(act, on=["time_series_id", "valid_time"], how="inner")
    j = s.pinball_crps(frame=j).filter(pl.col("lead_hours") < 240)
    caps = dict(zip(table["time_series_id"], table["capacity_mw"], strict=True))
    return j.with_columns(
        arm=pl.lit(name),
        label=pl.col("time_series_id").replace_strict(s.LABELS, return_dtype=pl.String),
        cap=pl.col("time_series_id").replace_strict(caps, return_dtype=pl.Float64),
    ).with_columns(
        group=pl.col("label").replace(s.KIND),
        crps_points=pl.col("crps") / pl.col("cap") * 100,
        mae_points=(pl.col("q50") - pl.col("actual")).abs() / pl.col("cap") * 100,
        width_points=pl.col("width") / pl.col("cap") * 100,
        day=(pl.col("lead_hours") // 24).cast(pl.Int32) + 1,
    ).select(
        "arm", "label", "group", "month", "day", "power_fcst_init_time", "valid_time", "lead_hours",
        "actual", "crps_points", "mae_points", "width_points", "covered", "below_q95",
    )  # fmt: skip


def contrast(*, rows: pl.DataFrame, tr: str, ref: str, metric: str, n_tests: int) -> dict:
    """Return the month-block contrast of `tr` minus `ref` on the rows both have."""
    both = rows.filter(pl.col("arm") == tr).join(
        rows.filter(pl.col("arm") == ref), on=KEYS, suffix="_ref"
    )
    per_month = (
        both.group_by("month")
        .agg((pl.col(metric) - pl.col(f"{metric}_ref")).mean().alias("d"), pl.len().alias("n"))
        .sort("month")
    )
    return {
        "n_rows": int(per_month["n"].sum()),
        **s.month_block_contrast(diff=per_month, n_tests=n_tests),
    }


def main() -> None:
    """Score everything and write the tables and the export."""
    OUT.mkdir(parents=True, exist_ok=True)
    s.OUT, s.EXPORT = OUT, EXPORT
    pool.WORK = ROOT
    table = s.series()
    act = s.actuals(table=table)
    arms = {name: load(arm=name, table=table) for name in FORECASTS}
    for name, frame in s.references(table=table).items():
        arms[name] = frame.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
    scored = pl.concat([score(name=n, frame=f, act=act, table=table) for n, f in arms.items()])
    own_counts = scored.group_by("arm").len().sort("arm")
    common = (
        scored.filter(pl.col("arm").is_in(COMMON))
        .group_by(KEYS)
        .agg(pl.col("arm").n_unique().alias("n_arms"))
        .filter(pl.col("n_arms") == len(COMMON))
        .select(KEYS)
    )
    rows = scored.join(common, on=KEYS, how="semi")
    counts = {
        "own_rows_by_arm": dict(zip(own_counts["arm"], own_counts["len"], strict=True)),
        "common_rows_per_arm": int(common.height),
        "common_rows_by_group": dict(
            zip(*common.group_by(pl.col("label").replace(s.KIND).alias("g")).len().to_dict().values(), strict=True)
        ),
    }
    (OUT / "row_counts.json").write_text(json.dumps(counts, indent=1, default=int))
    rows.write_parquet(OUT / "rows.parquet")
    agg = [
        pl.col("crps_points").mean().alias("crps_points"),
        pl.col("mae_points").mean().alias("mae_points"),
        pl.col("covered").mean().alias("coverage_q05_q95"),
        pl.col("below_q95").mean().alias("share_below_q95"),
        pl.col("width_points").mean().alias("width_points"),
        pl.len().alias("n"),
    ]
    rows.group_by("arm", "group", "day").agg(*agg).sort("arm", "group", "day").write_csv(OUT / "by_day.csv")
    rows.group_by("arm", "label", "day").agg(*agg).sort("arm", "label", "day").write_csv(
        OUT / "by_day_series.csv"
    )
    rows.group_by("arm", "group", "label", "month").agg(*agg).sort("arm", "label", "month").write_csv(
        OUT / "by_month.csv"
    )
    sets = pl.concat(
        [
            rows.filter((pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi)).with_columns(
                set=pl.lit(name)
            )
            for name, (lo, hi) in SETS.items()
        ]
    )
    sets.group_by("arm", "group", "set").agg(*agg).sort("arm", "group", "set").write_csv(OUT / "by_set.csv")
    sets.group_by("arm", "label", "set").agg(*agg).sort("arm", "label", "set").write_csv(OUT / "by_set_series.csv")
    # Tails: coverage and pinball loss at q01, q05, q95, q99 are scored from the export by e00619_tails.py.
    others = ("design1", "bg2", "climatology", "design2")
    out = []
    n_tests = len(SETS) * 2 * len(others)
    for (set_name, (lo, hi)), group, ref in itertools.product(SETS.items(), ("BSP", "GSP"), others):
        sub = sets.filter(
            (pl.col("group") == group) & (pl.col("set") == set_name)
        ).drop("set")
        if ref == "design2":
            # Design 2 has its own rows; contrast it on the rows it shares with the lag-enabled network.
            pass
        for metric in ("crps_points", "mae_points"):
            out.append(
                {
                    "treatment": ARM, "reference": ref, "group": group, "set": set_name,
                    "metric": metric, "bonferroni_count": n_tests,
                    **contrast(rows=sub, tr=ARM, ref=ref, metric=metric, n_tests=n_tests),
                }
            )  # fmt: skip
    pl.DataFrame(out).write_csv(OUT / "contrasts.csv")
    # Per-day contrasts against Design 1 (descriptive: unadjusted 95% month-block intervals).
    daily = []
    for group, day in itertools.product(("BSP", "GSP"), range(1, 11)):
        sub = rows.filter((pl.col("group") == group) & (pl.col("day") == day))
        for metric in ("crps_points", "mae_points"):
            daily.append(
                {"group": group, "day": day, "metric": metric,
                 **contrast(rows=sub, tr=ARM, ref="design1", metric=metric, n_tests=1)}
            )  # fmt: skip
    pl.DataFrame(daily).write_csv(OUT / "contrasts_by_day.csv")
    names = [ARM]
    s.write_export(arms=arms, act=act, table=table, names=names)
    print(json.dumps(counts, indent=1, default=int))
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=220):
        print(pl.read_csv(OUT / "by_set.csv"))
        print(pl.read_csv(OUT / "contrasts.csv"))
    _ = np


if __name__ == "__main__":
    main()

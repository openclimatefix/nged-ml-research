# ruff: noqa: E501, D103
"""Experiment 632: score arms on the export's keys.

CRPS is twice the mean pinball loss over the 11 levels, in points of capacity (percent of the
Primary's capacity times 100 is not applied; the unit is share of capacity x 100). Rows are the
event-free rows of the roster export at the test origins and leads 0.5 to 168 h, scored against the
cleaned `observed_power`. An arm's file for a seed is `predictions/<arm>/<month>_s<seed>.parquet`;
the arm's forecast is the mean over seeds of each quantile, then sorted.

Usage: python e00632_score.py ARM...
"""

import sys
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPORT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00632-gru-attention-sequence-network-for-primary-demand"
)
MONTHS: Final = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
Q: Final = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
TAU: Final = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
KEYS: Final = ["time_series_id", "power_fcst_init_time", "valid_time"]


def arm_forecast(*, arm: str, month: str, seeds: tuple[int, ...] | None = None) -> pl.DataFrame:
    """Return the seed-mean, sorted quantiles of an arm for a month."""
    files = sorted((ROOT / "predictions" / arm).glob(f"{month}_s*.parquet"))
    if seeds is not None:
        files = [f for f in files if int(f.stem.rsplit("_s", 1)[1]) in seeds]
    assert files, f"{arm} {month}: no files"
    frames = [pl.read_parquet(f).sort(KEYS) for f in files]
    base = frames[0].select(KEYS + ["lead_hours"])
    for f in frames[1:]:
        assert f.select(KEYS).equals(base.select(KEYS)), f"{arm} {month}: keys differ between seeds"
    import numpy as np

    stack = np.stack([f.select(Q).to_numpy() for f in frames]).mean(axis=0)
    stack.sort(axis=1)
    return base.with_columns(*[pl.Series(q, stack[:, i]) for i, q in enumerate(Q)])


def scored_rows(*, month: str) -> pl.DataFrame:
    """Return the export's event-free keys, targets, groups, and capacities for a month (K3 file)."""
    sub = pl.read_csv(EXPORT / "substations.csv")
    df = pl.read_parquet(EXPORT / f"K3_{month}.parquet").select(
        KEYS + ["observed_power", "event_free", "unit"]
    )
    return df.join(
        sub.select("time_series_id", "primary_name", "group", "capacity"), on="time_series_id"
    )


def score(*, arm_df: pl.DataFrame, month: str) -> pl.DataFrame:
    """Return the per-row loss frame (joined to the export keys; asserts identical key sets)."""
    rows = scored_rows(month=month).filter(
        pl.col("event_free")
        & pl.col("observed_power").is_not_null()
        & pl.col("observed_power").is_finite()
    )
    j = rows.join(arm_df.drop("lead_hours"), on=KEYS, how="inner")
    y = pl.col("observed_power")
    pin = [
        pl.max_horizontal(t * (y - pl.col(q)), (t - 1) * (y - pl.col(q)))
        for q, t in zip(Q, TAU, strict=True)
    ]
    return j.with_columns(
        crps=2 * pl.mean_horizontal(*pin) / pl.col("capacity") * 100,
        mae=(y - pl.col("q50")).abs() / pl.col("capacity") * 100,
        hit=((pl.col("q05") <= y) & (y <= pl.col("q95"))).cast(pl.Float64),
        above90=(y > pl.col("q90")).cast(pl.Float64),
        day=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours().floordiv(24)
        + 1,
        month=pl.lit(month),
    )


def main() -> None:
    arms = sys.argv[1:] or ["P2-pre", "K3-pre"]
    for arm in arms:
        parts = []
        for m in MONTHS:
            if not list((ROOT / "predictions" / arm).glob(f"{m}_s*.parquet")):
                continue
            a = arm_forecast(arm=arm, month=m)
            s = score(arm_df=a, month=m)
            parts.append(s)
            print(f"{arm} {m}: forecast rows {a.height}, scored rows {s.height}")
        if not parts:
            continue
        df = pl.concat(parts)
        print(
            df.group_by("group", "day")
            .agg(pl.col("crps").mean(), pl.col("hit").mean(), pl.col("above90").mean())
            .sort("group", "day")
            .filter(pl.col("day") <= 7)
        )


if __name__ == "__main__":
    main()

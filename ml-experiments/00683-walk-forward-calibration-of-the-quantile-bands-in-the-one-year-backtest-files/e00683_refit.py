"""Band width and median level of days 10 to 14 across the month-start refits (CPU only).

PYTHONPATH=ml-experiments/00668-*:ml-experiments/00683-* uv run --no-sync python \
    ml-experiments/00683-*/e00683_refit.py

Reads the stored normal-running Primaries file (two-seed average) and the per-seed fits of
experiment 667. Width is the q95 minus q05 of the half-hours at leads of 240 to 336 hours, in
percent of capacity; level is the median of `q50` over the same half-hours. Per origin the width is
the mean over those half-hours. The step at a refit is the median of the first 7 origins of the
month over the median of the last 7 origins of the month before. Writes `tables/refit_*.csv`.
"""

from itertools import pairwise
from typing import Final

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00683_calibrate import EXPERIMENT_DIR

SERIES: Final[tuple[int, ...]] = (11, 13, 8, 15, 17, 18, 24, 25)
EXP667: Final = (
    lib.FOLDER.parent / "00667-one-year-backtest-of-the-two-output-demand-network-at-primaries"
)
LONG_LEAD_H: Final[float] = 240.0


def long_lead_summary(*, frame: pl.DataFrame, capacity: pl.DataFrame) -> pl.DataFrame:
    """Return per series and origin: mean days 10 to 14 band width and median q50, in % capacity."""
    out = (
        frame.join(capacity.select("time_series_id", "capacity"), on="time_series_id")
        .with_columns(
            lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
        )
        .filter(pl.col("lead") >= LONG_LEAD_H)
        .group_by("time_series_id", "power_fcst_init_time")
        .agg(
            width=((pl.col("q95") - pl.col("q05")) * 100 / pl.col("capacity")).mean(),
            level=(pl.col("q50") * 100 / pl.col("capacity")).median(),
        )
        .sort("time_series_id", "power_fcst_init_time")
    )
    return out.with_columns(origin_month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))


def steps(*, summary: pl.DataFrame) -> pl.DataFrame:
    """Return the width ratio and level change at each month start, per series."""
    rows = []
    for tid in SERIES:
        part = summary.filter(pl.col("time_series_id") == tid).sort("power_fcst_init_time")
        months = part["origin_month"].unique(maintain_order=True).to_list()
        for prev, cur in pairwise(months):
            before = part.filter(pl.col("origin_month") == prev).tail(7)
            after = part.filter(pl.col("origin_month") == cur).head(7)
            rows.append(
                {
                    "time_series_id": tid,
                    "refit_month": cur,
                    "width_before": before["width"].median(),
                    "width_after": after["width"].median(),
                    "width_ratio": after["width"].median() / before["width"].median(),  # ty: ignore[unsupported-operator]
                    "level_change_pts": after["level"].median() - before["level"].median(),  # ty: ignore[unsupported-operator]
                }
            )
    return pl.DataFrame(rows)


def seed_spread(*, capacity: pl.DataFrame) -> pl.DataFrame:
    """Return per series and fold the difference between the two seeds' days 10 to 14 width."""
    rows = []
    for path in sorted(EXP667.glob("P7_D50_*/work/fits/P5/*_d0_s0.parquet")):
        other = path.with_name(path.name.replace("_s0.parquet", "_s1.parquet"))
        month = path.name[:7]
        widths = {}
        for seed, p in (("s0", path), ("s1", other)):
            frame = pl.read_parquet(p)
            summary = long_lead_summary(frame=frame, capacity=capacity)
            widths[seed] = summary.group_by("time_series_id").agg(w=pl.col("width").median())
        both = widths["s0"].join(widths["s1"], on="time_series_id", suffix="_s1")
        rows.append(both.with_columns(refit_month=pl.lit(month)))
    return pl.concat(rows).with_columns(seed_diff_ratio=(pl.col("w") / pl.col("w_s1")).abs())


def main() -> None:
    """Write the refit tables."""
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    stored = pl.read_parquet(lib.FOLDER / "primaries.parquet").filter(
        pl.col("NRA_or_prevailing") == "NRA", pl.col("time_series_id").is_in(list(SERIES))
    )
    summary = long_lead_summary(frame=stored, capacity=capacity)
    table = steps(summary=summary)
    out = EXPERIMENT_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "refit_steps.csv")
    pl.Config.set_tbl_rows(200)
    print(table.with_columns(pl.col(pl.Float64).round(3)))
    seeds = seed_spread(capacity=capacity).filter(pl.col("time_series_id").is_in(list(SERIES)))
    seeds.write_csv(out / "refit_seed_spread.csv")
    print(seeds.with_columns(pl.col(pl.Float64).round(3)).sort("time_series_id", "refit_month"))
    summary.write_parquet(out / "refit_origin_summary.parquet")
    print(
        "train rows per fold: see checks_fit_P7_D50_s0-1.json in each fold folder of experiment 667"
    )
    _ = np  # keep numpy imported for later figure code


if __name__ == "__main__":
    main()

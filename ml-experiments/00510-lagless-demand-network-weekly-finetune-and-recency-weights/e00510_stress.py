# ruff: noqa: E501
"""The injected-shift leak of experiment 510's stress runs.

For each variant and fold, the stress fit (refit week 3, trained on targets shifted by +3 or -3
standard deviations of the Primary's 7-day mean over the 7 days before the refit) is compared with the
matched uninjected fit of the main run. The leak is the injected sign times the change in the median
forecast of the following week, in points of capacity (percentage of each Primary's capacity),
averaged over the two signs, the 16 Primaries, and the week's forecast half-hours. Descriptive: one
seed, five folds.
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00510-lagless-demand-network-weekly-finetune-and-recency-weights/work"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
TAGS: Final[dict[str, str]] = {"hl182": "L-W", "hl90": "L-Ww", "hl90D": "L-Ww-D", "hl90O": "L-Ww-O"}


def leak(*, tag: str, month: str) -> dict:
    """Return the leak of one variant and fold, overall and by lead band."""
    control = pl.read_parquet(W / "fits" / tag / f"{month}_w2_s0.parquet").select(
        *KEYS, c=pl.col("q50")
    )
    out = {}
    per_sign = []
    for name, sign in (("p", 1), ("n", -1)):
        stressed = pl.read_parquet(W / "fits" / tag / f"{month}_w2_s0_stress{name}.parquet").select(
            *KEYS, s=pl.col("q50")
        )
        joined = control.join(stressed, on=KEYS).with_columns(
            lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
            shift=sign * 100.0 * (pl.col("s") - pl.col("c")),
        )
        per_sign.append(joined)
    out["rows"] = per_sign[0].height
    for label, cond in {
        "all": pl.lit(True),
        "day1": pl.col("lead") <= 24,
        "days2_3": (pl.col("lead") > 24) & (pl.col("lead") <= 72),
        "days4_10": pl.col("lead") > 72,
    }.items():
        out[label] = float(np.mean([p.filter(cond)["shift"].mean() for p in per_sign]))
    return out


def main() -> None:
    """Print and write the leak table."""
    records = []
    for tag, arm in TAGS.items():
        for month in MONTHS:
            records.append({"arm": arm, "month": month} | leak(tag=tag, month=month))  # noqa: PERF401
    table = pl.DataFrame(records)
    summary = table.group_by("arm", maintain_order=True).agg(
        pl.col("all").mean(), pl.col("all").min().alias("min"), pl.col("all").max().alias("max"),
        pl.col("day1").mean(), pl.col("days2_3").mean(), pl.col("days4_10").mean(),
    )  # fmt: skip
    table.write_csv(W / "scores" / "stress_leak_by_fold.csv")
    summary.write_csv(W / "scores" / "stress_leak.csv")
    with pl.Config(tbl_rows=40):
        print(table.with_columns(pl.col(pl.Float64).round(3)))
        print(summary.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()

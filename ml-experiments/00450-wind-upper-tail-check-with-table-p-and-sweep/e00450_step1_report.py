"""Experiment 450, step 1 report: coverage by farm with month-block intervals.

Reads `work/step1_cells.parquet`. The pooled number is a row-weighted share over the three farms.
The upper-tail rule: share above q99 at most 1.5% (coverage at least 0.985); q01 and q05 coverage
may move away from the raw arm's distance to nominal by at most 0.02. Intervals: 95% month-block
bootstrap (2,000 resamples of the 11 scored months). Output: `work/step1_report.parquet`.
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00450-wind-upper-tail-check-with-table-p-and-sweep" / "work"
)
FARM_LABEL: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}
NOMINAL: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q99": 0.99}
RESAMPLES: Final[int] = 2000
LIMIT_ABOVE_Q99: Final[float] = 0.015
MAX_HARM: Final[float] = 0.02


def main() -> None:
    """Write and print the coverage table."""
    cells = pl.read_parquet(WORK_DIR / "step1_cells.parquet").with_columns(
        farm=pl.col("site").replace(FARM_LABEL)
    )
    months = sorted(cells["month"].unique())
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(months), size=(RESAMPLES, len(months)))
    rows: list[dict] = []
    keys = cells.select("arm", "rowset", "variant").unique().sort("arm", "rowset", "variant")
    for arm, rowset, variant in keys.iter_rows():
        part = cells.filter(
            (pl.col("arm") == arm) & (pl.col("rowset") == rowset) & (pl.col("variant") == variant)
        )
        for farm in ("Farm A", "Farm B", "Farm C", "all farms"):
            sub = part if farm == "all farms" else part.filter(pl.col("farm") == farm)
            by_month = (
                sub.group_by("month").agg(pl.col("n").sum(), pl.col("^hit_.*$").sum()).sort("month")
            )
            n = by_month["n"].to_numpy().astype(float)
            row = {
                "arm": arm,
                "rowset": rowset,
                "variant": variant,
                "farm": farm,
                "n": int(n.sum()),
            }
            for q in NOMINAL:
                h = by_month[f"hit_{q}"].to_numpy().astype(float)
                cov = h.sum() / n.sum()
                boot = h[draws].sum(axis=1) / n[draws].sum(axis=1)
                row[f"cov_{q}"] = cov
                row[f"lo_{q}"], row[f"hi_{q}"] = np.quantile(boot, [0.025, 0.975])
            rows.append(row)
    table = pl.DataFrame(rows)
    # Harm against the raw arm: change in distance to nominal at q01 and q05.
    raw = table.filter(pl.col("variant") == "raw").select(
        "arm", "rowset", "farm", raw01=pl.col("cov_q01"), raw05=pl.col("cov_q05")
    )
    table = table.join(raw, on=["arm", "rowset", "farm"]).with_columns(
        above_q99=1.0 - pl.col("cov_q99"),
        above_q99_lo=1.0 - pl.col("hi_q99"),
        above_q99_hi=1.0 - pl.col("lo_q99"),
        harm_q01=(pl.col("cov_q01") - 0.01).abs() - (pl.col("raw01") - 0.01).abs(),
        harm_q05=(pl.col("cov_q05") - 0.05).abs() - (pl.col("raw05") - 0.05).abs(),
    )
    table = table.with_columns(
        passes=(pl.col("above_q99") <= LIMIT_ABOVE_Q99)
        & (pl.col("harm_q01") <= MAX_HARM)
        & (pl.col("harm_q05") <= MAX_HARM)
    )
    table.write_parquet(WORK_DIR / "step1_report.parquet")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    show = table.filter(pl.col("farm") == "all farms").select(
        "arm", "rowset", "variant", "n",
        pl.col("above_q99").round(4), pl.col("above_q99_lo").round(4),
        pl.col("above_q99_hi").round(4),
        pl.col("cov_q01").round(4), pl.col("cov_q05").round(4), pl.col("harm_q01").round(4),
        pl.col("harm_q05").round(4), "passes",
    )  # fmt: skip
    print(show)


if __name__ == "__main__":
    main()

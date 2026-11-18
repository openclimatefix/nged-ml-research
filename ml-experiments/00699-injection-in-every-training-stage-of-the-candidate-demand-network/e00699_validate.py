"""Experiment 699: validate the exported forecast files of every chain.

Usage: python e00699_validate.py

For each `backtests/*.parquet` under `roots/`, prints the row count, the number of duplicate keys
(`time_series_id`, `nwp_init_time`, `valid_time`, `seed`), the null and NaN quantile cells, the
number of crossings between adjacent quantiles, and the time zone and range of the valid times.
"""

from itertools import pairwise

import polars as pl
from e00699_common import ROOT

KEYS = ["time_series_id", "nwp_init_time", "valid_time", "seed"]
QUANTILES = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]


def main() -> None:
    """Print one validation block for each exported forecast file."""
    for path in sorted((ROOT / "roots").glob("*/backtests/*.parquet")):
        frame = pl.read_parquet(path)
        crossings = sum(
            frame.filter(pl.col(low) > pl.col(high)).height for low, high in pairwise(QUANTILES)
        )
        print(path.relative_to(ROOT / "roots"), frame.height)
        print("  duplicate keys", frame.height - frame.select(KEYS).unique().height)
        print(
            "  null quantile cells",
            sum(frame[q].null_count() for q in QUANTILES),
            "NaN",
            sum(frame[q].is_nan().sum() for q in QUANTILES),
        )
        print("  adjacent-quantile crossings", crossings)
        print(
            "  valid_time",
            frame.schema["valid_time"],
            frame["valid_time"].min(),
            frame["valid_time"].max(),
        )


if __name__ == "__main__":
    main()

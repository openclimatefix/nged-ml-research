# ruff: noqa: E501, I001
"""Experiment 701, GAM side: check that the export reproduces experiment 700's file for stored months.

    PYTHONPATH=<folders of 377, 700, 701> DATA_PATH_INTERNAL=<real data folder> \
        uv run --no-sync python ml-experiments/00701-*/e00701_gam_check_export.py D5 2026-06

Assembles default 3 for the named Primaries and one origin month from the sandbox (which holds
experiment 700's rows for months to 2026-06 until `e00701_gam_run.sh` appends), converts it with
`e00700_export.to_contract`, and compares it with experiment 700's `backtests/gam_year_sp1.parquet`
on (Primary, origin, valid time): row counts and the largest absolute difference of the 11 quantiles.
"""

import sys
from datetime import UTC, datetime

import e00701_gam_extend377  # noqa: F401  (dates and paths first)
import e00701_gam_extend as ext
import e00701_gam_export as export
import e00700_export as e700  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def main() -> None:
    """Print the comparison for each named Primary."""
    month = datetime.strptime(sys.argv[-1], "%Y-%m").replace(tzinfo=UTC)
    sites = sys.argv[1:-1]
    end = month.replace(year=month.year + (month.month == 12), month=month.month % 12 + 1)
    new = e700.to_contract(frame=export.gam_rows(start=month, end=end, sites=sites), unit="points")
    old = pl.read_parquet(
        ext.REAL_DATA / "ml-experiments" / e700.NAME / "backtests" / "gam_year_sp1.parquet"
    )
    series = e700.adapter.series_table().filter(pl.col("label").is_in(sites))
    ids = series["time_series_id"].cast(pl.Int8).to_list()
    old = old.filter(
        pl.col("time_series_id").is_in(ids)
        & (pl.col("power_fcst_init_time") >= month)
        & (pl.col("power_fcst_init_time") < end + (end - month))
    )
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = new.join(old, on=keys, how="full", suffix="_old", coalesce=True)
    both = joined.filter(pl.col("q50").is_not_null() & pl.col("q50_old").is_not_null())
    diff = max(
        float(
            np.nanmax(
                np.abs(
                    both[q].to_numpy().astype(np.float64)
                    - both[f"{q}_old"].to_numpy().astype(np.float64)
                )
            )
        )
        for q in e700.Q
    )
    print(
        f"{sites} {month:%Y-%m}: new rows {new.height}, 700 rows {old.filter(pl.col('power_fcst_init_time') < end).height}, joined {both.height}, only new {joined.filter(pl.col('q50_old').is_null()).height}, only 700 {joined.filter(pl.col('q50').is_null()).height}, max abs diff over 11 quantiles (MW) {diff}"
    )
    assert diff == 0.0


if __name__ == "__main__":
    main()

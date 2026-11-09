# ruff: noqa: D103, E501, PT018
"""Experiment 667: write `primaries.parquet` from the NRA rows and the prevailing rows, and check it.

Usage: python e00667_merge.py   (reads `exports/NRA_MONTH.parquet` and `exports/prevailing_MONTH.parquet`)
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import e00667_export as exp
import e00667_score as sc
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10" / "primaries.parquet"
SCHEMA: Final = {**exp.CONTRACT_SCHEMA, "NRA_or_prevailing": pl.String}
KEY: Final = ["time_series_id", "power_fcst_init_time", "valid_time", "NRA_or_prevailing"]


def main() -> None:
    parts = []
    for m in sc.MONTHS:
        nra = pl.read_parquet(exp.ROOT / "exports" / f"NRA_{m}.parquet").with_columns(
            NRA_or_prevailing=pl.lit("NRA")
        )
        prev = pl.read_parquet(exp.ROOT / "exports" / f"prevailing_{m}.parquet")
        assert prev.columns == nra.columns, (m, prev.columns)
        assert set(prev["NRA_or_prevailing"].unique()) == {"prevailing"}
        parts += [nra, prev.cast(dict(nra.schema))]  # ty: ignore[no-matching-overload]
    df = pl.concat(parts).sort(KEY)
    assert df.schema == pl.Schema(SCHEMA), df.schema
    n_origin = 365
    assert df.height == 16 * n_origin * 672 * 2, df.height
    assert df.height == df.unique(KEY).height, "duplicate keys"
    q = df.select(exp.QCOLS).to_numpy()
    assert np.isfinite(q).all(), "non-finite quantile"
    assert (np.diff(q, axis=1) >= 0).all(), "quantiles cross"
    origins = sorted(df["power_fcst_init_time"].unique().to_list())
    first = datetime(2025, 11, 1, 12, tzinfo=UTC)
    assert origins == [first + timedelta(days=d) for d in range(n_origin)], (
        "origin days are not 365 consecutive 12:00 UTC"
    )
    assert (df["nwp_init_time"] == df["power_fcst_init_time"].dt.truncate("1d")).all()
    ok = df.group_by("time_series_id", "power_fcst_init_time", "NRA_or_prevailing").agg(
        n=pl.len(),
        first=pl.col("valid_time").min() - pl.col("power_fcst_init_time").first(),
        last=pl.col("valid_time").max() - pl.col("power_fcst_init_time").first(),
    )
    assert (
        (ok["n"] == 672).all()
        and (ok["first"] == timedelta(0)).all()
        and (ok["last"] == timedelta(days=14) - timedelta(minutes=30)).all()
    )
    assert df["time_series_id"].n_unique() == 16
    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.rename(OUT)
    print("wrote", OUT, df.height, "rows; all checks passed")


if __name__ == "__main__":
    main()

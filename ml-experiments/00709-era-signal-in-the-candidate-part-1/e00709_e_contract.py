# ruff: noqa: E501, PT018
"""Phase E's contract rows: experiment 667's `contract_rows` with series-origins that may be incomplete.

The stored rows of experiment 667 before 2025-07-01 keep only half-hours with an observed target,
so a series-origin can hold fewer than 672 forecast rows (14 days of half-hours), for example when
the telemetry has a gap. Experiment 667's `contract_rows` asserts that every series-origin holds
all 672. This copy keeps every other assertion and requires only that no series-origin holds more
than 672 rows.
"""

from datetime import timedelta

import e00667_export as exp  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def contract_rows(*, mean: pl.DataFrame) -> pl.DataFrame:
    """Relabel the network origin (09:00) as the contract origin (12:00) and keep 12:00 to +14 d.

    Args:
        mean: The seed-mean forecasts at the network origin, in capacity units.

    Returns:
        The contract rows, in MW, with at most 14 * 48 rows per series-origin.
    """
    assert (mean["power_fcst_init_time"].dt.hour() == exp.NETWORK_ORIGIN_HOUR).all()
    assert (mean["power_fcst_init_time"].dt.minute() == 0).all()
    origin = pl.col("power_fcst_init_time") + timedelta(
        hours=exp.CONTRACT_HOUR - exp.NETWORK_ORIGIN_HOUR
    )
    cap = exp.capacity_mw().select(pl.col("time_series_id").cast(pl.Int64), "capacity_mw")
    out = (
        mean.with_columns(
            contract_origin=origin, time_series_id=pl.col("time_series_id").cast(pl.Int64)
        )
        .filter(
            (pl.col("valid_time") >= pl.col("contract_origin"))
            & (pl.col("valid_time") < pl.col("contract_origin") + timedelta(days=14))
        )
        .join(cap, on="time_series_id", how="left")
        .with_columns(
            *[
                (pl.col(q).cast(pl.Float64) * pl.col("capacity_mw")).cast(pl.Float32)
                for q in exp.QCOLS
            ]
        )
        .select(
            pl.col("time_series_id").cast(pl.Int8),
            pl.col("valid_time"),
            pl.col("contract_origin").dt.truncate("1d").alias("nwp_init_time"),
            pl.col("contract_origin").alias("power_fcst_init_time"),
            *exp.QCOLS,
        )
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )
    assert out.schema == pl.Schema(exp.CONTRACT_SCHEMA), out.schema
    assert (
        out["q99"].is_finite().all()
        and out.select(pl.all_horizontal(pl.col(exp.QCOLS).is_not_null())).to_series().all()
    )
    q = out.select(exp.QCOLS).to_numpy()
    assert (np.diff(q, axis=1) >= 0).all(), "quantiles cross"
    per = out.group_by("time_series_id", "power_fcst_init_time").len()["len"]
    assert per.max() <= exp.HALF_HOURS, per.max()
    assert out.height == out.unique(["time_series_id", "power_fcst_init_time", "valid_time"]).height
    assert out["time_series_id"].n_unique() == 16
    return out

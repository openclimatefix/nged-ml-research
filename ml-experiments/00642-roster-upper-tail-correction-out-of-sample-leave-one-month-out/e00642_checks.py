# ruff: noqa: E501
"""Experiment 642: the audit checks and the conformal-noise diagnostic, printed for the write-up."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00642_arms as arms
import e00642_score as score
import numpy as np
import polars as pl
from e00630_common import NETWORKS, load  # ty: ignore[unresolved-import]
from scipy.stats import norm
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00642-roster-upper-tail-correction-out-of-sample-leave-one-month-out"
)
ARMS: Final[tuple[str, ...]] = ("base", "S", "B", "C", "Q", "M")


def known_answer() -> None:
    """Rolling conformal on stationary synthetic data with a too-narrow forecast must fix q90."""
    rng = np.random.default_rng(1)
    rows = []
    for day in range(90):
        origin = datetime(2025, 10, 1, tzinfo=UTC) + timedelta(days=day)
        for half_hour in range(48):
            valid = origin + timedelta(hours=9 + half_hour * 0.5)
            rows.append(
                (1, "0-24 h", origin.date(), valid, rng.normal(), True, *(norm.ppf(arms.TAU) * 0.8))
            )
    cols = [
        "time_series_id",
        "band",
        "origin_date",
        "valid_time",
        "observed_power",
        "event_free",
        *score.Q,
    ]
    df = pl.DataFrame(rows, schema=cols, orient="row").with_columns(
        pl.col("valid_time").cast(pl.Datetime("us", "UTC"))
    )
    d = df.join(score.rolling_tables(df=df), on=["time_series_id", "band", "origin_date"])
    c = d.filter(pl.col("adj2").is_not_null())
    y = c["observed_power"]
    print(
        "known answer: q90 exceedance before",
        round((y > c["q90"]).mean(), 4),  # ty: ignore[no-matching-overload]
        "after",
        round((y > c["q90"] + c["adj2"]).mean(), 4),  # ty: ignore[no-matching-overload]
        "(nominal 0.10)",
    )


def main() -> None:
    """Print the median error per arm and the conformal adjustment's noise."""
    known_answer()
    for network in NETWORKS:
        df = pl.read_parquet(OUT / f"scored_{network}.parquet").filter(pl.col("event_free"))
        print(
            f"\n== {network}: mean error of the median (observed minus q50, MW) and its MAE, event-free"
        )
        print(
            df.select(
                *[pl.col(f"gap_{a}").mean().round(4).alias(f"mean_{a}") for a in ARMS],
                *[pl.col(f"gap_{a}").abs().mean().round(4).alias(f"mae_{a}") for a in ARMS],
            )
        )
    full = load(network="Candidate")
    tables = score.rolling_tables(df=full)
    d = score.attach(df=full, tables=tables).filter(pl.col("event_free"))
    corrected = d.filter(pl.col("adj2").is_not_null())
    y = corrected["observed_power"]
    print(
        "\n== Candidate, arm C diagnostic (event-free rows with a pool of at least 300 residuals)"
    )
    print("rows", corrected.height, "of", d.height)
    print(
        "q90 adjustment: mean",
        round(corrected["adj2"].mean(), 4),  # ty: ignore[no-matching-overload]
        "sd",
        round(corrected["adj2"].std(), 4),  # ty: ignore[no-matching-overload]
    )
    print(
        "exceedance of q90 before",
        round((y > corrected["q90"]).mean(), 4),  # ty: ignore[no-matching-overload]
        "after",
        round((y > corrected["q90"] + corrected["adj2"]).mean(), 4),  # ty: ignore[no-matching-overload]
    )
    print("rows with an adjustment above 0:", round((corrected["adj2"] > 0).mean(), 3))  # ty: ignore[no-matching-overload]
    _ = np
    # Share of rows whose pool is the whole previous 56 days, by month.
    print(
        tables.group_by("origin_date")
        .agg(pl.col("n_c").median())
        .sort("origin_date")
        .select(pl.col("n_c").median())
    )


if __name__ == "__main__":
    main()

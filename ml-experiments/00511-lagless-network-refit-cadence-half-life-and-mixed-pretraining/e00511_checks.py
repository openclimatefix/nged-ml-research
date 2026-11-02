# ruff: noqa: E501
"""Trigger-3 and median-minus-mean checks for experiment 511.

The mean forecast is approximated by the level-weighted mean of the 11 stored quantiles (trapezoid
over the levels, the tails extended flat). Prints, per arm and group (rows with no labelled event,
rows of the `main16` set), the mean absolute error of the median and of that mean forecast in points
of capacity against the cleaned actual, their contrasts, and per site the median minus the mean.
"""

from typing import Final

import e00470_build as build470  # ty: ignore[unresolved-import]
import e00510_common as c
import e00511_score as score
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = (
    "L-Ww",
    "L-3d",
    "L-1d",
    "L-Ww-30",
    "L-Ww-60",
    "L-mix",
    "K2-S1",
    "A-pool",
)
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
WEIGHTS: Final[np.ndarray] = np.diff(np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]]))


def main() -> None:
    """Print the checks."""
    score.configure()
    clean = c.clean_stack()
    names = c.sites()
    records = []
    for month in c.MONTHS:
        rows = pl.read_parquet(score.WORK_DIR / "scores" / f"rows_{month}.parquet").filter(
            pl.col("main16") & (pl.col("cls") == "clean") & (pl.col("group") != "report_only")
        )
        keys = rows.select("primary", "origin", "target", "group")
        actual = clean[keys["primary"].to_numpy(), keys["target"].to_numpy()]
        for arm in ARMS:
            f = pl.read_parquet(build470.prediction_path(arm=arm, month=month, cell=0))
            j = keys.join(f, on=["primary", "origin", "target"], how="left", maintain_order="left")
            q = j.select(list(build470.LEVELS)).to_numpy().astype(np.float64)
            mean = q @ WEIGHTS
            ok = np.isfinite(actual) & np.isfinite(mean)
            records.append(
                pl.DataFrame(
                    {"arm": arm, "group": j["group"].filter(ok), "primary": j["primary"].filter(ok),
                     "ae_median": np.abs(q[ok, 5] - actual[ok]) * 100, "ae_mean": np.abs(mean[ok] - actual[ok]) * 100,
                     "median_minus_mean": (q[ok, 5] - mean[ok]) * 100}
                )
            )  # fmt: skip
    t = pl.concat(records)
    summary = (
        t.group_by("arm", "group")
        .agg(
            pl.col("ae_median").mean(), pl.col("ae_mean").mean(), pl.col("median_minus_mean").mean()
        )
        .sort("group", "arm")
    )
    with pl.Config(tbl_rows=40):
        print(summary.with_columns(pl.col(pl.Float64).round(3)))
        per_site = (
            t.group_by("arm", "primary")
            .agg(m=pl.col("median_minus_mean").mean())
            .group_by("arm")
            .agg(max_abs=pl.col("m").abs().max(), mean_abs=pl.col("m").abs().mean())
            .sort("arm")
        )
        print(per_site.with_columns(pl.col(pl.Float64).round(3)))
    summary.write_csv(score.WORK_DIR / "scores" / "checks_mean_error.csv")
    _ = names


if __name__ == "__main__":
    main()

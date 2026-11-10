# ruff: noqa: D103, E501
"""Experiment 677: calibration of the 11 quantile levels of each ladder step, with the tails in
focus (private CSV).

Usage: python e00677_tails.py STEP[:SPAN:SEEDS] ...

For each step, the share of scored half-hours whose observed power is at or below each forecast quantile
(the nominal level is the quantile's own level), pooled over the Primaries with data and the
backtest year,
for the seed-mean forecast with the quantiles sorted within each row. Writes `scores/tail_calibration.csv`.
"""

import sys

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00677_export as exp
import e00677_score as sc
import numpy as np
import polars as pl


def main() -> None:
    truth = pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet").select(
        "time_series_id", "valid_time", "power"
    )
    rows = []
    for arg in sys.argv[1:]:
        step, _, rest = arg.partition(":")
        span, seeds = (
            (int(rest.split(":")[0]), tuple(int(c) for c in rest.split(":")[1]))
            if rest
            else (3, (0, 1, 2, 3, 4) if step[0] in "5678" else (0, 1, 2))
        )
        f = (
            sc.seed_mean(name=f"step{step}_{exp.VERSION[step]}_sp{span}", seeds=seeds)
            .join(truth, on=["time_series_id", "valid_time"])
            .filter(pl.col("power").is_not_null() & pl.col("power").is_finite())
        )
        y = f["power"].to_numpy().astype(np.float64)
        q = f.select(list(sc.QCOLS)).to_numpy().astype(np.float64)
        below = (y[:, None] <= q).mean(axis=0)
        for tau, share in zip(lib.TAUS, below, strict=True):
            rows.append(
                {
                    "step": f"{step}" + (" monthly" if span == 1 else ""),
                    "level": float(tau),
                    "share_at_or_below": float(share),
                    "half_hours": f.height,
                }
            )
        print(step, np.round(below, 3).tolist(), flush=True)
    pl.DataFrame(rows).write_csv(sc.ROOT / "scores" / "tail_calibration.csv")


if __name__ == "__main__":
    main()

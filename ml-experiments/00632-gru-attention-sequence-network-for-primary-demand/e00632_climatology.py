# ruff: noqa: D103
"""Experiment 632: month-by-half-hour climatology row.

For each Primary and fold month, the quantiles of the cleaned power (a share of capacity) at the
same calendar month and the same half-hour of the day, over all cleaned history that ends before the
fold's refit (3.5 h before the first of the month). A forecast ignores the origin and the weather.
It is the no-skill reference for the CRPS tables.

Usage: python e00632_climatology.py
"""

import e00632_data as d
import e00632_fold as fold_mod
import numpy as np
import polars as pl
from e00632_score import MONTHS, ROOT, TAU, Q

N_FUT = 336


def main() -> None:
    clean = d.clean_power()
    cap = fold_mod.capacity_of_primary()
    n_p, n = clean.shape
    grid_time = pl.from_epoch(
        pl.Series(d.prep422.GRID_START_US + np.arange(n) * d.prep422.STEP_US), time_unit="us"
    ).dt.replace_time_zone("UTC")
    month_of = grid_time.dt.month().to_numpy()
    slot = (grid_time.dt.hour() * 2 + grid_time.dt.minute() // 30).to_numpy()
    ids = d.prep.primaries().sort("primary")["time_series_id"].to_numpy()
    for month in MONTHS:
        origins = fold_mod.test_origins(month=month)
        refit = fold_mod.month_start(month=month)
        end = (
            fold_mod.idx_of(t=refit) - fold_mod.CUTOFF_AGE_STEPS
            if hasattr(fold_mod, "CUTOFF_AGE_STEPS")
            else fold_mod.idx_of(t=refit) - 7
        )
        fallbacks = 0
        table = np.full((n_p, 13, 48, 11), np.nan, dtype=np.float32)
        for p in range(n_p):
            for mo in range(1, 13):
                for sl in range(48):
                    v = clean[p, :end][(month_of[:end] == mo) & (slot[:end] == sl)]
                    v = v[np.isfinite(v)]
                    if (
                        len(v) < 10
                    ):  # no history in this calendar month: use every month at this half-hour
                        v = clean[p, :end][slot[:end] == sl]
                        v = v[np.isfinite(v)]
                        fallbacks += 1
                    table[p, mo, sl] = np.quantile(v, TAU)
        print(month, "cells using all-months fallback:", fallbacks)
        prim = np.repeat(np.arange(n_p), len(origins) * N_FUT)
        o = np.tile(np.repeat(origins, N_FUT), n_p)
        t = o + np.tile(np.arange(1, N_FUT + 1), n_p * len(origins))
        q = table[prim, month_of[t], slot[t]] * cap[prim, None]
        assert np.isfinite(q).all(), "a Primary lacks history for a month and half-hour"
        q = np.sort(q, axis=1)

        def to_t(idx: np.ndarray) -> pl.Series:
            return pl.from_epoch(
                pl.Series(d.prep422.GRID_START_US + idx * d.prep422.STEP_US), time_unit="us"
            ).dt.replace_time_zone("UTC")

        frame = pl.DataFrame(
            {"time_series_id": ids[prim], "power_fcst_init_time": to_t(o), "valid_time": to_t(t),
             "lead_hours": ((t - o) * 0.5).astype(np.float32),
             **{c: q[:, i] for i, c in enumerate(Q)}}
        )  # fmt: skip
        out = ROOT / "predictions" / "climatology" / f"{month}_s0.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(out)
        print(month, frame.height, flush=True)


if __name__ == "__main__":
    main()

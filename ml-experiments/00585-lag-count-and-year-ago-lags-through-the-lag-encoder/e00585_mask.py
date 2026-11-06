# ruff: noqa: E501
"""Experiment 585: Y1 against Y1-mask, the contamination check on the year-ago lag.

Y1-mask forecasts again with Y1's saved weights, with the year-ago lag set to missing wherever it
lies inside a verified load-shift event of experiment 580 (pass 200 in Y1's inject folder). For
each refit day and seed, this script compares the two forecasts' median on every row, and reports
the rows whose year-ago lag is inside an event separately from the rest.
"""

import e00470_build as build470  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00585-lag-count-and-year-ago-lags-through-the-lag-encoder"
W = ML_EXPERIMENTS_DATA_DIR / NAME / "arms" / "y1" / "work"
YEAR_STEPS = 364 * 48
MASK_PASS = 200
"""Pass number that `e00585_arm.py` gives the masked forecast (its own constant of the same name)."""


def real_events() -> list[dict]:
    """Return the verified real events (experiment 580) as per-Primary rows, as `e00585_arm.py` does."""
    sm = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    table = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00580-verified-load-shift-event-set-and-label-audit/work/events_verified.parquet"
    )
    return bank.verified_events(
        capacity_mw=dict(zip(sm["label"].to_list(), sm["capacity_mw"].to_list(), strict=True)),
        label_to_primary=dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True)),
        table=table,
    )


def main() -> None:
    """Write and print the Y1 against Y1-mask comparison."""
    sites = build470.site_map()
    prim = dict(zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True))
    events = pl.DataFrame(real_events()).select("primary", "onset_idx", "end_idx")
    rows = []
    for month in d6.MONTHS:
        for path in sorted((W / "fits" / "V3D5").glob(f"{month}_d*_s*.parquet")):
            if "stress" in path.name:
                continue
            day, seed = path.stem.split("_")[1:3]
            other = W / "inject" / f"{month}_r{MASK_PASS}_{day}_{seed}.parquet"
            if not other.exists():
                continue
            keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
            j = (
                pl.read_parquet(path)
                .select(*keys, y1="q50")
                .join(pl.read_parquet(other).select(*keys, masked="q50"), on=keys)
            )
            j = build470.grid_keys(frame=j).with_columns(
                primary=pl.col("time_series_id").replace_strict(prim, return_dtype=pl.Int16),
                lag_idx=pl.col("target") - YEAR_STEPS,
            )
            j = j.join(events, on="primary", how="left")
            inside = j.group_by(keys).agg(
                inside=(
                    (pl.col("lag_idx") >= pl.col("onset_idx"))
                    & (pl.col("lag_idx") < pl.col("end_idx"))
                ).any()
            )
            j = j.unique(subset=keys).join(inside, on=keys)
            for flag in (True, False):
                sub = j.filter(pl.col("inside") == flag)
                if sub.height:
                    d = (sub["masked"] - sub["y1"]).to_numpy()
                    rows.append(
                        {"month": month, "refit": day, "seed": seed, "year_ago_in_event": flag, "rows": sub.height,
                         "mean_abs_change_q50": float(np.mean(np.abs(d))), "mean_change_q50": float(np.mean(d))}
                    )  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(W.parent.parent / "mask_change.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(
            out.group_by("year_ago_in_event").agg(
                pl.col("rows").sum(),
                (pl.col("mean_abs_change_q50") * pl.col("rows")).sum() / pl.col("rows").sum(),
                (pl.col("mean_change_q50") * pl.col("rows")).sum() / pl.col("rows").sum(),
            )
        )


if __name__ == "__main__":
    main()

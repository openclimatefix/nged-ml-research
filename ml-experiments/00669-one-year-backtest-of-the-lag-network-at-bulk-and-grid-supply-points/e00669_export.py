# ruff: noqa: D103, RET504, PT018
"""Experiment 669: assemble the contract Parquet files `bsps.parquet` and `gsps.parquet` (CPU).

Reads each month's two seed forecasts of the network (native 09:00 origin, shares of effective
capacity), averages the quantiles over the seeds, sorts them within the row, keeps the valid times
from 12:00 on the origin day to 11:30 on day 14, relabels the origin as 12:00 UTC, multiplies by the
series' effective capacity (MW), and runs the contract's checks.

Usage: python e00669_export.py {bsp|gsp} [--out DIR]
"""

import argparse
from datetime import timedelta
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00669_common as k
import numpy as np
import polars as pl

QCOLS = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
FILE = {"bsp": "bsps.parquet", "gsp": "gsps.parquet"}


def seed_mean(*, group: str, month: str) -> pl.DataFrame:
    work = k.ROOT / "work" / group / "seed_preds" / k.TAG
    frames = [pl.read_parquet(work / f"{month}_s{s}.parquet") for s in (0, 1)]
    for fr in frames:
        assert fr.select(KEYS).n_unique() == fr.height, month
    return pl.concat(frames).group_by(KEYS).agg(*[pl.col(c).mean() for c in QCOLS])


def build(*, group: str) -> pl.DataFrame:
    series = common.series_table().filter(pl.col("site").is_in(list(k.GROUP_SITES[group])))
    cap = series.select("time_series_id", "effective_capacity_mw")
    assert series["units"].unique().to_list() == ["MW"]
    parts = [seed_mean(group=group, month=m) for m in k.MONTHS]
    frame = pl.concat(parts)
    origin9 = pl.col("power_fcst_init_time")
    day = origin9.dt.truncate("1d")
    start, stop = day + timedelta(hours=12), day + timedelta(days=14, hours=12)
    frame = frame.filter((pl.col("valid_time") >= start) & (pl.col("valid_time") < stop))
    frame = frame.join(cap, on="time_series_id", how="left", validate="m:1")
    vals = np.sort(frame.select(QCOLS).to_numpy().astype(np.float64), axis=1)
    mw = vals * frame["effective_capacity_mw"].to_numpy()[:, None]
    day_col = frame["power_fcst_init_time"].dt.truncate("1d")
    out = pl.DataFrame(
        {
            "time_series_id": frame["time_series_id"].cast(pl.Int8),
            "valid_time": frame["valid_time"],
            "nwp_init_time": day_col,
            "power_fcst_init_time": day_col + timedelta(hours=k.ORIGIN_HOUR_CONTRACT),
            **{c: pl.Series(c, mw[:, i]) for i, c in enumerate(QCOLS)},
        }
    ).sort("time_series_id", "power_fcst_init_time", "valid_time")
    return out


def checks(*, frame: pl.DataFrame, group: str) -> dict:
    n_series = len(k.GROUP_SITES[group])
    res = {"rows": frame.height, "expected_rows": n_series * k.N_ORIGINS * k.HALF_HOURS}
    assert frame.height == res["expected_rows"], res
    assert (
        frame.select(["time_series_id", "power_fcst_init_time", "valid_time"]).is_duplicated().sum()
        == 0
    )
    q = frame.select(QCOLS).to_numpy()
    assert np.isfinite(q).all(), "null or non-finite quantile"
    assert (np.diff(q, axis=1) >= 0).all(), "quantiles cross"
    for c in ("valid_time", "nwp_init_time", "power_fcst_init_time"):
        assert str(frame.schema[c]) == "Datetime(time_unit='us', time_zone='UTC')", frame.schema[c]
    assert frame.schema["time_series_id"] == pl.Int8
    lead = (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_minutes()
    assert lead.min() == 0 and lead.max() == 336 * 60 - 30, (lead.min(), lead.max())
    assert (frame["power_fcst_init_time"].dt.hour() == 12).all()
    assert (frame["nwp_init_time"].dt.hour() == 0).all()
    assert frame["power_fcst_init_time"].n_unique() == k.N_ORIGINS
    expected = {d.replace(hour=12) for d in k.origin_days()}
    assert set(frame["power_fcst_init_time"].unique().to_list()) == expected
    ids = set(
        common.series_table()
        .filter(pl.col("site").is_in(list(k.GROUP_SITES[group])))["time_series_id"]
        .to_list()
    )
    assert set(frame["time_series_id"].unique().to_list()) == ids
    res["columns"] = frame.columns
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=("bsp", "gsp"))
    p.add_argument("--out", type=Path, default=k.SHARED)
    a = p.parse_args()
    frame = build(group=a.group)
    res = checks(frame=frame, group=a.group)
    print(res)
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / FILE[a.group]
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)
    print("wrote", target)


if __name__ == "__main__":
    main()

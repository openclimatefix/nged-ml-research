"""Build the per-case wind inputs of experiment 659 from the 51 stored ensemble members.

For each farm and each daily run (the hindcast of experiment 259 to 2025-06-16, then experiment
615's rebuilt runs to 2026-06-30), this stores one row per case with the mean and standard
deviation across the 51 members of the 10 m and 100 m wind speeds, the sine and cosine of the 100 m
direction, and the air temperature, and the mean and standard deviation of the fixed physics-lite
power. Physics-lite is a power curve of the 100 m speed with cut-in 3 m/s, rated 12 m/s, cut-out
25 m/s, and a cubic rise between cut-in and rated. It has unit capacity, has no fitted parameter,
and is applied to each member before the mean and spread are taken.

    PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting:ml-experiments/00659-* \
        uv run --no-sync python ml-experiments/00659-*/e00659_wind_features.py
"""

import os
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as c303  # ty: ignore[unresolved-import]
import e00659_common as common
import numpy as np
import polars as pl

KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]


def physics_lite(*, speed: np.ndarray) -> np.ndarray:
    """Return the power curve of the 100 m speed as a fraction of capacity, per member."""
    rise = np.clip((speed - common.WIND_CUT_IN) / (common.WIND_RATED - common.WIND_CUT_IN), 0, 1)
    power = rise**3
    return np.where(speed >= common.WIND_CUT_OUT, 0.0, power)


def run_features(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the case rows (keys plus summaries) of one run's member rows for one farm."""
    rows = rows.sort(*KEYS, "ensemble_member")
    if rows.height % common.MEMBERS:
        msg = f"{rows.height} rows are not a whole number of cases"
        raise ValueError(msg)
    members = rows["ensemble_member"].to_numpy().reshape(-1, common.MEMBERS)
    if not (members == np.arange(common.MEMBERS)).all():
        msg = "members are not 0 to 50 in every case"
        raise ValueError(msg)

    def grid(name: str) -> np.ndarray:
        return rows[name].to_numpy().astype(np.float64).reshape(-1, common.MEMBERS)

    s10, s100 = grid("wind_speed_10m"), grid("wind_speed_100m")
    sin, cos, temp = grid("wind_dir_sin_100m"), grid("wind_dir_cos_100m"), grid("temperature_c")
    power = physics_lite(speed=s100)
    summaries = {}
    for name, array in (
        ("speed10", s10),
        ("speed100", s100),
        ("dir_sin", sin),
        ("dir_cos", cos),
        ("temp", temp),
    ):
        summaries[f"{name}_mean"] = array.mean(axis=1)
        summaries[f"{name}_sd"] = array.std(axis=1)
    summaries["pl_mean"] = power.mean(axis=1)
    summaries["pl_sd"] = power.std(axis=1)
    keys = rows.gather_every(common.MEMBERS).select(KEYS)
    return keys.with_columns(**{k: pl.Series(v.astype(np.float32)) for k, v in summaries.items()})


def main() -> None:
    """Write `wind_features/<farm>.parquet` for each of the three farms."""
    common.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    paths = sorted(c303.HINDCAST_DIR.glob("init=*.parquet")) + sorted(
        (common.E615_DIR / "wind_members").glob("init=*.parquet")
    )
    pieces: dict[str, list[pl.DataFrame]] = {f: [] for f in sites["site"].to_list()}
    started = time.time()
    for path in paths:
        frame = pl.read_parquet(path)
        for row in sites.iter_rows(named=True):
            part = frame.filter(pl.col("time_series_id") == row["time_series_id"])
            if part.height:
                pieces[row["site"]].append(run_features(rows=part))
    for farm, parts in pieces.items():
        table = pl.concat(parts).sort(*KEYS)
        out = common.FEATURE_DIR / f"{farm}.parquet"
        table.write_parquet(out.with_suffix(".tmp"))
        out.with_suffix(".tmp").rename(out)
        print(f"{farm}: {table.height:,} cases {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()

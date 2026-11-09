# ruff: noqa: E402
"""Build the per-case wind inputs for every run to 2026-10-31, gated on experiment 659's.

Copied from experiment 659's `e00659_wind_features.py`: for each farm and each daily run, one row
per case with the mean and standard deviation across the 51 members of the 10 m and 100 m wind
speeds, the sine and cosine of the 100 m direction, and the air temperature, and the mean and
standard deviation of the fixed physics-lite power (a cubic power curve of the 100 m speed with
cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s, applied to each member before the summary).

Runs before 2026-06-16 are read from experiment 259's and 615's member files. Runs from
2026-06-16 are read from this experiment's rebuilt member files (`e00666_wind_members.py`), which
run the full 15 days. The gate asserts that every case of experiment 659's stored summaries is
reproduced to 1e-5. The script then lists, per run from 2025-11-01, any missing member or valid
time of the 672 half-hours from 12:00 to 347.5 h after the 00 UTC run.

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00666-*/e00666_wind_features.py
"""

import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00666_common as common
import numpy as np
import polars as pl
from studies.pv_dataset import wind_sites

KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
GATE_TOLERANCE: Final[float] = 1e-5


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


def member_paths() -> list[Path]:
    """Return one member file per run: old runs from 259 and 615, rebuilt runs from this folder."""
    rebuilt = sorted(common.MEMBERS_DIR.glob("init=*.parquet"))
    rebuilt_names = {p.name for p in rebuilt}
    old = sorted(common.E259_HINDCAST_DIR.glob("init=*.parquet")) + sorted(
        common.E615_MEMBERS_DIR.glob("init=*.parquet")
    )
    return sorted([p for p in old if p.name not in rebuilt_names] + rebuilt, key=lambda p: p.name)


def gate(*, farm: str, table: pl.DataFrame) -> float:
    """Return the largest absolute difference from experiment 659, asserting the gate."""
    old = pl.read_parquet(common.E659_FEATURE_DIR / f"{farm}.parquet")
    mine = table.filter(pl.col("valid_time") < common.OLD_VALID_LIMIT)
    joined = old.join(mine, on=KEYS, how="inner", suffix="_new")
    if joined.height != old.height:
        msg = f"{farm}: only {joined.height} of {old.height} keys match"
        raise AssertionError(msg)
    # Experiment 615 cut its member files' NWP rows at 2026-06-30 23:30, so the last half-hours
    # before 2026-07-01 that interpolate towards a stamp after the cut are missing from
    # experiment 659's cases. The rebuilt runs have them.
    extra = mine.join(old, on=KEYS, how="anti")
    print(
        f"  gate {farm}: {extra.height} extra cases before 2026-07-01, earliest valid time "
        f"{extra['valid_time'].min()}",
        flush=True,
    )
    worst = 0.0
    for column in (c for c in old.columns if c not in KEYS):
        difference = float(
            joined.select((pl.col(column) - pl.col(f"{column}_new")).abs().max()).item()
        )
        print(f"  gate {farm} {column}: max abs difference {difference:.2e}", flush=True)
        worst = max(worst, difference)
    if worst > GATE_TOLERANCE:
        msg = f"{farm}: summaries differ from experiment 659 by up to {worst}"
        raise AssertionError(msg)
    return worst


def coverage(*, farm: str, table: pl.DataFrame) -> None:
    """Raise if a run from 2025-11-01 lacks any of the 672 half-hours (12 h to 347.5 h)."""
    cases = table.filter(pl.col("power_fcst_init_time") >= common.FIRST_ORIGIN).select(
        init=pl.col("power_fcst_init_time") - timedelta(hours=9), valid="valid_time"
    )
    counts = (
        cases.with_columns(lead=(pl.col("valid") - pl.col("init")).dt.total_minutes() / 60.0)
        .filter((pl.col("lead") >= 12.0) & (pl.col("lead") <= 347.5))
        .group_by("init")
        .agg(n=pl.len(), last=pl.col("lead").max())
    )
    expected = pl.datetime_range(
        common.FIRST_ORIGIN, common.LAST_RUN, interval="1d", time_zone="UTC", eager=True
    )
    short = (
        pl.DataFrame({"init": expected})
        .join(counts, on="init", how="left")
        .filter(pl.col("n").is_null() | (pl.col("n") != common.HORIZON_HALF_HOURS))
    )
    print(f"  coverage {farm}: {short.height} runs without all 672 half-hours", flush=True)
    if short.height:
        msg = f"{farm}: runs without all 672 half-hours:\n{short}"
        raise ValueError(msg)


def main() -> None:
    """Write `wind_features/<farm>.parquet` for each farm after the gate, then list gaps."""
    common.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    sites = wind_sites().select("time_series_id", "site")
    paths = member_paths()
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
        print(f"{farm}: {table.height:,} cases {time.time() - started:.0f} s", flush=True)
        gate(farm=farm, table=table)
        coverage(farm=farm, table=table)
        out = common.FEATURE_DIR / f"{farm}.parquet"
        table.write_parquet(out.with_suffix(".tmp"))
        out.with_suffix(".tmp").rename(out)


if __name__ == "__main__":
    main()

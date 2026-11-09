"""Experiment 661, step 5 inputs: known weather degraded to the ensemble's native time step.

Takes experiment 418's dense pretraining weather (ERA5 temperature and so on at half-hours, CAMS
shortwave irradiance, hourly values on the half-hour grid), replaces the shortwave column by its
3-hour or 6-hour window means on the ensemble's schedule, and rebuilds half-hourly values with the
production resampler (`ml_core.features._nwp._upsample_nwp_to_half_hourly`). The other six columns
are untouched. The raw input matrices that experiment 639 trains and scores on are then rebuilt
from the degraded weather, one folder per variant.

A 3-hour window sits at leads 3 to 24 of a run that starts at 00:00 UTC of the day. A 6-hour window
sits at leads 150 to 168 of a run that started 144 hours before the day, so that the resampler sees
the 6-hourly part of the ensemble schedule. The grid timestamp is the end of its half-hour, as in
experiment 418.

Usage: python e00661_degrade.py raw --variant w3|w6 [--sites D5 ...]
       python e00661_degrade.py check --variant w3|w6
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00418_prep as p418  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00661-pv-ens-time-resolution-shortwave-interpolation-audit-and-bias-adapter"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
WIDTH_H: Final[dict[str, int]] = {"w3": 3, "w6": 6}
SW: Final = "downward_short_wave_radiation_flux_surface"
SHORTWAVE_COLUMN: Final = p418.WEATHER.index("shortwave")


def raw_dir(*, variant: str) -> Path:
    """Return the folder of a variant's raw input matrices."""
    return WORK / "raw" / variant


def degrade(
    *, shortwave: np.ndarray, width_h: int, latitude: float, longitude: float
) -> np.ndarray:
    """Return the shortwave column degraded to windows of `width_h` hours and rebuilt at half-hours.

    Args:
        shortwave: Half-hourly irradiance on experiment 418's dense grid (NaN where missing).
        width_h: 3 or 6.
        latitude: Site latitude in degrees (the resampler's clear-sky shape needs it).
        longitude: Site longitude in degrees.

    Returns:
        An array of the same shape. A half-hour keeps NaN when its window has any NaN.
    """
    n = len(shortwave)
    start = common.GRID_START
    # The grid timestamp is the end of its half-hour; index i ends at start + 30 min * i.
    i = np.arange(n)
    ends_min = i * 30
    day = (ends_min - 1) // 1440  # half-hour ending 00:00 belongs to the previous day
    within = (
        ends_min - day * 1440
    )  # minutes after the day's 00:00 at the end of the half-hour, 30..1440
    window = (within - 1) // (width_h * 60)  # 0-based window of the day
    keys = day * (24 // width_h) + window
    ok = i >= 1
    uniq, inv = np.unique(keys[ok], return_inverse=True)
    sums = np.bincount(inv, weights=np.nan_to_num(shortwave[ok]))
    bad = np.bincount(inv, weights=np.isnan(shortwave[ok]).astype(float)) > 0
    counts = np.bincount(inv)
    means = sums / counts
    win_day = uniq // (24 // width_h)
    win_idx = uniq % (24 // width_h)
    day0 = datetime(start.year, start.month, start.day, tzinfo=UTC)
    ends = [
        day0 + timedelta(days=int(d), hours=width_h * (int(w) + 1))
        for d, w in zip(win_day, win_idx, strict=True)
    ]
    init = [day0 + timedelta(days=int(d), hours=0 if width_h == 3 else -144) for d in win_day]
    # Each trajectory also gets the null row that precedes its first window (lead 0, or lead 144 for
    # the 6-hourly part), as the archive has, because the resampler's half-hour grid starts at the
    # first row of a trajectory.
    n_days = int(win_day.max() - win_day.min()) + 1
    first_days = np.arange(win_day.min(), win_day.min() + n_days)
    lead_in = [day0 + timedelta(days=int(d)) for d in first_days]
    native = pl.DataFrame(
        {
            "time_series_id": np.zeros(len(uniq) + n_days, dtype=np.int32),
            "ensemble_member": np.zeros(len(uniq) + n_days, dtype=np.int64),
            "valid_time": [*ends, *lead_in],
            "nwp_init_time": [
                *init,
                *[d + timedelta(hours=0 if width_h == 3 else -144) for d in lead_in],
            ],
            SW: np.concatenate([np.where(bad, np.nan, means), np.full(n_days, np.nan)]),
        }
    ).with_columns(
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        pl.col("nwp_init_time").cast(pl.Datetime("us", "UTC")),
        pl.when(pl.col(SW).is_nan()).then(None).otherwise(pl.col(SW)).alias(SW),
    )
    coords = (
        pl.DataFrame({"time_series_id": [0], "latitude": [latitude], "longitude": [longitude]})
        .cast({"time_series_id": pl.Int32, "latitude": pl.Float32, "longitude": pl.Float32})
        .lazy()
    )
    rebuilt = _upsample_nwp_to_half_hourly(native.lazy(), solar_coordinates=coords).collect()
    index = np.asarray(common.grid_index(times=rebuilt["valid_time"]))
    out = np.full(n, np.nan, dtype=np.float32)
    inside = (index >= 0) & (index < n)
    values = rebuilt[SW].to_numpy()
    inside &= np.isfinite(values)  # the lead-in rows repeat other windows' stamps with null values
    out[index[inside]] = values[inside].astype(np.float32)
    assert len(np.unique(index[inside])) == int(inside.sum()), (
        "duplicate half-hour stamps with values"
    )
    return out


def write_raw(*, variant: str, sites: list[str] | None = None) -> None:
    """Write the degraded raw matrices of every Primary (or `sites`) into the variant's folder."""
    width = WIDTH_H[variant]
    original = p418.dense_weather
    table = {r["site"]: r for r in base.primaries().iter_rows(named=True)}

    def patched(*, series_id: int, site: str) -> np.ndarray:
        weather = original(series_id=series_id, site=site)
        weather[:, SHORTWAVE_COLUMN] = degrade(
            shortwave=weather[:, SHORTWAVE_COLUMN], width_h=width,
            latitude=float(table[site]["latitude"]), longitude=float(table[site]["longitude"]),
        )  # fmt: skip
        return weather

    p418.dense_weather = patched
    p418.RAW_DIR = raw_dir(variant=variant)
    try:
        for site in sites or list(table):
            print(p418.write_raw(site=site), flush=True)
    finally:
        p418.dense_weather = original


def check(*, variant: str) -> None:
    """Print the error of the degraded shortwave against the original for one site on sunny days."""
    width = WIDTH_H[variant]
    row = next(iter(base.primaries().iter_rows(named=True)))
    weather = p418.dense_weather(series_id=row["time_series_id"], site=row["site"])
    orig = weather[:, SHORTWAVE_COLUMN]
    started = time.monotonic()
    deg = degrade(
        shortwave=orig,
        width_h=width,
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
    )
    ok = np.isfinite(orig) & np.isfinite(deg)
    print(f"{variant}: {ok.sum():,} finite pairs, {time.monotonic() - started:.0f} s")
    print(
        "RMSE W/m2",
        float(np.sqrt(np.mean((orig[ok] - deg[ok]) ** 2))),
        "bias",
        float(np.mean(deg[ok] - orig[ok])),
        "peak orig",
        float(np.nanmax(orig)),
        "peak deg",
        float(np.nanmax(deg)),
    )
    print(
        "NaN where original finite:",
        int((np.isfinite(orig) & ~np.isfinite(deg)).sum()),
        "of",
        int(np.isfinite(orig).sum()),
    )
    # window-mean conservation
    n = len(orig)
    ends_min = np.arange(n) * 30
    key = ((ends_min - 1) // (width * 60))[1:]
    a, b = orig[1:], deg[1:]
    good = np.isfinite(a) & np.isfinite(b)
    _ua, inv = np.unique(key[good], return_inverse=True)
    ma = np.bincount(inv, weights=a[good]) / np.bincount(inv)
    mb = np.bincount(inv, weights=b[good]) / np.bincount(inv)
    print("max |window mean difference| W/m2:", float(np.abs(ma - mb).max()))


def main() -> None:
    """Run one subcommand."""
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["raw", "check"])
    p.add_argument("--variant", required=True, choices=list(WIDTH_H))
    p.add_argument("--sites", nargs="*")
    a = p.parse_args()
    if a.cmd == "raw":
        write_raw(variant=a.variant, sites=a.sites)
    else:
        check(variant=a.variant)


if __name__ == "__main__":
    main()

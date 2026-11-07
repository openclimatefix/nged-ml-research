# ruff: noqa: E501
"""Experiment 616: probe the stored stage-`pre` networks with fixed datetime inputs and coherent donor weather.

For a probe date `d` (a Tuesday in the development period) and each daylight half-hour slot `h`, the
datetime columns (14 to 34) and the drift position are fixed at `(d, h)`. The weather columns (0 to 6)
are taken from a donor: the same half-hour of every day within 21 days of `d`'s day of year, in every
year of the fold's pretraining window, at the Primary's own location. The donor's shortwave is
rescaled to the probe date's sun position (donor clear-sky index times the probe's clear-sky
irradiance). The network's mixture mean, converted to MW, is the probed response.

Usage: python e00616_probe.py FOLD [FOLD ...]
Writes `probe_<fold>.parquet` (donor table) and `probe_<fold>_y.npy` (float32, [arm, seed, row]).
"""

import sys
import time
from datetime import UTC, date, datetime, timedelta
from typing import Final

import e00616_lib as lib
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "00616-lag-free-network-pv-capacity-probe-on-stored-weights"
MW_PRIMARIES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
SUMMER_2025: Final[tuple[date, ...]] = tuple(
    date(2025, 4, 15) + timedelta(days=14 * i) for i in range(12)
)
"""Tuesdays every 14 days from 2025-04-15 to 2025-09-16."""
SPRING_2026: Final[tuple[date, ...]] = (date(2026, 4, 14), date(2026, 4, 28), date(2026, 5, 12))
WINDOW_DAYS: Final[int] = 21
MIN_ELEVATION_DEG: Final[float] = 10.0
SEEDS: Final[tuple[int, ...]] = tuple(range(10))
HALF_HOUR_US: Final[int] = 1_800_000_000
DAY_US: Final[int] = 86_400_000_000


def clear_sky(*, cos_zenith: np.ndarray) -> np.ndarray:
    """Return the Haurwitz clear-sky global horizontal irradiance in W/m2."""
    cz = np.clip(cos_zenith, 1e-3, None)
    return np.where(cos_zenith > 1e-3, 1098.0 * cz * np.exp(-0.057 / cz), 0.0)


def doy_distance(*, a: np.ndarray, b: int) -> np.ndarray:
    """Return the circular distance in days between day-of-year arrays `a` and the day `b`."""
    d = np.abs(a - b)
    return np.minimum(d, 366 - d)


def dates_for(*, month: str) -> list[date]:
    """Return the probe dates of a fold: the 2025 summer dates, and the 2026 spring dates before the fold."""
    refit = datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=UTC).date()
    out = list(SUMMER_2025)
    out += [d for d in SPRING_2026 if d < refit - timedelta(days=10)]
    return out


def build_table(
    *, fold: lib.Fold, primary: int, month: str, dates: list[date]
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the donor table of a Primary and the standardised network inputs (before the drift column)."""
    raw = lib.raw_rows(site=fold.sites[primary])
    xr, valid = raw["x"], raw["valid"]
    cutoff = lib.data._us(t=lib.c.month_start(month=month) - lib.data.regimes.REFIT_CUTOFF_AGE)
    tod = (valid // HALF_HOUR_US) % 48
    day_us = valid - tod * HALF_HOUR_US
    doy = (
        day_us.astype("datetime64[us]").astype("datetime64[D]")
        - day_us.astype("datetime64[us]").astype("datetime64[Y]").astype("datetime64[D]")
    ).astype(int)
    index = {int(v): i for i, v in enumerate(valid)}
    frames, inputs = [], []
    for d in dates:
        day0 = int(np.datetime64(d, "us").astype(np.int64))
        d_doy = (np.datetime64(d, "D") - np.datetime64(d, "Y").astype("datetime64[D]")).astype(int)
        for slot in range(48):
            probe_us = day0 + slot * HALF_HOUR_US
            pi = index.get(probe_us)
            if pi is None:
                continue
            cz_p = float(xr[pi, lib.COS_ZENITH])
            if np.degrees(np.arcsin(np.clip(cz_p, -1, 1))) < MIN_ELEVATION_DEG:
                continue
            assert xr[pi, 32] == 0.0, (d, slot, "bank holiday at a probe row")
            donors = np.nonzero(
                (tod == slot)
                & (doy_distance(a=doy, b=int(d_doy)) <= WINDOW_DAYS)
                & (valid <= cutoff)
                & (day_us != day0)
            )[0]
            if len(donors) == 0:
                continue
            x = xr[donors].copy()
            g_raw = x[:, lib.SHORTWAVE].copy()
            cz_d = x[:, lib.COS_ZENITH].copy()
            cs_d = clear_sky(cos_zenith=cz_d)
            index_d = np.where(cs_d > 20.0, g_raw / np.maximum(cs_d, 20.0), np.nan)
            cs_p = float(clear_sky(cos_zenith=np.array([cz_p]))[0])
            x[:, lib.SHORTWAVE] = np.nan_to_num(index_d * cs_p, nan=0.0)
            x[:, lib.DATETIME] = xr[pi, lib.DATETIME]
            inputs.append(x)
            frames.append(
                pl.DataFrame(
                    {
                        "primary": primary,
                        "date": d,
                        "slot": slot,
                        "probe_us": probe_us,
                        "donor_us": valid[donors],
                        "g_donor": g_raw,
                        "g_in": x[:, lib.SHORTWAVE],
                        "clear_probe": cs_p,
                        "clear_donor": cs_d,
                        "temp": xr[donors, lib.TEMPERATURE],
                        "elev_probe": float(np.degrees(np.arcsin(cz_p))),
                    }
                )
            )
    return pl.concat(frames), np.concatenate(inputs)


def run_fold(*, month: str) -> None:
    """Probe every stored network of a fold (arms D5 and M0, seeds 0 to 9) at the six MW Primaries."""
    started = time.monotonic()
    torch.set_num_threads(4)
    fold = lib.fold_statistics(month=month)
    dates = dates_for(month=month)
    print(
        month,
        "probe dates:",
        len(dates),
        "fold stats",
        f"{time.monotonic() - started:.0f}s",
        flush=True,
    )
    ids = [fold.sites.index(s) for s in MW_PRIMARIES]
    tables, xs = [], []
    for p in ids:
        t, x = build_table(fold=fold, primary=p, month=month, dates=dates)
        tables.append(t)
        xs.append(lib.standardise(fold=fold, x_raw=x))
    table = pl.concat(tables)
    x_std = np.concatenate(xs)
    primary = table["primary"].to_numpy()
    probe_us = table["probe_us"].to_numpy()
    print(month, "rows", len(table), flush=True)
    mean_t, std_t = fold.rows.target_mean[primary], fold.rows.target_std[primary]
    cap = fold.capacity_mw[primary]
    y = np.zeros((len(lib.ARMS), len(SEEDS), len(table)), dtype=np.float32)
    for a, arm in enumerate(lib.ARMS):
        x_arm = lib.with_tau(arm=arm, x=x_std, valid_us=probe_us)
        for s, seed in enumerate(SEEDS):
            net = lib.load_net(arm=arm, month=month, seed=seed)
            z = lib.mixture_mean(net=net, x=x_arm, primary=primary)
            y[a, s] = ((z * std_t + mean_t) * cap).astype(np.float32)
        print(month, arm, "done", f"{time.monotonic() - started:.0f}s", flush=True)
    table.write_parquet(OUT / f"probe_{month}.parquet")
    np.save(OUT / f"probe_{month}_y.npy", y)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        run_fold(month=m)

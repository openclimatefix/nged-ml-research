# ruff: noqa: E501
"""Pretraining rows (ERA5 and CAMS weather) and fold assembly for experiment 418.

The fine-tuning rows are experiment 397's summary-scheme rows. The pretraining rows use the same
input columns, but the weather columns come from ERA5 (the means) and CAMS (shortwave), and the
spread columns are filled per fold from the ENS rows' own median spread by lead band.

Commands (run with `PYTHONPATH` set as `e00418_fit.py` shows):

- `raw`: write each Primary's raw pretraining matrix (pseudo-runs every `STRIDE_DAYS` days from
  2019-09-01, leads 0.5 to 351 hours), with the spread columns left at zero.
- `check`: build one fold, print its row counts, compare the fine-tuning rows with experiment
  397's `build_train`, print the input-unit comparison with ENS, and print the noise ratio.
"""

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, STUDIES_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens"
)
RAW_DIR: Final[Path] = WORK_DIR / "work" / "raw_pretrain"
CAMS_PATH: Final[Path] = (
    STUDIES_DATA_DIR / "downloads" / "reanalysis" / "CAMS" / "demand_points"
) / "cams_demand_points.parquet"
PRETRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
STRIDE_DAYS: Final[int] = 2
MAX_STEPS: Final[int] = 702
"""351 hours of half-hours: the longest lead experiment 397's rows hold."""
LEAD_BAND_HOURS: Final[float] = 24.0
N_BANDS: Final[int] = 15
LOW_DATA_END: Final[datetime] = datetime(2024, 10, 1, tzinfo=UTC)
"""The first six months of ENS (2024-04-01 to 2024-09-30) for the low-data check."""
FIRST_USABLE: Final[dict[str, datetime]] = {
    "D11": datetime(2019, 9, 17, tzinfo=UTC),
    "D10": datetime(2019, 12, 1, tzinfo=UTC),
    "D5": datetime(2020, 2, 1, tzinfo=UTC),
    "D14": datetime(2020, 7, 1, tzinfo=UTC),
    "D16": datetime(2021, 2, 1, tzinfo=UTC),
}
"""First usable date after experiment 103's early-calibration cuts (copied from experiment 149)."""
STATE_VARIABLES: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "wind_speed_10m",
)
"""Interpolated linearly to the half-hour stamps, as the ENS upsampler does for point values."""
HELD_VARIABLES: Final[tuple[str, ...]] = ("longwave", "precipitation")
"""Hour-ending means held over both half-hours of the hour."""
WEATHER: Final[tuple[str, ...]] = base.WEATHER
CHUNK: Final[int] = 1_000_000


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


# --- dense weather ----------------------------------------------------------------------------


def dense_weather(*, series_id: int, site: str) -> np.ndarray:
    """Return a Primary's pretraining weather on the dense half-hourly grid, shape (grid, 7).

    Columns follow `WEATHER`. State variables are linearly interpolated between the hourly ERA5
    stamps; longwave and precipitation take the value of the hour ending at the stamp for both
    half-hours of that hour; shortwave is CAMS global horizontal irradiance, held the same way
    (experiment 14's convention: a CAMS stamp is the end of its hour).
    """
    size = common.grid_times().len()
    out = np.full((size, len(WEATHER)), np.nan, dtype=np.float32)
    era5 = (
        pl.read_parquet(common.ERA5_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .sort("valid_time")
    )
    index = np.asarray(common.grid_index(times=era5["valid_time"]))
    inside = (index >= 0) & (index < size)
    index = index[inside]
    for name in WEATHER:
        column = WEATHER.index(name)
        if name == "shortwave":
            continue
        values = era5[name].to_numpy().astype(np.float64)[inside]
        if name in STATE_VARIABLES:
            good = np.isfinite(values)
            grid = np.arange(index.min(), index.max() + 1)
            out[grid, column] = np.interp(grid, index[good], values[good]).astype(np.float32)
        else:
            out[index, column] = values.astype(np.float32)
            previous = index - 1
            out[previous[previous >= 0], column] = values[previous >= 0].astype(np.float32)
    cams = pl.read_parquet(CAMS_PATH).filter(pl.col("site") == site).sort("time")
    cams_index = np.asarray(common.grid_index(times=cams["time"]))
    ok = (cams_index > 0) & (cams_index < size)
    ghi = cams["ghi_w_m2"].to_numpy().astype(np.float32)
    column = WEATHER.index("shortwave")
    out[cams_index[ok], column] = ghi[ok]
    out[cams_index[ok] - 1, column] = ghi[ok]
    return out


# --- pretraining raw matrices -----------------------------------------------------------------


def pretrain_frame(
    *, row: dict, power: np.ndarray, weather: np.ndarray
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return a Primary's pseudo-run rows as a frame for `feature_matrix`, with the origin index.

    Rows with a missing power or weather value are dropped, as are rows before the Primary's first
    usable date. The spread columns are zero; `fold_pretrain` fills them per fold.
    """
    size = power.size
    first = common.grid_index(times=PRETRAIN_START + timedelta(hours=9))
    last_origin = size - 1 - MAX_STEPS
    origins = np.arange(first, last_origin, STRIDE_DAYS * common.STEPS_PER_DAY, dtype=np.int64)
    steps = np.arange(1, MAX_STEPS + 1, dtype=np.int64)
    origin = np.repeat(origins, steps.size)
    target = (origins[:, None] + steps[None, :]).ravel()
    keep = np.isfinite(power[target]) & np.isfinite(weather[target]).all(axis=1)
    cut = FIRST_USABLE.get(row["site"])
    if cut is not None:
        keep &= target >= int(common.grid_index(times=cut))
    origin, target = origin[keep], target[keep]
    grid = common.grid_times()
    derived = common.derived_columns(
        valid_time=grid, latitude=row["latitude"], longitude=row["longitude"]
    )
    lags = common.origin_lags(power=power, target=target, origin=origin)
    columns: dict[str, np.ndarray] = {}
    for column, name in enumerate(WEATHER):
        columns[f"mean_{name}"] = weather[target, column]
        columns[f"std_{name}"] = np.zeros(target.size, dtype=np.float32)
    for name in common.DERIVED:
        columns[name] = derived[name].to_numpy()[target]
    for column, name in enumerate(common.LAG_COLUMNS):
        columns[name] = lags[:, column]
    columns["lead_hours"] = ((target - origin) / 2.0).astype(np.float32)
    frame = pl.DataFrame(columns).with_columns(
        valid_time=grid.gather(target), power_fcst_init_time=grid.gather(origin)
    )
    return frame, origin


def write_raw(*, site: str) -> str:
    """Write a Primary's raw pretraining matrix if absent; return a progress line."""
    target = RAW_DIR / f"{site}.npz"
    if target.exists():
        return f"{site}: exists"
    started = time.monotonic()
    table = base.primaries()
    row = next(r for r in table.iter_rows(named=True) if r["site"] == site)
    power = common.load_power_arrays(series=table.filter(pl.col("site") == site))[site].power
    weather = dense_weather(series_id=row["time_series_id"], site=site)
    frame, _ = pretrain_frame(row=row, power=power, weather=weather)
    x = base.feature_matrix(frame=frame, scheme="summary")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    grid = frame["valid_time"].dt.epoch("us").to_numpy()
    np.savez(
        temporary,
        x=x,
        y=power[np.asarray(common.grid_index(times=frame["valid_time"]))].astype(np.float32),
        valid=grid,
        origin=frame["power_fcst_init_time"].dt.epoch("us").to_numpy(),
        lead=frame["lead_hours"].to_numpy().astype(np.float32),
    )
    temporary.rename(target)
    return f"{site}: {len(x):,} rows, {time.monotonic() - started:.0f} s"


# --- fold assembly ----------------------------------------------------------------------------


@dataclass
class EnsFold(base.FoldTrain):
    """Experiment 397's fold rows, plus the valid time of each row."""

    valid: np.ndarray


def build_ens_fold(*, refit: datetime, valid_before: datetime | None = None) -> EnsFold:
    """Return experiment 397's summary-scheme fold, optionally limited to early valid times.

    With `valid_before`, the standardiser and the target statistics are those of the whole
    window, and only rows with an earlier valid time remain (the low-data check), so the pretrained
    network's inputs stay on the same scale. Without it the result equals `e00397_prep.build_train`.
    """
    _, mask, layout = base.feature_names(scheme="summary")
    table = base.primaries()
    parts = []
    for row in table.iter_rows(named=True):
        z = np.load(base.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= _us(t=refit))
            & (z["valid"] > _us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if not keep.any():
            continue
        parts.append(
            {k: z[k][keep] for k in ("x", "y", "origin", "valid", "lead")}
            | {"primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64)}
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=base.FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / (86_400 * 1_000_000)
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(table)
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    if valid_before is not None:
        keep &= cat["valid"] < _us(t=valid_before)
    fields = {
        "x": standardiser.apply(x=cat["x"][keep]),
        "y": cat["y"][keep],
        "primary": cat["primary"][keep],
        "weight": weight[keep].astype(np.float32),
        "is_val": held[keep],
        "lead": cat["lead"][keep],
        "origin_day": origin_day[keep],
        "standardiser": standardiser,
        "target_mean": target_mean,
        "target_std": target_std,
        "layout": layout,
        "valid": cat["valid"][keep],
    }
    return EnsFold(**fields)


def column_indices() -> tuple[np.ndarray, np.ndarray]:
    """Return the column indices of the `mean_*` and the `std_*` inputs, in `WEATHER` order."""
    names, _, _ = base.feature_names(scheme="summary")
    return (
        np.array([names.index(f"mean_{w}") for w in WEATHER]),
        np.array([names.index(f"std_{w}") for w in WEATHER]),
    )


def spread_tables(*, fold: EnsFold) -> tuple[np.ndarray, np.ndarray]:
    """Return the median ENS spread by lead band, as standardised `std_*` inputs and noise sizes.

    Both arrays have shape (bands, 7). The first holds the median standardised `std_*` input of the
    fold's training rows (held-out weeks excluded). The second holds the same spread in the
    standardised units of the matching `mean_*` input, which is the noise standard deviation of
    arm P3.
    """
    mean_cols, std_cols = column_indices()
    train = ~fold.is_val
    band = np.minimum((fold.lead[train] // LEAD_BAND_HOURS).astype(int), N_BANDS - 1)
    x = fold.x[train]
    table = np.zeros((N_BANDS, len(WEATHER)), dtype=np.float32)
    for b in range(N_BANDS):
        rows = band == b
        if rows.any():
            table[b] = np.median(x[rows][:, std_cols], axis=0)
    st = fold.standardiser
    raw_spread = table * st.std[std_cols] + st.mean[std_cols]
    noise = np.maximum(raw_spread, 0.0) / st.std[mean_cols]
    return table, noise.astype(np.float32)


@dataclass
class PretrainSet:
    """Standardised pretraining rows of a fold."""

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    is_val: np.ndarray
    lead: np.ndarray


def fold_pretrain(*, fold: EnsFold, refit: datetime, noisy: bool, rng_seed: int) -> PretrainSet:
    """Return a fold's pretraining rows on the fold's input scale, with optional forecast-like noise.

    A row is kept only if its valid time is at least 3.5 hours before the refit and its origin is at
    or before it. The held-out weeks and their purge are the same absolute weeks as the fold's.
    """
    table, noise = spread_tables(fold=fold)
    mean_cols, std_cols = column_indices()
    cutoff_valid, cutoff_origin = _us(t=refit - regimes.REFIT_CUTOFF_AGE), _us(t=refit)
    files = []
    total = 0
    for row in base.primaries().iter_rows(named=True):
        z = np.load(RAW_DIR / f"{row['site']}.npz")
        origin = z["origin"]
        keep = (z["valid"] <= cutoff_valid) & (origin <= cutoff_origin)
        day = ((origin - _us(t=base.FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
        held, purged = base.held_out_weeks(origin_day=day)
        keep &= ~purged
        if keep.any():
            files.append((row["primary"], row["site"], keep, held))
            total += int(keep.sum())
    n_inputs = fold.x.shape[1]
    out = PretrainSet(
        x=np.empty((total, n_inputs), dtype=np.float32),
        y=np.empty(total, dtype=np.float32),
        primary=np.empty(total, dtype=np.int64),
        is_val=np.empty(total, dtype=bool),
        lead=np.empty(total, dtype=np.float32),
    )
    rng = np.random.default_rng(rng_seed)
    start = 0
    for primary, site, keep, held in files:
        z = np.load(RAW_DIR / f"{site}.npz")
        n = int(keep.sum())
        raw, lead = z["x"][keep], z["lead"][keep]
        assert (z["valid"][keep] <= cutoff_valid).all()
        assert (z["origin"][keep] <= cutoff_origin).all()
        band = np.minimum((lead // LEAD_BAND_HOURS).astype(int), N_BANDS - 1)
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            chunk = fold.standardiser.apply(x=raw[a:b])
            chunk[:, std_cols] = table[band[a:b]]
            if noisy:
                chunk[:, mean_cols] += (
                    rng.standard_normal((b - a, len(WEATHER))).astype(np.float32) * noise[band[a:b]]
                )
                chunk[:, mean_cols] = np.clip(
                    chunk[:, mean_cols], -base.INPUT_CLIP_SD, base.INPUT_CLIP_SD
                )
            out.x[start + a : start + b] = chunk
        out.y[start : start + n] = z["y"][keep]
        out.primary[start : start + n] = primary
        out.is_val[start : start + n] = held[keep]
        out.lead[start : start + n] = lead
        start += n
    return out


# --- checks -----------------------------------------------------------------------------------


def era5_dense_cache() -> dict[str, np.ndarray]:
    """Return every Primary's dense pretraining weather (raw units), loaded once."""
    return {
        r["site"]: dense_weather(series_id=r["time_series_id"], site=r["site"])
        for r in base.primaries().iter_rows(named=True)
    }


def input_checks(*, fold: EnsFold) -> None:
    """Print the unit comparison with ENS and the ratio of ENS-mean error to ENS spread.

    Both are computed on the fold's training rows (held-out weeks excluded). The ENS inputs are
    unstandardised from the fold's standardised matrix.
    """
    mean_cols, std_cols = column_indices()
    st = fold.standardiser
    weather = era5_dense_cache()
    sites = base.primaries()["site"].to_list()
    train = ~fold.is_val
    ens_mean = fold.x[train][:, mean_cols] * st.std[mean_cols] + st.mean[mean_cols]
    ens_std = fold.x[train][:, std_cols] * st.std[std_cols] + st.mean[std_cols]
    valid = fold.valid[train]
    primary = fold.primary[train]
    lead = fold.lead[train]
    idx = ((valid - _us(t=common.GRID_START)) // 1_800_000_000).astype(np.int64)
    era5 = np.full_like(ens_mean, np.nan)
    for p, site in enumerate(sites):
        rows = primary == p
        era5[rows] = weather[site][idx[rows]]
    era5[:, WEATHER.index("precipitation")] = np.log1p(
        np.clip(era5[:, WEATHER.index("precipitation")], 0, None)
    )
    print(
        "variable: ENS mean (sd) | ERA5/CAMS mean (sd) | sd ratio | mean difference | rms diff / ENS spread"
    )
    band = np.minimum((lead // LEAD_BAND_HOURS).astype(int), N_BANDS - 1)
    for c, name in enumerate(WEATHER):
        ok = np.isfinite(era5[:, c])
        a, b = ens_mean[ok, c], era5[ok, c]
        ratio = float(a.std() / max(b.std(), 1e-12))
        diff = a - b
        per_band = [
            float(
                np.sqrt(np.mean(diff[band[ok] == k] ** 2))
                / max(float(np.median(ens_std[ok][band[ok] == k, c])), 1e-12)
            )
            for k in (0, 3, 7, 14)
            if (band[ok] == k).any()
        ]
        print(
            f"{name}: {a.mean():.4g} ({a.std():.4g}) | {b.mean():.4g} ({b.std():.4g}) | {ratio:.2f} | "
            f"{diff.mean():+.4g} | days 1/4/8/15 {[round(v, 2) for v in per_band]}"
        )
        mean_ratio = float(abs(a.mean()) / max(abs(b.mean()), 1e-12))
        if not 1 / 3 <= mean_ratio <= 3:
            msg = f"{name}: ENS and ERA5/CAMS means differ by more than 3x; check the units"
            raise ValueError(msg)


def main() -> None:
    """Build the raw matrices or run the checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("raw", "check"))
    parser.add_argument("--month", default="2026-01")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "raw":
        sites = base.primaries()["site"].to_list()[: 2 if args.smoke else None]
        for site in sites:
            print(write_raw(site=site), flush=True)
        return
    refit = base.refit_of(month=args.month)
    fold = build_ens_fold(refit=refit)
    reference = base.build_train(scheme="summary", refit=refit)
    assert fold.x.shape == reference.x.shape, (fold.x.shape, reference.x.shape)
    assert np.array_equal(fold.is_val, reference.is_val)
    assert float(np.abs(fold.x[::997].astype(np.float64) - reference.x[::997]).max()) == 0.0
    print(
        f"fine-tuning rows {len(fold.x):,} (equal to experiment 397's build_train; {int(fold.is_val.sum()):,} held out)"
    )
    del reference
    for noisy in (False, True):
        pre = fold_pretrain(fold=fold, refit=refit, noisy=noisy, rng_seed=0)
        print(
            f"pretraining rows {len(pre.x):,} ({int(pre.is_val.sum()):,} held out), noisy={noisy}"
        )
        del pre
    input_checks(fold=fold)
    day = np.arange(-5000, 5000)
    held, _ = base.held_out_weeks(origin_day=day)
    phase = (day // 7) % base.HELD_WEEK_PERIOD
    assert (held == (phase == base.HELD_WEEK_PERIOD - 1)).all()
    print("held-out-week phase is consistent for origin days before 2024-04-01")


if __name__ == "__main__":
    main()

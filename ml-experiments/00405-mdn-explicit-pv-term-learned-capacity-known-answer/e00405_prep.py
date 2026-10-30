# ruff: noqa: E501
"""Inputs, injected copies, and folds for the solar-term mixture density network of experiment 405.

It extends experiment 397's `summary` scheme (the ENS mean and spread of each weather column, one
pass) with the extra per-row arrays the solar term needs, the D15 and D18 short-history cut, and
the known-answer injection.

Commands (run with `PYTHONPATH` as `e00405_fit.py` shows):

- `u99`: compute and store the scaling constant of the unit shape (from D10).
- `extras`: write each Primary's extra arrays (unit shape mean and spread, adapter inputs).
- `inject`: write an injected copy of the target and lags (`--farm C --share 0.5 [--placebo]`).
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = pv.WORK_DIR
EXTRAS_DIR: Final[Path] = WORK / "extras"
INJ_DIR: Final[Path] = WORK / "inj"
FARM_FRAME: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
SHORT_HISTORY_SITES: Final[tuple[str, ...]] = ("D15", "D18")
SHORT_HISTORY_CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
"""D15 and D18 drop every row, lag input, and scoring actual before this instant (`common.md`)."""
UNION_FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D")
"""A half-hour with a gap at any of these farms drops out of every known-answer condition, so that
the conditions train on identical rows."""
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI: Final[float] = 100.0
US_PER_HOUR: Final[int] = 3_600_000_000
N_YEAR_CELLS: Final[int] = 3
"""Season-years (September to August) 2023/24, 2024/25, 2025/26: the labels 2023 to 2025."""
N_MONTH_CELLS: Final[int] = 28
"""Months from 2024-04 to 2026-07."""
ADAPTER_COLUMNS: Final[tuple[str, ...]] = (
    "temperature",
    "temperature_sd",
    "longwave",
    "longwave_sd",
    "clearness_index",
    "unit_sd",
    "sin_doy",
    "cos_doy",
    "cos_zenith",
)
DROPPED_FOR_IRRADIANCE_FREE: Final[tuple[str, ...]] = (
    "mean_shortwave",
    "std_shortwave",
    "cos_zenith",
    "sin_azimuth",
    "cos_azimuth",
)
"""The demand-module columns removed in the C arms and A0-noirr: irradiance and solar geometry."""
NAMES, MASK, LAYOUT = base.feature_names(scheme="summary")
LAG_VALUE: Final[tuple[str, ...]] = ("lag_origin", "lag_day", "lag_week")


def col(*, name: str) -> int:
    """Return a raw-matrix column index."""
    return NAMES.index(name)


def kept_columns(*, irradiance_free: bool) -> np.ndarray:
    """Return the indices of the demand-module columns of an arm."""
    drop = set(DROPPED_FOR_IRRADIANCE_FREE) if irradiance_free else set()
    return np.array([i for i, n in enumerate(NAMES) if n not in drop], dtype=np.int64)


def extras_from_raw(*, x: np.ndarray, valid_us: np.ndarray) -> np.ndarray:
    """Return (rows, 2 + 9) float32: the unit shape mean and spread, then the adapter inputs.

    The unit shape mean and spread are divided by `pv.u99()`. The adapter inputs are the ENS mean
    and spread of temperature and longwave radiation, the clear-sky index (the shape mean), its
    spread, the sine and cosine of the day of year, and the cosine of the zenith.
    """
    t = pl.Series(valid_us).cast(pl.Datetime("us"))
    doy = t.dt.ordinal_day().to_numpy().astype(np.float64)
    c = {n: x[:, col(name=n)].astype(np.float64) for n in NAMES[:17]}
    mean, sd = pv.unit_shape(
        ghi_mean=c["mean_shortwave"],
        ghi_sd=c["std_shortwave"],
        cos_zenith=c["cos_zenith"],
        sin_azimuth=c["sin_azimuth"],
        cos_azimuth=c["cos_azimuth"],
        doy=doy,
        temperature_c=c["mean_temperature_2m"],
    )
    scale = pv.u99()
    out = np.empty((len(x), 2 + len(ADAPTER_COLUMNS)), dtype=np.float32)
    out[:, 0], out[:, 1] = mean / scale, sd / scale
    out[:, 2:] = np.stack(
        [
            c["mean_temperature_2m"] / 10.0,
            c["std_temperature_2m"],
            c["mean_longwave"] / 100.0,
            c["std_longwave"] / 10.0,
            np.where(
                c["cos_zenith"] > 0.05,
                np.clip(
                    c["mean_shortwave"] / (1000.0 * np.maximum(c["cos_zenith"], 0.05)), 0.0, 1.5
                ),
                0.0,
            ),
            sd / scale,
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            c["cos_zenith"],
        ],
        axis=1,
    )
    return out


def write_u99() -> str:
    """Store the 99th percentile of the unscaled mean shape over D10's daylight rows."""
    z = np.load(base.RAW_DIR / "summary" / "D10.npz")
    x = z["x"]
    t = pl.Series(z["valid"]).cast(pl.Datetime("us"))
    mean, _ = pv.unit_shape(
        ghi_mean=x[:, col(name="mean_shortwave")].astype(np.float64),
        ghi_sd=x[:, col(name="std_shortwave")].astype(np.float64),
        cos_zenith=x[:, col(name="cos_zenith")].astype(np.float64),
        sin_azimuth=x[:, col(name="sin_azimuth")].astype(np.float64),
        cos_azimuth=x[:, col(name="cos_azimuth")].astype(np.float64),
        doy=t.dt.ordinal_day().to_numpy().astype(np.float64),
        temperature_c=x[:, col(name="mean_temperature_2m")].astype(np.float64),
    )
    daylight = x[:, col(name="cos_zenith")] > 0.05
    value = float(np.percentile(mean[daylight], 99))
    pv.U99_PATH.parent.mkdir(parents=True, exist_ok=True)
    pv.U99_PATH.write_text(json.dumps({"u99": value, "rows": int(daylight.sum())}))
    return f"u99 = {value:.4f} over {int(daylight.sum()):,} daylight rows"


def write_extras(*, site: str) -> str:
    """Write a Primary's extra arrays if absent."""
    target = EXTRAS_DIR / f"{site}.npz"
    if target.exists():
        return f"{site}: exists"
    z = np.load(base.RAW_DIR / "summary" / f"{site}.npz")
    e = extras_from_raw(x=z["x"], valid_us=z["valid"])
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.npz")
    np.savez(tmp, e=e)
    tmp.rename(target)
    return f"{site}: {len(e):,} rows"


# --- the injected farm ---------------------------------------------------------------------


def farm_norm_series(*, farm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a farm's sorted times (microseconds) and power over its 99th percentile.

    The percentile is over the farm's readings with the sun under 75 degrees of zenith, as in
    experiment 129. A reading at or below zero while the irradiance exceeds 100 W/m2 is a dropout
    and returns NaN. Other negative readings are set to zero.
    """
    everything = pl.read_parquet(FARM_FRAME).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    daylight = everything.filter(pl.col("zenith") < FARM_MAX_ZENITH_DEG)
    p99 = float(
        np.nanpercentile(daylight.filter(pl.col("power").is_not_null())["power"].to_numpy(), 99)
    )
    dropout = (pl.col("power") <= 0.0) & (pl.col("ghi_u") > FARM_DROPOUT_GHI)
    frame = everything.select(
        "time",
        norm=pl.when(pl.col("zenith") >= FARM_MAX_ZENITH_DEG)
        .then(0.0)
        .when(pl.col("power").is_null() | dropout)
        .then(float("nan"))
        .otherwise(pl.col("power").clip(lower_bound=0.0) / p99),
    ).sort("time")
    return frame["time"].dt.epoch("us").to_numpy(), frame["norm"].to_numpy().astype(np.float64)


def placebo_day_map(*, days: np.ndarray) -> np.ndarray:
    """Map each day to another day of the same calendar month, by one fixed shuffle per month.

    The map depends only on the calendar month, never on which times were asked for, so the target
    and every lag see the same shuffled sky.
    """
    unique = np.unique(days)
    months = pl.Series(unique * 86_400 * 1_000_000).cast(pl.Datetime("us")).dt.truncate("1mo")
    mapped = {}
    for m in months.unique().to_list():
        first = int(pl.Series([m]).dt.epoch("us")[0]) // (86_400 * 1_000_000)
        n = ((m.replace(day=28) + timedelta(days=4)).replace(day=1) - m).days
        calendar = np.arange(first, first + n)
        rng = np.random.default_rng(405 + first)
        mapped.update(zip(calendar.tolist(), rng.permutation(calendar).tolist(), strict=True))
    return np.array([mapped[d] for d in days.tolist()])


def farm_at(
    *, times_us: np.ndarray, series: tuple[np.ndarray, np.ndarray], placebo: bool
) -> np.ndarray:
    """Return a farm's normalised power at the times; NaN in a dropout or a daytime gap, 0 at night.

    With `placebo`, each day takes the farm's output from another day of the same calendar month
    (a fixed shuffle of the days), so the daylight shape is right and the sky is wrong.
    """
    t_all, v_all = series
    query = times_us.copy()
    if placebo:
        day_us = 86_400 * 1_000_000
        day = query // day_us
        tod = query - day * day_us
        query = placebo_day_map(days=day) * day_us + tod
    pos = np.clip(np.searchsorted(t_all, query), 0, len(t_all) - 1)
    found = t_all[pos] == query
    out = np.where(found, v_all[pos], np.nan)
    hour = ((query // US_PER_HOUR) % 24).astype(np.int64)
    night = (hour < 5) | (hour > 20)
    return np.where(~found & night, 0.0, out)


def injection_tag(*, farm: str, share: float, placebo: bool) -> str:
    """Return the folder name of an injected copy."""
    return f"farm{farm}_share{round(share * 100):03d}" + ("_placebo" if placebo else "")


def studies_capacity_2025(*, site: str) -> float:
    """Return the studies page's capacity (MW) for the 2025/26 season-year."""
    table = pl.read_csv(pv.STUDIES_TABLE).filter(
        pl.col("site") == site, pl.col("season_year") == 2025
    )
    return float(table["capacity_mw"][0])


def write_injection(*, site: str, farm: str, share: float, placebo: bool) -> str:
    """Write an injected copy of a signed Primary's target and lags."""
    tag = injection_tag(farm=farm, share=share, placebo=placebo)
    target = INJ_DIR / tag / f"{site}.npz"
    if target.exists():
        return f"{tag} {site}: exists"
    z = np.load(base.RAW_DIR / "summary" / f"{site}.npz")
    row = next(r for r in base.primaries().iter_rows(named=True) if r["site"] == site)
    increment = share * studies_capacity_2025(site=site) / float(row["effective_capacity_mw"])
    series = farm_norm_series(farm=farm)
    union = [farm_norm_series(farm=f) for f in UNION_FARMS]
    valid = z["valid"]

    def injected(*, times: np.ndarray) -> np.ndarray:
        """Return the farm's output at the times, NaN wherever any of the union farms has a gap."""
        values = farm_at(times_us=times, series=series, placebo=placebo)
        gap = np.zeros(len(times), dtype=bool)
        for other in union:
            gap |= np.isnan(farm_at(times_us=times, series=other, placebo=False))
            gap |= np.isnan(farm_at(times_us=times, series=other, placebo=True))
        return np.where(gap, np.nan, values)

    unique, inverse = np.unique(valid, return_inverse=True)
    y = z["y"].astype(np.float64) - increment * injected(times=unique)[inverse]
    lags = np.empty((len(valid), 3), dtype=np.float32)
    for j, tag_lag in enumerate(("origin", "day", "week")):
        raw = z["x"][:, col(name=f"lag_{tag_lag}")].astype(np.float64)
        age = z["x"][:, col(name=f"lag_{tag_lag}_age")].astype(np.float64) * base.LEAD_SCALE_HOURS
        lag_time = valid - np.rint(age * 2.0).astype(np.int64) * (US_PER_HOUR // 2)
        uniq, inv = np.unique(lag_time, return_inverse=True)
        lags[:, j] = raw - increment * injected(times=uniq)[inv]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.npz")
    np.savez(tmp, y=y.astype(np.float32), lags=lags, increment_points=increment)
    tmp.rename(target)
    return (
        f"{tag} {site}: increment {increment:.4f} of capacity, {int(np.isnan(y).sum()):,} dropouts"
    )


# --- folds ---------------------------------------------------------------------------------


def mask_short_history_lags(*, x: np.ndarray, valid_us: np.ndarray, site: str) -> None:
    """Set the lags that fall before the short-history cut to missing, in place (D15 and D18)."""
    if site not in SHORT_HISTORY_SITES:
        return
    cut_us = int(SHORT_HISTORY_CUT.timestamp() * 1_000_000)
    for tag in ("origin", "day", "week"):
        age = x[:, col(name=f"lag_{tag}_age")].astype(np.float64) * base.LEAD_SCALE_HOURS
        lag_time = valid_us - np.rint(age * 2.0).astype(np.int64) * (US_PER_HOUR // 2)
        before = lag_time < cut_us
        x[before, col(name=f"lag_{tag}")] = np.nan
        x[before, col(name=f"lag_{tag}_missing")] = 1.0


@dataclass
class Fold:
    """The standardised training rows of one fold, with the solar term's side arrays."""

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    weight: np.ndarray
    is_val: np.ndarray
    lead: np.ndarray
    extras: np.ndarray
    year_cell: np.ndarray
    month_cell: np.ndarray
    standardiser: base.Standardiser
    keep: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    as_of_year_cell: int
    as_of_month_cell: int


def _us(*, t: datetime) -> int:
    return int(t.timestamp() * 1_000_000)


def build_fold(*, refit: datetime, irradiance_free: bool, injection: str | None = None) -> Fold:
    """Return a fold's rows: experiment 397's rule, the short-history cut, and optional injection."""
    table = base.primaries()
    parts = []
    for row in table.iter_rows(named=True):
        z = np.load(base.RAW_DIR / "summary" / f"{row['site']}.npz")
        valid, origin = z["valid"], z["origin"]
        keep = (
            (valid <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (origin <= _us(t=refit))
            & (valid > _us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if row["site"] in SHORT_HISTORY_SITES:
            keep &= valid >= _us(t=SHORT_HISTORY_CUT)
        if not keep.any():
            continue
        x = z["x"][keep].copy()
        y = z["y"][keep].astype(np.float32).copy()
        if injection is not None and row["site"] in pv.SIGNED_SITES:
            inj = np.load(INJ_DIR / injection / f"{row['site']}.npz")
            y = inj["y"][keep]
            for j, tag in enumerate(("origin", "day", "week")):
                x[:, col(name=f"lag_{tag}")] = inj["lags"][keep][:, j]
                x[np.isnan(x[:, col(name=f"lag_{tag}")]), col(name=f"lag_{tag}_missing")] = 1.0
        mask_short_history_lags(x=x, valid_us=valid[keep], site=row["site"])
        usable = np.isfinite(y)
        e = np.load(EXTRAS_DIR / f"{row['site']}.npz")["e"][keep]
        parts.append(
            {
                "x": x[usable],
                "y": y[usable],
                "origin": origin[keep][usable],
                "valid": valid[keep][usable],
                "lead": z["lead"][keep][usable],
                "primary": np.full(int(usable.sum()), row["primary"], dtype=np.int64),
                "e": e[usable],
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=base.FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(
        x=cat["x"][train], mask=base.feature_names(scheme="summary")[1]
    )
    age_days = (_us(t=refit) - cat["valid"]) / (86_400 * 1_000_000)
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n = len(table)
    target_mean, target_std = np.zeros(n), np.ones(n)
    for p in range(n):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    cols = kept_columns(irradiance_free=irradiance_free)
    last = refit - timedelta(days=1)
    last_us = np.array([_us(t=last)])
    return Fold(
        x=standardiser.apply(x=cat["x"][keep])[:, cols],
        y=cat["y"][keep],
        primary=cat["primary"][keep],
        weight=weight[keep].astype(np.float32),
        is_val=held[keep],
        lead=cat["lead"][keep],
        extras=cat["e"][keep],
        year_cell=np.clip(pv.season_year_index(valid_us=cat["valid"][keep]), 0, N_YEAR_CELLS - 1),
        month_cell=np.clip(pv.month_index(valid_us=cat["valid"][keep]), 0, N_MONTH_CELLS - 1),
        standardiser=standardiser,
        keep=cols,
        target_mean=target_mean,
        target_std=target_std,
        as_of_year_cell=int(
            np.clip(pv.season_year_index(valid_us=last_us)[0], 0, N_YEAR_CELLS - 1)
        ),
        as_of_month_cell=int(np.clip(pv.month_index(valid_us=last_us)[0], 0, N_MONTH_CELLS - 1)),
    )


def main() -> None:
    """Run a preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("u99", "extras", "inject"))
    parser.add_argument("--farm", default="C")
    parser.add_argument("--share", type=float, default=0.5)
    parser.add_argument("--placebo", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "u99":
        print(write_u99())
        return
    sites = base.primaries()["site"].to_list()
    if args.command == "extras":
        for s in sites[: 2 if args.smoke else None]:
            print(write_extras(site=s), flush=True)
    else:
        for s in pv.SIGNED_SITES[: 2 if args.smoke else None]:
            print(
                write_injection(site=s, farm=args.farm, share=args.share, placebo=args.placebo),
                flush=True,
            )


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Injected series, the day-of-year term, and folds for experiment 409.

It reuses experiment 405's per-Primary extras and `u99`, and copies its fold builder with three
changes: the demand module gets 20 day-of-year by hour inputs, the injected series can be a farm's
day-to-day anomaly or an ensemble control, and the fold keeps the valid times.

Commands (run with `PYTHONPATH` as `e00409_fit.py` shows):

- `inject --condition <name>`: write the injected copy of the target and lags for a condition.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00409-mdn-solar-term-day-of-year-and-anomaly-known-answer" / "work"
)
INJ_DIR: Final[Path] = WORK / "inj"
FARM: Final[str] = "C"
SHARE: Final[float] = 0.5
N_DOY_INPUTS: Final[int] = 20
CONDITIONS: Final[tuple[str, ...]] = (
    "control",
    "held_profile",
    "placebo_profile",
    "held_anomaly",
    "placebo_anomaly",
    "pos_anomaly",
    "pos_profile",
)
N_BENCH_SHUFFLES: Final[int] = 5
DAY_US: Final[int] = 86_400 * 1_000_000


# --- the day-of-year by hour inputs ----------------------------------------------------------


def doy_hour_inputs(*, x_raw: np.ndarray, valid_us: np.ndarray) -> np.ndarray:
    """Return (rows, 20) float32: day-of-year harmonics 1 and 2, alone and times the hour inputs.

    The four hour inputs (`sin_hour`, `cos_hour`, `sin_hour_2`, `cos_hour_2`) are read from the raw
    matrix, which does not standardise them.
    """
    doy = pl.Series(valid_us).cast(pl.Datetime("us")).dt.ordinal_day().to_numpy().astype(np.float64)
    hours = np.stack(
        [x_raw[:, p5.col(name=n)] for n in ("sin_hour", "cos_hour", "sin_hour_2", "cos_hour_2")],
        axis=1,
    ).astype(np.float64)
    cols = []
    for h in (1, 2):
        pair = np.stack(
            [np.sin(2 * np.pi * h * doy / 365.25), np.cos(2 * np.pi * h * doy / 365.25)], axis=1
        )
        cols.append(pair)
        cols.extend(pair[:, [j]] * hours for j in range(2))
    out = np.concatenate(cols, axis=1)
    assert out.shape[1] == N_DOY_INPUTS
    return out.astype(np.float32)


# --- injected series -------------------------------------------------------------------------


def _cell_keys(*, times_us: np.ndarray) -> pl.DataFrame:
    t = pl.Series(times_us).cast(pl.Datetime("us"))
    return pl.DataFrame(
        {
            "t": times_us,
            "ym": t.dt.year().to_numpy() * 100 + t.dt.month().to_numpy(),
            "hh": (t.dt.hour().to_numpy() * 2 + t.dt.minute().to_numpy() // 30),
        }
    )


def split_profile_anomaly(
    *, series: tuple[np.ndarray, np.ndarray]
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return (profile, anomaly) series: the mean per year-month and half-hour, and the remainder.

    The mean ignores gaps (NaN). Each month's anomaly therefore has zero mean at every half-hour.
    """
    t, v = series
    frame = _cell_keys(times_us=t).with_columns(
        v=pl.Series(v, dtype=pl.Float64).fill_nan(None)
    )
    frame = frame.with_columns(profile=pl.col("v").mean().over(["ym", "hh"]))
    profile = frame["profile"].fill_null(0.0).to_numpy()
    anomaly = np.where(np.isnan(v), np.nan, v - profile)
    return (t, profile), (t, anomaly)


def day_map(*, days: np.ndarray, salt: int) -> np.ndarray:
    """Map each day to another day of the same calendar month. Salt 0 is experiment 405's shuffle."""
    unique = np.unique(days)
    months = pl.Series(unique * DAY_US).cast(pl.Datetime("us")).dt.truncate("1mo")
    mapped = {}
    for m in months.unique().to_list():
        first = int(pl.Series([m]).dt.epoch("us")[0]) // DAY_US
        n = ((m.replace(day=28) + timedelta(days=4)).replace(day=1) - m).days
        calendar = np.arange(first, first + n)
        rng = np.random.default_rng(405 + first + 100_003 * salt)
        mapped.update(zip(calendar.tolist(), rng.permutation(calendar).tolist(), strict=True))
    return np.array([mapped[d] for d in days.tolist()])


def series_at(
    *, times_us: np.ndarray, series: tuple[np.ndarray, np.ndarray], shuffle_salt: int | None
) -> np.ndarray:
    """Return a normalised series at the times: NaN at a gap, 0 at night, optionally day-shuffled."""
    t_all, v_all = series
    query = times_us.copy()
    if shuffle_salt is not None:
        day = query // DAY_US
        query = day_map(days=day, salt=shuffle_salt) * DAY_US + (query - day * DAY_US)
    pos = np.clip(np.searchsorted(t_all, query), 0, len(t_all) - 1)
    found = t_all[pos] == query
    out = np.where(found, v_all[pos], np.nan)
    hour = ((query // p5.US_PER_HOUR) % 24).astype(np.int64)
    return np.where(~found & ((hour < 5) | (hour > 20)), 0.0, out)


def ensemble_series(*, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the Primary's ensemble unit shape (mean over rows sharing a valid time) by valid time."""
    z = np.load(base.RAW_DIR / "summary" / f"{site}.npz")
    u = np.load(p5.EXTRAS_DIR / f"{site}.npz")["e"][:, 0].astype(np.float64)
    frame = pl.DataFrame({"t": z["valid"], "u": u}).group_by("t").agg(pl.col("u").mean()).sort("t")
    return frame["t"].to_numpy(), frame["u"].to_numpy()


def condition_series(
    *, condition: str, site: str, salt: int = 0
) -> tuple[tuple[np.ndarray, np.ndarray] | None, int | None]:
    """Return the series to inject for a condition and its shuffle salt (None for no shuffle)."""
    if condition == "control":
        return None, None
    base_name, _, kind = condition.partition("_")
    shuffle = salt if base_name == "placebo" else None
    if kind == "profile" and base_name in ("held", "placebo"):
        return p5.farm_norm_series(farm=FARM), shuffle
    if kind == "anomaly" and base_name in ("held", "placebo"):
        return split_profile_anomaly(series=p5.farm_norm_series(farm=FARM))[1], shuffle
    parts = split_profile_anomaly(series=ensemble_series(site=site))
    return (parts[1] if kind == "anomaly" else parts[0]), None


def write_injection(*, site: str, condition: str) -> str:
    """Write an injected copy of a signed Primary's target and lags for a condition."""
    target = INJ_DIR / condition / f"{site}.npz"
    if target.exists():
        return f"{condition} {site}: exists"
    z = np.load(base.RAW_DIR / "summary" / f"{site}.npz")
    row = next(r for r in base.primaries().iter_rows(named=True) if r["site"] == site)
    increment = SHARE * p5.studies_capacity_2025(site=site) / float(row["effective_capacity_mw"])
    series, salt = condition_series(condition=condition, site=site)
    union = [p5.farm_norm_series(farm=f) for f in p5.UNION_FARMS]
    valid = z["valid"]

    def injected(*, times: np.ndarray) -> np.ndarray:
        """Return the series at the times, NaN wherever any union farm has a gap (as in 405)."""
        gap = np.zeros(len(times), dtype=bool)
        for other in union:
            gap |= np.isnan(series_at(times_us=times, series=other, shuffle_salt=None))
            gap |= np.isnan(series_at(times_us=times, series=other, shuffle_salt=0))
        if series is None:
            return np.where(gap, np.nan, 0.0)
        values = series_at(times_us=times, series=series, shuffle_salt=salt)
        # A time absent from the ensemble series (a lag before the matrix starts) injects nothing.
        return np.where(
            gap, np.nan, np.nan_to_num(values, nan=0.0) if condition.startswith("pos") else values
        )

    unique, inverse = np.unique(valid, return_inverse=True)
    y = z["y"].astype(np.float64) - increment * injected(times=unique)[inverse]
    lags = np.empty((len(valid), 3), dtype=np.float32)
    for j, tag_lag in enumerate(("origin", "day", "week")):
        raw = z["x"][:, p5.col(name=f"lag_{tag_lag}")].astype(np.float64)
        age = (
            z["x"][:, p5.col(name=f"lag_{tag_lag}_age")].astype(np.float64) * base.LEAD_SCALE_HOURS
        )
        lag_time = valid - np.rint(age * 2.0).astype(np.int64) * (p5.US_PER_HOUR // 2)
        uniq, inv = np.unique(lag_time, return_inverse=True)
        lags[:, j] = raw - increment * injected(times=uniq)[inv]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.npz")
    np.savez(tmp, y=y.astype(np.float32), lags=lags, increment_points=increment)
    tmp.rename(target)
    return f"{condition} {site}: increment {increment:.4f}, {int(np.isnan(y).sum()):,} dropouts"


# --- folds -----------------------------------------------------------------------------------


@dataclass
class Fold(p5.Fold):
    """Experiment 405's fold, with the valid times kept for the least-squares read-out."""

    valid: np.ndarray


def build_fold(
    *,
    refit: datetime,
    irradiance_free: bool = True,
    injection: str | None = None,
    use_doy: bool = True,
) -> Fold:
    """Return a fold's rows: experiment 405's rules plus the day-of-year inputs and the valid times."""
    table = base.primaries()
    parts = []
    for row in table.iter_rows(named=True):
        z = np.load(base.RAW_DIR / "summary" / f"{row['site']}.npz")
        valid, origin = z["valid"], z["origin"]
        keep = (
            (valid <= p5._us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (origin <= p5._us(t=refit))
            & (valid > p5._us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if row["site"] in p5.SHORT_HISTORY_SITES:
            keep &= valid >= p5._us(t=p5.SHORT_HISTORY_CUT)
        if not keep.any():
            continue
        x = z["x"][keep].copy()
        y = z["y"][keep].astype(np.float32).copy()
        if injection is not None and injection != "control" and row["site"] in pv.SIGNED_SITES:
            inj = np.load(INJ_DIR / injection / f"{row['site']}.npz")
            y = inj["y"][keep]
            for j, tag in enumerate(("origin", "day", "week")):
                x[:, p5.col(name=f"lag_{tag}")] = inj["lags"][keep][:, j]
                x[np.isnan(x[:, p5.col(name=f"lag_{tag}")]), p5.col(name=f"lag_{tag}_missing")] = (
                    1.0
                )
        elif injection == "control" and row["site"] in pv.SIGNED_SITES:
            inj = np.load(INJ_DIR / "control" / f"{row['site']}.npz")
            y = inj["y"][keep]
        p5.mask_short_history_lags(x=x, valid_us=valid[keep], site=row["site"])
        usable = np.isfinite(y)
        e = np.load(p5.EXTRAS_DIR / f"{row['site']}.npz")["e"][keep]
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
    origin_day = ((cat["origin"] - p5._us(t=base.FIRST_RUN)) // DAY_US).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(
        x=cat["x"][train], mask=base.feature_names(scheme="summary")[1]
    )
    age_days = (p5._us(t=refit) - cat["valid"]) / DAY_US
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n = len(table)
    target_mean, target_std = np.zeros(n), np.ones(n)
    for p in range(n):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    cols = p5.kept_columns(irradiance_free=irradiance_free)
    x_std = standardiser.apply(x=cat["x"][keep])[:, cols]
    if use_doy:
        x_std = np.concatenate(
            [x_std, doy_hour_inputs(x_raw=cat["x"][keep], valid_us=cat["valid"][keep])], axis=1
        )
    last_us = np.array([p5._us(t=refit - timedelta(days=1))])
    return Fold(
        x=x_std,
        y=cat["y"][keep],
        primary=cat["primary"][keep],
        weight=weight[keep].astype(np.float32),
        is_val=held[keep],
        lead=cat["lead"][keep],
        extras=cat["e"][keep],
        year_cell=np.clip(
            pv.season_year_index(valid_us=cat["valid"][keep]), 0, p5.N_YEAR_CELLS - 1
        ),
        month_cell=np.clip(pv.month_index(valid_us=cat["valid"][keep]), 0, p5.N_MONTH_CELLS - 1),
        standardiser=standardiser,
        keep=cols,
        target_mean=target_mean,
        target_std=target_std,
        as_of_year_cell=int(
            np.clip(pv.season_year_index(valid_us=last_us)[0], 0, p5.N_YEAR_CELLS - 1)
        ),
        as_of_month_cell=int(np.clip(pv.month_index(valid_us=last_us)[0], 0, p5.N_MONTH_CELLS - 1)),
        valid=cat["valid"][keep],
    )


def main() -> None:
    """Run a preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inject",))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for cond in args.conditions:
        for s in pv.SIGNED_SITES[: 2 if args.smoke else None]:
            print(write_injection(site=s, condition=cond), flush=True)


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""CAMS and ERA5 inputs, the injected farm, and the fold rows for experiment 417.

This copies the harness of experiment 412 (`e00412_inputs`) and changes three things:

- **Weather inputs are CAMS and ERA5 only.** The demand module reads the irradiance-free columns of
  experiment 405 with every ensemble mean and spread column removed, plus one ERA5 temperature
  column (hourly ERA5 temperature interpolated to the half-hour's midpoint). CAMS `u` and `k` are
  read from experiment 412's arrays, which this experiment does not rewrite.
- **A power factor.** The reading is `sqrt(P'^2 + (tan(phi) y)^2 + Q^2)`, where `P'` is the signed
  power with the farm injected, `y` is the Primary's own signed power before the injection (the
  demand-driven reactive power of experiment 366), and `Q` is the fixed floor.
- **The fold keeps each row's valid time and clear-sky index**, which the gates and labels need.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_prep as prep405  # ty: ignore[unresolved-import]
import e00412_inputs as i412  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_operator import tan_phi_from_power_factor
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

SLUG: Final[str] = "hidden-sign-mva-export-gate-zero-crossing-labels"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00417-{SLUG}" / "work"
ERA5_PATH: Final[Path] = (
    study_dir_for(study="stage1_pilot") / "demand" / "inputs" / "era5_features.parquet"
)
ERA5_COLUMNS: Final[tuple[str, ...]] = ("temperature_2m", "shortwave")
MVA_ONLY: Final[tuple[str, ...]] = i412.MVA_ONLY
SIGNED_SIX: Final[tuple[str, ...]] = i412.SIGNED_SIX
ROTATIONS: Final[dict[str, tuple[str, str]]] = i412.ROTATIONS
FARM_OF_SITE: Final[dict[str, str]] = i412.FARM_OF_SITE
SHORT_HISTORY_SITES: Final[tuple[str, ...]] = i412.SHORT_HISTORY_SITES
US_PER_DAY: Final[int] = i412.US_PER_DAY
US_PER_HALF_HOUR: Final[int] = i412.US_PER_HALF_HOUR
INPUT_CLIP_SD: Final[float] = base.INPUT_CLIP_SD
cams_for_rows = i412.cams_for_rows
shuffled_u = i412.shuffled_u
level_scale = i412.level_scale
hour_and_month = i412.hour_and_month
is_midday = i412.is_midday
SiteArrays = i412.SiteArrays


# --- ERA5 temperature ---------------------------------------------------------------------------


def write_era5() -> str:
    """Write each Primary's ERA5 temperature and shortwave on the dense half-hour grid."""
    out_dir = WORK_DIR / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = common.grid_times().dt.epoch("us").to_numpy().astype(np.float64)
    mid = grid - US_PER_HALF_HOUR / 2.0
    era5 = pl.read_parquet(ERA5_PATH, columns=["time_series_id", "valid_time", *ERA5_COLUMNS])
    lines = []
    for row in base.primaries().iter_rows(named=True):
        part = era5.filter(pl.col("time_series_id") == row["time_series_id"]).sort("valid_time")
        t_us = part["valid_time"].dt.epoch("us").to_numpy().astype(np.float64)
        arrays: dict[str, np.ndarray] = {}
        for name in ERA5_COLUMNS:
            v = part[name].to_numpy().astype(np.float64)
            arrays[name] = np.interp(mid, t_us, v, left=np.nan, right=np.nan).astype(np.float32)
        np.savez(out_dir / f"{row['site']}.npz", **arrays)  # ty: ignore[invalid-argument-type]
        lines.append(f"{row['site']}: {len(part):,} hourly rows")
    return "; ".join(lines)


def era5_for_rows(*, site: str, valid_us: np.ndarray, name: str = "temperature_2m") -> np.ndarray:
    """Return the ERA5 variable at the rows' valid times (NaN where ERA5 is missing)."""
    z = np.load(WORK_DIR / "era5" / f"{site}.npz")
    return z[name][i412.grid_index(valid_us=valid_us)].astype(np.float32)


# --- one Primary's reading matrix ---------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """How the folded pair's meter reading is made from the signed power.

    Attributes:
        level: The injected farm's size as a multiple of the Primary's median sunny-midday reading
            before 2025-07-01 (0 means no injection, the placebo).
        q_points: The reactive floor `Q`, in points of capacity.
        power_factor: The power factor of the Primary's own demand (1.0 means no proportional
            reactive power).
    """

    level: float
    q_points: float
    power_factor: float = 1.0
    farm_cap: float = 1.0
    """The injected farm is capped at this fraction of its own peak (1.0 means no cap)."""

    @property
    def tag(self) -> str:
        """Return a file-name tag."""
        pf = "" if self.power_factor == 1.0 else f"_PF{self.power_factor:g}"
        cap = "" if self.farm_cap == 1.0 else f"_CAP{self.farm_cap:g}"
        return f"L{self.level:g}_Q{self.q_points:g}{pf}{cap}"

    @property
    def tan_phi(self) -> float:
        """Return `tan(phi)` of the demand-driven reactive power."""
        return tan_phi_from_power_factor(power_factor=self.power_factor)


def parse_tag(*, tag: str) -> Setting:
    """Parse `L2_Q2` or `L2_Q2_PF0.97` into a setting."""
    parts = tag.split("_")
    pf = float(parts[2][2:]) if len(parts) > 2 else 1.0
    return Setting(float(parts[0][1:]), float(parts[1][1:]), pf)


def reading_from(
    *, signed_after: np.ndarray, signed_before: np.ndarray, setting: Setting
) -> np.ndarray:
    """Return the meter reading `sqrt(P'^2 + (tan(phi) y)^2 + Q^2)` in fractions of capacity."""
    q = setting.q_points / 100.0
    return np.sqrt(signed_after**2 + (setting.tan_phi * signed_before) ** 2 + q**2)


def site_arrays(*, site: str, fold: bool, setting: Setting | None) -> SiteArrays:
    """Return a Primary's raw rows with the reading and the lags as the meter gives them.

    Same as experiment 412's function, with the reading defined by `reading_from`.
    """
    z = np.load(base.RAW_DIR / "summary" / f"{site}.npz")
    x = z["x"].copy()
    y = z["y"].astype(np.float64).copy()
    valid, origin = z["valid"], z["origin"]
    y_truth = y.copy()
    if fold and site in SIGNED_SIX and setting is not None:
        increment = (
            level_scale(site=site, y_signed=y, valid_us=valid, level=setting.level)
            if setting.level > 0
            else 0.0
        )
        farm = prep405.farm_norm_series(farm=FARM_OF_SITE[site])
        others = [prep405.farm_norm_series(farm=f) for f in sorted(set(FARM_OF_SITE.values()))]

        peak = float(np.nanmax(farm[1]))

        def injected(*, times: np.ndarray) -> np.ndarray:
            values = prep405.farm_at(times_us=times, series=farm, placebo=False)
            if setting.farm_cap < 1.0:
                values = np.minimum(values, setting.farm_cap * peak)
            gap = np.zeros(len(times), dtype=bool)
            for other in others:
                gap |= np.isnan(prep405.farm_at(times_us=times, series=other, placebo=False))
            return np.where(gap, np.nan, values)

        unique, inverse = np.unique(valid, return_inverse=True)
        y_truth = y - increment * injected(times=unique)[inverse]
        y_read = reading_from(signed_after=y_truth, signed_before=y, setting=setting)
        for tag in ("origin", "day", "week"):
            raw = x[:, prep405.col(name=f"lag_{tag}")].astype(np.float64)
            age = (
                x[:, prep405.col(name=f"lag_{tag}_age")].astype(np.float64) * base.LEAD_SCALE_HOURS
            )
            lag_time = valid - np.rint(age * 2.0).astype(np.int64) * (prep405.US_PER_HOUR // 2)
            uniq, inv = np.unique(lag_time, return_inverse=True)
            signed_lag = raw - increment * injected(times=uniq)[inv]
            x[:, prep405.col(name=f"lag_{tag}")] = reading_from(
                signed_after=signed_lag, signed_before=raw, setting=setting
            )
            gone = np.isnan(x[:, prep405.col(name=f"lag_{tag}")])
            x[gone, prep405.col(name=f"lag_{tag}_missing")] = 1.0
        y = y_read
    prep405.mask_short_history_lags(x=x, valid_us=valid, site=site)
    return SiteArrays(
        x=x,
        y=y.astype(np.float32),
        y_truth=y_truth.astype(np.float32),
        valid=valid,
        origin=origin,
        lead=z["lead"],
    )


# --- the fold -----------------------------------------------------------------------------------


def demand_columns() -> np.ndarray:
    """Return the indices of the demand module's columns: no irradiance, no ensemble weather."""
    drop = set(prep405.DROPPED_FOR_IRRADIANCE_FREE)
    return np.array(
        [
            i
            for i, n in enumerate(prep405.NAMES)
            if n not in drop and not n.startswith(("mean_", "std_"))
        ],
        dtype=np.int64,
    )


@dataclass
class TemperatureScaler:
    """The training mean and standard deviation of the ERA5 temperature column."""

    mean: float
    std: float

    def apply(self, *, t: np.ndarray) -> np.ndarray:
        """Standardise, clip, and fill missing values with zero."""
        out = np.clip((t - self.mean) / self.std, -INPUT_CLIP_SD, INPUT_CLIP_SD)
        return np.nan_to_num(out, nan=0.0).astype(np.float32)


def model_inputs(
    *,
    x_raw: np.ndarray,
    valid_us: np.ndarray,
    site: str,
    standardiser: base.Standardiser,
    scaler: TemperatureScaler,
    cols: np.ndarray,
) -> np.ndarray:
    """Return the demand module's input matrix for rows: standardised columns plus ERA5 temperature."""
    t = scaler.apply(t=era5_for_rows(site=site, valid_us=valid_us))
    return np.concatenate([standardiser.apply(x=x_raw)[:, cols], t[:, None]], axis=1)


@dataclass
class Fold:
    """The standardised training rows of one refit, with CAMS `u`, `k`, the valid time, and flags.

    `u` is the solar term's input (permuted in the shuffle arm); `gate_u` is the real scaled unit
    shape. `valid` and `k` serve the Primary-level gate, the high-irradiance gate, and the labels.
    """

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    weight: np.ndarray
    is_val: np.ndarray
    u: np.ndarray
    gate_u: np.ndarray
    k: np.ndarray
    valid: np.ndarray
    standardiser: base.Standardiser
    scaler: TemperatureScaler
    cols: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    folded: np.ndarray
    setting: Setting
    pair: tuple[str, ...]
    n_dropped: dict[str, int]


def build_fold(*, refit, pair: tuple[str, ...], setting: Setting, shuffle_u: bool = False) -> Fold:  # noqa: ANN001
    """Return a refit's training rows: experiment 397's rule, the short-history cut, the fold."""
    table = base.primaries()
    parts = []
    dropped: dict[str, int] = {}
    for row in table.iter_rows(named=True):
        site = row["site"]
        arr = site_arrays(site=site, fold=site in pair, setting=setting)
        keep = (
            (arr.valid <= i412._us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (arr.origin <= i412._us(t=refit))
            & (arr.valid > i412._us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if site in SHORT_HISTORY_SITES:
            keep &= arr.valid >= i412._us(t=prep405.SHORT_HISTORY_CUT)
        usable = keep & np.isfinite(arr.y)
        dropped[site] = int(keep.sum() - usable.sum())
        if not usable.any():
            continue
        u_real, k_real = cams_for_rows(site=site, valid_us=arr.valid[usable])
        u = shuffled_u(site=site, valid_us=arr.valid[usable]) if shuffle_u else u_real
        parts.append(
            {
                "x": arr.x[usable],
                "y": arr.y[usable],
                "origin": arr.origin[usable],
                "valid": arr.valid[usable],
                "primary": np.full(int(usable.sum()), row["primary"], dtype=np.int64),
                "u": np.nan_to_num(u, nan=0.0).astype(np.float32),
                "g": np.nan_to_num(u_real, nan=0.0).astype(np.float32),
                "k": np.nan_to_num(k_real, nan=0.0).astype(np.float32),
                "t": era5_for_rows(site=site, valid_us=arr.valid[usable]),
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - i412._us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=i412.NAMES_MASK)
    scaler = TemperatureScaler(
        mean=float(np.nanmean(cat["t"][train])), std=max(float(np.nanstd(cat["t"][train])), 1e-3)
    )
    age_days = (i412._us(t=refit) - cat["valid"]) / US_PER_DAY
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n = len(table)
    target_mean, target_std = np.zeros(n), np.ones(n)
    for p in range(n):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep_rows = train | held
    cols = demand_columns()
    x_std = standardiser.apply(x=cat["x"][keep_rows])[:, cols]
    x = np.concatenate([x_std, scaler.apply(t=cat["t"][keep_rows])[:, None]], axis=1)
    return Fold(
        x=x,
        y=cat["y"][keep_rows],
        primary=cat["primary"][keep_rows],
        weight=weight[keep_rows].astype(np.float32),
        is_val=held[keep_rows],
        u=cat["u"][keep_rows],
        gate_u=cat["g"][keep_rows],
        k=cat["k"][keep_rows],
        valid=cat["valid"][keep_rows],
        standardiser=standardiser,
        scaler=scaler,
        cols=cols,
        target_mean=target_mean,
        target_std=target_std,
        folded=i412.folded_flags(pair=pair),
        setting=setting,
        pair=pair,
        n_dropped=dropped,
    )

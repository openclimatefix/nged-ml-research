# ruff: noqa: E501
"""ECMWF ensemble (ENS) mean and spread inputs in place of CAMS and ERA5, for experiment 424.

Experiment 417's fold (`e00417_inputs.build_fold`) reads three realised weather series that exist
at one value per valid time: the CAMS scaled unit shape `u`, the CAMS clear-sky index `k`, and ERA5
temperature. This module builds the same fold from the ENS mean and spread, which differ per row
(each row is one weather-model run at one lead). The weather variables are the same: irradiance
(through the unit shape and the clearness index) and temperature. Everything else is experiment
417's fold: rows, lags, injected farm, short-history cut, held-out weeks, weights, and targets.

The per-row ENS values come from experiment 405's `extras` arrays (`e00405_prep.extras_from_raw`),
which are row-aligned with experiment 397's `summary` arrays. Column meaning:

- `u` (column 0): the unit shape from the ENS mean shortwave irradiance, divided by `u99`.
- `unit_sd` (column 1): the spread of the unit shape across the 51 members, divided by `u99`.
- temperature (column 2 times 10, in degrees Celsius) and its spread across members (column 3).
- clearness index (column 6): the ENS mean shortwave over the clear-sky value, clipped to 0 to 1.5.

The demand module sees the standardised experiment 397 columns without any ensemble weather
columns (as in experiment 417), plus three ENS columns: temperature mean, temperature spread, and
unit-shape spread. The PV term sees the ENS unit-shape mean. The Primary gate and the open mask use
the ENS unit-shape mean and clearness index, because those are what a live service would hold.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_prep as prep405  # ty: ignore[unresolved-import]
import e00412_inputs as i412  # ty: ignore[unresolved-import]
import e00417_gates as g417  # ty: ignore[unresolved-import]
import e00417_inputs as i417  # ty: ignore[unresolved-import]
import numpy as np

COL_U: Final[int] = 0
COL_UNIT_SD: Final[int] = 1
COL_TEMP_C10: Final[int] = 2
COL_TEMP_SD: Final[int] = 3
COL_CLEARNESS: Final[int] = 6
N_ENS_COLUMNS: Final[int] = 3
"""Temperature mean, temperature spread, unit-shape spread."""
INPUT_CLIP_SD: Final[float] = base.INPUT_CLIP_SD
US_PER_DAY: Final[int] = i417.US_PER_DAY


@dataclass
class EnsFold:
    """Experiment 417's fold (same fields) plus each row's lead in hours, which the gate needs."""

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
    scaler: EnsScaler
    cols: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    folded: np.ndarray
    setting: i417.Setting
    pair: tuple[str, ...]
    n_dropped: dict[str, int]
    lead: np.ndarray


def primary_gates(*, fold: EnsFold) -> dict[str, g417.PrimaryGate]:
    """Return the Primary gates on each valid time's shortest-lead row.

    Experiment 417's gate keeps the first row of each valid time, which is the longest-lead row
    when the weather differs per run (ENS) and so blurs the unit shape. This keeps the shortest-lead
    row instead, so the gate statistic uses the ENS forecast closest to the valid time.
    """
    order = np.lexsort((fold.lead, fold.valid, fold.primary))
    reduced = replace(
        fold,
        primary=fold.primary[order],
        valid=fold.valid[order],
        y=fold.y[order],
        gate_u=fold.gate_u[order],
        k=fold.k[order],
    )
    return g417.primary_gates(fold=reduced)


@lru_cache(maxsize=4)
def _extras(*, site: str) -> np.ndarray:
    """Return a Primary's experiment 405 extras array, (rows, 11) float32, row-aligned with `summary`."""
    return np.load(prep405.EXTRAS_DIR / f"{site}.npz")["e"]


def ens_arrays(*, site: str) -> dict[str, np.ndarray]:
    """Return a Primary's per-row ENS values, aligned with `i417.site_arrays` (all rows, not filtered).

    Keys: `u` (unit-shape mean over `u99`), `k` (clearness index), and `inputs` (rows, 3): ENS
    temperature in degrees Celsius, its spread, and the unit-shape spread. Missing values are NaN.
    """
    e = _extras(site=site)
    return {
        "u": e[:, COL_U].astype(np.float32),
        "k": e[:, COL_CLEARNESS].astype(np.float32),
        "inputs": np.stack(
            [e[:, COL_TEMP_C10] * 10.0, e[:, COL_TEMP_SD], e[:, COL_UNIT_SD]], axis=1
        ).astype(np.float32),
    }


@dataclass
class EnsScaler:
    """The training mean and standard deviation of the three ENS input columns."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, *, e: np.ndarray) -> np.ndarray:
        """Standardise, clip, and fill missing values with zero (rows, 3)."""
        out = np.clip((e - self.mean) / self.std, -INPUT_CLIP_SD, INPUT_CLIP_SD)
        return np.nan_to_num(out, nan=0.0).astype(np.float32)


def model_inputs(
    *,
    x_raw: np.ndarray,
    ens_inputs: np.ndarray,
    standardiser: base.Standardiser,
    scaler: EnsScaler,
    cols: np.ndarray,
) -> np.ndarray:
    """Return the demand module's input matrix: standardised columns plus the three ENS columns."""
    return np.concatenate(
        [standardiser.apply(x=x_raw)[:, cols], scaler.apply(e=ens_inputs)], axis=1
    )


def build_fold(
    *,
    refit: datetime,
    pair: tuple[str, ...],
    setting: i417.Setting,
) -> tuple[EnsFold, dict[str, int]]:
    """Return a refit's training rows, as experiment 417's `build_fold` but with ENS weather.

    Row selection, standardisation of the experiment 397 columns, weights, targets, and held-out
    weeks follow `e00417_inputs.build_fold`. The CAMS and ERA5 series are replaced by the ENS mean
    and spread, and the fold's `scaler` is an `EnsScaler`.
    """
    table = base.primaries()
    parts = []
    dropped: dict[str, int] = {}
    missing: dict[str, int] = {}
    for row in table.iter_rows(named=True):
        site = row["site"]
        arr = i417.site_arrays(site=site, fold=site in pair, setting=setting)
        ens = ens_arrays(site=site)
        if len(ens["u"]) != len(arr.valid):
            msg = f"{site}: extras have {len(ens['u'])} rows, summary has {len(arr.valid)}"
            raise ValueError(msg)
        keep = (
            (arr.valid <= i412._us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (arr.origin <= i412._us(t=refit))
            & (arr.valid > i412._us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if site in i417.SHORT_HISTORY_SITES:
            keep &= arr.valid >= i412._us(t=prep405.SHORT_HISTORY_CUT)
        usable = keep & np.isfinite(arr.y)
        dropped[site] = int(keep.sum() - usable.sum())
        if not usable.any():
            continue
        u = ens["u"][usable]
        missing[site] = int(np.isnan(u).sum())
        parts.append(
            {
                "x": arr.x[usable],
                "y": arr.y[usable],
                "origin": arr.origin[usable],
                "lead": arr.lead[usable],
                "valid": arr.valid[usable],
                "primary": np.full(int(usable.sum()), row["primary"], dtype=np.int64),
                "u": np.nan_to_num(u, nan=0.0).astype(np.float32),
                "k": np.nan_to_num(ens["k"][usable], nan=0.0).astype(np.float32),
                "e": ens["inputs"][usable],
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - i412._us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=i412.NAMES_MASK)
    scaler = EnsScaler(
        mean=np.nanmean(cat["e"][train], axis=0).astype(np.float32),
        std=np.maximum(np.nanstd(cat["e"][train], axis=0), 1e-3).astype(np.float32),
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
    cols = i417.demand_columns()
    x = model_inputs(
        x_raw=cat["x"][keep_rows],
        ens_inputs=cat["e"][keep_rows],
        standardiser=standardiser,
        scaler=scaler,
        cols=cols,
    )
    return EnsFold(
        x=x,
        y=cat["y"][keep_rows],
        primary=cat["primary"][keep_rows],
        weight=weight[keep_rows].astype(np.float32),
        is_val=held[keep_rows],
        u=cat["u"][keep_rows],
        gate_u=cat["u"][keep_rows],
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
        lead=cat["lead"][keep_rows],
    ), missing

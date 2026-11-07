# ruff: noqa: E501, D107, PT018
"""Experiment 632: dense inputs and window sampling for the GRU plus self-attention network.

**Inputs on the dense half-hourly grid.** For each of the 16 Primaries the network reads, at every
half-hour, 28 weather and datetime columns: the 7 ERA5 and CAMS weather columns of experiment 418's
pre-training and the 21 solar-geometry, calendar, and holiday columns that experiment 397's
`feature_matrix` builds. The columns are standardised with the mean and standard deviation of the
fold's training region, clipped at 5 standard deviations, and stored as float16 on the GPU. The
whole array is about 110 MB, so a batch is assembled on the GPU by indexing, with no data loader.

**Windows.** A sample is a Primary and a forecast origin (a grid index). Its history tokens are the
`n_hist` half-hours ending at the origin, and its future tokens are the `n_fut` half-hours after it.
Power is visible at a history token only if the half-hour is at least 24 h (48 steps) before the
origin and the reading is not flagged. Everything newer is masked, so the last 24 h of the history
window carry weather and datetime inputs and no power.

**Neighbours.** For each Primary, the neighbour summary at a half-hour is the mean standardised
power of its electrical neighbours that are metered (experiment 383's table through experiment
400's `nged_neighbours`) and the share of those neighbours with a reading. The summary obeys the
same 24 h mask as the Primary's own power.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00559_extra as extra  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00632-gru-attention-sequence-network-for-primary-demand"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
CACHE: Final[Path] = ROOT / "cache"
STEPS_PER_DAY: Final[int] = 48
LAG_STEPS: Final[int] = STEPS_PER_DAY
"""Power is visible only at half-hours at least this many steps (24 h) before the origin."""
N_WEATHER: Final[int] = 7
N_DATETIME: Final[int] = 21
N_COLUMNS: Final[int] = N_WEATHER + N_DATETIME
WEATHER_IN_46: Final[list[int]] = list(range(7))
DATETIME_IN_46: Final[list[int]] = list(range(14, 35))
CLIP_SD: Final[float] = prep.INPUT_CLIP_SD
ORIGIN_HOUR_UTC: Final[int] = 9
"""Forecast origins of the roster export are 09:00 UTC."""


def dense_raw() -> np.ndarray:
    """Return the raw (unstandardised) 28 columns (primaries, grid, 28); NaN where weather is missing.

    Weather is `e00418_prep.dense_weather` (ERA5 plus CAMS shortwave). The datetime columns are the
    ones `e00397_prep.feature_matrix` builds, so they match experiment 397's columns 14 to 34.
    """
    f = CACHE / "dense_raw.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    grid = common.grid_times()
    table = prep.primaries()
    out = np.full((table.height, grid.len(), N_COLUMNS), np.nan, dtype=np.float32)
    for row in table.iter_rows(named=True):
        weather = prep418.dense_weather(series_id=row["time_series_id"], site=row["site"])
        derived = common.derived_columns(
            valid_time=grid, latitude=row["latitude"], longitude=row["longitude"]
        )
        columns: dict[str, np.ndarray] = {}
        for k, name in enumerate(prep.WEATHER):
            columns[f"mean_{name}"] = np.nan_to_num(weather[:, k], nan=0.0)
            columns[f"std_{name}"] = np.zeros(grid.len(), dtype=np.float32)
        for name in common.DERIVED:
            columns[name] = derived[name].to_numpy()
        for name in common.LAG_COLUMNS:
            columns[name] = np.zeros(grid.len(), dtype=np.float32)
        columns["lead_hours"] = np.zeros(grid.len(), dtype=np.float32)
        frame = pl.DataFrame(columns).with_columns(valid_time=grid, power_fcst_init_time=grid)
        x = prep.feature_matrix(frame=frame, scheme="summary")
        out[row["primary"], :, :N_WEATHER] = np.where(
            np.isfinite(weather), x[:, WEATHER_IN_46], np.nan
        )
        out[row["primary"], :, N_WEATHER:] = x[:, DATETIME_IN_46]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, out)
    return out


def clean_power() -> np.ndarray:
    """Return the cleaned power over capacity (primaries, grid), NaN where flagged or cut."""
    f = CACHE / "clean_power.npy"
    if f.exists():
        return np.load(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = c510.clean_stack()
    np.save(f, out)
    return out


def neighbour_lists() -> list[list[int]]:
    """Return, per Primary index, the Primary indices of its metered electrical neighbours."""
    import e00400_blend as b400  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sites = k578.sites()["label"].to_list()
    index = {label: i for i, label in enumerate(sites)}
    of: list[list[int]] = [[] for _ in sites]
    for label, near in b400.nged_neighbours().items():
        if label in index:
            of[index[label]] = [index[x] for x in near if x in index]
    return of


B1_GRID_NAMES: Final[tuple[str, ...]] = (
    "t_mean_3h", "t_mean_6h", "t_mean_24h", "t_change_3h", "t_change_6h", "t_min_day",
    "t_max_day", "t_cold_snap_days", "sw_mean_3h", "sw_mean_6h", "sw_mean_24h", "csi_change_3h",
)  # fmt: skip
"""The 12 B1 columns that depend only on the valid time; the other two are `cams_prev_day_*`."""
N_B1_GRID: Final[int] = len(B1_GRID_NAMES)
N_B1: Final[int] = N_B1_GRID + 2
DAY_US: Final[int] = 86_400 * 1_000_000
DAY0: Final[int] = prep418.PRETRAIN_START.toordinal() - 719163 - 5
"""A day code a few days before the grid start (days since 1970-01-01)."""
N_DAYS: Final[int] = 2700


def dense_b1() -> tuple[np.ndarray, np.ndarray]:
    """Return raw B1 columns: (primaries, grid, 12) valid-time columns and (primaries, N_DAYS, 2) CAMS day columns.

    The columns are experiment 437's, built exactly as `e00437_features.pretrain_extras` builds them
    for pre-training rows: trailing temperature and shortwave windows at the valid time (from the
    dense ERA5 and CAMS weather), and the CAMS mean irradiance and clear-sky ratio of the UTC day
    before the origin's day.
    """
    f1, f2 = CACHE / "dense_b1_grid.npy", CACHE / "dense_b1_cams.npy"
    if f1.exists() and f2.exists():
        return np.load(f1, mmap_mode="r"), np.load(f2)
    local_date = f437.local_date_codes()
    table = prep.primaries()
    grid = np.full((table.height, len(local_date), N_B1_GRID), np.nan, dtype=np.float32)
    cams = np.full((table.height, N_DAYS, 2), np.nan, dtype=np.float32)
    for row in table.iter_rows(named=True):
        dense = f437.dense_arrays(row=row, local_date=local_date)
        for k, name in enumerate(B1_GRID_NAMES):
            grid[row["primary"], :, k] = dense[name]
        for day, (mean, csi) in f437.cams_previous_day(site=row["site"]).items():
            if 0 <= day - DAY0 < N_DAYS:
                cams[row["primary"], day - DAY0] = (mean, csi)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f1, grid)
    np.save(f2, cams)
    return grid, cams


def b1_scale(
    *, grid: np.ndarray, cams: np.ndarray, mean: np.ndarray, std: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardise B1 with the fold's stored mean and standard deviation (14 numbers); missing becomes 0."""
    g = (np.asarray(grid) - mean[:N_B1_GRID]) / std[:N_B1_GRID]
    c = (np.asarray(cams) - mean[N_B1_GRID:]) / std[N_B1_GRID:]
    g = np.nan_to_num(g, nan=0.0).astype(np.float16)
    c = np.nan_to_num(c, nan=0.0).astype(np.float32)
    return torch.from_numpy(g).to(device), torch.from_numpy(c).to(device)


def tau_grid() -> np.ndarray:
    """Return the D5 knot position (months since 2019-09, fractional) of every grid half-hour."""
    us = (
        prep422.GRID_START_US
        + np.arange(common.grid_times().len(), dtype=np.int64) * prep422.STEP_US
    )
    return extra.DriftColumns.tau(valid_us=us)


@dataclass
class Fold:
    """Statistics of one fold's training region, applied unchanged to every window."""

    mean: np.ndarray  # (28,)
    std: np.ndarray  # (28,)
    standardise: np.ndarray  # (28,) bool
    target_mean: np.ndarray  # (primaries,) power over capacity
    target_std: np.ndarray  # (primaries,)
    train_end: int  # grid index of the first half-hour not used (refit minus 3.5 h)
    first_valid: np.ndarray  # (primaries,) first grid index with a reading


def standardised_inputs(
    *, raw: np.ndarray, fold: Fold, device: str, b1_grid: torch.Tensor | None = None
) -> torch.Tensor:
    """Return (primaries, grid, 28 [+ 12]) standardised inputs as float16 on `device`; NaN becomes 0."""
    out = np.empty(raw.shape, dtype=np.float16)
    for p in range(raw.shape[0]):
        z = np.clip((np.asarray(raw[p]) - fold.mean) / fold.std, -CLIP_SD, CLIP_SD)
        out[p] = np.nan_to_num(z, nan=0.0).astype(np.float16)
    x = torch.from_numpy(out).to(device)
    return x if b1_grid is None else torch.cat([x, b1_grid], dim=-1)


def standardised_power(*, power: np.ndarray, fold: Fold) -> np.ndarray:
    """Return z = (power - mean) / std per Primary, NaN where the reading is flagged or absent."""
    return ((power - fold.target_mean[:, None]) / fold.target_std[:, None]).astype(np.float32)


def neighbour_summary(*, z: np.ndarray, of: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean z of the neighbours, share of neighbours with a reading), both (primaries, grid)."""
    mean = np.zeros_like(z)
    share = np.zeros_like(z)
    for p, near in enumerate(of):
        if not near:
            continue
        block = z[near]
        ok = np.isfinite(block)
        total = np.where(ok, block, 0.0).sum(axis=0)
        count = ok.sum(axis=0)
        mean[p] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        share[p] = count / len(near)
    return mean, share


@dataclass
class Batch:
    """One batch of windows, all tensors on the device."""

    primary: torch.Tensor  # (B,)
    origin: torch.Tensor  # (B,) grid index
    x_hist: torch.Tensor  # (B, H, 40) weather, datetime, 12 B1 grid columns
    x_fut: torch.Tensor  # (B, F, 40)
    cams: torch.Tensor  # (B, 2) CAMS previous-day columns of the origin's day
    power: torch.Tensor  # (B, H) standardised own power, 0 where masked
    power_missing: torch.Tensor  # (B, H) 1 where masked
    nb_mean: torch.Tensor  # (B, H)
    nb_share: torch.Tensor  # (B, H)
    age: torch.Tensor  # (B, H) days before the origin
    target: torch.Tensor  # (B, F) standardised power; NaN where flagged or after the cut
    tau: torch.Tensor  # (B, F) D5 knot position of each future token
    hist_idx: torch.Tensor  # (B, H)
    fut_idx: torch.Tensor  # (B, F)


class Windows:
    """Holds the dense arrays on the device and cuts windows out of them by index."""

    def __init__(
        self, *, x: torch.Tensor, z: np.ndarray, nb_mean: np.ndarray, nb_share: np.ndarray,
        cams: torch.Tensor, tau: np.ndarray, n_hist: int, n_fut: int, device: str,
        lag_steps: int = LAG_STEPS, z_in: np.ndarray | None = None,
    ) -> None:  # fmt: skip
        """Hold `z` as the targets; the history power and neighbour summaries come from `z_in` if given.

        `z_in` is a lag-only altered copy of `z` (a synthetic or injected load-shift); the neighbour
        summaries passed in must have been computed from `z_in` too.
        """
        self.x, self.cams_arr = x, cams
        self.z = torch.from_numpy(z).to(device)
        self.z_in = self.z if z_in is None else torch.from_numpy(z_in).to(device)
        self.nb_mean = torch.from_numpy(nb_mean).to(device)
        self.nb_share = torch.from_numpy(nb_share).to(device)
        self.tau = torch.from_numpy(tau).to(device)
        self.n_hist, self.n_fut, self.device, self.lag_steps = n_hist, n_fut, device, lag_steps
        self.hist_off = torch.arange(-n_hist + 1, 1, device=device)
        self.fut_off = torch.arange(1, n_fut + 1, device=device)
        self.grid = x.shape[1]

    def cut(
        self, *, primary: torch.Tensor, origin: torch.Tensor, target_end: int | None = None
    ) -> Batch:
        """Return the windows of (primary, origin) pairs; targets at grid indices >= `target_end` are NaN."""
        h = origin[:, None] + self.hist_off[None, :]
        f = origin[:, None] + self.fut_off[None, :]
        assert int(h.min()) >= 0 and int(f.max()) < self.grid, "window leaves the grid"
        p = primary[:, None]
        visible = h <= (origin[:, None] - self.lag_steps)
        z = self.z_in[p, h]
        seen = visible & torch.isfinite(z)
        target = self.z[p, f].clone()
        if target_end is not None:
            target[f >= target_end] = float("nan")
        day_us = prep422.GRID_START_US + origin.cpu().numpy().astype(np.int64) * prep422.STEP_US
        day = torch.from_numpy(day_us // DAY_US - 1 - DAY0).to(self.device)
        return Batch(
            primary=primary, origin=origin, x_hist=self.x[p, h], x_fut=self.x[p, f],
            cams=self.cams_arr[primary, day], power=torch.where(seen, z, torch.zeros_like(z)),
            power_missing=(~seen).float(),
            nb_mean=torch.where(visible, self.nb_mean[p, h], torch.zeros_like(z)),
            nb_share=torch.where(visible, self.nb_share[p, h], torch.zeros_like(z)),
            age=(origin[:, None] - h).float() / STEPS_PER_DAY, target=target,
            tau=self.tau[f], hist_idx=h, fut_idx=f,
        )  # fmt: skip


def check_cheap(*, batch: Batch) -> None:
    """Row-by-row checks that cost nothing: run on every training batch.

    No own-power or neighbour value is newer than origin minus 24 h; the masked tail carries zeros;
    the targets are future tokens only (every target index is after the origin).
    """
    origin = batch.origin[:, None]
    newest = batch.hist_idx.masked_fill(batch.power_missing > 0.5, -(10**9)).amax(dim=1)
    assert (newest <= batch.origin - LAG_STEPS).all(), "own power newer than origin - 24 h"
    nb_seen = (batch.nb_share > 0) | (batch.nb_mean != 0)
    nb_newest = batch.hist_idx.masked_fill(~nb_seen, -(10**9)).amax(dim=1)
    assert (nb_newest <= batch.origin - LAG_STEPS).all(), "neighbour power newer than origin - 24 h"
    tail = batch.hist_idx > (origin - LAG_STEPS)
    assert (batch.power[tail] == 0).all()
    assert (batch.power_missing[tail] == 1).all()
    assert (batch.nb_mean[tail] == 0).all()
    assert (batch.nb_share[tail] == 0).all()
    assert (batch.fut_idx > origin).all(), "a future token is not after the origin"


def check_poisoned(*, batch: Batch, windows: Windows, limit: int = 64) -> str:
    """Per-sample check against a poisoned copy of the power arrays.

    For each of the first `limit` samples, every reading newer than origin minus 24 h of the sample's
    Primary (own power and neighbour summaries) is replaced by 1e6 in a private copy of that row, and
    the expected history tensors are rebuilt from the copy. They must equal the batch's tensors.
    """
    n = min(limit, len(batch.origin))
    for i in range(n):
        p, o = int(batch.primary[i]), int(batch.origin[i])
        rows = []
        for arr in (windows.z_in, windows.nb_mean, windows.nb_share):
            r = arr[p].clone()
            r[o - LAG_STEPS + 1 :] = 1e6
            rows.append(r)
        z, nm, ns = (r[batch.hist_idx[i]] for r in rows)
        visible = batch.hist_idx[i] <= o - LAG_STEPS
        seen = visible & torch.isfinite(z)
        zero = torch.zeros_like(z)
        assert torch.equal(batch.power[i], torch.where(seen, z, zero)), (
            f"sample {i}: power read poison"
        )
        assert torch.equal(batch.power_missing[i], (~seen).float()), f"sample {i}: flag read poison"
        assert torch.equal(batch.nb_mean[i], torch.where(visible, nm, zero)), (
            f"sample {i}: nb mean read poison"
        )
        assert torch.equal(batch.nb_share[i], torch.where(visible, ns, zero)), (
            f"sample {i}: nb share read poison"
        )
    return f"no-lookahead (poisoned copy): {n} windows identical"

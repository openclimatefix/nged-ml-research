# ruff: noqa: E501, D102, D103, D107, PT018
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
import e00510_common as c510  # ty: ignore[unresolved-import]
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


@dataclass
class Fold:
    """Statistics of one fold's training region, applied unchanged to every window."""

    mean: np.ndarray  # (28,)
    std: np.ndarray  # (28,)
    standardise: np.ndarray  # (28,) bool: columns that are standardised (others pass through)
    target_mean: np.ndarray  # (primaries,) power over capacity
    target_std: np.ndarray  # (primaries,)
    train_end: int  # grid index of the first half-hour not used for training targets or inputs
    first_valid: (
        np.ndarray
    )  # (primaries,) first grid index at or after which training windows start


def fit_fold(*, raw: np.ndarray, power: np.ndarray, train_first: int, train_end: int) -> Fold:
    """Fit the standardisation on grid indices [train_first, train_end) only."""
    _, mask, _ = prep.feature_names(scheme="summary")
    standardise = np.concatenate([mask[WEATHER_IN_46], mask[DATETIME_IN_46]]).astype(bool)
    region = np.asarray(raw[:, train_first:train_end]).reshape(-1, N_COLUMNS)
    mean = np.where(standardise, np.nanmean(region, axis=0), 0.0)
    std = np.maximum(np.where(standardise, np.nanstd(region, axis=0), 1.0), 1e-6)
    p = power[:, train_first:train_end]
    target_mean = np.nanmean(p, axis=1)
    target_std = np.maximum(np.nanstd(p, axis=1), 1e-3)
    first = np.array([int(np.argmax(np.isfinite(power[i]))) for i in range(power.shape[0])])
    return Fold(mean, std, standardise, target_mean, target_std, train_end, first)


def standardised_inputs(*, raw: np.ndarray, fold: Fold, device: str) -> torch.Tensor:
    """Return the (primaries, grid, 28) standardised inputs as float16 on `device`; NaN becomes 0."""
    out = np.empty(raw.shape, dtype=np.float16)
    for p in range(raw.shape[0]):
        z = np.clip((np.asarray(raw[p]) - fold.mean) / fold.std, -CLIP_SD, CLIP_SD)
        out[p] = np.nan_to_num(z, nan=0.0).astype(np.float16)
    return torch.from_numpy(out).to(device)


def standardised_power(*, power: np.ndarray, fold: Fold) -> np.ndarray:
    """Return z = (power - mean) / std per Primary, NaN where the reading is flagged or absent."""
    return ((power - fold.target_mean[:, None]) / fold.target_std[:, None]).astype(np.float32)


def neighbour_summary(*, z: np.ndarray, of: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean z of the neighbours, share of neighbours with a reading), both (primaries, grid).

    The mean is 0 and the share is 0 where no neighbour has a reading.
    """
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
    x_hist: torch.Tensor  # (B, H, 28) weather and datetime
    x_fut: torch.Tensor  # (B, F, 28)
    power: torch.Tensor  # (B, H) standardised own power, 0 where masked
    power_missing: (
        torch.Tensor
    )  # (B, H) 1 where masked (newer than origin - 24 h, flagged, or absent)
    nb_mean: torch.Tensor  # (B, H)
    nb_share: torch.Tensor  # (B, H)
    rel_hist: torch.Tensor  # (B, H) hours from the origin / 168, negative
    rel_fut: torch.Tensor  # (B, F) hours from the origin / 168, positive
    target: torch.Tensor  # (B, F) standardised power; NaN where flagged or after the cut
    hist_idx: torch.Tensor  # (B, H) grid index (kept for the assertions)
    fut_idx: torch.Tensor  # (B, F)


class Windows:
    """Holds the dense arrays on the device and cuts windows out of them by index."""

    def __init__(
        self, *, x: torch.Tensor, z: np.ndarray, nb_mean: np.ndarray, nb_share: np.ndarray,
        n_hist: int, n_fut: int, device: str,
    ) -> None:  # fmt: skip
        self.x = x
        self.z = torch.from_numpy(z).to(device)
        self.nb_mean = torch.from_numpy(nb_mean).to(device)
        self.nb_share = torch.from_numpy(nb_share).to(device)
        self.n_hist, self.n_fut, self.device = n_hist, n_fut, device
        self.hist_off = torch.arange(-n_hist + 1, 1, device=device)
        self.fut_off = torch.arange(1, n_fut + 1, device=device)
        self.grid = x.shape[1]

    def cut(
        self, *, primary: torch.Tensor, origin: torch.Tensor, target_end: int | None = None
    ) -> Batch:
        """Return the windows of (primary, origin) pairs.

        Args:
            primary: (B,) Primary indices.
            origin: (B,) grid indices of the forecast origins.
            target_end: if given, targets at grid indices at or after it are set to NaN (the fit
                sees no reading from the refit cutoff on).
        """
        h = origin[:, None] + self.hist_off[None, :]
        f = origin[:, None] + self.fut_off[None, :]
        assert int(h.min()) >= 0 and int(f.max()) < self.grid, "window leaves the grid"
        p = primary[:, None]
        visible = h <= (origin[:, None] - LAG_STEPS)
        z = self.z[p, h]
        seen = visible & torch.isfinite(z)
        target = self.z[p, f].clone()
        if target_end is not None:
            target[f >= target_end] = float("nan")
        return Batch(
            primary=primary,
            origin=origin,
            x_hist=self.x[p, h],
            x_fut=self.x[p, f],
            power=torch.where(seen, z, torch.zeros_like(z)),
            power_missing=(~seen).float(),
            nb_mean=torch.where(visible, self.nb_mean[p, h], torch.zeros_like(z)),
            nb_share=torch.where(visible, self.nb_share[p, h], torch.zeros_like(z)),
            rel_hist=(h - origin[:, None]).float() / (2 * 168),
            rel_fut=(f - origin[:, None]).float() / (2 * 168),
            target=target,
            hist_idx=h,
            fut_idx=f,
        )


def assert_no_lookahead(*, batch: Batch, windows: Windows) -> str:
    """Assert, row by row, that no power value the network reads is newer than origin - 24 h.

    Three checks. (1) Every history token with a visible own-power or neighbour value has a grid index
    at most `origin - 48`. (2) A poisoned copy of the power array, with every reading newer than the
    origin minus 24 h of each sample replaced by 1e6, gives the same own-power, missing-flag, and
    neighbour tensors. (3) The future tokens carry weather and datetime columns only: the batch has no
    future power field, and `target` is not an input.
    """
    origin = batch.origin[:, None]
    newest = batch.hist_idx.masked_fill(batch.power_missing > 0.5, -(10**9)).amax(dim=1)
    assert (newest <= batch.origin - LAG_STEPS).all(), "own power newer than origin - 24 h"
    nb_seen = (batch.nb_share > 0) | (batch.nb_mean != 0)
    nb_newest = batch.hist_idx.masked_fill(~nb_seen, -(10**9)).amax(dim=1)
    assert (nb_newest <= batch.origin - LAG_STEPS).all(), "neighbour power newer than origin - 24 h"
    masked_tail = batch.hist_idx > (origin - LAG_STEPS)
    assert (batch.power[masked_tail] == 0).all() and (batch.power_missing[masked_tail] == 1).all()
    assert (batch.nb_mean[masked_tail] == 0).all() and (batch.nb_share[masked_tail] == 0).all()
    # poisoned copy
    poisoned = Windows(
        x=windows.x, z=windows.z.cpu().numpy().copy(), nb_mean=windows.nb_mean.cpu().numpy().copy(),
        nb_share=windows.nb_share.cpu().numpy().copy(), n_hist=windows.n_hist, n_fut=windows.n_fut,
        device=windows.device,
    )  # fmt: skip
    for arr in (poisoned.z, poisoned.nb_mean, poisoned.nb_share):
        for i in range(len(batch.origin)):
            arr[batch.primary[i], batch.origin[i] - LAG_STEPS + 1 :] = 1e6
    again = poisoned.cut(primary=batch.primary, origin=batch.origin)
    for name in ("power", "power_missing", "nb_mean", "nb_share"):
        assert torch.equal(getattr(batch, name), getattr(again, name)), (
            f"{name} read poisoned power"
        )
    return (
        f"no-lookahead: {len(batch.origin)} windows, newest own power "
        f"{int((batch.origin - newest).min()) / 2:.1f} h before the origin at the closest, "
        "poisoned copy gives identical inputs"
    )

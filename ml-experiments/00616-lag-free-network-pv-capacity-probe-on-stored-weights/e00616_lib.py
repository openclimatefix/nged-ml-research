# ruff: noqa: E501
"""Shared loaders for experiment 616: the stored pretrained lag-free networks and their input rows.

Every function here is read-only on the stored weights. The networks are the stage-`pre` networks
(ERA5 and CAMS pretraining, before any ENS fine-tuning) of experiment 565's design 5 (arm D5, with the
per-Primary slow output offset) and of experiment 554's arm M0 (no offset).

The input rows are experiment 448's pretraining rows: one row per (weather run, valid time) with the
37 lag-free columns, standardised with the fold's own standardiser. The weather columns (0 to 6) and
the datetime columns (14 to 34) of a row depend on the valid time only.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m554  # ty: ignore[unresolved-import]
import e00565_model_variants as var  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W565: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00565-drift-embedding-variants-six-designs" / "work"
W554: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
    / "work"
)
FOLDS: Final[tuple[str, ...]] = c.MONTHS
"""The five development folds (test months); the pretrained weights of fold F end before F starts."""
COMPONENTS: Final[int] = 3
GRID_US: Final[int] = c.prep422.GRID_START_US
STEP_US: Final[int] = c.prep422.STEP_US
DRIFT_ORIGIN: Final[np.datetime64] = np.datetime64("2019-09", "M")
WEATHER: Final[slice] = slice(0, 7)
DATETIME: Final[slice] = slice(14, 35)
SHORTWAVE, LONGWAVE, TEMPERATURE = 0, 1, 2
COS_ZENITH = 14
ARMS: Final[tuple[str, ...]] = ("D5", "M0")


def n_knots(*, month: str) -> int:
    """Return the number of monthly knots of the drift tables of a fold (2019-09 to the fold's month)."""
    year, mon = (int(p) for p in month.split("-"))
    return (year - 2019) * 12 + (mon - 9) + 1


def tau(*, valid_us: np.ndarray) -> np.ndarray:
    """Return the knot position (months since 2019-09, fractional) of valid times in microseconds."""
    t = valid_us.astype("datetime64[us]")
    start = t.astype("datetime64[M]")
    lo, hi = start.astype("datetime64[us]"), (start + 1).astype("datetime64[us]")
    frac = (t - lo) / (hi - lo)
    return ((start - DRIFT_ORIGIN).astype(np.float64) + frac).astype(np.float32)


def weights_path(*, arm: str, month: str, seed: int) -> Path:
    """Return the stored stage-`pre` weights of an arm, fold, and seed."""
    if arm == "D5":
        return W565 / "pretrained_D5" / "real" / f"{month}_s{seed}.pt"
    if arm == "M0":
        return W554 / "pretrained" / "M0" / f"{month}_s{seed}.pt"
    raise ValueError(arm)


def load_net(*, arm: str, month: str, seed: int) -> torch.nn.Module:
    """Load a stored stage-`pre` network in evaluation mode, strictly (every key must match)."""
    n = 16
    if arm == "D5":
        net = var.OutputOffsetDrift(n_primaries=n, stage="pre", n_knots=n_knots(month=month))
    else:
        net = m554.ModularMdn(n_primaries=n, stage="pre")
    state = torch.load(weights_path(arm=arm, month=month, seed=seed), map_location="cpu")
    missing = net.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys, missing
    assert all(k.startswith("mixer_ft.") for k in missing.missing_keys), missing
    net.eval()
    return net


def with_tau(*, arm: str, x: np.ndarray, valid_us: np.ndarray) -> np.ndarray:
    """Append the drift position column for the D5 arm; the M0 arm takes the 37 columns as they are."""
    if arm == "M0":
        return x
    return np.concatenate([x, tau(valid_us=valid_us)[:, None]], axis=1)


def mixture_mean(*, net: torch.nn.Module, x: np.ndarray, primary: np.ndarray) -> np.ndarray:
    """Return the mixture mean (standardised target units) for each row, float64."""
    out = []
    with torch.no_grad():
        for a in range(0, len(x), 65_536):
            xb = torch.from_numpy(np.ascontiguousarray(x[a : a + 65_536]))
            pb = torch.from_numpy(primary[a : a + 65_536].astype(np.int64))
            log_w, mean, _ = base.mixture_parts(out=net(xb, pb).double(), components=COMPONENTS)
            out.append((log_w.exp() * mean).sum(dim=-1).numpy())
    return np.concatenate(out)


@dataclass
class Fold:
    """A development fold's rows and statistics."""

    month: str
    rows: data.Rows
    pre: data.Rows
    capacity_mw: np.ndarray
    sites: list[str]


def load_fold(*, month: str) -> Fold:
    """Build a fold's standardiser, target statistics, and ERA5-and-CAMS pretraining rows."""
    clean = c.clean_stack()
    refit = c.month_start(month=month)
    rows = data.build_fold(refit=refit, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean)
    table = prep.primaries()
    return Fold(
        month=month,
        rows=rows,
        pre=pre,
        capacity_mw=table["effective_capacity_mw"].to_numpy().astype(np.float64),
        sites=table["site"].to_list(),
    )


def valid_us_of(*, target_idx: np.ndarray) -> np.ndarray:
    """Return valid times in microseconds from grid indices."""
    return GRID_US + target_idx.astype(np.int64) * STEP_US


def to_datetime(*, us: int) -> datetime:
    """Return a UTC datetime from microseconds since the epoch."""
    return datetime.fromtimestamp(us / 1e6, UTC)


def raw_rows(*, site: str) -> dict[str, np.ndarray]:
    """Return one row per valid time of a Primary's stored pretraining array (raw, unstandardised).

    The weather and datetime columns of the 37 lag-free columns are identical across the weather
    runs that share a valid time, so the first row stands for all of them.
    """
    keep, _, _ = data.non_lag_layout()
    z = np.load(prep418.RAW_DIR / f"{site}.npz")
    _, first = np.unique(z["valid"], return_index=True)
    return {"x": z["x"][first][:, keep], "valid": z["valid"][first]}


def standardise(*, fold: Fold, x_raw: np.ndarray) -> np.ndarray:
    """Standardise raw 37-column rows with the fold's standardiser (spread columns set to 0)."""
    x = fold.rows.standardiser.apply(x=x_raw)
    x[:, 7:14] = 0.0
    x[:, 35:37] = 0.0
    return x


def fold_statistics(*, month: str) -> Fold:
    """Return a fold's standardiser, target statistics, and capacities, without the pretraining rows."""
    clean = c.clean_stack()
    refit = c.month_start(month=month)
    rows = data.build_fold(refit=refit, clean=clean)
    table = prep.primaries()
    return Fold(
        month=month,
        rows=rows,
        pre=None,  # ty: ignore[invalid-argument-type]
        capacity_mw=table["effective_capacity_mw"].to_numpy().astype(np.float64),
        sites=table["site"].to_list(),
    )

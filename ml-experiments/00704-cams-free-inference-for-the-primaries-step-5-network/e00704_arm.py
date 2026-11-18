# ruff: noqa: ANN001, ANN201, D103, E501
"""Experiments 703 and 704: replace the two previous-day CAMS columns of experiment 437's tables.

Importing this module (`e00704_run.py` does, before `runpy`) installs arm `E00704_ARM`:

- `replay`: nothing is patched; the chain reproduces the stored experiment.
- `proxy` (arm A): `f437.pretrain_extras` returns, for `cams_prev_day_mean` and
  `cams_prev_day_csi`, the mean of the dense stand-in shortwave `dense["valid_shortwave"]` over the
  48 half-hours of the UTC day before the origin's UTC date (stamps 00:30 to 00:00, period ends)
  and that sum over the sum of `f437.clear_sky_on_grid` (NaN if any of the 48 is not finite or the
  clear-sky sum is 0). Fine-tune, table, and scoring rows read the per-Primary tables of
  `e00704_tables.py build proxy`.
- `drop` (arm B): both columns are 0.0 in pre-training rows and in the tables of
  `e00704_tables.py build drop`; `Extras.fit` then stores mean 0 and std 1e-6.

The fold modules (`e00667_fold` and `e00677_fold`) point `f437.TRAILING_DIR` at experiment 667's
tables inside `redirect_to_extended_inputs`, which `fold.install` calls, so this module wraps
that function in each fold module and re-points `f437.TRAILING_DIR` at the arm's tables after it.
"""

import importlib
import os
from typing import Final

import e00437_features as f437  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARM: Final = os.environ.get("E00704_ARM", "")
assert ARM in {"replay", "proxy", "drop"}, f"set E00704_ARM to replay, proxy, or drop, not {ARM!r}"
CAMS_COLUMNS: Final = ("cams_prev_day_mean", "cams_prev_day_csi")
DAY_US: Final = 86_400 * 1_000_000
TABLES: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00704-cams-free-inference-for-the-primaries-step-5-network"
    / "inputs"
    / f"trailing437_{ARM}"
)
FOLD_MODULES: Final = ("e00667_fold", "e00677_fold")
_printed: dict[str, bool] = {}
_clear_cache: dict[str, np.ndarray] = {}


def proxy_columns(*, row: dict, dense: dict[str, np.ndarray], origin_us: np.ndarray) -> np.ndarray:
    """Return (rows, 2): previous-UTC-day mean shortwave and its ratio to the Haurwitz clear-sky sum."""
    site = row["site"]
    if site not in _clear_cache:
        _clear_cache[site] = f437.clear_sky_on_grid(
            latitude=float(row["latitude"]), longitude=float(row["longitude"])
        )
    clear = _clear_cache[site]
    shortwave = dense["valid_shortwave"]
    finite = np.isfinite(shortwave)
    count = np.concatenate([[0], np.cumsum(finite)])
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, shortwave, 0.0))])
    clear_total = np.concatenate([[0.0], np.cumsum(clear)])
    day = origin_us // DAY_US - 1
    first = f437.grid_of(us=day * DAY_US) + 1  # stamp 00:30 of that day
    stop = first + f437.STEPS_PER_DAY
    inside = (first >= 0) & (stop <= len(shortwave))
    lo, hi = np.where(inside, first, 0), np.where(inside, stop, 0)
    complete = inside & ((count[hi] - count[lo]) == f437.STEPS_PER_DAY)
    s_sum, c_sum = total[hi] - total[lo], clear_total[hi] - clear_total[lo]
    mean = np.where(complete, s_sum / f437.STEPS_PER_DAY, np.nan)
    csi = np.where(complete & (c_sum > 0), s_sum / np.where(c_sum > 0, c_sum, 1.0), np.nan)
    return np.stack([mean, csi], axis=1)


_orig_pretrain_extras = f437.pretrain_extras


def pretrain_extras(*, row, dense, columns, valid_us, origin_us, max_steps):
    """Return experiment 437's pre-training columns with the two CAMS columns of the arm."""
    here = [k for k, c in enumerate(columns) if c in CAMS_COLUMNS]
    if not here:
        return _orig_pretrain_extras(
            row=row,
            dense=dense,
            columns=columns,
            valid_us=valid_us,
            origin_us=origin_us,
            max_steps=max_steps,
        )
    rest = tuple(c for c in columns if c not in CAMS_COLUMNS)
    out = np.empty((len(valid_us), len(columns)), dtype=np.float32)
    if rest:
        other = _orig_pretrain_extras(
            row=row,
            dense=dense,
            columns=rest,
            valid_us=valid_us,
            origin_us=origin_us,
            max_steps=max_steps,
        )
        out[:, [k for k, c in enumerate(columns) if c not in CAMS_COLUMNS]] = other
    if ARM == "drop":
        values = np.zeros((len(valid_us), 2), dtype=np.float32)
    else:
        values = proxy_columns(row=row, dense=dense, origin_us=origin_us)
    for k in here:
        out[:, k] = values[:, CAMS_COLUMNS.index(columns[k])]
    if not _printed.get("pretrain"):
        _printed["pretrain"] = True
        print(
            f"E00704 {ARM} pretrain example {row['site']}:",
            {columns[k]: out[:3, k].tolist() for k in here},
            flush=True,
        )
    return out


def point_at_arm_tables() -> None:
    """Set `f437.TRAILING_DIR` to the arm's tables and print the proof."""
    assert TABLES.is_dir(), f"build the tables first: {TABLES}"
    assert len(list(TABLES.glob("*.parquet"))) == 16, "the arm's tables are incomplete"
    f437.TRAILING_DIR = TABLES
    example = pl.read_parquet(f437.table_path(site="D2"), columns=list(CAMS_COLUMNS))
    finite = example.filter(pl.all_horizontal(pl.all().is_finite()))
    print(
        f"E00704 {ARM}: f437.TRAILING_DIR = {f437.TRAILING_DIR}; D2 first finite raw columns:",
        finite.row(0, named=True),
        "; finite rows:",
        finite.height,
        "of",
        example.height,
        flush=True,
    )


def wrap_fold(*, module_name: str) -> None:
    try:
        fold = importlib.import_module(module_name)
    except ImportError:
        return
    original = fold.redirect_to_extended_inputs  # ty: ignore[unresolved-attribute]

    def redirect_to_extended_inputs(e) -> None:
        original(e)
        point_at_arm_tables()

    fold.redirect_to_extended_inputs = redirect_to_extended_inputs  # ty: ignore[unresolved-attribute]


if ARM != "replay":
    f437.pretrain_extras = pretrain_extras
    for _name in FOLD_MODULES:
        wrap_fold(module_name=_name)
else:
    print("E00704 replay: nothing patched", flush=True)

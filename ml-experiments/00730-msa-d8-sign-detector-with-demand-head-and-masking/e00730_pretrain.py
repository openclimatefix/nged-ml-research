# ruff: noqa: E501
"""Experiment 730: pre-training of experiment 545's encoders without the held-out months.

Every stored set of encoder weights from the earlier experiments saw D8's readings in the months that
this experiment holds out, so none is used. This module repeats experiment 545's M0 pre-training with
the held-out half-hours removed, and writes the weights, the standardiser, and D8's encoder input
table to this experiment's data folder.

The removal works through `clean`: the module sets `clean` to NaN on every removed half-hour of every
Primary before it calls experiment 428's `build_fold`, so that the standardiser, the target means and
spreads, and the lead-band medians of experiment 448's `spread_table` are all rebuilt on the
surviving half-hours. It then drops every `pretrain_rows` row whose valid time or origin falls in a
removed period. The helpers that decide what is removed are pure numpy and run in the unit tests. The
functions that read the stored files of the 16 Primaries import experiments 397 to 545 inside the
function, and need the real files.
"""

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00730_paths  # noqa: F401

# isort: split
import e00730_core as core
import numpy as np

WEATHER_SLICE: Final = slice(0, core.ENCODER_WEATHER_COLUMNS)
"""Columns 0 to 6 of experiment 545's 37 inputs: the 7 weather columns."""
DATETIME_SLICE: Final = slice(14, 14 + core.ENCODER_DATETIME_COLUMNS)
"""Columns 14 to 34: the 21 solar-geometry, calendar, and holiday columns."""
STANDARDISER_FILE: Final = "standardiser.npz"
D8_TABLE_FILE: Final = "d8_encoder_inputs.parquet"
N_PRIMARIES: Final = 16
"""Experiment 397's table of Primaries holds this many series."""
WEIGHT_FILE: Final = "2026-07_s{seed}.pt"
STEPS_PER_DAY: Final = 48


# --------------------------------------------------------------------------- what is removed


def month_days(*, months: Sequence[str]) -> np.ndarray:
    """Return every UTC day of the given `YYYY-MM` months as `datetime64[D]`."""
    parts = []
    for month in months:
        first = np.datetime64(f"{month}-01", "D")
        nxt = (first.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
        parts.append(np.arange(first, nxt))
    return np.concatenate(parts) if parts else np.array([], dtype="datetime64[D]")


def removed_days(
    *,
    held_out_months: Sequence[str],
    record_days: np.ndarray,
    margin: int = core.PRETRAIN_MARGIN_DAYS,
) -> np.ndarray:
    """Return every day to remove: the held-out months and the metered engine record's days, padded.

    Args:
        held_out_months: The 8 test and 4 validation months of this experiment.
        record_days: The days of the metered engine record.
        margin: The number of days either side of each that are removed too (2).

    Returns:
        The sorted unique days, `datetime64[D]`.
    """
    base = np.unique(np.concatenate([month_days(months=held_out_months), record_days]))
    shifts = np.arange(-margin, margin + 1).astype("timedelta64[D]")
    return np.unique((base[:, None] + shifts[None, :]).ravel())


def removed_half_hour_mask(*, grid_days: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """Return, for every half-hour of the grid (given by its UTC day), whether it is removed."""
    return np.isin(grid_days, removed)


def clean_with_removal(*, clean: np.ndarray, removed_mask: np.ndarray) -> np.ndarray:
    """Return a copy of `clean` (Primaries by grid) with every removed half-hour set to NaN.

    Args:
        clean: Cleaned power over capacity of every Primary, `(primaries, grid)`.
        removed_mask: The removed half-hours, `(grid,)`.

    Returns:
        The copy; every Primary is NaN on every removed half-hour.
    """
    out = clean.copy()
    out[:, removed_mask] = np.nan
    return out


def keep_rows(
    *, target_idx: np.ndarray, origin_idx: np.ndarray, removed_mask: np.ndarray
) -> np.ndarray:
    """Return the rows whose valid time and origin are both outside every removed period.

    The rows are (origin, lead) pairs, and the ERA5 and CAMS values of one valid time repeat across
    leads, so the test reads the valid time and the origin of each row.
    """
    size = len(removed_mask)
    bad = np.zeros(len(target_idx), dtype=bool)
    for idx in (target_idx, origin_idx):
        inside = (idx >= 0) & (idx < size)
        bad |= np.where(inside, removed_mask[np.clip(idx, 0, size - 1)], False)
    return ~bad


# --------------------------------------------------------------------------- the standardiser


@dataclass(frozen=True)
class StoredStandardiser:
    """The pre-training standardiser as stored beside the weights.

    Attributes:
        mean: The column means over the surviving rows.
        std: The column standard deviations over the surviving rows (at least 1e-6).
        mask: The columns that are standardised.
    """

    mean: np.ndarray
    std: np.ndarray
    mask: np.ndarray


def fit_standardiser(*, x: np.ndarray, mask: np.ndarray) -> StoredStandardiser:
    """Fit the standardiser of experiment 397 (`Standardiser.fit`) on raw surviving rows."""
    mean = np.where(mask, np.nanmean(x, axis=0), 0.0)
    std = np.where(mask, np.nanstd(x, axis=0), 1.0)
    return StoredStandardiser(mean=mean, std=np.maximum(std, 1e-6), mask=mask)


def standardiser_matches(
    *, stored: StoredStandardiser, surviving_x: np.ndarray, atol: float = 1e-5
) -> bool:
    """Return whether refitting on the surviving raw rows gives the stored standardiser.

    This is the leakage check of the plan: a standardiser fitted on any row that was removed differs
    from the refit on the surviving rows.
    """
    refit = fit_standardiser(x=surviving_x, mask=stored.mask)
    return bool(
        np.allclose(refit.mean, stored.mean, atol=atol, rtol=1e-5)
        and np.allclose(refit.std, stored.std, atol=atol, rtol=1e-5)
    )


def encoder_columns(*, standardiser: StoredStandardiser) -> StoredStandardiser:
    """Return the standardiser restricted to the 7 weather and 21 datetime columns, in that order."""
    index = np.r_[np.arange(*WEATHER_SLICE.indices(37)), np.arange(*DATETIME_SLICE.indices(37))]
    return StoredStandardiser(
        mean=standardiser.mean[index], std=standardiser.std[index], mask=standardiser.mask[index]
    )


def zscore_encoder_inputs(*, raw: np.ndarray, standardiser: StoredStandardiser) -> np.ndarray:
    """Return the 28 encoder columns z-scored as experiment 397's `Standardiser.apply` does.

    The values are standardised, clipped at five standard deviations, and NaN becomes 0.

    Args:
        raw: The raw 7 weather and 21 datetime columns, `(..., 28)`.
        standardiser: The 28-column standardiser from `encoder_columns`.

    Returns:
        The standardised array, float32.
    """
    out = (raw - standardiser.mean) / standardiser.std
    return np.nan_to_num(np.clip(out, -5.0, 5.0), nan=0.0).astype(np.float32)


def clear_sky_peak_hour(*, table: np.ndarray, cosz_column: int) -> int:
    """Return the UTC hour at which the mean of a clear-sky-driven column peaks, `(days, 48, cols)`.

    The smoke test prints whether the hour lies in `ENCODER_PEAK_HOURS_UTC` (11 to 13): a table that
    is misaligned in time peaks at another hour.
    """
    profile = table[:, :, cosz_column].mean(axis=0)
    return int(np.argmax(profile)) // 2


def share_non_constant(*, table: np.ndarray) -> float:
    """Return the share of the table's columns that are not constant over its half-hours."""
    flat = table.reshape(-1, table.shape[-1])
    return float(np.mean(np.ptp(flat, axis=0) > 0.0))


# --------------------------------------------------------------------------- the real run


def write_standardiser(*, standardiser: StoredStandardiser, folder: Path) -> None:
    """Write the standardiser beside the weights, atomically."""
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{STANDARDISER_FILE}.{os.getpid()}.tmp.npz"
    np.savez(tmp, mean=standardiser.mean, std=standardiser.std, mask=standardiser.mask)
    tmp.replace(folder / STANDARDISER_FILE)


def read_standardiser(*, folder: Path) -> StoredStandardiser:
    """Read the stored standardiser."""
    z = np.load(folder / STANDARDISER_FILE)
    return StoredStandardiser(mean=z["mean"], std=z["std"], mask=z["mask"])


def weight_path(*, folder: Path, seed: int) -> Path:
    """Return the weight file of a pre-training seed, `pretrained/M0x/2026-07_s<seed>.pt`."""
    return folder / "M0x" / WEIGHT_FILE.format(seed=seed)


def weight_digest(*, path: Path) -> str:
    """Return the SHA-256 of a weight file (recorded in each fit record)."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_removed_mask(*, held_out_months: Sequence[str], record_days: np.ndarray) -> np.ndarray:
    """Return the removed half-hours of experiment 087's dense grid (real run; needs the grid)."""
    import e00087_common as common

    grid = common.grid_times().dt.date().to_numpy().astype("datetime64[D]")
    return removed_half_hour_mask(
        grid_days=grid,
        removed=removed_days(held_out_months=held_out_months, record_days=record_days),
    )


def surviving_raw_x(*, refit, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """Return the raw training rows of the fold on the surviving half-hours, and the column mask.

    This repeats the row collection of experiment 428's `build_fold` up to the standardiser fit, so
    the leakage test can refit the standardiser from the surviving rows and compare it with the one
    that `build_fold` stored.
    """
    import e00087_regimes as regimes
    import e00397_prep as prep
    import e00422_prep as prep422
    import e00428_data as data428

    keep_cols, _, mask = data428.non_lag_layout()
    xs, origins = [], []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= data428._us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= data428._us(t=refit))
            & (
                z["valid"]
                > data428._us(t=refit - __import__("datetime").timedelta(days=prep.WINDOW_DAYS))
            )
        )
        if not keep.any():
            continue
        target_idx = ((z["valid"][keep] - prep422.GRID_START_US) // prep422.STEP_US).astype(
            np.int64
        )
        usable = np.isfinite(clean[row["primary"]][target_idx])
        xs.append(z["x"][keep][usable][:, keep_cols])
        origins.append(z["origin"][keep][usable])
    x = np.concatenate(xs)
    origin = np.concatenate(origins)
    origin_day = ((origin - data428._us(t=prep.FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
    held, purged = prep422.purge_masks(origin_day=origin_day)
    return x[~held & ~purged], mask


def pretraining_rows(*, removed_mask: np.ndarray):  # noqa: ANN201
    """Build experiment 428's fold for a refit at 2026-07-01 and the pre-training rows, minus removals.

    Args:
        removed_mask: The removed half-hours of the dense grid.

    Returns:
        The fold's rows (their standardiser and target spreads rebuilt on the surviving half-hours),
        the surviving pre-training rows, and the NaN-ed `clean` stack.
    """
    from datetime import UTC, datetime

    import e00428_data as data428
    import e00448_run as r448
    import e00510_common as c

    refit = datetime.fromisoformat(core.PRETRAIN_REFIT_ISO).replace(tzinfo=UTC)
    clean = clean_with_removal(clean=c.clean_stack(), removed_mask=removed_mask)
    rows = data428.build_fold(refit=refit, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean)
    ok = keep_rows(target_idx=pre.target_idx, origin_idx=pre.origin_idx, removed_mask=removed_mask)
    for name in ("x", "y", "primary", "origin_idx", "target_idx", "lead", "weight", "is_val"):
        setattr(pre, name, getattr(pre, name)[ok])
    return rows, pre, clean


def train_pretraining(
    *,
    x: Any,
    z: Any,
    w: Any,
    p: Any,
    is_val: Any,
    seed: int,
    device: str,
) -> tuple[Any, int, float]:
    """Experiment 545's training loop, with experiment 510's constants, copied from `e00545_fit`.

    `e00545_fit.py` parses its command line when it is imported, so the loop is copied here. The
    learning rate, weight decay, batch size, row fraction, gradient clipping, and the early stop on
    held-out weeks are experiment 510's (`e00510_common`). The held-out weeks are the rows of
    `is_val`, drawn from the rows that remain.

    Args:
        x: The inputs of the surviving rows.
        z: The standardised targets.
        w: The row weights.
        p: The Primary index of each row.
        is_val: Rows held out for the early stop.
        seed: The seed.
        device: "cuda" or "cpu".

    Returns:
        The network with its best weights, the epochs run, and the best held-out negative log likelihood.
    """
    import e00397_models as base
    import e00510_common as c
    import e00545_model as m
    import torch

    torch.manual_seed(seed)
    n_primaries = int(p.max().item()) + 1
    if n_primaries != N_PRIMARIES:
        msg = f"the pre-training rows cover {n_primaries} Primaries, not {N_PRIMARIES}"
        raise ValueError(msg)
    model = m.ModularMdn(n_primaries=n_primaries, aux=False, stage="pre").to(device)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    optimiser = torch.optim.AdamW(model.parameters(), lr=c.PRETRAIN_LR, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            nll = base.mixture_nll(out=model(x[b], p[b]), z=z[b], components=m.COMPONENTS)
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        base.mixture_nll(out=model(x[i], p[i]), z=z[i], components=m.COMPONENTS)
                        for i in val_idx.split(c.BATCH * 2)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, epochs, best


def run_pretraining(
    *, root: Path, held_out_months: Sequence[str], record_days: np.ndarray, seeds: Sequence[int],
    device: str, max_epochs: int | None = None,
) -> dict:  # fmt: skip
    """Pre-train one weight file per seed on the surviving rows, and write the standardiser.

    The stage asserts the removal on the real rows (no row of any Primary has a valid time or origin
    in a removed period, no valid time on or after 2026-07-01), refits the standardiser from the
    surviving raw rows and asserts that it equals the stored one, and then fits. Weights go to
    `root/pretrained/M0x/2026-07_s<seed>.pt` atomically.

    Args:
        root: This experiment's data folder.
        held_out_months: The 8 test and 4 validation months.
        record_days: The days of the metered engine record.
        seeds: The pre-training seeds (0 to 2).
        device: "cuda" or "cpu".
        max_epochs: Cap on epochs for the timing gate (None for the experiment's cap).

    Returns:
        Counts only: the rows before and after removal, the epochs, and the held-out losses.
    """
    import resource

    import e00510_common as c
    import torch

    folder = root / "pretrained"
    removed = build_removed_mask(held_out_months=held_out_months, record_days=record_days)
    rows, pre, clean = pretraining_rows(removed_mask=removed)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    stored = StoredStandardiser(
        mean=rows.standardiser.mean, std=rows.standardiser.std, mask=rows.standardiser.mask
    )
    raw, _mask = surviving_raw_x(
        refit=__import__("datetime").datetime.fromisoformat(
            core.PRETRAIN_REFIT_ISO + "T00:00:00+00:00"
        ),
        clean=clean,
    )
    check_pretraining_leakage(pre=pre, removed=removed, stored=stored, raw=raw)
    if not pre.is_val.any():
        msg = "no early-stopping week remains"
        raise ValueError(msg)
    write_standardiser(standardiser=stored, folder=folder)
    if max_epochs is not None:
        c.MAX_EPOCHS = max_epochs
    z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
    tensors = {
        "x": torch.from_numpy(pre.x).to(device), "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "w": torch.ones(len(pre.y), device=device), "p": torch.from_numpy(pre.primary).to(device),
        "v": torch.from_numpy(pre.is_val).to(device),
    }  # fmt: skip
    record: dict = {
        "rows": len(pre.y), "peak_memory_gb": peak_gb, "seeds": {},
        "d8_index": d8_primary_index(), "n_primaries": N_PRIMARIES,
    }  # fmt: skip
    for seed in seeds:
        target = weight_path(folder=folder, seed=seed)
        if target.exists():
            continue
        started = time.perf_counter()
        model, epochs, best = train_pretraining(
            x=tensors["x"], z=tensors["z"], w=tensors["w"], p=tensors["p"], is_val=tensors["v"],
            seed=seed, device=device,
        )  # fmt: skip
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        torch.save(model.state_dict(), tmp)
        tmp.replace(target)
        record["seeds"][str(seed)] = {
            "epochs": epochs, "held_out_nll": best,
            "seconds_per_epoch": (time.perf_counter() - started) / max(epochs, 1),
        }  # fmt: skip
    return record


HOLIDAY_RAW_COLUMN: Final = DATETIME_SLICE.start + core.ENCODER_DATETIME_COLUMNS - 1
"""Column of the raw 37 non-lag features that holds the signed distance to the nearest holiday."""


def check_pretraining_leakage(
    *, pre: Any, removed: np.ndarray, stored: StoredStandardiser, raw: np.ndarray
) -> None:
    """Run the plan's pre-training leakage assertions (each raises, and none is a bare `assert`).

    Args:
        pre: The pre-training rows (`target_idx`, `origin_idx`).
        removed: The removed half-hours of the grid.
        stored: The standardiser that the rows were built with.
        raw: The surviving raw feature rows `(n, 37)`.
    """
    from datetime import UTC, datetime

    import e00087_common as common
    import e00730_leakage as leak

    holdout = common.grid_index(
        times=datetime.fromisoformat(core.core729.HOLDOUT_START_ISO).replace(tzinfo=UTC)
    )
    leak.assert_pretrain_removal(
        target_idx=pre.target_idx, origin_idx=pre.origin_idx, removed_mask=removed,
        valid_end_idx=int(holdout),
    )  # fmt: skip
    leak.assert_standardiser_surviving(stored=stored, surviving_x=raw)
    leak.assert_holiday_clip(nearest_days=raw[:, HOLIDAY_RAW_COLUMN])


def d8_raw_encoder_table(*, base_dates: np.ndarray) -> np.ndarray:
    """Return D8's encoder inputs on `base_dates` before z-scoring, `(days, 48, 28)`.

    The weather columns come from experiment 418's `dense_weather` (CAMS shortwave at D8's point and
    ERA5 longwave, 2 m temperature, 2 m dew point, surface pressure, precipitation, and 10 m wind
    speed) and the datetime columns from experiment 87's `derived_columns` through experiment 397's
    `feature_matrix`, exactly as the pre-training rows were built. A half-hour that starts at `p`
    half-hours after midnight ends at stamp `p + 1`, so it reads the grid index of that stamp.

    Args:
        base_dates: The days to build, `datetime64[D]`.

    Returns:
        The unstandardised table.
    """
    import e00087_common as common
    import e00397_prep as prep
    import e00418_prep as prep418
    import e00428_data as data428
    import polars as pl

    series = d8_series()
    weather = prep418.dense_weather(series_id=series["time_series_id"], site="D8")
    grid = common.grid_times()
    derived = common.derived_columns(
        valid_time=grid, latitude=series["latitude"], longitude=series["longitude"]
    )
    columns: dict[str, np.ndarray] = {}
    for j, name in enumerate(prep418.WEATHER):
        columns[f"mean_{name}"] = weather[:, j]
        columns[f"std_{name}"] = np.zeros(len(grid), dtype=np.float32)
    for name in common.DERIVED:
        columns[name] = derived[name].to_numpy()
    for name in common.LAG_COLUMNS:
        columns[name] = np.full(len(grid), np.nan, dtype=np.float32)
    columns["lead_hours"] = np.zeros(len(grid), dtype=np.float32)
    raw = prep.feature_matrix(frame=pl.DataFrame(columns), scheme="summary")
    keep_cols, _, _ = data428.non_lag_layout()
    raw = raw[:, keep_cols]
    index = np.r_[np.arange(*WEATHER_SLICE.indices(37)), np.arange(*DATETIME_SLICE.indices(37))]
    day_start = np.array(
        [common.grid_index(times=_midnight(day=d)) for d in base_dates], dtype=np.int64
    )
    stamps = day_start[:, None] + 1 + np.arange(STEPS_PER_DAY)[None, :]
    return raw[stamps][:, :, index]


def d8_series() -> dict:
    """Return D8's row of experiment 397's table of Primaries, and check that the table holds 16."""
    import e00397_prep as prep
    import polars as pl

    table = prep.primaries()
    if table.height != N_PRIMARIES:
        msg = f"expected {N_PRIMARIES} Primaries, found {table.height}"
        raise ValueError(msg)
    return table.filter(pl.col("site") == "D8").row(0, named=True)


def d8_primary_index() -> int:
    """Return the index of D8 in experiment 397's table of Primaries (its embedding row)."""
    import e00397_prep as prep

    table = prep.primaries()
    return int(table["site"].to_list().index("D8"))


def build_d8_encoder_table(
    *, root: Path, base_dates: np.ndarray, standardiser: StoredStandardiser
) -> np.ndarray:
    """Build D8's encoder input table and write it as `d8_encoder_inputs.parquet`.

    Args:
        root: This experiment's data folder.
        base_dates: The days to build (every calendar day of the real table), `datetime64[D]`.
        standardiser: The stored standardiser of the 37 non-lag columns.

    Returns:
        The table `(days, 48, 28)`, z-scored with the stored standardiser.
    """
    table = zscore_encoder_inputs(
        raw=d8_raw_encoder_table(base_dates=base_dates),
        standardiser=encoder_columns(standardiser=standardiser),
    )
    write_table(table=table, base_dates=base_dates, path=root / "pretrained" / D8_TABLE_FILE)
    return table


def _midnight(*, day: np.datetime64):  # noqa: ANN202
    """Return midnight UTC of a day as a datetime."""
    from datetime import UTC, datetime

    return datetime.fromisoformat(f"{day}T00:00:00").replace(tzinfo=UTC)


def write_table(*, table: np.ndarray, base_dates: np.ndarray, path: Path) -> None:
    """Write D8's encoder input table as a parquet file of one row per base day, atomically."""
    import polars as pl

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {"day": base_dates.astype("datetime64[D]"), "values": table.reshape(len(base_dates), -1)}
    )
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def read_table(*, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read D8's encoder input table: the base days and the values `(days, 48, 28)`."""
    import polars as pl

    frame = pl.read_parquet(path)
    values = np.asarray(frame["values"].to_list(), dtype=np.float32)
    return frame["day"].to_numpy().astype("datetime64[D]"), values.reshape(
        len(frame), STEPS_PER_DAY, core.ENCODER_INPUT_COLUMNS
    )


def write_record(*, record: dict, path: Path) -> None:
    """Write a counts-only JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=float))

# ruff: noqa: E501, PT018, PLR0915
"""Experiment 713: the adapter from the Candidate chain of experiment 667 to the MSA network.

`load_pipeline` imports experiment 574's module the way `e00667_cand.py` does (with `sys.argv` set to
arm `P5` first, because the module parses its arguments at import), installs experiment 667's fold
patch, and only then builds any dense array: `e00667_fold.install` calls `e00667_extend.redirect`,
which moves `GRID_END` to 2026-11-20 and the ERA5 path to experiment 607's file. A dense array
built before the redirect would be short.

`build_fold_data` returns a `FoldData`: the fold's standardised dense arrays on the device (a
`Source`), the fold statistics, and the ensemble target-row table built from the Candidate's
`MonthRows` (training) and `TestRows` (prediction). The Candidate's rows are the only source of the
ensemble inputs, so the MSA network reads exactly the Candidate's standardised weather, spread, B1,
and date-time columns for the target row.
"""

import hashlib
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import e00410_stale_predict as stale  # ty: ignore[unresolved-import]
import e00713_batch as bt
import e00713_layout as lay
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00713-msa-axial-attention-demand-network-part-1"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
CACHE: Final[Path] = ROOT / "cache"
"""The dense-array cache of experiment 632's loaders; nothing is written into experiment 632's folder."""
NAME_667: Final = "00667-one-year-backtest-of-the-two-output-demand-network-at-primaries"
ROOT_667: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME_667
CANDIDATE_SEEDS_ROOT: Final[Path] = ROOT / "candidate_seeds"
"""Where `chain_seeds.sh` writes the refitted Candidate chain for seeds 2 to 4."""
ENS_WEATHER_COLUMNS: Final = list(range(7)) + list(range(46, 60))
"""Columns of a Candidate row: 7 ensemble-mean weather columns, then the 14 B1 columns."""
ENS_SPREAD_COLUMNS: Final = list(range(7, 14))
HOUR_US: Final = 3_600_000_000
ORIGIN_HOUR_US: Final = 9 * HOUR_US
ENSEMBLE_LEAD_HOURS: Final = 9.0
"""The ensemble run behind an origin was issued this long before the native origin."""
N_ENS_STEPS: Final = 14 * lay.STEPS_PER_DAY


@dataclass
class Pipeline:
    """The imported modules of the Candidate pipeline, after the fold patch."""

    e: ModuleType  # experiment 574's `e00574_fit`
    fold: ModuleType  # experiment 667's `e00667_fold`
    d632: ModuleType  # experiment 632's `e00632_data`
    month: str
    chain: Path  # `<root>/chains/<month>/work` of the Candidate chain that holds `pretrained_P2`


def run_frame_timing(*, origin: datetime) -> tuple[datetime, datetime, Path]:
    """Return the latest `nwp_init_time`, the `power_fcst_init_time`, and the path of an origin's stored run frame.

    The frame is the one the Candidate reads (`demand_frame_path` for the run issued one publication
    delay before the origin). A stub frame that `e00667_fold` writes for the months after 2026-06 holds
    the run date itself, so a path under `inputs/frames` is refused.
    """
    import e00087_common as common  # ty: ignore[unresolved-import]

    path = common.demand_frame_path(run=origin - common.PUBLICATION_DELAY)
    assert "inputs/frames" not in path.as_posix(), f"{path} is a stub frame, not a stored run"
    frame = pl.read_parquet(path, columns=["nwp_init_time", "power_fcst_init_time"])
    forecast_origin = frame["power_fcst_init_time"].unique()
    assert len(forecast_origin) == 1, (path, forecast_origin)
    return frame["nwp_init_time"].max(), forecast_origin[0], path  # ty: ignore[invalid-return-type]


def chain_dir(*, root: Path, month: str) -> Path:
    """Return the `work` folder of a Candidate chain for `month` under `root`."""
    return root / "chains" / month / "work"


def load_pipeline(*, month: str, phase: str, chain_root: Path = ROOT_667) -> Pipeline:
    """Import the Candidate pipeline for one fold month and phase (`train` or `predict`)."""
    assert phase in {"train", "predict"}, phase
    os.environ["E00667_MONTH"] = month
    os.environ["E00667_PHASE"] = phase
    os.environ.setdefault("E00667_ROOT", str(ROOT_667))
    sys.argv = ["e00574_fit.py", "fit", "--arm", "P5", "--months", month, "--seeds", "0"]
    import e00574_fit as e  # ty: ignore[unresolved-import]
    import e00632_data as d632  # ty: ignore[unresolved-import]
    import e00667_fold as fold  # ty: ignore[unresolved-import]

    chain = chain_dir(root=chain_root, month=month)
    stats = chain_dir(root=ROOT_667, month=month) / "stats" / e.SPEC.b1 / f"{month}.npz"
    e.stats_file = lambda *, month: stats
    assert month == fold.FINAL_MONTH, (
        fold.FINAL_MONTH,
        month,
    )  # the module reads E00667_MONTH once, at import
    fold.install(e, months=[month], reference_month=fold.PREVIOUS_MONTH)
    d632.CACHE = CACHE  # ty: ignore[invalid-assignment]  # nothing may be written into experiment 632's folder
    CACHE.mkdir(parents=True, exist_ok=True)
    return Pipeline(e=e, fold=fold, d632=d632, month=month, chain=chain)


def pretrained_encoder_file(*, chain_root: Path, month: str, seed: int) -> Path:
    """Return the Candidate chain's pre-trained encoders for (`month`, `seed`)."""
    path = chain_dir(root=chain_root, month=month) / "pretrained_P2" / f"{month}_s{seed}.pt"
    assert path.exists(), path
    return path


def chain_root_of_seed(*, seed: int) -> Path:
    """Return the Candidate chain root that holds `seed`: experiment 667 for 0 and 1, else the refit."""
    return ROOT_667 if seed in {0, 1} else CANDIDATE_SEEDS_ROOT


def fingerprint(*, mr_base: Any) -> str:
    """Return a hash of the fold's standardiser and target statistics."""
    std = mr_base.standardiser
    h = hashlib.sha256()
    for a in (std.mean, std.std, std.mask, mr_base.target_mean, mr_base.target_std):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def ens_table(
    *, x: np.ndarray, primary: np.ndarray, origin_idx: np.ndarray, target_idx: np.ndarray,
    origin_us: np.ndarray, first_day: int, n_days: int,
) -> bt.EnsTensors:  # fmt: skip
    """Scatter Candidate rows into the (Primary, origin day, position) table of the target row.

    Every row's origin must be 09:00 UTC (the 00Z run plus 9 hours): the origin-time assertion of
    the design, `nwp_init_time <= o - 9 h`, is then a statement about the run date, which the
    Candidate's `build_month` and `build_test` have checked to exist.
    """
    assert bool(((origin_us % (24 * HOUR_US)) == ORIGIN_HOUR_US).all()), (
        "an origin is not 09:00 UTC"
    )
    c_idx = origin_idx + lay.NATIVE_OFFSET_STEPS
    day = c_idx // lay.STEPS_PER_DAY - first_day
    rel = target_idx - c_idx
    keep = (rel >= 0) & (rel < N_ENS_STEPS) & (day >= 0) & (day < n_days)
    weather = np.zeros((bt.N_PRIMARIES, n_days, N_ENS_STEPS, bt.TOP_WIDTH), np.float16)
    spread = np.zeros((bt.N_PRIMARIES, n_days, N_ENS_STEPS, 7), np.float16)
    filled = np.zeros((bt.N_PRIMARIES, n_days, N_ENS_STEPS), bool)
    p, d, r = primary[keep], day[keep], rel[keep]
    weather[p, d, r] = x[keep][:, ENS_WEATHER_COLUMNS].astype(np.float16)
    spread[p, d, r] = x[keep][:, ENS_SPREAD_COLUMNS].astype(np.float16)
    filled[p, d, r] = True
    run_us = np.zeros((bt.N_PRIMARIES, n_days), np.int64)
    run_us[p, d] = (
        origin_us[keep] - lay.ENSEMBLE_LEAD_US
    )  # the 00Z run, issued 9 hours before the origin
    return bt.EnsTensors(
        weather=torch.from_numpy(weather), spread=torch.from_numpy(spread),
        filled=torch.from_numpy(filled), first_day=first_day, run_us=torch.from_numpy(run_us),
    )  # fmt: skip


@dataclass
class FoldData:
    """Everything one fold's training or prediction job reads."""

    month: str
    source: bt.Source
    cutoff_idx: int
    target_mean: np.ndarray
    target_std: np.ndarray
    clean: (
        np.ndarray
    )  # (16, G) cleaned power over capacity, NaN after the cutoff in the train phase
    fingerprint: str
    train_ens_days: tuple[int, int]
    test_origin_idx: np.ndarray | None  # native origin indices of the scored origins
    test_keys: dict[str, np.ndarray] | None
    dropped_per_fold: dict[str, int]
    rows: tuple[Any, Any] | None = None  # the Candidate's (MonthRows, test rows), kept for checks


def build_fold_data(*, pipe: Pipeline, device: str, keep_rows: bool = False) -> FoldData:
    """Build the fold's dense arrays, statistics, and ensemble table (the WP3 adapter).

    The training rows come from `e00574_fit.build_month`, and the scored rows from
    `e00574_fit.build_test` (prediction phase only). Dense ERA5 and CAMS columns are checked finite at
    every history token and every stage 1 target token of the scored origins (and of the previous
    month's reference origins in the training phase).
    """
    e, fold, d = pipe.e, pipe.fold, pipe.d632
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    clean = e.c.clean_stack()
    assert clean.shape[0] == bt.N_PRIMARIES
    cut_dt = datetime(fold.FINAL_START.year, fold.FINAL_START.month, 1, tzinfo=UTC)
    cutoff_idx = int((int(cut_dt.timestamp() * 1e6) - prep422.GRID_START_US) // prep422.STEP_US)
    mr = e.build_month(month=pipe.month, clean=clean)
    base = mr.base
    mean37, std37, mask37 = base.standardiser.mean, base.standardiser.std, base.standardiser.mask
    keep37 = list(range(7)) + list(range(14, 35))
    stat = d.Fold(
        mean=mean37[keep37], std=std37[keep37], standardise=mask37[keep37].astype(bool),
        target_mean=base.target_mean, target_std=base.target_std, train_end=0,
        first_valid=np.zeros(len(base.target_mean), dtype=np.int64),
    )  # fmt: skip
    raw = d.dense_raw()
    b1_grid, cams = d.dense_b1()
    npz = np.load(e.stats_file(month=pipe.month))
    b1_g, b1_c = d.b1_scale(grid=b1_grid, cams=cams, mean=npz["mean"], std=npz["std"], device="cpu")
    std28 = d.standardised_inputs(raw=raw, fold=stat, device="cpu")  # (16, G, 28) float16
    n_grid = std28.shape[1]
    stamp_us = prep422.GRID_START_US + np.arange(n_grid, dtype=np.int64) * prep422.STEP_US
    day_code = np.clip(stamp_us // d.DAY_US - 1 - d.DAY0, 0, d.N_DAYS - 1)
    cams_dense = b1_c[:, torch.from_numpy(day_code)].half()
    dense_weather = torch.cat([std28[..., : d.N_WEATHER], b1_g, cams_dense], dim=-1)
    dense_datetime = std28[..., d.N_WEATHER :]
    assert dense_weather.shape[-1] == bt.TOP_WIDTH and dense_datetime.shape[-1] == 21
    holidays = lay.england_bank_holidays(years=range(2019, 2029))
    day_type = torch.from_numpy(
        lay.day_types_for_grid(
            grid_start_us=prep422.GRID_START_US, n_grid=n_grid, holidays=holidays
        )
    ).long()
    power_z = torch.from_numpy(
        ((clean - base.target_mean[:, None]) / base.target_std[:, None]).astype(np.float32)
    )
    # the ensemble table: training rows, then (prediction phase) the scored rows
    origin_idx = ((mr.origin_us - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)
    first_day = int((origin_idx.min() + lay.NATIVE_OFFSET_STEPS) // lay.STEPS_PER_DAY)
    last_day = int((origin_idx.max() + lay.NATIVE_OFFSET_STEPS) // lay.STEPS_PER_DAY)
    test = None
    test_origin_idx = None
    test_keys = None
    if pipe.fold.PHASE == "predict":
        test = e.build_test(month=pipe.month, standardiser=base.standardiser)
        last_day = max(
            last_day, int((test.origin_idx.max() + lay.NATIVE_OFFSET_STEPS) // lay.STEPS_PER_DAY)
        )
        test_origin_idx = np.unique(test.origin_idx)
        test_keys = {"origin_us": np.unique(test.origin)}
    n_days = last_day - first_day + 1
    ens = ens_table(
        x=mr.x, primary=mr.primary, origin_idx=origin_idx, target_idx=mr.target_idx,
        origin_us=mr.origin_us, first_day=first_day, n_days=n_days,
    )  # fmt: skip
    if test is not None:
        scored = ens_table(
            x=test.x, primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx,
            origin_us=test.origin, first_day=first_day, n_days=n_days,
        )  # fmt: skip
        ens.weather = torch.where(scored.filled[..., None], scored.weather, ens.weather)
        ens.spread = torch.where(scored.filled[..., None], scored.spread, ens.spread)
        ens.run_us = torch.where(scored.filled.any(dim=-1), scored.run_us, ens.run_us)  # ty: ignore[no-matching-overload]
        ens.filled |= scored.filled
    ens = bt.EnsTensors(
        weather=ens.weather.to(device), spread=ens.spread.to(device),
        filled=ens.filled.to(device), first_day=first_day,
        run_us=None if ens.run_us is None else ens.run_us.to(device),
    )  # fmt: skip
    blocks = ens.filled.view(*ens.filled.shape[:2], lay.N_LEAD_DAYS, lay.STEPS_PER_DAY)
    partly = int((blocks.any(dim=-1) & ~blocks.all(dim=-1)).sum())
    total_rows = int(blocks.any(dim=-1).sum())
    source = bt.Source(
        power_z=power_z.to(device), dense_weather=dense_weather.to(device),
        dense_datetime=dense_datetime.to(device), day_type=day_type.to(device), ens=ens,
    )  # fmt: skip
    data = FoldData(
        month=pipe.month, source=source, cutoff_idx=cutoff_idx, target_mean=base.target_mean,
        target_std=base.target_std, clean=clean, fingerprint=fingerprint(mr_base=base),
        train_ens_days=(first_day, last_day), test_origin_idx=test_origin_idx, test_keys=test_keys,
        dropped_per_fold={
            **flagged_counts(month=pipe.month, cutoff_us=int(cut_dt.timestamp() * 1e6)),
            "target_rows_partly_filled": partly,
            "target_rows_with_any_weather": total_rows,
        },
    )  # fmt: skip
    check_weather_finite(
        raw=raw, test_origin_idx=test_origin_idx, b1_grid=b1_grid, cams=cams, day_code=day_code
    )
    if keep_rows:
        data.rows = (mr, test)
    return data


def flagged_counts(*, month: str, cutoff_us: int) -> dict[str, int]:
    """Return the exact count of `substation_zero` half-hours of the 16 Primaries that the fold drops.

    The count is read from the flagged-keys table (`e00410_stale_predict.FLAGGED_KEYS`), which the
    cleaning rule wrote, and not from the cleaned power. `training_and_history` holds the flagged
    half-hours before the fold's cutoff, and `scored_window` those from the cutoff to 15 days after
    the end of the fold month (the actuals of the scored 14-day horizons).
    """
    import e00578_common as k578  # ty: ignore[unresolved-import]

    ids = k578.sites()["time_series_id"].to_list()
    flagged = pl.read_parquet(stale.FLAGGED_KEYS).select("time_series_id", "valid_time")
    flagged = flagged.filter(pl.col("time_series_id").is_in(ids))
    cutoff = datetime.fromtimestamp(cutoff_us / 1e6, tz=UTC)
    year, mon = (int(part) for part in month.split("-"))
    month_end = datetime(year + (mon == 12), mon % 12 + 1, 1, tzinfo=UTC)
    times = flagged["valid_time"].dt.replace_time_zone("UTC")
    before = times < cutoff
    window = (times >= cutoff) & (times < month_end + timedelta(days=15))
    return {
        "flagged_before_cutoff": int(before.sum()),
        "flagged_in_scored_window": int(window.sum()),
    }


def check_weather_finite(
    *,
    raw: np.ndarray,
    test_origin_idx: np.ndarray | None,
    b1_grid: np.ndarray,
    cams: np.ndarray,
    day_code: np.ndarray,
) -> None:
    """Assert the raw ERA5, CAMS, and B1 inputs are finite at every history token and target token.

    The check covers every scored origin (prediction phase), on the raw arrays: the standardised
    arrays hold 0 where a value was missing, which would hide a gap. `b1_grid` holds the 12 valid-time
    B1 columns and `cams` the 2 CAMS day columns, read through `day_code` as the dense array reads them.
    """
    if test_origin_idx is None:
        return
    for o in test_origin_idx:
        c = int(o) + lay.NATIVE_OFFSET_STEPS
        lo, hi = c - lay.STEPS_PER_DAY * lay.MAX_AGE, c + lay.STEPS_PER_DAY * lay.N_LEAD_DAYS
        assert bool(np.isfinite(np.asarray(raw[:, lo:hi, :7])).all()), (
            f"non-finite ERA5 or CAMS weather around index {c}"
        )
        assert bool(np.isfinite(np.asarray(b1_grid[:, lo:hi])).all()), (
            f"non-finite B1 valid-time column around index {c}"
        )
        assert bool(np.isfinite(np.asarray(cams)[:, day_code[lo:hi]]).all()), (
            f"non-finite CAMS day column around index {c}"
        )

# ruff: noqa: E501
"""Experiment 733: the leakage guards and the origin-time weather assertion.

A failure raises `CorrectnessFailureError`, which is verdict row 1. Every guard fails when it was
given nothing to check, so no guard passes on an empty counter. No message holds a date, a series
identifier, or a real value. The one function that formats a date, `example_row`, formats the
example row that the plan prints, and the caller prints it to the log the plan names.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NoReturn

import e00733_channels as channels
import e00733_core as core
import e00733_paths as paths
import numpy as np

# isort: split
import e00730_leakage as leak730

CorrectnessFailureError = leak730.CorrectnessFailureError
assert_files_older = leak730.assert_files_older
assert_nothing_written_to = leak730.assert_nothing_written_to

GRID_START_US: Final = 1_567_296_000_000_000
"""2019-09-01 00:00 UTC in microseconds (the grid start of experiment 422)."""
STEP_US: Final = 1_800_000_000
HOUR_US: Final = 3_600_000_000
DAY_US: Final = 24 * HOUR_US
FORBIDDEN_SYNTHETIC_INPUT_WORDS: Final = (
    "world",
    "engine",
    "capacity",
    "truth",
    "injected",
    "param",
)
"""Words that no input of the synthetic forecaster may carry in its name."""


def _fail(*, message: str) -> NoReturn:
    raise CorrectnessFailureError(message)


def _stamp(*, idx_us: int) -> str:
    return datetime.fromtimestamp(idx_us / 1e6, UTC).strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------------- origin-time weather


def native_origin_us(*, c_idx: np.ndarray) -> np.ndarray:
    """Return the native origin (09:00 UTC) in microseconds: the stamp of grid index `c - 6`."""
    return GRID_START_US + (np.asarray(c_idx, dtype=np.int64) - core.NATIVE_OFFSET_STEPS) * STEP_US


def assert_origin_time_weather(
    *,
    c_idx: np.ndarray,
    run_init_us: np.ndarray,
    target_idx: np.ndarray,
    filled: np.ndarray,
    other_latest_valid_us: np.ndarray,
    run_available_us: np.ndarray | None = None,
) -> int:  # fmt: skip
    """Check.

    For every target token of every sample, that the weather is the origin-time run and no other
    column is later.

    The checks are, per target token that exists (`filled`): the run's initialisation time is 00:00
    UTC on the origin's day D; the run's availability time (initialisation plus 9 hours unless
    given) is at or before the native origin; the target token's valid time is at or after the
    contract origin; and the latest valid time that any other target-row column reads is before the
    native origin.

    Args:
        c_idx: `(B,)` the grid index of the 12:00 UTC contract origin.
        run_init_us: `(B,)` or `(B, 48)` the initialisation time of the ENS run behind each sample
            (or token), in microseconds since the epoch.
        target_idx: `(B, 48)` the grid index of each target token.
        filled: `(B, 48)` bool, whether the token exists.
        other_latest_valid_us: `(B,)` or `(B, 48)` the latest valid time that the trailing-temperature
            columns, the solar-gain columns, and the CAMS day summary of day D - 1 read, in microseconds.
        run_available_us: The availability time of the run, or None for initialisation plus 9 hours.

    Returns:
        The number of target tokens checked.

    Raises:
        CorrectnessFailureError: When a token fails a check, or when no token was checked.
    """
    c = np.asarray(c_idx, dtype=np.int64)
    b = len(c)
    filled = np.asarray(filled, dtype=bool)
    n = int(filled.sum())
    if b == 0 or n == 0:
        _fail(
            message="the origin-time weather assertion was given no target token, so it checked nothing"
        )
    init = np.broadcast_to(np.asarray(run_init_us, dtype=np.int64).reshape(b, -1), filled.shape)
    other = np.broadcast_to(
        np.asarray(other_latest_valid_us, dtype=np.int64).reshape(b, -1), filled.shape
    )
    available = (
        init + core.ENSEMBLE_AVAILABLE_AFTER_HOURS * HOUR_US
        if run_available_us is None
        else np.broadcast_to(
            np.asarray(run_available_us, dtype=np.int64).reshape(b, -1), filled.shape
        )
    )
    native = native_origin_us(c_idx=c)[:, None]
    day_start = (GRID_START_US + c * STEP_US) // DAY_US * DAY_US
    contract = (GRID_START_US + c * STEP_US)[:, None]
    valid_us = GRID_START_US + np.asarray(target_idx, dtype=np.int64) * STEP_US
    if (init[filled] != np.broadcast_to(day_start[:, None], filled.shape)[filled]).any():
        _fail(
            message="a target token reads an ENS run that was not initialised at 00:00 UTC on the origin's day"
        )
    if (available[filled] > np.broadcast_to(native, filled.shape)[filled]).any():
        _fail(
            message="a target token reads an ENS run that is available only after the native origin"
        )
    if (valid_us[filled] < np.broadcast_to(contract, filled.shape)[filled]).any():
        _fail(message="a target token lies before the contract origin")
    if (other[filled] >= np.broadcast_to(native, filled.shape)[filled]).any():
        _fail(message="a target-row column reads a valid time at or after the native origin")
    return n


def read_run_files(
    *, summary_dirs: Sequence[Path], epoch_days: Sequence[int]
) -> dict[int, tuple[int, int]]:
    """Read each ensemble run's initialisation and availability times from the run files.

    The initialisation time is 00:00 UTC of the date in the file's name (`init=YYYY-MM-DD.parquet`),
    and the availability time is the file's own `power_fcst_init_time` column, which must hold one
    value.

    Args:
        summary_dirs: The folders that hold the runs' summary files.
        epoch_days: The days (days since 1970-01-01) whose runs are read.

    Returns:
        For each day, the initialisation time and the availability time in microseconds.

    Raises:
        CorrectnessFailureError: When a run has no file, or its file holds several availability times.
    """
    import polars as pl

    out: dict[int, tuple[int, int]] = {}
    for day in epoch_days:
        name = f"init={datetime.fromtimestamp(int(day) * 86_400, UTC):%Y-%m-%d}.parquet"
        path = next((f / name for f in summary_dirs if (f / name).exists()), None)
        if path is None:
            _fail(message=f"the run file {name} does not exist, so its availability cannot be read")
        values = (
            pl.scan_parquet(path)
            .select(pl.col("power_fcst_init_time").dt.epoch("us").unique())
            .collect()
            .to_series()
            .to_list()
        )
        if len(values) != 1:
            _fail(message=f"the run file {name} holds {len(values)} availability times, not one")
        out[int(day)] = (int(day) * DAY_US, int(values[0]))
    return out


def example_row(
    *,
    c_idx: int,
    target_idx: int,
    run_init_us: int,
    other_latest_valid_us: int,
    passed: bool = True,
    run_available_us: int | None = None,
) -> str:
    """Return the example row that the assertion prints, in the form of the plan.

    Args:
        c_idx: The grid index of the contract origin of the first sample.
        target_idx: The grid index of a target token of that sample.
        run_init_us: The initialisation time of its ENS run.
        other_latest_valid_us: The latest valid time of any other target-row column.
        passed: Whether the assertion passed.
        run_available_us: The run's availability time as read from the data (default: the
            initialisation time plus the publication delay).

    Returns:
        The text of the example row.
    """
    contract = GRID_START_US + c_idx * STEP_US
    native = GRID_START_US + (c_idx - core.NATIVE_OFFSET_STEPS) * STEP_US
    valid = GRID_START_US + target_idx * STEP_US
    lead_day = int((target_idx - c_idx) // core.STEPS_PER_DAY) + 1
    return (
        f"origin-time weather: {core.SCORED_PRIMARY}, contract origin {_stamp(idx_us=contract)} "
        f"(native {native // 1000000 // 3600 % 24:02d}:00 UTC),\n"
        f"target valid {_stamp(idx_us=valid)}, lead day {lead_day}, ENS run {_stamp(idx_us=run_init_us)},\n"
        f"available {_stamp(idx_us=run_init_us + core.ENSEMBLE_AVAILABLE_AFTER_HOURS * HOUR_US if run_available_us is None else run_available_us)}, "
        f"latest valid time of any other target-row column\n"
        f"{_stamp(idx_us=other_latest_valid_us)}: {'pass' if passed else 'FAIL'}"
    )


# --------------------------------------------------------------------------- the banks precede the history


def assert_base_days_precede_history(
    *, base_days_epoch: np.ndarray, first_history_epoch_day_by_fold: Mapping[str, int]
) -> None:  # fmt: skip
    """Check that every base day of every bank precedes the history of every later-stage sample.

    The history is the first history half-hour of any stage 2 or stage 3 sample of any fold.

    Args:
        base_days_epoch: The base days of the banks that trained, stopped, or tested experiments 730
            and 732, as days since 1970-01-01.
        first_history_epoch_day_by_fold: The first history day of any stage 2 or stage 3 sample, per fold.

    Raises:
        CorrectnessFailureError: When a set is empty or a base day is not before the earliest history day.
    """
    base = np.asarray(base_days_epoch)
    if base.size == 0 or not first_history_epoch_day_by_fold:
        _fail(
            message="the base calendar or the fold layout is empty, so the base-days check read nothing"
        )
    first = min(first_history_epoch_day_by_fold.values())
    if int(base.max()) >= first:
        _fail(
            message="a base day of a bank that trained or tested the untangling network is not before the first history day of a fold"
        )


def count_overlap(
    *, sample_first_history_day: np.ndarray, sample_last_history_day: np.ndarray, base_days_epoch: np.ndarray,
    channel_days_present: np.ndarray,
) -> tuple[int, int]:  # fmt: skip
    """Return the number of samples whose history overlaps a base day.

    And the number of untangled channel days present on base days.

    Args:
        sample_first_history_day: `(N,)` the first history day of each sample, epoch days.
        sample_last_history_day: `(N,)` the last history day of each sample.
        base_days_epoch: The base days.
        channel_days_present: The epoch days that hold an untangled channel in the table.

    Returns:
        The two counts.

    Raises:
        CorrectnessFailureError: When there is no sample or no channel day, so a zero count would be vacuous.
    """
    first, last = np.asarray(sample_first_history_day), np.asarray(sample_last_history_day)
    if (
        first.size == 0
        or np.asarray(channel_days_present).size == 0
        or np.asarray(base_days_epoch).size == 0
    ):
        _fail(
            message="no sample, channel day, or base day was given, so the overlap count would be vacuous"
        )
    base = np.sort(np.asarray(base_days_epoch))
    lo = np.searchsorted(base, first, side="left")
    hi = np.searchsorted(base, last, side="right")
    n_samples = int((hi > lo).sum())
    n_channels = int(np.isin(np.asarray(channel_days_present), base).sum())
    return n_samples, n_channels


def assert_no_overlap(*, n_samples: int, n_channels: int) -> None:
    """Check that no sample's history overlaps a base day and no channel is present on a base day."""
    if n_samples != 0 or n_channels != 0:
        _fail(
            message=f"{n_samples} samples overlap a base day and {n_channels} channel days sit on base days"
        )


# --------------------------------------------------------------------------- the target rows carry no channel


def poison_target_cells(
    *, values: np.ndarray, ok: np.ndarray, primary: int, target_idx: np.ndarray
) -> tuple[np.ndarray, int]:  # fmt: skip
    """Return a copy of the channel values `(P.

    G, 3)` with every target cell of one sample set to NaN, and the number poisoned.

    Args:
        values: The channel values.
        ok: The flags `(P, G)`; a poisoned cell is counted only where the flag is on.
        primary: The sample's Primary.
        target_idx: The grid indices of the sample's target tokens.

    Returns:
        The poisoned copy and the number of flagged cells that were set to NaN.
    """
    out = np.array(values, dtype=np.float32, copy=True)
    idx = np.asarray(target_idx, dtype=np.int64).ravel()
    out[primary, idx] = np.nan
    return out, int(np.asarray(ok)[primary, idx].sum())


def assert_poisoned_targets_change_nothing(
    *, run: Callable[[bool], tuple[np.ndarray, int]], tol: float = core.TOLERANCE_POISON
) -> int:  # fmt: skip
    """Check that a copy of the channel table with every target-row value set to NaN gives the same forecast.

    Args:
        run: `run(poison)` returns the forecast and the number of target cells that were poisoned.
        tol: The largest allowed absolute difference.

    Returns:
        The number of target cells poisoned.

    Raises:
        CorrectnessFailureError: When no cell was poisoned (the check would be vacuous), the
            poisoned forecast is non-finite, or the two forecasts differ by more than `tol`.
    """
    clean, _ = run(False)
    poisoned, n_cells = run(True)
    if n_cells <= 0:
        _fail(message="the poisoned copy changed no flagged cell, so the check cannot fail")
    if clean.size == 0 or not np.isfinite(clean).all() or not np.isfinite(poisoned).all():
        _fail(message="a forecast of the target-poison check is empty or non-finite")
    if float(np.abs(clean - poisoned).max()) > tol:
        _fail(message="a forecast depends on an untangled value of a target half-hour")
    return n_cells


# --------------------------------------------------------------------------- the placebo and the training cut


def assert_placebo_from_28_days_earlier(*, source_idx: np.ndarray, token_idx: np.ndarray) -> int:
    """Check that every placebo value comes from a half-hour at least 28 days before the token it fills.

    Args:
        source_idx: The grid index that each placebo value comes from.
        token_idx: The grid index of the token that it fills.

    Returns:
        The number of tokens checked.
    """
    src, tok = np.asarray(source_idx, dtype=np.int64), np.asarray(token_idx, dtype=np.int64)
    if src.size == 0 or src.shape != tok.shape:
        _fail(message="the placebo check was given no token or mismatched arrays")
    if ((tok - src) < core.PLACEBO_SHIFT_DAYS * core.STEPS_PER_DAY).any():
        _fail(
            message="a placebo value comes from a half-hour less than 28 days before the token it fills"
        )
    return int(src.size)


def assert_training_power_cut(*, power: np.ndarray, fold_start_idx: int) -> int:
    """Check that the cleaned power is NaN from the fold month's start onwards in a training process.

    Args:
        power: The standardised power `(P, G)` that the training process holds.
        fold_start_idx: The grid index of the fold month's first half-hour.

    Returns:
        The number of half-hours checked.
    """
    tail = np.asarray(power)[:, fold_start_idx:]
    if tail.size == 0:
        _fail(message="the training cut check read no half-hour after the fold's start")
    if np.isfinite(tail).any():
        _fail(message="a training process holds cleaned power at or after the fold month's start")
    return int(tail.size)


# --------------------------------------------------------------------------- the synthetic world


def _bank_digest(*, path: Path) -> str:
    """Return the content digest that experiment 729's bank code computes for a stored bank."""
    import e00729_banks as banks729

    return str(banks729.load_bank(path=path).digest)


def assert_bank_digests(
    *, root_730: Path, digest_of: Callable[[Path], str] | None = None
) -> int:  # fmt: skip
    """Check that every bank under experiment 730's folder has the digest of its `banks/summary.json`.

    Args:
        root_730: Experiment 730's data folder (read only).
        digest_of: Returns the recomputed digest of a bank file (defaults to experiment 729's `load_bank`).

    Returns:
        The number of banks checked.

    Raises:
        StoppedError: When a bank's digest differs from the summary's (verdict row 0).
        CorrectnessFailureError: When the summary lists no bank.
    """
    digests = json.loads((root_730 / "banks" / "summary.json").read_text())["digests"]
    if not digests:
        _fail(message="the bank summary lists no bank, so no digest was checked")
    compute = digest_of if digest_of is not None else (lambda path: _bank_digest(path=path))
    for key, digest in digests.items():
        kind, _, seed = key.partition("_s")
        name = f"{kind}_s{seed}.pkl" if seed else f"{kind}.pkl"
        if compute(root_730 / "banks" / name) != digest:
            msg = "a stored bank's digest differs from experiment 730's summary (verdict row 0)"
            raise channels.StoppedError(msg)
    return len(digests)


def assert_synthetic_train_banks(*, seeds: Sequence[int], fit_seeds_732: Sequence[int]) -> None:
    """Check that the synthetic forecaster trains on banks of seeds 3 and 4 only.

    Which no fit of experiment 732 read.
    """
    if tuple(sorted(seeds)) != core.SYNTHETIC_TRAIN_BANK_SEEDS:
        _fail(message="the synthetic forecaster does not train on the banks of seeds 3 and 4")
    if set(seeds) & set(fit_seeds_732):
        _fail(
            message="a training bank of the synthetic forecaster was read by a fit of experiment 732"
        )


def assert_synthetic_inputs_clean(*, input_names: Sequence[str]) -> None:
    """Check that no input of the synthetic forecaster reads the world type.

    The engine parameters, the solar capacity, or a truth array.
    """
    if not input_names:
        _fail(
            message="the synthetic forecaster has no input names, so the input check read nothing"
        )
    bad = [n for n in input_names if any(w in n.lower() for w in FORBIDDEN_SYNTHETIC_INPUT_WORDS)]
    if bad:
        _fail(message=f"{len(bad)} inputs of the synthetic forecaster carry a forbidden name")


def assert_limits_before_fits(*, limits: Path, fit_records: Sequence[Path]) -> None:
    """Check that `limits.json` is older than every fit record, and that at least one record was given."""
    if not fit_records:
        _fail(message="no fit record was given, so the limits-before-fits check read nothing")
    assert_files_older(older=limits, newer=list(fit_records), what="limits before fits")


def assert_no_test_choice(*, events: Sequence[str]) -> None:
    """Check that the limits and the fits' stops came before the first test score."""
    if "test_score" not in events:
        return
    first = events.index("test_score")
    for needed in ("limits", "fits"):
        if needed not in events[:first]:
            _fail(message=f"{needed} was not fixed before the first test score")


def assert_wrote_only_here(*, written: Sequence[Path], data_root: Path) -> None:
    """Check that every written path lies in this experiment's data folder and none lies in an earlier experiment's."""
    if not written:
        _fail(message="no written path was given, so the write check read nothing")
    root = data_root.resolve()
    for path in written:
        if root != path.resolve() and root not in path.resolve().parents:
            _fail(message="a stage wrote outside this experiment's data folder")
    for forbidden in paths.forbidden_write_roots():
        assert_nothing_written_to(written=written, forbidden_root=forbidden)

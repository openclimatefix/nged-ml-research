"""Experiment 730: the leakage and correctness assertions.

A failure raises `CorrectnessFailureError`.

Each function takes the objects it checks and returns `None`. None prints a month, a date, a series
identifier, or a value; the messages give counts and rule names only. Experiment 729's assertions
(split integrity, seed spaces, independent banks, the hash match, the parameter ranges, no
lockdown month, no record day in the base days) are called unchanged from experiment 729's module.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00729_leakage as leak729
import e00730_arms as arms
import e00730_core as core
import e00730_pretrain as pretrain
import numpy as np

CorrectnessFailureError = leak729.CorrectnessFailureError
ALLOWED_INPUT_ROWS: Final = leak729.ALLOWED_INPUT_ROWS | {arms.FUTURE_ROW}
ALLOWED_PREFIXES: Final = ("enc_",)
"""The encoder arms read 28 inputs named `enc_0` to `enc_27`, none a month or an ordinal."""
HOLIDAY_COLUMN: Final = 26
"""Index among the 28 encoder inputs of the signed distance to the nearest bank holiday."""
HOLIDAY_COLUMN_NAME: Final = "nearest_days"
"""Name of the nearest-holiday distance in experiment 428's non-lag feature layout."""
FLOAT_SLACK: Final = 1e-6


def _fail(*, message: str) -> None:
    raise CorrectnessFailureError(message)


def assert_pretrain_removal(
    *, target_idx: np.ndarray, origin_idx: np.ndarray, removed_mask: np.ndarray, valid_end_idx: int
) -> None:
    """Check that no pre-training row has a valid time or origin in a removed period.

    Args:
        target_idx: The valid-time grid index of every pre-training row.
        origin_idx: The origin grid index of every row.
        removed_mask: The removed half-hours of the grid.
        valid_end_idx: The grid index of 2026-07-01 00:00 (no row reaches it).
    """
    if not pretrain.keep_rows(
        target_idx=target_idx, origin_idx=origin_idx, removed_mask=removed_mask
    ).all():
        _fail(message="a pre-training row has a valid time or origin in a removed period")
    if (target_idx >= valid_end_idx).any():
        _fail(message="a pre-training row has a valid time on or after the holdout start")


def assert_standardiser_surviving(
    *, stored: pretrain.StoredStandardiser, surviving_x: np.ndarray
) -> None:
    """Check that the stored standardiser equals a refit on the surviving raw rows."""
    if not pretrain.standardiser_matches(stored=stored, surviving_x=surviving_x):
        _fail(message="the standardiser was fitted on a row that the removal drops")


CHECK_BATCH: Final = 32
"""Windows of one training batch that `run_fit` checks before training."""


def assert_batch_loss_masks(*, windows: np.ndarray, targets: np.ndarray, keeps: np.ndarray) -> None:
    """Check a training batch's loss masks against its own hidden half-hours.

    A half-hour is hidden when the reading's presence row of the window is zero on a valid
    half-hour of the scored day.
    """
    start = core.core729.MIDDLE_START
    present = windows[:, 1, start : start + 48] > 0.5
    valid = targets[:, 1] > 0.5
    assert_keeps_follow_hidden(hidden=valid & ~present, keeps=keeps, valid=valid)


def assert_no_foreign_weights(*, used: Sequence[Path], pretrained_root: Path) -> None:
    """Check that every encoder weight file used lies under this experiment's `pretrained/`."""
    root = pretrained_root.resolve()
    for path in used:
        if root not in path.resolve().parents:
            _fail(message="an encoder weight file lies outside this experiment's pretrained folder")


def assert_inputs_not_targets(
    *, windows: np.ndarray, targets: np.ndarray, spec: arms.ArmSpec
) -> None:
    """Check that no input row of a window equals a target row (truth, engine, solar, base demand).

    Args:
        windows: A batch of inputs `(B, C, 96)`.
        targets: The batch's targets `(B, 6, 48)`.
        spec: The arm.
    """
    for w, t in zip(windows, targets, strict=True):
        if arms.input_equals_target_rows(window=w, targets=t):
            _fail(message=f"arm {spec.name} reads an input row equal to a target row")


def assert_no_month_or_ordinal(*, specs: Mapping[str, arms.ArmSpec]) -> None:
    """Check that no arm reads a row that could carry the month or the day of the year."""
    for spec in specs.values():
        for name in arms.stack_names(spec=spec):
            known = name in ALLOWED_INPUT_ROWS or name.startswith(ALLOWED_PREFIXES)
            if not known or "month" in name or "doy" in name:
                _fail(message=f"arm {spec.name} reads a row not on the allowed list ({name})")


def assert_holiday_clip(*, nearest_days: np.ndarray) -> None:
    """Check the stored nearest-holiday distance: clipped at 7 days and divided by 7.

    Experiment 397's `feature_matrix` stores the signed distance divided by the clip, so every
    finite value lies in [-1, 1]. The check also requires that the value 1 is reached, because a
    column that never reaches the clip is not the distance (it could be a flag or another column).

    Args:
        nearest_days: The stored column over many rows.
    """
    values = np.asarray(nearest_days, dtype=float)
    if not np.isfinite(values).all():
        _fail(message="the nearest-holiday distance has a non-finite value")
    peak = float(np.max(np.abs(values)))
    if peak > 1.0 + FLOAT_SLACK:
        _fail(message="the nearest-holiday distance is not clipped at 7 days")
    if peak < 1.0 - FLOAT_SLACK:
        _fail(message="the nearest-holiday distance never reaches the 7-day clip (wrong column?)")


def assert_keeps_follow_hidden(*, hidden: np.ndarray, keeps: np.ndarray, valid: np.ndarray) -> None:
    """Check the loss masks: sign loss only on visible valid half-hours, demand loss on all valid.

    Args:
        hidden: Hidden half-hours of the scored day, `(B, 48)` bool.
        keeps: `(B, 2, 48)` sign and demand keeps.
        valid: Valid half-hours of the scored day, `(B, 48)` bool.
    """
    if (keeps[:, 0] > 0.5)[hidden].any():
        _fail(message="the sign loss applies on a hidden half-hour")
    if not np.array_equal(keeps[:, 0] > 0.5, valid & ~hidden):
        _fail(message="the sign loss does not equal the visible valid half-hours")


def assert_causal_blind(*, p_a: np.ndarray, p_b: np.ndarray, visible: np.ndarray) -> None:
    """Check that two windows equal up to the cut give equal outputs on the visible half-hours."""
    if not np.allclose(p_a[visible], p_b[visible], atol=1e-6):
        _fail(message="a causal arm's output on visible half-hours depends on the hidden future")


def assert_choices_before_test(*, events: Sequence[str]) -> None:
    """Check that the architecture, style, and margin were fixed before the first test score."""
    if "test_score" not in events:
        return
    first = events.index("test_score")
    for needed in ("architecture", "style", "margin"):
        if needed not in events[:first]:
            _fail(message=f"{needed} was not fixed before the first test score")


def assert_files_older(*, older: Path, newer: Sequence[Path], what: str) -> None:
    """Check that `older` was written before every file of `newer` (limits before fits)."""
    t = older.stat().st_mtime_ns
    if any(p.stat().st_mtime_ns < t for p in newer):
        _fail(message=f"{what}: a file was written before {older.name}")


def assert_nothing_written_to(*, written: Sequence[Path], forbidden_root: Path) -> None:
    """Check that no written path lies inside experiment 729's data folder."""
    root = forbidden_root.resolve()
    for path in written:
        if root == path.resolve() or root in path.resolve().parents:
            _fail(message="a file was written inside experiment 729's data folder")

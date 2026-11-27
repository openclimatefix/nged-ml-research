"""Experiment 733: shared helpers of the two post hoc diagnostics (per-lead reading and probe).

Both diagnostics are report-only. They read the stored predictions and fitted networks of the
synthetic world, change no stored file, and leave the verdict as scored. The pure functions here
split rows by lead day, split tokens by segment, and score a probability.
"""

from pathlib import Path
from typing import Final

import e00733_core as core
import numpy as np

DIAGNOSTICS_FOLDER: Final = "diagnostics"
"""The folder under the experiment's data folder that holds every output of the diagnostics."""


def diagnostics_folder(*, root: Path) -> Path:
    """Return the folder of the diagnostics under `root`, created when missing."""
    folder = root / DIAGNOSTICS_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def lead_day_of_rows(*, lead_index: np.ndarray, width: int = core.STEPS_PER_DAY) -> np.ndarray:
    """Return the lead day (lead index plus 1) of each half-hour row, `(N * width,)`."""
    return np.repeat(np.asarray(lead_index, dtype=np.int64) + 1, width)


def restrict_to_lead_day(*, values: np.ndarray, row_lead_day: np.ndarray, day: int) -> np.ndarray:
    """Return `values` with NaN on every row whose lead day is not `day`."""
    return np.where(row_lead_day == day, np.asarray(values, dtype=np.float64), np.nan)


def split_by_segment(
    *, seg: np.ndarray, test_share: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return the train and test masks of rows, such that no segment is in both.

    Args:
        seg: The segment of each row `(T,)`.
        test_share: The share of segments that go to the test side.
        seed: Seeds the draw of the segments.

    Returns:
        `(train, test)`, each `(T,)` bool; every row is in exactly one of them.

    Raises:
        ValueError: When fewer than two segments exist, or a side would hold no segment.
    """
    seg = np.asarray(seg)
    ids = np.unique(seg)
    if len(ids) < 2:
        msg = "a split by segment needs at least two segments"
        raise ValueError(msg)
    n_test = min(len(ids) - 1, max(1, round(test_share * len(ids))))
    chosen = np.random.default_rng(seed).permutation(ids)[:n_test]
    test = np.isin(seg, chosen)
    return ~test, test


def auc(*, p: np.ndarray, y: np.ndarray) -> float:
    """Return the area under the ROC curve (the chance that a positive outranks a negative).

    Ties count half. NaN when `y` holds only one class.
    """
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y) > 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p))
    start = 0
    while start < len(p):
        stop = start
        while stop + 1 < len(p) and sorted_p[stop + 1] == sorted_p[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def probability_scores(*, p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return the accuracy at 0.5, the Brier score, the AUC, and the base rate of `y == 1`."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return {
        "accuracy": float(np.mean((p >= 0.5) == (y > 0.5))),
        "brier": float(np.mean((p - y) ** 2)),
        "auc": auc(p=p, y=y),
        "base_rate": float(np.mean(y)),
        "n": len(y),
    }


def pick_tokens(
    *, valid: np.ndarray, per_sample: int, rng: np.random.Generator
) -> np.ndarray:  # fmt: skip
    """Return a mask that keeps at most `per_sample` random valid tokens of each sample.

    Args:
        valid: `(B, ...)` bool, the tokens that may be used.
        per_sample: The most tokens kept per sample.
        rng: The generator.

    Returns:
        A bool mask of the shape of `valid`, a subset of it.
    """
    flat = np.asarray(valid).reshape(len(valid), -1)
    score = np.where(flat, rng.random(flat.shape), -1.0)
    rank = np.argsort(np.argsort(-score, axis=1), axis=1)
    return (flat & (rank < per_sample)).reshape(np.asarray(valid).shape)

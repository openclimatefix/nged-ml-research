"""The step heads of experiment 732 as event callers, from stored predictions only.

This module copies the match, mask, and isotonic code of the recalibration report
(`common.py` in its folder, which is not committed) and reads the frozen maps and thresholds of
its `frozen_thresholds.json`, whose SHA-256 is asserted before anything is read. No network runs.
A head call is a half-hour at which the mean over three seeds of the step probability, after the
frozen isotonic map, reaches the frozen F1-rule threshold. Calls of one sign less than 2 hours
apart merge, and the sign of a call is the sign of the mean of the seeds' size output (positive
when D8 gains load).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00736_common import (
    DATA_732,
    DIAG,
    FROZEN_THRESHOLDS_SHA256,
    HALF_HOURS_PER_DAY,
    RECAL,
    sha256_of,
)
from e00736_match import match_by_segment, merge_calls

ARMS: Final[dict[str, str]] = {"U*": "Ustar_unet", "U-noBSP": "U-noBSP_unet"}
"""Head arm to the stem of its prediction folders."""
HEAD_ARM: Final[str] = "U-noBSP"
"""The benchmark head: experiment 732's U-noBSP."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
RECAL_TOLERANCE: Final[int] = 1
"""The recalibration report's start tolerance in half-hours."""
BANDS: Final[dict[str, tuple[float, float]]] = {
    "under 2": (0.0, 2.0),
    "2 to 4": (2.0, 4.0),
    "4 to 8": (4.0, 8.0),
    "8 and more": (8.0, float("inf")),
}
"""The report's size bands in step noise units."""
STORED_DIGESTS: Final[dict[str, str]] = {
    "U-noBSP_unet_s0/test_transfer.npz": (
        "5d001cbd78ccbb524aa5797d8ddb3a1afe154b4f9544fe028ba0ec908a820669"
    ),
    "U-noBSP_unet_s1/test_transfer.npz": (
        "e967ca20a284223627d51e9d1ddf31b05ee20b16cdd68b4b5c92137765002795"
    ),
    "U-noBSP_unet_s2/test_transfer.npz": (
        "db376858834aa8299162cea197e087e246608d8a1975f77446129bdd1fb22d25"
    ),
    "Ustar_unet_s0/test_transfer.npz": (
        "022aa08770290805381fdf6015866c05096a408ee6b0fabed47e348d75f8a378"
    ),
    "Ustar_unet_s1/test_transfer.npz": (
        "b4dd121238a475263cd78f4bc6859a2a6773d6c90528ac5a1ff00bb8f24c202a"
    ),
    "Ustar_unet_s2/test_transfer.npz": (
        "9a7b9856b6dab568fa7c2d3ff95317ba8697dad4dbf4d25c8b16281a0c0673bc"
    ),
}
"""SHA-256 of the prediction files of both head arms that this experiment reads."""
TRUTH_CACHE_SHA256: Final[str] = "33e935a3cb8f52e8670f9584067906c3470204364e22099768a46c9e91054cfe"
"""SHA-256 of the diagnostics' `truth_cache.npz`."""


@dataclass(frozen=True)
class HeadOutputs:
    """A head arm's stored test outputs, the mean of its seeds, on the 94,080 valid half-hours."""

    arm: str
    step_p: np.ndarray
    size: np.ndarray
    row_seg: np.ndarray
    row_date: np.ndarray
    row_hh: np.ndarray


def frozen_thresholds() -> dict:
    """Read the frozen maps and thresholds after checking the file's SHA-256.

    Returns:
        The parsed JSON.

    Raises:
        ValueError: When the file's digest differs from the one the plan names.
    """
    path = RECAL / "frozen_thresholds.json"
    digest = sha256_of(path=path)
    if digest != FROZEN_THRESHOLDS_SHA256:
        msg = f"frozen_thresholds.json has digest {digest}, not {FROZEN_THRESHOLDS_SHA256}"
        raise ValueError(msg)
    return json.loads(path.read_text())


def check_stored_digests() -> dict[str, bool]:
    """Compare the SHA-256 of the six prediction files and the truth cache with the plan.

    Returns:
        File name to whether its digest matches.
    """
    out = {
        name: sha256_of(path=DATA_732 / "predictions" / name) == digest
        for name, digest in STORED_DIGESTS.items()
    }
    out["truth_cache.npz"] = sha256_of(path=DIAG / "truth_cache.npz") == TRUTH_CACHE_SHA256
    return out


def load_outputs(*, arm: str) -> HeadOutputs:
    """Read an arm's three seeds' test outputs and average them.

    Args:
        arm: `U*` or `U-noBSP`.

    Returns:
        The seed means of `step_p` and `size`, and the rows' segment, date, and half-hour.

    Raises:
        ValueError: When the seeds disagree on which half-hours the rows are.
    """
    stem = ARMS[arm]
    zs = [np.load(DATA_732 / "predictions" / f"{stem}_s{k}" / "test_transfer.npz") for k in SEEDS]
    keys = ("row_seg", "row_date", "row_hh")
    for z in zs[1:]:
        if any(not np.array_equal(zs[0][k], z[k]) for k in keys):
            msg = f"the seeds of {arm} hold different rows"
            raise ValueError(msg)
    return HeadOutputs(
        arm=arm,
        step_p=np.mean([z["step_p"].astype(np.float64) for z in zs], axis=0),
        size=np.mean([z["size"].astype(np.float64) for z in zs], axis=0),
        row_seg=zs[0]["row_seg"].astype(np.int64),
        row_date=zs[0]["row_date"].astype(np.int64),
        row_hh=zs[0]["row_hh"].astype(np.int64),
    )


def iso_apply(*, p: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Apply an isotonic map given by its knots (copied from the recalibration report).

    Args:
        p: The raw probabilities.
        xs: The map's knot positions.
        ys: The map's knot values.

    Returns:
        The calibrated probabilities.
    """
    return np.interp(p, xs, ys)


def frozen_threshold(*, arm: str, frozen: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Return an arm's isotonic knots and F1-rule threshold from the frozen file.

    Args:
        arm: `U*` or `U-noBSP`.
        frozen: The parsed `frozen_thresholds.json`.

    Returns:
        The knots' x values, the knots' y values, and the threshold on the calibrated scale.
    """
    a = frozen["arms"][arm]
    thr = float(a["maps"]["isotonic"]["f1_rule"]["cal_threshold"])
    return np.asarray(a["isotonic_knots_x"]), np.asarray(a["isotonic_knots_y"]), thr


def called_rows(*, out: HeadOutputs, frozen: dict) -> np.ndarray:
    """Flag the rows at which the head calls a step, after the frozen map and threshold.

    Args:
        out: The arm's outputs.
        frozen: The parsed `frozen_thresholds.json`.

    Returns:
        A boolean array over the rows.
    """
    xs, ys, thr = frozen_threshold(arm=out.arm, frozen=frozen)
    return iso_apply(p=out.step_p, xs=xs, ys=ys) >= thr


def merged_calls(*, out: HeadOutputs, frozen: dict, first_day: dict[int, int]) -> pl.DataFrame:
    """Return the head's merged calls on each segment's dense axis.

    Args:
        out: The arm's outputs.
        frozen: The parsed `frozen_thresholds.json`.
        first_day: Segment index to the epoch day of its first dense day.

    Returns:
        One row per merged call, with `seg`, `time` (the bank's dense half-hour), and `sign`.
    """
    called = called_rows(out=out, frozen=frozen)
    rows: list[dict[str, int]] = []
    for seg in np.unique(out.row_seg[called]).tolist():
        pick = called & (out.row_seg == seg)
        dense = (out.row_date[pick] - first_day[seg]) * HALF_HOURS_PER_DAY + out.row_hh[pick]
        sign = np.where(out.size[pick] >= 0.0, 1, -1)
        t, s = merge_calls(times=dense, signs=sign)
        rows += [{"seg": seg, "time": int(a), "sign": int(b)} for a, b in zip(t, s, strict=True)]
    return pl.DataFrame(rows, schema={"seg": pl.Int64, "time": pl.Int64, "sign": pl.Int64})


def unmerged_call_count(*, out: HeadOutputs, frozen: dict) -> int:
    """Return the number of calls before merging.

    Args:
        out: The arm's outputs.
        frozen: The parsed `frozen_thresholds.json`.

    Returns:
        The number of called rows.
    """
    return int(called_rows(out=out, frozen=frozen).sum())


def found_by_head(
    *,
    truth: pl.DataFrame,
    calls: pl.DataFrame,
    tolerance: int,
    time_col: str = "start_hh",
    end: bool = False,
) -> np.ndarray:
    """Return, for each transfer of `truth`, whether a head call matches its start (or end) edge.

    Args:
        truth: The transfer table, with `seg`, `sign`, `size_points`, and the time column.
        calls: The head's merged calls.
        tolerance: The start tolerance in half-hours.
        time_col: The truth column that holds the edge's half-hour on the dense axis.
        end: Whether to match the end edge, with the opposite sign.

    Returns:
        A boolean array in the order of `truth`.
    """
    matched = match_by_segment(
        truth=truth, calls=calls, tolerance=tolerance, time_col=time_col, end=end
    )
    return matched >= 0


# --- the recalibration report's one-half-hour match, copied for gate G1 -------------------------


def un(*, flat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Put flat valid rows back onto the `(windows, 48)` grid, zero elsewhere.

    Args:
        flat: Values on the valid rows.
        valid: The grid's valid mask.

    Returns:
        The grid.
    """
    out = np.zeros(valid.shape, dtype=flat.dtype)
    out[valid] = flat
    return out


def recal_match(
    *, called: np.ndarray, true: np.ndarray, tol: int = RECAL_TOLERANCE
) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one greedy match of called half-hours to true edges (the report's `match`).

    Args:
        called: Boolean grid of calls.
        true: Boolean grid of injected edges.
        tol: The tolerance in half-hours.

    Returns:
        Matched calls and matched edges, as boolean grids.
    """
    pad = 2 * tol + 2
    cp = np.pad(called, [(0, 0), (0, pad)])
    tp = np.pad(true, [(0, 0), (0, pad)])
    fc, ft = np.flatnonzero(cp.ravel()), np.flatnonzero(tp.ravel())
    cok = np.zeros(fc.size, bool)
    tok = np.zeros(ft.size, bool)
    i = j = 0
    while i < fc.size and j < ft.size:
        g = int(fc[i]) - int(ft[j])
        if abs(g) <= tol:
            cok[i] = tok[j] = True
            i += 1
            j += 1
        elif g < 0:
            i += 1
        else:
            j += 1
    oc = np.zeros(cp.size, bool)
    ot = np.zeros(cp.size, bool)
    oc[fc[cok]] = True
    ot[ft[tok]] = True
    k = slice(0, called.shape[1])
    return oc.reshape(cp.shape)[:, k], ot.reshape(cp.shape)[:, k]


def reproduce_recalibration(*, arm: str = "U-noBSP") -> dict[str, float]:
    """Re-score the stored test predictions as the recalibration report did (gate G1, first part).

    Args:
        arm: The head arm.

    Returns:
        The number of calls, matched calls, the injected edges of 4 step noise units or more, how
        many were matched, and the recall.
    """
    frozen = frozen_thresholds()
    out = load_outputs(arm=arm)
    cache = np.load(DIAG / "truth_cache.npz")
    valid = cache["valid"]
    event = (cache["event"] > 0.5) & valid
    noise = un(flat=cache["tr_edge_noise"].astype(np.float64), valid=valid)
    big = event & (noise >= BANDS["4 to 8"][0])
    called = un(flat=called_rows(out=out, frozen=frozen), valid=valid) & valid
    cok, tok = recal_match(called=called, true=event)
    n_big = int(big.sum())
    tp_big = int(tok[big].sum())
    return {
        "calls": int(called.sum()),
        "calls_matched": int(cok[called].sum()),
        "edges_4_and_more": n_big,
        "edges_4_and_more_found": tp_big,
        "recall_4_and_more": tp_big / n_big if n_big else float("nan"),
    }

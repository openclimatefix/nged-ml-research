"""Experiment 730: the scores of P(X < 0), the harm and detection checks, and the contrasts.

Everything here is numpy. A forecast is a seed-averaged probability `p` of X < 0 on the rows of one
bank, with the rows in experiment 729's `Rows` order. The counted contrasts call experiment 729's
two-way bootstrap (`read_contrast`) unchanged, at the Bonferroni level of `core.LEVEL`.
"""

from collections.abc import Mapping, Sequence

import e00730_paths  # noqa: F401

# isort: split
import e00729_dist as dist
import e00729_score as score729
import e00730_core as core
import numpy as np

# --------------------------------------------------------------------------- per-row scores


def row_brier(*, p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return the squared difference between P(X < 0) and the indicator of X < 0, per row."""
    return (p - (x < 0.0).astype(np.float64)) ** 2


def row_log_loss(*, p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return the negative log likelihood of the sign, per row, with the probability clipped."""
    q = np.clip(p, core.PROB_CLIP, 1.0 - core.PROB_CLIP)
    return -np.where(x < 0.0, np.log(q), np.log1p(-q))


def seed_mean(*, p_by_seed: Sequence[np.ndarray]) -> np.ndarray:
    """Return the mean over seeds of each row's probability."""
    return np.mean(np.stack(p_by_seed), axis=0)


def mean_brier(*, p: np.ndarray, x: np.ndarray) -> float:
    """Return the mean Brier score."""
    return float(row_brier(p=p, x=x).mean())


# --------------------------------------------------------------------------- the checks


def decidable(*, rows: score729.Rows) -> np.ndarray:
    """Return the rows whose reading is at least 1 MVA (the sign can be read from the reading)."""
    return rows.r >= core.DECIDABLE_READING_POINTS


def harm_check(*, rows: score729.Rows, p: np.ndarray, limit: float) -> dict[str, float | bool]:
    """Return H1: P(X < 0) on synthetic import rows with a reading of at least 1 MVA.

    Args:
        rows: The scored rows of the test bank.
        p: The arm's seed-mean P(X < 0).
        limit: The harm limit L from `limits.json`.

    Returns:
        The mean P(X < 0), the share above 0.5, the row count, and `pass` (both at most L).
    """
    keep = decidable(rows=rows) & (rows.x >= 0.0)
    return harm_from_p(p=p[keep], limit=limit)


def harm_from_p(*, p: np.ndarray, limit: float) -> dict[str, float | bool]:
    """Return the two harm numbers of an array of probabilities on import half-hours."""
    if p.size == 0:
        return {"mean_p": float("nan"), "share_above_half": float("nan"), "n": 0, "pass": False}
    mean_p = float(p.mean())
    share = float(np.mean(p > core.FALSE_ALARM_THRESHOLD))
    return {
        "mean_p": mean_p,
        "share_above_half": share,
        "n": int(p.size),
        "pass": bool(mean_p <= limit and share <= limit),
    }


def detection(*, rows: score729.Rows, p: np.ndarray) -> dict[str, float | bool]:
    """Return d: the mean P(X < 0) on reverse-flow rows with a reading of at least 1 MVA."""
    keep = decidable(rows=rows) & (rows.x < 0.0)
    if not keep.any():
        return {"d": float("nan"), "n": 0, "pass": False}
    d = float(p[keep].mean())
    return {"d": d, "n": int(keep.sum()), "pass": bool(d >= core.DETECTION_BAR)}


def calibration(*, rows: score729.Rows, p: np.ndarray, bar: float) -> dict:
    """Return the sign calibration bar on rows outside the dead band.

    Args:
        rows: The scored rows.
        p: The arm's seed-mean P(X < 0).
        bar: The reliability bar, a tenth of s (`Limits.reliability_bar`).

    Returns:
        The reliability term, the largest gap of a populated bin, the bins, and `pass`.
    """
    keep = np.abs(rows.x) > core.DEAD_BAND_POINTS
    rel = dist.reliability_term(prob=p[keep], observed=rows.x[keep] < 0)
    ok = bool(rel["reliability"] <= bar and rel["max_gap"] <= core.BIN_GAP_BAR)  # ty: ignore[unsupported-operator]
    return {**rel, "bar": bar, "pass": ok}


def real_harm_check(*, p: np.ndarray, limit: float) -> dict[str, float | bool]:
    """Return H2: the same two limits on every half-hour of the real held-out pre-engine days."""
    return harm_from_p(p=p, limit=limit)


# --------------------------------------------------------------------------- contrasts


def contrast_at_level(
    *,
    rows: score729.Rows,
    a: np.ndarray,
    b: np.ndarray,
    keep: np.ndarray,
    n_resamples: int,
    level: float = core.LEVEL,
) -> score729.ContrastReading:
    """Return the reading of `a - b` at this experiment's level and generator seed (730).

    Experiment 729's `contrast` reads at its own level and seed, so the table and the reading are
    built here and passed to `read_contrast` with `core.GENERATOR_SEED` and `level`.
    """
    table = score729.cell_table(rows=rows, diff=a - b, keep=keep, months=None)
    return score729.read_contrast(
        table=table, n_resamples=n_resamples, seed=core.GENERATOR_SEED, level=level
    )


def counted_contrasts(
    *,
    rows: score729.Rows,
    per_row: Mapping[str, np.ndarray],
    keep: np.ndarray | None = None,
    n_resamples: int = core.N_RESAMPLES,
) -> dict[str, score729.ContrastReading]:
    """Return the seven counted contrasts C1 to C7, each as (first minus second) on per-row losses.

    Args:
        rows: The scored rows.
        per_row: Each arm's (or, for C6, each architecture's) seed-averaged per-row Brier score.
            A contrast whose name is missing is left out.
        keep: The rows read (all by default).
        n_resamples: The resamples of each reading.

    Returns:
        The readings by contrast name, at level `core.LEVEL`.
    """
    keep = np.ones(rows.n, dtype=bool) if keep is None else keep
    out = {}
    for name, (a, b) in core.CONTRASTS.items():
        if a in per_row and b in per_row:
            out[name] = contrast_at_level(
                rows=rows, a=per_row[a], b=per_row[b], keep=keep, n_resamples=n_resamples
            )
    return out


def report_only_contrast(
    *,
    rows: score729.Rows,
    a: np.ndarray,
    b: np.ndarray,
    keep: np.ndarray | None = None,
    n_resamples: int = core.N_RESAMPLES,
) -> score729.ContrastReading:
    """Return one report-only contrast (a minus b), read at the same level as the counted ones."""
    keep = np.ones(rows.n, dtype=bool) if keep is None else keep
    return contrast_at_level(rows=rows, a=a, b=b, keep=keep, n_resamples=n_resamples)


def contrast_dict(*, reading: score729.ContrastReading) -> dict[str, float]:
    """Return a contrast reading as a plain dictionary."""
    r = reading.reading
    return {
        "point": float(r.point), "lower": float(r.lower), "upper": float(r.upper),
        "n_rows": float(reading.n_rows), "widening": float(reading.widening),
    }  # fmt: skip


def c1_reading(*, readings: Mapping[str, score729.ContrastReading]) -> core.core729.Reading:
    """Return the headline contrast C1 as a `Reading` for the verdict table."""
    return readings["C1"].reading


# --------------------------------------------------------------------------- demand read-out


def demand_mae(
    *, mu: np.ndarray, base: np.ndarray, valid: np.ndarray, hidden: np.ndarray
) -> dict[str, float]:
    """Return the demand output's mean absolute error on hidden and visible half-hours, in points.

    Args:
        mu: The predicted demand, `(P, 48)`.
        base: The true base demand, `(P, 48)`.
        valid: Valid half-hours, `(P, 48)`.
        hidden: Half-hours whose reading was hidden, `(P, 48)` bool.

    Returns:
        The error on hidden, visible, and all valid half-hours (NaN where there are none).
    """
    err = np.abs(mu - base)
    out = {}
    for name, keep in (("hidden", valid & hidden), ("visible", valid & ~hidden), ("all", valid)):
        out[name] = float(err[keep].mean()) if keep.any() else float("nan")
    return out


# --------------------------------------------------------------------------- causal protocol


def read_at_cuts(*, values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Return `values[i, cuts[i]]` for each window `i`: the output at the cut's own half-hour."""
    return values[np.arange(len(cuts)), cuts]


def causal_cut(*, t: int, k: int) -> int | None:
    """Return the scored-day position where the window is cut to read the output at half-hour `t`.

    The protocol shows `k` half-hours after `t`, so the cut is at `t + k`, held at the last scored
    half-hour (47). At `k = 48` nothing is cut: the two-sided protocol, returned as None.
    """
    return None if k >= 48 else min(t + k, 47)


def causal_output_unchanged(
    *, p_cut: np.ndarray, p_future_scrambled: np.ndarray, atol: float = 1e-6
) -> bool:
    """Return whether the output at the cut's visible half-hours ignores everything after the cut.

    The unit test builds two windows that agree up to the cut and differ after it, and asserts that
    the arm's predictions are the same on the visible half-hours.
    """
    return bool(np.allclose(p_cut, p_future_scrambled, atol=atol))

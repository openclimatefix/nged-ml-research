# ruff: noqa: E501
"""Experiment 732: the scores, the five solar bars, the step readings, and the contrasts (numpy only).

A forecast is a flat array on the valid half-hours of one bank, in experiment 729's `Rows` order. The
overlay readings (injection, raised sky, transfers) are `(P, 48)` arrays with a validity mask. A bar
whose input set is empty fails (it never passes by having nothing to read). Contrasts call experiment
729's two-way bootstrap unchanged through experiment 730's `contrast_at_level`, at `core.LEVEL`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import e00732_paths  # noqa: F401

# isort: split
import e00729_score as score729
import e00730_score as score730
import e00732_core as core
import numpy as np

# --------------------------------------------------------------------------- solar errors


def sun_up(*, cosz: np.ndarray) -> np.ndarray:
    """Return the half-hours with the sun up (the cosine of the zenith angle above the registered floor)."""
    return cosz > core.SUN_UP_COSZ


def solar_abs(*, pv: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return the per-row absolute error of the solar output, in points."""
    return np.abs(pv - truth)


def solar_sq(*, pv: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return the per-row squared error of the solar output."""
    return (pv - truth) ** 2


def mean_on(*, values: np.ndarray, keep: np.ndarray) -> float:
    """Return the mean of `values` on `keep` (NaN when no row is kept)."""
    return float(values[keep].mean()) if keep.any() else float("nan")


def rmse_on(*, values_sq: np.ndarray, keep: np.ndarray) -> float:
    """Return the root of the mean of squared errors on `keep` (NaN when no row is kept)."""
    return float(np.sqrt(values_sq[keep].mean())) if keep.any() else float("nan")


# --------------------------------------------------------------------------- the five solar bars


def injection_keep(
    *, delta: np.ndarray, amount: float, cosz: np.ndarray, pair: np.ndarray
) -> np.ndarray:
    """Return the half-hours an injection reading uses: paired window, sun up, and delta at least 5% of the amount.

    Args:
        delta: The scored delta `(P, 48)`.
        amount: The injected amount a, in points.
        cosz: The cosine of the zenith angle `(P, 48)`.
        pair: 1 where the window has a pair `(P,)`.
    """
    return (pair[:, None] > 0.5) & sun_up(cosz=cosz) & (delta >= core.DELTA_SHARE_OF_A * amount)


def recovery_bar(
    *, pv_inj: np.ndarray, pv_plain: np.ndarray, delta: np.ndarray, amount: float, keep: np.ndarray
) -> dict[str, float | bool]:
    """Return the recovery bar at one amount: the ratio of summed response to summed delta, and the median relative error.

    The bar fails when there is no half-hour to read.
    """
    if not keep.any() or float(delta[keep].sum()) <= 0.0:
        return {"rho": float("nan"), "median_error_over_a": float("nan"), "n": 0, "pass": False}
    rho = float((pv_inj - pv_plain)[keep].sum() / delta[keep].sum())
    med = float(np.median(np.abs(pv_inj - pv_plain - delta)[keep]) / amount)
    ok = core.RECOVERY_BAND[0] <= rho <= core.RECOVERY_BAND[1] and med <= core.RECOVERY_MEDIAN_ERROR
    return {"rho": rho, "median_error_over_a": med, "n": int(keep.sum()), "pass": bool(ok)}


def injection_error_rows(
    *, pv_inj: np.ndarray, pv_plain: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """Return the per-row absolute injection error |dPV - delta|."""
    return np.abs(pv_inj - pv_plain - delta)


def night_bar(*, pv: np.ndarray, cosz: np.ndarray, valid: np.ndarray) -> dict[str, float | bool]:
    """Return the night bar: the mean solar output at night and the share above 0.5 points."""
    keep = valid & (cosz <= core.NIGHT_COSZ)
    if not keep.any():
        return {"mean": float("nan"), "share_above": float("nan"), "n": 0, "pass": False}
    mean, share = float(pv[keep].mean()), float(np.mean(pv[keep] > core.NIGHT_EXCEED_POINTS))
    return {
        "mean": mean,
        "share_above": share,
        "n": int(keep.sum()),
        "pass": bool(mean <= core.NIGHT_MEAN_POINTS and share <= core.NIGHT_EXCEED_SHARE),
    }


def violations(*, pv: np.ndarray, pv_raised: np.ndarray, raisable: np.ndarray) -> np.ndarray:
    """Return, on raisable half-hours, 1.0 where PV(w) - PV(wc) exceeds the tolerance (a monotonicity violation)."""
    return ((pv - pv_raised) > core.MONOTONE_TOLERANCE_POINTS).astype(np.float64)


def monotone_bar(
    *, pv: np.ndarray, pv_raised: np.ndarray, raisable: np.ndarray
) -> dict[str, float | bool]:
    """Return the monotonicity bar: the share of raisable half-hours where the raised sky lowers the solar output."""
    if not raisable.any():
        return {"share": float("nan"), "n": 0, "pass": False}
    share = float(violations(pv=pv, pv_raised=pv_raised, raisable=raisable)[raisable].mean())
    return {
        "share": share,
        "n": int(raisable.sum()),
        "pass": bool(share <= core.MONOTONE_MAX_SHARE),
    }


def response_bar(
    *, pv: np.ndarray, pv_raised: np.ndarray, raisable: np.ndarray, solar_target: np.ndarray
) -> dict[str, float | bool]:
    """Return the response bar: the share of raisable half-hours of solar windows (target at least 1 point) that respond."""
    keep = raisable & (solar_target >= core.RESPONSE_MIN_TARGET_POINTS)
    if not keep.any():
        return {"share": float("nan"), "n": 0, "pass": False}
    share = float(np.mean((pv_raised - pv)[keep] > core.MONOTONE_TOLERANCE_POINTS))
    return {"share": share, "n": int(keep.sum()), "pass": bool(share >= core.RESPONSE_MIN_SHARE)}


# --------------------------------------------------------------------------- the step output


def step_brier_rows(*, step_p: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Return the per-row Brier score of the step event."""
    return (step_p - event) ** 2


def _dilate(*, a: np.ndarray, width: int) -> np.ndarray:
    """Return a boolean array that is True within `width` half-hours (along the last axis) of a True entry."""
    out = a.copy()
    for k in range(1, width + 1):
        out[..., k:] |= a[..., :-k]
        out[..., :-k] |= a[..., k:]
    return out


def step_f1(
    *,
    step_p: np.ndarray,
    event: np.ndarray,
    size: np.ndarray,
    valid: np.ndarray,
    threshold: float = core.STEP_THRESHOLD,
    min_points: float = core.BIG_STEP_POINTS,
) -> dict[str, float | bool]:
    """Return precision, recall, and F1 of the step output at `threshold` on steps of at least `min_points`.

    A true step counts as found when a prediction above the threshold lies within one half-hour of it, and a
    prediction counts as right when a true step of that size lies within one half-hour of it. All inputs are
    `(P, 48)`.
    """
    true_big = (event > 0.5) & (np.abs(size) >= min_points) & valid
    called = (step_p > threshold) & valid
    if not true_big.any():
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "n_true": 0,
            "n_called": int(called.sum()),
        }
    recall = float(_dilate(a=called, width=core.STEP_TOLERANCE_HALF_HOURS)[true_big].mean())
    precision = (
        float(_dilate(a=true_big, width=core.STEP_TOLERANCE_HALF_HOURS)[called].mean())
        if called.any()
        else 0.0
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_true": int(true_big.sum()),
        "n_called": int(called.sum()),
    }


def size_error(
    *, size_hat: np.ndarray, size: np.ndarray, event: np.ndarray, valid: np.ndarray
) -> float:
    """Return the mean absolute error of the step size on the true edge half-hours (NaN when there are none)."""
    keep = (event > 0.5) & valid
    return mean_on(values=np.abs(size_hat - size), keep=keep)


def transfer_scenarios(
    *, inside: np.ndarray, since_end_days: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the four scenario masks of table T2 on the transfer overlay.

    Args:
        inside: Half-hours inside a transfer `(P, 48)`.
        since_end_days: Days since the end of the latest transfer to end before the half-hour (inf where none), `(P, 48)`.
        valid: Valid half-hours.

    Returns:
        The masks "all", "outside", "inside", and "after" (within 7 days after a transfer ends).
    """
    after = valid & ~inside & (since_end_days >= 0) & (since_end_days < 7)
    outside = valid & ~inside & (since_end_days >= 7)
    return {"all": valid.copy(), "outside": outside, "inside": valid & inside, "after": after}


# --------------------------------------------------------------------------- harm by band


def band_masks(*, rows: score729.Rows) -> dict[str, np.ndarray]:
    """Return the import half-hours of each band (pooled, 1 to 3 MVA, 3 MVA and above) with a decidable reading."""
    imp = (rows.x >= 0.0) & (rows.r >= core.DECIDABLE_READING_POINTS)
    return {
        "pooled": imp,
        "1-3MVA": imp & (rows.r < core.BAND_EDGE_POINTS),
        "3MVA+": imp & (rows.r >= core.BAND_EDGE_POINTS),
    }


def harm_by_band(
    *, rows: score729.Rows, p: np.ndarray, limits: Mapping[str, core.BandLimit]
) -> dict[str, dict[str, float | bool]]:
    """Return H1 for each band: the mean P(X<0), the share above 0.5, and the pass flag against the band's limit."""
    return {
        name: score730.harm_from_p(p=p[keep], limit=limits[name].limit)
        for name, keep in band_masks(rows=rows).items()
    }


def band_boundary(*, band: Mapping[str, float | bool], b13_lower: float) -> str:
    """Return how the band of 1 to 3 MVA reads: "pass", "at the calibrated boundary", or "fail" (above its limit and B13's lower end above zero)."""
    if band["pass"]:
        return "pass"
    return "fail" if b13_lower > 0.0 else "at the calibrated boundary"


# --------------------------------------------------------------------------- the injection rows for the bootstrap


@dataclass(frozen=True)
class InjectionRows:
    """The rows of the injection overlay with the amount in the key (what the bootstrap reads as `Rows`).

    Attributes:
        seg: The segment number of each row.
        month: The month index.
        date: The base day.
        hh: The half-hour.
        amount: The injected amount of each row.
    """

    seg: np.ndarray
    month: np.ndarray
    date: np.ndarray
    hh: np.ndarray
    amount: np.ndarray

    @property
    def n(self) -> int:
        """The number of rows."""
        return int(self.seg.shape[0])

    def keys(self) -> np.ndarray:
        """Return the (segment, date, half-hour, amount) key of each row as a structured array."""
        return np.rec.fromarrays(
            [self.seg, self.date, self.hh, self.amount], names="seg,date,hh,amount"
        )


def stack_amounts(*, rows: score729.Rows, amounts: Sequence[float]) -> InjectionRows:
    """Tile the plain test rows once per amount, with the amount in the key (segments stay whole units)."""
    k = len(amounts)
    return InjectionRows(
        seg=np.tile(rows.seg, k), month=np.tile(rows.month, k), date=np.tile(rows.date, k),
        hh=np.tile(rows.hh, k), amount=np.repeat(np.asarray(amounts, dtype=np.float64), rows.n),
    )  # fmt: skip


def keys_are_unique(*, rows: InjectionRows) -> bool:
    """Return whether no two rows share a (segment, date, half-hour, amount) key."""
    return len(np.unique(rows.keys())) == rows.n


def injection_contrast(
    *,
    rows: InjectionRows,
    a: np.ndarray,
    b: np.ndarray,
    keep: np.ndarray,
    n_resamples: int = core.N_RESAMPLES,
) -> score729.ContrastReading:
    """Return C3: the contrast of the injection errors of two arms on the stacked rows.

    Fails when the keys repeat, because the bootstrap would then double-count a row.
    """
    if not keys_are_unique(rows=rows):
        msg = "the injection rows repeat a (segment, date, half-hour, amount) key"
        raise ValueError(msg)
    return score730.contrast_at_level(rows=rows, a=a, b=b, keep=keep, n_resamples=n_resamples)  # ty: ignore[invalid-argument-type]


# --------------------------------------------------------------------------- contrasts and shares


def contrast_of(
    *,
    rows: object,
    a: np.ndarray,
    b: np.ndarray,
    keep: np.ndarray | None = None,
    n_resamples: int = core.N_RESAMPLES,
) -> score729.ContrastReading:
    """Return the reading of `a - b` per row on `keep` (all rows by default), at the counted level, seed 732."""
    n = int(a.shape[0])
    keep = np.ones(n, dtype=bool) if keep is None else keep
    return score730.contrast_at_level(rows=rows, a=a, b=b, keep=keep, n_resamples=n_resamples)  # ty: ignore[invalid-argument-type]


def reading_dict(*, reading: score729.ContrastReading) -> dict[str, float]:
    """Return a contrast reading as a plain dictionary (point, ends, rows, Kish numbers, widening)."""
    return {
        **score730.contrast_dict(reading=reading),
        "kish_blocks": float(reading.kish_blocks),
        "kish_segments": float(reading.kish_segments),
    }


def log_loss_opposes(*, brier_point: float, log_loss_point: float) -> bool:
    """Return whether the log-loss point has the opposite sign to the Brier point (audit trigger 3)."""
    return bool(np.sign(brier_point) * np.sign(log_loss_point) < 0)


def c4_subsets(*, rows: score729.Rows) -> dict[str, np.ndarray]:
    """Return the rows of C4 and of R12: all rows, the solar-only world, and the no-injection world.

    The masks come from the world type of each row, so a world type missing from the bank gives an empty
    mask and the reading on it is void.
    """
    names = list(core.core729.WORLD_TYPES)
    world = np.asarray(rows.world)
    return {
        "all": np.ones(rows.n, dtype=bool),
        "solar_only": world == names.index("solar"),
        "no_injection": world == names.index("none"),
    }


def with_overrides(*, rows: score729.Rows, **fields: np.ndarray) -> score729.Rows:
    """Return a copy of `rows` with some arrays replaced (a convenience for the tests)."""
    return replace(rows, **fields)  # ty: ignore[invalid-argument-type]

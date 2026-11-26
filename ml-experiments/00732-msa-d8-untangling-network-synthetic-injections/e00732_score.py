# ruff: noqa: E501
"""Experiment 732: the scores, the bars, the contrasts, and the verdict.

This module holds the scores, the eight solar bars, the step readings, the contrasts, the words, and
the verdict (numpy only).

**A forecast is a flat array on the valid half-hours of one bank, in experiment 729's `Rows` order.**
The overlay readings (injection, raised sky, transfers) are flat arrays on the same rows, so the
bootstrap pairs every contrast on identical row keys.

**A bar fails when it reads nothing, and when it reads a non-finite value.** A bar never passes
because its input set is empty, and a head that outputs NaN is a silent component, not a pass.

**Contrasts call experiment 729's two-way bootstrap with Kish widening at `core.LEVEL`, with generator
seed `core.GENERATOR_SEED`.** They refuse a non-finite per-row difference on a kept row.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import e00729_score as score729
import e00730_score as score730
import e00732_core as core
import e00732_overlays as ov
import numpy as np

HB_LEVEL: Final = 1.0 - 2.0 * (1.0 - core.LEVEL)
"""The two-sided level whose lower end is the one-sided lower bound at `core.LEVEL` (HB reads that bound)."""
CASE_NAMES: Final = ("read_exact", "read_jittered", "unread_bsp_a", "other_bsp")
"""The partner cases of a transfer: a read neighbour with an exact mirror, the same with a jittered mirror, another Primary under BSP A, and a Primary under another BSP."""
MIRRORED_CASES: Final = (0, 1)
UNMIRRORED_CASES: Final = (2, 3)
SHAPE_NAMES: Final = core.SHAPES
BAND_LABELS_NOISE: Final = ("under_2", "2_to_4", "4_to_8", "8_and_more")
BAND_LABELS_POINTS: Final = ("3_to_5", "5_to_10", "10_to_20", "20_to_40")
BAND_LABELS_HOURS: Final = ("1_to_3h", "3_to_12h", "12h_to_2d", "2d_to_4d")
Bar = dict[str, float | bool]

# --------------------------------------------------------------------------- solar errors


def finite_on(*arrays: np.ndarray, keep: np.ndarray) -> bool:
    """Return whether every array is finite on every kept row (False when no row is kept)."""
    return bool(keep.any() and all(np.isfinite(a[keep]).all() for a in arrays))


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


# --------------------------------------------------------------------------- the eight solar bars


def injection_keep(
    *, delta: np.ndarray, amount: float, cosz: np.ndarray, pair: np.ndarray,
    exclude: np.ndarray | None = None,
) -> np.ndarray:  # fmt: skip
    """Return the half-hours an injection reading uses.

    They lie in a paired window, with the sun up and delta at least 5% of the amount.

    Args:
        delta: The scored delta (flat, or `(P, 48)`).
        amount: The injected amount a, in points.
        cosz: The cosine of the zenith angle, the same shape as `delta`.
        pair: 1 where the window has a pair: the same shape as `delta`, or one value per window.
        exclude: Unreliable half-hours of S1 solar output (zero-filled gaps) to leave out, or None.

    Returns:
        The boolean mask of the half-hours the reading uses.
    """
    pair_b = pair.reshape(pair.shape + (1,) * (delta.ndim - pair.ndim))
    keep = (pair_b > 0.5) & sun_up(cosz=cosz) & (delta >= core.DELTA_SHARE_OF_A * amount)
    return keep if exclude is None else keep & ~exclude


def recovery_bar(
    *, pv_inj: np.ndarray, pv_plain: np.ndarray, delta: np.ndarray, amount: float, keep: np.ndarray
) -> Bar:
    """Return the recovery bar at one amount.

    The bar reads the ratio of summed response to summed delta, and the median relative error.

    The bar fails when there is no half-hour to read, and when a kept value is not finite.
    """
    if not finite_on(pv_inj, pv_plain, delta, keep=keep) or float(delta[keep].sum()) <= 0.0:
        return {"rho": float("nan"), "median_error_over_a": float("nan"), "n": 0, "pass": False}
    rho = float((pv_inj - pv_plain)[keep].sum() / delta[keep].sum())
    med = float(np.median(np.abs(pv_inj - pv_plain - delta)[keep]) / amount)
    ok = core.RECOVERY_BAND[0] <= rho <= core.RECOVERY_BAND[1] and med <= core.RECOVERY_MEDIAN_ERROR
    return {"rho": rho, "median_error_over_a": med, "n": int(keep.sum()), "pass": bool(ok)}


def recovery_reverse_bar(
    *,
    pv_inj: Mapping[float, np.ndarray],
    pv_plain: np.ndarray,
    delta: Mapping[float, np.ndarray],
    keep: Mapping[float, np.ndarray],
    reverse: np.ndarray,
) -> Bar:
    """Return the recovery bar on half-hours where the plain member is in reverse flow, pooled over the amounts.

    Args:
        pv_inj: The injected member's solar output by amount.
        pv_plain: The plain member's solar output.
        delta: The scored delta by amount.
        keep: The rows each amount's injection reading uses (`injection_keep`).
        reverse: Where the plain member's signed truth is negative.

    Returns:
        The pooled ratio `rho`, the number of half-hours `n`, and the pass flag.
    """
    num = den = 0.0
    n = 0
    for amount, mask in keep.items():
        use = mask & reverse
        if not use.any():
            continue
        if not finite_on(pv_inj[amount], pv_plain, delta[amount], keep=use):
            return {"rho": float("nan"), "n": 0, "pass": False}
        num += float((pv_inj[amount] - pv_plain)[use].sum())
        den += float(delta[amount][use].sum())
        n += int(use.sum())
    if n == 0 or den <= 0.0:
        return {"rho": float("nan"), "n": 0, "pass": False}
    rho = num / den
    lo, hi = core.RECOVERY_REVERSE_BAND
    return {"rho": rho, "n": n, "pass": bool(lo <= rho <= hi)}


def injection_error_rows(
    *, pv_inj: np.ndarray, pv_plain: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """Return the per-row absolute injection error |dPV - delta|."""
    return np.abs(pv_inj - pv_plain - delta)


def absolute_bar(
    *, pv: np.ndarray, solar: np.ndarray, cosz: np.ndarray, world_is_solar: np.ndarray,
    exclude: np.ndarray | None = None,
) -> Bar:  # fmt: skip
    """Return the absolute bar.

    The bar passes when, on sun-up half-hours of solar worlds, the mean absolute error is at most
    half the mean target.

    The bar fails when no row is kept, when the mean target is zero (a silent solar world), and when
    a kept value is not finite.
    """
    keep = world_is_solar & sun_up(cosz=cosz)
    if exclude is not None:
        keep = keep & ~exclude
    if not finite_on(pv, solar, keep=keep):
        return {"mae": float("nan"), "mean_target": float("nan"), "n": 0, "pass": False}
    mae = float(np.abs(pv - solar)[keep].mean())
    target = float(solar[keep].mean())
    return {
        "mae": mae,
        "mean_target": target,
        "n": int(keep.sum()),
        "pass": bool(target > 0.0 and mae <= core.ABSOLUTE_BAR_SHARE * target),
    }


def night_bar(*, pv: np.ndarray, cosz: np.ndarray, valid: np.ndarray) -> Bar:
    """Return the night bar: the mean solar output at night and the share above 0.5 points."""
    keep = valid & (cosz <= core.NIGHT_COSZ)
    if not finite_on(pv, keep=keep):
        return {"mean": float("nan"), "share_above": float("nan"), "n": 0, "pass": False}
    mean, share = float(pv[keep].mean()), float(np.mean(pv[keep] > core.NIGHT_EXCEED_POINTS))
    return {
        "mean": mean,
        "share_above": share,
        "n": int(keep.sum()),
        "pass": bool(mean <= core.NIGHT_MEAN_POINTS and share <= core.NIGHT_EXCEED_SHARE),
    }


def violations(*, pv: np.ndarray, pv_raised: np.ndarray) -> np.ndarray:
    """Return 1.0 where PV(w) - PV(wc) exceeds the tolerance.

    Such a half-hour is a monotonicity violation. The result is NaN where an output is not finite.
    """
    out = ((pv - pv_raised) > core.MONOTONE_TOLERANCE_POINTS).astype(np.float64)
    return np.where(np.isfinite(pv) & np.isfinite(pv_raised), out, np.nan)


def monotone_bar(*, pv: np.ndarray, pv_raised: np.ndarray, raisable: np.ndarray) -> Bar:
    """Return the monotonicity bar: the share of raisable half-hours where the raised sky lowers the solar output."""
    if not finite_on(pv, pv_raised, keep=raisable):
        return {"share": float("nan"), "n": 0, "pass": False}
    share = float(violations(pv=pv, pv_raised=pv_raised)[raisable].mean())
    return {
        "share": share,
        "n": int(raisable.sum()),
        "pass": bool(share <= core.MONOTONE_MAX_SHARE),
    }


def response_bar(
    *, pv: np.ndarray, pv_raised: np.ndarray, raisable: np.ndarray, solar_target: np.ndarray
) -> Bar:
    """Return the response bar.

    The bar is the share of raisable half-hours of solar windows (target at least 1 point) that
    respond.
    """
    keep = raisable & (solar_target >= core.RESPONSE_MIN_TARGET_POINTS)
    if not finite_on(pv, pv_raised, solar_target, keep=keep):
        return {"share": float("nan"), "n": 0, "pass": False}
    share = float(np.mean((pv_raised - pv)[keep] > core.MONOTONE_TOLERANCE_POINTS))
    return {"share": share, "n": int(keep.sum()), "pass": bool(share >= core.RESPONSE_MIN_SHARE)}


def gain_bar(*, reading: score729.ContrastReading | None) -> Bar:
    """Return a bar that passes when a contrast is a gain.

    The contrast is counted or report-only, and a gain means its upper end lies below zero. A void
    reading fails.

    "Better than the sky alone" reads R23 through this bar, and "better than the demand output" reads C4.
    """
    if reading is None:
        return {"point": float("nan"), "upper": float("nan"), "pass": False}
    return {
        "point": float(reading.reading.point),
        "upper": float(reading.reading.upper),
        "pass": core.is_gain(reading=reading.reading),
    }


def solar_bars(
    *,
    recovery: Mapping[float, Bar] | None,
    recovery_reverse: Bar | None,
    absolute: Bar | None,
    sky: Bar | None,
    night: Bar | None,
    monotone: Bar | None,
    response: Bar | None,
    demand: Bar | None,
) -> dict[str, bool]:
    """Return the eight solar bars as pass flags.

    A bar that was not read fails, and so does a recovery bar that is missing an amount.
    """
    flags = {
        "recovery": bool(
            recovery is not None
            and set(recovery) == set(core.TEST_INJECTION_POINTS)
            and all(bool(v["pass"]) for v in recovery.values())
        ),
        "recovery_reverse": bool(recovery_reverse is not None and recovery_reverse["pass"]),
        "absolute": bool(absolute is not None and absolute["pass"]),
        "better_than_sky": bool(sky is not None and sky["pass"]),
        "night": bool(night is not None and night["pass"]),
        "monotone": bool(monotone is not None and monotone["pass"]),
        "response": bool(response is not None and response["pass"]),
        "better_than_demand": bool(demand is not None and demand["pass"]),
    }
    return {name: flags[name] for name in core.SOLAR_BARS}


# --------------------------------------------------------------------------- the step output


def step_brier_rows(*, step_p: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Return the per-row Brier score of the step event."""
    return (step_p - event) ** 2


def match_steps(
    *, called: np.ndarray, true: np.ndarray, tolerance: int = core.STEP_TOLERANCE_HALF_HOURS
) -> tuple[np.ndarray, np.ndarray]:
    """Match calls to true steps one to one, each within `tolerance` half-hours along the last axis.

    A call matches at most one true step and a true step at most one call (the plan's rule). The
    greedy pass along each row in time order gives the largest possible number of matches.

    Args:
        called: Boolean calls `(P, T)`.
        true: Boolean true steps `(P, T)`.
        tolerance: The largest distance in half-hours of a match.

    Returns:
        Which calls found a true step, and which true steps were found, both shaped like the inputs.
    """
    width = called.shape[-1] + 2 * tolerance + 2
    flat_called = np.flatnonzero(np.pad(called, [(0, 0), (0, 2 * tolerance + 2)]).reshape(-1))
    flat_true = np.flatnonzero(np.pad(true, [(0, 0), (0, 2 * tolerance + 2)]).reshape(-1))
    call_ok = np.zeros(flat_called.size, dtype=bool)
    true_ok = np.zeros(flat_true.size, dtype=bool)
    i = j = 0
    while i < flat_called.size and j < flat_true.size:
        gap = int(flat_called[i]) - int(flat_true[j])
        if abs(gap) <= tolerance:
            call_ok[i] = true_ok[j] = True
            i += 1
            j += 1
        elif gap < 0:
            i += 1
        else:
            j += 1
    out_called = np.zeros(called.size + called.shape[0] * (width - called.shape[-1]), dtype=bool)
    out_true = np.zeros_like(out_called)
    out_called[flat_called[call_ok]] = True
    out_true[flat_true[true_ok]] = True
    shape = (called.shape[0], width)
    keep = slice(0, called.shape[-1])
    return out_called.reshape(shape)[:, keep], out_true.reshape(shape)[:, keep]


def step_f1(
    *,
    step_p: np.ndarray,
    event: np.ndarray,
    size: np.ndarray,
    valid: np.ndarray,
    threshold: float = core.STEP_THRESHOLD,
    min_points: float = core.BIG_STEP_POINTS,
    select: np.ndarray | None = None,
) -> dict[str, float | bool]:
    """Return precision, recall, and F1 of the step output at `threshold` on steps of at least `min_points`.

    Calls and true steps of any size are matched one to one within one half-hour (`match_steps`), so
    a true step is found by at most one call and a call finds at most one true step. A true step of
    `min_points` or more counts as found when it is matched, and a call counts as right when it is
    matched to a true step of any size, because a call at a smaller real step is not a false alarm.
    The inputs are `(P, 48)`, or `(1, N)` for a flat series.

    Args:
        step_p: The step probability.
        event: The true step event.
        size: The signed step size at an edge.
        valid: Valid half-hours.
        threshold: The call threshold.
        min_points: The smallest step the recall and the F1 read.
        select: Restricts the true steps the recall reads (a band of table T6); the precision still
            reads every call.

    Returns:
        The precision, recall, F1, the number of true steps read, and the number of calls.
    """
    all_true = (event > 0.5) & valid
    true_big = all_true & (np.abs(size) >= min_points)
    if select is not None:
        true_big &= select
    called = (step_p > threshold) & valid
    if not true_big.any():
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "n_true": 0,
            "n_called": int(called.sum()),
        }
    call_ok, true_ok = match_steps(called=called, true=all_true)
    recall = float(true_ok[true_big].mean())
    precision = float(call_ok[called].mean()) if called.any() else 0.0
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
        inside: Half-hours inside a transfer.
        since_end_days: Days since the end of the latest transfer to end before the half-hour (inf
            where none).
        valid: Valid half-hours.

    Returns:
        The masks "all", "outside" (outside a transfer and at least 7 days after one ends),
        "inside", and "after" (within 7 days after a transfer ends).
    """
    after = valid & ~inside & (since_end_days >= 0) & (since_end_days < 7)
    outside = valid & ~inside & (since_end_days >= 7)
    return {"all": valid.copy(), "outside": outside, "inside": valid & inside, "after": after}


@dataclass(frozen=True)
class TransferRows:
    """What each scored half-hour of the transfer overlay knows about its transfers.

    The arrays are flat, in the order of the rows.

    Attributes:
        inside: The half-hour lies inside a transfer.
        since_end_days: Days since the end of the latest transfer to end before the half-hour (inf
            where none).
        edge: The half-hour holds a transfer edge.
        edge_points: The size of the transfer that holds the edge, in points (0 elsewhere).
        edge_noise: The same in D8's noise units.
        edge_hours: The duration of the transfer that holds the edge, in hours (0 elsewhere).
        edge_direction: +1 where the transfer gives D8 load, -1 where it takes load, 0 off an edge.
        edge_shape: The shape of the transfer that holds the edge, as an index into `SHAPE_NAMES`
            (-1 off an edge).
        edge_case: The partner case of that transfer, as an index into `CASE_NAMES` (-1 off an edge).
        win_case: The partner case of the transfer that holds the half-hour or, failing that, the
            first transfer that touches its window (-1 where the window holds none).
        win_shape: The shape of that transfer (-1 where none).
        win_edge: A transfer edge lies in the half-hour's input window.
    """

    inside: np.ndarray
    since_end_days: np.ndarray
    edge: np.ndarray
    edge_points: np.ndarray
    edge_noise: np.ndarray
    edge_hours: np.ndarray
    edge_direction: np.ndarray
    edge_shape: np.ndarray
    edge_case: np.ndarray
    win_case: np.ndarray
    win_shape: np.ndarray
    win_edge: np.ndarray


def partner_case(*, transfer: ov.Transfer) -> int:
    """Return a transfer's partner case: an index into `CASE_NAMES`."""
    if transfer.mirrored:
        return 0 if transfer.jitter == 1.0 else 1
    return 2 if transfer.inside_bsp_a else 3


def transfer_rows(
    *,
    transfers: Sequence[Sequence[ov.Transfer]],
    picks: Sequence[tuple[int, int]],
    valid: np.ndarray,
    noise_d8: float,
) -> TransferRows:
    """Return the per-row attributes of the transfer overlay, in the order of `valid` read flat.

    Args:
        transfers: The transfers of each prepared segment (`Prepared.transfers`).
        picks: `(segment index, dense day)` of each window, as `arms.all_picks` gives them.
        valid: The valid half-hours `(P, 48)`.
        noise_d8: D8's noise unit in points.

    Returns:
        The row attributes of the transfer overlay.
    """
    n = len(picks)
    cols = {
        "inside": np.zeros((n, 48), bool),
        "since_end_days": np.full((n, 48), np.inf),
        "edge": np.zeros((n, 48), bool),
        "edge_points": np.zeros((n, 48)),
        "edge_hours": np.zeros((n, 48)),
        "edge_direction": np.zeros((n, 48)),
        "edge_shape": np.full((n, 48), -1),
        "edge_case": np.full((n, 48), -1),
        "win_case": np.full((n, 48), -1),
        "win_shape": np.full((n, 48), -1),
        "win_edge": np.zeros((n, 48), bool),
    }
    for k, (seg, day) in enumerate(picks):
        base = day * 48
        flat = base + np.arange(48)
        lo_win = base - core.core729.MIDDLE_START
        hi_win = lo_win + core.core729.WINDOW_STEPS
        touching = sorted(
            (t for t in transfers[seg] if t.start_half_hour < hi_win and t.end_half_hour >= lo_win),
            key=lambda t: t.start_half_hour,
        )
        for t in transfers[seg]:
            cols["inside"][k] |= (flat >= t.start_half_hour) & (flat <= t.end_half_hour)
            ended = flat > t.end_half_hour
            days_since = (flat - t.end_half_hour) / 48.0
            cols["since_end_days"][k] = np.where(
                ended, np.minimum(cols["since_end_days"][k], days_since), cols["since_end_days"][k]
            )
            for edge_hh in (t.start_half_hour, t.end_half_hour):
                if base <= edge_hh < base + 48:
                    h = edge_hh - base
                    cols["edge"][k, h] = True
                    cols["edge_points"][k, h] = abs(t.size)
                    cols["edge_hours"][k, h] = t.ticks / 12.0
                    cols["edge_direction"][k, h] = np.sign(t.size)
                    cols["edge_shape"][k, h] = SHAPE_NAMES.index(t.shape)
                    cols["edge_case"][k, h] = partner_case(transfer=t)
        if touching:
            first = touching[0]
            cols["win_case"][k] = partner_case(transfer=first)
            cols["win_shape"][k] = SHAPE_NAMES.index(first.shape)
            for t in touching:
                held = (flat >= t.start_half_hour) & (flat <= t.end_half_hour)
                cols["win_case"][k, held] = partner_case(transfer=t)
                cols["win_shape"][k, held] = SHAPE_NAMES.index(t.shape)
            edges = [e for t in touching for e in (t.start_half_hour, t.end_half_hour)]
            cols["win_edge"][k] = any(lo_win <= e < hi_win for e in edges)
    pick = {name: value[valid] for name, value in cols.items()}
    return TransferRows(**pick, edge_noise=pick["edge_points"] / noise_d8)


def band_masks_of(
    *, values: np.ndarray, edges: Sequence[float], labels: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return one mask per band of `values`: below the first edge, between edges, and from the last edge up."""
    bounds = (-np.inf, *edges, np.inf)
    return {
        label: (values >= bounds[k]) & (values < bounds[k + 1]) for k, label in enumerate(labels)
    }


def step_table(
    *,
    step_p: np.ndarray,
    size_hat: np.ndarray | None,
    event: np.ndarray,
    size: np.ndarray,
    tr: TransferRows,
) -> dict[str, dict[str, dict[str, float | bool]]]:
    """Return table T6 for one arm or rule.

    The table holds recall and the step-size error by direction, noise-unit band, point band, and
    duration band.

    The inputs are flat rows of the transfer overlay, read as one series of `N` half-hours. The
    precision of a band is the precision over all calls, because a call has no band of its own.
    """
    one = {
        "p": step_p[None, :],
        "e": event[None, :],
        "s": size[None, :],
        "v": np.ones((1, event.size), dtype=bool),
    }
    groups = {
        "direction": {"gains_load": tr.edge_direction > 0, "loses_load": tr.edge_direction < 0},
        "size_noise_units": band_masks_of(
            values=tr.edge_noise, edges=core.SIZE_BANDS_NOISE_UNITS, labels=BAND_LABELS_NOISE
        ),
        "size_points": band_masks_of(
            values=tr.edge_points, edges=core.SIZE_BANDS_POINTS[1:-1], labels=BAND_LABELS_POINTS
        ),
        "duration": band_masks_of(
            values=tr.edge_hours, edges=core.DURATION_BANDS_HOURS[1:-1], labels=BAND_LABELS_HOURS
        ),
    }
    out: dict[str, dict[str, dict[str, float | bool]]] = {}
    for group, bands in groups.items():
        out[group] = {}
        for label, mask in bands.items():
            band = mask & tr.edge
            reading = step_f1(
                step_p=one["p"], event=one["e"], size=one["s"], valid=one["v"],
                min_points=0.0, select=band[None, :],
            )  # fmt: skip
            reading["size_error"] = (
                float("nan")
                if size_hat is None
                else mean_on(values=np.abs(size_hat - size), keep=band)
            )
            out[group][label] = reading
    return out


def precision_all(*, step_p: np.ndarray, event: np.ndarray) -> float:
    """Return the precision of the step calls over the whole transfer overlay.

    A call near any true step counts as right.
    """
    called = (step_p > core.STEP_THRESHOLD)[None, :]
    if not called.any():
        return 0.0
    true = (event > 0.5)[None, :]
    call_ok, _ = match_steps(called=called, true=true)
    return float(call_ok[called].mean())


def detectability(
    *, table: Mapping[str, Mapping[str, float | bool]], precision: float, counts: Mapping[str, int]
) -> dict[str, object]:
    """Return the detectability statement of one arm: the noise-unit bands that are detectable.

    A band is detectable when its recall and the precision over the whole transfer overlay both
    reach 0.5. The statement is void when the overlay holds fewer than 100 scored steps on either
    side of 4 noise units, and a void statement never stops the line.

    Args:
        table: T6's noise-unit bands for the arm (the `size_noise_units` entry of `step_table`).
        precision: The precision of the arm over every call.
        counts: The scored steps under 4 noise units ("below") and at 4 and more ("above").

    Returns:
        Whether the statement is void, the smallest detectable band, and every detectable band.
    """
    void = (
        counts.get("below", 0) < core.MIN_STEPS_PER_SIDE
        or counts.get("above", 0) < core.MIN_STEPS_PER_SIDE
    )
    found = [
        label
        for label in BAND_LABELS_NOISE
        if label in table
        and math.isfinite(float(table[label]["recall"]))
        and core.detectable_band(recall=float(table[label]["recall"]), precision=precision)
    ]
    return {
        "void": void,
        "smallest_detectable": None if void or not found else found[0],
        "detectable": found,
    }


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
    """Return how the band of 1 to 3 MVA reads.

    The reading is "pass", "at the calibrated boundary", or "fail". It fails when the band is above
    its limit with HB's lower bound above zero.
    """
    if band["pass"]:
        return "pass"
    return "fail" if b13_lower > 0.0 else "at the calibrated boundary"


# --------------------------------------------------------------------------- contrasts


def contrast_at_level(
    *,
    rows: object,
    a: np.ndarray,
    b: np.ndarray,
    keep: np.ndarray,
    n_resamples: int = core.N_RESAMPLES,
    level: float = core.LEVEL,
    months: Sequence[int] | None = None,
) -> score729.ContrastReading:
    """Return the reading of `a - b` per row on `keep`, from experiment 729's two-way bootstrap with Kish widening.

    The generator seed is `core.GENERATOR_SEED` (732). The call refuses a non-finite difference on a
    kept row, because a silent component would otherwise pass as an effect of zero.

    Args:
        rows: The rows (their segment, month, and date keys form the resampling units).
        a: The first arm's per-row loss.
        b: The second arm's per-row loss, paired on the same rows.
        keep: The rows read.
        n_resamples: The number of resamples.
        level: The two-sided level.
        months: The months to include (all by default).

    Raises:
        ValueError: When a kept row has a non-finite difference.

    Returns:
        The contrast reading.
    """
    diff = a - b
    if keep.any() and not np.isfinite(diff[keep]).all():
        msg = "a contrast reads a non-finite per-row difference"
        raise ValueError(msg)
    table = score729.cell_table(
        rows=rows,
        diff=np.where(keep, diff, 0.0),
        keep=keep,
        months=months,  # ty: ignore[invalid-argument-type]
    )
    return score729.read_contrast(
        table=table, n_resamples=n_resamples, seed=core.GENERATOR_SEED, level=level
    )


def contrast_of(
    *,
    rows: object,
    a: np.ndarray,
    b: np.ndarray,
    keep: np.ndarray | None = None,
    n_resamples: int = core.N_RESAMPLES,
    months: Sequence[int] | None = None,
) -> score729.ContrastReading:
    """Return the reading of `a - b` per row on `keep` (all rows by default), at the counted level."""
    keep = np.ones(a.shape[0], dtype=bool) if keep is None else keep
    return contrast_at_level(rows=rows, a=a, b=b, keep=keep, n_resamples=n_resamples, months=months)


def harm_hb(
    *, rows: score729.Rows, p_u: np.ndarray, p_s: np.ndarray, n_resamples: int = core.N_RESAMPLES
) -> score729.ContrastReading:
    """Return HB: U* minus S in the mean P(X<0) on import half-hours of the band of 1 to 3 MVA.

    The reading is a one-sided lower bound.

    The one-sided lower bound at `core.LEVEL` is the lower end of the two-sided reading at
    `HB_LEVEL`. HB is outside the Bonferroni count of seven.
    """
    return contrast_at_level(
        rows=rows,
        a=p_u,
        b=p_s,
        keep=band_masks(rows=rows)["1-3MVA"],
        n_resamples=n_resamples,
        level=HB_LEVEL,
    )


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
        """Return the (segment, date, half-hour, amount) key of each row as a record array."""
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
    """Return C3: the contrast of the injection errors of two arms on the stacked rows, keyed by amount.

    Args:
        rows: The stacked rows (`stack_amounts`).
        a: The first arm's per-row injection error.
        b: The second arm's per-row injection error.
        keep: The rows read.
        n_resamples: The number of resamples.

    Raises:
        ValueError: When the keys repeat, because the bootstrap would then double-count a row.

    Returns:
        The contrast reading.
    """
    if not keys_are_unique(rows=rows):
        msg = "the injection rows repeat a (segment, date, half-hour, amount) key"
        raise ValueError(msg)
    return contrast_at_level(rows=rows, a=a, b=b, keep=keep, n_resamples=n_resamples)


def reading_dict(*, reading: score729.ContrastReading | None) -> dict[str, float | bool]:
    """Return a contrast reading as a plain dictionary.

    The dictionary holds the point, the ends, the rows, the Kish numbers, the widening, and whether
    the reading is void.
    """
    if reading is None:
        nan = float("nan")
        return {
            "point": nan, "lower": nan, "upper": nan, "n_rows": 0.0, "widening": nan,
            "kish_blocks": nan, "kish_segments": nan, "void": True,
        }  # fmt: skip
    return {
        **score730.contrast_dict(reading=reading),
        "kish_blocks": float(reading.kish_blocks),
        "kish_segments": float(reading.kish_segments),
        "void": not finite_reading(reading=reading),
    }


def finite_reading(*, reading: score729.ContrastReading | None) -> bool:
    """Return whether a contrast reading exists and its point and both ends are finite."""
    return bool(
        reading is not None
        and all(
            math.isfinite(z)
            for z in (reading.reading.point, reading.reading.lower, reading.reading.upper)
        )
    )


def log_loss_opposes(*, brier_point: float, log_loss_point: float) -> bool:
    """Return whether the log-loss point has the opposite sign to the Brier point (audit trigger 3)."""
    return bool(np.sign(brier_point) * np.sign(log_loss_point) < 0)


def _world_codes(*, names: Sequence[str]) -> list[int]:
    return [list(core.core729.WORLD_TYPES).index(n) for n in names]


def c4_subsets(*, rows: score729.Rows) -> dict[str, np.ndarray]:
    """Return the rows of C4 and of R12: all rows, the solar-only world, and the no-injection world.

    A world type missing from the bank gives an empty mask, and the reading on it is void.
    """
    world = np.asarray(rows.world)
    solar, none = _world_codes(names=("solar", "none"))
    return {
        "all": np.ones(rows.n, dtype=bool),
        "solar_only": world == solar,
        "no_injection": world == none,
    }


def solar_world(*, rows: score729.Rows) -> np.ndarray:
    """Return the rows of the solar worlds (solar only, and engine and solar)."""
    return np.isin(np.asarray(rows.world), _world_codes(names=("solar", "both")))


def with_overrides(*, rows: score729.Rows, **fields: np.ndarray) -> score729.Rows:
    """Return a copy of `rows` with some arrays replaced."""
    return replace(rows, **fields)  # ty: ignore[invalid-argument-type]


# --------------------------------------------------------------------------- what the readings read


@dataclass
class ArmScores:
    """One arm's (or rule's) seed-averaged outputs on the flat valid rows of the test bank.

    None means the output does not exist.

    Attributes:
        p: P(X<0) on the plain rows.
        p_tr: P(X<0) on the transfer overlay.
        p_nobsp: P(X<0) on the plain rows with the BSP channel and its mask zeroed at scoring time.
        pv: The solar output on the plain rows.
        pv_tr: The solar output on the transfer overlay.
        pv_inj: The solar output of the injected member, by injected amount.
        pv_raised: The solar output of the raised-sky member.
        step: The step probability on the transfer overlay.
        size_hat: The step size on the transfer overlay.
    """

    p: np.ndarray | None = None
    p_tr: np.ndarray | None = None
    p_nobsp: np.ndarray | None = None
    pv: np.ndarray | None = None
    pv_tr: np.ndarray | None = None
    pv_inj: dict[float, np.ndarray] = field(default_factory=dict)
    pv_raised: np.ndarray | None = None
    step: np.ndarray | None = None
    size_hat: np.ndarray | None = None


@dataclass
class Truth:
    """The truth and the row attributes that the readings read, flat on the valid rows of the test bank.

    Attributes:
        rows: The scored rows (the plain test bank).
        x_tr: The signed truth on the transfer overlay.
        event: The step event on the transfer overlay.
        size: The signed step size on the transfer overlay.
        cosz: The cosine of the zenith angle on the plain rows.
        raisable: The raisable half-hours of the raised-sky member.
        delta: The injected output by amount.
        pair: Whether the window has a pair, by amount.
        tr: The transfer overlay's row attributes.
        unreliable: The unreliable S1 solar half-hours on the plain rows (None when absent).
    """

    rows: score729.Rows
    x_tr: np.ndarray
    event: np.ndarray
    size: np.ndarray
    cosz: np.ndarray
    raisable: np.ndarray
    delta: dict[float, np.ndarray]
    pair: dict[float, np.ndarray]
    tr: TransferRows
    unreliable: np.ndarray | None = None

    @property
    def exclude(self) -> np.ndarray:
        """The unreliable S1 solar half-hours on the plain rows (all False when none was recorded)."""
        if self.unreliable is None:
            return np.zeros(self.cosz.shape, dtype=bool)
        return self.unreliable


def injection_stack(*, truth: Truth, arm: ArmScores) -> np.ndarray | None:
    """Return an arm's per-row absolute injection error stacked over the three amounts.

    The result is None when the arm lacks an amount or the plain output.
    """
    if arm.pv is None or set(arm.pv_inj) != set(core.TEST_INJECTION_POINTS):
        return None
    return np.concatenate(
        [
            injection_error_rows(pv_inj=arm.pv_inj[a], pv_plain=arm.pv, delta=truth.delta[a])
            for a in core.TEST_INJECTION_POINTS
        ]
    )


def injection_keep_stack(*, truth: Truth) -> np.ndarray:
    """Return the rows of C3: paired, sun up, delta at least 5% of the amount, stacked over the three amounts."""
    return np.concatenate(
        [
            injection_keep(
                delta=truth.delta[a],
                amount=a,
                cosz=truth.cosz,
                pair=truth.pair[a].astype(float),
                exclude=truth.exclude,
            )
            for a in core.TEST_INJECTION_POINTS
        ]
    )


def split_by_case(*, case: np.ndarray) -> dict[str, np.ndarray]:
    """Return one mask per partner case (R16 and R22)."""
    return {name: case == k for k, name in enumerate(CASE_NAMES)}


def split_by_shape(*, shape: np.ndarray) -> dict[str, np.ndarray]:
    """Return one mask per transfer shape (R17)."""
    return {name: shape == k for k, name in enumerate(SHAPE_NAMES)}


def compute_readings(
    *,
    truth: Truth,
    arms: Mapping[str, ArmScores],
    n_resamples: int = core.N_RESAMPLES,
) -> dict[str, object]:
    """Return C1 to C7, HB, and the report-only contrasts as contrast readings.

    A reading is None, and so void, where an input is missing.

    Every reading pairs the two arms on the same rows. The arms are keyed `U*`, `U-NC`, `U-NCw`, `S+`,
    `U-SG`, `U*-L`, `U-N`, `U-noBSP`, the stored `S` and `S-L`, and the baselines `B-PV0`, `B-PV1`,
    `B-PV2`, `B-TN`, and `B-T`. A reading that needs an arm that is absent is None.

    Args:
        truth: The truth and the row attributes.
        arms: The outputs of each arm.
        n_resamples: The resamples of every reading.

    Returns:
        "counted" (C1 to C7), "hb", and "report" (R-readings; a split reading is keyed
        `R16_read_exact` and the like).
    """
    rows = truth.rows
    solar = rows.solar
    up = sun_up(cosz=truth.cosz) & ~truth.exclude
    bsp_keep = np.asarray(rows.has_bsp, dtype=bool)
    bsp_months = score729.bsp_months(rows=rows) if bsp_keep.any() else None
    tr = truth.tr
    allrows = np.ones(rows.n, dtype=bool)
    stacked = stack_amounts(rows=rows, amounts=core.TEST_INJECTION_POINTS)
    inj_keep = injection_keep_stack(truth=truth)
    subsets = c4_subsets(rows=rows)

    def arm(name: str) -> ArmScores:
        return arms.get(name, ArmScores())

    def brier(name: str) -> np.ndarray | None:
        p = arm(name).p
        return None if p is None else score730.row_brier(p=p, x=rows.x)

    def brier_tr(name: str) -> np.ndarray | None:
        p = arm(name).p_tr
        return None if p is None else score730.row_brier(p=p, x=truth.x_tr)

    def sol(name: str) -> np.ndarray | None:
        pv = arm(name).pv
        return None if pv is None else np.abs(pv - solar)

    def stp(name: str) -> np.ndarray | None:
        s = arm(name).step
        return None if s is None else (s - truth.event) ** 2

    def viol(name: str) -> np.ndarray | None:
        a = arm(name)
        return (
            None
            if a.pv is None or a.pv_raised is None
            else violations(pv=a.pv, pv_raised=a.pv_raised)
        )

    def read(
        a: np.ndarray | None,
        b: np.ndarray | None,
        keep: np.ndarray,
        months: Sequence[int] | None = None,
    ) -> score729.ContrastReading | None:
        if a is None or b is None or not keep.any():
            return None
        return contrast_at_level(
            rows=rows, a=a, b=b, keep=keep, n_resamples=n_resamples, months=months
        )

    def read_inj(name_a: str, name_b: str) -> score729.ContrastReading | None:
        a, b = (
            injection_stack(truth=truth, arm=arm(name_a)),
            injection_stack(truth=truth, arm=arm(name_b)),
        )
        if a is None or b is None or not inj_keep.any():
            return None
        return injection_contrast(rows=stacked, a=a, b=b, keep=inj_keep, n_resamples=n_resamples)

    counted = {
        "C1": read(brier("U*"), brier("S"), allrows),
        "C2": read(sol("U*"), sol("U-NC"), up),
        "C3": read_inj("U*", "U-NC"),
        "C4": read(sol("U*"), sol("B-PV2"), up),
        "C5": read(stp("U*"), stp("B-TN"), allrows),
        "C6": read(brier("U*"), brier("S+"), allrows),
        "C7": read(stp("U*"), stp("U-N"), allrows),
    }
    hb = None
    if arm("U*").p is not None and arm("S").p is not None:
        hb = harm_hb(rows=rows, p_u=arm("U*").p, p_s=arm("S").p, n_resamples=n_resamples)  # ty: ignore[invalid-argument-type]
    mirror, unmirrored = (
        np.isin(tr.win_case, MIRRORED_CASES),
        np.isin(tr.win_case, UNMIRRORED_CASES),
    )
    report: dict[str, score729.ContrastReading | None] = {
        "R1": read(brier("U-SG"), brier("S+"), allrows),
        "R2": read(brier("U*"), brier("U-SG"), allrows),
        "R3": read(brier("U*-L"), brier("S-L"), allrows),
        "R4": read(brier("S+"), brier("S"), allrows),
        "R5": read(brier("U*"), brier("U-NC"), allrows),
        "R6": read(sol("U*"), sol("B-PV1"), up),
        "R7": read(sol("U*-L"), sol("U*"), up),
        "R8": read(brier_tr("U*"), brier_tr("S+"), allrows),
        "R9": read(viol("U*"), viol("U-NC"), truth.raisable),
        "R10": read(sol("U*"), sol("U-NCw"), up),
        "R11": read_inj("U*", "U-NCw"),
        "R12_solar_only": read(sol("U*"), sol("B-PV2"), up & subsets["solar_only"]),
        "R12_no_injection": read(sol("U*"), sol("B-PV2"), up & subsets["no_injection"]),
        "R13": read(brier("U*"), brier("U-N"), allrows),
        "R14_mirror": read(brier_tr("U*"), brier_tr("U-N"), mirror),
        "R14_no_mirror": read(brier_tr("U*"), brier_tr("U-N"), unmirrored),
        "R15": read(stp("U-N"), stp("B-T"), allrows),
        "R19": read(brier("U*"), brier("U-noBSP"), bsp_keep, months=bsp_months),
        "R21": read(
            None if arm("U*").p_nobsp is None else score730.row_brier(p=arm("U*").p_nobsp, x=rows.x),
            brier("U*"), bsp_keep, months=bsp_months,
        ),
        "R23": read(sol("U*"), sol("B-PV0"), up & solar_world(rows=rows)),
    }  # fmt: skip
    for name, mask in split_by_case(case=tr.win_case).items():
        report[f"R16_{name}"] = read(stp("U*"), stp("U-N"), mask)
        report[f"R22_{name}"] = read(stp("U*"), stp("U-noBSP"), mask)
    for name, mask in split_by_shape(shape=tr.win_shape).items():
        report[f"R17_C5_{name}"] = read(stp("U*"), stp("B-TN"), mask)
        report[f"R17_C7_{name}"] = read(stp("U*"), stp("U-N"), mask)
    world = np.asarray(rows.world)
    for name in ("engine", "solar", "both"):
        report[f"R20_{name}"] = read(
            brier("U*"), brier("U-noBSP"),
            bsp_keep & (world == _world_codes(names=(name,))[0]), months=bsp_months,
        )  # fmt: skip
    return {"counted": counted, "hb": hb, "report": report}


def recall_gap_r18(
    *, steps: Mapping[str, np.ndarray], event: np.ndarray, size: np.ndarray, tr: TransferRows
) -> dict[str, dict[str, float]]:
    """Return R18: each rule's recall on solar-shaped transfers minus its recall on constant transfers.

    The transfers counted are steps of 5 points and more.

    Args:
        steps: The step probability of U*, U-N, B-TN, and B-T on the transfer overlay (flat).
        event: The step event.
        size: The signed step size.
        tr: The transfer overlay's row attributes.

    Returns:
        For each rule, its recall on solar-shaped transfers, its recall on constant transfers, and
        the gap between them.
    """
    one = {"e": event[None, :], "s": size[None, :], "v": np.ones((1, event.size), dtype=bool)}
    out = {}
    for name, p in steps.items():
        recall = {}
        for label in ("solar", "constant"):
            reading = step_f1(
                step_p=p[None, :], event=one["e"], size=one["s"], valid=one["v"],
                select=(tr.edge_shape == SHAPE_NAMES.index(label))[None, :],
            )  # fmt: skip
            recall[label] = float(reading["recall"])
        out[name] = {**recall, "gap": recall["solar"] - recall["constant"]}
    return out


# --------------------------------------------------------------------------- the report tables


def solar_error_table(
    *, rows: score729.Rows, cosz: np.ndarray, pv_by_arm: Mapping[str, np.ndarray]
) -> dict[str, dict[str, dict[str, float]]]:
    """Return table T1: the solar error of each arm by world type and by source.

    The error is the mean absolute and root mean squared error in points, on sun-up half-hours.

    A day that fell back to source S2 is not counted as source S1.
    """
    up = sun_up(cosz=cosz)
    groups = {w: np.asarray(rows.world) == i for i, w in enumerate(core.core729.WORLD_TYPES)}
    groups["source_S1"] = (np.asarray(rows.source) == 1) & ~np.asarray(rows.fallback, dtype=bool)
    groups["source_S2"] = np.asarray(rows.source) == 2
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, pv in pv_by_arm.items():
        err, sq = solar_abs(pv=pv, truth=rows.solar), solar_sq(pv=pv, truth=rows.solar)
        out[name] = {
            group: {
                "mae": mean_on(values=err, keep=mask & up),
                "rmse": rmse_on(values_sq=sq, keep=mask & up),
                "n": float((mask & up).sum()),
            }
            for group, mask in groups.items()
        }
    return out


def transfer_solar_table(
    *,
    rows: score729.Rows,
    cosz: np.ndarray,
    tr: TransferRows,
    pv_tr_by_arm: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return T1's transfer part: the solar error by transfer.

    The rows split by whether a transfer edge lies in the window, and by inside solar-shaped
    transfers with and without a constant partner.
    """
    up = sun_up(cosz=cosz)
    inside_solar = tr.inside & (tr.win_shape == SHAPE_NAMES.index("solar"))
    groups = {
        "edge_in_window": tr.win_edge,
        "no_edge_in_window": ~tr.win_edge,
        "inside_solar_shaped_mirrored": inside_solar & np.isin(tr.win_case, MIRRORED_CASES),
        "inside_solar_shaped_unmirrored": inside_solar & np.isin(tr.win_case, UNMIRRORED_CASES),
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, pv in pv_tr_by_arm.items():
        err = solar_abs(pv=pv, truth=rows.solar)
        out[name] = {
            g: {"mae": mean_on(values=err, keep=m & up), "n": float((m & up).sum())}
            for g, m in groups.items()
        }
    return out


def scenario_table(
    *, truth: Truth, arms: Mapping[str, ArmScores]
) -> dict[str, dict[str, dict[str, float]]]:
    """Return table T2: the four load-transfer scenarios.

    Each scenario shows the Brier score of P(X<0), the solar error, and the step Brier score of each
    arm.
    """
    tr = truth.tr
    masks = transfer_scenarios(
        inside=tr.inside,
        since_end_days=tr.since_end_days,
        valid=np.ones(tr.inside.shape, dtype=bool),
    )
    up = sun_up(cosz=truth.cosz) & ~truth.exclude
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, a in arms.items():
        out[name] = {}
        for scenario, keep in masks.items():
            cell: dict[str, float] = {"n": float(keep.sum())}
            if a.p_tr is not None:
                cell["brier_sign"] = mean_on(
                    values=score730.row_brier(p=a.p_tr, x=truth.x_tr), keep=keep
                )
            if a.pv_tr is not None:
                cell["solar_mae"] = mean_on(
                    values=np.abs(a.pv_tr - truth.rows.solar), keep=keep & up
                )
            if a.step is not None:
                cell["brier_step"] = mean_on(values=(a.step - truth.event) ** 2, keep=keep)
            out[name][scenario] = cell
    return out


def shifted_table(
    *, rows: score729.Rows, p_by_arm: Mapping[str, np.ndarray], limits: Mapping[str, core.BandLimit]
) -> dict[str, dict[str, object]]:
    """Return table T4 for the shifted-mix bank (no bar).

    The table holds the Brier score, the reliability term, and H1 of each arm.
    """
    out: dict[str, dict[str, object]] = {}
    for name, p in p_by_arm.items():
        rel = score729.dist.reliability_term(prob=p, observed=rows.x < 0)
        out[name] = {
            "brier": score730.mean_brier(p=p, x=rows.x),
            "reliability": float(rel["reliability"]),  # ty: ignore[invalid-argument-type]
            "h1": harm_by_band(rows=rows, p=p, limits=limits),
        }
    return out


def ambiguity_table(
    *, rows: score729.Rows, p_by_arm: Mapping[str, np.ndarray]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Return table T3: experiment 730's ambiguity read-out (reading bands and bands of |X|) for each arm."""
    return {name: score729.ambiguity_readout(rows=rows, p_neg=p) for name, p in p_by_arm.items()}


def recovery_by_group(
    *,
    truth: Truth,
    pv_plain: np.ndarray,
    pv_inj: Mapping[float, np.ndarray],
    groups: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Return the recovery ratio, pooled over the three amounts, in each group of rows (table T5).

    Args:
        truth: The truth and the row attributes.
        pv_plain: The plain member's solar output.
        pv_inj: The injected member's solar output by amount.
        groups: Row masks over the plain rows: terciles of the BSP coupling, the BSP half hidden, or
            import, reverse flow, and sign change.

    Returns:
        For each group, the pooled recovery ratio `rho` and the number of half-hours `n`.
    """
    out = {}
    for label, mask in groups.items():
        num = den = 0.0
        n = 0
        for a in core.TEST_INJECTION_POINTS:
            keep = (
                injection_keep(
                    delta=truth.delta[a],
                    amount=a,
                    cosz=truth.cosz,
                    pair=truth.pair[a].astype(float),
                )
                & mask
            )
            if not finite_on(pv_inj[a], pv_plain, truth.delta[a], keep=keep):
                continue
            num += float((pv_inj[a] - pv_plain)[keep].sum())
            den += float(truth.delta[a][keep].sum())
            n += int(keep.sum())
        out[label] = {"rho": num / den if den > 0 else float("nan"), "n": float(n)}
    return out


def sign_change_groups(*, truth: Truth, amount: float) -> dict[str, np.ndarray]:
    """Return the row groups of table T5.

    The groups are the rows where the plain member imports, where it is in reverse flow, and where
    the injection changes the sign.
    """
    x, d = truth.rows.x, truth.delta[amount]
    return {
        "import": (x >= 0) & (x - d >= 0),
        "reverse_flow": x < 0,
        "sign_changes": (x >= 0) & (x - d < 0),
    }


def beta_terciles(*, seg: np.ndarray, beta_by_segment: Mapping[int, float]) -> np.ndarray:
    """Return each row's tercile (0, 1, or 2) of its segment's BSP coupling beta."""
    ids = sorted(beta_by_segment)
    edges = np.quantile([beta_by_segment[i] for i in ids], [1 / 3, 2 / 3])
    code = {i: int(np.digitize(beta_by_segment[i], edges)) for i in ids}
    return np.array([code[int(s)] for s in seg])


# --------------------------------------------------------------------------- links to experiment 730's stored predictions


def row_key_digest(*, rows: score729.Rows) -> str:
    """Return the SHA-256 of the row keys (segment, day, and half-hour) of a bank's scored rows."""
    h = hashlib.sha256()
    for a in (rows.seg, rows.date, rows.hh):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()


def stored_prediction_problems(
    *,
    arrays: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    rows: score729.Rows,
    recorded_digest: str | None,
    canary_arm: str = "S",
) -> list[str]:
    """Return why experiment 730's stored predictions do not match the rows.

    Each reason is a correctness failure (verdict row 1).

    The three links: every stored array has `core.N_TEST_ROWS` rows (and as many as the rows); the
    digest of the rows' keys equals the digest recorded before any fit; and the Brier score of the
    seed-averaged `p` of arm S reproduces experiment 730's printed `core.BRIER_CANARY` to five
    decimal places. No stored arm, no seed, and no recorded digest are problems too.

    Args:
        arrays: The stored `test.npz` arrays of each arm, one mapping per seed (read only).
        rows: The test bank's rows.
        recorded_digest: The row-key digest recorded before any fit (None when none was recorded).
        canary_arm: The arm whose Brier score is the canary.

    Returns:
        One sentence for each link that fails (an empty list when every link holds).
    """
    problems = []
    if rows.n != core.N_TEST_ROWS:
        problems.append("the rebuilt test rows are not experiment 730's row count")
    for arm, seeds in arrays.items():
        if not seeds:
            problems.append(f"no stored seed for {arm}")
        for k, a in enumerate(seeds):
            if any(v.shape[0] != core.N_TEST_ROWS or v.shape[0] != rows.n for v in a.values()):
                problems.append(f"a stored array of {arm} (seed index {k}) has the wrong row count")
    if recorded_digest is None:
        problems.append("no row-key digest was recorded")
    elif row_key_digest(rows=rows) != recorded_digest:
        problems.append("the row keys differ from the recorded row-key digest")
    canary = arrays.get(canary_arm)
    if not canary:
        problems.append("no stored arm for the Brier canary")
    elif all(a["p"].shape[0] == rows.n for a in canary):
        p = score730.seed_mean(p_by_seed=[a["p"].astype(np.float64) for a in canary])
        got = round(score730.mean_brier(p=p, x=rows.x), core.BRIER_CANARY_DECIMALS)
        if got != round(core.BRIER_CANARY, core.BRIER_CANARY_DECIMALS):
            problems.append("the Brier canary of arm S does not reproduce")
    return problems


def overlay_reachability(
    *, n_transfer_steps: int, windows_by_amount: Mapping[float, int]
) -> dict[str, bool]:
    """Return whether the test overlay holds enough transfer steps and injected windows.

    A shortfall voids words and never stops the line.
    """
    return {
        "transfer_reachable": n_transfer_steps >= core.MIN_TRANSFER_STEPS_TEST,
        "injection_reachable": set(windows_by_amount) == set(core.TEST_INJECTION_POINTS)
        and all(n >= core.MIN_WINDOWS_PER_AMOUNT for n in windows_by_amount.values()),
    }


# --------------------------------------------------------------------------- the verdict file


@dataclass(frozen=True)
class VerdictFacts:
    """Everything the verdict file reads.

    Attributes:
        margin: The margin m of experiment 730.
        h1: U*'s harm readings by band (`harm_by_band`); a missing or empty band fails the run (row 2).
        h2: U*'s H2 reading on the real held-out pre-engine days.
        hb: HB's reading.
        c: The counted contrasts C1 to C7 (None where void).
        r4: R4's reading.
        r19: R19's reading.
        solar_bars: The eight solar bars' pass flags.
        injection_reachable: The test overlay holds enough injected windows at every amount.
        transfer_reachable: The test overlay holds enough transfer steps.
        f1_big: U*'s F1 at 0.5 on transfer steps of 5 points and more.
        n_bsp_reverse: Reverse-flow half-hours in the two BSP test months.
        stopped: Why the line stopped (verdict row 0), or None.
        invalid: Which correctness check failed (verdict row 1), or None.
        silent_test_missed: A known-answer test of a silenced component did not fail (row 1).
        failed: Why a counted fit raised and was not rerun (row 2), or None.
    """

    margin: float
    h1: Mapping[str, Mapping[str, float | bool]]
    h2: Mapping[str, float | bool] | None
    hb: score729.ContrastReading | None
    c: Mapping[str, score729.ContrastReading | None]
    r4: score729.ContrastReading | None
    r19: score729.ContrastReading | None
    solar_bars: Mapping[str, bool]
    injection_reachable: bool
    transfer_reachable: bool
    f1_big: float
    n_bsp_reverse: int
    stopped: str | None = None
    invalid: str | None = None
    silent_test_missed: bool = False
    failed: str | None = None


def unread_checks(*, facts: VerdictFacts) -> list[str]:
    """Return the checks that the row reads and that cannot be read.

    The checks are C1, H1 by band, and H2. Each makes the run `failed`, never a silent row 6.
    """
    unread = []
    if not finite_reading(reading=facts.c.get("C1")):
        unread.append("C1")
    unread.extend(
        f"H1 {band}"
        for band in ("pooled", "3MVA+", "1-3MVA")
        if band not in facts.h1 or int(facts.h1[band].get("n", 0)) == 0
    )
    if facts.h2 is None or int(facts.h2.get("n", 0)) == 0:
        unread.append("H2")
    return unread


def build_verdict(*, facts: VerdictFacts) -> dict[str, object]:
    """Return the content of `verdict.json`.

    The content is the first matching row of the verdict table, the four reading words, and the
    readings printed beside it.

    The row reads only C1, the margin, H1, H2, HB, and the stop, invalid, and failed flags. The
    solar, transfer, neighbour, and BSP words read their own readings and never change a row. A C1
    or a harm check that cannot be read is row 2 (`failed`), never a silent row 6.
    """
    c1 = facts.c.get("C1")
    unread = unread_checks(facts=facts)
    hb_lower = (
        float(facts.hb.reading.lower)
        if facts.hb is not None and finite_reading(reading=facts.hb)
        else float("nan")
    )
    band_status = (
        "unread"
        if "H1 1-3MVA" in unread
        else band_boundary(
            band=facts.h1["1-3MVA"], b13_lower=hb_lower if math.isfinite(hb_lower) else 0.0
        )
    )
    c1_reading = (
        c1.reading
        if c1 is not None and finite_reading(reading=c1)
        else core.core729.Reading(0.0, 0.0, 0.0)
    )
    inputs = core.VerdictInputs(
        stopped=facts.stopped is not None,
        invalid=facts.invalid is not None,
        silent_test_missed=facts.silent_test_missed,
        failed=facts.failed is not None or bool(unread),
        h1_pooled_pass=bool(facts.h1.get("pooled", {}).get("pass", False)),
        h1_high_pass=bool(facts.h1.get("3MVA+", {}).get("pass", False)),
        h2_pass=bool(facts.h2 is not None and facts.h2.get("pass", False)),
        band_fail=band_status == "fail",
        c1=c1_reading,
        margin=facts.margin,
    )
    row = core.verdict_row(v=inputs)
    word = core.VERDICT_WORDS[row]
    c5, c6, c7 = facts.c.get("C5"), facts.c.get("C6"), facts.c.get("C7")
    words = {
        "solar": core.solar_word(bars=facts.solar_bars, reachable=facts.injection_reachable),
        "transfer": core.transfer_word(
            c5=c5.reading if c5 is not None else None,
            f1=facts.f1_big,
            reachable=facts.transfer_reachable,
        ),
        "neighbour": core.neighbour_word(
            c7=c7.reading if c7 is not None else None, reachable=facts.transfer_reachable
        ),
        "bsp": core.bsp_word(
            r19=facts.r19.reading if facts.r19 is not None else None,
            n_reverse_half_hours=facts.n_bsp_reverse,
        ),
    }
    out: dict[str, object] = {
        "row": row,
        "verdict": word,
        "front_matter_verdict": core.front_matter_word(word=word),
        "reason": _reason(row=row, facts=facts, inputs=inputs, unread=unread),
        "band_1_3_mva": band_status,
        "words": words,
        "c1": reading_dict(reading=c1),
        "c1_word": core.reading_word(reading=c1.reading if c1 is not None else None),
        "ni_note": core.ni_note(c1=c1_reading, margin=facts.margin) if row in (4, 5, 6) else None,
    }
    if row >= 3:
        r4 = facts.r4
        out["c6"] = reading_dict(reading=c6)
        out["r4"] = reading_dict(reading=r4)
        both = finite_reading(reading=c6) and finite_reading(reading=r4)
        out["sign_change_carrier"] = (
            core.sign_change_carrier(c6=c6.reading, r4=r4.reading) if both else "unread"  # ty: ignore[unresolved-attribute]
        )
    return out


def _reason(
    *, row: int, facts: VerdictFacts, inputs: core.VerdictInputs, unread: Sequence[str]
) -> str | None:
    """Return the sentence that names why a row 0, 1, 2, or 3 applies (None for rows 4 to 6)."""
    if row == 0:
        return facts.stopped
    if row == 1:
        return facts.invalid or "a known-answer test of a silenced component did not fail"
    if row == 2:
        return facts.failed or ("cannot be read: " + ", ".join(unread))
    if row == 3:
        return core.harm_row_fails(v=inputs)
    return None

# ruff: noqa: E501
"""The known-answer gate, run after the screen fits and before any screen score is read.

Three checks must pass:

1. An oracle that adds the true shift to `M1`'s median returns a follow share of at least 0.95
   through the scorer, at a mid-day onset as well as at midnight.
2. LP adds at least 0.75 of the depth of a shift that is in force and far from its end, to the median
   at lead days 1 and 2 (zero-based), beyond `M1`'s median.
3. The depth head's median depth divided by the true depth is at least 0.7 on shifts of 3 to 10 days.

A first failure sets the remedy flag (cap 120, patience 15 for every head fit) and the 4 screen LP
fits run again once. A second failure exits with code 5 and nothing else may run. No worker waives it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import e00738_common as common
import e00738_score as score
import e00738_world as world
import numpy as np

ORACLE_MIN: Final[float] = 0.95
LP_FOLLOW_MIN: Final[float] = 0.75
DEPTH_RATIO_MIN: Final[float] = 0.7
REMEDY_FLAG: Final[str] = "gate_remedy.flag"
GATE_FILE: Final[str] = "gate_passed.json"
DEPTH_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("3-5", 3.0, 6.0),
    ("6-10", 6.0, 11.0),
    ("11-15", 11.0, 16.0),
)


def oracle_follow_synthetic(*, onset_half_hour_of_day: int, ramp: int = 12) -> float:
    """Return the follow share of an oracle for a shift that starts at the given half-hour of the day.

    The oracle's median is `M1`'s plus the true shift at each valid time, so the follow share must be 1
    whatever the onset's time of day. A divisor that started the day at midnight would not return 1.

    Args:
        onset_half_hour_of_day: 0 for midnight, 24 for 12:00 UTC.
        ramp: The ramp length in half-hours.

    Returns:
        The follow share.
    """
    rng = np.random.default_rng(0)
    shift = world.Shift(
        primary=0,
        kind="constant",
        onset_idx=48 * 3 + onset_half_hour_of_day,
        ramp_half_hours=ramp,
        end_idx=48 * 20,
        drawn_days=17.0,
        depth=0.2,
        sign=1,
        family="train",
    )
    valid = np.arange(48 * 2, 48 * 12)
    delta = world.true_delta_at(shift=shift, valid_idx=valid)
    base = rng.normal(size=len(valid))
    return score.follow_share(median_arm=base + delta, median_ref=base, delta=delta)


def decide(*, passed: bool, remedy_done: bool) -> str:
    """Return what the gate does next: `pass`, `remedy` (the first failure), or `stop` (the second).

    Args:
        passed: Whether all three checks passed.
        remedy_done: Whether the remedy fits already ran.

    Returns:
        One of `pass`, `remedy`, `stop`.
    """
    if passed:
        return "pass"
    return "stop" if remedy_done else "remedy"


def depth_ratio_table(
    *, depth_share: np.ndarray, true_depth: np.ndarray, age: np.ndarray, seen: np.ndarray
) -> dict[str, float]:
    """Return the median of the head's depth over the true depth, by band of true age.

    Args:
        depth_share: The head's signed depth over capacity at each in-force sample.
        true_depth: The shift's signed depth over capacity at each sample.
        age: The true age in days at each sample.
        seen: Whether A1 sees the shift (its age is above zero).

    Returns:
        The median ratio per band (NaN for an empty band), plus `3-10` for the gate.
    """
    out: dict[str, float] = {}
    ratio = depth_share / true_depth
    for name, lo, hi in DEPTH_BANDS:
        sel = seen & (age >= lo) & (age < hi)
        out[name] = float(np.median(ratio[sel])) if sel.any() else float("nan")
    sel = seen & (age >= 3.0) & (age < 11.0)
    out["3-10"] = float(np.median(ratio[sel])) if sel.any() else float("nan")
    return out


def evaluate(*, oracle: dict[str, float], lp_follow: float, depth_ratio: float) -> dict:
    """Return the gate record from the three readings.

    Args:
        oracle: The oracle follow share by onset (`midday` and `midnight`).
        lp_follow: LP's pooled follow share at lead days 1 and 2 on far-stratum rows.
        depth_ratio: The median depth ratio on shifts of 3 to 10 days.

    Returns:
        The record, with `passed` true only when all three checks hold.
    """
    checks = {
        "oracle": {"values": oracle, "passed": all(v >= ORACLE_MIN for v in oracle.values())},
        "lp_follow": {"value": lp_follow, "passed": bool(lp_follow >= LP_FOLLOW_MIN)},
        "depth_ratio": {"value": depth_ratio, "passed": bool(depth_ratio >= DEPTH_RATIO_MIN)},
    }
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values())}


def write_gate(*, record: dict, remedy_done: bool, code_hash: str, out: Path | None = None) -> str:
    """Write the gate file and the remedy flag as the decision requires, and return the decision.

    Args:
        record: The record from `evaluate`.
        remedy_done: Whether the remedy fits already ran.
        code_hash: The hash of the code that ran the gate.
        out: The output folder (`OUT` by default).

    Returns:
        `pass`, `remedy`, or `stop`.
    """
    root = out if out is not None else common.OUT
    decision = decide(passed=bool(record["passed"]), remedy_done=remedy_done)
    common.atomic_json(
        obj={**record, "decision": decision, "remedy_done": remedy_done, "code_hash": code_hash},
        path=root / ("gate_last_run.json" if decision != "pass" else GATE_FILE),
    )
    if decision == "remedy":
        (root / REMEDY_FLAG).write_text("cap 120, patience 15\n")
    return decision


def remedy_active(*, out: Path | None = None) -> bool:
    """Return whether the gate set the remedy flag, which every later head fit then follows."""
    return ((out if out is not None else common.OUT) / REMEDY_FLAG).exists()

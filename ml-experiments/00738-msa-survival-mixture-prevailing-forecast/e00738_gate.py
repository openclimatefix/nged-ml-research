# ruff: noqa: E501
"""The known-answer gate, run after the screen fits and before any screen score is read.

Three checks must pass:

1. An oracle that adds the true shift to `M1`'s median returns a follow share of at least 0.95
   through the scorer, at a mid-day onset as well as at midnight.
2. LP adds at least 0.75 of the depth of a shift that is in force and far from its end, to the median
   at lead days 1 and 2 (zero-based), beyond `M1`'s median, on the rows where A1 sees the shift
   (A1's age is above zero). The share over all far rows is recorded beside it as a diagnostic,
   because A1 cannot see a shift older than its 14-day history and LP adds nothing there.
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
import polars as pl

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


def evaluate(
    *,
    oracle: dict[str, float],
    lp_follow: float,
    depth_ratio: float,
    lp_follow_all_far: float | None = None,
    diagnostics: dict[str, object] | None = None,
) -> dict:
    """Return the gate record from the three readings.

    Args:
        oracle: The oracle follow share by onset (`midday` and `midnight`).
        lp_follow: LP's pooled follow share at lead days 1 and 2 on far-stratum rows where A1 sees
            the shift.
        depth_ratio: The median depth ratio on shifts of 3 to 10 days.
        lp_follow_all_far: The same share over all far-stratum rows (a diagnostic, not a check).
        diagnostics: Other readings to record, such as the share of far rows that A1 sees.

    Returns:
        The record, with `passed` true only when all three checks hold.
    """
    checks = {
        "oracle": {"values": oracle, "passed": all(v >= ORACLE_MIN for v in oracle.values())},
        "lp_follow": {"value": lp_follow, "passed": bool(lp_follow >= LP_FOLLOW_MIN)},
        "depth_ratio": {"value": depth_ratio, "passed": bool(depth_ratio >= DEPTH_RATIO_MIN)},
    }
    return {
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
        "diagnostics": {"lp_follow_all_far_rows": lp_follow_all_far, **(diagnostics or {})},
    }


def age_per_row(*, frame: pl.DataFrame, diag: pl.DataFrame) -> np.ndarray:
    """Return A1's age (days; 0 when A1 sees no shift) at each row of a forecast frame.

    Args:
        frame: A forecast frame (native-origin keys, `lead_day`).
        diag: The diagnostics frame of the same arm and panel (one row per Primary, origin, and lead).

    Returns:
        One value per frame row, in the frame's row order.
    """
    key = frame.select(
        "time_series_id",
        pl.col("power_fcst_init_time").dt.epoch("us").alias("origin_us"),
        "lead_day",
    )
    found = key.join(
        diag.select("time_series_id", "origin_us", "lead_day", "age"),
        on=["time_series_id", "origin_us", "lead_day"],
        how="left",
        maintain_order="left",
    )
    return found["age"].fill_null(0.0).to_numpy().astype(float)


def panel_readings(
    *,
    m1: pl.DataFrame,
    lp: pl.DataFrame,
    real: pl.DataFrame,
    diag: pl.DataFrame | None,
    shifts: list,
    ids: np.ndarray,
) -> dict:
    """Return the sums behind checks 2 and 3 for one scoring panel.

    Args:
        m1: `M1`'s forecast on the panel.
        lp: LP's forecast on the panel.
        real: `M1`'s forecast on the real record (its actuals are the unshifted power).
        diag: LP's diagnostics on the panel (None when the file does not exist).
        shifts: The panel's 16 shifts, indexed by the site table's `primary`.
        ids: The `time_series_id` of each Primary in `primary` order.

    Returns:
        `num` and `den` of the follow share over all far rows at lead days 1 and 2, `num_seen` and
        `den_seen` over those rows that A1 sees, the row counts, and the depth ratios of the
        3-to-10-day band (a list of at most one value).
    """
    assert m1.height == lp.height == real.height, "the panels differ in length"
    prim = score.primary_of(time_series_id=m1["time_series_id"].to_list(), ids=ids)
    origin, valid, lead = score.row_indices(frame=m1)
    masks = score.row_masks(
        prim=prim,
        origin_idx=origin,
        valid_idx=valid,
        lead_day=lead,
        onset=np.array([s.onset_idx for s in shifts]),
        end=np.array([s.end_idx for s in shifts]),
        clipped=np.array([s.clipped for s in shifts]),
    )
    sel = masks["far"] & np.isin(lead, (1, 2))
    delta = np.nan_to_num(m1["actual"].to_numpy() - real["actual"].to_numpy())
    step = lp["q50"].to_numpy() - m1["q50"].to_numpy()
    seen_row = (
        age_per_row(frame=lp, diag=diag) > 0 if diag is not None else np.zeros(m1.height, bool)
    )
    sel_seen = sel & seen_row
    out = {
        "num": float((step[sel] * delta[sel]).sum()),
        "den": float((delta[sel] ** 2).sum()),
        "num_seen": float((step[sel_seen] * delta[sel_seen]).sum()),
        "den_seen": float((delta[sel_seen] ** 2).sum()),
        "rows": int(sel.sum()),
        "rows_seen": int(sel_seen.sum()),
        "depth_ratio": [],
    }
    if diag is not None and diag.height:
        d = diag.filter(pl.col("lead_day") == 1)
        dprim = score.primary_of(time_series_id=d["time_series_id"].to_list(), ids=ids)
        c_idx = (
            d["origin_us"].to_numpy() - common.GRID_START_US
        ) // common.STEP_US + common.NATIVE_OFFSET_STEPS
        onset = np.array([s.onset_idx for s in shifts])[dprim]
        end = np.array([s.end_idx for s in shifts])[dprim]
        age = (c_idx - onset) / 48.0
        true = np.array([s.sign * s.depth for s in shifts])[dprim]
        in_force = (c_idx >= onset) & (c_idx < end)
        tab = depth_ratio_table(
            depth_share=d["depth_share"].to_numpy()[in_force],
            true_depth=true[in_force],
            age=age[in_force],
            seen=(d["age"].to_numpy() > 0)[in_force],
        )
        if not np.isnan(tab["3-10"]):
            out["depth_ratio"] = [tab["3-10"]]
    return out


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

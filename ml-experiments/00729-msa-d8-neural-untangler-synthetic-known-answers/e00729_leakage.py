# ruff: noqa: E501
"""Experiment 729: the leakage assertions. A failure raises `CorrectnessFailureError` (verdict row 1).

Each function takes the objects it checks and returns `None`; none prints a month, a date, or a
value. The messages say which rule failed and give counts only.
"""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import e00729_arms as arms
import e00729_banks as banks
import e00729_core as core
import e00729_sim as sim
import numpy as np

FORBIDDEN_IN_SIMULATOR: Final = ("engine_constants", "g2_window", "metered_record", "record_by_day")
"""Names that no simulator source may contain: the private constants file and the engine record."""
ALLOWED_INPUT_ROWS: Final = frozenset(
    {
        "r", "r_mask", "csi", "clear", "cosz", "cams_mask", "d_bsp", "bsp_mask", "local_sin",
        "local_cos", "dow_sin", "dow_cos", "holiday", "t2m",
    }
)  # fmt: skip
"""Every input row any arm may read; none is a month or a day of the year."""


class CorrectnessFailureError(ValueError):
    """A correctness check failed: the verdict is row 1 (`invalid`)."""


def _fail(*, message: str) -> None:
    raise CorrectnessFailureError(message)


def assert_split_integrity(
    *, days: Sequence[date], day_splits: Sequence[str], month_split: Mapping[str, str]
) -> None:
    """Check that no day sits in two splits and the 2-day guard margins hold.

    Args:
        days: The base days.
        day_splits: Each day's split from `core.assign_day_splits`.
        month_split: The month rule's output.
    """
    if len(set(days)) != len(days):
        _fail(message="a base day appears twice")
    for day, split in zip(days, day_splits, strict=True):
        if split == "dropped":
            continue
        if month_split.get(core.month_key(day=day)) != split:
            _fail(message="a day's split differs from its month's split")
        near = {
            month_split.get(core.month_key(day=day + timedelta(days=k)))
            for k in range(-core.DROP_MARGIN_DAYS, core.DROP_MARGIN_DAYS + 1)
        }
        if split == "training" and near & {"test", "validation"}:
            _fail(message="a training day lies within 2 days of a test or validation month")
        if split == "validation" and "test" in near:
            _fail(message="a validation day lies within 2 days of a test month")


def assert_not_in_lockdown(*, month_split: Mapping[str, str]) -> None:
    """Check that no test or validation month overlaps the fixed lockdown window."""
    for month, split in month_split.items():
        if split != "training" and core.LOCKDOWN_FIRST_MONTH <= month <= core.LOCKDOWN_LAST_MONTH:
            _fail(message="a test or validation month overlaps the lockdown window")


def assert_seed_spaces(*, train_seeds: Sequence[int], evaluation: Sequence[banks.Bank]) -> None:
    """Check that no evaluation bank uses a training seed space."""
    if not banks.seed_spaces_disjoint(train_seeds=train_seeds):
        _fail(message="a training seed space meets an evaluation seed space")
    train_roots = {banks.bank_root(kind="train", seed=s) for s in train_seeds}
    if any(b.root in train_roots for b in evaluation):
        _fail(message="an evaluation bank was built from a training root")
    if any(b.kind == "train" for b in evaluation):
        _fail(message="a training bank was listed as an evaluation bank")


def assert_hashes_match(*, digests_by_arm: Mapping[str, Mapping[str, str]]) -> None:
    """Check that every arm and baseline read identical evaluation banks (hash by bank kind)."""
    reference: Mapping[str, str] | None = None
    for digests in digests_by_arm.values():
        if reference is None:
            reference = digests
        elif dict(digests) != dict(reference):
            _fail(message="two arms read evaluation banks with different hashes")


def assert_cams_differs(*, cal: sim.Calendar) -> None:
    """Check that the CAMS irradiance at every farm differs from the series at D8."""
    for f in range(cal.farms.ghi.shape[0]):
        both = (cal.farms.ghi[f] > 0) & np.isfinite(cal.d8_ghi) & (cal.d8_ghi > 0)
        if both.sum() == 0:
            continue
        if float(np.max(np.abs(cal.farms.ghi[f][both] - cal.d8_ghi[both]))) <= 0.0:
            _fail(message="a farm's CAMS series equals the CAMS series at D8")


def assert_s1_same_day(*, seg: sim.Segment, cal: sim.Calendar) -> None:
    """Check that source S1 reads a farm's output on the scored base day only.

    On each non-fallback day of an S1 segment, the solar row over the segment's capacity must equal
    some farm's normalised output on that same calendar-axis day (night half-hours zeroed).
    """
    if seg.source != "S1":
        return
    cap = seg.params["solar_capacity_points"]
    for d in seg.base_days:
        if seg.s1_fallback[d]:
            continue
        row = seg.solar[d] / cap
        day = int(seg.cal_index[d])
        matches = False
        for f in range(cal.farms.power.shape[0]):
            if not cal.farms.power_ok[f, day]:
                continue
            shape = np.nan_to_num(cal.farms.power[f, day])
            night = cal.farms.zenith[f, day] >= core.SUN_UP_ZENITH_DEG
            ref = np.clip(np.where(night, 0.0, shape), 0.0, 1.2)
            if np.allclose(row, ref, atol=1e-4):
                matches = True
                break
        if not matches:
            _fail(message="an S1 day does not match any farm's output on the same base day")


def assert_bsp_construction(*, seg: sim.Segment, cal: sim.Calendar) -> None:
    """Check that the synthetic BSP is the same base day's real meter minus beta times the injection."""
    beta = seg.params["beta"]
    real = sim.to_dense(calendar_values=cal.bsp, cal_index=seg.cal_index)
    expected = real - beta * (seg.engine + seg.solar)
    both = np.isfinite(expected) & np.isfinite(seg.bsp)
    if not np.allclose(seg.bsp[both], expected[both], atol=1e-3):
        _fail(message="a synthetic BSP is not the real meter minus beta times the injection")
    if np.isfinite(seg.bsp).sum() != np.isfinite(expected).sum():
        _fail(message="a synthetic BSP has data where the real meter has none")


def assert_parameter_ranges(*, segments: Sequence[sim.Segment]) -> None:
    """Check that every drawn simulator parameter lies inside its range on the page."""
    for seg in segments:
        try:
            sim.assert_in_ranges(params=seg.params)
        except AssertionError:
            _fail(message="a simulator parameter lies outside its range")


def assert_simulator_source_clean(*, paths: Sequence[Path]) -> None:
    """Check that no simulator source names the metered engine record or the private constants."""
    for path in paths:
        text = path.read_text()
        if any(bad in text for bad in FORBIDDEN_IN_SIMULATOR):
            _fail(message=f"{path.name} names the engine record or the private constants file")


def assert_cut_applied(*, base_dates: np.ndarray, cut_day: np.datetime64) -> None:
    """Check that every base day lies before the private cut (the cut itself is never printed)."""
    if len(base_dates) and base_dates.max() >= cut_day:
        _fail(message="a base day lies on or after the private cut")


def assert_record_not_base(*, record_days: np.ndarray, base_dates: np.ndarray) -> None:
    """Check that no day of the metered engine record is a base day."""
    if np.isin(record_days, base_dates).any():
        _fail(message="a day of the metered engine record is a base day")


def assert_before_holdout(*, dates: np.ndarray) -> None:
    """Check that no valid time on or after 2026-07-01 is read."""
    if len(dates) and dates.max() >= np.datetime64(core.HOLDOUT_START_ISO, "D"):
        _fail(message="a date on or after 2026-07-01 was read")


def assert_no_calendar_memorisation(*, specs: Mapping[str, arms.ArmSpec]) -> None:
    """Check that no arm reads the month or the day of the year."""
    for spec in specs.values():
        names = set(arms.stack_names(spec=spec))
        if not names <= ALLOWED_INPUT_ROWS or any("month" in n or "doy" in n for n in names):
            _fail(message="an arm reads a row that could carry the month or the day of the year")


def assert_choices_before_test(*, events: Sequence[str]) -> None:
    """Check that the architecture, B*, and m were fixed before the first test score.

    Args:
        events: The names of events in the order they happened: "architecture", "bstar", "margin",
            and "test_score" (the first test score read).
    """
    if "test_score" not in events:
        return
    first_test = events.index("test_score")
    for needed in ("architecture", "bstar", "margin"):
        if needed not in events[:first_test]:
            _fail(message=f"{needed} was not fixed before the first test score")

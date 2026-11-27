# ruff: noqa: E501
"""Experiment 733: the synthetic forecasting world, as pure index arithmetic on per-segment arrays.

A forecasting sample is (segment, origin day D, lead day k). It has the layout of experiment 713:
the
contract origin is 12:00 UTC on dense day D (`c = 48 D + 24`, a stamp index), history row `a` holds
the
stamps `c - 48 a + j` for ages 2 to 15, and the target row of lead day `k` holds `c + 48 (k - 1) +
j`,
with the stamp `g` on slot `g - 1` of the segment's dense axis. A history half-hour is visible only
when its stamp is at most `c - 54`, as in `e00733_channels.visible_history_mask`.

This module holds no experiment-730 bank and no weather product. The real adapters build one
`SegmentArrays` per segment (the folded reading, the true sign, the untangled channels, the history
weather, and the ensemble rows of the base date) and this module cuts samples and batches from them.
The ensemble coverage of the base dates is a result, never an error: a base date without a 00Z run
of
the ensemble has the status `coverage_gap`, and the stage shrinks the world by the plan's rule
(segments with no covered origin are left out; more than half of the test segments left out stops
the line at verdict row 0).

No function prints a date. Every function that touches a real base date returns counts or arrays.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00733_core as core
import e00733_leakage as leak
import e00733_paths as paths  # noqa: F401
import numpy as np
from e00733_channels import StoppedError
from e00733_leakage import CorrectnessFailureError

SUMMARY_FILE: Final = re.compile(r"^init=(\d{4}-\d{2}-\d{2})\.parquet$")
"""The name of a run summary file."""
ENS_MODEL_ID: Final = "ECMWF_ENS_0_25_degree"
"""The `nwp_model_id` partition of the `NWP` Delta table that holds the ensemble runs."""
RUN_PARTITION: Final = re.compile(r"init_time=(\d{4}-\d{2}-\d{2})T00(?:%3A|:)00")
"""A 00Z partition folder of the `NWP` table (the colon is written `%3A` on disk)."""
STATUS_OK: Final = "ok"
STATUS_GAP: Final = "coverage_gap"
N_LEAD: Final = 14
N_WEATHER_COLUMNS: Final = 21
N_SPREAD_COLUMNS: Final = 7
STEPS: Final = core.STEPS_PER_DAY
TOP_ROWS: Final = 14
SCORED_K0: Final = tuple(d - 1 for d in core.SYNTHETIC_LEAD_DAYS)
"""The scored lead days 2 to 10 as zero-based lead indices."""


class SynthError(CorrectnessFailureError):
    """A check of the synthetic world failed (a correctness failure, verdict row 1)."""


# --------------------------------------------------------------------------- ensemble coverage


def nwp_run_days(*, nwp_root: Path, model_id: str = ENS_MODEL_ID) -> frozenset[np.datetime64]:
    """Return the days that have a 00Z run in the `NWP` table.

    Read from the partition folder names (read only).

    Args:
        nwp_root: The folder of the `NWP` Delta table.
        model_id: The `nwp_model_id` partition.

    Returns:
        The days (`datetime64[D]`).

    Raises:
        SynthError: When the partition folder is missing or holds no 00Z partition.
    """
    folder = nwp_root / f"nwp_model_id={model_id}"
    if not folder.is_dir():
        msg = "the NWP table has no partition folder for the ensemble"
        raise SynthError(msg)
    days = set()
    for entry in folder.iterdir():
        match = RUN_PARTITION.match(entry.name)
        if match is not None:
            days.add(np.datetime64(match.group(1), "D"))
    if not days:
        msg = "the NWP table holds no 00Z run of the ensemble"
        raise SynthError(msg)
    return frozenset(days)


def summary_run_days(*, summary_dirs: Sequence[Path]) -> frozenset[np.datetime64]:
    """Return the days that have a stored run summary.

    (`init=<date>.parquet`) in any of the folders (read only).

    The synthetic world's target rows read these summaries, which experiment 607 holds from
    2024-04-01 and experiment 707 holds from 2020-10-15 to 2024-03-31 (built from the `NWP` table).
    A base date without a summary is a `coverage_gap`; nothing is downloaded or built.

    Args:
        summary_dirs: The folders searched.

    Returns:
        The days (`datetime64[D]`).

    Raises:
        SynthError: When no folder holds a summary.
    """
    days = set()
    for folder in summary_dirs:
        if not folder.is_dir():
            continue
        for entry in folder.iterdir():
            match = SUMMARY_FILE.match(entry.name)
            if match is not None:
                days.add(np.datetime64(match.group(1), "D"))
    if not days:
        msg = "no folder of run summaries holds a file"
        raise SynthError(msg)
    return frozenset(days)


@dataclass(frozen=True)
class CoverageResult:
    """The ensemble coverage of one split's base dates.

    Attributes:
        covered: Per segment, a bool array over its base dates: whether the 00Z run of the base
            date exists.
        n_segments: The number of segments.
        n_left_out: The segments with no covered base date (left out of the world).
        n_fully_covered: The segments whose every base date is covered.
        n_base_dates: The base dates of all segments.
        n_covered: The base dates with a run.
    """

    covered: tuple[np.ndarray, ...]
    n_segments: int
    n_left_out: int
    n_fully_covered: int
    n_base_dates: int
    n_covered: int

    @property
    def left_out_share(self) -> float:
        """Return the share of segments left out."""
        return self.n_left_out / self.n_segments if self.n_segments else float("nan")

    @property
    def covered_share(self) -> float:
        """Return the share of base dates with a run."""
        return self.n_covered / self.n_base_dates if self.n_base_dates else float("nan")

    def statuses(self, *, segment: int) -> list[str]:
        """Return the `coverage_gap` result of each base date of one segment.

        "ok" or "coverage_gap".
        """
        return [STATUS_OK if c else STATUS_GAP for c in self.covered[segment]]

    def counts(self) -> dict[str, int | float]:
        """Return the counts (no date) that the private log and `limits.json` hold."""
        return {
            "n_segments": self.n_segments, "n_left_out": self.n_left_out,
            "n_fully_covered": self.n_fully_covered, "n_base_dates": self.n_base_dates,
            "n_covered": self.n_covered, "n_gap_base_dates": self.n_base_dates - self.n_covered,
        }  # fmt: skip


def ens_coverage(
    *, base_dates: Sequence[np.ndarray], run_days: Collection[np.datetime64]
) -> CoverageResult:
    """Return the coverage of each segment's base dates by the days that have a 00Z run.

    Args:
        base_dates: One `datetime64[D]` array per segment: the segment's base days.
        run_days: The days with a 00Z run.

    Returns:
        The result. A base date without a run is a `coverage_gap` in `statuses`; nothing is raised
        and
        nothing is downloaded.
    """
    runs = np.array(sorted(run_days), dtype="datetime64[D]")
    covered = tuple(np.isin(np.asarray(d).astype("datetime64[D]"), runs) for d in base_dates)
    return CoverageResult(
        covered=covered,
        n_segments=len(covered),
        n_left_out=sum(1 for c in covered if not c.any()),
        n_fully_covered=sum(1 for c in covered if len(c) and c.all()),
        n_base_dates=sum(len(c) for c in covered),
        n_covered=sum(int(c.sum()) for c in covered),
    )


def assert_enough_test_segments(*, result: CoverageResult) -> None:
    """Stop the line (verdict row 0) when more than half of the test segments are left out.

    Args:
        result: The coverage of the synthetic base dates by the ensemble runs.

    Raises:
        StoppedError: When the test split holds no segment or the left-out share is above the limit.
    """
    if result.n_segments == 0:
        msg = "the synthetic test split holds no segment"
        raise StoppedError(msg)
    if result.left_out_share > core.MAX_MISSING_TEST_SEGMENT_SHARE:
        msg = "more than half of the synthetic test segments lack ensemble runs (verdict row 0)"
        raise StoppedError(msg)


def assert_enough_test_segments_counted(*, n_segments: int, n_left_out: int) -> None:
    """Run `assert_enough_test_segments` on the counts that the world builder returns.

    Args:
        n_segments: The segments of the test bank.
        n_left_out: The segments with no covered base date.

    Raises:
        StoppedError: When the test split holds no segment or the left-out share is above the limit.
    """
    assert_enough_test_segments(
        result=CoverageResult(
            covered=(),
            n_segments=n_segments,
            n_left_out=n_left_out,
            n_fully_covered=0,
            n_base_dates=0,
            n_covered=0,
        )
    )


def kept_segments(*, result: CoverageResult) -> list[int]:
    """Return the indices of the segments that keep at least one covered origin."""
    return [i for i, c in enumerate(result.covered) if c.any()]


# --------------------------------------------------------------------------- per-segment arrays


@dataclass
class SegmentArrays:
    """One synthetic segment's arrays on its dense axis of `n_days` days.

    Attributes:
        present: `(n_days,)` whether the dense day is a real base day.
        covered: `(n_days,)` whether the base day has a 00Z ensemble run (an origin may sit there).
        reading_z: `(n_days * 48,)` the folded reading, standardised; NaN where the reading is
            invalid.
        x_neg: `(n_days * 48,)` the indicator of X < 0 (1.0 or 0.0), NaN where invalid.
        solar_points: `(n_days * 48,)` the injected solar output in points.
        u_raw: `(n_days * 48, 3)` the untangled channels `[P(X<0), solar points, step probability]`.
        u_ok: `(n_days * 48,)` whether the untangled channels exist at the half-hour.
        hist_weather: `(n_days * 48, 21)` the history weather columns (ERA5 and CAMS).
        hist_datetime: `(n_days * 48, 21)` the date-time columns.
        day_type: `(n_days * 48,)` int the day type (0 to 7).
        local_hh: `(n_days * 48,)` int the local half-hour of the day (0 to 47), the key of B-clim.
        month: `(n_days,)` int the local calendar month (1 to 12), the key of B-clim.
        top_weather: `(n_days, 14, 48, 21)` the target-row weather of the ensemble run of
            the base day, per lead day.
        top_spread: `(n_days, 14, 48, 7)` the ensemble spread columns.
        top_datetime: `(n_days, 14, 48, 21)` the date-time columns of the target rows.
        top_day_type: `(n_days, 14, 48)` int.
        u_rerun: `(n_days, 2, 48, 3)` the re-run outputs of UTC days D - 2 and D - 1 for an origin
            on
            dense day D (None: the stored outputs are used for every history day).
        u_rerun_ok: `(n_days, 2, 48)` bool, the flag of `u_rerun`.
        top_run_init_us: `(n_days,)` the initialisation time of the ensemble run that each origin's
            target rows read, from the run file's date, in microseconds (0 where uncovered).
        top_run_avail_us: `(n_days,)` the run file's own `power_fcst_init_time` column (0 where
            uncovered).
        top_other_valid_us: `(n_days,)` the valid time of the trailing-temperature, solar-gain,
            and CAMS columns copied into the target rows (0 where uncovered).
        top_c_grid: `(n_days,)` the real-calendar grid index of each origin's contract origin.
        top_filled: `(n_days, 14, 48)` bool, whether the ensemble run holds the target half-hour.
    """

    present: np.ndarray
    covered: np.ndarray
    reading_z: np.ndarray
    x_neg: np.ndarray
    solar_points: np.ndarray
    u_raw: np.ndarray
    u_ok: np.ndarray
    hist_weather: np.ndarray
    hist_datetime: np.ndarray
    day_type: np.ndarray
    local_hh: np.ndarray
    month: np.ndarray
    top_weather: np.ndarray
    top_spread: np.ndarray
    top_datetime: np.ndarray
    top_day_type: np.ndarray
    u_rerun: np.ndarray | None = None
    u_rerun_ok: np.ndarray | None = None
    top_run_init_us: np.ndarray | None = None
    top_run_avail_us: np.ndarray | None = None
    top_other_valid_us: np.ndarray | None = None
    top_c_grid: np.ndarray | None = None
    top_filled: np.ndarray | None = None

    @property
    def n_days(self) -> int:
        """Return the number of dense days."""
        return len(self.present)

    def check_shapes(self) -> None:
        """Raise when an array does not have the shape that its name promises."""
        n, s = self.n_days, self.n_days * STEPS
        want = {
            "covered": (n,), "reading_z": (s,), "x_neg": (s,), "solar_points": (s,),
            "u_raw": (s, core.N_CHANNELS), "u_ok": (s,), "hist_weather": (s, N_WEATHER_COLUMNS),
            "hist_datetime": (s, N_WEATHER_COLUMNS), "day_type": (s,), "local_hh": (s,), "month": (n,),
            "top_weather": (n, TOP_ROWS, STEPS, N_WEATHER_COLUMNS),
            "top_spread": (n, TOP_ROWS, STEPS, N_SPREAD_COLUMNS),
            "top_datetime": (n, TOP_ROWS, STEPS, N_WEATHER_COLUMNS),
            "top_day_type": (n, TOP_ROWS, STEPS),
        }  # fmt: skip
        for name in ("top_run_init_us", "top_run_avail_us", "top_other_valid_us", "top_c_grid"):
            want[name] = (n,)
        want["top_filled"] = (n, TOP_ROWS, STEPS)
        for name, shape in want.items():
            if getattr(self, name) is None:
                if name.startswith("top_") and name.endswith(("_us", "_grid", "_filled")):
                    msg = f"the segment array {name} is missing, so no origin-time check can run"
                    raise SynthError(msg)
            elif getattr(self, name).shape != shape:
                msg = f"the segment array {name} has shape {getattr(self, name).shape}, not {shape}"
                raise SynthError(msg)


def history_slots(*, origin_day: np.ndarray) -> np.ndarray:
    """Return the `(N, 14.

    48)` dense slots of the history rows of origins on dense days `origin_day`.

    The stamp is `c - 48 a + j` with `c = 48 D + 24`, and the slot is the stamp minus one.
    """
    c = np.asarray(origin_day, dtype=np.int64) * STEPS + 24
    ages = np.arange(core.MIN_AGE, core.MAX_AGE + 1)
    return c[:, None, None] - STEPS * ages[None, :, None] + np.arange(STEPS)[None, None, :] - 1


def target_slots(*, origin_day: np.ndarray, lead_index: np.ndarray) -> np.ndarray:
    """Return the `(N, 48)` dense slots of the target row of lead index `k - 1`."""
    c = np.asarray(origin_day, dtype=np.int64) * STEPS + 24
    k = np.asarray(lead_index, dtype=np.int64)
    return c[:, None] + STEPS * k[:, None] + np.arange(STEPS)[None, :] - 1


def visible_slots(*, origin_day: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Return which slots obey the 27-hour rule: the stamp (slot + 1) is at most `c - 54`."""
    c = np.asarray(origin_day, dtype=np.int64) * STEPS + 24
    limit = (c - core.NEWEST_VISIBLE_BEFORE_CONTRACT).reshape((-1,) + (1,) * (slots.ndim - 1))
    return (slots + 1) <= limit


def _slot_present(*, seg: SegmentArrays, slots: np.ndarray) -> np.ndarray:
    """Return whether each slot lies on a real base day of the segment."""
    s = np.asarray(slots)
    inside = (s >= 0) & (s < seg.n_days * STEPS)
    day = np.clip(s // STEPS, 0, seg.n_days - 1)
    return inside & seg.present[day]


def history_row_exists(*, seg: SegmentArrays, origin_day: np.ndarray) -> np.ndarray:
    """Return the `(N, 14)` mask of history rows whose 48 half-hours all lie on real base days."""
    return _slot_present(seg=seg, slots=history_slots(origin_day=origin_day)).all(axis=2)


@dataclass(frozen=True)
class SampleIndex:
    """The samples of one split: segment, origin day, and lead index (`k - 1`).

    Each `(N,)` int64.
    """

    seg: np.ndarray
    origin_day: np.ndarray
    lead_index: np.ndarray

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.seg)

    def take(self, *, mask: np.ndarray) -> SampleIndex:
        """Return the samples where `mask` holds."""
        return SampleIndex(self.seg[mask], self.origin_day[mask], self.lead_index[mask])


def enumerate_samples(*, segments: Sequence[SegmentArrays]) -> SampleIndex:
    """Return every sample of the segments that the plan keeps.

    An origin sits on a real base day with a covered ensemble run. A sample is kept when at least
    `MIN_HISTORY_ROWS` of its 14 history rows exist and its target row lies on real base days (the
    row has at least one finite reading or sign). A history row or a target row outside the segment
    is absent, never filled.

    Args:
        segments: The segments' arrays.

    Returns:
        The index, sorted by segment, origin day, and lead index.
    """
    seg_i, day_i, k_i = [], [], []
    for i, seg in enumerate(segments):
        days = np.flatnonzero(seg.present & seg.covered)
        if not len(days):
            continue
        rows = history_row_exists(seg=seg, origin_day=days).sum(axis=1)
        days = days[rows >= core.MIN_HISTORY_ROWS]
        for k in range(N_LEAD):
            t = target_slots(origin_day=days, lead_index=np.full(len(days), k))
            ok = _slot_present(seg=seg, slots=t).all(axis=1)
            d = days[ok]
            finite = np.isfinite(seg.x_neg[t[ok]]).any(axis=1)
            d = d[finite]
            seg_i.append(np.full(len(d), i, dtype=np.int64))
            day_i.append(d.astype(np.int64))
            k_i.append(np.full(len(d), k, dtype=np.int64))
    if not seg_i:
        return SampleIndex(*(np.zeros(0, dtype=np.int64) for _ in range(3)))
    out = SampleIndex(np.concatenate(seg_i), np.concatenate(day_i), np.concatenate(k_i))
    order = np.lexsort((out.lead_index, out.origin_day, out.seg))
    return SampleIndex(out.seg[order], out.origin_day[order], out.lead_index[order])


def scored(*, index: SampleIndex) -> np.ndarray:
    """Return the mask of the samples whose lead day is one of the scored lead days 2 to 10."""
    return np.isin(index.lead_index, np.asarray(SCORED_K0))


# --------------------------------------------------------------------------- batches

CHANNEL_MODES: Final = ("untangled", "absent", "true", "p_neg")
"""`untangled`: the stored and re-run outputs (YU); `absent`: no channel (Y0); `true`: the true
sign and solar output (YT, and the post hoc oracle YO); `p_neg`: the stored and re-run P(X<0) alone, with the solar and
transfer-step channels zero as an arm of the real folds with `channel_subset=("p_neg",)` has them
(YS)."""


@dataclass
class SynthBatch:
    """One batch of synthetic samples as numpy arrays, in the fields of `e00733_model.UntInputs`.

    Attributes:
        arrays: The `UntInputs` fields by name.
        target_z: `(B, 48)` the folded reading, standardised; NaN where invalid.
        sign_truth: `(B, 48)` the indicator of X < 0, NaN where unknown.
        history_x_neg: `(B, 14, 48)` the true sign of each history half-hour (the oracle's
            input; NaN where unknown).
    """

    arrays: dict[str, np.ndarray]
    target_z: np.ndarray
    sign_truth: np.ndarray
    history_x_neg: np.ndarray


@dataclass
class OriginTimes:
    """The run times that the target rows of a batch read, as the run files record them.

    Attributes:
        c_grid: `(B,)` the real-calendar grid index of each contract origin.
        target_grid: `(B, 48)` the real-calendar grid index of each target half-hour.
        filled: `(B, 48)` whether the run holds the half-hour.
        run_init_us: `(B,)` the run's initialisation time.
        run_avail_us: `(B,)` the run's availability time, read from the file's own column.
        other_valid_us: `(B,)` the valid time of the other target-row columns.
    """

    c_grid: np.ndarray
    target_grid: np.ndarray
    filled: np.ndarray
    run_init_us: np.ndarray
    run_avail_us: np.ndarray
    other_valid_us: np.ndarray

    @classmethod
    def empty(cls, *, b: int) -> OriginTimes:
        """Return a record of `b` samples with nothing read yet."""
        return cls(
            c_grid=np.zeros(b, dtype=np.int64), target_grid=np.zeros((b, STEPS), dtype=np.int64),
            filled=np.zeros((b, STEPS), dtype=bool), run_init_us=np.zeros(b, dtype=np.int64),
            run_avail_us=np.zeros(b, dtype=np.int64), other_valid_us=np.zeros(b, dtype=np.int64),
        )  # fmt: skip

    def take(self, *, n: int, seg: SegmentArrays, d: int, k: int) -> None:
        """Read sample `n`'s record out of its segment (raises when the segment holds none)."""
        if seg.top_c_grid is None or seg.top_filled is None:
            msg = "a segment holds no origin-time record, so its target rows cannot be checked"
            raise SynthError(msg)
        self.c_grid[n] = seg.top_c_grid[d]
        self.target_grid[n] = seg.top_c_grid[d] + STEPS * k + np.arange(STEPS)
        self.filled[n] = seg.top_filled[d, k]
        self.run_init_us[n] = seg.top_run_init_us[d]  # ty: ignore[not-subscriptable]
        self.run_avail_us[n] = seg.top_run_avail_us[d]  # ty: ignore[not-subscriptable]
        self.other_valid_us[n] = seg.top_other_valid_us[d]  # ty: ignore[not-subscriptable]

    def assert_origin_time_weather(self) -> int:
        """Check every target token of the batch against the run times that were read.

        Returns:
            The number of target tokens checked (0 when the batch has no covered origin).

        Raises:
            CorrectnessFailureError: When a token's run was not initialised at 00:00 UTC of the
                origin's base date, was available after the native origin, or another target-row
                column has a valid time at or after the native origin.
        """
        if not self.filled.any():
            return 0
        return leak.assert_origin_time_weather(
            c_idx=self.c_grid, run_init_us=self.run_init_us, target_idx=self.target_grid,
            filled=self.filled, other_latest_valid_us=self.other_valid_us,
            run_available_us=self.run_avail_us,
        )  # fmt: skip

    def example(self) -> str | None:
        """Return the plan's example row for the first sample with a covered origin (or None)."""
        picked = np.flatnonzero(self.filled.any(axis=1))
        if len(picked) == 0:
            return None
        i = int(picked[0])
        t = int(self.target_grid[i][self.filled[i]][0])
        return leak.example_row(
            c_idx=int(self.c_grid[i]), target_idx=t, run_init_us=int(self.run_init_us[i]),
            other_latest_valid_us=int(self.other_valid_us[i]),
            run_available_us=int(self.run_avail_us[i]),
        )  # fmt: skip


def gather_batch(
    *,
    segments: Sequence[SegmentArrays] | PackedBank,
    index: SampleIndex,
    rows: np.ndarray,
    d8_index: int,
    target_std: float,
    mode: str = "untangled",
    mask_history: tuple[np.ndarray, np.ndarray] | None = None,
) -> SynthBatch:
    """Cut the batch of samples `rows` out of the segments' arrays.

    A `PackedBank` in place of the list of segments takes the fast path, which returns the same
    arrays as the list does (`pack_segments` and the identity test say so).

    Args:
        segments: The segments' arrays, or their packed form.
        index: The split's samples.
        rows: The positions in `index` of the batch.
        d8_index: The Primary index of D8 (the forecaster's embedding of the synthetic series).
        target_std: D8's target standard deviation, the divisor of the solar channel.
        mode: One of `CHANNEL_MODES`.
        mask_history: The age-2 mask and the mask of ages 2 to 5 (each `(B,)` bool) of a training
            batch, applied to the channels only, never to the reading.

    Returns:
        The batch.

    Raises:
        SynthError: When `mode` is unknown, or a target row lies outside its segment (a gather that
            would read a half-hour of another segment).
    """
    if mode not in CHANNEL_MODES:
        msg = f"unknown channel mode {mode}"
        raise SynthError(msg)
    if isinstance(segments, PackedBank):
        return _gather_packed(
            bank=segments, index=index, rows=rows, d8_index=d8_index, target_std=target_std,
            mode=mode, mask_history=mask_history,
        )  # fmt: skip
    b = len(rows)
    out: dict[str, np.ndarray] = {
        k: np.zeros(s, dtype=dt)
        for k, s, dt in (
            ("top_weather", (b, STEPS, N_WEATHER_COLUMNS), np.float32),
            ("top_spread", (b, STEPS, N_SPREAD_COLUMNS), np.float32),
            ("top_datetime", (b, STEPS, N_WEATHER_COLUMNS), np.float32),
            ("hist_weather", (b, core.N_HISTORY_ROWS, STEPS, N_WEATHER_COLUMNS), np.float32),
            ("hist_datetime", (b, core.N_HISTORY_ROWS, STEPS, N_WEATHER_COLUMNS), np.float32),
            ("hist_power", (b, core.N_HISTORY_ROWS, STEPS), np.float32),
            ("hist_visible", (b, core.N_HISTORY_ROWS, STEPS), bool),
            ("row_exists", (b, core.N_HISTORY_ROWS), bool),
            ("day_type_top", (b, STEPS), np.int64),
            ("day_type_hist", (b, core.N_HISTORY_ROWS, STEPS), np.int64),
            ("hist_u", (b, core.N_HISTORY_ROWS, STEPS, core.N_CHANNELS), np.float32),
            ("hist_u_ok", (b, core.N_HISTORY_ROWS, STEPS), bool),
        )
    }  # fmt: skip
    target_z = np.full((b, STEPS), np.nan, dtype=np.float32)
    sign_truth = np.full((b, STEPS), np.nan, dtype=np.float32)
    hist_x = np.full((b, core.N_HISTORY_ROWS, STEPS), np.nan, dtype=np.float32)
    times = OriginTimes.empty(b=b)
    seg_of = index.seg[rows]
    for si in np.unique(seg_of):
        pos = np.flatnonzero(seg_of == si)
        _gather_segment(
            seg=segments[int(si)], index=index, rows=rows[pos], pos=pos, out=out,
            arrays=(target_z, sign_truth, hist_x), times=times, mode=mode, target_std=target_std,
        )  # fmt: skip
    times.assert_origin_time_weather()
    if mask_history is not None:
        _apply_masks(out=out, masks=mask_history)
    out["primary"] = np.full(b, d8_index, dtype=np.int64)
    out["lead_day"] = index.lead_index[rows].astype(np.int64)
    return SynthBatch(arrays=out, target_z=target_z, sign_truth=sign_truth, history_x_neg=hist_x)


def _gather_segment(
    *,
    seg: SegmentArrays,
    index: SampleIndex,
    rows: np.ndarray,
    pos: np.ndarray,
    out: dict[str, np.ndarray],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    times: OriginTimes,
    mode: str,
    target_std: float,
) -> None:
    """Write the samples `rows` of one segment into batch positions `pos`, vectorised.

    Args:
        seg: The segment that holds every sample of `rows`.
        index: The split's samples.
        rows: The positions in `index` of these samples.
        pos: The positions in the batch that receive them.
        out: The batch's arrays, written in place.
        arrays: The batch's `target_z`, `sign_truth`, and `hist_x`, written in place.
        times: The batch's origin-time record, written in place.
        mode: One of `CHANNEL_MODES`.
        target_std: D8's target standard deviation.

    Raises:
        SynthError: When a target row lies outside the segment, or a gathered array is not finite
            where the batch promises finite values.
    """
    target_z, sign_truth, hist_x = arrays
    d = index.origin_day[rows].astype(np.int64)
    k = index.lead_index[rows].astype(np.int64)
    for j, n in enumerate(pos):
        times.take(n=int(n), seg=seg, d=int(d[j]), k=int(k[j]))
    n_slots = seg.n_days * STEPS
    hs = history_slots(origin_day=d)
    inside = (hs >= 0) & (hs < n_slots)
    hs_c = np.clip(hs, 0, n_slots - 1)
    exists = _slot_present(seg=seg, slots=hs).all(axis=2)
    vis = visible_slots(origin_day=d, slots=hs) & inside & exists[:, :, None]
    reading = seg.reading_z[hs_c]
    finite = np.isfinite(reading) & vis
    out["row_exists"][pos] = exists
    out["hist_visible"][pos] = finite
    out["hist_power"][pos] = np.where(finite, reading, 0.0)
    out["hist_weather"][pos] = seg.hist_weather[hs_c] * inside[..., None]
    out["hist_datetime"][pos] = seg.hist_datetime[hs_c] * inside[..., None]
    out["day_type_hist"][pos] = seg.day_type[hs_c]
    out["top_weather"][pos] = seg.top_weather[d, k]
    out["top_spread"][pos] = seg.top_spread[d, k]
    out["top_datetime"][pos] = seg.top_datetime[d, k]
    out["day_type_top"][pos] = seg.top_day_type[d, k]
    ts = target_slots(origin_day=d, lead_index=k)
    if ts.min() < 0 or ts.max() >= n_slots:
        msg = "a target row lies outside its segment"
        raise SynthError(msg)
    target_z[pos] = seg.reading_z[ts]
    sign_truth[pos] = seg.x_neg[ts]
    hist_x[pos] = np.where(vis, seg.x_neg[hs_c], np.nan)
    u, u_ok = _channels(
        seg=seg, slots=hs_c, ok=finite, mode=mode, target_std=target_std, origin_day=d
    )
    if not np.isfinite(u).all():
        msg = "a gathered channel value is not finite"
        raise SynthError(msg)
    out["hist_u"][pos], out["hist_u_ok"][pos] = u, u_ok


# --------------------------------------------------------------------------- the packed bank

PACKED_DAY_FIELDS: Final = (
    "present", "top_weather", "top_spread", "top_datetime", "top_day_type", "top_run_init_us",
    "top_run_avail_us", "top_other_valid_us", "top_c_grid", "top_filled", "u_rerun", "u_rerun_ok",
)  # fmt: skip
"""The `SegmentArrays` fields whose first axis is the dense day axis."""
PACKED_SLOT_FIELDS: Final = (
    "reading_z", "x_neg", "solar_points", "u_raw", "u_ok", "hist_weather", "hist_datetime",
    "day_type",
)  # fmt: skip
"""The `SegmentArrays` fields whose first axis is the dense half-hour (slot) axis."""


@dataclass
class PackedBank:
    """Every segment of a split on one dense day axis and one slot axis, so a batch is one gather.

    A batch of 256 samples touches about 215 segments, so the per-segment gather spends most of its
    time in Python overhead. The packed bank holds each array of `SegmentArrays` once, with the
    segments one after another, and the batch is cut out of it in a single pass. The segments of
    the list are rebound to views of the packed arrays, so the bank adds no memory.

    Attributes:
        n_days: `(n_segments,)` the number of dense days of each segment.
        day_base: `(n_segments,)` the position of each segment's first day on the packed day axis.
        slot_base: `(n_segments,)` the position of each segment's first slot on the packed slot axis.
        present: `(total_days,)` as in `SegmentArrays`, concatenated.
        reading_z: `(total_slots,)` as in `SegmentArrays`, concatenated.
        x_neg: `(total_slots,)`.
        solar_points: `(total_slots,)`.
        u_raw: `(total_slots, 3)`.
        u_ok: `(total_slots,)`.
        hist_weather: `(total_slots, 21)`.
        hist_datetime: `(total_slots, 21)`.
        day_type: `(total_slots,)`.
        top_weather: `(total_days, 14, 48, 21)`.
        top_spread: `(total_days, 14, 48, 7)`.
        top_datetime: `(total_days, 14, 48, 21)`.
        top_day_type: `(total_days, 14, 48)`.
        top_run_init_us: `(total_days,)`.
        top_run_avail_us: `(total_days,)`.
        top_other_valid_us: `(total_days,)`.
        top_c_grid: `(total_days,)`.
        top_filled: `(total_days, 14, 48)`.
        u_rerun: `(total_days, 2, 48, 3)`, or None when no segment holds re-run outputs.
        u_rerun_ok: `(total_days, 2, 48)`, or None with `u_rerun`.
    """

    n_days: np.ndarray
    day_base: np.ndarray
    slot_base: np.ndarray
    present: np.ndarray
    reading_z: np.ndarray
    x_neg: np.ndarray
    solar_points: np.ndarray
    u_raw: np.ndarray
    u_ok: np.ndarray
    hist_weather: np.ndarray
    hist_datetime: np.ndarray
    day_type: np.ndarray
    top_weather: np.ndarray
    top_spread: np.ndarray
    top_datetime: np.ndarray
    top_day_type: np.ndarray
    top_run_init_us: np.ndarray
    top_run_avail_us: np.ndarray
    top_other_valid_us: np.ndarray
    top_c_grid: np.ndarray
    top_filled: np.ndarray
    u_rerun: np.ndarray | None = None
    u_rerun_ok: np.ndarray | None = None


def pack_segments(*, segments: Sequence[SegmentArrays]) -> PackedBank:
    """Pack the segments' arrays onto one day axis and one slot axis, and rebind the segments.

    Each field is concatenated once and each segment's array is then replaced by a view of its
    slice, so the old copy is freed and the memory stays at one copy. Values are unchanged.

    Args:
        segments: The segments of one split.

    Returns:
        The packed bank.

    Raises:
        SynthError: When a segment has no origin-time record, or only some segments hold re-run
            outputs.
    """
    if not segments:
        msg = "cannot pack zero segments"
        raise SynthError(msg)
    for seg in segments:
        seg.check_shapes()
    n_days = np.array([seg.n_days for seg in segments], dtype=np.int64)
    day_base = np.concatenate([[0], np.cumsum(n_days)[:-1]]).astype(np.int64)
    slot_base = day_base * STEPS
    rerun = {seg.u_rerun is not None and seg.u_rerun_ok is not None for seg in segments}
    if len(rerun) != 1:
        msg = "some segments hold re-run outputs and others do not, so they cannot be packed"
        raise SynthError(msg)
    fields = [*PACKED_DAY_FIELDS, *PACKED_SLOT_FIELDS]
    if rerun == {False}:
        fields = [f for f in fields if f not in ("u_rerun", "u_rerun_ok")]
    packed: dict[str, np.ndarray] = {}
    for name in fields:
        parts = [getattr(seg, name) for seg in segments]
        joined = np.concatenate(parts)
        base = day_base if name in PACKED_DAY_FIELDS else slot_base
        length = n_days if name in PACKED_DAY_FIELDS else n_days * STEPS
        del parts
        for seg, start, n in zip(segments, base, length, strict=True):
            setattr(seg, name, joined[start : start + n])
        packed[name] = joined
    return PackedBank(n_days=n_days, day_base=day_base, slot_base=slot_base, **packed)


def _masked_rows(*, table: np.ndarray, flat: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Return `table[flat]` as float32 with the rows outside the segment zeroed (one copy)."""
    rows = table[flat].astype(np.float32, copy=False)
    rows *= inside[..., None]
    return rows


def _gather_packed(
    *,
    bank: PackedBank,
    index: SampleIndex,
    rows: np.ndarray,
    d8_index: int,
    target_std: float,
    mode: str,
    mask_history: tuple[np.ndarray, np.ndarray] | None,
) -> SynthBatch:
    """Cut a batch out of a packed bank, giving the arrays that `_gather_segment` gives.

    Every position is worked out on the segment's own axis, clipped to the segment exactly as the
    per-segment gather clips it, and then moved to the packed axis by the segment's base.

    Args:
        bank: The packed bank.
        index: The split's samples.
        rows: The positions in `index` of the batch.
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation.
        mode: One of `CHANNEL_MODES`.
        mask_history: The two dropout masks of a training batch, or None.

    Returns:
        The batch.

    Raises:
        SynthError: When a target row lies outside its segment, or a gathered channel value is not
            finite.
    """
    b = len(rows)
    seg = index.seg[rows]
    d = index.origin_day[rows].astype(np.int64)
    k = index.lead_index[rows].astype(np.int64)
    n_slots = (bank.n_days[seg] * STEPS)[:, None, None]
    day0 = bank.day_base[seg]
    slot0 = bank.slot_base[seg][:, None, None]
    times = OriginTimes(
        c_grid=bank.top_c_grid[day0 + d],
        target_grid=bank.top_c_grid[day0 + d][:, None] + STEPS * k[:, None] + np.arange(STEPS),
        filled=bank.top_filled[day0 + d, k],
        run_init_us=bank.top_run_init_us[day0 + d],
        run_avail_us=bank.top_run_avail_us[day0 + d],
        other_valid_us=bank.top_other_valid_us[day0 + d],
    )
    hs = history_slots(origin_day=d)
    inside = (hs >= 0) & (hs < n_slots)
    hs_c = np.clip(hs, 0, n_slots - 1)
    flat = slot0 + hs_c
    day_of = np.clip(hs // STEPS, 0, (bank.n_days[seg] - 1)[:, None, None])
    exists = (inside & bank.present[day0[:, None, None] + day_of]).all(axis=2)
    vis = visible_slots(origin_day=d, slots=hs) & inside & exists[:, :, None]
    reading = bank.reading_z[flat]
    finite = np.isfinite(reading) & vis
    ts = target_slots(origin_day=d, lead_index=k)
    if ts.min() < 0 or (ts >= n_slots[:, :, 0]).any():
        msg = "a target row lies outside its segment"
        raise SynthError(msg)
    flat_t = bank.slot_base[seg][:, None] + ts
    u, u_ok = _channels(
        seg=bank, slots=hs_c, ok=finite, mode=mode, target_std=target_std, origin_day=d,
        flat_slots=flat, rerun_day=day0 + d,
    )  # fmt: skip
    if not np.isfinite(u).all():
        msg = "a gathered channel value is not finite"
        raise SynthError(msg)
    out: dict[str, np.ndarray] = {
        "row_exists": exists,
        "hist_visible": finite,
        "hist_power": np.where(finite, reading, 0.0).astype(np.float32, copy=False),
        "hist_weather": _masked_rows(table=bank.hist_weather, flat=flat, inside=inside),
        "hist_datetime": _masked_rows(table=bank.hist_datetime, flat=flat, inside=inside),
        "day_type_hist": bank.day_type[flat].astype(np.int64, copy=False),
        "top_weather": bank.top_weather[day0 + d, k].astype(np.float32, copy=False),
        "top_spread": bank.top_spread[day0 + d, k].astype(np.float32, copy=False),
        "top_datetime": bank.top_datetime[day0 + d, k].astype(np.float32, copy=False),
        "day_type_top": bank.top_day_type[day0 + d, k].astype(np.int64, copy=False),
        "hist_u": u,
        "hist_u_ok": u_ok,
    }
    times.assert_origin_time_weather()
    if mask_history is not None:
        _apply_masks(out=out, masks=mask_history)
    out["primary"] = np.full(b, d8_index, dtype=np.int64)
    out["lead_day"] = index.lead_index[rows].astype(np.int64)
    return SynthBatch(
        arrays=out,
        target_z=bank.reading_z[flat_t].astype(np.float32),
        sign_truth=bank.x_neg[flat_t].astype(np.float32),
        history_x_neg=np.where(vis, bank.x_neg[flat], np.nan).astype(np.float32),
    )


def _channels(
    *,
    seg: SegmentArrays | PackedBank,
    slots: np.ndarray,
    ok: np.ndarray,
    mode: str,
    target_std: float,
    origin_day: np.ndarray,
    flat_slots: np.ndarray | None = None,
    rerun_day: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the `(N, 14, 48, 3)` channels and `(N, 14, 48)` flag of `N` samples' histories.

    In the forecaster's input units. `slots` and `origin_day` are positions on each sample's own
    segment. A packed bank also passes `flat_slots`, the same slots on the packed axis, and
    `rerun_day`, the origin days on the packed day axis; both default to the segment's own.
    """
    flat = slots if flat_slots is None else flat_slots
    if mode == "absent":
        return np.zeros((*ok.shape, core.N_CHANNELS), np.float32), np.zeros_like(ok)
    if mode == "true":
        have = ok & np.isfinite(seg.x_neg[flat])
        u = np.stack(
            [
                np.nan_to_num(seg.x_neg[flat]),
                solar_unit(points=seg.solar_points[flat], target_std=target_std),
                np.zeros(slots.shape),
            ],
            axis=-1,
        )
    else:
        have = ok & seg.u_ok[flat]
        raw = seg.u_raw[flat]
        if seg.u_rerun is not None and seg.u_rerun_ok is not None:
            od = origin_day[:, None, None]
            rd = od if rerun_day is None else rerun_day[:, None, None]
            token_day = slots // STEPS
            age_days = od - token_day
            use = (age_days == 1) | (age_days == 2)
            r = np.flatnonzero(use.any(axis=(0, 2)))  # the history rows a re-run can replace
            if len(r):
                k = np.clip(2 - age_days[:, r], 0, 1)
                hh = slots[:, r] % STEPS
                raw[:, r] = np.where(use[:, r, :, None], seg.u_rerun[rd, k, hh], raw[:, r])
                have[:, r] = np.where(use[:, r], ok[:, r] & seg.u_rerun_ok[rd, k, hh], have[:, r])
        u = np.stack(
            [raw[..., 0], solar_unit(points=raw[..., 1], target_std=target_std), raw[..., 2]],
            axis=-1,
        )
        if mode == "p_neg":
            keep = np.array([name == "p_neg" for name in core.CHANNEL_NAMES])
            u = np.where(keep, u, 0.0)
    return np.where(have[..., None], u, 0.0).astype(np.float32), have


def solar_unit(*, points: np.ndarray, target_std: float) -> np.ndarray:
    """Return the solar channel `v`: points over D8's capacity and the standard deviation.

    Clipped to 0 to 10.
    """
    return np.clip(np.asarray(points) / 100.0 / target_std, 0.0, core.SOLAR_CLIP_MAX)


def _apply_masks(*, out: dict[str, np.ndarray], masks: tuple[np.ndarray, np.ndarray]) -> None:
    """Apply experiment 716's CAMS-rule mask and the mask of ages 2 to 5 to the channels.

    In place.
    """
    cams, convention = masks
    ok = out["hist_u_ok"]
    ok[:, 0] &= ~cams[:, None]
    ok[:, 1, core.CAMS_RULE_FIRST_MASKED_COLUMN :] &= ~cams[:, None]
    ok[:, :4] &= ~convention[:, None, None]
    out["hist_u"] *= ok[..., None]


def assert_no_world_leak(*, input_names: Sequence[str]) -> None:
    """Fail when an input of the synthetic forecaster is one that the plan forbids.

    The forecaster reads no world type, no engine parameter, no solar capacity, and no truth array
    except through the oracle arm's declared true channels.
    """
    forbidden = ("world", "engine", "capacity", "params", "truth", "x_neg", "solar_points")
    bad = [n for n in input_names if any(f in n for f in forbidden)]
    if bad:
        msg = f"the synthetic forecaster reads forbidden inputs: {bad}"
        raise SynthError(msg)


def share_reverse(*, index: SampleIndex, segments: Sequence[SegmentArrays]) -> tuple[float, int]:
    """Return the share of reverse-flow half-hours among the scored target rows, and their count."""
    sel = np.flatnonzero(scored(index=index))
    n_rev = n_all = 0
    for r in sel:
        seg = segments[int(index.seg[r])]
        t = target_slots(
            origin_day=index.origin_day[r : r + 1], lead_index=index.lead_index[r : r + 1]
        )[0]
        v = seg.x_neg[t]
        n_all += int(np.isfinite(v).sum())
        n_rev += int(np.nansum(v))
    return (n_rev / n_all if n_all else float("nan")), n_rev

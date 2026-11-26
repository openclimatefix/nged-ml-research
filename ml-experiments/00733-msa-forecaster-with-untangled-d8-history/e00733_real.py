# ruff: noqa: E501
"""Experiment 733.

The real-window builder that feeds `e00733_rerun.rerun_origins`, and three checks on it.

The builder turns experiment 729's day table, the three neighbours' real series, and the ensemble
summaries into the `(C, 96)` input window of one UTC day for one origin, with every input after the
visibility limit hidden. The three checks read the real files and write nothing:

- `reproduce_stored` runs the stored networks on windows with nothing hidden and compares the
  outputs with the stored real-day outputs of experiment 732;
- `poisoned_origin_run` replaces every input after the limit by NaN and by 1e6 and compares the
  re-run with the clean one;
- `ens_coverage_by_split` counts, for each split, the origins whose two ensemble runs cover every
  hidden column.

Experiment 732's code and torch are imported by the functions that need them, so the module imports
without them.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_paths as paths  # noqa: F401
import e00733_rerun as rr
import numpy as np
from e00733_channels import GRID_START_EPOCH_DAY, StoredOutputs

TOLERANCE_REPRODUCTION: Final = 1e-2
"""The largest allowed difference between a re-computed output with nothing hidden and the stored
output of a day away from the month edges. A CPU run differs from the GPU run that stored the file
by up to 3e-3 on 12 sampled days; the check is meant to catch a window built wrongly, which moves
an output by 0.1 or more."""
D8_SERIES_ID: Final = 26
"""The `time_series_id` of D8 in the ensemble summaries."""


class RealWindowError(RuntimeError):
    """The real-window builder met an input it cannot use (a correctness failure, verdict row 1)."""


@dataclass
class RealWindows:
    """The prepared real days and what the window builder needs to read them.

    Attributes:
        prepared: Experiment 732's `Prepared` of the span segment (the stack, the neighbours).
        spec: Experiment 732's arm specification of `U*`.
        first_epoch_day: The epoch day of the segment's first real day.
        arms: Experiment 732's `e00732_arms` module.
        stats: Experiment 729's `InputStats` (the standardisation of the ensemble means).
        summary_dir: The folder of the ensemble summaries, or several folders searched in order.
        series_id: D8's `time_series_id`.
        extra_ok: A function from the epoch day to a `(48,)` mask of half-hours that experiment
            713's
            reading mask also accepts (None: no further condition).
        runs: A cache of ensemble runs by epoch day.
        missing_ens_columns: How many hidden columns had no ensemble value (set to zero, as
            the hidden-span input).
    """

    prepared: Any
    spec: Any
    first_epoch_day: int
    arms: Any
    stats: Any
    summary_dir: Path | Sequence[Path]
    series_id: int
    extra_ok: Callable[[int], np.ndarray] | None = None
    runs: dict[int, rr.EnsRun] = field(default_factory=dict)
    missing_ens_columns: int = 0

    def dense(self, *, epoch_day: int) -> int:
        """Return the index of an epoch day on the segment's dense axis.

        (the first column is a padding day).
        """
        return epoch_day - self.first_epoch_day + 1

    def run(self, *, epoch_day: int) -> rr.EnsRun:
        """Return (and cache) the 00Z ensemble run of an epoch day."""
        if epoch_day not in self.runs:
            self.runs[epoch_day] = rr.read_ens_run(
                summary_dir=self.summary_dir, run_epoch_day=epoch_day, series_id=self.series_id
            )
        return self.runs[epoch_day]

    def window(self, *, origin_day: int, day: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the window of grid day `day` for an origin on grid day `origin_day`.

        And the middle's reading mask.

        Args:
            origin_day: The grid day number of the origin.
            day: The grid day number of the window.

        Returns:
            The `(C, 96)` window with every input after the limit hidden, and the `(48,)` mask of
            half-hours where D8 holds a reading (and the extra condition holds).
        """
        hidden = rr.window_hidden(origin_day=origin_day, day=day)
        dense = self.dense(epoch_day=day + GRID_START_EPOCH_DAY)
        plain = self.arms.plain_window(p=self.prepared, day=dense, spec=self.spec, hidden=None)
        reading_ok = plain[self.arms.ROW["r_mask"], rr.MID : rr.MID + core.STEPS_PER_DAY] > 0.5
        if self.extra_ok is not None:
            reading_ok = reading_ok & self.extra_ok(day + GRID_START_EPOCH_DAY)
        window = self.arms.plain_window(p=self.prepared, day=dense, spec=self.spec, hidden=hidden)
        era5 = None
        if hidden.any():
            origin_epoch = origin_day + GRID_START_EPOCH_DAY
            ens = rr.ens_for_window(
                previous=self.run(epoch_day=origin_epoch - 1),
                current=self.run(epoch_day=origin_epoch),
                origin_epoch_day=origin_epoch,
                window_stamps_us=rr.window_stamps_us(day_epoch=day + GRID_START_EPOCH_DAY),
            )
            self.missing_ens_columns += int((~np.isfinite(ens["temperature"]) & hidden).sum())
            era5 = rr.standardise_ens(
                temperature=ens["temperature"], wind=ens["wind"], precip_kg_m2_s=ens["precip"],
                t_mean=self.stats.t_mean, t_sd=self.stats.t_sd, w_mean=self.stats.w_mean, w_sd=self.stats.w_sd,
            )  # fmt: skip
        rr.hide_window(window=window, rows=self.arms.ROW, hidden=hidden, era5=era5)
        return window.astype(np.float32), reading_ok

    def build_window(
        self, origin_day: int, day: int, hidden: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `rerun_origins`'s callback.

        The window and the reading mask (the callback's `hidden` is checked, not used).
        """
        if not np.array_equal(hidden, rr.window_hidden(origin_day=origin_day, day=day)):
            msg = "the re-run's hidden columns differ from the visibility limit of the origin"
            raise RealWindowError(msg)
        return self.window(origin_day=origin_day, day=day)


def build_real_windows(
    *,
    cal: Any,
    reading_rule_use: np.ndarray,
    holidays: np.ndarray,
    stats: Any,
    medians: Mapping[str, float],
    neighbours: Mapping[str, np.ndarray] | None,
    summary_dir: Path | Sequence[Path],
    first_epoch_day: int,
    last_epoch_day: int,
    series_id: int = D8_SERIES_ID,
    extra_ok: Callable[[int], np.ndarray] | None = None,
) -> RealWindows:  # fmt: skip
    """Prepare the real days from `first_epoch_day` to `last_epoch_day` as one span segment.

    Args:
        cal: Experiment 729's calendar of the whole day table (`ctx.table.calendar`).
        reading_rule_use: Which calendar days may be used `(n,)`.
        holidays: The bank-holiday dates.
        stats: Experiment 729's `InputStats`.
        medians: The training median of each folded neighbour.
        neighbours: The neighbours' real series on the calendar axis (None: zero channels).
        summary_dir: The folder of the ensemble summaries, or several folders searched in order.
        first_epoch_day: The first real day, in days since 1970-01-01.
        last_epoch_day: The last real day.
        series_id: D8's `time_series_id`.
        extra_ok: Experiment 713's reading mask by epoch day (None: none).

    Returns:
        The builder's state.
    """
    import e00732_arms as arms
    import e00732_core as core732

    segment = rr.span_segment(
        cal=cal, first_epoch_day=first_epoch_day, last_epoch_day=last_epoch_day,
        reading=cal.reading, truth=cal.reading, use=reading_rule_use,
    )  # fmt: skip
    prepared = arms.prepare_plain(
        segments=[segment], holidays=holidays, stats=stats, medians=medians, neighbours=neighbours
    )[0]
    return RealWindows(
        prepared=prepared, spec=core732.ARMS["U*"], first_epoch_day=first_epoch_day, arms=arms,
        stats=stats, summary_dir=summary_dir, series_id=series_id, extra_ok=extra_ok,
    )  # fmt: skip


# --------------------------------------------------------------------------- checks on real files


def reproduce_stored(
    *, windows: RealWindows, nets: Sequence[Any], stored: StoredOutputs, epoch_days: Sequence[int], tolerance: float = TOLERANCE_REPRODUCTION,
) -> dict[str, float | int]:  # fmt: skip
    """Run the networks on windows with nothing hidden and compare with the stored real-day outputs.

    The comparison covers P(X<0) and the step probability (the solar output is in points, so it is
    compared relative to the stored range).

    Args:
        windows: The builder's state.
        nets: The stored networks, in evaluation mode.
        stored: Experiment 732's stored real-day outputs (seed-averaged).
        epoch_days: The days to compare.
        tolerance: The largest allowed difference of a probability.

    Returns:
        The number of rows compared, the largest difference of P(X<0), of the step probability, and
        of
        the solar output relative to the stored range.

    Raises:
        RealWindowError: When no row matches, or a difference exceeds `tolerance`.
    """
    arms = windows.arms
    built, keys = [], []
    for epoch_day in epoch_days:
        dense = windows.dense(epoch_day=epoch_day)
        built.append(
            arms.plain_window(p=windows.prepared, day=dense, spec=windows.spec, hidden=None)
        )
        keys.append(epoch_day)
    pred = rr.predict_windows(
        nets=nets,
        windows=np.stack(built).astype(np.float32),
        source=core.SOURCE_732,
        n_base=built[0].shape[0],
    ).mean(axis=0)  # (n, 3, 48)
    row = {(int(d), int(h)): i for i, (d, h) in enumerate(zip(stored.day, stored.hh, strict=True))}
    diffs: list[list[float]] = [[], [], []]
    for i, epoch_day in enumerate(keys):
        for hh in range(core.STEPS_PER_DAY):
            j = row.get((epoch_day, hh))
            if j is None:
                continue
            for c in range(3):
                diffs[c].append(abs(float(pred[i, c, hh]) - float(stored.values[j, c])))
    n = len(diffs[0])
    if n == 0:
        msg = "no recomputed half-hour matches a stored row, so nothing was reproduced"
        raise RealWindowError(msg)
    span = float(np.ptp(stored.values[:, 1])) or 1.0
    out = {
        "rows": n, "max_p_neg": max(diffs[0]), "max_step": max(diffs[2]), "max_solar_relative": max(diffs[1]) / span,
    }  # fmt: skip
    if out["max_p_neg"] > tolerance or out["max_step"] > tolerance:
        msg = f"a recomputed output with nothing hidden differs from the stored one by {max(out['max_p_neg'], out['max_step']):.2e}"
        raise RealWindowError(msg)
    return out


def poisoned_origin_run(
    *, make_windows: Callable[[Any, Mapping[str, np.ndarray] | None], RealWindows], cal: Any, neighbours: Mapping[str, np.ndarray], origin_day: int, predict: Callable[[np.ndarray], np.ndarray], n_seeds: int,
) -> int:  # fmt: skip
    """Check that poisoning every real input after the limit changes nothing.

    The check re-runs one origin.

    Args:
        make_windows: Builds the `RealWindows` from a calendar and neighbour arrays.
        cal: The calendar of the whole day table.
        neighbours: The neighbours' real series on the calendar axis.
        origin_day: The grid day number of the origin.
        predict: The windows to `(seeds, n, 3, 48)`.
        n_seeds: The number of seeds `predict` returns.

    Returns:
        The number of poison values checked.
    """
    origin_epoch = origin_day + GRID_START_EPOCH_DAY

    def run(value: float | None) -> np.ndarray:
        if value is None:
            windows = make_windows(cal, neighbours)
        else:
            poisoned = rr.poisoned_calendar(cal=cal, origin_epoch_day=origin_epoch, value=value)
            nb = rr.poisoned_neighbours(
                neighbours=neighbours, dates=cal.dates, origin_epoch_day=origin_epoch, value=value
            )
            windows = make_windows(poisoned, nb)
        outputs = rr.rerun_origins(
            origin_days=[origin_day],
            build_window=windows.build_window,
            predict=predict,
            n_seeds=n_seeds,
        )
        return outputs.values

    return rr.assert_rerun_ignores_inputs_after_limit(run=run)


def ens_coverage_by_split(
    *, origin_days: np.ndarray, splits: np.ndarray, summary_dir: Path | Sequence[Path], series_id: int = D8_SERIES_ID
) -> dict[str, dict[str, int]]:  # fmt: skip
    """Count, for each split.

    The origins whose two ensemble runs exist and cover every hidden column of both windows.

    A gap is data, not an error: the counts say how many origins have no ensemble weather.

    Args:
        origin_days: The grid day number of each origin `(n,)`.
        splits: The split of each origin `(n,)`.
        summary_dir: The folder of the ensemble summaries, or several folders searched in order.
        series_id: D8's `time_series_id`.

    Returns:
        For each split: `origins` and `covered`.
    """
    cache: dict[int, rr.EnsRun | None] = {}

    def run(epoch_day: int) -> rr.EnsRun | None:
        if epoch_day not in cache:
            try:
                cache[epoch_day] = rr.read_ens_run(
                    summary_dir=summary_dir, run_epoch_day=epoch_day, series_id=series_id
                )
            except rr.RerunError:
                cache[epoch_day] = None
        return cache[epoch_day]

    out: dict[str, dict[str, int]] = {}
    for origin_day, split in zip(origin_days, splits, strict=True):
        counts = out.setdefault(str(split), {"origins": 0, "covered": 0})
        counts["origins"] += 1
        epoch = int(origin_day) + GRID_START_EPOCH_DAY
        prev, cur = run(epoch - 1), run(epoch)
        if prev is None or cur is None:
            continue
        ok = True
        for off in rr.RERUN_OFFSETS:
            hidden = rr.window_hidden(origin_day=int(origin_day), day=int(origin_day) - off)
            got = rr.ens_for_window(
                previous=prev,
                current=cur,
                origin_epoch_day=epoch,
                window_stamps_us=rr.window_stamps_us(day_epoch=epoch - off),
            )
            ok &= bool(np.isfinite(got["temperature"][hidden]).all())
        counts["covered"] += int(ok)
    return out

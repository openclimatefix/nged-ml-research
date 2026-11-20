"""Experiment 709, arm L-stat: the alarm, the onset, the episodes, and the era ids.

The detector reads the year-on-year 28-day change of the 12 window statistics of
`e00709_stats`, scaled by a robust expanding scale. One forward pass over the days writes everything
as of the end of each day:

- **Alarm.** The largest absolute scaled change `|z|` is at least the threshold on 3 consecutive
  days. A 28-day change keeps `|z|` high for about 55 days after a step, so an alarm needs `|z|`
  to dip below the threshold since the last episode began or closed, and an open episode cannot be
  superseded before it is 56 days old. A gap of 7 days or more without readings suppresses alarms
  until both windows are full again. The threshold is tuned to at most 0.5 false alarms per
  Primary-year.
- **Onset.** One least-squares split of the last 56 daily values of the level statistic (5th,
  50th, or 95th percentile, night mean, or midday mean) with the largest `|z|`, with at least 7
  values before and 3 after the split.
- **Episodes** are `e00709_bocpd.Episode` objects. The previous segment, the change since the onset
  in the six band features, and the return closure (the 3-day mean back within 1.5 scaled units of
  the previous mean for 3 days) follow the change-point detector, so the evidence and the signal
  code treats both detectors alike. An episode closes only if the alarming statistic also has
  `|z|` below 1.5 for 3 days, because the six band features alone can read "back" on noise.
- **Seasonal reference.** A day that an alarm episode contaminates has no valid year-old reference,
  so `y(t)` uses `c(t - 728)` instead, or zero if that day is also contaminated. An episode
  contaminates the 56 days from the earlier of its onset and 28 days before its alarm (the span in
  which a 28-day change straddles the step) and, if it closed by a return, the 56 days that
  start 31 days before the closure (the return step).
- **Era ids.** `assign_era_ids` gives each confirmed era an id, reusing an earlier id when the
  changes since that era sum to within 1.5 scale units in every statistic.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import e00709_bocpd as bocpd
import e00709_stats as stats
import numpy as np

STAT_THRESHOLD_GRID: Final[tuple[float, ...]] = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0)
DEFAULT_STAT_THRESHOLD: Final[float] = 4.0
"""Used until the threshold has been tuned."""
CONSECUTIVE_DAYS: Final[int] = 3
ONSET_DAYS: Final[int] = 56
ONSET_MIN_BEFORE: Final[int] = 7
ONSET_MIN_AFTER: Final[int] = 3
CONTAMINATED_DAYS: Final[int] = 56
RETURN_Z: Final[float] = 1.5
"""An episode closes only once the alarming statistic is this close to its median."""
GAP_DAYS: Final[int] = 7
"""A run of days without readings this long suppresses alarms until both windows are full."""
RETURN_STEP_DAYS: Final[int] = 31
SCALE_REFRESH_DAYS: Final[int] = 7
"""The scale is recomputed on every 7th day and held between, which reads only older days."""
SCALE_LAG_DAYS: Final[int] = 28
RECURRENCE_SCALED: Final[float] = 1.5
PREVIOUS_WINDOW_DAYS: Final[int] = 180
MIN_PREVIOUS_DAYS: Final[int] = 14
VECTOR_DAY_OFFSET: Final[int] = stats.WINDOW_DAYS - 1
"""A change vector compares the window ending the day before the onset with the window that ends
27 days after the onset, so neither window straddles the step."""


@dataclass(frozen=True)
class StatOutput:
    """The output of the statistics detector for one Primary.

    Attributes:
        detector: The episodes and per-day view in the layout of the change-point detector:
            `delta_scaled` is the change in the six band features since the onset, and `p_short`
            holds the largest absolute scaled change of each day.
        z: Scaled year-on-year changes, shape (days, 12).
        scale: The scale in force on each day, shape (days, 12), NaN before it exists.
        start_day: First day with a scale, or the number of days if none.
        vectors: For each episode, the first day its change vector can be read and the vector (12
            statistics, in scale units); the day is beyond the grid if it cannot be read.
    """

    detector: bocpd.DetectorOutput
    z: np.ndarray
    scale: np.ndarray
    start_day: int
    vectors: dict[int, tuple[int, np.ndarray]]


def _mean(*, scaled: np.ndarray, first: int, last: int) -> np.ndarray:
    """Return the mean of the finite rows of days first..last, NaN if there are none."""
    block = scaled[max(first, 0) : last + 1]
    if not np.isfinite(block).any(axis=1).any():
        return np.full(scaled.shape[1], np.nan)
    return np.nanmean(block, axis=0)


def _previous_mean(*, scaled: np.ndarray, onset: int, floor: int) -> np.ndarray | None:
    """Return the previous segment's mean of the six band features, or None if too few days."""
    first = max(onset - PREVIOUS_WINDOW_DAYS, floor, 0)
    block = scaled[first:onset]
    if np.isfinite(block).all(axis=1).sum() < MIN_PREVIOUS_DAYS:
        return None
    return np.nanmean(block, axis=0)


def split_onset(*, daily: np.ndarray, alarm_day: int, statistic: int) -> int:
    """Place the onset by one least-squares split of the last 56 daily values of a statistic.

    Args:
        daily: The daily analogues, shape (days, 12).
        alarm_day: The day the alarm fired.
        statistic: Index of the alarming statistic.

    Returns:
        The first day of the second segment. The alarm day minus 14 if too few values are finite.
    """
    first = max(alarm_day - ONSET_DAYS + 1, 0)
    days = np.arange(first, alarm_day + 1)
    values = daily[first : alarm_day + 1, statistic]
    ok = np.isfinite(values)
    days, values = days[ok], values[ok]
    if values.size < ONSET_MIN_BEFORE + ONSET_MIN_AFTER:
        return alarm_day - 14
    cs, cs2 = np.cumsum(values), np.cumsum(values**2)
    best_cost, best = np.inf, alarm_day - 14
    for k in range(ONSET_MIN_BEFORE, values.size - ONSET_MIN_AFTER + 1):
        left = cs2[k - 1] - cs[k - 1] ** 2 / k
        right = (cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / (values.size - k)
        if left + right < best_cost:
            best_cost, best = left + right, int(days[k])
    return best


def detect_stat(  # noqa: PLR0912, PLR0915 - one forward pass over the days
    *,
    windows: stats.WindowStats,
    scaled: np.ndarray,
    threshold: float,
) -> StatOutput:
    """Run the statistics detector forward over one Primary.

    Args:
        windows: The Primary's window statistics.
        scaled: The Primary's six scaled band features (`SiteFeatures.scaled`), used for the
            previous segment, the change since the onset, and the return closure.
        threshold: The alarm threshold on the largest absolute scaled change.

    Returns:
        The episodes, the scaled changes, the scale, and the era-id change vectors.
    """
    n_days = windows.stat.shape[0]
    change = stats.change_statistics(stat=windows.stat)
    y = np.full((n_days, stats.N_STATS), np.nan)
    z = np.full((n_days, stats.N_STATS), np.nan)
    scale_by_day = np.full((n_days, stats.N_STATS), np.nan)
    contaminated = np.zeros(n_days, dtype=bool)
    episode_id = np.full(n_days, -1, dtype=np.int64)
    delta = np.full((n_days, scaled.shape[1]), np.nan)
    peak = np.full(n_days, np.nan)
    episodes: list[bocpd.Episode] = []
    current: bocpd.Episode | None = None
    median: np.ndarray | None = None
    scale: np.ndarray | None = None
    above = back = calm = floor = alarm_stat = invalid_run = suppress_until = 0
    armed = True
    start_day = n_days
    for day in range(n_days):
        if windows.valid_day[day]:
            if invalid_run >= GAP_DAYS:
                suppress_until = day + 2 * stats.WINDOW_DAYS - 1
            invalid_run = 0
        else:
            invalid_run += 1
        y[day] = change[day] - _reference(change=change, contaminated=contaminated, day=day)
        known = day - SCALE_LAG_DAYS
        if known >= 0 and (scale is None or day % SCALE_REFRESH_DAYS == 0):
            usable = np.where(contaminated[: known + 1, None], np.nan, y[: known + 1])
            median_new, scale_new = stats.robust_scale(values=usable)
            if np.isfinite(scale_new).all():
                median, scale = median_new, scale_new
        if scale is None or median is None:
            continue
        start_day = min(start_day, day)
        scale_by_day[day] = scale
        z[day] = (y[day] - median) / scale
        if current is not None:
            episode_id[day] = current.episode
            delta[day] = _mean(scaled=scaled, first=current.onset_day, last=day) - (
                current.previous_mean
            )
            recent = _mean(scaled=scaled, first=day - bocpd.RETURN_DAYS + 1, last=day)
            is_back = bool((np.abs(recent - current.previous_mean) < bocpd.RETURN_SCALED).all())
            back = back + 1 if is_back else 0
            level = abs(z[day, alarm_stat])
            calm = calm + 1 if (np.isfinite(level) and level < RETURN_Z) else 0
            if back >= bocpd.RETURN_DAYS and calm >= bocpd.RETURN_DAYS:
                current.end_day, current.ended_by = day, "return"
                contaminated[max(day - RETURN_STEP_DAYS, 0) : day + CONTAMINATED_DAYS] = True
                current, back, calm, above, floor, armed = None, 0, 0, 0, day, False
                continue
        finite = np.isfinite(z[day])
        if not finite.any():
            above = 0
            continue
        top = int(np.nanargmax(np.where(finite, np.abs(z[day]), -np.inf)))
        peak[day] = abs(z[day, top])
        if peak[day] < threshold:
            armed = True
        above = above + 1 if (peak[day] >= threshold and armed and day >= suppress_until) else 0
        frozen = current is not None and day - current.onset_day >= CONTAMINATED_DAYS
        if above < CONSECUTIVE_DAYS or (current is not None and not frozen):
            continue
        levels = np.where(
            finite[: stats.N_LEVEL_STATS], np.abs(z[day, : stats.N_LEVEL_STATS]), -1.0
        )
        split_stat = int(np.argmax(levels)) if (levels >= 0).any() else top
        onset = split_onset(daily=windows.daily, alarm_day=day, statistic=split_stat)
        previous = _previous_mean(scaled=scaled, onset=onset, floor=floor)
        if previous is None:
            continue
        if current is not None:
            current.end_day, current.ended_by = day - 1, "superseded"
        current = bocpd.Episode(
            episode=len(episodes), onset_day=onset, alarm_day=day, previous_mean=previous
        )
        episodes.append(current)
        contaminated[max(min(onset, day - stats.WINDOW_DAYS), 0) : onset + CONTAMINATED_DAYS] = True
        episode_id[day] = current.episode
        delta[day] = _mean(scaled=scaled, first=onset, last=day) - previous
        back, calm, floor, alarm_stat = 0, 0, onset, top
        above, armed = 0, False
    detector = bocpd.DetectorOutput(
        episodes=episodes, episode_id=episode_id, delta_scaled=delta, p_short=peak
    )
    vectors = {
        e.episode: _change_vector(stat=windows.stat, scale=scale_by_day, episode=e)
        for e in episodes
    }
    return StatOutput(
        detector=detector, z=z, scale=scale_by_day, start_day=start_day, vectors=vectors
    )


def _reference(*, change: np.ndarray, contaminated: np.ndarray, day: int) -> np.ndarray:
    """Return the year-old 28-day change, or the one two years old, or zero (see module doc)."""
    nan = np.full(change.shape[1], np.nan)
    first = day - stats.YEAR_DAYS
    if first < 0:
        return nan
    if not contaminated[first]:
        return change[first]
    second = day - 2 * stats.YEAR_DAYS
    if second < 0:
        return nan
    return np.zeros(change.shape[1]) if contaminated[second] else change[second]


def _change_vector(
    *, stat: np.ndarray, scale: np.ndarray, episode: bocpd.Episode
) -> tuple[int, np.ndarray]:
    """Return the day an episode's change vector can first be read, and the vector.

    The vector is the window ending 27 days after the onset (entirely after it) minus the window
    ending the day before the onset (entirely before it), over the scale in force that day.
    """
    n_days = stat.shape[0]
    read_day = max(episode.onset_day + VECTOR_DAY_OFFSET, episode.alarm_day)
    before = episode.onset_day - 1
    if read_day >= n_days or before < 0:
        return n_days + 1, np.zeros(stats.N_STATS)
    after = episode.onset_day + VECTOR_DAY_OFFSET
    vector = (stat[after] - stat[before]) / scale[read_day]
    return read_day, np.nan_to_num(vector)


def assign_era_ids(*, confirmed: Sequence[tuple[int, int, np.ndarray]]) -> dict[int, int]:
    """Give each confirmed era an id, reusing an earlier id when the level has come back to it.

    A Primary starts at id 0 (no era). A new era reuses the id of an earlier era, or 0, when the
    sum of the change vectors since that era (the later eras' vectors and the new one) is within
    `RECURRENCE_SCALED` scale units in every statistic; the most recent such era wins. Otherwise it
    takes the next unused id, and an id is never recycled for a different era.

    Args:
        confirmed: For each era in order of confirmation, its episode number, its confirmation day,
            and its change vector.

    Returns:
        The id of each episode.
    """
    ids: dict[int, int] = {}
    stack: list[tuple[int, np.ndarray]] = []
    next_id = 1
    for episode, _day, vector in sorted(confirmed, key=lambda item: item[1]):
        chosen = None
        for position in range(len(stack), -1, -1):
            since = vector + sum((v for _, v in stack[position:]), np.zeros_like(vector))
            if (np.abs(since) <= RECURRENCE_SCALED).all():
                chosen = stack[position - 1][0] if position > 0 else 0
                break
        if chosen is None:
            chosen, next_id = next_id, next_id + 1
        ids[episode] = chosen
        stack.append((chosen, vector))
    return ids


def tune_stat_threshold(
    *,
    windows: Sequence[stats.WindowStats],
    scaled: Sequence[np.ndarray],
    excluded: Sequence[np.ndarray],
    window: tuple[int, int],
) -> tuple[float, dict[float, float]]:
    """Choose the lowest threshold with at most 0.5 false alarms per Primary-year.

    Args:
        windows: The window statistics of each tuning Primary (real clean history).
        scaled: Each Primary's six scaled band features.
        excluded: Per Primary, a boolean per day that marks days inside a labelled event.
        window: First and last day (inclusive) of the tuning period.

    Returns:
        The chosen threshold (the largest on the grid if none meets the target) and the false
        alarm rate at every grid threshold.
    """
    rates: dict[float, float] = {}
    for threshold in STAT_THRESHOLD_GRID:
        outputs = [
            detect_stat(windows=w, scaled=x, threshold=threshold)
            for w, x in zip(windows, scaled, strict=True)
        ]
        rates[threshold] = bocpd.false_alarms_per_year(
            outputs=[o.detector for o in outputs],
            window=window,
            start_days=[o.start_day for o in outputs],
            excluded=excluded,
        )
    chosen = next(
        (t for t in STAT_THRESHOLD_GRID if rates[t] <= bocpd.FALSE_ALARM_TARGET),
        STAT_THRESHOLD_GRID[-1],
    )
    return chosen, rates

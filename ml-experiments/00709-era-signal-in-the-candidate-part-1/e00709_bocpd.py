"""Experiment 709: a forward-only online Bayesian change-point detector and its alarm rule.

The detector is the one of [Adams and MacKay (2007)](https://arxiv.org/abs/0710.3742). Each day it
reads the six scaled features of the Primary (`e00709_features`), keeps a posterior over the run
length (days since the last change), and updates that posterior with one new day. Nothing later than
the day just read is used, so the posterior after day `d` is a legitimate signal at the origin of
day `d + 1`.

**Model.** Within a segment each feature is an independent Normal with unknown mean and variance
(Normal-Gamma prior), so the one-day predictive density is a Student-t. The hazard is constant at
1/730 per day and the run length is truncated at 730 days, with the mass beyond it kept at 730.

**Alarm rule (plan section 5).** An alarm fires on the first day on which the posterior mass on run
lengths under 28 days has exceeded the threshold on 3 consecutive days *and* the mean of the
features since the estimated onset differs from the previous segment's mean by at least 3 scaled
units in one feature and at least 5% of capacity in that same feature. The threshold is tuned on
clean history to at most 0.5 false alarms per Primary-year (`tune_threshold`).

**Episodes.** An alarm opens an *episode*, which records the estimated onset day, the alarm day, and
the previous segment's mean. An episode closes when the 3-day mean of every feature is back within
1.5 scaled units of the previous mean for 3 consecutive days. The detector then stays disarmed until
the posterior mass on short run lengths has fallen below the threshold once, so the return step of
a switching event is not a new alarm, and the next previous segment starts after the return. An
episode older than 28 days is frozen, which lets a later alarm supersede it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy.special import gammaln, logsumexp

HAZARD: Final[float] = 1.0 / 730.0
MAX_RUN: Final[int] = 730
SHORT_RUN: Final[int] = 28
CONSECUTIVE_DAYS: Final[int] = 3
MIN_CHANGE_SCALED: Final[float] = 3.0
MIN_CHANGE_SHARE: Final[float] = 0.05
PREVIOUS_WINDOW_DAYS: Final[int] = 180
MIN_PREVIOUS_DAYS: Final[int] = 14
RETURN_SCALED: Final[float] = 1.5
RETURN_DAYS: Final[int] = 3
FALSE_ALARM_TARGET: Final[float] = 0.5
"""False alarms per Primary-year that the tuned threshold allows."""
THRESHOLD_GRID: Final[tuple[float, ...]] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
DEFAULT_THRESHOLD: Final[float] = 0.95
"""Used until `tune_threshold` has run."""

_MU0: Final[float] = 0.0
_KAPPA0: Final[float] = 0.1
_ALPHA0: Final[float] = 2.0
_BETA0: Final[float] = 1.5
_LOG_HAZARD: Final[float] = float(np.log(HAZARD))
_LOG_GROWTH: Final[float] = float(np.log1p(-HAZARD))

EndedByType = Literal["open", "return", "superseded"]


@dataclass
class Episode:
    """One detected shift, from its alarm to its end.

    Attributes:
        episode: Running number of the episode within the Primary, from 0.
        onset_day: Estimated first day of the shift, fixed on the alarm day.
        alarm_day: Day on which the alarm fired.
        previous_mean: Mean scaled feature vector of the segment before the shift, shape (6,).
        end_day: Last day of the episode, or -1 while the episode is open.
        ended_by: `open`, `return` (the level came back), or `superseded` (a later alarm).
    """

    episode: int
    onset_day: int
    alarm_day: int
    previous_mean: np.ndarray
    end_day: int = -1
    ended_by: EndedByType = "open"


@dataclass(frozen=True)
class RunLengthSeries:
    """The detector's posterior summary after each day.

    Attributes:
        p_short: Posterior mass on run lengths under 28 days, NaN on days without features.
        map_run: Most probable run length in 1 to 27 days, so the onset is `day - map_run + 1`.
    """

    p_short: np.ndarray
    map_run: np.ndarray


@dataclass(frozen=True)
class DetectorOutput:
    """The episodes of one Primary and the per-day view of them.

    Attributes:
        episodes: Every episode, in order of alarm.
        episode_id: The open episode of each day (alarm day to end day), or -1.
        delta_scaled: Mean of the features since the episode's onset minus the previous segment's
            mean, shape (days, 6), NaN while no episode is open.
        p_short: Posterior mass on short run lengths.
    """

    episodes: list[Episode]
    episode_id: np.ndarray
    delta_scaled: np.ndarray
    p_short: np.ndarray


def run_posterior(*, scaled: np.ndarray, start_day: int) -> RunLengthSeries:
    """Run the Bayesian online change-point recursion forward over the scaled features.

    A day with a missing feature leaves the posterior as it was and reports NaN.

    Args:
        scaled: Scaled daily features, shape (days, features).
        start_day: First day to read; earlier days are ignored.

    Returns:
        The summary of the posterior after each day.
    """
    n_days, n_features = scaled.shape
    runs = np.arange(MAX_RUN + 1)
    kappa = _KAPPA0 + runs
    alpha = _ALPHA0 + 0.5 * runs
    nu = 2.0 * alpha
    constant = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
    mu = np.full((n_features, MAX_RUN + 1), _MU0)
    beta = np.full((n_features, MAX_RUN + 1), _BETA0)
    log_r = np.full(MAX_RUN + 1, -np.inf)
    log_r[0] = 0.0
    p_short = np.full(n_days, np.nan)
    map_run = np.zeros(n_days, dtype=np.int64)
    for day in range(max(start_day, 0), n_days):
        x = scaled[day][:, None]
        if not np.isfinite(x).all():
            continue
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        z2 = (x - mu) ** 2 / scale2
        log_pred = constant - 0.5 * np.log(scale2) - (alpha + 0.5) * np.log1p(z2 / nu)
        joint = log_r + log_pred.sum(axis=0)
        new_log_r = np.empty_like(log_r)
        new_log_r[0] = logsumexp(joint) + _LOG_HAZARD
        new_log_r[1:] = joint[:-1] + _LOG_GROWTH
        new_log_r[-1] = np.logaddexp(new_log_r[-1], joint[-1] + _LOG_GROWTH)
        new_log_r -= logsumexp(new_log_r)
        k = kappa[:-1]
        new_mu = np.empty_like(mu)
        new_beta = np.empty_like(beta)
        new_mu[:, 0], new_beta[:, 0] = _MU0, _BETA0
        new_mu[:, 1:] = (k * mu[:, :-1] + x) / (k + 1.0)
        new_beta[:, 1:] = beta[:, :-1] + k * (x - mu[:, :-1]) ** 2 / (2.0 * (k + 1.0))
        mu, beta, log_r = new_mu, new_beta, new_log_r
        p_short[day] = float(np.exp(log_r[:SHORT_RUN]).sum())
        map_run[day] = 1 + int(np.argmax(log_r[1:SHORT_RUN]))
    return RunLengthSeries(p_short=p_short, map_run=map_run)


def _mean(*, scaled: np.ndarray, first: int, last: int) -> np.ndarray:
    """Return the mean of the finite scaled features of days first..last, NaN if there are none."""
    block = scaled[max(first, 0) : last + 1]
    if not np.isfinite(block).any(axis=1).any():
        return np.full(scaled.shape[1], np.nan)
    return np.nanmean(block, axis=0)


def _change_passes(*, delta_scaled: np.ndarray, spread: np.ndarray) -> bool:
    share = np.abs(delta_scaled) * spread
    return bool(((np.abs(delta_scaled) >= MIN_CHANGE_SCALED) & (share >= MIN_CHANGE_SHARE)).any())


def _previous_mean(*, scaled: np.ndarray, onset: int, floor: int) -> np.ndarray | None:
    """Return the previous segment's mean, or None if it has too few valid days."""
    first = max(onset - PREVIOUS_WINDOW_DAYS, floor, 0)
    block = scaled[first:onset]
    if np.isfinite(block).all(axis=1).sum() < MIN_PREVIOUS_DAYS:
        return None
    return np.nanmean(block, axis=0)


def detect_episodes(
    *,
    series: RunLengthSeries,
    scaled: np.ndarray,
    spread: np.ndarray,
    threshold: float,
    start_day: int,
) -> DetectorOutput:
    """Apply the alarm rule to a posterior and return the episodes.

    Args:
        series: The output of `run_posterior`.
        scaled: The scaled daily features the posterior was computed from.
        spread: Robust spread of each feature, to turn a scaled change into a share of capacity.
        threshold: Posterior mass on run lengths under 28 days that must be exceeded.
        start_day: First day the detector read.

    Returns:
        The episodes and the per-day view.
    """
    n_days = scaled.shape[0]
    episode_id = np.full(n_days, -1, dtype=np.int64)
    delta = np.full((n_days, scaled.shape[1]), np.nan)
    episodes: list[Episode] = []
    current: Episode | None = None
    above = 0
    back = 0
    floor = 0
    armed = True
    for day in range(max(start_day, 0), n_days):
        p = series.p_short[day]
        if current is not None:
            episode_id[day] = current.episode
            delta[day] = _mean(scaled=scaled, first=current.onset_day, last=day) - (
                current.previous_mean
            )
            recent = _mean(scaled=scaled, first=day - RETURN_DAYS + 1, last=day)
            is_back = bool((np.abs(recent - current.previous_mean) < RETURN_SCALED).all())
            back = back + 1 if is_back else 0
            if back >= RETURN_DAYS:
                current.end_day, current.ended_by = day, "return"
                current, back, above, floor, armed = None, 0, 0, day, False
                continue
        if np.isnan(p):
            continue
        if p <= threshold:
            armed = True
        above = above + 1 if (p > threshold and armed) else 0
        frozen = current is not None and day - current.onset_day > SHORT_RUN
        if above < CONSECUTIVE_DAYS or (current is not None and not frozen):
            continue
        onset = day - int(series.map_run[day]) + 1
        previous = _previous_mean(scaled=scaled, onset=onset, floor=floor)
        if previous is None:
            continue
        change = _mean(scaled=scaled, first=onset, last=day) - previous
        if not _change_passes(delta_scaled=change, spread=spread):
            continue
        if current is not None:
            current.end_day, current.ended_by = day - 1, "superseded"
        current = Episode(
            episode=len(episodes), onset_day=onset, alarm_day=day, previous_mean=previous
        )
        episodes.append(current)
        episode_id[day], delta[day], back, floor = current.episode, change, 0, onset
    return DetectorOutput(
        episodes=episodes, episode_id=episode_id, delta_scaled=delta, p_short=series.p_short
    )


def detect(
    *, scaled: np.ndarray, spread: np.ndarray, start_day: int, threshold: float = DEFAULT_THRESHOLD
) -> DetectorOutput:
    """Run the posterior and the alarm rule over one Primary's features.

    Args:
        scaled: Scaled daily features, shape (days, 6).
        spread: Robust spread of each feature.
        start_day: First day the detector reads (`SiteFeatures.ready_day`).
        threshold: The alarm threshold.

    Returns:
        The episodes and the per-day view.
    """
    series = run_posterior(scaled=scaled, start_day=start_day)
    return detect_episodes(
        series=series, scaled=scaled, spread=spread, threshold=threshold, start_day=start_day
    )


def month_shuffle(
    *, scaled: np.ndarray, start_day: int, month_of_year: np.ndarray, seed: int
) -> np.ndarray:
    """Return a stationary surrogate: days from `start_day` on, shuffled within each calendar month.

    Each day is swapped with a random day of the same month of the year (in any year), so the
    seasonal mean of every feature survives and every level shift is destroyed. Every alarm on the
    surrogate is therefore false.

    Args:
        scaled: Scaled daily features, shape (days, 6).
        start_day: First day to shuffle.
        month_of_year: Month of the year (1 to 12) of every day.
        seed: Random seed.

    Returns:
        A copy of `scaled` with the days from `start_day` on permuted within their month.
    """
    rng = np.random.default_rng(seed)
    out = scaled.copy()
    days = np.arange(start_day, scaled.shape[0])
    for month in range(1, 13):
        members = days[month_of_year[days] == month]
        out[members] = scaled[rng.permutation(members)]
    return out


def false_alarms_per_year(
    *,
    outputs: Sequence[DetectorOutput],
    window: tuple[int, int],
    start_days: Sequence[int],
    excluded: Sequence[np.ndarray],
) -> float:
    """Return the false alarms per Primary-year inside a window of days.

    Args:
        outputs: One detector output per Primary.
        window: First and last day (inclusive) that count.
        start_days: First day each detector read.
        excluded: Per Primary, a boolean per day that marks days inside a labelled event.

    Returns:
        Alarms on days outside labelled events, over the Primary-years the detectors covered.
    """
    alarms = 0
    days = 0
    for output, start, skip in zip(outputs, start_days, excluded, strict=True):
        first, last = max(window[0], start), window[1]
        if last < first:
            continue
        days += last - first + 1
        alarms += sum(
            1 for e in output.episodes if first <= e.alarm_day <= last and not skip[e.alarm_day]
        )
    return alarms / (days / 365.25) if days else float("nan")


def tune_threshold(
    *,
    series: Sequence[RunLengthSeries],
    scaled: Sequence[np.ndarray],
    spread: Sequence[np.ndarray],
    start_days: Sequence[int],
    excluded: Sequence[np.ndarray],
    window: tuple[int, int],
) -> tuple[float, dict[float, float]]:
    """Choose the lowest threshold with at most 0.5 false alarms per Primary-year.

    Args:
        series: The posterior of each clean tuning series (real history away from labelled events,
            or a block-shuffled surrogate).
        scaled: The scaled features behind each posterior.
        spread: Each Primary's robust spread.
        start_days: First day each detector read.
        excluded: Per series, a boolean per day that marks days inside a labelled event.
        window: First and last day (inclusive) of the tuning period.

    Returns:
        The chosen threshold (the largest on the grid if none meets the target) and the false
        alarm rate at every grid threshold.
    """
    rates: dict[float, float] = {}
    for threshold in THRESHOLD_GRID:
        outputs = [
            detect_episodes(series=s, scaled=x, spread=w, threshold=threshold, start_day=start)
            for s, x, w, start in zip(series, scaled, spread, start_days, strict=True)
        ]
        rates[threshold] = false_alarms_per_year(
            outputs=outputs, window=window, start_days=start_days, excluded=excluded
        )
    chosen = next((t for t in THRESHOLD_GRID if rates[t] <= FALSE_ALARM_TARGET), THRESHOLD_GRID[-1])
    return chosen, rates


def penalised_segmentation(*, x: np.ndarray, penalty: float, min_size: int) -> list[int]:
    """Return the change points of a piecewise-constant mean by penalised least squares.

    This is optimal partitioning with a minimum segment length: the segmentation that minimises
    the summed squared deviation from each segment's mean plus `penalty` per segment. It reads the
    whole series (it is a hindsight cross-check of the online detector, not a causal signal).

    Args:
        x: A finite one-dimensional series.
        penalty: Cost of each extra segment, in the units of `x` squared.
        min_size: Shortest allowed segment, in samples.

    Returns:
        The sample index at which each new segment starts, in increasing order.
    """
    n = x.size
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x**2)])
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=np.int64)
    for t in range(min_size, n + 1):
        starts = np.arange(0, t - min_size + 1)
        starts = starts[np.isfinite(best[starts])]
        cost = best[starts] + c2[t] - c2[starts] - (c1[t] - c1[starts]) ** 2 / (t - starts)
        j = int(np.argmin(cost))
        best[t], last[t] = cost[j] + penalty, starts[j]
    cuts: list[int] = []
    t = n
    while t > 0:
        s = int(last[t])
        if s > 0:
            cuts.append(s)
        t = s
    return cuts[::-1]

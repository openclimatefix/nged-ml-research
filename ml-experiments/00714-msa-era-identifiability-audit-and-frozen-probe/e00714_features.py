"""Experiment 714: the features of one event at one forecast origin.

**The origin rule.** The origin is `k` days after the onset day at 09:00 UTC. A feature reads power,
reference level, and weather only at half-hours before `origin - 6 hours`, the Candidate's data
cut. The onset is given to the feature builder (an upper bound on what an online detector knows).
The class, kind, duration, and mirror state of the event never enter a feature.

**Change.** The change at a half-hour is the power minus the reference level, in points of capacity
(100 times the share). The profile of the change is its mean at each half-hour of the day over the
post-onset half-hours of the window.

**Host features (primary).** `size` (median change), `flat` (shape-flatness of the profile: the
absolute mean over the root mean square, 1 for a constant change), `asym` (the night-to-midday
contrast, bounded), `corr` (correlation across days of the daily mean change with the daily mean
shortwave), `signdays` (share of days whose mean change has the sign of `size`), and `late` (the
median change of the last 3 days minus `size`).

**Step features (secondary).** `steps` (up and down steps per day) and `high` (fraction of
half-hours at the high level), from a Schmitt trigger on the 5-half-hour running median of the
change.

**Older-row features.** `old_change`, `old_spread`, `old_z`, and `flat_old`, from the median of the
single same-weekday days at ages 21, 28, 42, 63, 91, and 364 days that end before the onset.

**Neighbour features.** `mirror_share` and `mirror_corr`: each site's change is first measured
against its own mean change at each half-hour of the 14 days before the onset, then `mirror_share`
is the ratio of the neighbours' summed step in MW (negated) to the host's step in MW, and
`mirror_corr` is the correlation of the two centred series before the onset minus the
correlation over the whole span, which removes the shared error of the reference level.
"""

from dataclasses import dataclass
from typing import Final

import e00714_common as common
import numpy as np
from scipy.ndimage import median_filter

HOST_FEATURES: Final[tuple[str, ...]] = ("size", "flat", "asym", "corr", "signdays", "late")
STEP_FEATURES: Final[tuple[str, ...]] = ("steps", "high")
OLD_FEATURES: Final[tuple[str, ...]] = ("old_change", "old_spread", "old_z", "flat_old")
NEIGHBOUR_FEATURES: Final[tuple[str, ...]] = ("mirror_share", "mirror_corr")
ALL_FEATURES: Final[tuple[str, ...]] = (
    *HOST_FEATURES, *STEP_FEATURES, *OLD_FEATURES, *NEIGHBOUR_FEATURES
)  # fmt: skip
MIN_COVERAGE: Final[float] = 0.6
MIN_POST_HH: Final[int] = 96
"""A window needs this many finite post-onset half-hours (2 days)."""
NIGHT: Final[slice] = slice(0, 12)
MIDDAY: Final[slice] = slice(22, 28)
ASYM_EPS: Final[float] = 0.5
LATE_HH: Final[int] = 3 * common.DAY
SMOOTH_HH: Final[int] = 5
SCHMITT: Final[tuple[float, float]] = (0.35, 0.65)
MIN_STEP_POINTS: Final[float] = 1.0
MIRROR_CLIP: Final[float] = 2.0
MIN_DAYS_CORR: Final[int] = 4
MIN_OLD_ROWS: Final[int] = 2
MIN_STEP_MW: Final[float] = 0.3
MIN_OLD_COVERAGE: Final[float] = 0.5


@dataclass(frozen=True)
class EventInput:
    """What the feature builder needs to know about one event.

    Attributes:
        host: Index of the host Primary.
        onset_hh: First half-hour of the event on the grid.
        onset_day: Grid day of the onset.
        neighbours: Indices of the neighbour set that the neighbour features read.
    """

    host: int
    onset_hh: int
    onset_day: int
    neighbours: list[int]


def flatness(*, profile: np.ndarray) -> float:
    """Return the shape-flatness of a daily profile: |mean| over root mean square, in [0, 1].

    Args:
        profile: The change at each half-hour of the day.

    Returns:
        1 for a constant profile, 0 for a profile of zero mean.
    """
    rms = float(np.sqrt(np.mean(profile**2)))
    return 0.0 if rms == 0.0 else abs(float(np.mean(profile))) / rms


def _profile(*, change: np.ndarray, hh: np.ndarray, post: np.ndarray) -> np.ndarray | None:
    """Return the mean post-onset change at each half-hour of the day, or None if too sparse."""
    prof = np.full(common.DAY, np.nan)
    use = post & np.isfinite(change)
    for h in range(common.DAY):
        vals = change[use & (hh == h)]
        if vals.size >= 2:
            prof[h] = vals.mean()
    if np.isfinite(prof).sum() < common.DAY // 2:
        return None
    return np.where(np.isfinite(prof), prof, np.nanmean(prof))


def step_features(*, change: np.ndarray) -> tuple[float, float]:
    """Return the number of steps per day and the fraction of half-hours at the high level.

    The change is smoothed by a running median of 5 half-hours. The high level is the 90th
    percentile of its absolute value. A Schmitt trigger with thresholds at 35% and 65% of the high
    level switches the state, and each switch is one step (up or down).

    Args:
        change: The post-onset change, in points of capacity, NaN-free.

    Returns:
        Steps per day and the high fraction. Both are 0 when the high level is under 1 point.
    """
    smooth = np.abs(median_filter(change, size=SMOOTH_HH, mode="nearest"))
    high_level = float(np.percentile(smooth, 90))
    if high_level < MIN_STEP_POINTS:
        return 0.0, 0.0
    up, down = SCHMITT[1] * high_level, SCHMITT[0] * high_level
    state = bool(smooth[0] > up)
    switches = 0
    on = np.zeros(smooth.size, dtype=bool)
    for i, value in enumerate(smooth):
        if state and value < down:
            state = False
            switches += 1
        elif not state and value > up:
            state = True
            switches += 1
        on[i] = state
    return switches / (smooth.size / common.DAY), float(on.mean())


def build_features(
    *,
    event: EventInput,
    k: int,
    power: common.PassPower,
    reference: common.PassPower,
    shortwave: np.ndarray,
    capacity_mw: np.ndarray,
) -> dict[str, float] | None:
    """Return the features of one event at the origin `k` days after the onset day.

    Args:
        event: The event.
        k: Days from the onset day to the origin.
        power: Power of the pass (injected, or real for a noiseless known-answer).
        reference: The reference level (the real power for a noiseless known-answer).
        shortwave: ERA5 shortwave of the host over the whole grid (read before the origin only).
        capacity_mw: Effective capacity in MW of every site.

    Returns:
        A dict of the features, or None if the window lacks data.
    """
    origin_hh = (event.onset_day + k) * common.DAY + common.ORIGIN_HOUR_HH
    end = origin_hh - common.LATENCY_HH
    lo = end - common.WINDOW_HH
    idx = np.arange(lo, end)
    hh = idx % common.DAY
    post = idx >= event.onset_hh
    y = power.get(site=event.host, lo=lo, hi=end)
    r = reference.get(site=event.host, lo=lo, hi=end)
    change = (y - r) * 100.0
    if (
        np.isfinite(change).mean() < MIN_COVERAGE
        or (post & np.isfinite(change)).sum() < MIN_POST_HH
    ):
        return None
    prof = _profile(change=change, hh=hh, post=post)
    if prof is None:
        return None
    size = float(np.nanmedian(change[post]))
    out: dict[str, float] = {"size": size, "flat": flatness(profile=prof)}
    night, midday = float(prof[NIGHT].mean()), float(prof[MIDDAY].mean())
    out["asym"] = (night - midday) / (abs(night) + abs(midday) + ASYM_EPS)
    blocks = change.reshape(common.WINDOW_DAYS, common.DAY)
    block_post = post.reshape(blocks.shape)
    sw = shortwave[lo:end].reshape(blocks.shape)
    day_change, day_sw = [], []
    for row, mask, sw_row in zip(blocks, block_post, sw, strict=True):
        ok = mask & np.isfinite(row)
        if ok.sum() >= 36:
            day_change.append(row[ok].mean())
            day_sw.append(np.nanmean(sw_row[ok]))
    corr = 0.0
    if len(day_change) >= MIN_DAYS_CORR and np.std(day_change) > 0 and np.std(day_sw) > 0:
        corr = float(np.corrcoef(day_change, day_sw)[0, 1])
    out["corr"] = corr
    sign = np.sign(size) if size != 0 else 1.0
    out["signdays"] = float(np.mean(np.sign(day_change) == sign)) if day_change else 0.0
    late = change[-LATE_HH:]
    out["late"] = float(np.nanmedian(late) - size) if np.isfinite(late).any() else 0.0
    filled = np.where(np.isfinite(change), change, size)[post]
    out["steps"], out["high"] = step_features(change=filled)
    out.update(
        _older_features(event=event, power=power, lo=lo, end=end, change=change, post=post, hh=hh)
    )
    out.update(
        _neighbour_features(
            event=event, power=power, reference=reference, capacity_mw=capacity_mw,
            end=end,
        )
    )  # fmt: skip
    return out


def _older_features(
    *, event: EventInput, power: common.PassPower, lo: int, end: int, change: np.ndarray,
    post: np.ndarray, hh: np.ndarray,
) -> dict[str, float]:  # fmt: skip
    """Return the older-row features. Rows that end after the onset are left out."""
    rows = []
    for age in common.OLDER_AGES:
        a, b = end - common.DAY - age * common.DAY, end - age * common.DAY
        if b > event.onset_hh:
            continue
        row = power.get(site=event.host, lo=a, hi=b)
        if np.isfinite(row).mean() >= MIN_OLD_COVERAGE:
            rows.append(row)
    nan = {name: float("nan") for name in OLD_FEATURES}
    if len(rows) < MIN_OLD_ROWS:
        return nan
    stack = np.stack(rows) * 100.0
    with np.errstate(all="ignore"):
        ref = np.nanmedian(stack, axis=0)
        spread = float(np.nanmean(np.nanstd(stack, axis=0)))
    # the older row at position p has half-hour of the day (a + p) % 48; the window shares it
    pos = np.arange(common.DAY)
    ref_by_hh = np.empty(common.DAY)
    ref_by_hh[(end - common.DAY + pos) % common.DAY] = ref
    y = power.get(site=event.host, lo=lo, hi=end) * 100.0
    diff = y - ref_by_hh[hh]
    use = post & np.isfinite(diff)
    if use.sum() < MIN_POST_HH or not np.isfinite(spread):
        return nan
    prof = _profile(change=diff, hh=hh, post=post)
    old_change = float(np.nanmedian(diff[use]))
    return {
        "old_change": old_change,
        "old_spread": spread,
        "old_z": old_change / (spread + ASYM_EPS),
        "flat_old": flatness(profile=prof) if prof is not None else float("nan"),
    }


def centred_change(
    *, power: common.PassPower, reference: common.PassPower, site: int, onset_hh: int, lo: int,
    end: int,
) -> np.ndarray | None:  # fmt: skip
    """Return a site's change over `[lo, end)` against its own pre-onset behaviour.

    The change is power minus the reference level. The mean change at each half-hour of the day over
    the 14 days before the onset is subtracted. Those days lie before the origin, so no feature
    reads data after the origin. The subtraction removes the reference level's error that the host
    and its neighbours share.

    Args:
        power: Power of the pass.
        reference: The reference level.
        site: Site index.
        onset_hh: First half-hour of the event.
        lo: First half-hour of the window.
        end: Half-hour after the last of the window.

    Returns:
        The centred change in shares of capacity, or None if fewer than half of the 14 pre-onset
        days have readings.
    """
    pre_lo = onset_hh - common.WINDOW_HH
    pre = power.get(site=site, lo=pre_lo, hi=onset_hh) - reference.get(
        site=site, lo=pre_lo, hi=onset_hh
    )
    if np.isfinite(pre).mean() < MIN_OLD_COVERAGE:
        return None
    hh_pre = np.arange(pre_lo, onset_hh) % common.DAY
    baseline = np.zeros(common.DAY)
    for h in range(common.DAY):
        vals = pre[hh_pre == h]
        vals = vals[np.isfinite(vals)]
        baseline[h] = vals.mean() if vals.size else 0.0
    window = power.get(site=site, lo=lo, hi=end) - reference.get(site=site, lo=lo, hi=end)
    return window - baseline[np.arange(lo, end) % common.DAY]


def _neighbour_features(
    *, event: EventInput, power: common.PassPower, reference: common.PassPower,
    capacity_mw: np.ndarray, end: int,
) -> dict[str, float]:  # fmt: skip
    """Return the mirror features, each site measured against its own pre-onset days.

    The series run from 14 days before the onset to the data cut. `mirror_share` is the negated
    median post-onset step of the neighbours' summed change in MW over the host's median post-onset
    step in MW (0 when the host's step is under 0.3 MW). `mirror_corr` is the correlation of the
    host's change and the neighbours' summed change over the 14 days before the onset (the shared
    error of the reference level) minus the same correlation over the whole span: a step that the
    neighbours mirror lowers the correlation, a shared error does not. With no event both are
    near 0.
    """
    zero = {"mirror_share": 0.0, "mirror_corr": 0.0}
    lo = event.onset_hh - common.WINDOW_HH
    host_c = centred_change(
        power=power, reference=reference, site=event.host, onset_hh=event.onset_hh, lo=lo, end=end
    )
    if host_c is None:
        return zero
    total = np.zeros(end - lo)
    n_used = 0
    for j in event.neighbours:
        if j == event.host:
            continue
        c = centred_change(
            power=power, reference=reference, site=j, onset_hh=event.onset_hh, lo=lo, end=end
        )
        if c is None or np.isfinite(c).mean() < MIN_OLD_COVERAGE:
            continue
        total += np.where(np.isfinite(c), c, 0.0) * capacity_mw[j]
        n_used += 1
    host_mw = host_c * capacity_mw[event.host]
    post = np.arange(lo, end) >= event.onset_hh
    use = post & np.isfinite(host_mw)
    if n_used == 0 or use.sum() < MIN_POST_HH:
        return zero
    step = float(np.median(host_mw[use]))
    share = -float(np.median(total[use])) / step if abs(step) >= MIN_STEP_MW else 0.0
    finite = np.isfinite(host_mw)
    corr_pre = _correlation(x=host_mw[finite & ~post], z=total[finite & ~post])
    corr_all = _correlation(x=host_mw[finite], z=total[finite])
    return {
        "mirror_share": float(np.clip(share, -MIRROR_CLIP, MIRROR_CLIP)),
        "mirror_corr": corr_pre - corr_all,
    }


def _correlation(*, x: np.ndarray, z: np.ndarray) -> float:
    """Return the correlation of two series, 0 if either is constant or empty."""
    if x.size < 2 or np.std(x) == 0 or np.std(z) == 0:
        return 0.0
    return float(np.corrcoef(x, z)[0, 1])

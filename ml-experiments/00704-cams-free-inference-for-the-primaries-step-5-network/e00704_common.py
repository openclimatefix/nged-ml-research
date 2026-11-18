"""Experiments 703 and 704: the previous-UTC-day mean shortwave and clear-sky ratio of an origin."""

from typing import Final

import e00437_features as f437  # ty: ignore[unresolved-import]
import numpy as np

DAY_US: Final = 86_400 * 1_000_000


def previous_day_columns(
    *, shortwave: np.ndarray, clear: np.ndarray, origin_us: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per origin the previous UTC day's mean shortwave and its ratio to the clear-sky sum.

    Both arrays are on experiment 87's half-hourly grid. The 48 stamps are 00:30 of the UTC day
    before the origin's UTC date to 00:00 of the origin's day (period ends). A value is NaN if any
    of the 48 shortwave values is not finite, or (ratio only) if the clear-sky sum is 0.
    """
    finite = np.isfinite(shortwave)
    count = np.concatenate([[0], np.cumsum(finite)])
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, shortwave, 0.0))])
    clear_total = np.concatenate([[0.0], np.cumsum(clear)])
    first = f437.grid_of(us=(origin_us // DAY_US - 1) * DAY_US) + 1
    stop = first + f437.STEPS_PER_DAY
    inside = (first >= 0) & (stop <= len(shortwave))
    lo, hi = np.where(inside, first, 0), np.where(inside, stop, 0)
    complete = inside & ((count[hi] - count[lo]) == f437.STEPS_PER_DAY)
    s_sum, c_sum = total[hi] - total[lo], clear_total[hi] - clear_total[lo]
    mean = np.where(complete, s_sum / f437.STEPS_PER_DAY, np.nan)
    csi = np.where(complete & (c_sum > 0), s_sum / np.where(c_sum > 0, c_sum, 1.0), np.nan)
    return mean, csi

# ruff: noqa: E501
"""Experiment 693: per-row forecast scores in points of capacity (shared by the head scorers)."""

from typing import Final

import numpy as np

QUANTILES: Final = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)


def row_scores(*, q: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row scores: CRPS (twice the mean pinball loss), absolute error of the median, tail
    pinball, and the q05 to q95 cover.

    `q` holds the 11 quantiles as shares of capacity, sorted within the row; `y` is the target as a
    share of capacity.
    """
    lv = np.asarray(QUANTILES, np.float32)
    err = y[:, None] - q
    pin = np.maximum(lv * err, (lv - 1.0) * err) * 100.0
    return {
        "crps": 2.0 * pin.mean(axis=1),
        "ae": np.abs(err[:, 5]) * 100.0,
        "pin01": pin[:, 0], "pin05": pin[:, 1], "pin95": pin[:, 9], "pin99": pin[:, 10],
        "cover": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(np.float32),
    }  # fmt: skip

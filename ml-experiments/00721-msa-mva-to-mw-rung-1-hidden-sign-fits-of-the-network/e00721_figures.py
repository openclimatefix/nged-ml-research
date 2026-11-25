# ruff: noqa: E501
"""Experiment 721: the folder for dated detail and the guard that every figure's data passes first.

Dated detail about D5 and D10 (a prediction time series, an attention map at a dated half-hour) goes
in `PRIVATE/`, a subfolder of the experiment's data folder, and never in the repository or in a
published page. Every array that a figure would draw passes `assert_no_two_worlds_on_shared_dates`
first: a figure must never hold two of the host's reading, the injected reading, and the signed truth
on a common half-hour, because the difference of any two is the injected farm's output.
"""

from pathlib import Path

import e00721_fit as fit
import e00721_worlds as wd
import numpy as np

PRIVATE_FOLDER = "PRIVATE"
"""Subfolder of the experiment's data folder that holds dated detail."""


def private_dir(*, data_dir: Path) -> Path:
    """Create (with `Path.mkdir`) and return the folder of dated detail."""
    folder = fit.data_root(data_dir=data_dir) / PRIVATE_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_figure_data(
    *,
    data_dir: Path,
    name: str,
    series: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> Path:
    """Write the arrays of one figure to `PRIVATE/<name>.npz` after the guard has passed.

    Args:
        data_dir: The data folder.
        name: The figure's file stem.
        series: The `host`, `injected`, and `truth` arrays that the figure would draw, for the guard.
        arrays: The arrays to store.
        mask: Half-hours to consider in the guard, or `None` for all.

    Returns:
        The path of the stored file.

    Raises:
        ValueError: When the guard finds two worlds on a shared half-hour (nothing is written).
    """
    wd.assert_no_two_worlds_on_shared_dates(series=series, mask=mask)
    path = private_dir(data_dir=data_dir) / f"{name}.npz"
    np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]
    return path

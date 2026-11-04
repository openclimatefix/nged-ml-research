# ruff: noqa: E501
"""The three trailing-temperature columns (`W-T`) and their shuffled-day placebo for a month's rows.

Uses experiment 515's `Windows` (the ensemble-row tables built with the `previous` seam, and the
ERA5 prefix sums for pretraining rows) and experiment 526's rule for the placebo (the three `W-T`
columns of the `W-null` donor draw). Real columns of a row at (origin, valid) come from the weather run
issued at or before the origin; `Windows.real` raises if any row has no column.
"""

from datetime import UTC, datetime

import e00510_common as c  # ty: ignore[unresolved-import]
import e00515_features as feat  # ty: ignore[unresolved-import]
import e00515_fit as f515  # ty: ignore[unresolved-import]
import numpy as np

T_INDEX = [feat.ALL_COLUMNS.index(name) for name in feat.ARM_COLUMNS["T"]]
GRID = c.prep422.GRID_START_US
STEP = c.prep422.STEP_US


class WtColumns:
    """Standardised `W-T` columns (real and placebo) for the month rows, the test rows, and pretraining rows."""

    def __init__(self, *, month: str, mr: c.MonthRows) -> None:
        """Load the windows, fit the standardisation on the fold's training rows, and build the month's columns."""
        refit = c.month_start(month=month)
        self.windows = f515.Windows(fit_until_idx=int((c.us(t=refit) - GRID) // STEP))
        origin_idx = ((mr.origin_us - GRID) // STEP).astype(np.int64)
        self.month_real = self.windows.real(
            kind="ens", primary=mr.primary, origin_idx=origin_idx, valid_idx=mr.target_idx
        )[:, T_INDEX]
        self.month_null = self.windows.shuffled(
            kind="ens", primary=mr.primary, origin_idx=origin_idx, valid_idx=mr.target_idx
        )[:, T_INDEX]
        train = ~mr.is_val[: mr.n_base]
        self.mean = self.month_real[: mr.n_base][train].mean(axis=0)
        self.std = np.maximum(self.month_real[: mr.n_base][train].std(axis=0), 1e-6)
        self.example = self._example(mr=mr, origin_idx=origin_idx)

    def _example(self, *, mr: c.MonthRows, origin_idx: np.ndarray) -> str:
        """Describe one training row: its origin, target, the run its weather came from, and its columns."""
        i = int(len(origin_idx) // 2)
        origin = datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC)
        valid = datetime.fromtimestamp(mr.valid_us[i] / 1e6, UTC)
        assert valid > origin
        return (
            f"origin {origin}, valid {valid}, window columns (6, 24, 72 h mean temperature) "
            f"{np.round(self.month_real[i], 2).tolist()} from runs issued at or before the origin (`previous` seam)"
        )

    def scale(self, *, raw: np.ndarray) -> np.ndarray:
        """Standardise raw columns with the fold's training statistics; missing values become 0."""
        out = (raw - self.mean) / self.std
        return np.where(np.isfinite(out), out, 0.0).astype(np.float32)

    def month(self, *, null: bool) -> np.ndarray:
        """Return the standardised columns of every month row."""
        return self.scale(raw=self.month_null if null else self.month_real)

    def rows(self, *, kind: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray, null: bool) -> np.ndarray:  # fmt: skip
        """Return the standardised columns of another row set (`kind` is `ens` for test rows, `era5` for pretraining rows)."""
        args = {"kind": kind, "primary": primary, "origin_idx": origin_idx, "valid_idx": valid_idx}
        raw = self.windows.shuffled(**args) if null else self.windows.real(**args)
        return self.scale(raw=raw[:, T_INDEX])

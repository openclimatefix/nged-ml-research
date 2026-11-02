# ruff: noqa: E501
"""Shared pieces of experiment 511: refit cadences, the month's rows on the GPU, and the mixed pretraining.

Experiment 510's harness (`e00510_common`) supplies the network, the loop, the row selection, and the
forecast frame. This module adds what 510 hard-codes: weekly cutoffs only. A cadence is `d` (daily,
every day at 00:00), `t` (twice weekly, days 0 and 3 of each 7-day block, so gaps alternate 3 and 4
days), or `w` (weekly, days 0, 7, 14, 21, 28). Every cadence's first cutoff is the month's refit, so
all cadences share the fold's pretrained weights, standardiser, and early-stopping weeks.

Each model scores only the forecast origins from its own cutoff to the next cutoff of the same
cadence (the last cutoff to the end of the month). The windows tile the month, so every origin has
exactly one scoring model.
"""

from datetime import datetime, timedelta
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining"
    / "work"
)
E510_WORK: Final = c.WORK_DIR
CADENCES: Final[tuple[str, ...]] = ("d", "t", "w")
TWICE_WEEKLY_OFFSETS: Final[tuple[int, ...]] = (0, 3)
WEEKLY_OFFSETS: Final[tuple[int, ...]] = (0, 7, 14, 21, 28)


def cutoffs(*, month: str, cadence: str) -> list[datetime]:
    """Return the month's refit times at 00:00 UTC for a cadence, all inside the month."""
    start, stop = c.month_start(month=month), c.month_stop(month=month)
    days = (stop - start).days
    if cadence == "d":
        offsets = list(range(days))
    elif cadence == "t":
        offsets = [7 * b + o for b in range(5) for o in TWICE_WEEKLY_OFFSETS]
    elif cadence == "w":
        offsets = list(WEEKLY_OFFSETS)
    else:
        raise ValueError(cadence)
    return [start + timedelta(days=d) for d in offsets if d < days]


def window_of(*, month: str, cadence: str, index: int) -> tuple[int, int]:
    """Return the scored origin window [start, stop) in microseconds of cutoff `index`."""
    cuts = cutoffs(month=month, cadence=cadence)
    stop = cuts[index + 1] if index + 1 < len(cuts) else c.month_stop(month=month)
    return c.us(t=cuts[index]), c.us(t=stop)


def build_month(*, month: str, clean: np.ndarray) -> c.MonthRows:
    """Return the month's rows extended to the last daily cutoff.

    Experiment 510's `build_month` stops at day 29.
    """
    refit = c.month_start(month=month)
    rows = data.build_fold(refit=refit, clean=clean)
    last = cutoffs(month=month, cadence="d")[-1]
    ext = c._extension(refit=refit, last=last, clean=clean, rows=rows)
    grid_us, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    return c.MonthRows(
        month=month,
        x=np.concatenate([rows.x, ext["x"]]),
        y=np.concatenate([rows.y, ext["y"].astype(np.float32)]),
        primary=np.concatenate([rows.primary, ext["primary"]]),
        lead=np.concatenate([rows.lead, ext["lead"]]),
        valid_us=np.concatenate([grid_us + rows.target_idx.astype(np.int64) * step, ext["valid"]]),
        origin_us=np.concatenate(
            [grid_us + rows.origin_idx.astype(np.int64) * step, ext["origin"]]
        ),
        target_idx=np.concatenate([rows.target_idx.astype(np.int64), ext["target_idx"]]),
        is_val=np.concatenate([rows.is_val, ext["is_val"]]),
        base=rows,
        n_base=len(rows.x),
    )


class GpuMonth:
    """A month's inputs, standardised targets, and Primary indices held on the GPU once."""

    def __init__(self, *, mr: c.MonthRows, device: str, flag: float | None = None) -> None:
        """Upload the month's rows; `flag` appends a constant source-flag column (L-mix uses 1.0)."""
        x = torch.from_numpy(mr.x)
        if flag is not None:
            x = torch.cat([x, torch.full((len(x), 1), flag, dtype=x.dtype)], dim=1)
        z = (mr.y - mr.base.target_mean[mr.primary]) / mr.base.target_std[mr.primary]
        self.x = x.to(device)
        self.z = torch.from_numpy(z.astype(np.float32)).to(device)
        self.p = torch.from_numpy(mr.primary).to(device)
        self.mr = mr
        self.device = device

    def refit_tensors(
        self, *, refit: c.Refit, shift_z: np.ndarray | None = None
    ) -> dict[str, torch.Tensor]:
        """Return the selected rows' tensors; `shift_z` (per selected row) is added to the standardised target."""
        idx = torch.from_numpy(refit.index).to(self.device)
        z = self.z[idx]
        if shift_z is not None:
            z = z + torch.from_numpy(shift_z.astype(np.float32)).to(self.device)
        return {
            "x": self.x[idx],
            "z": z,
            "w": torch.from_numpy(refit.weight).to(self.device),
            "p": self.p[idx],
            "v": torch.from_numpy(refit.is_val).to(self.device),
        }

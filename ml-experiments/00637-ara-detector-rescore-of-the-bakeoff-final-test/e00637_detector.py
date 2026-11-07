"""Experiment 637: the frozen abnormal-running-arrangement (ARA) day detector.

The detector reads only cleaned observed power. For each Primary and UTC day it takes the mean over
the day of observed power minus a reference, for two references that exist for all five bake-off
systems alike: the manual heuristic with all 13 members (power 1 to 6 and 49 to 55 weeks earlier)
and the climatology fitted to data before `CLIM_CUTOFF`. It subtracts the median across the 16
Primaries on the same day, averages the two residuals, standardises each Primary by its median and
1.4826 times its median absolute deviation over the development year, flags days with |z| above
`k`, bridges one-day gaps, and keeps runs of at least `n` days. It never reads a network forecast.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00620_score as s  # ty: ignore[unresolved-import]
import numpy as np

PER_DAY: Final[int] = 48
CLIM_CUTOFF: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The climatology reference sees no power from this day on, so development and window days are both out of sample."""
DEV_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
DEV_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Development days are DEV_START up to but excluding DEV_END; the window starts at DEV_END."""
WINDOW_END: Final[datetime] = datetime(2026, 11, 1, tzinfo=UTC)
RULE_NAME: Final[str] = "ara_rule.json"


def day_of(when: datetime) -> int:
    """Return the day number since the grid start."""
    return int((when - s.START_US_DT) / (s.STEP_DT * PER_DAY))


def _flat(*, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = clean.shape[1] // PER_DAY * PER_DAY
    return np.repeat(np.arange(clean.shape[0]), n), np.tile(np.arange(n), clean.shape[0])


def climatology_reference(*, clean: np.ndarray) -> np.ndarray:
    """Return the (16, n) median climatology fitted to data before CLIM_CUTOFF."""
    prim, idx = _flat(clean=clean)
    cut = int((CLIM_CUTOFF - s.START_US_DT) / s.STEP_DT)
    ref = s.climatology(clean=clean, prim=prim, idx=idx, cutoff_idx=cut)[:, 5]
    return ref.reshape(clean.shape[0], -1)


def daily_residuals(*, clean: np.ndarray, ref_clim: np.ndarray) -> np.ndarray:
    """Return (2, 16, n_days): daily mean of observed minus the heuristic, and minus the climatology."""
    prim, idx = _flat(clean=clean)
    ref_h = s.manual_heuristic(clean=clean, prim=prim, idx=idx, lead_h=np.zeros(len(idx)))[:, 5]
    ref_h = ref_h.reshape(clean.shape[0], -1)
    obs = clean[:, : ref_h.shape[1]]
    out = []
    for ref in (ref_h, ref_clim):
        with np.errstate(all="ignore"):
            out.append(np.nanmean((obs - ref).reshape(clean.shape[0], -1, PER_DAY), axis=2))
    return np.stack(out).astype(np.float32)


def combine(*, daily: np.ndarray) -> np.ndarray:
    """Return the (16, n_days) average of the two cross-Primary-centred daily residuals."""
    centred = daily - np.nanmedian(daily, axis=1, keepdims=True)
    return np.nanmean(centred, axis=0)


def scale(*, resid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-Primary centre and 1.4826 x MAD over the development year."""
    sl = slice(day_of(DEV_START), day_of(DEV_END))
    centre = np.nanmedian(resid[:, sl], axis=1)
    mad = np.nanmedian(np.abs(resid[:, sl] - centre[:, None]), axis=1) * 1.4826
    return centre, mad


def flag_days(
    *, resid: np.ndarray, centre: np.ndarray, mad: np.ndarray, k: float, n: int
) -> np.ndarray:
    """Return the (16, n_days) boolean mask of days inside kept runs."""
    z = (resid - centre[:, None]) / mad[:, None]
    flag = np.nan_to_num(np.abs(z), nan=0.0) > k
    keep = np.zeros_like(flag)
    for p in range(flag.shape[0]):
        f = flag[p].copy()
        f[1:-1] |= f[:-2] & f[2:]
        i = 0
        while i < len(f):
            if f[i]:
                j = i
                while j < len(f) and f[j]:
                    j += 1
                if j - i >= n:
                    keep[p, i:j] = True
                i = j
            else:
                i += 1
    return keep


def load_rule(path: Path) -> dict:
    return json.loads(path.read_text())

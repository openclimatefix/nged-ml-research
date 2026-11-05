# ruff: noqa: E501, ANN001
"""Experiment 568: the relative-spread block appended to a row set's 37 columns.

Seven columns per row (see `e00568_model`): the standard deviation across the 51 ensemble members of
2 m temperature, 10 m wind speed, shortwave radiation, the u and v components of 10 m wind, and total
cloud cover, each divided by that variable's median over the fold's training rows of the same
Primary, plus a flag that is 1 where the run has no cloud cover file (runs before 2025-11-21).

- 2 m temperature, wind speed, and shortwave: experiment 397's `std_*` columns, un-standardised with
  the fold's `Standardiser` (the five-standard-deviation clip leaves the bulk untouched).
- u and v: the `std_wind_u_10m` and `std_wind_v_10m` columns of experiment 559's `UvSource`.
- Cloud: the `std_total_cloud_cover` column of experiment 559's `CloudSource` (percent / 100).

Every column of a row comes from the ensemble run keyed by that row's origin (the run issued at or
before the origin), found by the (Primary, origin, valid) key; a row with no key raises.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00559_extra as extra  # ty: ignore[unresolved-import]
import numpy as np

RATIO_CLIP: Final[float] = 10.0
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
"""Hours from an ECMWF run's initialisation to the forecast origin that may use it (experiment 87's constant)."""
STEPS_PER_DAY: Final[int] = 48
NAMES: Final[tuple[str, ...]] = (
    "std_temperature_2m",
    "std_wind_speed_10m",
    "std_shortwave",
    "std_wind_u_10m",
    "std_wind_v_10m",
    "std_total_cloud_cover",
    "cloud_missing",
)
_, COLUMNS, _ = data.non_lag_layout()
STORED: Final[list[int]] = [COLUMNS.index(n) for n in NAMES[:3]]
_UV: extra.UvSource | None = None
_CLOUD: extra.CloudSource | None = None


def sources() -> tuple[extra.UvSource, extra.CloudSource]:
    """Return the wind-component and cloud sources, loaded once per process."""
    global _UV, _CLOUD  # noqa: PLW0603
    if _UV is None:
        _UV = extra.UvSource()
        _CLOUD = extra.CloudSource()
    assert _CLOUD is not None
    return _UV, _CLOUD


def raw_block(
    *,
    x37: np.ndarray,
    standardiser,
    primary: np.ndarray,
    origin_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> np.ndarray:
    """Return the 7 raw columns (standard deviations, then the cloud flag) of a row set."""
    uv, cloud = sources()
    out = np.empty((len(x37), 7), dtype=np.float32)
    for j, col in enumerate(STORED):
        out[:, j] = x37[:, col] * standardiser.std[col] + standardiser.mean[col]
    vals, found = uv.ens(primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
    assert found.all(), f"{int((~found).sum())} rows have no u/v summary"
    out[:, 3:5] = vals[:, 4:6]
    cvals, _ = cloud.ens(primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
    out[:, 5] = cvals[:, 2]
    out[:, 6] = cvals[:, 1]
    return out


class SpreadBlock:
    """The fold's per-Primary medians, and the block for any row set of the fold."""

    def __init__(self, *, month: str, mr, placebo: bool) -> None:
        """Compute the medians on the fold's training rows (not the early-stopping weeks)."""
        self.placebo = placebo
        n = mr.n_base
        train = ~mr.is_val[:n]
        origin_idx = extra.grid_index(us=mr.origin_us)
        raw = raw_block(x37=mr.x[:, :37], standardiser=mr.base.standardiser, primary=mr.primary,
                        origin_idx=origin_idx, valid_idx=mr.target_idx)  # fmt: skip
        n_primaries = len(mr.base.target_mean)
        self.median = np.ones((n_primaries, 6), dtype=np.float64)
        for p in range(n_primaries):
            sel = train & (mr.primary[:n] == p)
            for j in range(6):
                v = raw[:n, j][sel]
                if j == 5:
                    v = v[raw[:n, 6][sel] == 0]
                v = v[np.isfinite(v) & (v > 0)]
                self.median[p, j] = float(np.median(v)) if len(v) else 1.0
        self.month = month
        self.example: str = ""

    def relative(self, *, raw: np.ndarray, primary: np.ndarray) -> np.ndarray:
        """Divide each standard deviation by its Primary's median, clip, and zero the missing cloud."""
        out = raw.copy()
        out[:, :6] = np.clip(raw[:, :6] / self.median[primary].astype(np.float32), 0.0, RATIO_CLIP)
        out[raw[:, 6] > 0, 5] = 0.0
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    def block(
        self, *, x37: np.ndarray, standardiser, primary: np.ndarray, origin_idx: np.ndarray,
        valid_idx: np.ndarray, describe: bool = False,
    ) -> np.ndarray:  # fmt: skip
        """Return the 7-column block of a row set (the shuffled one for the placebo arm)."""
        raw = raw_block(x37=x37, standardiser=standardiser, primary=primary,
                        origin_idx=origin_idx, valid_idx=valid_idx)  # fmt: skip
        rel = self.relative(raw=raw, primary=primary)
        if self.placebo:
            rel = self.shuffle(rel=rel, primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
        if describe:
            i = len(rel) // 2
            o = datetime.fromtimestamp((extra.GRID + int(origin_idx[i]) * extra.STEP) / 1e6, UTC)
            v = datetime.fromtimestamp((extra.GRID + int(valid_idx[i]) * extra.STEP) / 1e6, UTC)
            assert v > o
            self.example = (
                f"run {o - PUBLICATION_DELAY}, origin {o}, valid {v}, Primary {int(primary[i])}, spread block "
                f"{np.round(rel[i], 2).tolist()} from the ensemble run keyed by this origin"
                f"{' (shuffled between days of the same Primary and month)' if self.placebo else ''}"
            )
        return rel

    @staticmethod
    def shuffle(
        *, rel: np.ndarray, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Give each row the block of the same Primary, time of day, and lead on another origin day.

        The donor origin day is in the same month.
        """
        day = origin_idx // STEPS_PER_DAY
        month_id = np.array(
            [
                datetime.fromtimestamp(
                    (extra.GRID + int(d) * STEPS_PER_DAY * extra.STEP) / 1e6, UTC
                ).strftime("%Y-%m")
                for d in np.unique(day)
            ]
        )
        month_of = dict(zip(np.unique(day).tolist(), month_id.tolist(), strict=True))
        rng = np.random.default_rng(568)
        key = origin_idx.astype(np.int64) * (1 << 22) + valid_idx.astype(np.int64)
        by_primary = {}
        for p in np.unique(primary):
            sel = np.nonzero(primary == p)[0]
            order = np.argsort(key[sel])
            by_primary[int(p)] = (sel, key[sel][order], rel[sel][order])
        out = rel.copy()
        todo = np.ones(len(rel), dtype=bool)
        for _attempt in range(8):  # a row whose donor row was filtered out tries another donor day
            shift_day = {}
            for mid in np.unique(month_id):
                days = np.array(sorted(d for d, mm in month_of.items() if mm == mid))
                if len(days) > 1:  # a cyclic shift by a random offset leaves no day on itself
                    donors = np.roll(days, int(rng.integers(1, len(days))))
                    shift_day.update(zip(days.tolist(), (donors - days).tolist(), strict=True))
                else:
                    shift_day[int(days[0])] = 0
            shift = np.array([shift_day[int(d)] for d in day], dtype=np.int64) * STEPS_PER_DAY
            want = (origin_idx + shift).astype(np.int64) * (1 << 22) + (valid_idx + shift).astype(
                np.int64
            )
            for sel, keys, vals in by_primary.values():
                rows = sel[todo[sel]]
                if len(rows) == 0:
                    continue
                pos = np.clip(np.searchsorted(keys, want[rows]), 0, len(keys) - 1)
                hit = (keys[pos] == want[rows]) & (shift[rows] != 0)
                out[rows[hit]] = vals[pos[hit]]
                todo[rows[hit]] = False
        missing = int(todo.sum())
        assert missing < 0.02 * len(rel), (
            f"{missing} of {len(rel)} rows have no donor day after 8 tries"
        )
        for p in np.unique(primary):
            sel = primary == p
            own = (out[sel] == rel[sel]).all(axis=1).mean()
            assert own < 0.10, (
                f"Primary {p}: {own:.1%} of rows kept their own spread in the placebo"
            )
        return out

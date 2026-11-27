# ruff: noqa: E501
"""Experiment 733: the references of the two worlds.

Real folds: the stored forecasts of M* (experiment 713's arm `M1`) are the counted control of C1, read
from experiment 713's files (read only), and the F0r gate that those stored forecasts must be
reproduced by this experiment's loop with every new channel absent.

Synthetic world: the three rules that give P(X<0) for a target half-hour without a forecaster fit.

- B-hist, the persistence rule and the counted reference of C3: the weighted mean of the history rows'
  untangled P(X<0) at the same column, with the weights of experiment 713's starting day-type table
  through a softmax over the rows that hold a value. A column with no history value takes B-clim.
- B-clim: the training banks' share of reverse flow by local half-hour, weekday or weekend, and
  calendar month, written to `limits.json` before any fit.
- O: the untangling network's P(X<0) on the target day itself. O reads the target day's reading and
  observed weather, so it is a ceiling and no forecast.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00733_core as core
import e00733_paths as paths  # noqa: F401
import e00733_synth as sy
import numpy as np
import polars as pl
from e00733_channels import StoppedError

KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time")
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
N_MONTHS: Final = 12
WEEKEND_DAY_TYPES: Final = (5, 6, 7)
"""Saturday, Sunday, and a bank holiday (day type 7) form the second day class of B-clim."""
D8_LABEL: Final = "D8"


# --------------------------------------------------------------------------- real folds: M* and the F0r gate


def d8_series_id() -> int:
    """Return D8's `time_series_id` from experiment 578's site table."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sites = k578.sites().filter(pl.col("label") == D8_LABEL)
    if sites.height != 1:
        msg = "the site table does not hold exactly one D8"
        raise StoppedError(msg)
    return int(sites["time_series_id"][0])


def m_star_file(*, month: str, seed: int) -> Path:
    """Return the stored forecast file of M* (`M1`) for one fold and seed (experiment 713's data, read only)."""
    import e00713_arms as arms713  # ty: ignore[unresolved-import]
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    return arms713.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="M1", month=month, seed=seed)


def read_m_star(*, month: str, seed: int, path: Path | None = None, series_id: int | None = None):  # noqa: ANN201
    """Return M*'s stored forecasts of one fold and seed at D8 (native origin, the file's own columns).

    Args:
        month: The fold.
        seed: The seed.
        path: The file (default: experiment 713's stored file).
        series_id: D8's id (default: read from the site table).

    Returns:
        A frame with `KEYS` and `QCOLS`, sorted by the keys.

    Raises:
        StoppedError: When the file is missing or holds no D8 row (the control would be absent).
    """
    file = path if path is not None else m_star_file(month=month, seed=seed)
    if not file.exists():
        msg = f"the stored forecast of M* for {month} seed {seed} is missing (verdict row 0)"
        raise StoppedError(msg)
    sid = d8_series_id() if series_id is None else series_id
    frame = (
        pl.read_parquet(file, columns=[*KEYS, *QCOLS])
        .filter(pl.col("time_series_id") == sid)
        .sort(*KEYS)
    )
    if frame.is_empty():
        msg = "the stored forecast of M* holds no D8 row"
        raise StoppedError(msg)
    return frame


def f0r_gate(
    *, f0r: pl.DataFrame, stored: pl.DataFrame, tolerance: float = core.TOLERANCE_F0R
) -> dict[str, float | int]:
    """Check that F0r reproduces M*'s stored forecasts at D8 to `tolerance` of capacity.

    This is the gate of verdict row 0.

    Args:
        f0r: F0r's forecast frame of the fold and seed (all rows or D8 rows; D8 is selected
            from the keys of `stored`).
        stored: M*'s stored D8 frame (`read_m_star`).
        tolerance: The largest allowed difference of any quantile.

    Returns:
        The number of rows compared and the largest difference.

    Raises:
        StoppedError: When the keys differ, no row is compared, or a quantile differs by more than `tolerance`.
    """
    ids = stored["time_series_id"].unique()
    mine = f0r.filter(pl.col("time_series_id").is_in(ids)).sort(*KEYS)
    if mine.height == 0 or mine.height != stored.height:
        msg = "F0r and the stored M* forecast hold different numbers of D8 rows (verdict row 0)"
        raise StoppedError(msg)
    if not mine.select(*KEYS).equals(stored.select(*KEYS)):
        msg = "F0r and the stored M* forecast hold different keys (verdict row 0)"
        raise StoppedError(msg)
    worst = max(float((mine[c] - stored[c]).abs().max()) for c in QCOLS)  # ty: ignore[invalid-argument-type]
    if not worst <= tolerance:
        msg = f"F0r differs from M* by up to {worst:.2e} of capacity (verdict row 0)"
        raise StoppedError(msg)
    return {"rows": mine.height, "largest_difference": worst}


# --------------------------------------------------------------------------- synthetic world: B-clim


@dataclass(frozen=True)
class Clim:
    """B-clim: the share of reverse flow by local half-hour, day class, and calendar month.

    Attributes:
        share: `(48, 2, 12)` the share (NaN where the cell holds no half-hour).
        count: `(48, 2, 12)` the number of half-hours behind each cell.
        overall: The share over every half-hour.
    """

    share: np.ndarray
    count: np.ndarray
    overall: float

    def lookup(self, *, hh: np.ndarray, weekend: np.ndarray, month: np.ndarray) -> np.ndarray:
        """Return B-clim at the keys (month 1 to 12).

        An empty cell takes the share of its half-hour and day class, then the overall
        share.
        """
        s = self.share[hh, weekend.astype(int), month - 1]
        by_hh = self.pooled_by_half_hour()[hh, weekend.astype(int)]
        return np.where(np.isfinite(s), s, np.where(np.isfinite(by_hh), by_hh, self.overall))

    def pooled_by_half_hour(self) -> np.ndarray:
        """Return `(48, 2)` the share over every month (NaN for an empty pair)."""
        total = (np.nan_to_num(self.share) * self.count).sum(axis=2)
        n = self.count.sum(axis=2)
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON content of `limits.json`'s `b_clim`."""
        return {
            "share": np.nan_to_num(self.share, nan=-1.0).round(6).tolist(),
            "count": self.count.tolist(), "overall": self.overall,
        }  # fmt: skip


def fit_clim(*, segments: Sequence[sy.SegmentArrays]) -> Clim:
    """Fit B-clim on the training banks' segments.

    Args:
        segments: The training segments (the banks of seeds 3 and 4).

    Returns:
        The table.

    Raises:
        StoppedError: When no half-hour holds a sign (nothing to count).
    """
    pos = np.zeros((core.STEPS_PER_DAY, 2, N_MONTHS))
    cnt = np.zeros_like(pos)
    for seg in segments:
        x = seg.x_neg
        ok = np.isfinite(x)
        month = np.repeat(seg.month, sy.STEPS)
        weekend = np.isin(seg.day_type, WEEKEND_DAY_TYPES).astype(int)
        np.add.at(cnt, (seg.local_hh[ok], weekend[ok], month[ok] - 1), 1.0)
        np.add.at(pos, (seg.local_hh[ok], weekend[ok], month[ok] - 1), x[ok])
    if cnt.sum() == 0:
        msg = "no synthetic half-hour holds a sign, so B-clim cannot be fitted"
        raise StoppedError(msg)
    share = np.where(cnt > 0, pos / np.maximum(cnt, 1), np.nan)
    return Clim(share=share, count=cnt.astype(np.int64), overall=float(pos.sum() / cnt.sum()))


def clim_for_rows(
    *, clim: Clim, segments: Sequence[sy.SegmentArrays], index: sy.SampleIndex, rows: np.ndarray
) -> np.ndarray:
    """Return B-clim at every target half-hour of the samples `rows`, `(len(rows), 48)`."""
    out = np.zeros((len(rows), core.STEPS_PER_DAY))
    for n, r in enumerate(rows):
        seg = segments[int(index.seg[r])]
        t = sy.target_slots(
            origin_day=index.origin_day[r : r + 1], lead_index=index.lead_index[r : r + 1]
        )[0]
        out[n] = clim.lookup(
            hh=seg.local_hh[t],
            weekend=np.isin(seg.day_type[t], WEEKEND_DAY_TYPES),
            month=seg.month[t // sy.STEPS],
        )
    return out


# --------------------------------------------------------------------------- synthetic world: B-hist and O


def day_type_table() -> np.ndarray:
    """Return experiment 713's starting day-type prior `(8, 8)` (3.0 for the same type.

    1.5 for two working days).
    """
    import e00713_model as mdl  # ty: ignore[unresolved-import]

    return mdl.beta_day_start().detach().cpu().numpy().astype(np.float64)


def b_hist(
    *, p_hist: np.ndarray, ok: np.ndarray, type_top: np.ndarray, type_hist: np.ndarray,
    clim: np.ndarray, table: np.ndarray | None = None,
) -> np.ndarray:  # fmt: skip
    """Return B-hist, the persistence rule, for a batch.

    Args:
        p_hist: `(B, 14, 48)` the history's untangled P(X<0).
        ok: `(B, 14, 48)` whether the history value exists (visible, present, flagged).
        type_top: `(B, 48)` the day type of each target half-hour.
        type_hist: `(B, 14, 48)` the day type of each history half-hour.
        clim: `(B, 48)` B-clim, the value where no history row holds a value.
        table: The day-type prior (default: experiment 713's starting table).

    Returns:
        `(B, 48)` the probability. A weight is the exponential of the prior over the rows that hold a
        value (a softmax over rows), so a column with one value takes that value.
    """
    tab = day_type_table() if table is None else table
    logit = tab[type_top[:, None, :], type_hist]
    logit = np.where(ok, logit, -np.inf)
    any_ok = ok.any(axis=1)
    shifted = (
        logit - np.where(any_ok, np.max(np.where(ok, logit, -np.inf), axis=1), 0.0)[:, None, :]
    )
    w = np.where(ok, np.exp(shifted), 0.0)
    total = w.sum(axis=1)
    mean = (w * np.nan_to_num(p_hist)).sum(axis=1) / np.maximum(total, 1e-300)
    return np.where(any_ok, mean, clim)


def b_hist_for_batch(
    *, batch: sy.SynthBatch, clim: np.ndarray, table: np.ndarray | None = None
) -> np.ndarray:
    """Return B-hist for a synthetic batch gathered in the `untangled` mode (its channels and flags)."""
    a = batch.arrays
    return b_hist(
        p_hist=a["hist_u"][..., core.P_NEG], ok=a["hist_u_ok"], type_top=a["day_type_top"],
        type_hist=a["day_type_hist"], clim=clim, table=table,
    )  # fmt: skip


def o_for_rows(
    *, segments: Sequence[sy.SegmentArrays], index: sy.SampleIndex, rows: np.ndarray
) -> np.ndarray:
    """Return O, the untangling network's P(X<0) on the target day itself `(len(rows), 48)` (NaN where absent).

    O reads the target day's reading, so it is a ceiling and never a forecast.
    """
    out = np.full((len(rows), core.STEPS_PER_DAY), np.nan)
    for n, r in enumerate(rows):
        seg = segments[int(index.seg[r])]
        t = sy.target_slots(
            origin_day=index.origin_day[r : r + 1], lead_index=index.lead_index[r : r + 1]
        )[0]
        out[n] = np.where(seg.u_ok[t], seg.u_raw[t, core.P_NEG], np.nan)
    return out


def brier(*, p: np.ndarray, truth: np.ndarray) -> float:
    """Return the Brier score over the half-hours where the truth and the probability exist."""
    ok = np.isfinite(truth) & np.isfinite(p)
    if not ok.any():
        msg = "no half-hour holds both a probability and a truth, so no Brier score exists"
        raise StoppedError(msg)
    return float(np.mean((p[ok] - truth[ok]) ** 2))


def brier_skill(*, p: np.ndarray, reference: np.ndarray, truth: np.ndarray) -> float:
    """Return `1 - Brier(p) / Brier(reference)`, the skill against B-clim."""
    return 1.0 - brier(p=p, truth=truth) / brier(p=reference, truth=truth)

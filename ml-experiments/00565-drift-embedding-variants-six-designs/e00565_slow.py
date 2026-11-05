# ruff: noqa: E501
"""Experiment 565, design 2: the slow-trend summary input (a slow lag).

For each row, per-Primary trailing means and slopes of the cleaned target (capacity fraction) over
90, 180, and 365 days, each window ending `MIN_AGE_STEPS` half-hours (24 h) before the row's
forecast origin. The same rule holds for training rows, pretraining rows, and test rows, and every
call asserts it. The 6 columns are standardised per Primary on the fold's training rows.

The slope of a window is the mean of its second half minus the mean of its first half, per day.

The placebo (`null`) replaces a row's summaries by those of a donor end time 30 x k days earlier, with
k drawn from 1 to 24 per Primary and per block of 30 days of origin (fixed seed). A donor end time is
always before the row's own end time, so the placebo cannot carry later information.
"""

from datetime import UTC, datetime
from typing import Final

import e00510_common as c  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

GRID: Final[int] = c.prep422.GRID_START_US
STEP: Final[int] = c.prep422.STEP_US
STEPS_PER_DAY: Final[int] = 48
MIN_AGE_STEPS: Final[int] = STEPS_PER_DAY
"""A window ends at least 24 h (48 half-hours) before the row's origin."""
WINDOWS_DAYS: Final[tuple[int, ...]] = (90, 180, 365)
N_COLUMNS: Final[int] = 2 * len(WINDOWS_DAYS)
MIN_COVERAGE: Final[float] = 0.5
"""A window with fewer observed half-hours than this share gives no value (filled with the Primary's mean)."""
DONOR_MONTHS: Final[int] = 24
BLOCK_STEPS: Final[int] = 30 * STEPS_PER_DAY


def _mean(
    *, cs: np.ndarray, cn: np.ndarray, primary: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Return the mean of the observed values in the inclusive index range [lo, hi] (NaN when under-covered)."""
    lo = np.maximum(lo, 0)
    s = cs[primary, hi + 1] - cs[primary, lo]
    n = cn[primary, hi + 1] - cn[primary, lo]
    width = hi - lo + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= MIN_COVERAGE * width, s / np.maximum(n, 1), np.nan)


class SlowTrendColumns:
    """The 6 slow-trend columns, with the interface of `DriftColumns` (month rows, other row sets, placebo)."""

    def __init__(self, *, month: str, mr: c.MonthRows, clean: np.ndarray) -> None:
        """Build the cumulative sums, the placebo shifts, and the per-Primary standardisation."""
        self.clean = clean
        obs = np.isfinite(clean)
        self.cs = np.concatenate(
            [np.zeros((len(clean), 1)), np.cumsum(np.where(obs, clean, 0.0), axis=1)], axis=1
        )
        self.cn = np.concatenate(
            [np.zeros((len(clean), 1)), np.cumsum(obs, axis=1)], axis=1
        ).astype(np.int64)
        self.n_primaries = len(clean)
        rng = np.random.default_rng(0)
        self.donor_k = rng.integers(1, DONOR_MONTHS + 1, size=(self.n_primaries, 4096))
        self.null_rows_without_donor = 0
        self.rows_with_value = None
        self.max_end_gap = None
        self.last_end_minus_origin = 0
        origin_idx = ((mr.origin_us - GRID) // STEP).astype(np.int64)
        self.fit_until = int((c.us(t=c.month_start(month=month)) - GRID) // STEP)
        raw = self._raw(primary=mr.primary, origin_idx=origin_idx, null=False)
        train = ~mr.is_val[: mr.n_base]
        self.mean = np.zeros((self.n_primaries, N_COLUMNS), dtype=np.float64)
        self.std = np.ones((self.n_primaries, N_COLUMNS), dtype=np.float64)
        for p in range(self.n_primaries):
            sel = (mr.primary[: mr.n_base] == p) & train
            vals = raw[: mr.n_base][sel]
            if len(vals) > 10:
                self.mean[p] = np.nanmean(vals, axis=0)
                self.std[p] = np.maximum(np.nanstd(vals, axis=0), 1e-6)
        self.month_raw = raw
        self.month_origin_idx = origin_idx
        self.month_primary = mr.primary
        i = len(origin_idx) // 2
        self.example = (
            f"origin {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC)}; windows end at "
            f"{datetime.fromtimestamp((GRID + (origin_idx[i] - MIN_AGE_STEPS) * STEP) / 1e6, UTC)} (24 h earlier); "
            f"columns {np.round(self.scale(raw=raw[i : i + 1], primary=mr.primary[i : i + 1])[0], 2).tolist()}"
        )

    def _raw(self, *, primary: np.ndarray, origin_idx: np.ndarray, null: bool) -> np.ndarray:
        """Return the unstandardised columns (NaN where a window is under-covered); asserts the 24 h rule."""
        end = origin_idx - MIN_AGE_STEPS
        if null:
            block = (origin_idx // BLOCK_STEPS) % self.donor_k.shape[1]
            end = end - self.donor_k[primary, block] * BLOCK_STEPS
        assert (end <= origin_idx - MIN_AGE_STEPS).all(), (
            "a summary window ends inside 24 h of its origin"
        )
        self.last_end_minus_origin = int((end - origin_idx).max())
        end = np.minimum(end, self.clean.shape[1] - 1)
        cols = []
        for days in WINDOWS_DAYS:
            n = days * STEPS_PER_DAY
            half = n // 2
            whole = _mean(cs=self.cs, cn=self.cn, primary=primary, lo=end - n + 1, hi=end)
            first = _mean(cs=self.cs, cn=self.cn, primary=primary, lo=end - n + 1, hi=end - half)
            second = _mean(cs=self.cs, cn=self.cn, primary=primary, lo=end - half + 1, hi=end)
            cols += [whole, (second - first) / (half / STEPS_PER_DAY)]
        return np.stack(cols, axis=1)

    def scale(self, *, raw: np.ndarray, primary: np.ndarray) -> np.ndarray:
        """Return standardised columns; NaN becomes 0 (the Primary's mean)."""
        z = (raw - self.mean[primary]) / self.std[primary]
        return np.nan_to_num(z, nan=0.0).astype(np.float32)

    def month(self, *, null: bool) -> np.ndarray:
        """Return the columns of every month row."""
        raw = (
            self._raw(primary=self.month_primary, origin_idx=self.month_origin_idx, null=null)
            if null
            else self.month_raw
        )
        return self.scale(raw=raw, primary=self.month_primary)

    def rows(
        self,
        *,
        kind: str,
        primary: np.ndarray,
        origin_idx: np.ndarray,
        valid_idx: np.ndarray,
        null: bool,
    ) -> np.ndarray:
        """Return the columns of another row set (`ens` test rows or `era5` pretraining rows)."""
        assert (valid_idx > origin_idx).all(), f"{kind}: a target is not after its origin"
        raw = self._raw(
            primary=primary.astype(np.int64), origin_idx=origin_idx.astype(np.int64), null=null
        )
        return self.scale(raw=raw, primary=primary.astype(np.int64))


class ModularMdnSlow(m.ModularMdn):
    """`ModularMdn` whose mixers also read the 6 slow-trend columns (37 to 42)."""

    def __init__(self, *, n_primaries: int, stage: str) -> None:
        """Widen both mixers by `N_COLUMNS`."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        wide = 2 * m.ENCODER_OUT + m.EMBEDDING + N_COLUMNS
        self.mixer_pre = m.base.mlp(
            n_in=wide, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT
        )
        self.mixer_ft = m.base.mlp(
            n_in=wide + 7 + 2,
            hidden=m.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        for head in (self.mixer_pre, self.mixer_ft):
            m.base.init_head(head=head, components=m.COMPONENTS)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb, x[:, m.N_INPUTS : m.N_INPUTS + N_COLUMNS]], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))

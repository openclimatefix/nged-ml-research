# ruff: noqa: E501
"""The shuffled-day placebo of experiment 515: donor days, whole-day shifts, and window columns.

A local valid day (Europe/London midnight to midnight, 46, 48, or 50 half-hours) takes a donor day
with the same calendar month, day type (Monday to Friday, Saturday, Sunday), and length, at least
`MIN_SEPARATION_DAYS` days away, without replacement, never itself, and on the same side of the
fold's refit. The draw is seeded and the same at training and scoring. A row takes the window
columns of the donor day at the same half-hour of day and the same lead: its origin and valid time
both move by the whole-day shift between the two days. Three independent draws give fallback donors
for rows whose shifted run is missing.
"""

from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SEEDS: Final[tuple[int, ...]] = (515, 516, 517)
MIN_SEPARATION_DAYS: Final[int] = 3
STEPS_PER_DAY: Final[int] = 48
TRIES: Final[int] = 200


class DayStructure:
    """Local valid days of the whole grid and their placebo draws for one fold."""

    def __init__(self, *, fit_until_idx: int, allowed: tuple[int, int] | None = None) -> None:
        """Build the days and three donor-shift draws.

        Args:
            fit_until_idx: Grid index of the fold's refit; days starting at or after it are one
                side, earlier days the other.
            allowed: Grid index range `[low, high)` of days that may send or receive a donor, or
                None for every day. Ensemble rows use the span that holds ensemble runs.
        """
        times = common.grid_times().dt.replace_time_zone("UTC").dt.convert_time_zone("Europe/London")
        date = times.dt.date().cast(pl.Int32).to_numpy()
        _, self.day_id = np.unique(date, return_inverse=True)
        n_days = int(self.day_id.max()) + 1
        self.first = np.searchsorted(self.day_id, np.arange(n_days), side="left")
        self.length = np.bincount(self.day_id, minlength=n_days)
        first_date = date[self.first]
        month = first_date.astype("datetime64[D]").astype("datetime64[M]").astype(int) % 12
        weekday = (first_date + 3) % 7  # 1970-01-01 is a Thursday; Monday is 0
        kind = np.where(weekday < 5, 0, weekday - 4)
        side = self.first >= fit_until_idx
        self.in_range = (
            np.ones(n_days, dtype=bool)
            if allowed is None
            else (self.first >= allowed[0]) & (self.first < allowed[1])
        )
        self.shift = np.full((len(SEEDS), n_days), np.nan)
        self.unmatched = []
        self.widened = []
        for k, seed in enumerate(SEEDS):
            donor = self._draw(
                seed=seed, month=month, kind=kind, side=side, first_date=first_date
            )
            ok = donor >= 0
            self.widened.append(int(self.last_widened))
            self.shift[k, ok] = (
                np.round((self.first[donor[ok]] - self.first[ok]) / STEPS_PER_DAY) * STEPS_PER_DAY
            )
            self.unmatched.append(int((~ok).sum()))

    def _draw(
        self, *, seed: int, month: np.ndarray, kind: np.ndarray, side: np.ndarray, first_date: np.ndarray
    ) -> np.ndarray:
        """Return each day's donor day (-1 for none), shuffling within groups until donors are far enough."""
        rng = np.random.default_rng(seed)
        donor = np.full(len(self.first), -1, dtype=np.int64)
        self.last_widened = 0
        # Pass 0 shuffles within a calendar month; days left without a donor (a side with a single
        # such month day) widen to two-month, three-month, and then any-month groups.
        for widening, bucket in enumerate(
            (lambda m: m, lambda m: ((m + 1) // 2) % 6, lambda m: m // 3, lambda m: 0 * m)
        ):
            todo = (donor < 0) & self.in_range
            if not todo.any():
                break
            groups: dict[tuple, list[int]] = {}
            for d in np.nonzero(todo)[0]:
                key = (int(bucket(month[d])), int(kind[d]), int(self.length[d]), bool(side[d]))
                groups.setdefault(key, []).append(int(d))
            for key in sorted(groups):
                members = np.array(groups[key])
                if len(members) < 2:
                    continue
                best, best_bad = None, None
                for _ in range(TRIES):
                    perm = rng.permutation(members)
                    candidate = np.roll(perm, -1)
                    bad = np.abs(first_date[perm] - first_date[candidate]) < MIN_SEPARATION_DAYS
                    if best is None or bad.sum() < best_bad:
                        best, best_bad = (perm, candidate, bad), int(bad.sum())
                    if best_bad == 0:
                        break
                perm, candidate, bad = best  # ty: ignore[not-iterable]
                good = ~bad
                donor[perm[good]] = candidate[good]
                if widening:
                    self.last_widened += int(good.sum())
        return donor

    def shifts(self, *, valid_idx: np.ndarray, attempt: int) -> np.ndarray:
        """Return each row's whole-day shift in half-hours for a draw (NaN where it has no donor)."""
        return self.shift[attempt][self.day_id[valid_idx]]

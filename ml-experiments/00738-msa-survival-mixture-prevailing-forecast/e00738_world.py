"""The injected worlds: shift draws, the shift added to inputs and target, and the strata.

The generator copies experiment 650's draw (`e00650_bank.draw`), which is not edited. Two changes:
the mirrored shifts at neighbouring Primaries are left out (the forecaster reads only its own
Primary's power), and each shift keeps its drawn duration before the window clips it. A training
world is a renewal process of shifts per Primary, drawn from the seed of the fit and the pass. A
scoring world holds one shift per Primary per pass, drawn by experiment 650's own seed key.

Every index is a half-hour of the grid of experiment 713 (48 a day). The contract origin is at 12:00
UTC; a history row of age `a` covers the 24 hours that start at 12:00 UTC `a` days before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import e00738_common as common
import numpy as np

PER_DAY: Final[int] = 48
DEPTH_RANGE: Final[tuple[float, float]] = (0.03, 0.50)
SHORT_DAYS: Final[tuple[int, int]] = (3, 14)
LONG_DAYS: Final[tuple[int, int]] = (28, 112)
TRAIN_RAMP_HOURS: Final[tuple[int, int]] = (6, 48)
HELD_RAMP_HOURS: Final[tuple[int, int]] = (72, 120)
HELD_DEPTHS: Final[tuple[float, ...]] = (0.10, 0.20, 0.40, 0.65)
HELD_DAYS: Final[tuple[int, ...]] = (5, 10, 21, 45)
STRATA: Final[tuple[str, ...]] = (
    "far",
    "middle",
    "near",
    "ended_24h",
    "event_free",
    "onset_ahead",
    "ended_1_7",
)
IN_FORCE: Final[tuple[str, ...]] = ("far", "middle", "near")
FAR_DAYS: Final[float] = 10.0
MIDDLE_DAYS: Final[float] = 1.0
POST_END_DAYS: Final[float] = 7.0
REAL_TRANSFERS_FILE: Final[str] = (
    "00101-ro-nc-detector-on-real-hand-filtered-switching-events/events_real.parquet"
)


@dataclass(frozen=True)
class Shift:
    """One shift of one Primary, as a share of capacity added to its power.

    Attributes:
        primary: The Primary's index, 0 to 15.
        kind: `constant` or `proportional`.
        onset_idx: The first half-hour of the shift.
        ramp_half_hours: The ramp length (1 is a step).
        end_idx: The first half-hour after the shift, clipped to the window.
        drawn_days: The drawn duration in days, before clipping.
        depth: The unsigned depth as a share of capacity.
        sign: 1 or -1.
        family: `train`, `held_out`, or `resampled`.
    """

    primary: int
    kind: str
    onset_idx: int
    ramp_half_hours: int
    end_idx: int
    drawn_days: float
    depth: float
    sign: int
    family: str

    @property
    def clipped(self) -> bool:
        """Whether the window cut the shift short of its drawn duration."""
        return self.end_idx - self.onset_idx < round(self.drawn_days * PER_DAY)

    def row(self) -> dict[str, object]:
        """Return the shift as a flat dict for an event table."""
        return {
            "primary": self.primary,
            "kind": self.kind,
            "onset_idx": self.onset_idx,
            "ramp_half_hours": self.ramp_half_hours,
            "end_idx": self.end_idx,
            "drawn_days": self.drawn_days,
            "depth": self.depth,
            "sign": self.sign,
            "family": self.family,
            "clipped": self.clipped,
        }


def seed_key(*, space: str, **parts: int) -> list[int]:
    """Return the seed key of a draw, whose first element names the space it belongs to.

    Args:
        space: `train`, `valid`, `resampled`, or `scoring`.
        **parts: The remaining integers of the key, in order: for `train` and `valid` the fit seed,
            the month index, the pass, and the Primary; for `scoring` and `resampled` the month
            index, the pass, and the Primary.

    Returns:
        The key for `numpy.random.default_rng`.

    Raises:
        ValueError: For an unknown space.
    """
    first = {
        "train": common.SEED_SPACE_TRAIN,
        "valid": common.SEED_SPACE_VALID,
        "resampled": common.SEED_SPACE_RESAMPLED,
        "scoring": common.SEED_SPACE_650,
    }.get(space)
    if first is None:
        msg = f"unknown seed space {space}"
        raise ValueError(msg)
    return [first, *parts.values()]


def family_of(*, r: int) -> str:
    """Return `train` or `held_out` for scoring pass `r` (experiment 650's passes).

    Args:
        r: The pass number.

    Returns:
        The family.

    Raises:
        ValueError: For a pass that is in neither family.
    """
    if 0 <= r < common.SCORING_TRAIN_PASSES:
        return "train"
    if common.HELD_FIRST <= r < common.HELD_FIRST + common.SCORING_HELD_PASSES:
        return "held_out"
    msg = f"unknown pass {r}"
    raise ValueError(msg)


def scoring_passes() -> tuple[int, ...]:
    """Return the 12 training-family passes then the 4 held-out passes."""
    return (
        *range(common.SCORING_TRAIN_PASSES),
        *range(common.HELD_FIRST, common.HELD_FIRST + common.SCORING_HELD_PASSES),
    )


def _draw_train_family(*, rng: np.random.Generator, is_mw_primary: bool) -> tuple:
    """Draw kind, depth, sign, days, and ramp in experiment 650's order of random calls."""
    kind = "constant" if (rng.random() < 0.5 or not is_mw_primary) else "proportional"
    depth = float(np.exp(rng.uniform(np.log(DEPTH_RANGE[0]), np.log(DEPTH_RANGE[1]))))
    sign = 1 if rng.random() < 0.5 else -1
    if rng.random() < 0.5:
        days = int(rng.integers(SHORT_DAYS[0], SHORT_DAYS[1] + 1))
    else:
        days = int(rng.integers(LONG_DAYS[0], LONG_DAYS[1] + 1))
    ramp = 1 if rng.random() < 0.5 else int(rng.integers(*TRAIN_RAMP_HOURS)) * 2
    return kind, depth, sign, days, ramp


def draw_scoring_shift(
    *,
    month_idx: int,
    r: int,
    primary: int,
    month_start_idx: int,
    window_end_idx: int,
    is_mw: np.ndarray,
) -> Shift:
    """Draw the shift of one Primary in one scoring pass, as experiment 650's `draw` does.

    The draw depends only on (month index, pass, Primary), never on the data.

    Args:
        month_idx: The fold's month index (the month's start index, as experiment 650 keys it).
        r: The pass.
        primary: The Primary's index.
        month_start_idx: The grid index of 00:00 UTC on the first day of the month.
        window_end_idx: The first grid index that no shift may reach.
        is_mw: Per Primary, whether it reports signed megawatts.

    Returns:
        The shift.
    """
    rng = np.random.default_rng(seed_key(space="scoring", month=month_idx, r=r, primary=primary))
    family = family_of(r=r)
    if family == "train":
        kind, depth, sign, days, ramp = _draw_train_family(
            rng=rng, is_mw_primary=bool(is_mw[primary])
        )
        onset = month_start_idx + int(rng.integers(-10 * PER_DAY, 20 * PER_DAY))
    else:
        k = r - common.HELD_FIRST
        cell = (primary + 5 * k) % 16
        sign = 1 if k % 2 == 0 else -1
        depth = HELD_DEPTHS[cell % 4]
        days = HELD_DAYS[cell // 4]
        kind = "constant"
        long_ramp = (primary + k) % 2 == 0
        ramp = int(rng.integers(*(HELD_RAMP_HOURS if long_ramp else TRAIN_RAMP_HOURS))) * 2
        onset = month_start_idx + int(rng.integers(0, 10 * PER_DAY))
    end = min(onset + days * PER_DAY, window_end_idx)
    return Shift(
        primary=primary,
        kind=kind,
        onset_idx=onset,
        ramp_half_hours=ramp,
        end_idx=end,
        drawn_days=float(days),
        depth=depth,
        sign=sign,
        family=family,
    )


def draw_resampled_shift(
    *,
    month_idx: int,
    r: int,
    primary: int,
    month_start_idx: int,
    window_end_idx: int,
    is_mw: np.ndarray,
    real_days: np.ndarray,
) -> Shift:
    """Draw a shift whose duration is resampled from the real transfers (a report-only world).

    Args:
        month_idx: The fold's month index.
        r: The pass.
        primary: The Primary's index.
        month_start_idx: The grid index of the first day of the month.
        window_end_idx: The first grid index that no shift may reach.
        is_mw: Per Primary, whether it reports signed megawatts.
        real_days: The durations of the 66 real transfers, in days, sub-day durations included.

    Returns:
        The shift.
    """
    rng = np.random.default_rng(seed_key(space="resampled", month=month_idx, r=r, primary=primary))
    kind, depth, sign, _, ramp = _draw_train_family(rng=rng, is_mw_primary=bool(is_mw[primary]))
    days = float(rng.choice(real_days))
    onset = month_start_idx + int(rng.integers(-10 * PER_DAY, 20 * PER_DAY))
    end = min(onset + max(round(days * PER_DAY), 1), window_end_idx)
    return Shift(
        primary=primary,
        kind=kind,
        onset_idx=onset,
        ramp_half_hours=min(ramp, max(end - onset, 1)),
        end_idx=end,
        drawn_days=days,
        depth=depth,
        sign=sign,
        family="resampled",
    )


def renewal_shifts(
    *,
    space: str,
    fit_seed: int,
    month_idx: int,
    pass_no: int,
    primary: int,
    window_start_idx: int,
    window_end_idx: int,
    is_mw: np.ndarray,
) -> list[Shift]:
    """Draw the shifts of one Primary in one training or validation world, as a renewal process.

    Gaps between the end of one shift and the start of the next are exponential with a mean of 118
    days. The process starts 112 days before the window, so the window opens in a stationary mix.
    About 25% of origin days then lie inside a shift (39.25 days of mean shift length divided by
    39.25 + 118).

    Args:
        space: `train` or `valid`.
        fit_seed: The seed of the fit.
        month_idx: The fold's month index.
        pass_no: The training pass (always 0 for the fixed validation world).
        primary: The Primary's index.
        window_start_idx: The first grid index of the window.
        window_end_idx: The first grid index after the window (the fold's cutoff).
        is_mw: Per Primary, whether it reports signed megawatts.

    Returns:
        The shifts that reach into the window, clipped at its end.
    """
    rng = np.random.default_rng(
        seed_key(space=space, seed=fit_seed, month=month_idx, r=pass_no, primary=primary)
    )
    t = window_start_idx - common.BURN_IN_DAYS * PER_DAY
    out: list[Shift] = []
    while True:
        t += max(round(float(rng.exponential(common.GAP_MEAN_DAYS * PER_DAY))), 1)
        if t >= window_end_idx:
            return out
        kind, depth, sign, days, ramp = _draw_train_family(
            rng=rng, is_mw_primary=bool(is_mw[primary])
        )
        end_unclipped = t + days * PER_DAY
        if end_unclipped > window_start_idx:
            out.append(
                Shift(
                    primary=primary,
                    kind=kind,
                    onset_idx=t,
                    ramp_half_hours=ramp,
                    end_idx=min(end_unclipped, window_end_idx),
                    drawn_days=float(days),
                    depth=depth,
                    sign=sign,
                    family="train",
                )
            )
        t = end_unclipped


def envelope(*, n: int, shift: Shift) -> np.ndarray:
    """Return the 0 to 1 envelope of a shift on grid indices 0 to n."""
    i = np.arange(n)
    up = np.clip((i - shift.onset_idx + 1) / max(shift.ramp_half_hours, 1), 0.0, 1.0)
    return np.where((i >= shift.onset_idx) & (i < shift.end_idx), up, 0.0)


def shift_delta(*, clean_row: np.ndarray, shift: Shift) -> np.ndarray:
    """Return the share of capacity that one shift adds to a Primary's power (before the MVA floor).

    A proportional shape scales the Primary's own cleaned series so that its mean over the shift
    equals the drawn depth.

    Args:
        clean_row: The Primary's cleaned power over capacity, NaN where flagged.
        shift: The shift.

    Returns:
        A float32 array with the length of `clean_row`.
    """
    n = len(clean_row)
    env = envelope(n=n, shift=shift)
    if shift.kind == "constant":
        shape = np.ones(n, np.float32)
    else:
        seg = clean_row[max(shift.onset_idx, 0) : shift.end_idx]
        mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 1.0
        shape = (np.nan_to_num(clean_row, nan=mean) / max(abs(mean), 1e-6)).astype(np.float32)
    return (shift.sign * shift.depth * env * shape).astype(np.float32)


def build_own(*, clean: np.ndarray, shifts: list[Shift], is_mw: np.ndarray) -> np.ndarray:
    """Return the (16, grid) share of capacity that the shifts add to each Primary's power.

    NaN power stays NaN where the caller adds the result. An MVA-only Primary's power is a
    magnitude and stays at or above zero.

    Args:
        clean: Cleaned power over capacity, (16, grid), NaN where dropped.
        shifts: The shifts of every Primary.
        is_mw: Per Primary, whether it reports signed megawatts.

    Returns:
        A float32 array like `clean`.
    """
    own = np.zeros(clean.shape, np.float32)
    for s in shifts:
        if s.end_idx > s.onset_idx:
            own[s.primary] += shift_delta(clean_row=clean[s.primary], shift=s)
    for p in range(clean.shape[0]):
        if not bool(is_mw[p]):
            own[p] = np.where(
                np.isfinite(clean[p]), np.maximum(clean[p] + own[p], 0.0) - clean[p], own[p]
            ).astype(np.float32)
    return own


def inject(*, clean: np.ndarray, own: np.ndarray) -> np.ndarray:
    """Return the cleaned power with the shift added; a NaN stays NaN.

    Args:
        clean: Cleaned power over capacity.
        own: The shift over capacity.

    Returns:
        The injected power.
    """
    return clean + own


def event_table(*, shifts: list[Shift], month: str, r: int) -> list[dict[str, object]]:
    """Return the rows of the event table of one pass.

    Args:
        shifts: The shifts of the pass.
        month: The fold month.
        r: The pass.

    Returns:
        One dict per shift.
    """
    return [{"month": month, "pass": r, **s.row()} for s in shifts]


def load_real_durations_days() -> np.ndarray:
    """Return the durations of the 66 real transfers of experiment 101, in days.

    Returns:
        A float array of 66 durations.
    """
    import polars as pl
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    table = pl.read_parquet(ML_EXPERIMENTS_DATA_DIR / REAL_TRANSFERS_FILE)
    days = table["hours"].to_numpy().astype(float) / 24.0
    assert len(days) == common.REAL_TRANSFERS, len(days)
    assert bool((days > 0).all())
    return days


def strata(
    *,
    origin_idx: np.ndarray,
    valid_idx: np.ndarray,
    onset_idx: np.ndarray,
    end_idx: np.ndarray,
    clipped: np.ndarray,
) -> np.ndarray:
    """Return the stratum of each row, for one shift per row's Primary and pass.

    All intervals are half-open. With `R` the days from the origin to the end of the shift and `S`
    the days since the end:

    - `far`: in force at the origin (`onset <= origin < end`) and `R > 10`;
    - `middle`: in force and `1 < R <= 10`;
    - `near`: in force and `0 < R <= 1`;
    - `ended_24h`: `0 <= S <= 1` (an origin at the very half-hour of the end counts as ended, so the
      in-force and ended intervals leave no gap);
    - `ended_1_7`: `1 < S <= 7` (a context stratum);
    - `onset_ahead`: the shift starts after the origin and no later than the valid time (context);
    - `event_free`: none of the above, so the shift has not started by the valid time or ended more
      than 7 days before the origin.

    A shift clipped at the window edge has an end that is not a real end, so an in-force row of such
    a shift is `in_force_clipped`, and is left out of far, middle, and near. A clipped shift never
    reaches `ended` strata, because its end is the window's.

    Args:
        origin_idx: Grid index of the 12:00 UTC contract origin of each row.
        valid_idx: Grid index of each row's valid time.
        onset_idx: The shift's first half-hour.
        end_idx: The shift's end.
        clipped: Whether the window clipped the shift.

    Returns:
        An array of stratum names.
    """
    o = np.asarray(origin_idx)
    in_force = (np.asarray(onset_idx) <= o) & (o < np.asarray(end_idx))
    remaining = (np.asarray(end_idx) - o) / PER_DAY
    since = (o - np.asarray(end_idx)) / PER_DAY
    ended = (~in_force) & (o >= np.asarray(end_idx)) & ~np.asarray(clipped)
    ahead = (o < np.asarray(onset_idx)) & (np.asarray(onset_idx) <= np.asarray(valid_idx))
    clipped_ = np.asarray(clipped)
    return np.select(
        [
            in_force & clipped_,
            in_force & (remaining > FAR_DAYS),
            in_force & (remaining > MIDDLE_DAYS),
            in_force,
            ended & (since <= 1.0),
            ended & (since > 1.0) & (since <= POST_END_DAYS),
            ahead,
        ],
        ["in_force_clipped", "far", "middle", "near", "ended_24h", "ended_1_7", "onset_ahead"],
        default="event_free",
    )


def true_age_days(*, origin_idx: np.ndarray, onset_idx: np.ndarray) -> np.ndarray:
    """Return the days from the onset to the contract origin (negative before the onset)."""
    return (np.asarray(origin_idx) - np.asarray(onset_idx)) / PER_DAY


def true_delta_at(*, shift: Shift, valid_idx: np.ndarray) -> np.ndarray:
    """Return the share of capacity that a constant-shape shift adds at the given valid indices.

    Used by the oracle and the follow share, which need the shift's size at each valid time.

    Args:
        shift: The shift (its shape is treated as constant).
        valid_idx: Grid indices.

    Returns:
        The signed depth times the envelope.
    """
    v = np.asarray(valid_idx)
    up = np.clip((v - shift.onset_idx + 1) / max(shift.ramp_half_hours, 1), 0.0, 1.0)
    return (
        shift.sign * shift.depth * np.where((v >= shift.onset_idx) & (v < shift.end_idx), up, 0.0)
    )


def simulate_stratum_shares(
    *,
    n_months: int = 8,
    lead_days: tuple[float, ...] = (1.5, 3.5, 5.5, 7.5, 9.5),
    month_start_idx: int = 40_000,
) -> dict[str, float]:
    """Return the stratum shares of the scoring worlds, over origins on days 0 to 29 of a month.

    Args:
        n_months: How many month indices to simulate (keys of the draw).
        lead_days: The lead days after the contract origin at which each origin is scored.
        month_start_idx: The grid index of the simulated month's first day.

    Returns:
        The share of rows in each stratum (and `in_force_clipped`), plus the share of in-force
        origins older than 15 days under `older_than_15`.
    """
    is_mw = np.array([True] * 8 + [False] * 8)
    labels: list[np.ndarray] = []
    ages: list[np.ndarray] = []
    window_end = month_start_idx + 10_000 * PER_DAY  # no clipping, as for every month but the last
    origins = month_start_idx + np.arange(30) * PER_DAY + 24
    for m in range(n_months):
        for r in range(common.SCORING_TRAIN_PASSES):
            for p in range(16):
                s = draw_scoring_shift(
                    month_idx=m,
                    r=r,
                    primary=p,
                    month_start_idx=month_start_idx,
                    window_end_idx=window_end,
                    is_mw=is_mw,
                )
                for lead in lead_days:
                    valid = origins + int(lead * PER_DAY)
                    labels.append(
                        strata(
                            origin_idx=origins,
                            valid_idx=valid,
                            onset_idx=np.full(30, s.onset_idx),
                            end_idx=np.full(30, s.end_idx),
                            clipped=np.full(30, s.clipped),
                        )
                    )
                age = true_age_days(origin_idx=origins, onset_idx=np.full(30, s.onset_idx))
                ages.append(age[(origins >= s.onset_idx) & (origins < s.end_idx)])
    flat = np.concatenate(labels)
    shares = {name: float((flat == name).mean()) for name in (*STRATA, "in_force_clipped")}
    all_ages = np.concatenate(ages)
    shares["older_than_15"] = float((all_ages > 15).mean())
    return shares

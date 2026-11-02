"""Experiment 518: windows, injected worlds, and the split by time for the stage-0 encoder fit.

A window is one (Primary, day) pair: the 672 half-hours ending at the newest usable reading,
`origin - 7`, of the daily 09:00 UTC origin. Its events come from experiment 507's generator
(families A and B only; the plan drops family C), and a world is one generator seed. The functions
take plain arrays, so the self-test can run them on a synthetic spec.

Split for a fold with refit `R`:

- Test windows: origins in the scored month.
- Embargo: no training or validation window has its newest reading at or after `R - 24 days`.
- Validation block: the last 30 days before that embargo. Training windows end 14 days before the
  block starts, so no training history overlaps a validation history.
- A window whose history, padded by 48 half-hours, touches a labelled load-shift event is dropped
  everywhere. A window at a Primary with a start cut needs a history that starts after the cut.
"""

from dataclasses import dataclass
from typing import Final

import e00507_generator as gen  # ty: ignore[unresolved-import]
import e00518_model as model  # ty: ignore[unresolved-import]
import numpy as np

EMBARGO_DAYS: Final[int] = 24
VALIDATION_DAYS: Final[int] = 30
GAP_DAYS: Final[int] = 14
EVENT_PADDING: Final[int] = 48
MIN_OBSERVED: Final[float] = 0.9
"""A window needs a finite lagless median at this share of its half-hours."""
CHUNK: Final[int] = 1024
HISTORY: Final[int] = gen.HISTORY


@dataclass(frozen=True)
class Windows:
    """A set of windows: Primary index and day index per window."""

    primary: np.ndarray
    day: np.ndarray

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.primary)

    def take(self, index: np.ndarray) -> Windows:
        """Return the windows at `index`."""
        return Windows(primary=self.primary[index], day=self.day[index])


@dataclass(frozen=True)
class Arrays:
    """The inputs and labels of a set of windows for one world.

    Attributes:
        x: Encoder input (windows, CHANNELS, 672).
        step_true: The injected step in spreads, zero where none (windows, 672).
        valid: False where the half-hour is unobserved or masked from the step loss.
        genuine_target: 1 for a genuine change in force at the newest reading (windows,).
        genuine_valid: True where the window has a step in force at the newest reading whose onset
            lies inside the window.
        masked_mva: The number of observed half-hours masked by the apparent-power rule.
        observed: The number of observed half-hours.
    """

    x: np.ndarray
    step_true: np.ndarray
    valid: np.ndarray
    genuine_target: np.ndarray
    genuine_valid: np.ndarray
    masked_mva: int
    observed: int


def history_grid(*, day: np.ndarray) -> np.ndarray:
    """Return the grid indices (windows, 672) of each window's history."""
    newest = gen.newest_of(day=day)
    return newest[:, None] + np.arange(-(HISTORY - 1), 1)[None, :]


def _prefix(*, flags: np.ndarray) -> np.ndarray:
    """Return the cumulative count of `flags` along axis 1, with a leading zero column."""
    return np.concatenate(
        [np.zeros((flags.shape[0], 1), dtype=np.int64), np.cumsum(flags, axis=1, dtype=np.int64)],
        axis=1,
    )


def candidate_windows(
    *, spec: gen.Spec, med: np.ndarray, labelled: np.ndarray, first_history: np.ndarray
) -> Windows:
    """Return every window with a usable history and no labelled event.

    Args:
        spec: The generator spec.
        med: The lagless median (primaries, grid), NaN where absent.
        labelled: True where a labelled event covers the half-hour (primaries, grid).
        first_history: The earliest grid index allowed in a window's history, per Primary.

    Returns:
        The windows, ordered by Primary then day.
    """
    n_p, size = med.shape
    days = np.arange(spec.days)
    newest = gen.newest_of(day=days)
    first = newest - (HISTORY - 1)
    finite = _prefix(flags=np.isfinite(med))
    touched = _prefix(flags=labelled)
    out_p: list[np.ndarray] = []
    out_d: list[np.ndarray] = []
    for p in range(n_p):
        lo = np.maximum(first - EVENT_PADDING, 0)
        hi = np.minimum(newest + EVENT_PADDING + 1, size)
        ok = (first >= first_history[p]) & (newest < size)
        share = (
            finite[p, np.minimum(newest + 1, size)] - finite[p, np.maximum(first, 0)]
        ) / HISTORY
        ok &= share >= MIN_OBSERVED
        ok &= (touched[p, hi] - touched[p, lo]) == 0
        out_p.append(np.full(int(ok.sum()), p))
        out_d.append(days[ok])
    return Windows(primary=np.concatenate(out_p), day=np.concatenate(out_d))


@dataclass(frozen=True)
class Split:
    """The training, validation, and test windows of one fold."""

    train: Windows
    validation: Windows
    test: Windows


def split_by_time(
    *, windows: Windows, refit: int, month_half_hours: int, validation_days: int = VALIDATION_DAYS
) -> Split:
    """Split the candidate windows by the newest reading's grid index.

    Args:
        windows: The candidate windows.
        refit: The grid index of the refit (the first half-hour of the scored month).
        month_half_hours: The number of half-hours in the scored month.
        validation_days: The length of the validation block in days.

    Returns:
        The split, with the embargo and the gap applied.
    """
    newest = gen.newest_of(day=windows.day)
    origin = gen.origin_of(day=windows.day)
    day = 48
    embargo_start = refit - EMBARGO_DAYS * day
    val_start = embargo_start - validation_days * day
    train_end = val_start - GAP_DAYS * day
    return Split(
        train=windows.take(np.flatnonzero(newest < train_end)),
        validation=windows.take(np.flatnonzero((newest >= val_start) & (newest < embargo_start))),
        test=windows.take(np.flatnonzero((origin >= refit) & (origin < refit + month_half_hours))),
    )


def arrays_of(*, spec: gen.Spec, ev: gen.Events, med: np.ndarray, windows: Windows) -> Arrays:
    """Return the encoder inputs and labels of `windows` in the world `ev`.

    The surprise is `(shifted reading - lagless median) / spread`, clipped and scaled. The second
    channel is the median, standardised within the window. Unobserved readings and medians enter as
    zero (the plan drops the missing-mask channel). At the MVA-only Primaries a half-hour whose
    median is below twice the injected step is masked from the step loss.
    """
    n = len(windows)
    x = np.zeros((n, model.CHANNELS, HISTORY), dtype=np.float32)
    step_true = np.zeros((n, HISTORY), dtype=np.float32)
    valid = np.zeros((n, HISTORY), dtype=bool)
    gen_target = np.zeros(n, dtype=np.float32)
    gen_valid = np.zeros(n, dtype=bool)
    masked = observed = 0
    for lo in range(0, n, CHUNK):
        sl = slice(lo, min(lo + CHUNK, n))
        p, d = windows.primary[sl], windows.day[sl]
        k = len(p)
        g = history_grid(day=d)
        flat = (np.repeat(p, HISTORY), np.repeat(d, HISTORY), g.reshape(-1))
        shifted = gen.reading(spec=spec, ev=ev, p=flat[0], d=flat[1], g=flat[2], mode="shifted")
        shifted = shifted.reshape(k, HISTORY)
        real = spec.real[p[:, None], g]
        median = med[p[:, None], g]
        spread = spec.spread[p][:, None].astype(np.float32)
        step_cap = np.nan_to_num(shifted - real, nan=0.0)
        ok = np.isfinite(real) & np.isfinite(median)
        low = spec.mva_only[p][:, None] & (median < 2.0 * np.abs(step_cap)) & (step_cap != 0)
        masked += int((ok & low).sum())
        observed += int(ok.sum())
        z = np.nan_to_num((shifted - median) / spread, nan=0.0)
        z = np.clip(z, -model.SURPRISE_CLIP, model.SURPRISE_CLIP) / model.SURPRISE_SCALE
        mean = np.nanmean(median, axis=1, keepdims=True)
        std = np.nanstd(median, axis=1, keepdims=True) + 1e-3
        m = np.nan_to_num(np.clip((median - mean) / std, -4.0, 4.0), nan=0.0)
        x[sl, 0], x[sl, 1] = z, m
        step_true[sl] = np.where(ok, step_cap / spread, 0.0)
        valid[sl] = ok & ~low
        newest = gen.newest_of(day=d)
        for slot in range(gen.SLOTS):
            fam = ev.family[p, d, slot]
            live = (
                (fam != gen.NONE) & (ev.start[p, d, slot] <= newest) & (newest < ev.end[p, d, slot])
            )
            inside = ev.start[p, d, slot] >= newest - (HISTORY - 1)
            use = live & inside & ~gen_valid[sl]
            gen_valid[sl] |= use
            gen_target[sl] = np.where(use, (fam == gen.FAMILY_B).astype(np.float32), gen_target[sl])
    return Arrays(
        x=x,
        step_true=step_true,
        valid=valid,
        genuine_target=gen_target,
        genuine_valid=gen_valid,
        masked_mva=masked,
        observed=observed,
    )


def sampling_weights(*, spec: gen.Spec, med: np.ndarray, windows: Windows) -> np.ndarray:
    """Return the family D weight: 3 for the top decile of lagless-median range, else 1.

    The decile is taken over all of `windows`, not chunk by chunk.
    """
    span = np.empty(len(windows))
    for lo in range(0, len(windows), 8192):
        sl = slice(lo, lo + 8192)
        g = history_grid(day=windows.day[sl])
        m = med[windows.primary[sl][:, None], g]
        span[sl] = np.nanmax(m, axis=1) - np.nanmin(m, axis=1)
    return np.where(span >= np.nanquantile(span, 0.9), 3.0, 1.0)


def epoch_arrays(
    *,
    spec: gen.Spec,
    med: np.ndarray,
    windows: Windows,
    weights: np.ndarray,
    epoch: int,
    size: int,
    base_seed: int,
) -> Arrays:
    """Return one epoch's arrays: `size` windows drawn with `weights` in a fresh world.

    The world seed is `base_seed + epoch`, so each epoch sees different events on the same
    underlying readings. Hard negatives enter by their weights; every window, whatever its weight,
    receives the same family A and B rates (the plan's fix to the proposal).
    """
    rng = np.random.default_rng([base_seed, epoch])
    pick = rng.choice(len(windows), size=size, replace=True, p=weights / weights.sum())
    ev = gen.sample_world(spec=spec, seed=base_seed + epoch)
    return arrays_of(spec=spec, ev=ev, med=med, windows=windows.take(pick))


def in_force_shares(*, ev: gen.Events, windows: Windows) -> tuple[float, float]:
    """Return the share of windows with a step in force at the newest reading, and B's share."""
    in_a, in_b = gen.in_force_at_newest(ev=ev)
    a = in_a[windows.primary, windows.day]
    b = in_b[windows.primary, windows.day]
    any_step = float((a | b).mean())
    return any_step, float(b.sum() / max((a | b).sum(), 1))

"""Experiment 507: the injection generator for temporary (A) and genuine (B) shifts.

A window is one (Primary, origin) pair; origins are daily at 09:00 UTC (grid index = 18 mod 48).
Each window draws its own events, a pure function of (world seed, Primary, day), so rows that share
an origin share an injection. 45% of windows get family A (1 to 3 temporary shifts), 25% get
family B (one genuine level change that never ends), and 30% get neither. Family D (hard negatives)
is a sampling weight, `hard_negative_weight`, applied when windows are drawn for the encoder.

An event adds `size * profile(g) * shape(g)` to the Primary's stored reading, where `profile` ramps
up over `ramp + 1` half-hours from the onset, ramps down over `ramp + 1` half-hours to the end, and
a staircase runs at half size for its first `stair` half-hours. The shape is 1 (constant MW), the
Primary's own demand over its mean (proportional), or another Primary's demand over its mean at the
same time (a circuit profile), in the shares of `SHAPE_MIX`. At the MVA-only Primaries the reading is `abs(real +
step)`. The "history" is the 672 half-hours ending at the newest usable reading, `origin - 7`.

Derived expectation used by the self-test: with L = 1344 possible onsets and durations d,
P(one A event in force at the newest half-hour) = E[min(d, L)] / L, and with k events uniform on 1
to 3 the window's chance is 1 - (1 - p)^k averaged over k. Family B is always in force at its
origin, so its share of in-force windows is 0.25 / (0.25 + 0.45 * P(A)).

```bash
ml-experiments/00507-*/run.sh ml-experiments/00507-*/e00507_generator.py selftest
```
"""

import argparse
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

HISTORY: Final[int] = 672
NEWEST_AGE: Final[int] = 7
ORIGIN_PHASE: Final[int] = 18
"""Grid index of the first 09:00 UTC origin (the grid starts at 2019-09-01 00:00 UTC)."""
STEPS_PER_DAY: Final[int] = 48
SLOTS: Final[int] = 3
P_A: Final[float] = 0.45
P_B: Final[float] = 0.25
CAP_MW: Final[float] = 5.4
SIZE_SPREADS: Final[tuple[float, float]] = (0.5, 3.0)
NEVER: Final[int] = 1 << 30
MAX_DAYS: Final[int] = 4000
"""Every random vector is drawn at this length, so a window's events do not change when the grid grows."""
RAMP_SHARE: Final[float] = 0.3
STAIR_SHARE: Final[float] = 0.2
ONSET_RANGE_A: Final[int] = 2 * HISTORY
SHAPE_MIX: Final[tuple[float, float, float]] = (0.2, 0.5, 0.3)
"""Shares of constant, own-demand, and circuit shapes, set from N0's shift-shape table: 29 of the 34
events with a measured deviation correlate above 0.5 with the Primary's own demand shape."""
NONE, FAMILY_A, FAMILY_B = 0, 1, 2


@dataclass(frozen=True)
class Spec:
    """What the generator needs about the Primaries.

    Attributes:
        real: Stored readings over capacity (primaries, grid), NaN where unobserved.
        shape: Each Primary's demand over its mean, NaN replaced by 1 (primaries, grid).
        spread: Each Primary's lagless residual spread, a fraction of capacity.
        capacity_mw: Each Primary's effective capacity in MW.
        mva_only: True for the MVA-only Primaries.
        pool_steps: Durations of labelled events, in half-hours.
    """

    real: np.ndarray
    shape: np.ndarray
    spread: np.ndarray
    capacity_mw: np.ndarray
    mva_only: np.ndarray
    pool_steps: np.ndarray

    @property
    def days(self) -> int:
        """Return the number of origins (days) the grid holds."""
        return (self.real.shape[1] - ORIGIN_PHASE) // STEPS_PER_DAY


def shape_of(*, real: np.ndarray) -> np.ndarray:
    """Return each Primary's reading over its mean (1 where unobserved or not positive), clipped to 0..4."""
    mean = np.nanmean(np.abs(real), axis=1, keepdims=True)
    out = np.where(np.isfinite(real) & (real > 0), real / np.maximum(mean, 1e-6), 1.0)
    return np.clip(out, 0.0, 4.0).astype(np.float32)


@dataclass(frozen=True)
class Events:
    """Event parameters, all arrays (primaries, days, SLOTS); `family == NONE` marks no event."""

    family: np.ndarray
    start: np.ndarray
    end: np.ndarray
    size: np.ndarray
    source: np.ndarray
    ramp: np.ndarray
    stair: np.ndarray


def origin_of(*, day: np.ndarray) -> np.ndarray:
    """Return the grid index of the origin of day `day`."""
    return ORIGIN_PHASE + STEPS_PER_DAY * np.asarray(day, dtype=np.int64)


def newest_of(*, day: np.ndarray) -> np.ndarray:
    """Return the grid index of the newest usable reading for day `day`."""
    return origin_of(day=day) - NEWEST_AGE


def sample_world(*, spec: Spec, seed: int) -> Events:
    """Draw every window's events for one world; deterministic in (seed, Primary, day)."""
    days = spec.days
    n_p = spec.real.shape[0]
    assert days <= MAX_DAYS, days
    shape = (n_p, days, SLOTS)
    fam = np.zeros(shape, dtype=np.int8)
    start = np.zeros(shape, dtype=np.int64)
    end = np.zeros(shape, dtype=np.int64)
    size = np.zeros(shape, dtype=np.float32)
    source = np.full(shape, -1, dtype=np.int16)
    ramp = np.zeros(shape, dtype=np.int16)
    stair = np.zeros(shape, dtype=np.int16)
    newest = newest_of(day=np.arange(days))
    for p in range(n_p):
        rng = np.random.default_rng([seed, p])
        u = rng.random(MAX_DAYS)[:days]
        window_family = np.where(u < P_A, FAMILY_A, np.where(u < P_A + P_B, FAMILY_B, NONE))
        n_events = rng.integers(1, SLOTS + 1, size=MAX_DAYS)[:days]
        for k in range(SLOTS):
            active_a = (window_family == FAMILY_A) & (k < n_events)
            active_b = (window_family == FAMILY_B) & (k == 0)
            active = active_a | active_b
            onset_a = newest - (ONSET_RANGE_A - 1) + (rng.integers(0, ONSET_RANGE_A, size=MAX_DAYS))[:days]
            onset_b = newest - (HISTORY - 1) + (rng.integers(0, HISTORY, size=MAX_DAYS))[:days]
            from_pool = (rng.random(MAX_DAYS) < 0.5)[:days]
            pool_d = (spec.pool_steps[rng.integers(0, len(spec.pool_steps), size=MAX_DAYS)])[:days]
            log_d = np.rint(np.exp((rng.uniform(np.log(2), np.log(HISTORY), size=MAX_DAYS))[:days]))
            duration = np.where(from_pool, pool_d, log_d).astype(np.int64)
            magnitude = np.minimum(
                (rng.uniform(*SIZE_SPREADS, size=MAX_DAYS))[:days] * spec.spread[p],
                CAP_MW / spec.capacity_mw[p],
            )
            sign = np.where((rng.random(MAX_DAYS) < 0.5)[:days], 1.0, -1.0)
            circuit_fraction = (rng.uniform(0.3, 1.0, size=MAX_DAYS))[:days]
            kind = np.searchsorted(np.cumsum(SHAPE_MIX)[:-1], (rng.random(MAX_DAYS))[:days], side="right")
            other = (p + 1 + (rng.integers(0, n_p - 1, size=MAX_DAYS))[:days]) % n_p
            src = np.where(kind == 0, -1, np.where(kind == 1, p, other))
            ramped = (rng.random(MAX_DAYS) < RAMP_SHARE)[:days]
            ramp_len = (rng.integers(1, 5, size=MAX_DAYS))[:days]
            staired = (rng.random(MAX_DAYS) < STAIR_SHARE)[:days]
            stair_len = (rng.integers(2, 13, size=MAX_DAYS))[:days]
            fam[p, :, k] = np.where(active_a, FAMILY_A, np.where(active_b, FAMILY_B, NONE))
            s = np.where(active_a, onset_a, onset_b)
            start[p, :, k] = s
            end[p, :, k] = np.where(active_a, s + duration, NEVER)
            fraction = np.where(kind == 2, circuit_fraction, 1.0)
            size[p, :, k] = (sign * magnitude * fraction).astype(np.float32)
            source[p, :, k] = src
            ramp[p, :, k] = np.where(ramped, ramp_len, 0)
            stair[p, :, k] = np.where(staired, stair_len, 0)
    return Events(family=fam, start=start, end=end, size=size, source=source, ramp=ramp,
                  stair=stair)  # fmt: skip


def event_profile(*, ev: Events, p: np.ndarray, d: np.ndarray, k: int, g: np.ndarray) -> np.ndarray:
    """Return the profile (0 to 1, before size and shape) of slot `k` at grid index `g`."""
    s, e = ev.start[p, d, k], ev.end[p, d, k]
    r = ev.ramp[p, d, k].astype(np.float64)
    up = np.clip((g - s + 1) / (r + 1.0), 0.0, 1.0)
    down = np.clip((e - g) / (r + 1.0), 0.0, 1.0)
    st = ev.stair[p, d, k]
    stage = np.where((st > 0) & (g < s + st), 0.5, 1.0)
    return up * down * stage


def step_values(
    *,
    spec: Spec,
    ev: Events,
    p: np.ndarray,
    d: np.ndarray,
    g: np.ndarray,
    families: tuple[int, ...],
) -> np.ndarray:
    """Return the summed step (fraction of capacity) of the given families at (p, d, g) rows."""
    p, d, g = (np.asarray(a, dtype=np.int64) for a in (p, d, g))
    total = np.zeros(len(g), dtype=np.float64)
    inside = (g >= 0) & (g < spec.real.shape[1])
    gs = np.clip(g, 0, spec.real.shape[1] - 1)
    for k in range(SLOTS):
        fam = ev.family[p, d, k]
        use = np.isin(fam, families) & inside
        if not use.any():
            continue
        src = ev.source[p, d, k].astype(np.int64)
        shape = np.where(src < 0, 1.0, spec.shape[np.maximum(src, 0), gs])
        cap = CAP_MW / spec.capacity_mw[p]
        step = ev.size[p, d, k] * event_profile(ev=ev, p=p, d=d, k=k, g=g) * shape
        total += np.where(use, np.clip(step, -cap, cap), 0.0)
    return total


def reading(
    *, spec: Spec, ev: Events, p: np.ndarray, d: np.ndarray, g: np.ndarray, mode: str
) -> np.ndarray:
    """Return the reading at (p, d, g) rows: NaN where unobserved or outside the grid.

    Modes: `shifted` (families A and B), `true` (family B only, the target's value), `real` (the
    stored reading). At the MVA-only Primaries the reading is the magnitude of the sum.
    """
    p, d, g = (np.asarray(a, dtype=np.int64) for a in (p, d, g))
    inside = (g >= 0) & (g < spec.real.shape[1])
    base = np.where(inside, spec.real[p, np.clip(g, 0, spec.real.shape[1] - 1)], np.nan)
    families = {"shifted": (FAMILY_A, FAMILY_B), "true": (FAMILY_B,), "real": ()}[mode]
    step = step_values(spec=spec, ev=ev, p=p, d=d, g=g, families=families) if families else 0.0
    value = base + step
    return np.where(spec.mva_only[p], np.abs(value), value).astype(np.float32)


def window_family(*, ev: Events) -> np.ndarray:
    """Return the family of each window (primaries, days): A, B, or NONE."""
    return ev.family.max(axis=2)


def analytic_in_force(*, spec: Spec) -> tuple[float, float, float]:
    """Return the derived share of windows with an A step in force, with B, and B's share of both."""
    length = ONSET_RANGE_A
    pool = np.minimum(spec.pool_steps, length).astype(np.float64)
    log_mean = float(
        (HISTORY - 2) / (np.log(HISTORY) - np.log(2))
    )  # mean of a log-uniform on [2, HISTORY], below `length`
    p_event = 0.5 * float(pool.mean()) / length + 0.5 * log_mean / length
    p_a = P_A * float(np.mean([1 - (1 - p_event) ** k for k in range(1, SLOTS + 1)]))
    return p_a, P_B, P_B / (P_B + p_a)


def in_force_at_newest(*, ev: Events) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (primaries, days) arrays: an A step, and a B step, in force at the newest."""
    days = ev.family.shape[1]
    newest = newest_of(day=np.arange(days))[None, :, None]
    live = (ev.start <= newest) & (newest < ev.end)
    return ((ev.family == FAMILY_A) & live).any(2), ((ev.family == FAMILY_B) & live).any(2)


def hard_negative_weight(*, median: np.ndarray) -> np.ndarray:
    """Return 3 for windows whose lagless median range is in the top decile, else 1.

    Args:
        median: The lagless median over each window's history, shape (windows, 672).
    """
    span = np.nanmax(median, axis=1) - np.nanmin(median, axis=1)
    return np.where(span >= np.nanquantile(span, 0.9), 3.0, 1.0)


def synthetic_spec(*, seed: int = 0, primaries: int = 4, days: int = 200) -> Spec:
    """Return a small made-up spec for the self-test."""
    rng = np.random.default_rng(seed)
    n = ORIGIN_PHASE + days * STEPS_PER_DAY + 400
    t = np.arange(n)
    real = np.stack(
        [0.5 + 0.2 * np.sin(2 * np.pi * (t + 5 * p) / 48) + 0.01 * rng.normal(size=n)
         for p in range(primaries)]
    ).astype(np.float32)  # fmt: skip
    return Spec(
        real=real, shape=shape_of(real=real), spread=np.full(primaries, 0.02),
        capacity_mw=np.full(primaries, 40.0), mva_only=np.array([False, False, True, True][:primaries]),
        pool_steps=np.array([2, 8, 40, 96, 400]),
    )  # fmt: skip


def selftest(*, spec: Spec | None = None, strict: bool = True) -> dict[str, float]:
    """Check step algebra, determinism, and the shares; print and return the shares.

    Args:
        spec: Use the real Primaries; the default is a small synthetic spec.
        strict: Raise when a share or a distribution test misses its acceptance criterion.
    """
    from scipy import stats  # noqa: PLC0415

    # Step algebra on one hand-made event: constant 0.1, onset 100, end 110, ramp 1, no stair.
    toy = synthetic_spec(primaries=2, days=4)
    ev = sample_world(spec=toy, seed=0)
    fam = np.zeros_like(ev.family)
    fam[0, 0, 0] = FAMILY_A
    hand = Events(
        family=fam, start=np.full_like(ev.start, 100), end=np.full_like(ev.end, 110),
        size=np.full_like(ev.size, 0.1), source=np.full_like(ev.source, -1),
        ramp=np.ones_like(ev.ramp), stair=np.zeros_like(ev.stair),
    )  # fmt: skip
    g = np.array([98, 99, 100, 101, 105, 108, 109, 110, 120])
    got = step_values(spec=toy, ev=hand, p=np.zeros(9), d=np.zeros(9), g=g, families=(FAMILY_A,))
    want = np.array([0, 0, 0.05, 0.1, 0.1, 0.1, 0.05, 0.0, 0.0])
    assert np.allclose(got, want, atol=1e-9), (got, want)
    only_b = step_values(spec=toy, ev=hand, p=np.zeros(9), d=np.zeros(9), g=g, families=(FAMILY_B,))
    assert (only_b == 0).all(), "family filter leaks"
    again = sample_world(spec=toy, seed=0)
    assert all(np.array_equal(getattr(ev, f), getattr(again, f)) for f in ev.__dataclass_fields__)
    other = sample_world(spec=toy, seed=1)
    assert not np.array_equal(ev.start, other.start), "different seeds must differ"
    # Shares on a large world.
    big = spec or synthetic_spec(primaries=4, days=3000)
    ev = sample_world(spec=big, seed=7)
    in_a, in_b = in_force_at_newest(ev=ev)
    share_a, share_b = float(in_a.mean()), float(in_b.mean())
    any_step = float((in_a | in_b).mean())
    genuine = share_b / max(share_a + share_b, 1e-9)
    exp_a, exp_b, exp_genuine = analytic_in_force(spec=big)
    print(f"windows with an A step in force at the newest half-hour: {share_a:.3f} "
          f"(derived {exp_a:.3f})")  # fmt: skip
    print(f"windows with a B step in force: {share_b:.3f} (derived {exp_b:.3f}); any step "
          f"{any_step:.3f}")  # fmt: skip
    print(f"family B share of steps in force at the origin: {genuine:.3f} (derived {exp_genuine:.3f},"
          f" target about 0.60)")  # fmt: skip
    days = ev.family.shape[1]
    newest = newest_of(day=np.arange(days))[None, :, None]
    inside = (ev.family > 0) & (ev.start >= newest - (HISTORY - 1))
    results = {"share_a": share_a, "share_b": share_b, "genuine_share": genuine}
    rng = np.random.default_rng(0)
    for name, values in {
        "ramp length": ev.ramp, "size": np.abs(ev.size), "sign": np.sign(ev.size),
        "onset hour of day": (ev.start % 48) / 2.0,
    }.items():  # fmt: skip
        a = values[inside & (ev.family == FAMILY_A)]
        b = values[inside & (ev.family == FAMILY_B)]
        a, b = (rng.choice(x, size=min(len(x), 4000), replace=False) for x in (a, b))
        p_value = float(stats.ks_2samp(a, b).pvalue)
        print(
            f"A against B, events with the onset inside the history, {name}: KS p = {p_value:.3f}"
        )
        results[f"ks_{name}"] = p_value
        if strict:
            assert p_value > 0.01, f"{name} differs between families A and B"
    if strict:
        assert abs(share_a - exp_a) < 0.15 * exp_a, "A share is not within 15% of the derived value"
        assert share_a > 0.01, "the generator makes almost no A step in force at the origin"
        assert abs(genuine - exp_genuine) < 0.05, "genuine share is not within 5 points"
    print("selftest passed")
    return results


def main() -> None:
    """Run the self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["selftest"])
    parser.parse_args()
    selftest()


if __name__ == "__main__":
    main()

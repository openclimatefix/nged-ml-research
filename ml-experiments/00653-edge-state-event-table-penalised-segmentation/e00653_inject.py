"""Synthetic events added to the real residual, with the truth table that scores them."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, E580, GRID_LENGTH, GRID_START, SCORED_FROM as SCORED_START, Panel

OFFSET_2019: Final[int] = int((GRID_START - datetime(2019, 9, 1, tzinfo=UTC)).total_seconds() // 1800)
"""Half-hours from 2019-09-01 (the label tables' origin) to the grid origin."""
PER_YEAR: Final[float] = 1.2
"""Injected donor events per Primary-year (about 5% of time inside an event)."""
NEG_PER_YEAR: Final[float] = 1.0


@dataclass
class Truth:
    """One injected event at one Primary."""

    site: str
    start: int
    end: int | None
    amp_mw: float
    units: float
    kind: str
    group: int
    partners: list[str] = field(default_factory=list)
    shape: str = "constant"
    partner_amp: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GenParams:
    """Generator settings; `width_hh` and `shape_prop` come from step 0."""

    width_hh: tuple[int, int] = (0, 4)
    shape_prop: float = 0.5
    duration_mix: str = "default"
    share: tuple[float, float] = (0.5, 1.1)
    start_jitter_hh: int = 0
    permanent_frac: float = 0.1
    staged_frac: float = 0.10
    mid_step_frac: float = 0.05
    per_year: float = PER_YEAR


def logged_start_times(*, before: int | None = None) -> np.ndarray:
    """Hour-of-day (half-hours) and weekday of NGED's logged onsets, shape (n, 2)."""
    table = pl.read_parquet(E580 / "events_labelled_status.parquet")
    idx = table["label_onset_idx"].drop_nulls().to_numpy()
    if before is not None:
        idx = idx[idx < before + OFFSET_2019]
    base = np.datetime64("2019-09-01T00:00", "m")
    when = base + (idx * 30).astype("timedelta64[m]")
    minutes = (when - when.astype("datetime64[D]")).astype("timedelta64[m]").astype(int)
    weekday = (when.astype("datetime64[D]").astype(int) + 3) % 7
    return np.stack([minutes // 30, weekday], axis=1)


def _duration(rng: np.random.Generator, mix: str) -> int:
    lo, hi = {"default": (3 * 2, 180 * DAY), "short": (3 * 2, 7 * DAY), "long": (7 * DAY, 180 * DAY)}[mix]
    return int(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _units(rng: np.random.Generator) -> float:
    if rng.random() < 0.5:
        return float(np.exp(rng.uniform(np.log(2), np.log(8))))
    return float(np.exp(rng.uniform(np.log(1), np.log(20))))


def _start(rng: np.random.Generator, starts: np.ndarray, lo: int, hi: int) -> int:
    hh, wd = starts[rng.integers(len(starts))]
    for _ in range(50):
        t = int(rng.integers(lo, hi))
        day = GRID_START + timedelta(minutes=30 * t)
        if day.weekday() == wd:
            return (t // DAY) * DAY + int(hh)
    return int(rng.integers(lo, hi))


def _profile(*, n: int, width: int, drift: float, mid_step: float | None, staged: int,
             open_end: bool = False) -> np.ndarray:
    """Unit profile of an event of `n` half-hours with smooth edges and optional shape changes."""
    p = np.ones(n)
    w = max(width, 1)
    w = min(w, max(1, n // 2))
    ramp = np.linspace(0, 1, w + 1)[1:] if width > 0 else np.array([1.0])
    p[: len(ramp)] = np.minimum(p[: len(ramp)], ramp)
    if not open_end:
        p[n - len(ramp) :] = np.minimum(p[n - len(ramp) :], ramp[::-1])
    p *= 1 + drift * (np.arange(n) / max(n, 1) - 0.5)
    if mid_step is not None:
        p[n // 2 :] *= mid_step
    if staged > 0 and n > 4 * staged:
        p[n - staged :] *= 0.5
    return p


def inject(*, panel: Panel, residual: dict[str, np.ndarray], sigma: dict[str, float],
           rng: np.random.Generator, lo: int, hi: int, gen: GenParams = GenParams(),
           kinds: tuple[str, ...] = ("transfer", "negatives"),
           ) -> tuple[dict[str, np.ndarray], list[Truth]]:
    """Add events to a copy of the residual between half-hours `lo` and `hi`; return the truth."""
    starts = logged_start_times(before=lo + (hi - lo) if hi <= SCORED_START else None)
    out = {s: v.copy() for s, v in residual.items()}
    truth: list[Truth] = []
    years = (hi - lo) / (365.25 * DAY)
    group = 0

    def add(site: str, start: int, end: int | None, amp: float, prof: np.ndarray | None,
            kind: str, partners: list[str], units: float, shape: str, gid: int,
            tail: int | None = None) -> None:
        stop = hi if end is None else min(end, hi)
        n = stop - start
        if n <= 0:
            return
        if prof is None:
            prof = np.ones(n)
        elif len(prof) != n:
            prof = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(prof)), prof)
        out[site][start:stop] += amp * prof
        real_end = None if (end is None or end >= hi) else end
        truth.append(Truth(site, start, real_end, amp, units, kind, gid, partners, shape))

    def shape_of(site: str, start: int, n: int) -> tuple[np.ndarray, str]:
        if rng.random() < gen.shape_prop:
            p = panel.power[site][start : start + n]
            if len(p) < n or not np.isfinite(p).any():
                return np.ones(n), "constant"
            p = np.where(np.isfinite(p), p, np.nanmean(p))
            m = p.mean()
            return np.clip(p / m, 0, 3) if m > 0 else np.ones(n), "proportional"
        return np.ones(n), "constant"

    if "transfer" in kinds:
        for site in panel.sites:
            for _ in range(rng.poisson(gen.per_year * years)):
                group += 1
                st = _start(rng, starts, lo, hi - 6)
                perm = rng.random() < gen.permanent_frac
                dur = None if perm else _duration(rng, gen.duration_mix)
                end = None if perm else st + dur
                n = (hi if end is None else min(end, hi)) - st
                units = _units(rng)
                amp = -units * sigma[site]
                width = int(rng.integers(gen.width_hh[0], gen.width_hh[1] + 1))
                if perm and rng.random() < 0.5:
                    width = int(rng.integers(7 * DAY, 56 * DAY))
                mid = float(rng.uniform(0.5, 1.5)) if rng.random() < gen.mid_step_frac else None
                staged = int(max(1, 0.2 * n)) if (not perm and rng.random() < gen.staged_frac) else 0
                prof = _profile(n=n, width=width, drift=float(rng.uniform(-0.2, 0.2)), mid_step=mid,
                                staged=staged, open_end=(end is None or end >= hi))
                shape, name = shape_of(site, st, n)
                prof = prof * shape
                mode = rng.random()
                nbrs = panel.neighbours[site]
                kind = "permanent_shift" if perm else "solo"
                donor_seen, rec_seen, recs = True, False, []
                if mode < 0.65 and nbrs:
                    k = min(len(nbrs), int(rng.choice([1, 2, 3], p=[0.7, 0.23, 0.07])))
                    recs = list(rng.choice(nbrs, size=k, replace=False))
                    rec_seen = True
                elif mode >= 0.85 and nbrs:
                    k = min(len(nbrs), int(rng.choice([1, 2], p=[0.7, 0.3])))
                    recs = list(rng.choice(nbrs, size=k, replace=False))
                    donor_seen = False
                    rec_seen = True
                if rec_seen:
                    rho = float(rng.uniform(*gen.share))
                    split = rng.dirichlet(np.ones(len(recs)))
                    kind_pair = "permanent_transfer" if perm else "edge"
                    for r, frac in zip(recs, split, strict=True):
                        off = int(rng.integers(-gen.start_jitter_hh, gen.start_jitter_hh + 1))
                        r_start = max(lo, st + off)
                        r_amp = rho * frac * units * sigma[site]
                        r_prof = np.interp(np.linspace(0, 1, max(1, n - off)),
                                           np.linspace(0, 1, len(prof)), prof)
                        both = donor_seen
                        add(r, r_start, end, r_amp, r_prof, kind_pair if both else (
                            "permanent_shift" if perm else "solo"),
                            [site] if both else [], r_amp / sigma[r], name, group)
                if donor_seen:
                    partners = list(recs) if rec_seen else []
                    k2 = ("permanent_transfer" if perm else "edge") if rec_seen else kind
                    add(site, st, end, amp, prof, k2, partners, units, name, group)
                    if rec_seen:
                        truth[-1].partner_amp = {r: rho * f * units * sigma[site]
                                                 for r, f in zip(recs, split, strict=True)}
    if "negatives" in kinds:
        n_rout = 3
        for site in rng.choice(panel.sites, size=n_rout, replace=False):
            for _ in range(rng.poisson(12 * years)):
                group += 1
                st = int(rng.integers(lo, hi - 6))
                dur = int(np.exp(rng.uniform(np.log(6), np.log(3 * DAY))))
                q = float(rng.choice([0.5, 1.0, 1.5, 2.0])) * float(rng.choice([-1, 1]))
                shape, _ = shape_of(site, st, dur)
                prof = _profile(n=dur, width=2, drift=0.0, mid_step=None, staged=0) * shape
                add(site, st, st + dur, q, prof, "routine", [], abs(q) / sigma[site], "proportional", group)
        for site in panel.sites:
            for _ in range(rng.poisson(NEG_PER_YEAR * years)):
                group += 1
                st = int(rng.integers(lo, hi - 6))
                units = float(np.exp(rng.uniform(np.log(2), np.log(8))))
                sign = float(rng.choice([-1, 1]))
                dur = int(np.exp(rng.uniform(np.log(DAY), np.log(30 * DAY))))
                width = int(rng.integers(4, 25))
                add(site, st, st + dur, sign * units * sigma[site],
                    _profile(n=dur, width=width, drift=0, mid_step=None, staged=0),
                    "ramp", [], units, "constant", group)
            for _ in range(rng.poisson(NEG_PER_YEAR * years)):
                group += 1
                st = int(rng.integers(lo, hi - 6))
                dur = int(rng.integers(1, 6))
                units = float(rng.uniform(3, 10))
                add(site, st, st + dur, float(rng.choice([-1, 1])) * units * sigma[site], None,
                    "pulse", [], units, "constant", group)
            for _ in range(rng.poisson(NEG_PER_YEAR * years)):
                group += 1
                st = _start(rng, starts, lo, hi - 6)
                dur = _duration(rng, "default")
                units = _units(rng)
                add(site, st, st + dur, float(rng.choice([-1, 1])) * units * sigma[site], None,
                    "solo", [], units, "constant", group)
        for _ in range(rng.poisson(0.3 * years * len(panel.sites))):
            a, b = rng.choice(panel.sites, size=2, replace=False)
            if b in panel.neighbours[a]:
                continue
            group += 1
            st = _start(rng, starts, lo, hi - 6)
            dur = _duration(rng, "default")
            units = _units(rng)
            add(a, st, st + dur, -units * sigma[a], None, "placebo", [], units, "constant", group)
            add(b, st, st + dur, units * 0.8 * sigma[b], None, "placebo", [], units, "constant", group)
        t = np.arange(hi - lo)
        cm = sum(np.sin(2 * np.pi * t / (DAY * float(rng.uniform(3, 10))) + rng.uniform(0, 6.3))
                 for _ in range(3)) / 3
        for site in panel.sites:
            out[site][lo:hi] += float(rng.uniform(0.5, 1.5)) * 0.8 * sigma[site] * cm
    if "pv_growth" in kinds:
        for site in rng.choice(panel.sites, size=3, replace=False):
            group += 1
            rate = float(rng.uniform(0.2, 1.0))
            t = np.arange(hi - lo)
            hour = (t % DAY) / 2.0
            doy = ((lo + t) // DAY) % 365
            season = 0.6 + 0.4 * np.cos(2 * np.pi * (doy - 172) / 365)
            half_day = 4 + 4 * season
            sun = np.clip(np.sin(np.pi * (hour - (12 - half_day)) / (2 * half_day)), 0, None)
            out[site][lo:hi] -= rate * (t / (365.25 * DAY)) * season * sun
            truth.append(Truth(site, lo, None, rate, rate / sigma[site], "pv_growth", group))
    if "seasonal_bias" in kinds:
        for site in rng.choice(panel.sites, size=5, replace=False):
            group += 1
            t = np.arange(hi - lo)
            amp = float(rng.uniform(0.5, 1.5)) * sigma[site] * 2
            year = t // int(365.25 * DAY)
            gain = rng.uniform(0.5, 1.5, size=int(year.max()) + 1)[year]
            lag = rng.uniform(-20, 20, size=int(year.max()) + 1)[year] * DAY
            out[site][lo:hi] += amp * gain * np.sin(2 * np.pi * (t - lag) / (365.25 * DAY) + 1.0)
            truth.append(Truth(site, lo, None, amp, amp / sigma[site], "seasonal_bias", group))
    assert GRID_LENGTH == 78816
    return out, truth

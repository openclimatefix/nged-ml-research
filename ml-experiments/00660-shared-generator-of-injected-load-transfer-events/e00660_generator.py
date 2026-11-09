"""Shared generator of injected load-transfer events (experiment 660).

The generator adds synthetic load-shift events to a panel of real half-hourly power series, one
series per Primary substation, and returns the injected panel together with a ground-truth table
that says exactly what was added. The panel is a deterministic function of the base panel and the
table (`apply_events`), so a consumer can rebuild the injected delta from the table alone.

**Where it lives and how to import it.** This module belongs to experiment 660 and sits in that
experiment's folder. Another experiment imports it with that folder on the path, either
``PYTHONPATH=ml-experiments/00660-shared-generator-of-injected-load-transfer-events`` or, inside a
script, ``sys.path.insert(0, <that folder>)``, followed by ``import e00660_generator as gen``.
Moving it to a shared package would need the maintainer's review.

**Conventions.**

- Index ``i`` of every array is the half-hour starting at ``origin + 30 min * i`` (UTC).
- ``end`` is exclusive. ``end is None`` means the event is still running at the end of the window.
- A MW series is signed active power. An MVA series is apparent power (never negative), and an
  injected value there is the magnitude ``|base + delta|``; MW and MVA are never summed. An MVA
  magnitude can fold a large loss through zero; the table records the requested signed change,
  not the folded one.
- Time without an injected event is unlabelled, not negative.
- The panel may hold no finite reading at or after ``GeneratorConfig.cutoff`` (default
  2026-07-01), and no event starts there; an open event changes only half-hours that have a
  reading.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl

DAY: Final[int] = 48
"""Half-hours per day."""
YEAR_HH: Final[float] = 365.25 * DAY
CUTOFF: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""First instant of the final-test window; no panel reading or event window reaches it."""

POSITIVE_EDGE: Final[str] = "edge_transfer"
POSITIVE_SHIFT: Final[str] = "shift"
HARD_NEGATIVE: Final[str] = "hard_negative"

EDGE_CLASSES: Final[tuple[str, ...]] = (
    "edge_transfer",
    "edge_transfer_unseen_donor",
    "edge_transfer_unseen_recipient",
)
SHIFT_CLASSES: Final[tuple[str, ...]] = ("temporary_shift", "permanent_shift")
NEGATIVE_CLASSES: Final[tuple[str, ...]] = (
    "routine_step",
    "ramp",
    "pulse",
    "placebo_pair",
    "common_mode",
)
EVENT_COLUMNS: Final[dict[str, pl.DataType | type[pl.DataType]]] = {
    "event_id": pl.Int64,
    "group_id": pl.Int64,
    "site": pl.String,
    "role": pl.String,
    "event_class": pl.String,
    "label": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "residual_frac": pl.Float64,
    "delta": pl.Float64,
    "units": pl.Float64,
    "shape": pl.String,
    "prop_share": pl.Float64,
    "shape_source": pl.String,
    "ramp_in_hh": pl.Int64,
    "ramp_out_hh": pl.Int64,
    "drift": pl.Float64,
    "mid_factor": pl.Float64,
    "staged_hh": pl.Int64,
    "share": pl.Float64,
    "period_hh": pl.Float64,
    "phase": pl.Float64,
    "partners": pl.List(pl.String),
    "chain_parent": pl.Int64,
}
"""Schema of the ground-truth table. `delta` is the signed peak change in the panel's unit."""


@dataclass(frozen=True)
class PowerPanel:
    """A real power panel: one half-hourly series per Primary, with the data the generator needs.

    `power` holds MW for a signed-MW Primary and MVA for an MVA-only Primary (`is_mva`). A NaN is
    a half-hour with no reading. `noise` is one noise unit of each series in its own unit, the
    scale in which event sizes are drawn. `neighbours` is the electrical edge list.
    """

    power: dict[str, np.ndarray]
    is_mva: dict[str, bool]
    noise: dict[str, float]
    neighbours: dict[str, list[str]]
    origin: datetime

    @property
    def sites(self) -> list[str]:
        """Primaries in a fixed (sorted) order, so a seed gives the same world every time."""
        return sorted(self.power)

    @property
    def length(self) -> int:
        """Number of half-hours in the panel."""
        return len(next(iter(self.power.values())))

    def time_index(self, *, when: datetime) -> int:
        """Half-hour index of a UTC datetime (may fall outside the panel)."""
        return int((when - self.origin).total_seconds() // 1800)

    def validate(self, *, cutoff: datetime = CUTOFF) -> None:
        """Raise `ValueError` if the panel is malformed or holds a reading at or after `cutoff`."""
        if not self.power:
            raise ValueError("the panel holds no series")
        lengths = {len(v) for v in self.power.values()}
        if len(lengths) != 1:
            raise ValueError(f"series differ in length: {sorted(lengths)}")
        for site in self.power:
            if site not in self.is_mva or site not in self.noise:
                raise ValueError(f"{site}: missing is_mva or noise entry")
            if not self.noise[site] > 0:
                raise ValueError(f"{site}: noise must be positive")
            if self.is_mva[site] and np.nanmin(self.power[site], initial=0.0) < 0:
                raise ValueError(f"{site}: an MVA series must not be negative")
        for a, nbrs in self.neighbours.items():
            for b in nbrs:
                if b in self.power and a in self.power and a not in self.neighbours.get(b, []):
                    raise ValueError(f"edge {a}-{b} is not symmetric")
        stop = self.time_index(when=cutoff)
        if stop < self.length:
            for site, series in self.power.items():
                if np.isfinite(series[max(stop, 0) :]).any():
                    raise ValueError(f"{site}: a reading at or after the cutoff {cutoff}")


@dataclass(frozen=True)
class GeneratorConfig:
    """Every range and probability of the generator, with the default and its evidence.

    The page `docs/ml-experiments/00660-.../result.md` lists each default with its evidence and
    status (measured, literature, or unmeasured guess). Sensitivity arms are built with
    `dataclasses.replace` or the named constructors below.
    """

    cutoff: datetime = CUTOFF

    # --- how many events and of which class
    events_per_primary_year: float = 1.2
    class_probs: tuple[tuple[str, float], ...] = (
        ("edge_transfer", 0.55),
        ("edge_transfer_unseen_donor", 0.10),
        ("edge_transfer_unseen_recipient", 0.15),
        ("temporary_shift", 0.10),
        ("permanent_shift", 0.10),
    )
    recipient_count_probs: tuple[float, float, float] = (0.70, 0.23, 0.07)
    """Probability of one, two, or three recipients of a donor with a metered recipient."""
    overlap_prob: float = 0.10
    """Chance that an event starts inside an earlier event at the same Primary."""
    chain_prob: float = 0.10
    """Chance that an event's donor is a recipient of an earlier event, starting inside it."""

    # --- size
    units_range: tuple[float, float] = (1.0, 20.0)
    core_units_range: tuple[float, float] = (2.0, 8.0)
    core_mass: float = 0.5
    """Share of events drawn log-uniformly from `core_units_range` instead of `units_range`."""
    share_range: tuple[float, float] = (0.5, 1.1)
    """Recipients' total share of the donor's change (receiver share)."""

    # --- time
    duration_bands: tuple[tuple[float, float, float], ...] = ((0.125, 112.0, 1.0),)
    """(low days, high days, weight): log-uniform within a band; 0.125 days is 3 hours."""
    ramp_hh: tuple[int, int] = (0, 4)
    """Half-hours over which each end of an event ramps; 0 is a step within one half-hour."""
    staged_return_prob: float = 0.0
    """Chance that the last fifth of an event runs at half level (a staged return)."""
    partial_return_prob: float = 0.10
    """Chance that the level after the end settles at 0.2 to 0.8 of the event's level."""
    mid_step_prob: float = 0.05
    """Chance of a second step to 0.5 to 1.5 times the level halfway through an event."""
    drift_range: tuple[float, float] = (-0.2, 0.2)
    """Linear drift of the level across an event, as a fraction of the level."""
    recipient_start_offset_hh: tuple[int, int] = (0, 36)
    """Half-hours by which a recipient's start differs from the donor's (the end is shared)."""
    proportional_prob: float = 1.0
    """Chance that a step follows the donor's daily profile at least in part (else constant MW)."""
    proportional_share_range: tuple[float, float] = (0.3, 1.0)
    """Share of a proportional step that follows the donor's profile; the rest is constant MW."""

    # --- hard negatives
    hard_negatives: bool = True
    routine_primaries: tuple[int, int] = (3, 4)
    routine_steps_per_year: float = 12.0
    routine_steps: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    """Quantised step sizes (in the panel's unit) of the routine generator steps."""
    ramps_per_primary_year: float = 1.0
    ramp_hours: tuple[float, float] = (2.0, 12.0)
    pulses_per_primary_year: float = 1.0
    pulse_hh: tuple[int, int] = (1, 5)
    placebo_pairs_per_primary_year: float = 0.3
    common_mode_gain: float = 0.8
    """Peak of the shared weather-like perturbation, in noise units."""

    def __post_init__(self) -> None:
        """Reject a config whose probabilities cannot be sampled."""
        if abs(sum(p for _, p in self.class_probs) - 1.0) > 1e-9:
            raise ValueError("class_probs must sum to 1")
        if abs(sum(self.recipient_count_probs) - 1.0) > 1e-9:
            raise ValueError("recipient_count_probs must sum to 1")
        unknown = {c for c, _ in self.class_probs} - set(EDGE_CLASSES + SHIFT_CLASSES)
        if unknown:
            raise ValueError(f"unknown event classes: {sorted(unknown)}")


def default_config() -> GeneratorConfig:
    """The default configuration."""
    return GeneratorConfig()


def short_heavy(config: GeneratorConfig | None = None) -> GeneratorConfig:
    """Sensitivity mixture: 70% of events last 3 hours to 7 days, 30% 7 to 112 days."""
    return replace(
        config or GeneratorConfig(), duration_bands=((0.125, 7.0, 0.7), (7.0, 112.0, 0.3))
    )


def long_heavy(config: GeneratorConfig | None = None) -> GeneratorConfig:
    """Sensitivity mixture: 30% of events last 3 hours to 7 days, 70% 7 to 112 days."""
    return replace(
        config or GeneratorConfig(), duration_bands=((0.125, 7.0, 0.3), (7.0, 112.0, 0.7))
    )


def constant_mw_square(config: GeneratorConfig | None = None) -> GeneratorConfig:
    """Constant-MW square steps: no ramp, no drift, no second step, no staged or partial return."""
    return replace(
        config or GeneratorConfig(),
        proportional_prob=0.0,
        ramp_hh=(0, 0),
        drift_range=(0.0, 0.0),
        mid_step_prob=0.0,
        staged_return_prob=0.0,
        partial_return_prob=0.0,
    )


def shaped_transfers(config: GeneratorConfig | None = None) -> GeneratorConfig:
    """Shaped transfers: follow the donor's daily profile, ramp, stage the return, short-heavy."""
    return replace(
        short_heavy(config),
        proportional_prob=1.0,
        proportional_share_range=(0.5, 1.0),
        ramp_hh=(0, 4),
        staged_return_prob=0.5,
    )


@dataclass(frozen=True)
class StartTimes:
    """Half-hour of day (UTC) and weekday (0 is Monday) of logged event onsets."""

    half_hour_of_day: np.ndarray
    weekday: np.ndarray


@dataclass
class InjectedWorld:
    """The output of one injection."""

    power: dict[str, np.ndarray]
    events: pl.DataFrame
    base: PowerPanel
    config: GeneratorConfig
    seed: int
    window: tuple[int, int] = (0, 0)


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=EVENT_COLUMNS)


def window_of(*, row: dict, length: int) -> tuple[int, int]:
    """Half-hours [lo, hi) a row can change: to the panel end if open or if it leaves a residue."""
    end = row["end"]
    reach = length if (end is None or row["residual_frac"] != 0.0) else end
    return int(row["start"]), int(min(reach, length))


def unit_profile(*, row: dict, source: np.ndarray | None, length: int) -> np.ndarray:
    """The profile of a row over its window, such that the change is ``delta * profile``.

    The profile is 1 on the plateau, ramps linearly over `ramp_in_hh` and `ramp_out_hh`
    half-hours, drifts linearly by `drift` across the event, steps by `mid_factor` halfway, drops
    to 0.5 for the last `staged_hh` half-hours, and settles at `residual_frac` after the end. A
    proportional step is multiplied by ``(1 - c) + c * q``, where ``c`` is `prop_share` and ``q`` is
    the shape source's power over the event divided by its mean, clipped to [0, 3] and
    renormalised to mean 1, so `delta` is the mean change.
    """
    lo, hi = window_of(row=row, length=length)
    out = np.zeros(max(hi - lo, 0))
    if hi <= lo:
        return out
    if row["event_class"] == "common_mode":
        t = np.arange(hi - lo)
        return np.sin(2 * np.pi * t / row["period_hh"] + row["phase"])
    open_end = row["end"] is None or row["end"] >= length
    stop = hi if open_end else int(row["end"])
    n = stop - lo
    p = np.ones(n)
    w_in = min(int(row["ramp_in_hh"]), n - 1)
    if w_in > 0:
        p[:w_in] = np.minimum(p[:w_in], (np.arange(w_in) + 1) / (w_in + 1))
    if not open_end:
        w_out = min(int(row["ramp_out_hh"]), n - 1)
        if w_out > 0:
            p[n - w_out :] = np.minimum(
                p[n - w_out :], ((np.arange(w_out) + 1) / (w_out + 1))[::-1]
            )
    p *= 1 + row["drift"] * (np.arange(n) / max(n, 1) - 0.5)
    if row["mid_factor"] != 1.0:
        p[n // 2 :] *= row["mid_factor"]
    if row["staged_hh"] > 0 and not open_end and n > 2 * row["staged_hh"]:
        p[n - int(row["staged_hh"]) :] *= 0.5
    if row["shape"] == "proportional" and source is not None:
        seg = source[lo:stop]
        ok = np.isfinite(seg)
        mean = float(seg[ok].mean()) if ok.any() else 0.0
        if mean > 0:
            q = np.where(ok, np.clip(seg / mean, 0.0, 3.0), 1.0)
            c = float(row["prop_share"])
            p *= (1 - c) + c * q / q.mean()
    out[:n] = p
    out[n:] = row["residual_frac"]
    return out


def event_delta(*, row: dict, panel: PowerPanel) -> np.ndarray:
    """The change one table row makes to its Primary's series (length of the panel, mostly 0)."""
    length = panel.length
    delta = np.zeros(length)
    src = panel.power.get(row["shape_source"]) if row["shape"] == "proportional" else None
    prof = unit_profile(row=row, source=src, length=length)
    lo, hi = window_of(row=row, length=length)
    delta[lo:hi] = row["delta"] * prof
    return delta


def apply_events(*, panel: PowerPanel, events: pl.DataFrame) -> dict[str, np.ndarray]:
    """Rebuild the injected panel from the base panel and the ground-truth table.

    The change at a Primary is the sum of its rows' changes. A signed-MW series becomes base plus
    change. An MVA series becomes ``|base + change|`` where the change is non-zero, so MW and MVA
    are never summed and an MVA series never goes negative. Half-hours with no reading stay NaN.
    """
    rows = events.to_dicts()
    out: dict[str, np.ndarray] = {}
    for site in panel.sites:
        base = panel.power[site]
        acc = np.zeros(panel.length)
        for row in rows:
            if row["site"] == site:
                acc += event_delta(row=row, panel=panel)
        moved = base + acc
        if panel.is_mva[site]:
            moved = np.abs(moved)
        out[site] = np.where((acc != 0) & np.isfinite(base), moved, base)
    return out


def _weighted_choice(rng: np.random.Generator, pairs: tuple[tuple[str, float], ...]) -> str:
    names = [n for n, _ in pairs]
    probs = np.array([p for _, p in pairs])
    return str(names[int(rng.choice(len(names), p=probs / probs.sum()))])


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


class _Builder:
    """Draws rows for the ground-truth table."""

    def __init__(
        self,
        *,
        panel: PowerPanel,
        config: GeneratorConfig,
        rng: np.random.Generator,
        window: tuple[int, int],
        starts: StartTimes | None,
    ) -> None:
        self.panel, self.config, self.rng = panel, config, rng
        self.lo, self.hi = window
        self.rows: list[dict] = []
        self.next_event = 0
        self.next_group = 0
        n = panel.length
        t = np.arange(n)
        origin_hh = (panel.origin.hour * 60 + panel.origin.minute) // 30
        self.hod = (origin_hh + t) % DAY
        self.weekday = (panel.origin.weekday() + (origin_hh + t) // DAY) % 7
        self.starts = starts
        self.years = (self.hi - self.lo) / YEAR_HH
        self._finite = {
            site: np.concatenate([[0], np.cumsum(np.isfinite(v))])
            for site, v in panel.power.items()
        }

    def first_data(self, *, site: str) -> int:
        """Index of the first half-hour with a reading at the Primary (or the window start)."""
        c = self._finite[site]
        hit = np.flatnonzero(c[1:] > c[:-1])
        return max(self.lo, int(hit[0])) if len(hit) else self.lo

    def has_data(self, *, site: str, at: int) -> bool:
        """True if the Primary has readings for at least half of the day starting at `at`."""
        c = self._finite[site]
        hi = min(at + DAY, len(c) - 1)
        return bool(c[hi] - c[at] >= DAY // 2)

    def row(self, **kw: object) -> dict:
        """One table row with defaults filled in; returns it after appending."""
        base: dict = {
            "event_id": self.next_event,
            "group_id": self.next_group,
            "end": None,
            "residual_frac": 0.0,
            "units": 0.0,
            "shape": "constant",
            "prop_share": 0.0,
            "shape_source": "",
            "ramp_in_hh": 0,
            "ramp_out_hh": 0,
            "drift": 0.0,
            "mid_factor": 1.0,
            "staged_hh": 0,
            "share": 1.0,
            "period_hh": 0.0,
            "phase": 0.0,
            "partners": [],
            "chain_parent": -1,
        }
        base.update(kw)
        base["units"] = abs(base["delta"]) / self.panel.noise[base["site"]]
        self.next_event += 1
        self.rows.append(base)
        return base

    def start_time(self, *, lo: int | None = None, hi: int | None = None) -> int:
        """A start index in [lo, hi) at a logged hour and weekday when `starts` is given."""
        lo = self.lo if lo is None else lo
        hi = self.hi - 6 if hi is None else hi
        if self.starts is not None and len(self.starts.weekday) > 0:
            k = int(self.rng.integers(len(self.starts.weekday)))
            hh, wd = int(self.starts.half_hour_of_day[k]), int(self.starts.weekday[k])
            window = np.arange(lo, hi)
            ok = window[(self.hod[lo:hi] == hh) & (self.weekday[lo:hi] == wd)]
            if len(ok) > 0:
                return int(self.rng.choice(ok))
        return int(self.rng.integers(lo, hi))

    def duration(self) -> int:
        """A duration in half-hours from the configured bands."""
        bands = self.config.duration_bands
        w = np.array([b[2] for b in bands])
        lo_d, hi_d, _ = bands[int(self.rng.choice(len(bands), p=w / w.sum()))]
        return max(6, int(_log_uniform(self.rng, lo_d * DAY, hi_d * DAY)))

    def units(self) -> float:
        """A size in noise units."""
        c = self.config
        if self.rng.random() < c.core_mass:
            return _log_uniform(self.rng, *c.core_units_range)
        return _log_uniform(self.rng, *c.units_range)

    def shape_fields(self, *, n: int, permanent: bool) -> dict:
        """Ramp, drift, second step, staged and partial return for one event."""
        c, rng = self.config, self.rng
        out: dict = {
            "ramp_in_hh": int(rng.integers(c.ramp_hh[0], c.ramp_hh[1] + 1)),
            "ramp_out_hh": int(rng.integers(c.ramp_hh[0], c.ramp_hh[1] + 1)),
            "drift": float(rng.uniform(*c.drift_range)),
            "mid_factor": float(rng.uniform(0.5, 1.5)) if rng.random() < c.mid_step_prob else 1.0,
            "staged_hh": 0,
            "residual_frac": 0.0,
        }
        if not permanent:
            if rng.random() < c.staged_return_prob:
                out["staged_hh"] = max(1, int(0.2 * n))
            if rng.random() < c.partial_return_prob:
                out["residual_frac"] = float(rng.uniform(0.2, 0.8))
        return out

    def _pick_start_and_donor(
        self, *, parents: list[dict]
    ) -> tuple[int, list[str], str, int] | None:
        """Start index, Primaries with data then, the donor, and the chained parent (or -1)."""
        c, rng = self.config, self.rng
        parent: dict | None = None
        use_chain = False
        parent_win: tuple[int, int] | None = None
        if parents and rng.random() < c.chain_prob + c.overlap_prob:
            parent = parents[int(rng.integers(len(parents)))]
            use_chain = bool(rng.random() < c.chain_prob / (c.chain_prob + c.overlap_prob))
            end_p = parent["_end"] if parent["_end"] is not None else self.hi
            parent_win = (parent["start"], min(end_p, self.hi - 6))
            if parent_win[1] <= parent_win[0] + 1:
                parent = parent_win = None
        for _ in range(20):
            if parent_win is not None:
                st = self.start_time(lo=parent_win[0] + 1, hi=parent_win[1])
            else:
                st = self.start_time()
            able = [x for x in self.panel.sites if self.has_data(site=x, at=st)]
            if able:
                break
        else:
            return None
        donor = str(rng.choice(able))
        chain_parent = -1
        if parent is not None:
            pool = (
                [x for x in parent["_recipients"] if x in able]
                if use_chain
                else ([parent["_donor"]] if parent["_donor"] in able else [])
            )
            if pool:
                donor = str(rng.choice(pool))
                chain_parent = parent["event_id"]
        return st, able, donor, chain_parent

    def positive_event(self, *, parents: list[dict]) -> None:
        """Draw one positive event (an edge transfer or a shift) and append its rows."""
        c, rng, panel = self.config, self.rng, self.panel
        cls = _weighted_choice(rng, c.class_probs)
        picked = self._pick_start_and_donor(parents=parents)
        if picked is None:
            return
        st, able, donor, chain_parent = picked
        permanent = cls == "permanent_shift"
        dur = self.duration()
        end: int | None = None if permanent or st + dur >= self.hi else st + dur
        n = (self.hi if end is None else end) - st
        units = self.units()
        fields = self.shape_fields(n=n, permanent=permanent)
        proportional = rng.random() < c.proportional_prob
        shape = "proportional" if proportional else "constant"
        prop_share = float(rng.uniform(*c.proportional_share_range)) if proportional else 0.0
        group = self.next_group
        self.next_group += 1
        nbrs = [x for x in panel.neighbours.get(donor, []) if x in able]
        rho = float(rng.uniform(*c.share_range))
        recipients: list[str] = []
        if cls in ("edge_transfer", "edge_transfer_unseen_donor") and nbrs:
            k = min(len(nbrs), int(rng.choice([1, 2, 3], p=list(c.recipient_count_probs))))
            recipients = [str(x) for x in rng.choice(nbrs, size=k, replace=False)]
        elif cls in ("edge_transfer", "edge_transfer_unseen_donor"):
            cls = "temporary_shift"  # no metered neighbour exists to receive
        donor_seen = cls != "edge_transfer_unseen_donor"
        sign = -1.0 if cls in EDGE_CLASSES else float(rng.choice([-1.0, 1.0]))
        d_delta = sign * units * panel.noise[donor]
        label = POSITIVE_EDGE if cls in EDGE_CLASSES else POSITIVE_SHIFT
        common = {
            "group_id": group,
            "event_class": cls,
            "label": label,
            "start": st,
            "end": end,
            "shape": shape,
            "prop_share": prop_share,
            "share": rho,
            "chain_parent": chain_parent,
            **fields,
        }
        first: dict | None = None
        if donor_seen:
            first = self.row(
                site=donor,
                role="donor" if cls in EDGE_CLASSES else "solo",
                delta=d_delta,
                shape_source=donor,
                partners=recipients,
                **common,
            )
        if recipients:
            split = rng.dirichlet(np.ones(len(recipients)))
            for r, frac in zip(recipients, split, strict=True):
                offset = int(
                    rng.integers(c.recipient_start_offset_hh[0], c.recipient_start_offset_hh[1] + 1)
                )
                r_start = max(self.lo, min(st + offset, st + max(n - 6, 0)))
                self.row(
                    site=r,
                    role="recipient",
                    delta=-rho * float(frac) * d_delta,
                    shape_source=donor if donor_seen else r,
                    partners=[donor] if donor_seen else [],
                    **{**common, "start": r_start},
                )
        head = first if first is not None else self.rows[-1]
        parents.append(
            {
                "event_id": head["event_id"],
                "start": st,
                "_end": end,
                "_donor": donor,
                "_recipients": recipients,
            }
        )

    def hard_negatives(self) -> None:  # noqa: PLR0912 - one list of independent draws
        """Append the hard negatives: routine steps, ramps, pulses, placebo pairs, common mode."""
        c, rng, panel = self.config, self.rng, self.panel
        sites = panel.sites
        k = int(rng.integers(c.routine_primaries[0], c.routine_primaries[1] + 1))
        for routine_site in rng.choice(sites, size=min(k, len(sites)), replace=False):
            site = str(routine_site)
            for _ in range(rng.poisson(c.routine_steps_per_year * self.years)):
                st = self.start_time(hi=self.hi - 6)
                if not self.has_data(site=site, at=st):
                    continue
                dur = int(_log_uniform(rng, 6, 3 * DAY))
                size = float(rng.choice(c.routine_steps)) * float(rng.choice([-1, 1]))
                self.next_group += 1
                self.row(
                    group_id=self.next_group,
                    site=site,
                    role="solo",
                    event_class="routine_step",
                    label=HARD_NEGATIVE,
                    start=st,
                    end=min(st + dur, self.hi),
                    delta=size,
                    shape="proportional",
                    prop_share=1.0,
                    shape_source=site,
                    ramp_in_hh=2,
                    ramp_out_hh=2,
                )
        for site in sites:
            for _ in range(rng.poisson(c.ramps_per_primary_year * self.years)):
                st = int(rng.integers(self.lo, self.hi - 6))
                if not self.has_data(site=site, at=st):
                    continue
                dur = int(_log_uniform(rng, DAY, 30 * DAY))
                w = int(rng.integers(int(c.ramp_hours[0] * 2), int(c.ramp_hours[1] * 2) + 1))
                units = float(_log_uniform(rng, 2.0, 8.0))
                self.next_group += 1
                self.row(
                    group_id=self.next_group,
                    site=site,
                    role="solo",
                    event_class="ramp",
                    label=HARD_NEGATIVE,
                    start=st,
                    end=min(st + dur, self.hi),
                    delta=float(rng.choice([-1, 1])) * units * panel.noise[site],
                    ramp_in_hh=w,
                    ramp_out_hh=w,
                )
            for _ in range(rng.poisson(c.pulses_per_primary_year * self.years)):
                st = int(rng.integers(self.lo, self.hi - 6))
                if not self.has_data(site=site, at=st):
                    continue
                dur = int(rng.integers(c.pulse_hh[0], c.pulse_hh[1] + 1))
                units = float(rng.uniform(3.0, 10.0))
                self.next_group += 1
                self.row(
                    group_id=self.next_group,
                    site=site,
                    role="solo",
                    event_class="pulse",
                    label=HARD_NEGATIVE,
                    start=st,
                    end=min(st + dur, self.hi),
                    delta=float(rng.choice([-1, 1])) * units * panel.noise[site],
                )
        n_pairs = rng.poisson(c.placebo_pairs_per_primary_year * self.years * len(sites))
        for _ in range(n_pairs):
            a, b = (str(x) for x in rng.choice(sites, size=2, replace=False))
            if b in panel.neighbours.get(a, []):
                continue
            st = self.start_time()
            if not (self.has_data(site=a, at=st) and self.has_data(site=b, at=st)):
                continue
            dur = self.duration()
            end = None if st + dur >= self.hi else st + dur
            units = self.units()
            rho = float(rng.uniform(*c.share_range))
            self.next_group += 1
            common = {
                "group_id": self.next_group,
                "event_class": "placebo_pair",
                "label": HARD_NEGATIVE,
                "start": st,
                "end": end,
                "share": rho,
            }
            self.row(site=a, role="placebo_a", delta=-units * panel.noise[a], partners=[], **common)
            self.row(
                site=b, role="placebo_b", delta=rho * units * panel.noise[b], partners=[], **common
            )
        self.next_group += 1
        group = self.next_group
        for _ in range(3):
            period = float(rng.uniform(3, 10)) * DAY
            phase = float(rng.uniform(0, 2 * np.pi))
            for site in sites:
                gain = float(rng.uniform(0.5, 1.5)) * c.common_mode_gain / 3
                self.row(
                    group_id=group,
                    site=site,
                    role="common_mode",
                    event_class="common_mode",
                    label=HARD_NEGATIVE,
                    start=self.first_data(site=site),
                    end=None,
                    delta=gain * panel.noise[site],
                    period_hh=period,
                    phase=phase,
                )


def inject(
    *,
    panel: PowerPanel,
    config: GeneratorConfig | None = None,
    seed: int,
    start_times: StartTimes | None = None,
) -> InjectedWorld:
    """Inject events into a copy of the panel and return the injected panel and its truth table.

    Args:
        panel: the real, un-injected power panel (injection happens before cleaning and before
            any expected-power residual is computed, on real unshifted bases).
        config: ranges and probabilities; `default_config()` when omitted.
        seed: the random seed; the same panel, config, and seed give the same world.
        start_times: logged onset hours and weekdays to draw start times from; uniform when None.

    Returns:
        The `InjectedWorld` with the injected `power`, the ground-truth `events` table, and the
        window [lo, hi) the events start in (the whole panel before the cutoff). To inject into
        part of a series, slice the panel first.

    Raises:
        ValueError: if the panel is malformed, holds a reading at or after `config.cutoff`, or the
            window is too short to hold an event.
    """
    config = config or GeneratorConfig()
    cutoff = min(config.cutoff, CUTOFF)  # a later cutoff never reaches the final-test window
    panel.validate(cutoff=cutoff)
    stop = min(panel.length, panel.time_index(when=cutoff))
    lo, hi = 0, stop
    if hi - lo < 4 * DAY:
        raise ValueError("the window is shorter than four days")
    rng = np.random.default_rng(seed)
    builder = _Builder(panel=panel, config=config, rng=rng, window=(lo, hi), starts=start_times)
    parents: list[dict] = []
    for _ in range(
        sum(rng.poisson(config.events_per_primary_year * builder.years) for _ in panel.sites)
    ):
        builder.positive_event(parents=parents)
    if config.hard_negatives:
        builder.hard_negatives()
    events = (
        pl.DataFrame(builder.rows, schema=EVENT_COLUMNS) if builder.rows else _empty_events()
    ).sort(["event_id"])
    return InjectedWorld(
        power=apply_events(panel=panel, events=events),
        events=events,
        base=panel,
        config=config,
        seed=seed,
        window=(lo, hi),
    )


def event_windows(*, events: pl.DataFrame, length: int) -> dict[str, np.ndarray]:
    """For each Primary, a boolean mask of half-hours inside any of its event windows."""
    out: dict[str, np.ndarray] = {}
    for row in events.to_dicts():
        lo, hi = window_of(row=row, length=length)
        out.setdefault(row["site"], np.zeros(length, dtype=bool))[lo:hi] = True
    return out


def start_times_from(*, onset_index: np.ndarray, origin: datetime) -> StartTimes:
    """Hour-of-day and weekday of logged onsets, from half-hour indices counted from `origin`."""
    when = [origin + timedelta(minutes=30 * int(i)) for i in onset_index]
    hh = np.array([(w.hour * 60 + w.minute) // 30 for w in when], dtype=int)
    wd = np.array([w.weekday() for w in when], dtype=int)
    return StartTimes(half_hour_of_day=hh, weekday=wd)


def shuffled_placebo(*, panel: PowerPanel, events: pl.DataFrame, seed: int) -> pl.DataFrame:
    """The same events at random start times: sizes, durations, donors, and recipients unchanged.

    Every group (a donor with its recipients) moves rigidly by one random shift, so the marginal
    distribution of sizes, durations, and topology is kept while the link between an event and
    the real power and weather around it is broken. Use it as the placebo arm beside a contrast.
    Common-mode rows stay where they are.
    """
    rng = np.random.default_rng(seed)
    stop = min(panel.length, panel.time_index(when=CUTOFF))
    rows = events.to_dicts()
    shift: dict[int, int] = {}
    for row in rows:
        gid = row["group_id"]
        if gid in shift or row["event_class"] == "common_mode":
            continue
        group = [r for r in rows if r["group_id"] == gid]
        first = min(r["start"] for r in group)
        last = max((stop if r["end"] is None else r["end"]) for r in group)
        lo, hi = -first, stop - 1 - last
        shift[gid] = int(rng.integers(lo, hi + 1)) if hi > lo else 0
    for row in rows:
        d = shift.get(row["group_id"], 0)
        row["start"] += d
        if row["end"] is not None:
            row["end"] += d
    return pl.DataFrame(rows, schema=EVENT_COLUMNS)

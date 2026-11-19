"""Experiment 709: the era injector and the switching-event injector, with separate worlds.

**Era injector.** Four kinds from the plan, built in signed power (a share of capacity) and then
passed through an absolute value at the eight MVA-only Primaries, because an MVA series is never
negative:

- `level_m10`, `level_m25`: a level step of -10% or -25% of capacity (a generator connects);
- `level_p15`: a level step of +15% of capacity (a load connects);
- `solar_25`: a reduction with a peak of 25% of capacity that follows a *perturbed* ERA5 shortwave
  radiation field (each day multiplied by a log-normal factor), so the classifier, which reads the
  unperturbed ERA5 field, does not see the shape it is tested on;
- `curriculum_step`: a persistent step of 5% to 30% of capacity with a flat, midday-weighted, or
  evening-weighted daily profile, used by the training curriculum only;
- `engine`: an embedded engine of -30% of capacity that runs irregularly (stage 0 only, in its own
  `engine` world; the bank keeps the four kinds above).

**Switching-event injector.** A transfer built on the shared generator of experiment 660
(`e00660_generator`): a donor loses demand with a daily shape that follows its own demand, and the
neighbouring recipient gains the share `rho` of it, drawn uniformly from 0.9 to 1.0 so that the
pair's sum is nearly conserved, and the event reverts after a duration. `apply_events` of that
module turns the rows into power.
The `hidden_mirror` kind writes only the donor's row, which is an unmetered recipient.

**Worlds.** The tuning, curriculum, bank, and retrained worlds use disjoint seed ranges
(`WORLD_SEED_BASE`) and different generator settings (`WORLD_SETTINGS`), so no setting that is
fitted on one world is scored on the same draw. The host rules of plan section 20 item 13 are in
`choose_hosts`.

Run with the folder of experiment 660 on `PYTHONPATH`.
"""

from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import e00660_generator as gen  # ty: ignore[unresolved-import]
import e00709_features as features
import numpy as np
import polars as pl

EraKindType = Literal[
    "level_m10", "level_m25", "level_p15", "solar_25", "curriculum_step", "engine"
]
SwitchKindType = Literal["transfer", "hidden_mirror", "long", "very_long"]
WorldNameType = Literal["tuning", "curriculum", "bank", "retrained", "engine"]
ProfileType = Literal["flat", "midday", "evening"]
DurationBandType = Literal["<60", "60-120", ">120"]

SOLAR_PEAK_WM2: Final[float] = 700.0
"""Shortwave radiation at which the solar-shaped reduction reaches its peak."""
MIN_HISTORY_DAYS: Final[int] = 60
"""A host needs this many days of detector features before the onset."""
MAX_HOST_TRIES: Final[int] = 5000
EXCLUDED_HOSTS: Final[tuple[str, ...]] = ("D8", "D15", "D18")
"""D8 keeps its real eras unmixed with injected ones; D15 and D18 have no climatology."""
N_ERA_PER_PASS: Final[int] = 2
"""Era hosts per pass. Four was infeasible with about 13 hosts and the separation rules."""
SWITCH_ROTATION: Final[tuple[tuple[int, int], ...]] = ((0, 2), (1, 3), (2, 3), (0, 1))
"""Indices into a world's `switch_kinds` for the two switching pairs of pass `i % 4`.

Four consecutive passes carry every kind of a world (plain transfer, hidden mirror, long)."""
ENGINE_RUN_PROBABILITY: Final[float] = 0.5
"""Chance that the irregular engine of stage 0 runs on a given day."""
GENERATOR_NOISE: Final[float] = 0.01
"""Positive placeholder noise unit of the generator's panel; rows carry their own delta."""

WORLD_SEED_BASE: Final[dict[WorldNameType, int]] = {
    "tuning": 709_100_000,
    "curriculum": 709_200_000,
    "bank": 709_300_000,
    "retrained": 709_400_000,
    "engine": 709_500_000,
}
"""Disjoint seed ranges, 100,000 wide, so no two worlds can share a draw."""
SEED_RANGE: Final[int] = 100_000


@dataclass(frozen=True)
class WorldSettings:
    """The generator settings of one world.

    Attributes:
        era_kinds: Era kinds that the world draws from, in the order the hosts use them.
        era_size_range: Range of the absolute size (share of capacity) of a random-size era.
        transfer_size_range: Range of the donor's share of capacity that a transfer moves.
        duration_jitter: A bootstrapped verified duration is multiplied by `exp(U(-j, j))`.
        solar_noise: Standard deviation of the log of the daily solar perturbation factor.
        onset: `month_start` puts every onset at 00:00 UTC of one day, `random` draws it.
        switch_kinds: Kinds that `SWITCH_ROTATION` picks two from for each pass.
    """

    era_kinds: tuple[EraKindType, ...]
    era_size_range: tuple[float, float]
    transfer_size_range: tuple[float, float]
    duration_jitter: float
    solar_noise: float
    onset: Literal["month_start", "random"]
    switch_kinds: tuple[SwitchKindType, ...]


WORLD_SETTINGS: Final[dict[WorldNameType, WorldSettings]] = {
    "tuning": WorldSettings(
        era_kinds=("level_m10", "level_m25", "level_p15", "solar_25"),
        era_size_range=(0.07, 0.30),
        transfer_size_range=(0.06, 0.28),
        duration_jitter=0.30,
        solar_noise=0.12,
        onset="random",
        switch_kinds=("transfer", "transfer", "hidden_mirror", "long"),
    ),
    "curriculum": WorldSettings(
        era_kinds=("curriculum_step",) * 4,
        era_size_range=(0.05, 0.30),
        transfer_size_range=(0.05, 0.30),
        duration_jitter=0.20,
        solar_noise=0.15,
        onset="random",
        switch_kinds=("transfer", "transfer", "hidden_mirror", "long"),
    ),
    "bank": WorldSettings(
        era_kinds=("level_m10", "level_m25", "level_p15", "solar_25"),
        era_size_range=(0.10, 0.25),
        transfer_size_range=(0.08, 0.25),
        duration_jitter=0.10,
        solar_noise=0.20,
        onset="month_start",
        switch_kinds=("transfer", "transfer", "hidden_mirror", "long"),
    ),
    "engine": WorldSettings(
        era_kinds=("engine",) * 4,
        era_size_range=(0.10, 0.25),
        transfer_size_range=(0.08, 0.25),
        duration_jitter=0.10,
        solar_noise=0.20,
        onset="month_start",
        switch_kinds=("transfer", "transfer", "hidden_mirror", "long"),
    ),
    "retrained": WorldSettings(
        era_kinds=("level_m10", "level_m25", "level_p15", "solar_25"),
        era_size_range=(0.10, 0.25),
        transfer_size_range=(0.08, 0.25),
        duration_jitter=0.15,
        solar_noise=0.25,
        onset="month_start",
        switch_kinds=("transfer", "transfer", "hidden_mirror", "very_long"),
    ),
}
"""Each world differs from the others in size ranges, jitter, and kinds, on purpose (item 10)."""

ERA_SIZES: Final[dict[EraKindType, float]] = {
    "level_m10": -0.10,
    "level_m25": -0.25,
    "level_p15": 0.15,
    "solar_25": -0.25,
    "engine": -0.30,
}
"""Signed size of each fixed era kind, as a share of capacity (the peak for `solar_25`)."""


@dataclass(frozen=True)
class EraEvent:
    """One injected era.

    Attributes:
        host: Index of the Primary in the panel.
        kind: The era kind.
        size: Signed size as a share of capacity (the peak for the solar-shaped kind).
        onset_hh: First half-hour of the step on the grid.
        profile: Daily profile of a `curriculum_step`.
    """

    host: int
    kind: EraKindType
    size: float
    onset_hh: int
    profile: ProfileType = "flat"


@dataclass(frozen=True)
class SwitchEvent:
    """One injected switching event (a transfer).

    Attributes:
        donor: Index of the donor Primary.
        recipient: Index of the metered recipient, or None for the hidden-mirror kind.
        kind: The switching kind.
        size: The donor's loss as a share of its capacity (positive).
        onset_hh: First half-hour of the event.
        duration_days: Length of the event in days.
        share: The share `rho` of the donor's loss that the recipient receives.
    """

    donor: int
    recipient: int | None
    kind: SwitchKindType
    size: float
    onset_hh: int
    duration_days: float
    share: float


@dataclass(frozen=True)
class InjectedPass:
    """One injection pass: the injected panel and the truth about every event.

    Attributes:
        world: The world the pass belongs to.
        pass_index: Index of the pass within its world.
        seed: The seed that drew the pass.
        power: The injected power, shape (sites, half-hours).
        eras: The injected eras.
        switches: The injected switching events.
        truth: One row per event, with its host, class, kind, size, onset, end, and duration band.
    """

    world: WorldNameType
    pass_index: int
    seed: int
    power: np.ndarray
    eras: list[EraEvent]
    switches: list[SwitchEvent]
    truth: pl.DataFrame


def world_seed(*, world: WorldNameType, index: int) -> int:
    """Return the seed of draw `index` in a world.

    Args:
        world: The world.
        index: The draw's index, below 100,000.

    Returns:
        A seed inside the world's own range.

    Raises:
        ValueError: If the index leaves the world's range.
    """
    if not 0 <= index < SEED_RANGE:
        msg = f"seed index {index} is outside the world's range"
        raise ValueError(msg)
    return WORLD_SEED_BASE[world] + index


def duration_band(*, days: float) -> DurationBandType:
    """Return the duration band of an event (under 60, 60 to 120, or over 120 days).

    Args:
        days: Duration in days.

    Returns:
        The band label.
    """
    if days < 60:
        return "<60"
    return "60-120" if days <= 120 else ">120"


def _profile_shape(*, profile: ProfileType) -> np.ndarray:
    """Return the daily shape of a curriculum step, normalised to a mean of 1 over the day."""
    hh = np.arange(features.HALF_HOURS_PER_DAY)
    if profile == "flat":
        return np.ones(features.HALF_HOURS_PER_DAY)
    centre, width = (24, 6.0) if profile == "midday" else (38, 5.0)
    shape = 0.2 + np.exp(-0.5 * ((hh - centre) / width) ** 2)
    return shape / shape.mean()


def perturb_shortwave(
    *, shortwave: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a shortwave field with every day multiplied by a log-normal factor.

    Args:
        shortwave: Half-hourly shortwave radiation of one site.
        sigma: Standard deviation of the log of the daily factor.
        rng: The random generator.

    Returns:
        The perturbed field, the same shape.
    """
    days = shortwave.size // features.HALF_HOURS_PER_DAY
    factor = np.exp(rng.normal(0.0, sigma, size=days))
    return (shortwave.reshape(days, -1) * factor[:, None]).reshape(-1)


def _engine_schedule(*, n_half_hours: int, first: int, seed: int) -> np.ndarray:
    """Return an irregular on/off schedule of an embedded engine, 1 while it runs.

    On each day after `first` the engine runs with probability 0.5, for 4 to 12 hours that start at
    a random half-hour of the day, so no regular daily profile can absorb it. The runs of a day do
    not cross midnight.

    Args:
        n_half_hours: Length of the grid.
        first: First half-hour of the era.
        seed: Seed of the schedule.

    Returns:
        An array of zeros and ones over the grid.
    """
    rng = np.random.default_rng(seed)
    hh_day = features.HALF_HOURS_PER_DAY
    schedule = np.zeros(n_half_hours)
    for day in range(first // hh_day, n_half_hours // hh_day):
        if rng.random() >= ENGINE_RUN_PROBABILITY:
            continue
        length = int(rng.integers(8, 25))
        start = day * hh_day + int(rng.integers(0, hh_day - length + 1))
        schedule[start : start + length] = 1.0
    return schedule


def era_delta(
    *, event: EraEvent, n_half_hours: int, shortwave: np.ndarray | None, sigma: float, seed: int
) -> np.ndarray:
    """Return the signed change in power (share of capacity) that an era adds.

    Args:
        event: The era.
        n_half_hours: Length of the grid.
        shortwave: The host's ERA5 shortwave, needed for `solar_25`.
        sigma: Daily log-normal perturbation of the shortwave field.
        seed: Seed of the perturbation.

    Returns:
        The change at every half-hour, zero before the onset.
    """
    delta = np.zeros(n_half_hours)
    span = slice(event.onset_hh, n_half_hours)
    if event.kind == "solar_25":
        if shortwave is None:
            msg = "the solar-shaped era needs the shortwave field"
            raise ValueError(msg)
        field = perturb_shortwave(shortwave=shortwave, sigma=sigma, rng=np.random.default_rng(seed))
        delta[span] = event.size * np.clip(field[span] / SOLAR_PEAK_WM2, 0.0, 1.0)
    elif event.kind == "engine":
        delta[span] = (
            event.size
            * _engine_schedule(n_half_hours=n_half_hours, first=event.onset_hh, seed=seed)[span]
        )
    elif event.kind == "curriculum_step":
        shape = np.tile(_profile_shape(profile=event.profile), n_half_hours // 48 + 1)
        delta[span] = event.size * shape[span]
    else:
        delta[span] = event.size
    return delta


def apply_delta(*, base: np.ndarray, delta: np.ndarray, is_mva: bool) -> np.ndarray:
    """Add a signed change to a series, taking the absolute value at an MVA-only Primary.

    Args:
        base: The real series.
        delta: The signed change.
        is_mva: True for an MVA-only Primary.

    Returns:
        The injected series; readings that were missing, or that the change leaves alone, are kept.
    """
    moved = base + delta
    if is_mva:
        moved = np.abs(moved)
    return np.where((delta != 0) & np.isfinite(base), moved, base)


def generator_panel(*, panel: features.Panel) -> gen.PowerPanel:
    """Wrap the panel in the shared generator's `PowerPanel`.

    Args:
        panel: The real panel.

    Returns:
        A `PowerPanel` over capacity shares, with MVA flags and the electrical edge list.
    """
    sites = panel.sites
    return gen.PowerPanel(
        power={site: panel.power[i] for i, site in enumerate(sites)},
        is_mva={site: bool(panel.is_mva[i]) for i, site in enumerate(sites)},
        noise=dict.fromkeys(sites, GENERATOR_NOISE),
        neighbours={site: [sites[j] for j in panel.neighbours[i]] for i, site in enumerate(sites)},
        origin=features.GRID_START,
    )


def _transfer_rows(
    *, panel: features.Panel, event: SwitchEvent, first_id: int, rng: np.random.Generator
) -> list[dict]:
    """Return the generator's ground-truth rows of one transfer (donor, then recipient)."""
    donor, recipient = panel.sites[event.donor], event.recipient
    end = event.onset_hh + round(event.duration_days * features.HALF_HOURS_PER_DAY)
    ramp = (int(rng.integers(0, 5)), int(rng.integers(0, 5)))
    proportional = float(rng.uniform(0.3, 1.0))
    common = {
        "group_id": first_id,
        "label": gen.POSITIVE_EDGE,
        "start": event.onset_hh,
        "end": end,
        "residual_frac": 0.0,
        "shape": "proportional",
        "prop_share": proportional,
        "shape_source": donor,
        "ramp_in_hh": ramp[0],
        "ramp_out_hh": ramp[1],
        "drift": 0.0,
        "mid_factor": 1.0,
        "staged_hh": 0,
        "share": event.share,
        "period_hh": 0.0,
        "phase": 0.0,
        "chain_parent": -1,
    }
    donor_delta = -event.size
    rows = [
        {
            **common,
            "event_id": first_id,
            "site": donor,
            "role": "donor",
            "event_class": "edge_transfer"
            if recipient is not None
            else "edge_transfer_unseen_recipient",
            "delta": donor_delta,
            "units": abs(donor_delta) / GENERATOR_NOISE,
            "partners": [] if recipient is None else [panel.sites[recipient]],
        }
    ]
    if recipient is not None:
        moved = (
            -event.share
            * donor_delta
            * panel.capacity_mw[event.donor]
            / panel.capacity_mw[recipient]
        )
        moved = float(min(moved, 0.6))
        rows.append(
            {
                **common,
                "event_id": first_id + 1,
                "site": panel.sites[recipient],
                "role": "recipient",
                "event_class": "edge_transfer",
                "delta": moved,
                "units": abs(moved) / GENERATOR_NOISE,
                "partners": [donor],
            }
        )
    return rows


def inject_transfers(
    *, panel: features.Panel, events: list[SwitchEvent], seed: int
) -> dict[int, np.ndarray]:
    """Inject transfers with the shared generator's `apply_events`.

    Args:
        panel: The panel the transfers are added to.
        events: The transfers.
        seed: Seed of the ramp and shape draws.

    Returns:
        The injected series of every donor and metered recipient, by site index.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for k, event in enumerate(events):
        rows.extend(_transfer_rows(panel=panel, event=event, first_id=2 * k, rng=rng))
    table = pl.DataFrame(rows, schema=gen.EVENT_COLUMNS)
    injected = gen.apply_events(panel=generator_panel(panel=panel), events=table)
    touched = {e.donor for e in events} | {e.recipient for e in events if e.recipient is not None}
    return {i: np.asarray(injected[panel.sites[i]]) for i in touched}


def _metered(*, panel: features.Panel, site: int, onset_hh: int) -> list[int]:
    """Return the neighbours of a site that have readings in the 14 days before the onset."""
    lo = max(onset_hh - 14 * features.HALF_HOURS_PER_DAY, 0)
    return [j for j in panel.neighbours[site] if np.isfinite(panel.power[j, lo:onset_hh]).any()]


def choose_hosts(
    *,
    panel: features.Panel,
    ready_days: np.ndarray,
    onset_day: int,
    switch_kinds: tuple[SwitchKindType, ...],
    n_era: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[tuple[int, int | None]]]:
    """Choose the era hosts and the switching pairs of one pass (plan section 20, item 13).

    The donors and recipients of the switching pairs are all different Primaries, and D8, D15, and
    D18 are never recipients. An era host is never a donor, a recipient, or a neighbour of either
    in the same pass. A plain transfer needs a metered neighbour for its recipient, so a Primary
    with no metered neighbour can only be the donor of a hidden-mirror kind.

    Args:
        panel: The real panel.
        ready_days: First day with features of each site.
        onset_day: Day of the onset.
        switch_kinds: Kind of each switching pair.
        n_era: Number of era hosts.
        rng: The random generator.

    Returns:
        The era host indices and, per switching kind, the donor and its recipient (None if hidden).

    Raises:
        RuntimeError: If no assignment satisfies the rules in 5000 tries.
    """
    pool = host_pool(panel=panel, ready_days=ready_days, onset_day=onset_day)
    for _ in range(MAX_HOST_TRIES):
        drawn = try_hosts(
            panel=panel, pool=pool, onset_day=onset_day, switch_kinds=switch_kinds, n_era=n_era,
            rng=rng,
        )  # fmt: skip
        if drawn is not None:
            return drawn
    msg = "no host assignment satisfies the separation rules"
    raise RuntimeError(msg)


def host_pool(*, panel: features.Panel, ready_days: np.ndarray, onset_day: int) -> list[int]:
    """Return the Primaries that can host an event at an onset day.

    Args:
        panel: The real panel.
        ready_days: First day with features of each site.
        onset_day: Day of the onset.

    Returns:
        Indices of sites outside `EXCLUDED_HOSTS` with `MIN_HISTORY_DAYS` of features before the
        onset.
    """
    return [
        i
        for i, site in enumerate(panel.sites)
        if site not in EXCLUDED_HOSTS and ready_days[i] + MIN_HISTORY_DAYS <= onset_day
    ]


def try_hosts(
    *,
    panel: features.Panel,
    pool: list[int],
    onset_day: int,
    switch_kinds: tuple[SwitchKindType, ...],
    n_era: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[tuple[int, int | None]]] | None:
    """Draw one host assignment and return it if it satisfies the separation rules.

    Args:
        panel: The real panel.
        pool: The candidate hosts (`host_pool`).
        onset_day: Day of the onset.
        switch_kinds: Kind of each switching pair.
        n_era: Number of era hosts.
        rng: The random generator.

    Returns:
        The era hosts and the switching pairs, or None if the draw breaks a rule.
    """
    if len(pool) < len(switch_kinds):
        return None
    onset_hh = onset_day * features.HALF_HOURS_PER_DAY
    donors = [int(d) for d in rng.choice(pool, size=len(switch_kinds), replace=False)]
    pairs: list[tuple[int, int | None]] = []
    forbidden: set[int] = set()
    used = set(donors)
    for donor, kind in zip(donors, switch_kinds, strict=True):
        metered = [
            j
            for j in _metered(panel=panel, site=donor, onset_hh=onset_hh)
            if j not in used and panel.sites[j] not in EXCLUDED_HOSTS
        ]
        if kind == "hidden_mirror":
            recipient = None
        elif metered:
            recipient = int(rng.choice(metered))
            used.add(recipient)
        else:
            return None
        pairs.append((donor, recipient))
        forbidden |= {donor, *panel.neighbours[donor]}
        if recipient is not None:
            forbidden |= {recipient, *panel.neighbours[recipient]}
    free = [i for i in pool if i not in forbidden]
    if len(free) < n_era:
        return None
    return [int(h) for h in rng.choice(free, size=n_era, replace=False)], pairs


def _draw_duration(
    *, kind: SwitchKindType, verified_days: np.ndarray, jitter: float, rng: np.random.Generator
) -> float:
    """Return a duration in days: a verified one for plain kinds, a uniform draw for long kinds."""
    if kind == "long":
        return float(rng.uniform(60.0, 120.0))
    if kind == "very_long":
        return float(rng.uniform(121.0, 180.0))
    base = float(rng.choice(verified_days)) * float(np.exp(rng.uniform(-jitter, jitter)))
    return float(np.clip(base, 3.0, 59.9))


def build_pass(
    *,
    panel: features.Panel,
    weather_shortwave: np.ndarray,
    ready_days: np.ndarray,
    verified_days: np.ndarray,
    world: WorldNameType,
    pass_index: int,
    onset_day: int,
    n_era: int = N_ERA_PER_PASS,
) -> InjectedPass:
    """Inject `n_era` eras and two switching events into a copy of the panel.

    The two switching kinds come from `SWITCH_ROTATION` and the era kinds rotate with the pass
    index, so that consecutive passes cover every kind.

    Args:
        panel: The real panel.
        weather_shortwave: ERA5 shortwave of every site, shape (sites, half-hours).
        ready_days: First day with features of each site.
        verified_days: Durations of the verified switching events, without those in the scored
            window.
        world: The world, which fixes the seed range and the settings.
        pass_index: Index of the pass within the world.
        onset_day: The day of the onset (the first day of the scored month for `month_start`
            worlds; the earliest day of a random window otherwise, with a random half-hour).
        n_era: Number of era hosts.

    Returns:
        The injected pass.
    """
    settings = WORLD_SETTINGS[world]
    seed = world_seed(world=world, index=pass_index)
    rng = np.random.default_rng(seed)
    hh_day = features.HALF_HOURS_PER_DAY
    onset_hh = onset_day * hh_day + (
        0 if settings.onset == "month_start" else int(rng.integers(0, hh_day))
    )
    kinds = tuple(settings.switch_kinds[i] for i in SWITCH_ROTATION[pass_index % 4])
    if world == "bank" and pass_index % 4 == 3:
        kinds = ("very_long", *kinds[1:])
    era_kinds = tuple(settings.era_kinds[(n_era * pass_index + k) % 4] for k in range(n_era))
    era_hosts, pairs = choose_hosts(
        panel=panel,
        ready_days=ready_days,
        onset_day=onset_day,
        switch_kinds=kinds,
        n_era=n_era,
        rng=rng,
    )
    power = panel.power.copy()
    eras: list[EraEvent] = []
    for k, host in enumerate(era_hosts):
        kind = era_kinds[k]
        fixed = ERA_SIZES.get(kind)
        if fixed is not None and world in ("bank", "retrained", "engine"):
            size = fixed
        else:
            sign = 1.0 if kind == "level_p15" else -1.0
            if fixed is None:
                sign = 1.0 if rng.random() < 0.3 else -1.0
            size = sign * float(rng.uniform(*settings.era_size_range))
        profile: ProfileType = ("flat", "midday", "evening")[int(rng.integers(0, 3))]
        event = EraEvent(host=host, kind=kind, size=float(size), onset_hh=onset_hh, profile=profile)
        eras.append(event)
        delta = era_delta(
            event=event,
            n_half_hours=panel.power.shape[1],
            shortwave=weather_shortwave[host],
            sigma=settings.solar_noise,
            seed=seed + 1000 + k,
        )
        power[host] = apply_delta(
            base=panel.power[host], delta=delta, is_mva=bool(panel.is_mva[host])
        )
    switches = [
        SwitchEvent(
            donor=donor,
            recipient=recipient,
            kind=kind,
            size=float(rng.uniform(*settings.transfer_size_range)),
            onset_hh=onset_hh,
            duration_days=_draw_duration(
                kind=kind, verified_days=verified_days, jitter=settings.duration_jitter, rng=rng
            ),
            share=float(rng.uniform(0.9, 1.0)),
        )
        for (donor, recipient), kind in zip(pairs, kinds, strict=True)
    ]
    for site, series in inject_transfers(panel=panel, events=switches, seed=seed + 2000).items():
        power[site] = series
    return InjectedPass(
        world=world,
        pass_index=pass_index,
        seed=seed,
        power=power,
        eras=eras,
        switches=switches,
        truth=_truth_table(panel=panel, eras=eras, switches=switches),
    )


def _truth_table(
    *, panel: features.Panel, eras: list[EraEvent], switches: list[SwitchEvent]
) -> pl.DataFrame:
    rows = [
        {
            "host": e.host, "site": panel.sites[e.host], "truth": "era", "kind": e.kind,
            "size": e.size, "onset_day": e.onset_hh // 48, "end_day": -1, "duration_days": None,
            "band": None, "hidden_mirror": False,
        }
        for e in eras
    ]  # fmt: skip
    rows += [
        {
            "host": s.donor, "site": panel.sites[s.donor], "truth": "switch", "kind": s.kind,
            "size": -s.size, "onset_day": s.onset_hh // 48,
            "end_day": s.onset_hh // 48 + round(s.duration_days),
            "duration_days": s.duration_days, "band": duration_band(days=s.duration_days),
            "hidden_mirror": s.recipient is None,
        }
        for s in switches
    ]  # fmt: skip
    return pl.DataFrame(
        rows,
        schema={
            "host": pl.Int64, "site": pl.String, "truth": pl.String, "kind": pl.String,
            "size": pl.Float64, "onset_day": pl.Int64, "end_day": pl.Int64,
            "duration_days": pl.Float64, "band": pl.String, "hidden_mirror": pl.Boolean,
        },
    )  # fmt: skip


def month_start_days(*, first: date, last: date) -> list[int]:
    """Return the grid day index of the first day of every month from `first` to `last`.

    Args:
        first: A day in the first month.
        last: A day in the last month.

    Returns:
        Day indices of the month starts.
    """
    out = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        out.append(features.day_index(day=date(year, month, 1)))
        year, month = (year + (month == 12), month % 12 + 1)
    return out

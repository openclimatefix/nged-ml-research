"""Experiment 714: the second generator of permanent changes and temporary load moves.

A fresh draw that shares no function with experiment 709's injector (`e00709_inject`). It reads
only the base panel of experiment 709 (`e00709_features`) and experiment 660's generator
(`e00660_generator`, for the table of transfer rows and `apply_events`).

**Temporary moves** are transfers that follow the working assumption of the plan: a load move moves
a large number of connections, so at the donor it looks like a roughly constant decrease in MW and
at the receiver like a matching increase. Only 0 to 40% of the change follows the donor's daily
profile (`prop_share`). Durations are log-uniform from 1 to 112 days. The share `rho` that the
receiver gets is 0.9 to 1.0 (full mirror), 0.3 to 0.7 (partial mirror), or the receiver is not
metered (hidden mirror). A load move of any other behaviour is not modelled and would be misread as
a permanent change.

**Permanent changes** have four kinds in the pre-registered world `g2` and one further kind in the
secondary world `gas`:

- `flat`: a constant step, the case where the shape of the change cannot help;
- `profile_prop`: a step that follows the host's own daily profile in proportion;
- `daylight`: a step that is on whenever the sun is above the horizon, whatever the cloud;
- `quantised`: a step whose level is one of 2 or 3 values, redrawn at random every few days;
- `intermittent` (world `gas` only): constant-MW steps with random on and off times within and
  across days, with a running share that changes over the weeks. It uses experiment 660's pulse
  rows (`event_class="pulse"`), with `delta` in the range of the real gas-engine episodes
  (4% to 12% of capacity).

Step sizes of the four `g2` kinds come from a grid of -15%, -5%, +8%, and +30% of capacity. Every
pass has 2 permanent changes and 2 temporary moves at 4 different Primaries.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final, Literal

import e00660_generator as gen  # ty: ignore[unresolved-import]
import e00709_features as f709  # ty: ignore[unresolved-import]
import e00714_common as common
import numpy as np
import polars as pl

WorldType = Literal["g2", "gas"]
PermanentKindType = Literal["flat", "profile_prop", "daylight", "quantised", "intermittent"]
MirrorType = Literal["full", "partial", "hidden"]

PERMANENT_KINDS: Final[tuple[PermanentKindType, ...]] = (
    "flat",
    "profile_prop",
    "daylight",
    "quantised",
)
SIZE_GRID: Final[tuple[float, ...]] = (-0.15, -0.05, 0.08, 0.30)
"""Signed step sizes as a share of capacity; none of them is a step size of experiment 709."""
PASSES_PER_MONTH: Final[dict[WorldType, int]] = {"g2": 32, "gas": 8}
CANDIDATE_FACTOR: Final[int] = 4
"""The builder generates this many times as many passes as a world keeps, then keeps some."""
CELL_MODULUS: Final[dict[WorldType, int]] = {"g2": 8, "gas": 4}
"""A pass's index modulo this number fixes its mirror states, kinds, and step sizes, so a kept pass
replaces only a candidate of the same design cell."""
SELECTION_SEED: Final[int] = 714_500_000
SEED_BASE: Final[dict[WorldType, int]] = {"g2": 714_100_000, "gas": 714_200_000}
"""Seed ranges disjoint from experiment 709's worlds (709_100_000 to 709_500_000)."""
SWITCH_ROTATION: Final[tuple[tuple[MirrorType, MirrorType], ...]] = (
    ("full", "hidden"),
    ("partial", "full"),
    ("hidden", "partial"),
    ("partial", "hidden"),
)
"""Mirror state of the two transfers of pass `i % 4`."""
TRANSFER_SIZE: Final[tuple[float, float]] = (0.04, 0.20)
"""Range of the donor's loss as a share of capacity."""
DURATION_DAYS: Final[tuple[float, float]] = (1.0, 112.0)
MIRROR_SHARE: Final[dict[MirrorType, tuple[float, float]]] = {
    "full": (0.9, 1.0),
    "partial": (0.3, 0.7),
    "hidden": (0.9, 1.0),
}
PROP_SHARE: Final[tuple[float, float]] = (0.0, 0.4)
GAS_SIZE: Final[tuple[float, float]] = (0.04, 0.12)
"""Range of the intermittent step as a share of capacity (D8's episodes were 2 to 3 MW)."""
GAS_RUN_HH: Final[tuple[int, int]] = (4, 24)
LATITUDE_DEG: Final[float] = 52.5
LONGITUDE_DEG: Final[float] = -1.5
FLIP_PROBABILITY: Final[float] = 0.5
"""Chance that a transfer is scored at its receiving Primary."""
MAX_HOST_TRIES: Final[int] = 4000
MIN_PRE_COVERAGE: Final[float] = 0.9


@dataclass(frozen=True)
class GeneratedPass:
    """One generated pass.

    Attributes:
        power: Injected power over the stored window of the fold month.
        truth: One row per event.
        touched: Site indices of every host, donor, and recipient.
    """

    power: common.PassPower
    truth: pl.DataFrame
    touched: list[int]


def pass_seed(*, world: WorldType, month: str, index: int) -> int:
    """Return the seed of a pass: the world's base plus 1000 per fold month plus the index."""
    return SEED_BASE[world] + common.MONTHS.index(month) * 1000 + index


def sun_elevation_sine(*, n_days_from: int, n_days: int) -> np.ndarray:
    """Return the sine of the solar elevation at every half-hour, at a fixed UK location.

    Args:
        n_days_from: First grid day.
        n_days: Number of days.

    Returns:
        An array of length `n_days * 48`.
    """
    days = np.arange(n_days_from, n_days_from + n_days)
    doy = np.array([f709.day_of(index=int(d)).timetuple().tm_yday for d in days])
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.0)
    hours = (np.arange(common.DAY) + 0.5) / 2.0
    hour_angle = np.deg2rad((hours[None, :] + LONGITUDE_DEG / 15.0 - 12.0) * 15.0)
    lat = np.deg2rad(LATITUDE_DEG)
    sine = np.sin(lat) * np.sin(decl)[:, None] + np.cos(lat) * np.cos(decl)[:, None] * np.cos(
        hour_angle
    )
    return sine.reshape(-1)


def _daily_profile(*, real: np.ndarray, onset_hh: int) -> np.ndarray:
    """Return the host's median power at each half-hour of the day over the 28 days before onset."""
    seg = real[onset_hh - 28 * common.DAY : onset_hh].reshape(28, common.DAY)
    with np.errstate(all="ignore"):
        prof = np.nanmedian(seg, axis=0)
    return np.where(np.isfinite(prof), prof, np.nanmean(prof))


def permanent_delta(
    *, kind: PermanentKindType, size: float, onset_hh: int, n_hh: int, real: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:  # fmt: skip
    """Return the signed change (share of capacity) of one permanent change of a non-gas kind.

    Args:
        kind: The kind, other than `intermittent`.
        size: Signed size as a share of capacity (the mean change for `profile_prop`, the level
            when on for the others).
        onset_hh: First half-hour of the change.
        n_hh: Length of the grid.
        real: The host's real power (the profile of `profile_prop` reads only data before onset).
        rng: The random generator.

    Returns:
        The change at every half-hour, zero before the onset.

    Raises:
        ValueError: For the kind `intermittent`, which is built from table rows.
    """
    delta = np.zeros(n_hh)
    t = np.arange(onset_hh, n_hh)
    hh = t % common.DAY
    if kind == "flat":
        delta[onset_hh:] = size
    elif kind == "profile_prop":
        prof = _daily_profile(real=real, onset_hh=onset_hh)
        shape = np.clip(prof / max(float(prof.mean()), 1e-3), 0.2, 3.0)
        shape /= shape.mean()
        delta[onset_hh:] = size * shape[hh]
    elif kind == "daylight":
        sine = sun_elevation_sine(n_days_from=0, n_days=n_hh // common.DAY)
        delta[onset_hh:] = size * np.clip(sine[onset_hh:] / 0.08, 0.0, 1.0)
    elif kind == "quantised":
        n_levels = int(rng.integers(2, 4))
        levels = size * (np.arange(1, n_levels + 1) / n_levels)
        n_days = (n_hh - onset_hh) // common.DAY + 2
        day_level = np.empty(n_days)
        day = 0
        while day < n_days:
            run = int(rng.geometric(1 / 4.0))
            day_level[day : day + run] = rng.choice(levels)
            day += run
        delta[onset_hh:] = day_level[(t - onset_hh) // common.DAY]
    else:
        msg = f"kind {kind} is not built as a delta"
        raise ValueError(msg)
    return delta


def _row_template(*, group_id: int, start: int, end: int, share: float) -> dict:
    """Return the generator-table fields that every row of an event shares."""
    return {
        "group_id": group_id, "label": gen.POSITIVE_EDGE, "start": start, "end": end,
        "residual_frac": 0.0, "shape": "flat", "prop_share": 0.0, "shape_source": "",
        "ramp_in_hh": 0, "ramp_out_hh": 0, "drift": 0.0, "mid_factor": 1.0, "staged_hh": 0,
        "share": share, "period_hh": 0.0, "phase": 0.0, "chain_parent": -1,
    }  # fmt: skip


def intermittent_rows(
    *, site: str, size: float, onset_hh: int, stop_hh: int, first_id: int,
    rng: np.random.Generator,
) -> tuple[list[dict], float]:  # fmt: skip
    """Return experiment 660 pulse rows of an intermittent constant-MW step.

    Runs are 2 to 12 hours long. The off time before each run is drawn so that the running share of
    the current week matches a random walk between 5% and 85% that changes every week.

    Args:
        site: Site label of the host.
        size: Positive size of the step as a share of capacity (the host loses this much).
        onset_hh: First half-hour of the era.
        stop_hh: Half-hour after the last that is injected.
        first_id: Identifier of the first row.
        rng: The random generator.

    Returns:
        The rows and the mean running share.
    """
    weeks = (stop_hh - onset_hh) // (7 * common.DAY) + 2
    share = float(rng.uniform(0.1, 0.5))
    shares = []
    for _ in range(weeks):
        shares.append(share)
        share = float(np.clip(share + rng.uniform(-0.25, 0.25), 0.05, 0.85))
    rows: list[dict] = []
    t = onset_hh + int(rng.integers(0, 48))
    while t < stop_hh:
        p = shares[min((t - onset_hh) // (7 * common.DAY), weeks - 1)]
        on = int(rng.integers(GAS_RUN_HH[0], GAS_RUN_HH[1] + 1))
        end = min(t + on, stop_hh)
        rows.append(
            {
                **_row_template(group_id=first_id, start=t, end=end, share=1.0),
                "event_id": first_id + len(rows), "site": site, "role": "donor",
                "event_class": "pulse", "delta": -size, "units": size / 0.01, "partners": [],
            }
        )  # fmt: skip
        off = max(1, round(on * (1.0 - p) / p * float(np.exp(rng.uniform(-0.5, 0.5)))))
        t = end + off
    return rows, float(np.mean(shares))


def _metered(*, ctx: common.Context, site: int, onset_hh: int) -> list[int]:
    """Return neighbours of a site with real readings for at least 80% of the 14 days before."""
    lo = onset_hh - common.WINDOW_HH
    return [
        j
        for j in ctx.panel.neighbours[site]
        if np.isfinite(ctx.panel.power[j, lo:onset_hh]).mean() >= 0.8
    ]


def _pool(*, ctx: common.Context, onset_hh: int, ref_ok: np.ndarray) -> list[int]:
    """Return the Primaries that can host an event: covered before onset and with a reference."""
    lo = onset_hh - 60 * common.DAY
    return [
        i
        for i, site in enumerate(ctx.panel.sites)
        if site not in common.EXCLUDED_HOSTS
        and ref_ok[i]
        and np.isfinite(ctx.panel.power[i, lo:onset_hh]).mean() >= MIN_PRE_COVERAGE
    ]


def choose_hosts(
    *, ctx: common.Context, pool: list[int], onset_hh: int, mirrors: tuple[MirrorType, ...],
    n_era: int, rng: np.random.Generator,
) -> tuple[list[int], list[tuple[int, int | None]]]:  # fmt: skip
    """Draw donors, recipients, and permanent-change hosts that satisfy the separation rules.

    Events are placed one after another, in a random order of the transfers and the permanent
    changes. Every site of an event (a donor, a recipient, or a
    permanent-change host) must lie outside the sites and the neighbours of every earlier event, so
    no event sits in another event's neighbour set. A recipient is a metered neighbour of its
    donor and is the only site of its pair that lies in the neighbour set of the pair's donor.

    Args:
        ctx: The context.
        pool: The candidate hosts.
        onset_hh: First half-hour of the pass's events.
        mirrors: Mirror state of each transfer.
        n_era: Number of permanent-change hosts.
        rng: The random generator.

    Returns:
        The permanent-change hosts and the (donor, recipient or None) pairs.

    Raises:
        RuntimeError: If no assignment satisfies the rules.
    """
    panel = ctx.panel
    neighbours = panel.neighbours
    for _ in range(MAX_HOST_TRIES):
        order = rng.permutation([*range(len(mirrors)), *([-1] * n_era)])
        forbidden: set[int] = set()
        pairs: dict[int, tuple[int, int | None]] = {}
        era_hosts: list[int] = []
        failed = False
        for slot in order:
            options = [i for i in pool if i not in forbidden]
            if not options:
                failed = True
                break
            donor = int(rng.choice(options))
            if slot < 0:
                era_hosts.append(donor)
                forbidden |= {donor, *neighbours[donor]}
                continue
            recipient = None
            if mirrors[slot] != "hidden":
                metered = [
                    j
                    for j in _metered(ctx=ctx, site=donor, onset_hh=onset_hh)
                    if j not in forbidden and panel.sites[j] not in common.EXCLUDED_HOSTS
                ]
                if not metered:
                    failed = True
                    break
                recipient = int(rng.choice(metered))
            pairs[int(slot)] = (donor, recipient)
            forbidden |= {donor, *neighbours[donor]}
            if recipient is not None:
                forbidden |= {recipient, *neighbours[recipient]}
        if not failed:
            return era_hosts, [pairs[k] for k in range(len(mirrors))]
    msg = "no host assignment satisfies the separation rules"
    raise RuntimeError(msg)


def _transfer_rows(
    *, ctx: common.Context, donor: int, recipient: int | None, size: float, onset_hh: int,
    duration_hh: int, share: float, first_id: int, rng: np.random.Generator,
    receiver_only: bool = False,
) -> list[dict]:  # fmt: skip
    """Return experiment 660's rows of one transfer (the donor's row, then the recipient's).

    With `receiver_only` the donor is not metered: the single row is the gain of the scored site
    (`donor` is then the receiving Primary), of `size` times `share` of its capacity.
    """
    panel = ctx.panel
    sites = panel.sites
    common_fields = {
        **_row_template(group_id=first_id, start=onset_hh, end=onset_hh + duration_hh, share=share),
        "shape": "proportional",
        "prop_share": float(rng.uniform(*PROP_SHARE)),
        "shape_source": sites[donor],
        "ramp_in_hh": int(rng.integers(0, 9)),
        "ramp_out_hh": int(rng.integers(0, 9)),
    }
    if receiver_only:
        gain = size * share
        return [
            {
                **common_fields, "event_id": first_id, "site": sites[donor], "role": "recipient",
                "event_class": "edge_transfer_unseen_donor", "delta": gain, "units": gain / 0.01,
                "partners": [],
            }
        ]  # fmt: skip
    rows = [
        {
            **common_fields, "event_id": first_id, "site": sites[donor], "role": "donor",
            "event_class": "edge_transfer" if recipient is not None
            else "edge_transfer_unseen_recipient",
            "delta": -size, "units": size / 0.01,
            "partners": [] if recipient is None else [sites[recipient]],
        }
    ]  # fmt: skip
    if recipient is not None:
        moved = share * size * panel.capacity_mw[donor] / panel.capacity_mw[recipient]
        moved = float(min(moved, 0.6))
        rows.append(
            {
                **common_fields, "event_id": first_id + 1, "site": sites[recipient],
                "role": "recipient", "event_class": "edge_transfer", "delta": moved,
                "units": moved / 0.01, "partners": [sites[donor]],
            }
        )  # fmt: skip
    return rows


def _power_panel(*, ctx: common.Context) -> gen.PowerPanel:
    """Wrap the real panel in experiment 660's `PowerPanel`."""
    sites = ctx.panel.sites
    return gen.PowerPanel(
        power={s: ctx.panel.power[i] for i, s in enumerate(sites)},
        is_mva={s: bool(ctx.panel.is_mva[i]) for i, s in enumerate(sites)},
        noise=dict.fromkeys(sites, 0.01),
        neighbours={s: [sites[j] for j in ctx.panel.neighbours[i]] for i, s in enumerate(sites)},
        origin=f709.GRID_START,
    )


def _build_transfers(
    *, ctx: common.Context, pairs: list[tuple[int, int | None]], mirrors: tuple[MirrorType, ...],
    onset_hh: int, onset_day: int, truth: list[dict], rng: np.random.Generator,
) -> list[dict]:  # fmt: skip
    """Return the generator rows of every transfer, and append each transfer's truth row.

    Half of the transfers are scored at the receiving Primary, so that the sign of the change does
    not separate the temporary moves from the permanent changes.

    Args:
        ctx: The context.
        pairs: The (donor, recipient or None) pair of each transfer.
        mirrors: Mirror state of each transfer.
        onset_hh: First half-hour of the transfers.
        onset_day: Grid day of the onset.
        truth: The truth rows, appended to.
        rng: The random generator.

    Returns:
        The rows for `apply_events`.
    """
    panel = ctx.panel
    transfer_rows: list[dict] = []
    for k, ((donor, recipient), mirror) in enumerate(zip(pairs, mirrors, strict=True)):
        size = float(rng.uniform(*TRANSFER_SIZE))
        days = float(np.exp(rng.uniform(np.log(DURATION_DAYS[0]), np.log(DURATION_DAYS[1]))))
        share = float(rng.uniform(*MIRROR_SHARE[mirror]))
        flipped = bool(rng.random() < FLIP_PROBABILITY)
        new_rows = _transfer_rows(
            ctx=ctx, donor=donor, recipient=recipient, size=size, onset_hh=onset_hh,
            duration_hh=round(days * common.DAY), share=share, first_id=2 * k, rng=rng,
            receiver_only=flipped and recipient is None,
        )  # fmt: skip
        transfer_rows.extend(new_rows)
        if not flipped:
            scored, mirror_site, signed = donor, recipient, -size
        elif recipient is None:
            scored, mirror_site, signed = donor, None, size * share
        else:
            scored, mirror_site, signed = recipient, donor, float(new_rows[1]["delta"])
        band = "<60" if days < 60 else ("60-120" if days <= 120 else ">120")
        truth.append(
            {
                "host": scored, "site": panel.sites[scored], "truth": "switch", "kind": "transfer",
                "size": signed, "onset_hh": onset_hh, "onset_day": onset_day,
                "end_day": onset_day + round(days), "duration_days": days, "band": band,
                "mirror_state": mirror, "share": share,
                "recipient": -1 if mirror_site is None else mirror_site,
                "running_share": float("nan"), "receiving_host": flipped,
            }
        )  # fmt: skip
    return transfer_rows


def build_pass(*, ctx: common.Context, world: WorldType, month: str, index: int) -> GeneratedPass:
    """Generate one pass: 2 permanent changes and 2 temporary moves at 4 different Primaries.

    Args:
        ctx: The context (real panel and weather).
        world: `g2` (the four pre-registered permanent kinds) or `gas` (intermittent steps).
        month: Fold month of the pass.
        index: Index of the pass within the world's month.

    Returns:
        The generated pass.
    """
    seed = pass_seed(world=world, month=month, index=index)
    rng = np.random.default_rng(seed)
    panel = ctx.panel
    n_hh = panel.power.shape[1]
    month_day = common.month_start_day(month=month)
    onset_day = month_day + int(rng.integers(0, 2))
    onset_hh = onset_day * common.DAY + int(rng.integers(0, common.DAY))
    reference = common.reference_for(ctx=ctx, month_day=month_day)
    ref_ok = np.isfinite(reference.power).any(axis=1)
    pool = _pool(ctx=ctx, onset_hh=onset_hh, ref_ok=ref_ok)
    mirrors = SWITCH_ROTATION[index % 4]
    era_hosts, pairs = choose_hosts(
        ctx=ctx, pool=pool, onset_hh=onset_hh, mirrors=mirrors, n_era=2, rng=rng
    )
    power = panel.power.copy()
    truth: list[dict] = []
    rows: list[dict] = []
    stop_hh = min(onset_day + common.STORE_AFTER_DAYS, n_hh // common.DAY) * common.DAY
    for j, host in enumerate(era_hosts):
        e = 2 * index + j
        site = panel.sites[host]
        if world == "gas":
            size = float(rng.uniform(*GAS_SIZE))
            new_rows, running = intermittent_rows(
                site=site, size=size, onset_hh=onset_hh, stop_hh=stop_hh,
                first_id=10_000 + 1000 * j, rng=rng,
            )  # fmt: skip
            rows.extend(new_rows)
            kind: PermanentKindType = "intermittent"
            signed = -size
            extra = {"running_share": running}
        else:
            kind = PERMANENT_KINDS[e % 4]
            signed = SIZE_GRID[(e // 4) % 4]
            delta = permanent_delta(
                kind=kind, size=signed, onset_hh=onset_hh, n_hh=n_hh, real=panel.power[host],
                rng=rng,
            )  # fmt: skip
            moved = power[host] + delta
            if panel.is_mva[host]:
                moved = np.abs(moved)
            power[host] = np.where((delta != 0) & np.isfinite(power[host]), moved, power[host])
            extra = {"running_share": float("nan")}
        truth.append(
            {
                "host": host, "site": site, "truth": "era", "kind": kind, "size": signed,
                "onset_hh": onset_hh, "onset_day": onset_day, "end_day": -1,
                "duration_days": None, "band": None, "mirror_state": "none", "share": None,
                "recipient": -1, "receiving_host": False, **extra,
            }
        )  # fmt: skip
    transfer_rows = _build_transfers(
        ctx=ctx, pairs=pairs, mirrors=mirrors, onset_hh=onset_hh, onset_day=onset_day,
        truth=truth, rng=rng,
    )  # fmt: skip
    table = pl.DataFrame(transfer_rows + rows, schema=gen.EVENT_COLUMNS)
    injected = gen.apply_events(panel=_power_panel(ctx=ctx), events=table)
    touched = {t["host"] for t in truth} | {t["recipient"] for t in truth if t["recipient"] >= 0}
    for name in set(table["site"].to_list()):
        power[panel.sites.index(name)] = injected[name]
    lo, hi = common.window_bounds(onset_day=month_day)
    truth_frame = pl.DataFrame(
        truth,
        schema={
            "host": pl.Int64, "site": pl.String, "truth": pl.String, "kind": pl.String,
            "size": pl.Float64, "onset_hh": pl.Int64, "onset_day": pl.Int64,
            "end_day": pl.Int64, "duration_days": pl.Float64, "band": pl.String,
            "mirror_state": pl.String, "share": pl.Float64, "recipient": pl.Int64,
            "running_share": pl.Float64, "receiving_host": pl.Boolean,
        },
    ).with_columns(
        world=pl.lit(world), month=pl.lit(month), pass_index=pl.lit(index, dtype=pl.Int64)
    )  # fmt: skip
    return GeneratedPass(
        power=common.PassPower(start_hh=lo, power=power[:, lo:hi].astype(np.float32)),
        truth=truth_frame,
        touched=sorted(touched),
    )


def balance_vector(*, truth: pl.DataFrame, n_sites: int) -> np.ndarray:
    """Return, per Primary, the permanent changes minus the temporary moves scored there."""
    out = np.zeros(n_sites)
    for host, kind in zip(truth["host"].to_list(), truth["truth"].to_list(), strict=True):
        out[host] += 1.0 if kind == "era" else -1.0
    return out


def select_balanced(
    *, vectors: dict[tuple[str, int], np.ndarray], months: tuple[str, ...], per_month: int,
    modulus: int, seed: int = SELECTION_SEED,
) -> dict[str, list[int]]:  # fmt: skip
    """Keep, in each fold month, the candidate passes whose Primaries balance the two classes.

    A pass keeps its design cell (its index modulo `modulus`), so the mirror states, kinds, and
    sizes stay balanced. Within each cell the search swaps a kept pass for an unkept one while the
    sum over Primaries of the squared difference between permanent and temporary events falls,
    over all months together and within each month.

    Args:
        vectors: The `balance_vector` of each candidate, by (month, index).
        months: The fold months.
        per_month: Passes to keep per fold month.
        modulus: The number of design cells.
        seed: Seed of the starting draw.

    Returns:
        The kept indices of each month.
    """
    rng = np.random.default_rng(seed)
    per_cell = per_month // modulus
    cells: dict[tuple[str, int], list[int]] = {}
    for month, index in sorted(vectors):
        cells.setdefault((month, index % modulus), []).append(index)
    kept = {
        key: [int(i) for i in rng.permutation(sorted(v))[:per_cell]] for key, v in cells.items()
    }
    n_sites = len(next(iter(vectors.values())))
    month_sum = {m: np.zeros(n_sites) for m in months}
    for (month, _cell), indices in kept.items():
        for i in indices:
            month_sum[month] += vectors[(month, i)]

    def cost(sums: dict[str, np.ndarray]) -> float:
        total = np.sum(list(sums.values()), axis=0)
        return float(total @ total + sum(float(s @ s) for s in sums.values()))

    best = cost(month_sum)
    improved = True
    while improved:
        improved = False
        for (month, cell), indices in kept.items():
            for slot, old in enumerate(indices):
                for new in cells[(month, cell)]:
                    if new in indices:
                        continue
                    trial = dict(month_sum)
                    trial[month] = month_sum[month] - vectors[(month, old)] + vectors[(month, new)]
                    value = cost(trial)
                    if value < best - 1e-9:
                        best, month_sum, indices[slot], improved = value, trial, new, True
                        break
    result: dict[str, list[int]] = {m: [] for m in months}
    for (month, _cell), indices in kept.items():
        result[month].extend(indices)
    return {m: sorted(v) for m, v in result.items()}


_WORLD_FOLDER: str = "worlds"


def use_folder(*, name: str) -> None:
    """Store and read the generated passes under `name` in the experiment's data folder."""
    global _WORLD_FOLDER  # noqa: PLW0603
    _WORLD_FOLDER = name


def stored_indices(*, world: WorldType, month: str) -> list[int]:
    """Return the indices of the stored passes of a world's fold month, in increasing order."""
    prefix = f"{month}_p"
    found = (
        int(path.stem.removeprefix(prefix))
        for path in world_dir(world=world).glob(f"{prefix}*.npz")
        if not path.name.endswith(".tmp.npz")
    )
    return sorted(found)


def world_dir(*, world: WorldType) -> Path:
    """Return the folder that holds a generated world."""
    return common.DATA_DIR / _WORLD_FOLDER / world


def write_pass(*, world: WorldType, month: str, index: int, generated: GeneratedPass) -> None:
    """Write a generated pass atomically (an npz of power and a parquet of the truth table)."""
    folder = world_dir(world=world)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{month}_p{index}"
    tmp = stem.with_suffix(".tmp.npz")
    np.savez(
        tmp, start_hh=generated.power.start_hh, power=generated.power.power,
        touched=np.array(generated.touched),
    )  # fmt: skip
    tmp.rename(stem.with_suffix(".npz"))
    generated.truth.write_parquet(stem.with_suffix(".tmp.parquet"))
    stem.with_suffix(".tmp.parquet").rename(stem.with_suffix(".parquet"))


def read_pass(*, world: WorldType, month: str, index: int) -> GeneratedPass:
    """Read a generated pass written by `write_pass`."""
    stem = world_dir(world=world) / f"{month}_p{index}"
    with np.load(stem.with_suffix(".npz")) as data:
        power = common.PassPower(start_hh=int(data["start_hh"]), power=data["power"])
        touched = [int(t) for t in data["touched"]]
    return GeneratedPass(
        power=power, truth=pl.read_parquet(stem.with_suffix(".parquet")), touched=touched
    )


def onset_date_text(*, onset_day: int) -> str:
    """Return the calendar date of an onset day as text (for the notes)."""
    return str(f709.FIRST_DAY + timedelta(days=int(onset_day)))

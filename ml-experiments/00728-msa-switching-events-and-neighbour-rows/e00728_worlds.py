"""Experiment 728: the training world, the visible depth, the seed lists, and the counted event set.

**The training world** extends experiment 715's `mirror_world` (design section 2.1). Each event is a
constant step in shares of the host's capacity with an abrupt onset. The world differs from
experiment 715's `mirror_world` in seven ways: the lengths of temporary moves are log-uniform from 1
hour to 120 days, a permanent change never ends, a Primary takes at most two permanent changes with
a summed shift within +-0.50 of capacity, a move has no, one, or two to three metered recipients
(a third of the moves each), the sign is drawn independently of the kind and of the mirror class,
the events cover 15% of each Primary's half-hours, and the depth is drawn so that three quarters of
the events are of visible depth.

A temporary move sits in the history of the host and of every recipient, and never in a target. A
permanent change sits in the history and in the target of the host. A chance mirror of a permanent
change (1 event in 10) sits in the history of one neighbour only.

**The visible depth** (design section 2.4) of a step at a Primary is the larger of 0.10 of capacity
and 2 times the Primary's daily-mean reference error. `reference_error` rebuilds experiment 714's
weather-only reference with `e00714_common.build_reference`.

**The counted event set** (design section 2.2) is `g2`'s temporary moves of visible depth with a
metered mirror. `counted_moves` reads the truth tables of `g2` and applies the filter.

**Three flagged points for the reviewer.**

- *The reference error is in-sample.* `reference_error` scores the reference over the fold's
  training span `[month_day - 758, month_day - 28)`, the days that experiment 714's `fit_reference`
  fits it on, as the design says. The in-sample error is smaller than an out-of-sample error, so the
  visible depth is lower than a held-out reference error would give, and more `g2` moves pass the
  filter. The filter is the same for every arm.
- *The donor step ignores experiment 714's cap of 0.6.* `donor_step` recovers the donor's step of a
  move that `g2` scored at the receiving Primary from the recipient's step. The generator caps the
  recipient's step at 0.6 of the recipient's capacity, so for a capped row the recovered donor step
  is too small. No rule changes if a count differs from 81: the gate prints every count.
- *The MVA-only floor.* Every event at an MVA-only host, temporary or permanent, must keep the
  host's history at or above zero and the host's target at or above zero (as if the step were a
  fall). An event that breaks a test is redrawn at its own Primary with a new onset, its sign and
  depth kept, by one rule for both kinds, and is dropped when no onset passes. Only the cap of two
  permanent changes per Primary applies to permanent changes alone. A
  recipient's mirror is clipped at zero, as experiment 715 clips it. A Primary with no reference
  error (NaN) has a visible depth of the floor of 0.10 alone.
"""

import hashlib
from dataclasses import dataclass
from typing import Final

import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00714_common as common714  # ty: ignore[unresolved-import]
import e00714_generator2 as g2gen  # ty: ignore[unresolved-import]
import e00715_neighbours as nb  # ty: ignore[unresolved-import]
import e00715_worlds as w715  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.stats import chi2_contingency

PER_DAY: Final = lay.STEPS_PER_DAY
WORLD_KEY: Final = 728_001
"""First entry of the key of the training world's random stream."""
REDRAW_KEY: Final = 728_002
"""First entry of the key of the stream that redraws the onset of an event that fails a test."""
TEST_SEED_BASE: Final = 728_900_000
"""Held-out draws use `TEST_SEED_BASE` plus the fold's index; `g2`'s seeds are experiment 714's."""
TRAIN_SEED_BASE: Final = 728_100_000
"""Training draws use this base plus 1000 per fold month plus the training seed."""
FOLD_MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
"""Experiment 713's 8 folds, in order."""
INJECTED_MONTHS: Final = common714.MONTHS
"""The four folds that `g2` covers."""

TEMPORARY_SHARE: Final = 0.70
MIRROR_CLASS_SHARE: Final = 1.0 / 3.0
"""A third of moves have one metered recipient, a third two or three, a third none."""
MIN_LENGTH_HH: Final = 2
"""One hour."""
MAX_LENGTH_DAYS: Final = 120
PERMANENT_COVER_DAYS: Final = 28
"""A permanent change counts for this many days of the event cover."""
EVENT_COVER: Final = 0.15
MAX_PERMANENT_PER_PRIMARY: Final = 2
MAX_PERMANENT_SUM: Final = 0.50
"""The summed permanent shift at a Primary stays within this share of capacity."""
MAX_REDRAWS: Final = 400
SINGLE_SHARE: Final = (0.2, 1.0)
"""Share of the megawatts moved that one metered recipient takes (experiment 715's range)."""
MULTI_TOTAL_SHARE: Final = (0.6, 1.0)
"""Summed share of the megawatts moved that two or three metered recipients take."""
CHANCE_MIRROR: Final = w715.CHANCE_MIRROR
VISIBLE_SHARE: Final = 0.75
"""Share of events whose depth is at least the visible depth."""
DEPTH_MAX: Final = 0.50
DEPTH_MIN: Final = 0.05
VISIBLE_FLOOR: Final = 0.10
"""A step of visible depth is at least this share of capacity."""
VISIBLE_FACTOR: Final = 2.0
"""A step of visible depth is at least this many times the daily-mean reference error."""
INDEPENDENCE_LEVEL: Final = 0.01
"""A chi-squared test of independence that rejects at this level fails GH5."""
FLOOR_TOLERANCE: Final = 1e-9
RECIPIENT_CAP: Final = 0.6
"""Experiment 714 caps a recipient's step at this share of the recipient's capacity."""


@dataclass(frozen=True)
class DrawnEvent:
    """The random draws of one event, before any limit is applied.

    Attributes:
        temporary: True for a temporary move.
        length: Length in half-hours (a permanent change is lengthened to the end of the grid).
        onset_fraction: Uniform draw that places the onset.
        depth: Unsigned depth in shares of the host's capacity.
        sign: -1.0 for a fall, +1.0 for a rise.
        mirror_class: 0 for no metered recipient, 1 for one, 2 for two or three (a temporary move
            only).
    """

    temporary: bool
    length: int
    onset_fraction: float
    depth: float
    sign: float
    mirror_class: int


@dataclass
class TrainingWorld:
    """The two arrays of a training draw and its event table.

    Attributes:
        history: (16, G) power over capacity that the history rows read, all events in.
        target: (16, G) power over capacity that the targets read: the cleaned power plus the
            permanent changes only.
        events: One dict per event.
        seed: The world seed of the draw.
        redrawn: Number of events that failed the placement tests and were redrawn.
        dropped: Number of events that found no place that passed the placement tests.
        redraws: The same two counts by kind (`redrawn_temporary`, `redrawn_permanent`,
            `dropped_temporary`, `dropped_permanent`).
        first_try: For each `kind|group|sign` key, the number of events drawn and the number that
            passed the placement tests at their first placement.
    """

    history: np.ndarray
    target: np.ndarray
    events: list[dict]
    seed: int
    redrawn: int
    dropped: int
    redraws: dict[str, int]
    first_try: dict[str, list[int]]


# ---------------------------------------------------------------------- seeds


def train_seed(*, month: str, seed: int) -> int:
    """Return the world seed of the training draw of a fold month and a training seed."""
    return TRAIN_SEED_BASE + FOLD_MONTHS.index(month) * 1000 + seed


def held_out_seed(*, month: str) -> int:
    """Return the world seed of the held-out draw of an injected fold month."""
    assert month in INJECTED_MONTHS, month
    return TEST_SEED_BASE + INJECTED_MONTHS.index(month)


def g2_seeds(*, passes_per_month: dict[str, list[int]]) -> list[int]:
    """Return the world seeds of the stored `g2` passes, `passes_per_month` by fold month."""
    return [
        g2gen.pass_seed(world="g2", month=month, index=index)
        for month, indices in passes_per_month.items()
        for index in indices
    ]


def assert_disjoint(*, train: list[int], g2: list[int], held_out: list[int]) -> None:
    """GH6: assert that the training, `g2`, and held-out world seeds are pairwise disjoint."""
    sets = {"train": set(train), "g2": set(g2), "held_out": set(held_out)}
    for a, set_a in sets.items():
        for b, set_b in sets.items():
            if a < b:
                both = set_a & set_b
                assert not both, f"the {a} and {b} world seeds overlap: {sorted(both)[:5]}"


def assert_train_seed_allowed(*, world_seed: int, test_seeds: list[int]) -> None:
    """Refuse a fit whose training-world seed is in a test list (design section 2.1)."""
    assert world_seed not in set(test_seeds), (
        f"training-world seed {world_seed} is a test seed: refusing to fit"
    )


# ---------------------------------------------------------------------- visible depth


def reference_error(*, ctx: common714.Context, month: str) -> np.ndarray:
    """Return the daily-mean reference error of each of the 16 Primaries for a fold month.

    The error is the mean absolute difference between the daily mean of the Primary's real power
    and the daily mean of experiment 714's weather-only reference, in shares of capacity, **over the
    fold's training span** `[month_day - 758, month_day - 28)` that the reference is fitted on
    (experiment 714's `fit_reference`, `TRAIN_DAYS` and `TRAIN_LAG_DAYS`). A day's two means use
    the half-hours at which both the real power and the reference are finite. A Primary with no
    reference or no such day is NaN. The array follows `nb.LABELS`.
    """
    month_day = common714.month_start_day(month=month)
    day = common714.DAY
    hi = month_day - common714.TRAIN_LAG_DAYS
    lo = max(hi - common714.TRAIN_DAYS, 0)
    out = np.full(nb.N_PRIMARIES, np.nan)
    for label_index, label in enumerate(nb.LABELS):
        site = ctx.panel.sites.index(label)
        beta = common714.fit_reference(
            panel=ctx.panel, weather=ctx.weather, site=site, month_day=month_day
        )
        if beta is None:
            continue
        x = common714.design(weather=ctx.weather, site=site, day_lo=lo, day_hi=hi)
        pred = (x @ beta).reshape(hi - lo, day)
        real = ctx.panel.power[site, lo * day : hi * day].reshape(hi - lo, day)
        ok = np.isfinite(real) & np.isfinite(pred)
        count = ok.sum(axis=1)
        have = count > 0
        real_mean = np.where(ok, real, 0.0).sum(axis=1)[have] / count[have]
        pred_mean = np.where(ok, pred, 0.0).sum(axis=1)[have] / count[have]
        if have.any():
            out[label_index] = float(np.abs(real_mean - pred_mean).mean())
    return out


def visible_depth(*, reference_error_share: np.ndarray) -> np.ndarray:
    """Return the visible depth of each Primary: the larger of 0.10 and 2 times the reference error.

    A Primary with no reference error (NaN) takes the floor of 0.10 alone.
    """
    scaled = np.where(
        np.isfinite(reference_error_share), VISIBLE_FACTOR * reference_error_share, 0.0
    )
    return np.maximum(VISIBLE_FLOOR, scaled)


# ---------------------------------------------------------------------- the training world


def draw_event(*, rng: np.random.Generator, vis: float, n_grid: int) -> DrawnEvent:
    """Draw one event's kind, length, onset, depth, sign, and mirror class.

    Every draw is made for every event, so the stream does not depend on the kind. The kind, the
    sign, and the depth are drawn independently of each other. The depth is log-uniform between the
    Primary's visible depth and 0.50 for `VISIBLE_SHARE` of the events, and log-uniform between 0.05
    and the visible depth for the rest.
    """
    u_kind, u_len, u_onset, u_band, u_depth, u_sign, u_class = rng.random(7)
    temporary = bool(u_kind < TEMPORARY_SHARE)
    lo_len, hi_len = np.log(MIN_LENGTH_HH), np.log(MAX_LENGTH_DAYS * PER_DAY)
    length = round(float(np.exp(lo_len + u_len * (hi_len - lo_len))))
    length = min(max(length, MIN_LENGTH_HH), n_grid - 1)
    lo, hi = (vis, DEPTH_MAX) if u_band < VISIBLE_SHARE else (DEPTH_MIN, vis)
    depth = float(np.exp(np.log(lo) + u_depth * (np.log(hi) - np.log(lo))))
    sign = 1.0 if u_sign < 0.5 else -1.0
    mirror_class = min(int(u_class / MIRROR_CLASS_SHARE), 2) if temporary else 0
    return DrawnEvent(
        temporary=temporary, length=length, onset_fraction=float(u_onset), depth=depth, sign=sign,
        mirror_class=mirror_class,
    )  # fmt: skip


def place_onset(*, fraction: float, first: int, last: int, length: int) -> int:
    """Return the onset: uniform over the grid indices where the event fits (as experiment 581)."""
    hi = max(first + 1, last - length)
    return first + min(int(fraction * (hi - first)), hi - first - 1)


def _floor_ok(*, history_row: np.ndarray, onset: int, shift: float, mva: bool) -> bool:
    """Return True if the step may be added to this Primary at this onset (design section 2.1).

    **The same floor test applies to a temporary move and to a permanent change.** At an MVA-only
    Primary the history from the onset to the end of the training grid, with the step added, stays
    at or above zero, whatever the kind and the length of the event. The test therefore depends on
    the host, the onset, the sign, and the depth alone.
    """
    if not mva:
        return True
    seg = history_row[onset:] + shift
    finite = np.isfinite(seg)
    return not (finite.any() and seg[finite].min() < -FLOOR_TOLERANCE)


def _limits_ok(
    *, clean_row: np.ndarray, perm_sum: np.ndarray, count: int, onset: int, depth: float, mva: bool
) -> bool:  # fmt: skip
    """Return True if a step of this depth may join the permanent changes already at the Primary.

    **Both kinds of event face the same limits, and the limits ignore the sign.** The floor test
    depends on the sign, and it passes shallow falls more often than deep ones. A limit that depends
    on the depth and applied to one kind only would therefore pass a different mix of signs for that
    kind, and group and sign would predict the kind. So a rise and a fall of equal depth pass or
    fail these limits together, for a temporary move and for a permanent change alike. The limits
    are a summed permanent shift within `MAX_PERMANENT_SUM` even if the new step has the sign that
    adds to the sum, and, at an MVA-only Primary, a target at or above zero even if the new step is
    a fall. The count of permanent changes at the Primary, at most `MAX_PERMANENT_PER_PRIMARY`,
    depends on the host alone and is passed as zero for a temporary move.
    """
    if count >= MAX_PERMANENT_PER_PRIMARY:
        return False
    if (
        float(np.abs(perm_sum[onset:]).max(initial=0.0)) + depth
        > MAX_PERMANENT_SUM + FLOOR_TOLERANCE
    ):
        return False
    if mva:
        shifted = clean_row + perm_sum
        shifted[onset:] -= depth
        finite = np.isfinite(shifted)
        if finite.any() and shifted[finite].min() < -FLOOR_TOLERANCE:
            return False
    return True


def _recipient_shares(*, rng: np.random.Generator, mirror_class: int, n_near: int) -> list[float]:
    """Return the shares of the megawatts moved that each metered recipient takes."""
    if mirror_class == 1 or n_near == 1:
        return [float(rng.uniform(*SINGLE_SHARE))]
    n_recipients = int(rng.integers(2, min(3, n_near) + 1))
    total = float(rng.uniform(*MULTI_TOTAL_SHARE))
    weights = rng.dirichlet(np.ones(n_recipients))
    return [float(total * w) for w in weights]


def training_world(  # noqa: PLR0915
    *, clean: np.ndarray, end: int, first: int, last: int, world_seed: int,
    table: nb.NeighbourTable, capacity_mw: np.ndarray, mva_only: np.ndarray, vis: np.ndarray,
) -> TrainingWorld:  # fmt: skip
    """Return one draw of the training world of a fold.

    Args:
        clean: (16, G) cleaned power over capacity, NaN where flagged or missing. Only the
            columns before `end` are read.
        end: First grid index that the world must not read, the first half-hour of the fold month.
            A permanent change runs to `end`.
        first: First grid index of the schedule (experiment 713's `c_idx.min() - 768`).
        last: End of the schedule (`c_idx.max() + 672`).
        world_seed: The seed of the draw (`train_seed` or `held_out_seed`).
        table: The neighbour table (a recipient is a listed neighbour of the donor).
        capacity_mw: (16,) effective capacity in megawatts.
        mva_only: (16,) bool, True for MVA-only Primaries (steps are floored at zero).
        vis: (16,) visible depth in shares of capacity (`visible_depth`).

    Returns:
        The history and target arrays (G = `end` columns) and the event table. `redrawn` counts
        events (either kind) that failed the floor test of an MVA-only Primary or a limit on
        stacking and were moved to a new onset at their own Primary, with their sign and depth kept.
        `dropped` counts events for which no onset passed.
    """
    assert vis.shape == (nb.N_PRIMARIES,), vis.shape
    assert (vis < DEPTH_MAX).all(), vis
    assert 0 < last <= end <= clean.shape[1], (last, end, clean.shape)
    clean = clean[:, :end]
    n_grid = clean.shape[1]
    history = clean.copy()
    target = clean.copy()
    rng = np.random.default_rng([WORLD_KEY, world_seed])
    rng_redraw = np.random.default_rng([REDRAW_KEY, world_seed])
    perm_sum = np.zeros(clean.shape)
    perm_count = np.zeros(nb.N_PRIMARIES, dtype=int)
    events: list[dict] = []
    redraws = {
        "redrawn_temporary": 0,
        "redrawn_permanent": 0,
        "dropped_temporary": 0,
        "dropped_permanent": 0,
    }
    first_try: dict[str, list[int]] = {}
    span = last - first
    for p in range(nb.N_PRIMARIES):
        covered = 0
        while covered < EVENT_COVER * span:
            ev = draw_event(rng=rng, vis=float(vis[p]), n_grid=n_grid)
            kind = "temporary" if ev.temporary else "permanent"
            placing = ev.length if ev.temporary else PERMANENT_COVER_DAYS * PER_DAY
            covered += placing
            host = p
            onset = place_onset(fraction=ev.onset_fraction, first=first, last=last, length=placing)
            shift = ev.sign * ev.depth

            def fits(*, host: int, onset: int, shift: float = shift, ev: DrawnEvent = ev) -> bool:
                return _floor_ok(
                    history_row=history[host], onset=onset, shift=shift, mva=bool(mva_only[host])
                ) and _limits_ok(
                    clean_row=clean[host], perm_sum=perm_sum[host],
                    count=0 if ev.temporary else int(perm_count[host]), onset=onset,
                    depth=ev.depth, mva=bool(mva_only[host]),
                )  # fmt: skip

            ok = fits(host=host, onset=onset)
            group = "mva_only" if mva_only[host] else "signed_mw"
            stat = first_try.setdefault(f"{kind}|{group}|{'fall' if shift < 0 else 'rise'}", [0, 0])
            stat[0] += 1
            stat[1] += int(ok)
            tries = 0
            while not ok and tries < MAX_REDRAWS:
                onset = place_onset(
                    fraction=float(rng_redraw.random()), first=first, last=last, length=placing
                )  # fmt: skip
                ok = fits(host=host, onset=onset)
                tries += 1
            if tries:
                redraws[f"redrawn_{kind}"] += 1
            if not ok:
                redraws[f"dropped_{kind}"] += 1
                continue
            length = ev.length if ev.temporary else n_grid - onset
            if not ev.temporary:
                perm_sum[host, onset:] += shift
                perm_count[host] += 1
            w715.add_shift(
                array=history, primary=host, onset=onset, length=length, shift=shift
            )  # fmt: skip
            if not ev.temporary:
                w715.add_shift(
                    array=target, primary=host, onset=onset, length=length, shift=shift
                )  # fmt: skip
            near = table.neighbours[host]
            recipients: list[int] = []
            shares: list[float] = []
            megawatts: list[float] = []
            if ev.temporary:
                want = ev.mirror_class
            else:
                want = 1 if (near and rng.random() < CHANCE_MIRROR) else 0
            if want and near:
                shares = _recipient_shares(rng=rng, mirror_class=want, n_near=len(near))
                order = rng.permutation(len(near))[: len(shares)]
                recipients = [int(near[i]) for i in order]
                for r, s in zip(recipients, shares, strict=True):
                    mw = -shift * capacity_mw[host] * s
                    megawatts.append(float(mw))
                    w715.add_shift(
                        array=history, primary=r, onset=onset, length=length,
                        shift=mw / capacity_mw[r], floor=bool(mva_only[r]),
                    )  # fmt: skip
            events.append(
                {
                    "host": host, "onset": onset, "length": length,
                    "kind": "temporary" if ev.temporary else "permanent", "shift": shift,
                    "depth": ev.depth, "sign": ev.sign, "visible": ev.depth >= float(vis[host]),
                    "mirror_class": len(recipients), "recipients": recipients, "shares": shares,
                    "megawatts": megawatts, "chance": bool(recipients and not ev.temporary),
                }
            )  # fmt: skip
    return TrainingWorld(
        history=history, target=target, events=events, seed=world_seed,
        redrawn=redraws["redrawn_temporary"] + redraws["redrawn_permanent"],
        dropped=redraws["dropped_temporary"] + redraws["dropped_permanent"], redraws=redraws,
        first_try=first_try,
    )  # fmt: skip


# ---------------------------------------------------------------------- checks of a draw (GH5, GH7)


def check_targets_and_limits(
    *, world: TrainingWorld, clean: np.ndarray, mva_only: np.ndarray
) -> dict[str, float]:  # fmt: skip
    """GH7: assert clean targets and the three limits on stacked permanent changes; return counts.

    The target equals the cleaned power plus the permanent changes alone, so no target half-hour
    of a donor or of a recipient carries a temporary move, and every half-hour after the onset of a
    permanent change carries the change. A Primary has at most two permanent changes, the summed
    permanent shift stays within +-0.50, and the shifted power of an MVA-only Primary stays at or
    above zero. The realised permanent share of events is returned.
    """
    expected = clean.copy()
    count = np.zeros(nb.N_PRIMARIES, dtype=int)
    perm_sum = np.zeros(clean.shape)
    for e in world.events:
        if e["kind"] == "permanent":
            w715.add_shift(
                array=expected, primary=e["host"], onset=e["onset"], length=e["length"],
                shift=e["shift"],
            )  # fmt: skip
            count[e["host"]] += 1
            perm_sum[e["host"], e["onset"] : e["onset"] + e["length"]] += e["shift"]
            assert e["onset"] + e["length"] == clean.shape[1], "a permanent change must never end"
    same = np.array_equal(world.target, expected, equal_nan=True)
    assert same, "a target differs from the cleaned power plus the permanent changes"
    assert int(count.max()) <= MAX_PERMANENT_PER_PRIMARY, count
    assert float(np.abs(perm_sum).max()) <= MAX_PERMANENT_SUM + FLOOR_TOLERANCE, "stack above 0.50"
    for p in np.nonzero(mva_only)[0]:
        row = world.target[p]
        finite = np.isfinite(row)
        floor_ok = (not finite.any()) or row[finite].min() >= -FLOOR_TOLERANCE
        assert floor_ok, f"MVA-only Primary {p} goes below zero in the target"
    for p in np.nonzero(mva_only)[0]:
        row = world.history[p]
        finite = np.isfinite(row)
        floor_ok = (not finite.any()) or row[finite].min() >= -FLOOR_TOLERANCE
        assert floor_ok, f"MVA-only Primary {p} goes below zero in the history"
    n = len(world.events)
    return {
        "n_events": n,
        "permanent_share": sum(e["kind"] == "permanent" for e in world.events) / max(n, 1),
        **world.redraws,
    }


def realised_cover(*, world: TrainingWorld, clean: np.ndarray, first: int, last: int) -> float:
    """Return the share of half-hours in `[first, last)` that carry an event at their Primary."""
    inside = np.zeros(clean.shape, dtype=bool)
    for e in world.events:
        span = PERMANENT_COVER_DAYS * PER_DAY if e["kind"] == "permanent" else e["length"]
        inside[e["host"], e["onset"] : e["onset"] + span] = True
    return float(inside[:, first:last].mean())


def _chi2_p(*, frame: pl.DataFrame, column: str) -> tuple[pl.DataFrame, float]:
    """Return the table of kind by `column`, and the p-value of a chi-squared independence test."""
    table = frame.group_by("kind", column).len().pivot(on=column, index="kind", values="len")
    counts = table.drop("kind").fill_null(0).to_numpy().astype(float)
    counts = counts[:, counts.sum(axis=0) > 0]
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        return table, 1.0
    return table, float(chi2_contingency(counts).pvalue)


def independence_tables(
    *, worlds: list[TrainingWorld], mva_only: np.ndarray, level: float
) -> dict[str, object]:  # fmt: skip
    """GH5: pool the draws of one fold, return the tables of kind by sign, and test independence.

    The tables are kind by sign, kind by Primary, and kind by sign within each group (the MVA-only
    Primaries and the signed-MW Primaries), each with a chi-squared p-value.
    The unmirrored temporary moves must show as many rises as falls (a two-sided z-score). Every
    test is read at `level`,
    which the gate sets to the family level `INDEPENDENCE_LEVEL` over the number of tests of the
    gate (a Bonferroni correction). Pooling the draws of a fold gives the Primary table enough
    events for a chi-squared test.
    """
    from scipy.stats import norm

    frame = pl.DataFrame([e for w in worlds for e in w.events]).with_columns(
        group=pl.col("host").map_elements(
            lambda h: "mva_only" if mva_only[h] else "signed_mw", return_dtype=pl.String
        )
    )
    out: dict[str, object] = {"n_events": frame.height}
    notices: list[str] = []
    out["notices"] = notices
    tests = {
        "kind_by_sign": ("sign", frame),
        "kind_by_host": ("host", frame),
        "kind_by_sign_mva_only": ("sign", frame.filter(pl.col("group") == "mva_only")),
        "kind_by_sign_signed_mw": ("sign", frame.filter(pl.col("group") == "signed_mw")),
    }
    for name, (column, sub) in tests.items():
        table, p_value = _chi2_p(frame=sub, column=column)
        kinds = set(sub["kind"].to_list())
        if kinds != {"temporary", "permanent"}:
            notices.append(
                f"NOTICE: {name} has events of kinds {sorted(kinds)} only, so p = 1 is a default"
            )
        out[name] = table
        out[f"p_{name}"] = p_value
        assert p_value >= level, (
            f"kind is not independent: {name} (chi-squared p={p_value:.2e}, level {level:.2e})"
        )
    bare = frame.filter((pl.col("kind") == "temporary") & (pl.col("mirror_class") == 0))
    n = bare.height
    rises = int((bare["sign"] > 0).sum())
    z = (rises - n / 2) / max(np.sqrt(n / 4), 1e-9)
    out["unmirrored_temporary"] = {"n": n, "rises": rises, "z": float(z)}
    limit = float(norm.isf(level / 2))
    assert abs(z) < limit, (
        f"unmirrored temporary rises and falls are unbalanced (z={z:.2f}, limit {limit:.2f})"
    )
    return out


N_INDEPENDENCE_TESTS: Final = 5
"""Tests per fold in `independence_tables`: four chi-squared tests and one z-test."""


def world_hash(*, world: TrainingWorld) -> str:
    """Return the sha256 of the history and target arrays of a draw (GH9; a fit records it too)."""
    h = hashlib.sha256()
    for array in (world.history, world.target):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def fall_share_gap(
    *, worlds: list[TrainingWorld], mva_only: np.ndarray, level: float
) -> dict[str, dict[str, float]]:  # fmt: skip
    """Return, per group, the fall shares of the two kinds and a z-test of their difference.

    **This is the effect-size reading of GH5, and it counts towards the verdict of the gate.** It
    pools the given draws (all 400 of the gate). The chi-squared tests of a single fold have too few
    events to see a gap of 0.05 between the fall share of permanent changes and that of temporary
    moves, and a gap of that size lets a network read the kind from the group and the sign. The test
    is a z-test on the difference of two proportions with a pooled variance, in each group, read at
    `level` (the Bonferroni level of the other GH5 tests). `power_gap` is the smallest gap that the
    test finds with 80% power at the event counts of the draws. A group with no event of a kind
    fails the test (`passes` is 0, and the gap, z, and p are NaN).
    """
    from scipy.stats import norm

    frame = pl.DataFrame([e for w in worlds for e in w.events]).with_columns(
        group=pl.col("host").map_elements(
            lambda h: "mva_only" if mva_only[h] else "signed_mw", return_dtype=pl.String
        )
    )
    z_alpha = float(norm.isf(level / 2))
    z_power = float(norm.ppf(0.8))
    out: dict[str, dict[str, float]] = {}
    for group in ("mva_only", "signed_mw"):
        reading: dict[str, float] = {}
        falls: dict[str, int] = {}
        for kind in ("permanent", "temporary"):
            sub = frame.filter((pl.col("group") == group) & (pl.col("kind") == kind))
            falls[kind] = int(np.sum(sub["sign"].to_numpy() < 0))
            reading[f"n_{kind}"] = float(sub.height)
            reading[f"fall_share_{kind}"] = falls[kind] / sub.height if sub.height else float("nan")
        n_p, n_t = reading["n_permanent"], reading["n_temporary"]
        reading["gap"] = reading["fall_share_permanent"] - reading["fall_share_temporary"]
        pooled = (falls["permanent"] + falls["temporary"]) / max(n_p + n_t, 1.0)
        se = float(np.sqrt(pooled * (1 - pooled) * (1 / max(n_p, 1.0) + 1 / max(n_t, 1.0))))
        if n_p == 0 or n_t == 0 or se == 0:
            reading.update(gap=float("nan"), z=float("nan"), p=float("nan"), power_gap=float("nan"))
            reading["passes"] = 0.0
            out[group] = reading
            continue
        reading["z"] = reading["gap"] / se
        reading["p"] = float(2 * norm.sf(abs(reading["z"])))
        reading["power_gap"] = (z_alpha + z_power) * se
        reading["passes"] = float(reading["p"] >= level)
        out[group] = reading
    return out


def _event_frame(*, worlds: list[TrainingWorld], mva_only: np.ndarray) -> pl.DataFrame:
    """Return the events of the draws with their group (`mva_only`, or `signed_mw` for the rest)."""
    return pl.DataFrame([e for w in worlds for e in w.events]).with_columns(
        group=pl.col("host").map_elements(
            lambda h: "mva_only" if mva_only[h] else "signed_mw", return_dtype=pl.String
        )
    )


def kind_independence_pooled(
    *, worlds: list[TrainingWorld], mva_only: np.ndarray, level: float
) -> dict[str, object]:  # fmt: skip
    """GH5: test kind by group (2 by 2) and kind by Primary over all the draws, at `level`.

    **These two tests see a cue from the Primary to the kind that the per-fold tests cannot see.**
    The `signed_mw` group holds the 8 Primaries that are not MVA-only, which include the two whose
    unit is not signed megawatts. Both chi-squared tests count events. The kind-by-group table and
    the events per Primary and kind are returned for printing.
    """
    frame = _event_frame(worlds=worlds, mva_only=mva_only)
    table_group, p_group = _chi2_p(frame=frame, column="group")
    table_host, p_host = _chi2_p(frame=frame, column="host")
    return {
        "table_group": table_group, "p_kind_by_group": p_group,
        "table_host": table_host.sort("kind"), "p_kind_by_host_pooled": p_host,
        "per_primary": frame.group_by("host", "kind").len().sort("host", "kind"),
        "passes": bool(p_group >= level and p_host >= level),
    }  # fmt: skip


def coverage_problems(
    *, worlds_by_month: dict[str, list[TrainingWorld]], mva_only: np.ndarray, pooled: int,
    per_month: int,
) -> list[str]:  # fmt: skip
    """Return one message for each group and kind, and each Primary, short of its coverage floor.

    **The training world must hold events of both kinds in both groups, and at every Primary.** The
    floors of a group and kind are `pooled` events over all the draws and `per_month` events among
    the draws of each fold month. A group with no event of a kind fails both. Each Primary must hold
    at least one event of each kind over all the draws. An empty list means the coverage holds. The
    gate stops on any message, with no waiver and no fallback, and the maintainer is told.
    """
    counts: dict[tuple[str, str, str], int] = {}
    per_primary: dict[tuple[int, str], int] = {}
    for month, worlds in worlds_by_month.items():
        for world in worlds:
            for event in world.events:
                group = "mva_only" if mva_only[event["host"]] else "signed_mw"
                key = (group, event["kind"], month)
                counts[key] = counts.get(key, 0) + 1
                pk = (int(event["host"]), event["kind"])
                per_primary[pk] = per_primary.get(pk, 0) + 1
    problems = []
    for group in ("mva_only", "signed_mw"):
        for kind in ("permanent", "temporary"):
            by_month = {m: counts.get((group, kind, m), 0) for m in worlds_by_month}
            if sum(by_month.values()) < pooled:
                problems.append(f"{group} {kind}: {sum(by_month.values())} events, floor {pooled}")
            short = {m: n for m, n in by_month.items() if n < per_month}
            if short:
                problems.append(f"{group} {kind}: under {per_month} events in {short}")
    problems.extend(
        f"Primary {primary}: no {kind} event in the draws"
        for primary in range(nb.N_PRIMARIES)
        for kind in ("permanent", "temporary")
        if per_primary.get((primary, kind), 0) < 1
    )
    return problems


def check_shared_draw(*, host_a: np.ndarray, host_b: np.ndarray) -> None:
    """GH9: assert that the host arrays of `Rb` and `Rbn` for one fold and seed are identical."""
    assert np.array_equal(host_a, host_b, equal_nan=True), "Rb and Rbn read different host arrays"


# ---------------------------------------------------------------------- the counted event set


def donor_step(*, truth: pl.DataFrame, capacity_mw: np.ndarray) -> np.ndarray:
    """Return the unsigned step at the donor of each temporary move of a `g2` truth table.

    A row scored at the donor holds the donor's step in `size`. A row scored at the receiving
    Primary holds the recipient's step `moved` in shares of the recipient's capacity, which equals
    `share * donor_step * capacity_donor / capacity_recipient`, so the donor's step is recovered
    from `size`, `share`, and the two capacities (`moved` is capped at 0.6 in the generator, which
    understates the donor's step for a capped row). A row with no donor in the panel (no
    `recipient`) has a donor step of NaN.
    """
    out = np.full(truth.height, np.nan)
    for i, row in enumerate(truth.iter_rows(named=True)):
        if row["truth"] != "switch":
            continue
        if not row["receiving_host"]:
            out[i] = abs(row["size"])
        elif row["recipient"] is not None and row["recipient"] >= 0 and row["share"]:
            donor = row["recipient"]
            assert abs(row["size"]) < RECIPIENT_CAP, f"a receiving-host row is capped: {row}"
            out[i] = (
                abs(row["size"]) * capacity_mw[row["host"]] / (row["share"] * capacity_mw[donor])
            )
    return out


def g2_events(*, capacity_mw: np.ndarray, vis_by_month: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return one row per `g2` event, with the donor step and the visible-depth flags.

    The columns added to experiment 714's truth table are `donor_step` (unsigned, in shares of the
    donor's capacity; NaN for a permanent change), `donor` (the donor's Primary index, or -1),
    `depth_host` (the unsigned step at the scored host), `visible` (the step passes the visible
    depth at its own Primary, with the visible depth of the fold month), and `metered` (the mirror
    state is `full` or `partial`).
    """
    frames = []
    for month in INJECTED_MONTHS:
        for index in g2gen.stored_indices(world="g2", month=month):
            truth = g2gen.read_pass(world="g2", month=month, index=index).truth
            donors = np.where(
                truth["receiving_host"].to_numpy(),
                truth["recipient"].to_numpy(),
                truth["host"].to_numpy(),
            )
            donors = np.where(truth["truth"].to_numpy() == "switch", donors, -1)
            step = donor_step(truth=truth, capacity_mw=capacity_mw)
            vis = vis_by_month[month]
            host_vis = vis[truth["host"].to_numpy()]
            depth_host = np.abs(truth["size"].to_numpy())
            donor_vis = np.where(donors >= 0, vis[np.clip(donors, 0, None)], np.nan)
            frames.append(
                truth.with_columns(
                    pl.Series("donor_step", step),
                    pl.Series("donor", donors, dtype=pl.Int64),
                    pl.Series("depth_host", depth_host),
                    pl.Series(
                        "visible",
                        np.where(
                            truth["truth"].to_numpy() == "switch",
                            step >= donor_vis,
                            depth_host >= host_vis,
                        ),
                    ),
                    pl.col("mirror_state").is_in(["full", "partial"]).alias("metered"),
                )
            )
    return pl.concat(frames)


def counted_moves(*, events: pl.DataFrame, floor_only: bool = False) -> pl.DataFrame:
    """Return `g2`'s temporary moves of visible depth with a metered mirror (design section 2.2).

    The filter is the metered mirror and a donor step that passes the visible depth. With
    `floor_only` the filter keeps the donor step of at least 0.10 of capacity alone, which is the
    count of 81 before the reference-error term.
    """
    moves = events.filter((pl.col("truth") == "switch") & pl.col("metered"))
    if floor_only:
        return moves.filter(pl.col("donor_step") >= VISIBLE_FLOOR)
    return moves.filter(pl.col("visible"))


@dataclass(frozen=True)
class CountReport:
    """The counts of GH4.

    Attributes:
        all_temporary: Temporary moves of `g2`.
        metered: Of those, the moves with a metered mirror.
        floor_only: Of those, the moves with a donor step of at least 0.10 of capacity.
        visible_depth: Of those, the moves that pass the full visible-depth definition.
        lasting: Of the counted moves (visible depth), those that last 7 days or more.
        lasting_floor_only: Of the floor-only moves, those that last 7 days or more (the design's
            43 of 81).
        per_month: The counted moves per fold month.
        per_mirror_state: The counted moves per mirror state.
        per_band: The counted moves per duration band.
    """

    all_temporary: int
    metered: int
    floor_only: int
    visible_depth: int
    lasting: int
    lasting_floor_only: int
    per_month: pl.DataFrame
    per_mirror_state: pl.DataFrame
    per_band: pl.DataFrame


def count_report(*, events: pl.DataFrame) -> CountReport:
    """GH4: return the counts of the counted events (the gate applies the floors)."""
    switches = events.filter(pl.col("truth") == "switch")
    counted = counted_moves(events=events)
    return CountReport(
        all_temporary=switches.height,
        metered=switches.filter(pl.col("metered")).height,
        floor_only=counted_moves(events=events, floor_only=True).height,
        visible_depth=counted.height,
        lasting=counted.filter(pl.col("duration_days") >= 7).height,
        lasting_floor_only=counted_moves(events=events, floor_only=True)
        .filter(pl.col("duration_days") >= 7)
        .height,
        per_month=counted.group_by("month").len().sort("month"),
        per_mirror_state=counted.group_by("mirror_state").len().sort("mirror_state"),
        per_band=counted.group_by("band").len().sort("band"),
    )


def pooled_floor_ok(*, report: CountReport, pooled: int, per_month: int) -> bool:
    """Return True if the pooled count and the count per fold month meet the floors of GH4."""
    per = dict(
        zip(report.per_month["month"].to_list(), report.per_month["len"].to_list(), strict=True)
    )
    return report.visible_depth >= pooled and all(
        per.get(m, 0) >= per_month for m in INJECTED_MONTHS
    )

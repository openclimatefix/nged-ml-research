"""Tests of the windows, the neighbour channels, the batches, and the checks on the tensors."""

import dataclasses
from typing import Any, NamedTuple

import e00729_arms as arms729
import e00729_core as core729
import e00729_sim as sim
import e00730_arms as arms730
import e00730_masks as masks
import e00732_arms as arms
import e00732_core as core
import e00732_neighbours as nb
import e00732_overlays as ov
import numpy as np
import pytest
from world732 import make_bank, make_neighbours


class World(NamedTuple):
    """The invented bank, its overlays and prepared segments, and the neighbours' medians."""

    cal: sim.Calendar
    segs: list[sim.Segment]
    series: dict[str, np.ndarray]
    overlays: list[ov.TransferOverlay]
    prepared: list[arms.Prepared]
    index: ov.PairIndex
    stats: arms729.InputStats
    medians: dict[str, float]


@pytest.fixture(scope="module")
def world() -> World:
    cal, segs = make_bank(draws_per_world=3)
    series = make_neighbours(cal=cal)
    overlays = ov.overlay_segments(segments=segs, kind="train", seed=0, neighbours=series)
    stats = arms729.input_stats(cal=cal, train_days=np.arange(len(cal.dates)))
    medians = {
        n: nb.training_median(
            folded=nb.fold(signed=series[n], floor=0.0), train_rows=np.arange(len(cal.dates))
        )
        for n in nb.NEIGHBOURS
    }
    prepared = arms.prepare(overlays=overlays, holidays=cal.holidays, stats=stats, medians=medians)
    return World(
        cal, segs, series, overlays, prepared, ov.build_pair_index(segments=segs), stats, medians
    )


def need(array: np.ndarray | None) -> np.ndarray:
    """Return an optional batch member that the test built, failing when it is absent."""
    assert array is not None
    return array


def edge_picks(*, prepared: list[arms.Prepared], limit: int = 60) -> list[tuple[int, int]]:
    """Return windows whose scored day holds an edge of a mirrored transfer."""
    picks: list[tuple[int, int]] = []
    for i, p in enumerate(prepared):
        for tr in p.transfers:
            if not tr.mirrored:
                continue
            for hh in (tr.start_half_hour, tr.end_half_hour):
                if hh // 48 in p.days and (i, hh // 48) not in picks:
                    picks.append((i, hh // 48))
    return picks[:limit]


def train_batch(
    *, world: World, spec: core.ArmSpec732, picks: list[tuple[int, int]], seed: int = 0
) -> arms.Batch:
    """Return a training batch of the arm on the given windows."""
    return arms.make_batch(
        prepared=world.prepared, picks=picks, spec=spec, rng=np.random.default_rng(seed),
        index=world.index if spec.inject else None,
    )  # fmt: skip


def test_the_plain_window_starts_with_experiment_730s_arm_s_window(world: World) -> None:
    spec730 = arms730.ArmSpec("S", "d")
    for p in world.prepared[:6]:
        day = int(p.days[3])
        mine = arms.build_triple(
            p=p, day=day, spec=arms.eval_spec(), hidden=None, delta=None, eps=None
        )
        theirs, target, _ = arms730.build_one(prepared=p.base, dense_day=day, spec=spec730)
        assert mine.x.shape == (arms.N_BASE + 12, 96)
        np.testing.assert_array_equal(mine.x[: arms.N_BASE], theirs)
        assert tuple(arms730.stack_names(spec=spec730)) == arms.NAMES
        np.testing.assert_array_equal(arms.target_window(p=p, day=day)[:6], target)
    assert arms.N_CHANNELS == arms.N_BASE + nb.N_NEIGHBOUR_CHANNELS == len(arms.ALL_NAMES)
    assert arms.NEIGHBOUR_NAMES[0] == "D3_value"
    assert arms.NEIGHBOUR_NAMES[-1] == "D14_mask"


def test_neighbour_channels_are_the_real_loaders_values_over_the_median(world: World) -> None:
    p = world.prepared[0]
    day = int(p.days[5])
    w = arms.build_triple(p=p, day=day, spec=arms.eval_spec(), hidden=None, delta=None, eps=None).x
    s = arms.window_start(day=day)
    o = world.overlays[p.source_index]
    for name in nb.NEIGHBOURS:
        mirrored = o.neighbours[name].reshape(-1)
        want = nb.fold(signed=mirrored, floor=o.floors[name])[s : s + 96] / world.medians[name]
        ok = np.isfinite(want)
        assert ok.any(), name
        got = w[arms.NB_ROW[f"{name}_value"]]
        np.testing.assert_allclose(got[ok], want[ok], rtol=1e-5)
        assert (got[~ok] == 0).all()
        np.testing.assert_array_equal(w[arms.NB_ROW[f"{name}_mask"]], ok.astype(np.float32))
        change = w[arms.NB_ROW[f"{name}_change"]]
        both = ok[1:] & ok[:-1]
        np.testing.assert_allclose(change[1:][both], np.diff(want)[both], rtol=1e-4, atol=1e-6)
        assert change[0] == 0.0


def test_the_aligned_channel_follows_the_sign_of_the_reading_change(world: World) -> None:
    spec = arms.eval_spec()
    seen = 0
    for p in world.prepared[:8]:
        tr = arms.build_triple(
            p=p, day=int(p.days[4]), spec=spec, hidden=None, delta=None, eps=None
        )
        sign = np.sign(arms.reading_change(window=tr.x[: arms.N_BASE]))
        for name in nb.NEIGHBOURS:
            change = tr.x[arms.NB_ROW[f"{name}_change"]]
            np.testing.assert_allclose(
                tr.x[arms.NB_ROW[f"{name}_aligned"]], change * sign, atol=1e-7
            )
            seen += int(((change != 0) & (sign != 0)).sum())
    assert seen > 100


def test_hiding_zeroes_all_twelve_channels_and_never_leaks_the_hidden_steps_sign(
    world: World,
) -> None:
    spec = arms.eval_spec(inject=True, raised=True)
    p = world.prepared[0]
    day = int(p.days[4])
    hidden = np.zeros(96, dtype=bool)
    hidden[30:50] = True
    sib = ov.fixed_sibling(index=world.index, segment=p.source_index, day=day)
    delta = (
        None
        if sib is None
        else ov.window_delta(index=world.index, sibling=sib, amount=20.0, day=day)
    )
    triple = arms.build_triple(p=p, day=day, spec=spec, hidden=hidden, delta=delta, eps=0.2)
    for member in (triple.x, need(triple.xi), need(triple.xc)):
        assert (member[arms.N_BASE :, hidden] == 0).all()
        assert (member[arms.ROW["r"]][hidden] == 0).all()
        assert (member[arms.ROW["r_mask"]][hidden] == 0).all()
        assert (member[arms.N_BASE :, ~hidden] != 0).any()
        # the first shown half-hour after the span has no change: the previous half-hour is hidden
        for name in nb.NEIGHBOURS:
            assert member[arms.NB_ROW[f"{name}_change"], 50] == 0.0
            assert member[arms.NB_ROW[f"{name}_aligned"], 50] == 0.0
    # a hidden reading contributes no sign: the neighbour channels do not depend on it
    other = p.base.stack.copy()
    s = arms.window_start(day=day)
    other[arms.ROW["r"], s + 30 : s + 50] += 7.0
    swapped = dataclasses.replace(p, base=dataclasses.replace(p.base, stack=other))
    again = arms.build_triple(
        p=swapped, day=day, spec=arms.eval_spec(), hidden=hidden, delta=None, eps=None
    )
    first = arms.build_triple(
        p=p, day=day, spec=arms.eval_spec(), hidden=hidden, delta=None, eps=None
    )
    np.testing.assert_array_equal(again.x[arms.N_BASE :], first.x[arms.N_BASE :])


def test_injected_member_differs_only_in_the_reading_bsp_and_aligned_channels(
    world: World,
) -> None:
    spec = arms.eval_spec(inject=True)
    checked = 0
    for p in world.prepared:
        for day in map(int, p.days):
            sib = ov.fixed_sibling(index=world.index, segment=p.source_index, day=day)
            if sib is None:
                continue
            delta = ov.window_delta(index=world.index, sibling=sib, amount=20.0, day=day)
            tr = arms.build_triple(p=p, day=day, spec=spec, hidden=None, delta=delta, eps=None)
            changed = {
                arms.ALL_NAMES[i]
                for i in range(arms.N_CHANNELS)
                if not np.array_equal(tr.x[i], need(tr.xi)[i])
            }
            allowed = {"r", "d_bsp", *(f"{n}_aligned" for n in nb.NEIGHBOURS)}
            assert changed <= allowed, changed
            np.testing.assert_allclose(
                need(tr.ti)[arms.T["x"]], tr.t[arms.T["x"]] - tr.delta, atol=1e-5
            )
            np.testing.assert_allclose(
                need(tr.ti)[arms.T["solar"]], tr.t[arms.T["solar"]] + tr.delta, atol=1e-5
            )
            assert np.array_equal(need(tr.ti)[arms.T["event"]], tr.t[arms.T["event"]])
            # the reading is the original wherever delta is zero, exactly
            zero = delta[1:] == 0.0
            np.testing.assert_array_equal(
                need(tr.xi)[arms.ROW["r"]][zero], tr.x[arms.ROW["r"]][zero]
            )
            checked += 1
            if checked >= 12:
                return
    assert checked > 0


def test_injected_reading_matches_the_refold_of_the_injected_truth(world: World) -> None:
    spec = arms.eval_spec(inject=True)
    for p in world.prepared:
        for day in map(int, p.days):
            sib = ov.fixed_sibling(index=world.index, segment=p.source_index, day=day)
            if sib is None:
                continue
            delta = ov.window_delta(index=world.index, sibling=sib, amount=50.0, day=day)
            tr = arms.build_triple(p=p, day=day, spec=spec, hidden=None, delta=delta, eps=None)
            ok = need(tr.ti)[arms.T["valid"]] > 0.5
            got = need(tr.xi)[arms.ROW["r"], arms.MID : arms.MID + 48] * arms729.R_SCALE
            np.testing.assert_allclose(got[ok], need(tr.ti)[arms.T["r"]][ok], rtol=1e-4, atol=1e-3)
            assert tr.delta.max() > 0
            return
    pytest.fail("no window had a sibling")


def test_bsp_channel_of_the_injected_member_carries_beta_times_delta(world: World) -> None:
    spec = arms.eval_spec(inject=True)
    for p in (q for q in world.prepared if q.has_bsp):
        for day in map(int, p.days):
            sib = ov.fixed_sibling(index=world.index, segment=p.source_index, day=day)
            if sib is None:
                continue
            delta = ov.window_delta(index=world.index, sibling=sib, amount=50.0, day=day)
            tr = arms.build_triple(p=p, day=day, spec=spec, hidden=None, delta=delta, eps=None)
            want = -p.beta * np.diff(delta.astype(np.float64)) / arms729.BSP_SCALE
            got = need(tr.xi)[arms.ROW["d_bsp"]] - tr.x[arms.ROW["d_bsp"]]
            ok = tr.x[arms.ROW["bsp_mask"]] > 0.5
            assert ok.any()
            assert np.abs(want[ok]).max() > 0
            np.testing.assert_allclose(got[ok], want[ok], atol=1e-5)
            return
    pytest.fail("no BSP window with a sibling")


def test_raised_member_differs_only_in_the_clear_sky_index(world: World) -> None:
    spec = arms.eval_spec(raised=True)
    p = next(q for q in world.prepared if (q.base.stack[arms.ROW["csi"]] > 0).any())
    tr = arms.build_triple(p=p, day=int(p.days[5]), spec=spec, hidden=None, delta=None, eps=0.2)
    w, wc = tr.x, need(tr.xc)
    changed = {arms.ALL_NAMES[i] for i in range(arms.N_CHANNELS) if not np.array_equal(w[i], wc[i])}
    assert changed == {"csi"}
    mask = ov.raisable(csi=w[arms.ROW["csi"]], cosz=w[arms.ROW["cosz"]])
    assert mask.any()
    assert (wc[arms.ROW["csi"]][mask] > w[arms.ROW["csi"]][mask]).all()
    night = ~mask & (w[arms.ROW["cosz"]] <= 0.1)
    np.testing.assert_array_equal(wc[arms.ROW["csi"]][night], w[arms.ROW["csi"]][night])


def test_the_three_members_share_one_hidden_span_and_the_keeps_follow_it(world: World) -> None:
    spec = arms.eval_spec(inject=True, raised=True)
    p = world.prepared[0]
    day = int(p.days[4])
    hidden = np.zeros(96, dtype=bool)
    hidden[30:50] = True
    sib = ov.fixed_sibling(index=world.index, segment=p.source_index, day=day)
    delta = (
        None
        if sib is None
        else ov.window_delta(index=world.index, sibling=sib, amount=20.0, day=day)
    )
    tr = arms.build_triple(p=p, day=day, spec=spec, hidden=hidden, delta=delta, eps=0.2)
    for window in (tr.x, need(tr.xi), need(tr.xc)):
        assert (window[arms.ROW["r"]][hidden] == 0).all()
        assert (window[arms.ROW["r_mask"]][hidden] == 0).all()
    scored_hidden = hidden[arms.MID : arms.MID + 48]
    valid = tr.t[arms.T["valid"]] > 0.5
    for term in ("sign", "step", "size", "injection"):
        assert (tr.keeps[arms.K[term]][scored_hidden] == 0).all()
    np.testing.assert_array_equal(tr.keeps[arms.K["solar"]], valid.astype(np.float32))
    assert (tr.keeps[arms.K["size"]] <= (tr.t[arms.T["event"]] > 0.5)).all()
    night = tr.x[arms.ROW["cosz"], arms.MID : arms.MID + 48] <= 0
    np.testing.assert_array_equal(tr.keeps[arms.K["night"]] > 0, valid & night)


def test_events_and_sizes_are_aligned_with_the_transfers(world: World) -> None:
    seen = 0
    for p in world.prepared:
        o = world.overlays[p.source_index]
        for tr in o.transfers:
            for hh, sign in ((tr.start_half_hour, 1.0), (tr.end_half_hour, -1.0)):
                day = hh // 48
                if day in p.days:
                    t = arms.target_window(p=p, day=int(day))
                    assert t[arms.T["event"], hh % 48] == 1.0
                    assert np.sign(t[arms.T["size"], hh % 48]) == np.sign(sign * tr.size)
                    seen += 1
    assert seen > 0


def test_no_input_equals_a_target_and_a_planted_leak_is_caught(world: World) -> None:
    spec = arms.eval_spec(inject=True, raised=True)
    rng = np.random.default_rng(0)
    batch = arms.make_batch(
        prepared=world.prepared,
        picks=arms.sample_picks(prepared=world.prepared, n=32, rng=rng),
        spec=spec,
        rng=rng,
        index=world.index,
    )
    for i in range(32):
        for window, target in ((batch.x[i], batch.t[i]), (need(batch.xi)[i], need(batch.ti)[i])):
            # the solar row is left out: the invented sky is a pure clear-sky shape, so the
            # clear-sky channels equal it. The real-file smoke test checks real skies.
            probe = target.copy()
            probe[arms.T["solar"]] = 0.0
            assert not arms.input_equals_target(window=window, targets=probe)
    leak = batch.x[0].copy()
    leak[3, arms.MID + 5] = 1.0
    t = batch.t[0].copy()
    t[arms.T["event"]] = leak[3, arms.MID : arms.MID + 48]
    t[arms.T["event"], 5] = 1.0
    assert arms.input_equals_target(window=leak, targets=t)
    leak2 = batch.x[0].copy()
    leak2[3, arms.MID : arms.MID + 48] = batch.t[0][arms.T["size"]]
    if np.ptp(batch.t[0][arms.T["size"]]) > 0:
        assert arms.input_equals_target(window=leak2, targets=batch.t[0])


def test_training_batch_masks_half_the_windows_and_pairs_through_the_vectorised_draw(
    world: World,
) -> None:
    spec = arms.eval_spec(inject=True, raised=True)
    rng = np.random.default_rng(1)
    picks = arms.sample_picks(prepared=world.prepared, n=400, rng=rng)
    batch = arms.make_batch(
        prepared=world.prepared, picks=picks, spec=spec, rng=rng, index=world.index
    )
    plain = arms.make_eval_batch(prepared=world.prepared, picks=picks, spec=arms.eval_spec())
    masked = (batch.x[:, arms.ROW["r_mask"]] != plain.x[:, arms.ROW["r_mask"]]).any(axis=1).mean()
    assert 0.35 < masked < 0.65
    assert (
        batch.hidden.any(axis=1).mean() >= masked
    )  # a span over a missing reading changes nothing
    assert batch.pair.mean() == 1.0  # every window of this invented world has a sibling
    assert batch.x.shape[1:] == (arms.N_CHANNELS, 96)
    assert batch.delta_win.shape == (400, ov.WINDOW_DELTA_LEN)
    np.testing.assert_array_equal(batch.delta, batch.delta_win[:, 1 + arms.MID : 1 + arms.MID + 48])
    assert batch.picks == list(picks)
    first = arms.make_batch(
        prepared=world.prepared, picks=picks, spec=spec, rng=np.random.default_rng(1),
        index=world.index,
    )  # fmt: skip
    second = arms.make_batch(
        prepared=world.prepared, picks=picks, spec=spec, rng=np.random.default_rng(1),
        index=world.index,
    )  # fmt: skip
    for name in ("x", "xi", "xc", "amount", "delta_win", "hidden"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def one_solar_world(world: World) -> tuple[list[arms.Prepared], ov.PairIndex]:
    """Return the bank cut to one solar-world segment per month, so that segment has no sibling."""
    kept, seen = [], set()
    for s in world.segs:
        if s.world in ov.SOLAR_WORLDS:
            if (s.month, "solar") in seen:
                continue
            seen.add((s.month, "solar"))
        kept.append(s)
    overlays = ov.overlay_segments(segments=kept, kind="test", neighbours=world.series)
    prep = arms.prepare(
        overlays=overlays, holidays=world.cal.holidays, stats=world.stats, medians=world.medians
    )
    return prep, ov.build_pair_index(segments=kept)


def test_a_window_without_a_sibling_takes_no_pair(world: World) -> None:
    prep, idx = one_solar_world(world)
    spec = arms.eval_spec(inject=True, raised=True)
    rng = np.random.default_rng(1)
    picks = arms.sample_picks(prepared=prep, n=400, rng=rng)
    batch = arms.make_batch(prepared=prep, picks=picks, spec=spec, rng=rng, index=idx)
    assert 0 < batch.pair.mean() < 1
    no_pair = batch.pair == 0
    np.testing.assert_array_equal(need(batch.xi)[no_pair], batch.x[no_pair])
    assert (batch.delta[no_pair] == 0).all()
    assert (batch.amount[no_pair] == 0).all()
    assert (batch.delta_win[no_pair] == 0).all()
    paired = batch.pair == 1
    assert (batch.amount[paired] >= core.INJECTION_POINTS[0] - 1e-6).all()
    assert (batch.amount[paired] <= core.INJECTION_POINTS[1] + 1e-6).all()
    assert batch.delta[paired].max() > 0


def test_sign_arm_batches_carry_no_extra_members(world: World) -> None:
    rng = np.random.default_rng(2)
    batch = arms.make_batch(
        prepared=world.prepared,
        picks=arms.sample_picks(prepared=world.prepared, n=8, rng=rng),
        spec=core.ARMS["S+"],
        rng=rng,
        index=None,
    )
    assert batch.xi is None
    assert batch.xc is None
    assert batch.ti is None
    assert batch.x.shape[1] == arms.N_CHANNELS


def test_eval_batch_has_the_fixed_pair_and_skips_windows_without_a_sibling(world: World) -> None:
    spec = arms.eval_spec(inject=True)
    prep, idx = one_solar_world(world)
    picks = arms.all_picks(prepared=prep)
    batch = arms.make_eval_batch(prepared=prep, picks=picks, spec=spec, index=idx, amount=20.0)
    assert 0 < batch.pair.sum() < len(picks)
    assert (batch.amount[batch.pair == 1] == 20.0).all()
    assert (batch.amount[batch.pair == 0] == 0.0).all()
    plain = arms.make_eval_batch(prepared=prep, picks=picks, spec=arms.eval_spec())
    np.testing.assert_array_equal(plain.x, batch.x)  # an evaluation window is never masked
    assert masks.SCORED_START == arms.MID
    assert core729.WINDOW_STEPS == arms.STEPS


def test_hide_bsp_zeroes_only_the_bsp_channels_in_every_member(world: World) -> None:
    picks = arms.all_picks(prepared=world.prepared)[:6]
    kw: dict[str, Any] = {
        "prepared": world.prepared,
        "picks": picks,
        "spec": arms.eval_spec(inject=True, raised=True),
        "index": world.index,
        "amount": 20.0,
        "eps": 0.2,
    }
    shown = arms.make_eval_batch(**kw)
    hidden = arms.make_eval_batch(**kw, hide_bsp=True)
    bsp_rows = (arms.ROW["d_bsp"], arms.ROW["bsp_mask"])
    for a, b in (
        (shown.x, hidden.x),
        (need(shown.xi), need(hidden.xi)),
        (need(shown.xc), need(hidden.xc)),
    ):
        assert (b[:, arms.ROW["d_bsp"]] == 0).all()
        assert (b[:, arms.ROW["bsp_mask"]] == 0).all()
        keep = [i for i in range(arms.N_CHANNELS) if i not in bsp_rows]
        np.testing.assert_array_equal(a[:, keep], b[:, keep])
    assert (shown.x[:, arms.ROW["bsp_mask"]] > 0).any()
    np.testing.assert_array_equal(shown.t, hidden.t)


# --------------------------------------------------------------------------- the two ablated arms


def test_u_n_has_no_neighbour_channel_and_u_nobsp_has_no_bsp_channel(world: World) -> None:
    picks = edge_picks(prepared=world.prepared)[:20]
    kw: dict[str, Any] = {
        "prepared": world.prepared,
        "picks": picks,
        "index": world.index,
        "amount": 20.0,
        "eps": 0.2,
    }
    full = arms.make_eval_batch(spec=arms.eval_spec(inject=True, raised=True), **kw)
    no_nb = arms.make_eval_batch(
        spec=arms.eval_spec(inject=True, raised=True, neighbours=False), **kw
    )
    no_bsp = arms.make_eval_batch(
        spec=arms.eval_spec(inject=True, raised=True, bsp_zeroed=True), **kw
    )
    for a, b in (
        (full.x, no_nb.x),
        (need(full.xi), need(no_nb.xi)),
        (need(full.xc), need(no_nb.xc)),
    ):
        assert b.shape[1] == arms.N_BASE == arms.n_channels(spec=arms.eval_spec(neighbours=False))
        np.testing.assert_array_equal(a[:, : arms.N_BASE], b)
    bsp_rows = (arms.ROW["d_bsp"], arms.ROW["bsp_mask"])
    for a, b in (
        (full.x, no_bsp.x),
        (need(full.xi), need(no_bsp.xi)),
        (need(full.xc), need(no_bsp.xc)),
    ):
        assert b.shape == a.shape
        assert not b[:, list(bsp_rows)].any()
        keep = [i for i in range(arms.N_CHANNELS) if i not in bsp_rows]
        np.testing.assert_array_equal(a[:, keep], b[:, keep])
    assert full.x[:, arms.ROW["bsp_mask"]].any()
    assert core.ARMS["U-N"].neighbours is False
    assert core.ARMS["U-noBSP"].bsp_zeroed is True
    assert arms.n_channels(spec=core.ARMS["U*"]) == arms.N_CHANNELS


def test_a_training_batch_of_each_ablated_arm_has_the_right_channels(world: World) -> None:
    picks = arms.sample_picks(prepared=world.prepared, n=16, rng=np.random.default_rng(3))
    for name, channels in (("U-N", arms.N_BASE), ("U-noBSP", arms.N_CHANNELS)):
        batch = train_batch(world=world, spec=core.ARMS[name], picks=picks)
        for member in (batch.x, need(batch.xi), need(batch.xc)):
            assert member.shape[1] == channels
    zeroed = train_batch(world=world, spec=core.ARMS["U-noBSP"], picks=picks)
    for member in (zeroed.x, need(zeroed.xi), need(zeroed.xc)):
        assert not member[:, [arms.ROW["d_bsp"], arms.ROW["bsp_mask"]]].any()


# --------------------------------------------------------------------------- checks on the tensors

FULL = arms.eval_spec(inject=True, raised=True)


def make_checked(*, world: World) -> arms.Batch:
    """Return a training batch of windows with a mirrored edge, and of solar-world windows."""
    picks = edge_picks(prepared=world.prepared)
    assert len(picks) > 10
    picks += [
        (i, int(d))
        for i, p in enumerate(world.prepared)
        if p.world in ov.SOLAR_WORLDS
        for d in p.days[:6]
    ]
    return train_batch(world=world, spec=FULL, picks=picks)


def failing(*, world: World, batch: arms.Batch, spec: core.ArmSpec732 = FULL) -> list[str]:
    """Return the names of the checks that fail on a batch."""
    return arms.tensor_problems(
        checks=arms.tensor_checks(batch=batch, prepared=world.prepared, spec=spec)
    )


def test_every_tensor_check_passes_on_a_real_training_batch(world: World) -> None:
    batch = make_checked(world=world)
    checks = arms.tensor_checks(batch=batch, prepared=world.prepared, spec=FULL)
    assert set(checks) == set(arms.TENSOR_BARS) - {"bsp_zero"}
    assert arms.tensor_problems(checks=checks) == [], checks
    for name, share in checks.items():
        assert share.n > 0, name
    assert checks["mirror_edges"].n >= 10


def test_the_checks_pass_on_an_evaluation_batch_and_on_the_ablated_arms(world: World) -> None:
    picks = edge_picks(prepared=world.prepared) + [
        (i, int(d))
        for i, p in enumerate(world.prepared)
        if p.world in ov.SOLAR_WORLDS
        for d in p.days[:6]
    ]
    hidden = np.stack(
        [
            masks.hidden_positions(
                style=core.STYLE, rng=masks.fixed_rng(style=core.STYLE, index=k), utc_offset=0.0
            )
            for k in range(len(picks))
        ]
    )  # a validation batch has fixed hidden spans, so the hidden check has cases
    for spec in (
        FULL,
        arms.eval_spec(inject=True, raised=True, neighbours=False),
        arms.eval_spec(inject=True, raised=True, bsp_zeroed=True),
    ):
        batch = arms.make_eval_batch(
            prepared=world.prepared,
            picks=picks,
            spec=spec,
            index=world.index,
            amount=20.0,
            eps=0.2,
            hidden=hidden,
        )
        checks = arms.tensor_checks(batch=batch, prepared=world.prepared, spec=spec)
        assert arms.tensor_problems(checks=checks) == [], (spec, checks)
    assert "mirror_edges" in checks


def test_a_silent_mirror_fails_the_mirror_edge_check(world: World) -> None:
    picks = edge_picks(prepared=world.prepared)
    silent = [dataclasses.replace(p, nb_folded=p.nb_folded_real) for p in world.prepared]
    batch = arms.make_batch(
        prepared=silent, picks=picks, spec=FULL, rng=np.random.default_rng(0), index=world.index
    )
    assert "mirror_edges" in failing(world=world, batch=batch)


def test_a_mirror_at_the_wrong_neighbour_fails_the_mirror_edge_check(world: World) -> None:
    picks = edge_picks(prepared=world.prepared)
    wrong = [
        dataclasses.replace(p, nb_folded=np.roll(p.nb_folded, 1, axis=0)) for p in world.prepared
    ]
    batch = arms.make_batch(
        prepared=wrong, picks=picks, spec=FULL, rng=np.random.default_rng(0), index=world.index
    )
    assert "mirror_edges" in failing(world=world, batch=batch)


def test_a_mirror_that_moves_another_neighbour_fails_the_mirror_edge_check(world: World) -> None:
    batch = make_checked(world=world)
    spill = batch.x.copy()
    for b, (seg_i, day) in enumerate(batch.picks):
        for tr in world.prepared[seg_i].transfers:
            if tr.mirrored:
                col = tr.start_half_hour - arms.window_start(day=day)
                if arms.MID <= col < arms.MID + 48:
                    other = nb.NEIGHBOURS[(nb.NEIGHBOURS.index(tr.partner) + 1) % 3]
                    spill[b, arms.NB_ROW[f"{other}_value"], col] += 0.5
    assert "mirror_edges" in failing(world=world, batch=dataclasses.replace(batch, x=spill))


def test_a_silent_aligned_channel_fails_the_aligned_check(world: World) -> None:
    batch = make_checked(world=world)
    x = batch.x.copy()
    x[:, [arms.NB_ROW[f"{n}_aligned"] for n in nb.NEIGHBOURS]] = 0.0
    bad = failing(world=world, batch=dataclasses.replace(batch, x=x))
    assert {"aligned_nonzero", "aligned_exact"} <= set(bad)


def test_the_aligned_channel_built_on_the_truths_sign_fails_the_aligned_check(
    world: World,
) -> None:
    batch = make_checked(world=world)
    x = batch.x.copy()
    truth = np.sign(np.diff(batch.t[:, arms.T["x"]], axis=1))
    for n in nb.NEIGHBOURS:
        change = x[:, arms.NB_ROW[f"{n}_change"]]
        aligned = np.zeros_like(change)
        aligned[:, arms.MID + 1 : arms.MID + 48] = change[:, arms.MID + 1 : arms.MID + 48] * truth
        x[:, arms.NB_ROW[f"{n}_aligned"]] = aligned
    bad = failing(world=world, batch=dataclasses.replace(batch, x=x))
    assert "aligned_exact" in bad
    assert "aligned_reading_not_truth" in bad


def test_a_hidden_half_hour_that_shows_a_neighbour_fails_the_hidden_check(world: World) -> None:
    batch = make_checked(world=world)
    b = int(np.flatnonzero(batch.hidden.any(axis=1))[0])
    col = int(np.flatnonzero(batch.hidden[b])[0])
    x = batch.x.copy()
    x[b, arms.NB_ROW["D5_value"], col] = 0.7
    assert "hidden_zero" in failing(world=world, batch=dataclasses.replace(batch, x=x))


def test_an_injected_member_with_no_extra_output_fails_the_injection_check(world: World) -> None:
    batch = make_checked(world=world)
    assert (batch.pair > 0).any()
    bad = failing(world=world, batch=dataclasses.replace(batch, xi=batch.x.copy()))
    assert {"injected_reading_on_delta", "injected_bsp_on_delta"} <= set(bad)


def test_an_injected_member_that_changes_elsewhere_fails_the_injection_check(world: World) -> None:
    batch = make_checked(world=world)
    xi = need(batch.xi).copy()
    b = int(np.flatnonzero(batch.pair > 0)[0])
    step = np.diff(batch.delta_win[b])
    zero = np.flatnonzero((batch.delta_win[b][1:] == 0.0) & (step == 0.0))[0]
    xi[b, arms.ROW["r"], zero] += 0.25
    xi[b, arms.ROW["d_bsp"], zero] += 0.25
    bad = failing(world=world, batch=dataclasses.replace(batch, xi=xi))
    assert {"injected_reading_elsewhere", "injected_bsp_elsewhere"} <= set(bad)


def test_a_raised_member_that_does_not_raise_fails_the_raised_check(world: World) -> None:
    batch = make_checked(world=world)
    assert "raised_above" in failing(
        world=world, batch=dataclasses.replace(batch, xc=batch.x.copy())
    )


def test_a_silent_solar_target_fails_the_solar_target_check(world: World) -> None:
    batch = make_checked(world=world)
    t = batch.t.copy()
    t[:, arms.T["solar"]] = 0.0
    assert "solar_target_positive" in failing(world=world, batch=dataclasses.replace(batch, t=t))


def test_a_batch_with_the_wrong_channels_fails_the_arm_input_checks(world: World) -> None:
    picks = edge_picks(prepared=world.prepared)
    full = train_batch(world=world, spec=FULL, picks=picks)
    no_neighbours = arms.eval_spec(inject=True, raised=True, neighbours=False)
    assert "channel_count" in failing(world=world, batch=full, spec=no_neighbours)
    no_bsp = arms.eval_spec(inject=True, raised=True, bsp_zeroed=True)
    assert "bsp_zero" in failing(world=world, batch=full, spec=no_bsp)
    thin = train_batch(world=world, spec=no_neighbours, picks=picks)
    assert "channel_count" in failing(world=world, batch=thin)


def test_a_check_with_no_case_fails(world: World) -> None:
    batch = make_checked(world=world)
    nothing = dataclasses.replace(
        batch, pair=np.zeros_like(batch.pair), hidden=np.zeros_like(batch.hidden)
    )
    bad = failing(world=world, batch=nothing)
    assert {"hidden_zero", "injected_reading_on_delta", "injected_bsp_on_delta"} <= set(bad)
    assert "raised_above" in failing(world=world, batch=dataclasses.replace(batch, xc=None))
    assert arms.tensor_problems(checks={"hidden_zero": ov.Share(0.0, 0)}) == ["hidden_zero"]


def test_real_pseudo_segments_carry_the_neighbours_when_given_and_zeros_without(
    world: World,
) -> None:
    """A real run reaches the neighbour channels through `prepare_plain` (a pseudo-segment has no beta)."""
    use = np.ones(len(world.cal.dates), bool)
    reading = np.nan_to_num(world.cal.reading, nan=5.0)
    real = [
        sim.real_segment(cal=world.cal, month=m, use=use, reading=reading, truth=reading)
        for m in ("2022-03", "2022-04")
    ]
    assert not any("beta" in s.params for s in real)
    spec = arms.eval_spec()
    kwargs = {"holidays": world.cal.holidays, "stats": world.stats, "medians": world.medians}
    with_nb = arms.prepare_plain(segments=real, neighbours=world.series, **kwargs)
    without = arms.prepare_plain(segments=real, neighbours=None, **kwargs)
    day = int(with_nb[0].days[3])
    shown = arms.plain_window(p=with_nb[0], day=day, spec=spec, hidden=None)
    absent = arms.plain_window(p=without[0], day=day, spec=spec, hidden=None)
    nb_mask_rows = [arms.NB_ROW[f"{n}_mask"] for n in nb.NEIGHBOURS]
    assert shown[nb_mask_rows].min() == 1.0  # every neighbour present on a real day
    assert not absent[arms.N_BASE :].any()  # no neighbours: all twelve channels are zero
    assert np.isfinite(with_nb[0].nb_folded[:, day * 48 : day * 48 + 48]).all()

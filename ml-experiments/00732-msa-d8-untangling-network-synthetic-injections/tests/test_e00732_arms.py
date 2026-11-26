"""Tests of the windows, the triple, and the batches, built from the real simulator's segments."""

import e00729_arms as arms729
import e00729_core as core729
import e00730_arms as arms730
import e00730_masks as masks
import e00732_arms as arms
import e00732_core as core
import e00732_overlays as ov
import numpy as np
import pytest
from world732 import make_bank


@pytest.fixture(scope="module")
def world():
    cal, segs = make_bank(draws_per_world=3)
    overlays = ov.overlay_segments(segments=segs, kind="test")
    stats = arms729.input_stats(cal=cal, train_days=np.arange(len(cal.dates)))
    prepared = arms.prepare(overlays=overlays, holidays=cal.holidays, stats=stats)
    index = ov.build_pair_index(segments=segs)
    return cal, segs, overlays, prepared, index, stats


def test_the_plain_window_equals_experiment_730s_arm_s_window(world) -> None:
    _, _, _, prepared, _, _ = world
    spec730 = arms730.ArmSpec("S", "d")
    for p in prepared[:6]:
        day = int(p.days[3])
        mine, *_ = arms.build_triple(p=p, day=day, spec=arms.eval_spec(), hidden=None, delta_flat=None, eps=None)
        theirs, target, _ = arms730.build_one(prepared=p.base, dense_day=day, spec=spec730)
        np.testing.assert_array_equal(mine, theirs)
        assert arms.NAMES == tuple(arms730.stack_names(spec=spec730))
        np.testing.assert_array_equal(arms.target_window(p=p, day=day)[:6], target)


def test_picks_follow_the_rows_order(world) -> None:
    _, segs, overlays, prepared, _, _ = world
    picks = arms.all_picks(prepared=prepared)
    assert picks == arms729.all_picks(prepared=prepared)  # ty: ignore[invalid-argument-type]
    assert len(picks) == sum(len(p.days) for p in prepared)
    assert all(overlays[prepared[i].source_index].seg is not None for i, _ in picks[:3])


def test_injected_member_differs_only_in_the_reading_and_bsp_channels(world) -> None:
    _, segs, _, prepared, index, _ = world
    spec = arms.eval_spec(inject=True)
    checked = 0
    for p in prepared:
        for day in map(int, p.days):
            sib = ov.fixed_sibling(index=index, segment=p.source_index, day=day)
            if sib is None:
                continue
            delta_flat = ov.delta_flat(index=index, sibling=sib, amount=20.0)
            w, wi, _, t, ti, _, scored = arms.build_triple(p=p, day=day, spec=spec, hidden=None, delta_flat=delta_flat, eps=None)
            changed = {arms.NAMES[i] for i in range(arms.N_CHANNELS) if not np.array_equal(w[i], wi[i])}
            assert changed <= {"r", "d_bsp"}, changed
            assert "r" in changed or scored.max() == 0
            np.testing.assert_allclose(ti[arms.T["x"]], t[arms.T["x"]] - scored, atol=1e-5)
            np.testing.assert_allclose(ti[arms.T["solar"]], t[arms.T["solar"]] + scored, atol=1e-5)
            assert np.array_equal(ti[arms.T["event"]], t[arms.T["event"]])
            # the re-folded reading equals the original wherever delta is zero (the whole window)
            s = arms.window_start(day=day)
            zero = delta_flat[s : s + 96] == 0.0
            np.testing.assert_allclose(wi[arms.ROW["r"]][zero], w[arms.ROW["r"]][zero], rtol=1e-4, atol=1e-4)
            checked += 1
            if checked >= 12:
                return
    assert checked > 0


def test_injected_reading_matches_the_refold_of_the_injected_truth(world) -> None:
    _, _, _, prepared, index, _ = world
    spec = arms.eval_spec(inject=True)
    for p in prepared:
        for day in map(int, p.days):
            sib = ov.fixed_sibling(index=index, segment=p.source_index, day=day)
            if sib is None:
                continue
            delta_flat = ov.delta_flat(index=index, sibling=sib, amount=50.0)
            _, wi, _, t, ti, _, scored = arms.build_triple(p=p, day=day, spec=spec, hidden=None, delta_flat=delta_flat, eps=None)
            ok = ti[arms.T["valid"]] > 0.5
            got = wi[arms.ROW["r"], arms.MID : arms.MID + 48] * arms729.R_SCALE
            np.testing.assert_allclose(got[ok], ti[arms.T["r"]][ok], rtol=1e-4, atol=1e-3)
            assert scored.max() > 0
            return
    pytest.fail("no window had a sibling")


def test_bsp_channel_of_the_injected_member_carries_beta_times_delta(world) -> None:
    _, _, _, prepared, index, _ = world
    spec = arms.eval_spec(inject=True)
    for p in (q for q in prepared if q.has_bsp):
        for day in map(int, p.days):
            sib = ov.fixed_sibling(index=index, segment=p.source_index, day=day)
            if sib is None:
                continue
            delta_flat = ov.delta_flat(index=index, sibling=sib, amount=50.0)
            w, wi, *_ = arms.build_triple(p=p, day=day, spec=spec, hidden=None, delta_flat=delta_flat, eps=None)
            s = arms.window_start(day=day)
            want = -p.beta * np.diff(delta_flat[s - 1 : s + 96]) / arms729.BSP_SCALE
            got = (wi[arms.ROW["d_bsp"]] - w[arms.ROW["d_bsp"]])
            ok = w[arms.ROW["bsp_mask"]] > 0.5
            assert ok.any() and np.abs(want[ok]).max() > 0
            np.testing.assert_allclose(got[ok], want[ok], atol=1e-5)
            return
    pytest.fail("no BSP window with a sibling")


def test_raised_member_differs_only_in_the_clear_sky_index(world) -> None:
    _, _, _, prepared, _, _ = world
    spec = arms.eval_spec(raised=True)
    p = next(q for q in prepared if (q.base.stack[arms.ROW["csi"]] > 0).any())
    day = int(p.days[5])
    w, _, wc, *_ = arms.build_triple(p=p, day=day, spec=spec, hidden=None, delta_flat=None, eps=0.2)
    changed = {arms.NAMES[i] for i in range(arms.N_CHANNELS) if not np.array_equal(w[i], wc[i])}
    assert changed == {"csi"}
    mask = ov.raisable(csi=w[arms.ROW["csi"]], cosz=w[arms.ROW["cosz"]])
    assert mask.any() and (wc[arms.ROW["csi"]][mask] > w[arms.ROW["csi"]][mask]).all()
    np.testing.assert_array_equal(wc[arms.ROW["csi"]][~mask & (w[arms.ROW["cosz"]] <= 0.1)], w[arms.ROW["csi"]][~mask & (w[arms.ROW["cosz"]] <= 0.1)])


def test_the_three_members_share_one_hidden_span_and_the_keeps_follow_it(world) -> None:
    _, _, _, prepared, index, _ = world
    spec = arms.eval_spec(inject=True, raised=True)
    p = prepared[0]
    day = int(p.days[4])
    hidden = np.zeros(96, dtype=bool)
    hidden[30:50] = True
    sib = ov.fixed_sibling(index=index, segment=p.source_index, day=day)
    delta = None if sib is None else ov.delta_flat(index=index, sibling=sib, amount=20.0)
    w, wi, wc, t, _, keeps, _ = arms.build_triple(p=p, day=day, spec=spec, hidden=hidden, delta_flat=delta, eps=0.2)
    for window in (w, wi, wc):
        assert (window[arms.ROW["r"]][hidden] == 0).all() and (window[arms.ROW["r_mask"]][hidden] == 0).all()
    scored_hidden = hidden[arms.MID : arms.MID + 48]
    valid = t[arms.T["valid"]] > 0.5
    for term in ("sign", "step", "size", "injection"):
        assert (keeps[arms.K[term]][scored_hidden] == 0).all()
    np.testing.assert_array_equal(keeps[arms.K["solar"]], valid.astype(np.float32))
    assert (keeps[arms.K["size"]] <= (t[arms.T["event"]] > 0.5)).all()
    night = w[arms.ROW["cosz"], arms.MID : arms.MID + 48] <= 0
    np.testing.assert_array_equal(keeps[arms.K["night"]] > 0, valid & night)


def test_events_and_sizes_are_aligned_with_the_transfers(world) -> None:
    _, _, overlays, prepared, _, _ = world
    seen = 0
    for p in prepared:
        o = overlays[p.source_index]
        for tr in o.transfers:
            for hh, sign in ((tr.start_half_hour, 1.0), (tr.end_half_hour, -1.0)):
                day = hh // 48
                if day in p.days:
                    t = arms.target_window(p=p, day=int(day))
                    assert t[arms.T["event"], hh % 48] == 1.0
                    assert np.sign(t[arms.T["size"], hh % 48]) == np.sign(sign * tr.size)
                    seen += 1
    assert seen > 0


def test_no_input_equals_a_target_and_a_planted_leak_is_caught(world) -> None:
    _, _, _, prepared, index, _ = world
    spec = arms.eval_spec(inject=True, raised=True)
    rng = np.random.default_rng(0)
    batch = arms.make_batch(prepared=prepared, picks=arms.sample_picks(prepared=prepared, n=32, rng=rng), spec=spec, rng=rng, index=index)
    for i in range(32):
        for window, target in ((batch.x[i], batch.t[i]), (batch.xi[i], batch.ti[i])):
            # the solar row is left out: the invented calendar's sky is a pure clear-sky shape, so the
            # clear-sky channels equal it. The real-file smoke test checks the solar row on real skies.
            target = target.copy()
            target[arms.T["solar"]] = 0.0
            assert not arms.input_equals_target(window=window, targets=target)
    leak = batch.x[0].copy()
    leak[3] = np.concatenate([np.zeros(arms.MID), batch.t[0][arms.T["event"]], np.zeros(arms.MID)])
    leak[3, arms.MID] = 1.0
    t = batch.t[0].copy()
    t[arms.T["event"]] = leak[3, arms.MID : arms.MID + 48]
    t[arms.T["event"], 5] = 1.0
    leak[3, arms.MID + 5] = 1.0
    assert arms.input_equals_target(window=leak, targets=t)
    leak2 = batch.x[0].copy()
    leak2[3, arms.MID : arms.MID + 48] = batch.t[0][arms.T["size"]] + np.arange(48) * 1e-9
    leak2[3, arms.MID : arms.MID + 48] = batch.t[0][arms.T["size"]]
    if np.ptp(batch.t[0][arms.T["size"]]) > 0:
        assert arms.input_equals_target(window=leak2, targets=batch.t[0])


def one_solar_world(world):
    """Return the bank cut to one solar-world segment per month, so that segment has no sibling."""
    cal, segs, *_ , stats = world
    kept, seen = [], set()
    for s in segs:
        if s.world in ov.SOLAR_WORLDS:
            if (s.month, "solar") in seen:
                continue
            seen.add((s.month, "solar"))
        kept.append(s)
    overlays = ov.overlay_segments(segments=kept, kind="test")
    return kept, arms.prepare(overlays=overlays, holidays=cal.holidays, stats=stats), ov.build_pair_index(segments=kept)


def test_training_batch_masks_half_the_windows_and_counts_unpaired_windows(world) -> None:
    _, segs, _, prepared, index, _ = world
    spec = arms.eval_spec(inject=True, raised=True)
    rng = np.random.default_rng(1)
    picks = arms.sample_picks(prepared=prepared, n=400, rng=rng)
    batch = arms.make_batch(prepared=prepared, picks=picks, spec=spec, rng=rng, index=index)
    plain = arms.make_eval_batch(prepared=prepared, picks=picks, spec=arms.eval_spec())
    masked = (batch.x[:, arms.ROW["r_mask"]] != plain.x[:, arms.ROW["r_mask"]]).any(axis=1).mean()
    assert 0.35 < masked < 0.65
    assert batch.pair.mean() == 1.0  # every window of this invented world has a sibling
    kept, prep, idx = one_solar_world(world)
    rng = np.random.default_rng(1)
    picks = arms.sample_picks(prepared=prep, n=400, rng=rng)
    batch = arms.make_batch(prepared=prep, picks=picks, spec=spec, rng=rng, index=idx)
    assert 0 < batch.pair.mean() < 1  # the lone solar segment of each month has no sibling
    no_pair = batch.pair == 0
    np.testing.assert_array_equal(batch.xi[no_pair], batch.x[no_pair])
    assert (batch.delta[no_pair] == 0).all() and (batch.amount[no_pair] == 0).all()
    paired = batch.pair == 1
    assert (batch.amount[paired] >= core.INJECTION_POINTS[0] - 1e-6).all() and (batch.amount[paired] <= core.INJECTION_POINTS[1] + 1e-6).all()
    assert batch.delta[paired].max() > 0
    rng = np.random.default_rng(1)
    arms.sample_picks(prepared=prep, n=400, rng=rng)  # the same stream: picks, then the batch
    again = arms.make_batch(prepared=prep, picks=picks, spec=spec, rng=rng, index=idx)
    np.testing.assert_array_equal(again.x, batch.x)
    assert batch.x.shape[1:] == (arms.N_CHANNELS, 96)


def test_sign_arm_batches_carry_no_extra_members(world) -> None:
    _, _, _, prepared, _, _ = world
    spec = core.ARMS["S+"]
    rng = np.random.default_rng(2)
    batch = arms.make_batch(prepared=prepared, picks=arms.sample_picks(prepared=prepared, n=8, rng=rng), spec=spec, rng=rng, index=None)
    assert batch.xi is None and batch.xc is None and batch.ti is None


def test_eval_batch_has_the_fixed_pair_and_skips_windows_without_a_sibling(world) -> None:
    _, _, _, prepared, index, _ = world
    spec = arms.eval_spec(inject=True)
    kept, prep, idx = one_solar_world(world)
    picks = arms.all_picks(prepared=prep)
    batch = arms.make_eval_batch(prepared=prep, picks=picks, spec=spec, index=idx, amount=20.0)
    assert 0 < batch.pair.sum() < len(picks)
    assert (batch.amount[batch.pair == 1] == 20.0).all() and (batch.amount[batch.pair == 0] == 0.0).all()
    plain = arms.make_eval_batch(prepared=prep, picks=picks, spec=arms.eval_spec())
    np.testing.assert_array_equal(plain.x, batch.x)  # an evaluation window is never masked
    assert masks.SCORED_START == arms.MID and core729.WINDOW_STEPS == arms.STEPS

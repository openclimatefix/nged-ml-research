"""Tests of the transfer generator, the re-fold, the pairing, and the raised sky, on the real simulator."""

import dataclasses

import e00732_core as core
import e00732_overlays as ov
import numpy as np
import pytest
from world732 import make_bank


@pytest.fixture(scope="module")
def bank():
    return make_bank(draws_per_world=3)


def test_reactive_floor_is_recovered_exactly(bank) -> None:
    _, segs = bank
    for seg in segs[:8]:
        q = ov.reactive_floor(seg=seg)
        on = np.isfinite(seg.r) & np.isfinite(seg.x)
        np.testing.assert_allclose(np.sqrt(seg.x[on] ** 2 + q[on] ** 2), seg.r[on], rtol=1e-4, atol=1e-4)


def test_refold_known_answer() -> None:
    np.testing.assert_allclose(ov.refold(x=np.array([3.0, -4.0, 0.0]), q=np.array([4.0, 3.0, 2.0])), [5.0, 5.0, 2.0], rtol=1e-6)


def test_transfers_change_the_truth_by_the_drawn_size_inside_and_partially_at_an_edge(bank) -> None:
    _, segs = bank
    seg = segs[0]
    tr = ov.Transfer(start_tick=48 * 6 + 3, ticks=12 * 6 * 3 + 2, size=7.5, inside_bsp_a=True)  # 3 ticks into a half-hour
    out = ov.apply_transfers(seg=seg, transfers=[tr])
    d = (out.seg.x - seg.x).reshape(-1)
    inside = np.arange(tr.start_half_hour + 1, tr.end_half_hour)
    np.testing.assert_allclose(d[inside], 7.5, atol=1e-5)
    assert d[tr.start_half_hour] == pytest.approx(7.5 * 0.5, abs=1e-5)  # 3 of 6 ticks carry the load
    assert 0 < d[tr.end_half_hour] < 7.5
    assert out.event.sum() == 2 and out.size.reshape(-1)[tr.start_half_hour] == pytest.approx(7.5)
    assert out.size.reshape(-1)[tr.end_half_hour] == pytest.approx(-7.5)
    assert np.nansum(np.abs(d[:48])) == 0.0  # the padding day carries no transfer


def test_overlay_never_edits_the_stored_segment(bank) -> None:
    _, segs = bank
    seg = segs[3]
    before = [a.copy() for a in (seg.x, seg.r, seg.base, seg.bsp)]
    ov.apply_transfers(seg=seg, transfers=[ov.Transfer(48 * 6 * 2, 120, -9.0, False)])
    for b, a in zip(before, (seg.x, seg.r, seg.base, seg.bsp), strict=True):
        np.testing.assert_array_equal(np.nan_to_num(b, nan=-1), np.nan_to_num(a, nan=-1))


def test_refold_after_a_transfer_keeps_the_reactive_floor(bank) -> None:
    _, segs = bank
    seg = segs[5]
    out = ov.apply_transfers(seg=seg, transfers=[ov.Transfer(48 * 6 * 2, 12 * 6 * 4, -80.0, True)])
    on = np.isfinite(out.seg.r)
    np.testing.assert_array_equal(on, np.isfinite(seg.r))
    np.testing.assert_allclose(out.seg.r[on] ** 2 - out.seg.x[on] ** 2, out.q[on] ** 2, rtol=1e-3, atol=1e-3)
    assert (out.seg.r[on] >= out.q[on] - 1e-4).all()
    assert (np.sign(out.seg.x[on]) != np.sign(seg.x[on])).any()  # a removal of load flips a sign somewhere


def test_bsp_unchanged_inside_bsp_a_and_moves_by_beta_otherwise(bank) -> None:
    _, segs = bank
    seg = next(s for s in segs if s.has_bsp)
    inside = ov.apply_transfers(seg=seg, transfers=[ov.Transfer(48 * 6 * 2, 12 * 6 * 4, 10.0, True)])
    outside = ov.apply_transfers(seg=seg, transfers=[ov.Transfer(48 * 6 * 2, 12 * 6 * 4, 10.0, False)])
    ok = np.isfinite(seg.bsp)
    np.testing.assert_array_equal(inside.seg.bsp[ok], seg.bsp[ok])
    mid = slice(48 * 2 + 4, 48 * 2 + 40)
    np.testing.assert_allclose((outside.seg.bsp - seg.bsp).reshape(-1)[mid][ok.reshape(-1)[mid]], seg.params["beta"] * 10.0, rtol=1e-5)


def test_drawn_transfers_have_both_edges_on_scored_half_hours_and_never_overlap(bank) -> None:
    _, segs = bank
    for index, seg in enumerate(segs):
        rng = ov.overlay_rng(kind="test", seed=None, index=index)
        transfers, failed = ov.draw_transfers(seg=seg, rate=core.TRANSFER_RATE_FIXED, rng=rng)
        scored = ov.scored_half_hours(seg=seg)
        assert len(transfers) + failed == int(round(core.TRANSFER_RATE_FIXED * seg.n_dense / 30.0))
        for tr in transfers:
            assert scored[tr.start_half_hour] and scored[tr.end_half_hour]
            assert core.TRANSFER_HOURS[0] * 12 - 1 <= tr.ticks <= core.TRANSFER_HOURS[1] * 12 + 1
            assert core.TRANSFER_POINTS[0] - 1e-9 <= abs(tr.size) <= core.TRANSFER_POINTS[1] + 1e-9
        for a, b in zip(sorted(transfers, key=lambda t: t.start_tick), sorted(transfers, key=lambda t: t.start_tick)[1:], strict=False):
            assert a.end_tick <= b.start_tick


def test_overlays_are_deterministic_and_seeds_differ(bank) -> None:
    _, segs = bank
    a = ov.overlay_segments(segments=segs[:6], kind="train", seed=0)
    b = ov.overlay_segments(segments=segs[:6], kind="train", seed=0)
    c = ov.overlay_segments(segments=segs[:6], kind="train", seed=1)
    assert ov.overlay_digest(overlays=a) == ov.overlay_digest(overlays=b)
    assert ov.overlay_digest(overlays=a) != ov.overlay_digest(overlays=c)
    # one segment's overlay depends on its position alone, not on the segments before it
    solo = ov.overlay_segments(segments=segs[:6], kind="train", seed=0)[4]
    assert ov.overlay_digest(overlays=[solo]) == ov.overlay_digest(overlays=[a[4]])
    assert ov.overlay_seed(kind="train", seed=2) == 732_102 and ov.overlay_seed(kind="test", seed=None) == 732_000


def test_test_overlay_has_exactly_the_registered_rate(bank) -> None:
    _, segs = bank
    out = ov.overlay_segments(segments=segs, kind="test")
    assert all(o.rate == core.TRANSFER_RATE_FIXED for o in out)
    assert sum(len(o.transfers) for o in out) > 0


def test_component_checks_pass_and_an_empty_overlay_fails(bank) -> None:
    _, segs = bank
    out = ov.overlay_segments(segments=segs, kind="test")
    checks = ov.transfer_checks(overlays=out, originals=segs)
    for name, share in checks.items():
        assert ov.share_passes(share=share), (name, share)
    empty = [ov.apply_transfers(seg=s, transfers=[]) for s in segs]
    for share in ov.transfer_checks(overlays=empty, originals=segs).values():
        assert not ov.share_passes(share=share)  # an empty counter fails


def test_component_check_fails_when_the_bsp_is_not_moved(bank) -> None:
    _, segs = bank
    out = ov.overlay_segments(segments=segs, kind="test")
    broken = [dataclasses.replace(o, seg=dataclasses.replace(o.seg, bsp=s.bsp.copy())) for o, s in zip(out, segs, strict=True)]
    checks = ov.transfer_checks(overlays=broken, originals=segs)
    assert not ov.share_passes(share=checks["bsp_outside_moves"])
    assert ov.share_passes(share=checks["bsp_inside_unchanged"])


def test_component_check_fails_when_the_truth_is_not_shifted(bank) -> None:
    _, segs = bank
    out = ov.overlay_segments(segments=segs, kind="test")
    broken = [dataclasses.replace(o, seg=dataclasses.replace(o.seg, x=s.x.copy())) for o, s in zip(out, segs, strict=True)]
    assert not ov.share_passes(share=ov.transfer_checks(overlays=broken, originals=segs)["transfer_exact"])


def test_pair_index_uses_same_month_siblings_with_output_and_never_self(bank) -> None:
    _, segs = bank
    index = ov.build_pair_index(segments=segs)
    assert index.siblings
    for (i, d), sib in index.siblings.items():
        for j in sib:
            assert j != i and segs[j].month == segs[i].month and segs[j].world in ov.SOLAR_WORLDS
            assert index.units[int(j)].reshape(-1, 48)[d].max() > 0.0
    s2 = ov.build_pair_index(segments=segs, sources=("S2",))
    for sib in s2.siblings.values():
        assert all(segs[int(j)].source == "S2" for j in sib)
    assert len(s2.siblings) <= len(index.siblings)


def test_pair_index_counts_unpaired_windows_per_month(bank) -> None:
    _, segs = bank
    none_only = [s for s in segs if s.world == "none"]
    index = ov.build_pair_index(segments=none_only)
    assert not index.siblings
    assert all(v == 1.0 for v in index.share_unpaired().values()) and index.share_unpaired()
    assert ov.unpaired_problems(share_unpaired=index.share_unpaired())
    full = ov.build_pair_index(segments=segs)
    assert max(full.share_unpaired().values()) <= 1.0


def test_fixed_sibling_depends_on_the_window_alone(bank) -> None:
    _, segs = bank
    index = ov.build_pair_index(segments=segs, sources=("S2",))
    keys = sorted(index.siblings)[:20]
    first = [ov.fixed_sibling(index=index, segment=i, day=d) for i, d in keys]
    second = [ov.fixed_sibling(index=index, segment=i, day=d) for i, d in reversed(keys)][::-1]
    assert first == second
    assert ov.fixed_sibling(index=index, segment=0, day=0) is None  # day 0 is padding, never a window


def test_injection_known_answer(bank) -> None:
    _, segs = bank
    index = ov.build_pair_index(segments=segs)
    (i, d), sib = next(iter(sorted(index.siblings.items())))
    seg = segs[i]
    delta = ov.delta_flat(index=index, sibling=int(sib[0]), amount=20.0).reshape(seg.x.shape)
    assert delta.max() > 0 and delta.max() <= 20.0 * 1.3
    q = ov.reactive_floor(seg=seg)
    x_new, r_new, bsp_new = ov.inject(x=seg.x, q=q, delta=delta, bsp=seg.bsp, beta=seg.params["beta"])
    on = np.isfinite(seg.x)
    np.testing.assert_allclose(x_new[on], (seg.x - delta)[on], atol=1e-5)
    np.testing.assert_allclose(bsp_new[np.isfinite(seg.bsp)], (seg.bsp - seg.params["beta"] * delta)[np.isfinite(seg.bsp)], atol=1e-4)
    same = on & (delta == 0.0)
    np.testing.assert_allclose(r_new[same], seg.r[same], rtol=1e-4, atol=1e-4)
    assert (r_new[on & (delta > 0)] != seg.r[on & (delta > 0)]).any()


def test_pair_checks_pass_on_the_real_simulator_and_fail_without_siblings(bank) -> None:
    _, segs = bank
    index = ov.build_pair_index(segments=segs)
    checks = ov.pair_checks(segments=segs, index=index, amount=20.0, n_pairs=50)
    for name, share in checks.items():
        assert ov.share_passes(share=share), (name, share)
    empty = ov.pair_checks(segments=segs, index=ov.build_pair_index(segments=[s for s in segs if s.world == "none"]), amount=20.0)
    assert not any(ov.share_passes(share=s) for s in empty.values())


def test_raised_sky_changes_raisable_half_hours_only() -> None:
    csi = np.array([0.0, 0.5, 1.0, 1.3, 0.8, 0.6])
    cosz = np.array([0.0, 0.5, 0.9, 0.9, 0.05, 0.0])
    new = ov.raise_sky(csi=csi, cosz=cosz, eps=0.2)
    assert new[0] == 0.0 and new[4] == 0.8 and new[5] == 0.6  # night, dawn, and a zero index stay
    assert new[1] == pytest.approx(0.6) and new[2] == pytest.approx(1.2)
    assert new[3] == pytest.approx(1.5)  # clipped at 1.5
    mask = ov.raisable(csi=csi, cosz=cosz)
    assert mask.tolist() == [False, True, True, False, False, False]
    assert (new[mask] > csi[mask]).all()


def test_raised_check_on_the_simulated_segments(bank) -> None:
    _, segs = bank
    share = ov.raised_checks(segments=segs)
    assert ov.share_passes(share=share)


def test_reachability_counts(bank) -> None:
    _, segs = bank
    out = ov.overlay_segments(segments=segs, kind="test")
    r = ov.reachability(overlays=out, index=None, n_windows_by_amount={5.0: 10, 20.0: 5000})
    assert r["windows_at_5"][2] is False and r["windows_at_20"][2] is True
    assert r["transfer_steps_5_points"][0] >= 0

"""Tests of the transfer generator, the mirrors, the pairing, and the raised sky."""

import dataclasses
import itertools
from pathlib import Path
from typing import Any

import e00729_sim as sim
import e00732_core as core
import e00732_neighbours as nb
import e00732_overlays as ov
import numpy as np
import pytest
from world732 import make_bank, make_neighbours

Bank = tuple[list, list[sim.Segment], dict[str, np.ndarray]]
Overlays = list[ov.TransferOverlay]


@pytest.fixture(scope="module")
def bank() -> Bank:
    cal, segs = make_bank(draws_per_world=3)
    return cal, segs, make_neighbours(cal=cal)


@pytest.fixture(scope="module")
def test_overlays(bank: Bank) -> Overlays:
    _, segs, series = bank
    return ov.overlay_segments(segments=segs, kind="test", neighbours=series)


def _span(tr: ov.Transfer) -> slice:
    return slice(tr.start_half_hour, tr.end_half_hour + 1)


def test_reactive_floor_is_recovered_exactly(bank: Bank) -> None:
    _, segs, _ = bank
    for seg in segs[:8]:
        q = ov.reactive_floor(seg=seg)
        on = np.isfinite(seg.r) & np.isfinite(seg.x)
        np.testing.assert_allclose(
            np.sqrt(seg.x[on] ** 2 + q[on] ** 2), seg.r[on], rtol=1e-4, atol=1e-4
        )


def test_refold_known_answer() -> None:
    np.testing.assert_allclose(
        ov.refold(x=np.array([3.0, -4.0, 0.0]), q=np.array([4.0, 3.0, 2.0])),
        [5.0, 5.0, 2.0],
        rtol=1e-6,
    )


def test_transfers_change_the_truth_by_the_drawn_size_inside_and_partially_at_an_edge(
    bank: Bank,
) -> None:
    _, segs, _ = bank
    seg = segs[0]
    tr = ov.Transfer(
        start_tick=48 * 6 + 3, ticks=12 * 6 * 3 + 2, size=7.5
    )  # 3 ticks into a half-hour
    out = ov.apply_transfers(seg=seg, transfers=[tr])
    d = (out.seg.x - seg.x).reshape(-1)
    inside = np.arange(tr.start_half_hour + 1, tr.end_half_hour)
    np.testing.assert_allclose(d[inside], 7.5, atol=1e-5)
    assert d[tr.start_half_hour] == pytest.approx(
        7.5 * 0.5, abs=1e-5
    )  # 3 of 6 ticks carry the load
    assert 0 < d[tr.end_half_hour] < 7.5
    assert out.event.sum() == 2
    assert out.size.reshape(-1)[tr.start_half_hour] == pytest.approx(7.5)
    assert out.size.reshape(-1)[tr.end_half_hour] == pytest.approx(-7.5)
    assert np.nansum(np.abs(d[:48])) == 0.0  # the padding day carries no transfer


def test_overlay_never_edits_the_stored_segment_or_the_neighbours(bank: Bank) -> None:
    _, segs, series = bank
    seg = segs[3]
    dense = ov.dense_neighbours(seg=seg, series=series)
    before = [a.copy() for a in (seg.x, seg.r, seg.base, seg.bsp, *dense.values())]
    ov.apply_transfers(
        seg=seg,
        transfers=[ov.Transfer(48 * 6 * 2, 120, -9.0, partner="D5")],
        neighbours=dense,
    )
    after = (seg.x, seg.r, seg.base, seg.bsp, *dense.values())
    for b, a in zip(before, after, strict=True):
        np.testing.assert_array_equal(np.nan_to_num(b, nan=-1), np.nan_to_num(a, nan=-1))


def test_refold_after_a_transfer_keeps_the_reactive_floor(bank: Bank) -> None:
    _, segs, _ = bank
    seg = segs[5]
    out = ov.apply_transfers(
        seg=seg, transfers=[ov.Transfer(48 * 6 * 2, 12 * 6 * 4, -80.0, partner="unread")]
    )
    on = np.isfinite(out.seg.r)
    np.testing.assert_array_equal(on, np.isfinite(seg.r))
    np.testing.assert_allclose(
        out.seg.r[on] ** 2 - out.seg.x[on] ** 2, out.q[on] ** 2, rtol=1e-3, atol=1e-3
    )
    assert (out.seg.r[on] >= out.q[on] - 1e-4).all()
    assert (np.sign(out.seg.x[on]) != np.sign(seg.x[on])).any()  # a removal of load flips a sign


def test_bsp_unchanged_inside_bsp_a_and_moves_by_beta_otherwise(bank: Bank) -> None:
    _, segs, _ = bank
    seg = next(s for s in segs if s.has_bsp)
    ok = np.isfinite(seg.bsp)
    for partner in ("D3", "unread"):
        inside = ov.apply_transfers(
            seg=seg, transfers=[ov.Transfer(48 * 6 * 2, 12 * 6 * 4, 10.0, partner=partner)]
        )
        np.testing.assert_array_equal(inside.seg.bsp[ok], seg.bsp[ok])
    outside = ov.apply_transfers(
        seg=seg, transfers=[ov.Transfer(48 * 6 * 2, 12 * 6 * 4, 10.0, partner="other_bsp")]
    )
    mid = slice(48 * 2 + 4, 48 * 2 + 40)
    np.testing.assert_allclose(
        (outside.seg.bsp - seg.bsp).reshape(-1)[mid][ok.reshape(-1)[mid]],
        seg.params["beta"] * 10.0,
        rtol=1e-5,
    )


# --------------------------------------------------------------------------- the mirror


def test_the_mirror_moves_the_partner_by_minus_j_times_the_load_and_nothing_else(
    bank: Bank,
) -> None:
    _, segs, series = bank
    seg = segs[0]
    dense = ov.dense_neighbours(seg=seg, series=series)
    for partner, jitter in (("D3", 1.0), ("D5", 0.8), ("D14", 1.25)):
        tr = ov.Transfer(48 * 6 * 3 + 2, 6 * 20, 12.0, partner=partner, jitter=jitter)
        out = ov.apply_transfers(seg=seg, transfers=[tr], neighbours=dense)
        for name in nb.NEIGHBOURS:
            moved = np.nan_to_num(out.neighbours[name] - out.neighbours_real[name]).reshape(-1)
            if name == partner:
                inside = np.arange(tr.start_half_hour + 1, tr.end_half_hour)
                np.testing.assert_allclose(moved[inside], -jitter * 12.0, atol=1e-4)
                assert np.abs(moved[_span(tr)]).max() > 0
                outside = np.ones(moved.size, dtype=bool)
                outside[_span(tr)] = False
                assert (np.abs(moved[outside]) < 1e-9).all()
            else:
                assert (np.abs(moved) < 1e-9).all()
        # the real series are never edited
        np.testing.assert_allclose(
            np.nan_to_num(out.neighbours_real[partner], nan=-1),
            np.nan_to_num(dense[partner], nan=-1),
            rtol=1e-6,
        )


def test_a_transfer_to_a_partner_that_is_not_read_moves_no_neighbour(bank: Bank) -> None:
    _, segs, series = bank
    dense = ov.dense_neighbours(seg=segs[0], series=series)
    for partner in ("unread", "other_bsp"):
        out = ov.apply_transfers(
            seg=segs[0],
            transfers=[ov.Transfer(48 * 6 * 3, 6 * 20, -9.0, partner=partner)],
            neighbours=dense,
        )
        for name in nb.NEIGHBOURS:
            np.testing.assert_array_equal(
                np.nan_to_num(out.neighbours[name], nan=-1),
                np.nan_to_num(out.neighbours_real[name], nan=-1),
            )


def test_neighbours_are_placed_on_the_dense_axis_through_the_calendar_index(bank: Bank) -> None:
    _, segs, series = bank
    seg = segs[0]
    dense = ov.dense_neighbours(seg=seg, series=series)
    assert dense["D3"].shape == seg.x.shape
    off_base = seg.cal_index < 0
    assert np.isnan(dense["D3"][off_base]).all()  # padding days carry no neighbour reading
    for d in np.flatnonzero(~off_base)[:5]:
        np.testing.assert_array_equal(dense["D5"][d], series["D5"][seg.cal_index[d]])


def test_a_mirror_check_passes_on_a_real_overlay(test_overlays: Overlays, bank: Bank) -> None:
    _, segs, _ = bank
    checks = ov.transfer_checks(overlays=test_overlays, originals=segs)
    assert ov.check_problems(checks=checks) == []
    for name in ("mirror_exact", "mirror_nonzero", "mirror_nowhere_else"):
        assert checks[name].n > 0
        assert checks[name].value == 1.0


def test_the_mirror_check_fails_on_a_silent_mirror(test_overlays: Overlays, bank: Bank) -> None:
    _, segs, _ = bank
    silent = [dataclasses.replace(o, neighbours=dict(o.neighbours_real)) for o in test_overlays]
    checks = ov.transfer_checks(overlays=silent, originals=segs)
    assert not ov.share_passes(share=checks["mirror_exact"], bar=1.0)
    assert not ov.share_passes(share=checks["mirror_nonzero"], bar=1.0)
    assert "mirror_exact" in ov.check_problems(checks=checks)


def test_the_mirror_check_fails_on_a_mirror_at_the_wrong_neighbour(
    test_overlays: Overlays, bank: Bank
) -> None:
    _, segs, _ = bank
    rotate = {"D3": "D5", "D5": "D14", "D14": "D3"}
    moved = [
        dataclasses.replace(o, neighbours={rotate[n]: a for n, a in o.neighbours.items()})
        for o in test_overlays
    ]
    checks = ov.transfer_checks(overlays=moved, originals=segs)
    assert not ov.share_passes(share=checks["mirror_exact"], bar=1.0)
    assert not ov.share_passes(share=checks["mirror_nowhere_else"], bar=1.0)


def test_the_mirror_check_fails_when_a_neighbour_moves_outside_a_transfer(
    test_overlays: Overlays, bank: Bank
) -> None:
    _, segs, _ = bank
    nudged = []
    for o in test_overlays:
        extra = {n: a.copy() for n, a in o.neighbours.items()}
        extra["D3"][5, 5] += 1.0  # a move outside every transfer
        nudged.append(dataclasses.replace(o, neighbours=extra))
    checks = ov.transfer_checks(overlays=nudged, originals=segs)
    assert not ov.share_passes(share=checks["mirror_nowhere_else"], bar=1.0)


def test_the_mirror_check_fails_on_a_wrong_size(test_overlays: Overlays, bank: Bank) -> None:
    _, segs, _ = bank
    wrong = []
    for o in test_overlays:
        scaled = {
            n: o.neighbours_real[n] + 1.5 * (o.neighbours[n] - o.neighbours_real[n])
            for n in nb.NEIGHBOURS
        }
        wrong.append(dataclasses.replace(o, neighbours=scaled))
    assert not ov.share_passes(
        share=ov.transfer_checks(overlays=wrong, originals=segs)["mirror_exact"], bar=1.0
    )


def test_an_empty_overlay_fails_every_mirror_check(bank: Bank) -> None:
    _, segs, series = bank
    empty = ov.plain_overlays(segments=segs, neighbours=series)
    checks = ov.transfer_checks(overlays=empty, originals=segs)
    for name in ("mirror_exact", "mirror_nonzero", "transfer_exact", "shape_floor"):
        assert not ov.share_passes(share=checks[name])  # an empty counter fails


# --------------------------------------------------------------------------- the draw


def test_drawn_transfers_have_both_edges_on_scored_half_hours_and_never_overlap(bank: Bank) -> None:
    _, segs, series = bank
    for index, seg in enumerate(segs):
        rng = ov.overlay_rng(kind="test", seed=None, index=index)
        dense = ov.dense_neighbours(seg=seg, series=series)
        drawn = ov.draw_transfers(seg=seg, rate=core.TRANSFER_RATE_FIXED, rng=rng, neighbours=dense)
        scored = ov.scored_half_hours(seg=seg)
        assert len(drawn.transfers) + drawn.n_failed + drawn.n_dropped == drawn.n_draws
        assert drawn.n_draws == round(core.TRANSFER_RATE_FIXED * seg.n_dense / 30.0)
        for tr in drawn.transfers:
            assert scored[tr.start_half_hour]
            assert scored[tr.end_half_hour]
            assert core.TRANSFER_HOURS[0] * 12 - 1 <= tr.ticks <= core.TRANSFER_HOURS[1] * 12 + 1
            assert core.TRANSFER_POINTS[0] - 1e-9 <= abs(tr.size) <= core.TRANSFER_POINTS[1] + 1e-9
        ordered = sorted(drawn.transfers, key=lambda t: t.start_tick)
        for a, b in itertools.pairwise(ordered):
            assert a.end_tick <= b.start_tick


def test_partner_shape_and_mirror_mixes_follow_the_registered_shares(bank: Bank) -> None:
    _, segs, series = bank
    transfers = []
    for index, seg in enumerate(segs):
        rng = np.random.default_rng([7, index])
        dense = ov.dense_neighbours(seg=seg, series=series)
        transfers += ov.draw_transfers(seg=seg, rate=60.0, rng=rng, neighbours=dense).transfers
    n = len(transfers)
    assert n > 400
    read = sum(t.partner in core.READ_PARTNERS for t in transfers) / n
    unread = sum(t.partner == "unread" for t in transfers) / n
    other = sum(t.partner == "other_bsp" for t in transfers) / n
    assert read == pytest.approx(0.6, abs=0.07)
    assert unread == pytest.approx(0.2, abs=0.06)
    assert other == pytest.approx(0.2, abs=0.06)
    assert {t.partner for t in transfers} == {*core.READ_PARTNERS, "unread", "other_bsp"}
    for shape, share in core.SHAPE_MIX.items():
        assert sum(t.shape == shape for t in transfers) / n == pytest.approx(share, abs=0.07)
    mirrored = [t for t in transfers if t.mirrored]
    jittered = [t for t in mirrored if t.jitter != 1.0]
    assert len(jittered) / len(mirrored) == pytest.approx(0.3, abs=0.07)
    assert all(0.7 <= t.jitter <= 1.3 for t in jittered)
    assert all(t.jitter == 1.0 for t in transfers if not t.mirrored)
    assert sum(t.size > 0 for t in transfers) / n == pytest.approx(0.5, abs=0.07)


def test_every_shape_keeps_the_mean_the_floor_and_the_sign(bank: Bank) -> None:
    _, segs, series = bank
    seen = dict.fromkeys(core.SHAPES, 0)
    for index, seg in enumerate(segs):
        rng = np.random.default_rng([8, index])
        dense = ov.dense_neighbours(seg=seg, series=series)
        for tr in ov.draw_transfers(seg=seg, rate=30.0, rng=rng, neighbours=dense).transfers:
            seen[tr.shape] += 1
            profile = np.asarray(tr.profile)
            used = tr.used_weights() > 0
            weights = tr.used_weights()[used]
            assert (profile[used] >= core.SHAPE_FLOOR - 1e-9).all()  # T never changes sign
            assert (np.sign(tr.mean_levels()[used]) == np.sign(tr.size)).all()
            assert (profile[used] * weights).sum() / weights.sum() == pytest.approx(1.0, abs=1e-4)
            if tr.shape == "constant":
                assert (profile == 1.0).all()
    assert all(v > 0 for v in seen.values())


def test_a_solar_shaped_transfer_dips_when_the_sun_shines(bank: Bank) -> None:
    _, segs, _ = bank
    seg = next(s for s in segs if s.world == "none")
    probe = ov.Transfer(start_tick=48 * 3 * 6, ticks=48 * 6, size=10.0)
    span = slice(probe.start_half_hour, probe.end_half_hour + 1)
    sun = (np.nan_to_num(seg.csi) * np.nan_to_num(seg.clear)).reshape(-1)[span]
    donor = np.full(sun.size, 50.0)
    profile = ov.shape_profile(probe=probe, shape="solar", donor=donor, sun=sun, kappa=0.8)
    assert profile is not None
    used = probe.used_weights() > 0
    assert profile[used].min() >= core.SHAPE_FLOOR - 1e-9
    assert int(np.argmin(profile[used])) == int(np.argmax(sun[used]))
    assert profile[used].max() > 1.0 > profile[used].min()
    assert np.corrcoef(profile[used], sun[used])[0, 1] < -0.99
    # a depth that would push the load through zero is halved until the floor holds
    deep = ov.shape_profile(probe=probe, shape="solar", donor=donor, sun=sun * 1000.0, kappa=1.0)
    assert deep is not None
    assert deep[used].min() >= core.SHAPE_FLOOR - 1e-9


def test_a_demand_shaped_transfer_follows_the_donor_and_a_flat_donor_gives_a_constant() -> None:
    probe = ov.Transfer(start_tick=6 * 4, ticks=6 * 10, size=5.0)
    n = probe.end_half_hour - probe.start_half_hour + 1
    zeros = np.zeros(n)
    donor = np.linspace(20.0, 60.0, n)
    profile = ov.shape_profile(probe=probe, shape="demand", donor=donor, sun=zeros, kappa=0.5)
    assert profile is not None
    assert np.corrcoef(profile, donor)[0, 1] > 0.999
    flat = ov.shape_profile(
        probe=probe, shape="demand", donor=np.full(n, 33.0), sun=zeros, kappa=0.5
    )
    assert flat is not None
    np.testing.assert_allclose(flat, 1.0)
    nothing = ov.shape_profile(
        probe=probe, shape="demand", donor=np.full(n, np.nan), sun=zeros, kappa=0.5
    )
    assert nothing is None


def test_no_donor_gives_more_than_eight_tenths_of_what_it_has(bank: Bank) -> None:
    _, segs, series = bank
    checked = {"d8": 0, "partner": 0}
    for index, seg in enumerate(segs):
        rng = np.random.default_rng([9, index])
        dense = ov.dense_neighbours(seg=seg, series=series)
        base = np.nan_to_num(seg.base.reshape(-1))
        for tr in ov.draw_transfers(seg=seg, rate=60.0, rng=rng, neighbours=dense).transfers:
            used = tr.used_weights() > 0
            levels = np.abs(tr.size) * np.asarray(tr.profile)[used]
            if tr.size < 0:  # D8 loses the load
                assert (levels <= core.DONOR_CAP * base[_span(tr)][used] + 1e-9).all()
                checked["d8"] += 1
            elif tr.mirrored:  # the partner loses j T
                donor = dense[tr.partner].reshape(-1)[_span(tr)][used]
                finite = np.isfinite(donor)
                assert (tr.jitter * levels[finite] <= core.DONOR_CAP * donor[finite] + 1e-9).all()
                checked["partner"] += 1
    assert all(v > 0 for v in checked.values())


def test_a_donor_with_nothing_to_give_drops_the_transfer_and_counts_it(bank: Bank) -> None:
    _, segs, series = bank
    seg = segs[0]
    poor = {
        n: np.where(np.isfinite(a), 0.5, np.nan) for n, a in series.items()
    }  # far under 3 points
    dense = ov.dense_neighbours(seg=seg, series=poor)
    dropped = giving = 0
    for k in range(20):
        rng = np.random.default_rng([10, k])
        drawn = ov.draw_transfers(seg=seg, rate=30.0, rng=rng, neighbours=dense)
        dropped += drawn.n_dropped
        giving += sum(t.mirrored and t.size > 0 for t in drawn.transfers)
    assert dropped > 0
    assert giving == 0  # no read partner gives load when it holds under 3.75 points
    none = ov.draw_transfers(seg=seg, rate=30.0, rng=np.random.default_rng(1), neighbours=None)
    assert all(not t.mirrored for t in none.transfers)  # no neighbour has a reading to mirror
    assert none.n_failed + len(none.transfers) + none.n_dropped == none.n_draws


def test_overlays_are_deterministic_and_seeds_differ(bank: Bank) -> None:
    _, segs, series = bank
    kw: dict[str, Any] = {"segments": segs[:6], "kind": "train", "neighbours": series}
    a = ov.overlay_segments(seed=0, **kw)
    b = ov.overlay_segments(seed=0, **kw)
    c = ov.overlay_segments(seed=1, **kw)
    assert ov.overlay_digest(overlays=a) == ov.overlay_digest(overlays=b)
    assert ov.overlay_digest(overlays=a) != ov.overlay_digest(overlays=c)
    # one segment's overlay depends on its position alone, not on the segments before it
    solo = ov.overlay_segments(
        segments=[segs[4]], kind="train", seed=0, first_index=4, neighbours=series
    )[0]
    assert ov.overlay_digest(overlays=[solo]) == ov.overlay_digest(overlays=[a[4]])
    assert ov.overlay_seed(kind="train", seed=2) == 732_102
    assert ov.overlay_seed(kind="test", seed=None) == 732_000
    assert ov.overlay_seed(kind="validation", seed=None) == 732_050
    # the fold floors come from a stream of their own, and D3 has none
    assert a[0].floors["D3"] == 0.0
    assert 0.0 < a[0].floors["D5"] < 0.3 * 100.0 / 10.379 + 1e-9


def test_the_digest_sees_the_mirror(bank: Bank) -> None:
    _, segs, series = bank
    out = ov.overlay_segments(segments=segs[:4], kind="test", neighbours=series)
    silent = [dataclasses.replace(o, neighbours=dict(o.neighbours_real)) for o in out]
    assert ov.overlay_digest(overlays=out) != ov.overlay_digest(overlays=silent)


def test_test_overlay_has_exactly_the_registered_rate(test_overlays: Overlays) -> None:
    assert all(o.rate == core.TRANSFER_RATE_FIXED for o in test_overlays)
    assert sum(len(o.transfers) for o in test_overlays) > 0


def test_injection_and_raised_sky_overlays_carry_no_transfer(bank: Bank) -> None:
    _, segs, series = bank
    plain = ov.plain_overlays(segments=segs, neighbours=series)
    for o, seg in zip(plain, segs, strict=True):
        assert o.transfers == []
        assert not o.event.any()
        assert not o.t.any()
        assert not o.size.any()
        np.testing.assert_array_equal(np.nan_to_num(o.seg.x, nan=-1), np.nan_to_num(seg.x, nan=-1))
        for n in nb.NEIGHBOURS:
            np.testing.assert_array_equal(
                np.nan_to_num(o.neighbours[n], nan=-1), np.nan_to_num(o.neighbours_real[n], nan=-1)
            )


def test_plain_overlays_without_neighbours_hold_no_neighbour_reading(bank: Bank) -> None:
    _, segs, _ = bank
    out = ov.plain_overlays(segments=segs[:2], neighbours=None)
    assert all(np.isnan(o.neighbours[n]).all() for o in out for n in nb.NEIGHBOURS)


def test_component_checks_pass_and_an_empty_overlay_fails(
    test_overlays: Overlays, bank: Bank
) -> None:
    _, segs, series = bank
    checks = ov.transfer_checks(overlays=test_overlays, originals=segs)
    assert ov.check_problems(checks=checks) == []
    for share in checks.values():
        assert share.n > 0
    empty = ov.plain_overlays(segments=segs, neighbours=series)
    problems = ov.check_problems(checks=ov.transfer_checks(overlays=empty, originals=segs))
    assert {"transfer_exact", "shape_floor", "mirror_exact"} <= set(problems)


def test_component_check_fails_when_the_bsp_is_not_moved(
    test_overlays: Overlays, bank: Bank
) -> None:
    _, segs, _ = bank
    broken = [
        dataclasses.replace(o, seg=dataclasses.replace(o.seg, bsp=s.bsp.copy()))
        for o, s in zip(test_overlays, segs, strict=True)
    ]
    checks = ov.transfer_checks(overlays=broken, originals=segs)
    assert not ov.share_passes(share=checks["bsp_outside_moves"])
    assert ov.share_passes(share=checks["bsp_inside_unchanged"])


def test_component_check_fails_when_the_truth_is_not_shifted(
    test_overlays: Overlays, bank: Bank
) -> None:
    _, segs, _ = bank
    broken = [
        dataclasses.replace(o, seg=dataclasses.replace(o.seg, x=s.x.copy()))
        for o, s in zip(test_overlays, segs, strict=True)
    ]
    assert not ov.share_passes(
        share=ov.transfer_checks(overlays=broken, originals=segs)["transfer_exact"]
    )


def test_the_shape_floor_check_fails_on_a_load_that_changes_sign(
    test_overlays: Overlays, bank: Bank
) -> None:
    _, segs, _ = bank
    bad = []
    for o in test_overlays:
        flipped = []
        for t in o.transfers:
            if t.shape == "solar" and t.profile is not None and len(t.profile) > 2:
                broken = np.asarray(t.profile).copy()
                broken[1] = -0.5
                t = dataclasses.replace(t, profile=broken)  # noqa: PLW2901
            flipped.append(t)
        bad.append(dataclasses.replace(o, transfers=flipped))
    assert not ov.share_passes(
        share=ov.transfer_checks(overlays=bad, originals=segs)["shape_floor"], bar=1.0
    )


def test_the_donor_check_fails_when_too_many_draws_are_dropped(
    test_overlays: Overlays, bank: Bank
) -> None:
    _, segs, _ = bank
    many = [dataclasses.replace(o, n_dropped=3) for o in test_overlays]
    share = ov.transfer_checks(overlays=many, originals=segs)["donor_dropped"]
    assert not ov.share_passes(share=share, bar=ov.CHECK_BARS["donor_dropped"])


def test_shape_counts_list_every_partner_and_shape(test_overlays: Overlays) -> None:
    counts = ov.shape_counts(overlays=test_overlays)
    assert set(counts["shape"]) == set(core.SHAPES)
    assert sum(counts["partner"].values()) == sum(len(o.transfers) for o in test_overlays)


def test_the_training_overlay_is_built_once_and_read_back(tmp_path: Path, bank: Bank) -> None:
    _, segs, series = bank
    path = tmp_path / "overlays" / "train_s0.pkl"
    calls = []

    def build() -> list[ov.TransferOverlay]:
        calls.append(1)
        return ov.overlay_segments(segments=segs[:4], kind="train", seed=0, neighbours=series)

    first = ov.cached_overlays(path=path, build=build)
    second = ov.cached_overlays(path=path, build=build)
    assert len(calls) == 1
    assert ov.overlay_digest(overlays=first) == ov.overlay_digest(overlays=second)
    assert not list(path.parent.glob("*.tmp*"))


# ----------------------------------------------------------------- pairs and the raised sky


def test_pair_index_uses_same_month_siblings_with_output_and_never_self(bank: Bank) -> None:
    _, segs, _ = bank
    index = ov.build_pair_index(segments=segs)
    assert index.siblings
    for (i, d), sib in index.siblings.items():
        for j in sib:
            assert j != i
            assert segs[j].month == segs[i].month
            assert segs[j].world in ov.SOLAR_WORLDS
            assert index.units[int(j)].reshape(-1, 48)[d].max() > 0.0
    s2 = ov.build_pair_index(segments=segs, sources=("S2",))
    for sib in s2.siblings.values():
        assert all(segs[int(j)].source == "S2" for j in sib)
    assert len(s2.siblings) <= len(index.siblings)


def test_pair_index_counts_unpaired_windows_per_month(bank: Bank) -> None:
    _, segs, _ = bank
    none_only = [s for s in segs if s.world == "none"]
    index = ov.build_pair_index(segments=none_only)
    assert not index.siblings
    assert all(v == 1.0 for v in index.share_unpaired().values())
    assert index.share_unpaired()
    assert ov.unpaired_problems(share_unpaired=index.share_unpaired())
    full = ov.build_pair_index(segments=segs)
    assert max(full.share_unpaired().values()) <= 1.0
    assert ov.unpaired_problems(share_unpaired={"a": 0.05, "b": 0.0}) == []
    assert ov.unpaired_problems(share_unpaired={"a": 0.051, "b": 0.0}) == ["month 0"]


def test_fixed_sibling_depends_on_the_window_alone(bank: Bank) -> None:
    _, segs, _ = bank
    index = ov.build_pair_index(segments=segs, sources=("S2",))
    keys = sorted(index.siblings)[:20]
    first = [ov.fixed_sibling(index=index, segment=i, day=d) for i, d in keys]
    second = [ov.fixed_sibling(index=index, segment=i, day=d) for i, d in reversed(keys)][::-1]
    assert first == second
    assert ov.fixed_sibling(index=index, segment=0, day=0) is None  # day 0 is padding


def test_vectorised_pairing_matches_one_window_at_a_time(bank: Bank) -> None:
    _, segs, _ = bank
    index = ov.build_pair_index(segments=segs)
    keys = sorted(index.siblings)
    picks = [keys[k] for k in np.random.default_rng(0).choice(len(keys), 40, replace=False)]
    picks += [(0, 0)]  # a window with no sibling (padding day)
    seg_i = [i for i, _ in picks]
    days = [d for _, d in picks]
    draw = ov.draw_pairs(index=index, segments=seg_i, days=days, rng=np.random.default_rng(3))
    assert draw.delta.shape == (len(picks), ov.WINDOW_DELTA_LEN)
    assert draw.sibling[-1] == -1
    assert draw.amount[-1] == 0.0
    assert not draw.delta[-1].any()
    for k, (i, d) in enumerate(picks[:-1]):
        sibling, amount = int(draw.sibling[k]), float(draw.amount[k])
        assert sibling in {int(j) for j in index.siblings[i, d]}
        assert core.INJECTION_POINTS[0] - 1e-6 <= amount <= core.INJECTION_POINTS[1] + 1e-6
        one = ov.window_delta(index=index, sibling=sibling, amount=amount, day=d)
        np.testing.assert_allclose(draw.delta[k], one, rtol=1e-6)
        flat = ov.delta_flat(index=index, sibling=sibling, amount=amount)
        s = d * 48 - 24
        np.testing.assert_allclose(draw.delta[k], flat[s - 1 : s + 96], rtol=1e-6)
    again = ov.draw_pairs(index=index, segments=seg_i, days=days, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(draw.delta, again.delta)


def test_the_pairing_stream_does_not_depend_on_which_windows_have_a_sibling(bank: Bank) -> None:
    _, segs, _ = bank
    index = ov.build_pair_index(segments=segs)
    i, d = min(index.siblings)
    a = ov.draw_pairs(index=index, segments=[i, 0], days=[d, 0], rng=np.random.default_rng(5))
    b = ov.draw_pairs(index=index, segments=[i, i], days=[d, d], rng=np.random.default_rng(5))
    assert a.sibling[0] == b.sibling[0]
    assert a.amount[0] == b.amount[0]


def test_injection_known_answer(bank: Bank) -> None:
    _, segs, _ = bank
    index = ov.build_pair_index(segments=segs)
    (i, _d), sib = next(iter(sorted(index.siblings.items())))
    seg = segs[i]
    delta = ov.delta_flat(index=index, sibling=int(sib[0]), amount=20.0).reshape(seg.x.shape)
    assert delta.max() > 0
    assert delta.max() <= 20.0 * 1.3
    q = ov.reactive_floor(seg=seg)
    x_new, r_new, bsp_new = ov.inject(
        x=seg.x, q=q, delta=delta, bsp=seg.bsp, beta=seg.params["beta"]
    )
    on = np.isfinite(seg.x)
    np.testing.assert_allclose(x_new[on], (seg.x - delta)[on], atol=1e-5)
    np.testing.assert_allclose(
        bsp_new[np.isfinite(seg.bsp)],
        (seg.bsp - seg.params["beta"] * delta)[np.isfinite(seg.bsp)],
        atol=1e-4,
    )
    same = on & (delta == 0.0)
    np.testing.assert_allclose(r_new[same], seg.r[same], rtol=1e-4, atol=1e-4)
    assert (r_new[on & (delta > 0)] != seg.r[on & (delta > 0)]).any()


def test_pair_checks_pass_on_the_real_simulator_and_fail_without_siblings(bank: Bank) -> None:
    _, segs, _ = bank
    index = ov.build_pair_index(segments=segs)
    checks = ov.pair_checks(segments=segs, index=index, amount=20.0, n_pairs=50)
    for name, share in checks.items():
        assert ov.share_passes(share=share), (name, share)
    empty = ov.pair_checks(
        segments=segs,
        index=ov.build_pair_index(segments=[s for s in segs if s.world == "none"]),
        amount=20.0,
    )
    assert not any(ov.share_passes(share=s) for s in empty.values())


def test_raised_sky_changes_raisable_half_hours_only() -> None:
    csi = np.array([0.0, 0.5, 1.0, 1.3, 0.8, 0.6])
    cosz = np.array([0.0, 0.5, 0.9, 0.9, 0.05, 0.0])
    new = ov.raise_sky(csi=csi, cosz=cosz, eps=0.2)
    assert new[0] == 0.0
    assert new[4] == 0.8
    assert new[5] == 0.6  # night, dawn, and a zero index stay
    assert new[1] == pytest.approx(0.6)
    assert new[2] == pytest.approx(1.2)
    assert new[3] == pytest.approx(1.5)  # clipped at 1.5
    mask = ov.raisable(csi=csi, cosz=cosz)
    assert mask.tolist() == [False, True, True, False, False, False]
    assert (new[mask] > csi[mask]).all()


def test_raised_check_on_the_simulated_segments(bank: Bank) -> None:
    _, segs, _ = bank
    assert ov.share_passes(share=ov.raised_checks(segments=segs))


def test_reachability_counts(test_overlays: Overlays) -> None:
    r = ov.reachability(
        overlays=test_overlays, index=None, n_windows_by_amount={5.0: 10, 20.0: 5000}
    )
    assert r["windows_at_5"][2] is False
    assert r["windows_at_20"][2] is True
    assert r["transfer_steps_5_points"][0] >= 0

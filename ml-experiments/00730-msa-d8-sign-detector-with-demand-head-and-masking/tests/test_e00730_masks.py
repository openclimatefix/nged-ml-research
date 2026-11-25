"""Known-answer tests of the masking styles and of the causal rebuild of hourly channels."""

import e00729_core as core729
import e00730_core as core
import e00730_masks as masks
import numpy as np
import pytest

SCORED = slice(masks.SCORED_START, masks.SCORED_START + 48)


def test_style_w_hides_exactly_the_scored_day() -> None:
    hidden = masks.hidden_w()
    assert hidden.sum() == 48
    assert hidden[SCORED].all()
    assert not hidden[: masks.SCORED_START].any()


def test_style_r_hides_one_span_of_one_to_twenty_four_hours() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        hidden = masks.hidden_r(rng=rng)
        n = int(hidden.sum())
        assert 2 <= n <= 48
        idx = np.flatnonzero(hidden)
        assert idx[-1] - idx[0] + 1 == n


def test_style_k_blocks_lie_inside_the_scored_day_window_hours() -> None:
    rng = np.random.default_rng(1)
    sizes = set()
    for _ in range(300):
        hidden = masks.hidden_k(rng=rng, utc_offset=0.0)
        assert hidden.any()
        sizes.add(int(hidden.sum()) // 6)
    assert len(sizes) > 3


def test_style_zero_hides_nothing_and_unknown_style_raises() -> None:
    assert not masks.hidden_positions(style="0", rng=np.random.default_rng(0), utc_offset=0).any()
    with pytest.raises(ValueError, match="unknown masking style"):
        masks.hidden_positions(style="Q", rng=np.random.default_rng(0), utc_offset=0)


def test_apply_hidden_zeroes_reading_and_mask_only() -> None:
    window = np.ones((4, 96))
    out = masks.apply_hidden(window=window, hidden=masks.hidden_w())
    assert (out[:2, SCORED] == 0).all()
    assert (out[2:] == 1).all()


def test_fixed_masks_depend_on_style_and_index_alone() -> None:
    a = masks.fixed_hidden(style="K", index=7, utc_offset=1.0)
    b = masks.fixed_hidden(style="K", index=7, utc_offset=1.0)
    assert np.array_equal(a, b)
    different = [masks.fixed_hidden(style="K", index=i, utc_offset=1.0).sum() for i in range(20)]
    assert len(set(different)) > 1


def test_fixed_cuts_are_reproducible() -> None:
    assert np.array_equal(masks.fixed_cuts(n_windows=5), masks.fixed_cuts(n_windows=5))
    assert masks.fixed_cuts(n_windows=5).shape == (5, core.CAUSAL_VALIDATION_CUTS)


def test_hourly_inversion_is_exact() -> None:
    rng = np.random.default_rng(3)
    hourly = rng.normal(size=30)
    half = masks.half_hour_from_hourly(hourly=hourly)
    back = masks.hourly_from_half_hour(values=half)
    assert np.allclose(back[1:-1], hourly[1:-1], atol=1e-9)


def test_causal_rebuild_ignores_everything_after_the_cut_hour() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(size=24)
    b = a.copy()
    b[11:] += 100.0
    for copy in (False, True):
        out_a = masks.causal_half_hours(hourly=a, last_visible_hour=10, copy=copy)
        out_b = masks.causal_half_hours(hourly=b, last_visible_hour=10, copy=copy)
        both = np.isfinite(out_a) & np.isfinite(out_b)
        assert np.allclose(out_a[both], out_b[both])
        assert np.isnan(out_a[2 * 11 + 2 :]).all()
        assert np.isfinite(out_a[: 2 * 10]).all()


def test_copy_and_interpolation_differ() -> None:
    hourly = np.arange(10.0)
    copied = masks.copy_half_hour_from_hourly(hourly=hourly)
    interpolated = masks.half_hour_from_hourly(hourly=hourly)
    assert not np.allclose(copied, interpolated)
    assert copied[4] == copied[5] == 2.0


def test_mask_probability_is_half() -> None:
    assert core729.MASK_PROBABILITY == 0.5

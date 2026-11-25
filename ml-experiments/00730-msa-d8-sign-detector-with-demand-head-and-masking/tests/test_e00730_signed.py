"""Known-answer tests of the signed read-outs, the by-window masks, and the new result figures."""

from types import SimpleNamespace

import e00729_baselines as base
import e00729_core as core729
import e00730_core as core
import e00730_figures as figures
import e00730_masks as masks
import e00730_real as real
import e00730_signed as signed
import numpy as np
import pytest


def test_a_window_gets_the_same_fixed_mask_whatever_the_pick_order() -> None:
    picks = [(0, 3), (0, 4), (2, 3), (5, 9)]
    offsets = [[0.0] * 12, [0.0] * 12, [1.0] * 12, [0.0] * 12, [0.0] * 12, [1.0] * 12]
    for style in core.STYLES:
        forward = masks.fixed_hidden_for_picks(style=style, picks=picks, offsets=offsets)
        backward = masks.fixed_hidden_for_picks(style=style, picks=picks[::-1], offsets=offsets)
        assert np.array_equal(forward, backward[::-1])
        single = masks.fixed_hidden_for_picks(style=style, picks=[picks[2]], offsets=offsets)
        assert np.array_equal(single[0], forward[2])


def test_window_ids_differ_between_windows_and_reject_a_day_out_of_range() -> None:
    ids = {masks.window_id(segment=s, day=d) for s in range(5) for d in range(50)}
    assert len(ids) == 250
    with pytest.raises(ValueError, match="outside"):
        masks.window_id(segment=0, day=masks.WINDOW_ID_STRIDE)


def invented_rows(n: int = 600) -> SimpleNamespace:
    rng = np.random.default_rng(3)
    r = rng.uniform(1.0, 30.0, n)
    x = np.where(rng.random(n) < 0.2, -r, r)
    f = base.Features(
        r=r, local_hh=np.arange(n) % 48, local_month=np.full(n, 4), weekend=np.zeros(n, bool),
        weekday=np.zeros(n, int), holiday=np.zeros(n, bool), t2m=np.full(n, 10.0),
    )  # fmt: skip
    return SimpleNamespace(x=x, r=r, n=n, features=f)


def test_a_sign_only_prediction_becomes_a_mixture_around_plus_and_minus_the_reading() -> None:
    rows = invented_rows()
    lobes = signed.lobes_of(pred={"p": np.full(rows.n, 0.3)}, r=rows.r)
    assert np.allclose(lobes.p, 0.3)
    assert np.allclose(lobes.mu_i, -lobes.mu_r)
    assert np.allclose(lobes.s_i, core729.FLOOR_POINTS)
    stored = {
        "p": lobes.p,
        "mu_i": lobes.mu_i,
        "s_i": lobes.s_i,
        "mu_r": lobes.mu_r,
        "s_r": lobes.s_r,
    }
    again = signed.lobes_of(pred=stored, r=rows.r)
    assert np.array_equal(again.mu_i, lobes.mu_i)


def test_the_demand_climatology_is_the_cell_median() -> None:
    rows = invented_rows()
    clim = base.Climatology.fit(demand=rows.features.r, f=rows.features)
    d = signed.demand_climatology(climatology=clim, features=rows.features)
    assert d.shape == (rows.n,)
    assert np.isfinite(d).all()
    assert d.min() >= rows.r.min()
    assert d.max() <= rows.r.max()


def test_k1_scores_beside_b1_and_the_climatology() -> None:
    rows = invented_rows()
    clim = base.Climatology.fit(demand=rows.features.r, f=rows.features)
    p = np.where(rows.x < 0, 0.9, 0.05)
    out = real.k1_against_baselines(rows=rows, p=p, mu=rows.r, climatology=clim, pi=0.5)
    assert out["demand_mae"] == 0.0
    assert out["demand_mae_climatology"] > 0.0
    assert out["brier_b1"] > 0.0
    assert out["log_loss_p0"] > out["log_loss_b1"] or out["log_loss_p0"] > 0.0


def test_harm_and_detection_by_group_read_decidable_rows_only() -> None:
    mva = core.DECIDABLE_READING_POINTS
    rows = SimpleNamespace(
        x=np.array([5.0, 5.0, -5.0, -5.0, 5.0]),
        r=np.array([2 * mva, 2 * mva, 2 * mva, 2 * mva, 0.1 * mva]),
        world=np.array([0, 0, 1, 1, 0]),
    )
    p = np.array([0.0, 0.2, 0.9, 0.7, 0.9])
    out = signed.harm_detection_by_group(rows=rows, p=p)
    assert out["harm"]["world_none"] == pytest.approx(0.1)
    assert out["detection"]["world_engine"] == pytest.approx(0.8)
    assert np.isnan(out["detection"]["world_none"])


def test_the_new_result_figures_draw() -> None:
    dots = {
        "S": {"all": 0.03, "import": 0.02, "reverse": 0.2},
        "P0": {"all": 0.05, "import": 0.0, "reverse": 1.0},
    }
    assert figures.brier_dot_chart(scores=dots)
    assert figures.signed_crps_dot_chart(scores=dots)
    by_arm = {
        "S": {
            "harm": {"world_none": 0.01, "world_engine": 0.02},
            "detection": {"world_none": 0.5, "world_engine": 0.9},
        }
    }
    assert figures.harm_detection_chart(by_arm=by_arm, limit=0.15)
    assert figures.real_eras_chart(
        above_half={"net/0": 0.0, "solar/3": 0.01}, demand_above={"net/0": 0.4, "solar/3": 0.5}
    )

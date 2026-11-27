"""Known-answer tests of the offset model (gate G3 of experiment 734). Synthetic arrays only."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import e00734_hmm as hmm

PRIORS: Final[hmm.Priors] = hmm.Priors(rho_on=3e-4, rho_off=1 / 240, rho_cont=0.2)
SMALL: Final[hmm.Grid] = hmm.Grid(o_levels=(-1, 0, 1), l_levels=(-2, 0, 2), g_levels=(0.0,))
SMALL_PRIORS: Final[hmm.Priors] = hmm.Priors(rho_on=0.05, rho_off=0.3, rho_cont=0.4)
UNIT_SCALE: Final[np.ndarray] = np.ones(4)


def full(**kwargs: object) -> np.ndarray:
    """The full joint transition matrix: the sum of the stay, up, and down parts."""
    return np.sum(np.stack(hmm.offset_transition(**kwargs)), axis=0)  # ty: ignore[invalid-argument-type]


def student_noise(*, n: int, seed: int) -> np.ndarray:
    """Student-t noise with 4 degrees of freedom and scale 1."""
    return np.random.default_rng(seed).standard_t(4, n)


def joint_matrix(
    *, p: hmm.Priors, grid: hmm.Grid, boost: tuple[float, float] = (1.0, 1.0)
) -> np.ndarray:
    """The full joint transition matrix as a Kronecker product."""
    t = full(p=p, boost_up=boost[0], boost_down=boost[1], grid=grid)
    return np.kron(
        np.kron(t, hmm.level_transition(rho_l=p.rho_on, grid=grid)), hmm.gain_transition(grid=grid)
    )


def joint_emission(
    *, x: float, grid: hmm.Grid, sigma: float = 1.0, scale: float = 1.0
) -> np.ndarray:
    """Emission density on the joint state axis `(o, m), l, g`, flattened."""
    le = hmm.log_emission(x=x, f=0.0, sigma=sigma, scale=scale, grid=grid)
    return np.exp(np.concatenate([le] * hmm.N_M, axis=0)).ravel()


def test_offset_transition_rows_sum_to_one_for_every_boost() -> None:
    """Test offset transition rows sum to one for every boost and have no negative entry."""
    for boost in (1.0, 5.0, 21.0):
        for p in (PRIORS, hmm.Priors(rho_on=1e-3, rho_off=1 / 60, rho_cont=0.05, ramp_rule=False)):
            parts = hmm.offset_transition(p=p, boost_up=boost, boost_down=boost)
            total = np.sum(np.stack(parts), axis=0)
            np.testing.assert_allclose(total.sum(axis=1), 1.0, atol=1e-12)
            assert min(part.min() for part in parts) >= 0.0


def test_boost_changes_only_settled_rows_and_the_stay_probability_floor_holds() -> None:
    """Test a boost changes only rows with m = 0, and a boosted row keeps a stay of 0.5."""
    plain = full(p=PRIORS)
    boosted = full(p=PRIORS, boost_up=21.0, boost_down=5.0)
    n_o = hmm.N_O
    np.testing.assert_array_equal(plain[n_o:], boosted[n_o:])
    assert not np.allclose(plain[:n_o], boosted[:n_o])
    assert np.diag(boosted)[:n_o].min() >= hmm.STAY_FLOOR - 1e-12
    huge = full(p=hmm.Priors(rho_on=0.1, rho_off=0.2, rho_cont=0.2), boost_up=50.0)
    assert np.diag(huge)[:n_o].min() == pytest.approx(hmm.STAY_FLOOR)
    np.testing.assert_allclose(huge.sum(axis=1), 1.0, atol=1e-12)


def test_continuation_is_cheaper_than_a_fresh_step_and_noramp_prices_it_as_fresh() -> None:
    """Test a second up-step costs rho_cont, and the no-ramp setting prices it as a fresh step."""
    n_o = hmm.N_O
    from_up = hmm.N_O + 25  # o = +5 after an up-move (m = 1)
    settled_5 = 25
    ramp = full(p=PRIORS)
    flat = full(p=hmm.Priors(rho_on=3e-4, rho_off=1 / 240, rho_cont=0.2, ramp_rule=False))
    up_cols = [hmm.N_O + j for j in range(26, n_o)]
    assert ramp[from_up, up_cols].sum() == pytest.approx(0.2)
    assert flat[from_up, up_cols].sum() < 1e-3
    assert ramp[settled_5, up_cols].sum() < 3e-4


def test_return_to_zero_is_cheaper_than_a_new_level() -> None:
    """Test the return-to-zero prior is rho_off, far larger than the move to any other level."""
    parts = full(p=PRIORS)
    settled_5 = 25
    to_zero = parts[settled_5, 20]
    to_other = np.delete(parts[settled_5, :41], [20, 25]).max()
    assert to_zero == pytest.approx(PRIORS.rho_off)
    assert to_zero > 100 * to_other


def test_two_state_case_matches_the_hand_computation() -> None:
    """Test a two-block problem with offsets 0 and 1 against its closed-form posterior."""
    grid = hmm.Grid(o_levels=(0, 1), l_levels=(0,), g_levels=(0.0,))
    p = hmm.Priors(rho_on=0.2, rho_off=0.5, rho_cont=0.3)
    x = np.array([0.1, 1.3])
    post = hmm.forward_backward(x=x, f=np.zeros(2), sigma=1.0, scale=UNIT_SCALE, p=p, grid=grid)

    def dens(value: float, mean: float) -> float:
        return float(stats.t.pdf(value - mean, df=4))

    w_up = p.rho_on * dens(1.3, 1.0)
    w_stay = (1 - p.rho_on) * dens(1.3, 0.0)
    expected_up = w_up / (w_up + w_stay)
    assert post.o_marg[1, 1] == pytest.approx(expected_up, rel=1e-5)
    assert post.up_jump[1] == pytest.approx(expected_up, rel=1e-5)
    assert post.down_jump[1] == pytest.approx(0.0, abs=1e-12)
    assert post.o_marg[0, 0] == pytest.approx(1.0)
    assert post.loglik == pytest.approx(
        math.log(dens(0.1, 0.0)) + math.log(w_up + w_stay), rel=1e-6
    )


def test_posteriors_sum_to_one_and_equal_the_dense_matrix_computation() -> None:
    """Test forward-backward marginals sum to one and match a dense Kronecker forward-backward."""
    x = np.array([0.0, 1.2, 0.9, 1.4, 0.1, -0.2])
    x[2] = np.nan
    post = hmm.forward_backward(
        x=x,
        f=np.zeros(6),
        sigma=1.0,
        scale=UNIT_SCALE,
        p=SMALL_PRIORS,
        grid=SMALL,
        alpha_dtype=np.float64,
    )
    np.testing.assert_allclose(post.o_marg.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(post.l_marg.sum(axis=1), 1.0, atol=1e-6)
    t_full = joint_matrix(p=SMALL_PRIORS, grid=SMALL)
    init = np.zeros((9, 3, 1))
    init[1, :, 0] = 1 / 3
    alpha = [init.ravel() * joint_emission(x=x[0], grid=SMALL)]
    for t in range(1, 6):
        alpha.append((t_full.T @ alpha[-1]) * joint_emission(x=x[t], grid=SMALL))
    beta = [np.ones(27)]
    for t in range(5, 0, -1):
        beta.insert(0, t_full @ (joint_emission(x=x[t], grid=SMALL) * beta[0]))
    assert post.loglik == pytest.approx(math.log(alpha[-1].sum()), rel=1e-9)
    for t in range(6):
        gamma = (alpha[t] * beta[t]).reshape(9, 3)
        gamma /= gamma.sum()
        o_dense = gamma.sum(axis=1).reshape(3, 3).sum(axis=0)
        np.testing.assert_allclose(post.o_marg[t], o_dense, atol=1e-6)


def test_forward_backward_equals_brute_force_enumeration() -> None:
    """Test forward-backward matches enumeration of every path on a 3 by 3 by 1 grid."""
    x = np.array([0.0, 1.1, 0.8, 0.2, -0.4])
    n = len(x)
    post = hmm.forward_backward(
        x=x,
        f=np.zeros(n),
        sigma=1.0,
        scale=UNIT_SCALE,
        p=SMALL_PRIORS,
        grid=SMALL,
        alpha_dtype=np.float64,
    )
    t_full = joint_matrix(p=SMALL_PRIORS, grid=SMALL)
    init = np.zeros((9, 3, 1))
    init[1, :, 0] = 1 / 3
    total = _enumerate(x=x, t_full=t_full, init=init.ravel())
    assert post.loglik == pytest.approx(math.log(total.sum()), rel=1e-9)
    for t in range(n):
        axes = tuple(a for a in range(n) if a != t)
        marg = total.sum(axis=axes).reshape(9, 3).sum(axis=1).reshape(3, 3).sum(axis=0)
        np.testing.assert_allclose(post.o_marg[t], marg / marg.sum(), atol=1e-6)


def test_boosted_jump_probabilities_and_viterbi_equal_a_dense_enumeration() -> None:
    """Test boosted marginals, jump probabilities, and the Viterbi path against enumeration."""
    x = np.array([0.1, 1.2, 0.9, -0.3, -1.1, 0.2])
    n = len(x)
    bu = np.array([1.0, 4.0, 1.0, 2.5, 1.0, 1.0])
    bd = np.array([1.0, 1.0, 3.0, 1.0, 6.0, 1.0])
    post = hmm.forward_backward(
        x=x,
        f=np.zeros(n),
        sigma=1.0,
        scale=UNIT_SCALE,
        p=SMALL_PRIORS,
        boost_up=bu,
        boost_down=bd,
        grid=SMALL,
        alpha_dtype=np.float64,
    )
    mats = [joint_matrix(p=SMALL_PRIORS, grid=SMALL, boost=(bu[t], bd[t])) for t in range(n)]
    init = np.zeros((9, 3, 1))
    init[1, :, 0] = 1 / 3
    out = init.ravel() * joint_emission(x=x[0], grid=SMALL)
    for t in range(1, n):
        step = mats[t] * joint_emission(x=x[t], grid=SMALL)[None, :]
        out = out[..., None] * step.reshape((1,) * (t - 1) + (27, 27))
    total = out.sum()
    o_of = np.array([SMALL.o_levels[(k // 3) % 3] for k in range(27)])
    l_of = np.array([SMALL.l_levels[k % 3] for k in range(27)])
    assert post.loglik == pytest.approx(math.log(total), rel=1e-9)
    for t in range(1, n):
        axes = tuple(a for a in range(n) if a not in (t - 1, t))
        pair = out.sum(axis=axes) / total
        up = (o_of[None, :] > o_of[:, None]) * pair
        down = (o_of[None, :] < o_of[:, None]) * pair
        lj = (l_of[None, :] != l_of[:, None]) * pair
        assert post.up_jump[t] == pytest.approx(up.sum(), abs=1e-9)
        assert post.down_jump[t] == pytest.approx(down.sum(), abs=1e-9)
        assert post.l_jump[t] == pytest.approx(lj.sum(), abs=1e-9)
    best = np.unravel_index(int(np.argmax(out)), out.shape)
    o_path, l_path, _ = hmm.viterbi(
        x=x,
        f=np.zeros(n),
        sigma=1.0,
        scale=UNIT_SCALE,
        p=SMALL_PRIORS,
        boost_up=bu,
        boost_down=bd,
        grid=SMALL,
    )
    np.testing.assert_array_equal(o_path, o_of[list(best)])
    np.testing.assert_array_equal(l_path, l_of[list(best)])


def _enumerate(*, x: np.ndarray, t_full: np.ndarray, init: np.ndarray) -> np.ndarray:
    """Weight of every path of 27 states, as an array with one axis per block."""
    out = init * joint_emission(x=x[0], grid=SMALL)
    for t in range(1, len(x)):
        step = t_full * joint_emission(x=x[t], grid=SMALL)[None, :]
        out = out[..., None] * step.reshape((1,) * (t - 1) + (27, 27))
    return out


def test_a_mid_day_step_of_five_units_is_recovered_with_sign_and_timing() -> None:
    """Test a 30-day step of +5 units from 12:00 UTC is recovered with sign and timing."""
    n = 2 * 365 * hmm.BLOCKS_PER_DAY
    start = 200 * hmm.BLOCKS_PER_DAY + 2
    stop = start + 30 * hmm.BLOCKS_PER_DAY
    x = student_noise(n=n, seed=11)
    x[start:stop] += 5.0
    post = hmm.forward_backward(x=x, f=np.zeros(n), sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    median = np.argmax(np.cumsum(post.o_marg, axis=1) >= 0.5, axis=1) - 20
    inside = slice(start + 4, stop - 4)
    assert np.abs(median[inside] - 5).max() <= 0.5
    assert (1.0 - post.o_marg[inside, 20]).min() > 0.9
    assert (1.0 - post.o_marg[: start - 8, 20]).mean() < 0.05
    o_path, l_path, _ = hmm.viterbi(x=x, f=np.zeros(n), sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    nonzero = np.flatnonzero(o_path != 0)
    assert abs(int(nonzero[0]) - start) <= 1
    assert abs(int(nonzero[-1]) + 1 - stop) <= 1
    assert np.sign(o_path[nonzero]).min() == 1
    assert len(set(l_path.tolist())) == 1


def test_overlapping_steps_give_the_total_offset() -> None:
    """Test two overlapping steps of +5 units each give a total offset of +10 where both act."""
    n = 600
    x = student_noise(n=n, seed=5)
    x[100:260] += 5.0
    x[180:340] += 5.0
    post = hmm.forward_backward(x=x, f=np.zeros(n), sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    mean_level = (post.o_marg * np.arange(-20, 21)).sum(axis=1)
    assert abs(float(mean_level[200:250].mean()) - 10.0) < 0.5
    assert abs(float(mean_level[120:170].mean()) - 5.0) < 0.5
    assert abs(float(mean_level[280:330].mean()) - 5.0) < 0.5
    assert abs(float(mean_level[:80].mean())) < 0.3


def test_a_step_that_never_returns_goes_to_the_permanent_level() -> None:
    """Test a permanent +6 step puts the posterior mass on the permanent level."""
    n = 800
    x = student_noise(n=n, seed=8)
    x[300:] += 6.0
    post = hmm.forward_backward(x=x, f=np.zeros(n), sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    o_path, l_path, _ = hmm.viterbi(x=x, f=np.zeros(n), sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    assert l_path[-1] - l_path[0] == 6
    assert (o_path[310:] == 0).all()
    assert post.l_jump[295:306].max() > 0.5
    assert post.o_marg[500, 20] > 0.5


def test_a_demand_proportional_drift_is_absorbed_by_the_gain() -> None:
    """Test a drift of 5% of demand over 60 days is carried by the gain."""
    n = 720
    t = np.arange(n)
    f = 40.0 + 10.0 * np.sin(2 * np.pi * t / hmm.BLOCKS_PER_DAY)
    drift = np.clip((t - 200) / 240.0, 0.0, 1.0) * 0.05 * f
    x = student_noise(n=n, seed=2) + drift
    o_path, _, g_path = hmm.viterbi(x=x, f=f, sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    assert np.abs(o_path).max() <= 1
    assert g_path.max() >= 1


def test_a_daylight_only_drop_is_not_called_as_an_offset() -> None:
    """Test a 30-day drop of 4 units in the two daylight blocks only leaves the offset at zero."""
    n = 720
    x = student_noise(n=n, seed=4)
    scale = np.array([1.0, 2.5, 2.5, 1.0])
    x = x * scale[np.arange(n) % 4]
    daylight = (np.arange(n) % 4 == 1) | (np.arange(n) % 4 == 2)
    window = (np.arange(n) >= 300) & (np.arange(n) < 420)
    x = np.where(daylight & window, x - 4.0, x)
    post = hmm.forward_backward(x=x, f=np.zeros(n), sigma=1.0, scale=scale, p=PRIORS)
    assert (1.0 - post.o_marg[:, 20]).max() < 0.9
    o_path, _, _ = hmm.viterbi(x=x, f=np.zeros(n), sigma=1.0, scale=scale, p=PRIORS)
    assert (o_path == 0).all()


def test_the_mirror_boost_raises_a_weak_matching_step_and_kappa_zero_changes_nothing() -> None:
    """Test a neighbour's opposite step raises P(o != 0) of a weak step, and kappa = 0 does not."""
    n = 400
    x = student_noise(n=n, seed=9)
    x[150:250] += 3.0
    f = np.zeros(n)
    own = hmm.forward_backward(x=x, f=f, sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    neighbour_x = student_noise(n=n, seed=10)
    neighbour_x[151:250] -= 3.0
    nb = hmm.forward_backward(x=neighbour_x, f=f, sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    no_unknown = {"A": np.zeros(n, dtype=bool), "B": np.zeros(n, dtype=bool)}
    up, down = hmm.mirror_boost(
        site="A", pass1={"A": own, "B": nb}, neighbours=("B",), unknown=no_unknown, kappa=20.0
    )
    assert up[145:155].max() > 5.0
    weak = 0.4 * x
    base = hmm.forward_backward(x=weak, f=f, sigma=1.0, scale=UNIT_SCALE, p=PRIORS)
    helped = hmm.forward_backward(
        x=weak, f=f, sigma=1.0, scale=UNIT_SCALE, p=PRIORS, boost_up=up, boost_down=down
    )
    assert (1 - helped.o_marg[150:250, 20]).mean() > (1 - base.o_marg[150:250, 20]).mean()
    up0, down0 = hmm.mirror_boost(
        site="A", pass1={"A": own, "B": nb}, neighbours=("B",), unknown=no_unknown, kappa=0.0
    )
    same = hmm.forward_backward(
        x=weak, f=f, sigma=1.0, scale=UNIT_SCALE, p=PRIORS, boost_up=up0, boost_down=down0
    )
    np.testing.assert_allclose(same.o_marg, base.o_marg, atol=0.0)


def test_an_unknown_neighbour_block_gives_no_boost_and_the_own_series_is_never_read() -> None:
    """Test an unknown neighbour block gives no evidence, and a site never reads its own pass 1."""
    n = 30
    nb = hmm.Posterior(
        o_marg=np.zeros((n, 41), dtype=np.float32),
        l_jump=np.zeros(n),
        up_jump=np.zeros(n),
        down_jump=np.full(n, 0.5),
        loglik=0.0,
    )
    own = hmm.Posterior(
        o_marg=np.zeros((n, 41), dtype=np.float32),
        l_jump=np.zeros(n),
        up_jump=np.full(n, 0.9),
        down_jump=np.full(n, 0.9),
        loglik=0.0,
    )
    unknown = {"A": np.zeros(n, dtype=bool), "B": np.zeros(n, dtype=bool)}
    unknown["B"][10:20] = True
    up, down = hmm.mirror_boost(
        site="A", pass1={"A": own, "B": nb}, neighbours=("A", "B"), unknown=unknown, kappa=10.0
    )
    assert up[14] == 1.0
    assert up[0] == pytest.approx(1.0 + 10.0 * 0.5)
    assert up[7] == pytest.approx(6.0)
    assert up[6] == pytest.approx(6.0)
    assert up[24] == pytest.approx(6.0)
    assert down.max() == 1.0


def test_move_flags_follow_the_path() -> None:
    """Test the implied move flag is 1 after an up-move, 2 after a down-move, and 0 on a return."""
    path = np.array([0, 3, 3, 5, 5, 1, 1, 0, -2, -2, 0])
    expected = np.array([0, 1, 0, 1, 0, 2, 0, 0, 2, 0, 0])
    np.testing.assert_array_equal(hmm.move_flags(o_path=path), expected)


def _random_panel(*, seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A noisy series with steps, missing blocks, a forecast, and per-block boosts."""
    rng = np.random.default_rng(seed)
    x = rng.standard_t(4, n)
    for _ in range(3):
        a = int(rng.integers(0, n - 30))
        x[a : a + int(rng.integers(5, 30))] += rng.choice([-1, 1]) * rng.uniform(2, 8)
    x[rng.integers(0, n, 3)] = np.nan
    boost = 1 + rng.uniform(0, 20, n) * (rng.random(n) < 0.1)
    return x, rng.uniform(0, 3, n), boost, 1 + rng.uniform(0, 20, n) * (rng.random(n) < 0.1)


@pytest.mark.parametrize("seed", range(4))
def test_the_float32_chunked_viterbi_decodes_the_same_paths_as_the_float64_reference(
    seed: int,
) -> None:
    """Test the fast Viterbi pass decodes identical paths, with and without boosts."""
    x, f, bu, bd = _random_panel(seed=seed, n=90)
    scale = np.array([1.0, 1.3, 1.3, 1.0])
    for up, down in ((None, None), (bu, bd)):
        fast = hmm.viterbi(x=x, f=f, sigma=1.0, scale=scale, p=PRIORS, boost_up=up, boost_down=down)
        ref = hmm.viterbi_reference(
            x=x, f=f, sigma=1.0, scale=scale, p=PRIORS, boost_up=up, boost_down=down
        )
        for a, b in zip(fast, ref, strict=True):
            np.testing.assert_array_equal(a, b)


FULL_BLOCKS: Final[int] = 6_568
"""Blocks of a whole world, 2022-01-01 to 2026-06-30."""


@pytest.mark.skipif(
    bool(os.environ.get("CI")), reason="about 100 s and 0.6 GB; runs on the workstation only"
)
def test_the_fast_viterbi_equals_the_reference_on_a_whole_world_at_real_scale() -> None:
    """Test the fast pass decodes the reference's paths over 6,568 blocks of megawatt-scale data.

    The series is in megawatts with a noise unit of 0.22 MW, a demand forecast of 10 to 30 MW,
    long and short steps of 2 to 12 noise units (some never returning), outliers, missing runs, and
    sparse neighbour boosts, so float32 scores must stay resolved over the whole record.
    """
    rng = np.random.default_rng(6568)
    n, sigma = FULL_BLOCKS, 0.22
    scale = sigma * np.array([1.0, 1.4, 1.6, 1.1])
    day = np.arange(n) % hmm.BLOCKS_PER_DAY
    f = 20.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / (365 * hmm.BLOCKS_PER_DAY)) + 2.0 * day
    x = scale[day] * rng.standard_t(4, n) + 0.02 * f
    for _ in range(12):
        a = int(rng.integers(0, n - 10))
        width = int(rng.choice([2, 8, 60, 400, n]))
        x[a : a + width] += rng.choice([-1, 1]) * rng.uniform(2, 12) * sigma
    x[rng.integers(0, n, 20)] += 15 * sigma
    for _ in range(5):
        a = int(rng.integers(0, n - 50))
        x[a : a + int(rng.integers(1, 50))] = np.nan
    bu = 1 + rng.uniform(0, 20, n) * (rng.random(n) < 0.02)
    bd = 1 + rng.uniform(0, 20, n) * (rng.random(n) < 0.02)
    fast = hmm.viterbi(x=x, f=f, sigma=sigma, scale=scale, p=PRIORS, boost_up=bu, boost_down=bd)
    ref = hmm.viterbi_reference(
        x=x, f=f, sigma=sigma, scale=scale, p=PRIORS, boost_up=bu, boost_down=bd
    )
    assert np.count_nonzero(ref[0]) > 0
    for a, b in zip(fast, ref, strict=True):
        np.testing.assert_array_equal(a, b)


def test_the_fast_viterbi_equals_the_reference_on_the_small_grid() -> None:
    """Test the chunk size larger than the number of columns changes nothing."""
    x = np.array([0.1, 1.2, 0.9, -0.3, -1.1, 0.2, 4.0, 4.2, 0.1])
    f = np.zeros(len(x))
    fast = hmm.viterbi(x=x, f=f, sigma=1.0, scale=UNIT_SCALE, p=SMALL_PRIORS, grid=SMALL)
    ref = hmm.viterbi_reference(x=x, f=f, sigma=1.0, scale=UNIT_SCALE, p=SMALL_PRIORS, grid=SMALL)
    for a, b in zip(fast, ref, strict=True):
        np.testing.assert_array_equal(a, b)


def _kernel_panel(*, n: int, seed: int) -> dict:
    """A synthetic panel with a level change, missing blocks, and boosts."""
    rng = np.random.default_rng(seed)
    x = rng.standard_t(4, n) * 0.5
    start = int(rng.integers(0, n - 150))
    x[start : start + 100] += rng.choice([-1, 1]) * rng.uniform(1, 8)
    x[rng.random(n) < 0.03] = np.nan
    boost_up = np.where(rng.random(n) < 0.08, rng.choice([2.0, 5.0, 20.0], n), 1.0)
    boost_down = np.where(rng.random(n) < 0.08, 5.0, 1.0)
    return {
        "x": x,
        "f": rng.normal(size=n),
        "sigma": 0.7,
        "scale": np.array([1.0, 1.1, 0.9, 1.2]),
        "p": PRIORS,
        "boost_up": boost_up,
        "boost_down": boost_down,
    }


@pytest.mark.parametrize("seed", range(6))
def test_the_c_kernel_decodes_the_same_paths_as_the_numpy_pass_and_the_reference(seed: int) -> None:
    """Test the C Viterbi kernel equals `viterbi_numpy` and `viterbi_reference` path for path."""
    if hmm.native.load() is None:
        pytest.skip("no C compiler")
    kwargs = _kernel_panel(n=300 + 40 * seed, seed=seed)
    fast = hmm.viterbi(**kwargs, kernel="c")
    for other in (hmm.viterbi_numpy(**kwargs), hmm.viterbi_reference(**kwargs)):
        for a, b in zip(fast, other, strict=True):
            np.testing.assert_array_equal(a, b)


def test_the_c_kernel_carries_its_state_across_chunk_boundaries() -> None:
    """Test a record longer than one kernel chunk decodes as the NumPy pass does."""
    if hmm.native.load() is None:
        pytest.skip("no C compiler")
    kwargs = _kernel_panel(n=hmm.VITERBI_NATIVE_CHUNK * 2 + 37, seed=11)
    for a, b in zip(hmm.viterbi(**kwargs, kernel="c"), hmm.viterbi_numpy(**kwargs), strict=True):
        np.testing.assert_array_equal(a, b)


def test_without_a_compiler_viterbi_falls_back_to_numpy_and_the_c_switch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `auto` uses the NumPy pass when the kernel is missing, and `c` raises."""
    monkeypatch.setattr(hmm.native, "load", lambda: None)
    kwargs = _kernel_panel(n=200, seed=3)
    for a, b in zip(hmm.viterbi(**kwargs), hmm.viterbi_numpy(**kwargs), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(RuntimeError, match="unavailable"):
        hmm.viterbi(**kwargs, kernel="c")


@pytest.mark.parametrize("missing", ["run", "all"])
def test_the_c_kernel_stores_the_numpy_backpointers_bit_for_bit_including_ties(
    missing: str,
) -> None:
    """Test the C kernel's scores and backpointers equal the NumPy pass's, ties included.

    A run of missing blocks gives every state the same emission, so many candidates tie exactly
    and only the first-maximum rule decides the backpointer. The decoded path alone can hide a
    wrong tie rule, so the tables are compared.
    """
    if hmm.native.load() is None:
        pytest.skip("no C compiler")
    kwargs = _kernel_panel(n=hmm.VITERBI_NATIVE_CHUNK + 40, seed=5)
    if missing == "all":
        kwargs["x"] = np.full(len(kwargs["x"]), np.nan)
    else:
        kwargs["x"][100:200] = np.nan
    native_tables = hmm._native_tables(**kwargs, grid=hmm.FULL_GRID)
    assert native_tables is not None
    numpy_tables = hmm._numpy_tables(**kwargs, grid=hmm.FULL_GRID)
    for a, b in zip(native_tables, numpy_tables, strict=True):
        np.testing.assert_array_equal(a, b)


def test_viterbi_rejects_a_scale_that_would_make_a_score_nan() -> None:
    """Test a NaN or zero scale raises, so the two kernels never resolve a NaN score differently."""
    kwargs = _kernel_panel(n=200, seed=1)
    for bad in (np.array([1.0, np.nan, 1.0, 1.0]), np.array([1.0, 0.0, 1.0, 1.0])):
        kwargs["scale"] = bad
        with pytest.raises(ValueError, match="finite and positive"):
            hmm.viterbi(**kwargs)


def test_viterbi_records_the_kernel_it_used() -> None:
    """Test the NumPy switch adds `numpy` to the kernels recorded for the run metadata."""
    hmm.viterbi(**_kernel_panel(n=200, seed=2), kernel="numpy")
    assert "numpy" in hmm.KERNELS_USED

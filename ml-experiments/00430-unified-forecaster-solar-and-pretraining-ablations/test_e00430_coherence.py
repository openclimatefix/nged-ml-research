# ruff: noqa: D103, E402
"""Check experiment 430's coherence scores on cases with known answers."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path[:0] = [
    str(Path(__file__).parent),
    str(Path(__file__).parents[1] / "00310-metered-farm-long-lead-scoring"),
]

import e00430_coherence as coh
import numpy as np


def test_energy_score_of_two_members_around_the_observation() -> None:
    score = coh.energy_score(members=np.array([[0.0], [2.0]]), observed=np.array([1.0]))
    assert abs(score - 0.5) < 1e-12


def test_energy_score_is_zero_when_every_member_is_the_observation() -> None:
    observed = np.array([0.3, 0.5, 0.9])
    assert coh.energy_score(members=np.tile(observed, (4, 1)), observed=observed) == 0.0


def test_variogram_score_penalises_a_flat_path_when_the_observation_moves() -> None:
    first, second = np.array([0]), np.array([1])
    observed = np.array([0.0, 1.0])
    right = coh.variogram_score(
        members=np.tile(observed, (3, 1)), observed=observed, first=first, second=second
    )
    flat = coh.variogram_score(
        members=np.zeros((3, 2)), observed=observed, first=first, second=second
    )
    assert right == 0.0
    assert abs(flat - 1.0) < 1e-12


def test_pairs_keep_same_farm_neighbours_and_same_time_across_farms() -> None:
    farm = np.array([0, 0, 1, 0])
    hours = np.array([0.0, 1.0, 0.0, 9.0])
    first, second = coh.pairs(farm=farm, hours=hours)
    assert set(zip(first.tolist(), second.tolist(), strict=True)) == {(0, 1), (0, 2)}


def test_coupled_members_beat_independent_draws_on_a_correlated_target() -> None:
    rng = np.random.default_rng(0)
    level = rng.normal(size=51)
    members_coupled = np.sort(level)[:, None] * np.ones((51, 20)) + 0.01 * rng.normal(size=(51, 20))
    members_coupled = members_coupled[np.argsort(np.argsort(level))]
    shuffled = rng.permuted(members_coupled, axis=0)
    observed = np.full(20, 0.5)
    assert (
        coh.energy_score(members=members_coupled, observed=observed)
        <= coh.energy_score(members=shuffled, observed=observed) + 1e-9
    )


def test_variogram_score_prefers_members_that_keep_their_level_over_time() -> None:
    rng = np.random.default_rng(1)
    hours = np.arange(40)
    base = 0.4 + 0.3 * np.sin(hours / 6)
    coupled = base[None, :] + rng.normal(size=51)[:, None] * 0.08
    independent = rng.permuted(coupled, axis=0)
    first, second = np.triu_indices(len(hours), 1)
    near = (second - first) <= 12
    observed, firsts, seconds = base + 0.05, first[near], second[near]
    kept = coh.variogram_score(members=coupled, observed=observed, first=firsts, second=seconds)
    mixed = coh.variogram_score(
        members=independent, observed=observed, first=firsts, second=seconds
    )
    assert kept < mixed

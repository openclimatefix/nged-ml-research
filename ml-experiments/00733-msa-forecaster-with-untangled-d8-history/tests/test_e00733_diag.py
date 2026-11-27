"""Tests of the pure functions of the post hoc diagnostics of experiment 733."""

import e00733_diag as diag
import numpy as np
import pytest


def test_lead_day_of_rows_repeats_each_sample_for_its_half_hours() -> None:
    got = diag.lead_day_of_rows(lead_index=np.array([1, 9]), width=3)
    assert got.tolist() == [2, 2, 2, 10, 10, 10]


def test_restrict_to_lead_day_blanks_other_days() -> None:
    values = np.arange(6, dtype=float)
    got = diag.restrict_to_lead_day(values=values, row_lead_day=np.array([2, 2, 3, 3, 2, 3]), day=3)
    assert np.isnan(got[[0, 1, 4]]).all()
    assert got[[2, 3, 5]].tolist() == [2.0, 3.0, 5.0]


def test_split_by_segment_shares_no_segment_and_covers_every_row() -> None:
    seg = np.repeat(np.arange(10), 7)
    train_mask, test_mask = diag.split_by_segment(seg=seg, test_share=0.3, seed=1)
    assert not (train_mask & test_mask).any()
    assert (train_mask | test_mask).all()
    assert not set(seg[train_mask]) & set(seg[test_mask])
    assert len(set(seg[test_mask])) == 3


def test_split_by_segment_is_deterministic_and_keeps_both_sides() -> None:
    seg = np.array([0, 0, 1, 1])
    a = diag.split_by_segment(seg=seg, test_share=0.99, seed=5)
    b = diag.split_by_segment(seg=seg, test_share=0.99, seed=5)
    assert np.array_equal(a[1], b[1])
    assert a[0].any()
    assert a[1].any()


def test_split_by_segment_needs_two_segments() -> None:
    with pytest.raises(ValueError, match="two segments"):
        diag.split_by_segment(seg=np.zeros(4), test_share=0.5, seed=0)


def test_auc_matches_known_values() -> None:
    y = np.array([0, 0, 1, 1])
    assert diag.auc(p=np.array([0.1, 0.2, 0.8, 0.9]), y=y) == 1.0
    assert diag.auc(p=np.array([0.9, 0.8, 0.2, 0.1]), y=y) == 0.0
    assert diag.auc(p=np.full(4, 0.3), y=y) == 0.5
    assert np.isnan(diag.auc(p=np.array([0.1, 0.2]), y=np.array([1, 1])))


def test_probability_scores_of_a_base_rate_predictor() -> None:
    y = np.array([1.0, 0.0, 0.0, 0.0])
    got = diag.probability_scores(p=np.full(4, 0.25), y=y)
    assert got["brier"] == pytest.approx(0.1875)
    assert got["accuracy"] == 0.75
    assert got["auc"] == 0.5
    assert got["base_rate"] == 0.25


def test_pick_tokens_keeps_at_most_the_budget_of_valid_tokens() -> None:
    rng = np.random.default_rng(0)
    valid = rng.random((4, 3, 5)) < 0.6
    pick = diag.pick_tokens(valid=valid, per_sample=4, rng=rng)
    assert not (pick & ~valid).any()
    for b in range(4):
        assert pick[b].sum() == min(4, valid[b].sum())


def test_tiny_probe_separates_a_planted_sign() -> None:
    pytest.importorskip("torch")
    import e00733_diag_probe as probe

    rng = np.random.default_rng(3)
    n = 600
    truth = (rng.random(n) < 0.3).astype(np.float32)
    features = rng.normal(size=(n, 4)) + 2.0 * truth[:, None]
    seg = rng.integers(0, 12, n)
    train_mask, test_mask = diag.split_by_segment(seg=seg, test_share=0.3, seed=2)
    got = probe.fit_probe(
        features=features, truth=truth, train_mask=train_mask, test_mask=test_mask
    )
    assert got["auc"] > 0.9
    assert got["brier"] < got["base_rate"] * (1 - got["base_rate"])

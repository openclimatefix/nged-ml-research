"""Tests of the leakage assertions: each fires on a planted fault and stays quiet on the clean case (no torch)."""

import e00732_arms as arms
import e00732_leakage as leak
import e00732_overlays as ov
import numpy as np
import pytest
from world732 import make_bank


def test_inputs_not_targets_fires_on_a_planted_copy() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(size=(2, arms.N_CHANNELS, 96))
    t = rng.normal(size=(2, 8, 48))
    leak.assert_inputs_not_targets(windows=w, targets=t)
    w[1, 3, arms.MID : arms.MID + 48] = t[1, arms.T["event"]]
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_inputs_not_targets(windows=w, targets=t)
    w2 = rng.normal(size=(1, arms.N_CHANNELS, 96))
    d = rng.normal(size=(1, 48))
    w2[0, 5, arms.MID : arms.MID + 48] = d[0]
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_inputs_not_targets(windows=w2, targets=t[:1], delta=d)


def test_the_stack_reads_no_month_and_the_seed_spaces_are_disjoint() -> None:
    leak.assert_no_month_or_day_of_year()
    leak.assert_seed_spaces_disjoint()


def test_overlay_of_a_segment_does_not_depend_on_its_neighbours() -> None:
    _, segs = make_bank(draws_per_world=2)
    leak.assert_overlay_depends_only_on_its_segment(segments=segs, kind="test", seed=None)
    leak.assert_overlay_depends_only_on_its_segment(segments=segs, kind="train", seed=1)


def test_pair_index_must_come_from_the_same_bank() -> None:
    _, segs = make_bank(draws_per_world=2)
    index = ov.build_pair_index(segments=segs)
    leak.assert_pair_index_from_own_bank(index=index, segments=segs)
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_pair_index_from_own_bank(index=index, segments=segs[:-3])


def test_writes_into_other_experiments_data_folders_fire(tmp_path) -> None:
    other = tmp_path / "730"
    other.mkdir()
    mine = tmp_path / "732"
    mine.mkdir()
    leak.assert_stage_wrote_nothing_in(written=[mine / "a.json"], forbidden_roots=[other])
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_stage_wrote_nothing_in(written=[mine / "a.json", other / "b.json"], forbidden_roots=[other])


def test_limits_must_be_older_than_every_fit_record(tmp_path) -> None:
    limits = tmp_path / "limits.json"
    fit = tmp_path / "fit.json"
    limits.write_text("{}")
    fit.write_text("{}")
    leak.assert_limits_before_fits(limits=limits, fit_records=[fit])
    import os

    os.utime(fit, ns=(1, 1))
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_limits_before_fits(limits=limits, fit_records=[fit])


def test_sign_independence_and_stop_gradient_assertions() -> None:
    a = np.ones(4)
    leak.assert_sign_ignores_the_extra_heads(sign_before=a, sign_after=a)
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_sign_ignores_the_extra_heads(sign_before=a, sign_after=a + 0.1)
    leak.assert_stop_gradient_blocks_backbone(backbone_grad_sums=[0.0, 0.0])
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_stop_gradient_blocks_backbone(backbone_grad_sums=[0.0, 0.2])
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_stop_gradient_blocks_backbone(backbone_grad_sums=[])


def test_choices_come_before_the_first_test_score() -> None:
    leak.assert_no_test_choice(events=["limits", "bt", "fits", "test_score"])
    leak.assert_no_test_choice(events=["limits"])
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_no_test_choice(events=["limits", "test_score", "bt", "fits"])


def test_names_with_an_asterisk_fire() -> None:
    leak.assert_unit_ids_stable(names=["Ustar_unet_s0"])
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_unit_ids_stable(names=["U*_unet_s0"])

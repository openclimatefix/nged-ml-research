"""Tests of the leakage assertions: each fires on a planted fault and stays quiet on the clean case
(no torch).
"""

import copy
from pathlib import Path

import e00729_arms as arms729
import e00732_arms as arms
import e00732_core as core
import e00732_leakage as leak
import e00732_neighbours as nb
import e00732_overlays as ov
import numpy as np
import pytest
from world732 import make_bank, make_neighbours


class World:
    """The invented training bank through the real overlay, prepare, and batch builders."""

    def __init__(self) -> None:
        cal, self.segs = make_bank(draws_per_world=3)
        self.series = make_neighbours(cal=cal)
        self.overlays = ov.overlay_segments(
            segments=self.segs, kind="train", seed=0, neighbours=self.series
        )
        everyone = np.arange(len(cal.dates))
        stats = arms729.input_stats(cal=cal, train_days=everyone)
        medians = {
            n: nb.training_median(
                folded=nb.fold(signed=self.series[n], floor=0.0), train_rows=everyone
            )
            for n in nb.NEIGHBOURS
        }
        self.prepared = arms.prepare(
            overlays=self.overlays, holidays=cal.holidays, stats=stats, medians=medians
        )
        self.index = ov.build_pair_index(segments=self.segs)
        self.spec = core.ARMS["U*"]
        picks = arms.all_picks(prepared=self.prepared)
        self.picks = picks[:: max(len(picks) // 60, 1)][:60]

    def batch(self, *, seed: int = 0) -> arms.Batch:
        return arms.make_batch(
            prepared=self.prepared, picks=self.picks, spec=self.spec,
            rng=np.random.default_rng(seed), index=self.index,
        )  # fmt: skip


@pytest.fixture(scope="module")
def world() -> World:
    return World()


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


def test_the_stack_reads_no_month_and_the_seed_spaces_are_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leak.assert_no_month_or_day_of_year()
    leak.assert_seed_spaces_disjoint()
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_seed_spaces_disjoint(train_seeds=[])
    monkeypatch.setattr(core, "SEED_TEST_OVERLAY", core.SEED_TRAIN_OVERLAY + 2)
    with pytest.raises(leak.CorrectnessFailureError, match="shares a seed"):
        leak.assert_seed_spaces_disjoint()
    monkeypatch.setattr(core, "SEED_TEST_OVERLAY", core.SEED_VALIDATION_OVERLAY)
    with pytest.raises(leak.CorrectnessFailureError, match="validation and test"):
        leak.assert_seed_spaces_disjoint()


def test_no_test_month_is_used_by_training_pretraining_or_validation() -> None:
    used = {"train": ["2021-03", "2021-04"], "pretrain": ["2020-01"], "validation": ["2022-01"]}
    leak.assert_months_disjoint(used=used, test_months=["2022-03", "2022-04"])
    with pytest.raises(leak.CorrectnessFailureError, match="pretrain"):
        leak.assert_months_disjoint(
            used={**used, "pretrain": ["2020-01", "2022-04"]}, test_months=["2022-03", "2022-04"]
        )
    with pytest.raises(leak.CorrectnessFailureError, match="read nothing"):
        leak.assert_months_disjoint(used=used, test_months=[])
    with pytest.raises(leak.CorrectnessFailureError, match="read nothing"):
        leak.assert_months_disjoint(used={**used, "train": []}, test_months=["2022-03"])


def test_overlay_of_a_segment_does_not_depend_on_its_neighbours() -> None:
    cal, segs = make_bank(draws_per_world=2)
    series = make_neighbours(cal=cal)
    leak.assert_overlay_depends_only_on_its_segment(
        segments=segs, kind="test", seed=None, neighbours=series
    )
    leak.assert_overlay_depends_only_on_its_segment(
        segments=segs, kind="train", seed=1, neighbours=series
    )


def test_pair_index_must_come_from_the_same_bank() -> None:
    _, segs = make_bank(draws_per_world=2)
    index = ov.build_pair_index(segments=segs)
    leak.assert_pair_index_from_own_bank(index=index, segments=segs)
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_pair_index_from_own_bank(index=index, segments=segs[:-3])


def test_writes_into_other_experiments_data_folders_fire(tmp_path: Path) -> None:
    other = tmp_path / "730"
    other.mkdir()
    mine = tmp_path / "732"
    mine.mkdir()
    leak.assert_stage_wrote_nothing_in(written=[mine / "a.json"], forbidden_roots=[other])
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_stage_wrote_nothing_in(
            written=[mine / "a.json", other / "b.json"], forbidden_roots=[other]
        )


def test_limits_must_be_older_than_every_fit_record(tmp_path: Path) -> None:
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


def test_the_tensor_checks_run_and_fire_when_a_neighbour_channel_is_removed(world: World) -> None:
    batch = world.batch()
    checks = leak.assert_tensor_checks(batch=batch, prepared=world.prepared, spec=world.spec)
    assert "mirror_edges" in checks
    assert "hidden_zero" in checks
    broken = copy.deepcopy(batch)
    broken.x[:, list(leak.NEIGHBOUR_ROWS), :] = 0.0
    with pytest.raises(leak.CorrectnessFailureError, match="tensor checks failed"):
        leak.assert_tensor_checks(batch=broken, prepared=world.prepared, spec=world.spec)
    no_nb = arms.make_batch(
        prepared=world.prepared, picks=world.picks, spec=core.ARMS["U-N"],
        rng=np.random.default_rng(0), index=world.index,
    )  # fmt: skip
    with pytest.raises(
        leak.CorrectnessFailureError
    ):  # a batch without the 12 channels fails the wider arm's checks
        leak.assert_tensor_checks(batch=no_nb, prepared=world.prepared, spec=world.spec)


def test_hidden_spans_zero_all_twelve_neighbour_channels_and_the_check_fires_on_a_leak(
    world: World,
) -> None:
    batch = world.batch()
    assert (np.asarray(batch.hidden) > 0.5).any()
    leak.assert_hidden_neighbours_zero(batch=batch)
    broken = copy.deepcopy(batch)
    k, pos = map(int, np.argwhere(np.asarray(broken.hidden) > 0.5)[0])
    broken.xi[k, arms.NB_ROW["D5_change"], pos] = 0.5  # ty: ignore[invalid-assignment]
    with pytest.raises(leak.CorrectnessFailureError, match="hidden span"):
        leak.assert_hidden_neighbours_zero(batch=broken)
    unhidden = copy.deepcopy(batch)
    unhidden.hidden = np.zeros_like(unhidden.hidden)
    with pytest.raises(leak.CorrectnessFailureError, match="read nothing"):
        leak.assert_hidden_neighbours_zero(batch=unhidden)


def test_neighbour_values_are_folded_and_masks_binary(world: World) -> None:
    batch = world.batch()
    leak.assert_neighbours_unsigned_and_masks_binary(batch=batch)
    broken = copy.deepcopy(batch)
    broken.x[0, arms.NB_ROW["D14_value"], 5] = -1.0
    with pytest.raises(leak.CorrectnessFailureError, match="sign"):
        leak.assert_neighbours_unsigned_and_masks_binary(batch=broken)
    bad_mask = copy.deepcopy(batch)
    bad_mask.x[0, arms.NB_ROW["D3_mask"], 5] = 0.5
    with pytest.raises(leak.CorrectnessFailureError, match="mask"):
        leak.assert_neighbours_unsigned_and_masks_binary(batch=bad_mask)


def test_an_injected_member_differs_only_in_the_reading_the_bsp_step_and_the_aligned_changes(
    world: World,
) -> None:
    batch = world.batch()
    assert (np.asarray(batch.pair) > 0.5).any()
    leak.assert_injection_changes_only_allowed_channels(batch=batch)
    paired = int(np.argmax(np.asarray(batch.pair) > 0.5))
    # the rebuilt aligned channel may differ from the plain one, and the check allows it
    aligned = arms.NB_ROW["D3_aligned"]
    moved = copy.deepcopy(batch)
    moved.xi[paired, aligned] += 1.0  # ty: ignore[not-subscriptable]
    leak.assert_injection_changes_only_allowed_channels(batch=moved)
    for channel in (
        arms.ROW["csi"],
        arms.NB_ROW["D3_value"],
        arms.NB_ROW["D3_change"],
        arms.NB_ROW["D14_mask"],
        arms.ROW["r_mask"],
    ):
        broken = copy.deepcopy(batch)
        broken.xi[paired, channel, 7] += 1.0  # ty: ignore[not-subscriptable]
        with pytest.raises(leak.CorrectnessFailureError, match="may not change"):
            leak.assert_injection_changes_only_allowed_channels(batch=broken)
    silent = copy.deepcopy(batch)
    silent.xi = silent.x.copy()
    with pytest.raises(leak.CorrectnessFailureError, match="unchanged"):
        leak.assert_injection_changes_only_allowed_channels(batch=silent)
    unpaired = copy.deepcopy(batch)
    unpaired.pair = np.zeros_like(unpaired.pair)
    with pytest.raises(leak.CorrectnessFailureError, match="read nothing"):
        leak.assert_injection_changes_only_allowed_channels(batch=unpaired)
    no_member = copy.deepcopy(batch)
    no_member.xi = None
    with pytest.raises(leak.CorrectnessFailureError, match="no injected member"):
        leak.assert_injection_changes_only_allowed_channels(batch=no_member)


def test_neighbours_move_only_inside_a_transfer_that_names_them(world: World) -> None:
    leak.assert_neighbours_move_only_by_mirror(overlays=world.overlays)
    tampered = copy.deepcopy(world.overlays)
    target = next(o for o in tampered if o.transfers)
    series = target.neighbours["D3"].copy()
    flat = series.reshape(-1)
    covered = np.zeros(flat.shape, bool)
    for t in target.transfers:
        covered[t.start_half_hour : t.end_half_hour + 1] = True
    spot = int(np.flatnonzero(~covered & np.isfinite(flat))[0])
    flat[spot] += 5.0
    target.neighbours["D3"] = series
    with pytest.raises(leak.CorrectnessFailureError, match="outside a transfer"):
        leak.assert_neighbours_move_only_by_mirror(overlays=tampered)
    plain = ov.plain_overlays(segments=world.segs, neighbours=world.series)
    with pytest.raises(leak.CorrectnessFailureError, match="read nothing"):
        leak.assert_neighbours_move_only_by_mirror(overlays=plain)

"""Tests of the pre-training exclusions, the standardiser check, and the leakage assertions."""

from pathlib import Path

import e00730_arms as arms
import e00730_leakage as leak
import e00730_pretrain as pretrain
import numpy as np
import pytest


def test_month_days_and_the_two_day_margin() -> None:
    days = pretrain.month_days(months=["2022-02"])
    assert len(days) == 28
    removed = pretrain.removed_days(
        held_out_months=["2022-02"], record_days=np.array(["2023-05-10"], dtype="datetime64[D]")
    )
    assert np.datetime64("2022-01-30") in removed
    assert np.datetime64("2022-03-02") in removed
    assert np.datetime64("2022-03-03") not in removed
    assert np.datetime64("2023-05-12") in removed
    assert np.datetime64("2023-05-13") not in removed


def test_removal_nans_clean_and_drops_rows() -> None:
    grid_days = np.datetime64("2022-01-01") + np.repeat(np.arange(10), 4)
    mask = pretrain.removed_half_hour_mask(grid_days=grid_days, removed=grid_days[8:12])
    clean = np.ones((3, grid_days.size))
    out = pretrain.clean_with_removal(clean=clean, removed_mask=mask)
    assert np.isnan(out[:, mask]).all()
    assert np.isfinite(out[:, ~mask]).all()
    assert clean.sum() == clean.size
    target = np.array([0, 10, 20, 30])
    origin = np.array([0, 5, 5, 5])
    keep = pretrain.keep_rows(target_idx=target, origin_idx=origin, removed_mask=mask)
    assert keep.tolist() == [True, False, True, True]
    origin_bad = np.array([8, 0, 0, 0])
    keep2 = pretrain.keep_rows(
        target_idx=np.zeros(4, dtype=int), origin_idx=origin_bad, removed_mask=mask
    )
    assert keep2.tolist() == [False, True, True, True]


def test_a_standardiser_fitted_on_a_removed_row_is_caught() -> None:
    rng = np.random.default_rng(0)
    surviving = rng.normal(size=(500, 5))
    removed = rng.normal(5.0, 1.0, size=(100, 5))
    mask = np.ones(5, dtype=bool)
    good = pretrain.fit_standardiser(x=surviving, mask=mask)
    bad = pretrain.fit_standardiser(x=np.concatenate([surviving, removed]), mask=mask)
    leak.assert_standardiser_surviving(stored=good, surviving_x=surviving)
    with pytest.raises(leak.CorrectnessFailureError, match="standardiser"):
        leak.assert_standardiser_surviving(stored=bad, surviving_x=surviving)


def test_pretraining_removal_assertion_names_the_rule_and_prints_no_date() -> None:
    mask = np.zeros(100, dtype=bool)
    mask[40:50] = True
    leak.assert_pretrain_removal(
        target_idx=np.array([1, 60]),
        origin_idx=np.array([0, 55]),
        removed_mask=mask,
        valid_end_idx=90,
    )
    with pytest.raises(leak.CorrectnessFailureError, match="removed period") as e:
        leak.assert_pretrain_removal(
            target_idx=np.array([45]), origin_idx=np.array([0]), removed_mask=mask, valid_end_idx=90
        )
    assert "20" not in str(e.value)
    with pytest.raises(leak.CorrectnessFailureError, match="holdout"):
        leak.assert_pretrain_removal(
            target_idx=np.array([95]), origin_idx=np.array([0]), removed_mask=mask, valid_end_idx=90
        )


def test_foreign_weights_are_refused(tmp_path: Path) -> None:
    (tmp_path / "pretrained").mkdir()
    inside = tmp_path / "pretrained" / "M0x" / "a.pt"
    inside.parent.mkdir()
    inside.write_text("x")
    outside = tmp_path / "elsewhere.pt"
    outside.write_text("x")
    leak.assert_no_foreign_weights(used=[inside], pretrained_root=tmp_path / "pretrained")
    with pytest.raises(leak.CorrectnessFailureError, match="outside"):
        leak.assert_no_foreign_weights(used=[outside], pretrained_root=tmp_path / "pretrained")


def test_no_arm_reads_a_month_or_the_day_of_the_year() -> None:
    leak.assert_no_month_or_ordinal(specs=arms.arm_specs(style="K"))


def test_no_arm_reads_a_target_row() -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    rng = np.random.default_rng(0)
    window = rng.normal(size=(arms.n_channels(spec=spec), 96))
    targets = rng.normal(size=(6, 48))
    leak.assert_inputs_not_targets(windows=window[None], targets=targets[None], spec=spec)
    window[3, 24:72] = targets[0]
    with pytest.raises(leak.CorrectnessFailureError, match="target row"):
        leak.assert_inputs_not_targets(windows=window[None], targets=targets[None], spec=spec)


def test_holiday_distance_is_clipped_at_seven_days() -> None:
    leak.assert_holiday_clip(nearest_days=np.array([-1.0, 0.0, 0.5, 1.0]))
    with pytest.raises(leak.CorrectnessFailureError, match="clipped at 7 days"):
        leak.assert_holiday_clip(nearest_days=np.array([8.0, 1.0]))
    with pytest.raises(leak.CorrectnessFailureError, match="non-finite"):
        leak.assert_holiday_clip(nearest_days=np.array([np.nan, 1.0]))
    with pytest.raises(leak.CorrectnessFailureError, match="never reaches"):
        leak.assert_holiday_clip(nearest_days=np.array([0.0, 1.0, 0.0]) * 0.5)


def test_sign_loss_never_applies_on_a_hidden_half_hour() -> None:
    hidden = np.zeros((1, 48), dtype=bool)
    hidden[0, 10:20] = True
    valid = np.ones((1, 48), dtype=bool)
    keeps = np.stack([(valid & ~hidden), valid], axis=1).astype(np.float32)
    leak.assert_keeps_follow_hidden(hidden=hidden, keeps=keeps, valid=valid)
    keeps[0, 0, 12] = 1.0
    with pytest.raises(leak.CorrectnessFailureError, match="hidden"):
        leak.assert_keeps_follow_hidden(hidden=hidden, keeps=keeps, valid=valid)


def test_a_causal_arm_must_ignore_the_future() -> None:
    visible = np.zeros(48, dtype=bool)
    visible[:20] = True
    a = np.linspace(0, 1, 48)
    b = a.copy()
    b[20:] += 3.0
    leak.assert_causal_blind(p_a=a, p_b=b, visible=visible)
    b[5] += 0.1
    with pytest.raises(leak.CorrectnessFailureError, match="future"):
        leak.assert_causal_blind(p_a=a, p_b=b, visible=visible)


def test_choices_come_before_the_first_test_score() -> None:
    leak.assert_choices_before_test(events=["architecture", "style", "margin", "test_score"])
    leak.assert_choices_before_test(events=["architecture"])
    with pytest.raises(leak.CorrectnessFailureError, match="margin"):
        leak.assert_choices_before_test(events=["architecture", "style", "test_score", "margin"])


def test_limits_must_be_older_than_every_fit(tmp_path: Path) -> None:
    limits = tmp_path / "limits.json"
    limits.write_text("{}")
    fit = tmp_path / "fit.json"
    fit.write_text("{}")
    leak.assert_files_older(older=limits, newer=[fit], what="limits")
    fit_old = tmp_path / "old.json"
    fit_old.write_text("{}")
    import os

    os.utime(fit_old, ns=(1, 1))
    with pytest.raises(leak.CorrectnessFailureError, match="limits"):
        leak.assert_files_older(older=limits, newer=[fit_old], what="limits")


def test_nothing_is_written_inside_experiment_729s_folder(tmp_path: Path) -> None:
    other = tmp_path / "729"
    other.mkdir()
    leak.assert_nothing_written_to(written=[tmp_path / "730" / "x.json"], forbidden_root=other)
    with pytest.raises(leak.CorrectnessFailureError, match="729"):
        leak.assert_nothing_written_to(written=[other / "x.json"], forbidden_root=other)


def test_encoder_input_table_helpers() -> None:
    table = np.zeros((3, 48, 4))
    table[:, :, 0] = np.maximum(0, np.cos((np.arange(48) - 24) / 48 * 2 * np.pi))
    assert pretrain.clear_sky_peak_hour(table=table, cosz_column=0) == 12
    assert pretrain.share_non_constant(table=table) == 0.25
    z = pretrain.StoredStandardiser(mean=np.zeros(28), std=np.full(28, 2.0), mask=np.ones(28, bool))
    out = pretrain.zscore_encoder_inputs(
        raw=np.array([[100.0] * 14 + [np.nan] * 14]), standardiser=z
    )
    assert out[0, 0] == 5.0
    assert out[0, 20] == 0.0

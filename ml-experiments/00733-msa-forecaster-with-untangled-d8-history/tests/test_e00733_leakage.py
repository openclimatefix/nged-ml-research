# ruff: noqa: E501
"""Tests of `e00733_leakage`.

Every guard passes on a clean input, fails on a broken one, and fails on an empty one.
"""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import e00733_core as core
import e00733_leakage as lk
import numpy as np
import pytest

C = 48 * 100 + 24


def _tokens(n_b: int = 2) -> dict:
    c = np.array([C + 48 * i for i in range(n_b)], dtype=np.int64)
    day_us = (lk.GRID_START_US + c * lk.STEP_US) // lk.DAY_US * lk.DAY_US
    return {
        "c_idx": c,
        "run_init_us": day_us,
        "target_idx": c[:, None] + np.arange(48)[None, :] * 1,
        "filled": np.ones((n_b, 48), dtype=bool),
        "other_latest_valid_us": lk.native_origin_us(c_idx=c) - lk.STEP_US,
    }


def test_origin_time_weather_passes_and_counts() -> None:
    """A clean batch passes and the count is the number of target tokens."""
    assert lk.assert_origin_time_weather(**_tokens()) == 2 * 48


@pytest.mark.parametrize(
    "mutation",
    [
        lambda t: t.update(run_init_us=t["run_init_us"] - lk.DAY_US),
        lambda t: t.update(run_available_us=lk.native_origin_us(c_idx=t["c_idx"]) + lk.STEP_US),
        lambda t: t.update(target_idx=t["target_idx"] - 1),
        lambda t: t.update(other_latest_valid_us=lk.native_origin_us(c_idx=t["c_idx"])),
    ],
)
def test_origin_time_weather_fails_on_each_violation(mutation: Callable[[dict], None]) -> None:
    """Each of the four checks fails when its condition is broken."""
    t = _tokens()
    mutation(t)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_origin_time_weather(**t)


def test_origin_time_weather_fails_on_nothing_to_check() -> None:
    """An empty batch or an all-false `filled` raises, so the assertion cannot pass vacuously."""
    t = _tokens()
    t["filled"] = np.zeros_like(t["filled"])
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_origin_time_weather(**t)
    t = _tokens(0)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_origin_time_weather(**t)


def test_example_row_has_the_plans_fields() -> None:
    """The example row names the Primary, the origins, the lead day, the run, and the verdict."""
    text = lk.example_row(
        c_idx=C,
        target_idx=C + 49,
        run_init_us=lk.GRID_START_US + (C // 48) * lk.DAY_US,
        other_latest_valid_us=lk.GRID_START_US + (C - 6) * lk.STEP_US,
    )
    assert core.SCORED_PRIMARY in text
    assert "09:00 UTC" in text
    assert "lead day 2" in text
    assert text.rstrip().endswith("pass")


def test_base_days_precede_history() -> None:
    """Base days before every fold's first history day pass; a later one fails; empty inputs fail."""
    lk.assert_base_days_precede_history(
        base_days_epoch=np.array([100, 200]), first_history_epoch_day_by_fold={"a": 300, "b": 400}
    )
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_base_days_precede_history(
            base_days_epoch=np.array([100, 350]),
            first_history_epoch_day_by_fold={"a": 300, "b": 400},
        )
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_base_days_precede_history(
            base_days_epoch=np.array([]), first_history_epoch_day_by_fold={"a": 300}
        )


def test_overlap_counts_find_a_planted_overlap() -> None:
    """The count finds a sample whose history spans a base day, and a channel day on a base day.

    Empty inputs fail.
    """
    base = np.array([10, 20])
    n_s, n_c = lk.count_overlap(
        sample_first_history_day=np.array([5, 30]),
        sample_last_history_day=np.array([15, 40]),
        base_days_epoch=base,
        channel_days_present=np.array([20, 21]),
    )
    assert (n_s, n_c) == (1, 1)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_no_overlap(n_samples=n_s, n_channels=n_c)
    lk.assert_no_overlap(n_samples=0, n_channels=0)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.count_overlap(
            sample_first_history_day=np.array([]),
            sample_last_history_day=np.array([]),
            base_days_epoch=base,
            channel_days_present=np.array([1]),
        )


def test_target_poison_counts_cells_and_is_not_vacuous() -> None:
    """The poison guard counts the poisoned flagged cells.

    Passes for a forecast that ignores them, fails for one that reads them, and fails with no
    cell.
    """
    values = np.ones((3, 100, 3), dtype=np.float32)
    ok = np.ones((3, 100), dtype=bool)
    poisoned, n = lk.poison_target_cells(
        values=values, ok=ok, primary=1, target_idx=np.arange(10, 20)
    )
    assert n == 10
    assert np.isnan(poisoned[1, 10:20]).all()
    assert np.isfinite(values).all()

    def ignoring(poison: bool) -> tuple[np.ndarray, int]:
        return np.arange(5.0), 10 if poison else 0

    assert lk.assert_poisoned_targets_change_nothing(run=ignoring) == 10

    def reading(poison: bool) -> tuple[np.ndarray, int]:
        return np.arange(5.0) + (1.0 if poison else 0.0), 10 if poison else 0

    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_poisoned_targets_change_nothing(run=reading)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_poisoned_targets_change_nothing(run=lambda _p: (np.arange(5.0), 0))


def test_placebo_guard() -> None:
    """A placebo value 28 days before passes; a shorter shift or an empty input fails."""
    tok = np.array([5000, 6000])
    assert lk.assert_placebo_from_28_days_earlier(source_idx=tok - 28 * 48, token_idx=tok) == 2
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_placebo_from_28_days_earlier(source_idx=tok - 27 * 48, token_idx=tok)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_placebo_from_28_days_earlier(source_idx=np.array([]), token_idx=np.array([]))


def test_training_power_cut() -> None:
    """A training process holding power at or after the fold start fails, and an empty tail fails."""
    power = np.full((2, 100), np.nan)
    power[:, :60] = 1.0
    assert lk.assert_training_power_cut(power=power, fold_start_idx=60) == 80
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_training_power_cut(power=power, fold_start_idx=59)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_training_power_cut(power=power, fold_start_idx=100)


def test_bank_digests(tmp_path: Path) -> None:
    """A matching digest passes and counts the banks; a differing one stops the line; an empty summary fails."""
    import e00733_channels as channels

    (tmp_path / "banks").mkdir()
    (tmp_path / "banks" / "summary.json").write_text(
        json.dumps({"digests": {"validation": "a", "train_s3": "b"}})
    )
    seen: list[str] = []

    def digest_of(path: Path) -> str:
        seen.append(path.name)
        return {"validation.pkl": "a", "train_s3.pkl": "b"}[path.name]

    assert lk.assert_bank_digests(root_730=tmp_path, digest_of=digest_of) == 2
    assert seen == ["validation.pkl", "train_s3.pkl"]
    with pytest.raises(channels.StoppedError):
        lk.assert_bank_digests(root_730=tmp_path, digest_of=lambda _p: "other")
    (tmp_path / "banks" / "summary.json").write_text(json.dumps({"digests": {}}))
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_bank_digests(root_730=tmp_path, digest_of=lambda _p: "a")


def test_synthetic_guards() -> None:
    """The training banks are seeds 3 and 4 and unread by 732; the input names carry no forbidden word."""
    lk.assert_synthetic_train_banks(seeds=(4, 3), fit_seeds_732=(0, 1, 2))
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_synthetic_train_banks(seeds=(3, 4), fit_seeds_732=(3,))
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_synthetic_train_banks(seeds=(1, 2), fit_seeds_732=())
    lk.assert_synthetic_inputs_clean(input_names=["history_power", "ens_mean"])
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_synthetic_inputs_clean(input_names=["history_power", "world_type"])
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_synthetic_inputs_clean(input_names=[])


def test_ordering_guards(tmp_path: Path) -> None:
    """The limits file is older than the fit records; a test score needs the limits and fits before it."""
    limits = tmp_path / "limits.json"
    limits.write_text("{}")
    import os

    os.utime(limits, (1_000, 1_000))
    fit = tmp_path / "fit.json"
    fit.write_text("{}")
    os.utime(fit, (2_000, 2_000))
    lk.assert_limits_before_fits(limits=limits, fit_records=[fit])
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_limits_before_fits(limits=fit, fit_records=[limits])
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_limits_before_fits(limits=limits, fit_records=[])
    lk.assert_no_test_choice(events=["limits", "fits", "test_score"])
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_no_test_choice(events=["limits", "test_score", "fits"])


def test_write_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A write inside this experiment's folder passes.

    One under an earlier experiment's folder fails; an empty list fails.
    """
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    import e00733_paths as paths

    mine = paths.data_root()
    lk.assert_wrote_only_here(written=[mine / "x.json"], data_root=mine)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_wrote_only_here(written=[paths.root_732() / "x.json"], data_root=mine)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_wrote_only_here(written=[], data_root=mine)


def test_read_run_files_reads_name_and_availability_column(tmp_path: Path) -> None:
    """A run file gives its date and its availability column; a missing or ambiguous one fails."""
    import polars as pl

    when = [datetime(2026, 1, 14, 9, tzinfo=UTC)] * 3
    pl.DataFrame({"power_fcst_init_time": when}).write_parquet(tmp_path / "init=2026-01-14.parquet")
    day = (date(2026, 1, 14) - date(1970, 1, 1)).days
    got = lk.read_run_files(summary_dirs=[tmp_path], epoch_days=[day])
    assert got[day] == (day * lk.DAY_US, day * lk.DAY_US + 9 * lk.HOUR_US)
    with pytest.raises(lk.CorrectnessFailureError):
        lk.read_run_files(summary_dirs=[tmp_path], epoch_days=[day + 1])
    pl.DataFrame({"power_fcst_init_time": [when[0], when[0] + timedelta(hours=1)]}).write_parquet(
        tmp_path / "init=2026-01-15.parquet"
    )
    with pytest.raises(lk.CorrectnessFailureError):
        lk.read_run_files(summary_dirs=[tmp_path], epoch_days=[day + 1])


def test_732_training_banks_are_read_from_its_fit_records(tmp_path: Path) -> None:
    """Fit records name the training banks; a failed record does not count."""
    fits = tmp_path / "fits"
    fits.mkdir()
    for seed in (0, 1, 2):
        (fits / f"U-star_unet_s{seed}.json").write_text(json.dumps({"seed": seed}))
    (fits / "U-star_unet_s3.failed.json").write_text(json.dumps({"seed": 3}))
    assert lk.read_732_training_bank_seeds(root_732=tmp_path) == (0, 1, 2)
    lk.assert_synthetic_train_banks(seeds=(3, 4), fit_seeds_732=(0, 1, 2))
    (fits / "U-star_unet_s4.json").write_text(json.dumps({"seed": 4}))
    with pytest.raises(lk.CorrectnessFailureError):
        lk.assert_synthetic_train_banks(
            seeds=(3, 4), fit_seeds_732=lk.read_732_training_bank_seeds(root_732=tmp_path)
        )
    with pytest.raises(lk.CorrectnessFailureError):
        lk.read_732_training_bank_seeds(root_732=tmp_path / "other")

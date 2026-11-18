# ruff: noqa: ANN001, ANN202, C408, PT018
"""Tests of the pure helpers of experiment 702 (the fixed-epoch rule, the blend, the bootstrap)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00702_core as core
import numpy as np
import polars as pl
import pytest


def _held(*, weeks: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return origin days and held flags with `weeks[w]` held rows in week `w`."""
    day = np.concatenate([np.full(n, 7 * w + 3) for w, n in weeks.items()])
    return day, np.ones(len(day), dtype=bool)


def test_two_held_weeks_of_enough_rows_keep_early_stopping():
    """Two weeks with 1,000 rows each is the smallest history that early-stops."""
    day, held = _held(weeks={10: 1000, 22: 1000})
    assert not core.needs_fixed_epochs(origin_day=day, is_val=held)


def test_one_thin_week_forces_the_fixed_epochs():
    """One week of 999 rows leaves a single usable held week, so the phase is fixed."""
    day, held = _held(weeks={10: 1000, 22: 999})
    assert core.needs_fixed_epochs(origin_day=day, is_val=held)


def test_many_thin_weeks_do_not_add_up():
    """Five weeks of 500 rows hold 2,500 rows but no week reaches 1,000."""
    day, held = _held(weeks=dict.fromkeys(range(5), 500))
    assert core.needs_fixed_epochs(origin_day=day, is_val=held)


def test_training_rows_do_not_count_as_held_rows():
    """Only rows flagged as early-stopping rows are counted."""
    day = np.concatenate([np.full(5000, 3), np.full(5000, 17)])
    held = np.concatenate([np.ones(5000, bool), np.zeros(5000, bool)])
    assert core.needs_fixed_epochs(origin_day=day, is_val=held)


def test_no_held_rows_forces_the_fixed_epochs():
    """A Primary with no held rows at all trains for the fixed count."""
    assert core.needs_fixed_epochs(origin_day=np.array([1, 2]), is_val=np.zeros(2, bool))


def test_only_fit_and_bank_use_injected_shifts():
    """Pre-training and step 2 see the observed lags, as in experiment 677."""
    assert [core.uses_synthetic_rows(command=c) for c in ("pretrain", "step2", "fit", "bank")] == [
        False, False, True, True,
    ]  # fmt: skip


def test_fit_driver_picks_the_builder_and_the_gpu_class_from_the_rule():
    """The rule chooses both the builder and the GPU class (read from `e00702_fit.py`'s source)."""
    source = (Path(core.__file__).parent / "e00702_fit.py").read_text()
    for line in (
        "SYNTHETIC = core.uses_synthetic_rows(command=CMD)",
        "_month_build = arm.build_month_syn if SYNTHETIC else arm._build_month_p5",
        "class Gpu702(arm.AuxGpuMonth if SYNTHETIC else e.LagGpuMonth):",
        "assert issubclass(Gpu702, arm.AuxGpuMonth) == SYNTHETIC",
    ):
        assert source.count(line) == 1, line
    assert source.count("arm.build_month_syn") == 1 and source.count("arm.AuxGpuMonth") == 2


def test_global_arm_uses_the_big_batch_and_the_others_the_small_one():
    """The plan's 8,192 against 512."""
    assert core.default_batch(arm="global") == 8192
    assert {core.default_batch(arm=a) for a in core.ARMS} == {512}


def test_blend_is_the_mean_per_quantile_then_sorted():
    """The mean is [3, 1, 5], which crosses, so it is sorted within the row to [1, 3, 5]."""
    a = np.array([[1.0, 2.0, 9.0]])
    b = np.array([[5.0, 0.0, 1.0]])
    out = core.blend_quantiles(first=a, second=b)
    np.testing.assert_allclose(out, [[1.0, 3.0, 5.0]])


def test_blend_rejects_shapes_that_differ():
    """Two forecasts on different rows cannot be blended."""
    with pytest.raises(AssertionError):
        core.blend_quantiles(first=np.zeros((2, 11)), second=np.zeros((3, 11)))


def test_seed_pairs_match_global_seed_s_with_s_plus_three():
    """The blend of seed s is compared with the mean of global seeds s and s + 3."""
    assert core.pair_seeds() == [((0, 0), (0, 3)), ((1, 1), (1, 4)), ((2, 2), (2, 5))]


def test_array_hash_changes_with_one_value_a_name_or_a_dtype():
    """The hash is sensitive to a value, a name, and a dtype."""
    base = {"mean": np.array([1.0, 2.0])}
    h = core.array_hash(arrays=base)
    assert h == core.array_hash(arrays={"mean": np.array([1.0, 2.0])})
    assert h != core.array_hash(arrays={"mean": np.array([1.0, 2.0000001])})
    assert h != core.array_hash(arrays={"std": np.array([1.0, 2.0])})
    assert h != core.array_hash(arrays={"mean": np.array([1.0, 2.0], dtype=np.float32)})


def test_stats_record_detects_changed_statistics(tmp_path):
    """A stored record accepts the same arrays and rejects different ones."""
    rec = core.StatsRecord(month="2026-05", primary=3, arrays={"mean": np.arange(4.0)})
    path = tmp_path / "s.npz"
    rec.save(path=path)
    core.StatsRecord.check(path=path, arrays={"mean": np.arange(4.0)})
    with pytest.raises(AssertionError):
        core.StatsRecord.check(path=path, arrays={"mean": np.arange(4.0) + 1e-9})


def test_restricted_series_table_keeps_exactly_the_named_primary():
    """The one-row table holds the Primary at the given index and no other."""
    table = pl.DataFrame({"site": [f"D{i}" for i in range(16)], "primary": list(range(16))})
    one = core.restrict_series_table(series=table, label_index=5)
    assert one["site"].to_list() == ["D5"]


def test_blocks_do_not_cross_months_and_span_seven_origin_days():
    """Block ids restart in each month, so a block never straddles two months."""
    day = np.concatenate([np.arange(100, 130), np.arange(200, 231)])
    month = np.array(["a"] * 30 + ["b"] * 31)
    ids = core.block_ids(origin_day=day, month=month)
    assert len(set(ids[:30]) & set(ids[30:])) == 0
    assert len(set(ids[:30])) == 5  # days 0-6, 7-13, 14-20, 21-27, 28-29
    assert ids[0] == ids[6] != ids[7]


def _brute(*, a, b, block, draws):
    """Return the statistic of each (block draw, seed draw) the slow way."""
    out = []
    for picks, s in draws:
        rows = np.concatenate([np.flatnonzero(block == k) for k in picks])
        out.append((a[s, rows] - b[s, rows]).mean())
    return np.array(out)


def test_block_bootstrap_statistic_equals_the_brute_force_resample():
    """For identical draws, the vectorised ratio of sums equals the mean over the resampled rows."""
    rng = np.random.default_rng(0)
    n_seeds, n_rows = 3, 60
    a = rng.normal(size=(n_seeds, n_rows))
    b = rng.normal(size=(n_seeds, n_rows))
    block = np.repeat(np.arange(6), 10)
    # Replay the draws the function makes, with the same generator.
    n_resamples = 50
    gen = np.random.default_rng(7)
    pick = gen.integers(0, 6, size=(n_resamples, 6))
    seed = gen.integers(0, n_seeds, size=n_resamples)
    brute = _brute(
        a=a, b=b, block=block, draws=[(pick[i], int(seed[i])) for i in range(n_resamples)]
    )
    got = core.block_seed_bootstrap(
        loss_a=a, loss_b=b, block=block, n_resamples=n_resamples, level=0.9, rng_seed=7
    )
    lo, hi = np.quantile(brute, [0.05, 0.95])
    assert got["lo"] == pytest.approx(lo) and got["hi"] == pytest.approx(hi)
    assert got["mean"] == pytest.approx((a - b).mean())


def test_block_bootstrap_interval_excludes_zero_for_a_clear_gain_and_covers_it_for_noise():
    """A constant shift of -0.2 gives an interval below zero; pure noise gives one around zero."""
    rng = np.random.default_rng(1)
    block = np.repeat(np.arange(20), 25)
    base = rng.normal(size=(3, 500))
    gain = core.block_seed_bootstrap(loss_a=base - 0.2, loss_b=base, block=block, n_resamples=2000)
    assert gain["hi"] < 0
    noise = core.block_seed_bootstrap(
        loss_a=rng.normal(size=(3, 500)),
        loss_b=rng.normal(size=(3, 500)),
        block=block,
        n_resamples=2000,
    )
    assert noise["lo"] < 0 < noise["hi"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                hi=-0.06,
                lo=-0.2,
                month_means=[-0.1] * 4,
                hi_without_d15=-0.01,
                hi_without_d18=-0.02,
            ),
            "supported",
        ),
        (
            dict(
                hi=-0.02,
                lo=-0.2,
                month_means=[-0.1] * 4,
                hi_without_d15=-0.01,
                hi_without_d18=-0.02,
            ),
            "small gain",
        ),
        (
            dict(
                hi=-0.06,
                lo=-0.2,
                month_means=[-0.1, -0.1, 0.01, -0.1],
                hi_without_d15=-0.01,
                hi_without_d18=-0.02,
            ),
            "inconclusive",
        ),
        (
            dict(
                hi=-0.06, lo=-0.2, month_means=[-0.1] * 4, hi_without_d15=0.01, hi_without_d18=-0.02
            ),
            "inconclusive",
        ),
        (
            dict(
                hi=0.3,
                lo=0.1,
                month_means=[0.2] * 4,
                hi_without_d15=0.3,
                hi_without_d18=0.3,
                lo_without_d15=0.1,
                lo_without_d18=0.1,
            ),
            "refuted",
        ),
        (
            dict(
                hi=0.3,
                lo=0.1,
                month_means=[0.2] * 4,
                hi_without_d15=0.3,
                hi_without_d18=0.3,
                lo_without_d15=-0.1,
                lo_without_d18=0.1,
            ),
            "inconclusive",
        ),
    ],
)
def test_verdict_follows_the_decision_rule(kwargs, expected):
    """Each row of the plan's decision rule gives its verdict."""
    assert core.verdict(**{"lo_without_d15": -1.0, "lo_without_d18": -1.0, **kwargs}) == expected  # ty: ignore[invalid-argument-type]


def _purged_cycle() -> np.ndarray:
    """Return experiment 422's purge flags of one 84-day cycle: days 0 to 8 and 62 to 76."""
    day = np.arange(84)
    return (day < 9) | ((day >= 62) & (day < 77))


FIRST_ORIGIN_IDX: Final[int] = 80_370
"""2024-04-01 09:00 UTC on the grid that starts on 2019-09-01."""


def test_first_target_is_half_an_hour_after_the_first_unpurged_origin():
    """Days 0 to 8 are purged, so the first target is 2024-04-10 09:30 UTC (index 80,803)."""
    first = core.first_target_idx(first_origin_idx=FIRST_ORIGIN_IDX, purged_day=_purged_cycle())
    assert first == 80_803


@pytest.mark.parametrize(
    ("month", "expected"),
    [
        ("2025-11", (80_803, 108_137)),
        ("2026-01", (80_803, 111_065)),
        ("2026-04", None),
        ("2026-05", None),
        ("2026-07", None),
    ],
)
def test_fold_span_starts_at_the_floor_or_the_window(month, expected):
    """The three early folds start at the floor; 2026-05 and 2026-07 start 730 days before."""
    refit = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
    cutoff = int((refit - datetime(2019, 9, 1, tzinfo=UTC)) / timedelta(minutes=30))
    floor = core.first_target_idx(first_origin_idx=FIRST_ORIGIN_IDX, purged_day=_purged_cycle())
    first, last = core.fold_span(cutoff_idx=cutoff, first_idx=floor)
    assert last == cutoff - 7
    window = cutoff - 730 * 48 + 1
    assert first == max(window, floor)
    if expected is not None:
        assert (first, last) == expected
    assert (first == floor) == (month in {"2025-11", "2026-01", "2026-04"})


def test_scorable_months_are_the_four_test_months_and_the_five_approved_folds():
    """No month of the final-test window (2026-08 to 2026-10) can be exported or scored."""
    assert not set(core.TEST_MONTHS) & set(core.EXTRA_MONTHS)
    assert set(core.SCORABLE_MONTHS) == {*core.TEST_MONTHS, *core.EXTRA_MONTHS}
    assert len(core.SCORABLE_MONTHS) == 9
    assert max(core.SCORABLE_MONTHS) == "2026-07"

"""Known-answer tests of the constants, the bars, the widening, the MDE file, and the verdict
table."""

import itertools
import json
from pathlib import Path

import e00721_core as core
import numpy as np
import pytest


def test_level_is_bonferroni_over_three() -> None:
    assert pytest.approx(1 - 0.05 / 3) == core.LEVEL
    assert core.N_RESAMPLES == 20_000
    assert core.GENERATOR_SEED == 721


def test_kish_of_equal_counts_is_the_block_number() -> None:
    assert core.kish_effective_number(counts=[5, 5, 5, 5]) == pytest.approx(4.0)
    assert core.kish_effective_number(counts=[10, 0, 0]) == pytest.approx(1.0)
    assert core.kish_effective_number(counts=[]) == 0.0


def test_widening_grows_as_blocks_shrink() -> None:
    wide = core.widening_factors(kish_b=5.0)
    narrow = core.widening_factors(kish_b=200.0)
    assert wide[0] > narrow[0] > 1.0
    assert wide[1] > narrow[1] > 1.0
    assert core.widening_factors(kish_b=1.0) == (float("inf"), float("inf"))


def test_widened_interval_keeps_the_point_and_stretches_each_end() -> None:
    lo, hi = core.widened_interval(point=-0.5, lower=-0.8, upper=-0.3, kish_b=10.0)
    assert lo < -0.8
    assert hi > -0.3
    t_ratio, root = core.widening_factors(kish_b=10.0)
    assert (-0.5 - lo) == pytest.approx(0.3 * t_ratio * root)


def test_mde_is_the_larger_of_the_floor_and_the_seed_difference() -> None:
    assert core.mde_points(seed_difference_points=0.1) == pytest.approx(0.404)
    assert core.mde_points(seed_difference_points=-0.9) == pytest.approx(0.9)


def test_mde_file_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "mde.json"
    core.write_mde_file(path=path, seed_difference_points=0.2)
    with pytest.raises(FileExistsError):
        core.write_mde_file(path=path, seed_difference_points=0.9)
    assert core.read_mde_file(path=path) == pytest.approx(0.404)
    assert json.loads(path.read_text())["fold"] == "2026-05"
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileNotFoundError):
        core.read_mde_file(path=tmp_path / "missing.json")


def test_bar_g1_g2() -> None:
    assert core.bar_g1_g2(point=-0.5, upper=-0.1, mde=0.404) == "pass"
    assert core.bar_g1_g2(point=-0.5, upper=0.1, mde=0.404) == "miss"
    assert core.bar_g1_g2(point=-0.3, upper=-0.1, mde=0.404) == "miss"


def test_g3_limits_and_combination() -> None:
    assert core.g3_limit(scope="pooled") == pytest.approx(0.10)
    assert core.g3_limit(scope="D5") == pytest.approx(0.20)
    ok = core.G3Check(reading="plain", scope="pooled", point=0.05, lower=-0.1, limit=0.10)
    over = core.G3Check(reading="open", scope="D5", point=0.3, lower=-0.1, limit=0.20)
    harm = core.G3Check(reading="open", scope="D10", point=0.3, lower=0.1, limit=0.20)
    assert core.check_g3(checks=[ok, ok]) == "pass"
    assert core.check_g3(checks=[ok, over]) == "unresolved"
    assert core.check_g3(checks=[ok, over, harm]) == "fail"


def test_replay_stop_and_noise() -> None:
    curve = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    passes, best = core.replay_stop(val_history=curve, patience=8)
    assert passes == 11
    assert best == pytest.approx(0.8)
    assert core.replay_stop(val_history=curve[:5], patience=8)[0] == 5
    assert core.noise_of_curve(val_history=[1.0]) == 0.0


def test_flag_fires_on_a_curve_that_keeps_falling_after_the_stop() -> None:
    rng = np.random.default_rng(0)
    descent = 1.0 - 0.02 * np.arange(1, 21)
    stalled = 0.61 + np.abs(rng.normal(0, 0.004, 12))
    falling = 0.6 - 0.01 * np.arange(1, 40) + rng.normal(0, 0.002, 39)
    result = core.stopped_while_falling(val_history=[*descent, *stalled, *falling])
    assert result.flagged
    assert result.stop8 < result.stop30
    flat = [*descent, *(0.6 + rng.normal(0, 0.004, 40))]
    assert not core.stopped_while_falling(val_history=flat).flagged


def _verdict_inputs(**kw: object) -> core.VerdictInputs:
    base = {"g1": "pass", "g2": "pass", "g3": "pass"}
    return core.VerdictInputs(**{**base, **kw})  # ty: ignore[invalid-argument-type]


def test_every_verdict_row_has_a_single_matching_case() -> None:
    cases = {
        0: _verdict_inputs(stops=core.Stops(manager_halt=True), correctness_failed=True),
        1: _verdict_inputs(correctness_failed=True, synthetic_failed=True),
        2: _verdict_inputs(synthetic_failed=True, fit_failed=True),
        3: _verdict_inputs(fit_failed=True, g3="fail"),
        4: _verdict_inputs(g3="fail"),
        5: _verdict_inputs(),
        6: _verdict_inputs(g3="unresolved", sg_a_flagged=True),
        7: _verdict_inputs(g1="miss", sg_a_flagged=True),
        8: _verdict_inputs(g1="miss"),
        9: _verdict_inputs(g2="miss"),
    }
    words = {
        0: "stopped, no verdict",
        1: "invalid",
        3: "failed",
        4: "not supported",
        5: "supported",
    }
    for row, inputs in cases.items():
        got = core.verdict_row(v=inputs)
        assert got.row == row, (row, got)
        if row in words:
            assert got.verdict == words[row]
    assert (
        core.verdict_row(v=_verdict_inputs(synthetic_failed=True, synthetic_cap_hit=True)).verdict
        == "inconclusive"
    )
    assert core.verdict_row(v=_verdict_inputs(synthetic_failed=True)).verdict == "not supported"
    assert core.verdict_row(v=_verdict_inputs(correctness_failed=True)).correctness_in_first_line


def test_every_stop_alone_gives_row_zero_and_none_gives_a_verdict() -> None:
    for name in core.Stops.__dataclass_fields__:
        got = core.verdict_row(v=_verdict_inputs(stops=core.Stops(**{name: True})))
        assert got.row == 0, name
    assert not core.Stops().any()


def test_the_table_is_total_over_its_inputs() -> None:
    readings = ("pass", "miss")
    for g1, g2, g3, flagged in itertools.product(
        readings, readings, ("pass", "fail", "unresolved"), (False, True)
    ):
        got = core.verdict_row(v=core.VerdictInputs(g1=g1, g2=g2, g3=g3, sg_a_flagged=flagged))  # ty: ignore[invalid-argument-type]
        assert 4 <= got.row <= 9
        if got.row == 5:
            assert (g1, g2, g3) == ("pass", "pass", "pass")


def test_futility_and_seeds_two_to_four() -> None:
    inconclusive = core.verdict_row(v=_verdict_inputs(g2="miss"))
    assert core.seeds_2_to_4_may_run(result=inconclusive, g1_point=-0.5)
    assert not core.seeds_2_to_4_may_run(result=inconclusive, g1_point=0.0)
    assert not core.seeds_2_to_4_may_run(
        result=core.verdict_row(v=_verdict_inputs()), g1_point=-0.5
    )

# ruff: noqa: E501, SIM300
"""Known-answer tests of the constants, split rule, world mix, margin file, and verdict table."""

import json
from datetime import date
from pathlib import Path

import e00729_core as core
import numpy as np
import pytest


def calendar_months() -> tuple[list[str], list[int], list[int]]:
    """Return 72 months (2019-01 to 2024-12) with 25 base days each; the last 10 carry BSP data."""
    months = [f"{y}-{m:02d}" for y in range(2019, 2025) for m in range(1, 13)]
    return months, [25] * 72, [0] * 62 + [25] * 10


def test_split_rule_positions_and_precedence() -> None:
    months, n_days, bsp_days = calendar_months()
    split = core.month_splits(months=months, n_days=n_days, bsp_days=bsp_days)
    lockdown = [m for m in months if core.LOCKDOWN_FIRST_MONTH <= m <= core.LOCKDOWN_LAST_MONTH]
    assert all(split[m] == "training" for m in lockdown)
    other = [m for m in months[:62] if m not in lockdown]
    # the second, fifth, eighth month of the non-BSP group (numbered without the lockdown months) is a test month
    for position, month in enumerate(other, start=1):
        if position % 3 == 2:
            assert split[month] == "test"
        elif position % 6 == 1:
            assert split[month] == "validation"
        else:
            assert split[month] == "training"
    bsp = months[62:]
    for position, month in enumerate(bsp, start=1):
        if position % 3 == 2:
            assert split[month] == "test"
        elif position == len(bsp):
            assert split[month] == "validation"
        else:
            assert split[month] == "training"
    assert sum(split[m] == "test" for m in bsp) == 3


def test_split_rule_skips_short_months_and_numbers_the_rest() -> None:
    months = ["2023-01", "2023-02", "2023-03", "2023-04"]
    full = core.month_splits(months=months, n_days=[25, 25, 25, 25], bsp_days=[0, 0, 0, 0])
    short = core.month_splits(months=months, n_days=[25, 10, 25, 25], bsp_days=[0, 0, 0, 0])
    assert "2023-02" not in short
    assert full["2023-02"] == "test"
    assert short["2023-03"] == "test"  # the second remaining month


def test_day_splits_drop_the_two_day_margin() -> None:
    splits = {
        "2023-01": "training",
        "2023-02": "test",
        "2023-03": "validation",
        "2023-04": "training",
    }
    days = [
        date(2023, 1, 28), date(2023, 1, 29), date(2023, 1, 30), date(2023, 2, 10),
        date(2023, 2, 27), date(2023, 3, 1), date(2023, 3, 10), date(2023, 3, 30),
        date(2023, 4, 2), date(2023, 4, 3), date(2023, 5, 1),
    ]  # fmt: skip
    out = core.assign_day_splits(days=days, splits=splits)  # ty: ignore[invalid-argument-type]
    assert out == [
        "training", "training", "dropped", "test", "test", "dropped", "validation", "validation",
        "dropped", "training", "dropped",
    ]  # fmt: skip


def test_world_probabilities_sum_to_one_and_keep_the_proportions() -> None:
    p = core.world_probabilities(none_share=0.5)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["none"] == 0.5
    assert p["engine"] / p["solar"] == pytest.approx(0.30 / 0.20)
    assert core.world_probabilities(none_share=0.15) == pytest.approx(core.TEST_MIX)
    assert core.TEST_MIX_INJECTED == pytest.approx(0.85)


def test_training_none_share_range_contains_the_test_and_shifted_shares() -> None:
    lo, hi = core.NONE_SHARE_RANGE
    assert lo < core.TEST_MIX["none"] < hi
    assert lo < core.SHIFTED_NONE_SHARE < hi
    shares = [core.draw_none_share(rng=np.random.default_rng(i)) for i in range(50)]
    assert min(shares) >= lo
    assert max(shares) <= hi
    assert len(set(shares)) > 40


def test_level_and_constants_match_the_page() -> None:
    assert core.LEVEL == pytest.approx(1 - 0.05 / 12)
    assert core.N_RESAMPLES == 20_000
    assert core.GENERATOR_SEED == 729
    assert core.QUANTILE_LEVELS == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    assert core.QHAT_POINTS == pytest.approx(0.1 / 10.379 * 100)
    assert core.N_FITS == 50
    assert len(core.CONTRASTS) == 13
    assert core.N_CONTRASTS == 12


def test_harm_limit_is_half_the_floor_over_the_import_share() -> None:
    assert core.harm_limit_points(reverse_share=0.051) == pytest.approx(0.125 / 0.949)
    assert core.harm_limit_points(reverse_share=0.0) == pytest.approx(0.125)
    assert core.harm_limit_points(reverse_share=0.051) > core.HARM_CRPS_POINTS


def test_margin_is_the_larger_of_the_floor_and_twice_the_seed_sd() -> None:
    assert core.margin_points(val_crps_by_seed=[1.0, 1.01, 0.99, 1.0, 1.0]) == 0.25
    big = [1.0, 1.4, 0.6, 1.2, 0.8]
    assert core.margin_points(val_crps_by_seed=big) == pytest.approx(2 * np.std(big, ddof=1))
    with pytest.raises(ValueError, match="seeds"):
        core.margin_points(val_crps_by_seed=[1.0, 1.1, 0.9])


def test_margin_file_refuses_to_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "margin.json"
    with pytest.raises(FileNotFoundError):
        core.read_margin_file(path=path)
    core.write_margin_file(path=path, val_crps_by_seed=[1.0] * 5)
    assert core.read_margin_file(path=path) == 0.25
    with pytest.raises(FileExistsError):
        core.write_margin_file(path=path, val_crps_by_seed=[3.0, 1.0, 5.0, 2.0, 1.0])
    assert json.loads(path.read_text())["margin_points"] == 0.25
    assert not list(tmp_path.glob("*.tmp"))


def test_architecture_rule_prefers_fewer_weights_within_the_tie() -> None:
    scores = {"dae": 1.10, "unet": 1.00, "bilstm": 1.05}
    weights = {"dae": 100_000, "unet": 1_000_000, "bilstm": 500_000}
    assert (
        core.choose_architecture(scores=scores, weights=weights) == "dae"
    )  # within 0.125 of the best
    scores["dae"] = 1.20
    assert core.choose_architecture(scores=scores, weights=weights) == "bilstm"
    scores["bilstm"] = 1.30
    assert core.choose_architecture(scores=scores, weights=weights) == "unet"


def test_baseline_choice_takes_the_lower_validation_crps() -> None:
    assert core.choose_baseline(b1=1.0, b2=1.1) == "B1"
    assert core.choose_baseline(b1=1.1, b2=1.0) == "B2"
    assert core.choose_baseline(b1=1.0, b2=1.0) == "B1"


def test_kish_number_and_widening() -> None:
    assert core.kish_number(weights=[5, 5, 5, 5]) == pytest.approx(4.0)
    assert core.kish_number(weights=[10, 0, 0]) == pytest.approx(1.0)
    assert core.kish_number(weights=[]) == 0.0
    assert core.widening_ratio(kish=1.0) == float("inf")
    # the plan: 8 blocks widen the BSP contrasts by about 1.46 at the Bonferroni level
    assert core.widening_ratio(kish=8.0) == pytest.approx(1.46, abs=0.15)
    assert core.widening_ratio(kish=5.0) > core.widening_ratio(kish=50.0) > 1.0


def test_kish_matches_experiment_721_when_present() -> None:
    e721 = pytest.importorskip("e00721_core")
    counts = [3.0, 7.0, 1.0, 9.0]
    assert core.kish_number(weights=counts) == pytest.approx(
        e721.kish_effective_number(counts=counts)
    )


def _inputs(
    *, point: float = -0.5, lower: float = -0.8, upper: float = -0.2, **kw: object
) -> core.VerdictInputs:
    base: dict = {
        "stopped": False, "invalid": False, "failed": False, "h1_pass": True, "h2_pass": True,
        "c1": core.Reading(point, lower, upper), "margin": 0.25, "calibration_pass": True,
        "lobe_pass": True,
    }  # fmt: skip
    return core.VerdictInputs(**{**base, **kw})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("row", "kwargs"),
    [
        (0, {"stopped": True, "invalid": True}),
        (1, {"invalid": True, "failed": True}),
        (2, {"failed": True, "h1_pass": False}),
        (3, {"h1_pass": False}),
        (3, {"h2_pass": False}),
        (4, {}),
        (5, {"calibration_pass": False}),
        (5, {"lobe_pass": False}),
        (6, {"point": -0.1, "lower": -0.3, "upper": -0.05}),
        (7, {"point": -0.1, "lower": -0.9, "upper": 0.7}),
        (8, {"point": 0.3, "lower": 0.1, "upper": 0.5}),
        (8, {"point": -0.05, "lower": -0.2, "upper": 0.1}),
    ],
)
def test_every_verdict_row_is_reachable_and_first_match_wins(row: int, kwargs: dict) -> None:
    assert core.verdict_row(v=_inputs(**kwargs)) == row


def test_verdict_boundaries() -> None:
    # exactly at -m with an upper end below zero is a gain
    assert core.verdict_row(v=_inputs(point=-0.25, lower=-0.6, upper=-0.01)) == 4
    # just short of -m is a small gain, not support
    assert core.verdict_row(v=_inputs(point=-0.2499, lower=-0.6, upper=-0.01)) == 6
    # an upper end of exactly zero is not below zero
    assert core.verdict_row(v=_inputs(point=-0.5, lower=-1.0, upper=0.0)) in (7, 8)
    assert core.verdict_word(v=_inputs()) == "supported"
    assert core.verdict_word(v=_inputs(stopped=True)) == "stopped"
    assert core.verdict_word(v=_inputs(point=0.3, lower=0.1, upper=0.5)) == "not supported"


def test_harm_checks_fail_on_each_limit() -> None:
    assert core.harm_passes(crps_excess=0.12, mae_excess=0.25)
    assert not core.harm_passes(crps_excess=0.1201, mae_excess=0.0)
    assert not core.harm_passes(crps_excess=0.0, mae_excess=0.2501)
    assert core.harm_passes(crps_excess=0.0, mae_excess=0.0, false_sign_share=0.005)
    assert not core.harm_passes(crps_excess=0.0, mae_excess=0.0, false_sign_share=0.0051)


def test_reachability_and_gate_ratio() -> None:
    assert core.reachability_ok(n_reverse_all=2000, n_reverse_bsp=200)
    assert not core.reachability_ok(n_reverse_all=1999, n_reverse_bsp=500)
    assert not core.reachability_ok(n_reverse_all=5000, n_reverse_bsp=199)
    assert core.gate_ratio_ok(synthetic=0.2, real=0.17)
    assert core.gate_ratio_ok(synthetic=0.34, real=0.17)
    assert not core.gate_ratio_ok(synthetic=0.341, real=0.17)
    assert not core.gate_ratio_ok(synthetic=0.08, real=0.17)


def test_ranges_hold_every_simulator_parameter_of_the_page() -> None:
    assert core.RANGES["beta"] == (0.6, 1.0)
    assert core.RANGES["q_mva"] == (0.0, 0.3)
    assert core.RANGES["solar_capacity_points"] == (5.0, 150.0)
    assert core.RANGES["unit_points"] == (10.0, 50.0)
    assert core.RANGES["total_points_cap"][1] == 120.0
    assert core.RANGES["spell_days"] == (2, 19)


def test_the_ranges_are_the_registered_first_attempt() -> None:
    registered = {
        "no_run_share": (0.2, 0.7), "spells_per_30_days": (0.0, 2.0), "spell_days": (2, 19),
        "ramp_half_hours": (0.0, 2.0), "solar_capacity_points": (5.0, 150.0),
        "n_units": (1, 4), "unit_points": (10.0, 50.0), "evening_hours": (2.0, 6.0),
        "unit_output_share": (0.6, 1.0), "beta": (0.6, 1.0), "q_mva": (0.0, 0.3),
    }  # fmt: skip
    for name, value in registered.items():
        assert core.RANGES[name] == value


def test_quota_assigns_the_planned_mix_exactly() -> None:
    rng = np.random.default_rng(0)
    worlds = core.quota_world_types(rng=rng, n=64, none_share=0.15)
    assert {w: worlds.count(w) for w in core.WORLD_TYPES} == {
        "none": 10, "engine": 19, "solar": 13, "both": 22,
    }  # fmt: skip
    shifted = core.quota_world_types(rng=rng, n=64, none_share=0.5)
    assert shifted.count("none") == 32
    assert len(shifted) == 64
    assert core.quota_world_types(rng=np.random.default_rng(1), n=64, none_share=0.15) != worlds

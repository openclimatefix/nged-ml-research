"""Tests of the banks, their seed spaces and hashes, and the realism gate's functions."""

import e00729_banks as banks
import e00729_core as core
import e00729_sim as sim
import numpy as np
import pytest
from synthetic_world import make_calendar


def build(*, kind: str, seed: int | None, months: list[str] | None = None) -> banks.Bank:
    cal = make_calendar()
    use = {m: np.array([str(d)[:7] == m for d in cal.dates]) for m in ("2022-03", "2022-04")}
    return banks.build_bank(cal=cal, kind=kind, seed=seed, months=months or ["2022-03", "2022-04"], month_use=use)  # ty: ignore[invalid-argument-type]


def test_training_roots_never_meet_evaluation_roots() -> None:
    assert banks.seed_spaces_disjoint(train_seeds=core.MARGIN_SEEDS)
    roots = {banks.bank_root(kind="train", seed=s) for s in core.MARGIN_SEEDS}
    assert roots == {7290, 7291, 7292, 7293, 7294}
    assert banks.bank_root(kind="test", seed=None) == 729_000
    assert banks.bank_root(kind="shifted", seed=None) == 729_500
    assert not banks.seed_spaces_disjoint(train_seeds=[729_000 - core.TRAIN_SEED_BASE])


def test_bank_sizes_and_world_mix_for_each_kind() -> None:
    train = build(kind="train", seed=0)
    test = build(kind="test", seed=None)
    assert len(train.segments) == 2 * core.SEGMENT_DRAWS_TRAIN
    assert len(test.segments) == 2 * core.SEGMENT_DRAWS_EVAL
    assert all(core.NONE_SHARE_RANGE[0] <= v <= core.NONE_SHARE_RANGE[1] for v in train.none_share.values())
    assert all(v == core.TEST_MIX["none"] for v in test.none_share.values())
    shifted = build(kind="shifted", seed=None)
    assert all(v == core.SHIFTED_NONE_SHARE for v in shifted.none_share.values())


def test_training_banks_randomise_the_none_share_between_months_and_seeds() -> None:
    shares = {(s, m): v for s in (0, 1, 2) for m, v in build(kind="train", seed=s).none_share.items()}
    assert len({round(v, 6) for v in shares.values()}) == 6


def test_a_bank_is_reproducible_and_seed_dependent() -> None:
    a, b = build(kind="train", seed=0), build(kind="train", seed=0)
    assert a.digest == b.digest
    assert build(kind="train", seed=1).digest != a.digest
    assert build(kind="test", seed=None).digest != build(kind="shifted", seed=None).digest


def test_digest_changes_when_any_array_changes() -> None:
    bank = build(kind="test", seed=None)
    before = banks.bank_digest(segments=bank.segments)
    bank.segments[0].x[1, 3] += 1e-3
    assert banks.bank_digest(segments=bank.segments) != before


def test_save_and_load_roundtrip_and_detect_corruption(tmp_path) -> None:  # noqa: ANN001
    bank = build(kind="test", seed=None)
    path = tmp_path / "banks" / "test.pkl"
    banks.save_bank(bank=bank, path=path)
    assert banks.load_bank(path=path).digest == bank.digest
    assert not list(path.parent.glob("*.tmp"))
    bank.segments[0].x[1, 3] += 1.0
    banks.save_bank(bank=bank, path=path)
    with pytest.raises(AssertionError, match="changed on disk"):
        banks.load_bank(path=path)


def test_reverse_counts_split_by_bsp_months() -> None:
    bank = build(kind="test", seed=None)
    bank.segments[0].x[3, :5] = -10.0
    bank.segments[0].valid[3, :5] = True
    bank.segments[0].has_bsp = True
    bank.segments[1].x[3, :4] = -10.0
    bank.segments[1].valid[3, :4] = True
    bank.segments[1].has_bsp = False
    total, bsp = banks.reverse_counts(bank=bank)
    assert total >= 9
    assert bsp >= 5
    assert total - bsp >= 4


def test_near_zero_shares_use_one_megavolt_ampere() -> None:
    limit = banks.REAL_MIN_POINTS
    assert limit == pytest.approx(100.0 / 10.379)
    r = np.full((4, 48), 50.0)
    r[0, 3] = limit - 0.1
    r[1, :2] = limit - 1.0
    r[2, 0] = limit + 0.1
    day, half_hour = banks.near_zero_shares(r=r)
    assert day == 0.5
    assert half_hour == pytest.approx(3 / (4 * 48))


def test_step_fit_recovers_a_known_slope_and_its_error() -> None:
    rng = np.random.default_rng(0)
    d_eng = rng.normal(0, 10, 5000)
    d_bsp = -0.8 * d_eng + rng.normal(0, 4, 5000)
    fit = banks.step_fit(d_bsp=d_bsp, d_eng=d_eng)
    assert fit.slope == pytest.approx(-0.8, abs=0.01)
    assert fit.resid_sd == pytest.approx(4.0, rel=0.05)
    assert fit.se == pytest.approx(4.0 / (10 * np.sqrt(5000)), rel=0.1)
    nan_fit = banks.step_fit(d_bsp=np.array([1.0, np.nan]), d_eng=np.array([1.0, 1.0]))
    assert np.isnan(nan_fit.slope)


def test_size_matched_weights_give_each_quintile_an_equal_share() -> None:
    rng = np.random.default_rng(1)
    real = rng.normal(0, 5, 10_000)
    edges = banks.real_bin_edges(d_eng=real)
    synthetic = np.concatenate([rng.normal(0, 1, 9000), rng.normal(0, 30, 1000)])
    w = banks.size_matched_weights(d_eng=synthetic, edges=edges)
    bins = banks.size_bins(d_eng=synthetic, edges=edges)
    shares = [w[bins == b].sum() / w.sum() for b in range(core.GATE_STEP_BINS)]
    assert shares == pytest.approx([0.2] * 5)


def test_bsp_gate_each_condition_can_fail() -> None:
    synthetic = [banks.StepFit(-0.8, se, sd) for se, sd in zip(np.linspace(0.01, 0.02, 50), np.linspace(3, 5, 50), strict=True)]
    good = banks.bsp_gate(real=banks.StepFit(-0.76, 0.015, 4.0), real_ci=(-0.9, -0.6), synthetic=synthetic)
    assert good["pass"]
    assert not banks.bsp_gate(real=banks.StepFit(-0.2, 0.015, 4.0), real_ci=(-0.3, -0.1), synthetic=synthetic)["pass"]
    assert not banks.bsp_gate(real=banks.StepFit(-0.76, 0.05, 4.0), real_ci=(-0.9, -0.6), synthetic=synthetic)["pass"]
    assert not banks.bsp_gate(real=banks.StepFit(-0.76, 0.015, 9.0), real_ci=(-0.9, -0.6), synthetic=synthetic)["pass"]


def test_shares_by_world_read_only_the_named_worlds() -> None:
    bank = build(kind="train", seed=0)
    engine_only = banks.bank_shares_by_world(bank=bank, worlds=("engine",))
    nothing = banks.bank_shares_by_world(bank=bank, worlds=("missing_world",))
    assert np.isnan(nothing[0])
    assert 0.0 <= engine_only[0] <= 1.0
    assert isinstance(bank.segments[0], sim.Segment)

# ruff: noqa: E501
"""Tests of experiment 725's `LiveSource` on tiny synthetic arrays (CPU, torch, and the frozen checkout of experiment 713).

The tests skip when `e00713_batch` is not importable (put the frozen checkout's `00713-*` folder on `PYTHONPATH`).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
bt = pytest.importorskip("e00713_batch")
lay = pytest.importorskip("e00713_layout")

import e00725_batch as lb  # noqa: E402
import e00725_rule as rule  # noqa: E402

N_DAYS = 60
GRID = 48 * N_DAYS
C_DAY = 40
C_IDX = 48 * C_DAY + 24
ENS_VALUE = 3.0
ANALYSIS_VALUE = 1.0
SUMMARY_VALUE = 7.0


def make_base() -> bt.Source:
    """Return a synthetic parent source with constant analysis weather."""
    rng = np.random.default_rng(0)
    power = torch.from_numpy(rng.normal(size=(16, GRID)).astype(np.float32))
    weather = torch.full((16, GRID, 21), ANALYSIS_VALUE, dtype=torch.float16)
    weather[..., 19:21] = (
        SUMMARY_VALUE  # the day-summary columns of the analysis array equal `cams_days`
    )
    datetime_ = torch.zeros((16, GRID, 21), dtype=torch.float16)
    return bt.Source(
        power_z=power, dense_weather=weather, dense_datetime=datetime_,
        day_type=torch.zeros(GRID, dtype=torch.long), ens=None,
    )  # fmt: skip


def make_live(*, rule_name: str, phase: str = "train", found: bool = True) -> lb.LiveSource:
    """Return a `LiveSource` whose ENS cells read 3.0, each with a forecast origin 12 half-hours before its stamp."""
    n = torch.arange(GRID)
    return lb.build_live_source(
        base=make_base(), rule_name=rule_name, phase=phase,
        ens_z=torch.full((16, GRID, 19), ENS_VALUE, dtype=torch.float16),
        ens_found=torch.full((16, GRID), found, dtype=torch.bool),
        ens_origin=(n - 12)[None, :].expand(16, -1).clone(),
        cams_days=torch.full((16, N_DAYS + 5, 2), SUMMARY_VALUE),
        day0=0, grid_start_us=0,
    )  # fmt: skip


def batch(source: bt.Source) -> bt.Assembled:
    """Assemble one batch of two samples without the ensemble table."""
    return source.assemble(
        primary=torch.tensor([0, 3]), c_idx=torch.tensor([C_IDX, C_IDX + 48]),
        lead_day=torch.tensor([0, 0]), ensemble=False,
    )  # fmt: skip


def test_rule_off_batch_equals_the_parents() -> None:
    """KL1: with every cell on the analysis, the batch is bit-identical to `Source.assemble`."""
    live = make_live(rule_name="L").analysis_copy()
    ours, theirs = (
        batch(live),
        bt.Source.assemble(
            live,
            primary=torch.tensor([0, 3]),
            c_idx=torch.tensor([C_IDX, C_IDX + 48]),
            lead_day=torch.tensor([0, 0]),
            ensemble=False,
        ),
    )
    assert torch.equal(ours.inputs.hist_weather, theirs.inputs.hist_weather)
    assert torch.equal(ours.inputs.top_weather, theirs.inputs.top_weather)


def test_sources_of_every_cell_follow_the_table() -> None:
    """KL2: ENS cells read 3.0, analysis cells 1.0, and the day summary 7.0."""
    out = batch(make_live(rule_name="L"))
    hist = out.inputs.hist_weather[0].float().numpy()
    table = rule.reference_table(contract=rule_contract(), rule="L")
    assert (hist[..., :19][table[..., :19] == rule.SRC_ENS] == ENS_VALUE).all()
    assert (hist[..., :19][table[..., :19] != rule.SRC_ENS] == ANALYSIS_VALUE).all()
    assert (hist[..., 19:21] == SUMMARY_VALUE).all()


def rule_contract():
    """Return the contract origin of the test samples as a datetime."""
    from datetime import UTC, datetime, timedelta

    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(minutes=30 * C_IDX)


def test_l_and_z_differ_only_in_the_changed_cells() -> None:
    """The history of `L` and `Z` differ in the 3,715 ENS values only (the summaries are constant here)."""
    a, z = batch(make_live(rule_name="L")), batch(make_live(rule_name="Z"))
    diff = (a.inputs.hist_weather != z.inputs.hist_weather)[0].numpy()
    changed = rule.changed_cells(rule="L")
    assert not (diff & ~changed).any()
    assert int(diff[..., :19].sum()) == 3715


def test_with_power_in_keeps_the_type_and_the_counters() -> None:
    """Stage 3 reads `with_power_in`, which must return a `LiveSource` that shares the counters."""
    live = make_live(rule_name="L")
    new = live.with_power_in(power_in=torch.zeros_like(live.power_z))
    assert type(new) is lb.LiveSource
    assert new.counters is live.counters
    batch(new)
    assert live.counters["assemble_calls"] == 1


def test_missing_cell_reads_the_analysis_in_training_and_zero_at_prediction() -> None:
    """With no ENS row, a training cell keeps the analysis value and a prediction cell reads 0."""
    train = batch(make_live(rule_name="L", found=False))
    predict = batch(make_live(rule_name="L", phase="predict", found=False))
    table = rule.reference_table(contract=rule_contract(), rule="L")[..., :19] == rule.SRC_ENS
    assert (train.inputs.hist_weather[0].float().numpy()[..., :19][table] == ANALYSIS_VALUE).all()
    assert (predict.inputs.hist_weather[0].float().numpy()[..., :19][table] == 0.0).all()


def test_forecast_origin_after_the_stamp_raises() -> None:
    """KL4: a forecast origin after the token's stamp raises an assertion."""
    live = make_live(rule_name="L")
    live.ens_origin = live.ens_origin + 10**4
    with pytest.raises(AssertionError):
        batch(live)


def test_placebo_batch_raises() -> None:
    """The rule has no table for placebo ages."""
    live = make_live(rule_name="L")
    with pytest.raises(ValueError, match="placebo"):
        live.assemble(
            primary=torch.tensor([0]), c_idx=torch.tensor([C_IDX]), lead_day=torch.tensor([0]),
            ensemble=False, ages=torch.arange(2, 16)[None], age_exists=torch.ones((1, 14), dtype=torch.bool),
        )  # fmt: skip

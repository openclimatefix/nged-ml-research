# ruff: noqa: E501
"""Tests of `e00733_channels`: the rule, the stored-output reader, the table, the placebo, and the coverage."""

import json
import os
from pathlib import Path
from typing import Any

import e00733_channels as ch
import e00733_core as core
import numpy as np
import pytest


def _reading(
    row: int = 4,
    solar: str = "reasonable",
    transfer: str = "works",
    complete: bool = True,
    **checks: bool | None,
) -> ch.Reading732:
    base = {"h2": True, "p1": True, "p3": True}
    base.update(checks)
    return ch.Reading732(
        row=row, solar_word=solar, transfer_word=transfer, checks=base, chain_complete=complete
    )


def test_rule_uses_732_when_every_check_passes() -> None:
    """P(X<0) comes from experiment 732 and every channel enters."""
    choice = ch.decide(reading=_reading())
    assert choice.p_neg_source == core.SOURCE_732
    assert choice.usable == core.CHANNEL_NAMES
    assert choice.dropped == ()
    assert not choice.fu_equals_fs


def test_rule_falls_back_to_730_when_h2_missing() -> None:
    """A check that the file does not hold counts as not passed, so P(X<0) comes from 730's S."""
    choice = ch.decide(reading=_reading(h2=None))
    assert choice.p_neg_source == core.SOURCE_730
    assert choice.usable == ("p_neg",)
    assert choice.dropped == ("solar", "step_p")
    assert choice.fu_equals_fs


def test_k2_is_not_part_of_the_rule() -> None:
    """Experiment 732 never scored K2, so a reading that carries no K2 check still picks U*."""
    reading = _reading()
    assert "k2" not in reading.checks
    assert ch.decide(reading=reading).p_neg_source == core.SOURCE_732


def test_rule_drops_solar_alone_when_p3_fails() -> None:
    """The solar channel drops when P3 fails, and the others stay."""
    choice = ch.decide(reading=_reading(p3=False))
    assert choice.usable == ("p_neg", "step_p")


def test_rule_drops_the_transfer_channel_on_a_bad_word() -> None:
    """The transfer word 'fails' drops the transfer channel only."""
    choice = ch.decide(reading=_reading(transfer="fails"))
    assert choice.usable == ("p_neg", "solar")


def test_unfinished_732_stops_the_line() -> None:
    """A missing verdict, or an unfinished chain, is verdict row 0."""
    with pytest.raises(ch.StoppedError):
        ch.decide(reading=None)
    with pytest.raises(ch.StoppedError):
        ch.decide(reading=_reading(complete=False))


def test_choice_is_written_once(tmp_path: Path) -> None:
    """Writing the same choice twice is allowed; a different choice raises."""
    path = tmp_path / "channels.json"
    ch.write_choice(path=path, choice=ch.decide(reading=_reading()))
    ch.write_choice(path=path, choice=ch.decide(reading=_reading()))
    assert ch.read_choice(path=path) == ch.decide(reading=_reading())
    with pytest.raises(ch.CorrectnessFailureError):
        ch.write_choice(path=path, choice=ch.decide(reading=_reading(p3=False)))


def test_read_reading_732_reads_the_stored_formats(tmp_path: Path) -> None:
    """The reader returns None without a verdict and reads the words and flags of the stored files."""
    assert ch.read_reading_732(root=tmp_path) is None
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "verdict.json").write_text(
        json.dumps({"row": 5, "words": {"solar": "partly", "transfer": "works"}})
    )
    (tmp_path / "scores" / "real.json").write_text(
        json.dumps({"h2": {"pass": True}, "p1": {"all": {"status": "pass"}}, "p3": {"pass": False}})
    )
    (tmp_path / "chain_status.json").write_text(json.dumps({"state": "complete"}))
    got = ch.read_reading_732(root=tmp_path)
    assert got is not None
    assert (got.row, got.solar_word, got.chain_complete) == (5, "partly", True)
    assert got.checks == {"h2": True, "p1": True, "p3": False}


def _write_source(
    root: Path, job: str, seed: int, n: int, rng: np.random.Generator, *, keys: bool
) -> None:
    folder = root / "predictions" / job.format(seed=seed)
    folder.mkdir(parents=True)
    arrays: dict[str, Any] = {
        "p": rng.random(n),
        "pv": rng.random(n) * 50,
        "step_p": rng.random(n),
    }
    if keys:
        arrays |= {
            "row_date": np.repeat(np.arange(19000, 19000 + n // 4 + 1), 4)[:n],
            "row_hh": np.tile(np.arange(4), n)[:n],
        }
    np.savez(folder / f"{ch.STORED_SET}.npz", **arrays)


def test_load_outputs_averages_seeds_and_zeroes_dropped_channels(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    """The reader averages the three seeds of experiment 732's file and sets a dropped channel to zero."""
    for s in core.SOURCE_SEEDS:
        _write_source(tmp_path, ch.JOB_732, s, 40, rng, keys=True)
    choice = ch.decide(reading=_reading(p3=False))
    out = ch.load_outputs(choice=choice, root_732=tmp_path, root_730=tmp_path)
    assert out.values.shape == (40, 3)
    assert (out.values[:, 1] == 0).all()
    assert (out.values[:, 0] > 0).any()
    assert len(out.day) == len(out.hh) == 40


def test_load_outputs_fails_on_a_missing_seed(tmp_path: Path, rng: np.random.Generator) -> None:
    """A missing seed file raises, never a silent average of fewer seeds."""
    _write_source(tmp_path, ch.JOB_732, core.SOURCE_SEEDS[0], 10, rng, keys=True)
    with pytest.raises(ch.CorrectnessFailureError):
        ch.load_outputs(choice=ch.decide(reading=_reading()), root_732=tmp_path, root_730=tmp_path)


def test_730_rows_take_their_keys_from_the_era_keys(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    """Experiment 730's files hold no keys; a length mismatch with the era keys raises."""
    for s in core.SOURCE_SEEDS:
        _write_source(tmp_path, ch.JOB_730, s, 12, rng, keys=False)
    choice = ch.decide(reading=_reading(h2=None))
    keys = (np.arange(12) // 4 + 19000, np.arange(12) % 4)
    out = ch.load_outputs(
        choice=choice, root_732=tmp_path / "none", root_730=tmp_path, eras_keys=keys
    )
    assert out.values.shape == (12, 3)
    with pytest.raises(ch.CorrectnessFailureError):
        ch.load_outputs(
            choice=choice,
            root_732=tmp_path / "none",
            root_730=tmp_path,
            eras_keys=(keys[0][:5], keys[1][:5]),
        )


def test_silent_channel_check_cannot_pass_on_nothing() -> None:
    """The not-silent check fails on an empty table, an all-zero usable channel, and a non-zero dropped channel."""
    good = np.full((5, 3), 0.5, dtype=np.float32)
    ch.assert_channels_not_silent(values=good, usable=core.CHANNEL_NAMES)
    with pytest.raises(ch.CorrectnessFailureError):
        ch.assert_channels_not_silent(values=np.zeros((0, 3)), usable=core.CHANNEL_NAMES)
    silent = good.copy()
    silent[:, 1] = 0.0
    with pytest.raises(ch.CorrectnessFailureError):
        ch.assert_channels_not_silent(values=silent, usable=core.CHANNEL_NAMES)
    with pytest.raises(ch.CorrectnessFailureError):
        ch.assert_channels_not_silent(values=good, usable=("p_neg",))
    bad = good.copy()
    bad[0, 0] = 1.5
    with pytest.raises(ch.CorrectnessFailureError):
        ch.assert_channels_not_silent(values=bad, usable=core.CHANNEL_NAMES)


def test_grid_index_uses_period_end_stamps() -> None:
    """Slot (d, hh) has grid index (d - start) * 48 + hh + 1."""
    day = np.array([ch.GRID_START_EPOCH_DAY, ch.GRID_START_EPOCH_DAY + 2])
    assert ch.grid_index(day=day, hh=np.array([0, 47])).tolist() == [1, 2 * 48 + 48]


def test_table_roundtrip_and_dense(tmp_path: Path) -> None:
    """The table survives parquet and the dense form puts each value at its grid index."""
    outputs = ch.StoredOutputs(
        day=np.array([ch.GRID_START_EPOCH_DAY + 1] * 3), hh=np.array([5, 2, 9]),
        values=np.array([[0.1, 1, 0.2], [0.3, 2, 0.4], [0.5, 3, 0.6]], dtype=np.float32),
    )  # fmt: skip
    table = ch.build_table(outputs=outputs)
    assert table.grid_index.tolist() == [48 + 3, 48 + 6, 48 + 10]
    ch.write_table(table=table, path=tmp_path / "t.parquet")
    back = ch.read_table(path=tmp_path / "t.parquet")
    dense, ok = back.dense(n_grid=200)
    assert ok.sum() == 3
    assert dense[48 + 3, 0] == pytest.approx(0.3)
    with pytest.raises(ch.CorrectnessFailureError):
        back.dense(n_grid=10)


def test_duplicate_rows_raise() -> None:
    """Two rows with one grid index fail."""
    out = ch.StoredOutputs(
        day=np.array([19000, 19000]), hh=np.array([1, 1]), values=np.zeros((2, 3), dtype=np.float32)
    )
    with pytest.raises(ch.CorrectnessFailureError):
        ch.build_table(outputs=out)


def test_select_channels_zeroes_the_others() -> None:
    """A dropped channel is zero and the flag turns off when no channel is kept."""
    v = np.ones((4, 3), dtype=np.float32)
    ok = np.ones(4, dtype=bool)
    kept, flag = ch.select_channels(values=v, ok=ok, names=("p_neg",))
    assert kept[:, 0].all()
    assert not kept[:, 1:].any()
    assert flag.all()
    _, none = ch.select_channels(values=v, ok=ok, names=())
    assert not none.any()


def test_placebo_reads_28_days_earlier_and_refuses_less() -> None:
    """Each value is the stored value 28 days before, the first 28 days are absent, and a shorter shift raises."""
    shift = core.PLACEBO_SHIFT_DAYS * core.STEPS_PER_DAY
    n = shift + 100
    values = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    ok = np.ones(n, dtype=bool)
    out, flag = ch.placebo(values=values, ok=ok)
    assert (out[shift] == values[0]).all()
    assert not flag[:shift].any()
    assert flag[shift:].all()
    with pytest.raises(ch.CorrectnessFailureError):
        ch.placebo(values=values, ok=ok, shift_steps=shift - 1)


def test_coverage_counts_visible_readings_only() -> None:
    """Coverage is 1 when every visible reading has a channel, drops with a gap, and fails on an empty denominator."""
    n = 48 * 40
    reading = np.ones(n, dtype=bool)
    origin_days = np.array([30])
    assert ch.coverage(origin_days=origin_days, reading_ok=reading, channel_ok=reading) == 1.0
    half = reading.copy()
    half[: n // 2] = False
    cov = ch.coverage(origin_days=origin_days, reading_ok=reading, channel_ok=half)
    assert 0.0 < cov < 1.0
    with pytest.raises(ch.CorrectnessFailureError):
        ch.coverage(origin_days=origin_days, reading_ok=np.zeros(n, dtype=bool), channel_ok=reading)


def test_visible_history_stops_at_c_minus_54() -> None:
    """The newest visible stamp is c - 54: age-2 row positions after it are hidden."""
    n = 48 * 40
    mask = ch.visible_history_mask(origin_days=np.array([30]), reading_ok=np.ones(n, dtype=bool))
    assert mask.shape == (1, 14, 48)
    c = 30 * 48 + 24
    j = np.arange(48)
    expected_age2 = (c - 96 + j) <= c - 54
    assert (mask[0, 0] == expected_age2).all()
    assert mask[0, 1:].all()


def test_assert_coverage_stops_the_line() -> None:
    """A fold under 90% coverage, an empty mapping, or a NaN stops the line."""
    ch.assert_coverage(coverages={"f": 0.95})
    for bad in ({}, {"f": 0.89}, {"f": float("nan")}):
        with pytest.raises(ch.StoppedError):
            ch.assert_coverage(coverages=bad)


def test_absent_share() -> None:
    """The absent share is the visible half-hours without a flag, and an empty denominator raises."""
    vis = np.array([True, True, True, False])
    flag = np.array([True, False, True, True])
    assert ch.absent_share(flag=flag, visible=vis) == pytest.approx(1 / 3)
    with pytest.raises(ch.CorrectnessFailureError):
        ch.absent_share(flag=flag, visible=np.zeros(4, dtype=bool))


REAL_732 = Path(os.environ.get("DATA_PATH_INTERNAL", "/nonexistent")) / "ml-experiments" / "00732-msa-d8-untangling-network-synthetic-injections"


@pytest.mark.skipif(
    not (REAL_732 / "scores" / "verdict.json").exists(), reason="experiment 732's private files absent"
)
def test_the_rule_on_experiment_732s_real_score_files() -> None:
    """Experiment 732's verdict row 4, H2 passing, solar 'not shown', transfer 'works only in Brier score'.

    The rule gives P(X<0) from U*, the solar channel dropped, the transfer step from U*, and FU
    distinct from FS.
    """
    reading = ch.read_reading_732(root=REAL_732)
    assert reading is not None
    assert reading.row == 4
    assert reading.solar_word == "not shown"
    assert reading.transfer_word == "works only in Brier score"
    assert reading.checks["h2"] is True
    choice = ch.decide(reading=reading)
    assert choice.p_neg_source == core.SOURCE_732
    assert choice.usable == ("p_neg", "step_p")
    assert choice.dropped == ("solar",)
    assert not choice.fu_equals_fs

"""Tests of the component checks on a world whose clear-sky index is NaN at dawn and dusk."""

import copy
from pathlib import Path

import e00729_banks as banks729
import e00730_banks as banks
import numpy as np
import pytest
from fixture730 import make_root


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    root, _, cal = make_root(tmp_path, monkeypatch)
    bank = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    return bank, cal


def test_the_fixture_clear_sky_index_is_nan_at_night_and_at_dawn_and_dusk(world: tuple) -> None:
    _, cal = world
    assert np.isnan(cal.csi[:, 0]).all()
    up = cal.cosz[0] > 0
    first = np.flatnonzero(up)[0]
    assert np.isnan(cal.csi[0, first])
    assert np.isfinite(cal.csi[0, first + 2])


def test_a_clear_day_exists_although_dawn_and_dusk_are_nan(world: tuple) -> None:
    bank, _ = world
    seg = next(s for s in bank.segments if s.world == "solar")
    clear, clear_day = banks.clear_sun_up(csi=seg.csi, cosz=seg.cosz)
    assert clear_day[seg.cal_index >= 0].any()
    assert clear.any()


def test_silent_s2_farms_fail_the_component_check(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, cal = world
    monkeypatch.setattr(banks, "MIN_CASES", 1)
    ok = banks.component_checks(banks_by_name={"test": bank}, cal=cal)
    assert not any("/solar/" in k for k in ok["short"]), ok["short"]
    silent = copy.deepcopy(bank)
    for seg in silent.segments:
        if seg.source == "S2":
            seg.solar[:] = 0.0
    out = banks.component_checks(banks_by_name={"test": silent}, cal=cal)
    assert not out["pass"]
    assert any("/solar/" in k and "S2" in k for k in out["short"])


def test_a_key_without_cases_fails_as_missing(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, cal = world
    monkeypatch.setattr(banks, "MIN_CASES", 1)
    cloudy = copy.deepcopy(bank)
    for seg in cloudy.segments:
        seg.csi[:] = 0.1
    out = banks.component_checks(banks_by_name={"test": cloudy}, cal=cal)
    assert any(k.endswith("/missing") and "/solar/" in k for k in out["short"])


def test_too_few_cases_fail(world: tuple) -> None:
    bank, cal = world
    out = banks.component_checks(banks_by_name={"test": bank}, cal=cal)
    assert "cases" in out
    few = [k for k in out["short"] if k.endswith("/too_few")]
    assert few
    assert all(
        n < banks.MIN_CASES
        for c in out["cases"].values()
        for k, n in c.items()
        if any(f.endswith(f"{k}/too_few") for f in few)
    )


def test_a_silent_engine_fails_per_segment_and_quiet_days_are_read_apart(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank, cal = world
    monkeypatch.setattr(banks, "MIN_CASES", 1)
    silent = copy.deepcopy(bank)
    for seg in silent.segments:
        seg.engine[:] = 0.0
    out = banks.component_checks(banks_by_name={"test": silent}, cal=cal)
    assert any("/engine/" in k and "weekday" in k for k in out["short"])

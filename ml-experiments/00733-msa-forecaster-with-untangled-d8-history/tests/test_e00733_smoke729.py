"""Tests of `e00733_smoke729`: experiment 729's checks run against experiment 730's test bank."""

import sys
import types
from collections.abc import Callable
from pathlib import Path

import e00733_smoke729 as driver
import pytest


def _install(
    monkeypatch: pytest.MonkeyPatch, *, tmp: Path, bank_check: Callable[[Path], str]
) -> None:
    stages = types.ModuleType("e00729_stages")
    stages.bank_path = lambda *, kind: tmp / "superseded.pkl"  # ty: ignore[unresolved-attribute]
    banks = types.ModuleType("e00730_banks")
    banks.bank_path = lambda *, root, kind: root / "banks" / f"{kind}.pkl"  # ty: ignore[unresolved-attribute]
    script = types.ModuleType("smoke_real_files")
    script.CHECKS = {  # ty: ignore[unresolved-attribute]
        "bank_s2_solar": lambda: bank_check(stages.bank_path(kind="test"))  # ty: ignore[unresolved-attribute]
    }
    for module in (stages, banks, script):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def test_the_bank_check_reads_experiment_730s_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Path] = []

    def check(path: Path) -> str:
        seen.append(path)
        return "test bank present: True"

    _install(monkeypatch, tmp=tmp_path, bank_check=check)
    assert driver.main() == 0
    assert seen
    assert seen[0].name == "test.pkl"
    assert "bank_s2_solar: test bank present: True" in capsys.readouterr().out


def test_an_absent_bank_fails_instead_of_passing_on_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, tmp=tmp_path, bank_check=lambda _path: "test bank present: False")
    assert driver.main() == 1
    assert "FAILED" in capsys.readouterr().out


def test_a_silent_bank_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def check(_path: Path) -> str:
        msg = "S2 solar-world days are silent"
        raise AssertionError(msg)

    _install(monkeypatch, tmp=tmp_path, bank_check=check)
    assert driver.main() == 1

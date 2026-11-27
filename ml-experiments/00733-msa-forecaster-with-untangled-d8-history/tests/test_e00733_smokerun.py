"""Tests of `e00733_smokerun`: the later stages run on the CPU, as the chain runs them."""

import argparse
from pathlib import Path
from typing import Any

import e00733_smoke as smoke
import e00733_smokerun as smokerun
import pytest


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_the_smoke_runs_the_later_stages_on_the_cpu_whatever_the_fits_device(
    device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attention, series, and figures run on the CPU in the chain, so the smoke test does too."""
    seen: dict[str, Any] = {}

    def fake_later(*, scratch: Path, device: str) -> dict[str, Any]:
        seen["later_device"] = device
        return {}

    def fake_run_smoke(**kw: Any) -> dict[str, Any]:
        seen["fit_device"] = kw["device"]
        return {"pass": True, "steps": {}}

    root = tmp_path / "data" / "00733-x"
    root.mkdir(parents=True)
    monkeypatch.setattr(smokerun, "later_steps", fake_later)
    monkeypatch.setattr(smokerun, "link_inputs", lambda **_: [])
    monkeypatch.setattr(smokerun, "check_steps", lambda **_: {})
    monkeypatch.setattr(smokerun, "read_stored_step", lambda **_: None)
    monkeypatch.setattr(smokerun.rt, "make_run_job", lambda **_: None)
    monkeypatch.setattr(smoke, "run_smoke", fake_run_smoke)
    args = argparse.Namespace(scratch=str(tmp_path / "scratch"), device=device)
    smokerun.stage_smoke(args, root)
    assert seen == {"later_device": "cpu", "fit_device": device}

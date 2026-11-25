# ruff: noqa: E501
"""Tests of experiment 725's KL5 grid length, the KL8 statement for rule `Z`, and the GPU-process pattern of `run.sh`."""

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import e00725_sources as src
import e00725_weather_lookahead as wl
import pytest

RUN_SH = (
    Path(__file__).resolve().parents[1]
    / "ml-experiments/00725-msa-live-buildable-history-weather/run.sh"
)


def test_cache_grid_is_the_extended_grid_not_the_default_grid() -> None:
    """The cache has 126,624 half-hours (to 2026-11-20 23:30), the grid that `redirect` sets; the default grid has 121,248."""
    ext = pytest.importorskip("e00667_extend")
    common = pytest.importorskip("e00087_common")
    old = common.GRID_END
    try:
        assert len(common.grid_times()) == 121_248
        assert datetime(2026, 7, 31, 23, 30, tzinfo=UTC) == common.GRID_END
        ext.redirect()
        assert len(common.grid_times()) == src.N_GRID == 126_624
    finally:
        common.GRID_END = old


def test_lookahead_margins_l_and_e_are_never_negative_and_z_is_the_lab_arm() -> None:
    """KL8 requires no negative margin for `L` and `E`. `Z` reads the analysis in every cell, so it has negative margins by construction."""
    contract = datetime(2026, 3, 10, 12, tzinfo=UTC)
    for name in ("L", "E"):
        assert all(x.margin_hours >= 0 for x in wl.lines_for(contract=contract, rule_name=name))  # ty: ignore[invalid-argument-type]
    assert any(x.margin_hours < 0 for x in wl.lines_for(contract=contract, rule_name="Z"))


def gpu_pattern() -> str:
    """Return the `pgrep` pattern of `run.sh`."""
    text = RUN_SH.read_text()
    match = re.search(r'pgrep -fc "([^"]+)"', text)
    assert match is not None
    return match.group(1).replace("[.]", r"\.")


def test_gpu_process_pattern_matches_interpreters_only() -> None:
    """Only a python process of a fit, prediction, or per-fold check counts, not a shell that names the script."""
    pattern = re.compile(gpu_pattern())
    python = "/w/.venv/bin/python /w/ml-experiments/00725-x/"
    assert pattern.search(python + "e00725_fit.py L 2025-11 0")
    assert pattern.search(python + "e00725_predict.py L 2025-11 0")
    assert pattern.search(python + "e00725_checks.py fold 2025-11 train --device cuda")
    assert not pattern.search(python + "e00725_checks.py kl3r 2025-11")
    assert not pattern.search(python + "e00725_sources.py train")
    for shell in (
        "bash /tmp/go.sh job e00725_checks.py fold 2025-11 predict",
        "/bin/bash -c ./run.sh e00725_fit.py L 2025-11 0",
        '/usr/bin/sh -c "$@" sh /w/.venv/bin/python /w/x/e00725_checks.py fold 2025-11 train',
    ):
        assert not pattern.search(shell)


def test_run_sh_is_valid_bash() -> None:
    """The script parses."""
    assert subprocess.run(["bash", "-n", str(RUN_SH)], check=False).returncode == 0

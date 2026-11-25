# ruff: noqa: E501
"""Tests of experiment 725's KL5 grid length, the KL8 statement for rule `Z`, and the GPU-process pattern of `run.sh`."""

import dataclasses
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import e00725_checks as ck
import e00725_frozen as fz
import e00725_rule as rule
import e00725_sources as src
import e00725_weather_lookahead as wl
import pytest

RUN_SH = (
    Path(__file__).resolve().parents[1]
    / "ml-experiments/00725-msa-live-buildable-history-weather/run.sh"
)


def test_cache_grid_is_the_extended_grid_not_the_default_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache has 126,624 half-hours (to 2026-11-20 23:30), the grid that `redirect` sets; the default grid has 121,248."""
    ext = pytest.importorskip("e00667_extend")
    common = pytest.importorskip("e00087_common")
    prep = pytest.importorskip("e00397_prep")
    # `redirect` changes all four of these. monkeypatch restores each one after the test.
    for module, name in (
        (common, "GRID_END"),
        (common, "ERA5_PATH"),
        (common, "OBSERVED_PATH"),
        (prep, "SUMMARY_DIR"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    assert len(common.grid_times()) == 121_248
    assert datetime(2026, 7, 31, 23, 30, tzinfo=UTC) == common.GRID_END
    ext.redirect()
    assert len(common.grid_times()) == src.N_GRID == 126_624


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
    assert pattern.search(python + "e00725_t3_candidate.py T3-C 2025-11 0 --device cuda")
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


CONTRACT = datetime(2026, 3, 10, 12, tzinfo=UTC)


def test_kl8_l_and_e_fail_with_one_negative_margin() -> None:
    """A single margin of -0.5 h fails `L` and `E`."""
    for name in ("L", "E"):
        lines = wl.lines_for(contract=CONTRACT, rule_name=name)  # ty: ignore[invalid-argument-type]
        assert ck.kl8_arm_passes(name=name, lines=lines)[0]
        bad = [dataclasses.replace(lines[0], margin_hours=-0.5), *lines[1:]]
        passes, negative = ck.kl8_arm_passes(name=name, lines=bad)
        assert not passes
        assert len(negative) == 1


def test_kl8_z_needs_its_negative_lines() -> None:
    """`Z` passes while it shows negative lines, and fails when they are filtered out (the check can detect a look-ahead)."""
    lines = wl.lines_for(contract=CONTRACT, rule_name="Z")
    assert ck.kl8_arm_passes(name="Z", lines=lines)[0]
    kept = [x for x in lines if x.margin_hours >= 0]
    assert not ck.kl8_arm_passes(name="Z", lines=kept)[0]


def test_every_negative_z_line_overlaps_the_cells_that_the_live_rule_changes() -> None:
    """Each negative line of `Z` holds at least one cell whose source `L` changes, so the look-ahead lies in the changed cells."""
    changed = rule.changed_cells(rule="L")
    negative = [x for x in wl.lines_for(contract=CONTRACT, rule_name="Z") if x.margin_hours < 0]
    assert negative
    for line in negative:
        columns = list(wl.GROUPS[line.group])
        block = changed[line.age - rule.FIRST_AGE, line.first : line.last + 1][:, columns]
        assert block.any(), line.text


def test_run_sh_exports_the_environment(tmp_path: Path) -> None:
    """`run.sh` sets the frozen checkout first on PYTHONPATH, the dummy NGED variables, and one thread for every library."""
    text = RUN_SH.read_text()
    assert 'export PYTHONPATH="$EXTRA_PYTHONPATH:$here"' in text
    for name in (
        "DATA_PATH_INTERNAL", "NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "POLARS_MAX_THREADS=1", "RAYON_NUM_THREADS=1",
    ):  # fmt: skip
        assert f"export {name}" in text or f" {name}" in text, name
    assert text.index("export PYTHONPATH") < text.index("exec systemd-run")
    assert text.index("load average") < text.index("GB available") or text.index(
        "MAX_LOAD_WAIT_S"
    ) < text.index("MIN_AVAILABLE_GB")
    # run the script's exports for real, with `exec systemd-run` replaced by a printout of the environment
    fake = text.replace("exec systemd-run", "env | sort; exit 0 #")
    script = tmp_path / "run.sh"
    script.write_text(fake)
    frozen = tmp_path / "00713-frozen"
    frozen.mkdir()
    out = subprocess.run(
        ["bash", str(script), "e00725_fit.py", "L"], capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "DATA_PATH_INTERNAL": "/d", "EXTRA_PYTHONPATH": str(frozen), "MIN_AVAILABLE_GB": "0", "MAX_LOAD": "100000"},
    )  # fmt: skip
    env = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    assert env["PYTHONPATH"].startswith(str(frozen) + ":")
    assert env["DATA_PATH_INTERNAL"] == "/d"
    assert env["NGED_S3_BUCKET_URL"] == "https://example.com"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["POLARS_MAX_THREADS"] == env["RAYON_NUM_THREADS"] == "1"


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_limit_gpu_memory_passes_an_indexed_device(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """`set_per_process_memory_fraction` rejects the bare string `cuda`, so the cap must pass a device index."""
    torch = pytest.importorskip("torch")
    calls: list[tuple[str, object]] = []

    class Props:
        total_memory = 24 * 1024**3

    def properties(index: object) -> Props:
        calls.append(("properties", index))
        return Props()

    def fraction(value: float, index: object) -> None:
        calls.append(("fraction", index))
        assert isinstance(index, int)
        assert value == pytest.approx(0.25)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", fraction)
    fz.limit_gpu_memory(device=device)
    expected = 1 if device == "cuda:1" else 0
    assert calls == [("properties", expected), ("fraction", expected)]
    calls.clear()
    fz.limit_gpu_memory(device="cpu")
    assert calls == []

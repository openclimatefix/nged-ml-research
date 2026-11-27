# ruff: noqa: PT018
"""Tests of `e00733_cli`: exit codes, failure records, and that a stage with no function is refused
loudly."""

import argparse
import json
from pathlib import Path

import e00733_cli as cli
import e00733_stages as st
import pytest
from e00733_leakage import CorrectnessFailureError
from test_e00733_stages import DIGEST, _ready


def _args(stage: str, **kw: object) -> argparse.Namespace:
    return argparse.Namespace(**{"stage": stage, "interim": False, "measured": None, **kw})


def test_stage_without_function_is_refused_with_exit_2(tmp_path: Path) -> None:
    """A stage with no function in this build exits 2 (never 0, never 1)."""
    _ready(tmp_path, through="timing")
    (tmp_path / "scores" / st.TEST_MARKER).touch()
    assert cli.run_stage(args=_args("figures"), root=tmp_path, digest=DIGEST) == 2


@pytest.mark.parametrize(
    ("exc", "code"),
    [(st.StoppedError("x"), 3), (CorrectnessFailureError("x"), 4), (RuntimeError("x"), 5)],
)
def test_exit_codes_and_failure_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: Exception, code: int
) -> None:
    """A stopped stage exits 3, a correctness failure 4 (and writes `invalid.json`), any other
    failure 5; each leaves message and traceback."""
    _ready(tmp_path, through="timing")
    (tmp_path / "scores" / st.TEST_MARKER).touch()

    def boom(_a: argparse.Namespace, _r: Path) -> None:
        raise exc

    monkeypatch.setitem(cli.STAGE_FUNCTIONS, "figures", boom)
    assert cli.run_stage(args=_args("figures"), root=tmp_path, digest=DIGEST) == code
    rec = json.loads((tmp_path / "scores" / "failure_figures.json").read_text())
    assert rec["message"] == "x" and "Traceback" in rec["traceback"]
    assert (tmp_path / st.INVALID_FILE).exists() == (code == 4)


def test_guard_refusal_and_chain_status_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guard refusal exits 2; a chain whose status says failed exits 5."""
    assert cli.run_stage(args=_args("fit"), root=tmp_path, digest=DIGEST) == 2
    _ready(tmp_path, through="timing")
    monkeypatch.setitem(cli.STAGE_FUNCTIONS, "chain", lambda _a, _r: {"state": "failed"})
    assert cli.run_stage(args=_args("chain"), root=tmp_path, digest=DIGEST) == 5
    monkeypatch.setitem(cli.STAGE_FUNCTIONS, "chain", lambda _a, _r: {"state": "complete"})
    assert cli.run_stage(args=_args("chain"), root=tmp_path, digest=DIGEST) == 0


def test_timing_stage_writes_the_decision(tmp_path: Path) -> None:
    """The timing stage reads the measured numbers and writes `timing.json`; a counted-work overrun
    exits 3."""
    _ready(tmp_path, through="smoke")
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            {
                "seconds_per_real_fit": 288.0,
                "seconds_per_synthetic_fit": 1080.0,
                "peak_gpu_gb": 3.5,
                "peak_rss_gb": 8.0,
            }
        )
    )
    assert cli.run_stage(args=_args("timing", measured=str(m)), root=tmp_path, digest=DIGEST) == 0
    assert st.timing_passed(root=tmp_path)
    m.write_text(
        json.dumps(
            {
                "seconds_per_real_fit": 9999.0,
                "seconds_per_synthetic_fit": 1080.0,
                "peak_gpu_gb": 3.5,
                "peak_rss_gb": 8.0,
            }
        )
    )
    other = tmp_path / "o"
    other.mkdir()
    _ready(other, through="smoke")
    assert cli.run_stage(args=_args("timing", measured=str(m)), root=other, digest=DIGEST) == 3


def test_score_stage_writes_the_test_marker_only_after_the_choice_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The score stage needs finished predictions; the marker is written through
    `assert_no_test_choice` before scoring reads a test row."""
    _ready(tmp_path, through="timing")
    (tmp_path / "predictions" / "FU").mkdir(parents=True)
    (tmp_path / "predictions" / "FU" / "a.parquet").write_text("")
    st.write_chain_status(root=tmp_path, state="complete")
    import os
    import time

    old = time.time() - 100
    os.utime(tmp_path / "limits.json", (old, old))
    monkeypatch.setitem(
        cli.STAGE_FUNCTIONS,
        "score",
        lambda _a, r: (
            None if st.marker_exists(root=r) else (_ for _ in ()).throw(AssertionError("no marker"))
        ),
    )
    assert cli.run_stage(args=_args("score"), root=tmp_path, digest=DIGEST) == 0
    assert st.marker_exists(root=tmp_path)


def test_an_unexpected_error_in_the_setup_exits_5_never_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """An import error before the stage runs exits 5, which is not read as `slot busy`."""

    def broken() -> None:
        msg = "no module named torch"
        raise ImportError(msg)

    monkeypatch.setattr(cli, "install_runtime", broken)
    assert cli.main(["guard"]) == cli.st.EXIT_FAILED == 5


def test_run_sh_lists_exactly_the_stages_of_the_cli() -> None:
    """The wrapper's usage line names every stage of the cli, and no stage of another experiment."""
    text = (Path(cli.__file__).parent / "run.sh").read_text()
    usage = next(line for line in text.splitlines() if line.startswith("# Usage:"))
    listed = usage.split("<")[2].split(">")[0].split("|")
    assert tuple(listed) == cli.STAGES
    for stale in ("neighbours", "overlays", "solar_diag", "panels"):
        assert stale not in text


def test_the_cpu_stages_of_run_sh_are_the_stages_that_default_to_the_cpu() -> None:
    """`run.sh` skips the GPU check for exactly the stages of `CPU_ONLY_STAGES`, and `main` gives
    them the device `cpu` (the parser leaves the device unset)."""
    import re

    text = (Path(__file__).parent.parent / "run.sh").read_text()
    match = re.search(r"case \"\$\{1:-\}\" in ([a-z_ |]+)\) uses_gpu=0", text)
    assert match is not None
    in_run_sh = {name.strip() for name in match.group(1).split("|")}
    assert in_run_sh == set(st.CPU_ONLY_STAGES)
    assert set(st.CPU_ONLY_STAGES) <= set(cli.STAGES)
    for stage in cli.STAGES:
        parsed = cli.parser().parse_args([stage])
        assert parsed.device is None
    assert "attention" in st.CPU_ONLY_STAGES


def test_a_cpu_stage_refuses_the_gpu(tmp_path: Path) -> None:
    """`--device cuda` on a stage that `run.sh` runs without the GPU check exits 2."""
    _ready(tmp_path, through="timing")
    (tmp_path / "scores" / st.TEST_MARKER).touch()
    assert cli.run_stage(args=_args("attention", device="cuda"), root=tmp_path, digest=DIGEST) == 2

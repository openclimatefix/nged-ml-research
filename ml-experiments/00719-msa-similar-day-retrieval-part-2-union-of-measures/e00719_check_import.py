"""Experiment 719 check: part 1's code is found, and part 2's wrappers run part 1's self-tests."""

import subprocess
import sys
from pathlib import Path

import e00719_part1

HERE = Path(__file__).resolve().parent


def main() -> int:
    """Run the report self-test through the wrapper."""
    assert (e00719_part1.PART_1_FOLDER / "e00718_decision.py").is_file()
    for script in ("e00719_report.py",):
        result = subprocess.run(
            [sys.executable, str(HERE / script), "self-test"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout[-500:] + result.stderr[-500:]
        assert "all checks passed" in result.stdout, result.stdout[-300:]
    for command, want in (
        (["headline", "--seeds", "2"], ["headline", "2", "--seeds", "2"]),
        (["headline", "1"], ["headline", "1"]),
        (["self-test"], ["self-test"]),
    ):
        assert e00719_part1.with_part(args=command) == want, command
    print(f"e00719_check_import: part 1 found at {e00719_part1.PART_1_FOLDER}; all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

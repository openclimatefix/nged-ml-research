"""Fail if a tracked Python or shell script hard-codes a path under `/home/`.

Scripts resolve data through `studies.sources`, which honours `DATA_PATH_INTERNAL`, so that moving
the data folder changes one environment variable rather than every script.
"""

import subprocess
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "tests/test_no_absolute_home_paths.py",
        "tests/studies/test_capacity_labeller.py",
        "tests/studies/test_capacity_llm.py",
    }
)
"""This file names the prefix it searches for; the two capacity tests use it in sample strings."""
FORBIDDEN: Final[str] = "/home/"


def _tracked_scripts() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "*.py", "*.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [name for name in listing.splitlines() if name not in ALLOWED]


def test_no_tracked_script_hard_codes_a_home_path() -> None:
    offenders = [
        name
        for name in _tracked_scripts()
        if (REPO_ROOT / name).is_file() and FORBIDDEN in (REPO_ROOT / name).read_text()
    ]
    assert offenders == [], f"use studies.sources constants or $DATA_PATH_INTERNAL in: {offenders}"

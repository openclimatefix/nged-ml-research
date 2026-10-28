"""Repo-root pytest configuration.

``--run-studies`` gates the slow tests under ``tests/studies``. A plain ``uv run pytest``, which
local development and the per-pull-request CI use, skips them.
"""

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

# Must be set here, at import time: Polars reads `POLARS_MAX_THREADS` once, the first time it is
# imported, and the cap stops `-n auto` workers from each starting a thread per core.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

_STUDIES_TESTS_DIR = Path(__file__).parent / "tests" / "studies"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-studies`` opt-in flag."""
    parser.addoption(
        "--run-studies",
        action="store_true",
        default=False,
        help="Run the slow tests under tests/studies.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip the slow studies tests unless ``--run-studies`` was passed."""
    skip_studies = pytest.mark.skip(reason="slow studies test; pass --run-studies")
    for item in items:
        # A path check, not a `studies` keyword: the `tests/studies` directory node is itself a
        # keyword of every test beneath it.
        if item.path.is_relative_to(_STUDIES_TESTS_DIR) and not config.getoption("--run-studies"):
            item.add_marker(skip_studies)

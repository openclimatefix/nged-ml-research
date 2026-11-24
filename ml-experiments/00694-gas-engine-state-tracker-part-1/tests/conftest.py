"""Pytest configuration of the tests of the engine-figure guard.

`ml-experiments/`, which holds `engine_figure_guard`, is put on `sys.path` here. The tests are
skipped unless `--run-studies` is passed, as the tests under `tests/studies` are.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

_TESTS = Path(__file__).parent
sys.path.insert(0, str(_TESTS.parents[1]))


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip these tests unless `--run-studies` was passed."""
    skip = pytest.mark.skip(reason="slow studies test; pass --run-studies")
    for item in items:
        if item.path.is_relative_to(_TESTS) and not config.getoption("--run-studies"):
            item.add_marker(skip)

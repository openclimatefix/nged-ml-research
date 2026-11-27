"""Pytest configuration of the tests of experiment 722.

The experiment folder, which holds the scripts, is put on `sys.path` here, so the tests import
`e00722_core` and its siblings by bare name without a `pyproject.toml` entry. The tests are fast and
need no `torch`, and they are skipped unless `--run-studies` is passed, as the tests under
`tests/studies` are.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).parent
_FOLDER = _TESTS.parent
sys.path.insert(0, str(_FOLDER))


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip the tests of experiment 722 unless `--run-studies` was passed."""
    skip = pytest.mark.skip(reason="slow studies test; pass --run-studies")
    for item in items:
        if item.path.is_relative_to(_TESTS) and not config.getoption("--run-studies"):
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(722)

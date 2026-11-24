"""Pytest configuration of the tests of experiment 723.

The experiment folder, and experiment 722's folder whose modules the experiment imports by bare
name, are put on `sys.path` here, so no `pyproject.toml` entry is needed. The tests run on
synthetic arrays only, need no `torch`, and are skipped unless `--run-studies` is passed, as the
tests under `tests/studies` are.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).parent
_FOLDER = _TESTS.parent
sys.path.insert(0, str(_FOLDER))
sys.path.insert(0, str(_FOLDER.parent / "00722-msa-mva-bsp-closure-and-fast-hint"))


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip the tests of experiment 723 unless `--run-studies` was passed."""
    skip = pytest.mark.skip(reason="slow studies test; pass --run-studies")
    for item in items:
        if item.path.is_relative_to(_TESTS) and not config.getoption("--run-studies"):
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(723)

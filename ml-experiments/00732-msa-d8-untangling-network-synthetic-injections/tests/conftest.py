"""Pytest configuration of the tests of experiment 732.

The experiment folder (and experiment 730's, through `e00732_paths`) is put on `sys.path` here, so
the tests import `e00732_core` and its siblings by bare name. The tests need no data file and no
GPU, and they are skipped unless `--run-studies` is passed, as the tests under `tests/studies` are.
The tests of the networks need `torch` and skip without it: they call `pytest.importorskip("torch")`
before importing any module that imports torch.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).parent
_FOLDER = _TESTS.parent
sys.path.insert(0, str(_FOLDER))
sys.path.insert(0, str(_TESTS))

import e00732_paths  # noqa: E402  (puts experiments 729 and 730 on sys.path)

e00732_paths.pin_threads()


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip the tests of experiment 732 unless `--run-studies` was passed."""
    skip = pytest.mark.skip(reason="slow studies test; pass --run-studies")
    for item in items:
        if item.path.is_relative_to(_TESTS) and not config.getoption("--run-studies"):
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded generator."""
    return np.random.default_rng(732)

"""Pytest configuration of the tests of experiment 733.

The experiment folder is put on `sys.path` here, so the tests import `e00733_core` and its
siblings by bare name. The tests need no data file and no GPU, and they are skipped unless
`--run-studies` is passed. The tests that need `torch` skip without it.
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

import e00733_paths  # noqa: E402  (puts the sibling experiments on sys.path)

e00733_paths.pin_threads()


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip the tests of experiment 733 unless `--run-studies` was passed."""
    skip = pytest.mark.skip(reason="slow studies test; pass --run-studies")
    for item in items:
        if item.path.is_relative_to(_TESTS) and not config.getoption("--run-studies"):
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded generator."""
    return np.random.default_rng(733)


@pytest.fixture(scope="session", autouse=True)
def _data_folder_gains_no_file() -> Iterable[None]:
    """Fail the session when a test leaves a file in this experiment's real data folder.

    The folder is the one that `DATA_PATH_INTERNAL` names, which holds the real (or the scratch)
    run. Every test writes under `tmp_path`, so the listing must be the same after the session.
    """
    import os

    folder = e00733_paths.data_root() if os.environ.get("DATA_PATH_INTERNAL") else None

    def listing() -> list[tuple[str, int]]:
        if folder is None or not folder.exists():
            return []
        return sorted((str(p.relative_to(folder)), p.stat().st_mtime_ns) for p in folder.rglob("*"))

    before = listing()
    yield
    after = listing()
    assert after == before, f"the tests changed the data folder: {set(after) ^ set(before)}"

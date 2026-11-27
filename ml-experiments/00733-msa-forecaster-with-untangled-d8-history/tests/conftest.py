"""Pytest configuration of the tests of experiment 733.

The experiment folder is put on `sys.path` here, so the tests import `e00733_core` and its
siblings by bare name. The tests need no data file and no GPU, and they are skipped unless
`--run-studies` is passed. The tests that need `torch` are not collected without it (see
`NEEDS_TORCH`).
"""

import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np
import pytest

_TESTS = Path(__file__).parent
_FOLDER = _TESTS.parent
sys.path.insert(0, str(_FOLDER))
sys.path.insert(0, str(_TESTS))

import e00733_paths  # noqa: E402  (puts the sibling experiments on sys.path)

e00733_paths.pin_threads()

NEEDS_TORCH: Final = (
    "test_e00733_adapters.py",
    "test_e00733_cli.py",
    "test_e00733_figures.py",
    "test_e00733_fit.py",
    "test_e00733_model.py",
    "test_e00733_posthoc.py",
    "test_e00733_posthoc_guard.py",
    "test_e00733_predict.py",
    "test_e00733_predictions.py",
    "test_e00733_runtime.py",
    "test_e00733_scoring_synth.py",
    "test_e00733_series.py",
    "test_e00733_signdata.py",
    "test_e00733_smoke.py",
    "test_e00733_smokerun.py",
    "test_e00733_stages.py",
    "test_e00733_train.py",
    "test_e00733_store.py",
    "test_e00733_scoring.py",
    "test_e00733_score.py",
    "test_e00733_report.py",
    "test_e00733_gallery2.py",
    "test_e00733_core.py",
    "test_e00733_baselines.py",
)
"""The test modules that import `torch`, directly or through the modules of the experiment."""

collect_ignore: list[str] = [] if importlib.util.find_spec("torch") else list(NEEDS_TORCH)
"""Without `torch`, as in CI, pytest does not collect the modules in `NEEDS_TORCH`."""


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

"""Pytest configuration of the tests of experiment 721.

The experiment folder, which holds the scripts, is put on `sys.path` here, so the tests import
`e00721_core` and its siblings by bare name without a `pyproject.toml` entry. The tests need no
data file and no GPU, and they are skipped unless `--run-studies` is passed, as the tests under
`tests/studies` are. The tests of the network need `torch` and the folders of experiments 713 and
716 on `PYTHONPATH` (`run.sh` sets them), and are skipped without them.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).parent
_FOLDER = _TESTS.parent
sys.path.insert(0, str(_FOLDER))

_WORKTREES = _FOLDER.parents[2]
_SIBLINGS = (
    "frozen-713-7c29114a8/ml-experiments/00713-msa-axial-attention-demand-network-part-1",
    "frozen-713-7c29114a8/ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand",
    "msa-f7-pv-part-1/ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate",
    "msa-mva-to-mw-rung-0/ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters",
)
for _sibling in _SIBLINGS:
    _path = _WORKTREES / _sibling
    if _path.is_dir():
        sys.path.append(str(_path))
sys.path.extend(
    str(p) for p in sorted(_FOLDER.parent.glob("000[0-9][0-9]-*")) if p.name[:5] == "00087"
)
sys.path.extend(str(p) for p in sorted(_FOLDER.parent.glob("00506-*")))
_559 = sorted((_WORKTREES / "frozen-713-7c29114a8" / "ml-experiments").glob("00559-*"))
sys.path.extend(str(p) for p in _559)


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip the tests of experiment 721 unless `--run-studies` was passed."""
    skip = pytest.mark.skip(reason="slow studies test; pass --run-studies")
    for item in items:
        if item.path.is_relative_to(_TESTS) and not config.getoption("--run-studies"):
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded generator."""
    return np.random.default_rng(721)

# ruff: noqa: E501
"""Tests of `e00733_paths`: the data folders, the write guard, and the 732 locator."""

from pathlib import Path

import e00733_paths as paths
import pytest


def test_recipe_numbers_load() -> None:
    """The recipe of experiment 713 names 41 experiments, parsed without importing it."""
    nums = paths.recipe_numbers()
    assert len(nums) == 41
    assert all(isinstance(n, int) for n in nums)


def test_data_folders_are_distinct_and_under_the_internal_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every folder lies under `DATA_PATH_INTERNAL/ml-experiments`.

    And this experiment's differs from the earlier ones.
    """
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    roots = {
        paths.data_root(),
        paths.root_713(),
        paths.root_716(),
        paths.root_730(),
        paths.root_732(),
    }
    assert len(roots) == 5
    assert all(tmp_path / "ml-experiments" in r.parents for r in roots)


def test_forbidden_roots_exclude_this_experiment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The forbidden write roots are the six earlier folders and never this experiment's."""
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    forbidden = paths.forbidden_write_roots()
    assert len(forbidden) == 6
    assert paths.data_root() not in forbidden


def test_private_dir_is_created_under_the_given_root(tmp_path: Path) -> None:
    """The private folder is created under an explicit root, so a test never touches the shared data."""
    assert paths.private_dir(root=tmp_path) == tmp_path / "PRIVATE"
    assert (tmp_path / "PRIVATE").is_dir()
    assert paths.table_path(root=tmp_path).parent == paths.channels_dir(root=tmp_path)
    assert paths.rerun_dir(root=tmp_path).parent == paths.channels_dir(root=tmp_path)


def test_thread_pinning_is_exposed() -> None:
    """The thread helpers of experiment 730 are re-exported."""
    assert callable(paths.pin_threads)
    assert paths.MAX_SLOT_CORES >= 1

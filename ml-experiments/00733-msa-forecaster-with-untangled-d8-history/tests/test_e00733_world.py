"""Tests of `e00733_world`: the cached files, the ensemble-run lookup, and the sun-up threshold."""

from pathlib import Path

import e00733_world as world
import numpy as np
import pytest


def test_world_roundtrip(tmp_path: Path) -> None:
    """A written bank reads back with its extra fields, and the meta file round-trips."""
    (tmp_path / "world").mkdir()
    world.write_world(root=tmp_path, kind="validation", segments=[], extra={"n_kept": 3})
    segments, extra = world.read_world(root=tmp_path, kind="validation")
    assert segments == []
    assert extra == {"n_kept": 3}
    world.write_meta(root=tmp_path, payload={"mean": np.float64(0.5)})
    assert world.read_meta(root=tmp_path) == {"mean": 0.5}


def test_missing_world_file_raises(tmp_path: Path) -> None:
    """Reading a bank that was never written fails rather than returning an empty world."""
    (tmp_path / "world").mkdir()
    with pytest.raises(FileNotFoundError):
        world.read_world(root=tmp_path, kind="test")


def test_epoch_of_is_days_since_1970() -> None:
    """The ensemble-run epoch day of 1970-01-11 is 10."""
    assert world.epoch_of(date64=np.datetime64("1970-01-11")) == 10

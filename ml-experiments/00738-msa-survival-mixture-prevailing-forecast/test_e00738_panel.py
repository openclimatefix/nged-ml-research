"""Tests of the panel guards, the prediction panels, and the fold shifts."""
# ruff: noqa: D103, E402, PT018

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
# the tests never write to the experiment's data folder
os.environ.setdefault("E00738_OUT", str(Path(tempfile.gettempdir()) / "scratch" / "e738_tests"))
os.environ.setdefault("E00738_SCRATCH", "1")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 378, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))

import e00738_common as common
import e00738_panel as panel
import e00738_predict as pred
import e00738_world as world


def test_code_hash_guard_of_experiment_713() -> None:
    panel.assert_hash(actual=common.CODE_HASH_713)
    with pytest.raises(panel.CodeHashError, match="7c29114a8"):
        panel.assert_hash(actual="0000000000000000")


def test_panel_names_are_real_sixteen_passes_and_the_resampled_world() -> None:
    names = pred.panel_names()
    assert names[0] == "real"
    assert len(names) == 1 + 16 + 1
    assert names[-1] == "resampled00"
    assert len(set(names)) == len(names)


def test_panel_path_is_under_the_arm_and_the_fold(tmp_path: Path) -> None:
    p = pred.panel_path(arm="LP", month="2025-11", seed=0, name="pass03", out=tmp_path)
    assert tmp_path in p.parents
    assert "LP" in p.parts and p.name.startswith("pass03")


def test_all_panel_shifts_covers_every_panel_with_sixteen_shifts() -> None:
    is_mw = np.array([True] * 8 + [False] * 8)
    shifts = pred.all_panel_shifts(
        month="2025-11", month_start=60_000, window_end=60_000 + 10**5, is_mw=is_mw,
        real_days=np.array([0.5, 2.0, 9.0]),
    )  # fmt: skip
    assert set(shifts) == {n for n in pred.panel_names() if n != "real"}
    assert all(len(v) == 16 for v in shifts.values())
    assert {s.family for s in shifts["pass00"]} == {"train"}
    assert {s.family for s in shifts[f"pass{common.HELD_FIRST}"]} == {"held_out"}
    assert {s.family for s in shifts["resampled00"]} == {"resampled"}
    again = pred.all_panel_shifts(
        month="2025-11", month_start=60_000, window_end=60_000 + 10**5, is_mw=is_mw,
        real_days=np.array([0.5, 2.0, 9.0]),
    )  # fmt: skip
    assert again == shifts, "the draws depend only on the keys, never on the data"
    assert world.family_of(r=0) == "train"

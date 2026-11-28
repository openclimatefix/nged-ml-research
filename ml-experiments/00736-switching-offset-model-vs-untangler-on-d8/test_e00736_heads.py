"""Tests of the step-head recipe on invented arrays (no private data)."""
# ruff: noqa: D103, E402

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
sys.path.insert(0, str(Path(__file__).parent))
import e00736_common as common
import e00736_heads as heads


def test_isotonic_map_interpolates_between_knots() -> None:
    xs, ys = np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.2, 1.0])
    got = heads.iso_apply(p=np.array([0.25, 0.75]), xs=xs, ys=ys)
    assert got == pytest.approx([0.1, 0.6])


def test_recalibration_match_is_one_to_one_within_one_half_hour() -> None:
    called = np.zeros((1, 20), bool)
    true = np.zeros((1, 20), bool)
    called[0, [3, 4, 10]] = True
    true[0, [4, 11]] = True
    cok, tok = heads.recal_match(called=called, true=true)
    assert tok[0].sum() == 2
    assert cok[0].sum() == 2


def test_found_by_head_uses_the_sign_and_the_6_hour_tolerance() -> None:
    truth = pl.DataFrame(
        {
            "seg": [0, 0],
            "sign": [1, -1],
            "size_points": [20.0, 20.0],
            "start_hh": [100, 400],
            "end_hh": [200, 500],
        }
    )
    calls = pl.DataFrame({"seg": [0, 0], "time": [111, 395], "sign": [1, 1]})
    found = heads.found_by_head(truth=truth, calls=calls, tolerance=12)
    assert found.tolist() == [True, False]
    end = heads.found_by_head(
        truth=truth,
        calls=pl.DataFrame({"seg": [0], "time": [205], "sign": [-1]}),
        tolerance=12,
        time_col="end_hh",
        end=True,
    )
    assert end.tolist() == [True, False]


def test_merged_calls_take_the_sign_of_the_size_output() -> None:
    out = heads.HeadOutputs(
        arm="U-noBSP",
        step_p=np.array([0.9, 0.9, 0.9, 0.0]),
        size=np.array([1.0, 2.0, -3.0, 0.0]),
        row_seg=np.zeros(4, dtype=np.int64),
        row_date=np.full(4, 19000, dtype=np.int64),
        row_hh=np.array([0, 1, 20, 21]),
    )
    frozen = {
        "arms": {
            "U-noBSP": {
                "isotonic_knots_x": [0.0, 1.0],
                "isotonic_knots_y": [0.0, 1.0],
                "maps": {"isotonic": {"f1_rule": {"cal_threshold": 0.5}}},
            }
        }
    }
    calls = heads.merged_calls(out=out, frozen=frozen, first_day={0: 19000})
    assert calls["time"].to_list() == [0, 20]
    assert calls["sign"].to_list() == [1, -1]


def test_frozen_file_digest_is_the_one_in_the_plan() -> None:
    assert len(common.FROZEN_THRESHOLDS_SHA256) == 64
    path = common.RECAL / "frozen_thresholds.json"
    if not path.exists():
        pytest.skip("the recalibration report's folder is private")
    assert common.sha256_of(path=path) == common.FROZEN_THRESHOLDS_SHA256


def test_a_segment_with_no_calls_finds_nothing() -> None:
    truth = pl.DataFrame(
        {"seg": [3], "sign": [1], "size_points": [20.0], "start_hh": [100], "end_hh": [200]}
    )
    calls = pl.DataFrame({"seg": [0], "time": [100], "sign": [1]})
    assert heads.found_by_head(truth=truth, calls=calls, tolerance=12).tolist() == [False]

"""Known-answer test of `score_all` on invented segments, and tests of its helper checks."""
# ruff: noqa: D103, E402

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _n in ("732", "734", "653", "660"):
    for _d in HERE.parent.glob(f"00{_n}-*"):
        sys.path.append(str(_d))
import e00736_decode as decode
import e00736_score as score
import e00736_world as world

DAYS = 14
HEAD = "U-noBSP"


def segment() -> SimpleNamespace:
    """Fourteen dense days, the first and last of them padding, so 12 base days."""
    shape = (DAYS, 48)
    return SimpleNamespace(
        x=np.zeros(shape, dtype=np.float32),
        valid=np.ones(shape, dtype=bool),
        cal_index=np.array([-1, *range(DAYS - 2), -1]),
    )


def transfer_row(*, seg: int, start: int, end: int, sign: int, **over: Any) -> dict[str, Any]:
    hours = (end - start) / 2.0
    row = {
        "seg": seg,
        "month_idx": seg,
        "draw": 0,
        "start_hh": start,
        "end_hh": end,
        "start_panel": start + 1,
        "end_panel": end + 1,
        "size_points": 20.0 * sign,
        "sign": sign,
        "units": 5.0,
        "hours": hours,
        "mirrored": False,
        "has_arrays": False,
        "band": world.band_of(hours=hours),
        "big": True,
        "counted": hours >= 24.0,
        "gap": False,
    }
    return row | over


def calls(*rows: tuple[int, int, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "seg": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "sign": [r[2] for r in rows],
            "end": [-1] * len(rows),
            "amp_mw": [1.0] * len(rows),
        },
        schema={
            "seg": pl.Int64,
            "time": pl.Int64,
            "sign": pl.Int64,
            "end": pl.Int64,
            "amp_mw": pl.Float64,
        },
    )


def build() -> dict[str, Any]:
    segments = [segment(), segment()]
    rows = [
        transfer_row(seg=0, start=60, end=108, sign=1, mirrored=True),
        transfer_row(seg=0, start=140, end=188, sign=-1),
        transfer_row(seg=1, start=60, end=108, sign=1, mirrored=True, has_arrays=True),
        transfer_row(seg=1, start=140, end=188, sign=1),
        transfer_row(seg=1, start=200, end=230, sign=-1, big=False, units=1.0),
        transfer_row(seg=1, start=260, end=280, sign=1, gap=True),
    ]
    table = pl.DataFrame(rows)
    table = score.with_placebo(table=table, segments=segments, d8_power=np.ones(1000))
    head = calls((0, 62, 1), (0, 140, 1), (1, 60, 1), (1, 142, 1))
    offset_nbr = calls(
        (0, 110, -1), (0, 140, -1), (0, 398, 1), (1, 100, 1), (1, 140, 1), (1, 205, -1), (1, 262, 1)
    )
    layout = pl.DataFrame({"seg": [0, 1], "month": ["a", "b"], "month_idx": [0, 1], "draw": [0, 0]})
    return {
        "table": table,
        "head_calls": {HEAD: head},
        "offset": {"H-nbr": offset_nbr, "H-own": calls((0, 500, 1))},
        "segments": segments,
        "layout": layout,
    }


def test_score_all_reproduces_hand_counted_pairs_false_calls_and_placebo() -> None:
    result = score.score_all(**build(), gates_passed=True)
    assert result["n_transfers"] == 5  # the gap transfer is not scored
    # counted transfers are the first four: the head finds #0, #2, #3; H-nbr finds #1 and #3
    cell = result["counted_cell"]
    assert cell["paired"] == {"both": 1, "only_head": 2, "only_offset": 1, "neither": 0}
    assert cell["c1"] == pytest.approx(-0.25)
    assert cell["sign_test_p"] == pytest.approx(1.0)
    assert cell["months_without_discordant"] == 0
    assert result["verdict"] == "inconclusive"  # 4 counted transfers is fewer than 20
    # false calls: the head's (0, 140, +) only; H-nbr's (0, 398, +) and (1, 100, +). The H-nbr call
    # (1, 262, +) lies at the start of the gap transfer and is excused by it. Each segment has 12
    # base days.
    false = result["veto_v1"]["false_calls"]
    assert false[HEAD]["false"] == 1.0
    assert false[HEAD]["calls_on_base_days"] == 4.0
    assert false[HEAD]["base_days"] == 24.0
    assert false["H-nbr"]["false"] == 2.0
    assert false["H-nbr"]["calls_on_base_days"] == 7.0
    assert false["H-nbr"]["per_100_days"] == pytest.approx(100.0 * 2 / 24)
    assert result["veto_v1"]["passes"] is False
    # the placebo moves every start 336 half-hours later; only H-nbr's call at 398 matches (#0)
    assert result["placebo_starts_kept"] == 5
    assert result["placebo"]["H-nbr"]["counted"]["found"] == 1.0
    assert result["placebo"]["H-nbr"]["counted"]["n"] == 4.0
    assert result["placebo"][HEAD]["counted"]["found"] == 0.0
    band = result["shares_by_band_big"]
    assert band[HEAD]["24 to 96 hours"]["found"] == 3.0
    assert band["H-nbr"]["24 to 96 hours"]["found"] == 2.0
    secondary = result["paired_secondary"]
    assert secondary["counted_mirrored"]["only_head"] == 2
    assert secondary["counted_not_mirrored"]["both"] == 1
    assert secondary["counted_with_engine_or_solar"]["only_head"] == 1


def test_score_all_with_a_failed_gate_is_invalid() -> None:
    assert score.score_all(**build(), gates_passed=False)["verdict"] == "invalid"


def test_the_placebo_drops_moved_starts_near_any_real_edge_including_dropped_transfers() -> None:
    seg = segment()
    rows = [
        transfer_row(seg=0, start=60, end=108, sign=1),
        # the moved start of the first transfer is 396; an edge at 400 is within 6 hours of it
        transfer_row(seg=0, start=392, end=400, sign=-1, gap=True),
        transfer_row(seg=0, start=140, end=188, sign=1),
    ]
    table = score.with_placebo(table=pl.DataFrame(rows), segments=[seg], d8_power=np.ones(1000))
    assert table["placebo_ok"].to_list() == [False, False, True]


def test_event_values_compare_amplitude_confidence_and_grade() -> None:
    base = pl.DataFrame(
        {
            "primary": ["D3", "D8"],
            "start": [10, 20],
            "end": [30, -1],
            "amp_mw": [1.5, -2.0],
            "confidence": [0.9, 0.8],
            "grade": ["a", "b"],
        }
    )
    assert decode.compare_event_values(mine=base, stored=base)["equal"]
    nudged = base.with_columns(amp_mw=pl.Series([1.5 * (1 + 1e-9), -2.0]))
    assert decode.compare_event_values(mine=nudged, stored=base)["equal"]
    off = base.with_columns(confidence=pl.Series([0.9, 0.81]))
    assert not decode.compare_event_values(mine=off, stored=base)["equal"]
    regraded = base.with_columns(grade=pl.Series(["a", "c"]))
    assert decode.compare_event_values(mine=regraded, stored=base)["n_grade_mismatches"] == 1
    assert not decode.compare_event_values(mine=base.head(1), stored=base)["equal"]


def test_a_nonzero_engine_array_with_a_zero_addition_raises() -> None:
    raw = np.array([0.0, 2.0, 0.0])
    keep = np.array([True, True, True])
    world._check_array_written(
        name="engine", raw=raw, written=np.array([0.0, -2.0, 0.0]), keep=keep, segment=3
    )
    with pytest.raises(ValueError, match="segment 3"):
        world._check_array_written(
            name="engine", raw=raw, written=np.zeros(3), keep=keep, segment=3
        )
    world._check_array_written(
        name="solar", raw=np.zeros(3), written=np.zeros(3), keep=keep, segment=3
    )


def test_the_design_probe_raises_without_an_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(**_kwargs: object) -> None:
        msg = "no arrays"
        raise TypeError(msg)

    monkeypatch.setitem(sys.modules, "e00734_layers", SimpleNamespace(apply_additions=broken))
    monkeypatch.setitem(sys.modules, "e00734_decode", SimpleNamespace(WorldDecoder=object))
    with pytest.raises(RuntimeError, match="no fallback"):
        world.probe_design_a()
    record = world.probe_design_a(allow_fallback=True)
    assert record["design"] == world.DESIGN_TRANSFERS_ONLY

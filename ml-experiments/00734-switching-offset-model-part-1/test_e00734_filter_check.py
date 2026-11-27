"""Tests of the read-only filter diagnostic of experiment 734, on synthetic call tables."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest
from e00653_segment import Event  # ty: ignore[unresolved-import]

sys.path.insert(0, str(Path(__file__).parent))
import e00734_events as ev
import e00734_filter_check as fc

NB: dict[str, tuple[str, ...]] = {"A": ("B",), "B": ("A",), "C": ("A",)}


def calls(*rows: tuple) -> pl.DataFrame:
    """Calls from `(primary, start, end, amp, confidence, cls)` tuples."""
    return pl.DataFrame(
        [
            {"primary": p, "start": s, "end": e, "amp_mw": a, "confidence": c, "cls": k}
            for p, s, e, a, c, k in rows
        ],
        schema={
            "primary": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "amp_mw": pl.Float64,
            "confidence": pl.Float64,
            "cls": pl.String,
        },
    )


def test_mirror_flags_agree_with_the_grade_rule_of_the_events_module() -> None:
    """Test the recomputed mirror flag equals `events.grade` on a mix of cases."""
    spec = [
        ("A", 1000, 2000, -2.0, "shift_no_partner"),
        ("B", 1036, 2000, 2.0, "shift_no_partner"),
        ("C", 1037, -1, 2.0, "shift_no_partner"),
        ("A", 5000, -1, -3.0, "shift_no_partner"),
        ("B", 1000, -1, -2.0, ev.LEVEL_CLS),
    ]
    events = [
        Event(p, s, None if e < 0 else e, a, abs(a), 1.0, 0.0, cls=k) for p, s, e, a, k in spec
    ]
    ev.grade(events=events, neighbours=NB)
    graded = [e.kind.startswith("mirrored") for e in events]
    table = calls(*[(p, s, e, a, 1.0, k) for p, s, e, a, k in spec])
    assert fc.mirror_flags(calls=table, neighbours=NB) == graded


def test_each_filter_keeps_the_calls_it_should() -> None:
    """Test F1 to F6 on a table of five calls."""
    day = fc.DAY
    table = calls(
        ("A", 0, day // 2, -2.0, 0.9, "x"),  # 12 h, returned, partner at B
        ("B", 10, day // 2, 2.0, 0.9, "x"),  # 12 h, returned, partner at A
        ("A", 5000, 5000 + 3 * day, 2.0, 0.8, "x"),  # 3 days, no partner
        ("C", 9000, -1, 2.0, 0.7, "x"),  # open
        ("C", 9100, 9100 + 2 * day, 2.0, 0.6, "x"),  # overlaps the open call, lower confidence
    )
    out = fc.filtered_calls(calls=table, neighbours=NB)
    assert out["F0"].height == 5
    assert out["F1"]["start"].to_list() == [5000, 9000, 9100]
    assert out["F2"]["start"].to_list() == [5000, 9000, 9100]
    assert sorted(out["F3"]["start"].to_list()) == [0, 10]
    assert sorted(out["F4"]["start"].to_list()) == [0, 10, 5000, 9000]
    assert sorted(out["F5"]["start"].to_list()) == [0, 10, 5000, 9100]
    assert out["F6"].height == 0


def test_non_overlapping_keeps_the_higher_confidence_call() -> None:
    """Test of two overlapping calls at one Primary the more confident stays."""
    table = calls(
        ("A", 0, 100, 1.0, 0.6, "x"), ("A", 50, 150, 1.0, 0.9, "x"), ("B", 60, 90, 1.0, 0.5, "x")
    )
    kept = fc.non_overlapping(calls=table)
    assert sorted(zip(kept["primary"].to_list(), kept["start"].to_list(), strict=True)) == [
        ("A", 50),
        ("B", 60),
    ]


def world_rows() -> pl.DataFrame:
    """Truth rows of two found events, one closed and mirrored, one open, plus one missed."""
    base = {
        "site": "A",
        "kind": "solo",
        "group": "g",
        "units": 5.0,
        "world": 1,
        "tau": 0.5,
        "right_class": False,
        "decoded_cls": "x",
        "decoded_kind": "temporary",
    }
    rows = [
        base
        | {
            "amp_mw": 2.0,
            "start": 100,
            "end": 100 + 10 * 48,
            "days": 10.0,
            "edge_truth": True,
            "found": True,
            "start_err_hh": 2.0,
            "end_err_hh": 0.0,
        },
        base
        | {
            "amp_mw": 3.0,
            "start": 20000,
            "end": -1,
            "days": -1.0,
            "edge_truth": False,
            "found": True,
            "start_err_hh": 0.0,
            "end_err_hh": float("nan"),
        },
        base
        | {
            "amp_mw": 1.0,
            "start": 30000,
            "end": 30000 + 144,
            "days": 3.0,
            "edge_truth": False,
            "found": False,
            "start_err_hh": float("nan"),
            "end_err_hh": float("nan"),
        },
    ]
    return pl.DataFrame(rows)


def test_world_filters_use_the_matched_call_and_the_injected_event() -> None:
    """Test F1, F5 and the recall ceiling on truth rows."""
    d = fc.world_flags(rows=world_rows())
    assert d["keep_F1"].to_list()[:2] == [True, True]
    assert d["keep_F5"].to_list()[:2] == [True, False]
    assert d["elig_F5"].to_list() == [True, False, True]
    assert d["elig_F3"].to_list() == [True, False, False]


def test_found_share_and_ceiling_are_weighted_by_megawatts() -> None:
    """Test the long-cell share: F0 finds 5 of 6 MW, F5 finds 2 of 6 MW."""
    rows = fc.found_by_filter(rows=world_rows(), tau=0.5)
    pick = {(r["filter"], r["cell"]): r for r in rows}
    assert pick[("F0", "long+medium")]["found_share"] == pytest.approx(5 / 6)
    assert pick[("F5", "long, shallow")]["found_share"] == pytest.approx(2 / 5)
    assert pick[("F5", "long+medium")]["recall_ceiling"] == pytest.approx(3 / 6)
    assert pick[("F0", "long+medium")]["exact_in_worlds"] is True
    assert pick[("F3", "long+medium")]["exact_in_worlds"] is False


def test_equal_rate_tau_is_the_lowest_tau_within_both_limits() -> None:
    """Test the lowest tau with both filtered rates at or below E1-J-sym's."""
    by_tau = {
        0.5: {"F6": {"all_rate": 5.0, "big_rate": 1.0}},
        0.6: {"F6": {"all_rate": 3.0, "big_rate": 1.0}},
        0.7: {"F6": {"all_rate": 2.0, "big_rate": 0.5}},
    }
    rates = {"all": 3.0, "big": 1.0}
    assert fc.equal_rate_tau(by_tau=by_tau, rates=rates, filter_name="F6") == 0.6
    assert (
        fc.equal_rate_tau(by_tau=by_tau, rates={"all": 1.0, "big": 1.0}, filter_name="F6") is None
    )

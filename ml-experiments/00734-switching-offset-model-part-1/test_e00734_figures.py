"""Tests of the event figures and the picks of events for them (experiment 734)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import e00734_figures as figs

DAY = figs.DAY


def test_the_window_has_a_quarter_of_the_duration_on_each_side_and_at_least_two_days() -> None:
    """Test the margin rule, and the clipping at the ends of the record."""
    assert figs.draw_window(start=100 * DAY, end=140 * DAY, record_end=500 * DAY) == (
        90 * DAY,
        150 * DAY,
    )
    assert figs.draw_window(start=100 * DAY, end=101 * DAY, record_end=500 * DAY) == (
        98 * DAY,
        103 * DAY,
    )
    assert figs.draw_window(start=DAY, end=3 * DAY, record_end=4 * DAY) == (0, 4 * DAY)


def test_an_open_event_runs_sixty_days_and_an_end_past_the_record_is_clipped() -> None:
    """Test the extent of events with no end, and of one that ends after the last reading."""
    open_event = pl.DataFrame({"primary": ["A"], "start": [10 * DAY], "end": [-1]})
    assert figs.event_extent(event=open_event, record_end=500 * DAY) == (10 * DAY, 70 * DAY)
    short_record = pl.DataFrame(
        {"primary": ["A", "B"], "start": [10 * DAY, 12 * DAY], "end": [None, 400 * DAY]}
    )
    assert figs.event_extent(event=short_record, record_end=50 * DAY) == (10 * DAY, 50 * DAY)


def test_thin_averages_equal_groups_and_leaves_short_series_alone() -> None:
    """Test the averaging of a long series and the NaN handling."""
    x = np.arange(10.0)
    y = np.arange(10.0)
    assert figs.thin(x=x, y=y, max_points=20)[0] is x
    ys = figs.thin(x=x, y=y, max_points=5)[1]
    np.testing.assert_allclose(ys, [0.5, 2.5, 4.5, 6.5, 8.5])
    y2 = y.copy()
    y2[0] = np.nan
    assert np.isnan(figs.thin(x=x, y=y2, max_points=20)[1][0])
    assert np.isfinite(figs.thin(x=x, y=y2, max_points=5)[1]).all()


def test_the_involved_primaries_are_the_group_and_the_partners_of_the_matching_call() -> None:
    """Test the donor, recipient, and mirror partner all appear once, donor first."""
    calls = pl.DataFrame(
        {"primary": ["A", "A", "C"], "start": [100, 900, 100], "partners": ["B,D", "E", "F"]}
    )
    out = figs.involved_primaries(site="A", start=105, group_sites=["A", "B"], calls=calls)
    assert out == ["A", "B", "D"]
    assert figs.involved_primaries(site="A", start=105, group_sites=[], calls=calls.clear()) == [
        "A"
    ]


def calls_table() -> pl.DataFrame:
    """Called events of every grade, some of them false."""
    rows = []
    for i, g in enumerate(figs.GRADES * 4):
        rows.append(
            {
                "world": 900 + i,
                "primary": "A",
                "start": 100 * i,
                "end": 100 * i + 500,
                "grade": g,
                "partners": "",
                "matched": i % 2 == 0,
                "on_hard_negative": False,
            }
        )
    return pl.DataFrame(rows)


def matches_table() -> pl.DataFrame:
    """Match rows in both long cells, found and missed, some cascaded."""
    rows = [
        {
            "world": 900 + i,
            "site": "A",
            "start": 100 * i,
            "end": 100 * i + 500,
            "group": i,
            "found": i % 2 == 0,
            "cell": "long, shallow" if i % 4 < 2 else "long, deep",
            "cascaded": i % 5 == 0,
        }
        for i in range(40)
    ]
    return pl.DataFrame(rows)


def test_the_picks_cover_every_grade_hits_misses_false_calls_both_cells_and_cascades() -> None:
    """Test each category is present, capped at the constant, and the same on every run."""
    picks = figs.pick_categories(matches=matches_table(), calls=calls_table(), per_category=3)
    cats = {p["category"] for p in picks}
    for g in figs.GRADES:
        assert f"grade: {g}" in cats
    assert {
        "false call",
        "hit, long, shallow",
        "miss, long, shallow",
        "hit, long, deep",
        "miss, long, deep",
        "cascade, found",
        "cascade, missed",
    } <= cats
    assert max(sum(p["category"] == c for p in picks) for c in cats) <= 3
    assert picks == figs.pick_categories(
        matches=matches_table(), calls=calls_table(), per_category=3
    )
    assert figs.PER_GRADE == 6
    assert figs.PER_REAL == 12


def test_the_real_picks_split_called_events_and_found_and_missed_logged_shifts() -> None:
    """Test a logged shift is found when a call starts within 48 half-hours of its onset."""
    calls = pl.DataFrame(
        {
            "primary": ["A", "B"],
            "start": [1000, 5000],
            "end": [1500, 6000],
            "grade": ["mirrored and returned", "neither"],
        }
    )
    logged = pl.DataFrame({"primary": ["A", "B"], "start": [1010, 9000], "end": [1400, 9500]})
    picks = figs.pick_real(calls=calls, logged=logged, per_category=12)
    by = {p["category"]: p for p in picks}
    assert by["called, mirrored"]["primary"] == "A"
    assert by["called, not mirrored"]["primary"] == "B"
    assert by["logged shift found"]["primary"] == "A"
    assert by["logged shift missed"]["primary"] == "B"


def test_an_event_figure_draws_every_primary_with_its_band_and_overlays() -> None:
    """Test the chart holds a demand panel and an offset panel for each of two Primaries."""
    n = 200 * DAY
    rng = np.random.default_rng(0)
    power = {"A": 20 + rng.normal(size=n), "B": 15 + rng.normal(size=n)}
    power["A"][:5] = np.nan
    blocks = pl.DataFrame(
        {
            "block": np.arange(n // 12),
            **{c: np.linspace(-1, 1, n // 12) for c in ("q05", "q10", "q50", "q90", "q95")},
        }
    )
    calls = {
        "E1-J-sym": pl.DataFrame(
            {"primary": ["A"], "start": [60 * DAY], "end": [-1], "amp_mw": [-1.5]}
        )
    }
    src = figs.EventSource(
        power=power, record_end=n, blocks={"A": blocks, "B": blocks}, calls=calls
    )
    event = pl.DataFrame({"primary": ["A", "B"], "start": [60 * DAY] * 2, "end": [80 * DAY, None]})
    chart = figs.event_figure(event=event, primaries=["A", "B", "Z"], source=src, title="test")
    spec = chart.to_dict()
    assert len(spec["vconcat"]) == 2
    assert "subtitle" in spec["title"]
    assert spec["title"]["text"] == "test"


def test_events_after_the_record_are_not_drawn_as_zeros(tmp_path: Path) -> None:
    """Test a series that ends is cut at the record end and a NaN is not turned into a zero."""
    n = 100 * DAY
    series = np.full(n, 10.0)
    series[50 * DAY :] = np.nan
    src = figs.EventSource(power={"A": series}, record_end=50 * DAY)
    event = pl.DataFrame({"primary": ["A"], "start": [40 * DAY], "end": [90 * DAY]})
    chart = figs.event_figure(event=event, primaries=["A"], source=src)
    data = chart.to_dict()["datasets"]
    values = [row["mw"] for rows in data.values() for row in rows if "mw" in row]
    assert values
    assert all(v == 10.0 for v in values)
    size = figs.save_svg(chart=chart, path=tmp_path / "x.svg")
    assert 0 < size < figs.MAX_SVG_BYTES


def test_clip_spans_drops_spans_outside_the_window_and_cuts_the_rest() -> None:
    """Test a span wholly before or after the window is dropped and one across an edge is cut."""
    got = figs.clip_spans(spans=[(0, 5), (8, 30), (40, 50), (90, 120), (100, 110)], lo=10, hi=100)
    assert got == [(10, 30), (40, 50), (90, 100)]


def test_marks_far_outside_the_window_neither_widen_the_svg_nor_stretch_the_offset_axis(
    tmp_path: Path,
) -> None:
    """Test far events and a far large call add no data, so the figure stays within 1,000 px."""
    n = 400 * DAY
    rng = np.random.default_rng(1)
    power = {"A": 20 + rng.normal(size=n)}
    blocks = pl.DataFrame(
        {
            "block": np.arange(n // 12),
            **{c: np.linspace(-1, 1, n // 12) for c in ("q05", "q10", "q50", "q90", "q95")},
        }
    )
    calls = {
        "H-nbr called": pl.DataFrame(
            {
                "primary": ["A", "A"],
                "start": [60 * DAY, 350 * DAY],
                "end": [70 * DAY, -1],
                "amp_mw": [-1.5, -900.0],
            }
        )
    }
    src = figs.EventSource(power=power, record_end=n, blocks={"A": blocks}, calls=calls)
    event = pl.DataFrame({"primary": ["A"], "start": [60 * DAY], "end": [70 * DAY]})
    chart = figs.event_figure(event=event, primaries=["A"], source=src)
    datasets = chart.to_dict()["datasets"]
    amps = [row["amp"] for rows in datasets.values() for row in rows if "amp" in row]
    assert amps == [-1.5]
    path = tmp_path / "wide.svg"
    figs.save_svg(chart=chart, path=path)
    assert figs.svg_width(text=path.read_text()[:2000]) <= figs.MAX_WIDTH


def test_a_figure_wider_than_the_page_is_refused(tmp_path: Path) -> None:
    """Test `save_svg` raises for a chart wider than `MAX_WIDTH`."""
    import altair as alt
    import pytest

    wide = (
        alt.Chart(pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(x="x:Q", y="y:Q")
        .properties(width=1500, height=50)
    )
    with pytest.raises(ValueError, match="px wide"):
        figs.save_svg(chart=wide, path=tmp_path / "w.svg")  # ty: ignore[invalid-argument-type]
    assert not (tmp_path / "w.svg").exists()


def _injected_table() -> pl.DataFrame:
    """Twelve edge groups and twelve solo groups of rising size and varied length."""
    rows = []
    gid = 0
    for cls, partners in (("edge_transfer", ["B"]), ("temporary_shift", [])):
        for i in range(12):
            days = [1.0, 20.0, 6.0][i % 3]
            rows.append(
                {
                    "group_id": gid,
                    "site": "A",
                    "event_class": cls,
                    "label": "positive",
                    "units": 1.0 + i,
                    "start": 10 * DAY * (i + 1),
                    "end": int(10 * DAY * (i + 1) + days * DAY),
                    "partners": partners,
                }
            )
            gid += 1
    rows.append(
        {
            "group_id": gid,
            "site": "A",
            "event_class": "ramp",
            "label": "hard_negative",
            "units": 9.0,
            "start": 5 * DAY,
            "end": 6 * DAY,
            "partners": [],
        }
    )
    return pl.DataFrame(rows)


def test_decode_examples_cover_every_size_for_edge_and_solo_events_with_short_and_long_ones() -> (
    None
):
    """Test six slots (3 sizes, 2 kinds), no hard negative, both short and long durations."""
    slots, rest = figs.pick_decode_examples(table=_injected_table(), max_half_hours=400 * DAY)
    assert len(slots) == 6
    assert {(c["kind"], c["size"]) for c in slots} == {
        (k, s) for k in ("edge", "solo") for s in figs.SIZE_NAMES
    }
    assert any(c["days"] <= figs.SHORT_DAYS for c in slots)
    assert any(c["days"] >= figs.LONG_DAYS for c in slots)
    assert not {c["group"] for c in slots} & {c["group"] for c in rest}
    assert len({c["group"] for c in slots}) == 6
    assert all(c["group"] != 24 for c in slots + rest)


def test_decode_examples_skip_events_that_end_after_the_cut_record() -> None:
    """Test an event that ends after `max_half_hours` is not a candidate."""
    slots, rest = figs.pick_decode_examples(table=_injected_table(), max_half_hours=50 * DAY)
    assert all(c["group"] % 12 < 5 for c in slots + rest)


def test_window_stats_report_the_decode_inside_the_truth_window() -> None:
    """Test the statistics read the window's blocks, and the overlap with a call sets `decoded`."""
    n = 100
    blocks = pl.DataFrame(
        {
            "block": np.arange(n),
            "q10": np.full(n, -0.5),
            "q50": np.where((np.arange(n) >= 10) & (np.arange(n) < 20), 3.0, 0.0),
            "q90": np.where((np.arange(n) >= 10) & (np.arange(n) < 20), 4.0, 0.5),
        }
    )
    p_zero = np.where((np.arange(n) >= 10) & (np.arange(n) < 20), 0.1, 0.9)
    calls = pl.DataFrame({"primary": ["A"], "start": [120], "end": [240], "amp_mw": [3.0]})
    hit = figs.window_stats(
        blocks=blocks, p_zero=p_zero, calls=calls, primary="A", start=120, end=240
    )
    assert hit["decoded"] is True
    assert hit["best_iou"] == 1.0
    assert hit["max_abs_q50_mw"] == 3.0
    assert hit["max_q90_mw"] == 4.0
    assert abs(hit["mean_p_zero"] - 0.1) < 1e-9
    miss = figs.window_stats(
        blocks=blocks, p_zero=p_zero, calls=calls, primary="B", start=120, end=240
    )
    assert miss["decoded"] is False

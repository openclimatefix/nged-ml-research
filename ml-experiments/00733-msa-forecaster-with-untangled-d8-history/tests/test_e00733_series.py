# ruff: noqa: PT018, E501
"""Tests of `e00733_series` and `e00733_figstages`: the windows are typical or extreme by a stated
rule, never repeat, never come from source S1, and an empty scenario is an error."""

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import e00733_figstages as figstages
import e00733_series as series
import e00733_synth as sy
import numpy as np
import polars as pl
import pytest


def _frame(
    *, start: datetime, n_origins: int, offset: float = 0.0, bias: float = 0.0
) -> pl.DataFrame:
    rows = []
    for o in range(n_origins):
        origin = start + timedelta(days=o)
        for h in range(96):
            valid = origin + timedelta(hours=15, minutes=30 * h)
            actual = 0.3
            rows.append(
                {
                    "power_fcst_init_time": origin, "valid_time": valid, "lead_hours": 15.0 + 0.5 * h,
                    "q10": actual + offset - 0.1, "q50": actual + offset + bias * o, "q90": actual + offset + 0.1, "actual": actual,
                }
            )  # fmt: skip
    return pl.DataFrame(rows).with_columns(
        pl.col("power_fcst_init_time", "valid_time").dt.replace_time_zone("UTC")
    )


def test_the_typical_origin_is_the_median_error_of_the_season_and_never_repeats() -> None:
    """The origin with the median error is taken; an excluded origin is skipped; an absent season is an error without a fallback."""
    frame = _frame(start=datetime(2026, 1, 1), n_origins=5, bias=0.01)
    origin, note = series.pick_typical_origin(frame=frame, months=(1,), allow_fallback=False)
    assert origin == datetime(2026, 1, 3, tzinfo=UTC) and note == ""
    again, _ = series.pick_typical_origin(
        frame=frame, months=(1,), allow_fallback=False, exclude={origin}
    )
    assert again != origin
    with pytest.raises(series.SeriesError):
        series.pick_typical_origin(frame=frame, months=(7,), allow_fallback=False)
    fallback, note = series.pick_typical_origin(frame=frame, months=(7,), allow_fallback=True)
    assert fallback.month == 1 and "nearest available month" in note


def test_the_channel_origin_is_where_the_channels_move_the_forecast_most() -> None:
    """The origin with the largest mean absolute difference between FU and masked FU is taken, outside the excluded."""
    a = _frame(start=datetime(2026, 1, 1), n_origins=4)
    b = a.with_columns(
        q50=pl.col("q50")
        + pl.when(pl.col("power_fcst_init_time").dt.day() == 3).then(0.2).otherwise(0.01)
    )
    assert series.pick_channel_origin(with_channels=a, masked=b, exclude=set()) == datetime(
        2026, 1, 3, tzinfo=UTC
    )
    second = series.pick_channel_origin(
        with_channels=a, masked=b, exclude={datetime(2026, 1, 3, tzinfo=UTC)}
    )
    assert second != datetime(2026, 1, 3, tzinfo=UTC)
    with pytest.raises(series.SeriesError):
        series.pick_channel_origin(
            with_channels=a, masked=b, exclude=set(a["power_fcst_init_time"].to_list())
        )


def test_event_windows_keep_events_with_forecasts_at_both_ends_and_thin_a_long_list() -> None:
    """An event outside the forecast span is left out; more than the cap gives an evenly spaced subset; none gives an empty list and a count of 0."""
    times = pl.Series([datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(2000)])
    events = pl.DataFrame(
        {
            "start": [times[i] for i in range(100, 1800, 100)] + [datetime(2020, 1, 1, tzinfo=UTC)],
            "end": [times[i + 5] for i in range(100, 1800, 100)]
            + [datetime(2020, 1, 2, tzinfo=UTC)],
        }
    )
    shifts, n = series.event_windows(events=events, valid_times=times, cap=6)
    assert (
        n == 17
        and len(shifts) == 6
        and shifts[0]["first"] == shifts[0]["start"] - timedelta(days=14)
    )
    none, zero = series.event_windows(events=events.head(0), valid_times=times)
    assert none == [] and zero == 0


def test_a_repeated_window_is_an_error() -> None:
    """Two windows with one key raise, within or across scenarios."""
    series.assert_distinct(windows=[series.Window("a", (1,)), series.Window("b", (2,))])
    with pytest.raises(series.SeriesError):
        series.assert_distinct(windows=[series.Window("a", (1,)), series.Window("b", (1,))])


def _sign_data(*, sources: list[str], flow: dict[int, int]) -> SimpleNamespace:
    """Samples of 6 segments, 2 origins each, 9 lead days each; `flow` gives the reverse-flow half-hours of each segment's first origin."""
    seg, day, lead = [], [], []
    for s in range(len(sources)):
        for d in (0, 20):
            for k in range(1, 10):
                seg.append(s)
                day.append(d)
                lead.append(k)
    n = len(seg)
    truth = np.zeros((n, 48))
    for i, (s, d, k) in enumerate(zip(seg, day, lead, strict=True)):
        if d == 0 and k == 1:
            truth[i, : flow.get(s, 0)] = 1.0
    p = np.clip(truth * 0.7 + 0.05, 0, 1)
    index = sy.SampleIndex(np.array(seg), np.array(day), np.array(lead) - 1)
    return SimpleNamespace(
        index=index, sources=sources, truth=truth, keep=np.ones((n, 48), dtype=bool),
        probabilities={"YU": p, "Y0": p, "B-hist": p, "B-clim": p},
    )  # fmt: skip


def test_the_sign_windows_are_three_distinct_s2_segments_with_flow_and_one_without_and_never_s1() -> (
    None
):
    """Segments of source S1 are never chosen; the three windows with flow come from three segments."""
    data = _sign_data(
        sources=["S1", "S2", "S2", "S2", "S2", "none"], flow={0: 40, 1: 5, 2: 9, 3: 7, 5: 30}
    )
    windows = series.pick_sign_windows(data=data)
    segs = [w.seg for w in windows]
    assert [w.kind for w in windows] == ["reverse flow"] * 3 + ["no reverse flow"]
    assert 0 not in segs and 5 not in segs and len(set(segs)) == 4
    assert windows[0].seg == 2 and all(len(w.rows) == 9 for w in windows)
    series.assert_distinct(windows=[series.Window("sign", (w.seg, w.origin_day)) for w in windows])
    frame = series.sign_series_for(data=data, window=windows[0])
    assert frame["series"].n_unique() == 4 and frame.height == 4 * 9 * 48
    series.sign_figure(data=data, windows=windows)
    with pytest.raises(series.SeriesError):
        series.pick_sign_windows(
            data=_sign_data(sources=["S2", "S2", "S1", "S1"], flow={0: 3, 1: 3})
        )


def test_forecast_chart_refuses_an_empty_panel() -> None:
    """A panel that selects no row is an error."""
    frame = _frame(start=datetime(2026, 1, 1), n_origins=2)
    series.forecast_chart(
        frames={"FU": frame},
        origin=datetime(2026, 1, 1, tzinfo=UTC),
        lead=series.DAY_AHEAD_LEAD_HOURS,
        title="t",
    )
    with pytest.raises(series.SeriesError):
        series.forecast_chart(
            frames={"FU": frame}, origin=datetime(2030, 1, 1, tzinfo=UTC), lead=None, title="t"
        )


def test_svg_optimisation_refuses_nothing_to_optimise_a_missing_npx_and_an_oversize_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage fails on an empty file list, without `npx`, and on a file over 500 KB after `svgo`."""
    with pytest.raises(figstages.FigureStageError):
        figstages.optimise(folder=tmp_path, names=[])
    (tmp_path / "a.svg").write_text("x" * (figstages.MAX_SVG_BYTES + 1))
    monkeypatch.setattr(figstages.shutil, "which", lambda _n: None)
    with pytest.raises(figstages.FigureStageError, match="npx"):
        figstages.optimise(folder=tmp_path, names=["a.svg"])
    monkeypatch.setattr(figstages.shutil, "which", lambda _n: "/bin/true")
    monkeypatch.setattr(
        figstages.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 0, "", "")
    )
    with pytest.raises(figstages.FigureStageError, match="over the limit"):
        figstages.optimise(folder=tmp_path, names=["a.svg"])
    (tmp_path / "a.svg").write_text("x")
    assert figstages.optimise(folder=tmp_path, names=["a.svg"]) == {"a.svg": 1}


def test_real_arms_filtering_leaves_out_a_dropped_arm() -> None:
    """The drawn arms are FU, FS, and M* less the dropped ones."""
    assert series.drawn_arms(dropped=()) == ["FU", "FS", "M*"]
    assert series.drawn_arms(dropped=("YT", "FS")) == ["FU", "M*"]


def _folds(*, with_fs: bool) -> dict[str, dict[str, pl.DataFrame]]:
    fu = _frame(start=datetime(2026, 1, 1), n_origins=5, bias=0.01)
    frames = {"FU": fu, "M*": fu}
    if with_fs:
        frames["FS"] = fu
    return {"2026-01": frames}


def test_season_figure_draws_the_fitted_arms_and_names_a_dropped_one() -> None:
    """With FS dropped the panels carry FU and M* and say so; an undropped missing arm is an error."""
    folds = _folds(with_fs=False)
    chart, _, notes = series.season_figure(folds=folds, allow_fallback=True, dropped=("YT", "FS"))
    assert any("FS was not fitted: the timing gate dropped it" in n for n in notes)
    assert "FS was not fitted" in chart.vconcat[0].title.text
    with pytest.raises(series.SeriesError):
        series.season_figure(folds=folds, allow_fallback=False, dropped=())
    _, _, none = series.season_figure(folds=_folds(with_fs=True), allow_fallback=True)
    assert all("not fitted" not in n for n in none)


def test_read_real_skips_a_dropped_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dropped arm is never read, even before the stored forecasts are looked up."""
    frame = _frame(start=datetime(2026, 1, 1), n_origins=1).with_columns(time_series_id=pl.lit(1))
    monkeypatch.setattr(series.scoring, "has_job", lambda **kw: kw["arm"][0] in ("FU", "FS"))
    monkeypatch.setattr(series.store, "read_set", lambda **_: frame)
    monkeypatch.setattr(series.store, "set_name", lambda **_: "x")
    got = series.read_real(root=tmp_path, rows="test", d8=1, fold="2026-01", dropped=("FS",))
    assert set(got) == {"FU", "FU masked"}

# ruff: noqa: PT018, E501
"""Tests of `e00733_scoring`: the plan a contrast reads, the rows it keeps, R8's pieces, R9's tails,
the validation truth, and the two reads of the chain (F0r gate, screen point)."""

import json
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import e00733_baselines as baselines
import e00733_core as core
import e00733_scoring as sg
import e00733_store as store
import numpy as np
import polars as pl
import pytest
from e00733_channels import StoppedError


def _exists(held: set[tuple[str, str, int]]):  # noqa: ANN202
    def check(*, root: Path, arm: tuple[str, str | None], fold: str, seed: int, rows: str) -> bool:
        return (arm[0], fold, seed) in held

    return check


def _plan(**kw: Any) -> sg.Plan | None:
    return sg.choose_plan(**kw)


def test_choose_plan_never_falls_back_silently() -> None:
    """Only the full plan is read unless the screen or a partial plan is allowed; a counted one raises."""
    arms = [("FU", None), ("M*", None)]
    seeds = core.HEADLINE_SEEDS
    full = {(a, f, s) for a in ("FU", "M*") for f in core.ALL_FOLDS for s in seeds}
    plan = _plan(root=Path(), arms=arms, wanted_seeds=seeds, rows="test", exists=_exists(full))
    assert plan is not None and not plan.interim and plan.folds == core.ALL_FOLDS
    screen = {(a, f, s) for a in ("FU", "M*") for f in core.SCREEN_FOLDS for s in core.SCREEN_SEEDS}
    kw: dict[str, Any] = {
        "root": Path(), "arms": arms, "wanted_seeds": seeds, "rows": "test", "exists": _exists(screen),
    }  # fmt: skip
    assert _plan(**kw) is None
    plan = _plan(**kw, screen_ok=True)
    assert plan is not None and plan.interim and plan.folds == core.SCREEN_FOLDS
    assert plan.seeds == core.SCREEN_SEEDS
    with pytest.raises(sg.IncompletePlanError):
        _plan(**kw, required=True)
    one = {("FU", "2025-11", 0), ("M*", "2025-11", 0), ("M*", "2026-02", 0)}
    kw["exists"] = _exists(one)
    assert _plan(**kw) is None
    plan = _plan(**kw, partial_ok=True)
    assert plan is not None and plan.folds == ("2025-11",) and plan.seeds == (0,) and plan.interim
    assert _plan(**{**kw, "rows": "validation"}) is not None
    odd = {("FU", "2025-11", 0), ("M*", "2025-11", 0), ("FU", "2026-02", 1), ("M*", "2026-02", 1)}
    kw["exists"] = _exists(odd)
    assert _plan(**kw, partial_ok=True) is None
    with pytest.raises(sg.IncompletePlanError):
        _plan(**{**kw, "exists": _exists(set())}, required=True, partial_ok=True)


def test_the_screen_is_a_plan_of_a_counted_reading_only_when_it_stopped_the_rest(
    tmp_path: Path,
) -> None:
    """`scores/screen.json` is read for `stops_rest`."""
    assert not sg.screen_stops_rest(root=tmp_path)
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "screen.json").write_text('{"stops_rest": true}')
    assert sg.screen_stops_rest(root=tmp_path)
    (tmp_path / "scores" / "screen.json").write_text('{"stops_rest": false}')
    assert not sg.screen_stops_rest(root=tmp_path)


def test_validation_rows_swap_m_star_for_f0r() -> None:
    """On validation rows F0r stands in for M*; no other arm changes."""
    assert sg.swap_reference(arm=("M*", None), rows="validation") == ("F0r", None)
    assert sg.swap_reference(arm=("M*", None), rows="test") == ("M*", None)
    assert sg.swap_reference(arm=("FU", None), rows="validation") == ("FU", None)


def _losses(*, n: int = 40, offset: float = 0.0, ids: tuple[int, ...] = (26,)) -> pl.DataFrame:
    t0 = datetime(2025, 11, 1, 12, tzinfo=UTC)
    rows = []
    for i in range(n):
        for sid in ids:
            valid = t0 + timedelta(hours=24 + 12 * i)
            rows.append(
                (
                    sid,
                    t0 + timedelta(days=i % 3 + (i // 20) * 31),
                    valid,
                    24.0 + 12 * i,
                    1.0 + offset + 0.01 * i,
                )
            )
    frame = pl.DataFrame(
        rows,
        schema=["time_series_id", "power_fcst_init_time", "valid_time", "lead_hours", "crps"],
        orient="row",
    )
    return frame.with_columns(
        abs_err=pl.col("crps"), pin_q01=pl.col("crps"), pin_q99=pl.col("crps")
    )


def test_restrict_keeps_the_headline_lead_and_the_chosen_primaries() -> None:
    """Rows from 24 h up to 240 h of the named Primaries stay; others go."""
    frame = pl.concat([_losses(ids=(26,)), _losses(ids=(27,))])
    kept = sg.restrict(frame=frame, ids=[26])
    assert set(kept["time_series_id"].to_list()) == {26}
    assert (
        kept["lead_hours"].to_numpy().min() >= 24.0 and kept["lead_hours"].to_numpy().max() < 240.0
    )
    assert 0 < kept.height < 40


def test_common_keys_cuts_to_the_shared_keys_and_refuses_none() -> None:
    """Two frames are cut to the keys both hold; a pair with no key in common raises."""
    a, b = _losses(n=10), _losses(n=6, offset=0.5)
    ca, cb = sg.common_keys(a=a, b=b)
    assert ca.height == cb.height == 6 and ca.select(*sg.KEYS).equals(cb.select(*sg.KEYS))
    other = b.with_columns(time_series_id=pl.lit(99, dtype=pl.Int64))
    with pytest.raises(sg.sc.ScoreError):
        sg.common_keys(a=a, b=other)


def test_leave_one_month_out_returns_a_mean_without_each_fold_month() -> None:
    """The mean of the differences without each origin month (a polars Series, not an expression)."""
    t = _losses(n=40, offset=-0.2)
    r = _losses(n=40)
    out = sg.leave_one_month_out(t=t, r=r)
    assert set(out) == {"2025-10", "2025-11", "2025-12"} or len(out) >= 2
    assert all(v == pytest.approx(-0.2) for v in out.values())


def test_tail_table_counts_rows_and_coverage() -> None:
    """The tail table holds the pinball means, the coverage of each tail level, and the row count."""
    frame = pl.DataFrame(
        {
            "pin_q01": [1.0, 3.0],
            "pin_q05": [1.0, 1.0],
            "pin_q95": [2.0, 2.0],
            "pin_q99": [0.5, 0.5],
            "actual": [0.0, 5.0],
            "q01": [1.0, 1.0],
            "q05": [1.0, 1.0],
            "q95": [4.0, 4.0],
            "q99": [6.0, 6.0],
        }
    )
    out = sg.tail_table(frame=frame)
    assert out["n"] == 2 and out["pin_q01"] == pytest.approx(2.0)
    assert out["coverage_q01"] == pytest.approx(0.5) and out["coverage_q99"] == pytest.approx(1.0)
    assert sg.tail_table(frame=frame.head(0))["n"] == 0


def test_validation_truth_stacks_the_actuals_of_the_validation_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stack holds each set's actual at its primary and grid index, and no set raises."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    sites = pl.DataFrame({"time_series_id": [26, 27], "primary": [0, 1]})
    fake = types.SimpleNamespace(sites=lambda: sites)
    monkeypatch.setitem(sys.modules, "e00578_common", fake)
    with pytest.raises(sg.sc.ScoreError):
        sg.validation_truth(root=tmp_path)
    folder = tmp_path / "predictions" / "f0r_2025-11_s0_val"
    folder.mkdir(parents=True)
    t = datetime(2025, 5, 24, 12, tzinfo=UTC)
    pl.DataFrame(
        {
            "time_series_id": [26, 27],
            "valid_time": [t, t + timedelta(minutes=30)],
            "actual": [0.25, 0.5],
        }
    ).write_parquet(folder / "predictions.parquet")
    stack = sg.validation_truth(root=tmp_path)
    i = (int(t.timestamp() * 1e6) - prep422.GRID_START_US) // prep422.STEP_US
    assert (
        stack.shape[0] == 16
        and stack[0, i] == 0.25
        and stack[1, i + 1] == 0.5
        and np.isnan(stack[0, i + 1])
    )


def test_f0r_gate_needs_a_job_and_compares_to_m_star(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No F0r forecast stops the line; a forecast equal to M* passes; a different one stops it."""
    with pytest.raises(StoppedError):
        sg.f0r_gate(root=tmp_path, require_all=False)
    t0 = datetime(2025, 11, 1, 9, tzinfo=UTC)
    keys = {
        "time_series_id": [26, 26],
        "power_fcst_init_time": [t0, t0],
        "valid_time": [t0 + timedelta(hours=30), t0 + timedelta(hours=31)],
    }
    mine = pl.DataFrame({**keys, **{q: [0.1, 0.2] for q in sg.QCOLS}})
    monkeypatch.setattr(sg, "has_job", lambda **k: k["fold"] == "2025-11" and k["seed"] == 0)
    monkeypatch.setattr(sg.store, "read_set", lambda **_k: mine)
    monkeypatch.setattr(baselines, "read_m_star", lambda **_k: mine)
    got = sg.f0r_gate(root=tmp_path, require_all=False)
    assert got["jobs"] == 1 and got["rows"] == 2 and got["largest_difference"] == 0.0
    with pytest.raises(StoppedError):
        sg.f0r_gate(root=tmp_path, require_all=True)
    monkeypatch.setattr(
        baselines, "read_m_star", lambda **_k: mine.with_columns(q50=pl.col("q50") + 1e-3)
    )
    with pytest.raises(StoppedError):
        sg.f0r_gate(root=tmp_path, require_all=False)
    assert store.row_set_id(rows="test") == core.ROW_SET_ID_REAL


def _stub_losses(arm: tuple[str, str | None], plan: sg.Plan) -> pl.DataFrame:
    n = 4
    return pl.DataFrame(
        {
            "time_series_id": [1] * n,
            "lead_hours": [30.0] * n,
            "crps": [1.0] * n,
            "abs_err": [0.5] * n,
            **{c: [1.0] * n for c in sg.TAIL_COLUMNS},
            "actual": [0.0] * n,
            "q01": [1.0] * n,
            "q05": [1.0] * n,
            "q95": [1.0] * n,
            "q99": [1.0] * n,
            **{k: [1] * n for k in sg.KEYS if k not in ("time_series_id", "lead_hours")},
        }
    )


def _timing(root: Path, *, dropped: list[str]) -> None:
    (root / "scores").mkdir(parents=True)
    (root / "scores" / "timing.json").write_text(json.dumps({"dropped": dropped}))


def test_r9_and_r10_read_the_arms_that_were_fitted_when_fs_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With FS dropped R9 and R10 read FU and M* and record the arms used and the arm dropped."""
    _timing(tmp_path, dropped=["YT", "FS"])
    seen: list[str] = []

    def exists(**kw: Any) -> bool:
        seen.append(kw["arm"][0])
        return kw["arm"][0] != "FS"

    monkeypatch.setattr(sg, "has_job", exists)
    r9 = sg.reading_r9(root=tmp_path, rows="test", d8=1, truth=np.zeros(1), losses=_stub_losses)
    assert r9["arms_used"] == ["FU", "M*"] and r9["dropped"] == ["FS"]
    assert set(r9["arms"]) == {"FU", "M*"} and "FS" not in seen
    monkeypatch.setattr(
        sg,
        "slice_flags",
        lambda **kw: kw["rows"].with_columns(
            event_rows=pl.lit(False), event_following_7_days=pl.lit(False)
        ),
    )
    r10 = sg.reading_r10(root=tmp_path, rows="test", d8=1, truth=np.zeros(1), losses=_stub_losses)
    assert r10["arms_used"] == ["FU", "M*"] and r10["dropped"] == ["FS"]
    assert set(r10["scenarios"]["all rows"]) == {"n", "FU", "M*"}


def test_r9_keeps_every_arm_before_the_timing_gate_and_voids_when_no_plan_is_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a timing record all three arms are read; an undropped missing arm voids the reading."""
    monkeypatch.setattr(sg, "has_job", lambda **kw: True)
    r9 = sg.reading_r9(root=tmp_path, rows="test", d8=1, truth=np.zeros(1), losses=_stub_losses)
    assert r9["arms_used"] == ["FU", "FS", "M*"] and r9["dropped"] == []
    monkeypatch.setattr(sg, "has_job", lambda **kw: kw["arm"][0] != "FS")
    void = sg.reading_r9(root=tmp_path, rows="test", d8=1, truth=np.zeros(1), losses=_stub_losses)
    assert void["status"] == "void" and void["dropped"] == []

# ruff: noqa: PT018, E501
"""Tests of `e00733_scoring`: the plan a contrast reads, the rows it keeps, R8's pieces, R9's tails,
the validation truth, and the two reads of the chain (F0r gate, screen point)."""

import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

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


def test_choose_plan_prefers_the_full_plan_then_the_screen_then_what_exists() -> None:
    """The full plan needs every fold and seed; the screen needs four folds at seeds 0 and 1."""
    arms = [("FU", None), ("M*", None)]
    full = {(a, f, s) for a in ("FU", "M*") for f in core.ALL_FOLDS for s in core.HEADLINE_SEEDS}
    plan = sg.choose_plan(
        root=Path(), arms=arms, wanted_seeds=core.HEADLINE_SEEDS, rows="test", exists=_exists(full)
    )
    assert (
        plan is not None
        and not plan.interim
        and plan.folds == core.ALL_FOLDS
        and plan.seeds == core.HEADLINE_SEEDS
    )
    screen = {(a, f, s) for a in ("FU", "M*") for f in core.SCREEN_FOLDS for s in core.SCREEN_SEEDS}
    plan = sg.choose_plan(
        root=Path(),
        arms=arms,
        wanted_seeds=core.HEADLINE_SEEDS,
        rows="test",
        exists=_exists(screen),
    )
    assert (
        plan is not None
        and plan.interim
        and plan.folds == core.SCREEN_FOLDS
        and plan.seeds == core.SCREEN_SEEDS
    )
    odd = {("FU", "2025-11", 0), ("M*", "2025-11", 0), ("FU", "2026-02", 1), ("M*", "2026-02", 1)}
    assert (
        sg.choose_plan(
            root=Path(),
            arms=arms,
            wanted_seeds=core.HEADLINE_SEEDS,
            rows="test",
            exists=_exists(odd),
        )
        is None
    )
    one = {("FU", "2025-11", 0), ("M*", "2025-11", 0), ("M*", "2026-02", 0)}
    plan = sg.choose_plan(
        root=Path(), arms=arms, wanted_seeds=core.HEADLINE_SEEDS, rows="test", exists=_exists(one)
    )
    assert plan is not None and plan.folds == ("2025-11",) and plan.seeds == (0,) and plan.interim
    assert (
        sg.choose_plan(
            root=Path(),
            arms=arms,
            wanted_seeds=core.HEADLINE_SEEDS,
            rows="test",
            exists=_exists(set()),
        )
        is None
    )


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

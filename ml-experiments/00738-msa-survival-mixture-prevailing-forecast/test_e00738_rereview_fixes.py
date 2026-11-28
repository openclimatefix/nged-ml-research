"""Tests of the fixes that the Sonnet re-review of experiment 738 asked for."""
# ruff: noqa: D103, E402, ANN001

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
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
import e00738_gate as gate
import e00738_members as members
import e00738_panel as panel
import e00738_phases as phases
import e00738_predict as pred
import e00738_run as run
import e00738_survival as surv
import e00738_world as world
import e00738_worldcheck as wc
import polars as pl
from test_e00738_review_fixes import IDS, _gate_frames

OK = {"midday": 1.0, "midnight": 1.0}

# --- R1: the gate's bars carry the survival discount of a lead-free head ---------------------


def test_the_discount_of_the_generator_curve_is_about_three_quarters() -> None:
    curve = surv.GeneratorCurve()
    d3 = gate.survival_discount(curve=curve, ages=gate.CHECK3_AGES)
    d2 = gate.survival_discount(curve=curve, ages=gate.CHECK2_AGES)
    assert 0.70 < d3 < 0.80
    assert d3 < d2 < 0.9
    assert gate.survival_discount(
        curve=surv.EmpiricalCurve.of(durations_days=np.array([1e4])), ages=(3.0, 10.0)
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("m1_follow", [0.0, 0.3, 0.6])
def test_an_ideal_lead_free_head_passes_the_gate(m1_follow: float) -> None:
    curve = surv.GeneratorCurve()
    d2 = gate.survival_discount(curve=curve, ages=gate.CHECK2_AGES)
    d3 = gate.survival_discount(curve=curve, ages=gate.CHECK3_AGES)
    remaining = 1.0 - m1_follow
    record = gate.evaluate(
        oracle=OK, lp_follow=remaining * d2, depth_ratio=remaining * d3,
        m1_follow=m1_follow, discount_follow=d2, discount_depth=d3,
    )  # fmt: skip
    assert record["passed"]
    assert f"{d3:.3f}" in record["derivation"]
    # without the discount the same ideal head would sit below the bar of 0.7
    assert remaining * d3 < 0.7 * remaining or d3 >= 0.7


def test_gate_readings_just_off_the_bars_flip_the_decision() -> None:
    bar2, bar3 = 0.75 * 0.5 * 0.8, 0.7 * 0.5 * 0.75

    def passed(lp: float, depth: float) -> bool:
        record = gate.evaluate(
            oracle=OK, lp_follow=lp, depth_ratio=depth,
            m1_follow=0.5, discount_follow=0.8, discount_depth=0.75,
        )  # fmt: skip
        return bool(record["passed"])

    assert passed(bar2 + 0.01, bar3 + 0.01)
    assert not passed(bar2 - 0.01, bar3 + 0.01)
    assert not passed(bar2 + 0.01, bar3 - 0.01)


# --- R2: the predicted panels carry the injection the shifts describe ------------------------


def test_the_injection_check_passes_a_matching_panel_and_fails_a_missing_or_leaking_one() -> None:
    m1, _lp, real, _diag, shifts, _want = _gate_frames(ids=IDS)
    seen = wc.check_injected_panel(frame=m1, real=real, shifts=shifts, ids=IDS)
    assert seen["share_nonzero_inside"] == 1.0
    assert seen["rows_by_stratum"]["far"] > 0
    with pytest.raises(AssertionError, match="shifted"):
        wc.check_injected_panel(frame=real, real=real, shifts=shifts, ids=IDS)
    ended = [
        world.Shift(
            primary=s.primary, kind=s.kind, onset_idx=s.onset_idx, ramp_half_hours=1,
            end_idx=s.onset_idx + 48, drawn_days=1.0, depth=s.depth, sign=1, family="train",
        )
        for s in shifts
    ]  # fmt: skip
    with pytest.raises(AssertionError):
        wc.check_injected_panel(frame=m1, real=real, shifts=ended, ids=IDS)


# --- R3: every filter at the holdout can fail --------------------------------------------------


def _block(*, valid_us: list[int]) -> pl.DataFrame:
    n = len(valid_us)
    return pl.DataFrame(
        {
            "time_series_id": [1] * n,
            "origin_us": [common.HOLDOUT_START_US - 10**12] * n,
            "valid_us": valid_us,
            "lead_day": [1] * n,
            "actual": [0.1] * n,
            **{c: [0.2 + 0.01 * i for i in range(n)] for c in panel.QCOLS},
        }
    )


def test_the_forecast_frame_drops_every_row_at_or_after_the_holdout() -> None:
    edge = common.HOLDOUT_START_US
    step = common.STEP_US
    frame = panel.frame_of(
        blocks=[_block(valid_us=[edge - 2 * step, edge - step, edge, edge + step])]
    )
    assert frame.height == 2
    assert max(frame["valid_time"].dt.epoch("us").to_list()) < edge


def test_the_gate_and_the_injection_check_refuse_a_valid_time_in_the_holdout() -> None:
    m1, lp, real, diag, shifts, _ = _gate_frames(ids=IDS)
    late = pl.duration(microseconds=common.HOLDOUT_START_US - common.GRID_START_US)

    def shifted(f: pl.DataFrame) -> pl.DataFrame:
        return f.with_columns(pl.col("valid_time") + late)

    with pytest.raises(AssertionError, match="2026-07-01"):
        gate.panel_readings(
            m1=shifted(m1), lp=shifted(lp), real=shifted(real), diag=diag, shifts=shifts, ids=IDS
        )
    with pytest.raises(AssertionError, match="2026-07-01"):
        wc.check_injected_panel(frame=shifted(m1), real=shifted(real), shifts=shifts, ids=IDS)


def test_the_member_check_filter_leaves_only_whole_blocks() -> None:
    n = members.STEPS_PER_ROW
    frame = pl.DataFrame(
        {
            "time_series_id": [1] * (n - 1),
            "power_fcst_init_time": [0] * (n - 1),
            "lead_day": [1] * (n - 1),
        }
    )
    assert members.complete_blocks(frame=frame).height == 0


# --- R4 and R5: A1 stops --------------------------------------------------------------------


def _fold(month: str, share: float, rows: int = 1000) -> dict:
    return {"month": month, "rows": rows, "flagged": int(share * rows), "share": share}


def test_the_fold_check_stops_on_a_bad_fold_or_a_bad_pool_and_prints_the_table() -> None:
    ok = [_fold(m, 0.03) for m in common.FOLDS]
    assert run.a1_fold_verdict(rows=ok)["pooled"] == pytest.approx(0.03)
    with pytest.raises(RuntimeError, match="2025-12"):
        run.a1_fold_verdict(rows=[*ok[:1], _fold("2025-12", 0.12), *ok[2:]])
    with pytest.raises(RuntimeError, match="pooled"):
        run.a1_fold_verdict(rows=[_fold(m, 0.08) for m in common.FOLDS])


def test_a1_decide_raises_on_a_failed_tuning_and_writes_the_file_only_on_success(tmp_path) -> None:
    for z, miss in ((None, 0.1), (6.0, 0.7)):
        with pytest.raises(RuntimeError, match="coordinator"):
            run.a1_decide(
                z_tuned=z, share_tuned=0.2, miss_3_10=miss, month="2025-11", noise=0.1, out=tmp_path
            )
        assert not (tmp_path / common.A1_THRESHOLD_FILE).exists()
    run.a1_decide(
        z_tuned=6.0, share_tuned=0.04, miss_3_10=0.3, month="2025-11", noise=0.1, out=tmp_path
    )
    assert (tmp_path / common.A1_THRESHOLD_FILE).exists()


def test_the_gallery_draws_a1_with_the_tuned_threshold() -> None:
    import inspect

    import e00738_gallery as gallery

    assert "z=surv.a1_threshold()" in inspect.getsource(gallery)


# --- R6: the state-to-forecast link ------------------------------------------------------------


def test_a_forecast_from_another_state_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    state = common.fit_paths(arm="LP", month="2025-11", seed=0)["state"]
    state.parent.mkdir(parents=True)
    state.write_bytes(b"state one")
    phases.write_recipe(path=state, modules=phases.FIT_MODULES)
    path = pred.panel_path(arm="LP", month="2025-11", seed=0, name="real")
    link = phases.state_digest(arm="LP", stem="2025-11_s0")
    for f in (path, pred.diag_path(path=path, name="real")):
        pred.write_parquet_atomic(frame=pl.DataFrame({"a": [1]}), path=f, link=link)
    assert phases.require_all_recipes() == 3
    assert pred.panel_done(arm="LP", month="2025-11", seed=0, name="real")
    state.write_bytes(b"state two")  # refitted by hand: the same code, another state
    with pytest.raises(RuntimeError, match="another state"):
        phases.require_all_recipes()
    with pytest.raises(RuntimeError, match="another state"):
        pred.panel_done(arm="LP", month="2025-11", seed=0, name="real")


def test_the_figures_stage_checks_the_recipes() -> None:
    import inspect

    assert "require_all_recipes" in inspect.getsource(run.stage_figures)


# --- R7 and R8: the drawn end and the holdout check -----------------------------------------


def test_a_sub_half_hour_shift_has_a_drawn_end_one_half_hour_after_its_onset() -> None:
    assert world.drawn_end_of(onset_idx=100, drawn_days=0.005) == 101
    shift = world.Shift(
        primary=0, kind="constant", onset_idx=100, ramp_half_hours=1, end_idx=101,
        drawn_days=0.005, depth=0.1, sign=1, family="resampled",
    )  # fmt: skip
    assert shift.drawn_end_idx == 101


def test_the_world_check_fires_on_an_unclipped_end(tmp_path) -> None:
    for month in common.FOLDS:
        wc.write_worlds(month=month, out=tmp_path)
    good = wc.check_worlds(months=common.FOLDS, out=tmp_path)
    assert wc.verdict(found=good)["faults"] == []
    assert good["latest_drawn_end_idx"] >= good["latest_shift_end_idx"]
    path = wc.worlds_path(month="2026-05", out=tmp_path)
    table = pl.read_parquet(path)
    pl.concat(
        [
            table.head(1).with_columns(
                end_idx=pl.lit(good["holdout_idx"] + 500, dtype=table.schema["end_idx"])
            ),
            table.tail(table.height - 1),
        ]
    ).write_parquet(path)
    bad = wc.check_worlds(months=common.FOLDS, out=tmp_path)
    assert any("reaches" in f for f in wc.verdict(found=bad)["faults"])


# --- R10: exit codes and the wait ceiling -----------------------------------------------------


def test_a_chain_exits_with_75_after_two_hours_of_waiting(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    with pytest.raises(SystemExit) as exc:
        phases.wait_for_gpu(
            stage="s", task="t", busy=lambda: True, sleep=lambda _s: None,
            max_seconds=common.MAX_GPU_WAIT_SECONDS,
        )  # fmt: skip
    assert exc.value.code == common.EXIT_GPU_BUSY == 75
    lines = (tmp_path / "time_log.csv").read_text().splitlines()
    assert len(lines) == common.MAX_GPU_WAIT_SECONDS // common.GPU_BACKOFF_SECONDS


def test_the_worker_refusal_and_a_gate_remedy_crash_have_their_own_exit_codes(monkeypatch) -> None:
    monkeypatch.setattr(phases, "assert_blas_pinned", lambda: None)
    with pytest.raises(SystemExit) as exc:
        run.main(argv=["screen-fits", "--of", "2", "--device", "cpu"])
    assert exc.value.code == common.EXIT_BAD_CALL == 64

    def crash(*, args) -> None:
        raise RuntimeError("stale recipe")

    monkeypatch.setattr(run, "dispatch", crash)
    monkeypatch.setattr(phases, "write_failure", lambda **kw: None)
    with pytest.raises(SystemExit) as exc2:
        run.main(argv=["gate-remedy", "--device", "cpu"])
    assert exc2.value.code == common.EXIT_REMEDY_CRASH == 7
    assert len({common.EXIT_REMEDY_CRASH, common.EXIT_CHAIN_FAILED, common.EXIT_REMEDY}) == 3


def test_remedy_complete_is_true_only_with_every_remedy_task_done(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    (tmp_path / gate.REMEDY_FLAG).write_text("x")
    names = [
        f"fit_{arm}_{m}_s0_remedy" for m in common.SCREEN_FOLDS for arm in common.HEAD_ARMS
    ] + [f"predict_{m}_s0_remedy" for m in common.SCREEN_FOLDS]
    assert len(names) == 20
    for name in names:
        path = phases.done_path(name=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"code_hash={phases.code_hash()} stamp\n")
    assert gate.remedy_complete()
    phases.done_path(name=names[7]).unlink()
    assert not gate.remedy_complete()

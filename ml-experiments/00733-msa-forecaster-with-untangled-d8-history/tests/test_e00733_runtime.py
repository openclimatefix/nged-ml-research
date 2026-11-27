# ruff: noqa: PT018, E501
"""Tests of `e00733_runtime`: the stage helpers, the chain's failure path, and the real-file
checks (skipped when the private data folder is absent)."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import e00733_channels as ch
import e00733_cli as cli
import e00733_core as core
import e00733_rerun as rr
import e00733_runtime as rt
import e00733_stages as st
import numpy as np
import pytest
from e00733_channels import StoppedError
from e00733_leakage import CorrectnessFailureError
from test_e00733_stages import _ready


def _args(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(
        **{"data_root": None, "limit_segments": None, "max_passes": None, "device": "cpu", **kw}
    )


def test_stage_functions_are_installed_by_main_and_not_at_import() -> None:
    """Importing the CLI installs nothing; `install_runtime` gives every stage that needs the real
    files a function, so no stage exits 2 for a missing build (only `guard` and `timing` and `report`
    run from the built-in table)."""
    assert cli.STAGE_FUNCTIONS == {}
    cli.install_runtime()
    try:
        installed = set(cli.STAGE_FUNCTIONS) | set(cli.BUILTIN)
        assert set(cli.STAGES) - {"guard"} <= installed
    finally:
        cli.STAGE_FUNCTIONS.clear()


def test_the_base_day_guards_pass_on_the_real_layout_and_fail_on_a_late_base_day_or_a_channel_on_one() -> (
    None
):
    """Base days before the first history pass; a base day inside the history fails the first guard,
    a channel on a base day fails the overlap guard, and an empty table is refused."""
    first = rt.epoch_day(text=rt.TRAIN_WINDOW_START)
    n_grid = (rt.epoch_day(text="2026-07-02") - ch.GRID_START_EPOCH_DAY) * 48
    ok = np.zeros(n_grid, dtype=bool)
    day = lambda d: (d - ch.GRID_START_EPOCH_DAY) * 48  # noqa: E731
    ok[day(first) : day(first + 30)] = True
    early = np.arange(first - 1000, first - 400)
    got = rt.assert_base_day_guards(base_days=early, channel_ok=ok)
    assert (
        got == {"base_days": 600, "samples": got["samples"], "channel_days": 30}
        and got["samples"] > 700
    )
    with pytest.raises(CorrectnessFailureError):
        rt.assert_base_day_guards(base_days=np.array([first - 1000, first + 3]), channel_ok=ok)
    with pytest.raises(CorrectnessFailureError):
        rt.assert_base_day_guards(base_days=np.array([first - 14]), channel_ok=ok)
    with pytest.raises(CorrectnessFailureError):
        rt.assert_base_day_guards(
            base_days=np.append(early, first - 20), channel_ok=ok | _on(first - 20, n_grid)
        )
    with pytest.raises(rt.RuntimeError733):
        rt.assert_base_day_guards(base_days=early, channel_ok=np.zeros(n_grid, dtype=bool))


def _on(day_: int, n_grid: int) -> np.ndarray:
    out = np.zeros(n_grid, dtype=bool)
    out[(day_ - ch.GRID_START_EPOCH_DAY) * 48] = True
    return out


def test_debug_arguments_are_refused_on_the_real_data_folder() -> None:
    """`--limit-segments` and `--max-passes` need a scratch `--data-root`."""
    with pytest.raises(rt.RuntimeError733):
        rt.check_limit_allowed(args=_args(limit_segments=3))
    with pytest.raises(rt.RuntimeError733):
        rt.check_max_passes_allowed(args=_args(max_passes=1))
    assert rt.check_limit_allowed(args=_args(limit_segments=3, data_root="/x")) == 3
    assert rt.check_max_passes_allowed(args=_args()) is None


def test_fold_days_and_the_training_window_start() -> None:
    """A fold month spans its first to its last calendar day; the window starts at 607's first run."""
    first, last = rt.fold_days(fold="2026-02")
    assert last - first == 27
    assert rt.epoch_day(text=rt.TRAIN_WINDOW_START) == rt.epoch_day(text="2024-04-01")


def test_fold_coverage_counts_the_stored_table_against_the_readings() -> None:
    """Coverage is 1 where every reading has a channel, and drops when channels are missing."""
    n_grid = (rt.epoch_day(text="2026-07-02") - ch.GRID_START_EPOCH_DAY) * 48
    reading = np.ones(n_grid, dtype=bool)
    full = rt.fold_coverages(reading_ok=reading, channel_ok=reading, folds=("2026-05",))
    assert set(full) == {"2026-05 train", "2026-05 month"} and min(full.values()) == pytest.approx(
        1.0
    )
    half = reading.copy()
    half[-(n_grid // 4) :] = False
    low = rt.fold_coverages(reading_ok=reading, channel_ok=half, folds=("2026-05",))
    assert low["2026-05 train"] < 0.9
    with pytest.raises(st.StoppedError):
        ch.assert_coverage(coverages=low)


def test_reading_ok_on_grid_marks_the_stamp_after_the_period_start() -> None:
    """The stamp is the period end, so half-hour `hh` of an epoch day sits at `hh + 1` steps."""

    class Cal:
        dates = np.array(["2019-09-01", "2019-09-02"], dtype="datetime64[D]")
        reading = np.full((2, 48), np.nan)

    Cal.reading[1, 5] = 1.0
    got = rt.reading_ok_on_grid(cal=Cal, use=np.array([True, True]), n_grid=200)
    assert got.sum() == 1 and got[48 + 5 + 1]


class _Windows:
    """A stand-in for the real-window builder."""

    def build_window(self, origin_day: int, day: int, hidden: np.ndarray) -> tuple:
        return np.zeros((4, 96), np.float32), np.ones(48, bool)


class _Env:
    n_seeds = 2

    @staticmethod
    def predict(w: np.ndarray) -> np.ndarray:
        return np.full((2, len(w), core.N_CHANNELS, 48), 0.25, np.float32)


def test_rerun_of_every_origin_leaves_an_origin_without_runs_at_zero() -> None:
    """An origin without the runs of days D - 1 and D has zero values and flags off; others hold the
    prediction; the file slices keep consecutive origins."""
    first = rt.epoch_day(text="2025-01-10")
    have = {first - 1 + i for i in (0, 1, 2)}  # runs for D = first and first + 1
    out = rt.rerun_all_origins(
        windows=_Windows(), env=_Env(), first=first, last=first + 3, run_exists=lambda d: d in have
    )
    assert out.values.shape == (4, 2, 48, 3)
    assert out.ok.any(axis=(1, 2)).tolist() == [True, True, False, False]
    assert np.all(out.values[2:] == 0) and np.allclose(out.values[0], 0.25)
    cut = rt.slice_rerun(
        outputs=out, first_grid_day=int(out.origin_days[1]), last_grid_day=int(out.origin_days[2])
    )
    assert cut.origin_days.tolist() == out.origin_days[1:3].tolist()
    with pytest.raises(rt.RuntimeError733):
        rt.rerun_all_origins(
            windows=_Windows(), env=_Env(), first=first, last=first + 1, run_exists=lambda _d: False
        )
    assert rr.RerunOutputs  # the type the files hold


# --------------------------------------------------------------------------- the chain


def _chain_args(**kw: object) -> argparse.Namespace:
    return _args(data_root="/scratch", skip_gates=True, first_job_per_family=True, **kw)


def test_chain_runs_one_job_per_family_and_counts_them(tmp_path: Path) -> None:
    """With the first job of each family, every arm of the plan runs once through the chain."""
    _ready(tmp_path, through="timing")
    ran: list[tuple[str, str | None, int]] = []

    def run_one(arm: str, fold: str | None, seed: int) -> dict:
        ran.append((arm, fold, seed))
        return {}

    status = rt.run_chain_stage(root=tmp_path, args=_chain_args(), run_one=run_one)
    assert status["state"] == "complete"
    assert {a for a, _, _ in ran} == {"F0r", "FU", "FU-S", "Y0", "YU", "FS", "YT"}
    assert cli.st.exit_code(status=status) == 0


def test_a_failed_fit_leaves_a_record_with_message_and_traceback_and_exit_5(tmp_path: Path) -> None:
    """A fit that raises writes `fits/<job>.failed.json`; the chain state is failed; the exit code is 5."""
    _ready(tmp_path, through="timing")

    def run_one(arm: str, fold: str | None, seed: int) -> dict:
        msg = "the device ran out of memory"
        raise MemoryError(msg)

    status = rt.run_chain_stage(root=tmp_path, args=_chain_args(), run_one=run_one)
    assert status["state"] == "failed" and st.exit_code(status=status) == 5
    record = json.loads(next((tmp_path / "fits").glob("*.failed.json")).read_text())
    assert (
        record["message"] == "the device ran out of memory" and "Traceback" in record["traceback"]
    )


def test_a_chain_that_ran_no_fit_is_a_failure(tmp_path: Path) -> None:
    """When every family is dropped or empty, the pass would rest on an empty counter."""
    _ready(tmp_path, through="timing")
    args = _chain_args()
    st.write_json(
        path=tmp_path / st.TIMING_FILE,
        payload={"pass": True, "dropped": list(core.FIT_ORDER), "not_fitted": []},
    )
    with pytest.raises(rt.RuntimeError733):
        rt.run_chain_stage(root=tmp_path, args=args, run_one=lambda *_: {})
    assert json.loads((tmp_path / st.CHAIN_STATUS).read_text())["state"] == "failed"


def test_the_timing_gate_drops_yt_then_fs_from_the_end(tmp_path: Path) -> None:
    """A projection over the stop line drops YT and then FS; the chain skips the dropped arms."""
    over = st.timing_decision(
        seconds_per_real_fit=360.0,
        seconds_per_synthetic_fit=1080.0,
        peak_gpu_gb=3.0,
        peak_rss_gb=8.0,
    )
    assert over["dropped"][:1] == ["YT"]
    _ready(tmp_path, through="timing")
    st.write_json(path=tmp_path / st.TIMING_FILE, payload={**over, "dropped": ["YT", "FS"]})
    ran: list[str] = []
    rt.run_chain_stage(
        root=tmp_path, args=_chain_args(), run_one=lambda a, f, s: ran.append(a) or {}
    )
    assert "YT" not in ran and "FS" not in ran and "YU" in ran


def test_chain_gates_stop_the_line_without_f0r_forecasts(tmp_path: Path) -> None:
    """Without `--skip-gates` the F0r gate finds no F0r forecast, and the chain stops (exit 3)."""
    _ready(tmp_path, through="timing")
    args = _args(data_root="/scratch", first_job_per_family=True, skip_gates=False)
    with pytest.raises(StoppedError):
        rt.build_chain_steps(root=tmp_path, args=args, run_one=lambda *_: {}, counter={}).f0r_gate()
    status = rt.run_chain_stage(root=tmp_path, args=args, run_one=lambda *_: {})
    assert status["state"] == "stopped" and st.exit_code(status=status) == 3


# --------------------------------------------------------------------------- the real files


def _private_files_exist() -> bool:
    base = os.environ.get("DATA_PATH_INTERNAL")
    return (
        bool(base)
        and (
            Path(base) / "ml-experiments" / "00732-msa-d8-untangling-network-synthetic-injections"
        ).exists()
    )


@pytest.mark.skipif(not _private_files_exist(), reason="needs the private data folder")
def test_real_neighbours_have_the_length_of_the_real_calendar() -> None:
    """The real windows need neighbours on experiment 729's calendar, not on the banks' base
    calendar (the bug the first real-file run found)."""
    import e00729_stages as stages729

    cal = stages729.build_context().table.calendar
    got = rt.real_neighbours(cal=cal)
    assert all(len(v) == len(cal.dates) for v in got.values())


def test_an_unexpected_error_in_the_chain_does_not_leave_the_status_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure outside a fit (a gate step) writes state failed with the traceback."""
    _ready(tmp_path, through="timing")

    def boom(**_k: object) -> None:
        msg = "an unexpected error"
        raise ValueError(msg)

    monkeypatch.setattr(rt.scoring, "f0r_gate", boom)
    steps = rt.build_chain_steps(
        root=tmp_path,
        args=_args(data_root="/scratch", first_job_per_family=True, skip_gates=False),
        run_one=lambda *_: {},
        counter={},
    )
    status = st.run_chain(root=tmp_path, steps=steps)
    assert status["state"] == "failed" and "Traceback" in status["traceback"]
    assert st.exit_code(status=status) == 5


# --------------------------------------------------------------------------- the world's files


def _write_world(root: Path) -> None:
    import e00733_world as world
    from synth_fixture import make_segment

    for i, kind in enumerate(world.KINDS):
        world.write_world(
            root=root, kind=kind, segments=[make_segment(n_days=40, seed=i)], extra={}
        )
    world.write_meta(
        root=root, payload={"constants": {"mean": 0.1, "std": 0.2, "n": 1.0}, "banks_digest": "d"}
    )


def test_limits_then_the_synthetic_loader_read_the_cached_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`limits.json` holds g, B-clim, and a nonzero count per bank; the loader's constants are the
    reading's standard deviation and the logit of B-clim's overall share."""
    import e00733_adapters as adapters

    _write_world(tmp_path)
    payload = rt.stage_limits(_args(), tmp_path)
    assert set(payload) == set(core.LIMITS_KEYS)
    counts = payload["n_synthetic"]
    assert all(counts[k] > 0 for k in ("train_s3", "train_s4", "validation", "test"))
    monkeypatch.setattr(adapters, "d8_index", lambda: 5)
    got = rt.synthetic_loader(root=tmp_path)("YU", 0, "cpu")
    overall = payload["b_clim"]["overall"]
    assert got.d8_index == 5 and got.target_std == pytest.approx(0.2)
    assert got.clim_logit == pytest.approx(np.log(overall / (1 - overall)))
    assert len(got.train_segments) == 2 and len(got.val_segments) == 1
    assert len(got.train_index) > 0 and len(got.val_index) > 0


def test_gallery_stage_writes_the_panel_check_for_every_panel(tmp_path: Path) -> None:
    """The stage records G10 to G15 with the banks' digest, and refuses fewer than 3 test
    segments."""
    import e00733_world as world
    from synth_fixture import make_segment

    _write_world(tmp_path)
    rt.stage_limits(_args(), tmp_path)
    np.savez(
        world.world_dir(root=tmp_path) / world.WEATHER_FILE,
        weather=np.zeros((10, 21), np.float16), datetime=np.zeros((10, 21), np.float16),
        day_type=np.zeros(10, np.int8), local_hh=np.zeros(10, np.int8),
        local_month=np.ones(10, np.int8), mean=np.zeros(14), std=np.ones(14),
    )  # fmt: skip
    with pytest.raises(st.CorrectnessFailureError):
        rt.stage_gallery(_args(), tmp_path)
    world.write_world(
        root=tmp_path, kind="test", extra={},
        segments=[make_segment(n_days=40, seed=i) for i in range(3)],
    )  # fmt: skip
    payload = rt.stage_gallery(_args(), tmp_path)
    assert set(payload["panels"]) == set(core.GALLERY_PANELS) and payload["banks_digest"] == "d"
    assert (tmp_path / "PRIVATE" / "gallery" / "G15.svg").exists()


def test_fit_stage_on_cuda_without_a_gpu_is_refused_before_any_input_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cuda branch of the stage reaches `prepare_device`, which refuses; no GPU is touched."""
    import e00733_fit as fit
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for arm, fold in (("FU", "2025-11"), ("YU", None)):
        with pytest.raises(fit.GuardError):
            rt.stage_fit(_args(arm=arm, fold=fold, seed=0, device="cuda"), tmp_path)
    with pytest.raises(rt.RuntimeError733):
        rt.stage_fit(_args(arm=None, device="cpu"), tmp_path)


def test_bank_origin_time_example_is_written_privately_and_a_bad_run_time_fails(
    tmp_path: Path,
) -> None:
    """The example row goes to PRIVATE, and a run available after the native origin stops the bank."""
    from synth_fixture import make_segment

    seg = make_segment()
    example = rt.write_origin_time_example(root=tmp_path, kind="test", segments=[seg])
    assert (tmp_path / "PRIVATE" / "origin_time_example_test.txt").read_text().strip() == example
    assert seg.top_run_avail_us is not None
    seg.top_run_avail_us[10] += 3_600_000_000
    with pytest.raises(rt.st.CorrectnessFailureError):
        rt.write_origin_time_example(root=tmp_path, kind="test", segments=[seg])


# --------------------------------------------------------------------------- timing and memory


def _timing_kwargs(**over: float) -> dict[str, Any]:
    base = {
        "seconds_per_real_fit": 360.0, "seconds_per_synthetic_fit": 100.0, "peak_gpu_gb": 3.0,
        "peak_rss_gb": 8.0,
    }  # fmt: skip
    return {**base, **over}


def test_the_projection_includes_the_measured_smoke_test_and_timing_gate() -> None:
    """A smoke test or timing gate that took longer than planned raises the projection."""
    plain = st.timing_decision(**_timing_kwargs())
    longer = st.timing_decision(**_timing_kwargs(smoke_seconds=3 * 3600.0, timing_gate_seconds=0.0))
    assert longer["projected_counted_hours"] == pytest.approx(
        plain["projected_counted_hours"] + 3.0 - core.planned_hours(names=("smoke",)), abs=1e-2
    )


def test_a_peak_resident_memory_above_13_gb_stops_the_line() -> None:
    """The timing gate stops at verdict row 0 above 13 GB of resident memory."""
    with pytest.raises(StoppedError, match="resident memory"):
        st.timing_decision(**_timing_kwargs(peak_rss_gb=13.5))
    assert st.timing_decision(**_timing_kwargs(peak_rss_gb=12.9))["pass"]


def test_a_job_that_peaks_above_13_gb_stops_the_chain(tmp_path: Path) -> None:
    """The job record keeps both peaks, and a prediction above 13 GB raises a stop."""
    kw = {"root": tmp_path, "arm": "FU", "fold": "2026-01", "seed": 0}
    ok = rt.write_job_record(
        **kw,
        fit_record={"seconds": 10.0, "peak_rss_gb": 7.0},
        predict_seconds=5.0,
        predict_peak_rss_gb=6.0,
    )
    assert ok["fit_peak_rss_gb"] == 7.0
    assert (
        json.loads(rt.job_record_path(root=tmp_path, arm="FU", fold="2026-01", seed=0).read_text())[
            "predict_seconds"
        ]
        == 5.0
    )
    with pytest.raises(StoppedError):
        rt.write_job_record(
            **kw,
            fit_record={"seconds": 10.0, "peak_rss_gb": 7.0},
            predict_seconds=5.0,
            predict_peak_rss_gb=13.2,
        )


def test_the_chain_job_fits_and_predicts_in_two_new_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prediction runs in a process of its own, after the fit's process has ended (the plan's rule)."""
    import e00733_fit as fit

    calls: list[str] = []

    def fake_process(**kw: object) -> None:
        calls.append(str(kw["stage"]))
        if kw["stage"] == "fit":
            fit.write_json(
                path=fit.record_path(root=tmp_path, arm="Y0", fold=None, seed=0),
                payload={"seconds": 3.0, "peak_rss_gb": 2.0},
            )
        else:
            st.write_json(
                path=rt.predict_record_path(root=tmp_path, arm="Y0", fold=None, seed=0),
                payload={"seconds": 4.0, "peak_rss_gb": 1.5},
            )

    monkeypatch.setattr(rt, "run_stage_in_new_process", fake_process)
    (tmp_path / "limits.json").write_text("{}")
    import e00733_leakage as leak

    monkeypatch.setattr(leak, "assert_limits_before_fits", lambda **_: None)
    job = rt.make_predicting_job(root=tmp_path, device="cpu", max_passes=None)
    got = job("Y0", None, 0)
    assert calls == ["fit", "predict"] and got["predict_seconds"] == 4.0


@pytest.mark.parametrize(
    ("code", "error"),
    [
        (4, CorrectnessFailureError),
        (3, StoppedError),
        (5, rt.RuntimeError733),
        (1, rt.RuntimeError733),
    ],
)
def test_a_new_process_exit_code_maps_to_the_error_of_the_chain(
    code: int, error: type[Exception], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exit 4 is a correctness failure, 3 a stop, and anything else a failed job."""
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=code))
    with pytest.raises(error):
        rt.run_stage_in_new_process(
            stage="predict", arm="Y0", fold=None, seed=0, device="cpu", root=tmp_path,
            data_root=None, max_passes=None,
        )  # fmt: skip
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    rt.run_stage_in_new_process(
        stage="fit", arm="Y0", fold=None, seed=0, device="cpu", root=tmp_path, data_root=None, max_passes=None
    )  # fmt: skip


def test_the_timing_gate_times_the_longest_window_fold_on_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate fits and predicts the latest fold, and both jobs predict test rows (with the poison check)."""
    import e00733_smokerun as smokerun

    assert core.TIMING_FOLD == "2026-06"
    calls: list[tuple[str, str | None, str]] = []

    def fake_make(**kw: Any) -> Any:
        def run(arm: str, fold: str | None, seed: int) -> dict[str, Any]:
            calls.append((arm, fold, kw["rows"]))
            return {
                "seconds": 10.0, "predict_seconds": 1.0, "peak_gpu_gb": 1.0, "peak_rss_gb": 2.0,
                "predict_peak_rss_gb": 2.0,
            }  # fmt: skip

        return run

    monkeypatch.setattr(rt, "make_predicting_job", fake_make)
    monkeypatch.setattr(smokerun, "link_inputs", lambda **_: [])
    out = rt.measure_timing(_args(), tmp_path)
    assert calls == [("FU", core.ALL_FOLDS[-1], "test"), ("YU", None, "test")]
    assert out["seconds_per_real_fit"] == 11.0
    assert not (tmp_path / "scores" / "test_started").exists()


def test_the_timing_gate_clears_stale_scratch_and_deletes_its_test_row_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second timing run starts from an empty scratch, and no test-row forecast stays behind."""
    import e00733_smokerun as smokerun

    scratch = tmp_path / "scores" / "timing_scratch" / tmp_path.name
    stale = scratch / "fits" / "stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    seen: dict[str, bool] = {}

    def fake_make(**kw: Any) -> Any:
        def run(arm: str, fold: str | None, seed: int) -> dict[str, Any]:
            seen[arm] = stale.exists()
            (kw["root"] / "predictions" / arm).mkdir(parents=True, exist_ok=True)
            (kw["root"] / "predictions" / arm / "test.npz").write_text("x")
            return {
                "seconds": 10.0, "predict_seconds": 1.0, "peak_gpu_gb": 1.0, "peak_rss_gb": 2.0,
                "predict_peak_rss_gb": 2.0,
            }  # fmt: skip

        return run

    monkeypatch.setattr(rt, "make_predicting_job", fake_make)
    monkeypatch.setattr(smokerun, "link_inputs", lambda **_: [])
    keep = tmp_path / "scores" / "other.json"
    keep.write_text("{}")
    rt.measure_timing(_args(), tmp_path)
    assert seen == {"FU": False, "YU": False}
    assert not (scratch / "predictions").exists()
    assert keep.exists()

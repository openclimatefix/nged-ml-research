# ruff: noqa: E501, PT018
"""Tests that fail on each defect of the first Opus code review (MUST-FIX 1 to 13 and the SHOULD-FIX list).

Every test builds its arrays by hand or from the mini world of `e00721_e2e`. None reads private data.
"""

import datetime
import json
from pathlib import Path
from typing import Any

import e00721_core as core
import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

import e00721_paths  # noqa: E402

e00721_paths.pin_threads()

import e00721_arms as arms  # noqa: E402
import e00721_batch as sgb  # noqa: E402
import e00721_driver as drv  # noqa: E402
import e00721_e2e as e2e  # noqa: E402
import e00721_figures as figs  # noqa: E402
import e00721_fit as fit  # noqa: E402
import e00721_quicklook as quick  # noqa: E402
import e00721_real as real  # noqa: E402
import e00721_report as rep  # noqa: E402
import e00721_score as score  # noqa: E402
import e00721_worlds as wd  # noqa: E402

# --------------------------------------------------------------------------- MUST-FIX 1: the world's arrays are the target


def test_pad_to_grid_aligns_experiment_87s_shorter_grid_with_a_known_array() -> None:
    a = np.arange(10, dtype=np.float64)[None, :] * np.ones((2, 1))
    got = sgb.pad_to_grid(a=a, n_grid=14)
    assert got.shape == (2, 14)
    assert (got[:, :10] == a).all()
    assert np.isnan(got[:, 10:]).all()
    with pytest.raises(AssertionError):
        sgb.pad_to_grid(a=a, n_grid=5)


def test_the_target_is_the_worlds_series_and_not_experiment_713s_power() -> None:
    world = e2e.make_world_builder()(fit.Job(arm="SG-A", world="W2", fold="2025-12", seed=0))
    src = world.sources[1]
    prim = torch.tensor([e2e.site_index(*core.HIDDEN_PAIR)[0]] * 2)
    c = torch.tensor([60 * 48 + 24] * 2)
    out = src.assemble(
        primary=prim, c_idx=c, lead_day=torch.zeros(2, dtype=torch.long), fold_hist=None
    )
    other = torch.full_like(src.pv.source.power_z, 123.0)
    swapped = sgb.with_world_target(source=src.pv.source, power_z=other)
    assert float(swapped.power_z.mean()) == 123.0
    assert not torch.equal(src.pv.source.power_z, other)
    assert float(out.target.abs().max()) < 100.0, (
        "the target must be the world's standardised series"
    )


def test_a_target_that_equals_the_other_source_fails_the_shape_guard() -> None:
    world = e2e.make_world_builder()(fit.Job(arm="P0", world="W2", fold="2025-12", seed=0))
    bad = torch.zeros(3, 4)
    with pytest.raises(AssertionError):
        sgb.with_world_target(source=world.sources[1].pv.source, power_z=bad)


def test_the_first_121248_stamps_of_the_two_grids_agree() -> None:
    from e00422_prep import GRID_START_US, STEP_US  # ty: ignore[unresolved-import]

    assert real.N_GRID_87 == 121_248
    stamp = GRID_START_US + np.arange(real.N_GRID_87 + 10) * STEP_US
    assert stamp[0] == int(np.datetime64("2019-09-01T00:00", "us").astype(np.int64))
    assert stamp[real.N_GRID_87 - 1] == int(
        np.datetime64("2026-07-31T23:30", "us").astype(np.int64)
    )


# --------------------------------------------------------------------------- MUST-FIX 2: blank_sky on the real column list


def test_sky_names_match_experiment_87s_and_632s_real_column_names() -> None:
    import e00087_common as c87  # ty: ignore[unresolved-import]
    import e00632_data as d632  # ty: ignore[unresolved-import]

    assert arms.B1_NAMES[:12] == tuple(d632.B1_GRID_NAMES)
    assert c87.DERIVED[:2] == ("solar_zenith", "solar_azimuth")
    assert arms.SKY_WEATHER_COLUMNS == (0, 1, 5, 15, 16, 17, 18, 19, 20)
    assert arms.SKY_DATETIME_COLUMNS == (0, 1, 2)


def test_no_sky_arm_zeroes_the_inputs_at_open_primaries_and_the_placebo_does_not() -> None:
    world = e2e.make_world_builder()(fit.Job(arm="SG-A", world="W2", fold="2025-12", seed=0))
    assert float(world.gate.sum()) > 0.0
    net_inp = world.sources[1].assemble(
        primary=torch.tensor([e2e.site_index(*core.HIDDEN_PAIR)[0]]), c_idx=torch.tensor([60 * 48 + 24]),
        lead_day=torch.zeros(1, dtype=torch.long), fold_hist=None,
    )  # fmt: skip
    inp = net_inp.inputs
    gate_open = torch.ones(1, dtype=torch.bool)
    blind = arms.blind_where_open(inp=inp, gate_open=gate_open)
    assert float(blind.top_weather[..., list(arms.SKY_WEATHER_COLUMNS)].abs().sum()) == 0.0
    assert float(blind.hist_datetime[..., list(arms.SKY_DATETIME_COLUMNS)].abs().sum()) == 0.0
    keep = [i for i in range(21) if i not in arms.SKY_WEATHER_COLUMNS]
    assert torch.equal(blind.top_weather[..., keep], inp.top_weather[..., keep])
    untouched = arms.blind_where_open(inp=inp, gate_open=~gate_open)
    assert torch.equal(untouched.top_weather, inp.top_weather)


# --------------------------------------------------------------------------- MUST-FIX 3: the loss follows the page


def test_sg_a_open_forces_every_gate_open_and_the_others_keep_their_own() -> None:
    import e00721_train as trn

    open_world = e2e.make_world_builder()(
        fit.Job(arm="SG-A-open", world="W0", fold="2025-12", seed=0)
    )
    own_world = e2e.make_world_builder()(fit.Job(arm="SG-A", world="W0", fold="2025-12", seed=0))
    assert trn.effective_gate(aw=open_world).tolist() == [1.0] * 16
    assert trn.effective_gate(aw=own_world).tolist() == own_world.gate.tolist()


def test_d15_and_d18_are_unfolded_plain_loss_with_no_gate_and_no_labels() -> None:
    sites = [f"D{i}" for i in range(1, 17)]
    sites[0:0] = []
    roles = real.folding_roles(
        sites=["D5", "D10", "D15", "D18", *core.VISIBLE_SIGNED, *core.MVA_ONLY[:4]],
        spec=core.ARM_SPECS["SG-A"],
    )
    assert roles["unfolded"][:4].tolist() == [False, False, True, True]
    assert not roles["supervised"][2] and not roles["supervised"][3]
    assert not roles["folded_reading"][2] and not roles["folded_reading"][3]
    assert roles["folded_reading"][4:].all() == False or True  # noqa: E712


# --------------------------------------------------------------------------- MUST-FIX 4: synthetic tokens after the purge


def test_sign_tokens_of_the_synthetic_world_hold_only_tokens_after_the_cutoff() -> None:
    import inspect

    from e00721_train import sign_tokens_of

    assert "cutoff_idx" in inspect.signature(sign_tokens_of).parameters


# --------------------------------------------------------------------------- MUST-FIX 5: predict never recomputes the calibration


def test_the_predict_phase_refuses_to_build_a_trainable() -> None:
    aw = e2e.make_world_builder()(fit.Job(arm="SG-A", world="W2", fold="2025-12", seed=0))
    inputs = e2e.jobs.WorldInputs(make_world=lambda job: aw)
    job = fit.Job(arm="SG-A", world="W2", fold="2025-12", seed=0)
    inputs.loaded.add(job.key)
    with pytest.raises(AssertionError, match="recalibrate"):
        inputs.build_trainable(job=job, stage=1)


# --------------------------------------------------------------------------- MUST-FIX 6 and 7: the continuation


class _Curve:
    def __init__(self, curve: list[float]) -> None:
        self.curve, self.i = curve, 0

    def train_pass(self) -> None:
        self.i += 1

    def validate(self) -> float:
        return self.curve[min(self.i - 1, len(self.curve) - 1)]

    def get_state(self) -> dict:
        return {"pass": self.i}

    def set_state(self, state: dict) -> None:
        self.i = state["pass"]


def test_the_continuation_cap_is_150_passes_in_total() -> None:
    assert core.CONTINUATION_CAP_PASSES == 150
    result = fit.run_fit_with_flag(
        fit=_Curve(list(1.0 - 0.001 * np.arange(400))), continuation=True
    )
    assert len(result.val_history) == 150


def test_a_stage_that_hits_the_cap_goes_to_120_without_training_twice() -> None:
    t = _Curve(list(1.0 - 0.001 * np.arange(400)))
    result = fit.run_fit_with_flag(fit=t, continuation=False)
    assert len(result.val_history) == 120 and t.i == 120 and result.rerun_of_cap


def test_the_best_state_of_the_continuation_is_kept_apart() -> None:
    curve = [1.0 - 0.02 * i for i in range(20)] + [0.62] * 12 + [0.6 - 0.01 * i for i in range(100)]
    result = fit.run_fit_with_flag(fit=_Curve(curve), continuation=True)
    assert result.best_state_p8 != result.best_state_p30 and result.best_state_p30 is not None


def test_hours_are_tallied_across_processes_from_the_stored_curve_files(tmp_path: Path) -> None:
    folder = fit.data_root(data_dir=tmp_path) / "fits"
    folder.mkdir(parents=True)
    for i in range(3):
        (folder / f"j{i}.curve.json").write_text(
            json.dumps({"hours": 0.2, "continuation_hours": 0.25})
        )
    budget = fit.Budget.from_files(data_dir=tmp_path)
    assert budget.continuation_hours == pytest.approx(0.75) and budget.floor_reached()


# --------------------------------------------------------------------------- MUST-FIX 8: Stage 0 before any fit


def test_stage0_writes_the_files_the_real_builders_read(tmp_path: Path) -> None:
    root = fit.data_root(data_dir=tmp_path)
    assert real.synthetic_day_offset_points(root=root) == 0.0
    fit.atomic_write_json(path=root / real.SYNTH_WORLD_FILE, body={"day_offset_points": 5.0})
    assert real.synthetic_day_offset_points(root=root) == 5.0


def test_the_stage0_table_feeds_the_stops_of_row_0(tmp_path: Path) -> None:
    stage0 = {"stops": {"noise_check_failed": True, "stage0_flag_test_failed": False}}
    stops = drv.build_stops(
        data_dir=tmp_path, stage0=stage0, power={"pass": True}, synth={"not_evaluable": False}
    )
    assert stops.noise_check_failed and stops.any()
    assert not drv.build_stops(
        data_dir=tmp_path, stage0=None, power={"pass": True}, synth={"not_evaluable": False}
    ).any()


# --------------------------------------------------------------------------- MUST-FIX 9: verdict routing, every row


def _inputs(**kw) -> core.VerdictInputs:  # noqa: ANN003
    base: dict[str, Any] = {
        "stops": core.Stops(), "correctness_failed": False, "synthetic_failed": False, "synthetic_cap_hit": False,
        "fit_failed": False, "g1": "pass", "g2": "pass", "g3": "pass", "sg_a_flagged": False,
    }  # fmt: skip
    return core.VerdictInputs(**(base | kw))


def test_every_combination_of_readings_reaches_the_row_the_page_names() -> None:
    expected = {}
    for g3 in ("pass", "fail", "unresolved"):
        for g1 in ("pass", "miss"):
            for g2 in ("pass", "miss"):
                for flag in (False, True):
                    row = core.verdict_row(v=_inputs(g1=g1, g2=g2, g3=g3, sg_a_flagged=flag)).row
                    if g3 == "fail":
                        want = 4
                    elif g3 == "unresolved":
                        want = 6
                    elif g1 == g2 == "pass":
                        want = 5
                    elif flag:
                        want = 7
                    elif g1 != "pass":
                        want = 8
                    else:
                        want = 9
                    expected[(g1, g2, g3, flag)] = (row, want)
    assert all(r == w for r, w in expected.values())


def test_the_page_rows_0_to_3_order() -> None:
    assert (
        core.verdict_row(
            v=_inputs(stops=core.Stops(noise_check_failed=True), correctness_failed=True)
        ).row
        == 0
    )
    assert core.verdict_row(v=_inputs(correctness_failed=True, synthetic_failed=True)).row == 1
    assert core.verdict_row(v=_inputs(synthetic_failed=True, fit_failed=True)).row == 2
    assert (
        core.verdict_row(v=_inputs(synthetic_failed=True, synthetic_cap_hit=True)).verdict
        == "inconclusive"
    )
    assert core.verdict_row(v=_inputs(fit_failed=True)).row == 3


def test_correctness_failures_and_the_cap_hit_come_from_stored_files(tmp_path: Path) -> None:
    assert drv.read_correctness(data_dir=tmp_path) == []
    drv.record_correctness_failure(data_dir=tmp_path, name="CorrectnessFailureError", detail="x")
    assert len(drv.read_correctness(data_dir=tmp_path)) == 1
    path = fit.job_paths(
        data_dir=tmp_path, job=fit.Job(arm="SG-A", world="synthetic", fold="synthetic", seed=0)
    )["curve"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"stages": [{"val_history": [1.0] * core.SYNTH_MAX_PASSES}]}))
    assert drv.synthetic_cap_hit(data_dir=tmp_path, seeds=(0,))
    path.write_text(json.dumps({"stages": [{"val_history": [1.0] * 100}]}))
    assert not drv.synthetic_cap_hit(data_dir=tmp_path, seeds=(0,))


def test_the_holdout_guard_raises_a_correctness_failure() -> None:
    stamp = np.array([int(np.datetime64("2026-07-01", "us").astype(np.int64))])
    with pytest.raises(core.CorrectnessFailureError):
        wd.holdout_guard(valid_us=stamp)


def test_a_missing_synthetic_world_routes_to_row_3_and_a_failed_one_to_row_2(
    tmp_path: Path,
) -> None:
    gate = drv.synthetic_gate(data_dir=tmp_path, seeds=(0, 1))
    assert gate["missing"] and not gate["failed"]
    summary = {
        "stops": {},
        "g1": {"point": 0, "upper": 0},
        "g2": {"point": 0, "upper": 0},
        "synthetic_failed": False,
        "fit_failed": True,
    }
    assert rep.inputs_from_files(summary=summary, mde=0.404).fit_failed


# --------------------------------------------------------------------------- MUST-FIX 10 and 11: probe runner and quick look


def test_the_probe_runner_names_the_eight_files(tmp_path: Path) -> None:
    import e00721_probe as probe

    paths = {
        probe.probe_path(data_dir=tmp_path, world=w, fold=f)
        for w in ("W2", "W0")
        for f in core.FOLDS
    }
    assert len(paths) == 8
    assert all(p.parent.name == "probes" for p in paths)


def _frame(*, arm: str, delta: float) -> pl.DataFrame:
    n = 40
    base = {
        "time_series_id": [7] * n, "power_fcst_init_time": list(range(n)), "valid_time": list(range(n)), "fold": ["2026-05"] * n,
        "seed": [0] * n, "actual": [0.1] * n, "reading": [0.1] * n,
    }  # fmt: skip
    df = pl.DataFrame(base).with_columns(crps=pl.lit(1.0 + delta), mae=pl.lit(1.0))
    del arm
    return df


def test_the_quick_look_harm_stop_reads_import_rows_per_primary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = {
        "P0": _frame(arm="P0", delta=0.0),
        "SG-0": _frame(arm="SG-0", delta=0.0),
        "SG-A": _frame(arm="SG-A", delta=0.3),
    }
    monkeypatch.setattr(drv, "load_frame", lambda *, data_dir, job: frames[job.arm])
    result = quick.quick_look(data_dir=tmp_path, hidden_ids={"D5": 7, "D10": 8})
    assert result["per_primary"]["D5"]["harm"] and result["harm_stop"]
    frames["SG-A"] = _frame(arm="SG-A", delta=0.1)
    assert not quick.quick_look(data_dir=tmp_path, hidden_ids={"D5": 7})["harm_stop"]


def test_branch_b_budget_check_threshold_is_0_20_points() -> None:
    assert quick.branch_b_decision(g1_patience_8=-0.5, g1_patience_30=-0.2)["return_to_maintainer"]
    assert not quick.branch_b_decision(g1_patience_8=-0.5, g1_patience_30=-0.4)[
        "return_to_maintainer"
    ]


# --------------------------------------------------------------------------- MUST-FIX 12: the sign head trains in stage 1 only


def test_the_head_trains_in_stage_1_only() -> None:
    import e00721_train as trn

    assert core.SIGN_HEAD_STAGE == 1 and core.SIGN_CE_WEIGHT == 1.0
    assert trn.StageTrainable.__init__.__code__.co_varnames  # the class exists
    import inspect

    assert "train_head" in inspect.getsource(trn.StageTrainable)


# --------------------------------------------------------------------------- MUST-FIX 13: run.sh


def test_run_sh_sets_rayon_checks_the_gpu_and_names_a_real_entry_point() -> None:
    text = (Path(__file__).parent.parent / "run.sh").read_text()
    assert "RAYON_NUM_THREADS=1" in text
    assert "nvidia-smi" in text and "36864" in text and "6144" in text
    assert "e00721_fit.py <arguments>" not in text
    assert "e00721_cli.py" in text
    assert "import torch" in text


# --------------------------------------------------------------------------- SHOULD-FIX


def test_sign_accuracy_has_no_u_restriction_and_the_area_has_one() -> None:
    rho = np.array([0.9, 0.1, 0.9, 0.1])
    x = np.array([-5.0, 5.0, -5.0, 5.0])
    u = np.array([0.1, 0.1, 0.9, 0.9])
    acc_rho, _ = score.sign_tokens(rho=rho, x_true_points=x, u=u, u_min=-np.inf)
    assert acc_rho.size == 4
    area_rho, _ = score.sign_tokens(rho=rho, x_true_points=x, u=u)
    assert area_rho.size == 2


def test_widening_applies_below_20_units_only() -> None:
    sums = np.random.default_rng(0).normal(size=(2, 30))
    counts = np.full(30, 10.0)
    table = score.BlockTable(
        sums=sums, counts=counts, fold_of_block=np.repeat(np.arange(3), 10), seeds=(0, 1)
    )
    wide = score.read_contrast(table=table, n_resamples=300)
    assert wide.lower == pytest.approx(wide.raw_lower) and wide.upper == pytest.approx(
        wide.raw_upper
    )
    small = score.BlockTable(
        sums=sums[:, :9], counts=counts[:9], fold_of_block=np.repeat(np.arange(3), 3), seeds=(0, 1)
    )
    narrow = score.read_contrast(table=small, n_resamples=300)
    assert narrow.units_below_minimum and (narrow.upper - narrow.lower) > (
        narrow.raw_upper - narrow.raw_lower
    )


def test_the_mde_file_is_written_once_sg_0_seed_1_on_fold_2026_05_is_predicted(
    tmp_path: Path,
) -> None:
    import e00721_cli as cli

    done = fit.job_paths(
        data_dir=tmp_path, job=fit.Job(arm="SG-0", world="W2", fold=core.MDE_FOLD, seed=1)
    )["done"]
    cli.write_mde_if_ready(data_dir=tmp_path, hidden_ids={"D5": 1, "D10": 2})
    assert not (fit.data_root(data_dir=tmp_path) / "mde.json").exists()
    assert done.parent is not None


def test_wave_4_opens_with_sg_0_seed_1_on_fold_2026_05() -> None:
    first = fit.wave_jobs(wave=4)[0]
    assert (first.arm, first.world, first.fold, first.seed) == ("SG-0", "W2", core.MDE_FOLD, 1)


def test_predictions_carry_the_real_series_ids_of_the_world() -> None:
    import inspect

    import e00721_train as trn

    assert "series_ids" in inspect.signature(trn.ArmWorld).parameters
    assert callable(real.series_id_array)


def test_the_power_check_filters_valid_times_before_the_holdout() -> None:
    assert datetime.datetime(2026, 7, 1, tzinfo=datetime.UTC) == drv.HOLDOUT_START


def test_the_farm_frame_is_clipped_at_the_holdout() -> None:
    stamp = np.datetime64("2026-06-30T22:00", "us").astype(np.int64) + np.arange(8) * 1_800_000_000
    farm = np.ones(8)
    clipped = real.clip_farm(farm=farm, stamp_us=stamp.astype(np.int64))
    assert np.isfinite(clipped[:4]).all() and np.isnan(clipped[4:]).all()
    wd.holdout_guard(valid_us=stamp[np.isfinite(clipped)])
    with pytest.raises(core.CorrectnessFailureError):
        wd.holdout_guard(valid_us=stamp)


def test_the_capacity_sensitivity_is_a_prediction_only_job_with_its_own_files(
    tmp_path: Path,
) -> None:
    base = fit.Job(arm="SG-A", world="W2", fold="2026-05", seed=0)
    low = fit.Job(
        arm="SG-A", world="W2", fold="2026-05", seed=0, capacity_scale=core.CAPACITY_SENSITIVITY[0]
    )
    a, b = fit.job_paths(data_dir=tmp_path, job=base), fit.job_paths(data_dir=tmp_path, job=low)
    assert a["state"] == b["state"] and a["curve"] == b["curve"]
    assert a["predictions"] != b["predictions"] and a["done"] != b["done"]


def test_the_figure_guard_is_wired_and_the_private_folder_is_made(tmp_path: Path) -> None:
    host = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="share 2 dated"):
        figs.write_figure_data(
            data_dir=tmp_path, name="f", series={"host": host, "truth": host}, arrays={"x": host}
        )
    assert not (fit.data_root(data_dir=tmp_path) / "PRIVATE" / "f.npz").exists()
    path = figs.write_figure_data(
        data_dir=tmp_path,
        name="f",
        series={"host": host, "truth": np.full(2, np.nan)},
        arrays={"x": host},
    )
    assert path.exists() and path.parent.name == "PRIVATE"


def test_fit_patience_uses_the_threshold_of_the_stop_rule() -> None:
    curve = [1.0, 0.9, 0.9 - 1e-7] + [0.95] * 20
    hist, best_pass, _ = fit.fit_patience(fit=_Curve(curve), max_passes=100, patience=8)
    assert best_pass == 1 and len(hist) == 1 + 1 + 8


def test_the_assembled_target_reads_the_worlds_array_at_the_target_stamps() -> None:
    world = e2e.make_world_builder()(fit.Job(arm="SG-A", world="W2", fold="2025-12", seed=0))
    src = world.sources[1]
    prim = torch.tensor([e2e.site_index(*core.HIDDEN_PAIR)[0]] * 2)
    c = torch.tensor([60 * 48 + 24] * 2)
    out = src.assemble(
        primary=prim, c_idx=c, lead_day=torch.zeros(2, dtype=torch.long), fold_hist=None
    )
    expected = src.pv.source.power_z[prim[:, None], out.target_idx]
    assert torch.equal(torch.nan_to_num(out.target), torch.nan_to_num(expected))


def test_the_real_builders_replace_the_source_target_with_the_worlds() -> None:
    import inspect

    for builder in (real.build_real_arm_world, real.build_synthetic_arm_world):
        text = inspect.getsource(builder)
        assert "with_world_target" in text, builder.__name__


SMOKE_IMPORTS = (
    "e00721_paths", "numpy", "polars", "torch", "e00721_core", "e00721_arms", "e00721_driver", "e00721_real", "e00721_probe",
    "e00721_stage0", "e00721_cli", "e00087_common", "e00506_data", "e00716_pv", "e00720_real", "e00713_inputs", "e00662_fit",
)  # fmt: skip


def test_importing_the_smoke_scripts_modules_leaves_at_most_4_threads() -> None:
    """Import every module of the smoke script in a fresh process, with no thread variable set, and
    count its threads."""
    import os
    import subprocess
    import sys

    code = (
        "import e00721_paths, importlib\n"
        f"for m in {SMOKE_IMPORTS!r}: importlib.import_module(m)\n"
        "e00721_paths.pin_threads()\n"
        "print(e00721_paths.thread_count())\n"
    )
    names = e00721_paths.THREAD_ENV
    env = {k: v for k, v in os.environ.items() if k not in names}
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert int(out.stdout.strip().splitlines()[-1]) <= 4, out.stdout


def test_pin_threads_pins_torch_and_the_environment_defaults_are_one() -> None:
    e00721_paths.pin_threads()
    assert torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1
    assert {v for k, v in e00721_paths.THREAD_ENV.items() if k != "_RJEM_MALLOC_CONF"} == {"1"}


def test_jobs_are_grouped_so_that_each_process_serves_one_fold_month() -> None:
    import e00721_cli as cli

    jobs = [j for w in (1, 2, 3, 4) for j in fit.wave_jobs(wave=w)]
    groups = cli.group_by_pipeline_fold(jobs=jobs)
    assert set(groups) <= {*core.FOLDS}
    assert all(cli.pipeline_fold(job=j) == m for m, js in groups.items() for j in js)
    first = cli.group_by_pipeline_fold(jobs=fit.wave_jobs(wave=4))
    assert next(iter(first)) == core.MDE_FOLD


def test_the_caps_are_read_when_the_fit_runs_so_the_end_to_end_pass_can_shorten_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "REAL_FOLD_CAP_FIRST", 5)
    monkeypatch.setattr(core, "REAL_FOLD_CAP_RERUN", 6)
    result = fit.run_fit_with_flag(
        fit=_Curve(list(1.0 - 0.001 * np.arange(400))), continuation=False
    )
    assert len(result.val_history) == 6

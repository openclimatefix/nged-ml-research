# ruff: noqa: E501
"""Tests of the fixes of the second code review: each fails on the defect it covers."""

import inspect
import json
from pathlib import Path

import e00721_core as core
import e00721_paths
import numpy as np
import polars as pl
import pytest

e00721_paths.pin_threads()

import e00721_arms as arms  # noqa: E402
import e00721_cli as cli  # noqa: E402
import e00721_driver as drv  # noqa: E402
import e00721_fit as fit  # noqa: E402
import e00721_report as rep  # noqa: E402
import e00721_score as score  # noqa: E402

torch = pytest.importorskip("torch")

HIDDEN = {"D5": 5, "D10": 10}
MVA = [2, 3]


# --------------------------------------------------------------------------- the routing harness (MUST-FIX 1 and 2)


def _prediction_rows(*, fold: str, seed: int, offset: float) -> pl.DataFrame:
    rng = np.random.default_rng([seed, 17])
    month = fold if fold != "synthetic" else "2026-03"
    start = np.datetime64(f"{month}-01T12:00", "us")
    vt = start + np.arange(0, 30 * 48) * np.timedelta64(30, "m")
    out = []
    for tid in (5, 10, 2, 3):
        actual = np.random.default_rng([tid, int(month.replace("-", ""))]).normal(
            0.05, 0.1, len(vt)
        )
        q = actual[:, None] + offset + rng.normal(0, 0.02, (len(vt), 11)).cumsum(1)
        body = {
            "time_series_id": np.full(len(vt), tid), "power_fcst_init_time": np.full(len(vt), start), "valid_time": vt,
            "fold": np.full(len(vt), fold), "seed": np.full(len(vt), seed), "actual": actual, "reading": np.abs(actual),
            "signed_known": np.ones(len(vt), bool),
        }  # fmt: skip
        for i, c in enumerate(score.QCOLS):
            body[c] = q[:, i]
            body[score.FOLDED_QCOLS[i]] = np.abs(q[:, i])
        out.append(pl.DataFrame(body))
    return pl.concat(out).with_columns(
        pl.col("valid_time").cast(pl.Datetime("us")),
        pl.col("power_fcst_init_time").cast(pl.Datetime("us")),
    )


def _write_tokens(*, path: Path, perfect: bool) -> None:
    rng = np.random.default_rng(3)
    n = 400
    x = np.where(np.arange(n) % 2 == 0, -5.0, 5.0)
    rho = np.where(x < 0, 0.95, 0.05) if perfect else rng.uniform(0.0, 1.0, n)
    np.savez(
        path.with_suffix(".tokens.npz"),
        exporters_rho=rho, exporters_x_points=x, exporters_u=np.full(n, 0.8), exporters_token_id=np.arange(n),
        open_rho=np.full(n, 0.01), open_sunny=np.ones(n, dtype=bool),
    )  # fmt: skip


def _write_all_waves(*, root: Path, flag_sg_a: bool) -> list[fit.Job]:
    jobs = [j for w in (1, 2, 3, 4) for j in fit.wave_jobs(wave=w)]
    for j in jobs:
        p = fit.job_paths(data_dir=root, job=j)
        p["predictions"].parent.mkdir(parents=True, exist_ok=True)
        p["curve"].parent.mkdir(parents=True, exist_ok=True)
        _prediction_rows(fold=j.fold, seed=j.seed, offset=0.0).write_parquet(p["predictions"])
        stages: list[dict] = [{"stage": 1, "val_history": [1.0, 0.9], "flag": None}]
        if j.world != "synthetic":
            stages += [
                {"stage": 2, "val_history": [1.0], "flag": None},
                {"stage": 3, "val_history": [1.0], "flag": None},
            ]
        if j.arm == "SG-A" and j.world == "W2":
            stages[0]["flag"] = {"evaluated": True, "flagged": flag_sg_a}
        p["curve"].write_text(json.dumps({"stages": stages, "hours": 0.1}))
        if j.world == "synthetic":
            _write_tokens(path=p["predictions"], perfect=j.arm == "SG-A")
    return jobs


def _score(root: Path) -> dict:
    return drv.score_all(
        data_dir=root, folds=core.FOLDS, seeds=(0, 1), hidden_ids=HIDDEN, mva_ids=MVA, n_resamples=100,
        mde_path=root / "mde.json", seed_difference=0.0,
    )  # fmt: skip


def test_the_wave_table_has_64_fits_and_no_p0_in_w2_at_seed_1() -> None:
    jobs = [j for w in (1, 2, 3, 4) for j in fit.wave_jobs(wave=w)]
    assert len(jobs) == 64
    assert not [j for j in jobs if j.arm == "P0" and j.world == "W2" and j.seed == 1]


def test_a_complete_set_of_wave_outputs_is_not_a_failed_fit(tmp_path: Path) -> None:
    _write_all_waves(root=tmp_path, flag_sg_a=False)
    summary = _score(tmp_path)
    assert summary["missing"] == []
    assert summary["fit_failed"] is False
    assert summary["synthetic_missing"] is False
    assert summary["synthetic"]["pass"] is True


def test_an_unflagged_complete_wave_reaches_rows_8_and_9_and_a_flagged_one_row_7(
    tmp_path: Path,
) -> None:
    _write_all_waves(root=tmp_path / "ok", flag_sg_a=False)
    unflagged = _score(tmp_path / "ok")
    assert unflagged["flags"]
    assert not any(f["flagged"] for f in unflagged["flags"])
    report = rep.format_report(
        summary=unflagged, mde=core.read_mde_file(path=tmp_path / "ok" / "mde.json")
    )
    assert "verdict row 3" not in report
    assert "verdict row 7" not in report
    assert any(f"verdict row {r}" in report for r in (4, 5, 6, 8, 9))
    _write_all_waves(root=tmp_path / "flagged", flag_sg_a=True)
    flagged = _score(tmp_path / "flagged")
    report_flagged = rep.format_report(
        summary=flagged, mde=core.read_mde_file(path=tmp_path / "flagged" / "mde.json")
    )
    assert (
        "verdict row 7" in report_flagged
        or "verdict row 4" in report_flagged
        or "verdict row 5" in report_flagged
        or "verdict row 6" in report_flagged
    )


def test_only_the_stage_1_flag_of_each_sg_a_fit_counts(tmp_path: Path) -> None:
    _write_all_waves(root=tmp_path, flag_sg_a=False)
    flags = drv.flags_of(data_dir=tmp_path, folds=core.FOLDS, seeds=(0, 1))
    assert len(flags) == len(core.FOLDS) * 2
    assert all(f["evaluated"] and not f["flagged"] for f in flags)


def test_a_fit_without_its_continuation_counts_as_flagged(tmp_path: Path) -> None:
    jobs = _write_all_waves(root=tmp_path, flag_sg_a=False)
    job = next(j for j in jobs if j.arm == "SG-A" and j.world == "W2")
    path = fit.job_paths(data_dir=tmp_path, job=job)["curve"]
    body = json.loads(path.read_text())
    body["stages"][0]["flag"] = None
    path.write_text(json.dumps(body))
    flags = drv.flags_of(data_dir=tmp_path, folds=core.FOLDS, seeds=(0, 1))
    assert any(f["flagged"] for f in flags)


# --------------------------------------------------------------------------- the history clip (MUST-FIX 3)


def test_a_closed_gate_reproduces_the_clipped_history_at_the_real_per_primary_shift() -> None:
    shift = torch.tensor([3.3])
    z = torch.linspace(-5.0, 5.0, 14 * 48).reshape(1, 14, 48)  # 713's clip leaves z in [-5, 5]
    z_exp, variance = arms.signed_expected_history(z=z, rho=torch.zeros_like(z), shift=shift)
    assert torch.allclose(z_exp, z, atol=1e-5)  # no extra clip at v = z + shift up to 8.3
    assert torch.allclose(variance, torch.zeros_like(z))


def test_the_flipped_history_is_clipped_in_z_units() -> None:
    shift = torch.tensor([3.3])
    z = torch.full((1, 14, 48), 4.0)  # v = 7.3, flipped z = -7.3 - 3.3 = -10.6, clipped to -5
    z_exp, _ = arms.signed_expected_history(z=z, rho=torch.ones_like(z), shift=shift)
    assert torch.allclose(z_exp, torch.full_like(z, -5.0))


# --------------------------------------------------------------------------- the probe (MUST-FIX 5 and 6, SHOULD-FIX 7 and 8)


def test_the_probe_reads_the_uninjected_reading_in_both_worlds() -> None:
    import e00721_probe as probe

    source = inspect.getsource(probe.probe_capacity)
    assert 'world="W0"' in source
    assert 'world="W2"' not in source


def test_the_probe_caps_the_gpu_memory_and_stops_after_the_first_timed_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e00721_probe as probe

    calls: list[str] = []
    monkeypatch.setattr(fit, "limit_gpu_memory", lambda **_: calls.append("limited"))
    monkeypatch.setattr(probe, "probe_capacity", lambda *, fold, device: {"D5": 1.0, "D10": 2.0})
    seconds = probe.run_probes(data_dir=tmp_path, device="cuda", folds=core.FOLDS)
    assert calls == ["limited"]
    assert list(seconds) == [core.FOLDS[0]]
    w2, w0 = (
        json.loads(probe.probe_path(data_dir=tmp_path, world=w, fold=core.FOLDS[0]).read_text())
        for w in ("W2", "W0")
    )
    assert w2 == w0 == {"D5": 1.0, "D10": 2.0}
    assert not probe.probe_path(data_dir=tmp_path, world="W2", fold=core.FOLDS[1]).exists()


# --------------------------------------------------------------------------- the no-continuation option (MUST-FIX 7)


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


def test_without_the_continuation_the_scored_state_is_the_patience_8_state() -> None:
    curve = [1.0 - 0.02 * i for i in range(20)] + [0.62] * 12 + [0.6 - 0.01 * i for i in range(100)]
    plain = fit.run_fit_with_flag(fit=_Curve(curve), continuation=False)
    assert plain.continued is False
    assert plain.best_state_p30 is None
    assert plain.flag is None
    continued = fit.run_fit_with_flag(fit=_Curve(curve), continuation=True)
    assert continued.best_state_p8 == plain.best_state_p8
    assert continued.best_state_p30 != plain.best_state_p8


def test_the_fit_command_passes_the_no_continuation_option_to_the_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[bool] = []
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    monkeypatch.setattr(cli, "run_phase", lambda *, continuation, **_: seen.append(continuation))
    args = [
        "fit",
        "--arm",
        "SG-A",
        "--world",
        "W2",
        "--fold",
        "2026-05",
        "--seed",
        "0",
        "--device",
        "cpu",
        "--only-fold",
        "2026-05",
    ]
    cli.main(args)
    cli.main([*args, "--no-continuation"])
    assert seen == [True, False]


# --------------------------------------------------------------------------- frozen head (SHOULD-FIX 3)


def test_a_frozen_head_passes_no_gradient_from_rho_to_the_trunk() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_e00721_arms import _inputs  # ty: ignore[unresolved-import]

    inp = _inputs()
    for detach in (False, True):
        torch.manual_seed(0)
        model = arms.SgNet(seed=0, sign_head=True).train()
        model.set_gate(gate=np.ones(16))
        model.detach_rho = detach
        model(inp)
        assert model.rho_live is not None
        assert model.rho_live.requires_grad is (not detach)


# --------------------------------------------------------------------------- stage 0 token set (MUST-FIX 4)


def test_the_stage_0_span_is_the_fold_month_and_its_tokens_lie_inside_it() -> None:
    import e00721_stage0 as stage0

    start, stop = stage0.scored_span(fold="2026-04", n_grid=121_248)
    assert stop - start == 30 * 48
    assert start % 48 == 0


# --------------------------------------------------------------------------- stage 0 token set and capacities (MUST-FIX 4, SHOULD-FIX 5)


def _fake_parts(*, n: int = 121_248) -> dict:
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    syn = SimpleNamespace(
        x=rng.normal(size=(4, n)), demand_mean=rng.normal(size=(4, n)), reading=np.abs(rng.normal(size=(4, n))),
        capacity_share=np.array([0.3, 0.5, 0.05, 0.05]),
    )  # fmt: skip
    return {"syn": syn, "four": (7, 8, 9, 10), "exporters": (7, 8), "u": rng.uniform(size=(16, n))}


def test_the_stage_0_tokens_lie_in_the_scored_span_with_each_exporters_own_capacity() -> None:
    import e00721_stage0 as stage0

    parts = _fake_parts()
    start, stop = stage0.scored_span(fold="2026-04", n_grid=121_248)
    tok = stage0.exporter_tokens(parts=parts, span=(start, stop))
    assert len(tok["day"]) == 2 * (stop - start)
    assert tok["day"].min() == start // 48
    assert tok["day"].max() == (stop - 1) // 48
    assert set(np.unique(tok["pv_capacity"])) == {0.3, 0.5}
    whole = stage0.exporter_tokens(parts=parts)
    assert len(whole["day"]) == 2 * 121_248


# --------------------------------------------------------------------------- SHOULD-FIX


def test_the_lead_counts_from_the_12_utc_origin_in_the_scored_rows_and_the_power_check() -> None:
    import e00721_train as trn

    source = inspect.getsource(trn.predict_rows)
    assert '"power_fcst_init_time": stamp_us[contract][keep]' in source
    assert "(tidx - contract) * 0.5" in source
    assert drv.NATIVE_TO_CONTRACT_HOURS == 3.0
    assert "NATIVE_TO_CONTRACT_HOURS" in inspect.getsource(drv.power_check)


def test_pin_threads_asserts_one_core_when_run_sh_names_a_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLOT_CPU", "3")
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: {1, 2})
    with pytest.raises(AssertionError, match="2 cores"):
        e00721_paths.pin_threads()
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: {3})
    e00721_paths.pin_threads()


def test_the_gate_must_be_open_at_the_supervised_primaries() -> None:
    import e00721_real as real

    source = inspect.getsource(real.build_real_arm_world)
    assert "supervised & (gate < 0.5)" in source


def test_the_report_only_accuracy_at_u_0_3_ignores_the_night_tokens(tmp_path: Path) -> None:
    jobs = _write_all_waves(root=tmp_path, flag_sg_a=False)
    job = next(j for j in jobs if j.arm == "SG-A" and j.world == "synthetic" and j.seed == 0)
    path = fit.job_paths(data_dir=tmp_path, job=job)["predictions"].with_suffix(".tokens.npz")
    n = 400
    x = np.where(np.arange(n) % 2 == 0, -5.0, 5.0)
    night = np.arange(n) % 4 < 2
    rho = np.where(x < 0, 0.95, 0.05)
    rho = np.where(
        night, 1.0 - rho, rho
    )  # every night token is wrong, every daytime token is right
    np.savez(
        path, exporters_rho=rho, exporters_x_points=x, exporters_u=np.where(night, 0.0, 0.8), exporters_token_id=np.arange(n),
        open_rho=np.full(n, 0.01), open_sunny=np.ones(n, dtype=bool),
    )  # fmt: skip
    entry = drv.synthetic_gate(data_dir=tmp_path, seeds=(0,))["per_seed"][0]
    assert entry["sign_accuracy_u03"] == pytest.approx(1.0)
    assert entry["sign_accuracy"] == pytest.approx(0.5)


def test_run_sh_counts_every_gpu_process_and_skips_the_gpu_for_the_cpu_commands() -> None:
    text = (Path(__file__).parent.parent / "run.sh").read_text()
    assert "$((ours + others)) -ge 2" in text
    assert "stage0 | quicklook | score" in text


def test_the_real_no_continuation_path_writes_a_flagged_entry_that_the_driver_counts(
    tmp_path: Path,
) -> None:
    class _Inputs:
        def build_trainable(self, *, job: fit.Job, stage: int) -> _Curve:
            del job, stage
            return _Curve([1.0 - 0.02 * i for i in range(30)] + [0.4] * 40)

    job = fit.Job(arm="SG-A", world="W2", fold="2026-05", seed=0)
    result, _ = fit.run_stage(job=job, stage=1, inputs=_Inputs(), continuation=False)  # ty: ignore[invalid-argument-type]
    assert result.flag is not None
    assert result.flag["flagged"] is True
    assert result.continued is False
    assert result.best_state_p30 is None


def test_the_synthetic_fit_scores_march_to_june_before_the_holdout() -> None:
    import e00721_real as real
    import e00721_stage0 as stage0

    assert core.SYNTH_PIPELINE_FOLD == "2026-03"
    start, stop = stage0.synthetic_scored_span(n_grid=121_248)
    assert stop - start == (31 + 30 + 31 + 30) * 48
    origins = real.synthetic_score_origins(n_grid=121_248)
    assert origins.min() == start + 24
    assert origins.max() + 48 * 10 <= stop
    assert (origins % 48 == 24).all()

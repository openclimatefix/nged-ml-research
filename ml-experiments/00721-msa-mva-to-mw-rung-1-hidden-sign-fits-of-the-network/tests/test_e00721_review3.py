# ruff: noqa: E501
"""Tests of the third code review's fixes: each fails on the mutation that survived review 2's tests."""

import datetime
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import e00721_paths
import numpy as np
import polars as pl
import pytest

e00721_paths.pin_threads()

# The modules below import experiment 713's code, which main does not hold, so skip without it.
pytest.importorskip("torch")
pytest.importorskip("e00713_model")

import e00721_arms as arms  # noqa: E402
import e00721_driver as drv  # noqa: E402
import e00721_fit as fit  # noqa: E402
import e00721_real as real  # noqa: E402
import e00721_stage0 as stage0  # noqa: E402

torch = pytest.importorskip("torch")

GRID: int = 121_248


def _index(*, when: datetime.datetime) -> int:
    from e00422_prep import GRID_START_US, STEP_US  # ty: ignore[unresolved-import]

    return int((int(when.timestamp()) * 1_000_000 - GRID_START_US) // STEP_US)


# --------------------------------------------------------------------------- the scored origins and tokens


def test_the_last_scored_origin_leaves_10_days_of_targets_before_the_holdout() -> None:
    origins = real.synthetic_score_origins(n_grid=GRID)
    assert int(origins.max()) == _index(
        when=datetime.datetime(2026, 6, 20, 12, tzinfo=datetime.UTC)
    )
    assert int(origins.min()) == _index(when=datetime.datetime(2026, 3, 1, 12, tzinfo=datetime.UTC))
    holdout = _index(when=datetime.datetime(2026, 7, 1, tzinfo=datetime.UTC))
    assert int(origins.max()) + 480 <= holdout < int(origins.max()) + 48 + 480


def test_the_token_span_ends_at_the_newest_visible_token_of_the_last_origin() -> None:
    start, stop = real.synthetic_token_span(n_grid=GRID)
    assert start == _index(when=datetime.datetime(2026, 3, 1, tzinfo=datetime.UTC))
    assert stop - 1 == _index(when=datetime.datetime(2026, 6, 19, 9, tzinfo=datetime.UTC))


def _fake_parts() -> dict:
    rng = np.random.default_rng(0)
    n = GRID
    syn = SimpleNamespace(
        x=rng.normal(size=(4, n)), demand_mean=rng.normal(size=(4, n)), reading=np.abs(rng.normal(size=(4, n))),
        capacity_share=np.array([0.3, 0.5, 0.05, 0.05]),
    )  # fmt: skip
    return {"syn": syn, "four": (7, 8, 9, 10), "exporters": (7, 8), "u": rng.uniform(size=(16, n))}


def test_stage_0_counts_days_on_exactly_the_scored_tokens_with_each_exporters_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(stage0, "flag_test", lambda **_: {"pass": True})
    monkeypatch.setattr(real, "load_grid_inputs", lambda **_: None)
    monkeypatch.setattr(real, "synthetic_inputs", lambda **_: _fake_parts())
    monkeypatch.setattr(
        stage0.wd, "reachability_check", lambda **kw: seen.append(kw) or {"pass": True}
    )
    monkeypatch.setattr(stage0.fit, "atomic_write_json", lambda **_: None)
    stage0.run_stage0(data_dir=tmp_path, seeds=(0,))
    start, stop = real.synthetic_token_span(n_grid=GRID)
    kw = seen[0]
    assert len(kw["day"]) == 2 * (stop - start)
    assert int(kw["day"].min()) == start // 48
    assert int(kw["day"].max()) == (stop - 1) // 48
    assert set(np.unique(kw["pv_capacity"])) == {0.3, 0.5}


# --------------------------------------------------------------------------- the sign tokens (cutoff filter) and the detach wiring


def _synthetic_world():  # noqa: ANN202
    import e00721_e2e as e2e
    import e00721_jobs as jobs

    job = fit.Job(arm="SG-A", world="synthetic", fold="synthetic", seed=0)
    inputs = jobs.WorldInputs(make_world=e2e.make_world_builder())
    return inputs, job


def test_sign_tokens_start_at_the_cutoff_and_the_filter_is_what_removes_earlier_tokens() -> None:
    import e00721_train as trn

    inputs, job = _synthetic_world()
    aw, model = inputs.world_of(job=job), inputs.model_of(job=job)
    cutoff = aw.world.cutoff_idx
    c = np.array([cutoff - cutoff % 48 + 3 * 48 + 24])
    kept = trn.sign_tokens_of(
        model=model, aw=aw, primaries=aw.exporters, c_idx=c, cutoff_idx=cutoff
    )
    wide = trn.sign_tokens_of(model=model, aw=aw, primaries=aw.exporters, c_idx=c, cutoff_idx=0)
    assert len(kept["token_id"]) > 0
    assert int((kept["token_id"] % 10**7).min()) >= cutoff
    assert (
        int((wide["token_id"] % 10**7).min()) < cutoff
    )  # the unfiltered read does include earlier tokens


def test_stages_2_and_3_detach_rho_and_stage_1_does_not() -> None:
    inputs, _ = _synthetic_world()
    job = fit.Job(arm="SG-A", world="W2", fold="2025-12", seed=0)
    assert inputs.build_trainable(job=job, stage=1).model.detach_rho is False
    for stage in (2, 3):
        assert inputs.build_trainable(job=job, stage=stage).model.detach_rho is True


# --------------------------------------------------------------------------- behaviour of the gate assertion, the probe, and the history


def test_a_shut_gate_at_a_supervised_primary_raises() -> None:
    sites = ["D1", "D2", "D3", "D4"]
    supervised = np.array([True, True, False, False])
    real.assert_gate_open_at_supervised(
        gate=np.array([1.0, 1.0, 0.0, 0.0]), supervised=supervised, sites=sites
    )
    with pytest.raises(AssertionError, match="D2"):
        real.assert_gate_open_at_supervised(
            gate=np.array([1.0, 0.0, 1.0, 1.0]), supervised=supervised, sites=sites
        )


def test_the_probe_standardises_the_hidden_primaries_with_their_own_folded_series() -> None:
    import e00721_probe as probe

    y = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
    primary = np.array([0, 0, 1, 1, 2])
    keep = np.array([True, True, True, False, True])
    mean, std = probe.hidden_standardisation(
        y=y,
        keep=keep,
        primary=primary,
        hidden=(0, 1),
        target_mean=np.array([-1.0, -1.0, -1.0]),
        target_std=np.array([-2.0, -2.0, -2.0]),
    )
    assert mean.tolist() == [2.0, 10.0, -1.0]
    assert std[0] == pytest.approx(np.std([1.0, 3.0], ddof=1)) or std[0] == pytest.approx(
        np.std([1.0, 3.0])
    )
    assert std[2] == -2.0


def test_the_closed_gate_history_is_bit_exact_and_the_power_check_relabel_moves_both_columns() -> (
    None
):
    z = torch.randn(3, 14, 48).clamp(-5.0, 5.0)
    shift = torch.tensor([3.3, -0.9, 5.2])
    z_exp, variance = arms.signed_expected_history(z=z, rho=torch.zeros_like(z), shift=shift)
    assert torch.equal(z_exp, z)
    assert torch.equal(variance, torch.zeros_like(z))
    start = datetime.datetime(2026, 5, 1, 9)
    frame = pl.DataFrame(
        {
            "time_series_id": [5, 5, 5], "power_fcst_init_time": [start] * 3,
            "valid_time": [start + datetime.timedelta(hours=h) for h in (26, 28, 242)], "lead_hours": [26.0, 28.0, 242.0],
        }
    )  # fmt: skip
    out = drv.to_contract_rows(frame=frame, series_ids=[5])
    assert out["lead_hours"].to_list() == [
        25.0,
        239.0,
    ]  # 26 h native is 23 h from 12:00, which is below 24 h
    assert out["power_fcst_init_time"].to_list() == [start + datetime.timedelta(hours=3)] * 2


# --------------------------------------------------------------------------- run.sh by behaviour


def _run_sh(*, tmp_path: Path, pids: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "nvidia-smi"
    fake.write_text(
        f'#!/bin/bash\ncase "$*" in *memory.used*) echo 0 ;; *compute-apps*) printf "{pids}" ;; esac\n'
    )
    fake.chmod(0o755)
    env = {
        **{k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLOT_CPU": "0", "PYTHON": __import__("sys").executable,
        "DATA_PATH_INTERNAL": str(tmp_path), "HOME": str(tmp_path),
    }  # fmt: skip
    return subprocess.run(
        ["bash", str(Path(__file__).parent.parent / "run.sh"), "--go", "no_such_script.py", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _needs_launch_gates() -> None:
    avail = int(
        subprocess.run(
            "free -g | awk '/^Mem:/ {print $7}'",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        or 0
    )
    if avail < 25 or int(float(Path("/proc/loadavg").read_text().split()[0])) > 24:
        pytest.skip("the launch gates of run.sh refuse on this machine")


def test_run_sh_refuses_at_two_gpu_processes_and_not_at_one(tmp_path: Path) -> None:
    _needs_launch_gates()
    two = _run_sh(tmp_path=tmp_path / "a", pids="1\\n2\\n", args=["fit"])
    assert two.returncode == 1
    assert "GPU processes" in two.stderr
    one = _run_sh(tmp_path=tmp_path / "b", pids="1\\n", args=["fit"])
    assert "GPU processes" not in one.stderr


def test_run_sh_skips_the_gpu_gate_for_stage0_quicklook_and_score(tmp_path: Path) -> None:
    _needs_launch_gates()
    for n, command in enumerate(("stage0", "quicklook", "score")):
        out = _run_sh(tmp_path=tmp_path / str(n), pids="1\\n2\\n", args=[command])
        assert "GPU processes" not in out.stderr


# --------------------------------------------------------------------------- where the PV scales live


def test_the_pv_scales_are_state_dict_buffers_under_the_names_the_check_reads() -> None:
    import code_works_check as cwc
    import e00721_e2e as e2e
    import e00721_jobs as jobs

    job = fit.Job(arm="SG-A", world="W2", fold="2025-12", seed=0)
    inputs = jobs.WorldInputs(make_world=e2e.make_world_builder())
    state = inputs.network_state(job=job)
    keys = cwc.pv_scale_keys(state=state)
    assert sorted(k.rsplit(".", 1)[-1] for k in keys) == sorted(cwc.PV_SCALE_KEYS)
    assert all(
        float(state[k]) == 1.0 for k in keys
    )  # the uncalibrated start; `calibrate_scale` sets them
    assert cwc.pv_scale_keys(state={"trunk.weight": 0, "scale_like": 0}) == []


# --------------------------------------------------------------------------- wave1.sh


def test_wave1_script_names_the_jobs_of_wave_1_in_order_without_retries() -> None:
    script = Path(__file__).parent.parent / "wave1.sh"
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    text = script.read_text()
    assert not any(word in text for word in ("until ", "while ", "retry", "sleep"))
    listed = subprocess.run(
        [str(script), "--list"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    steps = [line.split(" ", 1)[1].split() for line in listed]
    assert [s[0] for s in steps][:2] == ["stage0", "probe"]
    assert steps[-1] == ["quicklook"]
    fits = [dict(zip(s[1::2], s[2::2], strict=False)) for s in steps if s[0] == "fit"]
    predicts = [dict(zip(s[1::2], s[2::2], strict=False)) for s in steps if s[0] == "predict"]
    keys = [
        fit.Job(arm=f["--arm"], world=f["--world"], fold=f["--fold"], seed=int(f["--seed"])).key
        for f in fits
    ]
    assert keys == [j.key for j in fit.wave_jobs(wave=1)]
    assert fits == predicts

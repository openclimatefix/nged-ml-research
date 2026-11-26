"""End-to-end tests of the fit stage, the timing gate, and the chain runner with tiny CPU fits on an
invented bank (need torch).
"""

import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import e00729_arms as arms729  # noqa: E402
import e00732_core as core  # noqa: E402
import e00732_fit as fit  # noqa: E402
import e00732_leakage as leak  # noqa: E402
import e00732_nets as nets  # noqa: E402
import e00732_stages as stages  # noqa: E402
import fake730  # noqa: E402
import numpy as np  # noqa: E402
from world732 import seed_neighbour_files  # noqa: E402


@pytest.fixture
def run(tmp_path: Any, monkeypatch: Any):
    root730, root = tmp_path / "730", tmp_path / "732"
    root.mkdir()
    world = fake730.build(root=root730)
    fake730.patch_constants(monkeypatch, root=root730, world=world)
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 32)
    monkeypatch.setattr(core, "BATCH_SIZE", 16)
    monkeypatch.setattr(core, "TIMING_EPOCHS", 1)
    holidays = world["cal"].holidays
    base = world["base"]
    stages.stage_banks(root=root, root730=root730, holidays=holidays, log=print)
    seed_neighbour_files(root=root, base=base, arrays=world["neighbours"])
    stages.stage_limits(
        root=root,
        root730=root730,
        holidays=holidays,
        log=print,
        base=base,
        arrays=world["neighbours"],
    )
    stages.stage_overlays(
        root=root,
        root730=root730,
        holidays=holidays,
        log=print,
        train_seeds=(0, 1),
        neighbours=world["neighbours"],
    )
    gallery = root / "PRIVATE" / "gallery"
    gallery.mkdir(parents=True)
    (gallery / "panel_check.json").write_text(
        json.dumps(
            {
                "banks_digest": stages.banks730.banks_digest_id(root=root730),
                "panels": dict.fromkeys(core.PANEL_REQUIREMENTS, True),
            }
        )
    )
    stats = arms729.input_stats(cal=world["cal"], train_days=np.arange(len(world["cal"].dates)))
    medians = stages.read_json(path=root / stages.MEDIANS_FILE)
    env = stages.Env(
        holidays=holidays, stats=stats, neighbours=world["neighbours"], medians=medians
    )
    stages.write_json(path=root / stages.TENSOR_FILE, payload={"pass": True})
    return root, root730, env


def test_timing_gate_then_one_fit_stores_record_network_and_predictions(
    run: Any, monkeypatch: Any
) -> None:
    root, root730, env = run
    monkeypatch.setattr(core, "STOP_LINE_HOURS", 1e9)
    with pytest.raises(RuntimeError, match="timing"):
        fit.fit_one(
            root=root,
            root730=root730,
            env=env,
            arm="U*",
            seed=0,
            device="cpu",
            log=print,
            max_epochs=1,
        )
    report = fit.stage_timing(
        root=root, root730=root730, env=env, device="cpu", manager_approved=False, log=print
    )
    assert report["pass"]
    assert set(report["seconds_per_epoch"]) == set(core.TIMING_ARMS)
    rec = fit.fit_one(
        root=root, root730=root730, env=env, arm="U*", seed=0, device="cpu", log=print, max_epochs=1
    )
    assert rec["epochs"] == 1
    assert len(rec["train_terms"]["sign"]) == 1
    assert (root / "networks" / "Ustar_unet_s0.pt").exists()
    folder = root / "predictions" / "Ustar_unet_s0"
    names = {p.stem for p in folder.glob("*.npz")}
    assert {"validation", "test", "shifted", "test_transfer", "test_raised"} <= names
    assert {
        f"test_inj_{a:g}{s}" for a in core.TEST_INJECTION_POINTS for s in ("", "_nobsp")
    } <= names
    with np.load(folder / "test.npz") as z:
        assert set(z.files) == {"p", "pv", "step_p", "size", "row_seg", "row_date", "row_hh"}
        assert len(z["p"]) == fake730_rows(root730)
    assert (
        fit.fit_one(root=root, root730=root730, env=env, arm="U*", seed=0, device="cpu", log=print)
        == rec
    )  # skipped, not refitted
    fit.fit_one(
        root=root, root730=root730, env=env, arm="S+", seed=0, device="cpu", log=print, max_epochs=1
    )
    with np.load(root / "predictions" / "Splus_unet_s0" / "test.npz") as z:
        assert set(z.files) == {"p", "row_seg", "row_date", "row_hh"}
    assert not (root / "predictions" / "Splus_unet_s0" / "test_raised.npz").exists()


def fake730_rows(root730: Any) -> int:
    return len(stages.load_stored(root730=root730, arm="S", arch="unet")[0]["p"])


def test_timing_gate_stops_over_the_line_unless_the_manager_approves(
    run: Any, monkeypatch: Any
) -> None:
    root, root730, env = run
    monkeypatch.setattr(core, "STOP_LINE_HOURS", 0.0)
    with pytest.raises(stages.StoppedError, match="timing gate"):
        fit.stage_timing(
            root=root, root730=root730, env=env, device="cpu", manager_approved=False, log=print
        )
    assert json.loads((root / "scores" / "timing.json").read_text())["pass"] is False
    assert fit.stage_timing(
        root=root, root730=root730, env=env, device="cpu", manager_approved=True, log=print
    )["pass"]


def test_chain_stops_and_reports_on_a_correctness_failure_and_a_failed_needed_arm(
    run: Any, monkeypatch: Any
) -> None:
    root, root730, _env = run
    (root / "scores").mkdir(exist_ok=True)
    stages.write_json(
        path=root / "scores" / "timing.json", payload={"pass": True, "hours_at_cap": 3.0}
    )
    lines = []

    def boom(arm: Any, arch: Any, seed: Any) -> Any:
        raise leak.CorrectnessFailureError("a planted leak")

    with pytest.raises(leak.CorrectnessFailureError):
        fit.stage_chain(root=root, root730=root730, run_one=boom, log=lines.append)
    status = json.loads((root / "chain_status.json").read_text())
    assert status["state"] == "invalid"
    assert status["family"] == "S+, U*"
    assert (root / "scores" / "invalid.json").exists()
    assert any(line.startswith("fits starting") for line in lines)
    with pytest.raises(RuntimeError, match="correctness"):
        stages.guard_run(root=root, root730=root730, stage="fit")


def test_chain_marks_a_failed_needed_arm_and_goes_on_past_a_failed_report_only_arm(
    run: Any,
) -> None:
    root, root730, _env = run
    stages.write_json(
        path=root / "scores" / "timing.json", payload={"pass": True, "hours_at_cap": 3.0}
    ) if (root / "scores").exists() else None
    (root / "scores").mkdir(exist_ok=True)
    stages.write_json(
        path=root / "scores" / "timing.json", payload={"pass": True, "hours_at_cap": 3.0}
    )
    calls = []

    def fake(arm: Any, arch: Any, seed: Any) -> Any:
        calls.append(arm)
        if arm == "U-SG":
            raise RuntimeError("a report-only fit raised")
        stages.write_json(
            path=stages.job_path(root=root, arm=arm, arch=arch, seed=seed), payload={"arm": arm}
        )
        return {}

    lines: list[str] = []
    status = fit.stage_chain(root=root, root730=root730, run_one=fake, log=lines.append)
    assert sum(line.startswith("fits starting") for line in lines) == 7
    assert status["state"] == "failed"
    assert status["arms"] == ["U-SG"]
    assert status["needed"] is False
    assert "U*-L" in calls
    assert len([c for c in calls if c == "U-SG"]) == 3
    assert (
        stages.job_path(root=root, arm="U-SG", arch="unet", seed=0).with_suffix(".failed.json")
    ).exists()
    assert sum(line.startswith("last fit ended") for line in lines) == 7
    for p in root.glob("fits/*.json"):
        p.unlink()

    def needed_fails(arm: Any, arch: Any, seed: Any) -> Any:
        if arm == "S+":
            raise RuntimeError("a needed fit raised")
        stages.write_json(
            path=stages.job_path(root=root, arm=arm, arch=arch, seed=seed), payload={"arm": arm}
        )
        return {}

    status = fit.stage_chain(root=root, root730=root730, run_one=needed_fails, log=lines.append)
    assert status["state"] == "failed"
    assert status["arms"] == ["S+"]
    assert json.loads((root / "chain_status.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("error", [stages.StoppedError, stages.GuardError])
def test_a_stop_or_a_closed_gate_is_never_hidden_as_a_failed_fit(
    run: Any, error: type[Exception]
) -> None:
    root, root730, _env = run
    (root / "scores").mkdir(exist_ok=True)
    stages.write_json(
        path=root / "scores" / "timing.json", payload={"pass": True, "hours_at_cap": 3.0}
    )

    def stop(arm: Any, arch: Any, seed: Any) -> Any:
        raise error("a gate is open")

    with pytest.raises(error):
        fit.run_jobs(root=root, jobs=[("U*", "unet", 0)], run_one=stop, log=print)
    assert not list(root.glob("fits/*.failed.json"))
    with pytest.raises(error):
        fit.stage_chain(root=root, root730=root730, run_one=stop, log=print)
    status = json.loads((root / "chain_status.json").read_text())
    assert status["state"] == "stopped"
    assert not list(root.glob("fits/*.failed.json"))


def test_a_failed_fit_is_skipped_unless_rerun_is_passed(run: Any) -> None:
    root, _root730, _env = run
    calls: list[str] = []

    def flaky(arm: Any, arch: Any, seed: Any) -> Any:
        calls.append(arm)
        if len(calls) == 1:
            raise ValueError("a fit raised")
        stages.write_json(
            path=stages.job_path(root=root, arm=arm, arch=arch, seed=seed), payload={"arm": arm}
        )
        return {}

    jobs = [("U*", "unet", 0)]
    assert fit.run_jobs(root=root, jobs=jobs, run_one=flaky, log=print)["failed"] == 1
    again = fit.run_jobs(root=root, jobs=jobs, run_one=flaky, log=print)
    assert again["skipped_failed"] == 1
    assert len(calls) == 1
    done = fit.run_jobs(root=root, jobs=jobs, run_one=flaky, log=print, rerun=True)
    assert done["run"] == 1
    assert len(calls) == 2
    assert (
        not stages.job_path(root=root, arm="U*", arch="unet", seed=0)
        .with_suffix(".failed.json")
        .exists()
    )


def test_net_checks_fire_on_a_head_that_leaks_into_the_sign_and_on_a_missing_stop_gradient() -> (
    None
):
    from e00732_arms import N_CHANNELS

    x = torch.randn(2, N_CHANNELS, 96)
    net = nets.Net(arch="unet", c_in=N_CHANNELS, heads=True, stop_gradient=True)
    fit.assert_net_checks(net=net, spec=core.ARMS["U-SG"], x=x)
    leaky = nets.Net(arch="unet", c_in=N_CHANNELS, heads=True, stop_gradient=False)
    with pytest.raises(leak.CorrectnessFailureError, match="backbone"):
        fit.assert_net_checks(
            net=leaky, spec=core.ARMS["U-SG"], x=x
        )  # the spec says stop-gradient, the net has none


def test_assert_fit_inputs_runs_the_channel_assertions_on_a_real_batch(
    run: Any, monkeypatch: Any
) -> None:
    import dataclasses

    import e00732_arms as arms

    _root, root730, env = run
    sets = fit.load_sets(root730=root730, env=env, seed=0)
    fit.assert_fit_inputs(spec=core.ARMS["U*"], sets=sets, seed=0)
    real = arms.make_batch

    def leaking(**kw: Any) -> Any:
        batch = real(**kw)
        assert batch.xi is not None
        xi = batch.xi.copy()
        xi[:, arms.ROW["csi"], 3] += 0.1
        return dataclasses.replace(batch, xi=xi)

    monkeypatch.setattr(arms, "make_batch", leaking)
    with pytest.raises(leak.CorrectnessFailureError, match="injected member"):
        fit.assert_fit_inputs(spec=core.ARMS["U*"], sets=sets, seed=0)

    def signed(**kw: Any) -> Any:
        batch = real(**kw)
        x = batch.x.copy()
        x[:, arms.NB_ROW["D14_value"], 5] = -1.0
        return dataclasses.replace(batch, x=x)

    monkeypatch.setattr(arms, "make_batch", signed)
    with pytest.raises(leak.CorrectnessFailureError, match="neighbour value"):
        fit.assert_fit_inputs(spec=core.ARMS["U*"], sets=sets, seed=0)


def test_load_sets_fails_on_a_pair_index_of_another_bank(run: Any, monkeypatch: Any) -> None:
    import e00732_overlays as ov

    _root, root730, env = run
    real = ov.build_pair_index
    monkeypatch.setattr(
        ov, "build_pair_index", lambda *, segments, **kw: real(segments=segments[:-1], **kw)
    )
    with pytest.raises(leak.CorrectnessFailureError, match="another bank"):
        fit.load_sets(root730=root730, env=env, seed=0)


def test_a_real_fit_stores_the_row_keys_of_every_test_set_and_a_wrong_key_stops_the_fit(
    run: Any, monkeypatch: Any
) -> None:
    import e00729_banks as banks729
    import e00729_score as score729
    import e00730_banks as banks730
    import e00732_report as report

    root, root730, env = run
    monkeypatch.setattr(core, "STOP_LINE_HOURS", 1e9)
    fit.stage_timing(
        root=root, root730=root730, env=env, device="cpu", manager_approved=True, log=print
    )
    fit.fit_one(
        root=root, root730=root730, env=env, arm="U*", seed=0, device="cpu", log=print, max_epochs=1
    )
    folder = root / "predictions" / "Ustar_unet_s0"
    te = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    keys = report.row_keys_of(
        rows=score729.rows_from_segments(segments=te.segments, holidays=env.holidays)
    )
    sets = sorted(p.stem for p in folder.glob("test*.npz"))
    assert "test_transfer" in sets
    assert {f"test_inj_{a:g}" for a in core.TEST_INJECTION_POINTS} <= set(sets)
    for name in sets:
        report.check_stored_keys(
            arrays=[report.read_set(folder=folder, name=name)], keys=keys, what=name
        )
    # a fit that stores a shifted key is stopped as a correctness failure
    real = fit.store_set

    def shifted(**kw: Any) -> None:
        real(**kw)
        path = kw["folder"] / f"{kw['name']}.npz"
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        if kw["name"] == "test_transfer":
            arrays["row_hh"] = (arrays["row_hh"] + 1) % 48
        np.savez(path, **arrays)

    monkeypatch.setattr(fit, "store_set", shifted)
    with pytest.raises(leak.CorrectnessFailureError, match="stored row keys differ"):
        fit.fit_one(
            root=root, root730=root730, env=env, arm="S+", seed=0, device="cpu", log=print,
            max_epochs=1,
        )  # fmt: skip


def test_a_failed_fit_stores_the_exception_type_message_and_traceback(run: Any) -> None:
    root, _root730, _env = run

    def broken(arm: Any, arch: Any, seed: Any) -> Any:
        raise AttributeError("module has no attribute 'limit'")

    fit.run_jobs(root=root, jobs=[("U*", "unet", 0)], run_one=broken, log=print)
    failed = json.loads(
        stages.job_path(root=root, arm="U*", arch="unet", seed=0)
        .with_suffix(".failed.json")
        .read_text()
    )
    assert failed["error"] == "AttributeError"
    assert "no attribute 'limit'" in failed["message"]
    assert "Traceback" in failed["traceback"]
    assert "broken" in failed["traceback"]


def test_the_chain_exit_status_is_non_zero_unless_every_fit_succeeded() -> None:
    assert fit.chain_exit_code(status={"state": "complete", "done": []}) == 0
    codes = {
        state: fit.chain_exit_code(status={"state": state})
        for state in ("failed", "stopped", "invalid", "running")
    }
    assert set(codes.values()) == {fit.CHAIN_FAILED_EXIT}
    assert fit.CHAIN_FAILED_EXIT not in (0, 1, 2, 3, 4, 64)  # 1 is a refused slot in `run.sh`


def test_the_cuda_branch_caps_the_gpu_memory_through_a_function_that_exists(
    run: Any, monkeypatch: Any
) -> None:
    """The invented-bank fits ran on the CPU only, so the `device == "cuda"` line never ran."""
    root, root730, env = run
    calls: list[str] = []
    monkeypatch.setattr(fit.train729, "limit_gpu_memory", lambda: calls.append("capped"))

    class StopHere(Exception):
        pass

    def stop(**_kwargs: Any) -> Any:
        raise StopHere

    monkeypatch.setattr(fit, "load_sets", stop)
    monkeypatch.setattr(fit.stages, "guard_run", lambda **_kwargs: None)
    with pytest.raises(StopHere):
        fit.fit_one(root=root, root730=root730, env=env, arm="U*", seed=0, device="cuda", log=print)
    assert calls == ["capped"]
    assert callable(fit.train729.limit_gpu_memory)


def test_every_attribute_that_the_code_takes_from_an_imported_module_exists() -> None:
    """A call such as `train730.limit_gpu_memory()` on the wrong module only fails when it runs."""
    import ast
    import importlib
    from pathlib import Path

    missing: list[str] = []
    for path in sorted(Path(fit.__file__).parent.glob("e00732_*.py")):
        tree = ast.parse(path.read_text())
        aliases = {
            (a.asname or a.name): a.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for a in node.names
            if a.name.startswith(("e00729_", "e00730_", "e00732_", "e00722_"))
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                module = importlib.import_module(aliases[node.value.id])
                if not hasattr(module, node.attr):
                    missing.append(f"{path.name}:{node.lineno} {node.value.id}.{node.attr}")
    assert not missing


def test_the_timing_gate_caps_the_gpu_memory_on_its_cuda_path(run: Any, monkeypatch: Any) -> None:
    root, root730, env = run
    calls: list[str] = []
    monkeypatch.setattr(fit.train729, "limit_gpu_memory", lambda: calls.append("capped"))

    class StopHere(Exception):
        pass

    def stop(**_kwargs: Any) -> Any:
        raise StopHere

    monkeypatch.setattr(fit, "load_sets", stop)
    monkeypatch.setattr(fit.stages, "guard_run", lambda **_kwargs: None)
    with pytest.raises(StopHere):
        fit.stage_timing(
            root=root, root730=root730, env=env, device="cuda", manager_approved=False, log=print
        )
    assert calls == ["capped"]


def test_a_stale_failed_file_is_ignored_once_the_fit_record_exists(run: Any) -> None:
    root, root730, _env = run
    (root / "scores").mkdir(exist_ok=True)
    stages.write_json(
        path=root / "scores" / "timing.json", payload={"pass": True, "hours_at_cap": 3.0}
    )
    first = stages.job_path(root=root, arm="S+", arch="unet", seed=0)
    first.parent.mkdir(parents=True, exist_ok=True)
    stages.write_json(path=first.with_suffix(".failed.json"), payload={"error": "Old"})
    assert fit.arm_failed(root=root, arm="S+")
    stages.write_json(path=first, payload={"arm": "S+"})
    assert not fit.arm_failed(root=root, arm="S+")  # the record exists beside the stale file

    def fake(arm: Any, arch: Any, seed: Any) -> Any:
        stages.write_json(
            path=stages.job_path(root=root, arm=arm, arch=arch, seed=seed), payload={"arm": arm}
        )
        return {}

    status = fit.stage_chain(root=root, root730=root730, run_one=fake, log=lambda _line: None)
    assert status["state"] == "complete"


def test_every_numpy_call_in_the_experiment_follows_a_cpu_call() -> None:
    """A CUDA tensor has no `.numpy()`: the receiver must have been moved with `.cpu()` first."""
    import ast
    from pathlib import Path

    bad: list[str] = []
    for path in sorted(Path(fit.__file__).parent.glob("e00732_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "numpy"
                and not node.args
            ):
                continue
            chain, recv = [], node.func.value
            while isinstance(recv, ast.Call | ast.Attribute | ast.Subscript):
                chain.append(
                    recv.func.attr
                    if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute)
                    else getattr(recv, "attr", "")
                )
                recv = (
                    recv.func.value
                    if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute)
                    else getattr(recv, "value", None)
                )
                if recv is None:
                    break
            if "cpu" not in chain:
                bad.append(f"{path.name}:{node.lineno}")
    assert not bad


def test_assert_net_checks_moves_its_outputs_to_the_cpu_before_numpy(monkeypatch: Any) -> None:
    """A tensor that refuses `.numpy()` unless `.cpu()` ran stands in for a CUDA tensor."""
    import torch

    class Gpu(torch.Tensor):
        on_cpu = False

        def cpu(self, *a: Any, **k: Any) -> Any:
            out = super().cpu(*a, **k).as_subclass(Gpu)
            out.on_cpu = True
            return out

        def numpy(self, *a: Any, **k: Any) -> Any:
            if not self.on_cpu:
                raise TypeError("can't convert cuda:0 device type tensor to numpy")
            return super().numpy(*a, **k)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(3, 3)
            self.extra = torch.nn.Linear(3, 3)

        def forward(self, x: Any) -> Any:
            return self.backbone(x).as_subclass(Gpu)

    fit.assert_net_checks(net=Net(), spec=core.ARMS["U*"], x=torch.zeros(2, 3, 3))

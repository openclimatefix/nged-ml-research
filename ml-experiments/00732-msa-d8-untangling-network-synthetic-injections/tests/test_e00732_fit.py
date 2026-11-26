"""End-to-end tests of the fit stage, the timing gate, and the chain runner with tiny CPU fits on an invented bank (need torch)."""

import json

import pytest

torch = pytest.importorskip("torch")

import e00732_core as core  # noqa: E402
import e00732_fit as fit  # noqa: E402
import e00732_leakage as leak  # noqa: E402
import e00732_nets as nets  # noqa: E402
import e00732_stages as stages  # noqa: E402
import fake730  # noqa: E402
import numpy as np  # noqa: E402
import e00729_arms as arms729  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    root730, root = tmp_path / "730", tmp_path / "732"
    root.mkdir()
    world = fake730.build(root=root730)
    fake730.patch_constants(monkeypatch, root=root730, world=world)
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 32)
    monkeypatch.setattr(core, "BATCH_SIZE", 16)
    monkeypatch.setattr(core, "TIMING_EPOCHS", 1)
    holidays = world["cal"].holidays
    stages.stage_banks(root=root, root730=root730, holidays=holidays, log=print)
    stages.stage_limits(root=root, root730=root730, holidays=holidays, log=print)
    stages.stage_overlays(
        root=root, root730=root730, holidays=holidays, log=print, train_seeds=(0, 1)
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
    env = stages.Env(holidays=holidays, stats=stats)
    return root, root730, env


def test_timing_gate_then_one_fit_stores_record_network_and_predictions(run, monkeypatch) -> None:
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
    assert report["pass"] and set(report["seconds_per_epoch"]) == set(core.TIMING_ARMS)
    rec = fit.fit_one(
        root=root, root730=root730, env=env, arm="U*", seed=0, device="cpu", log=print, max_epochs=1
    )
    assert (
        rec["epochs"] == 1
        and len(rec["train_terms"]["sign"]) == 1
        and (root / "networks" / "Ustar_unet_s0.pt").exists()
    )
    folder = root / "predictions" / "Ustar_unet_s0"
    names = {p.stem for p in folder.glob("*.npz")}
    assert {"validation", "test", "shifted", "test_transfer", "test_raised"} <= names
    assert {
        f"test_inj_{a:g}{s}" for a in core.TEST_INJECTION_POINTS for s in ("", "_nobsp")
    } <= names
    with np.load(folder / "test.npz") as z:
        assert set(z.files) == {"p", "pv", "step_p", "size"} and len(z["p"]) == fake730_rows(
            root730
        )
    assert (
        fit.fit_one(root=root, root730=root730, env=env, arm="U*", seed=0, device="cpu", log=print)
        == rec
    )  # skipped, not refitted
    fit.fit_one(
        root=root, root730=root730, env=env, arm="S+", seed=0, device="cpu", log=print, max_epochs=1
    )
    with np.load(root / "predictions" / "Splus_unet_s0" / "test.npz") as z:
        assert set(z.files) == {"p"}
    assert not (root / "predictions" / "Splus_unet_s0" / "test_raised.npz").exists()


def fake730_rows(root730) -> int:
    return len(stages.load_stored(root730=root730, name="S", arm="S", arch="unet")[0]["p"])


def test_timing_gate_stops_over_the_line_unless_the_manager_approves(run, monkeypatch) -> None:
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
    run, monkeypatch
) -> None:
    root, root730, env = run
    (root / "scores").mkdir(exist_ok=True)
    stages.write_json(
        path=root / "scores" / "timing.json", payload={"pass": True, "hours_at_cap": 3.0}
    )
    lines = []

    def boom(arm, arch, seed):
        raise leak.CorrectnessFailureError("a planted leak")

    with pytest.raises(leak.CorrectnessFailureError):
        fit.stage_chain(root=root, root730=root730, run_one=boom, log=lines.append)
    status = json.loads((root / "chain_status.json").read_text())
    assert (
        status["state"] == "invalid"
        and status["family"] == "U*, U-NC"
        and (root / "scores" / "invalid.json").exists()
    )
    assert any(line.startswith("fits starting") for line in lines)
    with pytest.raises(RuntimeError, match="correctness"):
        stages.guard_run(root=root, root730=root730, stage="fit")


def test_chain_marks_a_failed_needed_arm_and_goes_on_past_a_failed_report_only_arm(run) -> None:
    root, root730, env = run
    stages.write_json(
        path=root / "scores" / "timing.json", payload={"pass": True, "hours_at_cap": 3.0}
    ) if (root / "scores").exists() else None
    (root / "scores").mkdir(exist_ok=True)
    stages.write_json(
        path=root / "scores" / "timing.json", payload={"pass": True, "hours_at_cap": 3.0}
    )
    calls = []

    def fake(arm, arch, seed):
        calls.append(arm)
        if arm == "U-SG":
            raise RuntimeError("a report-only fit raised")
        stages.write_json(
            path=stages.job_path(root=root, arm=arm, arch=arch, seed=seed), payload={"arm": arm}
        )
        return {}

    lines: list[str] = []
    status = fit.stage_chain(root=root, root730=root730, run_one=fake, log=lines.append)
    assert (
        status["state"] == "complete"
        and "U*-L" in calls
        and len([c for c in calls if c == "U-SG"]) == 3
    )
    assert (
        stages.job_path(root=root, arm="U-SG", arch="unet", seed=0).with_suffix(".failed.json")
    ).exists()
    assert sum(line.startswith("last fit ended") for line in lines) == 4
    for p in root.glob("fits/*.json"):
        p.unlink()

    def needed_fails(arm, arch, seed):
        if arm == "S+":
            raise RuntimeError("a needed fit raised")
        stages.write_json(
            path=stages.job_path(root=root, arm=arm, arch=arch, seed=seed), payload={"arm": arm}
        )
        return {}

    status = fit.stage_chain(root=root, root730=root730, run_one=needed_fails, log=lines.append)
    assert status["state"] == "failed" and status["arms"] == ["S+"]
    assert json.loads((root / "chain_status.json").read_text())["state"] == "failed"


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

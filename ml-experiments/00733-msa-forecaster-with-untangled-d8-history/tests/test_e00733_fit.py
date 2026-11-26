# ruff: noqa: E501, PT018
"""Tests of `e00733_fit`: the device branches (stubbed), both fits, the skip rule, and the failure records."""

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import e00733_channels as channels
import e00733_core as core
import e00733_model as model
import e00733_synth as sy
import numpy as np
import pytest
from synth_fixture import make_segment

torch = pytest.importorskip("torch")
import e00730_leakage as leak730  # noqa: E402  # ty: ignore[unresolved-import]
import e00733_fit as fit  # noqa: E402
import e00733_train as tn  # noqa: E402
from torch_inputs import FakeSource, make_channel_tensors  # noqa: E402


def _synthetic_inputs() -> fit.SyntheticInputs:
    segs = [make_segment(n_days=40, seed=0)]
    idx = sy.enumerate_samples(segments=segs)
    sel = np.arange(len(idx))[:: max(1, len(idx) // 32)]
    sub = sy.SampleIndex(idx.seg[sel], idx.origin_day[sel], idx.lead_index[sel])
    return fit.SyntheticInputs(
        train_segments=segs, train_index=sub, val_segments=segs, val_index=sub,
        d8_index=3, target_std=1.0, clim_logit=-1.0,
    )  # fmt: skip


@pytest.fixture
def short_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap the synthetic loop at 2 passes so the test is quick."""
    monkeypatch.setattr(
        tn, "train_synthetic", partial(tn.train_synthetic, max_epochs=2, patience=3, batch_size=16)
    )


# --------------------------------------------------------------------------- the device


def test_prepare_device_caps_the_gpu_share_through_the_function_of_experiment_729(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On "cuda" the cap is 729's `limit_gpu_memory` at 6 GB; "cpu" calls nothing."""
    calls: list[float] = []
    monkeypatch.setattr(
        fit.train729, "limit_gpu_memory", lambda *, gigabytes: calls.append(gigabytes)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fit.prepare_device(device="cuda")
    assert calls == [core.GPU_GB]
    fit.prepare_device(device="cpu")
    assert calls == [core.GPU_GB]


def test_prepare_device_refuses_cuda_without_a_gpu_and_unknown_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit that asks for the GPU where none exists is a guard, not a failed fit."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(fit.GuardError):
        fit.prepare_device(device="cuda")
    with pytest.raises(fit.GuardError):
        fit.prepare_device(device="mps")


def test_peak_gpu_gb_reads_the_cuda_counter_only_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cuda branch reads `max_memory_allocated`; the cpu branch is zero."""
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3 * 10**9)
    assert fit.peak_gpu_gb(device="cuda") == pytest.approx(3.0)
    assert fit.peak_gpu_gb(device="cpu") == 0.0


# --------------------------------------------------------------------------- the synthetic fit


def test_synthetic_fit_stores_record_network_and_loss_curve(
    tmp_path: Path, short_loop: None
) -> None:
    """A fit writes its record (with the code version and the stored pass) and its network.

    And the curve data reads back.
    """
    rec = fit.fit_synthetic(
        root=tmp_path, arm="YU", seed=0, device="cpu", loader=lambda *_: _synthetic_inputs()
    )
    assert rec["code_version"] == core.CODE_VERSION and rec["mode"] == "untangled"
    assert (tmp_path / "networks" / "YU_s0.pt").exists()
    on_disk = json.loads(fit.record_path(root=tmp_path, arm="YU", fold=None, seed=0).read_text())
    curve = fit.loss_curve_data(record=on_disk)
    s = curve["series"][0]
    assert len(s["train"]) == len(s["val"]) >= 1 and 1 <= s["stored_pass"] <= len(s["val"])
    net = model.UntangledNet(seed=0, sign_head=True)
    net.load_state_dict(torch.load(tmp_path / "networks" / "YU_s0.pt"))


def test_a_finished_fit_is_skipped(tmp_path: Path, short_loop: None) -> None:
    """A second call reads the record and does not call the loader."""
    fit.fit_synthetic(
        root=tmp_path, arm="Y0", seed=1, device="cpu", loader=lambda *_: _synthetic_inputs()
    )

    def boom(*_: object) -> fit.SyntheticInputs:
        msg = "the loader must not run"
        raise AssertionError(msg)

    again = fit.fit_synthetic(root=tmp_path, arm="Y0", seed=1, device="cpu", loader=boom)
    assert again["arm"] == "Y0" and again["mode"] == "absent"


def test_synthetic_modes_follow_the_arm_specification() -> None:
    """Y0 reads no channel, YU the untangled ones, YT the true ones; a real arm is refused."""
    assert [fit.synthetic_mode(arm=a) for a in ("Y0", "YU", "YT")] == [
        "absent",
        "untangled",
        "true",
    ]
    with pytest.raises(fit.GuardError):
        fit.synthetic_mode(arm="FU")


def test_synthetic_fit_on_cuda_takes_the_capped_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the device stubbed.

    The cuda branch caps the GPU and moves the network to the device; the CPU runs it.
    """
    calls: list[float] = []
    monkeypatch.setattr(
        fit.train729, "limit_gpu_memory", lambda *, gigabytes: calls.append(gigabytes)
    )
    asked: list[bool] = []

    def _available_once() -> bool:
        # Only the fit's own device check sees a GPU; torch's optimiser asks again and must not.
        asked.append(True)
        return len(asked) == 1

    monkeypatch.setattr(torch.cuda, "is_available", _available_once)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 10**9)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *_a, **_k: self)
    monkeypatch.setattr(tn, "train_synthetic", _cpu_train)
    rec = fit.fit_synthetic(
        root=tmp_path, arm="YT", seed=0, device="cuda", loader=lambda *_: _synthetic_inputs()
    )
    assert calls == [core.GPU_GB] and rec["peak_gpu_gb"] == pytest.approx(1.0)


_real_train_synthetic = tn.train_synthetic


def _cpu_train(**kw: object) -> tn.History:
    kw["device"] = "cpu"
    return _real_train_synthetic(**kw, max_epochs=2, patience=3, batch_size=16)  # ty: ignore[invalid-argument-type]


# --------------------------------------------------------------------------- the real-fold fit


class _World:
    def __init__(self, source: tn.UntSource) -> None:
        self.source = source


def _real_inputs(tmp_path: Path) -> fit.RealInputs:
    src = tn.UntSource(source=FakeSource(), tensors=make_channel_tensors(n_grid=48 * 400))
    state = tmp_path / "stage1.pt"
    torch.save(model.UntangledNet(seed=0, path=False).state_dict(), state)
    return fit.RealInputs(
        world=_World(src),
        stage1_state=state,
        dropped={"n": 0},
        fingerprint="abc",
        encoders="frozen",
        has_channels=True,
    )


def _samples(*, world: object, stage: int, placebo: bool) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    n = 40
    return SimpleNamespace(
        primary=np.full(n, 3), c_idx=(300 + rng.integers(0, 50, n)) * 48 + 24,
        is_val=np.arange(n) % 5 == 0, ages=None,
    )  # fmt: skip


def _patch_713(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tn.tr, "token_weights", lambda **_: torch.ones(1, 1))
    monkeypatch.setattr(fit.tr713, "target_end", lambda **_: 10**9)
    monkeypatch.setattr(tn.tr, "target_end", lambda **_: 10**9)
    monkeypatch.setattr(tn.tr, "BATCH", 8)
    monkeypatch.setattr(tn.tr, "VALIDATION_LEAD_DAYS", (0, 3))
    monkeypatch.setattr(
        tn, "block_dropout_masks", lambda **kw: (np.zeros(len(kw["origin_idx"]), bool),) * 2
    )
    monkeypatch.setattr(tn.tr, "synthetic_source", lambda *, world, samples: world.source)
    monkeypatch.setattr(fit, "CAP_FIRST", 2)
    monkeypatch.setattr(fit, "CAP_RERUN", 3)


def test_real_fold_fit_stores_both_stages_and_reruns_a_stage_that_hits_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage that ends at the cap is rerun from its starting state at the larger cap; both histories are stored."""
    _patch_713(monkeypatch)
    rec = fit.fit_real(
        root=tmp_path, arm="FU", fold="a", seed=0, device="cpu", loader=lambda *_: _real_inputs(tmp_path), make_samples=_samples,
    )  # fmt: skip
    assert [s["stage"] for s in rec["stages"]] == [2, 3] and rec["stages"][0]["scale"] is not None
    assert rec["stages"][1]["scale"] is None
    assert len(rec["stages"][0]["histories"]) == 2 and rec["stages"][0]["cap_hit"]
    assert (tmp_path / "networks" / "FU_a_s0_stage2.pt").exists() and (
        tmp_path / "networks" / "FU_a_s0_stage3.pt"
    ).exists()
    curve = fit.loss_curve_data(record=rec)
    assert [s["label"] for s in curve["series"]] == ["stage 2", "stage 3"]


def test_real_fold_fit_refuses_a_stored_or_synthetic_arm(tmp_path: Path) -> None:
    """A stored arm and a synthetic arm are not fitted by the real-fold path."""
    for arm in ("M*", "YU"):
        with pytest.raises(fit.GuardError):
            fit.fit_real(
                root=tmp_path,
                arm=arm,
                fold="a",
                seed=0,
                device="cpu",
                loader=lambda *_: None,  # ty: ignore[invalid-argument-type]
            )  # ty: ignore[invalid-argument-type]


def test_loss_curve_refuses_an_empty_record() -> None:
    """A record with no pass has no curve."""
    with pytest.raises(ValueError, match="no loss curve"):
        fit.loss_curve_data(
            record={
                "arm": "FU",
                "fold": "a",
                "seed": 0,
                "stages": [{"stage": 2, "histories": [{"train": [], "val": [], "stored_pass": 0}]}],
            }
        )


# --------------------------------------------------------------------------- the runner


def _jobs() -> list[tuple[str, str | None, int]]:
    return [("YU", None, 0), ("YU", None, 1), ("Y0", None, 0)]


def test_a_failed_fit_leaves_message_and_traceback_and_the_run_goes_on(tmp_path: Path) -> None:
    """The failure record holds the message and the traceback; the other fits still run.

    A rerun skips the failed fit.
    """
    ran: list[tuple] = []

    def run_one(arm: str, fold: str | None, seed: int) -> dict:
        if seed == 1:
            msg = "out of memory"
            raise RuntimeError(msg)
        ran.append((arm, seed))
        fit.write_json(
            path=fit.record_path(root=tmp_path, arm=arm, fold=fold, seed=seed), payload={"ok": True}
        )
        return {}

    logs: list[str] = []
    counts = fit.run_jobs(root=tmp_path, jobs=_jobs(), run_one=run_one, log=logs.append)
    assert counts == {"run": 2, "skipped": 0, "skipped_failed": 0, "failed": 1} and len(ran) == 2
    failed = json.loads((tmp_path / "fits" / "YU_s1.failed.json").read_text())
    assert (
        failed["message"] == "out of memory"
        and "Traceback" in failed["traceback"]
        and failed["error"] == "RuntimeError"
    )
    assert fit.arm_failed(root=tmp_path, arm="YU", folds=[None], seeds=[0, 1])
    assert not fit.arm_failed(root=tmp_path, arm="Y0", folds=[None], seeds=[0])
    again = fit.run_jobs(root=tmp_path, jobs=_jobs(), run_one=run_one, log=logs.append)
    assert again == {"run": 0, "skipped": 2, "skipped_failed": 1, "failed": 0}
    retry = fit.run_jobs(root=tmp_path, jobs=_jobs(), run_one=run_one, log=logs.append, rerun=True)
    assert retry["failed"] == 1 and retry["skipped"] == 2


@pytest.mark.parametrize(
    "exc",
    [
        leak730.CorrectnessFailureError("leak"),
        channels.StoppedError("stop"),
        fit.GuardError("gate"),
    ],
)
def test_a_correctness_failure_a_stop_and_a_guard_are_never_recorded_as_a_failed_fit(
    tmp_path: Path, exc: Exception
) -> None:
    """These three propagate, and no failure file is written."""

    def run_one(*_: object) -> dict:
        raise exc

    with pytest.raises(type(exc)):
        fit.run_jobs(root=tmp_path, jobs=_jobs(), run_one=run_one, log=lambda _: None)
    assert not list(tmp_path.glob("fits/*.failed.json"))


def test_a_failed_chain_exits_with_5_never_1() -> None:
    """The chain's exit status is 0 only on completion, and 5 otherwise."""
    assert fit.chain_exit_code(status={"state": "complete"}) == 0
    assert fit.chain_exit_code(status={"state": "failed"}) == 5 == core.CHAIN_FAILED_EXIT
    assert fit.chain_exit_code(status={}) != 1


def test_fits_must_start_after_the_limits_are_written(tmp_path: Path) -> None:
    """A missing `limits.json` is a guard; a fit record older than it is a correctness failure."""
    with pytest.raises(fit.GuardError):
        fit.assert_fits_after_limits(root=tmp_path, fit_records=[])
    old = tmp_path / "fits" / "a.json"
    fit.write_json(path=old, payload={})
    import os

    os.utime(old, (1, 1))
    fit.write_json(path=tmp_path / "limits.json", payload={})
    with pytest.raises(leak730.CorrectnessFailureError):
        fit.assert_fits_after_limits(root=tmp_path, fit_records=[old])


def test_progress_lines_and_names() -> None:
    """The announcement lines carry the family, the projection, and the core; file names avoid the star."""
    assert "projection 1.50 GPU-process-hours" in fit.fits_starting_line(
        family="synthetic", hours=1.5, cores="22-27", now="t"
    )
    assert "last fit ended" in fit.last_fit_ended_line(family="real", counts={"run": 1}, now="t")
    assert (
        fit.job_name(arm="M*", fold="a", seed=2) == "Mstar_a_s2"
        and fit.job_name(arm="YU", fold=None, seed=0) == "YU_s0"
    )
    assert fit.d8_index_from(labels=["D1", core.SCORED_PRIMARY, "D9"]) == 1
    with pytest.raises(fit.GuardError):
        fit.d8_index_from(labels=["D1"])
    assert fit.zero_check(values=np.zeros(3)) and not fit.zero_check(values=np.ones(3))

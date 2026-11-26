# ruff: noqa: E501, PT018, PT017, PLR0915, E731, ANN003
"""Experiment 716: the fit driver's file handling, on a stubbed training loop (WP4 check; CPU, no real data).

The training loop, the sample builder, and the calibration are replaced by stand-ins, so the check
tests only what `e00716_fit.fit` itself does: the gate refusals, the warm start from the stored stage
1 state of `M*`, the atomic stage files and their logs, a resume after a crash between stages, the
skip of a finished job, the refusal of a state written by other code (and the logged override), the
experiment 713 hash refusal and its override, the cap-hit rerun, and the peak-memory record.
The state files go under a throwaway folder in the home directory, removed at the end.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_fit as fit
import e00716_model as pvm
import numpy as np
import torch  # ty: ignore[unresolved-import]

MONTH, SEED = "2025-11", 0


def expect_refusal(*, fn, needle: str) -> None:  # noqa: ANN001
    """Assert that `fn()` raises an AssertionError whose message contains `needle`."""
    try:
        fn()
    except AssertionError as err:
        assert needle in str(err), f"wrong refusal: {err}"
        return
    msg = f"no refusal containing {needle!r}"
    raise AssertionError(msg)


def main() -> int:
    """Run the checks."""
    base = Path(tempfile.mkdtemp(prefix="f7fit_", dir=Path.home()))
    try:
        run(base=base)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("e00716_check_fit: all checks passed")
    return 0


def run(*, base: Path) -> None:
    """Build the throwaway data folder and exercise `fit`."""
    data_dir = base / "data"
    star_state = base / "m_star_stage1.pt"
    star_log = base / "m_star_stage1.json"
    torch.save(pvm.PvNet(arm="Z", seed=SEED).state_dict(), star_state)
    star_log.write_text(json.dumps({"code_hash": arms713.code_hash()}))
    arms.m_star_stage1_state = lambda **_: star_state  # ty: ignore[invalid-assignment]
    arms.m_star_stage1_log = lambda **_: star_log  # ty: ignore[invalid-assignment]
    arms.gate713_summary = lambda **_: {"waived": True, "failed_checks": ["KA3", "KA12"]}  # ty: ignore[invalid-assignment]
    arms.watched_mtimes = lambda **_: {}  # ty: ignore[invalid-assignment]
    gate = arms.gate_file(data_dir=data_dir)
    gate.parent.mkdir(parents=True)

    calls: list[tuple[int, int]] = []
    crash_at: list[int | None] = [None]
    cap_hit_first = [False]

    def stub_train(*, model, stage, **_):  # noqa: ANN001, ANN202
        calls.append((stage, len(calls)))
        if crash_at[0] == stage:
            raise RuntimeError("simulated crash")
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.001 * stage)
        first = tr.StageLog(
            stage=stage,
            passes=60 if cap_hit_first[0] else 7,
            best_val=0.5,
            cap_hit=cap_hit_first[0],
        )
        rerun = tr.StageLog(stage=stage, passes=70, best_val=0.45) if cap_hit_first[0] else None
        return first, rerun

    fit.train_with_rerun = stub_train  # ty: ignore[invalid-assignment]

    class FakeSamples(list):
        """Three samples with the `is_val` flags that the fit log counts."""

        is_val = np.array([False, False, True])

    tr.make_samples = lambda **_: FakeSamples([0, 1, 2])  # ty: ignore[invalid-assignment]
    tr.target_end = lambda **_: 10  # ty: ignore[invalid-assignment]
    pvm.PvNet.calibrate_scale = lambda self, batches: {"r_hist": 1.0}  # ty: ignore[invalid-assignment]
    import e00716_train as pvt

    pvt.calibration_batches = lambda **_: []  # ty: ignore[invalid-assignment]

    loader = lambda arm, month, seed, device: fit.FitInputs(
        world=SimpleNamespace(source=None), dropped={}, fingerprint="abc", encoders="frozen"
    )  # ty: ignore[invalid-argument-type]

    def go(arm: str = "H", **kw):  # noqa: ANN202
        return fit.fit(
            arm=arm, month=MONTH, seed=SEED, device="cpu", data_dir=data_dir, loader=loader, **kw
        )

    # 1. The gate: refused when missing, refused when written by other code, accepted with the override.
    expect_refusal(fn=go, needle="is missing")
    gate.write_text(json.dumps({"code": {"code_hash": "oldhash"}}))
    expect_refusal(fn=go, needle="oldhash")
    log = go(accept_gate="oldhash")
    assert log["code_overrides"][0]["kind"] == "gate", log["code_overrides"]
    for stage in (2, 3):
        assert arms.model_file(
            data_dir=data_dir, arm="H", month=MONTH, seed=SEED, stage=stage
        ).exists()
        assert fit.stage_log_file(
            data_dir=data_dir, arm="H", month=MONTH, seed=SEED, stage=stage
        ).exists()
    assert "peak_host_mb" in log and log["peak_host_mb"] > 0
    assert log["e00713_gate"]["waived"] is True
    print("gate refusals, override, state and log files, peak memory: ok")

    # 2. A finished job is skipped without training.
    gate.write_text(json.dumps({"code": arms.code_record()}))
    n = len(calls)
    go()
    assert len(calls) == n, "a finished job trained again"
    print("finished job skipped: ok")

    # 3. A resume after a crash between stages trains only the missing stage, and warm-starts from the stage 2 file.
    crash_at[0] = 3
    expect_refusal(fn=lambda: _crash_wrap(go, arm="HT"), needle="simulated crash")
    assert arms.model_file(data_dir=data_dir, arm="HT", month=MONTH, seed=SEED, stage=2).exists()
    assert not arms.model_file(
        data_dir=data_dir, arm="HT", month=MONTH, seed=SEED, stage=3
    ).exists()
    crash_at[0] = None
    calls.clear()
    go(arm="HT")
    assert [c[0] for c in calls] == [3], calls
    print("resume after a crash: ok")

    # 4. A state written by other code is refused, and an override names the exact old hash and is logged.
    path = fit.stage_log_file(data_dir=data_dir, arm="HT", month=MONTH, seed=SEED, stage=2)
    entry = json.loads(path.read_text())
    entry["code_hash"] = "deadbeef"
    path.write_text(json.dumps(entry))
    expect_refusal(fn=lambda: go(arm="HT"), needle="deadbeef")
    log = go(arm="HT", accept=["deadbeef"])
    assert any(o["old_hash"] == "deadbeef" for o in log["code_overrides"]), log
    print("code-hash refusal and logged override: ok")

    # 5. The experiment 713 hash on the stored stage 1 state: refused, then accepted with the exact old hash.
    star_log.write_text(json.dumps({"code_hash": "old713"}))
    expect_refusal(fn=lambda: go(arm="T"), needle="old713")
    log = go(arm="T", accept_713=["old713"])
    assert log["code_overrides"][0]["kind"] == "713 warm start", log["code_overrides"]
    star_log.write_text(json.dumps({"code_hash": arms713.code_hash()}))
    print("713 hash refusal and override: ok")

    # 6. A stage that hits the cap records the rerun.
    cap_hit_first[0] = True
    log = go(arm="H-S")
    assert log["stages"][0]["rerun"] is not None and log["stages"][0]["passes"] == 70, log[
        "stages"
    ][0]
    print("cap-hit rerun recorded: ok")

    # 7. A changed standardiser is refused.
    bad = lambda arm, month, seed, device: fit.FitInputs(
        world=SimpleNamespace(source=None), dropped={}, fingerprint="zzz", encoders="frozen"
    )  # ty: ignore[invalid-argument-type]
    fit.fit(arm="H-X", month=MONTH, seed=SEED, device="cpu", data_dir=data_dir, loader=loader)
    shutil.rmtree(
        arms.model_file(data_dir=data_dir, arm="H-X", month=MONTH, seed=SEED, stage=3).parent
    )
    expect_refusal(
        fn=lambda: fit.fit(
            arm="H-X", month=MONTH, seed=SEED, device="cpu", data_dir=data_dir, loader=bad
        ),
        needle="standardiser changed",
    )
    print("standardiser refusal: ok")


def _crash_wrap(go, **kw) -> None:  # noqa: ANN001
    """Run `go` and re-raise a simulated crash as an AssertionError so `expect_refusal` can match it."""
    try:
        go(**kw)
    except RuntimeError as err:
        raise AssertionError(str(err)) from err


if __name__ == "__main__":
    sys.exit(main())

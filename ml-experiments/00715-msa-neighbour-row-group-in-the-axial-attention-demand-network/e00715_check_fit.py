# ruff: noqa: E501, ANN001, ANN003, ANN201, D102, D107
"""Experiment 715: check the fit driver's resume, skip, and code-policy logic with a stub that trains nothing.

Usage: e00715_check_fit.py   (CPU only; run it through `run.sh`)

The stub replaces `load_world` and `train_stage` (`e00715_fit.Deps`) and `make_samples`, and the check runs
`e00715_fit.fit` against a temporary data folder that holds a fake experiment 713 stage 1 state. It checks
that (a) a fresh job runs stages 2 and 3 once each and saves both states, (b) a job that crashed in stage 3
resumes without retraining stage 2, (c) a finished job is skipped, (d) a stage log with another code hash
is refused, and accepted only when the exact old hash is named, (e) a stage that hits the 60-pass cap is
rerun from its starting state with a cap of 120 passes, (f) the log prints the peak host memory, and
(g) a missing gate file refuses the job. Each refusal check has a matching accepting call, so a driver that
refuses everything fails.
"""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import e00713_arms as a713
import e00713_model as mdl
import e00715_arms as arms
import e00715_fit as fit
import e00715_train as tr
import torch  # ty: ignore[unresolved-import]

MONTH, SEED = "2025-11", 0


def build_folder(*, root: Path) -> None:
    """Write the fake stage 1 state and log of `M1`, the selection, and the gate file."""
    sel = a713.selection_file(data_dir=root)
    sel.parent.mkdir(parents=True, exist_ok=True)
    sel.write_text(json.dumps({"m_star": "M1"}))
    state = arms.stage1_state_file(data_dir=root, start_arm="M1", month=MONTH, seed=SEED)
    state.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1)
    torch.save(mdl.MsaNet(switches=mdl.ARM_SWITCHES["M"]).state_dict(), state)
    log = arms.stage1_log_file(data_dir=root, start_arm="M1", month=MONTH, seed=SEED)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"code_hash": a713.code_hash()}))
    gate = arms.gate_file(data_dir=root)
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps({"code": arms.code_record()}))


class Stub:
    """Counts the stage calls and optionally crashes in, or caps, one stage."""

    def __init__(self, *, crash_stage: int | None = None, cap_stage: int | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.crash_stage, self.cap_stage = crash_stage, cap_stage

    def train_stage(self, *, model, stage, max_passes, **_):
        self.calls.append((stage, max_passes))
        if stage == self.crash_stage:
            raise RuntimeError("simulated crash")
        with torch.no_grad():
            next(model.parameters()).add_(0.5)
        cap = stage == self.cap_stage and max_passes == fit.CAP_FIRST
        return fit.tr713.StageLog(stage=stage, passes=max_passes, best_val=0.5, cap_hit=cap)

    def deps(self) -> fit.Deps:
        return fit.Deps(load_world=lambda **_: (None, {"stub": 0}), train_stage=self.train_stage)


def run(*, root: Path, stub: Stub, accept=()) -> tuple[dict, str]:
    """Run `fit.fit` with the stub and return the log and the printed text."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        log = fit.fit(
            arm="F4",
            month=MONTH,
            seed=SEED,
            device="cpu",
            data_dir=root,
            accept=accept,
            deps=stub.deps(),
        )
    return log, buf.getvalue()


def refused(*, root: Path, accept=()) -> bool:
    """Return whether `fit.fit` refuses with an AssertionError."""
    try:
        run(root=root, stub=Stub(), accept=accept)
    except AssertionError:
        return True
    return False


def main() -> int:
    """Run the checks."""
    tr.make_samples = lambda *, world, stage: [0, 1, 2]  # ty: ignore[invalid-assignment]
    results: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(dir=Path.home() / ".cache") as tmp:
        root = Path(tmp)
        build_folder(root=root)
        gate = arms.gate_file(data_dir=root)
        gate_text = gate.read_text()
        gate.unlink()
        results["g: no gate file is refused"] = refused(root=root)
        gate.write_text(gate_text)
        crash = Stub(crash_stage=3)
        try:
            run(root=root, stub=crash)
            results["b: the stub crashed"] = False
        except RuntimeError:
            results["b: the stub crashed"] = [c[0] for c in crash.calls] == [2, 3]
        stage2 = arms.model_file(data_dir=root, arm="F4", month=MONTH, seed=SEED, stage=2)
        results["b: stage 2 state saved before the crash"] = stage2.exists()
        resume = Stub()
        log, text = run(root=root, stub=resume)
        results["b: the resume trains stage 3 only"] = [c[0] for c in resume.calls] == [3]
        results["b: the resume log holds both stages"] = [s["stage"] for s in log["stages"]] == [
            2,
            3,
        ]
        results["f: the peak host memory is printed and logged"] = (
            "peak host" in text and log["peak_host_mb"] > 0
        )
        again = Stub()
        _, text = run(root=root, stub=again)
        results["c: a finished job runs nothing"] = again.calls == [] and "done" in text
        stage_log = arms.stage_log_file(data_dir=root, arm="F4", month=MONTH, seed=SEED, stage=3)
        original = stage_log.read_text()
        record = json.loads(original)
        record["code_hash"] = "deadbeefdeadbeef"
        stage_log.write_text(json.dumps(record))
        results["d: another code hash is refused"] = refused(root=root)
        accepted = run(root=root, stub=Stub(), accept=["deadbeefdeadbeef"])[0]
        results["d: the exact old hash is accepted and logged"] = "deadbeefdeadbeef" in json.dumps(
            accepted.get("code_overrides", accepted)
        )
        stage_log.write_text(original)
        for stage in (2, 3):
            arms.model_file(data_dir=root, arm="F4", month=MONTH, seed=SEED, stage=stage).unlink()
        arms.log_file(data_dir=root, arm="F4", month=MONTH, seed=SEED, kind="fit").unlink()
        cap = Stub(cap_stage=2)
        log, _ = run(root=root, stub=cap)
        results["e: a capped stage reruns once at 120 passes"] = cap.calls == [
            (2, 60),
            (2, 120),
            (3, 60),
        ]
        results["e: the log records both runs"] = log["stages"][0]["rerun"] is not None
    for name, ok in results.items():
        print(("PASS" if ok else "FAIL"), name)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: E501
"""Experiment 733: the `smoke` stage, wired to the real fits, predictions, and later stages.

One epoch of every arm runs in a scratch folder outside the data folder (`--scratch`), on the
device of `--device` (the real GPU for the official run). The scratch folder links to the data
folder's read-only inputs (`limits.json`, `channels.json`, `channels/`, and `world/`), holds the
states, records, and prediction sets of the smoke fits, and receives the outputs of the later
stages. The data folder gets `scores/smoke.json` only (`scores/smoke_cpu.json` on the CPU, which never
opens the chain).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_leakage as leak
import e00733_paths as paths
import e00733_runtime as rt
import e00733_smoke as smoke
import e00733_stages as st
import numpy as np

LINKED_INPUTS: Final = ("limits.json", "channels.json", "channels", "world")
SMOKE_729_TIMEOUT_SECONDS: Final = 3600


def scratch_root(*, outer: Path, root: Path) -> Path:
    """Return the smoke test's own data folder, named like `root`.

    The experiment store finds the folder of experiment 733 by its name, so the folder under the
    scratch folder carries the name of the data folder.
    """
    return outer / root.name


def link_inputs(*, root: Path, scratch: Path) -> list[str]:
    """Link the data folder's read-only inputs into the scratch folder.

    Args:
        root: This experiment's data folder.
        scratch: The scratch folder.

    Returns:
        The names linked.

    Raises:
        RuntimeError733: When an input the fits read does not exist (the smoke test would read none).
    """
    scratch.mkdir(parents=True, exist_ok=True)
    linked = []
    for name in LINKED_INPUTS:
        source = root / name
        if not source.exists():
            msg = f"the input {name} does not exist in the data folder, so the smoke fits cannot read it"
            raise rt.RuntimeError733(msg)
        target = scratch / name
        if not target.exists():
            target.symlink_to(source.resolve())
        linked.append(name)
    return linked


def read_stored_step(*, scratch: Path):  # noqa: ANN201
    """Return `read_stored(arm, path) -> n`: the scoring code's read of a stored smoke output."""
    import e00733_predict as pred
    import e00733_scoring as scoring
    import e00733_scoring_synth as synth
    import e00733_synth as sy
    import e00733_world as world

    def read(arm: str, path: Path) -> int:
        spec = core.ARMS[arm]
        if spec.where == "real":
            fold = spec.folds[0]
            plan = scoring.Plan(folds=(fold,), seeds=(0,), interim=True)
            return scoring.read_arm(
                root=scratch, arm=(arm, None), plan=plan, rows="validation"
            ).height
        segments, _ = world.read_world(root=scratch, kind="validation")
        index = sy.enumerate_samples(segments=segments)
        arrays = synth.read_arm(root=scratch, arm=arm, seeds=(0,), rows="validation")
        assert arrays is not None, path
        pred.check_stored_synthetic(
            arrays={**arrays, "p_neg": arrays["p_neg"]}, n_expected=len(index)
        )
        return len(arrays["p_neg"])

    return read


def check_steps(*, root: Path) -> dict[str, Any]:
    """Return the extra real-file checks of the smoke test, by name."""

    def banks() -> str:
        return f"{leak.assert_bank_digests(root_730=paths.root_730())} banks match their digests"

    def rerun_coverage() -> str:
        import e00733_rerun as rr

        shares = []
        for fold in core.ALL_FOLDS:
            out = rr.read_rerun(path=paths.rerun_dir(root=root) / f"{fold}.npz")
            covered = int(out.ok.any(axis=(1, 2)).sum())
            if covered == 0 or not np.isfinite(out.values[out.ok]).all():
                msg = f"fold {fold} holds no finite re-run output"
                raise rt.RuntimeError733(msg)
            shares.append(f"{fold}: {covered}/{len(out.origin_days)}")
        return "origins with a re-run by fold: " + ", ".join(shares)

    def poison() -> str:
        if not rt.poison_passed(root=root):
            msg = "scores/poison_test.json does not record a passed poison test: run `channels`"
            raise rt.RuntimeError733(msg)
        return "the poison test of the real re-run passed"

    def synthetic_samples() -> str:
        import e00733_synth as sy
        import e00733_world as world

        test, _ = world.read_world(root=root, kind="test")
        seg = next(s for s in test if (s.present & s.covered).any())
        n = len(sy.enumerate_samples(segments=[seg]))
        if n == 0:
            msg = "the first covered test segment yields no sample"
            raise rt.RuntimeError733(msg)
        return f"{n} samples with ENS rows from one test segment"

    def smoke_729() -> str:
        folder = paths.ROOT / "00729-msa-d8-neural-untangler-synthetic-known-answers"
        here = Path(__file__).resolve().parent
        extra = [
            here,
            folder,
            folder / "tests",
            paths.ROOT / "00730-msa-d8-sign-detector-with-demand-head-and-masking",
        ]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([*map(str, extra), *sys.path])}
        done = subprocess.run(
            [sys.executable, str(here / "e00733_smoke729.py")], capture_output=True,
            text=True, check=False, timeout=SMOKE_729_TIMEOUT_SECONDS, env=env,
        )  # fmt: skip
        lines = done.stdout.splitlines()
        if done.returncode != 0 or not any(
            line.startswith("bank_s2_solar: test bank present: True") for line in lines
        ):
            msg = f"experiment 729's checks on experiment 730's banks exited {done.returncode}: {(done.stdout + done.stderr)[-600:]}"
            raise rt.RuntimeError733(msg)
        return f"experiment 729's {len(lines)} real-file checks passed on experiment 730's banks"

    return {
        "bank digests": banks, "re-run outputs per fold": rerun_coverage, "poison test": poison,
        "synthetic samples": synthetic_samples, "729 smoke script": smoke_729,
    }  # fmt: skip


def later_steps(*, scratch: Path, device: str = "cpu") -> dict[str, Any]:
    """Return the later stages, each runnable on validation rows in the scratch folder."""
    import e00733_figstages as figstages
    import e00733_report as rep
    import e00733_scorestage as scorestage

    args = argparse.Namespace(validation_only=True, device=device)

    def score(validation_only: bool = True) -> str:
        verdict = scorestage.run_score(root=scratch, rows="validation")
        return f"verdict row {verdict['row']} on validation rows"

    def report(validation_only: bool = True) -> str:
        return f"wrote {rep.write_report(root=scratch, counted_read=False).name}"

    def figure_stage(function: Any) -> Any:
        def run(validation_only: bool = True) -> str:
            result = function(args, scratch)
            return f"wrote {len(result['files'])} figures: {', '.join(result['files'])}"

        return run

    return {
        "score": score, "report": report, "figures": figure_stage(figstages.stage_figures),
        "loss_curves": figure_stage(figstages.stage_loss_curves),
        "attention": figure_stage(figstages.stage_attention), "series": figure_stage(figstages.stage_series),
    }  # fmt: skip


def gpu_info() -> dict[str, Any]:
    """Return the name of the GPU and this process's peak GPU memory in gigabytes."""
    import torch

    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def stage_smoke(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Run the smoke test: one epoch of every arm, then every later stage on validation rows.

    Args:
        args: The parsed arguments (`--scratch`, `--device`).
        root: This experiment's data folder (it receives `scores/smoke.json` only, or
            `scores/smoke_cpu.json` on the CPU).

    Returns:
        The smoke record.

    Raises:
        RuntimeError733: When no scratch folder was given or a step failed.
    """
    import e00733_predictions as pr

    if getattr(args, "scratch", None) is None:
        msg = "the smoke stage needs --scratch, a folder outside the data folder"
        raise rt.RuntimeError733(msg)
    outer = Path(args.scratch)
    smoke.assert_scratch_outside(scratch=outer, data_root=root)
    scratch = scratch_root(outer=outer, root=root)
    link_inputs(root=root, scratch=scratch)
    run_job = rt.make_run_job(root=scratch, device=args.device, max_passes=1)

    def predict(arm: str, fold: str | None, seed: int) -> Path:
        return pr.predict_job(
            root=scratch, arm=arm, fold=fold, seed=seed, device=args.device, rows="validation"
        )[0]

    steps = smoke.SmokeSteps(
        fit_real=run_job,
        fit_synthetic=lambda arm, seed: run_job(arm, None, seed),
        predict=predict, read_stored=read_stored_step(scratch=scratch),
        checks=check_steps(root=root), later=later_steps(scratch=scratch, device="cpu"),
        gpu_info=gpu_info if args.device == "cuda" else None,
    )  # fmt: skip
    started = time.monotonic()
    record = smoke.run_smoke(root=root, scratch=scratch, steps=steps, device=args.device)
    record["seconds"] = round(time.monotonic() - started, 1)
    out_file = root / st.smoke_file(device=args.device)
    st.write_json(path=out_file, payload=record)
    leak.assert_wrote_only_here(written=[out_file], data_root=root)
    if not record["pass"]:
        failed = [k for k, v in record["steps"].items() if not v["ok"]]
        msg = f"the smoke test failed at: {failed}"
        raise rt.RuntimeError733(msg)
    return record

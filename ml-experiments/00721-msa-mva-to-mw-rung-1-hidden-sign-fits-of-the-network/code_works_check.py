# ruff: noqa: E501
"""Experiment 721: the first GPU code-works check on one tiny real fold (not run until told to).

The check fits `SG-A` in `W2` on the real fold 2026-05 for a handful of passes and asserts the ten
properties that the second code review asked for. It makes no claim about any forecast. Every
output goes under a scratch path under `$HOME`, and the process holds at most 6 GB of the GPU.

A real fold is built in one phase per process, so the check has four subcommands, each run
through `run.sh`:

1. `fit SCRATCH`: assertions 1 (no-continuation scored state), 2 (training target), 3 (sign
   labels), 4 (gate at the supervised Primaries), 5 (closed gate equals the base at the real
   shift), and the fit's GPU peak (8).
2. `predict SCRATCH`: assertions 6 (ensemble and PV runs at least 9 h before the native origin),
   7 (the lead window), 9 (the predict phase reproduces the fitted PV scales), the predict peak
   (8), and the share of PV target rows that are missing per Primary.
3. `probe SCRATCH`: assertion 8 for the capacity probe (GPU peak at most 6 GB).
4. `routing SCRATCH`: assertion 10 (`score_all` on the complete set of wave outputs reaches rows
   4 to 9; CPU only).

Usage: `SLOT_CPU=3 run.sh --go code_works_check.py fit SCRATCH --device cuda`, with `fit`,
`predict`, `probe`, or `routing` as the subcommand.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Final

import e00721_paths

e00721_paths.pin_threads()

# isort: split
import e00721_core as core  # noqa: E402
import e00721_fit as fit  # noqa: E402
import numpy as np  # noqa: E402

FOLD: Final = "2026-05"
JOB: Final = fit.Job(arm="SG-A", world="W2", fold=FOLD, seed=0)
PASSES: Final = 2
GPU_CAP_GB: Final = 6.0
REPORT_FILE: Final = "code_works_report.json"


def say(*, name: str, ok: bool, detail: str = "") -> None:
    """Print one assertion's result and raise when it fails."""
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
    assert ok, f"{name} failed: {detail}"


def peak_gb(*, torch: object, device: str) -> float:
    """Return the GPU's peak allocation of this process in GB, or 0 on the CPU."""
    if not device.startswith("cuda"):
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 2**30  # ty: ignore[unresolved-attribute]


def prepare_scratch(*, scratch: Path, data_dir: Path) -> None:
    """Copy the fold's probe files, or write stand-in capacities (named as such) when the probe has not run."""
    import e00721_probe as probe

    for world in ("W2", "W0"):
        dest = probe.probe_path(data_dir=scratch, world=world, fold=FOLD)
        dest.parent.mkdir(parents=True, exist_ok=True)
        real_file = probe.probe_path(data_dir=data_dir, world=world, fold=FOLD)
        if real_file.exists():
            shutil.copy(real_file, dest)
        elif not dest.exists():
            fit.atomic_write_json(path=dest, body={"D5": 1.0, "D10": 1.0})
            print(f"stand-in probe capacities in {dest.name}: the probe has not run", flush=True)


def tiny_caps() -> None:
    """Cut the pass caps so that a fit takes a few passes."""
    vars(core).update(
        REAL_FOLD_CAP_FIRST=PASSES, REAL_FOLD_CAP_RERUN=PASSES + 1, CONTINUATION_CAP_PASSES=PASSES + 2,
    )  # fmt: skip


def inputs_of(*, scratch: Path, device: str, phase: str):  # noqa: ANN201
    """Return the `WorldInputs` of one phase."""
    import e00721_jobs as jobs
    import e00721_real as real

    os.environ["E00667_PHASE"] = phase
    return jobs.WorldInputs(
        make_world=real.make_world_builder(data_dir=scratch, device=device, phase=phase),
        device=device,
    )


def check_fit(*, scratch: Path, data_dir: Path, device: str) -> dict:
    """Run the fit-phase assertions."""
    import e00721_arms as arms
    import e00721_real as real
    import torch  # ty: ignore[unresolved-import]

    if device.startswith("cuda"):
        fit.limit_gpu_memory(torch=torch)
        torch.cuda.reset_peak_memory_stats()
    tiny_caps()
    prepare_scratch(scratch=scratch, data_dir=data_dir)
    report: dict = {}
    # 1. the scored state equals a no-continuation run when the continuation option is off
    states = {}
    for label, continuation in (("without", False), ("with", True)):
        root = scratch / label
        prepare_scratch(scratch=root, data_dir=data_dir)
        inputs = inputs_of(scratch=root, device=device, phase="train")
        curve = fit.fit_job(
            job=JOB, inputs=inputs, data_dir=root, budget=fit.Budget(), continuation=continuation
        )
        stage1 = curve["stages"][0]
        states[label] = {
            k: v.cpu()
            for k, v in torch.load(
                fit.job_paths(data_dir=root, job=JOB)["state"], map_location="cpu"
            ).items()
        }
        if not continuation:
            say(
                name="1a the skipped continuation is flagged",
                ok=bool(stage1["flag"]["flagged"]) and not stage1["continued"],
                detail=str(stage1["flag"]),
            )
    same = all(
        torch.allclose(states["with"][k].float(), states["without"][k].float(), atol=1e-4)
        for k in states["with"]
    )
    say(name="1b the scored state does not depend on the continuation option", ok=same)
    # 2 to 5 read the world of the continuation-off fit
    aw = inputs.world_of(job=JOB)
    pre = np.arange(aw.world.clean.shape[1]) < aw.world.cutoff_idx
    say(
        name="2a the target is NaN at and after the cutoff",
        ok=bool(np.isnan(aw.world.clean[:, ~pre]).all()),
    )
    say(
        name="2b the target is finite before the cutoff for some Primary",
        ok=bool(
            np.isfinite(aw.world.clean[:, pre]).any(axis=1).all()
            or np.isfinite(aw.world.clean[:, pre]).any()
        ),
    )
    gi = real.load_grid_inputs(fold=FOLD, seed=0, phase="train")
    sites = list(gi.panel.sites)
    sign = aw.sources[1].sign_grid.cpu().numpy()
    hidden = [sites.index(s) for s in core.HIDDEN_PAIR]
    visible = [sites.index(s) for s in core.VISIBLE_SIGNED]
    say(name="3a no sign label at D5 or D10", ok=bool(np.isnan(sign[hidden]).all()))
    say(
        name="3b sign labels at all four visible signed Primaries",
        ok=all(bool(np.isfinite(sign[i]).any()) for i in visible),
        detail=f"{len(visible)} Primaries",
    )
    say(
        name="4 the gate is open at the four supervised Primaries",
        ok=bool((aw.gate[visible] > 0.5).all()),
    )
    # 5. a closed gate reads out 716's base at the real per-Primary shift
    import e00716_model as pvm  # ty: ignore[unresolved-import]

    source = aw.sources[1]
    c_idx = torch.as_tensor(
        np.linspace(aw.world.cutoff_idx - 48 * 30, aw.world.cutoff_idx - 48 * 12, 8).astype(
            np.int64
        )
        // 48
        * 48
        + 24,
        device=source.device,
    )
    primary = torch.as_tensor(np.array(visible[:2] * 4)[:8], device=source.device)
    batch = source.assemble(
        primary=primary, c_idx=c_idx, lead_day=torch.zeros_like(primary), fold_hist=None
    )
    torch.manual_seed(0)
    plain = pvm.PvNet(arm="Z", seed=0).to(source.device).eval()
    torch.manual_seed(0)
    sg = arms.SgNet(seed=0, sign_head=True).to(source.device).eval()
    sg.set_gate(gate=np.zeros(16))
    sg.set_shift(shift=aw.shift)
    with torch.no_grad():
        want, got = plain(batch.inputs).out, sg(batch.inputs).out
    say(
        name="5 a closed gate reproduces 716's read-out at the real shift",
        ok=bool(torch.allclose(want, got, atol=1e-4)),
        detail=f"largest gap {float((want - got).abs().max()):.2e}",
    )
    gb = peak_gb(torch=torch, device=device)
    say(name="8a the fit's GPU peak is at most 6 GB", ok=gb <= GPU_CAP_GB, detail=f"{gb:.2f} GB")
    report["fit_peak_gb"] = gb
    return report


def check_predict(*, scratch: Path, data_dir: Path, device: str) -> dict:
    """Run the predict-phase assertions on the fit of the `without` scratch folder."""
    import e00721_train as trn
    import torch  # ty: ignore[unresolved-import]

    if device.startswith("cuda"):
        fit.limit_gpu_memory(torch=torch)
        torch.cuda.reset_peak_memory_stats()
    root = scratch / "without"
    prepare_scratch(scratch=root, data_dir=data_dir)
    inputs = inputs_of(scratch=root, device=device, phase="predict")
    state = torch.load(fit.job_paths(data_dir=root, job=JOB)["state"], map_location="cpu")
    inputs.load_network_state(job=JOB, state=state)
    aw = inputs.world_of(job=JOB)
    model = inputs.model_of(job=JOB)
    # 9. the predict phase reproduces the fitted PV scales (and every other stored tensor) exactly
    loaded = inputs.network_state(job=JOB)
    scale_keys = [k for k in state if "scale" in k or k.rsplit(".", 1)[-1] in {"r_h", "r_t"}]
    say(
        name="9 the predict phase reproduces the fit's PV scales exactly",
        ok=all(torch.equal(loaded[k], state[k].cpu()) for k in state) and bool(scale_keys),
        detail=f"{len(scale_keys)} scale tensors",
    )
    # 6. every ensemble and PV run is at or before native origin - 9 h (the batch builders assert it; check it directly as well)
    assert aw.score_c_idx is not None
    source = aw.sources[aw.predict_stage]
    c_all = torch.as_tensor(
        aw.score_c_idx[:: max(len(aw.score_c_idx) // 20, 1)], device=source.device
    )
    for p in aw.score_primaries:
        primary = torch.full_like(c_all, p)
        source.assemble(
            primary=primary, c_idx=c_all, lead_day=torch.zeros_like(primary), fold_hist=None
        )
        day = c_all // 48 - source.pv.tables.first_day
        source.pv.check_run_times(primary=primary, day=day.clamp(min=0), c_idx=c_all)
    say(
        name="6 every ensemble and PV run is at or before the native origin minus 9 h",
        ok=True,
        detail=f"{len(c_all)} origins, {len(aw.score_primaries)} Primaries",
    )
    # 7. the lead window counts from the 12:00 UTC origin
    rows = trn.predict_rows(
        model=model,
        aw=aw,
        c_idx=aw.score_c_idx[:3],
        primaries=aw.score_primaries[:2],
        stamp_us=aw.stamp_us,
    )
    lead_h = (
        (rows["valid_time"] - rows["power_fcst_init_time"]).dt.total_minutes() / 60.0
    ).to_numpy()
    say(
        name="7a leads run from 24 h up to but not including 240 h",
        ok=bool(
            float(lead_h.min()) >= core.LEAD_MIN_HOURS
            and float(lead_h.max()) < core.LEAD_MAX_HOURS_EXCLUSIVE
        ),
        detail=f"{float(lead_h.min())} to {float(lead_h.max())}",
    )
    say(
        name="7b the origin is 12:00 UTC",
        ok=bool(
            (rows["power_fcst_init_time"].dt.hour() == 12).all()
            and (rows["power_fcst_init_time"].dt.minute() == 0).all()
        ),
    )
    gb = peak_gb(torch=torch, device=device)
    say(
        name="8b the predict phase's GPU peak is at most 6 GB",
        ok=gb <= GPU_CAP_GB,
        detail=f"{gb:.2f} GB",
    )
    # 12. the share of PV target rows that are missing, per Primary (printed as a share only)
    tables = source.pv.tables
    missing = {}
    for p in range(16):
        top = getattr(tables, "top_ok", None)
        if top is not None:
            missing[str(p)] = round(float(1.0 - top[p].float().mean()), 3)
    print(f"missing share of PV target rows per Primary: {missing}", flush=True)
    return {"predict_peak_gb": gb, "missing_pv_target_share": missing}


def check_probe(*, scratch: Path, device: str) -> dict:
    """Run the probe of the fold and assert its GPU peak."""
    import e00721_probe as probe
    import torch  # ty: ignore[unresolved-import]

    if device.startswith("cuda"):
        fit.limit_gpu_memory(torch=torch)
        torch.cuda.reset_peak_memory_stats()
    out = probe.probe_capacity(fold=FOLD, device=device)
    gb = peak_gb(torch=torch, device=device)
    say(
        name="8c the probe's GPU peak is at most 6 GB",
        ok=gb <= GPU_CAP_GB and all(v > 0 for v in out.values()),
        detail=f"{gb:.2f} GB",
    )
    del scratch
    return {"probe_peak_gb": gb}


def check_routing(*, scratch: Path) -> dict:
    """Assert that `score_all` on the complete set of wave outputs reaches rows 4 to 9."""
    sys.path.insert(0, str(Path(__file__).parent / "tests"))
    import e00721_report as rep
    from test_e00721_review2 import _score, _write_all_waves  # ty: ignore[unresolved-import]

    root = scratch / "routing"
    jobs = _write_all_waves(root=root, flag_sg_a=False)
    summary = _score(root)
    report = rep.format_report(summary=summary, mde=core.read_mde_file(path=root / "mde.json"))
    row = next(
        int(line.split("row ")[1].split(":")[0])
        for line in report.splitlines()
        if line.startswith("verdict row")
    )
    say(
        name="10 score_all on the complete wave outputs reaches rows 4 to 9",
        ok=len(jobs) == 64 and not summary["fit_failed"] and 4 <= row <= 9,
        detail=f"row {row}",
    )
    return {"row": row}


def main() -> int:
    """Run one subcommand and write its part of the report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("fit", "predict", "probe", "routing"))
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert str(args.scratch).startswith(str(Path.home())), "the scratch path must sit under $HOME"
    data_dir = Path(os.environ["DATA_PATH_INTERNAL"])
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.command == "fit":
        report = check_fit(scratch=args.scratch, data_dir=data_dir, device=args.device)
    elif args.command == "predict":
        report = check_predict(scratch=args.scratch, data_dir=data_dir, device=args.device)
    elif args.command == "probe":
        report = check_probe(scratch=args.scratch, device=args.device)
    else:
        report = check_routing(scratch=args.scratch)
    path = args.scratch / f"{args.command}_{REPORT_FILE}"
    path.write_text(json.dumps(report, indent=1))
    print(f"{args.command}: all assertions passed; report in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

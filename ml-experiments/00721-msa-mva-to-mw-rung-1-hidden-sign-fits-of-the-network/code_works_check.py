# ruff: noqa: E501, PLR0915
"""Experiment 721: the first GPU code-works check on one tiny real fold (not run until told to).

The check fits `SG-A` in `W2` on the real fold 2026-05 for a handful of passes and asserts the ten
properties that the second code review asked for. It makes no claim about any forecast. Every
output goes under a scratch path under `$HOME`, and the process holds at most 6 GB of the GPU.

A real fold is built in one phase per process, and a process must not build two real worlds in
12 GB, so the check has eight subcommands, each run through `run.sh`. Every one prints the host
memory peak of its process.

1. `fit-without SCRATCH`: assertions 1a (the skipped continuation is flagged), 2 (training target),
   3 (sign labels), 4 (gate at the supervised Primaries), 5 (a closed gate equals 716's `HT`
   read-out at the real shift), and the fit's GPU peak (8).
2. `fit-with SCRATCH`: the same fit with the continuation. `compare SCRATCH` then reports the
   largest gap between the two scored states (1b) and fails only above a loose bar.
3. `predict SCRATCH`: assertions 6 (ensemble and PV runs at least 9 h before the native origin),
   7 (the lead window), 9 (predicting leaves the stored state, PV scales included, unchanged),
   the predict peak (8), and the missing share of PV target rows per Primary (12).
4. `probe SCRATCH`: assertion 8 for the capacity probe (GPU peak at most 6 GB).
5. `synth-fit SCRATCH` and `synth-predict SCRATCH`: the synthetic fit on the pipeline fold
   2026-03, with the cutoff, the scored origins, the token properties (none before the cutoff, the
   predicted last token, scored valid times before the holdout, days per class equal to Stage 0's),
   seconds, and GPU peak printed so the extra cost can be re-measured.
6. `routing SCRATCH`: assertion 10 (`score_all` on the complete set of wave outputs reaches rows
   4 to 9; CPU only).

Usage: `SLOT_CPU=3 run.sh --go code_works_check.py fit-without SCRATCH --device cuda`.
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
LOOSE_STATE_GAP: Final = 1e-2
"""Hard bar on the gap between two fits' scored states; rounding differences sit far below it."""
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


def check_fit(*, scratch: Path, data_dir: Path, device: str, continuation: bool) -> dict:
    """Fit once (with or without the continuation) and, without it, run the fit-phase assertions.

    The two fits run in two processes (`fit-without`, `fit-with`), so that no process builds two real
    worlds in 12 GB. `compare` then reads the two stored states.
    """
    import e00721_arms as arms
    import e00721_real as real
    import torch  # ty: ignore[unresolved-import]

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.startswith("cuda"):
        fit.limit_gpu_memory(torch=torch)
        torch.cuda.reset_peak_memory_stats()
    tiny_caps()
    report: dict = {}
    label = "with" if continuation else "without"
    root = scratch / label
    prepare_scratch(scratch=root, data_dir=data_dir)
    inputs = inputs_of(scratch=root, device=device, phase="train")
    curve = fit.fit_job(
        job=JOB, inputs=inputs, data_dir=root, budget=fit.Budget(), continuation=continuation
    )
    if continuation:
        gb = peak_gb(torch=torch, device=device)
        say(
            name="8a the fit's GPU peak is at most 6 GB", ok=gb <= GPU_CAP_GB, detail=f"{gb:.2f} GB"
        )
        return {"fit_peak_gb": gb}
    stage1 = curve["stages"][0]
    say(
        name="1a the skipped continuation is flagged",
        ok=bool(stage1["flag"]["flagged"]) and not stage1["continued"],
        detail=str(stage1["flag"]),
    )
    # 2 to 5 read the world of this (continuation-off) fit
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
    plain = pvm.PvNet(arm="HT", seed=0).to(source.device).eval()
    torch.manual_seed(0)
    sg = arms.SgNet(seed=0, sign_head=True).to(source.device).eval()
    fitted = torch.load(fit.job_paths(data_dir=root, job=JOB)["state"], map_location="cpu")
    plain.load_state_dict(
        fitted, strict=False
    )  # the shared trunk and mixture weights; the head keys are SgNet's own
    sg.load_state_dict(fitted)
    sg.set_gate(gate=np.zeros(16))
    sg.set_shift(shift=aw.shift)
    with torch.no_grad():
        want, got = plain(batch.inputs).out, sg(batch.inputs).out
    say(
        name="5 a closed gate reproduces 716's `HT` read-out at the real shift (fitted weights)",
        ok=bool(torch.allclose(want, got, atol=1e-4)),
        detail=f"largest gap {float((want - got).abs().max()):.2e}",
    )
    gb = peak_gb(torch=torch, device=device)
    say(name="8a the fit's GPU peak is at most 6 GB", ok=gb <= GPU_CAP_GB, detail=f"{gb:.2f} GB")
    report["fit_peak_gb"] = gb
    return report


def check_compare(*, scratch: Path) -> dict:
    """Report the largest gap between the scored states of the two fits (item 1b).

    Two GPU fits can differ by rounding, so the gap is reported and only a loose bar fails: a
    continuation that changed the scored state would show a gap far above it.
    """
    import torch  # ty: ignore[unresolved-import]

    states = {
        label: torch.load(
            fit.job_paths(data_dir=scratch / label, job=JOB)["state"], map_location="cpu"
        )
        for label in ("with", "without")
    }
    gap = max(
        float((states["with"][k].float() - states["without"][k].float()).abs().max())
        for k in states["with"]
    )
    say(
        name="1b the scored state does not depend on the continuation option",
        ok=gap <= LOOSE_STATE_GAP,
        detail=f"largest gap {gap:.2e} (bar {LOOSE_STATE_GAP})",
    )
    return {"largest_state_gap": gap}


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
    # 9. predicting leaves the network (the PV scales included) exactly as stored: nothing is recalibrated
    after = inputs.network_state(job=JOB)
    scale_keys = [k for k in state if "scale" in k or k.rsplit(".", 1)[-1] in {"r_h", "r_t"}]
    say(
        name="9 the network after predicting equals the stored state, PV scales included",
        ok=all(torch.equal(after[k], state[k].cpu()) for k in state) and bool(scale_keys),
        detail=f"{len(scale_keys)} scale tensors",
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
        missing[str(p)] = round(float(1.0 - tables.top_filled[p].float().mean()), 3)
    say(
        name="12 the missing share of PV target rows is printed per Primary",
        ok=len(missing) == 16,
        detail=str(missing),
    )
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


SYNTH_JOB: Final = fit.Job(arm="SG-A", world="synthetic", fold="synthetic", seed=0)


def check_synth(*, scratch: Path, device: str, phase: str) -> dict:
    """Fit (phase `train`) or predict (phase `predict`) the synthetic `SG-A` and assert its span."""
    import time

    import e00721_real as real
    import torch  # ty: ignore[unresolved-import]

    if device.startswith("cuda"):
        fit.limit_gpu_memory(torch=torch)
        torch.cuda.reset_peak_memory_stats()
    vars(core).update(SYNTH_MAX_PASSES=PASSES)
    inputs = inputs_of(scratch=scratch, device=device, phase=phase)
    started = time.monotonic()
    if phase == "train":
        fit.fit_job(
            job=SYNTH_JOB, inputs=inputs, data_dir=scratch, budget=fit.Budget(), continuation=False
        )
        aw = inputs.world_of(job=SYNTH_JOB)
        say(
            name="11a the synthetic pipeline fold is 2026-03",
            ok=core.SYNTH_PIPELINE_FOLD == "2026-03",
        )
        say(
            name="11b the synthetic target is NaN at and after the cutoff",
            ok=bool(np.isnan(aw.world.clean[:, aw.world.cutoff_idx :]).all()),
        )
    else:
        fit.predict_job(job=SYNTH_JOB, inputs=inputs, data_dir=scratch)
        aw = inputs.world_of(job=SYNTH_JOB)
        c = aw.score_c_idx
        say(
            name="11c scored origins lie in March to June 2026",
            ok=bool(
                c is not None and (c == real.synthetic_score_origins(n_grid=len(aw.stamp_us))).all()
            ),
            detail=f"{len(c)} origins",
        )
        tokens = fit.job_paths(data_dir=scratch, job=SYNTH_JOB)["predictions"].with_suffix(
            ".tokens.npz"
        )
        say(name="11d the sign tokens are written", ok=tokens.exists())
        import e00721_stage0 as stage0

        tok = dict(np.load(tokens))
        grid = (
            tok["exporters_token_id"] % 10**7
        )  # the token id is the Primary times 10**7 plus the grid index
        start, stop = real.synthetic_token_span(n_grid=len(aw.stamp_us))
        say(
            name="11f no sign token before the cutoff",
            ok=bool(grid.min() >= aw.world.cutoff_idx),
            detail=str(int(grid.min()) - aw.world.cutoff_idx),
        )
        say(name="11g the last sign token is the predicted one", ok=int(grid.max()) == stop - 1)
        holdout = int(
            (np.datetime64(core.HOLDOUT_START_ISO, "us").astype(np.int64) - aw.stamp_us[0])
            // (aw.stamp_us[1] - aw.stamp_us[0])
        )
        say(
            name="11h every scored valid time is before the holdout",
            ok=bool(int(c.max()) + int(core.LEAD_MAX_HOURS_EXCLUSIVE * 2) <= holdout),
        )
        days = stage0.class_days(
            day=grid // 48, x_true=tok["exporters_x_points"], u=tok["exporters_u"]
        )
        gi = real.load_grid_inputs(
            fold=core.SYNTH_PIPELINE_FOLD, seed=0, phase="predict", with_farms=False
        )
        parts = real.synthetic_inputs(
            gi=gi,
            seed=0,
            day_offset_points=real.synthetic_day_offset_points(
                root=fit.data_root(data_dir=scratch)
            ),
        )
        want = stage0.class_days(
            **{
                k: v
                for k, v in stage0.exporter_tokens(parts=parts, span=(start, stop)).items()
                if k in {"day", "x_true", "u"}
            }
        )
        say(
            name="11i the token days per class equal Stage 0's",
            ok=days == want,
            detail=f"{days} against {want}",
        )
    seconds = time.monotonic() - started
    gb = peak_gb(torch=torch, device=device)
    say(
        name="11e the GPU peak is at most 6 GB",
        ok=gb <= GPU_CAP_GB,
        detail=f"{gb:.2f} GB, {seconds:.0f} s",
    )
    return {"phase": phase, "seconds": seconds, "peak_gb": gb}


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
    parser.add_argument(
        "command",
        choices=(
            "fit-without",
            "fit-with",
            "compare",
            "predict",
            "probe",
            "routing",
            "synth-fit",
            "synth-predict",
        ),
    )
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert str(args.scratch).startswith(str(Path.home())), "the scratch path must sit under $HOME"
    data_dir = Path(os.environ["DATA_PATH_INTERNAL"])
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.command in {"fit-without", "fit-with"}:
        report = check_fit(
            scratch=args.scratch,
            data_dir=data_dir,
            device=args.device,
            continuation=args.command == "fit-with",
        )
    elif args.command == "compare":
        report = check_compare(scratch=args.scratch)
    elif args.command == "predict":
        report = check_predict(scratch=args.scratch, data_dir=data_dir, device=args.device)
    elif args.command in {"synth-fit", "synth-predict"}:
        report = check_synth(
            scratch=args.scratch / "synth",
            device=args.device,
            phase="train" if args.command == "synth-fit" else "predict",
        )
    elif args.command == "probe":
        report = check_probe(scratch=args.scratch, device=args.device)
    else:
        report = check_routing(scratch=args.scratch)
    import resource

    report["host_peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(f"host memory peak of this process: {report['host_peak_mb']} MB (the cap is 12288 MB)")
    path = args.scratch / f"{args.command}_{REPORT_FILE}"
    path.write_text(json.dumps(report, indent=1))
    print(f"{args.command}: all assertions passed; report in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

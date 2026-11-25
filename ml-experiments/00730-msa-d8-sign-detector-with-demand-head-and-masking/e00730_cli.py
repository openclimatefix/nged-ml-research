# ruff: noqa: E402, PLR0912, PLR0915, E501
"""Experiment 730: the command line. Run it through `run.sh`, never directly.

    SLOT_CPUS=<list> run.sh --go e00730_cli.py <stage> [options]

Stages, in the order of the plan: `banks`, `gate`, `gallery` (then `panels` once the design lead has
read the gallery), `pretrain`, `timing`, `screen`, `choose`, `margin`, `styles`, `later`,
`refit729`, `fit`, `real`, `score`, `report`, `figures`, `series`, and `components`. Every stage
prints counts and scores only, and every stage but `banks` runs its guard first.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import e00730_paths as paths

paths.pin_threads()

import e00730_banks as banks
import e00730_core as core
import e00730_stages as stages


def log(line: str) -> None:
    """Print one progress line with a UTC time stamp."""
    print(f"{datetime.now(UTC):%H:%M:%S} {line}", flush=True)


def device_runner(*, root: Path, device: str):  # noqa: ANN201
    """Return a function that fits one new arm on the real context."""
    ctx = stages.real_context()
    return lambda arm, arch, seed: stages.fit_one(
        root=root, ctx=ctx, arm=arm, arch=arch, seed=seed, device=device, log=log
    )


def run_pretrain(*, root: Path, device: str, max_epochs: int | None, tmp: bool) -> dict:
    """Run the three pre-training runs (or, for the timing gate, 2 epochs of one into a scratch folder)."""
    import e00730_pretrain as pretrain

    ctx = stages.real_context()
    held = [*ctx.months(split="test"), *ctx.months(split="validation")]
    record_days = stages.stages729.data.record_days()  # ty: ignore[unresolved-attribute]
    target = root / "timing_scratch" if tmp else root
    record = pretrain.run_pretraining(
        root=target, held_out_months=held, record_days=record_days,
        seeds=[0] if tmp else core.PRETRAIN_SEEDS, device=device, max_epochs=max_epochs,
    )  # fmt: skip
    pretrain.write_record(record=record, path=target / "pretrained" / "record.json")
    if not tmp:
        standardiser = pretrain.read_standardiser(folder=root / "pretrained")
        base = ctx.cal.dates
        pretrain.build_d8_encoder_table(root=root, base_dates=base, standardiser=standardiser)
    return record


def main(argv: list[str]) -> int:
    """Run one stage and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*stages.stage_names(), "panels"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm")
    parser.add_argument("--arch", choices=core.ARCHITECTURES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--style", choices=core.STYLES, default="K")
    parser.add_argument("--confirm-panels", help="JSON file of the design lead's per-panel reading")
    parser.add_argument("--manager-approved", action="store_true")
    args = parser.parse_args(argv)
    root = paths.data_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        return dispatch(args=args, root=root)
    except banks.StoppedError as exc:
        log(f"STOPPED (verdict row 0): {exc}")
        return 3


def dispatch(*, args: argparse.Namespace, root: Path) -> int:
    """Run the stage that `args` names."""
    stage = args.stage
    if stage == "banks":
        banks.guard_run(root=root, stage="banks")
        ctx = stages.real_context()
        banks.stage_banks(
            root=root, inputs=stages.bank_inputs(ctx=ctx), train_seeds=range(5), log=log
        )
    elif stage == "gate":
        ctx = stages.real_context()
        banks.stage_gate(root=root, gate_inputs=stages.real_gate_inputs(ctx=ctx), log=log)
    elif stage == "gallery":
        stages.stage_gallery(root=root, shares_real=None, log=log)
    elif stage == "panels":
        confirms = json.loads(Path(args.confirm_panels).read_text())
        record = stages.record_panel_check(root=root, design_lead_confirms=confirms)
        log(f"panel check recorded: all passed {record['all_passed']}")
    elif stage == "pretrain":
        banks.guard_run(root=root, stage="pretrain")
        log(
            f"pre-training: {run_pretrain(root=root, device=args.device, max_epochs=None, tmp=False)}"
        )
    elif stage == "timing":
        banks.guard_run(root=root, stage="timing")
        ctx = stages.real_context()
        secs = {
            a: stages.measure_seconds_per_epoch(root=root, ctx=ctx, arch=a, device=args.device)
            for a in core.ARCHITECTURES
        }
        scratch = run_pretrain(
            root=root, device=args.device, max_epochs=core.TIMING_PRETRAIN_EPOCHS, tmp=True
        )
        stages.stage_timing(
            root=root, seconds_per_epoch=secs, peak_memory_gb=scratch["peak_memory_gb"],
            manager_approved=args.manager_approved, log=log,
        )  # fmt: skip
    elif stage in ("screen", "styles", "later"):
        run_one = device_runner(root=root, device=args.device)
        counts = getattr(stages, f"stage_{stage}")(root=root, run_one=run_one, log=log)
        log(f"{stage}: {counts}")
    elif stage == "refit729":
        ctx = stages.real_context()
        counts = stages.stage_refit729(
            root=root,
            run_one=lambda a, ar, s: stages.run_refit_729(
                root=root, ctx=ctx, arm=a, arch=ar, seed=s, device=args.device
            ),
            log=log,
        )
        log(f"refit729: {counts}")
    elif stage == "choose":
        log(str(stages.choose_architecture(root=root)))
    elif stage == "margin":
        log(str(stages.write_margin(root=root)))
    elif stage == "fit":
        run_one = device_runner(root=root, device=args.device)
        run_one(args.arm, args.arch, args.seed)
    elif stage == "components":
        log(str(stages.stage_components(root=root)["pass"]))
    elif stage == "report":
        print(stages.stage_report(root=root))
    elif stage == "figures":
        stages.stage_figures(root=root, log=log)
    elif stage == "real":
        ctx = stages.real_context()
        rows, era = stages.real_inputs(ctx=ctx)
        stages.stage_real(root=root, real_rows=rows, era_of_row=era, log=log)
    elif stage == "score":
        ctx = stages.real_context()
        stages.stage_score(root=root, holidays=ctx.cal.holidays, log=log)
    elif stage == "series":
        ctx = stages.real_context()
        table, windows = stages.series_inputs(root=root, holidays=ctx.cal.holidays, real_rows=None)
        stages.stage_series(root=root, table=table, windows=windows, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
